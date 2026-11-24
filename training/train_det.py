#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    COCO_NAMES,
    RoundPaths,
    check_round_name,
    load_config,
    parse_frame_id,
    read_csv,
    segment_key,
    update_meta,
    write_json,
)

# YOLOv8s 검출 모델을 이번 회차 데이터로 이어서 학습한다.
# 데이터 섞기: 이번 회차 학습 프레임 전부 + 이전 회차 학습 프레임 일부 + COCO 일부(망각 방지)
# 시작점: 첫 회차는 yolov8s.pt(COCO 사전학습), 둘째 회차부터는 models/<직전 회차>/best.pt
#
# 예) 학습 서버(WSL2, GPU)에서
#   python3 -m training.train_det --config training/configs/round.yaml               # 목록 만들고 학습
#   python3 -m training.train_det --config training/configs/round.yaml --prepare-only  # 목록과 data.yaml 만
#
# 결과: out/training/datasets/<회차>/own_train.txt, val.txt, train.txt, data.yaml
#       models/<회차>/det/(Ultralytics 학습 기록), models/<회차>/best.pt, train_info.json
#
# RTX 4070 12GB 기본값 근거
#   imgsz 640: 파이의 HEF, dxnn 입력이 640x640 이라 학습도 같은 크기로 맞춘다(다르게 하면 변환 뒤 정확도가 흔들림).
#   batch 16: Ultralytics 기본값. YOLOv8s 640 은 배치 16 에서 GPU 메모리 약 4~6GB 를 쓴다고 알려져 있어(확인 필요)
#             12GB 에 여유가 있다. WSL2 는 Windows 화면도 같은 GPU 를 써서 1GB 남짓 덜 쓸 수 있다. 첫 학습 로그의
#             GPU_mem 열이 8GB 아래면 32 로 올려도 된다. batch: -1 이면 Ultralytics 가 메모리 60% 에 맞춰 정한다.
#   workers 8: 7950X3D 는 16코어지만 WSL2 기본 메모리 한도(호스트 64GB 의 절반)에서 데이터 로더 복사본이 늘면
#              메모리가 모자랄 수 있어 절반만 쓴다.
#   epochs 50, patience 20: 사전학습 모델 미세조정이라 처음부터 학습(기본 100)보다 짧게. 검증 점수가 20번 안 오르면 멈춘다.
#   cache False: 회차 데이터는 수천 장이라 디스크로도 충분하다. 'ram' 은 COCO 를 많이 섞을 때 WSL2 메모리를 넘길 수 있다.


# ---------------------------------------------------------------------------
# 시험 세트 누설 막기 (학습 목록을 만드는 모든 길이 이 함수를 지난다)
# ---------------------------------------------------------------------------
def known_frame_times(cfg: dict) -> dict:
    """모든 회차 frames.csv 에서 {프레임 이름: 세그먼트 안 시각(초)}. exclude=window 판정에 쓴다."""
    paths = RoundPaths(cfg)
    out: dict = {}
    if not os.path.isdir(paths.datasets):
        return out
    for name in os.listdir(paths.datasets):
        for r in read_csv(os.path.join(paths.datasets, name, "frames.csv")):
            try:
                out[r["frame_id"]] = float(r["t"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def filter_test_leak(image_paths: list, lock, times: dict | None = None) -> tuple[list, list]:
    """이미지 경로 목록에서 시험 세트와 겹치는 것을 뺀다. (남은 목록, 뺀 목록)

    프레임 이름 규칙(주행__세그먼트__카메라__번호)에 맞는 것만 검사한다. COCO 이미지는 그대로 둔다.
    exclude=window 일 때는 times(known_frame_times)에서 시각을 찾고, 시각을 모르는 프레임은
    시험 프레임이 있는 세그먼트를 통째로 빼서 보수적으로 막는다.
    """
    times = times or {}
    kept, dropped = [], []
    for p in image_paths:
        fid = os.path.splitext(os.path.basename(p))[0]
        info = parse_frame_id(fid)
        if info is not None and lock.excludes(fid, segment_key(info["drive"], info["segment"]), times.get(fid)):
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


def assert_no_test_leak(image_paths: list, lock, times: dict | None = None) -> None:
    _, bad = filter_test_leak(image_paths, lock, times)
    if bad:
        raise RuntimeError(f"학습 목록에 시험 세트와 겹치는 이미지가 {len(bad)}장 있습니다. 예: {bad[:3]}")


# ---------------------------------------------------------------------------
# 이번 회차 학습, 검증 나누기 (세그먼트 단위로 나눠 옆 프레임이 양쪽에 들어가지 않게)
# ---------------------------------------------------------------------------
def split_own(cfg: dict) -> tuple[list, list]:
    paths = RoundPaths(cfg)
    src = {r["frame_id"]: r["label_source"] for r in read_csv(paths.sources_csv)}
    frames = [r for r in read_csv(paths.frames_csv) if src.get(r["frame_id"], "missing") != "missing"]
    rng = random.Random(int(cfg["train_det"]["seed"]))
    ratio = float(cfg["train_det"]["val_ratio"])
    drive_frames = [r for r in frames if r["source"] == "drive"]
    segs = sorted({r["seg_key"] for r in drive_frames})
    val_ids = set()
    target = ratio * len(drive_frames)
    if len(segs) >= 2:
        # 세그먼트를 섞어 차례로 검증에 넣되, 목표의 1.5배를 넘기는 세그먼트는 건너뛴다(큰 세그먼트 하나가
        # 학습 데이터를 다 가져가지 않게). 하나는 반드시 학습에 남긴다.
        rng.shuffle(segs)
        count = {s: 0 for s in segs}
        for r in drive_frames:
            count[r["seg_key"]] += 1
        n, used = 0, 0
        for s in segs:
            if n >= target or used >= len(segs) - 1:
                break
            if n + count[s] > max(1.5 * target, 1):
                continue
            val_ids.update(r["frame_id"] for r in drive_frames if r["seg_key"] == s)
            n += count[s]
            used += 1
    if not val_ids:
        # 세그먼트로 나눌 수 없으면 세그먼트마다 시간 순 마지막 몫을 검증으로(옆 프레임이 섞일 수 있어 차선책)
        by_seg: dict = {}
        for r in drive_frames:
            by_seg.setdefault(r["seg_key"], []).append(r)
        for rows in by_seg.values():
            order = sorted(rows, key=lambda r: float(r["t"]))
            k = int(round(ratio * len(order)))
            val_ids.update(r["frame_id"] for r in order[len(order) - k:] if k > 0)
    img = lambda fid: os.path.join(paths.images, fid + ".jpg")  # noqa: E731
    train = [img(r["frame_id"]) for r in frames if r["frame_id"] not in val_ids]
    val = [img(r["frame_id"]) for r in frames if r["frame_id"] in val_ids]
    if not val and train:
        val = train[-1:]  # 검증이 비면 Ultralytics 가 멈추므로 한 장이라도 둔다
    return train, val


# ---------------------------------------------------------------------------
# 섞기: 이전 회차 일부, COCO 일부
# ---------------------------------------------------------------------------
def previous_round_images(cfg: dict) -> list:
    paths = RoundPaths(cfg)
    me = check_round_name(cfg["round"])
    out = []
    if not os.path.isdir(paths.datasets):
        return out
    for name in sorted(os.listdir(paths.datasets)):
        try:
            if check_round_name(name) >= me:
                continue
        except ValueError:
            continue
        p = os.path.join(paths.datasets, name, "own_train.txt")
        if os.path.exists(p):
            out += [x.strip() for x in open(p, encoding="utf-8") if x.strip()]
    return out


def coco_images(cfg: dict) -> list:
    """coco_root/images/<split>/ 의 이미지(라벨은 Ultralytics 규칙대로 coco_root/labels/<split>/)."""
    root = RoundPaths(cfg).coco_root
    split = cfg["train_det"]["coco_split"]
    if not root:
        return []
    d = os.path.join(root, "images", split)
    if not os.path.isdir(d) or not os.path.isdir(os.path.join(root, "labels", split)):
        print(f"[경고] COCO 폴더가 없어 섞지 않습니다: {d}", file=sys.stderr)
        return []
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png")))


def build_data_yaml(cfg: dict) -> dict:
    from training.select_frames import load_lock

    paths = RoundPaths(cfg)
    tcfg = cfg["train_det"]
    lock = load_lock(paths)
    if not lock.exists:
        raise RuntimeError("고정 시험 세트가 없습니다. 학습 전에 select_frames 로 정하세요.")
    rng = random.Random(int(tcfg["seed"]))

    times = known_frame_times(cfg)
    own, val = split_own(cfg)
    own, d1 = filter_test_leak(own, lock, times)
    val, d2 = filter_test_leak(val, lock, times)
    prev, d3 = filter_test_leak(previous_round_images(cfg), lock, times)
    prev = [p for p in prev if os.path.exists(p)]
    n_prev = min(len(prev), int(round(float(tcfg["prev_ratio"]) * len(own))))
    prev_pick = rng.sample(prev, n_prev) if n_prev else []
    coco = coco_images(cfg)
    n_coco = min(len(coco), int(round(float(tcfg["coco_ratio"]) * len(own))))
    coco_pick = rng.sample(coco, n_coco) if n_coco else []

    train = own + prev_pick + coco_pick
    assert_no_test_leak(train + val, lock, times)

    os.makedirs(paths.rd, exist_ok=True)
    files = {
        "own_train": os.path.join(paths.rd, "own_train.txt"),
        "train": os.path.join(paths.rd, "train.txt"),
        "val": os.path.join(paths.rd, "val.txt"),
        "data": os.path.join(paths.rd, "data.yaml"),
    }
    for key, items in (("own_train", own), ("train", train), ("val", val)):
        with open(files[key], "w", encoding="utf-8") as f:
            f.write("".join(p + "\n" for p in items))
    # 80 클래스 머리를 유지한다(파이 후처리가 COCO 번호를 읽음). 우리 라벨은 6 클래스만 있으므로
    # 신호등 같은 다른 클래스는 우리 영상에서 배경으로 배우게 된다. COCO 를 섞는 이유 중 하나다.
    data = {
        "path": paths.rd,
        "train": files["train"],
        "val": files["val"],
        "names": {i: n for i, n in enumerate(COCO_NAMES)},
    }
    with open(files["data"], "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    info = {"own_train": len(own), "val": len(val), "prev": len(prev_pick), "coco": len(coco_pick),
            "test_leak_dropped": len(d1) + len(d2) + len(d3), "data_yaml": files["data"]}
    update_meta(paths, "data", info)
    return info


# ---------------------------------------------------------------------------
# 학습기
# ---------------------------------------------------------------------------
class UltralyticsTrainer:
    name = "ultralytics"

    def train(self, data_yaml: str, weights: str, out_dir: str, params: dict) -> str:
        from ultralytics import YOLO  # GPU 서버에서만 불러온다

        model = YOLO(weights)
        model.train(data=data_yaml, project=out_dir, name="det", exist_ok=True, **params)
        best = os.path.join(out_dir, "det", "weights", "best.pt")
        if not os.path.exists(best):
            raise RuntimeError(f"학습은 끝났지만 best.pt 가 없습니다: {best}")
        return best


def start_weights(cfg: dict) -> str:
    """직전 회차 best.pt. 직전이 r000 이면 설정의 base_weights(models/ 아래에 있으면 그 파일)."""
    paths = RoundPaths(cfg)
    prev = cfg["prev_round"]
    if prev == "r000":
        base = cfg["train_det"]["base_weights"]
        local = os.path.join(paths.models, os.path.basename(base))
        return local if os.path.exists(local) else base
    p = paths.best_pt(prev)
    if not os.path.exists(p):
        raise FileNotFoundError(f"직전 회차 모델이 없습니다: {p}. prev_round 를 확인하세요.")
    return p


def train_params(cfg: dict) -> dict:
    t = cfg["train_det"]
    return {k: t[k] for k in ("imgsz", "batch", "epochs", "patience", "workers", "cache", "device", "seed")}


def train_round(cfg: dict, trainer=None, prepare_only: bool = False) -> dict:
    paths = RoundPaths(cfg)
    info = build_data_yaml(cfg)
    if prepare_only:
        return info
    trainer = trainer or UltralyticsTrainer()
    weights = start_weights(cfg)
    out_dir = paths.model_dir()
    best = trainer.train(info["data_yaml"], weights, out_dir, train_params(cfg))
    final = paths.best_pt()
    if os.path.abspath(best) != os.path.abspath(final):
        shutil.copyfile(best, final)
    info.update({"start_weights": weights, "best": final, "trainer": trainer.name, "params": train_params(cfg)})
    write_json(os.path.join(out_dir, "train_info.json"), info)
    update_meta(paths, "train_det", info)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YOLOv8s 검출 회차 학습")
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--prepare-only", action="store_true", help="학습 목록과 data.yaml 만 만든다")
    ap.add_argument("--fake", action="store_true", help="가짜 학습기(흐름 점검용)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    trainer = None
    if args.fake:
        from training.fakes import FakeTrainer

        trainer = FakeTrainer()
    info = train_round(cfg, trainer, args.prepare_only)
    print(f"학습 목록: 이번 회차 {info['own_train']}장, 이전 회차 {info['prev']}장, COCO {info['coco']}장, "
          f"검증 {info['val']}장 (시험 세트와 겹쳐 뺀 것 {info['test_leak_dropped']}장)")
    if "best" in info:
        print(f"결과: {info['best']} (시작점 {info['start_weights']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
