#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.detcompare import summarize_pairs  # noqa: E402
from realityview.types import COCO_ALL  # noqa: E402

# 같은 영상을 Hailo-8과 DX-M1 검출 모델에 차례로 넣어 FPS, 검출 수, 상자 위치 차이를 비교한다.
# 두 NPU를 동시에 돌리지 않는다(각자의 처리 능력을 따로 재려는 비교).
#
# 예) 파이 저장소 폴더, .venv 켠 뒤 (NPU를 잡으므로 timeout을 붙인다)
#   # record_frames.py로 녹화한 카메라 2대 100프레임
#   timeout -s INT -k 10 300 python3 scripts/compare_det.py --config configs/two_cam_current.yaml \
#       --rec out/rec/room_100 --dx models/dxnn/yolov8-s_640x640_ppu.dxnn \
#       --dx models/dxnn/yolov8-s_640x640.dxnn --out out/measure/06_compare_rec.json
#   # 정지 영상 몇 장 (카메라 구분 없이 한 장씩)
#   timeout -s INT -k 10 120 python3 scripts/compare_det.py --config configs/two_cam_current.yaml \
#       --images out/measure/01_frame_cam0.jpg out/measure/01_frame_cam1.jpg \
#       --dx models/dxnn/yolov8-s_640x640_ppu.dxnn --out out/measure/06_compare_still.json
#
# FPS는 영상을 모두 메모리에 올린 뒤 backend.infer()(전처리, 추론, 후처리)만 잰 값이다.
# 녹화 입력이면 한 번의 infer()에 카메라 수만큼 영상이 들어가므로 "프레임(카메라 묶음)/초"다.


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="npu 항목(score_thr, det_hef 등)을 읽을 설정 yaml")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rec", help="record_frames.py 녹화 폴더(timestamps.csv가 있는 곳)")
    src.add_argument("--images", nargs="+", help="정지 영상 경로들(한 장씩 따로 넣는다)")
    ap.add_argument("--dx", action="append", default=[], help="비교할 DX-M1 .dxnn(여러 번 가능)")
    ap.add_argument("--hef", help="Hailo HEF. 없으면 설정의 npu.det_hef")
    ap.add_argument("--iou", type=float, default=0.5, help="같은 물체로 볼 IoU 기준")
    ap.add_argument("--warmup", type=int, default=5, help="시간 측정에서 뺄 앞쪽 호출 수")
    ap.add_argument("--all-classes", action="store_true",
                    help="COCO 80 클래스를 모두 남긴다(방 안처럼 사람, 차가 없을 때 의자 등으로 비교)")
    ap.add_argument("--score-thr", type=float, help="점수 문턱값 덮어쓰기. 없으면 설정의 npu.score_thr")
    ap.add_argument("--out", required=True, help="결과 JSON 경로")
    return ap


# ---------------------------------------------------------------------------
# 입력 영상 읽기
# ---------------------------------------------------------------------------
def load_rec(folder: str) -> tuple[list[list[np.ndarray]], list[str], list[str]]:
    """녹화 폴더 → (프레임마다 카메라 영상 목록, 카메라 이름 순서, 프레임마다 대표 이름)."""
    with open(os.path.join(folder, "timestamps.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    cams: list[str] = []
    by_frame: dict[int, dict[str, str]] = {}
    for r in rows:
        if r["camera"] not in cams:
            cams.append(r["camera"])
        by_frame.setdefault(int(r["frame"]), {})[r["camera"]] = r["file"]
    frames, names = [], []
    for k in sorted(by_frame):
        if set(by_frame[k]) != set(cams):
            continue  # 한 카메라라도 빠진 프레임은 쓰지 않는다
        frames.append([cv2.imread(os.path.join(folder, by_frame[k][c])) for c in cams])
        names.append(f"frame{k:05d}")
    return frames, cams, names


def load_images(paths: list[str]) -> tuple[list[list[np.ndarray]], list[str], list[str]]:
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        frames.append([img])
    return frames, ["image"], [os.path.basename(p) for p in paths]


# ---------------------------------------------------------------------------
# 검출기 하나 돌리기
# ---------------------------------------------------------------------------
def run_backend(backend, frames: list[list[np.ndarray]], warmup: int) -> tuple[list[list[list]], dict]:
    """모든 프레임을 넣고 (프레임별, 카메라별 Detection 목록, 시간 통계)를 돌려준다."""
    for f in frames[: min(warmup, len(frames))]:
        backend.infer(f)
    out, times = [], []
    for f in frames:
        t0 = time.perf_counter()
        res = backend.infer(f)
        times.append(time.perf_counter() - t0)
        out.append([r.detections for r in res])
    total = float(np.sum(times))
    stats = {
        "calls": len(times),
        "images_per_call": len(frames[0]) if frames else 0,
        "fps": round(len(times) / total, 2) if total > 0 else None,
        "ms_mean": round(1000 * float(np.mean(times)), 2),
        "ms_p95": round(1000 * float(np.percentile(times, 95)), 2),
    }
    return out, stats


def make_hailo(npu: dict, hef: str | None, score_thr: float, labels: dict | None):
    from realityview.detectors import HailoBackend

    return HailoBackend(hef or npu["det_hef"], None, score_thr=score_thr, labels=labels)


def make_dx(npu: dict, model: str, score_thr: float, labels: dict | None):
    from realityview.detectors import DXM1Backend

    return DXM1Backend(
        model,
        None,
        score_thr=score_thr,
        labels=labels,
        iou_thr=float(npu.get("dx_iou_thr", 0.45)),
        det_format=npu.get("dx_det_format", "yolov8"),
        input_rgb=bool(npu.get("dx_input_rgb", True)),
    )


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    _, _, cfg = load_config(args.config)
    npu = cfg.get("npu", {})
    if args.rec:
        frames, cam_names, frame_names = load_rec(args.rec)
    else:
        frames, cam_names, frame_names = load_images(args.images)
    if not frames:
        print("입력 영상이 없습니다", file=sys.stderr)
        return 2

    # ---- 두 검출기에 같은 조건(점수 문턱값, 남길 클래스)을 준다 ----
    score_thr = args.score_thr if args.score_thr is not None else float(npu.get("score_thr", 0.4))
    labels = COCO_ALL if args.all_classes else None  # None이면 각 백엔드가 COCO_KEEP(주행 클래스)
    rep: dict = {
        "input": args.rec or args.images,
        "cameras": cam_names,
        "frames": len(frames),
        "classes": "COCO 80" if args.all_classes else "COCO_KEEP",
        "score_thr": score_thr,
        "dx_iou_thr": float(npu.get("dx_iou_thr", 0.45)),
        "match_iou": args.iou,
        "runs": {},
    }
    dets: dict[str, list[list[list]]] = {}

    # ---- 검출기마다 차례로: 열고, 돌리고, 닫는다(두 NPU를 동시에 잡지 않는다) ----
    specs = [("hailo", lambda: make_hailo(npu, args.hef, score_thr, labels))]
    specs += [(f"dx:{os.path.basename(m)}", (lambda m=m: make_dx(npu, m, score_thr, labels))) for m in args.dx]
    for name, make in specs:
        backend = make()
        try:
            dets[name], rep["runs"][name] = run_backend(backend, frames, args.warmup)
        finally:
            backend.close()
        print(name, json.dumps(rep["runs"][name]), flush=True)

    # ---- Hailo를 기준으로 카메라별, 전체 비교 ----
    rep["compare"] = {}
    for name in dets:
        if name == "hailo":
            continue
        per_cam = {}
        for ci, cam in enumerate(cam_names):
            ref = [fr[ci] for fr in dets["hailo"]]
            oth = [fr[ci] for fr in dets[name]]
            per_cam[cam] = summarize_pairs(ref, oth, args.iou)
        ref_all = [d for fr in dets["hailo"] for d in fr]
        oth_all = [d for fr in dets[name] for d in fr]
        rep["compare"][name] = {"all": summarize_pairs(ref_all, oth_all, args.iou), "per_camera": per_cam}

    # ---- 클래스별 검출 수(어떤 물체가 잡혔는지 보려고) ----
    rep["label_counts"] = {}
    for name, fr_list in dets.items():
        counts: dict[str, int] = {}
        for fr in fr_list:
            for cam_dets in fr:
                for d in cam_dets:
                    counts[d.label] = counts.get(d.label, 0) + 1
        rep["label_counts"][name] = counts

    # ---- 정지 영상은 상자까지 남긴다(눈으로 대조하려고) ----
    if args.images:
        rep["boxes"] = {
            name: {fn: [dataclasses.asdict(d) for d in fr[0]] for fn, fr in zip(frame_names, fr_list)}
            for name, fr_list in dets.items()
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v["all"] for k, v in rep["compare"].items()}, ensure_ascii=False))
    print(json.dumps(rep["label_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
