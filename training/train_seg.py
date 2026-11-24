#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    MASK_IGNORE,
    MASK_ROAD,
    RoundPaths,
    load_config,
    resolve_path,
    update_meta,
    write_json,
)

# BiSeNetV2 도로 세그 미세조정용 데이터를 만든다. 학습 코드는 mmsegmentation 1.x 를 쓸 예정이며
# 그 부분(run_mmseg)은 설치한 버전에서 확인 필요다. 입력 데이터 준비까지는 이 파일만으로 끝난다.
#
# 예) python3 -m training.train_seg --config training/configs/round.yaml           # 데이터 준비만
#     python3 -m training.train_seg --config training/configs/round.yaml --run     # 서버에서 mmseg 학습까지
#
# 결과: out/training/datasets/<회차>/seg/img_dir/{train,val}/*.jpg
#       out/training/datasets/<회차>/seg/ann_dir/{train,val}/*.png
#       out/training/datasets/<회차>/seg/info.json, models/<회차>/seg/ (학습 기록, --run 일 때)
#
# 라벨 규칙(label_mode)
#   road_first:      0 도로, 1 그 밖, 255 무시. 2 클래스 모델이 되고, 도로 번호가 0 이라 파이의
#                    road_classes=(0,) 설정을 바꾸지 않아도 된다.
#   cityscapes_keep: 0 도로, 나머지 전부 255(무시). Cityscapes 19 클래스 머리를 그대로 두고 도로만 더 배운다.
#                    도로 아닌 화소를 배우지 않아 도로를 넓게 칠하는 쪽으로 치우칠 수 있어 기본값이 아니다.

LABEL_MODES = ("road_first", "cityscapes_keep")


# ---------------------------------------------------------------------------
# 마스크 변환
# ---------------------------------------------------------------------------
def convert_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    """회차 마스크(0 도로 아님, 1 도로, 255 모름)를 학습 라벨로."""
    if mode not in LABEL_MODES:
        raise ValueError(f"label_mode 는 {LABEL_MODES} 중 하나: {mode}")
    out = np.full(mask.shape, 255, np.uint8)
    road = mask == MASK_ROAD
    out[road] = 0
    if mode == "road_first":
        out[(~road) & (mask != MASK_IGNORE)] = 1
    return out


def num_classes(mode: str) -> int:
    return 2 if mode == "road_first" else 19


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def prepare_seg(cfg: dict) -> dict:
    """검출과 같은 학습, 검증 나누기(세그먼트 단위)를 쓰고, 마스크가 있는 프레임만 넣는다."""
    from training.train_det import filter_test_leak, known_frame_times, split_own
    from training.select_frames import load_lock

    paths = RoundPaths(cfg)
    mode = cfg["train_seg"]["label_mode"]
    seg_root = os.path.join(paths.rd, "seg")
    lock = load_lock(paths)
    train, val = split_own(cfg)
    times = known_frame_times(cfg)
    train, _ = filter_test_leak(train, lock, times)
    val, _ = filter_test_leak(val, lock, times)
    counts = {}
    for sub in ("img_dir", "ann_dir"):
        # 다시 준비할 때 옛 나누기의 파일이 남아 학습과 검증이 섞이지 않게 비운다(회차 폴더 안 사본만)
        d = os.path.join(seg_root, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
    for split, items in (("train", train), ("val", val)):
        idir = os.path.join(seg_root, "img_dir", split)
        adir = os.path.join(seg_root, "ann_dir", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(adir, exist_ok=True)
        n = 0
        for img in items:
            fid = os.path.splitext(os.path.basename(img))[0]
            mp = os.path.join(paths.masks, fid + ".png")
            if not os.path.exists(mp):
                continue
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(adir, fid + ".png"), convert_mask(m, mode))
            dst = os.path.join(idir, fid + ".jpg")
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(img, dst)  # 같은 디스크면 복사하지 않고 하드링크
            except OSError:
                shutil.copyfile(img, dst)
            n += 1
        counts[split] = n
    info = {"label_mode": mode, "num_classes": num_classes(mode), "root": seg_root, **counts}
    write_json(os.path.join(seg_root, "info.json"), info)
    update_meta(paths, "seg_prepare", info)
    return info


# ---------------------------------------------------------------------------
# mmsegmentation 학습 (서버에서만. API 와 기본 설정 파일 이름은 설치한 버전에서 확인 필요)
# ---------------------------------------------------------------------------
def build_mmseg_config(cfg: dict, info: dict):
    """mmseg 의 Cityscapes BiSeNetV2 설정을 읽어 우리 데이터 경로와 클래스 수로 바꾼다."""
    from mmengine.config import Config  # 확인 필요(mmengine 0.10.x)

    s = cfg["train_seg"]
    base = os.path.join(resolve_path(s["mmseg_root"]), s["base_config"])
    c = Config.fromfile(base)
    nc = info["num_classes"]
    ch, cw = s["crop"]
    classes = ("road", "other") if nc == 2 else None
    c.model.data_preprocessor.size = (ch, cw)
    c.model.decode_head.num_classes = nc
    aux = c.model.get("auxiliary_head")
    for h in (aux if isinstance(aux, list) else [aux] if aux else []):
        h.num_classes = nc
    for loader, split in (("train_dataloader", "train"), ("val_dataloader", "val"), ("test_dataloader", "val")):
        ds = c[loader].dataset
        ds.type = "BaseSegDataset"
        ds.data_root = info["root"]
        ds.data_prefix = dict(img_path=f"img_dir/{split}", seg_map_path=f"ann_dir/{split}")
        ds.img_suffix, ds.seg_map_suffix = ".jpg", ".png"
        ds.reduce_zero_label = False
        if classes:
            ds.metainfo = dict(classes=classes, palette=[[128, 64, 128], [0, 0, 0]])
        for t in ds.get("pipeline", []):
            if t.get("type") in ("RandomResize", "Resize"):
                t["scale"] = (cw, ch)
            if t.get("type") == "RandomCrop":
                t["crop_size"] = (ch, cw)
    c.train_dataloader.batch_size = int(s["batch"])
    c.train_cfg.max_iters = int(s["iters"])
    c.train_cfg.val_interval = max(1000, int(s["iters"]) // 10)
    c.load_from = resolve_path(s.get("load_from"))
    c.work_dir = os.path.join(RoundPaths(cfg).model_dir(), "seg")
    return c


def run_mmseg(cfg: dict, info: dict) -> str:
    from mmengine.runner import Runner  # 확인 필요

    c = build_mmseg_config(cfg, info)
    os.makedirs(c.work_dir, exist_ok=True)
    c.dump(os.path.join(c.work_dir, "config.py"))
    Runner.from_cfg(c).train()
    return c.work_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BiSeNetV2 도로 세그 데이터 준비와 학습")
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--run", action="store_true", help="준비 뒤 mmsegmentation 학습까지(서버)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    info = prepare_seg(cfg)
    print(f"세그 데이터: 학습 {info['train']}장, 검증 {info['val']}장, 클래스 {info['num_classes']}개 ({info['label_mode']})")
    print(f"결과: {info['root']}")
    if args.run:
        print(f"학습 기록: {run_mmseg(cfg, info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
