#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    KEEP_CLASSES,
    MASK_ROAD,
    RoundPaths,
    load_config,
    match_boxes,
    read_csv,
    read_yolo,
    update_meta,
    write_json,
)

# 사람이 검수한 라벨과 자동 라벨을 같은 프레임에서 비교해 자동 라벨 품질을 잰다.
# 이 값으로 다음 회차 검수 비율(review.random_ratio)을 정한다.
#
# 예) python3 -m training.label_qa --config training/configs/round.yaml
# 결과: out/training/datasets/<회차>/qa/label_qa.json, label_qa.md
#
# 지표 정의(같은 클래스끼리 IoU 0.5 이상을 짝으로 봄)
#   박스 IoU: 짝지어진 박스들의 IoU 평균
#   놓친 비율: 사람 박스 가운데 자동 라벨에 짝이 없는 비율
#   잘못 잡은 비율: 자동 박스 가운데 사람 라벨에 짝이 없는 비율
# 의심 프레임은 일부러 어려운 것만 모은 것이라 전체 품질은 무작위 표본 열로 판단한다.


# ---------------------------------------------------------------------------
# 비교
# ---------------------------------------------------------------------------
def compare_sets(fids: list, human_dir: str, auto_dir: str, img_dir: str, iou_thr: float = 0.5) -> dict:
    tot = {"frames": 0, "human": 0, "auto": 0, "matched": 0, "iou_sum": 0.0}
    per = {c: {"human": 0, "auto": 0, "matched": 0} for c in KEEP_CLASSES}
    mask_ious = []
    for fid in fids:
        hp = os.path.join(human_dir, "labels", fid + ".txt")
        ip = os.path.join(img_dir, fid + ".jpg")
        if not os.path.exists(hp) or not os.path.exists(ip):
            continue
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        hb = [b for b in read_yolo(hp, w, h) if b.cls in KEEP_CLASSES]
        ab = [b for b in read_yolo(os.path.join(auto_dir, "labels", fid + ".txt"), w, h) if b.cls in KEEP_CLASSES]
        pairs, _, _ = match_boxes(hb, ab, iou_thr)
        tot["frames"] += 1
        tot["human"] += len(hb)
        tot["auto"] += len(ab)
        tot["matched"] += len(pairs)
        tot["iou_sum"] += sum(p[2] for p in pairs)
        for b in hb:
            per[b.cls]["human"] += 1
        for b in ab:
            per[b.cls]["auto"] += 1
        for i, _j, _ in pairs:
            per[hb[i].cls]["matched"] += 1
        hm, am = os.path.join(human_dir, "masks", fid + ".png"), os.path.join(auto_dir, "masks", fid + ".png")
        if os.path.exists(hm) and os.path.exists(am):
            a = cv2.imread(hm, cv2.IMREAD_GRAYSCALE) == MASK_ROAD
            b = cv2.imread(am, cv2.IMREAD_GRAYSCALE) == MASK_ROAD
            union = np.logical_or(a, b).sum()
            if union:
                mask_ious.append(float(np.logical_and(a, b).sum() / union))
    return summarize(tot, per, mask_ious)


def _rate(num: float, den: float):
    return round(num / den, 4) if den else None


def summarize(tot: dict, per: dict, mask_ious: list) -> dict:
    return {
        "frames": tot["frames"],
        "human_boxes": tot["human"],
        "auto_boxes": tot["auto"],
        "box_iou": _rate(tot["iou_sum"], tot["matched"]),
        "miss_rate": _rate(tot["human"] - tot["matched"], tot["human"]),
        "false_rate": _rate(tot["auto"] - tot["matched"], tot["auto"]),
        "road_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else None,
        "per_class": {
            KEEP_CLASSES[c]: {
                "human": v["human"], "auto": v["auto"],
                "miss_rate": _rate(v["human"] - v["matched"], v["human"]),
                "false_rate": _rate(v["auto"] - v["matched"], v["auto"]),
            } for c, v in per.items() if v["human"] or v["auto"]
        },
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def to_markdown(rnd: str, res: dict) -> str:
    a, r = res["all"], res["random"]
    lines = [
        f"## 자동 라벨 품질 ({rnd})",
        "",
        "사람 검수 라벨을 정답으로 보고, 같은 클래스끼리 IoU 0.5 이상을 짝으로 셌습니다.",
        "",
        "| 지표 | 검수 전체 | 무작위 표본 |",
        "|---|---|---|",
    ]
    for key, name in (("frames", "프레임 수"), ("human_boxes", "사람 박스 수"), ("box_iou", "박스 IoU 평균"),
                      ("miss_rate", "놓친 비율"), ("false_rate", "잘못 잡은 비율"), ("road_iou", "도로 마스크 IoU")):
        lines.append(f"| {name} | {_fmt(a.get(key))} | {_fmt(r.get(key))} |")
    lines += ["", "클래스별 값은 검수 전체 기준입니다.", "", "| 클래스 | 놓친 비율 | 잘못 잡은 비율 |", "|---|---|---|"]
    for name, v in a["per_class"].items():
        lines.append(f"| {name} ({v['human']}개) | {_fmt(v['miss_rate'])} | {_fmt(v['false_rate'])} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 회차 단위 실행
# ---------------------------------------------------------------------------
def qa_round(cfg: dict) -> dict:
    paths = RoundPaths(cfg)
    review = read_csv(os.path.join(paths.review, "review_list.csv"))
    all_ids = [r["frame_id"] for r in review]
    rand_ids = [r["frame_id"] for r in review if r.get("why") == "random"]
    auto = paths.auto_dir(cfg["autolabel"]["primary"])
    iou = float(cfg["autolabel"]["match_iou"])
    res = {
        "all": compare_sets(all_ids, paths.reviewed, auto, paths.images, iou),
        "random": compare_sets(rand_ids, paths.reviewed, auto, paths.images, iou),
    }
    out = os.path.join(paths.rd, "qa")
    write_json(os.path.join(out, "label_qa.json"), res)
    with open(os.path.join(out, "label_qa.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(cfg["round"], res))
    update_meta(paths, "label_qa", {k: {kk: vv for kk, vv in v.items() if kk != "per_class"} for k, v in res.items()})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="사람 라벨과 자동 라벨 비교")
    ap.add_argument("--config", default="training/configs/round.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    res = qa_round(cfg)
    a = res["all"]
    print(f"검수 {a['frames']}장: 박스 IoU {_fmt(a['box_iou'])}, 놓친 비율 {_fmt(a['miss_rate'])}, "
          f"잘못 잡은 비율 {_fmt(a['false_rate'])}")
    print(f"결과: {os.path.join(RoundPaths(cfg).rd, 'qa', 'label_qa.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
