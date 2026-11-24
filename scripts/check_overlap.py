#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.calib_overlap import overlap_pairs, overlap_score  # noqa: E402
from realityview.config import load_config, scene_scale  # noqa: E402

# 카메라 간 겹침 확인: 같은 순간에 찍은 카메라 영상들로, 함께 보는 바닥이 탑뷰에서 맞물리는지 점수를 낸다.
# 영상은 카메라 이름=경로로 준다. 차와 사람이 없는 빈 도로나 주차장 바닥에서 찍은 것이 좋다.
#
# 예) 4대 설정에서 겹치는 쌍을 모두 보기
#   python3 scripts/check_overlap.py --config configs/four_cam.yaml \
#       --image rear=out/calib/ov/rear_00000.png --image left=out/calib/ov/left_00000.png \
#       --image right=out/calib/ov/right_00000.png --json out/measure/overlap.json
#
# 결과: 쌍마다 특징점 어긋남(shift_m), 차선 횡위치 차이(lane_dy_m), 밝기 상관(zncc)과 판정(ok)을 화면과 --json에.
# 겹치는 바닥이 없는 쌍(지금 two_cam_current.yaml의 전방, 후방)은 "겹침 없음"으로 적는다.


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", action="append", default=[], metavar="CAM=PATH", help="카메라 영상(여러 번)")
    ap.add_argument("--near", type=float, default=12.0, help="두 카메라에서 이 거리(실차 m) 안의 겹친 바닥만 본다")
    ap.add_argument("--pass-m", type=float, default=0.15, help="통과 기준 어긋남(실차 m, 축척을 곱해 쓴다)")
    ap.add_argument("--json", help="결과 json 저장 경로")
    return ap


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cams, grid, cfg = load_config(args.config)
    scale = scene_scale(cfg)
    names = [c.name for c in cams]
    images = [None] * len(cams)
    for spec in args.image:
        name, path = spec.split("=", 1)
        if name not in names:
            raise SystemExit(f"설정에 없는 카메라: {name}")
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"영상을 읽지 못했습니다: {path}")
        images[names.index(name)] = img
    pairs = overlap_pairs(cams, grid, args.near, scale)
    if not pairs:
        print("겹치는 바닥이 있는 카메라 쌍이 없습니다(이 설정은 카메라 사이가 사각지대).")
    results = []
    for i, j, cells in pairs:
        if images[i] is None or images[j] is None:
            print(f"{names[i]}-{names[j]}: 겹친 셀 {cells}개, 영상이 없어 건너뜀")
            continue
        r = overlap_score(cams, images, grid, i, j, scale, args.near, args.pass_m)
        results.append(r)
        fmt = lambda v, k=3, u="m": "못 잼" if v is None else f"{v:.{k}f}{u}"  # noqa: E731
        verdict = {True: "통과", False: "어긋남", None: "판정 못 함"}[r["ok"]]
        print(f"{names[i]}-{names[j]}: 겹친 바닥 {r['area_m2']:.1f}m2, 특징점 어긋남 {fmt(r['shift_m'])}"
              f"(짝 {r['matches']}개), 차선 차이 {fmt(r['lane_dy_m'])}(짝 {r['lane_pairs']}개), "
              f"밝기 상관 {fmt(r['zncc'], 2, '')}, 기준 {r['pass_m']:.3f}m: {verdict}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"config": args.config, "results": results}, f, ensure_ascii=False, indent=1)
    return 1 if any(r["ok"] is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
