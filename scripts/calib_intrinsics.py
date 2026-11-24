#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.calib import board_points, calibrate_intrinsics, find_corners, intrinsics_yaml  # noqa: E402

# 체커보드 사진 폴더로 카메라 내부 파라미터(K, 왜곡 계수)를 구한다.
# 사진은 scripts/record_frames.py로 설정 파일과 같은 해상도, 센서 모드로 찍는다.
#
# 예) 모형용 작은 보드(내부 코너 9x6, 칸 20mm), 전방 130도 렌즈는 어안 모델
#   python3 scripts/calib_intrinsics.py --images 'out/calib/front/front_*.png' --cols 9 --rows 6 \
#       --square 0.020 --model fisheye --out out/calib/front_intrinsics.yaml
#
# 결과: 화면과 --out 파일에 설정 파일 intrinsics 항목에 붙여 넣을 글, 같은 이름의 .json에 사진별 오차


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="사진 경로나 glob 패턴(따옴표로 감싼다)")
    ap.add_argument("--cols", type=int, default=9, help="내부 코너 가로 개수(칸 수 - 1)")
    ap.add_argument("--rows", type=int, default=6, help="내부 코너 세로 개수(칸 수 - 1)")
    ap.add_argument("--square", type=float, default=0.020, help="칸 한 변 길이(m). 인쇄 후 자로 잰 값")
    ap.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole")
    ap.add_argument("--out", help="결과 yaml 조각 저장 경로. 같은 이름 .json에 사진별 오차도 남긴다")
    ap.add_argument("--min-images", type=int, default=10, help="코너를 찾은 사진이 이보다 적으면 멈춘다")
    return ap


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    paths = sorted({p for pat in args.images for p in glob.glob(pat)})
    objp = board_points(args.cols, args.rows, args.square)
    obj_pts, img_pts, names, size = [], [], [], None
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"읽지 못함: {p}", file=sys.stderr)
            continue
        if size is not None and g.shape[::-1] != size:
            print(f"해상도가 다른 사진이라 뺍니다: {p} {g.shape[::-1]} (앞 사진 {size})", file=sys.stderr)
            continue
        size = g.shape[::-1]
        c = find_corners(g, args.cols, args.rows)
        if c is None:
            print(f"코너 못 찾음: {os.path.basename(p)}")
            continue
        obj_pts.append(objp)
        img_pts.append(c)
        names.append(p)
    print(f"코너를 찾은 사진: {len(obj_pts)}/{len(paths)}")
    if len(obj_pts) < args.min_images:
        print(f"{args.min_images}장 미만입니다. 보드를 화면 가장자리, 가까이, 기울여 가며 더 찍어 주세요.", file=sys.stderr)
        return 2

    res = calibrate_intrinsics(obj_pts, img_pts, size, args.model)
    for i, why in res["dropped"]:
        print(f"뺀 사진: {os.path.basename(names[i])} ({why})")
    # 사진별 오차가 큰 것부터 다섯 장(다시 찍을 사진 고르기용)
    worst = sorted(zip(res["per_view"], res["used"]), reverse=True)[:5]
    print("오차가 큰 사진:", ", ".join(f"{os.path.basename(names[i])} {e:.2f}px" for e, i in worst))
    print(f"재투영 오차 RMS: {res['rms']:.3f} px (완료 기준 1px 이하)")
    text = intrinsics_yaml(res)
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        detail = {
            "rms": res["rms"], "model": res["model"], "size_wh": res["size_wh"],
            "K": res["K"].tolist(), "D": res["D"].tolist(),
            "per_view": {os.path.basename(names[i]): e for i, e in zip(res["used"], res["per_view"])},
            "dropped": [os.path.basename(names[i]) for i, _ in res["dropped"]],
            "square_m": args.square, "cols": args.cols, "rows": args.rows,
        }
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")
    return 0 if res["rms"] <= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
