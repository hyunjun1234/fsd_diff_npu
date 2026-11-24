#!/usr/bin/env python3
"""체커보드 사진 폴더로 카메라 내부 파라미터를 구한다.

예) python scripts/calib_intrinsics.py --images calib/front/*.jpg --cols 9 --rows 6 --square 0.025 --model fisheye
결과를 configs/*.yaml의 intrinsics에 붙여 넣는다. 촬영 해상도는 데모 해상도(예: 768x432)와 같아야 한다.
"""
from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--cols", type=int, default=9, help="내부 코너 가로 개수")
    ap.add_argument("--rows", type=int, default=6, help="내부 코너 세로 개수")
    ap.add_argument("--square", type=float, default=0.025, help="칸 크기 (m)")
    ap.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole")
    args = ap.parse_args()

    paths = sorted(p for pat in args.images for p in glob.glob(pat))
    objp = np.zeros((args.rows * args.cols, 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square
    obj_pts, img_pts, size = [], [], None
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = g.shape[::-1]
        ok, corners = cv2.findChessboardCorners(g, (args.cols, args.rows))
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            g, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        )
        obj_pts.append(objp)
        img_pts.append(corners.astype(np.float64))
    print(f"사용한 이미지: {len(obj_pts)}/{len(paths)}")
    if len(obj_pts) < 8:
        raise SystemExit("코너를 찾은 이미지가 8장 미만입니다. 각도와 거리를 바꿔 더 찍어 주세요.")

    if args.model == "fisheye":
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj_pts, img_pts, size, K, D, flags=flags)
    else:
        rms, K, D, _, _ = cv2.calibrateCamera([o.astype(np.float32) for o in obj_pts],
                                             [i.astype(np.float32) for i in img_pts], size, None, None)
    print(f"재투영 오차 RMS: {rms:.3f} px (1px 이하 권장)")
    print("intrinsics:")
    print(f"  model: {args.model}")
    print(f"  fx: {K[0,0]:.3f}\n  fy: {K[1,1]:.3f}\n  cx: {K[0,2]:.3f}\n  cy: {K[1,2]:.3f}")
    print(f"  dist: [{', '.join(f'{v:.6f}' for v in np.ravel(D))}]")


if __name__ == "__main__":
    main()
