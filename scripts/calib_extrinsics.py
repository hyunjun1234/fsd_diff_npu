#!/usr/bin/env python3
"""바닥 기준점으로 카메라 외부 파라미터(위치, yaw/pitch/roll)를 구한다.

준비: 차 주변 바닥에 테이프/마커로 점을 6개 이상 표시하고 차량 좌표(X 전방, Y 좌측, m)를 줄자로 잰다.
CSV 형식(헤더 포함): u,v,X,Y   (u,v는 해당 카메라 영상의 픽셀 좌표)

예) python scripts/calib_extrinsics.py --config configs/two_cam.yaml --camera front --points calib/front_points.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview.camera_model import pose_from_pnp  # noqa: E402
from realityview.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--points", required=True)
    args = ap.parse_args()

    cams, _, _ = load_config(args.config)
    cam = next(c for c in cams if c.name == args.camera)
    with open(args.points, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    XY = np.array([[float(r["X"]), float(r["Y"]), 0.0] for r in rows])
    if len(rows) < 4:
        raise SystemExit("점이 4개 이상 필요합니다(6개 이상 권장).")

    # 왜곡을 먼저 제거해 정규화 좌표로 만든 뒤 PnP (fisheye/pinhole 공통 처리)
    if cam.model == "fisheye":
        n = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist[:4])
    else:
        n = cv2.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist)
    ok, rvec, tvec = cv2.solvePnP(XY, n, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("solvePnP 실패")
    pos, yaw, pitch, roll = pose_from_pnp(rvec, tvec)

    cam.position, cam.yaw_deg, cam.pitch_deg, cam.roll_deg = pos, yaw, pitch, roll
    pred, _ = cam.project(XY)
    err = np.linalg.norm(pred - uv, axis=1)
    g, _ = cam.pixel_to_ground(uv)
    gerr = np.linalg.norm(g - XY[:, :2], axis=1)
    print(f"재투영 오차: 평균 {err.mean():.2f}px, 최대 {err.max():.2f}px")
    print(f"지면 역투영 오차: 평균 {gerr.mean()*100:.1f}cm, 최대 {gerr.max()*100:.1f}cm")
    print("extrinsics:")
    print(f"  position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"  yaw: {yaw:.2f}\n  pitch: {pitch:.2f}\n  roll: {roll:.2f}")


if __name__ == "__main__":
    main()
