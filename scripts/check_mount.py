from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realityview.bev import IpmCompositor  # noqa: E402
from realityview.config import load_config, scene_scale  # noqa: E402

# 카메라 장착 위치와 화각으로 BEV 바닥 영역이 어디에 놓이는지 계산한다(카메라, NPU 없이 설정만 읽음).
#   - 카메라마다 맡은 바닥 셀의 X, Y 범위(전방 카메라 셀이 차 앞, 후방 카메라 셀이 차 뒤에 있는지)
#   - 카메라 정면 방향으로 가장 가까이 보이는 바닥 거리
#   - 정면 방향 거리별로 영상 한 화소가 바닥에서 몇 m인지(BEV 셀 크기와 비교)
# 예:
#   python3 scripts/check_mount.py --config configs/two_cam_current.yaml --out out/measure/07_mount_current.json
# 결과는 화면과 --out JSON에 남는다.

# 실차 기준 정면 거리(m). 모형 설정이면 scene.scale을 곱해 잰다.
DISTANCES_CAR = (2.0, 5.0, 10.0, 25.0)


# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------
def _heading(cam) -> np.ndarray:
    """카메라가 바라보는 수평 방향(차량 좌표 X, Y 단위 벡터)."""
    a = np.radians(cam.yaw_deg)
    return np.array([np.cos(a), np.sin(a)])


def nearest_ground_ahead(cam, step: float, far: float) -> float | None:
    """카메라 정면 방향 바닥을 가까운 곳부터 훑어 처음 영상 안에 들어오는 거리(카메라 발밑 기준)."""
    h = _heading(cam)
    for d in np.arange(step, far, step):
        xy = cam.position[:2] + d * h
        uv, ok = cam.project(np.array([[xy[0], xy[1], 0.0]]))
        if ok[0] and 0 <= uv[0, 0] <= cam.width - 1 and 0 <= uv[0, 1] <= cam.height - 1:
            return float(d)
    return None


def ground_m_per_px(cam, d: float, eps: float) -> float | None:
    """정면 거리 d의 바닥에서 앞뒤로 eps만큼 움직일 때 영상에서 몇 화소 움직이는지로 화소당 바닥 길이를 구한다."""
    h = _heading(cam)
    a = cam.position[:2] + d * h
    b = cam.position[:2] + (d + eps) * h
    uv, ok = cam.project(np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]]))
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] <= cam.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= cam.height - 1)
    if not inside.all():
        return None  # 영상 밖(화각 밖이거나 수평선 위)
    px = float(np.linalg.norm(uv[1] - uv[0]))
    return eps / px if px > 0 else None


def check(config: str) -> dict:
    cams, grid, cfg = load_config(config)
    s = scene_scale(cfg)
    ipm = IpmCompositor(cams, grid)
    X, Y = grid.cell_centers()
    out = {
        "config": config,
        "scale": s,
        "bev": {"x": [grid.x_min, grid.x_max], "y": [grid.y_min, grid.y_max], "res": grid.res,
                "cells": [grid.height, grid.width]},
        "coverage_ratio": float(ipm.coverage().mean()),
        "cameras": [],
    }
    for i, cam in enumerate(cams):
        own = ipm.owner == i
        row = {
            "name": cam.name,
            "position": [round(float(v), 4) for v in cam.position],
            "yaw": cam.yaw_deg, "pitch": cam.pitch_deg, "roll": cam.roll_deg,
            "cells": int(own.sum()),
        }
        if own.any():
            row["x_range"] = [float(X[own].min()), float(X[own].max())]
            row["y_range"] = [float(Y[own].min()), float(Y[own].max())]
            # 바닥 영역이 카메라의 어느 쪽에 놓였는지: 정면 방향 성분이 모두 양수면 카메라 앞쪽
            rel = np.stack([X[own] - cam.position[0], Y[own] - cam.position[1]], 1) @ _heading(cam)
            row["all_cells_in_front_of_camera"] = bool((rel > 0).all())
        row["nearest_ground_ahead"] = nearest_ground_ahead(cam, step=0.01 * s, far=60.0 * s)
        row["m_per_px"] = {str(d): ground_m_per_px(cam, d * s, eps=0.01 * s) for d in DISTANCES_CAR}
        out["cameras"].append(row)
    return out


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="결과 JSON 경로. 예: out/measure/07_mount_current.json")
    args = ap.parse_args()
    r = check(args.config)
    b = r["bev"]
    print(f"{r['config']}  scale {r['scale']}  BEV x {b['x']} y {b['y']} 셀 {b['res']:.4f}m {b['cells']}"
          f"  바닥이 보이는 셀 {r['coverage_ratio'] * 100:.1f}%")
    for c in r["cameras"]:
        print(f"- {c['name']}: 위치 {c['position']} yaw {c['yaw']} pitch {c['pitch']}  셀 {c['cells']}개")
        if c["cells"]:
            print(f"    X {c['x_range'][0]:.3f} ~ {c['x_range'][1]:.3f}, Y {c['y_range'][0]:.3f} ~ {c['y_range'][1]:.3f}"
                  f", 모두 카메라 앞쪽: {c['all_cells_in_front_of_camera']}")
        print(f"    정면 바닥이 처음 보이는 거리: {c['nearest_ground_ahead']}")
        mpp = ", ".join(f"{d}m(실차): {'-' if v is None else f'{v:.3f}'}" for d, v in c["m_per_px"].items())
        print(f"    화소당 바닥 길이(m): {mpp}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
