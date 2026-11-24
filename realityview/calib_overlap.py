from __future__ import annotations

import cv2
import numpy as np

from .bev import BevGrid, IpmCompositor
from .camera_model import Camera
from .lanes import LaneConfig, LaneDetector

# 카메라 간 겹침 확인: 두 카메라가 함께 보는 바닥을 각자 탑뷰로 펴서 서로 맞물리는지 점수를 낸다.
# 외부 값이 맞으면 같은 바닥 무늬(차선, 균열, 맨홀)가 두 탑뷰에서 같은 자리에 놓인다. 한쪽 각도가 틀어지면
# 그쪽 탑뷰가 밀리거나 돌아가서 같은 무늬가 어긋난다.
#
# 점수 세 가지
#   shift_m  : 특징점(ORB)을 두 탑뷰에서 짝지어, 짝마다 위치 차이(m)를 재고 그 중앙값. 작을수록 잘 맞음
#   lane_dy_m: 겹친 곳에서 각자 찾은 차선을 짝지어, 같은 X에서 횡위치 차이(m)의 평균
#   zncc     : 겹친 곳 밝기의 정규화 상관(-1~1). 1에 가까울수록 같은 그림. 두 카메라 노출이 다르면 낮아진다
# 판정은 shift_m과 lane_dy_m 중 잰 것이 모두 기준(실차 0.15m x 축척) 이하이면 통과로 본다. 둘 다 못 재면
# (특징점 짝이 모자람) 겹친 바닥이 5m2 이상이고 밝기 상관이 0.5 미만일 때 불합격, 그 밖에는 판정하지 않는다(ok = None).
#
# 주의
#   바닥 위에 서 있는 물체(차, 사람, 벽)는 IPM에서 카메라마다 다른 쪽으로 늘어지므로 특징점이 어긋나 보인다.
#   그런 화소는 occluders(카메라별 마스크, 0이 아니면 뺌)로 빼거나 빈 도로에서 잰다.


def single_camera_ipm(cameras: list[Camera], grid: BevGrid, max_range: float = 30.0) -> list[IpmCompositor]:
    """카메라마다 따로 탑뷰를 펴는 합성기. 합성 탑뷰는 겹친 셀을 한 카메라만 맡아 비교에 쓸 수 없다."""
    return [IpmCompositor([c], grid, max_range) for c in cameras]


def overlap_pairs(cameras: list[Camera], grid: BevGrid, near_m: float = 12.0, scale: float = 1.0,
                  min_cells: int = 200) -> list[tuple[int, int, int]]:
    """함께 보는 바닥 셀이 min_cells 이상인 카메라 쌍 (i, j, 셀 수). 셀은 두 카메라에서 near_m(실차 m) 안만 센다."""
    ipms = single_camera_ipm(cameras, grid)
    out = []
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            n = int(overlap_mask(cameras[i], cameras[j], grid, ipms[i], ipms[j], near_m * scale).sum())
            if n >= min_cells:
                out.append((i, j, n))
    return out


def overlap_mask(cam_a: Camera, cam_b: Camera, grid: BevGrid, ipm_a: IpmCompositor, ipm_b: IpmCompositor,
                 near: float) -> np.ndarray:
    """두 카메라가 모두 보고, 둘 다에서 near(m) 안인 셀(bool). 먼 셀은 IPM이 늘어져 무늬가 번진다."""
    X, Y = grid.cell_centers()
    da = np.hypot(X - cam_a.position[0], Y - cam_a.position[1])
    db = np.hypot(X - cam_b.position[0], Y - cam_b.position[1])
    return ipm_a.coverage() & ipm_b.coverage() & (da <= near) & (db <= near)


# ---------------------------------------------------------------------------
# 점수 계산
# ---------------------------------------------------------------------------
def _feature_shift(ga: np.ndarray, gb: np.ndarray, mask: np.ndarray, res: float, max_shift: float) -> dict:
    """ORB 특징점 짝의 위치 차이(b - a, m). 차량 좌표로 X(위쪽이 +), Y(왼쪽이 +)."""
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=7, edgeThreshold=15, patchSize=15)
    m8 = mask.astype(np.uint8) * 255
    ka, da = orb.detectAndCompute(ga, m8)
    kb, db = orb.detectAndCompute(gb, m8)
    if da is None or db is None or len(ka) < 5 or len(kb) < 5:
        return {"matches": 0, "shift_m": None, "shift_xy_m": None}
    ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if not ms:
        return {"matches": 0, "shift_m": None, "shift_xy_m": None}
    pa = np.array([ka[m.queryIdx].pt for m in ms])
    pb = np.array([kb[m.trainIdx].pt for m in ms])
    d = (pb - pa) * res  # (열, 행) 차이 m
    dxy = np.stack([-d[:, 1], -d[:, 0]], axis=1)  # 행이 늘면 X가 줄고, 열이 늘면 Y가 준다
    keep = np.linalg.norm(dxy, axis=1) <= max_shift  # 엉뚱한 짝(먼 곳의 비슷한 무늬)은 뺀다
    if keep.sum() < 5:
        return {"matches": int(keep.sum()), "shift_m": None, "shift_xy_m": None}
    # 중앙값 벡터 근처(max_shift의 1/10 안)의 짝만 남겨 평균한다. 반복 무늬의 엉뚱한 짝이 값을 끌지 않게
    med = np.median(dxy[keep], axis=0)
    inl = keep & (np.linalg.norm(dxy - med, axis=1) <= max_shift / 10.0)
    if inl.sum() < 5:
        return {"matches": int(inl.sum()), "shift_m": None, "shift_xy_m": None}
    mean = dxy[inl].mean(axis=0)
    return {"matches": int(inl.sum()), "shift_m": float(np.linalg.norm(mean)), "shift_xy_m": mean.tolist()}


def _lane_gap(grid: BevGrid, wa: np.ndarray, wb: np.ndarray, mask: np.ndarray, scale: float,
              lane_cfg: LaneConfig | None, match_m: float) -> dict:
    """겹친 곳에서 두 탑뷰가 각자 찾은 차선을 짝지어 같은 X에서의 횡위치 차이 평균(m)."""
    # 겹친 곳은 좁아서 차선이 짧게 보인다. 기본 설정의 최소 길이(3m)를 1.5m로 줄여 짧은 조각도 쓴다
    cfg = lane_cfg or LaneConfig(min_support_m=1.5)
    la = LaneDetector(grid, cfg, mask, scale).detect(wa)
    lb = LaneDetector(grid, cfg, mask, scale).detect(wb)
    gaps = []
    for a in la:
        best = None
        for b in lb:
            x0, x1 = max(a.x_range[0], b.x_range[0]), min(a.x_range[1], b.x_range[1])
            if x1 - x0 < 1.0 * scale:
                continue
            xs = np.linspace(x0, x1, 20)
            g = float(np.mean(np.abs(a.y_at(xs) - b.y_at(xs))))
            if g <= match_m and (best is None or g < best):
                best = g
        if best is not None:
            gaps.append(best)
    return {"lanes_a": len(la), "lanes_b": len(lb), "lane_pairs": len(gaps),
            "lane_dy_m": float(np.mean(gaps)) if gaps else None}


def overlap_score(cameras: list[Camera], images: list[np.ndarray], grid: BevGrid, i: int, j: int,
                  scale: float = 1.0, near_m: float = 12.0, pass_m: float = 0.15,
                  occluders: list[np.ndarray | None] | None = None, lane_cfg: LaneConfig | None = None,
                  zncc_min: float = 0.5) -> dict:
    """카메라 i, j의 겹친 바닥이 맞물리는지 점수(dict). 겹친 곳이 없으면 cells=0, ok=None."""
    ca, cb = cameras[i], cameras[j]
    ia, ib = IpmCompositor([ca], grid), IpmCompositor([cb], grid)
    mask = overlap_mask(ca, cb, grid, ia, ib, near_m * scale)
    out = {"pair": [ca.name, cb.name], "cells": int(mask.sum()), "area_m2": float(mask.sum() * grid.res ** 2),
           "zncc": None, "matches": 0, "shift_m": None, "shift_xy_m": None, "lane_pairs": 0, "lane_dy_m": None,
           "pass_m": pass_m * scale, "ok": None}
    if out["cells"] < 200:
        return out
    wa, wb = ia.warp([images[i]]), ib.warp([images[j]])
    if occluders is not None:
        for w, ipm, oc in ((wa, ia, occluders[i]), (wb, ib, occluders[j])):
            if oc is not None:
                mask &= ipm.warp([oc], interpolation=cv2.INTER_NEAREST) == 0
    ga = cv2.cvtColor(wa, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    # 밝기 상관(겹친 곳만)
    a, b = ga[mask].astype(np.float64), gb[mask].astype(np.float64)
    if a.std() > 1e-6 and b.std() > 1e-6:
        out["zncc"] = float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))
    out.update(_feature_shift(ga, gb, mask, grid.res, 3.0 * scale))
    out.update(_lane_gap(grid, wa, wb, mask, scale, lane_cfg, 1.0 * scale))
    # 판정: 특징점이나 차선으로 잰 어긋남이 있으면 그것으로, 둘 다 못 재면 밝기 상관이 zncc_min 미만일 때만 불합격
    vals = [v for v in (out["shift_m"], out["lane_dy_m"]) if v is not None]
    if vals:
        out["ok"] = max(vals) <= out["pass_m"]
    elif out["zncc"] is not None and out["zncc"] < zncc_min and out["area_m2"] >= 5.0 * scale * scale:
        # 겹친 바닥이 너무 좁으면(5m2 x 축척^2 미만) 밝기 상관 하나로는 판정하지 않는다
        out["ok"] = False
    return out
