from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .camera_model import Camera
from .types import Detection

# 단안 깊이 모델 출력 다루기.
# DEEPX 모델 저장소 2_4_0의 깊이 모델(yolo26-depth, scdepthv3, fastdepth)은 출력이 (1,1,H,W) float32
# 하나이고, DEEPX dx_app v3.2.2 예제(FastDepthPostprocessor)는 이 값을 최솟값~최댓값으로 0~1에 펴서
# 보여 주기만 한다. 값이 미터인지, 깊이(클수록 멂)인지 역깊이(클수록 가까움)인지는 문서에 없어서
# 모델마다 kind 옵션으로 정하고, guess_kind()로 바닥 쪽 값의 흐름을 보고 확인한다.
#   kind="depth"  : 클수록 멀다. 축척만 맞추면 된다(metric = s * raw).
#   kind="inverse": 클수록 가깝다. 역깊이 공간에서 축척과 이동을 맞춘다(1 / metric = a * raw + b).
DEPTH_KINDS = ("depth", "inverse")


# ---------------------------------------------------------------------------
# 출력 모양 정리
# ---------------------------------------------------------------------------
def depth_from_output(out: np.ndarray) -> np.ndarray:
    """(1,1,H,W), (1,H,W), (H,W) 출력을 (H,W) float32로."""
    d = np.asarray(out, dtype=np.float32)
    while d.ndim > 2 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 2:
        raise ValueError(f"깊이 출력은 (H,W)로 줄일 수 있어야 합니다. 받은 모양: {np.shape(out)}")
    return d


def resize_depth(depth: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """깊이 맵을 원본 영상 크기로(선형 보간). 입력은 단순 리사이즈로 넣었으므로 그대로 늘리면 화소가 맞는다."""
    if depth.shape == tuple(out_hw):
        return depth
    return cv2.resize(depth, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)


def _check_kind(kind: str) -> None:
    if kind not in DEPTH_KINDS:
        raise ValueError(f"kind는 {DEPTH_KINDS} 중 하나여야 합니다: {kind}")


# ---------------------------------------------------------------------------
# 컬러 칠하기 (눈으로 확인용)
# ---------------------------------------------------------------------------
def colorize_depth(depth: np.ndarray, kind: str = "depth", lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """깊이 맵을 BGR 컬러 영상으로. 가까운 곳이 밝고(노랑) 먼 곳이 어둡게(보라, 검정) 칠한다.

    양 끝 2%는 잘라서 튀는 값 몇 개 때문에 전체가 한 색이 되지 않게 한다.
    """
    _check_kind(kind)
    d = np.asarray(depth, dtype=np.float32)
    near = -d if kind == "depth" else d  # 클수록 가깝게 바꾼 값
    finite = np.isfinite(near)
    if not finite.any():
        return np.zeros(d.shape + (3,), np.uint8)
    lo, hi = np.percentile(near[finite], [lo_pct, hi_pct])
    span = max(float(hi - lo), 1e-6)
    n = np.clip((np.where(finite, near, lo) - lo) / span, 0.0, 1.0)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------------------
# 깊이인지 역깊이인지 보기
# ---------------------------------------------------------------------------
def guess_kind(depth: np.ndarray, rows: Sequence[float] = (0.95, 0.85, 0.75, 0.65)) -> tuple[str, list[float]]:
    """영상 아래쪽 가운데 띠의 값을 아래에서 위로 읽어 kind를 추정한다.

    차나 방 바닥처럼 아래쪽이 지면인 장면이면, 영상 위로 갈수록 지면이 멀어진다.
    값이 위로 갈수록 커지면 "depth", 작아지면 "inverse". 아래쪽에 물체가 가까이 있으면 틀릴 수 있어
    추정일 뿐이고, 결정은 설정(kind)으로 한다. 돌려주는 값은 (추정, 줄마다 중앙값).
    """
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    meds = []
    for r in rows:
        y = int(np.clip(r * h, 0, h - 1))
        band = d[max(0, y - h // 40) : y + h // 40 + 1, w // 3 : 2 * w // 3]
        meds.append(float(np.nanmedian(band)))
    slope = np.polyfit(np.arange(len(meds)), meds, 1)[0]  # 위로 갈수록의 기울기
    return ("depth" if slope > 0 else "inverse"), meds


# ---------------------------------------------------------------------------
# 검출 박스 아래쪽의 깊이 값
# ---------------------------------------------------------------------------
def box_bottom_depth(
    depth: np.ndarray, box: Sequence[float], strip_frac: float = 0.08, inset_frac: float = 0.02, min_px: int = 3
) -> float:
    """박스 아래 가장자리 바로 안쪽 띠의 중앙값(모델 값 그대로).

    가장자리 화소는 배경(지면)과 물체가 섞이므로 박스 높이의 inset_frac만큼 안으로 들어간 뒤,
    strip_frac 높이, 가운데 절반 폭의 띠에서 중앙값을 낸다. 영상 밖이면 nan.
    """
    d = np.asarray(depth)
    h, w = d.shape
    x1, y1, x2, y2 = (float(v) for v in box)
    bh = max(y2 - y1, 1.0)
    bot = int(round(min(y2, h) - inset_frac * bh))
    top = int(round(bot - max(strip_frac * bh, min_px)))
    cx, half = (x1 + x2) / 2.0, max((x2 - x1) / 4.0, min_px / 2.0)
    left, right = int(round(max(cx - half, 0))), int(round(min(cx + half, w)))
    top, bot = max(top, 0), min(max(bot, top + 1), h)
    if right <= left or bot <= top:
        return float("nan")
    v = d[top:bot, left:right]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


# ---------------------------------------------------------------------------
# 지면 평면(IPM) 기준 거리
# ---------------------------------------------------------------------------
def ground_depth(cam: Camera, uv: np.ndarray, max_range: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """픽셀이 가리키는 지면 점까지의 카메라 광축 방향 거리 z(m)와 직선거리(m).

    깊이 모델은 보통 광축 방향 거리(z)를 내므로 비교는 z로 한다. 지면과 안 만나면 nan.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xy, valid = cam.pixel_to_ground(uv, max_range=max_range)
    p = np.hstack([xy, np.zeros((len(xy), 1))])
    p_cam = (cam.R_cv @ (p - cam.position).T).T
    z = np.where(valid, p_cam[:, 2], np.nan)
    rng = np.where(valid, np.linalg.norm(p_cam, axis=1), np.nan)
    return z, rng


def ground_samples(
    cam: Camera,
    depth: np.ndarray,
    rows: tuple[float, float] = (0.6, 0.98),
    step: int = 16,
    mask: np.ndarray | None = None,
    max_range: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """축척 맞추기용 (모델 값, 지면 z) 쌍. 영상 아래쪽 격자점 중 지면과 만나는 점만 쓴다.

    mask(도로 마스크 등, 원본 크기 0/1)를 주면 그 안의 점만 쓴다. 물체가 덮은 곳을 빼려면 이렇게 한다.
    """
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    ys = np.arange(int(rows[0] * h), int(rows[1] * h), step)
    xs = np.arange(step // 2, w, step)
    gx, gy = np.meshgrid(xs, ys)
    uv = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    keep = np.ones(len(uv), bool)
    if mask is not None:
        keep &= np.asarray(mask)[uv[:, 1].astype(int), uv[:, 0].astype(int)] > 0
    # 깊이 맵 크기와 카메라 해상도가 다르면 픽셀 좌표를 카메라 해상도로 옮긴다
    uv_cam = uv * [cam.width / w, cam.height / h]
    z, _ = ground_depth(cam, uv_cam, max_range=max_range)
    raw = d[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    keep &= np.isfinite(z) & np.isfinite(raw) & (z > 0)
    return raw[keep], z[keep]


# ---------------------------------------------------------------------------
# 축척 맞추기 (상대 깊이 -> 미터)
# ---------------------------------------------------------------------------
@dataclass
class DepthScale:
    """모델 값을 미터 z로 바꾸는 식. kind="depth"면 a*raw, "inverse"면 1/(a*raw+b)."""

    kind: str
    a: float
    b: float = 0.0
    n: int = 0  # 맞출 때 쓴 점 수

    def apply(self, raw: np.ndarray | float) -> np.ndarray:
        r = np.asarray(raw, dtype=np.float64)
        if self.kind == "depth":
            return self.a * r + self.b
        inv = self.a * r + self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(inv > 1e-9, 1.0 / inv, np.nan)


def fit_depth_scale(raw: np.ndarray, ref_z: np.ndarray, kind: str = "depth") -> DepthScale:
    """지면 점의 (모델 값, IPM z) 쌍으로 축척을 맞춘다.

    - depth: 비율 ref/raw의 중앙값 하나(튀는 점에 강하다).
    - inverse: 1/ref = a*raw + b를 최소제곱으로 풀고, 잔차가 중앙값 절대편차의 3배를 넘는 점을 한 번 빼고 다시 푼다.
    점이 모자라면 ValueError.
    """
    _check_kind(kind)
    raw = np.asarray(raw, dtype=np.float64).ravel()
    ref = np.asarray(ref_z, dtype=np.float64).ravel()
    ok = np.isfinite(raw) & np.isfinite(ref) & (ref > 0)
    if kind == "depth":
        ok &= raw > 0
        if ok.sum() < 3:
            raise ValueError(f"축척을 맞출 지면 점이 모자랍니다: {int(ok.sum())}개")
        return DepthScale("depth", float(np.median(ref[ok] / raw[ok])), 0.0, int(ok.sum()))
    if ok.sum() < 4:
        raise ValueError(f"축척을 맞출 지면 점이 모자랍니다: {int(ok.sum())}개")
    x, y = raw[ok], 1.0 / ref[ok]
    # 1) 첫 직선은 Theil-Sen(두 점씩 이은 기울기의 중앙값)으로 잡는다. 지면 위 물체처럼 튀는 점이
    #    30% 가까이 섞여도 흔들리지 않는다. 점이 많으면 고르게 400개만 골라 계산량을 줄인다.
    idx = np.linspace(0, len(x) - 1, min(len(x), 400)).astype(int)
    xs, ys = x[idx], y[idx]
    i, j = np.triu_indices(len(xs), 1)
    dx = xs[j] - xs[i]
    good = np.abs(dx) > 1e-12
    if not good.any():
        raise ValueError("모델 값이 모두 같아 축척을 맞출 수 없습니다")
    a = float(np.median((ys[j] - ys[i])[good] / dx[good]))
    b = float(np.median(y - a * x))
    # 2) 그 직선에서 멀리 떨어진 점(중앙값 절대편차의 3배 초과)을 빼고 최소제곱으로 다듬는다
    res = np.abs(y - (a * x + b))
    inl = res <= 3.0 * 1.4826 * np.median(res) + 1e-12
    if inl.sum() >= 4:
        a, b = np.polyfit(x[inl], y[inl], 1)
    return DepthScale("inverse", float(a), float(b), int(inl.sum()))


# ---------------------------------------------------------------------------
# 검출 박스: 깊이 모델 거리와 IPM 거리 비교
# ---------------------------------------------------------------------------
def compare_boxes_with_ipm(
    cam: Camera,
    depth: np.ndarray,
    dets: Sequence[Detection],
    scale: DepthScale | None = None,
    max_range: float = 80.0,
) -> list[dict]:
    """검출마다 박스 아래 가운데 점의 IPM 거리(z)와 같은 자리의 깊이 모델 값을 나란히 둔다.

    depth는 원본 영상 크기여야 한다(resize_depth). scale이 있으면 모델 값을 미터로 바꾼 값과 차이도 넣는다.
    보정 전에는 IPM 거리 자체가 의미 없으므로 이 표는 코드 확인용이다.
    """
    rows = []
    for det in dets:
        x1, y1, x2, y2 = det.box
        uv = np.array([[(x1 + x2) / 2.0, y2]])
        z, rng = ground_depth(cam, uv * [cam.width / depth.shape[1], cam.height / depth.shape[0]], max_range)
        raw = box_bottom_depth(depth, det.box)
        row = {
            "label": det.label,
            "score": round(float(det.score), 3),
            "box": [round(float(v), 1) for v in det.box],
            "ipm_z_m": None if not np.isfinite(z[0]) else round(float(z[0]), 3),
            "ipm_range_m": None if not np.isfinite(rng[0]) else round(float(rng[0]), 3),
            "depth_raw": None if not np.isfinite(raw) else round(raw, 4),
        }
        if scale is not None and np.isfinite(raw):
            m = float(scale.apply(raw))
            row["depth_m"] = round(m, 3) if np.isfinite(m) else None
            if row["depth_m"] is not None and row["ipm_z_m"] is not None:
                row["diff_m"] = round(row["depth_m"] - row["ipm_z_m"], 3)
        rows.append(row)
    return rows
