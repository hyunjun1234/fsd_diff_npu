from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import Detection

# 두 검출기(예: Hailo-8과 DX-M1)의 결과를 같은 영상에서 비교하는 함수 모음.
# 하드웨어 없이 맥에서도 시험할 수 있게 계산만 담는다. 실행 스크립트는 scripts/compare_det.py.


# ---------------------------------------------------------------------------
# 상자 하나끼리의 비교 값
# ---------------------------------------------------------------------------
def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """두 상자(x1, y1, x2, y2)의 IoU. 겹치지 않으면 0."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """두 상자 중심 사이 거리(픽셀)."""
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


# ---------------------------------------------------------------------------
# 한 영상 안에서 짝 맞추기
# ---------------------------------------------------------------------------
def match_detections(
    ref: Sequence[Detection], other: Sequence[Detection], iou_thr: float = 0.5
) -> list[tuple[int, int, float, float]]:
    """같은 클래스끼리 IoU가 큰 순서대로 한 번씩만 짝을 짓는다(탐욕 매칭).

    돌려주는 값: (ref 번호, other 번호, IoU, 중심 거리 px) 목록. IoU가 iou_thr 미만이면 짝이 아니다.
    """
    pairs = []
    for i, a in enumerate(ref):
        for j, b in enumerate(other):
            if a.label != b.label:
                continue
            v = iou_xyxy(a.box, b.box)
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(key=lambda t: -t[0])  # IoU 큰 것부터
    used_i, used_j, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, v, center_distance(ref[i].box, other[j].box)))
    return out


# ---------------------------------------------------------------------------
# 여러 영상을 모은 요약
# ---------------------------------------------------------------------------
def summarize_pairs(
    ref_frames: Sequence[Sequence[Detection]],
    other_frames: Sequence[Sequence[Detection]],
    iou_thr: float = 0.5,
) -> dict:
    """영상마다의 검출 목록 두 벌을 비교해 수를 센다.

    ref_frames[k]와 other_frames[k]는 같은 영상(같은 카메라, 같은 프레임)의 검출이어야 한다.
    """
    if len(ref_frames) != len(other_frames):
        raise ValueError(f"영상 수가 다릅니다: {len(ref_frames)} 대 {len(other_frames)}")
    n_ref = n_other = 0
    ious: list[float] = []
    dists: list[float] = []
    for a, b in zip(ref_frames, other_frames):
        n_ref += len(a)
        n_other += len(b)
        for _, _, v, d in match_detections(a, b, iou_thr):
            ious.append(v)
            dists.append(d)
    n_img = len(ref_frames)
    return {
        "images": n_img,
        "ref_total": n_ref,
        "other_total": n_other,
        "ref_per_image": round(n_ref / n_img, 3) if n_img else 0.0,
        "other_per_image": round(n_other / n_img, 3) if n_img else 0.0,
        "matched": len(ious),
        "ref_unmatched": n_ref - len(ious),
        "other_unmatched": n_other - len(ious),
        "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
        "iou_min": round(float(np.min(ious)), 4) if ious else None,
        "center_px_mean": round(float(np.mean(dists)), 2) if dists else None,
        "center_px_max": round(float(np.max(dists)), 2) if dists else None,
        "iou_thr": iou_thr,
    }
