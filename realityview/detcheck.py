from __future__ import annotations

import cv2
import numpy as np

from .types import Detection

# 검출 결과가 원본 영상 좌표에 맞게 나오는지 사람이 눈으로, 그리고 숫자로 확인하는 도구.
# scripts/check_hailo_det.py가 쓴다.


# ---------------------------------------------------------------------------
# 박스 계산
# ---------------------------------------------------------------------------
def box_iou(a, b) -> float:
    """두 박스 (x1, y1, x2, y2)의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def unmirror_box(box, width: int) -> tuple[float, float, float, float]:
    """좌우로 뒤집은 영상(폭 width)에서 나온 박스를 원래 영상 좌표로 되돌린다."""
    x1, y1, x2, y2 = box
    return (width - x2, y1, width - x1, y2)


def flip_consistency(dets: list[Detection], dets_flipped: list[Detection], width: int) -> list[float]:
    """원본 검출과 좌우 반전 영상 검출을 짝지어 IoU 목록을 돌려준다.

    왜 하는가: parse_hailo_nms가 x와 y 순서를 바꿔 읽고 있다면, 영상을 좌우로 뒤집었을 때
    박스가 위아래 방향으로 움직이므로 되돌린 박스가 원본 박스와 겹치지 않는다(IoU가 낮다).
    순서가 맞다면 같은 물체의 박스가 거의 그대로 겹친다(IoU가 높다).
    같은 이름끼리 IoU가 큰 순서로 한 번씩만 짝짓는다.
    """
    back = [Detection(d.label, d.score, unmirror_box(d.box, width)) for d in dets_flipped]
    pairs = []
    for i, a in enumerate(dets):
        for j, b in enumerate(back):
            if a.label == b.label:
                pairs.append((box_iou(a.box, b.box), i, j))
    pairs.sort(reverse=True)
    used_a, used_b, ious = set(), set(), []
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ious.append(iou)
    return ious


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------
def draw_detections(img: np.ndarray, dets: list[Detection], title: str = "") -> np.ndarray:
    """원본 영상 위에 박스와 이름, 점수를 그린다(원본은 건드리지 않음)."""
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{d.label} {d.score:.2f}", (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            cv2.LINE_AA,
        )
    if title:
        cv2.putText(out, title, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out
