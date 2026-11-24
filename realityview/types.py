from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO 클래스 중 주행 장면에 필요한 것만 사용
COCO_KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    """한 카메라 영상 위의 검출 결과. box는 픽셀 (x1, y1, x2, y2)."""

    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass
class FrameResult:
    detections: list[Detection] = field(default_factory=list)
    road_mask: np.ndarray | None = None  # HxW uint8, 1=주행 가능 영역


@dataclass
class GroundObject:
    label: str
    score: float
    xy: np.ndarray  # 차량 좌표 (X, Y) m
    size: tuple[float, float]  # (길이, 폭) m
    cam_index: int
    distance: float = 0.0
