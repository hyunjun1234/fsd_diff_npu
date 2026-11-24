from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO 클래스 중 주행 장면에 필요한 것만 사용
COCO_KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# COCO 80개 클래스 전체 이름(번호 = 목록 위치). 검출 확인 스크립트처럼 모든 클래스를 볼 때 쓴다.
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)
COCO_ALL = dict(enumerate(COCO_NAMES))


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
