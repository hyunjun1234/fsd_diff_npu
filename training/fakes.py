from __future__ import annotations

import json
import os

import cv2
import numpy as np

from training.common import MASK_BG, MASK_ROAD, Box

# 큰 모델 없이 파이프라인 전체를 시험하기 위한 가짜 라벨러, 학습기, 예측기.
# 합성 영상(training/tests/synth.py)은 차를 파란 사각형, 사람을 빨간 사각형, 도로를 회색 바닥으로 그린다.
# 가짜 라벨러는 그 색을 찾아 박스와 도로 마스크를 만든다. 실제 서버에서도 --fake 로 흐름만 점검할 수 있다.


# ---------------------------------------------------------------------------
# 가짜 라벨러: 색으로 물체 찾기
# ---------------------------------------------------------------------------
class FakeColorLabeler:
    """파랑이 강한 덩어리는 차(COCO 2), 빨강이 강한 덩어리는 사람(COCO 0), 회색 아래쪽은 도로.

    drop_classes 의 클래스는 일부러 놓쳐서 두 라벨러가 어긋나는 상황을 만든다.
    ambiguous_score 를 주면 사람 박스 점수를 그 값으로 낮춰 애매한 박스를 만든다.
    """

    def __init__(self, name: str = "fake", score: float = 0.9, drop_classes=(), ambiguous_score: float | None = None,
                 with_road: bool = True):
        self.name = name
        self.score = score
        self.drop = set(drop_classes)
        self.amb = ambiguous_score
        self.with_road = with_road

    def label(self, img: np.ndarray, path: str | None = None):
        from training.autolabel import LabelResult  # 순환 import 를 피하려고 여기서

        b, g, r = [img[:, :, i].astype(np.int16) for i in range(3)]
        boxes = []
        for cls, mask in ((2, (b > r + 40) & (b > g + 40)), (0, (r > b + 40) & (r > g + 40))):
            if cls in self.drop:
                continue
            n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = st[i]
                if area < 30:
                    continue
                s = self.amb if (self.amb is not None and cls == 0) else self.score
                boxes.append(Box(cls, float(x), float(y), float(x + w), float(y + h), float(s)))
        road = None
        if self.with_road:
            gray_like = (np.abs(b - g) < 12) & (np.abs(g - r) < 12) & (g > 15) & (g < 220)
            road = np.full(img.shape[:2], MASK_BG, np.uint8)
            road[gray_like] = MASK_ROAD
            road[: img.shape[0] // 2] = MASK_BG  # 도로는 아래 절반에만 있다고 본다
        return LabelResult(boxes=boxes, road=road)


# ---------------------------------------------------------------------------
# 가짜 학습기와 예측기
# ---------------------------------------------------------------------------
class FakeTrainer:
    """학습 대신 학습 목록을 읽어 개수만 세고 best.pt 자리에 json 을 쓴다."""

    name = "fake"

    def train(self, data_yaml: str, weights: str, out_dir: str, params: dict) -> str:
        import yaml

        with open(data_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        n_train = sum(1 for _ in open(data["train"], encoding="utf-8"))
        os.makedirs(out_dir, exist_ok=True)
        best = os.path.join(out_dir, "best.pt")
        with open(best, "w", encoding="utf-8") as f:
            json.dump({"fake": True, "start": weights, "n_train": n_train, "params": params}, f)
        return best


class FakePredictor:
    """가짜 라벨러를 그대로 예측기로 쓴다. miss_classes 로 일부러 성능을 떨어뜨린 '나쁜 회차'를 만든다."""

    def __init__(self, miss_classes=(), score: float = 0.9):
        self.lab = FakeColorLabeler("pred", score=score, drop_classes=miss_classes, with_road=False)

    def predict(self, img: np.ndarray, path: str | None = None) -> list[Box]:
        return self.lab.label(img, path).boxes
