from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import Camera
from .types import Detection, FrameResult

# 하드웨어 없이 파이프라인 전체를 확인하기 위한 간단한 합성 장면.
# 3차로 도로와 주변 차량/보행자를 차량 좌표계에서 움직이고, 각 카메라 영상을 광선-지면 교차로 그린다.
# SimBackend는 정답 박스에 잡음을 섞어 '검출기 출력'처럼 돌려준다.
# ForcedMaskSeg는 세그 출력 마스크를 무작위 마스크로 바꾸는 렌더링 비용 시험용 감싸개다.

LANE_W = 3.5
ROAD_HALF = 1.5 * LANE_W  # 3차로
SKY = (200, 170, 140)


@dataclass
class SimObject:
    label: str
    x: float
    y: float
    vx: float  # 자차 대비 상대속도 (m/s)
    vy: float
    length: float
    width: float
    height: float
    color: tuple[int, int, int]


def default_objects() -> list[SimObject]:
    return [
        SimObject("car", 12.0, 0.0, -1.0, 0.0, 4.5, 1.9, 1.5, (60, 60, 200)),
        SimObject("car", 6.0, LANE_W, 2.0, 0.0, 4.6, 1.9, 1.5, (200, 200, 200)),
        SimObject("truck", 22.0, -LANE_W, -0.5, 0.0, 8.0, 2.5, 3.0, (40, 140, 220)),
        SimObject("car", -9.0, 0.0, 1.2, 0.0, 4.4, 1.8, 1.5, (50, 50, 50)),
        SimObject("car", -14.0, -LANE_W, 0.4, 0.0, 4.6, 1.9, 1.5, (180, 120, 60)),
        SimObject("person", 9.0, ROAD_HALF + 1.5, -12.0, 0.0, 0.5, 0.5, 1.7, (40, 180, 60)),
    ]


class SimWorld:
    def __init__(self, cameras: list[Camera], objects: list[SimObject] | None = None, ego_speed: float = 12.0):
        self.cameras = cameras
        self.objects = objects if objects is not None else default_objects()
        self.ego_speed = ego_speed
        self.s = 0.0  # 자차 누적 이동거리 (차선 점선 스크롤용)
        self.t = 0.0
        self.gt: list[list[Detection]] = []
        # gt와 같은 모양으로, 각 정답 박스가 objects의 몇 번째 물체인지 적는다(추적 평가용).
        # SimBackend는 이 값을 쓰지 않으므로 잡음 난수의 순서는 예전과 같다
        self.gt_ids: list[list[int]] = []
        # 물체가 반대편으로 다시 등장한 횟수(물체마다). 다시 등장하면 평가에서 다른 물체로 본다
        self.wraps: list[int] = [0] * len(self.objects)
        self.gt_masks: list[np.ndarray] = []
        self._rays = []
        for cam in cameras:
            u, v = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
            uv = np.stack([u.ravel(), v.ravel()], 1)
            xy, valid = cam.pixel_to_ground(uv, max_range=150.0)
            self._rays.append((xy.reshape(cam.height, cam.width, 2), valid.reshape(cam.height, cam.width)))

    def step(self, dt: float) -> None:
        self.t += dt
        self.s += self.ego_speed * dt
        for i, o in enumerate(self.objects):
            o.x += o.vx * dt
            o.y += o.vy * dt
            # 너무 멀어지면 반대편에서 다시 등장
            if o.x > 45:
                o.x = -35
                self.wraps[i] += 1
            elif o.x < -35:
                o.x = 45
                self.wraps[i] += 1

    # ---- 렌더링 -------------------------------------------------------
    def _ground_color(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        H, W = X.shape
        img = np.empty((H, W, 3), np.uint8)
        img[:] = (70, 110, 80)  # 풀밭
        absY = np.abs(Y)
        side = (absY > ROAD_HALF) & (absY < ROAD_HALF + 3.0)
        img[side] = (150, 150, 150)  # 보도
        road = absY <= ROAD_HALF
        img[road] = (70, 70, 72)
        Xs = X + self.s
        dash = (np.mod(Xs, 9.0) < 4.0)
        for yl in (-LANE_W / 2, LANE_W / 2):
            m = (np.abs(Y - yl) < 0.08) & dash
            img[m] = (235, 235, 235)
        edge = (np.abs(absY - ROAD_HALF) < 0.1)
        img[edge] = (60, 200, 230)  # 가장자리 실선(노란색)
        # 아스팔트 질감(IPM 확인용)
        noise = ((np.floor(Xs * 2) * 7 + np.floor(Y * 2) * 13) % 5).astype(np.int16) * 3
        img[road] = np.clip(img[road].astype(np.int16) + noise[road, None], 0, 255).astype(np.uint8)
        return img

    def render(self) -> list[np.ndarray]:
        frames, self.gt, self.gt_ids, self.gt_masks = [], [], [], []
        for cam, (xy, valid) in zip(self.cameras, self._rays):
            img = self._ground_color(xy[..., 0], xy[..., 1])
            img[~valid] = SKY
            occl = np.zeros(img.shape[:2], np.uint8)
            dets, ids = [], []
            # 먼 물체부터 그려서 가까운 물체가 위에 덮이게 한다(번호는 objects 순서 그대로 기억)
            order = sorted(
                range(len(self.objects)),
                key=lambda i: -np.hypot(self.objects[i].x - cam.position[0], self.objects[i].y - cam.position[1]),
            )
            for i in order:
                o = self.objects[i]
                box = self._draw_object(img, cam, o, occl)
                if box is not None:
                    dets.append(Detection(o.label, 0.9, box))
                    ids.append(i)
            road = valid & (np.abs(xy[..., 1]) <= ROAD_HALF) & (occl == 0)
            self.gt_masks.append(road.astype(np.uint8))
            frames.append(img)
            self.gt.append(dets)
            self.gt_ids.append(ids)
        return frames

    def _draw_object(self, img: np.ndarray, cam: Camera, o: SimObject, occl: np.ndarray):
        L, W, Hh = o.length / 2, o.width / 2, o.height
        base = np.array([[L, W], [L, -W], [-L, -W], [-L, W]]) + [o.x, o.y]
        corners = np.vstack([np.c_[base, np.zeros(4)], np.c_[base, np.full(4, Hh)]])
        uv, front = cam.project(corners)
        if not front.all():
            return None
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        if x2 < 0 or y2 < 0 or x1 >= cam.width or y1 >= cam.height:
            return None
        pts = uv.round().astype(np.int32)
        faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]
        shade = [0.8, 0.65, 0.55, 0.7, 1.0]
        centers = corners.reshape(8, 3)
        fd = [np.linalg.norm(centers[f].mean(0) - cam.position) for f in faces]
        for k in np.argsort(fd)[::-1]:
            col = tuple(int(c * shade[k]) for c in o.color)
            cv2.fillConvexPoly(img, pts[faces[k]], col, cv2.LINE_AA)
            cv2.fillConvexPoly(occl, pts[faces[k]], 1)
        bx = (max(0.0, x1), max(0.0, y1), min(cam.width - 1.0, x2), min(cam.height - 1.0, y2))
        if (bx[2] - bx[0]) * (bx[3] - bx[1]) < 60:
            return None
        if y2 > cam.height - 1:  # 접지점이 화면 밖이면 검출기도 거리 추정 불가 → 제외
            return None
        return bx


class SimBackend:
    """SimWorld의 정답을 '검출기 출력'처럼 돌려준다(픽셀 잡음, 가끔 놓침)."""

    def __init__(self, world: SimWorld, noise_px: float = 2.0, miss_rate: float = 0.05, seed: int = 0):
        self.world = world
        self.noise_px = noise_px
        self.miss_rate = miss_rate
        self.rng = np.random.default_rng(seed)

    def infer(self, frames) -> list[FrameResult]:
        out = []
        for dets, mask in zip(self.world.gt, self.world.gt_masks):
            noisy = []
            for d in dets:
                if self.rng.random() < self.miss_rate:
                    continue
                b = np.array(d.box) + self.rng.normal(0, self.noise_px, 4)
                noisy.append(Detection(d.label, float(self.rng.uniform(0.6, 0.95)), tuple(b.tolist())))
            out.append(FrameResult(detections=noisy, road_mask=mask))
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 렌더링 비용 시험용: 세그 출력 마스크를 무작위 마스크로 바꾸는 감싸개
# ---------------------------------------------------------------------------
class ForcedMaskSeg:
    """세그 백엔드를 감싸서, NPU 추론은 그대로 돌리고 돌려줄 마스크만 무작위 마스크로 바꾼다.

    진짜 세그 모델이 없을 때(드론 모델을 세그 자리에 넣은 맛보기) 마스크 칠하기 비용을
    정해진 도로 비율로 재려고 쓴다. run_demo.py의 --seg-mask-test random50 이 이것을 켠다.

    - 마스크는 화소마다 fraction 확률로 1이 되는 무작위 0/1(uint8)이다(같은 seed면 늘 같다).
    - 매번 새로 만들면 세그 스레드의 CPU를 잡아먹어 측정을 흐리므로, 카메라 크기마다 pool 장을
      미리 만들어 두고 호출마다 돌려 가며 준다. 이웃한 두 호출은 늘 다른 배열을 받으므로
      파이프라인 입장에서는 "세그가 새 마스크를 냈다"와 같다.
    - 돌려주는 배열은 여러 번 다시 쓰이므로 받는 쪽에서 고치면 안 된다(파이프라인은 읽기만 한다).
    """

    def __init__(self, inner, fraction: float = 0.5, pool: int = 2, seed: int = 0):
        self.inner = inner  # 진짜 세그 백엔드(예: DXM1Backend). segment()와 close()만 쓴다
        self.fraction = float(fraction)
        self.pool = max(2, int(pool))
        self._rng = np.random.default_rng(seed)
        self._masks: dict[tuple[int, int], list[np.ndarray]] = {}  # (h, w) → 미리 만든 마스크들
        self.calls = 0

    def _pool_for(self, hw: tuple[int, int]) -> list[np.ndarray]:
        if hw not in self._masks:
            self._masks[hw] = [(self._rng.random(hw) < self.fraction).astype(np.uint8) for _ in range(self.pool)]
        return self._masks[hw]

    def segment(self, frames) -> list[np.ndarray]:
        # 1) NPU 추론과 그 후처리는 원래대로 돌린다(NPU 부하와 세그 스레드 CPU 사용을 그대로 두려고)
        self.inner.segment(frames)
        # 2) 결과는 버리고 미리 만든 무작위 마스크를 돌려준다
        k = self.calls % self.pool
        self.calls += 1
        return [self._pool_for(tuple(f.shape[:2]))[k] for f in frames]

    def close(self) -> None:
        self.inner.close()
