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
# 차선과 신호등 시험용 옵션(기본은 꺼져 있어 예전 장면과 똑같다)
#   curvature: 도로를 굽힌다. 자차 좌표에서 도로 가운데가 Y = curvature/2 x X^2 로 휜다(자차는 차로 가운데)
#   traffic_lights: 공중에 신호등(세로형, 가로형)을 띄운다. 정답 색은 gt_lights, 상자는 SimBackend가 검출처럼 준다

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


# ---------------------------------------------------------------------------
# 신호등(시험용)
# ---------------------------------------------------------------------------
# 켜진 등 색(BGR). 초록은 실제 신호등처럼 푸른 기가 있다
SIM_LAMP_BGR = {"red": (40, 40, 245), "yellow": (0, 190, 255), "green": (170, 255, 20)}
SIM_LAMP_OFF = (45, 45, 45)
SIM_HOUSING = (25, 25, 25)


@dataclass
class SimTrafficLight:
    """공중에 뜬 신호등 하나. 면이 -X(다가오는 자차) 쪽을 본다. 위치와 크기는 m."""

    x: float
    y: float
    z: float  # 상자 아래끝 높이
    states: tuple = ("red",)  # 차례로 바뀌는 색. 하나면 고정
    period: float = 2.0  # 색 하나가 이어지는 시간(초)
    orientation: str = "vertical"  # "vertical" 또는 "horizontal"
    lamp_m: float = 0.3  # 등 한 칸 크기

    def state_at(self, t: float) -> str:
        k = int(np.floor(t / max(self.period, 1e-6))) % len(self.states)
        return self.states[k]

    def size(self) -> tuple[float, float]:
        """(폭 Y, 높이 Z). 등 세 칸 + 테두리."""
        long, short = 3 * self.lamp_m + 0.1, self.lamp_m + 0.1
        return (short, long) if self.orientation == "vertical" else (long, short)


def default_traffic_lights() -> list[SimTrafficLight]:
    """앞쪽 교차로의 세로형, 가로형 신호등 두 개. 색은 2초마다 빨강, 초록, 노랑 순서로 바뀐다."""
    return [
        SimTrafficLight(18.0, 3.0, 4.0, ("red", "green", "yellow"), 2.0, "vertical"),
        SimTrafficLight(22.0, -2.5, 4.5, ("green", "yellow", "red"), 2.0, "horizontal"),
    ]


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
    def __init__(
        self,
        cameras: list[Camera],
        objects: list[SimObject] | None = None,
        ego_speed: float = 12.0,
        curvature: float = 0.0,  # 도로 곡률(1/m). 0이면 곧은 도로(예전과 같음)
        traffic_lights: list[SimTrafficLight] | None = None,  # 신호등. None이면 없음(예전과 같음)
    ):
        self.cameras = cameras
        self.curvature = float(curvature)
        self.traffic_lights = list(traffic_lights) if traffic_lights else []
        self.gt_lights: list[list[tuple]] = []  # 카메라별 (상자, 색, 모양) 목록(신호등이 있을 때)
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

    # ---- 도로 모양 -----------------------------------------------------
    def road_offset(self, X):
        """X에서 도로 가운데의 Y(곡률이 0이면 0)."""
        return 0.5 * self.curvature * np.asarray(X, dtype=np.float64) ** 2

    def gt_lanes(self) -> list[dict]:
        """정답 차선 목록. 곡선은 Y = a X^2 + c (a = curvature/2). 왼쪽(+Y)부터.

        차로 경계 점선 두 줄(흰색)과 도로 가장자리 실선 두 줄(노란색, 폭 0.2m의 가운데)이다.
        """
        a = 0.5 * self.curvature
        return [
            {"c": ROAD_HALF, "a": a, "style": "solid", "color": "yellow", "side": "other"},
            {"c": LANE_W / 2, "a": a, "style": "dashed", "color": "white", "side": "left"},
            {"c": -LANE_W / 2, "a": a, "style": "dashed", "color": "white", "side": "right"},
            {"c": -ROAD_HALF, "a": a, "style": "solid", "color": "yellow", "side": "other"},
        ]

    # ---- 렌더링 -------------------------------------------------------
    def _ground_color(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.curvature != 0.0:
            Y = Y - self.road_offset(X)  # 굽은 도로: 도로 가운데 기준 횡위치로 바꿔 같은 무늬를 칠한다
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

    # ---- 시각을 직접 정하기(카메라마다 다른 시각에 찍는 4대 시뮬레이션용) -------------
    def set_time(self, t: float) -> None:
        """물체와 자차를 시각 t(초)의 상태로 바로 옮긴다.

        step()은 dt만큼 차례로 움직이지만, 카메라마다 찍는 시각이 다른 장면을 만들려면 아무 시각으로나
        오갈 수 있어야 한다. 위치 = 처음 위치 + 속도 x t 이고, x가 -35~45 m를 벗어나면 80 m씩 돌려
        반대편에 다시 나온다(step()의 다시 등장과 같은 범위, 넘친 거리는 남긴다).
        처음 위치는 이 함수를 처음 부를 때의 위치다. 이 함수를 쓴 뒤 step()을 섞어 쓰지 않는다.
        """
        if not hasattr(self, "_t0_state"):
            self._t0_state = [(o.x, o.y) for o in self.objects]
        self.t = float(t)
        self.s = self.ego_speed * self.t
        for i, (o, (x0, y0)) in enumerate(zip(self.objects, self._t0_state)):
            x = x0 + o.vx * self.t
            k = int(np.floor((x + 35.0) / 80.0))
            o.x = x - 80.0 * k
            o.y = y0 + o.vy * self.t
            self.wraps[i] = abs(k)

    def render(self) -> list[np.ndarray]:
        frames, self.gt, self.gt_ids, self.gt_masks = [], [], [], []
        for k in range(len(self.cameras)):
            img, dets, ids, road = self.render_camera(k)
            self.gt_masks.append(road)
            frames.append(img)
            self.gt.append(dets)
            self.gt_ids.append(ids)
        return frames

    def render_camera(self, k: int, ground: bool = True):
        """카메라 k 한 대만 지금 상태로 그린다. (영상, 정답 상자, 물체 번호, 도로 마스크)를 돌려준다.

        ground=False면 바닥 무늬를 칠하지 않는다(정답 상자만 필요할 때 빠르게. 상자 값은 같다).
        """
        cam = self.cameras[k]
        xy, valid = self._rays[k]
        if ground:
            img = self._ground_color(xy[..., 0], xy[..., 1])
            img[~valid] = SKY
        else:
            img = np.empty((cam.height, cam.width, 3), np.uint8)
            img[:] = SKY
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
        if self.traffic_lights:
            self._draw_lights(img, k)
        ylat = xy[..., 1] if self.curvature == 0.0 else xy[..., 1] - self.road_offset(xy[..., 0])
        road = valid & (np.abs(ylat) <= ROAD_HALF) & (occl == 0)
        return img, dets, ids, road.astype(np.uint8)

    def _draw_lights(self, img: np.ndarray, k: int) -> None:
        """신호등을 카메라 k 영상에 그리고 정답(상자, 색, 모양)을 gt_lights[k]에 적는다."""
        cam = self.cameras[k]
        while len(self.gt_lights) <= k:
            self.gt_lights.append([])
        self.gt_lights[k] = []
        for tl in self.traffic_lights:
            if cam.position[0] >= tl.x:
                continue  # 신호등 면은 -X 쪽을 본다. 뒤에서는 등이 안 보인다
            w, h = tl.size()
            face = np.array([[tl.x, tl.y + w / 2, tl.z], [tl.x, tl.y - w / 2, tl.z],
                             [tl.x, tl.y - w / 2, tl.z + h], [tl.x, tl.y + w / 2, tl.z + h]])
            uv, front = cam.project(face)
            if not front.all():
                continue
            x1, y1 = uv.min(0)
            x2, y2 = uv.max(0)
            if x2 < 0 or y2 < 0 or x1 >= cam.width or y1 >= cam.height:
                continue
            cv2.fillConvexPoly(img, uv.round().astype(np.int32), SIM_HOUSING, cv2.LINE_AA)
            state = tl.state_at(self.t)
            for j, name in enumerate(("red", "yellow", "green")):
                # 세로형은 위에서부터, 가로형은 왼쪽(+Y)부터 빨강, 노랑, 초록
                off = (1 - j) * tl.lamp_m
                ctr = (np.array([tl.x, tl.y, tl.z + h / 2 + off]) if tl.orientation == "vertical"
                       else np.array([tl.x, tl.y + off, tl.z + h / 2]))
                edge = ctr + [0.0, 0.0, 0.4 * tl.lamp_m]
                p, ok = cam.project(np.stack([ctr, edge]))
                if not ok.all():
                    continue
                r = max(1, int(round(float(np.linalg.norm(p[1] - p[0])))))
                col = SIM_LAMP_BGR[name] if name == state else SIM_LAMP_OFF
                cv2.circle(img, tuple(int(round(v)) for v in p[0]), r, col, -1, cv2.LINE_AA)
            box = (max(0.0, x1), max(0.0, y1), min(cam.width - 1.0, x2), min(cam.height - 1.0, y2))
            if (box[2] - box[0]) * (box[3] - box[1]) >= 30:
                self.gt_lights[k].append((box, state, tl.orientation))

    def _draw_object(self, img: np.ndarray, cam: Camera, o: SimObject, occl: np.ndarray):
        L, W, Hh = o.length / 2, o.width / 2, o.height
        oy = o.y if self.curvature == 0.0 else o.y + float(self.road_offset(o.x))  # 굽은 도로면 차로를 따라 옮긴다
        base = np.array([[L, W], [L, -W], [-L, -W], [-L, W]]) + [o.x, oy]
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
        # 신호등 상자 잡음은 따로 뽑는다(신호등이 없을 때 기존 물체 잡음 순서가 예전과 똑같게)
        self.rng_lights = np.random.default_rng(seed + 1000)

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
        # 신호등(있을 때만): 정답 상자에 작은 잡음을 섞어 COCO traffic light 검출처럼 붙인다
        # (SimWorld 대신 정답만 흉내 낸 객체를 넘기는 곳도 있어 속성이 없으면 신호등이 없는 것으로 본다)
        has_lights = bool(getattr(self.world, "traffic_lights", None))
        for r, lights in zip(out, self.world.gt_lights if has_lights else []):
            for box, _state, _ori in lights:
                b = np.array(box) + self.rng_lights.normal(0, min(self.noise_px, 1.0), 4)
                r.detections.append(Detection("traffic light", 0.8, tuple(b.tolist())))
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 카메라별 촬영 시각 만들기(시각 어긋남, 흔들림, 프레임 누락)
# ---------------------------------------------------------------------------
def camera_timelines(
    n_cams: int,
    frames: int,
    fps: float,
    skew_ms: list[float],
    drop: list[float],
    jitter_ms: list[float],
    seed: int = 0,
) -> list[list[tuple[int, float]]]:
    """카메라마다 (시뮬레이션 프레임 번호, 찍은 시각 초) 목록을 만든다.

    카메라 k의 n번째 프레임 시각 = n / fps + skew_ms[k] + 정규분포 흔들림(jitter_ms[k]).
    drop[k] 확률로 그 프레임을 뺀다(USB 카메라가 프레임을 놓친 것처럼). 기준 카메라(0번)는 빼지 않는 편이
    비교하기 쉬우므로 drop[0]은 보통 0으로 둔다. 같은 seed면 늘 같은 결과다.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_cams):
        tl = []
        for n in range(frames):
            j = rng.normal(0.0, jitter_ms[k]) if jitter_ms[k] > 0 else 0.0
            dropped = drop[k] > 0 and rng.random() < drop[k]
            if not dropped:
                tl.append((n, n / fps + (skew_ms[k] + j) / 1e3))
        out.append(tl)
    return out


def per_camera(values: str | None, n: int, default: float = 0.0) -> list[float]:
    """'0,0,20,50' 같은 쉼표 문자열을 카메라 수만큼의 숫자 목록으로. 값 하나면 모든 카메라에 같게."""
    if not values:
        return [default] * n
    v = [float(x) for x in str(values).split(",") if x.strip()]
    if len(v) == 1:
        return v * n
    if len(v) != n:
        raise ValueError(f"카메라 {n}대에 값 {len(v)}개가 왔습니다: {values}")
    return v


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
