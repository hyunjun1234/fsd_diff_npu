from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera_model import Camera
from .types import Detection

# 신호등 색 판별: YOLOv8s가 찾은 COCO traffic light(클래스 9) 상자를 잘라 HSV로 빨강, 노랑, 초록, 모름을 고른다.
#
# 한 상자 판별 순서(classify_light)
#   1) 상자를 영상 안으로 자르고, 너무 작으면 "모름"
#   2) 밝고 채도 높은 화소(켜진 등)만 골라 색상(H)으로 빨강, 노랑, 초록 화소 수를 센다
#      빨강은 색상 원의 0도 근처라 0~9와 160~180 두 구간을 합친다
#   3) 한 색이 켜진 화소의 60% 이상이면 그 색
#   4) 두 색이 비슷하면(빨강과 노랑이 자주 섞인다) 켜진 등의 위치로 정한다
#      세로형은 위에서부터, 가로형은 왼쪽부터 빨강, 노랑, 초록 순서라고 본다
#   5) 켜진 화소가 너무 적으면 "모름"
# 프레임 사이에서는 LightTracker가 같은 신호등을 이어 붙이고 최근 N 프레임 다수결로 깜빡임을 줄인다.
# 신호등은 공중에 있어 지면 투영(박스 하단 = 바닥)이 맞지 않으므로 추적기와 탑뷰 물체에는 넣지 않는다.

TL_LABEL = "traffic light"
TL_COCO_ID = 9
STATES = ("red", "yellow", "green", "unknown")
# 화면 표시 색(BGR)
STATE_BGR = {"red": (60, 60, 240), "yellow": (30, 200, 250), "green": (120, 220, 60), "unknown": (130, 130, 130)}
# 지면에 투영하지 않는(공중에 있는) 클래스. 파이프라인이 추적기에 넣기 전에 뺀다
AERIAL_LABELS = (TL_LABEL,)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class TrafficLightConfig:
    """신호등 색 판별 설정. 색 값은 OpenCV HSV(H 0~180)."""

    v_min: int = 150  # 켜진 등: 밝기 하한
    s_min: int = 90  # 켜진 등: 채도 하한
    red_h: tuple = (9, 160)  # 빨강: H <= 9 또는 H >= 160
    yellow_h: tuple = (10, 34)  # 노랑(주황 포함)
    green_h: tuple = (35, 100)  # 초록(청록 포함. 신호등 초록은 푸른 기가 있다)
    min_px: int = 4  # 켜진 화소가 이보다 적으면 모름
    min_frac: float = 0.02  # 켜진 화소가 상자 넓이의 이 비율보다 적으면 모름
    dominant: float = 0.6  # 한 색이 켜진 화소의 이 비율 이상이면 그 색으로 정한다
    min_box_px: int = 6  # 상자 짧은 변이 이보다 작으면 모름(너무 멀다)
    aspect: float = 1.3  # 긴 변 / 짧은 변이 이 값 이상이면 세로형 또는 가로형, 아니면 모양 모름
    vote_frames: int = 5  # 다수결에 쓸 최근 프레임 수
    hold_frames: int = 3  # 못 찾아도 이 프레임 수까지 지난 상태를 유지
    match_frac: float = 1.0  # 같은 신호등으로 볼 상자 중심 거리(상자 긴 변 x 이 값)
    pole_m: float = 15.0  # 3D 화면에서 기둥을 세울 거리(실차 m). 거리 값은 보정 전이라 표시용이다


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
@dataclass
class TrafficLight:
    """한 신호등의 이번 프레임 상태."""

    cam_index: int
    box: tuple[float, float, float, float]  # 카메라 영상 픽셀 (x1, y1, x2, y2)
    raw_state: str  # 이번 프레임만 보고 판별한 색
    state: str  # 최근 프레임 다수결로 정한 색(화면에 쓰는 값)
    orientation: str  # "vertical", "horizontal", "square"
    score: float = 0.0  # 검출 점수
    tid: int = 0  # 신호등 번호(이어 붙인 같은 신호등은 같은 번호)
    xy: np.ndarray | None = None  # 3D 화면에 기둥을 세울 차량 좌표(m). 카메라 방향 x pole_m, 거리 의미 없음


def orientation_of(w: float, h: float, aspect: float = 1.3) -> str:
    """상자 모양으로 세로형, 가로형, 모름(정사각형에 가까움)을 정한다."""
    if h >= aspect * w:
        return "vertical"
    if w >= aspect * h:
        return "horizontal"
    return "square"


# ---------------------------------------------------------------------------
# 한 상자 판별
# ---------------------------------------------------------------------------
def classify_light(crop_bgr: np.ndarray, cfg: TrafficLightConfig | None = None,
                   orientation: str | None = None) -> str:
    """잘라 낸 신호등 영상 한 장의 색. "red", "yellow", "green", "unknown" 중 하나."""
    c = cfg or TrafficLightConfig()
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    h, w = crop_bgr.shape[:2]
    if min(h, w) < c.min_box_px:
        return "unknown"
    if orientation is None:
        orientation = orientation_of(w, h, c.aspect)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (V >= c.v_min) & (S >= c.s_min)
    masks = {
        "red": lit & ((H <= c.red_h[0]) | (H >= c.red_h[1])),
        "yellow": lit & (H >= c.yellow_h[0]) & (H <= c.yellow_h[1]),
        "green": lit & (H >= c.green_h[0]) & (H <= c.green_h[1]),
    }
    counts = {k: int(np.count_nonzero(m)) for k, m in masks.items()}
    total = sum(counts.values())
    if total < max(c.min_px, c.min_frac * h * w):
        return "unknown"
    ranked = sorted(counts, key=lambda k: -counts[k])
    if counts[ranked[0]] >= c.dominant * total:
        return ranked[0]
    # 두 색이 비슷하면 켜진 등의 위치(세 칸 중 켜진 화소가 가장 많은 칸)로 정한다
    pos = _lit_position(lit, orientation)
    if pos is not None and pos in ranked[:2]:
        return pos
    return "unknown"


def _lit_position(lit: np.ndarray, orientation: str) -> str | None:
    """긴 축을 세 칸으로 나눠 켜진 화소가 가장 많은 칸의 색(신호 순서 빨강, 노랑, 초록). 모양을 모르면 None."""
    if orientation == "vertical":
        prof = lit.sum(axis=1)  # 위에서 아래로
    elif orientation == "horizontal":
        prof = lit.sum(axis=0)  # 왼쪽에서 오른쪽으로
    else:
        return None
    thirds = np.array_split(prof, 3)
    k = int(np.argmax([t.sum() for t in thirds]))
    return ("red", "yellow", "green")[k]


def crop_box(img: np.ndarray, box) -> np.ndarray:
    """상자를 영상 안으로 잘라 낸 영역(보기, 복사 아님). 영상 밖이면 빈 배열."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


def light_xy(cam: Camera, box, dist: float) -> np.ndarray | None:
    """상자 가운데 방향으로 카메라에서 dist(m)만큼 떨어진 바닥 점(차량 좌표). 방향을 못 구하면 None.

    신호등의 실제 거리는 모르므로(공중에 있고 보정도 전) 방향만 쓰고 거리는 표시용 고정값이다.
    """
    u = (box[0] + box[2]) / 2.0
    v = (box[1] + box[3]) / 2.0
    d = cam.pixel_rays(np.array([[u, v]]))[0]
    if not np.all(np.isfinite(d)):
        return None
    dxy = d[:2]
    n = float(np.linalg.norm(dxy))
    if n < 1e-6:
        return None
    return cam.position[:2] + dxy / n * dist


# ---------------------------------------------------------------------------
# 프레임 간 다수결
# ---------------------------------------------------------------------------
@dataclass
class _LightTrack:
    tid: int
    cam_index: int
    center: np.ndarray
    size: float
    history: deque = field(default_factory=deque)
    miss: int = 0
    last: TrafficLight | None = None


class LightTracker:
    """카메라마다 상자 중심이 가까운 신호등끼리 이어 붙이고, 최근 vote_frames 프레임의 색을 다수결로 정한다."""

    def __init__(self, cfg: TrafficLightConfig | None = None):
        self.cfg = cfg or TrafficLightConfig()
        self.tracks: list[_LightTrack] = []
        self._next = 1

    def update(self, observed: list[TrafficLight]) -> list[TrafficLight]:
        c = self.cfg
        pairs = []
        for j, ob in enumerate(observed):
            ctr = np.array([(ob.box[0] + ob.box[2]) / 2, (ob.box[1] + ob.box[3]) / 2])
            size = max(ob.box[2] - ob.box[0], ob.box[3] - ob.box[1], 1.0)
            for i, t in enumerate(self.tracks):
                if t.cam_index != ob.cam_index:
                    continue
                d = float(np.linalg.norm(t.center - ctr))
                if d <= c.match_frac * max(size, t.size):
                    pairs.append((d, i, j, ctr, size))
        pairs.sort(key=lambda p: p[0])
        used_t, used_o = set(), set()
        out: list[TrafficLight] = []
        for d, i, j, ctr, size in pairs:
            if i in used_t or j in used_o:
                continue
            used_t.add(i)
            used_o.add(j)
            t = self.tracks[i]
            t.center, t.size, t.miss = ctr, size, 0
            out.append(self._vote(t, observed[j]))
        for j, ob in enumerate(observed):
            if j in used_o:
                continue
            ctr = np.array([(ob.box[0] + ob.box[2]) / 2, (ob.box[1] + ob.box[3]) / 2])
            size = max(ob.box[2] - ob.box[0], ob.box[3] - ob.box[1], 1.0)
            t = _LightTrack(self._next, ob.cam_index, ctr, size)
            self._next += 1
            self.tracks.append(t)
            used_t.add(len(self.tracks) - 1)
            out.append(self._vote(t, ob))
        # 못 찾은 신호등은 hold_frames까지 지난 결과를 그대로 보여 준다
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.miss += 1
                if t.miss <= c.hold_frames and t.last is not None:
                    out.append(t.last)
        self.tracks = [t for t in self.tracks if t.miss <= c.hold_frames]
        return out

    def _vote(self, t: _LightTrack, ob: TrafficLight) -> TrafficLight:
        t.history.append(ob.raw_state)
        while len(t.history) > max(1, self.cfg.vote_frames):
            t.history.popleft()
        cnt = Counter(t.history)
        best = max(cnt.values())
        state = next(s for s in reversed(t.history) if cnt[s] == best)  # 같으면 최근 값
        res = TrafficLight(ob.cam_index, ob.box, ob.raw_state, state, ob.orientation, ob.score, t.tid, ob.xy)
        t.last = res
        return res


# ---------------------------------------------------------------------------
# 한 프레임 처리(파이프라인이 부른다)
# ---------------------------------------------------------------------------
class TrafficLightReader:
    """카메라별 검출 결과에서 신호등 상자를 골라 색을 판별하고 다수결까지 한 결과를 돌려준다."""

    def __init__(self, cameras: list[Camera], cfg: TrafficLightConfig | None = None, scale: float = 1.0):
        self.cameras = cameras
        self.cfg = cfg or TrafficLightConfig()
        self.scale = float(scale)
        self.tracker = LightTracker(self.cfg)

    def update(self, frames: list[np.ndarray], dets_per_cam: list[list[Detection]]) -> list[TrafficLight]:
        c = self.cfg
        observed = []
        for i, (img, dets) in enumerate(zip(frames, dets_per_cam)):
            for d in dets:
                if d.label != TL_LABEL:
                    continue
                w, h = d.box[2] - d.box[0], d.box[3] - d.box[1]
                ori = orientation_of(w, h, c.aspect)
                state = classify_light(crop_box(img, d.box), c, ori)
                xy = light_xy(self.cameras[i], d.box, c.pole_m * self.scale) if i < len(self.cameras) else None
                observed.append(TrafficLight(i, tuple(d.box), state, state, ori, d.score, 0, xy))
        return self.tracker.update(observed)


def main_light(lights: list[TrafficLight] | None, cameras: list[Camera] | None = None) -> TrafficLight | None:
    """화면 위쪽 아이콘에 보여 줄 대표 신호등. 앞을 보는 카메라의, 색을 아는, 가장 큰 상자를 고른다."""
    if not lights:
        return None

    def forward(lt: TrafficLight) -> bool:
        if cameras is None or lt.cam_index >= len(cameras):
            return True
        return float(cameras[lt.cam_index].R_vc[0, 2]) > 0.5  # 카메라 광축의 X 성분(앞쪽이면 +)

    def area(lt: TrafficLight) -> float:
        return (lt.box[2] - lt.box[0]) * (lt.box[3] - lt.box[1])

    for pool in ([l for l in lights if forward(l) and l.state != "unknown"],
                 [l for l in lights if l.state != "unknown"], list(lights)):
        if pool:
            return max(pool, key=area)
    return None
