from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .bev import BevGrid
from .camera_model import Camera, intrinsics_from_hfov
from .render import BG, CLASS_COLOR, EGO_REAR_OFFSET, EGO_SIZE, _even, render_ground_layer
from .tracker import Track

# 리얼리티 뷰처럼 자차 뒤쪽 위에서 내려다보는 3D 시점 렌더러(numpy와 OpenCV만 쓴다).
#
# 그리는 순서(한 프레임)
#   1) 바닥: 탑뷰 크기의 바닥 그림(IPM 바닥 영상, 도로 마스크, 격자, 거리 고리)을 만든 뒤
#      "가상 카메라 화소 -> 바닥 셀" 표(미리 계산)로 cv2.remap 해서 원근 화면에 편다.
#      바닥은 ground refresh 때만 다시 만들고, 그 사이 프레임은 지난 결과를 다시 쓴다.
#   2) 물체와 자차: 가상 카메라에서 먼 것부터 그린다(화가 알고리즘). 물체는 반투명.
#      차는 박스, 사람은 세로 원기둥(다각기둥)으로 그린다.
#   3) 저해상도로 그렸으면 화면 크기로 키운 뒤, 글자(ID, 거리)는 키운 화면에 또렷하게 쓴다.
#
# 좌표계는 다른 모듈과 같다: 차량 X 전방, Y 좌측, Z 위(m), 원점은 뒷바퀴 축 가운데 바닥.
# 모든 길이(가상 카메라 위치, 물체 높이, 자차 크기, 고리 간격)에 장면 축척(scale)을 곱한다.
# 원근 투영은 크기를 같은 비율로 줄여도 그림이 같으므로 모형(scale 0.1)에서도 같은 화면이 나온다.


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class View3DConfig:
    """3D 시점 설정. 길이는 실차 m 기준이고, 렌더러가 장면 축척을 곱해 쓴다."""

    width: int = 960  # 화면 폭(px). 짝수로 맞춘다
    height: int = 540  # 화면 높이(px)
    cam_back: float = 7.0  # 가상 카메라가 원점(뒷바퀴 축)보다 뒤에 있는 거리(m)
    cam_height: float = 4.5  # 가상 카메라 높이(m)
    pitch_deg: float = 18.0  # 아래로 내려다보는 각(도)
    hfov_deg: float = 70.0  # 수평 화각(도)
    render_scale: float = 1.0  # 1보다 작으면 그만큼 작게 그린 뒤 화면 크기로 키운다(CPU 절약)
    box_alpha: float = 0.6  # 물체 박스 불투명도(0~1). 1이면 불투명
    show_grid: bool = True  # 바닥 격자(5m 간격)
    show_rings: bool = True  # 거리 고리(5, 10, 15, 20m)
    show_ids: bool = True  # 트랙 ID 글자
    show_velocity: bool = True  # 상대 속도 화살표(1초 뒤 위치까지)
    show_ego: bool = True  # 자차 모형
    interp: bool = False  # 트랙 표시 위치 보간(부드럽게). 기본은 추적 결과 그대로
    interp_tau: float = 0.15  # 보간 시간 상수(초). 작을수록 새 위치를 빨리 따라간다

    def checked(self) -> "View3DConfig":
        """값을 확인하고 폭과 높이를 짝수로 맞춘 사본을 돌려준다."""
        if not 0.1 <= self.render_scale <= 1.0:
            raise ValueError(f"render_scale은 0.1~1.0이어야 합니다: {self.render_scale}")
        if not 10.0 <= self.hfov_deg <= 150.0:
            raise ValueError(f"hfov_deg는 10~150도여야 합니다: {self.hfov_deg}")
        if self.cam_height <= 0:
            raise ValueError(f"cam_height는 0보다 커야 합니다: {self.cam_height}")
        if not 0.0 <= self.box_alpha <= 1.0:
            raise ValueError(f"box_alpha는 0~1이어야 합니다: {self.box_alpha}")
        return dataclasses.replace(self, width=_even(self.width), height=_even(self.height))


# 클래스별 높이(m, 실차). 없는 클래스는 DEFAULT_HEIGHT
CLASS_HEIGHT = {"car": 1.5, "truck": 3.0, "bus": 3.2, "person": 1.7, "bicycle": 1.5, "motorcycle": 1.5}
DEFAULT_HEIGHT = 1.5
CYLINDER_CLASSES = ("person",)  # 세로 원기둥으로 그릴 클래스
CYLINDER_SIDES = 10  # 원기둥 옆면 수(많을수록 둥글고 느리다)

# 색
SKY_TOP = (70, 52, 40)  # 화면 맨 위 하늘색(BGR)
SKY_HORIZON = (120, 104, 92)  # 지평선 근처 하늘색
GRID_COLOR = (54, 50, 47)
RING_COLOR = (88, 82, 76)
RING_TEXT = (150, 145, 140)
EGO_BODY = (225, 225, 228)
EGO_CABIN = (95, 92, 90)
LIGHT_XY = np.array([0.5, 0.6]) / np.hypot(0.5, 0.6)  # 옆면 명암용 빛 방향(앞 왼쪽에서 비춤)

RINGS_M = (5, 10, 15, 20)
GRID_STEP_M = 5.0
NEAR_M = 0.3  # 가상 카메라 앞 이 거리보다 가까운 꼭짓점이 있는 물체는 그리지 않는다(실차 m)
ARROW_SEC = 1.0  # 속도 화살표 길이 = 상대 속도 x 이 시간
ARROW_MAX_M = 8.0  # 화살표 최대 길이(실차 m). 새 트랙의 튀는 속도 추정이 화면을 가로지르지 않게


# ---------------------------------------------------------------------------
# 트랙 표시 위치 보간
# ---------------------------------------------------------------------------
class TrackSmoother:
    """트랙 표시 위치를 지난 표시 위치와 새 추적 위치 사이에서 보간한다.

    표시 위치 <- 표시 위치 + a x (추적 위치 - 표시 위치),  a = 1 - exp(-dt / tau)
    - 시간은 step()에 넘긴 dt를 더한 값이라 시뮬레이션에서도 같은 결과가 나온다(벽시계를 쓰지 않음).
    - 처음 보는 ID는 바로 그 자리에 둔다. 사라진 ID는 지운다.
    - 추적기의 Track을 고치지 않고, 위치만 바꾼 사본을 돌려준다.
    """

    def __init__(self, tau: float = 0.15):
        self.tau = max(1e-3, float(tau))
        self._pos: dict[int, np.ndarray] = {}
        self._t: float | None = None

    def update(self, tracks: list[Track], t: float) -> list[Track]:
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        a = 1.0 - math.exp(-dt / self.tau)
        out, seen = [], {}
        for tr in tracks:
            target = np.asarray(tr.xy, dtype=np.float64)
            prev = self._pos.get(tr.tid)
            xy = target.copy() if prev is None else prev + a * (target - prev)
            seen[tr.tid] = xy
            out.append(dataclasses.replace(tr, xy=xy.copy(), vel=np.array(tr.vel, dtype=np.float64)))
        self._pos = seen
        return out


# ---------------------------------------------------------------------------
# 그릴 물체 한 개(바닥 다각형을 위로 세운 기둥)
# ---------------------------------------------------------------------------
@dataclass
class _Prism:
    base_xy: np.ndarray  # (N,2) 바닥 다각형, 위에서 볼 때 반시계 방향
    z0: float  # 바닥 높이(m)
    z1: float  # 윗면 높이(m)
    color: tuple  # BGR
    alpha: float  # 불투명도


def box_base(center_xy, length: float, width: float) -> np.ndarray:
    """차량 축과 나란한 직사각형 바닥(반시계 방향: 앞왼쪽, 뒤왼쪽, 뒤오른쪽, 앞오른쪽)."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    L, W = length / 2.0, width / 2.0
    return np.array([[cx + L, cy + W], [cx - L, cy + W], [cx - L, cy - W], [cx + L, cy - W]])


def circle_base(center_xy, radius: float, sides: int = CYLINDER_SIDES) -> np.ndarray:
    """원기둥 바닥을 정다각형으로(반시계 방향)."""
    a = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return np.stack([center_xy[0] + radius * np.cos(a), center_xy[1] + radius * np.sin(a)], axis=1)


# ---------------------------------------------------------------------------
# 렌더러
# ---------------------------------------------------------------------------
class Renderer3D:
    """자차 뒤쪽 위 가상 카메라에서 본 3D 화면을 그린다.

    grid: 바닥 영상(IPM 결과)과 도로 마스크가 놓인 탑뷰 격자
    cfg: View3DConfig
    scale: 장면 축척(모형이면 0.1 같은 값). 길이 설정에 모두 곱한다
    timer: StageTimer(선택). 바닥을 다시 만든 시간을 "render3d_ground"로 적는다
    """

    def __init__(self, grid: BevGrid, cfg: View3DConfig | None = None, scale: float = 1.0, timer=None):
        self.grid = grid
        self.cfg = (cfg or View3DConfig()).checked()
        self.scale = float(scale)
        self.timer = timer
        c, s = self.cfg, self.scale
        # ---- 가상 카메라(핀홀). 그리는 해상도는 render_scale만큼 작다 ----
        self.out_w, self.out_h = c.width, c.height
        self.rw, self.rh = _even(c.width * c.render_scale), _even(c.height * c.render_scale)
        self.cam = Camera(
            name="view3d",
            width=self.rw,
            height=self.rh,
            K=intrinsics_from_hfov(self.rw, self.rh, c.hfov_deg),
            position=np.array([-c.cam_back * s, 0.0, c.cam_height * s]),
            pitch_deg=c.pitch_deg,
        )
        self._R = self.cam.R_cv  # 차량 -> 카메라 회전
        self._pos = self.cam.position
        self._K = self.cam.K
        self._near = NEAR_M * s
        # ---- 바닥 표(화면 화소 -> 탑뷰 셀)와 하늘 배경은 한 번만 만든다 ----
        self.horizon_row, self._maps = self._build_ground_maps()
        self._background = self._build_background()
        self._ground_view: np.ndarray | None = None  # 지난번에 편 바닥(다시 쓰기용)
        self.ground_refreshes = 0  # 바닥을 다시 만든 횟수
        self._smoother = TrackSmoother(c.interp_tau) if c.interp else None
        self._ego_parts = self._build_ego()

    # ---- 투영 ---------------------------------------------------------
    def to_cam(self, pts: np.ndarray) -> np.ndarray:
        """차량 좌표 점(N,3) -> 가상 카메라 광학 좌표(N,3: x 오른쪽, y 아래, z 앞)."""
        return (np.asarray(pts, dtype=np.float64).reshape(-1, 3) - self._pos) @ self._R.T

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """차량 좌표 점(N,3) -> 그리는 해상도의 화소(N,2)와 카메라 앞 거리 z(N,).

        z가 near보다 작은 점은 화소 값이 의미 없다(카메라 뒤이거나 너무 가깝다). 호출하는 쪽이 z를 보고 거른다.
        """
        pc = self.to_cam(pts)
        z = pc[:, 2]
        zs = np.where(z > 1e-9, z, 1e-9)
        u = self._K[0, 0] * pc[:, 0] / zs + self._K[0, 2]
        v = self._K[1, 1] * pc[:, 1] / zs + self._K[1, 2]
        return np.stack([u, v], axis=1), z

    def project_output(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """project()와 같지만 최종 화면 크기(width x height)의 화소로 돌려준다."""
        uv, z = self.project(pts)
        return uv * np.array([self.out_w / self.rw, self.out_h / self.rh]), z

    # ---- 바닥 표 만들기 -------------------------------------------------
    def _build_ground_maps(self):
        """화면 화소마다 시선이 바닥(Z=0)과 만나는 점을 구해 탑뷰 셀 좌표로 바꾼 remap 표.

        - 지평선 위(시선이 바닥과 만나지 않는) 줄은 하늘이라 표에서 뺀다. 돌려주는 horizon_row부터 아래만 표가 있다.
          cv2.warpPerspective로 바로 펴면 지평선 위 화소가 카메라 뒤쪽 바닥을 거울처럼 끌어오므로 표를 직접 만든다.
        - 탑뷰 밖 셀은 -10으로 두어 remap이 배경색(BG)을 넣게 한다. int16 표(CV_16SC2)가 넘치지 않게 범위를 자른다.
        """
        g, cam = self.grid, self.cam
        u, v = np.meshgrid(np.arange(self.rw) + 0.5, np.arange(self.rh) + 0.5)
        # 화소의 광학 좌표 방향 (x, y, 1). 큰 배열 행렬곱 대신 성분별로 계산한다
        # (맥의 numpy가 큰 행렬곱에서 근거 없는 0 나눗셈 경고를 내는 일이 있어서)
        K = self._K
        dx = (u - K[0, 2]) / K[0, 0]
        dy = (v - K[1, 2]) / K[1, 1]
        R = cam.R_vc  # 광학 -> 차량 회전
        d_veh = [R[i, 0] * dx + R[i, 1] * dy + R[i, 2] for i in range(3)]  # 차량 좌표 방향 X, Y, Z
        down = d_veh[2] < -1e-9  # 아래를 향해야 바닥과 만난다
        rows_sky = np.nonzero(~down.all(axis=1))[0]
        h0 = int(rows_sky.max()) + 1 if rows_sky.size else 0
        if h0 >= self.rh:
            return self.rh, None  # 화면 전체가 하늘(카메라가 위를 봄)
        t = -self._pos[2] / d_veh[2][h0:]
        X = self._pos[0] + t * d_veh[0][h0:]
        Y = self._pos[1] + t * d_veh[1][h0:]
        # 탑뷰 화소 좌표(바닥 영상 배열의 열, 행). 셀 i의 중심이 좌표 i에 오도록 0.5를 뺀다
        col = (g.y_max - Y) / g.res - 0.5
        row = (g.x_max - X) / g.res - 0.5
        outside = (col < -1) | (col > g.width) | (row < -1) | (row > g.height)
        mx = np.where(outside, -10.0, col).astype(np.float32)
        my = np.where(outside, -10.0, row).astype(np.float32)
        return h0, cv2.convertMaps(mx, my, cv2.CV_16SC2)

    def _build_background(self) -> np.ndarray:
        """하늘(위에서 지평선으로 갈수록 밝게)과 빈 바닥(배경색). 바닥 영상이 없을 때 그대로 보인다."""
        img = np.empty((self.rh, self.rw, 3), np.uint8)
        img[:] = BG
        h0 = self.horizon_row
        if h0 > 0:
            k = np.linspace(0.0, 1.0, h0)[:, None]
            sky = (1 - k) * np.array(SKY_TOP, np.float64) + k * np.array(SKY_HORIZON, np.float64)
            img[:h0] = sky.astype(np.uint8)[:, None, :]
        return img

    # ---- 바닥 그리기 ---------------------------------------------------
    def _ground_layer(self, ground_img, road_mask, coverage) -> np.ndarray:
        """탑뷰 크기의 바닥 그림에 격자와 거리 고리를 더한다(원근으로 펴기 전)."""
        g, c, s = self.grid, self.cfg, self.scale
        layer = render_ground_layer(g, ground_img, road_mask, coverage)
        if c.show_grid:
            step = GRID_STEP_M * s
            for x in np.arange(math.ceil(g.x_min / step) * step, g.x_max + 1e-9, step):
                r = int(round((g.x_max - x) / g.res))
                cv2.line(layer, (0, r), (g.width - 1, r), GRID_COLOR, 1)
            for y in np.arange(math.ceil(g.y_min / step) * step, g.y_max + 1e-9, step):
                col = int(round((g.y_max - y) / g.res))
                cv2.line(layer, (col, 0), (col, g.height - 1), GRID_COLOR, 1)
        if c.show_rings:
            o = g.to_pixel(np.array([[0.0, 0.0]]))[0]
            for base in RINGS_M:
                cv2.circle(layer, (int(round(o[0])), int(round(o[1]))), int(round(base * s / g.res)), RING_COLOR, 3,
                           cv2.LINE_AA)
        return layer

    def _refresh_ground(self, ground_img, road_mask, coverage) -> None:
        """바닥을 다시 만들어 원근 화면으로 편다. 결과는 다음 프레임들이 다시 쓴다."""
        img = self._background.copy()
        if self._maps is not None:
            layer = self._ground_layer(ground_img, road_mask, coverage)
            m1, m2 = self._maps
            img[self.horizon_row:] = cv2.remap(layer, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=BG)
        self._ground_view = img
        self.ground_refreshes += 1

    # ---- 자차 모형 -----------------------------------------------------
    def _build_ego(self) -> list[_Prism]:
        """자차를 몸통 박스와 그 위 실내(유리) 박스 두 개로 만든다(저폴리)."""
        s = self.scale
        L, W = EGO_SIZE[0] * s, EGO_SIZE[1] * s
        cx = EGO_SIZE[0] / 2 * s - EGO_REAR_OFFSET * s  # 자차 중심 X
        body = _Prism(box_base((cx, 0.0), L, W), 0.12 * s, 0.85 * s, EGO_BODY, 1.0)
        cabin = _Prism(box_base((cx - 0.25 * s, 0.0), 2.3 * s, 1.5 * s), 0.85 * s, 1.4 * s, EGO_CABIN, 1.0)
        return [body, cabin]

    # ---- 기둥 하나 그리기 ------------------------------------------------
    def _draw_prism(self, canvas: np.ndarray, p: _Prism) -> bool:
        """바닥 다각형을 세운 기둥을 그린다. 보이는 면만 칠하고(볼록이라 면끼리 겹치지 않음) 테두리를 긋는다.

        꼭짓점 하나라도 가상 카메라 near보다 가까우면(뒤쪽 포함) 통째로 건너뛴다. 그렸으면 True.
        반투명은 물체를 둘러싼 사각형(ROI)에서만 섞는다(화면 전체를 섞으면 물체마다 수 ms가 든다).
        """
        n = len(p.base_xy)
        pts3 = np.vstack([np.c_[p.base_xy, np.full(n, p.z0)], np.c_[p.base_xy, np.full(n, p.z1)]])
        uv, z = self.project(pts3)
        if np.any(z < self._near):
            return False
        H, W = canvas.shape[:2]
        x0, y0 = np.floor(uv.min(axis=0)).astype(int)
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            return False  # 화면 밖
        if (x1 - x0) > 4 * W or (y1 - y0) > 4 * H:
            return False  # 카메라에 너무 붙어 화면보다 훨씬 크게 투영됨(그려도 의미 없음)
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        cam_xy = self._pos[:2]
        faces, shades = [], []
        # 옆면: 바깥 법선이 카메라 쪽을 향하면 보인다
        for i in range(n):
            j = (i + 1) % n
            e = p.base_xy[j] - p.base_xy[i]
            nrm = np.array([e[1], -e[0]])
            mid = (p.base_xy[i] + p.base_xy[j]) / 2.0
            if float(nrm @ (mid - cam_xy)) < 0.0:
                faces.append([i, j, n + j, n + i])
                nn = nrm / (np.linalg.norm(nrm) + 1e-12)
                shades.append(0.55 + 0.3 * max(0.0, float(nn @ LIGHT_XY)))
        # 윗면: 카메라가 윗면보다 높으면 보인다
        if self._pos[2] > p.z1:
            faces.append(list(range(n, 2 * n)))
            shades.append(1.0)
        if not faces:
            return False
        ipts = np.round(uv - [x0c, y0c]).astype(np.int32)
        roi = canvas[y0c:y1c, x0c:x1c]
        layer = roi.copy() if p.alpha < 1.0 else roi
        for f, sh in zip(faces, shades):
            col = tuple(int(min(255, ch * sh)) for ch in p.color)
            cv2.fillConvexPoly(layer, ipts[f], col, cv2.LINE_AA)
        if p.alpha < 1.0:
            cv2.addWeighted(layer, p.alpha, roi, 1.0 - p.alpha, 0.0, dst=roi)
        edge = tuple(int(min(255, ch * 0.5 + 128)) for ch in p.color)
        cv2.polylines(roi, [ipts[f] for f in faces], True, edge, 1, cv2.LINE_AA)
        return True

    # ---- 속도 화살표 ---------------------------------------------------
    def _draw_arrow(self, canvas: np.ndarray, t: Track, color: tuple) -> None:
        vel = np.asarray(t.vel, dtype=np.float64)
        if float(np.linalg.norm(vel)) < t.min_speed:
            return  # 느리면 방향을 믿지 않는다(추적기와 같은 기준)
        zg = 0.03 * self.scale
        step = vel * ARROW_SEC
        n = float(np.linalg.norm(step))
        if n > ARROW_MAX_M * self.scale:
            step = step * (ARROW_MAX_M * self.scale / n)
        a = np.array([[t.xy[0], t.xy[1], zg], [t.xy[0] + step[0], t.xy[1] + step[1], zg]])
        uv, z = self.project(a)
        if np.any(z < self._near):
            return
        p0, p1 = (tuple(int(round(c)) for c in q) for q in uv)
        if max(abs(p0[0]), abs(p0[1]), abs(p1[0]), abs(p1[1])) > 8 * max(self.rw, self.rh):
            return
        cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)

    # ---- 한 프레임 -----------------------------------------------------
    def render(
        self,
        tracks: list[Track],
        ground_img: np.ndarray | None = None,
        road_mask: np.ndarray | None = None,
        coverage: np.ndarray | None = None,
        refresh_ground: bool = True,
        t: float | None = None,
    ) -> np.ndarray:
        """3D 화면 한 장(width x height, BGR)을 돌려준다.

        refresh_ground=False이고 지난 바닥이 있으면 바닥을 다시 만들지 않는다(ground_img 등은 무시).
        t: 누적 시간(초). interp가 켜져 있으면 지난 호출과의 차이로 보간 비율을 정한다
        """
        c, s = self.cfg, self.scale
        # 1) 바닥
        if refresh_ground or self._ground_view is None:
            if self.timer is not None:
                with self.timer.stage("render3d_ground"):
                    self._refresh_ground(ground_img, road_mask, coverage)
            else:
                self._refresh_ground(ground_img, road_mask, coverage)
        canvas = self._ground_view.copy()

        # 2) 트랙 보간(선택)
        if self._smoother is not None:
            tracks = self._smoother.update(tracks, 0.0 if t is None else float(t))

        # 3) 그릴 것 목록: (가상 카메라와의 거리, 종류, 내용). 먼 것부터 그린다(화가 알고리즘)
        items = []
        if c.show_ego:
            ego_c = np.array([EGO_SIZE[0] / 2 * s - EGO_REAR_OFFSET * s, 0.0, 0.7 * s])
            items.append((float(np.linalg.norm(ego_c - self._pos)), 0, "ego", None))
        for k, tr in enumerate(tracks):
            h = CLASS_HEIGHT.get(tr.label, DEFAULT_HEIGHT) * s
            ctr = np.array([tr.xy[0], tr.xy[1], h / 2.0])
            items.append((float(np.linalg.norm(ctr - self._pos)), k + 1, "track", tr))
        items.sort(key=lambda it: (-it[0], it[1]))  # 거리가 같으면 입력 순서로(결과가 늘 같게)

        labels = []  # (u, v, 글자, 색): 키운 화면에 마지막에 쓴다
        for _, _, kind, tr in items:
            if kind == "ego":
                for part in self._ego_parts:
                    self._draw_prism(canvas, part)
                continue
            color = CLASS_COLOR.get(tr.label, (200, 200, 200))
            h = CLASS_HEIGHT.get(tr.label, DEFAULT_HEIGHT) * s
            if c.show_velocity:
                self._draw_arrow(canvas, tr, color)  # 바닥에 그리므로 박스보다 먼저(반투명 박스 너머로 보인다)
            if tr.label in CYLINDER_CLASSES:
                r = max(tr.size[1], 0.4 * s) / 2.0
                prism = _Prism(circle_base(tr.xy, r), 0.0, h, color, c.box_alpha)
            else:
                prism = _Prism(box_base(tr.xy, tr.size[0], tr.size[1]), 0.0, h, color, c.box_alpha)
            if self._draw_prism(canvas, prism) and c.show_ids:
                labels.append((np.array([tr.xy[0], tr.xy[1], h + 0.3 * s]), str(tr.tid), (240, 240, 240)))

        # 4) 저해상도면 화면 크기로 키운다
        if (self.rw, self.rh) != (self.out_w, self.out_h):
            canvas = cv2.resize(canvas, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

        # 5) 글자: 거리 고리 숫자와 트랙 ID(키운 화면에 바로 써서 흐려지지 않게)
        if c.show_rings:
            d = math.sqrt(0.5)
            for base in RINGS_M:
                r = base * s
                labels.append((np.array([r * d, -r * d, 0.0]), f"{r:g}m", RING_TEXT))
        for p3, text, col in labels:
            uv, z = self.project_output(p3[None])
            if z[0] < self._near:
                continue
            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
            if 0 <= u < self.out_w and 0 <= v < self.out_h:
                cv2.putText(canvas, text, (u - 6, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return canvas
