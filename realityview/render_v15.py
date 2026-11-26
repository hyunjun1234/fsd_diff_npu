from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .bev import BevGrid
from .camera_model import Camera, intrinsics_from_hfov
from .raster import raster_vc
from .render import _even, dash_runs, draw_light_icon, ego_lane_polygon
from .tracker import Track

# FSD v15 리얼리티 뷰를 흉내 낸 3D 화면(--view v15). numpy, OpenCV, 작은 C 래스터라이저(_raster.c)만 쓴다.
#
# v15는 약 1B 생성 모델이 카메라 특징에서 화면을 바로 만들지만, 파이 5에서는 돌릴 수 없다. 그래서 화면에서
# 보이는 특징을 기존 방식(검출, 추적, IPM 바닥)으로 따라 한다.
#   1) 바닥: 카메라 영상을 바닥에 편 실제 색 영상(IPM)에 색 보정, 가장자리 번짐, 거리 안개를 입힌다.
#      카메라가 못 보는 곳은 아스팔트 무늬로 채운다. 원근 바닥과 안개, 하늘은 시점이 그대로면 다시 쓴다.
#   2) 바닥 위: 차선(폭 있는 띠, 실선과 점선), 계획 경로(파란 띠, 멀수록 흐림), 물체 그림자(부드러운 그림자).
#   3) 물체: 박스 대신 3D 모델. 자차는 사용자 차량 모델(assets/vehicles, 흰색 도장), 다른 차는 저폴리 승용차,
#      트럭, 버스, 이륜차, 걷는 사람, 신호등 기둥. 면마다 햇빛 방향 명암과 하이라이트, 거리 안개를 넣고
#      z-버퍼로 가림을 맞춘다.
#   4) 화면 글자와 표시: 속도와 기어(알 때만), 신호등 아이콘, 시점 이름.
#   5) 시점: chase(기본), high, rear(후방), side, top, orbit(자차 둘레를 도는 가상 카메라). 바꾸면 부드럽게 옮겨 간다.
#
# 좌표계는 다른 모듈과 같다: 차량 X 전방, Y 좌측, Z 위(m), 원점은 뒷바퀴 축 가운데 바닥.
# 모든 길이에 장면 축척(scale)을 곱하므로 모형(scale 0.1)에서도 같은 화면이 나온다.

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "vehicles"
DEFAULT_EGO_MODEL = ASSET_DIR / "ego_carnival_2011.npz"


# ---------------------------------------------------------------------------
# 시점
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewPose:
    """가상 카메라 자세. 바라보는 점(target)을 중심으로 방위각, 수평 거리, 높이로 정한다(실차 m, 도)."""

    target_x: float
    target_z: float
    azimuth: float  # 0이면 카메라가 target 앞쪽(+X), 180이면 뒤쪽, 90이면 왼쪽
    dist: float  # target에서 카메라까지 수평 거리
    height: float  # 카메라 높이
    hfov: float  # 수평 화각


VIEWS: dict[str, ViewPose] = {
    "chase": ViewPose(10.0, 0.0, 180.0, 17.5, 4.6, 64.0),  # 자차 뒤쪽 위(기본, v15 기본 화면과 비슷한 높이)
    "high": ViewPose(7.0, 0.0, 180.0, 14.0, 10.0, 66.0),  # 더 높이서 넓게
    "rear": ViewPose(-7.0, 0.0, 0.0, 15.0, 4.2, 70.0),  # 자차 앞쪽 위에서 뒤를 본다(후방 카메라 영상이 바닥에 깔림)
    "side": ViewPose(1.5, 0.3, 90.0, 11.0, 3.0, 66.0),  # 왼쪽 옆에서
    "top": ViewPose(4.0, 0.0, 180.0, 0.6, 30.0, 62.0),  # 거의 바로 위
    "orbit": ViewPose(1.5, 0.3, 180.0, 10.5, 4.0, 66.0),  # 방위각이 시간에 따라 돈다
}
VIEW_KEYS = {ord("1"): "chase", ord("2"): "high", ord("3"): "rear", ord("4"): "side", ord("5"): "top", ord("6"): "orbit"}


def _lerp_pose(a: ViewPose, b: ViewPose, k: float) -> ViewPose:
    k = k * k * (3 - 2 * k)  # 부드럽게 출발하고 멈춘다
    da = (b.azimuth - a.azimuth + 180.0) % 360.0 - 180.0  # 짧은 쪽으로 돈다
    return ViewPose(*(x + (y - x) * k for x, y in zip(
        (a.target_x, a.target_z, a.azimuth, a.dist, a.height, a.hfov),
        (b.target_x, b.target_z, a.azimuth + da, b.dist, b.height, b.hfov))))


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class V15Config:
    """v15풍 화면 설정. 길이는 실차 m 기준."""

    width: int = 960
    height: int = 540
    view: str = "chase"  # 처음 시점
    render_scale: float = 1.0  # 1보다 작으면 작게 그린 뒤 키운다(글자는 키운 화면에 쓴다)
    transition_s: float = 1.0  # 시점을 바꿀 때 옮겨 가는 시간(초)
    orbit_deg_s: float = 14.0  # orbit 시점의 회전 속도(도/초)
    tour_s: float = 0.0  # 0보다 크면 이 간격(초)으로 시점을 차례로 돌린다(시연 영상용)
    tour_views: tuple = ("chase", "high", "orbit", "rear", "side", "chase")
    ego_model: str | None = None  # 자차 모델 npz. None이면 assets/vehicles/ego_carnival_2011.npz
    ego_paint: tuple | None = None  # 자차 도장 색 BGR로 덮어쓰기(None이면 모델에 저장된 색, 기본 흰색)
    fog_start_m: float = 18.0  # 이 거리부터 안개가 낀다
    fog_end_m: float = 70.0  # 이 거리에서 안개가 가장 짙다
    shadows: bool = True
    path: bool = True  # 계획 경로 파란 띠
    path_len_m: float = 38.0
    show_ui: bool = True
    show_ids: bool = False
    interp: bool = True  # 트랙 위치를 프레임 사이에서 부드럽게(1 - exp(-dt / tau))
    interp_tau: float = 0.12
    ground_saturation: float = 0.75  # 바닥 영상 채도(1이면 원본)
    ground_scale: float = 0.5  # 원근 바닥과 하늘을 이 비율 해상도로 편 뒤 키운다(IPM 바닥이 원래 거칠어 차이가 작다)

    def checked(self) -> "V15Config":
        if not 0.25 <= self.ground_scale <= 1.0:
            raise ValueError(f"ground_scale은 0.25~1.0이어야 합니다: {self.ground_scale}")
        if not 0.25 <= self.render_scale <= 1.0:
            raise ValueError(f"render_scale은 0.25~1.0이어야 합니다: {self.render_scale}")
        if self.view not in VIEWS:
            raise ValueError(f"view는 {', '.join(VIEWS)} 중 하나여야 합니다: {self.view}")
        for v in self.tour_views:
            if v not in VIEWS:
                raise ValueError(f"tour_views에 모르는 시점이 있습니다: {v}")
        return dataclasses.replace(self, width=_even(self.width), height=_even(self.height))


# 색(BGR)
SKY_TOP = (58, 44, 36)
SKY_HORIZON = (150, 136, 124)
FOG = (128, 118, 110)
ASPHALT = (62, 60, 58)
LANE_WHITE = (232, 234, 236)
LANE_YELLOW = (40, 190, 240)
PATH_BLUE = (255, 150, 40)
OTHER_CAR = (196, 196, 200)
OTHER_GLASS = (58, 46, 40)
TIRE = (30, 30, 32)
TRUCK = (186, 188, 192)
PERSON = (205, 210, 222)
BIKE = (110, 214, 120)
POLE = (96, 96, 100)
SUN = np.array([-0.45, 0.35, 0.82])  # 햇빛 방향(물체 쪽에서 해를 향함). 뒤쪽 왼쪽 위
SUN = SUN / np.linalg.norm(SUN)

# 클래스별 표시 크기(실차 m: 길이, 폭, 높이). 추적기의 크기 추정은 흔들려서 표시는 이 값을 쓴다
CLASS_DIMS = {
    "car": (4.5, 1.85, 1.45),
    "truck": (8.0, 2.5, 3.2),
    "bus": (11.0, 2.55, 3.2),
    "person": (0.5, 0.5, 1.72),
    "bicycle": (1.75, 0.6, 1.7),
    "motorcycle": (2.0, 0.8, 1.6),
}
NEAR_M = 0.3
EGO_FRONT_M = 4.0  # 뒷바퀴 축에서 자차 앞끝까지(대략, 경로 띠 시작점)


# ---------------------------------------------------------------------------
# 메시(정점, 삼각형, 면 색, 면 재질)
# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    V: np.ndarray  # (N,3) 로컬 좌표(X 앞, Y 왼쪽, Z 위), 원점은 바닥 가운데(자차는 뒷바퀴 축 가운데)
    F: np.ndarray  # (M,3) int32
    C: np.ndarray  # (M,3) uint8 BGR
    shiny: np.ndarray  # (M,) float 하이라이트 세기(도장 0.5, 유리 0.8, 나머지 0.1)
    CN: np.ndarray | None = None  # (M,3,3) 꼭짓점별 법선(로컬). corner_normals()가 한 번 계산해 둔다

    def corner_normals(self, crease_deg: float = 38.0) -> np.ndarray:
        """삼각형 꼭짓점마다 쓸 법선. 이웃 면과 평균한 정점 법선을 쓰되, 면 법선과 crease_deg보다 벌어지면
        (상자 모서리, 유리와 차체 경계처럼 각진 곳) 면 법선을 쓴다. 명암 보간이 둥근 곳만 부드럽게 한다."""
        if self.CN is None:
            a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
            fn = np.cross(b - a, c - a)
            fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
            # 면 방향(감긴 순서)이 들쭉날쭉한 원본 모델도 있어서 정점 법선은 방향을 맞춰 더한다:
            # 먼저 바깥 방향(메시 중심에서 멀어지는 쪽)으로 면 법선을 뒤집어 맞춘다
            ctr = self.V.mean(axis=0)
            fc = (a + b + c) / 3.0
            flip = np.sum(fn * (fc - ctr), axis=1) < 0
            fn[flip] *= -1
            vn = np.zeros_like(self.V)
            for k in range(3):
                np.add.at(vn, self.F[:, k], fn)
            vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
            cn = vn[self.F]  # (M,3,3)
            cos = np.sum(cn * fn[:, None, :], axis=2)
            sharp = cos < math.cos(math.radians(crease_deg))
            cn[sharp] = np.broadcast_to(fn[:, None, :], cn.shape)[sharp]
            self.CN = cn
        return self.CN

    @staticmethod
    def concat(parts: list["Mesh"]) -> "Mesh":
        V, F, C, S, n = [], [], [], [], 0
        for p in parts:
            V.append(p.V)
            F.append(p.F + n)
            C.append(p.C)
            S.append(p.shiny)
            n += len(p.V)
        return Mesh(np.vstack(V), np.vstack(F).astype(np.int32), np.vstack(C).astype(np.uint8), np.concatenate(S))


def _mesh(V, F, color, shiny=0.1) -> Mesh:
    F = np.asarray(F, np.int32).reshape(-1, 3)
    return Mesh(np.asarray(V, np.float64), F, np.tile(np.array(color, np.uint8), (len(F), 1)),
                np.full(len(F), shiny))


def _quad_tris(quads) -> list:
    out = []
    for a, b, c, d in quads:
        out += [(a, b, c), (a, c, d)]
    return out


def box_mesh(cx, cy, z0, L, W, H, color, shiny=0.1, top_scale=1.0) -> Mesh:
    """축과 나란한 상자(top_scale < 1이면 윗면이 좁은 사다리꼴 기둥)."""
    l, w = L / 2, W / 2
    lt, wt = l * top_scale, w * top_scale
    V = [(cx + l, cy + w, z0), (cx - l, cy + w, z0), (cx - l, cy - w, z0), (cx + l, cy - w, z0),
         (cx + lt, cy + wt, z0 + H), (cx - lt, cy + wt, z0 + H), (cx - lt, cy - wt, z0 + H), (cx + lt, cy - wt, z0 + H)]
    q = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return _mesh(V, _quad_tris(q), color, shiny)


def cylinder_mesh(c0, c1, r, sides, color, shiny=0.1) -> Mesh:
    """c0에서 c1까지 반지름 r 다각기둥(양 끝 막음)."""
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    ax = c1 - c0
    ax /= np.linalg.norm(ax)
    u = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(a)[:, None] * u * r + np.sin(a)[:, None] * v * r
    V = np.vstack([c0 + ring, c1 + ring, c0, c1])
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, sides + j), (i, sides + j, sides + i), (2 * sides, j, i), (2 * sides + 1, sides + i, sides + j)]
    return _mesh(V, F, color, shiny)


def sphere_mesh(c, r, color, rings=4, sides=8, shiny=0.1) -> Mesh:
    V = [(c[0], c[1], c[2] + r), (c[0], c[1], c[2] - r)]
    for i in range(1, rings):
        ph = np.pi * i / rings
        for j in range(sides):
            th = 2 * np.pi * j / sides
            V.append((c[0] + r * np.sin(ph) * np.cos(th), c[1] + r * np.sin(ph) * np.sin(th), c[2] + r * np.cos(ph)))
    F = []
    for j in range(sides):
        k = (j + 1) % sides
        F.append((0, 2 + j, 2 + k))
        F.append((1, 2 + (rings - 2) * sides + k, 2 + (rings - 2) * sides + j))
    for i in range(rings - 2):
        for j in range(sides):
            k = (j + 1) % sides
            a, b = 2 + i * sides + j, 2 + i * sides + k
            c2, d = 2 + (i + 1) * sides + j, 2 + (i + 1) * sides + k
            F += [(a, c2, d), (a, d, b)]
    return _mesh(V, F, color, shiny)


def loft_mesh(sections, color_body, color_glass, shiny_body=0.5) -> Mesh:
    """옆에서 본 단면 여러 개를 이어 차체를 만든다.

    sections: (x, z_bottom, z_belt, z_top, half_w_bottom, half_w_belt, half_w_top, glass) 목록(뒤에서 앞으로).
    glass가 True인 구간은 허리선과 지붕 사이(옆창)와 윗면(앞유리, 뒷유리)을 유리색으로 칠한다.
    """
    rings = []
    for x, zb, zm, zt, wb, wm, wt, _ in sections:
        rings.append([(x, wb, zb), (x, wm, zm), (x, wt, zt), (x, -wt, zt), (x, -wm, zm), (x, -wb, zb)])
    V = [p for r in rings for p in r]
    F, C, S = [], [], []
    n = 6
    for s in range(len(rings) - 1):
        glass = sections[s][7] and sections[s + 1][7]
        slope = abs(sections[s + 1][3] - sections[s][3]) > 0.15 * abs(sections[s + 1][0] - sections[s][0])
        for e in range(n):
            a, b = s * n + e, s * n + (e + 1) % n
            c, d = (s + 1) * n + (e + 1) % n, (s + 1) * n + e
            F += [(a, b, c), (a, c, d)]
            is_glass = (glass and e in (1, 3)) or (e == 2 and slope and sections[s][7] != sections[s + 1][7]) \
                or (e == 2 and slope and glass)
            col = color_glass if is_glass else color_body
            C += [col, col]
            S += [0.35 if is_glass else shiny_body] * 2
    # 앞뒤 막기
    for s in (0, len(rings) - 1):
        base = s * n
        for e in range(1, n - 1):
            F.append((base, base + e, base + e + 1))
            C.append(color_body)
            S.append(shiny_body)
    return Mesh(np.asarray(V, float), np.asarray(F, np.int32), np.asarray(C, np.uint8), np.asarray(S, float))


def _wheels(L, W, r, xs, color=TIRE) -> list[Mesh]:
    out = []
    for x in xs:
        for y in (W / 2 - 0.12, -W / 2 + 0.12):
            out.append(cylinder_mesh((x, y - 0.12, r), (x, y + 0.12, r), r, 8, color))
    return out


def car_mesh(L=4.5, W=1.85, H=1.45, body=OTHER_CAR) -> Mesh:
    """저폴리 승용차(뒤에서 앞으로 단면 8개). 원점은 바닥 가운데."""
    hl, hw = L / 2, W / 2
    secs = [
        (-hl, 0.32, 0.78, 0.92, hw * 0.92, hw * 0.95, hw * 0.88, False),
        (-hl + 0.25, 0.25, 0.80, 0.98, hw, hw, hw * 0.9, False),
        (-hl + 0.75, 0.22, 0.82, H * 0.98, hw, hw, hw * 0.78, True),
        (-0.35, 0.22, 0.84, H, hw, hw, hw * 0.78, True),
        (0.55, 0.22, 0.84, H * 0.97, hw, hw, hw * 0.76, True),
        (hl - 1.25, 0.22, 0.82, 0.98, hw, hw, hw * 0.86, False),
        (hl - 0.2, 0.26, 0.74, 0.84, hw, hw * 0.97, hw * 0.86, False),
        (hl, 0.34, 0.62, 0.70, hw * 0.9, hw * 0.9, hw * 0.82, False),
    ]
    body_m = loft_mesh(secs, body, OTHER_GLASS)
    return Mesh.concat([body_m] + _wheels(L, W, 0.33, (-hl + 0.9, hl - 0.95)))


def truck_mesh(L=8.0, W=2.5, H=3.2) -> Mesh:
    cab_l = 2.1
    cab = loft_mesh([
        (L / 2 - cab_l, 0.5, 1.6, H * 0.9, W / 2, W / 2, W / 2 * 0.96, False),
        (L / 2 - 0.9, 0.5, 1.6, H * 0.9, W / 2, W / 2, W / 2 * 0.96, True),
        (L / 2 - 0.25, 0.5, 1.55, H * 0.82, W / 2, W / 2, W / 2 * 0.9, True),
        (L / 2, 0.5, 1.2, 1.5, W / 2 * 0.96, W / 2 * 0.96, W / 2 * 0.9, False),
    ], TRUCK, OTHER_GLASS)
    cargo = box_mesh(-cab_l / 2 + 0.05, 0, 0.55, L - cab_l - 0.2, W, H - 0.55, (176, 178, 182), 0.2)
    return Mesh.concat([cab, cargo] + _wheels(L, W, 0.5, (-L / 2 + 1.0, -L / 2 + 2.2, L / 2 - 1.3)))


def bus_mesh(L=11.0, W=2.55, H=3.2) -> Mesh:
    hl, hw = L / 2, W / 2
    secs = [
        (-hl, 0.35, 1.2, H * 0.97, hw, hw, hw * 0.96, False),
        (-hl + 0.4, 0.35, 1.25, H, hw, hw, hw * 0.96, True),
        (hl - 0.5, 0.35, 1.25, H, hw, hw, hw * 0.96, True),
        (hl, 0.35, 1.05, H * 0.95, hw * 0.97, hw * 0.97, hw * 0.93, True),
    ]
    return Mesh.concat([loft_mesh(secs, (170, 186, 200), OTHER_GLASS, 0.3)] + _wheels(L, W, 0.5, (-hl + 2.3, hl - 2.4)))


def person_mesh(phase: float = 0.0, height: float = 1.72, color=PERSON) -> Mesh:
    """걷는 사람(다리와 팔이 phase에 따라 흔들린다). 원점은 발 가운데 바닥, +X가 앞."""
    k = height / 1.72
    sw = 0.45 * math.sin(phase)
    parts = []
    hip = 0.92 * k
    for side, s in ((0.1, sw), (-0.1, -sw)):
        foot = (hip * math.sin(s), side * k, 0.0)
        parts.append(cylinder_mesh(foot, (0, side * k, hip), 0.075 * k, 5, (80, 84, 92)))
    parts.append(box_mesh(0, 0, hip - 0.04 * k, 0.24 * k, 0.40 * k, 0.56 * k, color, 0.2, top_scale=1.08))
    for side, s in ((0.25, -sw), (-0.25, sw)):
        sh = (0, side * k, 1.42 * k)
        hand = (0.55 * k * math.sin(s * 0.8), side * 1.05 * k, 1.42 * k - 0.55 * k * math.cos(s * 0.8))
        parts.append(cylinder_mesh(sh, hand, 0.05 * k, 5, color))
    parts.append(sphere_mesh((0, 0, 1.6 * k), 0.12 * k, (150, 170, 200), rings=4, sides=7))
    return Mesh.concat(parts)


def bike_mesh(L=1.75, motor=False, phase=0.0) -> Mesh:
    r = 0.34 if not motor else 0.32
    parts = []
    for x in (-L / 2 + r, L / 2 - r):
        parts.append(cylinder_mesh((x, -0.04, r), (x, 0.04, r), r, 10, TIRE))
    frame_col = BIKE
    parts.append(box_mesh(0, 0, r * 0.9, L - 2 * r, 0.12 if not motor else 0.35, 0.35 if not motor else 0.45,
                          frame_col, 0.4))
    rider = person_mesh(phase if not motor else 0.0, 1.55)
    rider.V = rider.V + np.array([-0.1, 0, 0.45])
    rider.V[:, 2] = np.maximum(rider.V[:, 2], r * 1.2)
    parts.append(rider)
    return Mesh.concat(parts)


def light_pole_mesh(state_bgr, horizontal: bool, height=5.2, arm=3.0) -> tuple[Mesh, np.ndarray]:
    """신호등 기둥(로컬 원점=기둥 밑, 팔은 -Y쪽, 즉 오른쪽 도로 위로 뻗음). 켜진 등의 로컬 위치도 돌려준다."""
    parts = [cylinder_mesh((0, 0, 0), (0, 0, height), 0.11, 6, POLE, 0.3),
             cylinder_mesh((0, 0, height - 0.2), (0, -arm, height - 0.2), 0.07, 6, POLE, 0.3)]
    hx = (0.35, 1.05, 0.38) if horizontal else (0.35, 0.38, 1.05)
    cy = -arm + 0.3
    z0 = height - 0.2 - (0.2 if horizontal else 0.55)
    parts.append(box_mesh(0.0, cy, z0, hx[0], hx[1], hx[2], (40, 40, 44), 0.2))
    lamp = np.array([-0.19, cy, z0 + hx[2] / 2])
    parts.append(box_mesh(-0.18, cy, lamp[2] - 0.13, 0.02, 0.26, 0.26, state_bgr, 0.0))
    return Mesh.concat(parts), lamp


def load_vehicle_model(path: str | Path, paint_bgr: tuple | None = None) -> tuple[Mesh, Mesh]:
    """scripts/convert_vehicle_model.py가 만든 npz를 (높은 LOD, 낮은 LOD) 메시로 읽는다."""
    d = np.load(str(path))
    out = []
    for tag in ("", "_low"):
        R = d["R" + tag]
        C = d["C" + tag].copy()
        if paint_bgr is not None:
            C[R == 1] = paint_bgr
        shiny = np.select([R == 1, R == 2, R == 4], [0.5, 0.35, 0.6], 0.12)
        out.append(Mesh(d["V" + tag].astype(np.float64), d["F" + tag].astype(np.int32), C, shiny))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# 물체가 바닥 영상에 늘어진 자리(IPM 번짐) 마스크
# ---------------------------------------------------------------------------
def stretch_mask(cameras, owner_u8: list[np.ndarray], grid: BevGrid, tracks: list[Track], scale: float = 1.0,
                 far_m: float = 60.0) -> np.ndarray | None:
    """추적 중인 물체가 카메라 영상에서 가린 바닥(IPM에서 물체가 바닥으로 길게 늘어진 자리)을 탑뷰 마스크로.

    카메라 i에서 물체 상자(클래스 표시 크기, 15% 크게)의 윗면 꼭짓점을 바닥으로 투영하면, 바닥 위 상자와
    그 투영점을 감싼 볼록 다각형이 카메라 i의 영상에서 물체에 가려진 바닥이다. 그 다각형 중 카메라 i가
    맡은 셀(owner)만 1로 둔다. 검출 상자 마스크와 달리 검출이 한두 프레임 빠져도 트랙이 있으면 지운다.
    """
    if not tracks:
        return None
    H, W = grid.height, grid.width
    out = np.zeros((H, W), np.uint8)
    tmp = np.zeros((H, W), np.uint8)
    far = far_m * scale
    for cam, own in zip(cameras, owner_u8):
        c = cam.position
        fwd = cam.R_vc[:2, 2]
        polys = []
        for tr in tracks:
            L, Wd, Ht = (v * scale for v in CLASS_DIMS.get(tr.label, CLASS_DIMS["car"]))
            L, Wd = L * 1.15, Wd * 1.15
            x, y = float(tr.xy[0]), float(tr.xy[1])
            base = np.array([[x + L / 2, y + Wd / 2], [x - L / 2, y + Wd / 2], [x - L / 2, y - Wd / 2],
                             [x + L / 2, y - Wd / 2]])
            rel = base - c[:2]
            if np.all(rel @ fwd < 0):
                continue  # 카메라 뒤
            if Ht >= c[2] * 0.95:
                q = c[:2] + rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9) * far
            else:
                q = c[:2] + rel * (c[2] / (c[2] - Ht))
                d = np.linalg.norm(q - c[:2], axis=1, keepdims=True)
                q = c[:2] + (q - c[:2]) * np.minimum(1.0, far / (d + 1e-9))
            hull = cv2.convexHull(grid.to_pixel(np.vstack([base, q])).astype(np.float32))
            polys.append(np.round(hull[:, 0, :] * 4).astype(np.int32))
        if not polys:
            continue
        tmp[:] = 0
        cv2.fillPoly(tmp, polys, 1, cv2.LINE_8, shift=2)
        cv2.bitwise_or(out, cv2.bitwise_and(tmp, own), dst=out)
    return out


# ---------------------------------------------------------------------------
# 렌더러
# ---------------------------------------------------------------------------
class RendererV15:
    """v15 리얼리티 뷰를 닮은 3D 화면. Renderer3D와 같은 render() 인자를 받는다(ego_speed가 더 있다).

    grid: 바닥 영상과 도로 마스크가 놓인 탑뷰 격자
    scale: 장면 축척(모형이면 0.1 같은 값)
    """

    def __init__(self, grid: BevGrid, cfg: V15Config | None = None, scale: float = 1.0, timer=None):
        self.grid = grid
        self.cfg = (cfg or V15Config()).checked()
        self.scale = float(scale)
        self.timer = timer
        c = self.cfg
        self.out_w, self.out_h = c.width, c.height
        self.rw, self.rh = _even(c.width * c.render_scale), _even(c.height * c.render_scale)
        self.gw, self.gh = _even(self.rw * c.ground_scale), _even(self.rh * c.ground_scale)
        self._near = NEAR_M * self.scale
        # 자차 모델(없으면 저폴리 승용차를 흰색으로)
        path = Path(c.ego_model) if c.ego_model else DEFAULT_EGO_MODEL
        if path.exists():
            self.ego_hi, self.ego_lo = load_vehicle_model(path, c.ego_paint)
        else:
            m = car_mesh(4.6, 1.9, 1.5, body=c.ego_paint or (242, 242, 242))
            m.V[:, 0] += 1.3
            self.ego_hi = self.ego_lo = m
        self.ego_model_name = path.stem if path.exists() else "builtin"
        for m in (self.ego_hi, self.ego_lo):
            m.V = m.V * self.scale
        self._mesh_cache: dict = {}
        # 시점 상태
        self.view = c.view
        self._pose = VIEWS[c.view]
        self._from: ViewPose | None = None
        self._trans_t = 0.0
        self._orbit_az = VIEWS["orbit"].azimuth
        self._t_last: float | None = None
        self._tour_next = c.tour_s if c.tour_s > 0 else None
        self._tour_i = 0
        # 시점별 캐시(카메라가 움직이지 않으면 다시 쓴다)
        self._cam_key = None
        self._ground_layer = None  # 탑뷰 크기 바닥 그림(색 보정 끝)
        self._ground_view = None  # 원근으로 편 바닥 + 하늘 + 안개
        self._ground_dirty = True
        self._ego_cache = None  # (색, 깊이, 마스크) 자차만 그린 그림
        self._asphalt = self._make_asphalt()
        self._cov_alpha = None
        self.ground_refreshes = 0
        self._pos: dict[int, np.ndarray] = {}  # 트랙 보간 위치
        self._yaw: dict[int, float] = {}
        self._phase: dict[int, float] = {}
        self._set_camera(self._pose)

    # ---- 시점 ----------------------------------------------------------
    def set_view(self, name: str, immediate: bool = False) -> None:
        """시점을 바꾼다. immediate가 아니면 transition_s 동안 옮겨 간다."""
        if name not in VIEWS:
            raise ValueError(f"모르는 시점: {name}")
        if name == self.view and self._from is None:
            return
        if name == "orbit":
            self._orbit_az = self._pose.azimuth
        self.view = name
        if immediate or self.cfg.transition_s <= 0:
            self._from = None
            self._pose = self._target_pose()
        else:
            self._from = self._pose
            self._trans_t = 0.0

    def handle_key(self, key: int) -> bool:
        """창 모드 키 입력(1~6). 시점을 바꿨으면 True."""
        name = VIEW_KEYS.get(key)
        if name is None:
            return False
        self.set_view(name)
        return True

    def _target_pose(self) -> ViewPose:
        p = VIEWS[self.view]
        if self.view == "orbit":
            p = dataclasses.replace(p, azimuth=self._orbit_az)
        return p

    def _advance(self, dt: float) -> None:
        c = self.cfg
        if self._tour_next is not None:
            self._tour_next -= dt
            if self._tour_next <= 0:
                self._tour_i = (self._tour_i + 1) % len(c.tour_views)
                self.set_view(c.tour_views[self._tour_i])
                self._tour_next = c.tour_s
        if self.view == "orbit":
            self._orbit_az = (self._orbit_az + c.orbit_deg_s * dt) % 360.0
        target = self._target_pose()
        if self._from is not None:
            self._trans_t += dt
            k = min(1.0, self._trans_t / max(c.transition_s, 1e-6))
            self._pose = _lerp_pose(self._from, target, k)
            if k >= 1.0:
                self._from = None
        else:
            self._pose = target

    @property
    def moving(self) -> bool:
        """카메라가 움직이는 중이면 True(바닥 표와 자차 그림을 매 프레임 다시 만든다)."""
        return self._from is not None or self.view == "orbit"

    def _set_camera(self, p: ViewPose) -> None:
        s = self.scale
        key = tuple(round(v, 4) for v in dataclasses.astuple(p))
        if key == self._cam_key:
            return
        self._cam_key = key
        az = math.radians(p.azimuth)
        tgt = np.array([p.target_x, 0.0, p.target_z]) * s
        pos = tgt + np.array([p.dist * math.cos(az), p.dist * math.sin(az), 0.0]) * s
        pos[2] = p.height * s
        fwd = tgt - pos
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        pitch = math.degrees(math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1])))
        self.cam = Camera("v15", self.rw, self.rh, intrinsics_from_hfov(self.rw, self.rh, p.hfov), pos,
                          yaw_deg=yaw, pitch_deg=pitch)
        self._R = self.cam.R_cv
        self._P = self.cam.position
        self._K = self.cam.K
        self.horizon_row, self._maps, self._fog_w, self._dist_px = self._build_ground_maps()
        self._background = self._build_background()
        self._ground_dirty = True
        self._ego_cache = None

    # ---- 투영 ----------------------------------------------------------
    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pc = (np.asarray(pts, np.float64).reshape(-1, 3) - self._P) @ self._R.T
        z = pc[:, 2]
        zs = np.where(z > 1e-9, z, 1e-9)
        uv = np.stack([self._K[0, 0] * pc[:, 0] / zs + self._K[0, 2], self._K[1, 1] * pc[:, 1] / zs + self._K[1, 2]], 1)
        return uv, z

    # ---- 바닥 표, 하늘, 안개 ---------------------------------------------
    def _build_ground_maps(self):
        g = self.grid
        # 바닥은 gw x gh 해상도로 편다. 그 화소 중심을 그리는 해상도(rw x rh)의 좌표로 바꿔 시선을 구한다
        u, v = np.meshgrid((np.arange(self.gw, dtype=np.float32) + 0.5) * (self.rw / self.gw),
                           (np.arange(self.gh, dtype=np.float32) + 0.5) * (self.rh / self.gh))
        K = self._K
        dx = (u - K[0, 2]) / K[0, 0]
        dy = (v - K[1, 2]) / K[1, 1]
        R = self.cam.R_vc
        dX = R[0, 0] * dx + R[0, 1] * dy + R[0, 2]
        dY = R[1, 0] * dx + R[1, 1] * dy + R[1, 2]
        dZ = R[2, 0] * dx + R[2, 1] * dy + R[2, 2]
        down = dZ < -1e-6
        rows_sky = np.nonzero(~down.all(axis=1))[0]
        h0 = int(rows_sky.max()) + 1 if rows_sky.size else 0
        if h0 >= self.gh:
            return self.gh, None, None, None
        t = -self._P[2] / dZ[h0:]
        X = self._P[0] + t * dX[h0:]
        Y = self._P[1] + t * dY[h0:]
        col = (g.y_max - Y) / g.res - 0.5
        row = (g.x_max - X) / g.res - 0.5
        outside = (col < -1) | (col > g.width) | (row < -1) | (row > g.height)
        mx = np.where(outside, -10.0, col).astype(np.float32)
        my = np.where(outside, -10.0, row).astype(np.float32)
        maps = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        # 안개 가중치: 카메라에서 바닥점까지 수평 거리
        dist = np.hypot(X - self._P[0], Y - self._P[1]) / self.scale
        c = self.cfg
        f = np.clip((dist - c.fog_start_m) / max(c.fog_end_m - c.fog_start_m, 1e-3), 0.0, 1.0) ** 1.3 * 0.9
        # 지평선 바로 아래는 안개가 가장 짙게 번지도록(먼 바닥 무늬의 계단 현상을 감춘다)
        fog_w = f.astype(np.float32)
        return h0, maps, fog_w, dist.astype(np.float32)

    def _build_background(self) -> np.ndarray:
        img = np.empty((self.gh, self.gw, 3), np.uint8)
        h0 = max(1, self.horizon_row)
        k = np.linspace(0.0, 1.0, h0)[:, None] ** 1.6
        sky = (1 - k) * np.array(SKY_TOP, float) + k * np.array(SKY_HORIZON, float)
        img[:h0] = sky.astype(np.uint8)[:, None, :]
        img[h0:] = FOG
        return img

    def _make_asphalt(self) -> np.ndarray:
        """카메라가 못 보는 바닥에 깔 아스팔트 무늬(탑뷰 크기, 한 번만 만든다)."""
        g = self.grid
        rng = np.random.default_rng(7)
        n = rng.normal(0, 1, (g.height, g.width)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 1.2) * 6.0
        base = np.empty((g.height, g.width, 3), np.float32)
        base[:] = ASPHALT
        return np.clip(base + n[:, :, None], 0, 255).astype(np.uint8)

    def _ground_layer_from(self, ground_img, road_mask, coverage, obj_mask=None) -> np.ndarray:
        """탑뷰 크기 바닥 그림: 실제 색 IPM 영상(채도 낮추고 조금 어둡게)을 아스팔트와 부드럽게 섞는다."""
        layer = self._asphalt
        if ground_img is None:
            return layer.copy()
        c = self.cfg
        img = ground_img
        if c.ground_saturation < 1.0:
            gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            img = cv2.addWeighted(img, c.ground_saturation, gray, 1.0 - c.ground_saturation, -6.0)
        if coverage is None:
            return img
        if self._cov_alpha is None or self._cov_alpha[0] is not coverage:
            a = cv2.GaussianBlur(coverage.astype(np.float32), (0, 0), max(1.0, 0.35 / self.grid.res))
            a = cv2.erode(a, np.ones((3, 3), np.uint8))  # 가장자리 검은 화소가 번지지 않게 조금 안쪽부터
            self._cov_alpha = (coverage, a, 1.0 - a)
        _, a, ia = self._cov_alpha
        if obj_mask is not None:
            # 검출 상자가 바닥으로 늘어진 자리(차체가 길게 번진 화소)는 아스팔트로 덮는다
            hole = cv2.GaussianBlur(obj_mask.astype(np.float32), (0, 0), max(1.0, 0.4 / self.grid.res))
            a = a * (1.0 - np.clip(hole * 2.5, 0.0, 1.0))
            ia = 1.0 - a
        return cv2.blendLinear(img, layer, a, ia)

    def _refresh_ground(self, ground_img, road_mask, coverage, obj_mask=None) -> None:
        if ground_img is not None or self._ground_layer is None:
            self._ground_layer = self._ground_layer_from(ground_img, road_mask, coverage, obj_mask)
        self._ground_dirty = True

    def _ground_view_now(self) -> np.ndarray:
        if self._ground_dirty or self._ground_view is None:
            img = self._background.copy()
            if self._maps is not None:
                h0 = self.horizon_row
                flat = cv2.remap(self._ground_layer, self._maps[0], self._maps[1], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=ASPHALT)
                fogimg = np.empty_like(flat)
                fogimg[:] = FOG
                img[h0:] = cv2.blendLinear(fogimg, flat, self._fog_w, 1.0 - self._fog_w)
            if (self.gw, self.gh) != (self.rw, self.rh):
                img = cv2.resize(img, (self.rw, self.rh), interpolation=cv2.INTER_LINEAR)
            self._ground_view = img
            self._ground_dirty = False
            self.ground_refreshes += 1
        return self._ground_view

    # ---- 면 명암 -------------------------------------------------------
    def _shade(self, P: np.ndarray, base: np.ndarray, shiny: np.ndarray, cn: np.ndarray) -> np.ndarray:
        """꼭짓점마다 햇빛 명암, 하늘빛, 하이라이트, 거리 안개를 입힌 색 (M,3,3) uint8.

        P: 꼭짓점 위치 (M,3,3), base: 면 색 (M,3), shiny: (M,), cn: 꼭짓점 법선 (M,3,3), 모두 차량 좌표.
        """
        view = self._P - P
        vd = np.linalg.norm(view, axis=2, keepdims=True) + 1e-12
        view /= vd
        # 양면 조명: 면 중심에서 카메라를 등진 법선은 뒤집는다
        fsign = np.sign(np.einsum("ij,ij->i", cn.sum(axis=1), view.sum(axis=1)) + 1e-12)
        n = cn * fsign[:, None, None]
        diff = np.clip(n @ SUN, 0.0, 1.0)
        sky = np.clip(n[..., 2], 0.0, 1.0)
        h = SUN + view
        h /= np.linalg.norm(h, axis=2, keepdims=True) + 1e-12
        sh = shiny[:, None]
        spec = np.clip(np.einsum("ijk,ijk->ij", n, h), 0.0, 1.0) ** 28 * sh
        rim = (1.0 - np.clip(np.einsum("ijk,ijk->ij", n, view), 0.0, 1.0)) ** 3 * 0.18 * sh
        k = 0.40 + 0.50 * diff + 0.14 * sky
        col = base.astype(np.float32)[:, None, :] * k[..., None] + (255.0 * (spec + rim))[..., None]
        cfg = self.cfg
        d = vd[..., 0] / self.scale
        f = np.clip((d - cfg.fog_start_m) / max(cfg.fog_end_m - cfg.fog_start_m, 1e-3), 0, 1) ** 1.3 * 0.85
        col = col * (1 - f[..., None]) + np.array(FOG, np.float32) * f[..., None]
        return np.clip(col, 0, 255).astype(np.uint8)

    def _draw_meshes(self, img, zbuf, items, mask=None) -> list[bool]:
        """(메시, yaw 회전 R2 또는 None, 위치 t3) 목록을 한 번에 그린다(파이썬 호출 수를 줄이려고 합친다).

        near보다 가까운 면과 화면에서 너무 먼 면은 뺀다. 물체마다 그린 면이 있었는지 돌려준다.
        """
        if not items:
            return []
        Vs, Fs, Cs, Ss, Ns, n0 = [], [], [], [], [], 0
        for m, R2, t3 in items:
            V = m.V.copy()
            cn = m.corner_normals()
            if R2 is not None:
                V[:, :2] = V[:, :2] @ R2.T
                cn = cn.copy()
                cn[..., :2] = cn[..., :2] @ R2.T
            V += t3
            Vs.append(V)
            Fs.append(m.F + n0)
            Cs.append(m.C)
            Ss.append(m.shiny)
            Ns.append(cn)
            n0 += len(V)
        V = np.vstack(Vs)
        F = np.vstack(Fs)
        uv, z = self.project(V)
        lim = 8 * max(self.rw, self.rh)
        keep = (z[F] >= self._near).all(axis=1) & (np.abs(uv[F]) < lim).all(axis=(1, 2))
        drawn, start = [], 0
        for f in Fs:
            drawn.append(bool(keep[start:start + len(f)].any()))
            start += len(f)
        if not keep.any():
            return drawn
        F = F[keep]
        col = self._shade(V[F], np.vstack(Cs)[keep], np.concatenate(Ss)[keep], np.concatenate(Ns)[keep])
        raster_vc(img, zbuf, uv, 1.0 / np.maximum(z, 1e-6), F, col, mask=mask)
        return drawn

    def _draw_mesh(self, img, zbuf, m: Mesh, R2: np.ndarray | None, t3: np.ndarray, mask=None) -> bool:
        return self._draw_meshes(img, zbuf, [(m, R2, t3)], mask)[0]

    # ---- 자차 ----------------------------------------------------------
    def _ego_layer(self):
        """자차만 그린 (색, 깊이, 마스크). 카메라가 그대로면 다시 쓴다(가장 비싼 메시라서)."""
        if self._ego_cache is None:
            img = np.zeros((self.rh, self.rw, 3), np.uint8)
            z = np.zeros((self.rh, self.rw), np.float32)
            mask = np.zeros((self.rh, self.rw), np.uint8)
            m = self.ego_lo if self.moving else self.ego_hi
            self._draw_mesh(img, z, m, None, np.zeros(3), mask=mask)
            self._ego_cache = (img, z, mask)
        return self._ego_cache

    # ---- 트랙 ----------------------------------------------------------
    def _smooth_tracks(self, tracks: list[Track], dt: float, ego_speed: float | None):
        c = self.cfg
        a = 1.0 - math.exp(-dt / max(c.interp_tau, 1e-3)) if c.interp else 1.0
        out = []
        pos, yaw, phase = {}, {}, {}
        for tr in tracks:
            target = np.asarray(tr.xy, float)
            prev = self._pos.get(tr.tid)
            xy = target.copy() if (prev is None or dt <= 0) else prev + a * (target - prev)
            pos[tr.tid] = xy
            v = np.asarray(tr.vel, float).copy()
            if ego_speed is not None:
                v[0] += ego_speed * self.scale  # 상대 속도 + 자차 속도 = 땅에 대한 속도
            sp = float(np.linalg.norm(v))
            y_prev = self._yaw.get(tr.tid)
            moving = sp > max(tr.min_speed, 0.6 * self.scale) and (ego_speed is not None or tr.label == "person")
            if moving:
                y_new = math.atan2(v[1], v[0])
                if tr.label not in ("person",) and ego_speed is None:
                    y_new = 0.0
            else:
                y_new = 0.0 if y_prev is None else y_prev
            if y_prev is None:
                yv = y_new
            else:
                d = (y_new - y_prev + math.pi) % (2 * math.pi) - math.pi
                yv = y_prev + d * min(1.0, a * 0.8 + 0.05)
            yaw[tr.tid] = yv
            phase[tr.tid] = self._phase.get(tr.tid, 0.0) + dt * sp / self.scale * 3.2
            out.append((tr, xy, yv, phase[tr.tid], sp))
        self._pos, self._yaw, self._phase = pos, yaw, phase
        return out

    def _object_mesh(self, label: str, phase: float) -> Mesh:
        s = self.scale
        if label == "person":
            key = ("person", int(phase / (2 * math.pi) * 12) % 12)
            ph = key[1] / 12 * 2 * math.pi
        elif label in ("bicycle", "motorcycle"):
            key = (label, int(phase / (2 * math.pi) * 8) % 8)
            ph = key[1] / 8 * 2 * math.pi
        else:
            key = (label, 0)
            ph = 0.0
        m = self._mesh_cache.get(key)
        if m is None:
            if label == "person":
                m = person_mesh(ph)
            elif label == "bicycle":
                m = bike_mesh(1.75, False, ph)
            elif label == "motorcycle":
                m = bike_mesh(2.0, True, ph)
            elif label == "truck":
                m = truck_mesh()
            elif label == "bus":
                m = bus_mesh()
            else:
                m = car_mesh()
            m = Mesh(m.V * s, m.F, m.C, m.shiny)
            self._mesh_cache[key] = m
        return m

    # ---- 바닥 위 그림(차선, 경로, 그림자) -----------------------------------
    def _ground_poly(self, pts_xy: np.ndarray) -> np.ndarray | None:
        uv, z = self.project(np.c_[pts_xy, np.zeros(len(pts_xy))])
        if np.any(z < self._near) or np.abs(uv).max() > 8 * max(self.rw, self.rh):
            return None
        return np.round(uv * 4).astype(np.int32)  # 소수 2비트(shift=2)로 매끄럽게

    def _ribbon(self, pts: np.ndarray, half_w: float) -> list[np.ndarray]:
        """곡선(N,2)을 폭 2*half_w 띠의 사각형 조각(바닥 좌표) 목록으로."""
        if len(pts) < 2:
            return []
        d = np.gradient(pts, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) * half_w
        L, R = pts + nrm, pts - nrm
        return [np.array([L[i], L[i + 1], R[i + 1], R[i]]) for i in range(len(pts) - 1)]

    def _clip_visible(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        _, z = self.project(np.c_[pts, np.zeros(len(pts))])
        return pts[z > self._near + 0.5 * self.scale]

    def _draw_lanes(self, canvas: np.ndarray, lanes) -> None:
        s = self.scale
        step = 0.5 * s
        polys = {LANE_WHITE: [], LANE_YELLOW: []}
        for ln in lanes:
            col = LANE_YELLOW if ln.color == "yellow" else LANE_WHITE
            for run in dash_runs(ln.points(step / 2), ln.style == "dashed", s):
                run = self._clip_visible(run)
                for q in self._ribbon(run, 0.08 * s):
                    p = self._ground_poly(q)
                    if p is not None:
                        polys[col].append(p)
        for col, ps in polys.items():
            if ps:
                cv2.fillPoly(canvas, ps, col, cv2.LINE_AA, shift=2)

    def _path_center(self, lanes) -> np.ndarray:
        """계획 경로 가운데 선: 내 차선 양쪽 경계의 가운데(없으면 한쪽에서 1.75m, 둘 다 없으면 직진)."""
        s = self.scale
        x = np.linspace(EGO_FRONT_M * s, self.cfg.path_len_m * s, 40)
        left = right = None
        for ln in lanes or []:
            if ln.side == "left":
                left = ln
            elif ln.side == "right":
                right = ln
        if left is not None and right is not None:
            y = (left.y_at(x) + right.y_at(x)) / 2
        elif left is not None:
            y = left.y_at(x) - 1.75 * s
        elif right is not None:
            y = right.y_at(x) + 1.75 * s
        else:
            y = np.zeros_like(x)
        y = y - y[0] * np.clip(1 - (x - x[0]) / (12 * s), 0, 1)  # 자차 앞에서 시작해 차선 가운데로 이어지게
        return np.stack([x, y], 1)

    def _draw_path_and_shadows(self, canvas: np.ndarray, lanes, objs) -> None:
        """경로 띠(파랑, 멀수록 흐림)와 그림자(어둡게)를 한 번에 섞는다. 둘 다 바닥에 깔린다."""
        s = self.scale
        H, W = canvas.shape[:2]
        shadow = np.zeros((H, W), np.uint8)
        path = np.zeros((H, W), np.uint8)
        if self.cfg.shadows:
            sun_off = -SUN[:2] / SUN[2]  # 높이 1당 그림자가 밀리는 거리
            items = [(EGO_FRONT_M / 2 * s - 0.6 * s, 0.0, 0.0, 5.13 * s, 2.0 * s, 1.8 * s)]
            for tr, xy, yaw, _, _ in objs:
                L, Wd, Ht = CLASS_DIMS.get(tr.label, CLASS_DIMS["car"])
                items.append((xy[0], xy[1], yaw, L * s, Wd * s, Ht * s))
            for x, y, yaw, L, Wd, Ht in items:
                cy, sy = math.cos(yaw), math.sin(yaw)
                R2 = np.array([[cy, -sy], [sy, cy]])
                base = np.array([[L / 2, Wd / 2], [-L / 2, Wd / 2], [-L / 2, -Wd / 2], [L / 2, -Wd / 2]]) * 1.05 @ R2.T
                base += [x, y]
                cast = base + sun_off * Ht * 0.55
                hull = cv2.convexHull(np.vstack([base, cast]).astype(np.float32))[:, 0, :]
                p = self._ground_poly(hull.astype(float))
                if p is not None:
                    cv2.fillPoly(shadow, [p], 110, cv2.LINE_AA, shift=2)
                pc = self._ground_poly(base * 0.92 + np.array([x, y]) * 0.08)
                if pc is not None:
                    cv2.fillPoly(shadow, [pc], 170, cv2.LINE_AA, shift=2)
        if self.cfg.path:
            ctr = self._clip_visible(self._path_center(lanes))
            quads = self._ribbon(ctr, 0.95 * s)
            n = max(1, len(quads))
            for i, q in enumerate(quads):
                p = self._ground_poly(q)
                if p is not None:
                    a = int(150 * (1 - i / n) ** 1.4 + 20)
                    cv2.fillPoly(path, [p], a, cv2.LINE_AA, shift=2)
        both = cv2.bitwise_or(shadow, path)
        x0, y0, bw, bh = cv2.boundingRect(both)
        if bw == 0 or bh == 0:
            return
        k = max(3, int(round(5 * self.cfg.render_scale)) | 1)
        x0, y0 = max(x0 - k, 0), max(y0 - k, 0)
        x1, y1 = min(x0 + bw + 2 * k, W), min(y0 + bh + 2 * k, H)
        roi = canvas[y0:y1, x0:x1]
        if shadow[y0:y1, x0:x1].any():
            sh = cv2.GaussianBlur(shadow[y0:y1, x0:x1], (0, 0), 2.5 * self.cfg.render_scale + 0.5)
            f = cv2.cvtColor(cv2.subtract(255, cv2.multiply(sh, 0.6)), cv2.COLOR_GRAY2BGR)
            cv2.multiply(roi, f, dst=roi, scale=1.0 / 255)
        pa = path[y0:y1, x0:x1]
        if pa.any():
            a = pa.astype(np.float32) * (1.0 / 255)
            blue = _solid_like(canvas, PATH_BLUE)[y0:y1, x0:x1]
            roi[:] = cv2.blendLinear(blue, roi, a, 1.0 - a)

    # ---- 신호등 ---------------------------------------------------------
    def _light_item(self, lt):
        """신호등 하나의 (메시, 기둥 밑 위치, 켜진 등의 로컬 위치)."""
        from .traffic_lights import STATE_BGR

        s = self.scale
        col = STATE_BGR.get(lt.state, STATE_BGR["unknown"])
        lit = tuple(int(min(255, c * 1.25 + 30)) for c in col)
        key = ("light", lt.state, lt.orientation)
        m = self._mesh_cache.get(key)
        if m is None:
            mesh, lamp = light_pole_mesh(lit, lt.orientation == "horizontal")
            m = (Mesh(mesh.V * s, mesh.F, mesh.C, mesh.shiny), lamp * s)
            self._mesh_cache[key] = m
        mesh, lamp = m
        # 등이 신호등 위치(xy)에 오도록 기둥을 왼쪽에 세우고 팔을 오른쪽(도로 위)으로 뻗는다
        xy = np.asarray(lt.xy, float)
        return mesh, np.array([xy[0], xy[1] - lamp[1], 0.0]), lamp

    # ---- 한 프레임 -----------------------------------------------------
    def render(self, tracks: list[Track], ground_img=None, road_mask=None, coverage=None, refresh_ground: bool = True,
               t: float | None = None, lanes=None, lights=None, ego_speed: float | None = None,
               obj_mask: np.ndarray | None = None) -> np.ndarray:
        """화면 한 장(width x height BGR). ego_speed(m/s, 실차 기준)를 주면 물체 방향과 속도 표시에 쓴다."""
        c = self.cfg
        dt = 0.0 if (t is None or self._t_last is None) else max(0.0, float(t) - self._t_last)
        if t is not None:
            self._t_last = float(t)
        self._advance(dt)
        self._set_camera(self._pose)
        timer = self.timer
        # 1) 바닥
        if refresh_ground or self._ground_layer is None:
            if timer is not None:
                with timer.stage("render3d_ground"):
                    self._refresh_ground(ground_img, road_mask, coverage, obj_mask)
            else:
                self._refresh_ground(ground_img, road_mask, coverage, obj_mask)
        canvas = self._ground_view_now().copy()
        objs = self._smooth_tracks(tracks, dt, ego_speed)
        # 2) 바닥 위: 차선, 경로, 그림자
        if lanes:
            self._draw_lanes(canvas, lanes)
        self._draw_path_and_shadows(canvas, lanes, objs)
        # 3) 자차(캐시) 얹기
        ego_img, ego_z, ego_mask = self._ego_layer()
        cv2.copyTo(ego_img, ego_mask, canvas)
        zbuf = ego_z.copy()
        # 4) 물체와 신호등 기둥: 한 번에 모아 z-버퍼로 그린다
        items, meta = [], []
        for tr, xy, yaw, phase, sp in objs:
            m = self._object_mesh(tr.label, phase)
            cy, sy = math.cos(yaw), math.sin(yaw)
            items.append((m, np.array([[cy, -sy], [sy, cy]]), np.array([xy[0], xy[1], 0.0])))
            meta.append(("track", tr, xy))
        for lt in lights or []:
            if lt.xy is not None:
                m, base, lamp = self._light_item(lt)
                items.append((m, None, base))
                meta.append(("light", lt, lamp + base))
        drawn = self._draw_meshes(canvas, zbuf, items)
        labels, glows = [], []
        from .traffic_lights import STATE_BGR

        for ok, (kind, obj, p) in zip(drawn, meta):
            if not ok:
                continue
            if kind == "track" and c.show_ids:
                h = CLASS_DIMS.get(obj.label, CLASS_DIMS["car"])[2] * self.scale
                labels.append((np.array([p[0], p[1], h + 0.3 * self.scale]), str(obj.tid)))
            elif kind == "light" and obj.state in ("red", "yellow", "green"):
                uv, z = self.project(p[None])
                if z[0] > self._near:
                    glows.append((uv[0], z[0], STATE_BGR[obj.state]))
        # 5) 키우기, 빛 번짐, 글자
        if (self.rw, self.rh) != (self.out_w, self.out_h):
            canvas = cv2.resize(canvas, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        k = self.out_w / self.rw
        for uv, z, col in glows:
            r = int(np.clip(0.9 * self.scale * self._K[0, 0] / z * k, 3, 40))
            _glow(canvas, (int(uv[0] * k), int(uv[1] * k)), r, col)
        for p3, text in labels:
            uv, z = self.project(p3[None])
            if z[0] > self._near:
                cv2.putText(canvas, text, (int(uv[0, 0] * k) - 6, int(uv[0, 1] * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (240, 240, 240), 1, cv2.LINE_AA)
        if c.show_ui:
            self._draw_ui(canvas, lights, ego_speed)
        return canvas

    # ---- 화면 표시 -----------------------------------------------------
    def _draw_ui(self, canvas: np.ndarray, lights, ego_speed: float | None) -> None:
        H, W = canvas.shape[:2]
        f = cv2.FONT_HERSHEY_DUPLEX
        if ego_speed is not None:
            kmh = int(round(abs(ego_speed) * 3.6))
            txt = str(kmh)
            (tw, th), _ = cv2.getTextSize(txt, f, 1.7, 2)
            cv2.putText(canvas, txt, (28, 20 + th), f, 1.7, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.putText(canvas, "km/h", (30 + tw + 6, 20 + th), f, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            x = 30
            for g in "PRND":
                on = g == "D"
                cv2.putText(canvas, g, (x, 26 + th + 26), f, 0.55, (250, 250, 250) if on else (130, 130, 130),
                            1, cv2.LINE_AA)
                x += 20
        if lights:
            from .traffic_lights import main_light

            lt = main_light(lights)
            if lt is not None:
                horiz = lt.orientation == "horizontal"
                iw = 3 * 22 + 4 if horiz else 26
                draw_light_icon(canvas, lt.state, W - iw - 26, 16, lt.orientation)
        label = self.view.upper()
        (tw, _), _ = cv2.getTextSize(label, f, 0.45, 1)
        cv2.putText(canvas, label, (W - tw - 16, H - 14), f, 0.45, (200, 200, 205), 1, cv2.LINE_AA)


_SOLID: dict = {}


def _solid_like(img: np.ndarray, col: tuple) -> np.ndarray:
    """img와 같은 크기의 단색 영상(크기와 색별로 한 번 만들어 다시 쓴다, 읽기 전용)."""
    key = (img.shape, col)
    out = _SOLID.get(key)
    if out is None:
        if len(_SOLID) > 64:
            _SOLID.clear()
        out = np.empty(img.shape, np.uint8)
        out[:] = col
        _SOLID[key] = out
    return out


def _glow(img: np.ndarray, c: tuple, r: int, col: tuple) -> None:
    """켜진 등 둘레의 빛 번짐(작은 사각형 안에서만 더한다)."""
    H, W = img.shape[:2]
    R = r * 3
    x0, y0, x1, y1 = max(c[0] - R, 0), max(c[1] - R, 0), min(c[0] + R + 1, W), min(c[1] + R + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / float(r * r)
    a = (np.exp(-d2 * 0.9) * 0.9)[:, :, None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    roi += a * np.array(col, np.float32)
    img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
