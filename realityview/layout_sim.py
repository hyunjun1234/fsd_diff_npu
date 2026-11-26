from __future__ import annotations

import math
from functools import lru_cache
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bev import BevGrid, merge_cross_camera, project_detections
from .camera_model import Camera, fisheye_f_from_fov
from .tracker import BevTracker
from .types import Detection

# 카메라 배치(대수, 자리, 렌즈)를 바꿔 가며 주변 물체가 화면에 얼마나 끊김 없이 나오는지 재는 시뮬레이션.
#
# 흐름
#   1) 배치(Layout): 렌즈(LENSES)와 장착 자리(MOUNTS)를 골라 Camera 목록을 만든다.
#   2) 장면(Scenario): 자차 주변을 움직이는 물체(Actor)의 경로를 정한다. 좌표는 자차 기준(차량 좌표계)이다.
#   3) 정답 상자: 프레임마다 카메라별로 물체 상자를 계산한다. 물체 겉면에 점을 찍어 화각과 영상 안에 들어오고
#      자차 차체, 창틀, 다른 물체에 가리지 않는 점만으로 상자를 만든다(object_boxes의 partial 규칙).
#      일부만 보이는 물체는 보이는 부분만 상자가 되고, 보이는 비율이 함께 남는다.
#   4) 검출: 정답 상자에 화소 잡음을 섞고, NPU 입력(640) 기준 상자 크기가 작을수록, 보이는 비율이 낮을수록
#      더 자주 놓친다(DetModel).
#   5) 추적: 실제 파이프라인과 같은 함수(project_detections -> merge_cross_camera -> BevTracker)를 그대로 쓴다.
#   6) 지표: 화면 범위 안의 모든 물체(카메라에 안 보이는 것 포함)에 대해 화면에 나온 비율, 끊긴 시간,
#      ID 바뀜, 중복 표시, 위치 오차를 센다.
#
# 렌즈 값은 보정 전 근사다. 어안 렌즈는 등거리 투영(r = f x theta, 왜곡 계수 0)으로 본다.

VEHICLES = ("car", "truck", "bus")


# ---------------------------------------------------------------------------
# 렌즈
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lens:
    model: str  # "pinhole" 또는 "fisheye"
    width: int
    height: int
    f: float  # 초점거리(px)
    max_fov_deg: float | None = None  # 어안 렌즈가 보는 전체 화각(대각). None이면 영상 대각 끝까지

    @property
    def hfov_deg(self) -> float:
        half = self.width / 2.0
        if self.model == "pinhole":
            return 2.0 * math.degrees(math.atan(half / self.f))
        return 2.0 * math.degrees(half / self.f)

    @property
    def vfov_deg(self) -> float:
        half = self.height / 2.0
        if self.model == "pinhole":
            return 2.0 * math.degrees(math.atan(half / self.f))
        return 2.0 * math.degrees(half / self.f)

    def fov_limit(self) -> float | None:
        if self.model != "fisheye":
            return None
        if self.max_fov_deg is not None:
            return self.max_fov_deg
        diag = 2.0 * math.degrees(math.hypot(self.width / 2.0, self.height / 2.0) / self.f)
        return min(diag, 179.0)


def _fish(width: int, height: int, fov_deg: float, diagonal: bool, max_fov: float | None = None) -> Lens:
    return Lens("fisheye", width, height, fisheye_f_from_fov(width, height, fov_deg, diagonal), max_fov)


def _pin(width: int, height: int, hfov_deg: float) -> Lens:
    return Lens("pinhole", width, height, (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0))


# 영상 폭은 설정 파일과 같은 768px로 맞춘다.
LENSES: dict[str, Lens] = {
    # 저장소 설정과 같은 값(configs/four_cam.yaml, two_cam_current.yaml)
    "cm3_wide": Lens("pinhole", 768, 432, 311.0),  # Camera Module 3 Wide, 수평 102도
    "cm3": Lens("pinhole", 768, 432, 564.3),  # Camera Module 3, 수평 68.5도
    "ov5647_130": Lens("fisheye", 768, 576, 423.1, 140.0),  # ov5647 130도(대각으로 봄)
    # Arducam USB 카메라(제품 표기 화각을 등거리 어안으로 옮긴 값)
    "starvis2_86": _fish(768, 432, 86.0, False),  # STARVIS 2 IP65, 수평 86도
    "imx291_120": _fish(768, 432, 120.0, True),  # IMX291 120도(대각)
    "imx291_160": _fish(768, 432, 160.0, True),  # IMX291 160도(대각), 수평 약 139도
    "imx291_140": _fish(768, 432, 140.0, True),  # 160도 제품의 실제 화각이 표기보다 좁을 때를 보는 가정값
    "imx462_141": _fish(768, 432, 141.4, False),  # IMX462 CSI, 수평 141.4도
    # 테슬라 HW4 기준 화각(04 문서). 센서 비율 2896x1876에 맞춰 768x498
    "tesla_main_50": _pin(768, 498, 50.0),
    "tesla_wide_120": _fish(768, 498, 120.0, False),
    "tesla_bumper_140": _fish(768, 498, 140.0, False),
    "tesla_bpillar_90": _pin(768, 498, 90.0),
    "tesla_fender_80": _pin(768, 498, 80.0),
    "tesla_rear_130": _fish(768, 498, 130.0, False),
    # 좌우 렌즈의 수평 화각 하한을 찾을 때 쓰는 가정값(등거리 어안, 수평 화각으로 지정)
    **{f"h{h}": _fish(768, 432, float(h), False) for h in (126, 130, 134, 145, 150)},
}


# ---------------------------------------------------------------------------
# 장착 자리 (왼쪽 기준. 오른쪽은 y와 yaw 부호를 뒤집는다)
# ---------------------------------------------------------------------------
# 값: (위치 x, y, z m), yaw 도, 아래로 숙인 각 도. 테슬라 모델 Y 자리는 04 문서의 추정치다.
MOUNTS: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "wind_wide": ((1.80, -0.06, 1.45), 0.0, 3.0),  # 앞유리 상단, 전방 광각 자리
    "wind_main": ((1.80, 0.00, 1.45), 0.0, 3.0),  # 앞유리 상단, 전방 메인 자리
    "bumper": ((3.72, 0.00, 0.55), 0.0, 12.0),  # 앞범퍼 그릴 위
    "rear": ((-0.95, 0.00, 1.00), 180.0, 20.0),  # 뒷번호판 위
    "fender": ((2.25, 0.97, 0.92), 135.0, 5.0),  # 앞펜더 리피터, 뒤쪽 옆을 봄
    "bpillar": ((1.20, 0.93, 1.15), 60.0, 5.0),  # B필러, 앞쪽 옆을 봄
    # 유리 안쪽 장착(07 문서의 첫 차량 시험 배치). 좌표는 추정이다.
    "rear_glass": ((-0.30, 0.00, 1.35), 180.0, 15.0),  # 뒷유리 상단 안쪽
    "side_glass": ((0.95, 0.82, 1.20), 90.0, 15.0),  # 2열 옆유리 앞쪽 안쪽
    # 사이드미러 아래(주차 보조 카메라가 달리는 자리). 좌표는 추정이다.
    "mirror": ((2.05, 1.03, 1.02), 90.0, 20.0),
}

# 유리 안쪽에 달면 창틀(기둥)이 좌우 시야를 막는다. 값: (창이 향한 방향 도, 그 방향에서 좌우로 보이는 각 도).
# 카메라를 돌려 달아도 창틀은 그대로이므로 카메라 방향이 아니라 창 방향을 기준으로 자른다. 각은 추정이다.
MOUNT_H_CLIP: dict[str, tuple[float, float]] = {"rear_glass": (180.0, 65.0), "side_glass": (90.0, 62.0)}


@dataclass(frozen=True)
class CamSpec:
    name: str
    lens: str
    mount: str
    yaw: float | None = None  # None이면 MOUNTS의 값
    pitch: float | None = None
    right: bool = False  # True면 오른쪽(왼쪽 자리의 y, yaw 부호를 뒤집음)
    link: str = "usb"  # "csi" 또는 "usb"
    h_clip: tuple[float, float] | None = None  # 좌우 시야 한계 (가운데 방향, 반각). None이면 MOUNT_H_CLIP의 값

    def clip(self) -> tuple[float, float] | None:
        """창틀이 막는 좌우 시야 (가운데 방향 도, 반각 도). 한계가 없으면 None."""
        c = MOUNT_H_CLIP.get(self.mount) if self.h_clip is None else self.h_clip
        if c is None:
            return None
        return (-c[0], c[1]) if self.right else c

    def build(self) -> Camera:
        lens = LENSES[self.lens]
        (x, y, z), yaw0, pitch0 = MOUNTS[self.mount]
        yaw = yaw0 if self.yaw is None else self.yaw
        pitch = pitch0 if self.pitch is None else self.pitch
        if self.right:
            y, yaw = -y, -yaw
        K = np.array([[lens.f, 0, lens.width / 2.0], [0, lens.f, lens.height / 2.0], [0, 0, 1.0]])
        return Camera(
            name=self.name, width=lens.width, height=lens.height, K=K, position=np.array([x, y, z]),
            yaw_deg=yaw, pitch_deg=pitch, model=lens.model, max_fov_deg=lens.fov_limit(),
            source={"type": "picamera2" if self.link == "csi" else "usb"},
        )


def side_pair(lens: str, mount: str, yaw: float | None = None, pitch: float | None = None,
              prefix: str = "") -> list[CamSpec]:
    """좌우 한 쌍. prefix로 이름을 구분한다(예: bp_left, fd_left)."""
    return [
        CamSpec(f"{prefix}left", lens, mount, yaw, pitch, right=False),
        CamSpec(f"{prefix}right", lens, mount, yaw, pitch, right=True),
    ]


@dataclass(frozen=True)
class Layout:
    key: str
    title: str  # 표에 쓰는 짧은 설명
    specs: tuple[CamSpec, ...]

    def cameras(self) -> list[Camera]:
        return [s.build() for s in self.specs]

    def h_clips(self) -> list[tuple[float, float] | None]:
        return [s.clip() for s in self.specs]

    @property
    def n(self) -> int:
        return len(self.specs)

    @property
    def n_usb(self) -> int:
        return sum(s.link == "usb" for s in self.specs)


def _front(wide: str = "cm3_wide", main: bool = False) -> list[CamSpec]:
    out = [CamSpec("front", wide, "wind_wide", link="csi")]
    if main:
        out.append(CamSpec("front_main", "cm3", "wind_main", link="csi"))
    return out


def side_layout(key: str, title: str, lens: str, mount: str, yaw: float | None = None, pitch: float | None = None,
                main: bool = True, rear_lens: str = "imx291_160", rear_mount: str = "rear",
                front_wide: str = "cm3_wide") -> Layout:
    """전방(광각, main이면 메인도), 후방, 좌우 한 쌍으로 된 배치. 좌우 카메라의 렌즈, 자리, 방향을 바꿔 볼 때 쓴다."""
    return Layout(key, title, (*_front(front_wide, main), CamSpec("rear", rear_lens, rear_mount),
                               *side_pair(lens, mount, yaw, pitch)))


def make_layouts(side_pitch: float = 20.0, rear_lens: str = "imx291_160") -> dict[str, Layout]:
    """비교할 배치 목록. side_pitch는 정측면을 보는 좌우 카메라를 아래로 숙인 각이다."""
    rear = [CamSpec("rear", rear_lens, "rear")]
    side90 = lambda lens: side_pair(lens, "bpillar", 90.0, side_pitch)  # noqa: E731 - 짧은 묶음
    L: list[Layout] = [
        Layout("c2_now", "지금 2대: 전방 ov5647, 후방 CM3", (
            CamSpec("front", "ov5647_130", "wind_wide", link="csi"),
            CamSpec("rear", "cm3", "rear", link="csi"))),
        Layout("c4_fender", "4대: 전방, 후방, 펜더 좌우(뒤쪽 옆, CM3 Wide)", (
            *_front(), CamSpec("rear", "cm3_wide", "rear", link="csi"), *side_pair("cm3_wide", "fender"))),
        Layout("c4_side160", "4대: 전방, 후방, 정측면 좌우 160도", (*_front(), *rear, *side90("imx291_160"))),
        Layout("c4_now_side160", "4대: 지금 2대(전방 ov5647, 후방 CM3)에 정측면 좌우 160도", (
            CamSpec("front", "ov5647_130", "wind_wide", link="csi"),
            CamSpec("rear", "cm3", "rear", link="csi"), *side90("imx291_160"))),
        Layout("c5_fender86", "5대: 전방 2, 후방, 펜더 좌우 86도(뒤쪽 옆)", (
            *_front(main=True), *rear, *side_pair("starvis2_86", "fender"))),
        Layout("c5_side86", "5대: 전방 2, 후방, 정측면 좌우 86도", (*_front(main=True), *rear, *side90("starvis2_86"))),
        Layout("c5_side120", "5대: 전방 2, 후방, 정측면 좌우 120도", (*_front(main=True), *rear, *side90("imx291_120"))),
        Layout("c5_side160", "5대: 전방 2, 후방, 정측면 좌우 160도", (*_front(main=True), *rear, *side90("imx291_160"))),
        # 160도 제품의 실제 화각이 표기보다 좁을 때(대각 140도, 수평 약 122도)
        Layout("c5_side140", "5대: 후방과 정측면 좌우의 화각이 대각 140도일 때", (
            *_front(main=True), CamSpec("rear", "imx291_140", "rear"), *side90("imx291_140"))),
        Layout("c5_ov_side160", "5대: 전방 ov5647과 CM3, 후방, 정측면 좌우 160도", (
            *_front("ov5647_130", main=True), *rear, *side90("imx291_160"))),
        Layout("c6_nomain", "6대: 전방 광각, 후방, B필러 좌우, 펜더 좌우", (
            *_front(), *rear, *side_pair("starvis2_86", "bpillar", prefix="bp_"),
            *side_pair("starvis2_86", "fender", prefix="fd_"))),
        Layout("c6_side160_bumper", "6대: 5대(정측면 160도)에 범퍼", (
            *_front(main=True), *rear, *side90("imx291_160"), CamSpec("bumper", "imx291_160", "bumper"))),
        Layout("c7_tesla", "7대: 전방 2, 후방, B필러 좌우, 펜더 좌우", (
            *_front(main=True), *rear, *side_pair("starvis2_86", "bpillar", prefix="bp_"),
            *side_pair("starvis2_86", "fender", prefix="fd_"))),
        # 7대 배치의 좌우 카메라를 더 숙이거나(가까운 바닥이 보이게) 더 넓은 렌즈로 바꾼 경우
        Layout("c7_pitch20", "7대: B필러와 펜더 카메라를 20도 숙임", (
            *_front(main=True), *rear, *side_pair("starvis2_86", "bpillar", pitch=20.0, prefix="bp_"),
            *side_pair("starvis2_86", "fender", pitch=20.0, prefix="fd_"))),
        Layout("c7_wide120", "7대: B필러와 펜더 카메라를 120도 렌즈로, 20도 숙임", (
            *_front(main=True), *rear, *side_pair("imx291_120", "bpillar", pitch=20.0, prefix="bp_"),
            *side_pair("imx291_120", "fender", pitch=20.0, prefix="fd_"))),
        Layout("c8_tesla", "8대: 7대에 범퍼", (
            *_front(main=True), *rear, *side_pair("starvis2_86", "bpillar", prefix="bp_"),
            *side_pair("starvis2_86", "fender", prefix="fd_"), CamSpec("bumper", "imx291_160", "bumper"))),
        # 5대(정측면 160도)에 테슬라의 펜더 카메라 자리(뒤쪽 옆)와 범퍼 자리를 더한 경우
        Layout("c7_side160_fender", "7대: 5대(정측면 160도)에 펜더 좌우 86도(뒤쪽 옆)", (
            *_front(main=True), *rear, *side90("imx291_160"), *side_pair("starvis2_86", "fender", prefix="fd_"))),
        Layout("c8_side160_all", "8대: 5대(정측면 160도)에 펜더 좌우와 범퍼", (
            *_front(main=True), *rear, *side90("imx291_160"), *side_pair("starvis2_86", "fender", prefix="fd_"),
            CamSpec("bumper", "imx291_160", "bumper"))),
        # 유리 안쪽 장착(07 문서의 첫 차량 시험). 후방은 뒷유리 안쪽, 좌우는 2열 옆유리 안쪽이다.
        Layout("g4_cm3wide", "유리 안쪽 4대: 전방, 뒷유리, 2열 옆유리(모두 CM3 Wide)", (
            *_front(), CamSpec("rear", "cm3_wide", "rear_glass", link="csi"), *side_pair("cm3_wide", "side_glass"))),
        side_layout("g4_side160", "유리 안쪽 4대: 전방, 뒷유리 160도, 2열 옆유리 160도", "imx291_160", "side_glass",
                    main=False, rear_lens=rear_lens, rear_mount="rear_glass"),
        side_layout("g5_side160", "유리 안쪽 5대: 전방 2, 뒷유리 160도, 2열 옆유리 160도", "imx291_160", "side_glass",
                    rear_lens=rear_lens, rear_mount="rear_glass"),
        Layout("t8_ref", "테슬라 화각 그대로 8대(참고)", (
            CamSpec("front", "tesla_wide_120", "wind_wide", link="csi"),
            CamSpec("front_main", "tesla_main_50", "wind_main", link="csi"),
            CamSpec("rear", "tesla_rear_130", "rear"),
            *side_pair("tesla_bpillar_90", "bpillar", prefix="bp_"),
            *side_pair("tesla_fender_80", "fender", prefix="fd_"),
            CamSpec("bumper", "tesla_bumper_140", "bumper"))),
    ]
    return {x.key: x for x in L}


# ---------------------------------------------------------------------------
# 자차 차체가 가리는 곳
# ---------------------------------------------------------------------------
# 차체 치수는 04 문서(모델 Y). 불투명한 차체 윗면 높이(벨트라인, 보닛, 트렁크)는 추정값이다.
EGO_X_REAR, EGO_X_FRONT, EGO_HALF_W = -1.011, 3.783, 0.96
_EGO_TOP_X = np.array([-1.011, -0.60, 2.35, 3.783])  # 뒤끝, 뒷유리 아래, 앞유리 아래(카울), 앞끝
_EGO_TOP_Z = np.array([1.10, 1.05, 1.05, 0.80])


def ego_blocks(cam_pos: np.ndarray, pts: np.ndarray, skip: float = 0.25) -> np.ndarray:
    """카메라에서 점(N,3 또는 바닥 점 N,2)을 잇는 시선이 자차 차체(유리 아래 불투명한 부분)를 지나는지(N,) bool.

    차체는 바닥 네모(EGO_X_REAR~EGO_X_FRONT, 폭 2 x EGO_HALF_W) 위에 윗면 높이가 x에 따라 꺾은선으로 바뀌는 덩어리다.
    시선의 높이와 차체 윗면 높이는 둘 다 시선을 따라 구간별 직선이므로, 시선이 네모 안에 있는 구간의 양 끝과
    윗면이 꺾이는 x에서만 "시선 높이 < 윗면 높이"를 보면 된다.
    카메라가 차체 속에 묻혀 있으면(범퍼, 번호판 위) 카메라에서 skip m까지는 보지 않는다(스스로 가리지 않게).
    차체 윗면보다 위에 달린 카메라는 skip 없이 계산해서, 바로 아래 문짝 윗선에 가리는 바닥도 가린 것으로 본다.
    """
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if p.shape[1] == 2:
        p = np.concatenate([p, np.zeros((len(p), 1))], axis=1)
    if p.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    c = np.asarray(cam_pos, dtype=np.float64)
    if not (EGO_X_REAR <= c[0] <= EGO_X_FRONT and abs(c[1]) <= EGO_HALF_W
            and c[2] < float(np.interp(c[0], _EGO_TOP_X, _EGO_TOP_Z))):
        skip = 0.0  # 차체 윗면보다 위(B필러, 유리 안쪽)나 차체 밖에 달린 카메라는 처음부터 계산한다
    d = p - c
    length = np.linalg.norm(d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 시선(c + tau x d, tau 0~1)이 네모 안에 있는 구간 [lo, hi]
        tx1, tx2 = (EGO_X_REAR - c[0]) / d[:, 0], (EGO_X_FRONT - c[0]) / d[:, 0]
        ty1, ty2 = (-EGO_HALF_W - c[1]) / d[:, 1], (EGO_HALF_W - c[1]) / d[:, 1]
    x_par = np.abs(d[:, 0]) < 1e-12
    y_par = np.abs(d[:, 1]) < 1e-12
    x_in = EGO_X_REAR <= c[0] <= EGO_X_FRONT
    y_in = -EGO_HALF_W <= c[1] <= EGO_HALF_W
    lo_x = np.where(x_par, -np.inf if x_in else np.inf, np.minimum(tx1, tx2))
    hi_x = np.where(x_par, np.inf if x_in else -np.inf, np.maximum(tx1, tx2))
    lo_y = np.where(y_par, -np.inf if y_in else np.inf, np.minimum(ty1, ty2))
    hi_y = np.where(y_par, np.inf if y_in else -np.inf, np.maximum(ty1, ty2))
    lo = np.maximum(np.maximum(lo_x, lo_y), skip / np.maximum(length, 1e-9))
    hi = np.minimum(np.minimum(hi_x, hi_y), 1.0)
    has = hi > lo
    blocked = np.zeros(len(p), dtype=bool)
    cands = [lo, hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        for xk in _EGO_TOP_X[1:-1]:
            tk = (xk - c[0]) / d[:, 0]
            cands.append(np.where((tk > lo) & (tk < hi), tk, lo))
    for tau in cands:
        tau = np.where(has, tau, 0.0)
        x = c[0] + tau * d[:, 0]
        z = c[2] + tau * d[:, 2]
        blocked |= has & (z < np.interp(x, _EGO_TOP_X, _EGO_TOP_Z) - 1e-9)
    return blocked


# ---------------------------------------------------------------------------
# 정답 상자
# ---------------------------------------------------------------------------
# 두 가지 규칙이 있다.
#   full   : SimWorld._draw_object와 같다. 상자 꼭짓점 8개가 모두 카메라 앞(어안은 화각 안)에 있고 접지점이
#            영상 안에 있어야 한다. 옆 차로에서 나란히 달리는 차처럼 영상보다 넓게 걸친 물체는 빠진다.
#   partial: 물체 겉면 가운데 카메라를 향한 면(옆면, 앞뒤면, 윗면)에 점을 0.15m 간격으로 찍고, 화각과 영상 안에
#            들어오면서 자차 차체나 다른 물체에 가리지 않는 점만으로 상자를 만든다(기본). 실제 검출기가 화면
#            가장자리에 걸리거나 보닛 위로만 보이는 차를 보이는 부분만 상자로 잡는 것과 같다. 함께 돌려주는 값:
#              frac    보이는 비율. 카메라에서 본 물체의 겉보기 크기(입체각) 가운데 보이는 몫이다. 가리는 것이
#                      없고 영상 안에 다 들어오면 1이고, 검출 모형은 이 값이 낮으면 더 자주 놓친다.
#              contact 상자 아래 끝이 실제 접지점(물체 바닥)인지. 아래가 영상 밖으로 잘리거나 가리면 파이프라인은
#                      잘린 자리를 접지점으로 알고 거리를 계산하므로 위치가 어긋난다. contact="drop"이면 이런
#                      상자를 뺀다.
MIN_AREA = 60.0  # 영상 상자 최소 넓이(px^2). SimWorld와 같다


SURFACE_SPACING = 0.15  # 겉면 점 간격(m)
SURFACE_N_MIN, SURFACE_N_MAX = 6, 56  # 한 변에 찍는 점 수의 한계(11m 버스는 0.2m 간격이 된다)


@lru_cache(maxsize=128)
def _box_template(length: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """상자 겉면(옆면 2, 앞뒤면 2, 윗면)에 찍은 점. 좌표는 바닥 중심 기준이다.

    돌려주는 값: 점(P,3), 면 번호(P,), 그 점이 대표하는 넓이(P,), 바닥 줄(높이 0)인지(P,).
    면 번호: 0 = y가 작은 쪽 옆면, 1 = y가 큰 쪽 옆면, 2 = x가 작은 쪽 끝면, 3 = x가 큰 쪽 끝면, 4 = 윗면.
    모서리에도 점을 두어 상자 끝이 정확히 잡히게 하고, 넓이는 사다리꼴 규칙으로 나눈다(끝 점은 절반).
    """
    def axis(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        n = int(min(SURFACE_N_MAX, max(SURFACE_N_MIN, math.ceil((hi - lo) / SURFACE_SPACING) + 1)))
        w = np.ones(n)
        w[0] = w[-1] = 0.5
        return np.linspace(lo, hi, n), w * (hi - lo) / w.sum()

    a, wa = axis(-length / 2.0, length / 2.0)
    b, wb = axis(-width / 2.0, width / 2.0)
    c, wc = axis(0.0, height)
    pts, face, area = [], [], []

    def add(f: int, p: np.ndarray, w: np.ndarray) -> None:
        pts.append(p)
        face.append(np.full(len(p), f))
        area.append(w)

    A, C = np.meshgrid(a, c, indexing="ij")
    for f, v in ((0, -width / 2.0), (1, width / 2.0)):
        add(f, np.stack([A.ravel(), np.full(A.size, v), C.ravel()], axis=1), np.outer(wa, wc).ravel())
    B, C = np.meshgrid(b, c, indexing="ij")
    for f, v in ((2, -length / 2.0), (3, length / 2.0)):
        add(f, np.stack([np.full(B.size, v), B.ravel(), C.ravel()], axis=1), np.outer(wb, wc).ravel())
    A, B = np.meshgrid(a, b, indexing="ij")
    add(4, np.stack([A.ravel(), B.ravel(), np.full(A.size, height)], axis=1), np.outer(wa, wb).ravel())
    p = np.concatenate(pts)
    out = (p, np.concatenate(face), np.concatenate(area), p[:, 2] <= 1e-12)
    for v in out:
        v.setflags(write=False)
    return out


def fast_project(cam: Camera, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera.project와 같은 결과를 numpy만으로 낸다(왜곡 계수가 0일 때). 점이 많을 때 OpenCV 호출보다 빠르다.

    핀홀: u = fx x/z + cx. 어안(등거리): 반지름 = f x theta. 왜곡 계수가 있으면 Camera.project를 그대로 부른다.
    """
    if np.any(cam.dist != 0):
        return cam.project(pts)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pc = (pts - cam.position) @ cam.R_cv.T  # (N,3) 카메라 광학 좌표
    xc, yc, zc = pc[:, 0], pc[:, 1], pc[:, 2]
    front = zc > 1e-3
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        if cam.model == "fisheye":
            r = np.hypot(xc, yc)
            theta = np.arctan2(r, zc)
            front &= theta <= cam.fisheye_theta_max()
            k = np.where(r > 1e-12, theta / np.maximum(r, 1e-12), 1.0)
            uv = np.stack([fx * k * xc + cx, fy * k * yc + cy], axis=1)
        else:
            uv = np.stack([fx * xc / zc + cx, fy * yc / zc + cy], axis=1)
    return uv, front


def _corners(x, y, length, width, height) -> np.ndarray:
    """도로와 나란한 상자 N개의 꼭짓점 (N,8,3). 0~3은 바닥, 4~7은 윗면."""
    n = x.size
    sx = np.array([1.0, 1.0, -1.0, -1.0])
    sy = np.array([1.0, -1.0, -1.0, 1.0])
    bx = x[:, None] + sx[None, :] * (length[:, None] / 2.0)
    by = y[:, None] + sy[None, :] * (width[:, None] / 2.0)
    low = np.stack([bx, by, np.zeros((n, 4))], axis=-1)
    high = np.stack([bx, by, np.repeat(height[:, None], 4, axis=1)], axis=-1)
    return np.concatenate([low, high], axis=1)


def _in_h_clip(cam: Camera, pts: np.ndarray, h_clip: tuple[float, float]) -> np.ndarray:
    """점(N,3)이 카메라 자리에서 볼 때 h_clip = (가운데 방향 도, 반각 도) 안에 있는지(N,). 바닥에 내린 방향으로 잰다."""
    az = np.arctan2(pts[:, 1] - cam.position[1], pts[:, 0] - cam.position[0]) - math.radians(h_clip[0])
    az = (az + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(az) <= math.radians(h_clip[1])


def object_boxes(cam: Camera, x, y, length, width, height, rule: str = "partial", ego_body: bool = True,
                 min_frac: float = 0.05, h_clip: tuple[float, float] | None = None,
                 blockers: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """도로와 나란한 상자 N개의 영상 상자(N,4: x1,y1,x2,y2), 유효 여부(N,), 보이는 비율(N,), 접지점이 보이는지(N,).

    h_clip = (가운데 방향 도, 반각 도)를 주면 그 밖에 있는 점은 보이지 않는 것으로 한다(창틀에 가림).
    blockers = occluder_boxes()의 결과를 주면 다른 물체에 가린 점도 뺀다(partial 규칙에서만).
    """
    x, y, length, width, height = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x, y, length, width, height))
    n = x.size
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0, dtype=bool)
    w1, h1 = cam.width - 1.0, cam.height - 1.0
    if rule == "full":
        corners = _corners(x, y, length, width, height).reshape(-1, 3)
        uv, front = fast_project(cam, corners)
        if h_clip is not None:
            front = front & _in_h_clip(cam, corners, h_clip)
        uv = uv.reshape(n, 8, 2)
        ok = front.reshape(n, 8).all(axis=1)
        with np.errstate(invalid="ignore"):
            x1, y1 = uv[..., 0].min(axis=1), uv[..., 1].min(axis=1)
            x2, y2 = uv[..., 0].max(axis=1), uv[..., 1].max(axis=1)
            ok &= ~((x2 < 0) | (y2 < 0) | (x1 >= cam.width) | (y1 >= cam.height))
            boxes = np.stack([np.maximum(0.0, x1), np.maximum(0.0, y1), np.minimum(w1, x2), np.minimum(h1, y2)], axis=1)
            ok &= (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= MIN_AREA
            ok &= ~(y2 > h1)
        if ego_body:
            ok &= ~ground_contact_hidden(cam, boxes, ok)
        boxes[~ok] = 0.0
        return boxes, ok, ok.astype(np.float64), ok.copy()
    if rule != "partial":
        raise ValueError(f"rule은 full 또는 partial이어야 합니다: {rule}")
    cpos = np.asarray(cam.position, dtype=np.float64)
    boxes = np.zeros((n, 4))
    ok = np.zeros(n, dtype=bool)
    frac = np.zeros(n)
    contact = np.zeros(n, dtype=bool)
    live = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e5) & (np.abs(y) < 1e5)
    size = np.round(np.stack([length, width, height], axis=1), 4)
    p_all, o_all, w_all, b_all = [], [], [], []
    for L, W, H in np.unique(size[live], axis=0):
        mem = np.nonzero(live & (size[:, 0] == L) & (size[:, 1] == W) & (size[:, 2] == H))[0]
        tp, tf, ta, tb = _box_template(float(L), float(W), float(H))
        # 카메라가 각 면의 바깥쪽으로 떨어진 거리. 양수인 면만 카메라를 향한다(뒷면은 물체 자신에 가려 안 보인다).
        out_d = np.stack([(y[mem] - W / 2.0) - cpos[1], cpos[1] - (y[mem] + W / 2.0),
                          (x[mem] - L / 2.0) - cpos[0], cpos[0] - (x[mem] + L / 2.0),
                          np.full(mem.size, cpos[2] - H)], axis=1)  # (물체, 면 5개)
        hs = out_d[:, tf]  # (물체, 점)
        oi, pi = np.nonzero(hs > 1e-9)  # 물체 순서대로 나온다
        if oi.size == 0:
            continue
        p = tp[pi].copy()
        p[:, 0] += x[mem][oi]
        p[:, 1] += y[mem][oi]
        dist = np.maximum(np.linalg.norm(p - cpos, axis=1), 1e-6)
        p_all.append(p)
        o_all.append(mem[oi])
        w_all.append(ta[pi] * hs[oi, pi] / dist ** 3)  # 카메라에서 본 입체각 = 넓이 x cos / 거리^2
        b_all.append(tb[pi])
    if not p_all:
        return boxes, ok, frac, contact
    flat = np.concatenate(p_all)
    owner = np.concatenate(o_all)
    wgt = np.concatenate(w_all)
    bottom = np.concatenate(b_all)
    uv, front = fast_project(cam, flat)
    u, v = uv[:, 0], uv[:, 1]
    with np.errstate(invalid="ignore"):
        in_u = front & (u >= 0) & (u <= w1)
        vis = in_u & (v >= 0) & (v <= h1)
        below = in_u & (v > h1)  # 영상 아래 끝 밖으로 이어지는 부분
    if h_clip is not None:
        in_clip = _in_h_clip(cam, flat, h_clip)
        vis &= in_clip
        below &= in_clip
    if ego_body and vis.any():
        idx = np.nonzero(vis)[0]
        vis[idx] = ~ego_blocks(cpos, flat[idx])
    if blockers is not None and vis.any():
        idx = np.nonzero(vis)[0]
        vis[idx] = ~blocked_by_boxes(cpos, flat[idx], owner[idx], *blockers)
    first = np.r_[True, owner[1:] != owner[:-1]]
    start = np.nonzero(first)[0]
    seg = owner[start]  # 구간마다 물체 번호
    seg_of = np.cumsum(first) - 1
    big = 1e9
    x1 = np.minimum.reduceat(np.where(vis, u, big), start)
    y1 = np.minimum.reduceat(np.where(vis, v, big), start)
    x2 = np.maximum.reduceat(np.where(vis, u, -big), start)
    y2 = np.maximum.reduceat(np.where(vis, v, -big), start)
    share = np.add.reduceat(np.where(vis, wgt, 0.0), start) / np.maximum(np.add.reduceat(wgt, start), 1e-300)
    seen = np.add.reduceat(vis.astype(np.int64), start) > 0
    # 접지점: 영상에서 가장 아래에 보이는 점이 물체 바닥(높이 0)이고, 물체가 영상 아래 끝 밖으로 이어지지 않아야 한다
    lowest = vis & bottom & (v >= y2[seg_of] - 1e-6)
    touch = (np.add.reduceat(lowest.astype(np.int64), start) > 0) & (np.add.reduceat(below.astype(np.int64), start) == 0)
    good = seen & (share >= min_frac)
    with np.errstate(invalid="ignore"):
        good &= (x2 - x1) * (y2 - y1) >= MIN_AREA
    boxes[seg[good]] = np.stack([x1, y1, x2, y2], axis=1)[good]
    ok[seg[good]] = True
    frac[seg[good]] = share[good]
    contact[seg[good]] = touch[good]
    return boxes, ok, frac, contact


def ground_contact_hidden(cam: Camera, boxes: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """상자 아래 가운데(파이프라인이 거리 계산에 쓰는 화소)의 바닥 점이 자차 차체에 가리는지(N,). full 규칙용."""
    out = np.zeros(len(boxes), dtype=bool)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return out
    bottom = np.stack([(boxes[idx, 0] + boxes[idx, 2]) / 2.0, boxes[idx, 3]], axis=1)
    g, valid = cam.pixel_to_ground(bottom, max_range=200.0)
    hid = np.zeros(idx.size, dtype=bool)
    if valid.any():
        hid[valid] = ego_blocks(cam.position, g[valid])
    out[idx] = hid
    return out


# 다른 물체가 가리는 곳 ------------------------------------------------------
# 승용차는 통짜 상자가 아니라 아래 몸통과 가운데 객실, 두 덩어리로 본다. 보닛과 트렁크 위로는 뒤에 있는 물체가
# 보이기 때문이다. 비율은 추정값이다(높이 1.5m 차에서 몸통 0.98m, 길이 4.5m 차에서 객실 2.5m).
CAR_BODY_H = 0.65  # 몸통(보닛, 트렁크) 높이 / 전체 높이
CAR_CABIN_L = 0.55  # 객실 길이 / 전체 길이


def occluder_boxes(labels: list[str], x, y, length, width, height, along_y=None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """물체마다 시선을 가리는 덩어리를 만든다. 돌려주는 값: 아래 모서리(K,3), 위 모서리(K,3), 물체 번호(K,).

    length와 width는 Actor.size()처럼 X방향, Y방향 길이다. along_y[i]가 True면 차체가 Y축 방향이라 객실도 Y방향으로 줄인다.
    x가 유한하지 않거나 장면 밖 표시(1e5 이상)인 물체는 넣지 않는다.
    """
    lo, hi, owner = [], [], []
    for i, lab in enumerate(labels):
        if not (np.isfinite(x[i]) and abs(x[i]) < 1e5):
            continue
        hx, hy, h = length[i] / 2.0, width[i] / 2.0, height[i]
        if lab == "car":
            lo.append((x[i] - hx, y[i] - hy, 0.0)); hi.append((x[i] + hx, y[i] + hy, CAR_BODY_H * h)); owner.append(i)
            if along_y is not None and along_y[i]:
                hy *= CAR_CABIN_L
            else:
                hx *= CAR_CABIN_L
            lo.append((x[i] - hx, y[i] - hy, CAR_BODY_H * h)); hi.append((x[i] + hx, y[i] + hy, h)); owner.append(i)
        else:
            lo.append((x[i] - hx, y[i] - hy, 0.0)); hi.append((x[i] + hx, y[i] + hy, h)); owner.append(i)
    return np.array(lo, dtype=np.float64).reshape(-1, 3), np.array(hi, dtype=np.float64).reshape(-1, 3), \
        np.array(owner, dtype=int)


def blocked_by_boxes(cam_pos: np.ndarray, pts: np.ndarray, pt_owner: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     box_owner: np.ndarray) -> np.ndarray:
    """카메라에서 점(M,3)까지의 시선이 다른 물체의 덩어리(축에 나란한 상자 K개)를 지나는지(M,) bool.

    pt_owner[m]과 box_owner[k]가 같으면 자기 자신이라 보지 않는다(물체가 자기 뒷면을 가리는 것은 세지 않는다.
    보이는 비율 frac의 기준이 "가리는 것이 없을 때 1"이 되도록).
    시선 c + tau x d (tau 0~1)가 상자 안에 있는 구간을 축마다 구해 겹치는지 본다(slab 방법).
    """
    m = len(pts)
    out = np.zeros(m, dtype=bool)
    if m == 0 or len(lo) == 0:
        return out
    c = np.asarray(cam_pos, dtype=np.float64)
    d = np.asarray(pts, dtype=np.float64) - c
    dist = np.linalg.norm(d, axis=1)
    unit = d / np.maximum(dist, 1e-9)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
    ctr = (lo + hi) / 2.0 - c
    rad = np.linalg.norm(hi - lo, axis=1) / 2.0  # 상자를 감싸는 공의 반지름
    cd = np.linalg.norm(ctr, axis=1)
    for k in range(len(lo)):
        # 먼저 걸러 낸다: 자기 자신, 상자보다 앞에 있는 점, 상자를 감싸는 공 밖으로 지나가는 시선
        sel = (pt_owner != box_owner[k]) & ~out & (dist > cd[k] - rad[k])
        if cd[k] > rad[k]:
            sel &= unit @ (ctr[k] / cd[k]) >= math.sqrt(1.0 - (rad[k] / cd[k]) ** 2)
        idx = np.nonzero(sel)[0]
        if idx.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            t1 = (lo[k] - c) * inv[idx]
            t2 = (hi[k] - c) * inv[idx]
            t_in = np.fmin(t1, t2).max(axis=1)
            t_out = np.fmax(t1, t2).min(axis=1)
            out[idx] = (t_out > np.maximum(t_in, 0.0)) & (t_in < 1.0 - 1e-6)
    return out


def occluded_by_others(boxes: np.ndarray, ok: np.ndarray, dist: np.ndarray, width: int, height: int,
                       thr: float = 0.6, shrink: float = 0.15, cell: int = 8) -> np.ndarray:
    """더 가까운 물체의 상자에 넓이의 thr 이상이 덮인 물체(N,) bool. 예전 어림 방식이고 비교용으로 남겨 둔다.

    상자끼리 겹침으로 어림한다. 3D 상자의 영상 상자는 빈 모서리를 포함하므로 가리는 쪽 상자는 사방을
    shrink 비율만큼 줄여서 쓴다. 영상은 cell 화소 격자로 줄여 계산한다.
    """
    out = np.zeros(len(boxes), dtype=bool)
    idx = np.nonzero(ok)[0]
    if idx.size < 2:
        return out
    gw, gh = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    mask = np.zeros((gh, gw), dtype=bool)
    for i in idx[np.argsort(dist[idx])]:  # 가까운 것부터
        x1, y1, x2, y2 = boxes[i] / cell
        a1, b1 = int(math.floor(x1)), int(math.floor(y1))
        a2, b2 = max(a1 + 1, int(math.ceil(x2))), max(b1 + 1, int(math.ceil(y2)))
        sub = mask[b1:b2, a1:a2]
        if sub.size and sub.mean() >= thr:
            out[i] = True
            continue  # 가려진 물체는 뒤 물체를 가리는 데 넣지 않는다(이미 앞 물체가 가리고 있다)
        w, h = x2 - x1, y2 - y1
        s1, t1 = int(round(x1 + shrink * w)), int(round(y1 + shrink * h))
        s2, t2 = int(round(x2 - shrink * w)), int(round(y2 - shrink * h))
        if s2 > s1 and t2 > t1:
            mask[t1:t2, s1:s2] = True
    return out


# ---------------------------------------------------------------------------
# 장면: 자차 주변을 움직이는 물체
# ---------------------------------------------------------------------------
SIZES = {  # (길이, 폭, 높이) m
    "car": (4.5, 1.9, 1.5),
    "truck": (8.0, 2.5, 3.0),
    "bus": (11.0, 2.6, 3.2),
    "person": (0.5, 0.5, 1.7),
    "motorcycle": (2.0, 0.8, 1.5),
}
LANE = 3.5


@dataclass
class Actor:
    label: str
    path: Callable[[float], tuple[float, float, int] | None]  # t -> (x, y, 다시 등장 횟수) 또는 None(장면 밖)
    along_y: bool = False  # True면 차체가 Y축 방향(교차로를 가로지르는 차)

    def size(self) -> tuple[float, float, float]:
        length, width, height = SIZES[self.label]
        return (width, length, height) if self.along_y else (length, width, height)


def _wrap(v: float, lo: float, hi: float) -> tuple[float, int]:
    span = hi - lo
    k = math.floor((v - lo) / span)
    return v - span * k, abs(int(k))


def lane_path(y: float, x0: float, v: float, lo: float = -35.0, hi: float = 45.0):
    """차로를 따라 등속으로 가다 범위를 벗어나면 반대편에서 다시 나온다(SimWorld.set_time과 같은 범위)."""
    def f(t: float):
        x, k = _wrap(x0 + v * t, lo, hi)
        return x, y, k
    return f


def cross_path(x: float, y0: float, v: float, lo: float = -30.0, hi: float = 30.0):
    """교차로를 가로지르는 경로(Y 방향 등속, 범위를 벗어나면 반대편에서 다시)."""
    def f(t: float):
        y, k = _wrap(y0 + v * t, lo, hi)
        return x, y, k
    return f


def pingpong_path(y: float, x_lo: float, x_hi: float, v: float, phase: float = 0.0):
    """x_lo와 x_hi 사이를 등속으로 오간다(앞차와의 거리가 벌어졌다 좁혀지는 경우)."""
    span = x_hi - x_lo

    def f(t: float):
        s = (phase * 2 * span + abs(v) * t) % (2 * span)
        return x_lo + (s if s <= span else 2 * span - s), y, 0
    return f


def loop_path(offset: float, period: float, phase: float = 0.0):
    """자차 차체에서 offset m 떨어져 period 초에 차를 한 바퀴 도는 경로(모서리는 둥글게). 반시계 방향.

    여러 사람을 같은 period로 돌리면 phase 차이만큼 늘 떨어져 있어 서로 겹치지 않는다.
    """
    lx, ly = EGO_X_FRONT - EGO_X_REAR, 2 * EGO_HALF_W
    arc = math.pi * offset / 2.0
    seg = [lx, arc, ly, arc, lx, arc, ly, arc]  # 오른쪽 변(앞으로), 앞오른쪽 모서리, 앞 변, ...
    total = sum(seg)

    def f(t: float):
        s = ((phase + t / period) % 1.0) * total
        hw, d = EGO_HALF_W, offset
        if s < seg[0]:  # 오른쪽 변: 뒤에서 앞으로
            return EGO_X_REAR + s, -(hw + d), 0
        s -= seg[0]
        if s < arc:  # 앞오른쪽 모서리
            a = -math.pi / 2 + s / d
            return EGO_X_FRONT + d * math.cos(a), -hw + d * math.sin(a), 0
        s -= arc
        if s < seg[2]:  # 앞 변: 오른쪽에서 왼쪽으로
            return EGO_X_FRONT + d, -hw + s, 0
        s -= seg[2]
        if s < arc:  # 앞왼쪽 모서리
            a = s / d
            return EGO_X_FRONT + d * math.cos(a), hw + d * math.sin(a), 0
        s -= arc
        if s < seg[4]:  # 왼쪽 변: 앞에서 뒤로
            return EGO_X_FRONT - s, hw + d, 0
        s -= seg[4]
        if s < arc:  # 뒤왼쪽 모서리
            a = math.pi / 2 + s / d
            return EGO_X_REAR + d * math.cos(a), hw + d * math.sin(a), 0
        s -= arc
        if s < seg[6]:  # 뒤 변: 왼쪽에서 오른쪽으로
            return EGO_X_REAR - d, hw - s, 0
        s -= seg[6]
        a = math.pi + s / d  # 뒤오른쪽 모서리
        return EGO_X_REAR + d * math.cos(a), -hw + d * math.sin(a), 0
    return f


@dataclass
class Scenario:
    key: str
    title: str
    actors: list[Actor]
    duration: float  # 초


def make_scenarios() -> dict[str, Scenario]:
    """평가 장면. 속도는 자차 대비 상대 속도(m/s)다."""
    A = Actor
    lead = EGO_X_FRONT + 2.25  # 앞차 중심 x = 앞범퍼 + 차 길이 절반 + 간격
    S = [
        Scenario("overtake", "옆 차로 추월(추월당함, 추월함)", [
            A("car", lane_path(LANE, -30.0, 3.0)), A("car", lane_path(LANE, 10.0, 3.0)),
            A("car", lane_path(-LANE, 38.0, -3.0)), A("truck", lane_path(-LANE, -2.0, -3.0)),
            A("car", lane_path(2 * LANE, -20.0, 8.0)), A("car", lane_path(-2 * LANE, 20.0, -6.0)),
            A("car", pingpong_path(0.0, 14.0, 18.0, 0.3)), A("car", pingpong_path(0.0, -12.0, -9.0, 0.2)),
        ], 80.0),
        Scenario("dense", "정체 구간에서 나란히 달림", [
            *[A("truck" if i == 2 else "car", lane_path(LANE, -32.0 + 10.0 * i, 0.5)) for i in range(8)],
            A("bus", lane_path(-LANE, 40.0, -0.7)), A("car", lane_path(-LANE, 27.0, -0.7)),
            A("truck", lane_path(-LANE, 13.0, -0.7)), A("car", lane_path(-LANE, 0.0, -0.7)),
            A("car", lane_path(-LANE, -12.0, -0.7)), A("car", lane_path(-LANE, -24.0, -0.7)),
            *[A("car", lane_path(2 * LANE, -25.0 + 20.0 * i, 1.2)) for i in range(4)],
            A("car", pingpong_path(0.0, lead + 2.5, lead + 5.0, 0.2)), A("car", pingpong_path(0.0, -9.5, -7.5, 0.15)),
        ], 120.0),
        Scenario("oncoming", "왕복 2차로: 마주 오는 차, 길가 주차 차, 보행자", [
            A("car", lane_path(LANE, 40.0, -25.0)), A("car", lane_path(LANE, 0.0, -25.0)),
            *[A("car", lane_path(-3.3, 30.0 - 25.0 * i, -12.0)) for i in range(3)],
            A("person", lane_path(6.9, 20.0, -10.6)), A("person", lane_path(-6.5, -10.0, -13.4)),
            A("car", pingpong_path(0.0, 16.0, 20.0, 0.3)),
        ], 60.0),
        Scenario("cross", "교차로 정지: 가로지르는 차와 보행자", [
            A("car", cross_path(10.5, -28.0, 8.0), along_y=True), A("car", cross_path(10.5, 2.0, 8.0), along_y=True),
            A("car", cross_path(14.0, 25.0, -8.0), along_y=True), A("truck", cross_path(14.0, -5.0, -8.0), along_y=True),
            A("person", cross_path(7.0, -8.0, 1.4, -9.0, 9.0)), A("person", cross_path(8.3, 6.0, -1.3, -9.0, 9.0)),
            A("car", pingpong_path(LANE, 0.6, 1.0, 0.02)), A("car", pingpong_path(-LANE, -1.2, -0.8, 0.02)),
            A("car", pingpong_path(0.0, -8.2, -7.8, 0.02)),
        ], 60.0),
        Scenario("walk", "정차 중 사람이 차를 한 바퀴 돎", [
            A("person", loop_path(1.0, 24.0, 0.0)), A("person", loop_path(2.0, 24.0, 1 / 3)),
            A("person", loop_path(3.5, 24.0, 2 / 3)),
        ], 96.0),
        Scenario("far", "먼 앞차(15~42m)", [
            A("car", pingpong_path(0.0, 15.0, 42.0, 1.2)), A("car", pingpong_path(LANE, 12.0, 44.0, 1.5, 0.3)),
            A("truck", pingpong_path(-LANE, 20.0, 44.0, 0.8, 0.6)),
            A("person", lane_path(-6.5, 44.0, -12.0)),
        ], 90.0),
        Scenario("queue", "정차 대기열: 바짝 붙은 앞차, 차 사이로 지나가는 오토바이", [
            A("car", pingpong_path(0.0, lead + 1.2, lead + 6.0, 0.6)),
            A("car", pingpong_path(LANE, 0.8, 1.2, 0.03)), A("car", pingpong_path(LANE, 7.3, 7.7, 0.03)),
            A("car", pingpong_path(LANE, -5.7, -5.3, 0.03)),
            A("truck", pingpong_path(-LANE, 2.8, 3.2, 0.03)), A("car", pingpong_path(-LANE, -7.2, -6.8, 0.03)),
            A("car", pingpong_path(0.0, -7.6, -7.2, 0.03)),
            A("motorcycle", lane_path(1.75, -30.0, 4.0)), A("motorcycle", lane_path(-1.75, -10.0, 3.0)),
        ], 80.0),
        Scenario("sidewalk", "서행 중 길가: 보도의 보행자, 길가에 선 차, 옆으로 지나가는 오토바이", [
            A("person", lane_path(7.5, 24.0, -5.4, -16.0, 26.0)), A("person", lane_path(10.5, 5.0, -2.6, -16.0, 26.0)),
            A("person", lane_path(-8.0, 12.0, -5.3, -16.0, 26.0)), A("person", lane_path(-10.5, -6.0, -2.7, -16.0, 26.0)),
            A("person", lane_path(-5.6, 20.0, -4.0, -16.0, 26.0)),
            A("car", lane_path(-3.6, 30.0, -4.0)), A("car", lane_path(-3.6, 2.0, -4.0)), A("car", lane_path(-3.6, -22.0, -4.0)),
            A("motorcycle", lane_path(4.2, -30.0, 2.5)),
        ], 70.0),
        Scenario("backout", "주차 칸에서 후진 출차: 양옆에 선 차, 뒤를 가로지르는 차와 사람", [
            A("car", pingpong_path(2.9, 1.3, 1.5, 0.01)), A("car", pingpong_path(-2.9, 1.3, 1.5, 0.01)),
            A("car", cross_path(-7.5, -20.0, 4.0, -20.0, 20.0), along_y=True),
            A("car", cross_path(-11.0, 6.0, -5.0, -20.0, 20.0), along_y=True),
            A("person", cross_path(-3.2, -7.0, 1.3, -9.0, 9.0)), A("person", cross_path(-4.6, 4.0, -1.1, -9.0, 9.0)),
            A("person", cross_path(6.0, 3.0, 1.2, -9.0, 9.0)),
        ], 50.0),
    ]
    return {s.key: s for s in S}


# ---------------------------------------------------------------------------
# 프레임별 정답 만들기
# ---------------------------------------------------------------------------
@dataclass
class FrameTruth:
    t: float
    xy: np.ndarray  # (A,2) 기준 시각의 물체 위치(장면 밖이면 nan)
    epoch: np.ndarray  # (A,) 다시 등장 횟수
    alive: np.ndarray  # (A,) bool
    cams: list[list[tuple[int, np.ndarray, float, float]]]  # 카메라별 [(물체 번호, 상자, 광축에서 벗어난 각 rad, 보이는 비율), ...]


def cam_skew_s(layout: Layout) -> list[float]:
    """카메라별 촬영 시각 어긋남(초). 카메라가 서로 따로 도는 30fps라 가장 가까운 프레임끼리 묶어도 남는 차이다.

    14번 측정에서 CSI끼리는 평균 2~17ms, USB(가짜)는 평균 18ms(-24~+48ms)였다. 그 크기 안에서
    카메라 순서로 정한 고정값을 쓴다(난수 대신 고정값이라 seed와 무관하게 같은 정답이 나온다).
    """
    out = []
    for k, s in enumerate(layout.specs):
        half = 8.0 if s.link == "csi" else 25.0
        out.append(((k * 37 % 17) / 16.0 * 2.0 - 1.0) * half / 1e3 if k else 0.0)
    return out


def actor_state(actors: list[Actor], t: float):
    xs, ys, ep, alive = [], [], [], []
    for a in actors:
        p = a.path(t)
        if p is None:
            xs.append(np.nan); ys.append(np.nan); ep.append(0); alive.append(False)
        else:
            xs.append(p[0]); ys.append(p[1]); ep.append(p[2]); alive.append(True)
    return np.array(xs), np.array(ys), np.array(ep, dtype=int), np.array(alive, dtype=bool)


def build_truth(layout: Layout, scenario: Scenario, fps: float, ego_body: bool = True, occlusion: bool | str = "exact",
                skew: bool = True, rule: str = "partial", contact: str = "keep") -> tuple[list[Camera], list[FrameTruth]]:
    """장면을 fps 간격으로 훑어 프레임마다 카메라별 정답 상자를 만든다(잡음과 놓침은 아직 없음).

    occlusion: 물체끼리 가리는 것을 어떻게 볼지.
             "exact" 시선이 다른 물체의 덩어리(occluder_boxes)를 지나는 겉면 점을 뺀다. 반쯤 가린 물체는 보이는
                     부분만 상자가 되고 보이는 비율이 줄어 검출 확률이 낮아진다(기본).
             "box"   더 가까운 물체의 영상 상자에 넓이의 60% 넘게 덮인 물체를 통째로 뺀다(예전 어림 방식).
             "none"  가리지 않는다.
             True는 "exact", False는 "none"과 같다. rule이 "full"이면 겉면 점이 없으므로 "exact" 대신 "box"를 쓴다.
    contact: "keep"이면 접지점이 잘린 상자도 그대로 둔다(파이프라인이 잘린 자리로 거리를 계산해 위치가 어긋난다).
             "drop"이면 접지점이 보이지 않는 상자를 뺀다(SimWorld의 규칙과 같은 보수적인 기준).
    """
    if contact not in ("keep", "drop"):
        raise ValueError(f"contact는 keep 또는 drop이어야 합니다: {contact}")
    occlusion = {True: "exact", False: "none"}.get(occlusion, occlusion)
    if occlusion not in ("exact", "box", "none"):
        raise ValueError(f"occlusion은 exact, box, none 가운데 하나여야 합니다: {occlusion}")
    if occlusion == "exact" and rule == "full":
        occlusion = "box"
    labels = [a.label for a in scenario.actors]
    along_y = [a.along_y for a in scenario.actors]
    cams = layout.cameras()
    clips = layout.h_clips()
    skews = cam_skew_s(layout) if skew else [0.0] * len(cams)
    sizes = np.array([a.size() for a in scenario.actors])  # (A,3)
    frames = []
    n = int(round(scenario.duration * fps))
    for i in range(n):
        t = i / fps
        x, y, ep, alive = actor_state(scenario.actors, t)
        per_cam = []
        for cam, dt, clip in zip(cams, skews, clips):
            if dt != 0.0:
                xk, yk, _, ak = actor_state(scenario.actors, t + dt)
            else:
                xk, yk, ak = x, y, alive
            xs = np.where(ak, xk, 1e6)
            ys = np.where(ak, yk, 1e6)
            blockers = occluder_boxes(labels, xs, ys, sizes[:, 0], sizes[:, 1], sizes[:, 2], along_y) \
                if occlusion == "exact" else None
            boxes, ok, frac, touch = object_boxes(cam, xs, ys, sizes[:, 0], sizes[:, 1], sizes[:, 2], rule, ego_body,
                                                  h_clip=clip, blockers=blockers)
            ok &= ak
            if contact == "drop":
                ok &= touch
            if occlusion == "box":
                dist = np.hypot(xs - cam.position[0], ys - cam.position[1])
                ok &= ~occluded_by_others(boxes, ok, dist, cam.width, cam.height)
            idx = np.nonzero(ok)[0]
            if idx.size:
                ctr = np.stack([(boxes[idx, 0] + boxes[idx, 2]) / 2.0, (boxes[idx, 1] + boxes[idx, 3]) / 2.0], axis=1)
                off = off_axis_angle(cam, ctr)
            else:
                off = np.zeros(0)
            per_cam.append([(int(j), boxes[j].copy(), float(o), float(frac[j])) for j, o in zip(idx, off)])
        frames.append(FrameTruth(t, np.stack([x, y], axis=1), ep, alive, per_cam))
    return cams, frames


def off_axis_angle(cam: Camera, uv: np.ndarray) -> np.ndarray:
    """화소(N,2)가 가리키는 방향이 광축에서 벗어난 각(rad)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    r = np.hypot(uv[:, 0] - cam.K[0, 2], uv[:, 1] - cam.K[1, 2]) / cam.K[0, 0]
    return r if cam.model == "fisheye" else np.arctan(r)


# ---------------------------------------------------------------------------
# 검출 모형
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DetModel:
    """정답 상자를 검출기 출력처럼 바꾸는 규칙.

    크기 s는 NPU 입력(긴 변 640)으로 줄였을 때 상자 넓이의 제곱근(px)이다. s0 이하는 검출하지 못하고
    s1 이상은 p_max 확률로 검출하며 그 사이는 직선으로 잇는다. s0 = s1 = 0이면 크기와 무관하게 p_max
    (SimBackend와 같은 방식)이다. 크기별 검출률은 실측이 아니라 가정이므로 세 가지를 함께 돌려 본다.
    """

    key: str
    s0: float
    s1: float
    p_max: float = 0.95
    noise_px: float = 2.0
    edge_penalty: float = 0.0  # 광축에서 edge_from도 넘게 벗어나면 검출 확률을 이 비율까지 깎는다
    edge_from_deg: float = 50.0
    edge_full_deg: float = 80.0
    difficulty_sigma: float = 0.25  # 물체와 카메라 쌍마다 고정된 크기 배율(로그 정규)의 표준편차
    frac0: float = 0.0  # 물체가 이 비율 이하로만 보이면 검출 0
    frac1: float = 0.0  # 이 비율 이상 보이면 잘림으로는 깎지 않는다(frac0 = frac1 = 0이면 잘림 무시)
    edge_fisheye_only: bool = False  # True면 가장자리 깎기를 어안 렌즈 카메라에만 적용한다

    def prob(self, box: np.ndarray, off_axis: float, cam: Camera, factor: float = 1.0, frac: float = 1.0) -> float:
        if self.s1 <= 0:
            p = self.p_max
        else:
            scale = 640.0 / max(cam.width, cam.height)
            s = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1]))) * scale * factor
            p = self.p_max * min(1.0, max(0.0, (s - self.s0) / (self.s1 - self.s0)))
        if self.frac1 > 0:
            p *= min(1.0, max(0.0, (frac - self.frac0) / (self.frac1 - self.frac0)))
        if self.edge_penalty > 0 and (cam.model == "fisheye" or not self.edge_fisheye_only):
            a, b = math.radians(self.edge_from_deg), math.radians(self.edge_full_deg)
            p *= 1.0 - self.edge_penalty * min(1.0, max(0.0, (off_axis - a) / (b - a)))
        return p


DET_MODELS = {
    "flat": DetModel("flat", 0.0, 0.0, difficulty_sigma=0.0),  # 저장소 SimBackend와 같음(크기 무관 5% 놓침)
    "easy": DetModel("easy", 5.0, 14.0, frac0=0.10, frac1=0.30),
    "base": DetModel("base", 8.0, 24.0, frac0=0.15, frac1=0.45),
    "hard": DetModel("hard", 12.0, 36.0, edge_penalty=0.5, frac0=0.25, frac1=0.65),
    # 어안 렌즈 가장자리(광축에서 65도 이상)에서는 검출기가 거의 못 잡는다고 본 경우. 광각 좌우 카메라에 불리한 가정이다.
    "edge": DetModel("edge", 8.0, 24.0, edge_penalty=0.9, edge_from_deg=35.0, edge_full_deg=65.0, frac0=0.15, frac1=0.45,
                     edge_fisheye_only=True),
}


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
SECTORS = ("front", "front_side", "side", "rear_side", "rear")
BANDS = ("near", "mid", "far")
EGO_CENTER_X = (EGO_X_REAR + EGO_X_FRONT) / 2.0


def sector_of(x: float, y: float) -> str:
    """자차 가운데에서 본 방위. 정면 0도, 좌우 대칭."""
    az = abs(math.degrees(math.atan2(y, x - EGO_CENTER_X)))
    if az <= 30:
        return "front"
    if az <= 75:
        return "front_side"
    if az <= 105:
        return "side"
    if az <= 150:
        return "rear_side"
    return "rear"


def band_of(x: float, y: float) -> str:
    """자차 차체에서 떨어진 거리. 가까움 5m 이하, 중간 15m 이하, 그 밖은 멂."""
    dx = max(EGO_X_REAR - x, 0.0, x - EGO_X_FRONT)
    dy = max(abs(y) - EGO_HALF_W, 0.0)
    d = math.hypot(dx, dy)
    return "near" if d <= 5.0 else ("mid" if d <= 15.0 else "far")


LONG_VEHICLES = ("truck", "bus")


def match_radius(x: float, y: float) -> float:
    """정답과 트랙을 같은 물체로 볼 최대 거리(m). scripts/eval_tracking.py와 같다(가까우면 3m, 멀면 거리의 20%)."""
    return max(3.0, 0.2 * math.hypot(x, y))


def body_distance(tx: float, ty: float, x: float, y: float, label: str, sx: float, sy: float) -> float:
    """트랙 위치(tx, ty)가 물체에서 떨어진 거리(m). sx, sy는 물체의 X방향, Y방향 길이다.

    보통은 물체 중심까지의 거리다. 트럭과 버스는 차체가 길어 트랙이 차체 옆에 붙어 있어도 중심에서는 3m 넘게
    떨어질 수 있으므로, 길이 방향 중심선(양 끝을 폭의 절반씩 줄인 선분)까지의 거리로 잰다.
    """
    dx, dy = abs(tx - x), abs(ty - y)
    if label in LONG_VEHICLES:
        half = abs(sx - sy) / 2.0
        if sx >= sy:
            dx = max(dx - half, 0.0)
        else:
            dy = max(dy - half, 0.0)
    return math.hypot(dx, dy)


def _group(label: str) -> str:
    return "vehicle" if label in VEHICLES else label


@dataclass
class Metrics:
    """화면 범위 안 물체 기준 지표. 범위는 탑뷰(grid)와 3D 화면 전방(grid 앞끝~far_x, 옆 far_y 이내)이다.

    표시 비율과 끊김은 탑뷰 범위 안의 물체로, 먼 앞쪽 비율은 3D 화면 전방의 물체로 센다. ID 바뀜은 화면 범위 안에
    있는 물체의 것만, 중복과 헛 표시는 화면 범위 안에 그려지는 트랙만 센다(화면 밖에서 일어난 일은 보이지 않는다).
    """

    grid: BevGrid
    dt: float
    far_x: float = 40.0
    far_y: float = 6.0
    total: int = 0  # 탑뷰 범위 안 (물체, 프레임) 수
    shown: int = 0
    visible: int = 0  # 그중 어느 카메라든 정답 상자가 있던 수(잡음과 놓침 전)
    far_total: int = 0
    far_shown: int = 0
    switches: int = 0
    dup: int = 0
    false_tracks: int = 0
    frames: int = 0
    cells: dict = field(default_factory=dict)  # (방위, 거리, 종류) -> [전체, 표시]
    errs: list = field(default_factory=list)  # 짝지어진 트랙의 위치 오차(m), 가까움+중간 거리
    gaps: list = field(default_factory=list)  # 끊긴 구간 길이(초)
    _last: dict = field(default_factory=dict)
    _run: dict = field(default_factory=dict)
    _ids: dict = field(default_factory=dict)

    def in_bev(self, x: float, y: float) -> bool:
        g = self.grid
        return g.x_min <= x <= g.x_max and g.y_min <= y <= g.y_max

    def in_far(self, x: float, y: float) -> bool:
        return self.grid.x_max < x <= self.far_x and abs(y) <= self.far_y

    def add(self, actors: list[Actor], truth: FrameTruth, tracks: list) -> None:
        self.frames += 1
        seen = {e[0] for per in truth.cams for e in per}
        gts = [(i, actors[i].label, float(truth.xy[i, 0]), float(truth.xy[i, 1]))
               for i in range(len(actors)) if truth.alive[i] and math.hypot(*truth.xy[i]) <= 70.0]
        sizes = [actors[i].size() for i, _, _, _ in gts]
        # 정답과 트랙 사이 거리(같은 종류이고 반경 안일 때만)
        dist = {}
        for gi, (_, lab, x, y) in enumerate(gts):
            r = match_radius(x, y)
            for ti, t in enumerate(tracks):
                if _group(t.label) != _group(lab):
                    continue
                d = body_distance(float(t.xy[0]), float(t.xy[1]), x, y, lab, sizes[gi][0], sizes[gi][1])
                if d < r:
                    dist[(gi, ti)] = d
        # 일대일 짝짓기. 지난 프레임에 짝이던 트랙이 아직 반경 안에 있으면 그대로 두고, 나머지는 가까운 쌍부터 잇는다.
        # 한 물체에 트랙이 둘 붙어 있을 때 프레임마다 더 가까운 쪽으로 짝이 오가는 것을 ID 바뀜으로 세지 않기 위해서다.
        used_g, used_t, assign = set(), set(), {}
        by_tid = {t.tid: ti for ti, t in enumerate(tracks)}
        for gi, (i, _, _, _) in enumerate(gts):
            ti = by_tid.get(self._last.get((i, int(truth.epoch[i]))))
            if ti is not None and ti not in used_t and (gi, ti) in dist:
                used_g.add(gi)
                used_t.add(ti)
                assign[gi] = (tracks[ti].tid, dist[(gi, ti)])
        for d, gi, ti in sorted((d, gi, ti) for (gi, ti), d in dist.items()):
            if gi in used_g or ti in used_t:
                continue
            used_g.add(gi)
            used_t.add(ti)
            assign[gi] = (tracks[ti].tid, d)
        live_keys = set()
        for gi, (i, lab, x, y) in enumerate(gts):
            key = (i, int(truth.epoch[i]))
            hit = assign.get(gi)
            on_screen = self.in_bev(x, y) or self.in_far(x, y)
            if hit is not None:
                prev = self._last.get(key)
                if prev is not None and prev != hit[0] and on_screen:
                    self.switches += 1
                self._last[key] = hit[0]
                if on_screen:
                    self._ids.setdefault(key, set()).add(hit[0])
            if self.in_far(x, y):
                self.far_total += 1
                self.far_shown += hit is not None
            if not self.in_bev(x, y):
                continue
            live_keys.add(key)
            self.total += 1
            self.visible += i in seen
            c = self.cells.setdefault((sector_of(x, y), band_of(x, y), _group(lab)), [0, 0])
            c[0] += 1
            if hit is not None:
                self.shown += 1
                c[1] += 1
                if band_of(x, y) != "far":
                    self.errs.append(hit[1])
                run = self._run.pop(key, 0)
                if run:
                    self.gaps.append(run * self.dt)
            else:
                self._run[key] = self._run.get(key, 0) + 1
        # 화면 범위를 벗어난 물체의 끊김 구간은 거기서 끝난 것으로 센다
        for key in [k for k in self._run if k not in live_keys]:
            self.gaps.append(self._run.pop(key) * self.dt)
        # 짝이 없는 트랙: 화면 범위 안에 그려지는 것만 센다. 같은 종류의 물체 곁에 있으면 중복, 아니면 헛 표시다
        for ti, t in enumerate(tracks):
            tx, ty = float(t.xy[0]), float(t.xy[1])
            if ti in used_t or not (self.in_bev(tx, ty) or self.in_far(tx, ty)):
                continue
            if any((gi, ti) in dist for gi in range(len(gts))):
                self.dup += 1
            else:
                self.false_tracks += 1

    def finish(self) -> None:
        for key in list(self._run):
            self.gaps.append(self._run.pop(key) * self.dt)

    def summary(self) -> dict:
        self.finish()
        gaps = np.array(self.gaps) if self.gaps else np.zeros(0)
        errs = np.array(self.errs) if self.errs else np.zeros(0)
        minutes = self.frames * self.dt / 60.0
        counts = [len(v) for v in self._ids.values()]
        return {
            "frames": self.frames,
            "total": self.total,
            "shown": self.shown,
            "visible": self.visible,
            "far_total": self.far_total,
            "far_shown": self.far_shown,
            "id_switches": self.switches,
            "dup": self.dup,
            "false_tracks": self.false_tracks,
            # 프레임 수는 FPS가 높을수록 커지므로 배치끼리 비교할 때는 시간(초)으로 바꾼 값을 쓴다
            "dup_s": self.dup * self.dt,
            "false_s": self.false_tracks * self.dt,
            "minutes": minutes,
            "gap_sum_s": float(gaps.sum()),
            "gap_max_s": float(gaps.max()) if gaps.size else 0.0,
            "gaps_over_1s": int((gaps > 1.0).sum()),
            "gaps_over_3s": int((gaps > 3.0).sum()),
            "err_median": float(np.median(errs)) if errs.size else None,
            "err_p90": float(np.percentile(errs, 90)) if errs.size else None,
            "tracks_per_object": float(np.mean(counts)) if counts else None,
            "cells": {"|".join(k): v for k, v in self.cells.items()},
        }


# ---------------------------------------------------------------------------
# 추적 돌리기
# ---------------------------------------------------------------------------
def track_step(cams: list[Camera], tracker: BevTracker, dets: list[list[Detection]], dt: float,
               vehicle_center: str = "ray") -> list:
    """한 프레임의 검출을 지면에 내리고 합쳐 추적한다. RealityViewPipeline.step의 2~4단계와 같은 호출이다."""
    objs = []
    for i, (cam, d) in enumerate(zip(cams, dets)):
        objs += project_detections(i, cam, d, max_range=40.0, scale=1.0, vehicle_center=vehicle_center)
    objs = merge_cross_camera(objs, scale=1.0)
    return tracker.update(objs, dt)


def noisy_detections(cams: list[Camera], truth: FrameTruth, actors: list[Actor], det: DetModel,
                     rng: np.random.Generator, factors: np.ndarray) -> list[list[Detection]]:
    """정답 상자에 검출 모형(놓침, 화소 잡음)을 적용한다. factors는 (카메라, 물체)별 고정 크기 배율이다."""
    out = []
    for k, (cam, per) in enumerate(zip(cams, truth.cams)):
        dets = []
        for j, box, off, frac in per:
            if rng.random() >= det.prob(box, off, cam, float(factors[k, j]), frac):
                continue
            b = box + rng.normal(0.0, det.noise_px, 4)
            dets.append(Detection(actors[j].label, float(rng.uniform(0.6, 0.95)), tuple(b.tolist())))
        out.append(dets)
    return out


def difficulty_factors(cams: list[Camera], n_actors: int, sigma: float, seed: int) -> np.ndarray:
    """(카메라, 물체)별 고정 크기 배율. 같은 seed에서 같은 이름의 카메라는 배치가 달라도 같은 값을 받는다.

    카메라 순서로 뽑으면 4대 배치의 후방 카메라와 5대 배치의 전방 메인 카메라가 같은 값을 받게 되어,
    배치를 비교할 때 카메라 구성과 상관없는 차이가 섞인다.
    """
    out = np.ones((len(cams), n_actors))
    if sigma > 0:
        for k, cam in enumerate(cams):
            rng = np.random.default_rng([int(seed), zlib.crc32(cam.name.encode("utf-8"))])
            out[k] = np.exp(rng.normal(0.0, sigma, n_actors))
    return out


def run_once(layout: Layout, scenario: Scenario, det: DetModel, fps: float, seed: int, grid: BevGrid | None = None,
             truth: tuple[list[Camera], list[FrameTruth]] | None = None, tracker_mode: str = "robust",
             vehicle_center: str = "ray", **truth_opts) -> dict:
    """배치 하나, 장면 하나를 한 번 돌려 지표 요약을 돌려준다."""
    grid = grid or BevGrid()
    cams, frames = truth if truth is not None else build_truth(layout, scenario, fps, **truth_opts)
    rng = np.random.default_rng(seed)
    factors = difficulty_factors(cams, len(scenario.actors), det.difficulty_sigma, seed)
    tracker = BevTracker(mode=tracker_mode)
    dt = 1.0 / fps
    m = Metrics(grid, dt)
    for fr in frames:
        dets = noisy_detections(cams, fr, scenario.actors, det, rng, factors)
        tracks = track_step(cams, tracker, dets, dt, vehicle_center)
        m.add(scenario.actors, fr, tracks)
    return m.summary()


def merge_summaries(items: list[dict]) -> dict:
    """seed별 요약을 합친다(횟수는 더하고 비율은 합친 횟수로 다시 계산)."""
    keys_sum = ("frames", "total", "shown", "visible", "far_total", "far_shown", "id_switches", "dup", "false_tracks",
                "dup_s", "false_s", "minutes", "gap_sum_s", "gaps_over_1s", "gaps_over_3s")
    out = {k: sum(it[k] for it in items) for k in keys_sum}
    out["gap_max_s"] = max(it["gap_max_s"] for it in items)
    for k in ("err_median", "err_p90", "tracks_per_object"):
        vals = [it[k] for it in items if it[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    cells: dict = {}
    for it in items:
        for k, (a, b) in it["cells"].items():
            c = cells.setdefault(k, [0, 0])
            c[0] += a
            c[1] += b
    out["cells"] = cells
    out["shown_ratio"] = out["shown"] / out["total"] if out["total"] else None
    out["visible_ratio"] = out["visible"] / out["total"] if out["total"] else None
    out["far_ratio"] = out["far_shown"] / out["far_total"] if out["far_total"] else None
    mins = out["minutes"] or 1e-9
    out["gaps_over_1s_per_min"] = out["gaps_over_1s"] / mins
    out["id_switches_per_min"] = out["id_switches"] / mins
    out["dup_s_per_min"] = out["dup_s"] / mins  # 한 물체에 상자가 하나 더 떠 있던 시간(초, 트랙 수만큼 더함)
    out["false_s_per_min"] = out["false_s"] / mins
    return out


def cell_ratio(summary: dict, sectors=None, bands=None, groups=None) -> float | None:
    """요약의 칸(방위, 거리, 종류) 가운데 고른 것만 합친 표시 비율."""
    a = b = 0
    for k, (tot, shown) in summary["cells"].items():
        s, bd, g = k.split("|")
        if (sectors is None or s in sectors) and (bands is None or bd in bands) and (groups is None or g in groups):
            a += tot
            b += shown
    return b / a if a else None


# ---------------------------------------------------------------------------
# 처리량 모형
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FpsModel:
    """카메라 수에 따른 루프 시간(ms)과 FPS.

    기본 시간 = base_ms + per_cam_ms x N. 파이 실측 두 점을 잇는 직선이다: 2대 38.2ms(03번 측정),
    4대 62.2ms(14번 측정의 두 실행 61.6, 62.8ms의 가운데). 둘 다 Hailo 검출만, 탑뷰 화면, 바닥 영상을 켠 상태다.
    5대부터는 실측이 없어 이 직선을 늘린 값이고, 그 밖의 항도 실측이 없는 어림이다.
      ground   : 탑뷰 바닥에 카메라 영상을 까는지. 끄면(run_demo.py --no-ground) 바닥 합성 단계가 빠진다.
                 이 단계의 실측은 2대 9.5ms, 4대 19.8ms라서 ground_base_ms + ground_per_cam_ms x N으로 본다.
      resize_ms: USB 카메라 한 대의 영상을 설정 크기로 줄이는 시간. 주 스레드에서 돌아 루프 시간에 그대로 더해진다.
      decode_ms: USB 카메라 한 프레임의 MJPEG 해제 시간. 해제는 카메라별 스레드에서 카메라 속도(cam_fps)로 돌아
                 루프 시간에 바로 더해지지 않고, 코어를 나눠 쓰는 만큼(coupling) 루프를 늦춘다. 4대 실측은 가짜 USB
                 카메라 2대가 영상 파일을 푸는 상태에서 잰 값이라, 그 시간(decode_in_base_ms, 추정)을 넘는 만큼만 센다.
      seg      : DX-M1 도로 세그를 함께 돌리는 경우. 2대 실측(38.2 -> 51.0ms)에 맞춘 식이다.
    """

    base_ms: float = 14.2
    per_cam_ms: float = 12.0
    ground: bool = True
    ground_base_ms: float = -0.8
    ground_per_cam_ms: float = 5.15
    resize_ms: float = 0.0
    decode_ms: float = 0.0
    decode_in_base_ms: float = 1.5
    coupling: float = 0.4
    cores: float = 4.0
    cam_fps: float = 30.0
    seg: bool = False

    def frame_ms(self, n: int, n_usb: int = 0) -> float:
        t = self.base_ms + self.per_cam_ms * n
        if not self.ground:
            t -= self.ground_base_ms + self.ground_per_cam_ms * n
        t += self.resize_ms * n_usb
        cores = self.cores
        if self.seg:
            t = 1.13 * t + 2.7 * n + 2.4
            cores -= 0.5
        busy = n_usb * self.cam_fps * max(self.decode_ms - self.decode_in_base_ms, 0.0) / 1000.0  # 해제 스레드가 붙잡는 코어 수
        return t * (1.0 + self.coupling * busy / max(cores - busy, 0.5))

    def fps(self, n: int, n_usb: int = 0) -> float:
        return min(self.cam_fps, 1000.0 / self.frame_ms(n, n_usb))


# ---------------------------------------------------------------------------
# 정지 시야 지도
# ---------------------------------------------------------------------------
def coverage_map(layout: Layout, label: str = "car", det: DetModel | None = None, step: float = 0.5,
                 x_range: tuple[float, float] = (-15.0, 40.0), y_range: tuple[float, float] = (-12.0, 12.0),
                 ego_body: bool = True, rule: str = "partial", contact: str = "keep") -> dict:
    """바닥 격자의 칸마다 그 자리에 물체(도로와 나란히)를 놓았을 때 보이는 카메라 수와 검출 확률을 구한다.

    돌려줌: xs, ys(칸 중심), n_cam(H,W), p_any(H,W: 한 프레임에 어느 카메라든 검출할 확률), free(H,W: 자차와 겹치지 않는 칸)
    """
    det = det or DET_MODELS["base"]
    length, width, height = SIZES[label]
    xs = np.arange(x_range[0] + step / 2, x_range[1], step)
    ys = np.arange(y_range[0] + step / 2, y_range[1], step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    n = x.size
    n_cam = np.zeros(n, dtype=int)
    p_miss = np.ones(n)
    for cam, clip in zip(layout.cameras(), layout.h_clips()):
        boxes, ok, frac, touch = object_boxes(cam, x, y, np.full(n, length), np.full(n, width), np.full(n, height),
                                              rule, ego_body, h_clip=clip)
        if contact == "drop":
            ok &= touch
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            continue
        ctr = np.stack([(boxes[idx, 0] + boxes[idx, 2]) / 2.0, (boxes[idx, 1] + boxes[idx, 3]) / 2.0], axis=1)
        off = off_axis_angle(cam, ctr)
        p = np.array([det.prob(boxes[j], float(o), cam, 1.0, float(frac[j])) for j, o in zip(idx, off)])
        n_cam[idx] += 1
        p_miss[idx] *= 1.0 - p
    # 자차와 겹치는 자리는 뺀다(물체 상자와 자차 상자가 닿지 않는 칸만)
    free = (np.abs(y) - width / 2.0 > EGO_HALF_W + 0.2) | (x - length / 2.0 > EGO_X_FRONT + 0.2) | \
           (x + length / 2.0 < EGO_X_REAR - 0.2)
    shape = X.shape
    return {"xs": xs, "ys": ys, "n_cam": n_cam.reshape(shape), "p_any": (1.0 - p_miss).reshape(shape),
            "free": free.reshape(shape)}


def coverage_summary(cov: dict, grid: BevGrid | None = None) -> dict:
    """시야 지도를 탑뷰 범위 안에서 요약한다: 보이는 칸 비율, 평균 검출 확률, 방위별 값."""
    grid = grid or BevGrid()
    X, Y = np.meshgrid(cov["xs"], cov["ys"], indexing="ij")
    roi = cov["free"] & (X >= grid.x_min) & (X <= grid.x_max) & (Y >= grid.y_min) & (Y <= grid.y_max)
    out = {"seen_ratio": float((cov["n_cam"][roi] > 0).mean()), "p_mean": float(cov["p_any"][roi].mean())}
    sec = np.vectorize(sector_of)(X, Y)
    band = np.vectorize(band_of)(X, Y)
    for s in SECTORS:
        m = roi & (sec == s) & (band != "far")
        out[f"seen_{s}"] = float((cov["n_cam"][m] > 0).mean()) if m.any() else None
        out[f"p_{s}"] = float(cov["p_any"][m].mean()) if m.any() else None
    near = roi & (band == "near")
    out["seen_near"] = float((cov["n_cam"][near] > 0).mean()) if near.any() else None
    return out
