from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import Camera
from .types import Detection, GroundObject

# BEV 격자, 카메라 영상/마스크의 IPM 합성, 검출 결과의 지면 투영.


# ---------------------------------------------------------------------------
# 탑뷰 격자
# ---------------------------------------------------------------------------

@dataclass
class BevGrid:
    """탑뷰 격자. 화면 위쪽이 차량 전방(+X), 왼쪽이 차량 좌측(+Y)."""

    x_min: float = -15.0  # 후방 (m)
    x_max: float = 25.0  # 전방
    y_min: float = -12.0  # 우측
    y_max: float = 12.0  # 좌측
    res: float = 0.05  # m/px

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.res))

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """각 픽셀 중심의 차량 좌표 X, Y (H, W)."""
        rows = np.arange(self.height) + 0.5
        cols = np.arange(self.width) + 0.5
        X = self.x_max - rows * self.res
        Y = self.y_max - cols * self.res
        return np.meshgrid(X, Y, indexing="ij")

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """차량 좌표 (N,2: X,Y) → BEV 픽셀 (N,2: col,row), float."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        col = (self.y_max - xy[:, 1]) / self.res
        row = (self.x_max - xy[:, 0]) / self.res
        return np.stack([col, row], axis=1)


# ---------------------------------------------------------------------------
# IPM 합성(카메라 영상, 마스크 → 탑뷰)
# ---------------------------------------------------------------------------
class IpmCompositor:
    """카메라마다 BEV 셀 → 이미지 픽셀 LUT를 미리 계산해 cv2.remap으로 합성한다.

    Fast-BEV의 LUT 투영과 같은 아이디어를 지면 평면 한 장에만 적용한 것이다.
    여러 카메라가 겹치는 셀은 카메라에 더 가까운(해상도가 높은) 쪽을 쓴다.
    """

    def __init__(self, cameras: list[Camera], grid: BevGrid, max_range: float = 30.0):
        self.cameras = cameras
        self.grid = grid
        X, Y = grid.cell_centers()
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
        H, W = grid.height, grid.width
        self.maps: list[tuple[np.ndarray, np.ndarray]] = []
        best_dist = np.full(H * W, np.inf)
        self.owner = np.full(H * W, -1, dtype=np.int32)
        for i, cam in enumerate(cameras):
            uv, in_front = cam.project(pts)
            inside = (
                in_front
                & (uv[:, 0] >= 0)
                & (uv[:, 0] <= cam.width - 1)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] <= cam.height - 1)
            )
            dist = np.linalg.norm(pts[:, :2] - cam.position[:2], axis=1)
            inside &= dist <= max_range
            take = inside & (dist < best_dist)
            best_dist[take] = dist[take]
            self.owner[take] = i
            mx = np.where(inside, uv[:, 0], -1).astype(np.float32).reshape(H, W)
            my = np.where(inside, uv[:, 1], -1).astype(np.float32).reshape(H, W)
            self.maps.append(cv2.convertMaps(mx, my, cv2.CV_16SC2))
        self.owner = self.owner.reshape(H, W)
        # 매 프레임 쓰는 "카메라 i가 맡은 셀" 마스크를 미리 uint8로 만들어 둔다.
        # 예전에는 warp()마다 owner == i 비교와 numpy 불리언 색인 복사를 해서 파이5에서 바닥 영상
        # 합성에 22ms가 들었다. cv2.copyTo는 같은 일을 마스크 자리만 골라 복사해서 훨씬 빠르고 결과는 똑같다.
        self._owner_u8 = [np.ascontiguousarray((self.owner == i).astype(np.uint8)) for i in range(len(cameras))]
        self._coverage = self.owner >= 0
        self._coverage.flags.writeable = False  # 여러 프레임이 같이 쓰므로 고치지 못하게 막는다

    def coverage(self) -> np.ndarray:
        """카메라가 하나라도 보는 셀(bool, H x W). 읽기 전용 배열을 매번 같은 것으로 돌려준다."""
        return self._coverage

    def warp(self, images: list[np.ndarray], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """카메라 영상(또는 마스크) 목록을 BEV 한 장으로 합성. 채널 수는 입력과 같다.

        카메라마다 1) LUT로 remap 해서 탑뷰 크기로 펴고 2) 그 카메라가 맡은 셀만 결과에 복사한다.
        """
        if len(images) != len(self.cameras):
            raise ValueError("images and cameras length mismatch")
        out = None
        for i, (img, (m1, m2)) in enumerate(zip(images, self.maps)):
            if img is None:
                continue
            w = cv2.remap(img, m1, m2, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if out is None:
                out = np.zeros_like(w)
            cv2.copyTo(w, self._owner_u8[i], out)  # 맡은 셀만 복사(out을 그 자리에서 고침)
        if out is None:
            out = np.zeros((self.grid.height, self.grid.width, 3), np.uint8)
        return out


# ---------------------------------------------------------------------------
# 검출 결과의 지면 투영과 카메라 간 병합
# ---------------------------------------------------------------------------
# 클래스별 기본 크기 (길이, 폭) m. 단안 박스 폭이 부정확할 때 표시용으로 쓴다.
DEFAULT_SIZE = {
    "car": (4.5, 1.9),
    "truck": (8.0, 2.5),
    "bus": (11.0, 2.6),
    "motorcycle": (2.0, 0.8),
    "bicycle": (1.8, 0.6),
    "person": (0.6, 0.6),
}


def project_detections(
    cam_index: int,
    cam: Camera,
    dets: list[Detection],
    max_range: float = 40.0,
    scale: float = 1.0,
    vehicle_center: str = "ray",
) -> list[GroundObject]:
    """박스 하단 중앙(접지점)을 지면으로 투영한다.

    차량은 박스 하단이 가장 가까운 면이므로, 투영점을 카메라 시선 방향으로 차체 길이의 절반만큼
    밀어 중심을 추정한다. 보행자/이륜차는 그대로 쓴다.
    scale: 장면 축척. 책상 위 모형(장난감 차)이면 실제 차에 대한 모형 크기 비율(예: 0.15)을 주어
           기본 크기와 거리 문턱값을 같은 비율로 줄인다. 1.0이면 예전과 같다.
    vehicle_center: 접지점에서 차 중심까지 미는 거리를 정하는 방식.
      "ray"(기본, 예전과 같음): 시선 방향 어느 쪽에서 보든 차 길이의 절반만큼 민다.
      "box": 차가 도로와 나란히(차량 X축 방향) 서 있다고 보고, 카메라 광축 방향과 차 상자 크기로 중심을 구한다
             (box_center). 뒤쪽 옆을 보는 측면 카메라에서 차 중심이 앞으로 1m 넘게 치우치던 것을 줄인다.
    """
    if vehicle_center not in ("ray", "box"):
        raise ValueError(f"vehicle_center는 ray 또는 box여야 합니다: {vehicle_center}")
    if not dets:
        return []
    bottoms = np.array([[(d.box[0] + d.box[2]) / 2.0, d.box[3]] for d in dets])
    left = np.array([[d.box[0], d.box[3]] for d in dets])
    right = np.array([[d.box[2], d.box[3]] for d in dets])
    gc, vc = cam.pixel_to_ground(bottoms, max_range)
    gl, vl = cam.pixel_to_ground(left, max_range)
    gr, vr = cam.pixel_to_ground(right, max_range)
    axis = camera_ground_axis(cam) if vehicle_center == "box" else None
    out = []
    for k, d in enumerate(dets):
        if not vc[k]:
            continue
        length, width = DEFAULT_SIZE.get(d.label, (1.0, 1.0))
        length, width = length * scale, width * scale  # 모형이면 크기도 축척만큼 작다
        xy = gc[k].copy()
        ray = xy - cam.position[:2]
        n = np.linalg.norm(ray)
        if n > 1e-6 and length > 1.5 * scale:
            u = ray / n
            if vehicle_center == "box" and axis is not None:
                xy = box_center(xy, cam.position[:2], axis, length, width)
            else:
                xy = xy + u * (length / 2.0)
        if vl[k] and vr[k]:
            w_meas = float(np.linalg.norm(gl[k] - gr[k]))
            if 0.3 * scale < w_meas < 4.0 * scale:
                width = w_meas
        out.append(
            GroundObject(
                label=d.label,
                score=d.score,
                xy=xy,
                size=(length, width),
                cam_index=cam_index,
                distance=float(n),
            )
        )
    return out


def box_center(g: np.ndarray, cam_xy: np.ndarray, axis: np.ndarray, length: float, width: float) -> np.ndarray:
    """차 상자(차량 X축과 나란한 길이 x 폭)의 중심을 박스 아래 끝 투영점 g에서 추정한다.

    박스 아래 끝(영상에서 가장 낮은 줄)은 차 바닥 네모 중 카메라 앞쪽 깊이(광축 방향 거리)가 가장 작은 점이다.
    그래서 g는 "가장 가까운 깊이" 선 위에 있고, 중심은 그보다 깊이가
        h = 길이/2 x |광축_x| + 폭/2 x |광축_y|
    만큼 더 깊다(네모를 광축 방향으로 본 반폭). 중심의 방위는 박스 가운데 열, 곧 카메라에서 g로 가는 시선이다.
    그래서 중심 = 카메라 + 시선 x t, t = (광축 . (g - 카메라) + h) / (광축 . 시선).
    전방 카메라가 정면의 차를 보면 h = 길이/2가 되어 예전 방식과 같고, 뒤쪽 옆을 보는 측면 카메라에서는
    차를 비스듬히 보는 만큼 덜 밀거나 더 민다.
    g, cam_xy: 지면 좌표(m). axis: 카메라 광축을 지면에 내린 단위 벡터.
    """
    ray = g - cam_xy
    n = float(np.linalg.norm(ray))
    if n < 1e-6:
        return g
    ray = ray / n
    cos = float(axis @ ray)
    if cos < 0.2:  # 시선이 광축과 거의 수직이면(화면 가장자리) 깊이로 풀기 어렵다. 예전처럼 길이 절반만 민다
        return g + ray * (length / 2.0)
    h = length / 2.0 * abs(float(axis[0])) + width / 2.0 * abs(float(axis[1]))
    t = (float(axis @ (g - cam_xy)) + h) / cos
    return cam_xy + ray * t


def camera_ground_axis(cam: Camera) -> np.ndarray | None:
    """카메라 광축(카메라 좌표 +Z)을 차량 좌표 지면에 내린 단위 벡터. 거의 수직으로 내려다보면 None."""
    a = np.asarray(cam.R_vc, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])
    v = a[:2]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-3 else None


def merge_cross_camera(objs: list[GroundObject], radius: float = 1.5, scale: float = 1.0) -> list[GroundObject]:
    """카메라 경계에서 중복된 같은 객체를 합친다(점수 높은 순으로 흡수).

    scale: 장면 축척(project_detections와 같은 값). 합칠 반경과 거리 가중의 하한을 같이 줄인다.
    """
    radius = radius * scale
    floor = 1.0 * scale  # 아주 가까운 카메라 하나가 가중치를 독차지하지 않게 두는 거리 하한
    objs = sorted(objs, key=lambda o: -o.score)
    kept: list[GroundObject] = []
    for o in objs:
        dup = False
        for k in kept:
            same_group = _group(k.label) == _group(o.label)
            if same_group and np.linalg.norm(k.xy - o.xy) < radius and k.cam_index != o.cam_index:
                # 거리가 가까운 카메라의 추정이 더 정확하므로 가중 평균
                wk, wo = 1.0 / max(k.distance, floor), 1.0 / max(o.distance, floor)
                k.xy = (k.xy * wk + o.xy * wo) / (wk + wo)
                dup = True
                break
        if not dup:
            kept.append(o)
    return kept


def _group(label: str) -> str:
    return "vehicle" if label in ("car", "truck", "bus") else label
