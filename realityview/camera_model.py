"""카메라 모델과 지면(Z=0) 투영.

좌표계
- 차량 좌표계: X 전방, Y 좌측, Z 위. 원점은 뒷바퀴 축 중앙 바닥(또는 원하는 기준점).
- 카메라 광학 좌표계: x 오른쪽, y 아래, z 광축(전방).
- 자세: yaw(차량 Z축 기준, 좌회전 +), pitch(아래로 숙이면 +), roll(광축 기준 시계 방향 +). 단위는 도(degree).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# 광학 좌표축을 "정면을 보는 카메라"의 차량 좌표로 표현한 회전 (열 = 광학 x, y, z)
_R_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray  # 3x3 내부 파라미터
    position: np.ndarray  # 차량 좌표계에서 카메라 위치 (m)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    dist: np.ndarray = field(default_factory=lambda: np.zeros(4))
    model: str = "pinhole"  # "pinhole" 또는 "fisheye"
    source: dict = field(default_factory=dict)  # 입력 장치 설정(실행 스크립트에서 사용)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        if self.model not in ("pinhole", "fisheye"):
            raise ValueError(f"unknown camera model: {self.model}")

    # ---- 외부 파라미터 -------------------------------------------------
    @property
    def R_vc(self) -> np.ndarray:
        """카메라 광학 좌표 → 차량 좌표 회전."""
        y, p, r = np.radians([self.yaw_deg, self.pitch_deg, self.roll_deg])
        # roll은 광축(차량 기준으로는 정면 X축) 기준 회전
        return _rz(y) @ _ry(p) @ _rx(r) @ _R_BASE

    @property
    def R_cv(self) -> np.ndarray:
        """차량 좌표 → 카메라 광학 좌표 회전."""
        return self.R_vc.T

    @property
    def rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.R_cv
        t = -R @ self.position
        rvec, _ = cv2.Rodrigues(R)
        return rvec, t.reshape(3, 1)

    # ---- 투영 ---------------------------------------------------------
    def project(self, pts_vehicle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """차량 좌표 점(N,3)을 (왜곡 포함) 픽셀 좌표(N,2)로. 두 번째 값은 카메라 앞쪽 여부."""
        pts = np.asarray(pts_vehicle, dtype=np.float64).reshape(-1, 3)
        z_cam = (self.R_cv @ (pts - self.position).T)[2]
        in_front = z_cam > 1e-3
        rvec, tvec = self.rvec_tvec
        if self.model == "fisheye":
            uv, _ = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, self.K, self.dist[:4])
        else:
            uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, self.K, self.dist)
        return uv.reshape(-1, 2), in_front

    def pixel_rays(self, uv: np.ndarray) -> np.ndarray:
        """픽셀 좌표(N,2) → 차량 좌표계 단위 방향 벡터(N,3). 왜곡 보정 포함."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            n = cv2.fisheye.undistortPoints(uv, self.K, self.dist[:4])
        else:
            n = cv2.undistortPoints(uv, self.K, self.dist)
        n = n.reshape(-1, 2)
        d_cam = np.hstack([n, np.ones((len(n), 1))])
        d_veh = (self.R_vc @ d_cam.T).T
        return d_veh / np.linalg.norm(d_veh, axis=1, keepdims=True)

    def pixel_to_ground(self, uv: np.ndarray, max_range: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
        """픽셀이 가리키는 지면(Z=0) 위의 점(N,2: X,Y)과 유효 여부(N,)."""
        d = self.pixel_rays(uv)
        valid = d[:, 2] < -1e-6  # 아래를 향해야 지면과 만난다
        t = np.where(valid, -self.position[2] / np.where(valid, d[:, 2], -1.0), 0.0)
        xy = self.position[:2] + t[:, None] * d[:, :2]
        valid &= np.linalg.norm(xy - self.position[:2], axis=1) <= max_range
        return xy, valid


def camera_from_dict(d: dict) -> Camera:
    intr = d["intrinsics"]
    if "K" in intr:
        K = np.asarray(intr["K"], dtype=np.float64)
    else:
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    ext = d["extrinsics"]
    return Camera(
        name=d["name"],
        width=int(d["width"]),
        height=int(d["height"]),
        K=K,
        dist=np.asarray(intr.get("dist", [0, 0, 0, 0]), dtype=np.float64),
        model=intr.get("model", "pinhole"),
        position=np.asarray(ext["position"], dtype=np.float64),
        yaw_deg=float(ext.get("yaw", 0.0)),
        pitch_deg=float(ext.get("pitch", 0.0)),
        roll_deg=float(ext.get("roll", 0.0)),
        source=d.get("source", {}),
    )


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """수평 화각으로 대략적인 핀홀 K를 만든다(보정 전 임시값)."""
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


def pose_from_pnp(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """solvePnP 결과(차량→카메라) → (카메라 위치, yaw, pitch, roll) 도 단위."""
    R_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    position = -R_cv.T @ t
    M = R_cv.T @ _R_BASE.T  # = Rz(yaw) Ry(pitch) Rx(roll)
    yaw = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    pitch = np.degrees(np.arcsin(np.clip(-M[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(M[2, 1], M[2, 2]))
    return position, float(yaw), float(pitch), float(roll)
