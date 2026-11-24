from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .bev import BevGrid
from .camera_model import Camera
from .lanes import LaneConfig, LaneCurve, LaneDetector

# 주행 중 외부 보정 자동 점검. 설정을 고치지 않고 경고만 낸다.
#
# 생각의 줄기
#   곧은 도로의 차선은 지면에서 서로 나란하고 차량 X축과도 거의 나란하다. 이 직선들이 영상에서 만나는 점
#   (소실점)은 카메라에서 본 "도로 방향"이고, 지평선은 그 소실점을 지난다. 설정 값이 맞으면 소실점 방향을
#   차량 좌표로 돌렸을 때 정확히 +X(후방 카메라는 -X)가 나온다. pitch가 틀리면 위아래로, yaw가 틀리면
#   좌우로 어긋나므로 그 어긋난 각도가 곧 설정 값과 실제 각도의 차이다.
#
# 계산 순서(카메라마다, 몇 프레임에 한 번)
#   1) 탑뷰 바닥 영상에서 그 카메라가 맡은 셀만으로 차선을 찾는다(lanes.py의 LaneDetector를 카메라별로 하나씩).
#      파이프라인의 차선 곡선은 앞뒤 카메라 화소를 한 곡선으로 맞춘 것이라 한 카메라의 틀어짐이 묽어진다.
#   2) 곡선 위 점을 카메라 위치에서 본 광선으로 바꾼다. 바닥 영상은 설정 값으로 편 것이므로, 이 광선은
#      그 화소의 실제 광선을 설정 값 기준 차량 좌표로 나타낸 것이다(어안 왜곡도 이미 풀려 있다).
#   3) 곧은 차선 하나의 광선들은 카메라 중심을 지나는 평면 하나에 놓인다. 차선 둘 이상의 평면이 공통으로
#      품는 방향이 소실 방향이다(평면 법선들에 모두 수직인 방향, SVD로 구함).
#   4) "설정 값에서 yaw, pitch를 얼마 바꾸면 도로 방향(+X 또는 -X)이 이 소실 방향으로 보이는가"를 풀어
#      yaw, pitch 차이를 낸다. 지평선 위치 차이(px)는 설정 값으로 예상한 소실점과 잰 소실점의 세로 차이다.
#   5) 굽은 길(보인 구간에서 직선과 많이 다름), 차선 하나뿐, 한쪽 각도가 너무 큰 표본(차선 변경, 오검출)은 버리고, 최근 window_s 초
#      평균이 warn_deg를 넘으면 경고한다.
#
# 한계
#   yaw 차이는 "차가 차선과 나란히 달린다"는 가정에 기대므로 한 프레임 값은 차선 안 흔들림, 차선 변경만큼
#   틀린다. 오래 평균한 값만 의미가 있다. roll은 소실점으로 알 수 없어 보지 않는다. 오르막 내리막에서는 pitch
#   차이가 길의 기울기만큼 섞인다.

log = logging.getLogger("calib_monitor")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class CalibMonitorConfig:
    every: int = 10  # 몇 프레임마다 볼지(파이프라인이 쓴다)
    window_s: float = 60.0  # 평균을 내는 최근 시간(초)
    min_samples: int = 10  # 창 안 표본이 이보다 적으면 판정하지 않는다
    warn_deg: float = 1.0  # 평균 차이가 이 각도(도)를 넘으면 경고
    max_sample_deg: float = 6.0  # 표본 하나의 차이가 이보다 크면 버린다(차선 변경, 오검출)
    # 굽은 차선 거르기: 보인 구간의 양 끝을 이은 직선에서 곡선이 가장 많이 벗어난 거리(|a| x (구간/2)^2, 실차 m)가
    # 이 값보다 크면 쓰지 않는다. 2차 계수만으로 거르면 먼 화소가 적은 후방 카메라의 맞춤 잡음까지 굽은 길로 본다
    max_sag_m: float = 0.15
    min_span_m: float = 4.0  # 차선이 보인 구간 길이 하한(실차 m)
    min_lanes: int = 2  # 한 카메라에서 곧은 차선이 이 수 이상이어야 표본을 만든다
    sample_step_m: float = 0.5  # 곡선에서 점을 뽑는 간격(실차 m)
    log_every_s: float = 10.0  # 같은 카메라 경고를 로그에 다시 남기는 최소 간격(초)


# ---------------------------------------------------------------------------
# 기하 계산
# ---------------------------------------------------------------------------
def lane_plane_normal(cam: Camera, pts_xy: np.ndarray) -> np.ndarray | None:
    """지면 점들(N,2)을 카메라 위치에서 본 광선으로 바꾸고, 그 광선들이 놓인 평면의 단위 법선을 구한다."""
    if len(pts_xy) < 3:
        return None
    rays = np.hstack([pts_xy, np.zeros((len(pts_xy), 1))]) - cam.position
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    # 법선은 모든 광선과 수직 = 광선 행렬의 가장 작은 특이값 방향
    _, s, vt = np.linalg.svd(rays, full_matrices=False)
    if s[1] < 1e-4:  # 점이 한 방향에 몰려 평면이 정해지지 않음
        return None
    return vt[-1]


def road_axis(cam: Camera) -> np.ndarray:
    """이 카메라가 보는 도로 방향(차량 좌표 +X 또는 -X). 광축이 앞을 보면 +X."""
    fwd = cam.R_vc[:, 2]
    return np.array([1.0 if fwd[0] >= 0 else -1.0, 0.0, 0.0])


def vanishing_direction(cam: Camera, normals: list[np.ndarray]) -> np.ndarray | None:
    """차선 평면 법선 둘 이상에 모두 수직인 단위 방향(설정 값 기준 차량 좌표). 도로 방향 쪽을 향하게 부호를 맞춘다."""
    if len(normals) < 2:
        return None
    N = np.asarray(normals)
    # 법선이 2개뿐이면 줄인 SVD는 3번째 방향(법선들에 수직인 방향)을 주지 않으므로 full_matrices=True로 구한다
    _, s, vt = np.linalg.svd(N, full_matrices=True)
    if s[1] < 1e-3:  # 평면들이 거의 같음(차선이 겹침): 교선이 정해지지 않음
        return None
    d = vt[2]
    return d if d @ road_axis(cam) > 0 else -d


def angle_offsets(cam: Camera, d: np.ndarray, iters: int = 8) -> tuple[float, float]:
    """소실 방향 d(설정 값 기준 차량 좌표)를 만드는 실제 카메라의 yaw, pitch 차이(도, 실제 - 설정).

    실제 카메라가 (yaw+dy, pitch+dp)면 도로 방향 e는 실제 광학 좌표에서 R_cv(실제) e로 찍히고, 설정 값으로 펴면
    R_vc(설정) R_cv(실제) e로 보인다. 이것이 d와 같아지는 (dy, dp)를 가우스-뉴턴(수치 미분)으로 푼다.
    """
    import dataclasses

    e = road_axis(cam)
    Rvc0 = cam.R_vc

    def f(t):
        c = dataclasses.replace(cam, yaw_deg=cam.yaw_deg + t[0], pitch_deg=cam.pitch_deg + t[1])
        v = Rvc0 @ (c.R_cv @ e)
        return v / np.linalg.norm(v) - d

    t = np.zeros(2)
    for _ in range(iters):
        r = f(t)
        J = np.stack([(f(t + 1e-4 * k) - r) / 1e-4 for k in np.eye(2)], axis=1)
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        t += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return float(t[0]), float(t[1])


def horizon_shift_px(cam: Camera, d: np.ndarray) -> float | None:
    """잰 소실점과 설정 값으로 예상한 소실점의 세로 차이(px, +면 잰 것이 아래). 영상 밖 방향이면 None."""
    far = 1e4
    uv, ok = cam.project(np.stack([cam.position + far * road_axis(cam), cam.position + far * d]))
    if not ok.all():
        return None
    return float(uv[1, 1] - uv[0, 1])


def estimate_from_lanes(cam: Camera, lanes: list[LaneCurve], cfg: CalibMonitorConfig, scale: float = 1.0) -> dict | None:
    """한 카메라가 본 차선 곡선들로 yaw, pitch 차이 표본 하나. 조건이 안 맞으면 None."""
    normals = []
    for ln in lanes:
        a = float(ln.coeffs[0])
        span = ln.x_range[1] - ln.x_range[0]
        if abs(a) * (span / 2.0) ** 2 > cfg.max_sag_m * scale or span < cfg.min_span_m * scale:
            continue
        n = lane_plane_normal(cam, ln.points(cfg.sample_step_m * scale))
        if n is not None:
            normals.append(n)
    if len(normals) < cfg.min_lanes:
        return None
    d = vanishing_direction(cam, normals)
    if d is None:
        return None
    dyaw, dpitch = angle_offsets(cam, d)
    return {"yaw": dyaw, "pitch": dpitch, "horizon_px": horizon_shift_px(cam, d), "lanes": len(normals)}


# ---------------------------------------------------------------------------
# 점검기: 카메라별 차선 검출, 표본 모으기, 시간 평균, 경고
# ---------------------------------------------------------------------------
class CalibMonitor:
    """주행 중 외부 보정이 틀어졌는지 본다. update()에 탑뷰 바닥 영상과 시각을 넣는다.

    owner: 탑뷰 셀마다 그 셀을 맡은 카메라 번호(IpmCompositor.owner). 카메라마다 자기 셀만으로 차선을 찾는다.
    """

    def __init__(self, cameras: list[Camera], grid: BevGrid, owner: np.ndarray, scale: float = 1.0,
                 cfg: CalibMonitorConfig | None = None, lane_cfg: LaneConfig | None = None):
        self.cameras = cameras
        self.cfg = cfg or CalibMonitorConfig()
        self.scale = float(scale)
        lc = lane_cfg or LaneConfig()
        self.detectors = [LaneDetector(grid, lc, owner == i, self.scale) for i in range(len(cameras))]
        self.samples: list[deque] = [deque() for _ in cameras]  # (시각, yaw 차이, pitch 차이, 지평선 px)
        self.runs = 0
        self.used = [0] * len(cameras)  # 표본으로 쓴 횟수(카메라별)
        self._last_log = [-1e18] * len(cameras)
        self._warnings: list[str] = []

    def update(self, ground_bgr: np.ndarray | None, t: float, road_mask: np.ndarray | None = None) -> list[str]:
        """바닥 영상 한 장을 보고 지금 경고 목록(글)을 돌려준다."""
        self.runs += 1
        c = self.cfg
        if ground_bgr is not None:
            for i, (cam, det) in enumerate(zip(self.cameras, self.detectors)):
                est = estimate_from_lanes(cam, det.detect(ground_bgr, road_mask), c, self.scale)
                if est is None or max(abs(est["yaw"]), abs(est["pitch"])) > c.max_sample_deg:
                    continue
                self.samples[i].append((float(t), est["yaw"], est["pitch"], est["horizon_px"]))
                self.used[i] += 1
        # 창 밖 표본 버리기
        for q in self.samples:
            while q and q[0][0] < t - c.window_s:
                q.popleft()
        self._warnings = []
        for i, st in enumerate(self.status()):
            if st["warn"]:
                msg = (f"보정 점검: {st['camera']} 카메라 각도 차이 yaw {st['yaw']:+.1f}도, pitch {st['pitch']:+.1f}도"
                       f"(최근 {c.window_s:.0f}초 표본 {st['n']}개 평균, 기준 {c.warn_deg:.1f}도)")
                self._warnings.append(msg)
                if t - self._last_log[i] >= c.log_every_s:
                    log.warning(msg)
                    self._last_log[i] = t
        return list(self._warnings)

    def status(self) -> list[dict]:
        """카메라별 최근 창 평균. 표본이 부족하면 값은 None, warn은 False."""
        out = []
        for cam, q in zip(self.cameras, self.samples):
            n = len(q)
            st = {"camera": cam.name, "n": n, "yaw": None, "pitch": None, "horizon_px": None, "warn": False}
            if n >= self.cfg.min_samples:
                arr = np.array([(s[1], s[2]) for s in q])
                st["yaw"], st["pitch"] = float(arr[:, 0].mean()), float(arr[:, 1].mean())
                hz = [s[3] for s in q if s[3] is not None]
                st["horizon_px"] = float(np.mean(hz)) if hz else None
                st["warn"] = max(abs(st["yaw"]), abs(st["pitch"])) > self.cfg.warn_deg
            out.append(st)
        return out

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def summary(self) -> dict:
        """끝날 때 남길 요약(json용)."""
        return {"runs": self.runs, "used": dict(zip([c.name for c in self.cameras], self.used)),
                "status": self.status()}


def draw_warnings(img: np.ndarray, warnings: list[str]) -> None:
    """화면 왼쪽 아래에 경고 글을 주황색으로 적는다(img를 고침). 한글은 OpenCV 글꼴에 없어 영문 약식으로 쓴다."""
    import cv2
    import re

    y = img.shape[0] - 10
    for w in reversed(warnings):
        m = re.search(r"(\S+) 카메라 각도 차이 yaw ([+-][\d.]+)도, pitch ([+-][\d.]+)도", w)
        text = f"CALIB CHECK {m.group(1)}: yaw {m.group(2)} pitch {m.group(3)} deg" if m else "CALIB CHECK"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (4, y - th - 6), (12 + tw, y + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        y -= th + 14
