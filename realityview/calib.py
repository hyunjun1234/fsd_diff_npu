from __future__ import annotations

import re
from typing import Sequence

import cv2
import numpy as np

from .camera_model import Camera, pose_from_pnp

# 카메라 보정 계산 모음. scripts/calib_intrinsics.py, calib_extrinsics.py가 쓰고,
# 하드웨어 없이 합성 코너로 시험할 수 있게 파일 읽기와 화면 출력은 넣지 않는다.
# 길이 단위는 모두 m (체커보드 칸 20mm = 0.02).


# ---------------------------------------------------------------------------
# 체커보드 코너
# ---------------------------------------------------------------------------
def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    """보드 좌표계의 내부 코너 위치 (cols*rows, 3). 첫 코너가 원점, 가로(cols)가 x, 세로(rows)가 y."""
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def find_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """흑백 영상에서 내부 코너 (cols*rows, 2)를 찾는다. 못 찾으면 None.

    어안 렌즈 가장자리처럼 휜 보드에서도 잘 잡히는 findChessboardCornersSB를 먼저 쓰고,
    안 되면 예전 방식(findChessboardCorners + cornerSubPix)으로 한 번 더 찾는다.
    """
    flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags_sb)
    if ok:
        return c.reshape(-1, 2).astype(np.float64)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if not ok:
        return None
    # 작은 칸(모형용 15~25mm)은 영상에서 수십 px라 창을 작게(5x5) 둔다
    c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return c.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------------------
# 내부 파라미터
# ---------------------------------------------------------------------------
def _bad_view_index(err: cv2.error) -> int | None:
    """cv2.fisheye.calibrate의 CALIB_CHECK_COND 오류 문장에서 문제 사진 번호를 꺼낸다."""
    m = re.search(r"input array (\d+)", str(err))
    return int(m.group(1)) if m else None


def calibrate_intrinsics(
    obj_pts: Sequence[np.ndarray],
    img_pts: Sequence[np.ndarray],
    size_wh: tuple[int, int],
    model: str = "pinhole",
) -> dict:
    """체커보드 코너로 K와 왜곡 계수를 구한다.

    obj_pts[i]: (N,3) 보드 좌표, img_pts[i]: (N,2) 픽셀 좌표. size_wh: 영상 (폭, 높이).
    돌려주는 값: rms(px), K, D, per_view(사진별 RMS px), used(쓴 사진 번호), dropped(뺀 사진 번호와 이유)
    어안 보정은 조건이 나쁜 사진(보드가 너무 작거나 비스듬함)이 하나만 있어도 실패하므로,
    그 사진 번호를 오류에서 읽어 빼고 다시 푼다.
    """
    idx = list(range(len(obj_pts)))
    dropped: list[tuple[int, str]] = []
    while True:
        if len(idx) < 3:
            raise ValueError(f"쓸 수 있는 사진이 3장 미만입니다(뺀 사진: {dropped})")
        O = [np.asarray(obj_pts[i], np.float64).reshape(-1, 1, 3) for i in idx]
        I = [np.asarray(img_pts[i], np.float64).reshape(-1, 1, 2) for i in idx]
        if model == "fisheye":
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_CHECK_COND
            crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
            try:
                rms, K, D, rv, tv = cv2.fisheye.calibrate(O, I, size_wh, np.zeros((3, 3)), np.zeros((4, 1)),
                                                          flags=flags, criteria=crit)
            except cv2.error as e:
                bad = _bad_view_index(e)
                if bad is None or bad >= len(idx):
                    raise
                dropped.append((idx[bad], "어안 보정 조건 나쁨(CALIB_CHECK_COND)"))
                del idx[bad]
                continue
        elif model == "pinhole":
            rms, K, D, rv, tv = cv2.calibrateCamera(
                [o.astype(np.float32) for o in O], [i.astype(np.float32) for i in I], size_wh, None, None
            )
        else:
            raise ValueError(f"model은 pinhole 또는 fisheye: {model}")
        break

    # ---- 사진별 재투영 오차(어느 사진이 나쁜지 보려고) ----
    per_view = []
    for o, i, r, t in zip(O, I, rv, tv):
        if model == "fisheye":
            p, _ = cv2.fisheye.projectPoints(o, r, t, K, D)
        else:
            p, _ = cv2.projectPoints(o, r, t, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return {
        "rms": float(rms), "K": np.asarray(K), "D": np.asarray(D).ravel(), "model": model,
        "per_view": per_view, "used": idx, "dropped": dropped, "size_wh": list(size_wh),
    }


def intrinsics_yaml(res: dict) -> str:
    """calibrate_intrinsics 결과를 설정 파일 intrinsics 항목에 붙여 넣을 글로."""
    K, D = res["K"], res["D"]
    lines = [
        "intrinsics:",
        f"  model: {res['model']}",
        f"  fx: {K[0, 0]:.3f}",
        f"  fy: {K[1, 1]:.3f}",
        f"  cx: {K[0, 2]:.3f}",
        f"  cy: {K[1, 2]:.3f}",
        f"  dist: [{', '.join(f'{v:.6f}' for v in D)}]",
        f"  # RMS {res['rms']:.3f}px, 사진 {len(res['used'])}장, 영상 {res['size_wh'][0]}x{res['size_wh'][1]}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 외부 파라미터 (바닥 기준점)
# ---------------------------------------------------------------------------
def floor_board_xy(cols: int, rows: int, square: float, origin_xy: Sequence[float], yaw_deg: float) -> np.ndarray:
    """바닥에 깐 체커보드의 내부 코너를 차량(모형) 좌표 (N,2)로.

    origin_xy: 첫 코너(find_corners가 돌려주는 0번)의 차량 좌표 X, Y (m, 줄자로 잼)
    yaw_deg: 보드 가로 방향(코너 0번에서 1번으로)이 차량 X축과 이루는 각(좌회전 +)
    보드 세로 방향(0번에서 cols번으로)은 위에서 내려다볼 때 가로 방향의 오른쪽 90도다.
    OpenCV 코너 검출기가 영상에서 가로 다음 세로를 시계 방향으로 맞추고, 바닥 보드를 위에서 보면
    같은 순서가 되기 때문이다(바닥 보드를 어안 카메라 영상으로 그려 find_corners로 확인한 시험 있음).
    0번 코너가 어느 것인지는 촬영 영상에서 확인한다(calib_extrinsics.py가 그림으로 남김).
    """
    b = board_points(cols, rows, square)[:, :2]
    b[:, 1] *= -1.0  # 세로 방향을 가로의 오른쪽(시계 방향)으로
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return b @ R.T + np.asarray(origin_xy, np.float64)


def solve_extrinsics(cam: Camera, uv: np.ndarray, XY: np.ndarray) -> dict:
    """픽셀 좌표 uv(N,2)와 지면 좌표 XY(N,2)로 카메라 위치와 자세를 구한다. cam은 내부 파라미터만 쓴다.

    왜곡을 먼저 풀어 정규화 좌표로 바꾼 뒤 PnP를 푼다(어안, 핀홀 공통).
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    XYZ = np.hstack([np.asarray(XY, np.float64).reshape(-1, 2), np.zeros((len(uv), 1))])
    if len(uv) < 4:
        raise ValueError("점이 4개 이상 필요합니다(6개 이상 권장)")
    if cam.model == "fisheye":
        n = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist[:4])
    else:
        n = cv2.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist)
    ok, rvec, tvec = cv2.solvePnP(XYZ, n, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP 실패")
    pos, yaw, pitch, roll = pose_from_pnp(rvec, tvec)
    return {"position": pos, "yaw": yaw, "pitch": pitch, "roll": roll}


def apply_pose(cam: Camera, pose: dict) -> None:
    cam.position = np.asarray(pose["position"], np.float64)
    cam.yaw_deg, cam.pitch_deg, cam.roll_deg = pose["yaw"], pose["pitch"], pose["roll"]


def ground_errors(cam: Camera, uv: np.ndarray, XY: np.ndarray) -> dict:
    """지금 cam의 내부, 외부 파라미터로 재투영 오차(px)와 지면 역투영 오차(m)를 잰다."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    XY = np.asarray(XY, np.float64).reshape(-1, 2)
    pred, _ = cam.project(np.hstack([XY, np.zeros((len(XY), 1))]))
    px = np.linalg.norm(pred - uv, axis=1)
    g, valid = cam.pixel_to_ground(uv)
    gerr = np.where(valid, np.linalg.norm(g - XY, axis=1), np.nan)
    return {
        "reproj_px_mean": float(np.nanmean(px)), "reproj_px_max": float(np.nanmax(px)),
        "ground_m_mean": float(np.nanmean(gerr)), "ground_m_max": float(np.nanmax(gerr)),
        "invalid": int(np.count_nonzero(~valid)), "n": int(len(uv)),
    }


def extrinsics_yaml(cam: Camera) -> str:
    p = cam.position
    return (
        "extrinsics:\n"
        f"  position: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]\n"
        f"  yaw: {cam.yaw_deg:.2f}\n  pitch: {cam.pitch_deg:.2f}\n  roll: {cam.roll_deg:.2f}\n"
    )
