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


# ---------------------------------------------------------------------------
# 빠른 재보정: 바닥 마커 (scripts/recalib_markers.py가 쓴다)
# ---------------------------------------------------------------------------
# 내부 보정과 외부 보정 초기값이 이미 있을 때, 바닥에 놓은 마커 4개 이상의 실측 중심 위치로
# 카메라 각도(기본)나 각도와 위치(옵션)를 다시 구한다. 체커보드를 깔 자리가 없는 차에서도 쓴다.
# 마커는 두 가지를 쓴다.
#   aruco : OpenCV ArUco 마커(기본 사전 DICT_4X4_50). 번호가 찍혀 있어 어느 마커인지 헷갈리지 않는다
#   circle: 흰 바탕의 검은 원. ArUco가 없는 OpenCV 판에서 쓴다. 번호가 없으므로 지금 외부 값으로
#           마커 자리를 영상에 투영해 가장 가까운 원을 그 마커로 본다(초기값이 크게 틀리면 못 쓴다)
# 마커의 "중심"만 쓴다. 바닥에 놓을 때 줄자로 재기 쉬운 것이 중심이고, 마커를 어느 방향으로 돌려 놓았는지는
# 몰라도 된다. ArUco 중심은 네 꼭짓점의 왜곡을 푼 뒤 두 대각선의 교점으로 구한다(원근에서도 정확한 중심).
ARUCO_DICT = "DICT_4X4_50"


def aruco_available() -> bool:
    """이 OpenCV에 ArUco 검출기(4.7 이후 API)가 있는지."""
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")


def marker_image(marker_id: int, side_px: int, kind: str = "aruco", dict_name: str = ARUCO_DICT) -> np.ndarray:
    """인쇄용 마커 그림(흑백 uint8). 둘레에 마커 한 변의 1/4만큼 흰 여백을 둔다(검출에 여백이 필요하다)."""
    pad = side_px // 4
    out = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, np.uint8)
    if kind == "aruco":
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        out[pad:pad + side_px, pad:pad + side_px] = cv2.aruco.generateImageMarker(d, int(marker_id), side_px)
    elif kind == "circle":
        c = pad + side_px // 2
        cv2.circle(out, (c, c), side_px // 2, 0, -1, cv2.LINE_AA)
    else:
        raise ValueError(f"kind는 aruco 또는 circle: {kind}")
    return out


def render_floor_markers(cam: Camera, img: np.ndarray, markers: Sequence[tuple], kind: str = "aruco",
                         dict_name: str = ARUCO_DICT, supersample: int = 3) -> np.ndarray:
    """카메라 영상 img 위에 바닥(Z=0)에 놓인 마커를 그려 넣는다(시험과 모의 확인용, img를 고쳐서 돌려줌).

    markers: (번호, X, Y, 한 변 m, 돌린 각 도) 목록. 마커 그림의 위쪽이 차량 +X를 보는 것이 각도 0이다.
    화소마다 지면 위치를 구해 마커 그림에서 색을 읽는다. 가장자리 계단을 줄이려고 화소를 supersample^2개로
    나눠 평균 낸다.
    """
    h, w = img.shape[:2]
    s = max(1, int(supersample))
    off = (np.arange(s) + 0.5) / s
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    acc = np.zeros((h, w), np.float64)
    hit = np.zeros((h, w), np.float64)
    tex_px = 200
    for mid, X0, Y0, side, yaw in markers:
        tex = marker_image(int(mid), tex_px, kind, dict_name).astype(np.float64)
        full = side * tex.shape[0] / tex_px  # 여백까지 포함한 한 변(m)
        a = np.radians(yaw)
        # 마커가 영상에 맺히는 대략의 자리만 계산한다(여백 포함 네 꼭짓점의 투영 상자 + 여유)
        hc = full / 2 * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        corners = hc @ Rm.T + [X0, Y0]
        uv, ok = cam.project(np.hstack([corners, np.zeros((4, 1))]))
        if not ok.all():
            continue
        u0, v0 = np.clip(np.floor(uv.min(0)) - 3, 0, [w - 1, h - 1]).astype(int)
        u1, v1 = np.clip(np.ceil(uv.max(0)) + 3, 0, [w - 1, h - 1]).astype(int)
        if u1 <= u0 or v1 <= v0:
            continue
        uu, vv = u[v0:v1 + 1, u0:u1 + 1], v[v0:v1 + 1, u0:u1 + 1]
        for dy in off:
            for dx in off:
                with np.errstate(all="ignore"):  # 어안 가장자리처럼 왜곡을 못 푸는 화소는 valid가 거짓으로 걸러 준다
                    g, valid = cam.pixel_to_ground(np.stack([(uu + dx).ravel(), (vv + dy).ravel()], 1))
                g = np.where(valid[:, None], g, 1e9)
                # 지면 점을 마커 좌표로(마커 위쪽 = +X를 돌린 방향). 그림 행은 위가 0이라 위쪽 방향을 뒤집는다
                loc = (g - [X0, Y0]) @ Rm  # 열 0: 마커 위쪽 방향, 열 1: 마커 왼쪽 방향
                col = (0.5 - loc[:, 1] / full) * tex.shape[1]
                row = (0.5 - loc[:, 0] / full) * tex.shape[0]
                inside = valid & (col >= 0) & (col < tex.shape[1]) & (row >= 0) & (row < tex.shape[0])
                val = np.zeros(len(g))
                val[inside] = tex[row[inside].astype(int), col[inside].astype(int)]
                acc[v0:v1 + 1, u0:u1 + 1] += val.reshape(uu.shape)
                hit[v0:v1 + 1, u0:u1 + 1] += inside.reshape(uu.shape)
    m = hit > 0
    # 마커가 덮은 비율만큼 원래 색과 섞는다(마커 가장자리 화소)
    frac = (hit / (s * s))[m][:, None]
    gray = (acc[m] / np.maximum(hit[m], 1))[:, None]
    img[m] = np.clip(img[m] * (1 - frac) + gray * frac, 0, 255).astype(np.uint8)
    return img


def undistort_norm(cam: Camera, uv: np.ndarray) -> np.ndarray:
    """픽셀 (N,2)의 왜곡을 풀어 정규화 좌표 (N,2: x/z, y/z)로."""
    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    if cam.model == "fisheye":
        return cv2.fisheye.undistortPoints(uv, cam.K, cam.dist[:4]).reshape(-1, 2)
    return cv2.undistortPoints(uv, cam.K, cam.dist).reshape(-1, 2)


def _diag_center(q: np.ndarray) -> np.ndarray:
    """사각형 네 꼭짓점(차례대로)의 두 대각선 교점. 원근 투영에서도 실제 중심의 상이다."""
    p0, p1, p2, p3 = (np.append(p, 1.0) for p in q)
    x = np.cross(np.cross(p0, p2), np.cross(p1, p3))
    return x[:2] / x[2] if abs(x[2]) > 1e-12 else q.mean(0)


def detect_aruco(gray: np.ndarray, cam: Camera, dict_name: str = ARUCO_DICT) -> dict[int, np.ndarray]:
    """ArUco 마커를 찾아 {번호: 중심의 정규화 좌표(2,)}로. 같은 번호가 둘 이상이면 그 번호는 뺀다."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    out: dict[int, np.ndarray] = {}
    if ids is None:
        return out
    seen: dict[int, int] = {}
    for c, i in zip(corners, ids.ravel()):
        seen[int(i)] = seen.get(int(i), 0) + 1
        out[int(i)] = _diag_center(undistort_norm(cam, c.reshape(4, 2)))
    return {k: v for k, v in out.items() if seen[k] == 1}


def detect_circles(gray: np.ndarray, cam: Camera, layout: dict[int, tuple[float, float]],
                   gate_px: float = 40.0) -> dict[int, np.ndarray]:
    """검은 원 마커를 찾아 지금 외부 값으로 예상한 자리에 가장 가까운 것을 그 번호로 본다.

    원의 무게중심은 원근 때문에 실제 중심과 조금 어긋난다(가까운 쪽이 크게 찍힘). 마커를 크게, 가까이 둘수록
    어긋남이 커지므로 ArUco를 쓸 수 있으면 ArUco를 쓴다.
    """
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor, p.blobColor = True, 0
    p.filterByArea, p.minArea, p.maxArea = True, 30.0, float(gray.size) / 4
    p.filterByCircularity = False  # 바닥 원은 납작한 타원으로 찍힌다
    p.filterByConvexity, p.minConvexity = True, 0.8
    p.filterByInertia, p.minInertiaRatio = True, 0.02
    kps = cv2.SimpleBlobDetector_create(p).detect(gray)
    if not kps:
        return {}
    pts = np.array([k.pt for k in kps])
    ids = list(layout)
    pred, ok = cam.project(np.array([[*layout[i], 0.0] for i in ids]))
    out: dict[int, np.ndarray] = {}
    used: set[int] = set()
    # 예상 자리와 가까운 짝부터 하나씩 정한다(한 원을 두 마커가 나눠 갖지 않게)
    pairs = sorted((float(np.linalg.norm(pts[j] - pred[k])), k, j)
                   for k in range(len(ids)) if ok[k] for j in range(len(pts)))
    for dist, k, j in pairs:
        if dist > gate_px or ids[k] in out or j in used:
            continue
        out[ids[k]] = undistort_norm(cam, pts[j])[0]
        used.add(j)
    return out


def _norm_project(R_cv: np.ndarray, pos: np.ndarray, XY: np.ndarray) -> np.ndarray:
    """지면 점 (N,2)를 회전 R_cv, 위치 pos인 카메라의 정규화 좌표 (N,2)로(왜곡 없는 이상 투영)."""
    P = np.hstack([XY, np.zeros((len(XY), 1))]) - pos
    c = P @ R_cv.T
    return c[:, :2] / c[:, 2:3]


def _cam_with(cam: Camera, yaw: float, pitch: float, roll: float, pos=None) -> Camera:
    import dataclasses

    return dataclasses.replace(cam, yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=float(roll),
                               position=np.array(cam.position if pos is None else pos, np.float64))


def solve_pose_markers(cam: Camera, norm: np.ndarray, XY: np.ndarray, solve_position: bool = False,
                       iters: int = 30) -> dict:
    """마커 중심(정규화 좌표 norm (N,2))과 실측 지면 위치 XY (N,2)로 카메라 자세를 다시 구한다.

    solve_position=False(기본): 위치는 지금 값 그대로 두고 yaw, pitch, roll 세 각도만 맞춘다.
      진동이나 탈착으로 바뀌는 것은 주로 각도이고, 점 4개로 위치까지 풀면 높이와 pitch가 서로 바뀌며 흔들린다.
      지금 각도에서 출발해 가우스-뉴턴(수치 미분)으로 푼다. scipy가 없는 파이에서도 돈다.
    solve_position=True: 지금 외부 값을 시작값으로 solvePnP(반복법)를 풀어 위치까지 구한다(6개 이상 권장).
      시작값 없이 풀면 거의 수평인 전방 카메라에서 거울상 해로 빠질 수 있어 늘 지금 값에서 출발한다.
    """
    norm = np.asarray(norm, np.float64).reshape(-1, 2)
    XY = np.asarray(XY, np.float64).reshape(-1, 2)
    if len(norm) < 4:
        raise ValueError("마커가 4개 이상 보여야 합니다")
    if solve_position:
        rvec, tvec = cam.rvec_tvec
        XYZ = np.hstack([XY, np.zeros((len(XY), 1))])
        ok, rvec, tvec = cv2.solvePnP(XYZ, norm.reshape(-1, 1, 2), np.eye(3), None, rvec.copy(), tvec.copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP 실패")
        pos, yaw, pitch, roll = pose_from_pnp(rvec, tvec)
        return {"position": pos, "yaw": yaw, "pitch": pitch, "roll": roll}
    # ---- 각도만: 가우스-뉴턴 ----
    th = np.array([cam.yaw_deg, cam.pitch_deg, cam.roll_deg], np.float64)

    def resid(t):
        return (_norm_project(_cam_with(cam, *t).R_cv, cam.position, XY) - norm).ravel()

    eps = 1e-4
    for _ in range(iters):
        r = resid(th)
        J = np.stack([(resid(th + eps * e) - r) / eps for e in np.eye(3)], axis=1)
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        th += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return {"position": cam.position.copy(), "yaw": float(th[0]), "pitch": float(th[1]), "roll": float(th[2])}


def marker_errors(cam: Camera, norm: np.ndarray, XY: np.ndarray) -> dict:
    """cam 값으로 마커 중심의 재투영 오차(px, 정규화 오차 x fx로 근사)와 지면 역투영 오차(m)를 잰다."""
    norm = np.asarray(norm, np.float64).reshape(-1, 2)
    XY = np.asarray(XY, np.float64).reshape(-1, 2)
    pred = _norm_project(cam.R_cv, cam.position, XY)
    px = np.linalg.norm(pred - norm, axis=1) * float(cam.K[0, 0])
    ray = np.hstack([norm, np.ones((len(norm), 1))]) @ cam.R_vc.T
    valid = ray[:, 2] < -1e-9
    t = np.where(valid, -cam.position[2] / np.where(valid, ray[:, 2], -1.0), np.nan)
    g = cam.position[:2] + t[:, None] * ray[:, :2]
    gerr = np.linalg.norm(g - XY, axis=1)
    return {
        "reproj_px_mean": float(np.mean(px)), "reproj_px_max": float(np.max(px)),
        "ground_m_mean": float(np.nanmean(gerr)) if valid.any() else float("nan"),
        "ground_m_max": float(np.nanmax(gerr)) if valid.any() else float("nan"),
        "invalid": int(np.count_nonzero(~valid)), "n": int(len(norm)),
    }


def pose_delta(cam: Camera, pose: dict) -> dict:
    """새 자세 - 지금 값. 각도는 도(yaw는 -180~180으로 감음), 위치는 m."""
    dyaw = (pose["yaw"] - cam.yaw_deg + 180.0) % 360.0 - 180.0
    dpos = np.asarray(pose["position"], np.float64) - cam.position
    return {"yaw": float(dyaw), "pitch": float(pose["pitch"] - cam.pitch_deg),
            "roll": float(pose["roll"] - cam.roll_deg), "position": dpos, "position_m": float(np.linalg.norm(dpos))}


# ---------------------------------------------------------------------------
# 설정 파일의 외부 값 고치기(주석을 지우지 않게 해당 줄의 숫자만 바꾼다)
# ---------------------------------------------------------------------------
def update_extrinsics_text(text: str, cam_name: str, pose: dict, write_position: bool, scale: float = 1.0) -> str:
    """설정 글(text)에서 카메라 cam_name의 extrinsics 줄(yaw, pitch, roll, 필요하면 position)만 바꾼 글을 돌려준다.

    yaml을 읽어 다시 쓰면 설명 주석이 모두 사라지므로, "- name: <카메라>" 블록 안의 extrinsics 아래
    해당 키 줄에서 값 부분만 갈아 끼우고 줄 끝 주석은 남긴다.
    위치는 position 줄이 있으면 그 값을, position_car 줄만 있으면 실차 m(위치 / scale)로 바꾼다.
    """
    lines = text.splitlines(keepends=True)
    name_re = re.compile(r"^(\s*)-\s*name:\s*['\"]?([^'\"#\s]+)")
    start = end = None
    indent = None
    for i, ln in enumerate(lines):
        m = name_re.match(ln)
        if m and start is None and m.group(2) == cam_name:
            start, indent = i, len(m.group(1))
        elif start is not None and (m and len(m.group(1)) <= indent or
                                    (ln.strip() and not ln.startswith(" ") and not ln.lstrip().startswith("#"))):
            end = i
            break
    if start is None:
        raise ValueError(f"설정에 카메라 {cam_name} 블록이 없습니다")
    end = len(lines) if end is None else end
    ext = next((i for i in range(start, end) if re.match(r"^\s*extrinsics:\s*(#.*)?$", lines[i])), None)
    if ext is None:
        raise ValueError(f"카메라 {cam_name}에 extrinsics 항목이 없습니다")
    ext_ind = len(lines[ext]) - len(lines[ext].lstrip())
    # extrinsics 블록의 끝: 들여쓰기가 extrinsics 이하인 다음 줄(빈 줄, 주석 줄은 건너뜀)
    ext_end = end
    for i in range(ext + 1, end):
        s = lines[i]
        if s.strip() and not s.lstrip().startswith("#") and len(s) - len(s.lstrip()) <= ext_ind:
            ext_end = i
            break
    vals = {"yaw": pose["yaw"], "pitch": pose["pitch"], "roll": pose["roll"]}
    done = set()
    for i in range(ext + 1, ext_end):
        m = re.match(r"^(\s*)(yaw|pitch|roll|position|position_car):(\s*)([^#\n]*?)(\s*)(#.*)?(\r?\n)?$", lines[i])
        if not m:
            continue
        key = m.group(2)
        if key in vals:
            new = f"{vals[key]:.2f}"
        elif write_position and key in ("position", "position_car"):
            p = np.asarray(pose["position"], np.float64) / (scale if key == "position_car" else 1.0)
            new = "[" + ", ".join(f"{v:.4f}" for v in p) + "]"
        else:
            continue
        # 줄 끝 주석의 앞 칸 맞춤이 흐트러지지 않게 값 길이가 바뀐 만큼 공백을 늘리거나 줄인다
        pad = m.group(5) or ""
        if m.group(6):
            pad = " " * max(1, len(m.group(4)) + len(pad) - len(new))
        lines[i] = f"{m.group(1)}{key}:{m.group(3) or ' '}{new}{pad}{m.group(6) or ''}{m.group(7) or ''}"
        done.add("position" if key.startswith("position") else key)
    need = {"yaw", "pitch", "roll"} | ({"position"} if write_position else set())
    if need - done:
        raise ValueError(f"카메라 {cam_name}의 extrinsics에 없는 줄: {sorted(need - done)}")
    return "".join(lines)
