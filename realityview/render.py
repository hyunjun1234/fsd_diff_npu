from __future__ import annotations

import cv2
import numpy as np

from .bev import BevGrid
from .tracker import Track
from .types import Detection

# 리얼리티 뷰 스타일 탑뷰 렌더링(OpenCV, 라즈베리파이에서도 가볍게 돈다).
#
# 속도 메모(2026-11-24, 파이5에서 잼, out/measure/03_render_mask.md):
#   예전에는 마스크 덧칠을 numpy 불리언 색인(out[m] = 0.6*out[m] + ...)으로 해서
#   카메라 2대에 40ms가 들었다. 선택된 화소를 모으고(gather) 다시 흩뿌리는(scatter) 데 대부분이 나갔다.
#   지금은 "영상 전체를 OpenCV로 섞은 뒤(addWeighted) 마스크 자리만 복사(copyTo)"한다.
#   섞는 식은 같고, OpenCV는 반올림, 예전 식은 버림이라 마스크 자리 화소가 최대 1 다를 수 있다.


# ---------------------------------------------------------------------------
# 색과 크기
# ---------------------------------------------------------------------------
BG = (28, 24, 22)
ROAD = (92, 84, 78)  # 탑뷰 도로 덧칠 색
CAM_ROAD = (200, 120, 60)  # 카메라 영상 도로 덧칠 색
EGO = (235, 235, 235)
CLASS_COLOR = {
    "car": (230, 200, 120),
    "truck": (200, 170, 100),
    "bus": (200, 170, 100),
    "person": (90, 170, 250),
    "bicycle": (120, 220, 120),
    "motorcycle": (120, 220, 120),
}
EGO_SIZE = (4.6, 1.9)  # 길이, 폭 (m)
EGO_REAR_OFFSET = 1.0  # 원점(뒷축)에서 차량 뒤끝까지 (m)

# 바닥 영상 색조: 회색값 g를 (B, G, R) = g * 0.45 * (1.0, 0.95, 0.9)로 어둡고 차갑게 바꾼다
GROUND_GAIN = 0.45
GROUND_TINT = (1.0, 0.95, 0.9)


# ---------------------------------------------------------------------------
# 덧칠 도우미(마스크 자리만 색을 섞기)
# ---------------------------------------------------------------------------
_SOLID_CACHE: dict[tuple, np.ndarray] = {}


def _solid(shape: tuple, color: tuple) -> np.ndarray:
    """한 가지 색으로 채운 영상. 크기와 색이 같으면 처음 만든 것을 다시 쓴다(읽기 전용).

    addWeighted는 두 영상을 섞으므로 "색 한 가지"도 영상으로 만들어 줘야 한다.
    매 프레임 만들면 그만큼 메모리 쓰기가 늘어서 모양별로 한 번만 만든다.
    렌더링 스레드와 메인 스레드가 같이 불러도 사전에 같은 값을 넣을 뿐이라 안전하다.
    """
    key = (tuple(shape), tuple(int(c) for c in color))
    img = _SOLID_CACHE.get(key)
    if img is None:
        img = np.empty(shape, np.uint8)
        img[:] = key[1]
        img.flags.writeable = False  # 실수로 고치면 다음 프레임 색이 틀어지므로 막아 둔다
        _SOLID_CACHE[key] = img
    return img


def as_mask_u8(mask: np.ndarray) -> np.ndarray:
    """마스크를 OpenCV가 받는 uint8로. 0이 아닌 화소가 "선택"이다.

    - uint8: 그대로(복사 없음)
    - bool: 같은 메모리를 uint8로 보기만 한다(복사 없음, 값은 0/1)
    - 그 밖의 정수, 실수: 0이 아니면 1로 바꾼 새 배열
    """
    if mask.dtype == np.uint8:
        return np.ascontiguousarray(mask)
    if mask.dtype == np.bool_:
        return np.ascontiguousarray(mask).view(np.uint8)
    return (mask != 0).astype(np.uint8)


def blend_where(dst: np.ndarray, mask_u8: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    """dst에서 mask가 0이 아닌 화소만 (1-alpha)*dst + alpha*color로 바꾼다(dst를 그 자리에서 고침).

    1) 영상 전체를 color와 섞는다(cv2.addWeighted, SIMD로 한 번에 처리해서 빠르다)
    2) 섞인 영상에서 mask 자리만 dst로 복사한다(cv2.copyTo)
    마스크 밖 화소는 손대지 않으므로 그대로다.
    """
    mixed = cv2.addWeighted(dst, 1.0 - alpha, _solid(dst.shape, color), alpha, 0.0)
    cv2.copyTo(mixed, mask_u8, dst)
    return dst


# ---------------------------------------------------------------------------
# 바닥 영상 색조 표(회색값 0~255 → BGR)
# ---------------------------------------------------------------------------
def _ground_lut() -> np.ndarray:
    """회색값마다 색조를 입힌 결과를 미리 계산한 표. 모양 (1, 256, 3), cv2.LUT용.

    예전 식 (float32 회색 * 0.45 * tint).clip(0, 255).astype(uint8)을 회색값 256개에 그대로 적용했으므로
    화소마다 계산하던 예전 결과와 똑같다(버림까지 같음).
    """
    g = np.arange(256, dtype=np.float32)[:, None]
    tint = np.array(GROUND_TINT, np.float32)
    return (g * GROUND_GAIN * tint).clip(0, 255).astype(np.uint8).reshape(1, 256, 3)


_GROUND_LUT = _ground_lut()


# ---------------------------------------------------------------------------
# 탑뷰
# ---------------------------------------------------------------------------
def render_bev(
    grid: BevGrid,
    tracks: list[Track],
    ground_img: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
    show_ids: bool = True,
) -> np.ndarray:
    """탑뷰 한 장을 새로 그린다.

    ground_img: IPM으로 합성한 바닥 영상(BGR). 어둡고 차갑게 바꿔 깐다
    road_mask: 탑뷰 크기의 도로 마스크(0이 아니면 도로). 도로색을 반쯤 섞는다
    coverage: 카메라가 보는 셀(bool 또는 uint8). 이 밖은 배경색으로 둔다
    """
    H, W = grid.height, grid.width
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG
    cov = as_mask_u8(coverage) if coverage is not None else None

    # ---- 1) 바닥 영상: 회색으로 바꾼 뒤 색조 표로 한 번에 변환, 카메라가 보는 셀만 복사 ----
    if ground_img is not None:
        g = cv2.cvtColor(ground_img, cv2.COLOR_BGR2GRAY)
        layer = cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), _GROUND_LUT)
        if cov is None:
            canvas = layer
        else:
            cv2.copyTo(layer, cov, canvas)

    # ---- 2) 도로 마스크: 도로이면서 카메라가 보는 셀에만 도로색을 반쯤 섞는다 ----
    if road_mask is not None:
        m = as_mask_u8(road_mask)
        if cov is not None:
            # 값이 1이 아닌 마스크(예: 255)도 있으니 먼저 0/255로 맞춘 뒤 coverage와 겹친다
            m = cv2.bitwise_and(cv2.compare(m, 0, cv2.CMP_GT), cov)
        blend_where(canvas, m, ROAD, 0.5)

    # ---- 3) 거리 고리, 자차 ----
    _draw_rings(canvas, grid)
    _draw_box(canvas, grid, np.array([EGO_SIZE[0] / 2 - EGO_REAR_OFFSET, 0.0]), EGO_SIZE, 0.0, EGO, fill=True)

    # ---- 4) 트랙: 먼 것부터 그려서 가까운 것이 위에 오게 ----
    for t in sorted(tracks, key=lambda t: -np.linalg.norm(t.xy)):
        color = CLASS_COLOR.get(t.label, (200, 200, 200))
        # 자차 기준 상대좌표라 실제 진행 방향은 자차 오도메트리 없이는 알 수 없다.
        # 차량은 자차와 평행하다고 가정하고, 상대속도로 방향을 추정할 수 있을 때만 쓴다.
        yaw = 0.0
        if t.label in ("person",):
            c = grid.to_pixel(t.xy[None])[0]
            r = max(3, int(0.35 / grid.res))
            cv2.circle(canvas, (int(c[0]), int(c[1])), r, color, -1, cv2.LINE_AA)
        else:
            _draw_box(canvas, grid, t.xy, t.size, yaw, color, fill=True)
        if show_ids:
            c = grid.to_pixel(t.xy[None])[0]
            cv2.putText(
                canvas, str(t.tid), (int(c[0]) + 8, int(c[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1,
                cv2.LINE_AA,
            )
    return canvas


def _draw_rings(canvas: np.ndarray, grid: BevGrid) -> None:
    c = grid.to_pixel(np.array([[0.0, 0.0]]))[0].astype(int)
    for r_m in (5, 10, 15, 20):
        cv2.circle(canvas, tuple(c), int(r_m / grid.res), (60, 56, 52), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, f"{r_m}m", (c[0] + 4, c[1] - int(r_m / grid.res) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (110, 105, 100), 1, cv2.LINE_AA,
        )


def _draw_box(canvas, grid, xy, size, yaw, color, fill=True) -> None:
    L, Wd = size
    corners = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = (corners @ R.T) + np.asarray(xy)
    px = grid.to_pixel(pts).round().astype(np.int32)
    if fill:
        cv2.fillPoly(canvas, [px], color, cv2.LINE_AA)
        cv2.polylines(canvas, [px], True, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [px], True, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 카메라 영상 주석(도로 덧칠, 검출 상자, 이름)
# ---------------------------------------------------------------------------
def annotate_camera(img: np.ndarray, dets: list[Detection], name: str, road_mask: np.ndarray | None = None) -> np.ndarray:
    """카메라 영상 복사본에 도로 마스크(색 40%), 검출 상자, 카메라 이름을 그린다. img는 고치지 않는다."""
    out = img.copy()
    if road_mask is not None:
        blend_where(out, as_mask_u8(road_mask), CAM_ROAD, 0.4)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        col = CLASS_COLOR.get(d.label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(out, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# 대시보드 합치기
# ---------------------------------------------------------------------------
def compose_dashboard(bev: np.ndarray, cam_views: list[np.ndarray], fps: float | None = None) -> np.ndarray:
    """왼쪽에 탑뷰, 오른쪽에 카메라 영상들을 세로로 쌓는다."""
    H = bev.shape[0]
    if cam_views:
        col_w = int(H / len(cam_views) * cam_views[0].shape[1] / cam_views[0].shape[0])
        col_w = min(col_w, int(bev.shape[1] * 1.6))
        tiles = []
        for v in cam_views:
            h = int(v.shape[0] * col_w / v.shape[1])
            tiles.append(cv2.resize(v, (col_w, h)))
        right = np.vstack(tiles)
        if right.shape[0] < H:
            right = np.vstack([right, np.zeros((H - right.shape[0], col_w, 3), np.uint8)])
        right = right[:H]
        out = np.hstack([bev, right])
    else:
        out = bev.copy()
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} FPS", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
