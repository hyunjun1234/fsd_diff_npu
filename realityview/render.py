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
# 바닥 층(탑뷰와 3D 시점이 같이 쓴다)
# ---------------------------------------------------------------------------
def render_ground_layer(
    grid: BevGrid,
    ground_img: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
) -> np.ndarray:
    """탑뷰 크기의 바닥 그림만 새로 만든다(물체, 자차, 고리는 없음).

    1) 바닥 영상: 회색으로 바꾼 뒤 색조 표로 한 번에 변환, 카메라가 보는 셀만 복사
    2) 도로 마스크: 도로이면서 카메라가 보는 셀에만 도로색을 반쯤 섞는다
    바닥 영상이 없으면 배경색 한 장이다. 돌려주는 배열은 새로 만든 것이라 받는 쪽이 고쳐도 된다.
    """
    H, W = grid.height, grid.width
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG
    cov = as_mask_u8(coverage) if coverage is not None else None

    if ground_img is not None:
        g = cv2.cvtColor(ground_img, cv2.COLOR_BGR2GRAY)
        layer = cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), _GROUND_LUT)
        if cov is None:
            canvas = layer
        else:
            cv2.copyTo(layer, cov, canvas)

    if road_mask is not None:
        m = as_mask_u8(road_mask)
        if cov is not None:
            # 값이 1이 아닌 마스크(예: 255)도 있으니 먼저 0/255로 맞춘 뒤 coverage와 겹친다
            m = cv2.bitwise_and(cv2.compare(m, 0, cv2.CMP_GT), cov)
        blend_where(canvas, m, ROAD, 0.5)
    return canvas


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
    scale: float = 1.0,
    lanes=None,
    lights=None,
) -> np.ndarray:
    """탑뷰 한 장을 새로 그린다.

    lanes: 차선 곡선 목록(lanes.LaneCurve). 주면 내 차선을 칠하고 곡선을 그린다(기본 없음)
    lights: 신호등 목록(traffic_lights.TrafficLight). 주면 왼쪽 위에 대표 신호등 아이콘을 그린다(기본 없음)

    scale: 장면 축척. 모형이면 자차 크기, 보행자 점 크기, 거리 고리 간격을 같은 비율로 줄인다.

    ground_img: IPM으로 합성한 바닥 영상(BGR). 어둡고 차갑게 바꿔 깐다
    road_mask: 탑뷰 크기의 도로 마스크(0이 아니면 도로). 도로색을 반쯤 섞는다
    coverage: 카메라가 보는 셀(bool 또는 uint8). 이 밖은 배경색으로 둔다
    """
    # ---- 1), 2) 바닥 영상과 도로 마스크(3D 시점도 같은 함수로 바닥을 만든다) ----
    canvas = render_ground_layer(grid, ground_img, road_mask, coverage)
    if lanes:
        draw_lanes_bev(canvas, grid, lanes, scale)

    # ---- 3) 거리 고리, 자차 ----
    _draw_rings(canvas, grid, scale)
    ego = (EGO_SIZE[0] * scale, EGO_SIZE[1] * scale)
    _draw_box(canvas, grid, np.array([ego[0] / 2 - EGO_REAR_OFFSET * scale, 0.0]), ego, 0.0, EGO, fill=True)

    # ---- 4) 트랙: 먼 것부터 그려서 가까운 것이 위에 오게 ----
    for t in sorted(tracks, key=lambda t: -np.linalg.norm(t.xy)):
        color = CLASS_COLOR.get(t.label, (200, 200, 200))
        # 자차 기준 상대좌표라 실제 진행 방향은 자차 오도메트리 없이는 알 수 없다.
        # 차량은 자차와 평행하다고 가정하고, 상대속도로 방향을 추정할 수 있을 때만 쓴다.
        yaw = 0.0
        if t.label in ("person",):
            c = grid.to_pixel(t.xy[None])[0]
            r = max(3, int(0.35 * scale / grid.res))
            cv2.circle(canvas, (int(c[0]), int(c[1])), r, color, -1, cv2.LINE_AA)
        else:
            _draw_box(canvas, grid, t.xy, t.size, yaw, color, fill=True)
        if show_ids:
            c = grid.to_pixel(t.xy[None])[0]
            cv2.putText(
                canvas, str(t.tid), (int(c[0]) + 8, int(c[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1,
                cv2.LINE_AA,
            )
    if lights:
        from .traffic_lights import main_light

        lt = main_light(lights)
        if lt is not None:
            draw_light_icon(canvas, lt.state, 10, 10, lt.orientation)
    return canvas


# ---------------------------------------------------------------------------
# 차선과 신호등(탑뷰와 3D 시점이 같이 쓴다)
# ---------------------------------------------------------------------------
EGO_LANE = (230, 150, 40)  # 내 차선 덧칠 색(파랑, BGR)
DASH_ON_M = 3.0  # 점선 한 칸 길이(실차 m)
DASH_PERIOD_M = 9.0  # 점선 한 주기(칠한 칸 + 빈칸)


def lane_bgr(color: str) -> tuple:
    from .lanes import LANE_BGR

    return LANE_BGR.get(color, LANE_BGR["white"])


def dash_runs(pts: np.ndarray, dashed: bool, scale: float = 1.0) -> list[np.ndarray]:
    """곡선 위 점(N,2: X, Y)을 그릴 조각으로 나눈다. 실선은 통째로, 점선은 X 기준 DASH_ON_M씩 끊는다.

    점선 무늬의 위치는 실제 도로 점선과 맞추지 않은 표시용이다(실선과 구별되기만 하면 된다).
    """
    if len(pts) < 2:
        return []
    if not dashed:
        return [pts]
    on = np.mod(pts[:, 0], DASH_PERIOD_M * scale) < DASH_ON_M * scale
    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if (not v or i == len(on) - 1) and start is not None:
            end = i + 1 if v else i
            if end - start >= 2:
                runs.append(pts[start:end])
            start = None
    return runs


def ego_lane_polygon(lanes, step: float, x_min: float | None = None, x_max: float | None = None):
    """내 차선(왼쪽, 오른쪽 경계 사이) 다각형의 차량 좌표 점(N,2). 경계가 둘 다 없으면 None."""
    left = next((l for l in lanes if l.side == "left"), None)
    right = next((l for l in lanes if l.side == "right"), None)
    if left is None or right is None:
        return None
    x0 = max(left.x_range[0], right.x_range[0], -np.inf if x_min is None else x_min)
    x1 = min(left.x_range[1], right.x_range[1], np.inf if x_max is None else x_max)
    if x1 - x0 < step:
        return None
    xs = np.linspace(x0, x1, max(2, int((x1 - x0) / step) + 1))
    return np.vstack([np.c_[xs, left.y_at(xs)], np.c_[xs[::-1], right.y_at(xs[::-1])]])


def draw_lanes_bev(canvas: np.ndarray, grid: BevGrid, lanes, scale: float = 1.0) -> None:
    """탑뷰에 내 차선 덧칠(반투명)과 차선 곡선(실선, 점선, 색)을 그린다(canvas를 그 자리에서 고침)."""
    step = 0.5 * scale
    poly = ego_lane_polygon(lanes, step)
    if poly is not None:
        px = grid.to_pixel(poly).round().astype(np.int32)
        x0, y0 = np.maximum(px.min(axis=0), 0)
        x1, y1 = np.minimum(px.max(axis=0) + 1, [canvas.shape[1], canvas.shape[0]])
        if x1 > x0 and y1 > y0:
            roi = canvas[y0:y1, x0:x1]
            layer = roi.copy()
            cv2.fillPoly(layer, [px - [x0, y0]], EGO_LANE, cv2.LINE_AA)
            cv2.addWeighted(layer, 0.35, roi, 0.65, 0.0, dst=roi)
    thick = max(2, int(round(0.15 * scale / grid.res)))
    for ln in lanes:
        col = lane_bgr(ln.color)
        for run in dash_runs(ln.points(step / 2), ln.style == "dashed", scale):
            px = grid.to_pixel(run).round().astype(np.int32)
            cv2.polylines(canvas, [px], False, col, thick, cv2.LINE_AA)


def draw_light_icon(canvas: np.ndarray, state: str, x: int, y: int, orientation: str = "vertical",
                    lamp: int = 9) -> None:
    """(x, y)를 왼쪽 위로 하는 신호등 아이콘. 켜진 색 하나만 밝게, 나머지는 어둡게 그린다.

    가로형 신호등이면 아이콘도 가로로 눕힌다. 모름이면 세 등 모두 어둡고 가운데에 '?'를 쓴다.
    """
    from .traffic_lights import STATE_BGR

    horiz = orientation == "horizontal"
    gap = lamp * 2 + 4
    w, h = (gap * 3 + 4, gap + 4) if horiz else (gap + 4, gap * 3 + 4)
    H, W = canvas.shape[:2]
    if x + w > W or y + h > H or x < 0 or y < 0:
        return
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)
    for k, s in enumerate(("red", "yellow", "green")):
        cx = x + 2 + gap // 2 + (gap * k if horiz else 0)
        cy = y + 2 + gap // 2 + (0 if horiz else gap * k)
        col = STATE_BGR[s] if s == state else (55, 55, 55)
        cv2.circle(canvas, (cx, cy), lamp, col, -1, cv2.LINE_AA)
    if state == "unknown":
        cv2.putText(canvas, "?", (x + w + 4, y + h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
                    cv2.LINE_AA)


def _draw_rings(canvas: np.ndarray, grid: BevGrid, scale: float = 1.0) -> None:
    """자차 기준 거리 고리. 실차는 5, 10, 15, 20m, 모형은 같은 고리를 축척만큼 줄여 그린다."""
    c = grid.to_pixel(np.array([[0.0, 0.0]]))[0].astype(int)
    for base in (5, 10, 15, 20):
        r_m = base * scale
        cv2.circle(canvas, tuple(c), int(r_m / grid.res), (60, 56, 52), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, f"{r_m:g}m", (c[0] + 4, c[1] - int(r_m / grid.res) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
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


def _even(v: float) -> int:
    """짝수 정수로 내림(최소 2). H.264(avc1) 녹화는 폭과 높이가 짝수여야 열린다."""
    return max(2, int(v) // 2 * 2)


def _fit_width(img: np.ndarray, w: int) -> np.ndarray:
    """비율을 지키며 폭을 w로 맞춘다."""
    h = max(1, int(round(img.shape[0] * w / img.shape[1])))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def compose_dashboard_3d(
    main: np.ndarray,
    cam_views: list[np.ndarray],
    bev: np.ndarray | None = None,
    fps: float | None = None,
    side_ratio: float = 1 / 3,
) -> np.ndarray:
    """3D 시점을 크게, 카메라 영상을 작게 두는 배치.

    [탑뷰(있을 때, 높이를 3D에 맞춤)] [3D 시점(큰 화면)] [카메라 영상 세로 줄(폭 = 3D 폭 x side_ratio)]
    - 카메라 줄이 3D 높이보다 길면 모든 칸을 같은 비율로 줄여 넣고, 남는 자리는 배경색으로 채운다.
    - 결과 크기는 입력 크기로만 정해지므로 프레임마다 같다(녹화 파일 크기가 바뀌지 않는다).
    - main 폭과 높이가 짝수이면 결과도 짝수 크기다.
    """
    H, W = main.shape[:2]
    parts = []
    if bev is not None:
        bw = _even(bev.shape[1] * H / bev.shape[0])
        parts.append(cv2.resize(bev, (bw, H), interpolation=cv2.INTER_AREA))
    parts.append(main)
    if cam_views:
        col_w = _even(W * side_ratio)
        col = np.empty((H, col_w, 3), np.uint8)
        col[:] = BG
        tiles = [_fit_width(v, col_w) for v in cam_views]
        total = sum(t.shape[0] for t in tiles)
        if total > H:
            # 세로로 다 들어가지 않으면 모든 칸을 같은 비율로 줄이고 가운데 정렬한다
            k = H / total
            tiles = [cv2.resize(t, (max(1, int(t.shape[1] * k)), max(1, int(t.shape[0] * k))), interpolation=cv2.INTER_AREA)
                     for t in tiles]
        y = 0
        for t in tiles:
            h, w = t.shape[:2]
            h = min(h, H - y)
            if h <= 0:
                break
            x = (col_w - w) // 2
            col[y:y + h, x:x + w] = t[:h]
            y += h
        parts.append(col)
    out = np.hstack(parts) if len(parts) > 1 else main.copy()
    if fps is not None:
        x0 = parts[0].shape[1] if bev is not None else 0
        cv2.putText(out, f"{fps:.1f} FPS", (x0 + 10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
