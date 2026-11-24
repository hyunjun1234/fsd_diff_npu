"""리얼리티 뷰 스타일 탑뷰 렌더링 (OpenCV, 라즈베리파이에서도 가볍게 돈다)."""
from __future__ import annotations

import cv2
import numpy as np

from .bev import BevGrid
from .tracker import Track
from .types import Detection

BG = (28, 24, 22)
ROAD = (92, 84, 78)
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


def render_bev(
    grid: BevGrid,
    tracks: list[Track],
    ground_img: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
    show_ids: bool = True,
) -> np.ndarray:
    H, W = grid.height, grid.width
    canvas = np.full((H, W, 3), BG, np.uint8)

    if ground_img is not None:
        # 실제 바닥 영상을 어둡고 차갑게 깔아 형태만 보이게 한다
        g = cv2.cvtColor(ground_img, cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
        tint = np.array([1.0, 0.95, 0.9], np.float32)
        layer = (g * 0.45 * tint).clip(0, 255).astype(np.uint8)
        sel = coverage if coverage is not None else np.ones((H, W), bool)
        canvas[sel] = layer[sel]
    if road_mask is not None:
        m = road_mask.astype(bool)
        if coverage is not None:
            m &= coverage
        canvas[m] = (0.5 * canvas[m] + 0.5 * np.array(ROAD)).astype(np.uint8)

    _draw_rings(canvas, grid)
    _draw_box(canvas, grid, np.array([EGO_SIZE[0] / 2 - EGO_REAR_OFFSET, 0.0]), EGO_SIZE, 0.0, EGO, fill=True)

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


def annotate_camera(img: np.ndarray, dets: list[Detection], name: str, road_mask: np.ndarray | None = None) -> np.ndarray:
    out = img.copy()
    if road_mask is not None:
        m = road_mask.astype(bool)
        out[m] = (0.6 * out[m] + 0.4 * np.array([200, 120, 60])).astype(np.uint8)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        col = CLASS_COLOR.get(d.label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(out, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


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
