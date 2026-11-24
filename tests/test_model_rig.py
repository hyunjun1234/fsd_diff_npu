import os

import cv2
import numpy as np
import pytest

from realityview.bev import IpmCompositor, merge_cross_camera, project_detections
from realityview.calib import (
    apply_pose,
    board_points,
    calibrate_intrinsics,
    find_corners,
    floor_board_xy,
    ground_errors,
    solve_extrinsics,
)
from realityview.camera_model import Camera, fisheye_f_from_fov
from realityview.config import load_config
from realityview.detcompare import center_distance, iou_xyxy, match_detections, summarize_pairs
from realityview.pipeline import RealityViewPipeline
from realityview.sim import SimBackend, SimWorld
from realityview.tracker import BevTracker
from realityview.types import Detection

# 모형 시험(책상 위 장난감 차) 준비물 시험: 어안 카메라, 보정 계산, 장면 축척, 두 검출기 비교.
# 모두 하드웨어 없이 합성 값으로 돈다.

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_CFG = os.path.join(ROOT, "configs", "two_cam_model.yaml")


def _fisheye(dist=(0.0, 0.0, 0.0, 0.0), max_fov=None, pitch=25.0):
    f = fisheye_f_from_fov(768, 576, 130.0, diagonal=True)
    K = np.array([[f, 0, 384.0], [0, f, 288.0], [0, 0, 1]])
    return Camera("front", 768, 576, K, [0.15, 0.0, 0.20], 0.0, pitch, 0.0, dist=list(dist),
                  model="fisheye", max_fov_deg=max_fov)


# ---------------------------------------------------------------------------
# 모형 설정 파일
# ---------------------------------------------------------------------------
def test_model_config_loads_small_grid_and_fisheye_front():
    cams, grid, cfg = load_config(MODEL_CFG)
    assert [c.name for c in cams] == ["front", "rear"]
    assert cams[0].model == "fisheye" and cams[1].model == "pinhole"
    assert cams[0].max_fov_deg == 140.0
    assert 0.15 <= cams[0].position[2] <= 0.30  # 모형 카메라 높이(실측 필요)
    assert grid.res <= 0.02 and grid.x_max - grid.x_min <= 4.0 + 1e-9
    assert (grid.height, grid.width) == (400, 240)
    assert cfg["scene"]["scale"] == pytest.approx(0.15)
    # 근사 초점거리: 대각 130도 등거리 어안
    assert cams[0].K[0, 0] == pytest.approx(fisheye_f_from_fov(768, 576, 130.0, diagonal=True), abs=0.1)


def test_fisheye_f_from_fov_horizontal_and_diagonal():
    assert fisheye_f_from_fov(768, 576, 130.0) == pytest.approx(384 / np.radians(65), rel=1e-9)
    assert fisheye_f_from_fov(768, 576, 130.0, diagonal=True) == pytest.approx(480 / np.radians(65), rel=1e-9)


# ---------------------------------------------------------------------------
# 어안 가장자리: 화각 밖 점이 영상 안으로 접혀 들어오지 않는지
# ---------------------------------------------------------------------------
def test_fisheye_theta_max_limits():
    assert np.degrees(_fisheye().fisheye_theta_max()) == pytest.approx(89.9, abs=0.1)
    assert np.degrees(_fisheye(max_fov=140).fisheye_theta_max()) == pytest.approx(70.0, abs=1e-6)
    # k1이 크게 음수면 투영식이 90도 전에 줄어든다: 그 지점에서 자른다
    cam = _fisheye(dist=(-0.35, 0.0, 0.0, 0.0))
    lim = cam.fisheye_theta_max()
    th = np.linspace(0, lim, 50)
    td = th * (1 - 0.35 * th**2)
    assert np.degrees(lim) < 89 and np.all(np.diff(td) > 0)


def test_fisheye_project_rejects_points_beyond_fov():
    cam = _fisheye(max_fov=140, pitch=0.0)
    # 광축에서 80도 옆 점(화각 70도 밖)은 앞쪽이어도 무효, 30도 점은 유효
    a80, a30 = np.radians(80), np.radians(30)
    pts = np.array([[0.15 + np.cos(a80), np.sin(a80), 0.20], [0.15 + np.cos(a30), np.sin(a30), 0.20]])
    _, ok = cam.project(pts)
    assert ok.tolist() == [False, True]


def test_ipm_with_folding_fisheye_uses_only_cells_inside_fov():
    # 접힘이 생기는 왜곡 계수: 화각 밖 바닥 셀이 영상 안 좌표로 투영될 수 있다
    cam = _fisheye(dist=(-0.35, 0.0, 0.0, 0.0), pitch=25.0)
    _, grid, _ = load_config(MODEL_CFG)
    ipm = IpmCompositor([cam], grid)
    X, Y = grid.cell_centers()
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    p_cam = (cam.R_cv @ (pts - cam.position).T)
    theta = np.arctan2(np.hypot(p_cam[0], p_cam[1]), p_cam[2])
    used = ipm.owner.ravel() == 0
    assert used.any()
    assert theta[used].max() <= cam.fisheye_theta_max() + 1e-9


def test_fisheye_ground_round_trip_and_bad_pixels():
    cams, _, _ = load_config(MODEL_CFG)
    cam = cams[0]
    XY = np.array([[0.5, 0.0], [0.8, 0.3], [0.4, -0.25], [1.5, 0.6]])
    uv, ok = cam.project(np.hstack([XY, np.zeros((4, 1))]))
    assert ok.all()
    g, valid = cam.pixel_to_ground(uv)
    assert valid.all()
    np.testing.assert_allclose(g, XY, atol=1e-6)
    # 왜곡을 풀 수 없는 아주 먼 픽셀은 nan이 아니라 무효로 나와야 한다
    g2, v2 = cam.pixel_to_ground(np.array([[1e9, 1e9]]))
    assert not v2[0] and np.all(np.isfinite(g2))


# ---------------------------------------------------------------------------
# 보정 계산 (합성 코너)
# ---------------------------------------------------------------------------
def _synthetic_views(K, D, n=14, seed=0, noise=0.1):
    """칸 20mm, 내부 9x6 보드를 카메라 앞 20~40cm에서 여러 자세로 찍은 코너."""
    rng = np.random.default_rng(seed)
    objp = board_points(9, 6, 0.020)
    center = objp.mean(axis=0)
    obj, img = [], []
    while len(obj) < n:
        rvec = rng.uniform(-0.6, 0.6, 3)
        R, _ = cv2.Rodrigues(rvec)
        # 화면 가운데와 가장자리를 고루 덮도록 보드 위치를 흔든다
        t = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(0.2, 0.4)]) - R @ center
        uv, _ = cv2.fisheye.projectPoints(objp.reshape(-1, 1, 3), rvec, t, K, D)
        uv = uv.reshape(-1, 2) + rng.normal(0, noise, (len(objp), 2))
        if (uv.min() < 5) or (uv[:, 0].max() > 763) or (uv[:, 1].max() > 571):
            continue
        obj.append(objp)
        img.append(uv)
    return obj, img


def test_fisheye_intrinsics_recovered_from_small_board():
    f = fisheye_f_from_fov(768, 576, 130.0, diagonal=True)
    K = np.array([[f, 0, 380.0], [0, f, 290.0], [0, 0, 1.0]])
    D = np.array([0.03, -0.01, 0.002, 0.0])
    obj, img = _synthetic_views(K, D)
    res = calibrate_intrinsics(obj, img, (768, 576), "fisheye")
    assert res["rms"] < 0.5
    assert res["K"][0, 0] == pytest.approx(f, rel=0.01)
    assert res["K"][0, 2] == pytest.approx(380.0, abs=3.0)
    assert len(res["per_view"]) == len(res["used"])


def test_find_corners_on_rendered_board():
    # 칸 20px 보드를 그려서 코너 찾기(SB 방식)가 내부 코너 수와 순서를 맞게 주는지
    sq, cols, rows = 24, 9, 6
    img = np.full(((rows + 3) * sq, (cols + 3) * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = (r + 1) * sq, (c + 1) * sq
                img[y0 : y0 + sq, x0 : x0 + sq] = 0
    corners = find_corners(img, cols, rows)
    assert corners is not None and corners.shape == (cols * rows, 2)
    d = np.diff(corners[:cols, 0])
    assert np.allclose(np.abs(d), sq, atol=1.0)


def test_extrinsics_from_floor_board_recovers_pose():
    cams, _, _ = load_config(MODEL_CFG)
    truth = cams[0]
    XY = floor_board_xy(9, 6, 0.020, (0.40, 0.08), -90.0)
    assert np.allclose(XY[1] - XY[0], [0.0, -0.020])  # 보드 가로 = 차량 -Y 방향
    uv, ok = truth.project(np.hstack([XY, np.zeros((len(XY), 1))]))
    assert ok.all()
    # 처음 값을 일부러 틀리게 두고 푼다
    cam = Camera(truth.name, truth.width, truth.height, truth.K, [0.0, 0.0, 0.5], 10.0, 5.0, 0.0,
                 dist=truth.dist, model="fisheye", max_fov_deg=truth.max_fov_deg)
    apply_pose(cam, solve_extrinsics(cam, uv, XY))
    np.testing.assert_allclose(cam.position, truth.position, atol=1e-4)
    assert cam.pitch_deg == pytest.approx(truth.pitch_deg, abs=0.05)
    e = ground_errors(cam, uv, XY)
    assert e["ground_m_max"] < 1e-4 and e["invalid"] == 0


# ---------------------------------------------------------------------------
# 장면 축척 (모형)
# ---------------------------------------------------------------------------
def test_scale_shrinks_object_size_merge_and_gate():
    cams, _, _ = load_config(MODEL_CFG)
    cam = cams[0]
    uv, _ = cam.project(np.array([[0.8, 0.0, 0.0]]))
    det = Detection("car", 0.9, (uv[0, 0] - 20, uv[0, 1] - 15, uv[0, 0] + 20, uv[0, 1]))
    real = project_detections(0, cam, [det])
    model = project_detections(0, cam, [det], scale=0.15)
    assert real[0].size[0] == pytest.approx(4.5) and model[0].size[0] == pytest.approx(0.675)
    # 실차 크기로 밀면 중심이 2.25m 뒤로 가서 ±2m 격자 밖이 된다. 축척을 주면 0.34m만 민다
    assert real[0].xy[0] > 2.5 and model[0].xy[0] < 1.3
    # 병합 반경: 실차 1.5m, 모형 0.225m
    a, b = model[0], project_detections(1, cam, [det], scale=0.15)[0]
    b.xy = a.xy + np.array([0.3, 0.0])
    assert len(merge_cross_camera([a, b], scale=0.15)) == 2
    assert len(merge_cross_camera([a, b])) == 1
    assert BevTracker(scale=0.15).gate == pytest.approx(0.375)


def test_pipeline_runs_with_model_config():
    cams, grid, cfg = load_config(MODEL_CFG)
    world = SimWorld(cams)
    pipe = RealityViewPipeline(cams, SimBackend(world), grid, scale=cfg["scene"]["scale"])
    try:
        out = pipe.step(world.render(), 0.1)
        assert out.bev.shape[:2] == (grid.height, grid.width)
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# 두 검출기 비교 (compare_det.py 계산부)
# ---------------------------------------------------------------------------
def test_iou_and_center_distance():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou_xyxy((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert center_distance((0, 0, 10, 10), (3, 4, 13, 14)) == pytest.approx(5.0)


def test_match_detections_same_class_greedy():
    ref = [Detection("chair", 0.9, (0, 0, 10, 10)), Detection("tv", 0.8, (50, 50, 60, 60))]
    oth = [Detection("chair", 0.7, (1, 0, 11, 10)), Detection("chair", 0.6, (50, 50, 60, 60))]
    pairs = match_detections(ref, oth, 0.5)
    assert len(pairs) == 1 and pairs[0][:2] == (0, 0)  # tv는 클래스가 달라 짝이 없다
    s = summarize_pairs([ref], [oth], 0.5)
    assert (s["matched"], s["ref_unmatched"], s["other_unmatched"]) == (1, 1, 1)
    assert s["center_px_mean"] == pytest.approx(1.0)
    empty = summarize_pairs([[]], [[]])
    assert empty["iou_mean"] is None and empty["matched"] == 0
    with pytest.raises(ValueError):
        summarize_pairs([[]], [])


def _render_floor_board(cam, cols, rows, sq, origin, yaw_deg):
    """카메라 영상의 화소마다 바닥 위치를 구해 바닥에 깐 체커보드를 그린다(어안도 그대로 됨)."""
    u, v = np.meshgrid(np.arange(cam.width) + 0.0, np.arange(cam.height) + 0.0)
    g, ok = cam.pixel_to_ground(np.stack([u.ravel(), v.ravel()], 1))
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    b = (g - np.asarray(origin)) @ R
    b[:, 1] *= -1.0  # floor_board_xy와 같은 규칙: 세로는 가로의 오른쪽
    ix, iy = np.floor(b[:, 0] / sq) + 1, np.floor(b[:, 1] / sq) + 1
    inside = ok & (ix >= 0) & (ix <= cols) & (iy >= 0) & (iy <= rows)
    img = np.full(len(g), 230, np.uint8)
    img[inside & ((ix + iy) % 2 == 0)] = 20
    img[~ok] = 120
    return img.reshape(cam.height, cam.width)


@pytest.mark.parametrize("which,origin,yaw", [(0, (0.40, 0.08), -90.0), (0, (0.35, -0.10), 0.0), (1, (-0.55, -0.08), 90.0)])
def test_floor_board_corner_order_matches_detector(which, origin, yaw):
    # 진짜 코너 검출기가 돌려주는 순서와 floor_board_xy의 순서가 같아야 외부 보정이 맞는다
    cams, _, _ = load_config(MODEL_CFG)
    cam = cams[which]
    c = find_corners(_render_floor_board(cam, 9, 6, 0.02, origin, yaw), 9, 6)
    assert c is not None
    g, ok = cam.pixel_to_ground(c)
    assert ok.all()
    # 사람이 재는 값: 0번 코너 위치와 0번에서 1번으로 가는 방향
    d = g[1] - g[0]
    pred = floor_board_xy(9, 6, 0.02, g[0], np.degrees(np.arctan2(d[1], d[0])))
    assert np.abs(pred - g).max() < 0.003
