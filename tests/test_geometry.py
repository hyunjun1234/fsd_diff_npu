import cv2
import numpy as np
import pytest

from realityview.bev import BevGrid, IpmCompositor, merge_cross_camera, project_detections
from realityview.camera_model import Camera, intrinsics_from_hfov, pose_from_pnp
from realityview.config import load_config
from realityview.types import Detection, GroundObject


def make_cam(yaw=0.0, pitch=12.0, roll=0.0, pos=(1.9, 0.0, 1.3), model="pinhole", dist=None):
    return Camera(
        "c", 768, 432, intrinsics_from_hfov(768, 432, 100), np.array(pos), yaw, pitch, roll,
        dist=np.zeros(4) if dist is None else np.array(dist), model=model,
    )


def test_forward_camera_center_ray_hits_ground_ahead():
    cam = make_cam(pitch=12.0)
    xy, ok = cam.pixel_to_ground(np.array([[384.0, 216.0]]))
    assert ok[0]
    # 높이 1.3m, 12도 숙임 → 1.3 / tan(12°) ≈ 6.12m 앞
    assert xy[0, 0] == pytest.approx(1.9 + 1.3 / np.tan(np.radians(12)), abs=1e-6)
    assert xy[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_left_of_image_is_left_of_vehicle():
    cam = make_cam()
    xy, ok = cam.pixel_to_ground(np.array([[100.0, 400.0]]))
    assert ok[0] and xy[0, 1] > 0


@pytest.mark.parametrize("yaw", [0.0, 90.0, 180.0, -90.0, 35.0])
@pytest.mark.parametrize("model", ["pinhole", "fisheye"])
def test_project_then_backproject_roundtrip(yaw, model):
    dist = [0.05, -0.01, 0.001, 0.0] if model == "fisheye" else [-0.1, 0.02, 0.0, 0.0]
    cam = make_cam(yaw=yaw, pitch=20.0, roll=2.0, model=model, dist=dist)
    rng = np.random.default_rng(0)
    # 카메라가 보는 방향 앞쪽 지면 점들
    d = rng.uniform(3, 12, 50)
    a = np.radians(yaw) + rng.uniform(-0.5, 0.5, 50)
    pts = np.c_[cam.position[0] + d * np.cos(a), cam.position[1] + d * np.sin(a), np.zeros(50)]
    uv, front = cam.project(pts)
    assert front.all()
    xy, ok = cam.pixel_to_ground(uv)
    assert ok.all()
    np.testing.assert_allclose(xy, pts[:, :2], atol=1e-3)


def test_sky_pixels_are_invalid():
    cam = make_cam(pitch=5.0)
    _, ok = cam.pixel_to_ground(np.array([[384.0, 5.0]]))
    assert not ok[0]


def test_pose_from_pnp_recovers_extrinsics():
    cam = make_cam(yaw=-90.0, pitch=28.0, roll=-3.0, pos=(2.0, -0.95, 1.05))
    rvec, tvec = cam.rvec_tvec
    pos, yaw, pitch, roll = pose_from_pnp(rvec, tvec)
    np.testing.assert_allclose(pos, cam.position, atol=1e-9)
    assert (yaw, pitch, roll) == pytest.approx((-90.0, 28.0, -3.0), abs=1e-6)


def test_solvepnp_pipeline_matches_calib_script_logic():
    truth = make_cam(yaw=0.0, pitch=14.0, roll=1.0)
    XY = np.array([[4, 2, 0], [4, -2, 0], [7, 0, 0], [10, 3, 0], [10, -3, 0], [6, 1, 0]], float)
    uv, _ = truth.project(XY)
    n = cv2.undistortPoints(uv.reshape(-1, 1, 2), truth.K, truth.dist)
    ok, rvec, tvec = cv2.solvePnP(XY, n, np.eye(3), None)
    pos, yaw, pitch, roll = pose_from_pnp(rvec, tvec)
    np.testing.assert_allclose(pos, truth.position, atol=1e-4)
    assert pitch == pytest.approx(14.0, abs=1e-3)


def test_bev_grid_pixel_mapping():
    g = BevGrid(x_min=-10, x_max=10, y_min=-5, y_max=5, res=0.1)
    assert (g.height, g.width) == (200, 100)
    px = g.to_pixel(np.array([[10.0, 5.0], [0.0, 0.0]]))
    np.testing.assert_allclose(px, [[0, 0], [50, 100]])
    X, Y = g.cell_centers()
    np.testing.assert_allclose(g.to_pixel(np.c_[X[3, 7], Y[3, 7]]), [[7.5, 3.5]])


def test_ipm_coverage_and_owner():
    cams, grid, _ = load_config("configs/two_cam.yaml")
    ipm = IpmCompositor(cams, grid)
    X, Y = grid.cell_centers()
    # 차량 바로 앞 8m는 전방, 바로 뒤 8m는 후방 카메라가 맡는다
    r_front = int((grid.x_max - 8) / grid.res)
    r_rear = int((grid.x_max + 8) / grid.res)
    c = grid.width // 2
    assert ipm.owner[r_front, c] == 0
    assert ipm.owner[r_rear, c] == 1
    img = [np.full((c_.height, c_.width, 3), 50 + 100 * i, np.uint8) for i, c_ in enumerate(cams)]
    out = ipm.warp(img)
    assert out[r_front, c, 0] == 50 and out[r_rear, c, 0] == 150


def test_project_detection_car_center_pushed_back():
    cam = make_cam()
    target = np.array([[10.0, 0.0, 0.0]])
    uv, _ = cam.project(target)
    u, v = uv[0]
    det = Detection("car", 0.9, (u - 40, v - 60, u + 40, v))
    objs = project_detections(0, cam, [det])
    assert len(objs) == 1
    # 접지점(10m) + 차 길이 절반(2.25m)
    assert objs[0].xy[0] == pytest.approx(10.0 + 2.25, abs=0.05)
    assert objs[0].xy[1] == pytest.approx(0.0, abs=0.05)


def test_merge_cross_camera():
    a = GroundObject("car", 0.9, np.array([0.0, 5.0]), (4.5, 1.9), 0, distance=5)
    b = GroundObject("car", 0.8, np.array([0.5, 5.2]), (4.5, 1.9), 1, distance=10)
    c = GroundObject("person", 0.8, np.array([0.5, 5.2]), (0.6, 0.6), 1, distance=10)
    out = merge_cross_camera([a, b, c])
    assert len(out) == 2
