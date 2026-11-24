import os

import numpy as np
import pytest
import yaml

from realityview.bev import IpmCompositor
from realityview.config import apply_scene_scale, load_config

# 카메라 자리를 테슬라 모델 Y 배치(docs/04_tesla_camera_layout.md)로 둔 설정과,
# 실차 단위 값(position_car, bev_car)을 scene.scale로 줄이는 설정 로더를 시험한다. 하드웨어 없이 돈다.

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = lambda name: os.path.join(ROOT, "configs", name)  # noqa: E731

# 04 문서의 추정 좌표(실차 m)와 이 저장소가 쓰기로 한 각도(도)
TESLA_FRONT_WIDE = ([1.80, -0.06, 1.45], 0.0, 3.0)
TESLA_REAR = ([-0.95, 0.0, 1.00], 180.0, 20.0)
TESLA_FENDER_L = ([2.25, 0.97, 0.92], 135.0, 5.0)
TESLA_FENDER_R = ([2.25, -0.97, 0.92], -135.0, 5.0)


def _cam_dict(ext: dict) -> dict:
    return {"name": "c", "width": 640, "height": 480,
            "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 320.0, "cy": 240.0}, "extrinsics": ext}


def _write(tmp_path, cfg: dict) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 설정 로더: 실차 단위 값을 축척으로 줄이기
# ---------------------------------------------------------------------------
def test_position_car_and_bev_car_are_scaled(tmp_path):
    cfg = {"scene": {"scale": 0.1},
           "bev_car": {"x_min": -15.0, "x_max": 25.0, "y_min": -12.0, "y_max": 12.0, "res": 0.1},
           "cameras": [_cam_dict({"position_car": [1.8, -0.06, 1.45], "yaw": 0.0, "pitch": 3.0})]}
    cams, grid, out = load_config(_write(tmp_path, cfg))
    np.testing.assert_allclose(cams[0].position, [0.18, -0.006, 0.145], atol=1e-12)
    assert cams[0].pitch_deg == 3.0  # 각도는 줄이지 않는다
    assert (grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.res) == (-1.5, 2.5, -1.2, 1.2, 0.01)
    assert "bev_car" not in out and "position_car" not in out["cameras"][0]["extrinsics"]


def test_measured_position_is_not_scaled_again(tmp_path):
    # 외부 보정 결과(모형 m)를 붙여 넣으면 position이 생긴다. 이 값에는 축척을 다시 곱하지 않는다
    cfg = {"scene": {"scale": 0.1}, "bev": {"res": 0.01, "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1},
           "cameras": [_cam_dict({"position": [0.2, 0.0, 0.15], "position_car": [1.8, 0.0, 1.45]})]}
    cams, _, _ = load_config(_write(tmp_path, cfg))
    np.testing.assert_allclose(cams[0].position, [0.2, 0.0, 0.15])


def test_config_without_car_keys_is_unchanged():
    raw = yaml.safe_load(open(CFG("two_cam.yaml"), encoding="utf-8"))
    assert apply_scene_scale(raw) == raw
    cams, grid, _ = load_config(CFG("two_cam.yaml"))
    assert grid.res == raw["bev"]["res"]
    np.testing.assert_allclose(cams[0].position, raw["cameras"][0]["extrinsics"]["position"])


def test_scale_loader_rejects_ambiguous_or_bad_input():
    base = {"cameras": [_cam_dict({"position": [0, 0, 1]})]}
    with pytest.raises(ValueError, match="bev_car"):
        apply_scene_scale({**base, "bev": {"res": 0.1}, "bev_car": {"res": 1.0}})
    with pytest.raises(ValueError, match="모르는 키"):
        apply_scene_scale({**base, "bev_car": {"resolution": 1.0}})
    with pytest.raises(ValueError, match="0보다"):
        apply_scene_scale({**base, "scene": {"scale": 0}})
    with pytest.raises(ValueError, match="position"):
        apply_scene_scale({"cameras": [_cam_dict({"yaw": 0.0})]})


# ---------------------------------------------------------------------------
# 테슬라 배치 값
# ---------------------------------------------------------------------------
def _assert_pose(cam, pose, scale=1.0):
    pos, yaw, pitch = pose
    np.testing.assert_allclose(cam.position, np.asarray(pos) * scale, atol=1e-9)
    assert (cam.yaw_deg, cam.pitch_deg, cam.roll_deg) == (yaw, pitch, 0.0)


def test_current_and_four_cam_use_tesla_positions():
    cams, _, _ = load_config(CFG("two_cam_current.yaml"))
    _assert_pose(cams[0], TESLA_FRONT_WIDE)
    _assert_pose(cams[1], TESLA_REAR)
    four, _, _ = load_config(CFG("four_cam.yaml"))
    assert [c.name for c in four] == ["front", "rear", "left", "right"]
    for cam, pose in zip(four, (TESLA_FRONT_WIDE, TESLA_REAR, TESLA_FENDER_L, TESLA_FENDER_R)):
        _assert_pose(cam, pose)


def test_model_is_current_layout_scaled_with_same_angles_and_lenses():
    cur, cur_grid, _ = load_config(CFG("two_cam_current.yaml"))
    mod, mod_grid, cfg = load_config(CFG("two_cam_model.yaml"))
    s = cfg["scene"]["scale"]
    for a, b in zip(cur, mod):
        np.testing.assert_allclose(b.position, a.position * s, atol=1e-9)
        assert (b.yaw_deg, b.pitch_deg, b.roll_deg) == (a.yaw_deg, a.pitch_deg, a.roll_deg)
        # 렌즈(내부 파라미터)와 영상 크기는 축척과 상관없이 같다
        assert (b.model, b.width, b.height) == (a.model, a.width, a.height)
        np.testing.assert_allclose(b.K, a.K)
    # BEV 범위는 실차 설정을 같은 축척으로 줄인 것
    for k in ("x_min", "x_max", "y_min", "y_max"):
        assert getattr(mod_grid, k) == pytest.approx(getattr(cur_grid, k) * s)


# ---------------------------------------------------------------------------
# 바닥 영역이 차 앞뒤에 놓이는지 (IPM 셀 담당 카메라)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["two_cam_current.yaml", "two_cam_model.yaml", "four_cam.yaml"])
def test_each_camera_owns_ground_on_its_own_side(name):
    cams, grid, cfg = load_config(CFG(name))
    s = float((cfg.get("scene") or {}).get("scale", 1.0))
    ipm = IpmCompositor(cams, grid)
    X, Y = grid.cell_centers()
    for i, cam in enumerate(cams):
        own = ipm.owner == i
        assert own.any(), cam.name
        a = np.radians(cam.yaw_deg)
        ahead = (X[own] - cam.position[0]) * np.cos(a) + (Y[own] - cam.position[1]) * np.sin(a)
        assert (ahead > 0).all(), cam.name  # 모든 셀이 카메라가 바라보는 쪽에 있다
    front, rear = ipm.owner == 0, ipm.owner == 1
    # 전방 셀은 앞유리보다 앞(x > 1.8 x s 근처), 후방 셀은 뒤범퍼(-1.011 x s)보다 뒤
    assert X[front].min() > 1.0 * s and X[rear].max() < -1.011 * s
