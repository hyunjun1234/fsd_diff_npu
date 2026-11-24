import importlib.util
import os
import sys

import cv2
import numpy as np
import pytest

from realityview.camera_model import Camera
from realityview.depth import (
    DepthScale,
    box_bottom_depth,
    colorize_depth,
    compare_boxes_with_ipm,
    depth_from_output,
    fit_depth_scale,
    ground_depth,
    ground_samples,
    guess_kind,
    resize_depth,
)
from realityview.detectors import DXM1Backend, check_hailo_seg_outputs
from realityview.types import Detection

sys.path.insert(0, os.path.dirname(__file__))
from test_dxm1 import FakeEngine  # noqa: E402

# 단안 깊이(realityview/depth.py), DXM1Backend 깊이 경로, Hailo seg HEF 확인, 새 스크립트 옵션 시험.
# 합성 장면: 높이 1.5m 카메라가 앞쪽 아래를 보고, 영상 아래쪽 절반이 평평한 지면이다.

ROOT = os.path.join(os.path.dirname(__file__), "..")
W, H = 320, 240


# ---------------------------------------------------------------------------
# 합성 장면 만들기
# ---------------------------------------------------------------------------
def _cam() -> Camera:
    K = np.array([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]])
    return Camera("front", W, H, K, np.array([0.0, 0.0, 1.5]), pitch_deg=15.0)


def _ground_z_map(cam: Camera) -> np.ndarray:
    """화소마다 지면 점까지의 광축 거리 z. 지면과 안 만나는 화소(하늘)는 30m로 채운다.

    ground_samples()가 정수 화소 좌표를 읽으므로 같은 좌표에서 만든다.
    """
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    z, _ = ground_depth(cam, np.stack([gx.ravel(), gy.ravel()], 1), max_range=200.0)
    z = z.reshape(H, W)
    return np.where(np.isfinite(z), z, 30.0).astype(np.float32)


def test_synthetic_scene_has_ground_at_bottom():
    z = _ground_z_map(_cam())
    # 아래쪽은 가깝고 위로 갈수록 멀어져야 한다(카메라가 아래를 보는지 확인)
    assert z[H - 5, W // 2] < z[H // 2 + 20, W // 2] < 30.0


# ---------------------------------------------------------------------------
# 출력 모양과 크기
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1, 8, 12), (1, 8, 12), (8, 12)])
def test_depth_from_output_shapes(shape):
    assert depth_from_output(np.ones(shape)).shape == (8, 12)


def test_depth_from_output_rejects_channels():
    with pytest.raises(ValueError):
        depth_from_output(np.ones((1, 3, 8, 12)))


def test_resize_depth_keeps_values():
    d = np.full((24, 32), 2.5, np.float32)
    r = resize_depth(d, (48, 64))
    assert r.shape == (48, 64) and np.allclose(r, 2.5)


# ---------------------------------------------------------------------------
# 컬러와 kind 추정
# ---------------------------------------------------------------------------
def test_colorize_near_is_bright_for_both_kinds():
    z = _ground_z_map(_cam())
    for kind, d in (("depth", z), ("inverse", 1.0 / z)):
        c = colorize_depth(d, kind)
        assert c.shape == (H, W, 3) and c.dtype == np.uint8
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        assert g[H - 3, W // 2] > g[5, W // 2]  # 가까운 아래쪽이 더 밝다


def test_guess_kind_from_floor():
    z = _ground_z_map(_cam())
    assert guess_kind(z)[0] == "depth"
    assert guess_kind(1.0 / z)[0] == "inverse"


# ---------------------------------------------------------------------------
# 박스 아래쪽 깊이
# ---------------------------------------------------------------------------
def test_box_bottom_depth_uses_strip_inside_box():
    d = np.full((100, 100), 9.0, np.float32)
    d[60:80, 40:60] = 3.0  # 물체
    # 박스 아래 가장자리(80)는 바닥(9)과 맞닿지만 안쪽 띠만 읽으므로 3이 나온다
    assert box_bottom_depth(d, (40, 60, 60, 80)) == pytest.approx(3.0)
    assert np.isnan(box_bottom_depth(d, (200, 200, 220, 220)))


# ---------------------------------------------------------------------------
# 지면 점으로 축척 맞추기
# ---------------------------------------------------------------------------
def test_fit_scale_depth_recovers_unknown_scale():
    cam = _cam()
    z = _ground_z_map(cam)
    raw, ref = ground_samples(cam, z / 2.5)  # 모델이 실제의 1/2.5로 낸다고 가정
    assert len(raw) > 50
    s = fit_depth_scale(raw, ref, "depth")
    assert s.a == pytest.approx(2.5, rel=1e-3)


def test_fit_scale_inverse_recovers_scale_and_shift_with_outliers():
    cam = _cam()
    z = _ground_z_map(cam)
    rel = 3.0 / z + 0.1  # 역깊이에 축척과 이동이 붙은 상대값
    raw, ref = ground_samples(cam, rel)
    raw = raw.copy()
    raw[::10] += 5.0  # 튀는 점 10%(지면 위 물체)
    s = fit_depth_scale(raw, ref, "inverse")
    assert s.kind == "inverse"
    assert np.allclose(s.apply(raw[1:10]), ref[1:10], rtol=1e-3)
    assert s.n == len(raw) - len(raw[::10])


def test_fit_scale_needs_points():
    with pytest.raises(ValueError):
        fit_depth_scale(np.array([1.0]), np.array([2.0]), "depth")
    with pytest.raises(ValueError):
        fit_depth_scale(np.array([1.0]), np.array([2.0]), "bad")


def test_ground_samples_respects_mask():
    cam = _cam()
    z = _ground_z_map(cam)
    mask = np.zeros((H, W), np.uint8)
    mask[:, : W // 2] = 1
    raw_all, _ = ground_samples(cam, z)
    raw_half, _ = ground_samples(cam, z, mask=mask)
    assert 0 < len(raw_half) < len(raw_all)


# ---------------------------------------------------------------------------
# 검출 박스: 깊이와 IPM 비교
# ---------------------------------------------------------------------------
def test_compare_boxes_with_ipm_zero_diff_on_flat_ground():
    cam = _cam()
    z = _ground_z_map(cam)
    # 박스 아래쪽 띠도 지면이라 깊이 값과 IPM 거리가 거의 같아야 한다(띠 높이만큼 조금 멀다)
    det = Detection("car", 0.9, (140.0, 170.0, 180.0, 200.0))
    rows = compare_boxes_with_ipm(cam, z / 2.0, [det], DepthScale("depth", 2.0))
    r = rows[0]
    assert r["ipm_z_m"] is not None and r["depth_m"] is not None
    assert abs(r["diff_m"]) < 0.1 * r["ipm_z_m"]
    # 축척이 없으면 미터 값과 차이는 넣지 않는다
    assert "diff_m" not in compare_boxes_with_ipm(cam, z, [det])[0]


# ---------------------------------------------------------------------------
# DXM1Backend 깊이 경로 (가짜 엔진)
# ---------------------------------------------------------------------------
def _depth_engine(path):
    if "depth" in path:
        return FakeEngine(
            path,
            input_shape=(1, 64, 96, 3),
            outputs=[("depth", [1, 1, 64, 96])],
            out_fn=lambda x: [x[..., 0].astype(np.float32).reshape(1, 1, 64, 96) / 100.0],
        )
    return FakeEngine(path)


def test_dxm1_backend_depth_only(tmp_path):
    p = tmp_path / "fake-depth.dxnn"
    p.write_bytes(b"x")
    be = DXM1Backend(None, depth_model=str(p), engine_factory=_depth_engine)
    frames = [np.full((120, 160, 3), 50, np.uint8), np.full((90, 160, 3), 200, np.uint8)]
    maps = be.estimate_depth(frames)
    # 원본 크기로 늘어나고, 단순 리사이즈 입력(레터박스 회색 114가 없음)이라 값이 그대로다
    assert maps[0].shape == (120, 160) and maps[1].shape == (90, 160)
    assert np.allclose(maps[0], 0.5) and np.allclose(maps[1], 2.0)
    assert be.estimate_depth(frames, full_size=False)[0].shape == (64, 96)
    assert len(be.infer_raw(frames, which="depth")) == 2
    with pytest.raises(RuntimeError):
        be.infer(frames)  # 검출 모델이 없다
    eng = be.depth.engine
    be.close()
    assert eng.disposed and be.depth is None


def test_dxm1_backend_requires_some_model_and_file(tmp_path):
    with pytest.raises(ValueError):
        DXM1Backend(None)
    with pytest.raises(FileNotFoundError):
        DXM1Backend(None, depth_model=str(tmp_path / "none.dxnn"), engine_factory=_depth_engine)


def test_dxm1_backend_without_depth_model_raises(tmp_path):
    p = tmp_path / "det.dxnn"
    p.write_bytes(b"x")
    be = DXM1Backend(str(p), engine_factory=_depth_engine)
    with pytest.raises(RuntimeError):
        be.estimate_depth([np.zeros((10, 10, 3), np.uint8)])


# ---------------------------------------------------------------------------
# Hailo seg HEF 확인, 스크립트 옵션
# ---------------------------------------------------------------------------
def test_check_hailo_seg_outputs():
    check_hailo_seg_outputs(["stdc1/argmax1"])
    with pytest.raises(ValueError):
        check_hailo_seg_outputs(["conv63", "conv61", "conv55", "conv48"], "yolov5n_seg_h8.hef")


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_09", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_demo_det_hef_default_none():
    mod = _load("run_demo")
    assert mod.build_parser().parse_args([]).det_hef is None
    assert mod.build_parser().parse_args(["--det-hef", "a.hef"]).det_hef == "a.hef"


def test_probe_image_input_nchw_and_resize(tmp_path):
    mod = _load("probe_dxm1")
    p = str(tmp_path / "img.png")
    cv2.imwrite(p, np.full((40, 80, 3), (10, 20, 30), np.uint8))
    x = mod._image_input(p, [1, 3, 16, 16], resize=True)
    assert x.shape == (1, 3, 16, 16) and x[0, 0, 0, 0] == 30  # RGB로 바뀌어 R 채널이 먼저
    y = mod._image_input(p, [1, 16, 16, 3])
    assert y.shape == (1, 16, 16, 3) and y[0, 0, 0, 0] == 114  # 레터박스 여백


def test_depth_check_on_images(tmp_path, monkeypatch):
    mod = _load("depth_check")
    import realityview.detectors as det_mod

    orig = det_mod.DxModel.__init__

    def fake_init(self, path, key="dx_det_model", engine_factory=None):
        orig(self, path, key, _depth_engine)

    monkeypatch.setattr(det_mod.DxModel, "__init__", fake_init)
    p = tmp_path / "fake-depth.dxnn"
    p.write_bytes(b"x")
    img = str(tmp_path / "f.png")
    cv2.imwrite(img, np.random.default_rng(0).integers(0, 255, (576, 768, 3), dtype=np.uint8))
    out = str(tmp_path / "res")
    argv = ["depth_check.py", "--config", os.path.join(ROOT, "configs", "two_cam_current.yaml"), "--depth", str(p),
            "--frames", "4", "--warmup", "1", "--images", img, img, "--out", out]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    assert os.path.isfile(out + ".json") and os.path.isfile(out + "_front.jpg")
