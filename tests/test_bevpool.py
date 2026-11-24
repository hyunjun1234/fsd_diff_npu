import numpy as np
import pytest

from realityview import bevpool as bp
from realityview.camera_model import Camera

# =====================================================================
# 공통 준비: 작은 카메라 배치와 격자
# =====================================================================
# 맥과 파이 모두에서 1초 안에 끝나도록 작은 크기를 씁니다.

IMG_HW = (32, 48)
FEAT_HW = (4, 6)
GRID = bp.BEVGrid(x_min=-8, x_max=8, y_min=-8, y_max=8, z_min=-1.0, z_max=3.0, cell=1.0)
BINS = bp.DepthBins(d_min=1.0, d_max=13.0, step=1.0)  # 12개, 일부는 격자(8 m) 밖으로 나감


def _cam(name, yaw, pos, pitch=0.0):
    K = [[24.0, 0, 23.5], [0, 24.0, 15.5], [0, 0, 1]]
    return Camera(name, IMG_HW[1], IMG_HW[0], K, pos, yaw_deg=yaw, pitch_deg=pitch)


def _rig(cams):
    return bp.cameras_to_arrays(cams, IMG_HW)


def _random_inputs(n_cams, C, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n_cams, BINS.count) + FEAT_HW)
    depth = np.exp(logits)
    depth /= depth.sum(axis=1, keepdims=True)  # 깊이 방향 softmax처럼 합이 1
    feat = rng.normal(size=(n_cams, C) + FEAT_HW)
    return depth.astype(np.float32), feat.astype(np.float32)


def _fast_methods():
    """이 장비에서 쓸 수 있는 빠른 구현 목록. C는 빌드되면 포함합니다."""
    methods = ["reduceat", "bincount"]
    if bp.get_c_backend() is not None:
        methods.append("c")
    return methods


# =====================================================================
# 1. 빠른 구현이 기준 구현과 같은지
# =====================================================================


@pytest.mark.parametrize("seed", [0, 1])
def test_fast_methods_match_reference(seed):
    cams = [_cam("f", 0, [1.0, 0, 1.5]), _cam("b", 180, [-1.0, 0, 1.5]), _cam("l", 90, [0, 0.8, 1.2], pitch=10)]
    K, R, t = _rig(cams)
    depth, feat = _random_inputs(len(cams), C=5, seed=seed)
    ref = bp.bev_pool_reference(depth, feat, K, R, t, GRID, BINS, IMG_HW)
    plan = bp.precompute_plan(K, R, t, GRID, BINS, FEAT_HW, IMG_HW)
    assert ref.shape == (5, GRID.nx, GRID.ny)
    assert np.abs(ref).max() > 0  # 시험이 비어 있지 않은지
    for m in _fast_methods():
        for nth in (1, 4):
            out = bp.bev_pool(depth, feat, plan, method=m, nthreads=nth)
            assert out.shape == ref.shape and out.dtype == np.float32
            np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0, err_msg=f"{m} threads={nth}")


def test_hand_computed_single_point():
    # 카메라 한 대가 정면(+X)을 보고 (0, 0, 1)에 있음. 특징맵 1x1, 광축 위 픽셀, 깊이 2 m 하나.
    # 점은 차량 좌표 (2, 0, 1) 이므로 x 셀 floor((2 + 8) / 1) = 10, y 셀 floor((0 + 8) / 1) = 8.
    K = np.array([[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1]]])
    R = _cam("f", 0, [0, 0, 1]).R_vc[None]
    t = np.array([[0.0, 0.0, 1.0]])
    bins = bp.DepthBins(2.0, 3.0, 1.0)
    depth = np.array([0.25], dtype=np.float32).reshape(1, 1, 1, 1)
    feat = np.array([1.0, -2.0, 4.0], dtype=np.float32).reshape(1, 3, 1, 1)
    plan = bp.precompute_plan(K, R, t, GRID, bins, (1, 1), (1, 1))
    assert plan.n_kept == 1 and plan.iv_cell.tolist() == [10 * GRID.ny + 8]
    expected = np.zeros((3, GRID.nx, GRID.ny), dtype=np.float32)
    expected[:, 10, 8] = [0.25, -0.5, 1.0]
    np.testing.assert_allclose(bp.bev_pool_reference(depth, feat, K, R, t, GRID, bins, (1, 1)), expected, atol=1e-6)
    for m in _fast_methods():
        np.testing.assert_allclose(bp.bev_pool(depth, feat, plan, method=m), expected, atol=1e-6)


# =====================================================================
# 2. 격자 밖 점은 버려지는지
# =====================================================================


def _single_point_plan(cam_pos):
    """정면 카메라 한 대, 깊이 2 m 점 하나짜리 계획."""
    K = np.array([[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1]]])
    R = _cam("f", 0, cam_pos).R_vc[None]
    t = np.array([cam_pos], dtype=np.float64)
    return bp.precompute_plan(K, R, t, GRID, bp.DepthBins(2.0, 3.0, 1.0), (1, 1), (1, 1))


def test_point_just_below_lower_bound_is_dropped():
    # 점 x = -8.05: 정수 변환만 하면 0번 셀로 잘못 들어가지만 floor를 쓰면 -1이라 버려져야 합니다.
    assert _single_point_plan([-10.05, 0.0, 1.0]).n_kept == 0
    # 점 x = -7.95: 0번 셀에 들어가야 합니다.
    plan = _single_point_plan([-9.95, 0.0, 1.0])
    assert plan.n_kept == 1 and plan.iv_cell.tolist() == [0 * GRID.ny + 8]


def test_points_outside_xy_and_z_are_dropped():
    assert _single_point_plan([7.0, 0.0, 1.0]).n_kept == 0  # x = 9 > 8
    assert _single_point_plan([0.0, 8.5, 1.0]).n_kept == 0  # y = 8.5 > 8
    assert _single_point_plan([0.0, 0.0, -1.5]).n_kept == 0  # z = -1.5 < -1
    assert _single_point_plan([0.0, 0.0, 3.0]).n_kept == 0  # z = 3.0, 위 경계는 포함하지 않음


def test_dropped_points_carry_no_weight():
    # 특징을 모두 1로 두면 BEV 합계는 격자 안 점들의 깊이 확률 합과 같아야 합니다.
    cams = [_cam("f", 0, [1.0, 0, 1.5]), _cam("r", -90, [0, -0.8, 1.5])]
    K, R, t = _rig(cams)
    depth, _ = _random_inputs(2, C=1, seed=3)
    feat = np.ones((2, 1) + FEAT_HW, dtype=np.float32)
    plan = bp.precompute_plan(K, R, t, GRID, BINS, FEAT_HW, IMG_HW)
    assert 0 < plan.n_kept < plan.n_total  # 일부는 실제로 격자 밖
    kept_mass = depth.reshape(-1)[plan.depth_idx].astype(np.float64).sum()
    assert kept_mass < depth.sum() - 1e-3
    for m in _fast_methods():
        out = bp.bev_pool(depth, feat, plan, method=m)
        assert abs(float(out.sum(dtype=np.float64)) - kept_mass) < 1e-3


# =====================================================================
# 3. 카메라가 보지 않는 셀은 0인지
# =====================================================================


def test_cells_without_camera_are_zero():
    # 정면 카메라 한 대(차량 x = 1 m)는 뒤쪽을 볼 수 없으므로 x < 1 m 셀은 모두 0이어야 합니다.
    cams = [_cam("f", 0, [1.0, 0, 1.5])]
    K, R, t = _rig(cams)
    depth, feat = _random_inputs(1, C=4, seed=5)
    plan = bp.precompute_plan(K, R, t, GRID, BINS, FEAT_HW, IMG_HW)
    behind = int(np.floor((1.0 - GRID.x_min) / GRID.cell))  # x = 1 m가 들어가는 셀 번호
    covered = np.zeros(GRID.nx * GRID.ny, dtype=bool)
    covered[plan.iv_cell] = True
    covered = covered.reshape(GRID.nx, GRID.ny)
    assert not covered[:behind].any()
    for m in _fast_methods() + ["reference"]:
        if m == "reference":
            out = bp.bev_pool_reference(depth, feat, K, R, t, GRID, BINS, IMG_HW)
        else:
            out = bp.bev_pool(depth, feat, plan, method=m)
        assert np.all(out[:, :behind, :] == 0), m
        assert np.all(out[:, ~covered] == 0), m  # 점이 하나도 없는 셀 전부
        assert np.abs(out[:, covered]).max() > 0, m


# =====================================================================
# 4. C 빌드가 안 될 때 NumPy로 돌아가는지
# =====================================================================


def test_missing_compiler_returns_none(tmp_path):
    # 빈 폴더라 캐시된 라이브러리가 없으므로 반드시 빌드를 시도하고, 없는 컴파일러라 실패해야 합니다.
    assert bp.load_c_backend(cc=str(tmp_path / "no_such_cc"), build_dir=tmp_path) is None
    assert not list(tmp_path.glob("*.so"))


def test_auto_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(bp, "get_c_backend", lambda: None)
    assert bp.resolve_method("auto") == "bincount"
    with pytest.raises(RuntimeError):
        bp.bev_pool(*_random_inputs(1, 2), bp.precompute_plan(*_rig([_cam("f", 0, [1, 0, 1.5])]), GRID, BINS, FEAT_HW, IMG_HW), method="c")
    cams = [_cam("f", 0, [1.0, 0, 1.5])]
    K, R, t = _rig(cams)
    depth, feat = _random_inputs(1, C=3, seed=7)
    plan = bp.precompute_plan(K, R, t, GRID, BINS, FEAT_HW, IMG_HW)
    ref = bp.bev_pool_reference(depth, feat, K, R, t, GRID, BINS, IMG_HW)
    np.testing.assert_allclose(bp.bev_pool(depth, feat, plan), ref, atol=1e-4, rtol=0)


def test_c_backend_reports_build_info():
    backend = bp.get_c_backend()
    if backend is None:
        pytest.skip("C compiler not available on this machine")
    assert backend.path.exists() and backend.path.name == bp._lib_name()
    assert backend.threads_used(1) == 1
    if backend.openmp:
        assert backend.threads_used(4) >= 1
