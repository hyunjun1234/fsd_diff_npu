from __future__ import annotations

import importlib.util
import os
import sys

import cv2
import numpy as np
import pytest

from realityview import render
from realityview.bev import BevGrid, IpmCompositor
from realityview.config import load_config
from realityview.pipeline import RealityViewPipeline
from realityview.sim import ForcedMaskSeg
from realityview.types import FrameResult

ROOT = os.path.join(os.path.dirname(__file__), "..")

# 도로 마스크 덧칠과 BEV 합성을 OpenCV 방식으로 바꾼 뒤에도 그림이 예전과 같은지(반올림 차이 1 이하) 본다.
# 예전 식은 아래 "기준 함수"에 그대로 옮겨 두었다(커밋 bb2bc78의 render.py, bev.py).


# ---------------------------------------------------------------------------
# 기준 함수(예전 식)
# ---------------------------------------------------------------------------
def ref_annotate_overlay(img, road_mask):
    out = img.copy()
    m = road_mask.astype(bool)
    out[m] = (0.6 * out[m] + 0.4 * np.array([200, 120, 60])).astype(np.uint8)
    return out


def ref_bev(grid, ground_img, road_mask, coverage):
    H, W = grid.height, grid.width
    canvas = np.full((H, W, 3), render.BG, np.uint8)
    if ground_img is not None:
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
        canvas[m] = (0.5 * canvas[m] + 0.5 * np.array(render.ROAD)).astype(np.uint8)
    render._draw_rings(canvas, grid)
    render._draw_box(canvas, grid, np.array([render.EGO_SIZE[0] / 2 - render.EGO_REAR_OFFSET, 0.0]), render.EGO_SIZE,
                     0.0, render.EGO, fill=True)
    return canvas


def ref_warp(ipm, images, interpolation=cv2.INTER_LINEAR):
    out = None
    for i, (img, (m1, m2)) in enumerate(zip(images, ipm.maps)):
        w = cv2.remap(img, m1, m2, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if out is None:
            out = np.zeros_like(w)
        sel = ipm.owner == i
        out[sel] = w[sel]
    return out


def max_diff(a, b) -> int:
    return int(np.abs(a.astype(int) - b.astype(int)).max())


# ---------------------------------------------------------------------------
# 시험 데이터: 0~255 값을 모두 담은 영상과 무작위 마스크
# ---------------------------------------------------------------------------
@pytest.fixture
def img():
    # 모든 화소값(0~255)이 B, G, R 각 채널에 다 나오도록 만든 64x96 영상
    rng = np.random.default_rng(1)
    a = np.zeros((64, 96, 3), np.uint8)
    flat = np.arange(64 * 96) % 256
    a[..., 0] = flat.reshape(64, 96)
    a[..., 1] = rng.permutation(flat).reshape(64, 96)
    a[..., 2] = 255 - flat.reshape(64, 96)
    return a


@pytest.fixture
def mask():
    return (np.random.default_rng(2).random((64, 96)) < 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# 카메라 영상 덧칠
# ---------------------------------------------------------------------------
def test_annotate_overlay_within_one_and_unmasked_exact(img, mask):
    out = render.annotate_camera(img, [], "", mask)  # 이름이 빈 글자라 덧칠만 남는다
    ref = ref_annotate_overlay(img, mask)
    assert max_diff(out, ref) <= 1  # 반올림(OpenCV)과 버림(예전 식) 차이만 허용
    off = mask == 0
    np.testing.assert_array_equal(out[off], img[off])  # 마스크 밖은 정확히 그대로


def test_annotate_does_not_modify_input(img, mask):
    before = img.copy()
    render.annotate_camera(img, [], "cam", mask)
    np.testing.assert_array_equal(img, before)


def test_annotate_mask_variants(img, mask):
    # 마스크 없음, 전부 0은 입력 그대로
    np.testing.assert_array_equal(render.annotate_camera(img, [], "", None), img)
    np.testing.assert_array_equal(render.annotate_camera(img, [], "", np.zeros_like(mask)), img)
    # 전부 1은 모든 화소가 섞인다
    full = np.ones_like(mask)
    assert max_diff(render.annotate_camera(img, [], "", full), ref_annotate_overlay(img, full)) <= 1
    # bool 마스크, 255 값 마스크, int32 마스크도 0/1 uint8 마스크와 같은 결과
    base = render.annotate_camera(img, [], "", mask)
    for m in (mask.astype(bool), mask * 255, mask.astype(np.int32)):
        np.testing.assert_array_equal(render.annotate_camera(img, [], "", m), base)


def test_solid_cache_is_read_only():
    s = render._solid((4, 5, 3), (1, 2, 3))
    assert s is render._solid((4, 5, 3), (1, 2, 3))
    with pytest.raises(ValueError):
        s[0, 0, 0] = 9


# ---------------------------------------------------------------------------
# 탑뷰(바닥 영상, 도로 덧칠)
# ---------------------------------------------------------------------------
@pytest.fixture
def small_grid():
    return BevGrid(x_min=-3.2, x_max=3.2, y_min=-4.8, y_max=4.8, res=0.1)  # 64 x 96


def test_ground_lut_is_exact(small_grid, img):
    cov = np.zeros((64, 96), bool)
    cov[5:50, 10:80] = True
    # 바닥 영상만(도로 없음)은 예전과 화소까지 똑같아야 한다(색조 표를 같은 식으로 만들었으므로)
    np.testing.assert_array_equal(render.render_bev(small_grid, [], img, None, cov), ref_bev(small_grid, img, None, cov))
    np.testing.assert_array_equal(render.render_bev(small_grid, [], img, None, None), ref_bev(small_grid, img, None, None))


@pytest.mark.parametrize("cov_kind", ["bool", "uint8", "none"])
def test_bev_road_within_one(small_grid, img, mask, cov_kind):
    cov = np.zeros((64, 96), bool)
    cov[5:50, 10:80] = True
    cov_ref = None if cov_kind == "none" else cov
    cov_arg = {"bool": cov, "uint8": cov.astype(np.uint8), "none": None}[cov_kind]
    out = render.render_bev(small_grid, [], img, mask, cov_arg)
    ref = ref_bev(small_grid, img, mask, None if cov_ref is None else cov_ref.copy())
    assert max_diff(out, ref) <= 1
    # 도로가 아닌 셀(또는 카메라가 보지 않는 셀)은 정확히 같다
    road_sel = mask.astype(bool) if cov_ref is None else (mask.astype(bool) & cov_ref)
    np.testing.assert_array_equal(out[~road_sel], ref[~road_sel])


def test_bev_road_mask_values_and_inputs_untouched(small_grid, img, mask):
    cov = np.ones((64, 96), bool)
    cov_before, mask_before = cov.copy(), mask.copy()
    a = render.render_bev(small_grid, [], img, mask, cov)
    np.testing.assert_array_equal(render.render_bev(small_grid, [], img, mask * 255, cov), a)
    np.testing.assert_array_equal(render.render_bev(small_grid, [], img, mask.astype(bool), cov), a)
    np.testing.assert_array_equal(cov, cov_before)  # 예전 식은 m &= coverage로 복사본만 고쳤다. 지금도 입력은 그대로
    np.testing.assert_array_equal(mask, mask_before)


# ---------------------------------------------------------------------------
# IPM 합성
# ---------------------------------------------------------------------------
def test_warp_matches_reference_exactly():
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    ipm = IpmCompositor(cams, grid)
    rng = np.random.default_rng(3)
    imgs = [rng.integers(0, 256, (c.height, c.width, 3), dtype=np.uint8) for c in cams]
    masks = [(rng.random((c.height, c.width)) < 0.5).astype(np.uint8) for c in cams]
    np.testing.assert_array_equal(ipm.warp(imgs), ref_warp(ipm, imgs))
    np.testing.assert_array_equal(ipm.warp(masks, interpolation=0), ref_warp(ipm, masks, 0))
    cov = ipm.coverage()
    assert cov is ipm.coverage() and not cov.flags.writeable
    np.testing.assert_array_equal(cov, ipm.owner >= 0)


# ---------------------------------------------------------------------------
# 탑뷰 도로 마스크 캐시(세그가 갱신될 때만 다시 합성)
# ---------------------------------------------------------------------------
class MaskBackend:
    """정해 둔 마스크 목록을 그대로 붙여 돌려주는 가짜 검출 백엔드."""

    def __init__(self):
        self.masks = None

    def infer(self, frames):
        ms = self.masks or [None] * len(frames)
        return [FrameResult(detections=[], road_mask=m) for m in ms]

    def close(self):
        pass


def test_road_cache_hit_and_invalidate():
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    be = MaskBackend()
    pipe = RealityViewPipeline(cams, be, grid, show_ground=False, show_cameras=False)
    frames = [np.zeros((c.height, c.width, 3), np.uint8) for c in cams]
    rng = np.random.default_rng(4)
    new_masks = lambda: [(rng.random((c.height, c.width)) < 0.5).astype(np.uint8) for c in cams]  # noqa: E731
    try:
        # 마스크 없음: 합성 안 함
        pipe.step(frames, 0.1)
        assert pipe._road_cache is None and (pipe.road_cache_hits, pipe.road_cache_misses) == (0, 0)
        # 새 마스크: 합성(빗나감 1), 같은 객체가 다시 오면 재사용(적중 2)
        be.masks = new_masks()
        pipe.step(frames, 0.1)
        road1 = pipe._road_cache[1]
        pipe.step(frames, 0.1)
        pipe.step(frames, 0.1)
        assert (pipe.road_cache_hits, pipe.road_cache_misses) == (2, 1)
        assert pipe._road_cache[1] is road1
        np.testing.assert_array_equal(road1, pipe.ipm.warp(be.masks, interpolation=0))
        # 카메라 하나만 새 마스크로 바뀌어도 다시 합성하고, 결과는 새로 합성한 것과 같다
        be.masks = [be.masks[0], new_masks()[1]]
        pipe.step(frames, 0.1)
        assert pipe.road_cache_misses == 2 and pipe._road_cache[1] is not road1
        np.testing.assert_array_equal(pipe._road_cache[1], pipe.ipm.warp(be.masks, interpolation=0))
        # 한 카메라만 마스크가 있어도 합성하고(나머지는 빈 마스크), 마스크가 사라지면 캐시를 비운다
        be.masks = [be.masks[0], None]
        pipe.step(frames, 0.1)
        assert pipe.road_cache_misses == 3
        be.masks = None
        out = pipe.step(frames, 0.1)
        assert pipe._road_cache is None and out.bev.shape[:2] == (grid.height, grid.width)
    finally:
        pipe.close()


def test_road_cache_misses_every_frame_with_sim():
    # sim은 매 프레임 새 마스크 배열을 만든다: 캐시가 틀린 그림을 쓰지 않는지(늘 빗나가는지) 본다
    from realityview.sim import SimBackend, SimWorld

    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    world = SimWorld(cams)
    pipe = RealityViewPipeline(cams, SimBackend(world), grid)
    try:
        for _ in range(3):
            world.step(0.1)
            pipe.step(world.render(), 0.1)
        assert (pipe.road_cache_hits, pipe.road_cache_misses) == (0, 3)
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# 무작위 50% 마스크 시험 옵션
# ---------------------------------------------------------------------------
class CountSeg:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def segment(self, frames):
        self.calls += 1
        return [np.zeros(f.shape[:2], np.uint8) for f in frames]

    def close(self):
        self.closed = True


def test_forced_mask_seg():
    inner = CountSeg()
    seg = ForcedMaskSeg(inner, fraction=0.5)
    frames = [np.zeros((48, 64, 3), np.uint8), np.zeros((40, 64, 3), np.uint8)]
    a = seg.segment(frames)
    b = seg.segment(frames)
    c = seg.segment(frames)
    assert inner.calls == 3  # 진짜 세그(NPU)도 매번 돈다
    assert [m.shape for m in a] == [(48, 64), (40, 64)] and all(m.dtype == np.uint8 for m in a)
    assert all(0.4 < m.mean() < 0.6 for m in a)  # 화소 약 50%가 1
    assert all(x is not y for x, y in zip(a, b))  # 이웃한 호출은 다른 배열(캐시가 새 마스크로 본다)
    assert all(x is y for x, y in zip(a, c))  # 미리 만든 2장을 돌려 쓴다
    seg.close()
    assert inner.closed


def _load_run_demo():
    spec = importlib.util.spec_from_file_location("run_demo", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_demo_seg_mask_test_option(monkeypatch):
    mod = _load_run_demo()
    assert mod.build_parser().parse_args([]).seg_mask_test == "none"
    a = mod.build_parser().parse_args(["--seg-npu", "dxm1", "--seg-mask-test", "random50"])
    assert a.seg_mask_test == "random50"
    # --seg-npu 없이 쓰면 효과가 없으므로 오류로 알려 준다
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--frames", "1", "--headless", "--seg-mask-test", "random50"])
    with pytest.raises(SystemExit):
        mod.main()
