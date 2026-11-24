from __future__ import annotations

import importlib.util
import math
import os
import threading

import numpy as np
import pytest

from realityview.bev import BevGrid
from realityview.config import load_config
from realityview.pipeline import RealityViewPipeline, _RenderJob, _RenderWorker
from realityview.profiling import StageTimer
from realityview.render import CLASS_COLOR, compose_dashboard_3d
from realityview.render3d import ARROW_MAX_M, Renderer3D, TrackSmoother, View3DConfig
from realityview.sim import SimBackend, SimWorld
from realityview.tracker import Track

ROOT = os.path.join(os.path.dirname(__file__), "..")

# 3D 시점 렌더러 시험: 바닥 투영이 맞는지(좌우 뒤집힘 없음), 카메라 뒤 물체, 가려짐 순서,
# 트랙 보간, 대시보드 배치, 파이프라인의 --view 옵션.


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------
def _plain_cfg(**kw) -> View3DConfig:
    """격자, 고리, 자차, 글자, 화살표를 모두 끈 설정(그림 비교가 흔들리지 않게)."""
    base = dict(show_grid=False, show_rings=False, show_ego=False, show_ids=False, show_velocity=False)
    base.update(kw)
    return View3DConfig(**base)


def _track(tid, label, x, y, size=(4.5, 1.9), vel=(0.0, 0.0)) -> Track:
    return Track(tid, label, np.array([x, y], float), vel=np.array(vel, float), size=size)


def _load_run_demo():
    spec = importlib.util.spec_from_file_location("run_demo_10", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 바닥 투영
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("render_scale", [1.0, 0.5])
def test_ground_spot_lands_where_projected(render_scale):
    """바닥 영상의 밝은 점이 3D 화면에서 그 점을 직접 투영한 자리에 나와야 한다(좌우, 앞뒤 뒤집힘 검사)."""
    grid = BevGrid()
    r = Renderer3D(grid, _plain_cfg(render_scale=render_scale))
    img = np.zeros((grid.height, grid.width, 3), np.uint8)
    X, Y = 10.0, 3.0  # 앞 10m, 왼쪽 3m
    c, rr = grid.to_pixel(np.array([[X, Y]]))[0]
    img[int(rr) - 2:int(rr) + 3, int(c) - 2:int(c) + 3] = 255
    out = r.render([], ground_img=img)
    assert out.shape == (540, 960, 3)
    gray = out.sum(axis=2)
    gray[: r.horizon_row * 2] = 0  # 하늘은 빼고 본다
    v, u = np.unravel_index(np.argmax(gray), gray.shape)
    uv, z = r.project_output(np.array([[X, Y, 0.0]]))
    assert z[0] > 0
    assert abs(u - uv[0, 0]) <= 3 and abs(v - uv[0, 1]) <= 3
    assert u < out.shape[1] / 2  # 왼쪽 점은 화면 왼쪽에


def test_sky_above_horizon_is_not_mirrored_ground():
    """지평선 위는 하늘 배경이어야 한다. 바닥을 흰색으로 채워도 하늘 줄에 흰색이 새지 않는다."""
    grid = BevGrid()
    r = Renderer3D(grid, _plain_cfg())
    assert 0 < r.horizon_row < r.rh
    img = np.full((grid.height, grid.width, 3), 255, np.uint8)
    out = r.render([], ground_img=img)
    assert out[: r.horizon_row].max() < 200
    assert out[-1].min() > 50  # 바로 아래 바닥(자차 뒤)은 흰 바닥 영상이 깔림(색조 때문에 어둡게)


def test_scale_invariance():
    """장면 축척을 0.1로 줄이면 0.1배 한 점이 같은 화소에 투영된다(모형 설정에서도 같은 화면)."""
    p = np.array([[12.0, -2.0, 1.0], [4.0, 3.0, 0.0]])
    a = Renderer3D(BevGrid(), View3DConfig(), scale=1.0).project_output(p)[0]
    b = Renderer3D(BevGrid(-1.5, 2.5, -1.2, 1.2, 0.005), View3DConfig(), scale=0.1).project_output(p * 0.1)[0]
    assert np.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# 물체 그리기
# ---------------------------------------------------------------------------
def test_object_behind_or_at_camera_is_skipped():
    """가상 카메라 뒤나 카메라 옆(가까운 면)에 걸친 물체는 그리지 않고 오류도 없다."""
    grid = BevGrid()
    r = Renderer3D(grid, View3DConfig())
    empty = r.render([], refresh_ground=True)
    cam_x = r.cam.position[0]
    tracks = [_track(1, "car", -20.0, 0.0), _track(2, "car", cam_x, 0.0), _track(3, "person", cam_x - 0.1, 0.5)]
    out = r.render(tracks, refresh_ground=False)
    assert np.array_equal(out, empty)


def test_near_object_covers_far_object_regardless_of_input_order():
    """화가 알고리즘: 가까운 차가 먼 트럭을 가린다. 입력 순서를 바꿔도 그림이 같다."""
    grid = BevGrid()
    r = Renderer3D(grid, _plain_cfg(box_alpha=1.0))
    car, truck = _track(1, "car", 8.0, 0.0), _track(2, "truck", 16.0, 0.0, size=(8.0, 2.5))
    a = r.render([car, truck], refresh_ground=False)
    b = r.render([truck, car], refresh_ground=False)
    assert np.array_equal(a, b)
    # 차 윗면 앞쪽 가장자리 근처 점: 이 시선 뒤에 트럭 뒷면이 있다(높이 1.2m쯤)
    uv, _ = r.project_output(np.array([[10.0, 0.0, 1.5]]))
    u, v = (int(round(x)) for x in uv[0])
    px = a[v + 2, u].astype(int)
    assert np.abs(px - np.array(CLASS_COLOR["car"])).max() <= 6
    # 가려진 점이 정말 트럭 투영 안에 있는지(시험이 헛돌지 않게): 트럭만 그리면 같은 화소가 트럭색 계열
    t_only = r.render([truck], refresh_ground=False)
    assert np.abs(t_only[v + 2, u].astype(int) - px).max() > 20


def test_boxes_are_translucent():
    """box_alpha가 1보다 작으면 박스 너머 바닥색이 섞인다."""
    grid = BevGrid()
    solid = Renderer3D(grid, _plain_cfg(box_alpha=1.0)).render([_track(1, "car", 10.0, 0.0)])
    clear = Renderer3D(grid, _plain_cfg(box_alpha=0.5)).render([_track(1, "car", 10.0, 0.0)])
    r = Renderer3D(grid, _plain_cfg())
    uv, _ = r.project_output(np.array([[10.0, 0.0, 1.5]]))
    u, v = (int(round(x)) for x in uv[0])
    assert clear[v, u].astype(int).sum() < solid[v, u].astype(int).sum()


def test_person_cylinder_ids_and_arrows_draw():
    """사람(원기둥), ID 글자, 속도 화살표를 켜고 그려도 오류 없이 그림이 바뀐다."""
    grid = BevGrid()
    r = Renderer3D(grid, View3DConfig())
    base = r.render([])
    t = _track(7, "person", 9.0, 4.0, size=(0.6, 0.6), vel=(-2.0, 0.0))
    out = r.render([t], refresh_ground=False)
    assert (out != base).any()


# ---------------------------------------------------------------------------
# 트랙 보간
# ---------------------------------------------------------------------------
def test_track_smoother_snaps_then_eases():
    sm = TrackSmoother(tau=0.2)
    t1 = _track(1, "car", 0.0, 0.0)
    out = sm.update([t1], t=0.0)
    assert np.allclose(out[0].xy, [0.0, 0.0])
    t1b = _track(1, "car", 10.0, 0.0)
    out = sm.update([t1b, _track(2, "car", 5.0, 5.0)], t=0.1)
    a = 1.0 - math.exp(-0.1 / 0.2)
    assert np.allclose(out[0].xy, [10.0 * a, 0.0])
    assert np.allclose(out[1].xy, [5.0, 5.0])  # 처음 본 ID는 바로 그 자리
    assert np.allclose(t1b.xy, [10.0, 0.0])  # 원래 트랙은 고치지 않는다
    out = sm.update([t1b], t=0.1)  # 시간이 안 흘렀으면 그대로
    assert np.allclose(out[0].xy, [10.0 * a, 0.0])
    assert sm.update([], t=0.2) == [] and sm._pos == {}


# ---------------------------------------------------------------------------
# 설정과 대시보드
# ---------------------------------------------------------------------------
def test_config_checked_makes_even_and_rejects_bad():
    c = View3DConfig(width=961, height=541).checked()
    assert (c.width, c.height) == (960, 540)
    for bad in (dict(render_scale=0.05), dict(hfov_deg=5.0), dict(cam_height=0.0), dict(box_alpha=1.5)):
        with pytest.raises(ValueError):
            View3DConfig(**bad).checked()


def test_compose_dashboard_3d_fixed_even_size():
    main = np.zeros((540, 960, 3), np.uint8)
    cams = [np.zeros((576, 768, 3), np.uint8), np.zeros((432, 768, 3), np.uint8)]
    a = compose_dashboard_3d(main, cams, fps=12.3)
    assert a.shape == (540, 1280, 3)
    tall = [np.zeros((900, 300, 3), np.uint8)] * 3  # 세로로 다 안 들어가는 칸들
    b = compose_dashboard_3d(main, tall)
    assert b.shape == (540, 1280, 3)
    bev = np.zeros((800, 480, 3), np.uint8)
    c = compose_dashboard_3d(main, cams, bev=bev)
    assert c.shape[0] == 540 and c.shape[1] % 2 == 0 and c.shape[1] == 324 + 960 + 320


# ---------------------------------------------------------------------------
# 파이프라인과 실행 스크립트
# ---------------------------------------------------------------------------
def _sim_pipe(view="bev", **kw):
    cams, grid, cfg = load_config(os.path.join(ROOT, "configs", "two_cam.yaml"))
    world = SimWorld(cams)
    pipe = RealityViewPipeline(cams, SimBackend(world), grid, view=view, **kw)
    return world, pipe


def _run(world, pipe, n=5):
    outs = []
    for _ in range(n):
        world.step(0.1)
        outs.append(pipe.step(world.render(), 0.1))
    return outs


def test_pipeline_bev_view_unchanged_by_default():
    w1, p1 = _sim_pipe()
    w2, p2 = _sim_pipe(view="bev")
    for a, b in zip(_run(w1, p1), _run(w2, p2)):
        assert np.array_equal(a.bev, b.bev)
        # 맨 아래 FPS 글자는 벽시계로 재서 실행마다 다르므로 빼고 비교한다
        assert np.array_equal(a.dashboard[:-30], b.dashboard[:-30])
        assert a.view3d is None and a.bev is not None
    assert p1.r3d is None


@pytest.mark.parametrize("view", ["3d", "both"])
def test_pipeline_3d_views(view):
    world, pipe = _sim_pipe(view=view, view3d=View3DConfig(interp=True))
    outs = _run(world, pipe, 6)
    shapes = {o.dashboard.shape for o in outs}
    assert len(shapes) == 1
    h, w = outs[0].dashboard.shape[:2]
    assert h % 2 == 0 and w % 2 == 0
    assert outs[-1].view3d.shape == (540, 960, 3)
    assert (outs[-1].bev is None) == (view == "3d")
    pipe.close()


def test_pipeline_ground_every_skips_warp():
    """3D만 그릴 때 ground_every=3이면 바닥을 3프레임에 한 번만 다시 만든다."""
    world, pipe = _sim_pipe(view="3d", ground_every=3)
    _run(world, pipe, 7)
    assert pipe.r3d.ground_refreshes == 3  # 0, 3, 6번째 프레임
    pipe.close()


def test_pipeline_3d_render_thread():
    world, pipe = _sim_pipe(view="3d", render_thread=True)
    outs = _run(world, pipe, 4)
    assert outs[0].view3d is not None and outs[-1].dashboard.shape == outs[0].dashboard.shape
    pipe.close()


def test_pipeline_rejects_unknown_view():
    with pytest.raises(ValueError):
        _sim_pipe(view="side")


def test_run_demo_view_options():
    mod = _load_run_demo()
    ap = mod.build_parser()
    args = ap.parse_args([])
    assert args.view == "bev" and args.view3d_ground_every == 1 and not args.view3d_interp
    args = ap.parse_args(["--view", "3d", "--view3d-size", "640x360", "--view3d-scale", "0.5", "--view3d-interp",
                          "--view3d-cam", "6,4,20,60"])
    c = mod.make_view3d_config(args, {"view3d": {"box_alpha": 0.4}})
    assert (c.width, c.height, c.render_scale, c.interp) == (640, 360, 0.5, True)
    assert (c.cam_back, c.cam_height, c.pitch_deg, c.hfov_deg, c.box_alpha) == (6.0, 4.0, 20.0, 60.0, 0.4)
    with pytest.raises(SystemExit):
        mod.make_view3d_config(ap.parse_args(["--view3d-cam", "1,2"]), {})


def test_road_mask_is_painted_on_3d_ground():
    """도로 마스크가 있으면 3D 바닥의 그 자리 색이 바뀐다."""
    grid = BevGrid()
    r = Renderer3D(grid, _plain_cfg())
    ground = np.full((grid.height, grid.width, 3), 100, np.uint8)
    road = np.zeros((grid.height, grid.width), np.uint8)
    road[: grid.height // 2] = 1  # 앞쪽 절반(x 5~25m)을 도로로
    a = r.render([], ground_img=ground)
    b = r.render([], ground_img=ground, road_mask=road)
    uv, _ = r.project_output(np.array([[12.0, 0.0, 0.0]]))
    u, v = (int(round(x)) for x in uv[0])
    assert np.abs(a[v, u].astype(int) - b[v, u].astype(int)).max() > 10
    uv, _ = r.project_output(np.array([[2.0, 4.0, 0.0]]))  # 도로 아닌 곳은 그대로
    u, v = (int(round(x)) for x in uv[0])
    assert np.array_equal(a[v, u], b[v, u])


def test_velocity_arrow_is_clamped():
    """아주 큰 속도도 화살표는 ARROW_MAX_M 길이에서 끊긴다(상한 속도와 같은 그림)."""
    grid = BevGrid()
    r = Renderer3D(grid, _plain_cfg(show_velocity=True))
    big = r.render([_track(1, "car", 10.0, 0.0, vel=(0.0, 500.0))])
    cap = r.render([_track(1, "car", 10.0, 0.0, vel=(0.0, ARROW_MAX_M))], refresh_ground=False)
    assert np.array_equal(big, cap)


def test_render_worker_keeps_dropped_ground_refresh():
    """렌더링 스레드가 바닥 갱신 작업을 버리면 다음 작업이 갱신을 이어받는다."""
    started, release = threading.Event(), threading.Event()
    seen = []

    def slow_render(job):
        seen.append(job.refresh_ground)
        started.set()
        release.wait(5.0)
        return ("dash", None, None)

    w = _RenderWorker(slow_render, StageTimer())
    mk = lambda refresh: _RenderJob([], None, None, [], [], None, refresh_ground=refresh)  # noqa: E731
    w.submit(mk(False))
    assert started.wait(5.0)  # 첫 작업을 스레드가 가져가 멈춰 있는 동안
    j1, j2 = mk(True), mk(False)
    w.submit(j1)
    w.submit(j2)  # j1(갱신 차례)은 버려진다
    assert w.dropped == 1 and j2.refresh_ground is True
    release.set()
    w.close()


def test_bench_render3d_smoke():
    spec = importlib.util.spec_from_file_location("bench_r3d", os.path.join(ROOT, "scripts", "bench_render3d.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tr = mod.fake_tracks(8, scale=1.0)
    assert len(tr) == 8 and {t.label for t in tr} == {"car", "truck", "person"}
    grid = BevGrid()
    r = Renderer3D(grid, View3DConfig())
    ms = mod.bench(r, tr, None, None, None, refresh=False, reps=1)
    assert ms > 0
