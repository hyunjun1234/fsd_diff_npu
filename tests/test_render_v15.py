from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from realityview import raster as rz
from realityview.bev import BevGrid, IpmCompositor
from realityview.config import load_config
from realityview.pipeline import RealityViewPipeline
from realityview.render_v15 import (
    DEFAULT_EGO_MODEL,
    VIEWS,
    RendererV15,
    V15Config,
    car_mesh,
    load_vehicle_model,
    person_mesh,
    stretch_mask,
)
from realityview.sim import SimBackend, SimWorld
from realityview.tracker import Track

ROOT = os.path.join(os.path.dirname(__file__), "..")

# v15풍 화면 시험: 삼각형 z-버퍼, 자차 모델, 시점 전환, IPM 번짐 마스크, 파이프라인과 실행 옵션.


def _track(tid, label, x, y, vel=(0.0, 0.0)) -> Track:
    return Track(tid, label, np.array([x, y], float), vel=np.array(vel, float), size=(4.5, 1.9))


def _ground(grid: BevGrid, val=120):
    g = np.full((grid.height, grid.width, 3), val, np.uint8)
    return g, np.ones((grid.height, grid.width), bool)


# ---------------------------------------------------------------------------
# 래스터라이저
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_c", [True, False])
def test_raster_nearer_triangle_wins(use_c, monkeypatch):
    if use_c and not rz.c_available():
        pytest.skip("C 컴파일러 없음")
    if not use_c:
        monkeypatch.setattr(rz, "c_available", lambda: False)
    img = np.zeros((40, 40, 3), np.uint8)
    z = np.zeros((40, 40), np.float32)
    uv = np.array([[2, 2], [38, 2], [20, 38], [2, 2], [38, 2], [20, 38]], float)
    iz = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])  # 앞 삼각형 1/z = 0.5(가까움), 뒤 0.2
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    col = np.array([[[0, 0, 255]] * 3, [[255, 0, 0]] * 3], np.uint8)
    rz.raster_vc(img, z, uv, iz, tris, col)
    assert tuple(img[15, 20]) == (0, 0, 255)  # 그린 순서와 상관없이 가까운 빨강
    assert img[0, 0].sum() == 0


def test_raster_vc_interpolates_colour():
    if not rz.c_available():
        pytest.skip("C 컴파일러 없음")
    img = np.zeros((10, 101, 3), np.uint8)
    z = np.zeros((10, 101), np.float32)
    uv = np.array([[0, 0], [101, 0], [101, 10], [0, 10]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    c0, c1 = [0, 0, 0], [200, 200, 200]
    col = np.array([[c0, c1, c1], [c0, c1, c0]], np.uint8)
    rz.raster_vc(img, z, uv, np.ones(4), tris, col)
    assert img[5, 5, 0] < 30 and img[5, 95, 0] > 170 and 80 < img[5, 50, 0] < 120


# ---------------------------------------------------------------------------
# 메시와 자차 모델
# ---------------------------------------------------------------------------
def test_ego_model_is_light_and_white():
    hi, lo = load_vehicle_model(DEFAULT_EGO_MODEL)
    assert len(hi.F) <= 15000 and len(lo.F) < len(hi.F)
    d = np.load(DEFAULT_EGO_MODEL)
    paint = d["C"][d["R"] == 1]
    assert len(paint) > 1000 and paint.min() >= 230  # 흰색 도장
    # 크기: 카니발 2011 길이 5.13m, 원점은 뒷바퀴 축(뒤끝이 -1m 근처), 바닥이 0
    assert abs(np.ptp(hi.V[:, 0]) - 5.13) < 0.05
    assert -1.4 < hi.V[:, 0].min() < -0.8 and abs(hi.V[:, 2].min()) < 0.05
    assert abs(hi.V[:, 1].mean()) < 0.1  # 좌우 가운데


def test_paint_override():
    hi, _ = load_vehicle_model(DEFAULT_EGO_MODEL, paint_bgr=(10, 20, 200))
    d = np.load(DEFAULT_EGO_MODEL)
    assert np.all(hi.C[d["R"] == 1] == (10, 20, 200))


def test_procedural_meshes_have_valid_faces():
    for m in (car_mesh(), person_mesh(0.7)):
        assert m.F.min() >= 0 and m.F.max() < len(m.V)
        assert len(m.C) == len(m.F) == len(m.shiny)
        cn = m.corner_normals()
        assert cn.shape == (len(m.F), 3, 3) and np.allclose(np.linalg.norm(cn, axis=2), 1, atol=1e-6)


# ---------------------------------------------------------------------------
# 렌더러
# ---------------------------------------------------------------------------
def test_render_size_and_deterministic():
    grid = BevGrid()
    g, cov = _ground(grid)
    tr = [_track(1, "car", 12, 0, (2, 0)), _track(2, "person", 8, 4, (0, -1.2)), _track(3, "truck", 20, -3.5)]
    outs = []
    for _ in range(2):
        r = RendererV15(grid, V15Config(width=640, height=360))
        for k in range(3):
            o = r.render(tr, g, None, cov, t=k * 0.1, ego_speed=10.0)
        outs.append(o)
    assert outs[0].shape == (360, 640, 3)
    assert np.array_equal(outs[0], outs[1])


def test_object_is_drawn_where_projected():
    grid = BevGrid()
    g, cov = _ground(grid, 90)
    r = RendererV15(grid, V15Config(path=False, shadows=False, show_ui=False))
    empty = r.render([], g, None, cov, t=0.0)
    r2 = RendererV15(grid, V15Config(path=False, shadows=False, show_ui=False))
    full = r2.render([_track(1, "car", 15.0, 3.5)], g, None, cov, t=0.0)
    uv, _ = r2.project(np.array([[15.0, 3.5, 0.8]]))
    u, v = int(uv[0, 0]), int(uv[0, 1])
    assert np.abs(full[v, u].astype(int) - empty[v, u].astype(int)).sum() > 30
    assert u < r2.rw / 2  # 왼쪽(+Y) 차는 화면 왼쪽


def test_scale_invariance_of_projection():
    p = np.array([[12.0, -2.0, 1.0], [4.0, 3.0, 0.0]])
    a = RendererV15(BevGrid(), V15Config()).project(p)[0]
    b = RendererV15(BevGrid(-1.5, 2.5, -1.2, 1.2, 0.005), V15Config(), scale=0.1).project(p * 0.1)[0]
    assert np.allclose(a, b, atol=1e-6)


def test_view_transition_eases_to_target():
    r = RendererV15(BevGrid(), V15Config(width=320, height=180, transition_s=1.0))
    g, cov = _ground(r.grid)
    r.render([], g, None, cov, t=0.0)
    r.set_view("rear")
    r.render([], g, None, cov, t=0.5)
    mid = r._pose
    assert VIEWS["rear"].azimuth != mid.azimuth != VIEWS["chase"].azimuth and r.moving
    r.render([], g, None, cov, t=1.2)
    assert r._pose == VIEWS["rear"] and not r.moving
    assert r.handle_key(ord("5")) and r.view == "top"
    assert not r.handle_key(ord("x"))
    with pytest.raises(ValueError):
        r.set_view("nope")


def test_tour_cycles_views():
    r = RendererV15(BevGrid(), V15Config(width=320, height=180, tour_s=1.0, tour_views=("chase", "rear")))
    g, cov = _ground(r.grid)
    seen = set()
    for k in range(30):
        r.render([], g, None, cov, t=k * 0.1)
        seen.add(r.view)
    assert seen == {"chase", "rear"}


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        V15Config(view="sky").checked()
    with pytest.raises(ValueError):
        V15Config(render_scale=2.0).checked()
    assert V15Config(width=641, height=361).checked().width == 640


def test_oncoming_car_faces_backward_only_with_ego_speed():
    r = RendererV15(BevGrid(), V15Config(width=320, height=180))
    g, cov = _ground(r.grid)
    tr = [_track(1, "car", 20, 3.5, (-25.0, 0.0))]  # 상대 -25 m/s
    for k in range(10):
        r.render(tr, g, None, cov, t=k * 0.1, ego_speed=12.0)
    assert abs(abs(r._yaw[1]) - np.pi) < 0.2  # 땅 기준 -13 m/s: 마주 오는 차
    r2 = RendererV15(BevGrid(), V15Config(width=320, height=180))
    for k in range(10):
        r2.render(tr, g, None, cov, t=k * 0.1)
    assert abs(r2._yaw[1]) < 1e-6  # 자차 속도를 모르면 차는 앞을 본다고 둔다


# ---------------------------------------------------------------------------
# IPM 번짐 마스크
# ---------------------------------------------------------------------------
def test_stretch_mask_covers_ground_behind_object():
    cams, grid, _ = load_config(os.path.join(ROOT, "configs", "two_cam.yaml"))
    ipm = IpmCompositor(cams, grid)
    assert stretch_mask(cams, ipm._owner_u8, grid, []) is None
    m = stretch_mask(cams, ipm._owner_u8, grid, [_track(1, "car", 10.0, 0.0)])
    px = lambda x, y: tuple(int(v) for v in grid.to_pixel(np.array([[x, y]]))[0])  # noqa: E731
    c, r = px(13.0, 0.0)  # 물체 바로 뒤(전방 카메라에서 볼 때 가려진 바닥)
    assert m[r, c] == 1
    c, r = px(10.0, 6.0)  # 옆은 가려지지 않는다
    assert m[r, c] == 0


# ---------------------------------------------------------------------------
# 파이프라인과 실행 스크립트
# ---------------------------------------------------------------------------
def test_pipeline_v15_view():
    cams, grid, _ = load_config(os.path.join(ROOT, "configs", "two_cam.yaml"))
    world = SimWorld(cams)
    pipe = RealityViewPipeline(cams, SimBackend(world), grid, view="v15", view3d=V15Config(width=640, height=360))
    pipe.ego_speed = world.ego_speed
    outs = []
    for _ in range(4):
        world.step(0.1)
        outs.append(pipe.step(world.render(), 0.1))
    assert outs[-1].view3d.shape == (360, 640, 3)
    assert len({o.dashboard.shape for o in outs}) == 1
    assert isinstance(pipe.r3d, RendererV15)
    pipe.close()


def test_run_demo_v15_options():
    spec = importlib.util.spec_from_file_location("run_demo_v15", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ap = mod.build_parser()
    args = ap.parse_args(["--view", "v15", "--v15-view", "rear", "--v15-tour", "3", "--v15-paint", "250,10,20",
                          "--view3d-size", "640x360"])
    c = mod.make_view3d_config(args, {})
    assert isinstance(c, V15Config)
    assert (c.view, c.tour_s, c.ego_paint, c.width) == ("rear", 3.0, (20, 10, 250), 640)
