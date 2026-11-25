from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview import layout_sim as ls  # noqa: E402
from realityview.pipeline import RealityViewPipeline  # noqa: E402
from realityview.sim import SimObject, SimWorld  # noqa: E402
from realityview.tracker import BevTracker  # noqa: E402
from realityview.types import FrameResult  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _body_distance(x: float, y: float) -> float:
    dx = max(ls.EGO_X_REAR - x, 0.0, x - ls.EGO_X_FRONT)
    dy = max(abs(y) - ls.EGO_HALF_W, 0.0)
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# 1. 투영과 정답 상자가 저장소의 기존 코드와 같은지
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["c4_fender", "c5_side160", "t8_ref"])
def test_fast_project_matches_camera_project(key):
    """numpy 투영이 Camera.project(OpenCV)와 같은 화소와 같은 앞뒤 판정을 낸다(핀홀, 어안 모두)."""
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-30, 45, 4000), rng.uniform(-15, 15, 4000), rng.uniform(0, 3.2, 4000)]
    for cam in ls.make_layouts()[key].cameras():
        uv0, f0 = cam.project(pts)
        uv1, f1 = ls.fast_project(cam, pts)
        assert np.array_equal(f0, f1)
        assert np.abs(uv0[f0] - uv1[f0]).max() < 1e-3


def test_full_rule_matches_simworld():
    """full 규칙(자차 차체 가림 끔)은 SimWorld가 그리는 정답 상자와 물체 목록, 상자 값이 같다."""
    cams = ls.make_layouts()["c4_fender"].cameras()
    objs = []
    for x in (-22.0, -12.0, -6.0, 0.0, 6.0, 12.0, 21.0, 33.0):
        for y in (-7.0, -3.5, 3.5, 7.0):
            objs.append(SimObject("car", x, y, 0.0, 0.0, 4.5, 1.9, 1.5, (90, 90, 90)))
    objs += [SimObject("truck", 15.0, 0.0, 0.0, 0.0, 8.0, 2.5, 3.0, (90, 90, 90)),
             SimObject("person", 5.0, 2.0, 0.0, 0.0, 0.5, 0.5, 1.7, (90, 90, 90)),
             SimObject("car", -9.0, 0.0, 0.0, 0.0, 4.5, 1.9, 1.5, (90, 90, 90))]
    world = SimWorld(cams, objects=objs)
    world.set_time(0.0)
    arr = lambda name: np.array([getattr(o, name) for o in objs])  # noqa: E731
    seen = 0
    for k, cam in enumerate(cams):
        _, dets, ids, _ = world.render_camera(k, ground=False)
        boxes, ok, _, _ = ls.object_boxes(cam, arr("x"), arr("y"), arr("length"), arr("width"), arr("height"),
                                          rule="full", ego_body=False)
        assert sorted(ids) == np.nonzero(ok)[0].tolist()
        for d, i in zip(dets, ids):
            assert np.allclose(d.box, boxes[i], atol=1e-3)
        seen += len(ids)
    assert seen >= 20  # 비교한 상자가 충분한지


def test_partial_rule_keeps_object_wider_than_view():
    """옆 차로에 나란히 선 트럭은 영상보다 넓어 full 규칙에서는 빠지고, partial 규칙에서는 보이는 부분이 상자로 남는다."""
    cam = ls.CamSpec("left", "starvis2_86", "bpillar", 90.0, 20.0).build()
    args = (cam, [1.2], [3.5], [8.0], [2.5], [3.0])
    _, ok_full, _, _ = ls.object_boxes(*args, rule="full", ego_body=False)
    boxes, ok, frac, contact = ls.object_boxes(*args, rule="partial")
    assert not ok_full[0] and ok[0]
    assert 0.05 < frac[0] < 1.0
    assert boxes[0, 2] - boxes[0, 0] > 0.8 * cam.width
    assert contact[0]  # 가까운 쪽 바닥선이 영상 안에 있다
    # 차 전체가 영상 안에 들어오면 핀홀 렌즈에서는 두 규칙의 상자가 같다(직선이 직선으로 찍히므로 꼭짓점이 곧 끝이다)
    pin = ls.CamSpec("left", "cm3_wide", "bpillar", 90.0, 20.0).build()
    far = ([1.2], [9.0], [4.5], [1.9], [1.5])
    b_full, ok_f, _, _ = ls.object_boxes(pin, *far, rule="full", ego_body=False)
    b_part, ok_p, frac_p, touch_p = ls.object_boxes(pin, *far, rule="partial", ego_body=False)
    assert ok_f[0] and ok_p[0] and touch_p[0] and frac_p[0] > 0.99
    assert np.allclose(b_full, b_part, atol=1e-6)
    # 어안 렌즈에서는 모서리 사이가 바깥으로 휘어 찍히므로 겉면 점으로 만든 상자가 꼭짓점 상자를 감싼다
    b_full, _, _, _ = ls.object_boxes(cam, *far, rule="full", ego_body=False)
    b_part, _, _, _ = ls.object_boxes(cam, *far, rule="partial", ego_body=False)
    assert np.all(b_part[0, :2] <= b_full[0, :2] + 1e-6) and np.all(b_part[0, 2:] >= b_full[0, 2:] - 1e-6)
    assert b_full[0, 1] - b_part[0, 1] > 1.0


# ---------------------------------------------------------------------------
# 2. 자차 차체와 창틀이 가리는 곳
# ---------------------------------------------------------------------------
def test_ego_body_blocks_ground_near_the_car():
    wind = ls.MOUNTS["wind_main"][0]
    # 앞유리 카메라: 보닛에 가려 앞범퍼(3.78m) 앞 2m쯤까지 바닥이 안 보인다
    assert ls.ego_blocks(wind, np.array([[4.5, 0.0], [5.5, 0.0], [8.0, 0.0], [20.0, 0.0]])).tolist() == [True, True, False, False]
    # 같은 자리라도 보닛 위로 올라온 점은 보인다
    assert ls.ego_blocks(wind, np.array([[5.0, 0.0, 1.2], [5.0, 0.0, 0.3]])).tolist() == [False, True]
    # 번호판 위 후방 카메라는 차체 끝에 있어 가리지 않는다
    assert not ls.ego_blocks(ls.MOUNTS["rear"][0], np.array([[-1.5, 0.0], [-3.0, 1.0], [-10.0, -4.0]])).any()
    # 옆유리 안쪽 카메라는 문짝에 가려 차 바로 옆 바닥이 안 보인다
    side = ls.MOUNTS["side_glass"][0]
    assert ls.ego_blocks(side, np.array([[0.95, 1.2], [0.95, 1.5], [0.95, 2.5], [0.95, 6.0]])).tolist() == [True, True, False, False]
    # 뒷유리 안쪽 카메라는 트렁크에 가려 뒤범퍼 뒤 바닥이 안 보인다
    assert ls.ego_blocks(ls.MOUNTS["rear_glass"][0], np.array([[-2.0, 0.0], [-3.5, 0.0], [-8.0, 0.0]])).tolist() == [True, True, False]


def test_window_frame_limits_inside_glass_camera():
    """유리 안쪽 카메라는 창이 향한 방향에서 정해진 각 밖을 보지 못한다. 카메라를 돌려 달아도 한계는 창 기준이다."""
    spec = ls.CamSpec("left", "imx291_160", "side_glass")
    assert spec.clip() == (90.0, 62.0)
    assert ls.CamSpec("right", "imx291_160", "side_glass", right=True).clip() == (-90.0, 62.0)
    assert ls.CamSpec("left", "imx291_160", "bpillar").clip() is None
    cam = spec.build()
    az = math.radians(25.0)  # 차 앞쪽에서 25도 = 창 방향(90도)에서 65도. 렌즈 화각(좌우 69.7도) 안, 창틀 한계(62도) 밖
    px, py = cam.position[0] + 6.0 * math.cos(az), cam.position[1] + 6.0 * math.sin(az)
    person = ([px], [py], [0.5], [0.5], [1.7])
    assert ls.object_boxes(cam, *person, ego_body=False)[1][0]
    assert not ls.object_boxes(cam, *person, ego_body=False, h_clip=spec.clip())[1][0]
    turned = ls.CamSpec("left", "imx291_160", "side_glass", yaw=60.0)
    assert turned.clip() == (90.0, 62.0)
    assert not ls.object_boxes(turned.build(), *person, ego_body=False, h_clip=turned.clip())[1][0]


# ---------------------------------------------------------------------------
# 3. 장면 경로, 검출 모형, 처리량 모형
# ---------------------------------------------------------------------------
def test_loop_path_keeps_offset_and_closes():
    f = ls.loop_path(1.5, 20.0)
    prev = f(0.0)
    for t in np.linspace(0.0, 20.0, 801)[1:]:
        x, y, _ = f(float(t))
        assert abs(_body_distance(x, y) - 1.5) < 1e-6
        assert math.hypot(x - prev[0], y - prev[1]) < 0.06  # 뚝 끊기는 곳이 없다(한 바퀴 약 22.8m를 800칸으로)
        prev = (x, y)
    # 같은 주기로 도는 사람들은 늘 떨어져 있다
    a, b = ls.loop_path(1.0, 24.0, 0.0), ls.loop_path(2.0, 24.0, 1 / 3)
    assert min(math.hypot(a(t)[0] - b(t)[0], a(t)[1] - b(t)[1]) for t in np.linspace(0, 24, 241)) > 2.5


def test_lane_and_pingpong_paths():
    f = ls.lane_path(3.5, 40.0, 5.0)
    assert f(0.0) == (40.0, 3.5, 0)
    x, y, k = f(1.5)  # 47.5m는 범위(-35~45m) 밖이라 반대편에서 다시 나온다
    assert (round(x, 6), y, k) == (-32.5, 3.5, 1)
    g = ls.pingpong_path(0.0, 14.0, 18.0, 0.5)
    xs = [g(float(t))[0] for t in np.linspace(0.0, 40.0, 401)]
    assert min(xs) >= 14.0 - 1e-9 and max(xs) <= 18.0 + 1e-9
    assert max(xs) - min(xs) > 3.9


@pytest.mark.parametrize("key", list(ls.make_scenarios()))
def test_scenario_actors_never_overlap(key):
    """장면의 물체끼리, 물체와 자차가 겹치는 순간이 없다(겹치면 가림 계산과 정답 짝짓기가 뜻을 잃는다)."""
    scen = ls.make_scenarios()[key]
    sizes = [a.size() for a in scen.actors]
    for t in np.arange(0.0, scen.duration, 0.1):
        st = [a.path(float(t)) for a in scen.actors]
        for i, (xi, yi, _) in enumerate(st):
            li, wi, _ = sizes[i]
            inside_x = xi - li / 2 < ls.EGO_X_FRONT and xi + li / 2 > ls.EGO_X_REAR
            assert not (inside_x and abs(yi) - wi / 2 < ls.EGO_HALF_W), (key, t, i)
            for j in range(i + 1, len(st)):
                xj, yj, _ = st[j]
                lj, wj, _ = sizes[j]
                assert max(abs(xi - xj) - (li + lj) / 2, abs(yi - yj) - (wi + wj) / 2) > 0.0, (key, t, i, j)


def test_det_model_size_truncation_and_edge():
    cam = ls.make_layouts()["c5_side160"].cameras()[2]
    big, small = np.array([100.0, 100.0, 220.0, 200.0]), np.array([100.0, 100.0, 108.0, 108.0])
    base, flat, hard = ls.DET_MODELS["base"], ls.DET_MODELS["flat"], ls.DET_MODELS["hard"]
    assert base.prob(big, 0.0, cam) == pytest.approx(0.95)
    assert base.prob(small, 0.0, cam) == 0.0  # NPU 입력 기준 6.7px
    assert flat.prob(small, 0.0, cam) == pytest.approx(0.95)
    assert base.prob(big, 0.0, cam, frac=0.2) < base.prob(big, 0.0, cam, frac=0.4) < base.prob(big, 0.0, cam, frac=0.9)
    assert hard.prob(big, math.radians(75.0), cam) < hard.prob(big, math.radians(40.0), cam)


def test_fps_model_matches_measured_points():
    fm = ls.FpsModel()
    assert fm.frame_ms(2) == pytest.approx(38.2, abs=0.05)  # 2대 실측
    assert fm.frame_ms(4) == pytest.approx(62.2, abs=0.05)  # 4대 실측 두 실행의 가운데
    assert fm.fps(4, 2) > fm.fps(5, 3) > fm.fps(6, 4) > fm.fps(7, 5) > fm.fps(8, 6)
    assert ls.FpsModel(decode_ms=10.0).fps(5, 3) < fm.fps(5, 3)
    assert ls.FpsModel(decode_ms=10.0).fps(5, 0) == pytest.approx(fm.fps(5, 0))  # USB 카메라가 없으면 해제 부담도 없다
    assert ls.FpsModel(seg=True).frame_ms(2) == pytest.approx(51.0, abs=1.0)  # 2대 세그 동시 실행 실측 51.0ms


def test_layouts_are_consistent():
    layouts = ls.make_layouts()
    for key, lay in layouts.items():
        cams = lay.cameras()
        assert len(cams) == lay.n == len(lay.h_clips())
        assert len({c.name for c in cams}) == lay.n
        assert lay.n_usb <= lay.n
    assert [layouts[k].n for k in ("c2_now", "c4_fender", "c5_side160", "c6_nomain", "c7_tesla", "c8_tesla")] == [2, 4, 5, 6, 7, 8]
    left, right = layouts["c5_side160"].cameras()[3:5]
    assert left.position[1] == -right.position[1] and left.yaw_deg == -right.yaw_deg


# ---------------------------------------------------------------------------
# 4. 추적 단계가 실제 파이프라인과 같은지
# ---------------------------------------------------------------------------
class _ListBackend:
    """미리 정한 검출 목록을 프레임마다 차례로 돌려주는 백엔드."""

    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def infer(self, frames):
        out = [FrameResult(detections=list(d)) for d in self.seq[self.i]]
        self.i += 1
        return out

    def close(self):
        pass


def test_track_step_matches_pipeline():
    """같은 검출을 넣으면 track_step과 RealityViewPipeline.step이 같은 트랙(ID, 위치)을 낸다."""
    layout = ls.make_layouts()["c5_side160"]
    scen = ls.make_scenarios()["overtake"]
    scen.duration = 6.0
    fps = 10.0
    cams, frames = ls.build_truth(layout, scen, fps)
    rng = np.random.default_rng(3)
    ones = np.ones((len(cams), len(scen.actors)))
    seq = [ls.noisy_detections(cams, fr, scen.actors, ls.DET_MODELS["base"], rng, ones) for fr in frames]
    assert sum(len(d) for dets in seq for d in dets) > 100
    pipe = RealityViewPipeline(cams, _ListBackend(seq), show_ground=False, show_cameras=False, tracker="robust")
    tracker = BevTracker(mode="robust")
    blank = [np.zeros((c.height, c.width, 3), np.uint8) for c in cams]
    shown = 0
    for dets in seq:
        a = pipe.step(blank, 1.0 / fps).tracks
        b = ls.track_step(cams, tracker, dets, 1.0 / fps)
        assert [(t.tid, t.label) for t in a] == [(t.tid, t.label) for t in b]
        for ta, tb in zip(a, b):
            assert np.allclose(ta.xy, tb.xy)
        shown += len(a)
    assert shown > 50


# ---------------------------------------------------------------------------
# 5. 지표와 배치 비교
# ---------------------------------------------------------------------------
def _walk(key: str, seconds: float = 24.0, fps: float = 13.0) -> dict:
    scen = ls.make_scenarios()["walk"]
    scen.duration = seconds
    return ls.merge_summaries([ls.run_once(ls.make_layouts()[key], scen, ls.DET_MODELS["base"], fps, 0)])


def test_metrics_count_objects_no_camera_sees():
    """앞뒤 2대만 있으면 차 옆을 지나는 사람은 어느 카메라에도 안 잡힌다. 그 시간도 분모에 들어간다."""
    m = _walk("c2_now")
    assert m["total"] == 3 * 24 * 13
    assert m["visible"] < 0.8 * m["total"]
    assert m["shown_ratio"] < m["visible_ratio"] + 0.25  # 예측으로 조금 더 이어 그리는 것까지만
    assert m["gaps_over_1s"] >= 3


def test_walk_around_needs_wide_side_cameras():
    """사람이 차를 한 바퀴 도는 장면. 펜더에서 뒤쪽 옆을 보는 5대는 앞쪽 옆에서 1초 넘게 끊기고,
    정측면을 160도로 보는 5대는 끊기지 않는다."""
    fender, side = _walk("c5_fender86"), _walk("c5_side160")
    assert side["shown_ratio"] > 0.97 and side["gaps_over_1s"] == 0
    assert fender["gaps_over_1s"] >= 3
    assert fender["shown_ratio"] < side["shown_ratio"] - 0.05


def test_cli_run_and_report(tmp_path):
    out = tmp_path / "run.json"
    script = os.path.join(ROOT, "scripts", "sim_camera_count.py")
    cmd = [sys.executable, script, "run", "--layouts", "c2_now", "--scenarios", "far", "--det", "flat", "--seeds", "1",
           "--fps", "8", "--jobs", "1", "--quiet", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    row = data["rows"][0]
    assert (row["layout"], row["scenario"], row["det"], row["fps"]) == ("c2_now", "far", "flat", 8.0)
    assert 0.0 < row["shown_ratio"] <= 1.0
    assert data["layouts"]["c2_now"]["n"] == 2
    rep = subprocess.run([sys.executable, script, "report", "--in", str(out)], capture_output=True, text=True, timeout=60)
    assert rep.returncode == 0 and "c2_now" in rep.stdout


# ---------------------------------------------------------------------------
# 차량용 설정 파일이 시뮬레이션에서 고른 배치와 같은지
# ---------------------------------------------------------------------------
def test_five_cam_config_matches_simulated_layout():
    """configs/five_cam.yaml의 카메라 5대가 배치 c5_side160과 같은 자리, 방향, 렌즈를 쓴다."""
    from realityview.config import load_config

    cams, grid, _ = load_config(os.path.join(ROOT, "configs", "five_cam.yaml"))
    sim = ls.make_layouts()["c5_side160"].cameras()
    assert [c.name for c in cams] == [c.name for c in sim]
    for c, r in zip(cams, sim):
        assert (c.width, c.height, c.model) == (r.width, r.height, r.model)
        assert np.allclose(c.K, r.K, atol=0.06)  # 설정 파일은 초점거리를 소수 첫째 자리까지 적는다
        assert np.allclose(c.position, r.position) and c.yaw_deg == r.yaw_deg and c.pitch_deg == r.pitch_deg
        assert c.max_fov_deg == r.max_fov_deg
        assert c.source["type"] == r.source["type"]
    assert (grid.x_min, grid.x_max, grid.y_min, grid.y_max) == (-15.0, 25.0, -12.0, 12.0)
    # USB 3대는 16:9 크기로 받아야 한다(받는 크기를 적지 않으면 드라이버가 4:3 모드를 고를 수 있다)
    usb = [c for c in cams if c.source["type"] == "usb"]
    assert len(usb) == 3 and all(c.source["capture"] == [1280, 720] and c.source["fourcc"] == "MJPG" for c in usb)
