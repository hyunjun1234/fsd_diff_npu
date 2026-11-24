from __future__ import annotations

import importlib.util
import os

import cv2
import numpy as np
import pytest

from realityview.bev import BevGrid, IpmCompositor
from realityview.config import load_config
from realityview.lanes import LaneConfig, LaneCurve, LaneDetector, LaneSmoother, assign_sides
from realityview.pipeline import RealityViewPipeline
from realityview.render import dash_runs, render_bev
from realityview.render3d import Renderer3D, View3DConfig
from realityview.sim import SimBackend, SimTrafficLight, SimWorld, default_traffic_lights
from realityview.traffic_lights import (
    LightTracker,
    TrafficLight,
    TrafficLightConfig,
    classify_light,
    crop_box,
    main_light,
    orientation_of,
)
from realityview.types import Detection

ROOT = os.path.join(os.path.dirname(__file__), "..")

# 차선 검출(고전 방식)과 신호등 색 판별 시험.
# 정답은 합성 장면(sim.py의 curvature, traffic_lights 옵션)에서 나온다. 실제 도로 사진은 저장소와 파이에 없다.
# 거리 정확도는 카메라 보정 전이라 판정하지 않고, 합성 장면 안에서 IPM이 맞을 때의 곡선 오차만 본다.

NEAR_X = (3.0, 15.0)  # 곡선 오차를 재는 X 범위(m). 먼 차선은 영상에서 화소 한 개보다 가늘다


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------
def _sim_ground(config: str, curvature: float = 0.0, t: float = 0.1, lights=None):
    cams, grid, _ = load_config(os.path.join(ROOT, config))
    world = SimWorld(cams, objects=[], curvature=curvature, traffic_lights=lights)
    world.step(t)
    frames = world.render()
    ipm = IpmCompositor(cams, grid)
    ground = ipm.warp(frames)
    road = ipm.warp(world.gt_masks, interpolation=0)
    return world, grid, ipm, frames, ground, road


def _match(lanes: list[LaneCurve], gt: dict, tol: float = 0.6) -> LaneCurve | None:
    """정답 차선과 X=0 횡위치가 가장 가까운 검출 곡선(tol 안에 없으면 None)."""
    best = min(lanes, key=lambda l: abs(l.offset - gt["c"]), default=None)
    if best is None or abs(best.offset - gt["c"]) > tol:
        return None
    return best


def _curve_error(ln: LaneCurve, gt: dict) -> float:
    """NEAR_X 안에서 검출 곡선과 정답 곡선의 횡위치 차이 최댓값(m). 검출 구간 밖은 재지 않는다."""
    x0, x1 = max(NEAR_X[0], ln.x_range[0]), min(NEAR_X[1], ln.x_range[1])
    xs = np.linspace(x0, x1, 25)
    return float(np.max(np.abs(ln.y_at(xs) - (gt["a"] * xs ** 2 + gt["c"]))))


# ---------------------------------------------------------------------------
# 차선 검출: 합성 장면 정답과 비교
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config", ["configs/two_cam.yaml", "configs/two_cam_current.yaml"])
@pytest.mark.parametrize("curvature", [0.0, 0.006, -0.006])
def test_lanes_match_sim_truth(config, curvature):
    """네 차선(점선 흰색 둘, 실선 노란색 둘)을 모두 찾고 곡선 오차, 실선/점선, 색, 좌우가 맞아야 한다."""
    world, grid, ipm, _, ground, road = _sim_ground(config, curvature)
    lanes = LaneDetector(grid, None, ipm.coverage()).detect(ground, road)
    for gt in world.gt_lanes():
        ln = _match(lanes, gt)
        assert ln is not None, f"차선을 못 찾음: {gt}"
        assert _curve_error(ln, gt) < 0.15, (gt, ln.coeffs)
        assert ln.style == gt["style"], (gt, ln.fill)
        assert ln.color == gt["color"]
        assert ln.side == gt["side"]
        assert ln.x_range[1] - ln.x_range[0] > 10.0


def test_left_is_positive_y():
    """왼쪽 차선(+Y)은 탑뷰에서 열 번호가 작은 쪽에 그려진다(좌우 뒤집힘 검사)."""
    world, grid, ipm, _, ground, road = _sim_ground("configs/two_cam.yaml")
    lanes = LaneDetector(grid, None, ipm.coverage()).detect(ground, road)
    left = next(l for l in lanes if l.side == "left")
    right = next(l for l in lanes if l.side == "right")
    assert left.offset > 0 > right.offset
    col_l = grid.to_pixel(np.array([[10.0, left.y_at(10.0)]]))[0, 0]
    col_r = grid.to_pixel(np.array([[10.0, right.y_at(10.0)]]))[0, 0]
    assert col_l < col_r


def test_road_mask_drops_lines_outside_road():
    """도로 밖(보도 위)에 그은 흰 선은 도로 마스크가 있으면 빠지고, 없으면 잡힌다.

    자차 가까운 띠(X -8~8m)에서 카메라가 보는 곳이어야 출발점이 잡히므로 보도 가운데(Y 7m)에 긋는다.
    """
    _, grid, ipm, _, ground, road = _sim_ground("configs/two_cam.yaml")
    g = ground.copy()
    y_out = 7.0  # 도로 가장자리(5.25m) 바깥, 보도(5.25~8.25m) 위
    px = grid.to_pixel(np.array([[-12.0, y_out], [20.0, y_out]])).round().astype(int)
    cv2.line(g, tuple(px[0]), tuple(px[1]), (240, 240, 240), max(2, int(0.15 / grid.res)))
    det = LaneDetector(grid, None, ipm.coverage())
    with_road = det.detect(g, road)
    without = det.detect(g, None)
    assert not any(abs(l.offset - y_out) < 0.5 for l in with_road)
    assert any(abs(l.offset - y_out) < 0.5 for l in without)


def test_no_lanes_on_blank_ground():
    grid = BevGrid()
    det = LaneDetector(grid)
    assert det.detect(np.full((grid.height, grid.width, 3), 70, np.uint8)) == []
    assert det.detect(None) == []


def test_lane_scale_invariance():
    """모형 축척(0.1)에서도 같은 차선을 같은 비율의 위치로 찾는다(길이 설정에 scale을 곱하는지)."""
    s = 0.1
    grid = BevGrid(-15 * s, 25 * s, -12 * s, 12 * s, 0.05 * s)
    img = np.full((grid.height, grid.width, 3), 70, np.uint8)
    for y in (1.75 * s, -1.75 * s):
        px = grid.to_pixel(np.array([[-14 * s, y], [24 * s, y]])).round().astype(int)
        cv2.line(img, tuple(px[0]), tuple(px[1]), (235, 235, 235), 3)
    lanes = LaneDetector(grid, None, None, scale=s).detect(img)
    offs = sorted(l.offset for l in lanes)
    assert len(offs) == 2 and np.allclose(offs, [-1.75 * s, 1.75 * s], atol=0.05 * s)


# ---------------------------------------------------------------------------
# 차선 평활
# ---------------------------------------------------------------------------
def _ln(c, style="dashed", color="white", a=0.0):
    return LaneCurve(np.array([a, 0.0, c]), (-10.0, 20.0), style=style, color=color, n_points=50)


def test_smoother_holds_and_votes():
    sm = LaneSmoother(LaneConfig(hold_frames=2, vote_frames=5, smooth_alpha=0.5))
    out = sm.update([_ln(1.75), _ln(-1.75)])
    assert {l.side for l in out} == {"left", "right"}
    # 한 프레임 놓쳐도 지난 곡선을 보여 준다
    out = sm.update([_ln(1.75)])
    assert len(out) == 2
    # 한 번 실선으로 잘못 나와도 다수결로 점선 유지
    out = sm.update([_ln(1.75, style="solid"), _ln(-1.75)])
    assert next(l for l in out if l.side == "left").style == "dashed"
    # 횡위치 튐은 반만 반영
    out = sm.update([_ln(2.05), _ln(-1.75)])
    assert abs(next(l for l in out if l.side == "left").offset - 1.90) < 1e-6
    # hold_frames보다 오래 못 찾으면 지운다
    for _ in range(3):
        out = sm.update([_ln(-1.75)])
    assert [l.side for l in out] == ["right"]


def test_assign_sides_far_lane_is_other():
    lanes = assign_sides([_ln(4.0), _ln(-1.6), _ln(-5.0)], ego_max=2.6)
    assert [l.side for l in lanes] == ["other", "right", "other"]


def test_lanes_smooth_over_sim_frames():
    """움직이는 합성 장면 20프레임에서 내 차선 양쪽이 매 프레임 있고 점선으로 유지된다(점선이 흘러가도)."""
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    world = SimWorld(cams, objects=[], curvature=0.004)
    ipm = IpmCompositor(cams, grid)
    det, sm = LaneDetector(grid, None, ipm.coverage()), LaneSmoother()
    for _ in range(20):
        world.step(0.1)
        frames = world.render()
        out = sm.update(det.detect(ipm.warp(frames), ipm.warp(world.gt_masks, interpolation=0)))
        sides = {l.side: l for l in out}
        assert "left" in sides and "right" in sides
        assert sides["left"].style == "dashed" and sides["right"].style == "dashed"


# ---------------------------------------------------------------------------
# 신호등 색
# ---------------------------------------------------------------------------
LIT = {"red": (40, 40, 245), "yellow": (0, 190, 255), "green": (170, 255, 20)}


def _light_img(state: str, orientation: str, lamp: int = 12) -> np.ndarray:
    """신호등 한 개 그림(어두운 상자에 등 세 개, 켜진 것 하나)."""
    n = 3 * lamp * 2 + 8
    w, h = (n, lamp * 2 + 8) if orientation == "horizontal" else (lamp * 2 + 8, n)
    img = np.full((h, w, 3), 25, np.uint8)
    for k, s in enumerate(("red", "yellow", "green")):
        c = (4 + lamp + 2 * lamp * k, 4 + lamp) if orientation == "horizontal" else (4 + lamp, 4 + lamp + 2 * lamp * k)
        cv2.circle(img, c, lamp - 2, LIT[s] if s == state else (45, 45, 45), -1)
    return img


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("state", ["red", "yellow", "green"])
def test_classify_synthetic(state, orientation):
    assert classify_light(_light_img(state, orientation)) == state


def test_classify_unknown_and_red_wrap():
    assert classify_light(np.full((40, 14, 3), 25, np.uint8)) == "unknown"  # 꺼짐
    assert classify_light(np.zeros((3, 3, 3), np.uint8)) == "unknown"  # 너무 작음
    assert classify_light(np.zeros((0, 0, 3), np.uint8)) == "unknown"
    # 색상 원의 반대쪽 빨강(H 약 175)도 빨강
    img = _light_img("red", "vertical")
    img[img[..., 2] == 245] = (80, 20, 245)
    assert classify_light(img) == "red"


def test_classify_position_breaks_tie():
    """빨강과 노랑 화소가 반반이면 켜진 등의 위치로 정한다(세로형 위 = 빨강, 가로형 오른쪽 = 초록 순서)."""
    img = _light_img("red", "vertical")
    top = img[: img.shape[0] // 3]
    m = top[..., 2] == 245
    ys, xs = np.nonzero(m)
    half = xs < np.median(xs)
    top[ys[half], xs[half]] = LIT["yellow"]  # 위 칸 등의 반을 노랑으로
    assert classify_light(img) == "red"


def test_orientation_and_crop():
    assert orientation_of(10, 30) == "vertical"
    assert orientation_of(30, 10) == "horizontal"
    assert orientation_of(10, 11) == "square"
    img = np.zeros((50, 60, 3), np.uint8)
    assert crop_box(img, (-5, -5, 10, 20)).shape[:2] == (20, 10)
    assert crop_box(img, (70, 0, 80, 10)).size == 0


def test_sim_light_color_accuracy():
    """합성 장면 신호등(세로형, 가로형, 색이 2초마다 바뀜)을 정답 상자로 잘라 판별한 정확도 95% 이상."""
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam_current.yaml"))
    world = SimWorld(cams, objects=[], traffic_lights=default_traffic_lights())
    ok = total = 0
    seen = set()
    for _ in range(60):
        world.step(0.1)
        frames = [world.render_camera(k, ground=True)[0] for k in range(len(cams))]
        for k, lights in enumerate(world.gt_lights):
            for box, state, ori in lights:
                total += 1
                ok += classify_light(crop_box(frames[k], box)) == state
                seen.add((state, ori))
    assert total >= 100
    assert ok / total >= 0.95
    assert len(seen) == 6  # 세 색 x 두 모양을 모두 거쳤다


def test_light_tracker_majority_vote():
    """깜빡임(모름, 한 번 다른 색)이 섞여도 다수결로 같은 색을 유지하고, 계속 바뀌면 따라간다."""
    tr = LightTracker(TrafficLightConfig(vote_frames=5, hold_frames=1))
    box = (100.0, 50.0, 110.0, 80.0)
    seq = ["red", "red", "unknown", "red", "green", "red", "green", "green", "green"]
    states = []
    for k, s in enumerate(seq):
        b = (box[0] + k % 2, box[1], box[2] + k % 2, box[3])  # 상자가 조금씩 흔들린다
        out = tr.update([TrafficLight(0, b, s, s, "vertical")])
        assert len(out) == 1 and out[0].tid == 1
        states.append(out[0].state)
    assert states[:6] == ["red"] * 6
    assert states[-1] == "green"
    # 다른 카메라의 같은 자리 상자는 다른 신호등이다
    out = tr.update([TrafficLight(1, box, "red", "red", "vertical")])
    assert {o.tid for o in out} == {1, 2}


def test_main_light_prefers_known_large():
    a = TrafficLight(0, (0, 0, 10, 30), "unknown", "unknown", "vertical")
    b = TrafficLight(0, (0, 0, 5, 15), "green", "green", "vertical")
    assert main_light([a, b]) is b
    assert main_light([a]) is a
    assert main_light([]) is None


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------
def test_dash_runs():
    pts = np.c_[np.linspace(0, 18, 73), np.zeros(73)]
    assert len(dash_runs(pts, False)) == 1
    runs = dash_runs(pts, True)
    assert len(runs) == 2 and all(r[:, 0].max() - r[:, 0].min() <= 3.0 for r in runs)


def test_render3d_draws_lane_where_projected():
    grid = BevGrid()
    cfg = View3DConfig(show_grid=False, show_rings=False, show_ego=False, show_ids=False, show_velocity=False)
    r = Renderer3D(grid, cfg)
    base = r.render([])
    lanes = assign_sides([_ln(1.75, style="solid"), _ln(-1.75, style="solid", color="yellow")], 2.6)
    out = r.render([], lanes=lanes)
    for ln, col in zip(lanes, [(235, 235, 235), (40, 200, 245)]):
        uv, _ = r.project_output(np.array([[10.0, ln.offset, 0.02]]))
        u, v = uv[0].round().astype(int)
        patch = out[v - 2:v + 3, u - 2:u + 3].reshape(-1, 3).astype(int)
        assert np.min(np.abs(patch - col).sum(axis=1)) < 60
    # 내 차선 가운데는 파랗게 칠해진다
    uv, _ = r.project_output(np.array([[10.0, 0.0, 0.0]]))
    u, v = uv[0].round().astype(int)
    assert int(out[v, u, 0]) > int(base[v, u, 0]) + 20
    # lanes를 안 주면 예전과 똑같다
    assert np.array_equal(r.render([]), base)


def test_render_lights_icon_and_pole():
    grid = BevGrid()
    r = Renderer3D(grid, View3DConfig(show_grid=False, show_rings=False, show_ego=False))
    base = r.render([])
    lt = TrafficLight(0, (0, 0, 10, 30), "red", "red", "vertical", xy=np.array([15.0, 2.0]))
    out = r.render([], lights=[lt])
    assert not np.array_equal(out, base)
    top_right = out[:120, -120:]
    assert np.any((top_right[..., 2] > 200) & (top_right[..., 1] < 100))  # 빨간 등이 켜진 아이콘
    bev = render_bev(grid, [], lanes=[_ln(1.75)], lights=[lt])
    assert np.any((bev[:120, :120, 2] > 200) & (bev[:120, :120, 1] < 100))


# ---------------------------------------------------------------------------
# 파이프라인과 시뮬레이션 옵션
# ---------------------------------------------------------------------------
def test_sim_defaults_unchanged():
    """곡률 0, 신호등 없음이면 영상과 검출 잡음이 예전 장면과 똑같다."""
    cams, _, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    a, b = SimWorld(cams), SimWorld(cams, curvature=0.0, traffic_lights=None)
    a.step(0.1), b.step(0.1)
    fa, fb = a.render(), b.render()
    assert all(np.array_equal(x, y) for x, y in zip(fa, fb))
    ra, rb = SimBackend(a).infer(fa), SimBackend(b).infer(fb)
    assert [[d.box for d in r.detections] for r in ra] == [[d.box for d in r.detections] for r in rb]
    assert b.gt_lights == []


def test_pipeline_lanes_and_lights():
    """옵션을 켜면 차선과 신호등이 나오고, 신호등은 추적(바닥 물체)에 들어가지 않는다."""
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    world = SimWorld(cams, curvature=0.004, traffic_lights=default_traffic_lights())
    pipe = RealityViewPipeline(cams, SimBackend(world), grid, view="both", lanes=LaneConfig(),
                               traffic_lights=TrafficLightConfig())
    states = []
    for _ in range(25):
        world.step(0.1)
        out = pipe.step(world.render(), 0.1)
        assert all(t.label != "traffic light" for t in out.tracks)
        states += [l.state for l in out.lights]
    assert {"left", "right"} <= {l.side for l in out.lanes}
    assert any(r.detections and any(d.label == "traffic light" for d in r.detections) for r in out.results)
    assert {"red", "green"} <= set(states)
    assert pipe.lane_light_stats["ego_both_frac"] > 0.9
    pipe.close()


def test_pipeline_defaults_off_and_aerial_filtered():
    """기본은 꺼져 있고(None), 신호등 상자가 와도 바닥 물체로 투영하지 않는다."""
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))

    class _Fake:
        def infer(self, frames):
            from realityview.types import FrameResult

            return [FrameResult([Detection("traffic light", 0.9, (300, 100, 310, 130))]) for _ in frames]

        def close(self):
            pass

    pipe = RealityViewPipeline(cams, _Fake(), grid)
    frames = [np.zeros((c.height, c.width, 3), np.uint8) for c in cams]
    for _ in range(3):
        out = pipe.step(frames, 0.1)
    assert out.lanes is None and out.lights is None and out.tracks == []
    assert pipe.lane_light_stats is None


def test_pipeline_lanes_with_no_ground_and_every():
    """--no-ground여도 차선용 바닥 영상은 합성하고, lanes_every 사이 프레임은 지난 결과를 쓴다."""
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    world = SimWorld(cams, objects=[])
    pipe = RealityViewPipeline(cams, SimBackend(world), grid, show_ground=False, view="3d",
                               lanes=LaneConfig(), lanes_every=3)
    for _ in range(6):
        world.step(0.1)
        out = pipe.step(world.render(), 0.1)
    assert pipe.lane_light_stats["lane_runs"] == 2
    assert {"left", "right"} <= {l.side for l in out.lanes}
    pipe.close()


def test_run_demo_options_default_off():
    spec = importlib.util.spec_from_file_location("run_demo_15", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args([])
    assert not args.lanes and not args.traffic_lights and not args.sim_lights
    assert args.sim_curvature == 0.0 and args.lanes_every == 1
    args = mod.build_parser().parse_args(["--lanes", "--traffic-lights", "--lanes-every", "2"])
    assert args.lanes and args.traffic_lights and args.lanes_every == 2


def test_sim_light_state_cycle():
    tl = SimTrafficLight(10, 0, 4, ("red", "green", "yellow"), 2.0)
    assert [tl.state_at(t) for t in (0.0, 1.9, 2.0, 4.5, 6.1)] == ["red", "red", "green", "yellow", "red"]


# ---------------------------------------------------------------------------
# 가림 처리(검출 상자 자리 빼기, 짧게 보인 차선)
# ---------------------------------------------------------------------------
def test_occluder_removes_lane_pixels():
    """검출 상자 자리를 편 탑뷰 마스크를 주면 그 자리 차선 화소는 빠진다."""
    from realityview.lanes import box_masks

    world, grid, ipm, frames, ground, road = _sim_ground("configs/two_cam.yaml")
    det = LaneDetector(grid, None, ipm.coverage())
    full = det.lane_mask(ground, road)
    # 앞 카메라 영상 전체를 상자로 덮는다(앞 바닥이 모두 가려진 것처럼)
    h, w = frames[0].shape[:2]
    dets = [[Detection("car", 0.9, (0, 0, w - 1, h - 1))], []]
    masks = box_masks([f.shape[:2] for f in frames], dets)
    occ = ipm.warp(masks, interpolation=0)
    part = det.lane_mask(ground, road, occ)
    near = (det.row_x > 3.0) & (det.row_x < 12.0)  # 앞 카메라 영상 아래쪽에 찍히는 가까운 바닥(점선 한 칸 이상)
    assert np.count_nonzero(full[near]) > 20
    assert np.count_nonzero(part[near]) < 0.2 * np.count_nonzero(full[near])
    back = det.row_x < -3.0  # 뒤 카메라 쪽은 그대로
    assert np.count_nonzero(part[back]) == np.count_nonzero(full[back])
    assert box_masks([f.shape[:2] for f in frames], [[], []]) is None
    # 신호등 상자는 바닥을 가리지 않으므로 빼지 않는다
    assert box_masks([(10, 10)], [[Detection("traffic light", 0.9, (0, 0, 5, 5))]], ("traffic light",)) is None


def test_short_visible_lane_does_not_vote_style():
    """보인 구간이 짧아 실선/점선을 못 정한 검출(unsure)은 다수결에 넣지 않고, 처음이면 실선으로 보여 준다."""
    sm = LaneSmoother(LaneConfig(vote_frames=5))
    out = sm.update([_ln(1.75, style="unsure")])
    assert out[0].style == "solid"
    sm.update([_ln(1.75, style="dashed")])
    for _ in range(4):
        out = sm.update([_ln(1.75, style="unsure")])
    assert out[0].style == "dashed"
