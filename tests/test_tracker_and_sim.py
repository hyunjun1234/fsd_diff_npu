import numpy as np

from realityview.config import load_config
from realityview.detectors import letterbox, parse_hailo_nms, seg_to_road_mask
from realityview.pipeline import RealityViewPipeline
from realityview.sim import SimBackend, SimWorld
from realityview.tracker import BevTracker
from realityview.types import GroundObject


def test_tracker_keeps_id_and_estimates_velocity():
    tr = BevTracker()
    ids = set()
    for k in range(20):
        o = GroundObject("car", 0.9, np.array([10.0 + 0.5 * k, 3.5]), (4.5, 1.9), 0)
        tracks = tr.update([o], dt=0.1)
        ids |= {t.tid for t in tracks}
    assert ids == {1}
    assert abs(tracks[0].vel[0] - 5.0) < 0.5


def test_tracker_drops_lost_tracks():
    tr = BevTracker(max_misses=2)
    o = GroundObject("car", 0.9, np.array([10.0, 0.0]), (4.5, 1.9), 0)
    tr.update([o], 0.1)
    tr.update([o], 0.1)
    for _ in range(4):
        tr.update([], 0.1)
    assert tr.tracks == []


def test_parse_hailo_nms_undoes_letterbox():
    img = np.zeros((432, 768, 3), np.uint8)
    lb, s, pad = letterbox(img, (640, 640))
    # 원본 (100,200)-(300,400) 박스를 레터박스 좌표의 정규화 값으로
    x1, y1, x2, y2 = 100 * s + pad[0], 200 * s + pad[1], 300 * s + pad[0], 400 * s + pad[1]
    nms = [np.zeros((0, 5)) for _ in range(80)]
    nms[2] = np.array([[y1 / 640, x1 / 640, y2 / 640, x2 / 640, 0.8]])
    nms[4] = np.array([[0.1, 0.1, 0.2, 0.2, 0.9]])  # airplane → 버림
    dets = parse_hailo_nms(nms, (640, 640), s, pad, 0.4)
    assert len(dets) == 1 and dets[0].label == "car"
    np.testing.assert_allclose(dets[0].box, (100, 200, 300, 400), atol=1e-6)


def test_seg_to_road_mask():
    logits = np.zeros((4, 8, 19), np.float32)
    logits[:2, :, 10] = 5.0  # 위 절반은 sky(10)
    logits[2:, :, 0] = 5.0  # 아래 절반은 road(0)
    m = seg_to_road_mask(logits, (8, 16), [0])
    assert m.shape == (8, 16) and m[6].all() and not m[1].any()


def test_sim_pipeline_end_to_end_positions():
    cams, grid, _ = load_config("configs/two_cam.yaml")
    world = SimWorld(cams)
    pipe = RealityViewPipeline(cams, SimBackend(world, noise_px=1.0, miss_rate=0.0), grid)
    for _ in range(15):
        world.step(0.1)
        frames = world.render()
        out = pipe.step(frames, 0.1)
    assert out.dashboard.ndim == 3 and out.bev.shape[:2] == (grid.height, grid.width)
    # 15m 이내 차량은 추적 위치가 정답과 1m 이내여야 한다
    near = [o for o in world.objects if o.label == "car" and np.hypot(o.x, o.y) < 15]
    assert near
    for o in near:
        d = min(np.hypot(t.xy[0] - o.x, t.xy[1] - o.y) for t in out.tracks)
        assert d < 1.0, (o, d)
