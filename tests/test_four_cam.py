import csv
import importlib.util
import os
import time
from argparse import Namespace

import cv2
import numpy as np
import pytest

from realityview.config import load_config
from realityview.multicam import (
    FrameSyncer,
    GtReplayBackend,
    QuadSplitSource,
    SyncedSources,
    UsbCameraSource,
    compensate_offsets,
    split_quad,
)
from realityview.pipeline import RealityViewPipeline
from realityview.sim import SimBackend, SimWorld, camera_timelines, per_camera
from realityview.sources import open_replay_sources, open_source
from realityview.tracker import Track
from realityview.types import GroundObject

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 가짜 장치
# ---------------------------------------------------------------------------
class FakeCapture:
    """cv2.VideoCapture 흉내. n장을 주고 끝난다. 각 영상의 [0,0,0] 값이 프레임 번호."""

    def __init__(self, n=5, size=(64, 48)):
        self.n, self.size, self.i = n, size, 0
        self.props = {}
        self.released = False

    def isOpened(self):
        return True

    def set(self, k, v):
        self.props[k] = v
        if k == cv2.CAP_PROP_POS_FRAMES:
            self.i = int(v)
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        f = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        f[0, 0, 0] = self.i
        self.i += 1
        return True, f

    def release(self):
        self.released = True


class FakeQuadInner:
    """2x2 합쳐진 영상. 칸 t의 값 = 10 x 프레임 번호 + t. 시각은 33ms 간격."""

    def __init__(self, n=10):
        self.n, self.i = n, 0
        self.last_timestamp_ns = None
        self.closed = False

    def read(self):
        if self.i >= self.n:
            return None
        f = np.zeros((40, 80, 3), np.uint8)
        for t, (y, x) in enumerate([(0, 0), (0, 40), (20, 0), (20, 40)]):
            f[y : y + 20, x : x + 40] = 10 * self.i + t
        self.last_timestamp_ns = 1_000_000_000 + self.i * 33_000_000
        self.i += 1
        return f

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# USB 카메라
# ---------------------------------------------------------------------------
def test_usb_source_stamps_monotonic_with_offset_and_resizes():
    cap = FakeCapture(n=3)
    t0 = time.monotonic_ns()
    s = UsbCameraSource(0, 32, 24, fps=30, timestamp_offset_ms=5.0, capture_factory=lambda d: cap, threaded=False)
    f = s.read()
    assert f.shape == (24, 32, 3)
    # 받은 시각 - 5ms. time.monotonic_ns()와 같은 시계라 지금 시각 근처
    assert t0 - 6_000_000 <= s.last_timestamp_ns <= time.monotonic_ns()
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 32 and cap.props[cv2.CAP_PROP_FPS] == 30
    s.read(), s.read()
    assert s.read() is None  # 장치가 끝나면 None
    s.close()
    assert cap.released


def test_usb_source_capture_size_is_requested_from_device_and_resized():
    cap = FakeCapture(n=2, size=(1280, 720))
    s = UsbCameraSource(0, 768, 432, fourcc="MJPG", capture_size=(1280, 720), capture_factory=lambda d: cap, threaded=False)
    f = s.read()
    # 장치에는 1280x720을 요청하고, 파이프라인에는 768x432로 넘긴다
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280 and cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert cap.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert f.shape == (432, 768, 3) and not s.ratio_mismatch
    s.close()


def test_usb_source_warns_when_device_ratio_differs(capsys):
    # 768x432를 요청했는데 장치가 4:3(640x480)으로 내는 경우: 줄이면 영상이 찌그러지므로 알린다
    cap = FakeCapture(n=2, size=(640, 480))
    s = UsbCameraSource(0, 768, 432, capture_factory=lambda d: cap, threaded=False)
    assert s.read().shape == (432, 768, 3)
    s.read()
    s.close()
    err = capsys.readouterr().err
    assert s.ratio_mismatch and err.count("비율이 달라") == 1  # 첫 영상에서 한 번만


def test_open_source_passes_capture_size(monkeypatch):
    import realityview.multicam as mc

    cap = FakeCapture(n=1, size=(1280, 720))
    monkeypatch.setattr(mc, "_default_capture", lambda d: cap)
    cam = [c for c in load_config(os.path.join(ROOT, "configs/five_cam.yaml"))[0] if c.name == "left"][0]
    s = open_source(cam)
    try:
        assert s.capture_size == (1280, 720) and s.size == (768, 432)
        assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    finally:
        s.close()


def test_usb_source_thread_gives_each_frame_once_and_ends():
    s = UsbCameraSource("fake.mp4", 64, 48, capture_factory=lambda d: FakeCapture(n=4), pace_fps=200)
    seen, stamps = [], []
    while True:
        f = s.read()
        if f is None:
            break
        seen.append(int(f[0, 0, 0]))
        stamps.append(s.last_timestamp_ns)
    s.close()
    assert seen == sorted(set(seen)) and seen[-1] == 3  # 같은 프레임을 두 번 주지 않음, 마지막까지 받음
    assert stamps == sorted(stamps)


def test_open_source_video_type_reads_file(tmp_path):
    p = str(tmp_path / "side.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for _ in range(3):
        vw.write(np.full((48, 64, 3), 100, np.uint8))
    vw.release()
    cam = load_config(os.path.join(ROOT, "configs/four_cam.yaml"))[0][2]
    cam.source = {"type": "video", "uri": p, "fps": 100}
    s = open_source(cam)
    try:
        for _ in range(5):  # 3장짜리 파일이지만 처음으로 돌아가 계속 준다
            f = s.read()
            assert f is not None and f.shape == (cam.height, cam.width, 3)
            assert s.last_timestamp_ns is not None
    finally:
        s.close()


# ---------------------------------------------------------------------------
# 쿼드 키트
# ---------------------------------------------------------------------------
def test_split_quad_order():
    f = FakeQuadInner().read()
    assert [int(t[0, 0, 0]) for t in split_quad(f)] == [0, 1, 2, 3]


def test_quad_align_delays_fast_tiles_to_match_late_tiles():
    hub = QuadSplitSource(FakeQuadInner(), align=True)
    views = [hub.view(t, lag) for t, lag in enumerate([0, 0, 1, 1])]
    vals, stamps = [], []
    for _ in range(3):
        vals.append([int(v.read()[0, 0, 0]) for v in views])
        stamps.append([v.last_timestamp_ns for v in views])
    # 세 번째 루프(합쳐진 프레임 2): 칸 0,1은 프레임 1에서, 칸 2,3은 프레임 2에서(=실제로 프레임 1 순간)
    assert vals[2] == [10, 11, 22, 23]
    assert len(set(stamps[2])) == 1 and stamps[2][0] == 1_000_000_000 + 33_000_000
    assert hub.seq == 3  # 루프마다 합쳐진 프레임을 한 번만 읽음


def test_quad_hub_turns_on_inner_sensor_timestamp():
    class PiLike(FakeQuadInner):
        want_timestamp = False  # Picamera2처럼 켜야만 센서 시각을 채움

        def read(self):
            f = super().read()
            if not self.want_timestamp:
                self.last_timestamp_ns = None
            return f

    hub = QuadSplitSource(PiLike())
    v = hub.view(0)
    v.read()
    assert v.last_timestamp_ns == 1_000_000_000


def test_quad_no_align_shifts_timestamps_only():
    hub = QuadSplitSource(FakeQuadInner(), align=False)
    views = [hub.view(t, lag) for t, lag in enumerate([0, 0, 1, 1])]
    for _ in range(2):
        vals = [int(v.read()[0, 0, 0]) for v in views]
    assert vals == [10, 11, 12, 13]
    st = [v.last_timestamp_ns for v in views]
    assert st[0] - st[2] == 33_000_000  # 늦은 칸은 한 프레임 간격만큼 앞당긴 시각
    with pytest.raises(ValueError):
        hub.view(0)  # 같은 칸을 두 카메라가 쓰면 오류


def test_open_source_quad_shares_one_device(tmp_path):
    p = str(tmp_path / "quad.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 15, (160, 96))
    for _ in range(3):
        vw.write(np.zeros((96, 160, 3), np.uint8))
    vw.release()
    cams = load_config(os.path.join(ROOT, "configs/four_cam.yaml"))[0]
    hubs = {}
    srcs = []
    for k, c in enumerate(cams):
        c.source = {"type": "quad", "group": "kit0", "tile": k, "lag_frames": 1 if k >= 2 else 0,
                    "inner": {"type": "video", "uri": p, "fps": 100, "width": 160, "height": 96}}
        srcs.append(open_source(c, hubs=hubs))
    assert list(hubs) == ["kit0"] and hubs["kit0"].lags == {0: 0, 1: 0, 2: 1, 3: 1}
    frames = [s.read() for s in srcs]
    assert all(f.shape == (432, 768, 3) for f in frames) and hubs["kit0"].seq == 1
    for s in srcs:
        s.close()


# ---------------------------------------------------------------------------
# 시간 맞춤
# ---------------------------------------------------------------------------
def _img(v):
    return np.full((4, 4, 3), v, np.uint8)


def test_frame_syncer_ok_hold_blank_and_csv(tmp_path):
    path = str(tmp_path / "sync.csv")
    sy = FrameSyncer([(4, 4)] * 3, ["a", "b", "c"], tolerance_ms=10, on_miss="hold", csv_path=path)
    ms = 1_000_000
    sy.push(0, _img(1), 100 * ms)
    sy.push(1, _img(2), 92 * ms)
    sy.push(1, _img(3), 104 * ms)  # 더 가까운 쪽(4ms)을 고른다
    b = sy.bundle(100 * ms)
    assert b.status == ["ok", "ok", "blank"]  # c는 아직 한 장도 없어 빈 칸
    assert int(b.frames[1][0, 0, 0]) == 3 and b.offsets_ms[1] == pytest.approx(4.0)
    assert b.frames[2].shape == (4, 4, 3) and b.frames[2].max() == 0
    sy.push(0, _img(4), 133 * ms)
    sy.push(2, _img(5), 160 * ms)  # 27ms 차이 > 10ms
    b = sy.bundle(133 * ms)
    # b는 이전 프레임 재사용(29ms 묵음), c는 오차 밖이지만 가진 것 중 가장 가까운 27ms 뒤 프레임
    assert b.status == ["ok", "hold", "hold"]
    assert b.offsets_ms[1] == pytest.approx(-29.0) and b.offsets_s[2] == pytest.approx(0.027)
    summ = sy.summary()
    assert summ["cameras"]["b"]["hold"] == 1 and summ["cameras"]["c"]["blank"] == 1 and summ["cameras"]["c"]["hold"] == 1
    sy.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and rows[1]["b_status"] == "hold"


def test_frame_syncer_hold_does_not_go_stale():
    # 받은 시각을 쓰는 카메라는 새 프레임이 기준보다 30ms 앞서 온다. 직전 프레임만 고집하면 계속 묵는다
    sy = FrameSyncer([(4, 4)] * 2, tolerance_ms=25, on_miss="hold")
    ms = 1_000_000
    for k in range(10):
        sy.push(0, _img(1), k * 66 * ms)
        sy.push(1, _img(2), (k * 66 + 30) * ms)
        b = sy.bundle(k * 66 * ms)
    assert abs(b.offsets_ms[1]) <= 36.0


def test_frame_syncer_blank_policy():
    sy = FrameSyncer([(4, 4)] * 2, tolerance_ms=5, on_miss="blank")
    sy.push(0, _img(1), 0)
    sy.push(1, _img(2), 0)
    sy.bundle(0)
    sy.push(0, _img(3), 50_000_000)
    b = sy.bundle(50_000_000)
    assert b.status == ["ok", "blank"] and b.stamps_ns[1] is None


class _Stream:
    def __init__(self, stamps):
        self.stamps, self.pos = stamps, 0
        self.recorded_timestamp_ns = None
        self.recorded_frame = None

    def next_ns(self):
        return self.stamps[self.pos] if self.pos < len(self.stamps) else None

    def read(self):
        if self.pos >= len(self.stamps):
            return None
        self.recorded_timestamp_ns, self.recorded_frame = self.stamps[self.pos], self.pos
        self.pos += 1
        return _img(self.pos)


def test_synced_sources_pairs_by_time_with_drops():
    ms = 1_000_000
    ref = _Stream([0, 100 * ms, 200 * ms, 300 * ms])
    other = _Stream([20 * ms, 120 * ms, 320 * ms])  # 220ms 프레임이 빠짐
    ss = SyncedSources([ref, other], FrameSyncer([(4, 4)] * 2, tolerance_ms=25))
    metas = []
    while ss.read() is not None:
        metas.append((ss.last.meta[1], ss.last.status[1]))
    assert metas == [(0, "ok"), (1, "ok"), (1, "hold"), (2, "ok")]


def test_compensate_offsets_moves_by_track_velocity():
    tr = Track(1, "car", np.array([10.0, 0.0]), vel=np.array([20.0, 0.0]), hits=5)
    o = GroundObject("car", 0.9, np.array([11.5, 0.0]), (4.5, 1.9), cam_index=1)
    far = GroundObject("person", 0.9, np.array([-30.0, 5.0]), (0.5, 0.5), cam_index=1)
    compensate_offsets([o, far], [tr], [0.0, 0.05], dt=0.025, gate=2.5)
    # 카메라 1은 기준보다 50ms 늦게 찍었다. 기준 시각 위치 = 11.5 - 20 x 0.05 = 10.5
    assert o.xy[0] == pytest.approx(10.5)
    assert far.xy[0] == -30.0  # 맞는 트랙이 없으면 그대로


def test_pipeline_offsets_ignored_unless_skew_comp():
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/four_cam.yaml"))
    outs = []
    for comp, offs in ((False, None), (False, [0, 0, 0.05, 0.05]), (True, [0, 0, 0, 0])):
        w = SimWorld(cams)
        pipe = RealityViewPipeline(cams, SimBackend(w, seed=3), grid, show_ground=False, show_cameras=False,
                                   tracker="robust", skew_comp=comp)
        for _ in range(6):
            w.step(0.1)
            out = pipe.step(w.render(), 0.1, offs)
        pipe.close()
        outs.append(sorted((t.tid, round(float(t.xy[0]), 6)) for t in out.tracks))
    assert outs[0] == outs[1] == outs[2]


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------
def test_sim_render_camera_matches_render_and_set_time_wraps():
    cams = load_config(os.path.join(ROOT, "configs/four_cam.yaml"))[0]
    w = SimWorld(cams)
    frames = w.render()
    img, dets, ids, road = w.render_camera(2)
    assert np.array_equal(img, frames[2]) and ids == w.gt_ids[2] and np.array_equal(road, w.gt_masks[2])
    _, dets_ng, _, _ = w.render_camera(2, ground=False)
    assert [d.box for d in dets_ng] == [d.box for d in dets]
    w.set_time(10.0)  # 사람(vx -12)은 9 - 120 = -111 m, 80 m씩 돌려 -31 m
    assert w.objects[5].x == pytest.approx(-31.0) and w.wraps[5] == 1
    w.set_time(0.0)
    assert w.objects[5].x == pytest.approx(9.0) and w.wraps[5] == 0


def test_camera_timelines_skew_and_drop():
    tl = camera_timelines(2, 100, 10.0, [0, 50], [0, 0.2], [0, 0], seed=1)
    assert len(tl[0]) == 100 and 60 < len(tl[1]) < 95
    n, t = tl[1][0]
    assert t == pytest.approx(n / 10.0 + 0.05)
    assert per_camera("0,0,20,50", 4) == [0, 0, 20, 50] and per_camera("5", 3) == [5, 5, 5]
    with pytest.raises(ValueError):
        per_camera("1,2", 4)


def test_make_sim4_replay_with_gt_backend(tmp_path):
    mk = _load_script("make_sim4")
    out = str(tmp_path / "sim4")
    args = mk.build_parser().parse_args(["--out", out, "--frames", "8", "--drop", "0,0,0,0.3", "--seed", "2"])
    info = mk.make(args)
    assert info["frames_per_camera"][:3] == [8, 8, 8]
    cams = load_config(os.path.join(ROOT, "configs/four_cam.yaml"))[0]
    srcs = open_replay_sources(out, cams)
    assert all(hasattr(s, "next_ns") for s in srcs)
    ss = SyncedSources(srcs, FrameSyncer([(c.width, c.height) for c in cams], tolerance_ms=25))
    be = GtReplayBackend(os.path.join(out, "gt.json"), noise_px=0.0)
    n = 0
    while ss.read() is not None:
        be.set_frame_meta(ss.last.meta)
        res = be.infer(ss.last.frames)
        assert len(res) == 4
        n += 1
    assert n == 8
    assert ss.syncer.summary()["cameras"]["cam3"]["abs_offset_ms_max"] is not None
    for s in srcs:
        s.close()


def test_eval_tracking_sync_mode_runs():
    ev = _load_script("eval_tracking")
    m, summ = ev.run_sim_sync(os.path.join(ROOT, "configs/four_cam.yaml"), 15.0, 20, 0, "robust", 0,
                              [0, 0, 50, 50], [0, 0, 0, 0], [0, 0, 0, 0], 25.0, "hold", True)
    assert m.gt_frames > 0 and summ["bundles"] == 20
    # 50ms 어긋남은 15fps에서 한 프레임 앞 프레임과 묶여 66.7 - 50 = 16.7ms만 남는다
    assert summ["cameras"]["left"]["abs_offset_ms_max"] == pytest.approx(1000 / 15 - 50, abs=0.01)


def test_run_demo_sync_options_default_off():
    rd = _load_script("run_demo")
    a = rd.build_parser().parse_args([])
    assert a.sync is False and a.skew_comp is False and a.sync_miss == "hold"
    assert isinstance(a, Namespace)
