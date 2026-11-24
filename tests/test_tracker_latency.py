import importlib.util
import os
import sys
import types

import cv2
import numpy as np
import pytest

from realityview.config import load_config
from realityview.sim import SimBackend, SimWorld
from realityview.sources import ReplaySource, picamera_config_kwargs, replay_files
from realityview.tracker import BevTracker
from realityview.types import GroundObject

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_11", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 추적기 ID 쪼개짐 재현 (3D 시뮬레이션 영상과 같은 설정: two_cam.yaml, 15 fps, 잡음 기본값, seed 0)
# ---------------------------------------------------------------------------
def _car1_tracks(tracker: str, frames: int = 170) -> tuple[int, int]:
    """오른쪽 차선 앞차(정답 물체 1번)에 붙은 트랙 ID 수와 전체 ID 바뀜 횟수."""
    ev = _load_script("eval_tracking")
    m = ev.run_sim(os.path.join(ROOT, "configs/two_cam.yaml"), 15.0, frames, 0, tracker)
    car1 = [ids for key, ids in m._ids.items() if key[0] == 1]
    return max(len(s) for s in car1), m.switches


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sim_far_car_split_basic_vs_robust():
    n_basic, sw_basic = _car1_tracks("basic")
    n_robust, sw_robust = _car1_tracks("robust")
    # 기존 추적기는 먼 앞차 하나에 트랙이 여럿 생긴다(영상의 ID 32, 35, 37 ...)
    assert n_basic >= 6
    # 고친 추적기는 몇 개 안쪽으로 줄고 전체 ID 바뀜도 크게 준다
    assert n_robust <= 3
    assert sw_robust * 3 < sw_basic


def test_robust_keeps_side_by_side_cars_apart():
    # 같은 속도로 나란히 가는 두 차(y=0, y=3.5). 시선 방향으로만 크게 튀는 잡음을 넣어도 ID가 바뀌면 안 된다
    rng = np.random.default_rng(1)
    tr = BevTracker(mode="robust")
    first = None
    for k in range(60):
        objs = []
        for y in (0.0, 3.5):
            p = np.array([20.0 + 0.1 * k, y])
            u = p / np.linalg.norm(p)
            objs.append(GroundObject("car", 0.9, p + u * rng.normal(0, 1.5), (4.5, 1.9), 0))
        tracks = tr.update(objs, 1 / 15)
        if len(tracks) == 2:
            now = {t.tid: t.xy[1] > 1.75 for t in tracks}
            if first is None:
                first = now
            assert now == first


def test_robust_outlier_does_not_blow_up_velocity():
    tr = BevTracker(mode="robust")
    for k in range(10):
        tr.update([GroundObject("car", 0.9, np.array([25.0, 3.5]), (4.5, 1.9), 0)], 1 / 15)
    # 시선 방향으로 3m 튄 측정 한 번
    tr.update([GroundObject("car", 0.9, np.array([28.0, 3.5]), (4.5, 1.9), 0)], 1 / 15)
    assert len(tr.tracks) == 1
    assert np.linalg.norm(tr.tracks[0].vel) < 6.0  # basic이면 약 22m/s
    basic = BevTracker()
    for k in range(10):
        basic.update([GroundObject("car", 0.9, np.array([25.0, 3.5]), (4.5, 1.9), 0)], 1 / 15)
    basic.update([GroundObject("car", 0.9, np.array([28.0, 3.5]), (4.5, 1.9), 0)], 1 / 15)
    assert len(basic.tracks) == 2  # 기존 동작: 고정 2.5m 밖이라 새 트랙이 생긴다


def test_robust_does_not_spawn_duplicate_next_to_track():
    tr = BevTracker(mode="robust")
    for _ in range(3):
        tr.update([GroundObject("car", 0.9, np.array([10.0, 0.0]), (4.5, 1.9), 0)], 0.1)
    # 같은 차의 측정이 두 개(병합에서 놓친 겹침) 들어와도 트랙은 하나
    objs = [GroundObject("car", 0.9, np.array([10.0, 0.0]), (4.5, 1.9), 0),
            GroundObject("car", 0.8, np.array([11.2, 0.3]), (4.5, 1.9), 1)]
    tr.update(objs, 0.1)
    assert len(tr.tracks) == 1


def test_robust_scales_with_model_rig():
    tr = BevTracker(mode="robust", scale=0.1)
    assert tr.gate == pytest.approx(0.25) and tr.max_speed == pytest.approx(6.0)


def test_tracker_mode_checked():
    with pytest.raises(ValueError):
        BevTracker(mode="fancy")


# ---------------------------------------------------------------------------
# 평가 지표
# ---------------------------------------------------------------------------
class _T:
    def __init__(self, tid, xy, label="car"):
        self.tid, self.xy, self.label = tid, np.array(xy, float), label


def test_tracking_metrics_counts():
    ev = _load_script("eval_tracking")
    m = ev.TrackingMetrics()
    g = [((0, 0), "car", np.array([10.0, 0.0])), ((1, 0), "car", np.array([10.0, 3.5]))]
    m.add(0, g, [_T(1, [10, 0]), _T(2, [10, 3.5])])
    m.add(1, g, [_T(1, [10, 0]), _T(3, [10.2, 3.5]), _T(4, [10.5, 3.6])])  # 1번 물체: ID 바뀜 + 중복
    m.add(2, g, [_T(1, [10, 0]), _T(9, [-10, 0])])  # 1번 물체 놓침, 헛트랙 하나
    s = m.summary()
    assert s["id_switches"] == 1 and s["dup_frames"] == 1 and s["false_tracks"] == 1
    assert s["miss_ratio"] == pytest.approx(1 / 6, abs=1e-3) and s["tracks_per_object_max"] == 2


def test_match_radius_grows_with_range():
    ev = _load_script("eval_tracking")
    assert ev.match_radius(np.array([5.0, 0.0])) == 3.0
    assert ev.match_radius(np.array([30.0, 0.0])) == pytest.approx(6.0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sim_gt_ids_follow_gt_and_keep_noise():
    cams, _, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    w = SimWorld(cams)
    w.step(0.1)
    w.render()
    assert [len(a) for a in w.gt_ids] == [len(a) for a in w.gt]
    for dets, ids in zip(w.gt, w.gt_ids):
        assert all(d.label == w.objects[i].label for d, i in zip(dets, ids))
    # 정답 ID를 더해도 검출 잡음 난수는 그대로(같은 seed면 같은 출력)
    a = SimBackend(w, seed=0).infer(None)
    b = SimBackend(w, seed=0).infer(None)
    assert [d.box for r in a for d in r.detections] == [d.box for r in b for d in r.detections]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sim_wrap_counter():
    cams, _, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    w = SimWorld(cams)
    w.objects[0].x = 44.99
    w.objects[0].vx = 1.0
    w.step(0.1)
    assert w.wraps[0] == 1 and w.objects[0].x == -35


# ---------------------------------------------------------------------------
# 카메라 지연 옵션 (가짜 picamera2)
# ---------------------------------------------------------------------------
def test_picamera_kwargs_default_unchanged():
    kw = picamera_config_kwargs(768, 432, 30)
    assert "buffer_count" not in kw and "queue" not in kw
    kw = picamera_config_kwargs(768, 432, 30, buffer_count=3, queue=False)
    assert kw["buffer_count"] == 3 and kw["queue"] is False


class _Req:
    def __init__(self, cam):
        self.cam = cam

    def make_array(self, name):
        return np.zeros((432, 768, 3), np.uint8)

    def get_metadata(self):
        return {"SensorTimestamp": 5_000_000}

    def release(self):
        self.cam.released += 1


class _Cam:
    def __init__(self, index):
        self.released = 0
        self.flush_args = []
        self.cfg = None

    def create_video_configuration(self, **kw):
        self.cfg = dict(kw)
        return dict(kw)

    def configure(self, cfg):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def capture_array(self, name):
        return np.zeros((432, 768, 3), np.uint8)

    def capture_request(self, flush=None):
        self.flush_args.append(flush)
        return _Req(self)


def test_picamera_flush_and_buffers(monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=_Cam))
    from realityview.sources import PiCameraSource

    src = PiCameraSource(0, 768, 432, fps=24, buffer_count=3, queue=False, flush=True)
    assert src.cam.cfg["buffer_count"] == 3 and src.cam.cfg["queue"] is False
    assert src.cam.cfg["controls"]["FrameRate"] == 24.0
    src.read()  # want_timestamp 없이도 flush면 요청 경로를 쓰고, 타임스탬프는 그 프레임 것
    assert src.cam.flush_args == [True] and src.cam.released == 1 and src.last_timestamp_ns == 5_000_000


def test_run_demo_camera_options_default_off():
    mod = _load_script("run_demo")
    a = mod.build_parser().parse_args([])
    assert (a.tracker, a.replay, a.cam_fps, a.cam_buffers, a.cam_queue, a.cam_flush) == (
        "basic", None, None, None, None, False)
    assert mod.camera_override(a) == {}
    b = mod.build_parser().parse_args(["--cam-fps", "24", "--cam-buffers", "3", "--cam-queue", "off", "--cam-flush"])
    assert mod.camera_override(b) == {"fps": 24.0, "buffer_count": 3, "queue": False, "flush": True}


# ---------------------------------------------------------------------------
# 저장 이미지 재생
# ---------------------------------------------------------------------------
def test_replay_source_reads_loops_and_resizes(tmp_path):
    for k in range(2):
        cv2.imwrite(str(tmp_path / f"front_{k:05d}.png"), np.full((100, 200, 3), k * 50, np.uint8))
    cv2.imwrite(str(tmp_path / "rear_00000.jpg"), np.zeros((50, 60, 3), np.uint8))
    (tmp_path / "front_notes.txt").write_text("x")
    assert len(replay_files(str(tmp_path), "front")) == 2
    src = ReplaySource(str(tmp_path), "front", 64, 48)
    vals = [int(src.read()[0, 0, 0]) for _ in range(3)]
    assert vals == [0, 50, 0]
    assert src.read().shape == (48, 64, 3) and src.last_timestamp_ns is None
    once = ReplaySource(str(tmp_path), "rear", 64, 48, loop=False)
    assert once.read() is not None and once.read() is None
    with pytest.raises(FileNotFoundError):
        ReplaySource(str(tmp_path), "left", 64, 48)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_demo_replay_with_sim_free_backend(tmp_path, monkeypatch):
    # --replay는 카메라를 열지 않는다. 검출기 대신 아무것도 못 찾는 가짜 백엔드로 끝까지 도는지만 본다
    mod = _load_script("run_demo")
    monkeypatch.chdir(ROOT)
    for name in ("front", "rear"):
        cv2.imwrite(str(tmp_path / f"{name}_00000.png"), np.zeros((432, 768, 3), np.uint8))

    class _NoDet:
        def __init__(self, *a, **k):
            pass

        def infer(self, frames):
            from realityview.types import FrameResult

            return [FrameResult() for _ in frames]

        def close(self):
            pass

    import realityview.detectors as det

    monkeypatch.setattr(det, "UltralyticsBackend", _NoDet)
    snap = tmp_path / "snap.png"
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--config", "configs/two_cam.yaml", "--backend", "ultralytics",
                                      "--replay", str(tmp_path), "--frames", "3", "--headless", "--view", "3d",
                                      "--tracker", "robust", "--snapshot", str(snap)])
    mod.main()
    assert snap.exists()


def test_robust_merges_overlapping_tracks_but_not_lanes():
    # 두 카메라가 같은 차를 1.2m 떨어진 두 자리로 계속 투영하는 경우: 트랙은 하나로 정리된다
    tr = BevTracker(mode="robust")
    tr.tracks = []
    from realityview.tracker import Track

    tr.tracks = [Track(1, "car", np.array([12.0, 0.0]), hits=10), Track(2, "car", np.array([13.2, 0.2]), hits=3),
                 Track(3, "car", np.array([12.0, 3.5]), hits=10)]
    tr._next_id = 4
    objs = [GroundObject("car", 0.9, np.array([12.0, 0.0]), (4.5, 1.9), 0),
            GroundObject("car", 0.9, np.array([13.2, 0.2]), (4.5, 1.9), 2),
            GroundObject("car", 0.9, np.array([12.0, 3.5]), (4.5, 1.9), 0)]
    tr.update(objs, 0.1)
    assert sorted(t.tid for t in tr.tracks) == [1, 3]  # 오래된 1번이 남고 옆 차선 3번은 그대로
