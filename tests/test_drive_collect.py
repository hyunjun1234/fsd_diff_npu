import json
import os
import time

import numpy as np
import pytest

from realityview.camera_model import camera_from_dict
from realityview.drive_rec import DriveRecorder, list_drives, segment_of
from realityview.mjpeg import (
    MjpegReplaySource,
    MjpegWriter,
    frames_missing,
    nearest_indices,
    read_index,
    synced_refs,
)
from realityview.sources import ReplaySource, open_replay_sources


# ---------------------------------------------------------------------------
# 시험용 카메라
# ---------------------------------------------------------------------------
def make_cam(name, w=64, h=48, fps=50):
    return camera_from_dict(
        {
            "name": name,
            "width": w,
            "height": h,
            "intrinsics": {"fx": 50.0, "fy": 50.0, "cx": w / 2, "cy": h / 2},
            "extrinsics": {"position": [0.0, 0.0, 1.0], "yaw": 0.0, "pitch": 5.0},
            "source": {"type": "opencv", "fps": fps},
        }
    )


class FakeCam:
    """일정 간격(주기)으로 영상을 내는 가짜 카메라. skip_every 번째마다 한 프레임을 건너뛴 시각을 준다."""

    def __init__(self, w, h, period_s=0.02, skip_every=0, value=0):
        self.w, self.h, self.period = w, h, period_s
        self.skip_every = skip_every
        self.n = 0
        self.value = value
        self.want_timestamp = False
        self.last_timestamp_ns = None
        self.t = time.monotonic_ns()
        self.closed = False

    def read(self):
        time.sleep(self.period)
        step = 2 if (self.skip_every and self.n and self.n % self.skip_every == 0) else 1
        self.t += int(step * self.period * 1e9)
        self.n += 1
        self.last_timestamp_ns = self.t if self.want_timestamp else None
        img = np.full((self.h, self.w, 3), (self.n * 7 + self.value) % 256, np.uint8)
        return img

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# 순수 함수
# ---------------------------------------------------------------------------
def test_frames_missing_and_segment_of():
    p = 1e9 / 30
    assert frames_missing(None, 100, p) == 0
    assert frames_missing(0, int(p), p) == 0
    assert frames_missing(0, int(3 * p), p) == 2
    assert frames_missing(0, int(1.4 * p), p) == 0
    assert segment_of(0, 0, 60) == 0
    assert segment_of(int(59.9e9), 0, 60) == 0
    assert segment_of(int(60e9), 0, 60) == 1
    assert segment_of(-5, 0, 60) == 0


def test_nearest_indices():
    ref = np.array([0, 10, 20, 30])
    other = np.array([1, 12, 14, 29, 40])
    assert nearest_indices(ref, other).tolist() == [0, 1, 2, 3]
    assert nearest_indices(ref, np.array([5])).tolist() == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# mjpeg 쓰기와 읽기
# ---------------------------------------------------------------------------
def test_mjpeg_roundtrip_and_truncated_tail(tmp_path):
    w = MjpegWriter(str(tmp_path), "front", quality=90)
    imgs = [np.full((48, 64, 3), v, np.uint8) for v in (10, 120, 240)]
    for i, im in enumerate(imgs):
        w.write(im, i, 1000 + i, 0.0)
    st = w.close()
    assert st["frames"] == 3 and st["bytes"] > 0
    refs = read_index(str(tmp_path), "front")
    assert [r.sensor_ns for r in refs] == [1000, 1001, 1002]
    src = MjpegReplaySource(refs, 64, 48, loop=False)
    for im in imgs:
        got = src.read()
        assert abs(int(got.mean()) - int(im.mean())) <= 2
    assert src.read() is None
    assert src.last_timestamp_ns is None and src.recorded_timestamp_ns == 1002
    src.close()
    # 전원이 꺼진 것처럼 영상 끝을 잘라 내면 마지막 프레임은 버린다
    path = os.path.join(str(tmp_path), "front.mjpeg")
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 10)
    assert len(read_index(str(tmp_path), "front")) == 2


def test_synced_refs_pairs_by_time(tmp_path):
    a = MjpegWriter(str(tmp_path), "front")
    b = MjpegWriter(str(tmp_path), "rear")
    img = np.zeros((48, 64, 3), np.uint8)
    for i in range(10):
        a.write(img, i, i * 100, 0.0)
    for i in range(7):  # 뒤 카메라는 더 드물게
        b.write(img, i, i * 150 + 3, 0.0)
    a.close()
    b.close()
    refs, skew = synced_refs(str(tmp_path), ["front", "rear"])
    assert len(refs[0]) == len(refs[1]) == 10
    for ra, rb in zip(*refs):
        assert abs(ra.sensor_ns - rb.sensor_ns) <= 75
    assert skew <= 75 / 1e6


# ---------------------------------------------------------------------------
# 주행 녹화기
# ---------------------------------------------------------------------------
def _record(tmp_path, cams, srcs, **kw):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cameras: []\n")
    logs = []
    rec = DriveRecorder(
        str(tmp_path / "drives"), cams, srcs, config_path=str(cfg), name="d1", log=logs.append,
        health_fn=lambda: {"soc_temp_c": 50.0, "throttled": "0x0"}, **kw,
    )
    rec.start()
    rec.run_until_stopped(poll_s=0.02, disk_check_s=0.05)
    meta = rec.close()
    return rec, meta, logs


def test_drive_recorder_segments_and_replay(tmp_path):
    # 설정의 fps와 가짜 카메라 주기를 맞춘다(앞 100fps = 10ms, 뒤 80fps = 12.5ms)
    cams = [make_cam("front", fps=100), make_cam("rear", 32, 24, fps=80)]
    srcs = [FakeCam(64, 48, 0.01, skip_every=10), FakeCam(32, 24, 0.0125, value=100)]
    rec, meta, logs = _record(tmp_path, cams, srcs, segment_s=0.25, duration_s=0.8, status_interval=0.3, min_free_gb=0.0)
    d = rec.drive_dir
    assert meta["stop_reason"] == "duration"
    assert os.path.exists(os.path.join(d, "config.yaml"))
    with open(os.path.join(d, "drive.json")) as f:
        dj = json.load(f)
    assert dj["config_text"] == "cameras: []\n"
    assert [c["name"] for c in dj["cameras"]] == ["front", "rear"]
    segs = sorted(s for s in os.listdir(d) if s.startswith("seg_"))
    assert len(segs) >= 3
    # 모든 세그먼트가 깨끗하게 닫혔고(segment.json), 프레임 수 합이 전체와 같다
    total = {"front": 0, "rear": 0}
    for s in segs:
        with open(os.path.join(d, s, "segment.json")) as f:
            sj = json.load(f)
        assert sj["closed_cleanly"]
        for c in sj["cameras"]:
            total[c["camera"]] += c["frames"]
            assert len(read_index(os.path.join(d, s), c["camera"])) == c["frames"]
    tot = {t["camera"]: t for t in meta["totals"]}
    assert total == {"front": tot["front"]["frames"], "rear": tot["rear"]["frames"]}
    # 10장마다 한 장 건너뛴 앞 카메라의 누락이 잡힌다
    assert tot["front"]["gaps"] >= 1 and tot["rear"]["gaps"] == 0
    # 상태 기록: 시작 줄 + 주기 기록 + 끝 줄
    with open(os.path.join(d, "status.csv")) as f:
        lines = f.read().strip().splitlines()
    assert lines[0].startswith("elapsed_s") and "front_gaps_interval" in lines[0]
    assert len(lines) >= 3 and logs
    # 회차 폴더와 세그먼트 폴더 둘 다 --replay 로 넣을 수 있다
    for folder in (d, os.path.join(d, segs[0])):
        rs = open_replay_sources(folder, cams)
        f0, f1 = rs[0].read(), rs[1].read()
        assert f0.shape == (48, 64, 3) and f1.shape == (24, 32, 3)
        for r in rs:
            r.close()
    # 회차 목록
    rows = list_drives(str(tmp_path / "drives"))
    assert rows[0]["drive"] == "d1" and rows[0]["segments_clean"] == len(segs) and rows[0]["bytes"] > 0


def test_drive_recorder_stop_signal_and_disk_guard(tmp_path):
    cams = [make_cam("front")]
    # 디스크가 모자라면 시작하자마자 한 장도 쓰지 않고 멈춘다
    rec, meta, _ = _record(tmp_path, cams, [FakeCam(64, 48)], min_free_gb=10.0, free_fn=lambda: 1.0)
    assert meta["stop_reason"].startswith("disk_low")
    assert not [s for s in os.listdir(rec.drive_dir) if s.startswith("seg_")]

    # 녹화 도중 공간이 줄면 멈추고 세그먼트를 닫는다
    free = {"v": 100.0}
    cams2 = [make_cam("front")]
    rec2 = DriveRecorder(str(tmp_path / "drives2"), cams2, [FakeCam(64, 48)], name="d2", min_free_gb=5.0,
                         free_fn=lambda: free["v"], status_interval=0, log=lambda s: None,
                         health_fn=lambda: {})
    rec2.start()
    time.sleep(0.1)
    free["v"] = 1.0
    rec2.run_until_stopped(poll_s=0.02, disk_check_s=0.02)
    meta2 = rec2.close()
    assert meta2["stop_reason"].startswith("disk_low")
    assert os.path.exists(os.path.join(rec2.drive_dir, "seg_0000", "segment.json"))

    # 시그널 처리기가 부르는 request_stop: 이유가 남고 세그먼트가 닫힌다
    rec3 = DriveRecorder(str(tmp_path / "drives3"), cams2, [FakeCam(64, 48)], name="d3", min_free_gb=0,
                         status_interval=0, log=lambda s: None, health_fn=lambda: {})
    rec3.start()
    time.sleep(0.1)
    rec3.request_stop("signal:SIGTERM")
    rec3.run_until_stopped(poll_s=0.02)
    meta3 = rec3.close()
    assert meta3["stop_reason"] == "signal:SIGTERM"
    with open(os.path.join(rec3.drive_dir, "seg_0000", "segment.json")) as f:
        assert json.load(f)["cameras"][0]["frames"] >= 1


def test_open_replay_sources_keeps_image_folders(tmp_path):
    import cv2

    cv2.imwrite(str(tmp_path / "front_00000.png"), np.zeros((48, 64, 3), np.uint8))
    rs = open_replay_sources(str(tmp_path), [make_cam("front")])
    assert isinstance(rs[0], ReplaySource)
    with pytest.raises(FileNotFoundError):
        open_replay_sources(str(tmp_path), [make_cam("rear")])


# ---------------------------------------------------------------------------
# 회차 모델
# ---------------------------------------------------------------------------
from realityview.config import load_config  # noqa: E402
from realityview.model_rounds import primary_det_model, resolve_round_model, round_file_name, shadow_det_model  # noqa: E402


def test_round_models():
    assert round_file_name("yolov8s", 1, "hef") == "yolov8s_r001.hef"
    assert round_file_name("yolov8s", 12, "dxnn") == "yolov8s_r012.dxnn"
    with pytest.raises(ValueError):
        round_file_name("yolov8s", 1, "onnx")
    _, _, cfg = load_config("configs/two_cam_current.yaml")
    # 0회차는 예전 파일을 그대로 가리킨다(npu 항목과 같은 경로)
    assert primary_det_model(cfg, "hailo") == (cfg["npu"]["det_hef"], 0)
    assert primary_det_model(cfg, "dxm1") == (cfg["npu"]["dx_det_model"], 0)
    assert shadow_det_model(cfg, "dxm1") == (cfg["npu"]["dx_det_model"], 0)
    # 등록되지 않은 회차는 이름 규칙
    assert resolve_round_model(cfg, 1, "hef") == os.path.join("models", "yolov8s_r001.hef")
    # models 항목이 없으면 예전 설정 그대로
    assert primary_det_model({"npu": {}}, "hailo") == (None, None)
    assert shadow_det_model({"npu": {"det_hef": "a.hef", "dx_det_model": "b.dxnn"}}, "dxm1") == ("b.dxnn", None)


# ---------------------------------------------------------------------------
# 섀도 비교와 점수 엿보기
# ---------------------------------------------------------------------------
from realityview.shadow import ScoreTap, ShadowBackend, compare_dets  # noqa: E402
from realityview.types import Detection, FrameResult  # noqa: E402


class ListBackend:
    """미리 정한 검출 목록을 돌려주는 가짜 백엔드."""

    def __init__(self, dets, temp=None):
        self.dets = dets
        self.closed = False
        if temp is not None:
            self.chip_temperature = lambda: temp

    def infer(self, frames):
        return [FrameResult(detections=list(self.dets)) for _ in frames]

    def close(self):
        self.closed = True


def test_compare_dets():
    a = [Detection("car", 0.9, (0, 0, 10, 10)), Detection("person", 0.6, (20, 20, 30, 40))]
    b = [Detection("car", 0.8, (1, 0, 11, 10)), Detection("car", 0.5, (50, 50, 60, 60))]
    c = compare_dets(a, b, 0.5)
    assert c["matched"] == 1 and c["shadow_missed"] == 1 and c["shadow_extra"] == 1
    assert c["score_diff_mean"] == pytest.approx(-0.1, abs=1e-6)
    assert 0.8 < c["iou_mean"] < 0.9
    e = compare_dets([], [], 0.5)
    assert e["matched"] == 0 and e["iou_mean"] is None


def test_score_tap_keeps_display_results_and_forwards_temperature():
    dets = [Detection("car", 0.9, (0, 0, 10, 10)), Detection("car", 0.35, (20, 0, 30, 10))]
    tap = ScoreTap(ListBackend(dets, temp=70.0), display_thr=0.4)
    out = tap.infer([None, None])
    # 화면과 추적에는 원래 문턱 이상만, 낮은 상자는 last_raw에만
    assert [d.score for d in out[0].detections] == [0.9]
    assert [d.score for d in tap.last_raw[0].detections] == [0.9, 0.35]
    assert tap.chip_temperature() == 70.0
    assert getattr(ScoreTap(ListBackend(dets), 0.4), "chip_temperature", None) is None


def test_shadow_backend_returns_primary_and_writes_csv(tmp_path):
    prim = ListBackend([Detection("car", 0.9, (0, 0, 10, 10))])
    sh = ListBackend([Detection("car", 0.7, (0, 0, 10, 12))], temp=81.0)
    path = str(tmp_path / "shadow.csv")
    sb = ShadowBackend(prim, sh, path, ["front", "rear"])
    for _ in range(3):
        out = sb.infer([np.zeros((4, 4, 3), np.uint8)] * 2)
    assert out[0].detections[0].score == 0.9  # 기본 결과만 돌려준다
    s = sb.summary()
    assert s["frames"] == 3 and s["matched"] == 6 and s["shadow_missed"] == 0
    # Hailo가 섀도 쪽에 있어도 온도 멈춤이 꺼지지 않는다
    assert getattr(sb, "chip_temperature")() == 81.0
    sb.close()
    assert prim.closed and sh.closed
    with open(path) as f:
        lines = f.read().strip().splitlines()
    assert lines[0].startswith("frame,camera,n_primary") and len(lines) == 1 + 3 * 2


# ---------------------------------------------------------------------------
# 트리거 클립
# ---------------------------------------------------------------------------
from realityview.tracker import Track  # noqa: E402
from realityview.triggers import TrackIdWatcher, TriggerClipper, TriggerEvent, ambiguous_event, mismatch_event  # noqa: E402


def test_trigger_events():
    res = [FrameResult([Detection("car", 0.45, (0, 0, 5, 5)), Detection("car", 0.9, (0, 0, 5, 5))]), FrameResult([])]
    ev = ambiguous_event(7, res, ["front", "rear"], 0.3, 0.5)
    assert ev.kind == "ambiguous" and ev.score == 0.45 and ev.frame == 7
    assert ambiguous_event(7, [FrameResult([Detection("car", 0.5, (0, 0, 1, 1))])], ["front"], 0.3, 0.5) is None

    w = TrackIdWatcher(switch_dist=2.0)
    t1 = Track(1, "car", np.array([5.0, 0.0]))
    assert [e.kind for e in w.update(0, [t1])] == ["track_new"]
    assert w.update(1, [t1]) == []
    t2 = Track(2, "car", np.array([5.5, 0.2]))  # 1번이 사라진 자리에서 2번이 생김
    evs = w.update(2, [t2])
    assert [e.kind for e in evs] == ["track_switch"] and evs[0].detail["old_id"] == 1

    assert mismatch_event(3, [{"shadow_missed": 0, "shadow_extra": 0}]) is None
    m = mismatch_event(3, [{"camera": "front", "shadow_missed": 1, "shadow_extra": 1}], 1)
    assert m.kind == "npu_mismatch" and m.score == 2.0


def test_trigger_clipper_ring_cooldown_and_replay(tmp_path):
    cams = [make_cam("front", 32, 24), make_cam("rear", 32, 24)]
    clip = TriggerClipper(str(tmp_path), ["front", "rear"], pre_s=0.25, post_s=0.2, cooldown_s=1.0, log=lambda s: None)
    t = 100.0
    for n in range(40):  # 10 fps
        frames = [np.full((24, 32, 3), n, np.uint8), np.full((24, 32, 3), 255 - n, np.uint8)]
        clip.push(n, frames, t)
        if n == 10:
            clip.fire([TriggerEvent("ambiguous", 0.42, n)], t)
        if n == 11:
            clip.fire([TriggerEvent("track_new", 0.0, n)], t)  # 모으는 중이라 같은 클립에 합쳐진다
        if n == 15:
            clip.fire([TriggerEvent("ambiguous", 0.41, n)], t)  # 1초 안이라 막힌다
        if n == 30:
            clip.fire([TriggerEvent("ambiguous", 0.40, n)], t)
        t += 0.1
    st = clip.close()
    assert st["clips_written"] == 2 and st["merged"] == 1 and st["suppressed"] == {"ambiguous": 1}
    first = sorted(st["clips"])[0]
    with open(os.path.join(first, "trigger.json")) as f:
        meta = json.load(f)
    assert [e["kind"] for e in meta["events"]] == ["ambiguous", "track_new"]
    # 앞 0.25초(트리거 프레임 포함 3장) + 뒤 0.2초(2장)
    assert meta["first_frame"] == 8 and meta["trigger_frame"] == 10 and meta["last_frame"] == 12
    rs = open_replay_sources(first, cams)
    assert int(rs[0].read().mean()) in range(6, 11) and rs[0].recorded_index == 8
    with open(os.path.join(first, "front.csv")) as f:
        wall = float(f.read().splitlines()[1].split(",")[3])
    assert abs(wall - time.time()) < 60  # 벽시계(time.time())를 적는다
    for r in rs:
        r.close()


def test_trigger_clipper_memory_cap(tmp_path):
    clip = TriggerClipper(str(tmp_path), ["front"], pre_s=10.0, post_s=0.0, max_buffer_mb=0.01, log=lambda s: None)
    for n in range(10):
        clip.push(n, [np.zeros((48, 64, 3), np.uint8)], float(n))  # 9KB씩, 한도 10KB
    assert len(clip._ring) == 1 and clip.stats["ring_trimmed_by_memory"] >= 1
    clip.close()
