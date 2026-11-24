import csv
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

from realityview.profiling import HealthLogger, LatencyRecorder, _parse_vcgencmd, read_proc_mem
from realityview.recorder import VideoRecorder, frames_due

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_run_demo():
    spec = importlib.util.spec_from_file_location("run_demo_08", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 상태 기록 (HealthLogger)
# ---------------------------------------------------------------------------
class _FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_parse_vcgencmd():
    assert _parse_vcgencmd("throttled=0x0") == "0x0"
    assert _parse_vcgencmd("temp=55.4'C") == "55.4"
    assert _parse_vcgencmd(None) is None and _parse_vcgencmd("oops") is None


def test_read_proc_mem_parses_and_tolerates_missing(tmp_path):
    st = tmp_path / "status"
    st.write_text("Name:\tpython\nVmRSS:\t  204800 kB\n")
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 4000000 kB\nMemAvailable:    1048576 kB\n")
    assert read_proc_mem(str(st), str(mi)) == {"rss_mb": 200.0, "mem_available_mb": 1024.0}
    assert read_proc_mem(str(tmp_path / "x"), str(tmp_path / "y")) == {"rss_mb": None, "mem_available_mb": None}


def test_health_logger_writes_interval_rows(tmp_path):
    clock = _FakeClock()
    path = tmp_path / "soak" / "h.csv"
    hl = HealthLogger(
        str(path),
        interval_s=60,
        clock=clock,
        health_fn=lambda: {"throttled": "throttled=0x0", "temp": "temp=61.2'C"},
        mem_fn=lambda: {"rss_mb": 300.0, "mem_available_mb": 2000.0},
    )
    hl.start()
    # 1/32초 간격(32 FPS, 2진수로 정확한 값)으로 150초: 60초, 120초에 한 줄씩, 닫을 때 남은 30초 한 줄
    for _ in range(4800):
        clock.t += 0.03125
        hl.tick()
    hl.close()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["elapsed_s"] for r in rows] == ["0.0", "60.0", "120.0", "150.0"]
    assert float(rows[1]["fps"]) == pytest.approx(32.0, abs=0.1)
    assert int(rows[1]["frames"]) + int(rows[2]["frames"]) + int(rows[3]["frames"]) == 4800
    assert rows[1]["throttled"] == "0x0" and rows[1]["soc_temp_c"] == "61.2" and rows[1]["rss_mb"] == "300.0"
    hl.tick()  # 닫은 뒤에는 아무것도 하지 않는다
    hl.close()


# ---------------------------------------------------------------------------
# 지연 기록 (LatencyRecorder)
# ---------------------------------------------------------------------------
def test_latency_recorder_per_camera_and_warmup():
    lr = LatencyRecorder(["front", "rear"], warmup=2)
    ms = 1_000_000
    for i in range(12):
        base = i * 40 * ms
        # front: 센서 시각 base, rear: base+10ms. 화면 완성은 base+60ms
        lr.add([base, base + 10 * ms], [base + 5 * ms, base + 12 * ms], base + 60 * ms, seg_mask_sensor_ns=base - 80 * ms)
    s = lr.summary()
    assert s["frames_seen"] == 12
    assert s["sensor_to_display_ms"]["front"]["n"] == 10
    assert s["sensor_to_display_ms"]["front"]["mean_ms"] == 60.0
    assert s["sensor_to_display_ms"]["rear"]["p50_ms"] == 50.0
    assert s["sensor_to_display_oldest_ms"]["p95_ms"] == 60.0
    assert s["sensor_to_read_ms"]["rear"]["mean_ms"] == 2.0
    assert s["seg_mask_sensor_to_display_ms"]["mean_ms"] == 140.0


def test_latency_recorder_skips_unknown_timestamps():
    lr = LatencyRecorder(["a", "b"])
    lr.add([None, None], [None, None], 10**9)
    s = lr.summary()
    assert s["sensor_to_display_oldest_ms"] is None and s["seg_mask_sensor_to_display_ms"] is None


# ---------------------------------------------------------------------------
# 녹화 (VideoRecorder)
# ---------------------------------------------------------------------------
def test_frames_due_keeps_real_time():
    assert frames_due(0.0, 10, 0) == 1  # 첫 장
    assert frames_due(0.05, 10, 1) == 0  # 루프가 빠르면 건너뜀
    assert frames_due(0.1, 10, 1) == 1
    assert frames_due(0.35, 10, 2) == 2  # 루프가 느리면 반복


class _FakeWriter:
    opened_codecs = []
    ok_codecs = {"mp4v"}

    def __init__(self, path, fourcc, fps, size):
        self.cc = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        _FakeWriter.opened_codecs.append(self.cc)
        self.frames = []
        _FakeWriter.last = self

    def isOpened(self):
        return self.cc in _FakeWriter.ok_codecs

    def write(self, f):
        self.frames.append(int(f[0, 0, 0]))

    def release(self):
        pass


def test_video_recorder_falls_back_and_resamples(tmp_path):
    _FakeWriter.opened_codecs = []
    clock = _FakeClock()
    rec = VideoRecorder(str(tmp_path / "v" / "a.mp4"), fps=10, writer_factory=_FakeWriter, clock=clock, queue_size=100)
    # 20 FPS로 2초(40장) 넣으면 10 fps 파일에는 약 20장이 들어가야 한다
    for i in range(40):
        rec.submit(np.full((4, 6, 3), i % 256, np.uint8))
        clock.t += 0.05
    st = rec.close()
    assert _FakeWriter.opened_codecs[:2] == ["avc1", "mp4v"] and st["codec"] == "mp4v"
    assert st["size_wh"] == [6, 4] and st["dropped"] == 0
    assert st["written"] == 20 and st["duration_s"] == 2.0
    assert _FakeWriter.last.frames[:3] == [0, 2, 4]  # 짝수 번째 영상만 남는다


def test_video_recorder_fails_loudly_when_no_codec(tmp_path):
    _FakeWriter.ok_codecs = set()
    try:
        rec = VideoRecorder(str(tmp_path / "b.mp4"), fps=10, writer_factory=_FakeWriter)
        with pytest.raises(RuntimeError):
            rec.submit(np.zeros((4, 6, 3), np.uint8))
        rec.close()
    finally:
        _FakeWriter.ok_codecs = {"mp4v"}


# ---------------------------------------------------------------------------
# 카메라 타임스탬프 (가짜 picamera2)
# ---------------------------------------------------------------------------
class _FakeRequest:
    def __init__(self, cam):
        self.cam = cam

    def make_array(self, name):
        return np.zeros((432, 768, 3), np.uint8)

    def get_metadata(self):
        return {"SensorTimestamp": 123_000_000}

    def release(self):
        self.cam.released += 1


class _FakeCam:
    def __init__(self, index):
        self.released = 0
        self.array_calls = 0
        self.camera_config = None

    def create_video_configuration(self, **kw):
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
        self.array_calls += 1
        return np.zeros((432, 768, 3), np.uint8)

    def capture_request(self):
        return _FakeRequest(self)


def test_picamera_timestamp_only_when_enabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=_FakeCam))
    from realityview.sources import PiCameraSource

    src = PiCameraSource(0, 768, 432)
    src.read()
    assert src.cam.array_calls == 1 and src.last_timestamp_ns is None  # 기본 동작은 그대로
    src.want_timestamp = True
    f = src.read()
    assert f.shape == (432, 768, 3) and src.last_timestamp_ns == 123_000_000 and src.cam.released == 1


# ---------------------------------------------------------------------------
# run_demo 새 옵션
# ---------------------------------------------------------------------------
def test_run_demo_new_options_default_off():
    a = _load_run_demo().build_parser().parse_args([])
    assert (a.duration, a.health_csv, a.record_video, a.record_fps) == (0.0, None, None, 0.0)


def test_run_demo_sim_health_and_record(tmp_path, monkeypatch):
    mod = _load_run_demo()
    monkeypatch.chdir(ROOT)
    import realityview.recorder as recmod

    monkeypatch.setattr(recmod.cv2, "VideoWriter", _FakeWriter)
    csv_path = tmp_path / "h.csv"
    prof = tmp_path / "p.json"
    argv = ["run_demo.py", "--backend", "sim", "--headless", "--frames", "10", "--warmup", "3",
            "--health-csv", str(csv_path), "--health-interval", "0", "--record-video", str(tmp_path / "v.mp4"),
            "--record-fps", "5", "--profile", "--profile-out", str(prof)]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) >= 2 and rows[0]["elapsed_s"] == "0.0"
    rep = json.loads(prof.read_text(encoding="utf-8"))
    assert rep["record"]["submitted"] == 7 and rep["record"]["codec"] == "mp4v"
    assert "latency" not in rep  # sim에는 센서 타임스탬프가 없다


def test_health_logger_extra_columns(tmp_path):
    clock = _FakeClock()
    path = tmp_path / "h.csv"
    hl = HealthLogger(str(path), interval_s=1, clock=clock, health_fn=lambda: {}, mem_fn=lambda: {},
                      extra_fn=lambda: {"hailo_temp_c": 71.5}, extra_columns=("hailo_temp_c",))
    hl.start()
    clock.t += 0.5
    assert hl.tick() is None  # 간격 전에는 줄을 쓰지 않는다
    clock.t += 0.6
    row = hl.tick()
    hl.close()
    assert row["hailo_temp_c"] == 71.5
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["hailo_temp_c"] == "71.5" and rows[0]["throttled"] == ""


def test_hailo_chip_temperature_reads_hottest_sensor_or_none():
    from realityview.detectors import HailoBackend

    b = HailoBackend.__new__(HailoBackend)  # 장치 없이 메서드만 시험
    temp = types.SimpleNamespace(ts0_temperature=70.04, ts1_temperature=72.36)
    dev = types.SimpleNamespace(control=types.SimpleNamespace(get_chip_temperature=lambda: temp))
    b.vdevice = types.SimpleNamespace(get_physical_devices=lambda: [dev])
    assert b.chip_temperature() == 72.4
    b.vdevice = types.SimpleNamespace(get_physical_devices=lambda: [])
    assert b.chip_temperature() is None


def test_run_demo_hailo_stop_temp_default_off():
    a = _load_run_demo().build_parser().parse_args([])
    assert a.hailo_stop_temp == 0.0


def test_run_demo_hailo_stop_temp_needs_health_csv(monkeypatch):
    mod = _load_run_demo()
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--hailo-stop-temp", "95"])
    with pytest.raises(SystemExit):
        mod.main()
