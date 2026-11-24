import configparser
import json
import os
import shlex
import time

import numpy as np
import pytest

from realityview.power import ShutdownWatcher, finish
from realityview.thermal import (
    CAUTION,
    CRITICAL,
    DEFAULT_LIMITS,
    NORMAL,
    WARNING,
    SensorLimits,
    ThermalConfig,
    ThermalController,
    ThermalGateBackend,
    ThermalMonitor,
    actions_for,
    parse_limits,
    step_level,
    throttled_now,
)
from realityview.types import Detection, FrameResult

ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# 센서별 단계와 히스테리시스
# ---------------------------------------------------------------------------
def test_step_level_up_down_with_hysteresis():
    lim = SensorLimits(caution=90, warning=95, critical=100, hysteresis=4)
    assert step_level(NORMAL, 85.0, lim) == NORMAL
    assert step_level(NORMAL, 90.0, lim) == CAUTION
    # 급하게 오르면 한 번에 위험까지
    assert step_level(NORMAL, 101.0, lim) == CRITICAL
    # 경계 바로 아래로 내려와도 해제 온도(경계-4) 아래가 아니면 그대로
    assert step_level(WARNING, 93.0, lim) == WARNING
    assert step_level(WARNING, 91.5, lim) == WARNING
    assert step_level(WARNING, 90.9, lim) == CAUTION
    assert step_level(CAUTION, 86.5, lim) == CAUTION
    assert step_level(CAUTION, 85.9, lim) == NORMAL
    # 많이 식으면 여러 단계를 한 번에 내려간다
    assert step_level(CRITICAL, 70.0, lim) == NORMAL


def test_missing_reading_keeps_previous_level():
    lim = DEFAULT_LIMITS["hailo"]
    assert step_level(WARNING, None, lim) == WARNING
    assert step_level(NORMAL, None, lim) == NORMAL


def test_default_release_is_above_measured_idle():
    # 부하 없을 때 측정 온도(방열 없음): Hailo-8 최고 84도, DX-M1 75도, SoC 연속 실행 중 67.5도.
    # 해제 온도가 이보다 낮으면 주의 단계가 영영 풀리지 않는다
    idle = {"hailo": 84.0, "dxm1": 75.0, "soc": 67.5}
    for name, t in idle.items():
        lim = DEFAULT_LIMITS[name]
        assert lim.caution - lim.hysteresis > t, name
        assert step_level(CAUTION, t, lim) == NORMAL, name
    # 근거 수치: Hailo 위험은 HailoRT 경고(약 104도) 아래, DX-M1 위험은 85도
    assert DEFAULT_LIMITS["hailo"].critical < 104.0
    assert DEFAULT_LIMITS["dxm1"].critical <= 85.0


def test_parse_limits_and_throttled_bits():
    lim = parse_limits("hailo=80,85,90;soc=60,70,80,2")
    assert lim["hailo"].bounds() == [80, 85, 90] and lim["hailo"].hysteresis == DEFAULT_LIMITS["hailo"].hysteresis
    assert lim["soc"].hysteresis == 2 and lim["dxm1"] == DEFAULT_LIMITS["dxm1"]
    with pytest.raises(ValueError):
        parse_limits("hailo=95,90,100")
    assert throttled_now("0x0") == 0
    assert throttled_now("0x50000") == 0  # 지난 일 비트만 서 있음
    assert throttled_now("0x50005") == 5
    assert throttled_now(None) is None and throttled_now("bad") is None


# ---------------------------------------------------------------------------
# 감시기: 가짜 온도 순서로 단계 전환
# ---------------------------------------------------------------------------
def make_monitor(tmp_path=None, **kw):
    logs = []
    t = {"now": 0.0}
    mon = ThermalMonitor(ThermalConfig(**kw), {"hailo": lambda: None, "dxm1": lambda: None, "soc": lambda: None},
                         throttled_fn=None, csv_path=str(tmp_path / "th.csv") if tmp_path else None,
                         log=logs.append, clock=lambda: t["now"])
    return mon, logs, t


def test_monitor_transitions_and_csv(tmp_path):
    mon, logs, clk = make_monitor(tmp_path)
    seq = [
        ({"hailo": 84, "dxm1": 75, "soc": 60}, "normal"),
        ({"hailo": 91, "dxm1": 75, "soc": 60}, "caution"),
        ({"hailo": 93, "dxm1": 80, "soc": 60}, "warning"),  # DX-M1 80도가 경고를 만든다
        ({"hailo": 93, "dxm1": 79, "soc": 60}, "warning"),  # 79도는 해제 온도(78) 위라 유지
        ({"hailo": 93, "dxm1": 77, "soc": 60}, "caution"),  # Hailo 93도는 아직 주의
        ({"hailo": 101, "dxm1": None, "soc": 60}, "critical"),
        ({"hailo": 97, "dxm1": None, "soc": 60}, "critical"),  # 해제 온도 96 위
        ({"hailo": 85, "dxm1": None, "soc": 60}, "caution"),  # dxm1 못 읽음: 직전 단계(caution) 유지
        ({"hailo": 85, "dxm1": 74, "soc": 60}, "normal"),
    ]
    for temps, want in seq:
        clk["now"] += 5
        st = mon.sample(temps=temps, throttled=None)
        assert st.name == want, (temps, st.name)
    s = mon.close()
    assert [tr["to"] for tr in s["transitions"]] == ["caution", "warning", "caution", "critical", "caution", "normal"]
    assert s["max_temps"]["hailo"] == 101
    assert sum("->" in x for x in logs) == 6
    assert any("3번 연속" in x for x in logs)  # dxm1 을 세 번 연속 못 읽음
    rows = (tmp_path / "th.csv").read_text().strip().splitlines()
    assert rows[0].startswith("t_s,wall_time,level,hailo_c,dxm1_c,soc_c") and len(rows) == 1 + len(seq)


def test_throttled_now_forces_caution():
    mon, _, _ = make_monitor()
    assert mon.sample(temps={"soc": 50}, throttled="0x0").level == NORMAL
    st = mon.sample(temps={"soc": 50}, throttled="0x4")
    assert st.level == CAUTION and "throttled" in st.cause
    assert mon.sample(temps={"soc": 50}, throttled="0x40000").level == NORMAL


def test_monitor_thread_with_failing_readers():
    # 맥처럼 센서가 전부 없어도 예외 없이 돌고 정상 단계로 남는다
    def boom():
        raise OSError("no sensor")

    mon = ThermalMonitor(ThermalConfig(interval_s=0.01), {"hailo": boom, "soc": lambda: None},
                         throttled_fn=lambda: None, log=lambda s: None).start()
    time.sleep(0.1)
    s = mon.close()
    assert s["final_level"] == "normal" and s["samples"] >= 2


def test_on_change_called_from_sample():
    seen = []
    mon = ThermalMonitor(ThermalConfig(), {"hailo": lambda: None}, throttled_fn=None, log=lambda s: None,
                         on_change=lambda st: seen.append(st.name))
    mon.sample(temps={"hailo": 101}, throttled=None)
    mon.sample(temps={"hailo": 101}, throttled=None)
    assert seen == ["critical"]


# ---------------------------------------------------------------------------
# 조치: 주기, DX-M1 멈춤, 검출 상한, 되돌리기
# ---------------------------------------------------------------------------
class FakeInner:
    def __init__(self):
        self.calls = 0
        self.seg_every = 3

    def infer(self, frames):
        self.calls += 1
        return [FrameResult(detections=[Detection("car", 0.9, (0, 0, 1, 1))]) for _ in frames]

    def chip_temperature(self):
        return 77.0

    def close(self):
        pass


class FakePipe:
    def __init__(self, backend):
        self.backend = backend
        self.seg_every, self.lanes_every, self.ground_every = 3, 1, 2
        self.seg_paused = False


def test_actions_for_levels():
    cfg = ThermalConfig()
    assert actions_for(NORMAL, cfg).period_mult == 1
    assert actions_for(CAUTION, cfg).period_mult == 2 and not actions_for(CAUTION, cfg).pause_dxm1
    w = actions_for(WARNING, cfg)
    assert w.pause_dxm1 and w.det_fps_cap == 8.0 and not w.stop_npu
    assert actions_for(CRITICAL, cfg).exit
    c = actions_for(CRITICAL, ThermalConfig(critical_action="camera_only"))
    assert c.stop_npu and not c.exit
    k = actions_for(CRITICAL, ThermalConfig(critical_action="continue"))
    assert not k.stop_npu and not k.exit


def test_controller_applies_and_restores():
    mon, logs, _ = make_monitor()
    gate = ThermalGateBackend(FakeInner())
    pipe = FakePipe(gate)
    ctl = ThermalController(mon, pipe, gate, log=logs.append)
    mon.sample(temps={"hailo": 91}, throttled=None)
    ctl.poll()
    assert (pipe.seg_every, pipe.lanes_every, pipe.ground_every) == (6, 2, 4)
    assert gate.inner.seg_every == 6 and not pipe.seg_paused and gate.fps_cap == 0
    mon.sample(temps={"hailo": 96}, throttled=None)
    ctl.poll()
    assert pipe.seg_paused and gate.fps_cap == 8.0 and pipe.seg_every == 12
    mon.sample(temps={"hailo": 101}, throttled=None)
    assert ctl.poll().exit
    mon.sample(temps={"hailo": 80}, throttled=None)
    ctl.poll()
    assert (pipe.seg_every, pipe.lanes_every, pipe.ground_every, gate.inner.seg_every) == (3, 1, 2, 3)
    assert not pipe.seg_paused and gate.fps_cap == 0 and not gate.stopped


def test_controller_dxm1_detector_paused_on_warning():
    mon, _, _ = make_monitor()
    gate = ThermalGateBackend(FakeInner())
    ctl = ThermalController(mon, FakePipe(gate), gate, det_on_dxm1=True, log=lambda s: None)
    mon.sample(temps={"dxm1": 81}, throttled=None)
    ctl.poll()
    assert gate.stopped


def test_gate_backend_fps_cap_and_stop():
    t = {"now": 0.0}
    inner = FakeInner()
    gate = ThermalGateBackend(inner, clock=lambda: t["now"])
    frames = [np.zeros((4, 4, 3), np.uint8)] * 2
    assert gate.chip_temperature() == 77.0  # 안쪽 속성 통과
    gate.fps_cap = 5.0  # 0.2초마다 한 번
    r1 = gate.infer(frames)
    t["now"] = 0.1
    r2 = gate.infer(frames)
    assert inner.calls == 1 and gate.skipped == 1
    assert r2[0] is not r1[0] and r2[0].detections[0].label == "car"  # 새 객체, 같은 내용
    t["now"] = 0.25
    gate.infer(frames)
    assert inner.calls == 2
    gate.stopped = True
    r = gate.infer(frames)
    assert inner.calls == 2 and r[0].detections == [] and len(r) == 2


# ---------------------------------------------------------------------------
# 안전 종료: 신호 파일, GPIO 디바운스, 마무리 명령
# ---------------------------------------------------------------------------
def test_stop_file_stale_removed_then_triggers(tmp_path):
    f = tmp_path / "STOP"
    f.write_text("old")
    got = []
    w = ShutdownWatcher(got.append, stop_file=str(f), log=lambda s: None)
    assert not f.exists()  # 지난 실행이 남긴 파일은 지운다
    assert w.check() is None
    f.write_text("now")
    assert w.check().startswith("stop_file:") and len(got) == 1
    assert w.check() is None and len(got) == 1  # 한 번만 부른다


def test_gpio_debounce_ignores_short_drop():
    t = {"now": 0.0}
    val = {"v": 0}
    got = []
    w = ShutdownWatcher(got.append, gpio_read=lambda: val["v"], acc_on_value=0, debounce_s=2.0,
                        clock=lambda: t["now"], log=lambda s: None)
    w.check()
    val["v"] = 1  # 시동 걸 때처럼 잠깐 꺼짐
    t["now"] = 1.0
    w.check()
    t["now"] = 2.5
    val["v"] = 0
    w.check()
    assert got == []
    val["v"] = 1
    for now in (3.0, 4.0, 4.9):
        t["now"] = now
        w.check()
    assert got == []
    t["now"] = 5.0
    w.check()
    assert len(got) == 1 and got[0].startswith("acc_off")


def test_gpio_read_error_is_not_power_loss():
    def bad():
        raise OSError("gpio busy")

    t = {"now": 0.0}
    got = []
    w = ShutdownWatcher(got.append, gpio_read=bad, debounce_s=0.1, clock=lambda: t["now"], log=lambda s: None)
    for now in (0, 1, 2):
        t["now"] = now
        w.check()
    assert got == []


def test_finish_runs_cmd_only_on_power_loss(tmp_path):
    marker = tmp_path / "poweroff_called"
    cmd = f"touch {shlex.quote(str(marker))}"
    r = finish(None, cmd, log=lambda s: None)
    assert not marker.exists() and r["after_cmd_rc"] is None
    r = finish("stop_file:x", cmd, log=lambda s: None)
    assert marker.exists() and r["after_cmd_rc"] == 0 and r["sync_ms"] >= 0


def test_drive_recorder_stops_cleanly_on_stop_file(tmp_path):
    from test_drive_collect import FakeCam, make_cam

    from realityview.drive_rec import DriveRecorder
    from realityview.mjpeg import read_index

    cams = [make_cam("front"), make_cam("rear")]
    rec = DriveRecorder(str(tmp_path / "drives"), cams, [FakeCam(64, 48), FakeCam(64, 48)], name="d",
                        segment_s=0.3, min_free_gb=0, status_interval=0, log=lambda s: None, health_fn=lambda: {})
    stop = tmp_path / "STOP"
    w = ShutdownWatcher(lambda r: rec.request_stop(f"power_loss:{r}"), stop_file=str(stop), poll_s=0.02,
                        log=lambda s: None)
    rec.start()
    w.start()
    time.sleep(0.45)  # 두 번째 세그먼트 중간
    stop.write_text("")
    rec.run_until_stopped(poll_s=0.02)
    meta = rec.close()
    w.close()
    assert meta["stop_reason"].startswith("power_loss:stop_file:")
    d = tmp_path / "drives" / "d"
    segs = sorted(p for p in os.listdir(d) if p.startswith("seg_"))
    assert len(segs) >= 2
    for s in segs:
        assert (d / s / "segment.json").exists(), s  # 마지막 세그먼트까지 깨끗하게 닫힘
        for cam in ("front", "rear"):
            refs = read_index(str(d / s), cam)
            with open(d / s / f"{cam}.csv") as f:
                rows = len(f.read().strip().splitlines()) - 1
            assert len(refs) == rows > 0  # csv 줄과 영상 안 JPEG 수가 같다(잘린 줄 없음)
            last = refs[-1]
            assert last.offset + last.size == os.path.getsize(d / s / f"{cam}.mjpeg")
    assert json.loads((d / "drive.json").read_text())["stop_reason"] == meta["stop_reason"]


# ---------------------------------------------------------------------------
# 서비스 파일: 형식과 옵션이 실제 스크립트와 맞는지
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,script", [("fsd-record.service", "record_drive.py"), ("fsd-demo.service", "run_demo.py")])
def test_service_files(name, script):
    import importlib.util

    p = os.path.join(ROOT, "deploy", "systemd", name)
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(p)
    svc = cp["Service"]
    assert cp["Install"]["WantedBy"] == "default.target"
    assert svc["RestartPreventExitStatus"] == "75" and int(svc["TimeoutStopSec"]) >= 30
    assert "Conflicts" in cp["Unit"]
    argv = shlex.split(svc["ExecStart"])
    assert argv[1] == f"scripts/{script}" and os.path.exists(os.path.join(ROOT, argv[1]))
    spec = importlib.util.spec_from_file_location("svc_" + script[:-3], os.path.join(ROOT, argv[1]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(argv[2:])  # 옵션 이름이 틀리면 여기서 SystemExit
    assert args.thermal and args.stop_file.startswith("out/")
    assert os.path.exists(os.path.join(ROOT, args.config))
