import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import pytest

from realityview.config import load_config
from realityview.detcheck import box_iou, flip_consistency, unmirror_box
from realityview.detectors import letterbox, parse_hailo_nms
from realityview.pipeline import RealityViewPipeline
from realityview.profiling import STAGE_ORDER, StageTimer, format_table, read_pi_health
from realityview.sim import SimBackend, SimWorld
from realityview.sources import picamera_config_kwargs, rotation_flips
from realityview.types import COCO_ALL, Detection

# 1단계 실기 검증에서 새로 넣은 코드(시간 측정, 검출 확인, 카메라 설정, 렌더링 스레드)를 시험한다.
# 하드웨어가 없어도 돌도록 Picamera2는 가짜 모듈로 바꿔 끼운다.

ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# 시간 측정(StageTimer)
# ---------------------------------------------------------------------------
def test_timer_disabled_records_nothing():
    tm = StageTimer(enabled=False)
    with tm.stage("capture"):
        pass
    tm.add("render", 5.0)
    tm.end_frame(10.0)
    assert tm.summary() == {} and tm.frames == 0


def test_timer_sums_within_frame_and_skips_warmup():
    tm = StageTimer(enabled=True, warmup=2)
    for k in range(6):
        # 카메라 2대처럼 한 프레임에 두 번 불리는 단계는 더해져야 한다
        tm.add("preprocess", 1.0)
        tm.add("preprocess", 2.0)
        tm.add("render", float(k))
        tm.end_frame(total_ms=10.0 + k)
    s = tm.summary()
    assert s["preprocess"]["n"] == 4 and s["preprocess"]["mean_ms"] == pytest.approx(3.0)
    # 워밍업 2프레임(k=0,1)이 빠져 render 표본은 2,3,4,5
    assert s["render"]["mean_ms"] == pytest.approx(3.5)
    assert s["render"]["p50_ms"] == pytest.approx(3.5)
    assert s["frame_total"]["p95_ms"] == pytest.approx(np.percentile([12, 13, 14, 15], 95))
    # 표 순서는 STAGE_ORDER를 따른다
    assert list(s) == [n for n in STAGE_ORDER if n in s]


def test_timer_stage_measures_elapsed_time():
    tm = StageTimer(enabled=True)
    with tm.stage("track"):
        time.sleep(0.01)
    tm.end_frame()
    assert tm.summary()["track"]["mean_ms"] >= 9.0


def test_format_table_has_three_columns():
    tm = StageTimer(enabled=True)
    tm.add("capture", 1.25)
    tm.end_frame(2.0)
    lines = format_table(tm.summary()).splitlines()
    assert all(line.count("|") == 4 for line in lines)  # 3열 표
    assert "카메라 획득" in lines[2] and "프레임 전체" in lines[3]


def test_read_pi_health_never_raises():
    h = read_pi_health()
    assert set(h) == {"throttled", "temp"}


# ---------------------------------------------------------------------------
# 검출 확인 도구(detcheck)와 NMS 파싱
# ---------------------------------------------------------------------------
def test_box_iou_and_unmirror():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert unmirror_box((10, 20, 30, 40), 100) == (70, 20, 90, 40)


def test_flip_consistency_detects_swapped_axes():
    w = 768
    dets = [Detection("car", 0.9, (100, 300, 220, 400))]
    good = [Detection("car", 0.9, unmirror_box(dets[0].box, w))]  # 좌우 반전 영상에서 나올 박스
    assert flip_consistency(dets, good, w) == [pytest.approx(1.0)]
    # x, y를 바꿔 읽은 경우처럼 엉뚱한 곳에 나온 박스는 IoU가 낮다
    bad = [Detection("car", 0.9, (300, 100, 400, 220))]
    assert flip_consistency(dets, bad, w)[0] < 0.3
    # 이름이 다르면 짝짓지 않는다
    assert flip_consistency(dets, [Detection("person", 0.9, good[0].box)], w) == []


def test_parse_hailo_nms_labels_override_keeps_all_classes():
    img = np.zeros((576, 768, 3), np.uint8)
    _, s, pad = letterbox(img, (640, 640))
    nms = [np.zeros((0, 5)) for _ in range(80)]
    nms[4] = np.array([[0.3, 0.2, 0.5, 0.6, 0.9]])  # airplane
    assert parse_hailo_nms(nms, (640, 640), s, pad, 0.4) == []  # 기본은 주행 클래스만
    dets = parse_hailo_nms(nms, (640, 640), s, pad, 0.4, labels=COCO_ALL)
    assert len(dets) == 1 and dets[0].label == "airplane"
    x1, y1, x2, y2 = dets[0].box
    # 정규화 좌표 [ymin, xmin, ymax, xmax] -> 원본 픽셀 (x1, y1, x2, y2)
    assert x1 == pytest.approx((0.2 * 640 - pad[0]) / s) and y1 == pytest.approx((0.3 * 640 - pad[1]) / s)
    assert x2 == pytest.approx((0.6 * 640 - pad[0]) / s) and y2 == pytest.approx((0.5 * 640 - pad[1]) / s)


# ---------------------------------------------------------------------------
# 카메라 설정 만들기(순수 함수)
# ---------------------------------------------------------------------------
def test_picamera_config_kwargs():
    kw = picamera_config_kwargs(768, 576, 30, sensor_size=[1296, 972])
    assert kw["main"] == {"size": (768, 576), "format": "RGB888"}
    assert kw["controls"] == {"FrameRate": 30.0}
    assert kw["sensor"] == {"output_size": (1296, 972)}
    assert "sensor" not in picamera_config_kwargs(768, 432, 15)


def test_rotation_flips():
    assert rotation_flips(0) == (False, False)
    assert rotation_flips(180) == (True, True)
    assert rotation_flips(-180) == (True, True)
    with pytest.raises(ValueError):
        rotation_flips(90)


# ---------------------------------------------------------------------------
# PiCameraSource (가짜 picamera2 모듈로 시험)
# ---------------------------------------------------------------------------
class _FakePicamera2:
    instances = []
    fail_configure = False

    def __init__(self, index):
        self.index = index
        self.calls = []
        self.cfg_kwargs = None
        self.camera_config = None
        _FakePicamera2.instances.append(self)

    def create_video_configuration(self, **kw):
        self.cfg_kwargs = kw
        return dict(kw)

    def configure(self, cfg):
        self.calls.append("configure")
        if _FakePicamera2.fail_configure:
            raise RuntimeError("configure failed")
        self.camera_config = {"sensor": {"output_size": (1296, 972), "bit_depth": 10}}

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")

    def capture_array(self, name):
        return np.zeros((480, 640, 4), np.uint8)  # 4채널, 설정과 다른 크기

    def capture_metadata(self):
        return {"ScalerCrop": (0, 0, 2592, 1944)}


class _FakeTransform:
    def __init__(self, hflip=0, vflip=0):
        self.hflip, self.vflip = hflip, vflip


@pytest.fixture
def fake_picam(monkeypatch):
    _FakePicamera2.instances = []
    _FakePicamera2.fail_configure = False
    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=_FakePicamera2))
    monkeypatch.setitem(sys.modules, "libcamera", types.SimpleNamespace(Transform=_FakeTransform))
    return _FakePicamera2


def test_picamera_close_releases_camera(fake_picam):
    from realityview.sources import PiCameraSource

    src = PiCameraSource(0, 768, 576, 30, sensor_size=(1296, 972))
    cam = fake_picam.instances[0]
    assert cam.cfg_kwargs["sensor"] == {"output_size": (1296, 972)}
    assert src.sensor_config["output_size"] == (1296, 972)
    src.close()
    src.close()  # 두 번 불러도 한 번만 놓는다
    assert cam.calls == ["configure", "start", "stop", "close"]


def test_picamera_configure_failure_still_closes(fake_picam):
    from realityview.sources import PiCameraSource

    fake_picam.fail_configure = True
    with pytest.raises(RuntimeError):
        PiCameraSource(1, 768, 432)
    assert fake_picam.instances[0].calls == ["configure", "close"]


def test_picamera_rotation_and_read_shape(fake_picam):
    from realityview.sources import PiCameraSource

    src = PiCameraSource(1, 768, 432, rotation=180)
    t = fake_picam.instances[0].cfg_kwargs["transform"]
    assert (t.hflip, t.vflip) == (1, 1)
    f = src.read()
    assert f.shape == (432, 768, 3)  # 3채널, 설정 크기로 맞춤
    src.close()


def test_open_source_passes_sensor_size_and_rotation(fake_picam):
    from realityview.sources import open_source

    cams, _, _ = load_config(os.path.join(ROOT, "configs/two_cam_current.yaml"))
    srcs = [open_source(c) for c in cams]
    try:
        assert [c.index for c in fake_picam.instances] == [0, 1]
        assert fake_picam.instances[0].cfg_kwargs["sensor"] == {"output_size": (1296, 972)}
        assert fake_picam.instances[1].cfg_kwargs["sensor"] == {"output_size": (2304, 1296)}
        assert fake_picam.instances[1].cfg_kwargs["main"]["size"] == (768, 432)
    finally:
        for s in srcs:
            s.close()


# ---------------------------------------------------------------------------
# 지금 장비용 설정 파일
# ---------------------------------------------------------------------------
def test_two_cam_current_config_matches_sensors():
    cams, grid, cfg = load_config(os.path.join(ROOT, "configs/two_cam_current.yaml"))
    front, rear = cams
    assert (front.name, front.source["index"], front.width, front.height) == ("front", 0, 768, 576)
    assert (rear.name, rear.source["index"], rear.width, rear.height) == ("rear", 1, 768, 432)
    # 전방 ov5647은 130도 광각 렌즈라 어안(대각 130도 등거리 근사), 후방 imx708은 핀홀 수평 약 68.5도
    assert front.model == "fisheye" and front.max_fov_deg == 140.0
    assert front.K[0, 0] == pytest.approx(480 / np.radians(65), abs=0.1)
    hfov = lambda c: np.degrees(2 * np.arctan(c.width / 2 / c.K[0, 0]))  # noqa: E731
    assert rear.model == "pinhole" and hfov(rear) == pytest.approx(68.5, abs=0.2)
    # 센서 모드 비율이 출력 비율과 같아야 ISP가 가장자리를 잘라 내지 않는다
    for c in cams:
        sw, sh = c.source["sensor_size"]
        assert sw / sh == pytest.approx(c.width / c.height, rel=0.01)
    assert cfg["npu"]["det_hef"] == "models/yolov8s_h8.hef"


# ---------------------------------------------------------------------------
# 파이프라인: 시간 측정과 렌더링 스레드
# ---------------------------------------------------------------------------
def _sim_pipe(**kw):
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    world = SimWorld(cams)
    return world, RealityViewPipeline(cams, SimBackend(world, noise_px=1.0, miss_rate=0.0), grid, **kw)


def test_pipeline_profile_records_cpu_stages():
    tm = StageTimer(enabled=True)
    world, pipe = _sim_pipe(timer=tm)
    for _ in range(3):
        world.step(0.1)
        pipe.step(world.render(), 0.1)
        tm.end_frame(1.0)
    s = tm.summary()
    for name in ("detect_total", "ipm_project", "merge", "track", "bev_warp", "render"):
        assert s[name]["n"] == 3, name
    assert pipe.render_count == 3
    pipe.close()


def test_pipeline_render_thread_matches_sync_output():
    world, pipe = _sim_pipe(render_thread=True)
    try:
        outs = []
        for _ in range(12):
            world.step(0.1)
            outs.append(pipe.step(world.render(), 0.1))
            time.sleep(0.02)  # 렌더링 스레드가 따라올 시간
        # 첫 프레임부터 대시보드가 있어야 한다(None이면 --save, --snapshot이 깨짐)
        assert all(o.dashboard is not None and o.dashboard.ndim == 3 for o in outs)
        deadline = time.time() + 5
        while pipe.render_count < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert pipe.render_count >= 2
        assert outs[-1].dashboard.shape == outs[0].dashboard.shape
    finally:
        pipe.close()
    assert not pipe._worker._thread.is_alive()


def test_pipeline_no_ground_skips_warp():
    tm = StageTimer(enabled=True)
    world, pipe = _sim_pipe(timer=tm, show_ground=False)
    world.step(0.1)
    out = pipe.step(world.render(), 0.1)
    tm.end_frame(1.0)
    assert out.bev.shape[:2] == (pipe.grid.height, pipe.grid.width)
    pipe.close()


# ---------------------------------------------------------------------------
# run_demo.py: 옵션 기본값과 측정 JSON
# ---------------------------------------------------------------------------
def _load_run_demo():
    spec = importlib.util.spec_from_file_location("run_demo", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_demo_defaults_keep_old_behavior():
    a = _load_run_demo().build_parser().parse_args([])
    assert (a.profile, a.bev_res, a.render_thread, a.no_ground) == (False, None, False, False)


def test_run_demo_profile_writes_json(tmp_path, monkeypatch):
    mod = _load_run_demo()
    out = tmp_path / "prof.json"
    monkeypatch.chdir(ROOT)
    argv = ["run_demo.py", "--backend", "sim", "--headless", "--frames", "8", "--warmup", "3",
            "--profile", "--profile-out", str(out), "--bev-res", "0.1", "--render-thread"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["frames_total"] == 8 and rep["frames_measured"] == 5
    assert rep["options"]["bev_res"] == 0.1 and rep["options"]["render_thread"] is True
    assert rep["options"]["bev_size_hw"] == [400, 240]
    assert rep["stages_ms"]["capture"]["n"] == 5 and rep["stages_ms"]["frame_total"]["n"] == 5
    assert rep["loop_fps"] > 0 and set(rep["health_start"]) == {"throttled", "temp"}
