import importlib.util
import os
import sys
import threading
import time

import numpy as np
import pytest

from realityview.config import load_config
from realityview.detectors import (
    DXM1Backend,
    DxModel,
    batched_nms,
    decode_yolo_raw,
    dx_seg_to_road_mask,
    letterbox,
    nms_numpy,
    parse_dx_detections,
    parse_dx_ppu_detections,
    ppu_boxes,
    PPU_BBOX_DTYPE,
)
from realityview.pipeline import RealityViewPipeline
from realityview.profiling import STAGE_LABEL_KO, STAGE_ORDER, StageTimer
from realityview.sim import SimBackend, SimWorld

# 2단계 DX-M1 백엔드 시험. dx_engine이 없는 맥에서도 돌도록 InferenceEngine을 흉내 낸 가짜 엔진을 쓴다.
# 가짜 엔진은 파이의 dx_engine 3.4.2에서 확인한 동작을 따른다.
#   - get_input_tensors_info(): [{"name", "shape", "dtype", "elem_size"}], 입력은 [1,H,W,3] uint8
#   - run(x) / wait(job): 출력 텐서 순서대로 배열 목록, run_async(x, user_arg)는 job 번호
#   - 입력 dtype이 다르면 ValueError (실제 _validate_inputs_dtype와 같은 자리에서 거른다)
# 맨 아래 두 시험은 파이 .venv에서만(진짜 dx_engine) 돈다.

ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# 가짜 dx_engine.InferenceEngine
# ---------------------------------------------------------------------------
class FakeEngine:
    def __init__(self, path, input_shape=(1, 640, 640, 3), input_dtype=np.uint8, outputs=None, out_fn=None, ppu=False):
        self.path = path
        self.input_shape = list(input_shape)
        self.input_dtype = np.dtype(input_dtype)
        # outputs: [(이름, 모양)] 목록. out_fn(x)가 실제 값을 만든다(없으면 0)
        self.outputs = outputs or [("output0", [1, 84, 8400])]
        self.out_fn = out_fn
        self.ppu = ppu
        self.inputs = []  # 받은 입력(검사용)
        self._jobs = {}
        self._next = 0
        self.disposed = False

    # ---- 정보 조회 ----
    def get_input_tensors_info(self):
        return [{"name": "images", "shape": list(self.input_shape), "dtype": self.input_dtype.type, "elem_size": self.input_dtype.itemsize}]

    def get_output_tensors_info(self):
        return [{"name": n, "shape": list(s), "dtype": np.float32, "elem_size": 4} for n, s in self.outputs]

    def is_ppu(self):
        return self.ppu

    # ---- 실행 ----
    def _check(self, x):
        assert not self.disposed, "닫힌 엔진을 썼다"
        if np.dtype(x.dtype) != self.input_dtype:
            raise ValueError(f"Input dtype mismatch: expected {self.input_dtype}, got {x.dtype}")
        if list(x.shape) != self.input_shape:
            raise ValueError(f"shape mismatch: expected {self.input_shape}, got {list(x.shape)}")
        assert x.flags["C_CONTIGUOUS"]
        self.inputs.append(x.copy())

    def _outs(self, x):
        if self.out_fn is not None:
            return self.out_fn(x)
        return [np.zeros(s, np.float32) for _, s in self.outputs]

    def run(self, x):
        self._check(x)
        return self._outs(x)

    def run_async(self, x, user_arg=None, output_buffer=None):
        self._check(x)
        self._next += 1
        self._jobs[self._next] = self._outs(x)
        return self._next

    def wait(self, job_id):
        return self._jobs.pop(job_id)

    def dispose(self):
        self.disposed = True


def _touch(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"fake")
    return str(p)


def _raw_with_boxes(rows, n=8400, nc=80):
    """YOLOv8 원시 출력 (1, 4+nc, n)을 만든다. rows: [(앵커 번호, cx, cy, w, h, 클래스, 점수)]."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    out[0, 0] = np.linspace(0, 640, n)  # 앵커 중심처럼 영상 전체에 퍼뜨린다
    out[0, 1] = np.linspace(0, 640, n)
    for a, cx, cy, w, h, c, s in rows:
        out[0, :4, a] = (cx, cy, w, h)
        out[0, 4 + c, a] = s
    return out


# ---------------------------------------------------------------------------
# CPU NMS
# ---------------------------------------------------------------------------
def test_nms_numpy_suppresses_overlap_and_limits_count():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert list(nms_numpy(boxes, scores, 0.5)) == [0, 2]
    assert list(nms_numpy(boxes, scores, 0.5, max_det=1)) == [0]
    assert nms_numpy(boxes[:0], scores[:0], 0.5).size == 0


def test_batched_nms_keeps_overlapping_boxes_of_other_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    classes = np.array([2, 2, 0])
    assert sorted(batched_nms(boxes, scores, classes, 0.5).tolist()) == [0, 2]


# ---------------------------------------------------------------------------
# YOLO 원시 출력 해석
# ---------------------------------------------------------------------------
ROWS = [
    (100, 320, 320, 100, 50, 2, 0.9),  # car
    (101, 322, 321, 100, 50, 2, 0.8),  # 같은 차를 조금 어긋나게 다시 찾은 것: NMS로 지워져야 함
    (102, 320, 320, 100, 50, 0, 0.7),  # 같은 자리의 person: 클래스가 달라 남아야 함
    (103, 100, 100, 20, 20, 9, 0.95),  # traffic light: COCO_KEEP에 없어 걸러져야 함
    (104, 500, 500, 30, 30, 2, 0.2),  # 점수가 낮아 걸러져야 함
]


@pytest.mark.parametrize("transpose", [False, True])
def test_decode_yolov8_both_layouts(transpose):
    raw = _raw_with_boxes(ROWS)
    if transpose:
        raw = raw.transpose(0, 2, 1)  # (1, 8400, 84)
    rows = decode_yolo_raw(raw, (640, 640), 0.4, 0.45, "yolov8", keep_classes=[0, 1, 2, 3, 5, 7])
    assert [(r[0], round(r[1], 2)) for r in rows] == [(2, 0.9), (0, 0.7)]
    np.testing.assert_allclose(rows[0][2:], (270, 295, 370, 345), atol=1e-4)


def test_decode_normalized_coords_and_logits():
    raw = _raw_with_boxes(ROWS)
    raw[0, :4] /= 640.0  # 0~1 정규화 좌표로 나오는 모델
    sc = raw[0, 4:]
    raw[0, 4:] = np.where(sc > 0, np.log(sc / (1 - sc + 1e-9) + 1e-9), -10.0)  # 시그모이드 전 로짓
    rows = decode_yolo_raw(raw, (640, 640), 0.4, 0.45, "yolov8", keep_classes=[0, 2])
    assert [r[0] for r in rows] == [2, 0]
    np.testing.assert_allclose(rows[0][2:], (270, 295, 370, 345), atol=0.05)
    assert rows[0][1] == pytest.approx(0.9, abs=1e-3)


def test_decode_yolov5_multiplies_objectness():
    nc = 80
    out = np.zeros((1, 100, 5 + nc), np.float32)
    out[0, :, 0] = np.linspace(0, 640, 100)
    out[0, 7, :5] = (320, 320, 100, 50, 0.5)
    out[0, 7, 5 + 2] = 0.9  # 최종 점수 0.45
    out[0, 8, :5] = (100, 100, 10, 10, 0.3)
    out[0, 8, 5 + 2] = 0.9  # 0.27: 기준 아래
    rows = decode_yolo_raw(out, (640, 640), 0.4, 0.45, "yolov5")
    assert len(rows) == 1 and rows[0][0] == 2 and rows[0][1] == pytest.approx(0.45)


def test_decode_rejects_bad_format_and_shape():
    with pytest.raises(ValueError):
        decode_yolo_raw(np.zeros((1, 84, 8400)), (640, 640), 0.4, fmt="yolov99")
    with pytest.raises(ValueError):
        decode_yolo_raw(np.zeros((1, 3, 80, 80)), (640, 640), 0.4)


def test_parse_dx_detections_undoes_letterbox():
    # 768x432 영상 위 (100,200)-(300,400) 차 한 대를 640x640 레터박스 좌표로 옮겨 넣는다
    img = np.zeros((432, 768, 3), np.uint8)
    _, s, (px, py) = letterbox(img, (640, 640))
    x1, y1, x2, y2 = 100 * s + px, 200 * s + py, 300 * s + px, 400 * s + py
    raw = _raw_with_boxes([(50, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 2, 0.8)])
    dets = parse_dx_detections([raw], ["output0"], (640, 640), s, (px, py), 0.4)
    assert len(dets) == 1 and dets[0].label == "car"
    np.testing.assert_allclose(dets[0].box, (100, 200, 300, 400), atol=1e-3)


def test_parse_dx_detections_multi_output_names_tensors():
    outs = [np.zeros((1, 4, 80, 80)), np.zeros((1, 1, 80, 80))]
    with pytest.raises(ValueError, match=r"reg_s8\[1, 4, 80, 80\].*infer_raw"):
        parse_dx_detections(outs, ["reg_s8", "obj_s8"], (640, 640), 1.0, (0, 0), 0.4)


# ---------------------------------------------------------------------------
# 세그 출력
# ---------------------------------------------------------------------------
def test_dx_seg_mask_handles_nchw_and_classmap():
    logits = np.zeros((1, 19, 4, 8), np.float32)
    logits[0, 10, :2] = 5.0  # 위: sky
    logits[0, 0, 2:] = 5.0  # 아래: road
    m = dx_seg_to_road_mask(logits, (8, 16), [0])
    assert m.shape == (8, 16) and m[6].all() and not m[1].any()
    classmap = np.argmax(logits, axis=1)  # (1, 4, 8) 클래스 번호
    np.testing.assert_array_equal(dx_seg_to_road_mask(classmap, (8, 16), [0]), m)


# ---------------------------------------------------------------------------
# DxModel, DXM1Backend (가짜 엔진)
# ---------------------------------------------------------------------------
def test_missing_model_error_says_where_to_put_it(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        DXM1Backend(str(tmp_path / "nope.dxnn"))
    msg = str(e.value)
    assert "nope.dxnn" in msg and "models/" in msg and "dx_det_model" in msg and "02_dxm1_backend.md 6장" in msg
    det = _touch(tmp_path, "det.dxnn")
    with pytest.raises(FileNotFoundError, match="dx_seg_model"):
        DXM1Backend(det, str(tmp_path / "seg.dxnn"), engine_factory=lambda p: FakeEngine(p))
    with pytest.raises(ValueError):
        DXM1Backend(None, None)


def test_dxmodel_reads_layout_and_prepares_input(tmp_path):
    p = _touch(tmp_path, "m.dxnn")
    nhwc = DxModel(p, engine_factory=lambda q: FakeEngine(q))
    assert (nhwc.layout, nhwc.input_hw, nhwc.input_dtype) == ("NHWC", (640, 640), np.uint8)
    img = np.random.randint(0, 255, (640, 640, 3), np.uint8)
    x = nhwc.prepare(img)
    assert x.shape == (1, 640, 640, 3) and x.dtype == np.uint8
    np.testing.assert_array_equal(x[0], img)
    # 정규화가 모델 밖에 남은 float NCHW 입력 모델
    f = DxModel(p, engine_factory=lambda q: FakeEngine(q, input_shape=(1, 3, 320, 480), input_dtype=np.float32))
    assert (f.layout, f.input_hw) == ("NCHW", (320, 480))
    y = f.prepare(np.full((320, 480, 3), 255, np.uint8))
    assert y.shape == (1, 3, 320, 480) and y.dtype == np.float32 and y.max() == pytest.approx(1.0)
    f.engine.run(y)  # 가짜 엔진의 dtype, 모양 검사를 통과해야 한다
    with pytest.raises(ValueError):
        f.prepare(np.zeros((10, 10, 3), np.uint8))


def _det_backend(tmp_path, frames_hw, box, **kw):
    """box(원본 좌표)에 차 한 대를 돌려주는 가짜 검출 엔진으로 백엔드를 만든다."""
    img = np.zeros(frames_hw + (3,), np.uint8)
    _, s, (px, py) = letterbox(img, (640, 640))
    x1, y1, x2, y2 = box[0] * s + px, box[1] * s + py, box[2] * s + px, box[3] * s + py
    raw = _raw_with_boxes([(50, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 2, 0.8)])
    engines = []

    def factory(p):
        engines.append(FakeEngine(p, out_fn=lambda x: [raw.copy()]))
        return engines[-1]

    be = DXM1Backend(_touch(tmp_path, "det.dxnn"), engine_factory=factory, **kw)
    return be, engines


def test_dxm1_backend_infer_two_cameras(tmp_path):
    tm = StageTimer(enabled=True)
    be, engines = _det_backend(tmp_path, (432, 768), (100, 200, 300, 400), timer=tm)
    frames = [np.zeros((432, 768, 3), np.uint8) for _ in range(2)]
    frames[0][:] = (255, 0, 0)  # BGR 파랑
    res = be.infer(frames)
    tm.end_frame(1.0)
    assert len(res) == 2
    for r in res:
        assert len(r.detections) == 1 and r.detections[0].label == "car"
        np.testing.assert_allclose(r.detections[0].box, (100, 200, 300, 400), atol=1e-3)
    # 엔진은 카메라마다 [1,640,640,3] uint8을 받았고, 파랑(BGR)은 RGB 입력에서 마지막 채널에 있어야 한다
    eng = engines[0]
    assert len(eng.inputs) == 2 and eng.inputs[0].shape == (1, 640, 640, 3)
    assert tuple(eng.inputs[0][0, 320, 320]) == (0, 0, 255)
    s = tm.summary()
    for name in ("preprocess", "dxm1_infer", "nms_parse"):
        assert s[name]["n"] == 1, name
    be.close()
    assert eng.disposed


def test_dxm1_backend_bgr_input_option(tmp_path):
    be, engines = _det_backend(tmp_path, (432, 768), (100, 200, 300, 400), input_rgb=False)
    f = np.zeros((432, 768, 3), np.uint8)
    f[:] = (255, 0, 0)
    be.infer([f])
    assert tuple(engines[0].inputs[0][0, 320, 320]) == (255, 0, 0)
    be.close()


# ---------------------------------------------------------------------------
# 장치 후처리(PPU) 모델: DEEPX yolov8-s_640x640_ppu.dxnn에서 확인한 출력 형식
#   (1, N, 32) uint8, 상자 없으면 (0, 0). 32바이트 = DeviceBoundingBox_t
# ---------------------------------------------------------------------------
def _ppu_raw(rows):
    """rows: [(cx, cy, w, h, 점수, 라벨)] → 진짜 엔진처럼 (1, N, 32) uint8."""
    if not rows:
        return np.zeros((0, 0), np.uint8)
    b = np.zeros(len(rows), PPU_BBOX_DTYPE)
    for i, (cx, cy, w, h, sc, lb) in enumerate(rows):
        b[i]["x"], b[i]["y"], b[i]["w"], b[i]["h"] = cx, cy, w, h
        b[i]["score"], b[i]["label"] = sc, lb
        b[i]["layer_idx"] = 1
    return b.view(np.uint8).reshape(1, len(rows), 32)


def test_ppu_struct_is_32_bytes_and_reads_fields():
    raw = _ppu_raw([(100.0, 200.0, 40.0, 60.0, 0.9, 2)])
    assert raw.shape == (1, 1, 32) and raw.dtype == np.uint8
    b = ppu_boxes(raw)
    assert b["x"][0] == 100.0 and b["h"][0] == 60.0 and b["label"][0] == 2
    np.testing.assert_allclose(b["score"][0], 0.9, rtol=1e-6)
    # 상자가 하나도 없을 때 진짜 엔진은 (0, 0) 배열을 준다
    assert len(ppu_boxes(np.zeros((0, 0), np.uint8))) == 0
    with pytest.raises(ValueError, match="32"):
        ppu_boxes(np.zeros((1, 3, 6), np.float32))


def test_parse_ppu_filters_nms_and_undoes_letterbox():
    # 레터박스 배율 0.5, 패딩 (0, 80): 원본 (x - 0) / 0.5, (y - 80) / 0.5
    rows = [
        (100.0, 180.0, 40.0, 40.0, 0.9, 2),  # 차
        (101.0, 181.0, 40.0, 40.0, 0.8, 2),  # 같은 차와 겹침: NMS로 빠진다
        (100.0, 180.0, 40.0, 40.0, 0.7, 0),  # 같은 자리 사람: 다른 클래스라 남는다
        (300.0, 300.0, 20.0, 20.0, 0.3, 2),  # 점수 미달
        (400.0, 400.0, 20.0, 20.0, 0.9, 56),  # 의자: COCO_KEEP 밖
    ]
    dets = parse_dx_ppu_detections([_ppu_raw(rows)], 0.5, (0, 80), score_thr=0.4)
    assert [d.label for d in dets] == ["car", "person"]
    np.testing.assert_allclose(dets[0].box, (160, 160, 240, 240), atol=1e-4)
    assert parse_dx_ppu_detections([_ppu_raw([])], 1.0, (0, 0), 0.4) == []
    # labels를 주면 그 클래스만(번호 → 이름)
    d2 = parse_dx_ppu_detections([_ppu_raw(rows)], 0.5, (0, 80), 0.4, labels={56: "chair"})
    assert [d.label for d in d2] == ["chair"]


def test_dxm1_backend_reads_ppu_model(tmp_path):
    raw = _ppu_raw([(320.0, 320.0, 64.0, 32.0, 0.85, 2)])
    be = DXM1Backend(
        _touch(tmp_path, "ppu.dxnn"),
        engine_factory=lambda p: FakeEngine(p, outputs=[("BBOX", [8400])], ppu=True, out_fn=lambda x: [raw.copy()]),
    )
    f = np.zeros((640, 640, 3), np.uint8)  # 레터박스가 없는 크기라 좌표가 그대로
    res = be.infer([f, f])
    assert [len(r.detections) for r in res] == [1, 1]
    np.testing.assert_allclose(res[0].detections[0].box, (288, 304, 352, 336), atol=1e-4)
    be.close()


def test_dxm1_backend_rejects_yolov5_ppu_model(tmp_path):
    be = DXM1Backend(
        _touch(tmp_path, "ppu.dxnn"),
        engine_factory=lambda p: FakeEngine(p, outputs=[("BBOX", [8400])], ppu=True,
                                            out_fn=lambda x: [np.zeros((0, 0), np.uint8)]),
        det_format="yolov5",
    )
    frames = [np.zeros((432, 768, 3), np.uint8)]
    with pytest.raises(ValueError, match=r"PPU.*BBOX\[8400\].*infer_raw"):
        be.infer(frames)
    assert be.infer_raw(frames)[0][0].shape == (0, 0)  # 원시 출력은 그대로 받을 수 있다
    be.close()


def test_decode_keeps_probabilities_with_tiny_negative_values():
    # DEEPX yolov8-s_640x640.dxnn 실측: 확률 출력인데 최솟값이 -1.2e-7. 시그모이드를 또 씌우면
    # 점수 0인 앵커 8400개가 모두 0.5가 되어 문턱값 0.4를 넘는다.
    raw = _raw_with_boxes([(10, 100, 100, 50, 50, 2, 0.9)])
    raw[0, 4 + 5, 20] = -1.1920929e-07
    rows = decode_yolo_raw(raw, (640, 640), 0.4)
    assert len(rows) == 1 and rows[0][0] == 2
    np.testing.assert_allclose(rows[0][1], 0.9, rtol=1e-6)


def test_dx_seg_uint16_classmap_like_bisenetv2():
    # BiSeNetV2 실측 출력: (1, 1024, 2048) uint16 클래스 번호. 줄인 뒤 비교해도 결과가 같아야 한다
    rng = np.random.default_rng(0)
    seg = rng.integers(0, 19, size=(1, 1024, 2048)).astype(np.uint16)
    seg[0, :512] = 0  # 위쪽 절반은 도로
    m = dx_seg_to_road_mask(seg, (432, 768), [0])
    assert m.shape == (432, 768) and m.dtype == np.uint8
    import cv2

    ref = cv2.resize(np.isin(seg[0].astype(np.int32), [0]).astype(np.uint8), (768, 432), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(m, ref)
    m2 = dx_seg_to_road_mask(seg, (432, 768), [0, 1])
    ref2 = cv2.resize(np.isin(seg[0].astype(np.int32), [0, 1]).astype(np.uint8), (768, 432), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(m2, ref2)
    # int64 클래스 맵(cv2.resize가 못 받는 자료형)도 같은 결과
    m3 = dx_seg_to_road_mask(seg.astype(np.int64), (432, 768), [0])
    np.testing.assert_array_equal(m3, ref)


def _seg_factory(hw=(64, 128), nc=19):
    """아래 절반을 도로(0)로 내는 NCHW 세그 가짜 엔진."""

    def out_fn(x):
        lg = np.zeros((1, nc, hw[0] // 4, hw[1] // 4), np.float32)
        lg[0, 10, : hw[0] // 8] = 5.0
        lg[0, 0, hw[0] // 8 :] = 5.0
        return [lg]

    return lambda p: FakeEngine(p, input_shape=(1,) + hw + (3,), outputs=[("seg", [1, nc, hw[0] // 4, hw[1] // 4])], out_fn=out_fn)


def test_dxm1_seg_only_backend(tmp_path):
    be = DXM1Backend(None, _touch(tmp_path, "seg.dxnn"), engine_factory=_seg_factory())
    frames = [np.zeros((432, 768, 3), np.uint8)] * 2
    masks = be.segment(frames)
    assert len(masks) == 2 and masks[0].shape == (432, 768)
    assert masks[0][400].all() and not masks[0][10].any()
    raw = be.infer_raw(frames, "seg")
    assert len(raw) == 2 and raw[0][0].shape == (1, 19, 16, 32)
    with pytest.raises(RuntimeError, match="dx_det_model"):
        be.infer(frames)
    be.close()


def test_dxm1_backend_with_seg_reuses_mask(tmp_path):
    engines = []

    def factory(p):
        e = _seg_factory()(p) if p.endswith("seg.dxnn") else FakeEngine(p)
        engines.append(e)
        return e

    be = DXM1Backend(_touch(tmp_path, "det.dxnn"), _touch(tmp_path, "seg.dxnn"), seg_every=3, engine_factory=factory)
    frames = [np.zeros((432, 768, 3), np.uint8)]
    for _ in range(4):
        r = be.infer(frames)
        assert r[0].road_mask is not None
    # 4프레임 중 세그는 0, 3번째에만
    assert len(engines[1].inputs) == 2
    be.close()


# ---------------------------------------------------------------------------
# 두 NPU 동시 실행: 별도 스레드 세그 스케줄
# ---------------------------------------------------------------------------
class GatedSeg:
    """gate가 열릴 때까지 segment()가 멈춰 있는 가짜 세그 백엔드(시간에 기대지 않는 시험용)."""

    def __init__(self, fail=False):
        self.gate = threading.Event()
        self.calls = []
        self.started = threading.Event()
        self.closed = False
        self.fail = fail
        self.thread_names = set()

    def segment(self, frames):
        self.thread_names.add(threading.current_thread().name)
        self.calls.append(len(frames))
        self.started.set()
        if not self.gate.wait(10):
            raise TimeoutError("gate가 열리지 않음")
        if self.fail:
            raise RuntimeError("세그 실패")
        return [np.ones(f.shape[:2], np.uint8) for f in frames]

    def close(self):
        self.closed = True


def _sim_pipe(**kw):
    cams, grid, _ = load_config(os.path.join(ROOT, "configs/two_cam.yaml"))
    world = SimWorld(cams)
    return world, RealityViewPipeline(cams, SimBackend(world, noise_px=1.0, miss_rate=0.0), grid, **kw)


def test_pipeline_default_has_no_seg_thread():
    _, pipe = _sim_pipe()
    assert pipe.seg_stats is None and pipe._seg is None
    pipe.close()


def test_pipeline_async_seg_schedule_and_mask_reuse():
    seg = GatedSeg()
    tm = StageTimer(enabled=True)
    world, pipe = _sim_pipe(seg_backend=seg, seg_every=3, timer=tm)
    safety = threading.Timer(15, seg.gate.set)  # 시험이 잘못돼도 영원히 멈추지 않게
    safety.start()
    try:
        # 0~6 프레임: 세그 스레드는 첫 작업(0번 프레임)에서 막혀 있다. 메인 루프는 기다리지 않아야 한다
        t0 = time.perf_counter()
        for _ in range(7):
            world.step(0.1)
            out = pipe.step(world.render(), 0.1)
            # 아직 세그 결과가 없으니 시뮬레이터가 만든 마스크(하늘은 0)가 그대로 있다
            assert all(not r.road_mask.all() for r in out.results)
            tm.end_frame(1.0)
        assert time.perf_counter() - t0 < 10, "메인 루프가 세그를 기다렸다"
        assert seg.started.is_set() and seg.thread_names == {"seg"}
        # 0번은 실행 중, 3번은 대기 중에 6번에 밀려 버려진다
        st = pipe.seg_stats
        assert (st["submitted"], st["done"], st["dropped"]) == (3, 0, 1)
        assert st["frames_without_mask"] == 7 and st["mask_age_mean"] is None

        # 세그를 풀어 주면 0번, 6번 순서로 끝난다. 7번 프레임은 6번 영상의 마스크(1프레임 묵음)를 쓴다
        seg.gate.set()
        assert pipe._seg.wait_idle(5)
        world.step(0.1)
        out = pipe.step(world.render(), 0.1)
        tm.end_frame(1.0)
        # 세그 결과(모두 1)가 시뮬레이터 마스크를 대신한다
        assert all(r.road_mask.all() and r.road_mask.shape == f.shape[:2]
                   for r, f in zip(out.results, world.render()))
        # 8번 프레임: 새 작업이 없으니 같은 마스크를 다시 쓴다(2프레임 묵음)
        world.step(0.1)
        pipe.step(world.render(), 0.1)
        tm.end_frame(1.0)
        st = pipe.seg_stats
        assert (st["done"], st["mask_age_mean"], st["mask_age_max"]) == (2, 1.5, 2)
        assert seg.calls == [2, 2]  # 카메라 2대분씩 두 번
        assert tm.summary()["seg_async"]["n"] >= 1
    finally:
        safety.cancel()
        pipe.close()
    assert seg.closed and not pipe._seg._thread.is_alive()


def test_pipeline_async_seg_error_is_raised_in_main_loop():
    seg = GatedSeg(fail=True)
    seg.gate.set()
    world, pipe = _sim_pipe(seg_backend=seg, seg_every=1)
    try:
        world.step(0.1)
        pipe.step(world.render(), 0.1)
        pipe._seg.wait_idle(5)
        with pytest.raises(RuntimeError, match="세그 스레드"):
            pipe.step(world.render(), 0.1)
    finally:
        pipe.close()


def test_stage_names_for_dxm1_have_labels():
    for name in ("dxm1_infer", "seg_async"):
        assert name in STAGE_ORDER and name in STAGE_LABEL_KO


# ---------------------------------------------------------------------------
# run_demo.py와 설정 파일
# ---------------------------------------------------------------------------
def _load_run_demo():
    spec = importlib.util.spec_from_file_location("run_demo", os.path.join(ROOT, "scripts", "run_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_demo_dxm1_options():
    mod = _load_run_demo()
    a = mod.build_parser().parse_args([])
    assert (a.backend, a.seg_npu) == ("sim", "none")  # 기본 동작은 그대로
    a = mod.build_parser().parse_args(["--backend", "dxm1", "--seg-npu", "dxm1"])
    assert (a.backend, a.seg_npu) == ("dxm1", "dxm1")


def test_run_demo_seg_npu_without_model_explains(monkeypatch, capsys):
    mod = _load_run_demo()
    monkeypatch.chdir(ROOT)
    argv = ["run_demo.py", "--backend", "sim", "--headless", "--frames", "2", "--seg-npu", "dxm1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "dx_seg_model" in str(e.value) and "models/" in str(e.value)


@pytest.mark.parametrize("cfg", ["two_cam.yaml", "two_cam_current.yaml", "four_cam.yaml", "two_cam_model.yaml"])
def test_configs_have_dxm1_keys(cfg):
    _, _, c = load_config(os.path.join(ROOT, "configs", cfg))
    npu = c["npu"]
    assert npu["dx_det_model"].startswith("models/") and npu["dx_det_model"].endswith(".dxnn")
    assert npu["dx_det_format"] in ("yolov8", "yolov5")
    assert npu["dx_seg_channels_first"] is True
    if cfg in ("two_cam_current.yaml", "two_cam_model.yaml"):
        # 지금 달린 카메라 설정은 DEEPX 공식 모델 저장소에서 받은 실제 모델을 가리킨다
        assert npu["dx_det_model"] == "models/dxnn/yolov8-s_640x640_ppu.dxnn"
        assert npu["dx_seg_model"] == "models/dxnn/bisenetv2_1024x2048.dxnn"
        assert npu["road_classes"] == [0]  # Cityscapes road
    else:
        assert npu["dx_seg_model"] is None


# ---------------------------------------------------------------------------
# 파이 전용: 진짜 dx_engine (맥에서는 건너뜀)
# ---------------------------------------------------------------------------
def test_real_dx_engine_import_and_device():
    dx = pytest.importorskip("dx_engine")
    from dx_engine import Configuration, DeviceStatus

    assert dx.__version__.startswith("3.")
    assert Configuration().get_version()
    n = DeviceStatus.get_device_count()
    assert n >= 1
    st = DeviceStatus.get_current_status(0)
    assert 0 < st.get_temperature(0) < 125


@pytest.mark.skipif(not os.environ.get("RV_DX_TEST_MODEL"), reason="RV_DX_TEST_MODEL(.dxnn 경로)이 없으면 건너뜀")
def test_real_dx_model_raw_outputs():
    pytest.importorskip("dx_engine")
    path = os.environ["RV_DX_TEST_MODEL"]
    be = DXM1Backend(path, input_rgb=False)
    try:
        frames = [np.random.randint(0, 255, (432, 768, 3), np.uint8), np.random.randint(0, 255, (576, 768, 3), np.uint8)]
        raw = be.infer_raw(frames, "det")
        assert len(raw) == 2
        for outs in raw:
            if be.det.is_ppu:
                # PPU 모델은 정보가 [8400]이라고 해도 실제로는 (1, N, 32) uint8 또는 (0, 0)을 준다
                assert outs[0].dtype == np.uint8 and (outs[0].size == 0 or outs[0].shape[-1] == 32)
            else:
                assert [list(o.shape) for o in outs] == be.det.output_shapes
        if len(be.det.output_names) > 1:
            with pytest.raises(ValueError, match="infer_raw"):
                be.infer(frames)
    finally:
        be.close()


# 파이에 DEEPX 공식 모델이 있으면(models/dxnn/) 실제 모델로 검출과 세그를 한 번씩 돌린다
_DXNN = os.path.join(ROOT, "models", "dxnn")


@pytest.mark.skipif(not os.path.isfile(os.path.join(_DXNN, "yolov8-s_640x640_ppu.dxnn")), reason="models/dxnn에 PPU 모델이 없으면 건너뜀")
def test_real_ppu_and_raw_models_agree():
    pytest.importorskip("dx_engine")
    from realityview.types import COCO_ALL

    rng = np.random.default_rng(1)
    black = np.zeros((432, 768, 3), np.uint8)
    for name in ("yolov8-s_640x640_ppu.dxnn", "yolov8-s_640x640.dxnn"):
        path = os.path.join(_DXNN, name)
        if not os.path.isfile(path):
            continue
        be = DXM1Backend(path, labels=COCO_ALL, score_thr=0.25)
        try:
            res = be.infer([black, rng.integers(0, 255, (576, 768, 3), dtype=np.uint8)])
            assert len(res) == 2 and res[0].detections == []  # 검은 영상은 상자 0개(PPU는 (0, 0) 출력)
        finally:
            be.close()


@pytest.mark.skipif(not os.path.isfile(os.path.join(_DXNN, "bisenetv2_1024x2048.dxnn")), reason="models/dxnn에 BiSeNetV2가 없으면 건너뜀")
def test_real_bisenetv2_segment_shape():
    pytest.importorskip("dx_engine")
    be = DXM1Backend(None, os.path.join(_DXNN, "bisenetv2_1024x2048.dxnn"))
    try:
        frames = [np.zeros((576, 768, 3), np.uint8), np.zeros((432, 768, 3), np.uint8)]
        masks = be.segment(frames)
        assert [m.shape for m in masks] == [(576, 768), (432, 768)]
        assert all(m.dtype == np.uint8 and m.max() <= 1 for m in masks)
    finally:
        be.close()
