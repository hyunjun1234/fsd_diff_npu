from __future__ import annotations

import os
from typing import Protocol, Sequence

import cv2
import numpy as np

from .profiling import StageTimer
from .types import COCO_KEEP, Detection, FrameResult

# 카메라별 검출 + 세그멘테이션 백엔드.
# - HailoBackend: 라즈베리파이 5 + Hailo-8/8L. Hailo Model Zoo의 HEF를 그대로 쓴다.
#   (검출: yolov8s/yolov8m 등 NMS 내장 HEF, 세그: Cityscapes 학습 시맨틱 세그 HEF 선택)
# - DXM1Backend: 라즈베리파이 5 + DEEPX DX-M1. .dxnn을 dx_engine으로 돌리고 NMS는 CPU에서 한다.
# - UltralyticsBackend: PC 개발용. ultralytics YOLO로 검출만 한다.
# - 시뮬레이션은 sim.SimBackend 참고.


class Backend(Protocol):
    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]: ...

    def close(self) -> None: ...


class SegBackend(Protocol):
    """도로 세그만 하는 백엔드. 파이프라인의 별도 세그 스레드가 쓴다(예: DXM1Backend)."""

    def segment(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# 공통 전처리
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size_hw: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """비율 유지 리사이즈 + 패딩. (출력, 배율, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    th, tw = size_hw
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((th, tw, 3), 114, dtype=np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    out[py : py + nh, px : px + nw] = resized
    return out, s, (px, py)


def parse_hailo_nms(
    nms_out,
    size_hw: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    score_thr: float,
    labels: dict | None = None,
) -> list[Detection]:
    """HailoRT NMS(by class) 출력 → Detection.

    nms_out: 클래스 개수만큼의 배열 목록, 각 배열은 (n, 5) = [ymin, xmin, ymax, xmax, score] (0~1 정규화).
    labels: 클래스 번호 → 이름. 없으면 주행에 필요한 COCO_KEEP만 남긴다.
    """
    th, tw = size_hw
    keep = COCO_KEEP if labels is None else labels
    dets = []
    for cls_id, arr in enumerate(nms_out):
        label = keep.get(cls_id)
        if label is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        for ymin, xmin, ymax, xmax, score in arr.reshape(-1, 5):
            if score < score_thr:
                continue
            x1 = (xmin * tw - pad[0]) / scale
            y1 = (ymin * th - pad[1]) / scale
            x2 = (xmax * tw - pad[0]) / scale
            y2 = (ymax * th - pad[1]) / scale
            dets.append(Detection(label, float(score), (float(x1), float(y1), float(x2), float(y2))))
    return dets


def seg_to_road_mask(
    seg_out: np.ndarray,
    out_hw: tuple[int, int],
    road_classes: Sequence[int],
    channels_first: bool = False,
) -> np.ndarray:
    """세그 출력(H,W,C 로짓 또는 H,W 클래스맵) → 원본 크기의 도로 마스크(uint8 0/1).

    channels_first=True면 (C,H,W) 로짓으로 본다. DX-M1은 출력이 NCHW로 나오므로 이 값을 켠다
    (Hailo는 NHWC라 기본값 False 그대로).
    """
    seg = np.asarray(seg_out)
    seg = np.squeeze(seg)
    if seg.ndim == 3:
        cls = np.argmax(seg, axis=0 if channels_first else -1)
    else:
        cls = seg.astype(np.int32)
    mask = np.isin(cls, list(road_classes)).astype(np.uint8)
    return cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Hailo
# ---------------------------------------------------------------------------
class _HefModel:
    """HEF 하나를 VDevice 스케줄러 위에 올려 둔다(여러 모델 동시 사용 가능)."""

    def __init__(self, vdevice, hef_path: str, output_float: bool = True):
        from hailo_platform import (  # type: ignore
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )

        self.hef = HEF(hef_path)
        cfg = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = vdevice.configure(self.hef, cfg)[0]
        in_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = in_info.name
        self.input_hw = tuple(in_info.shape[:2])
        self.output_names = [o.name for o in self.hef.get_output_vstream_infos()]
        in_params = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(
            self.ng, format_type=FormatType.FLOAT32 if output_float else FormatType.UINT8
        )
        self._pipe_ctx = InferVStreams(self.ng, in_params, out_params)
        self.pipe = self._pipe_ctx.__enter__()

    def infer(self, batch: np.ndarray) -> dict:
        return self.pipe.infer({self.input_name: np.ascontiguousarray(batch)})

    def close(self) -> None:
        self._pipe_ctx.__exit__(None, None, None)


# 파이의 /usr/share/hailo-models 에 있는 Hailo-8용 HEF(2026-11-24 확인)와 이 백엔드에서 쓸 수 있는지.
#   yolov8s_h8.hef    검출, NMS 내장(점수 0.2, IoU 0.7)              det_hef로 씀(기본)
#   yolov6n_h8.hef    검출, NMS 내장(점수 0.2, IoU 0.65), 약 2배 빠름  det_hef로 쓸 수 있음(--det-hef)
#   yolov5n_seg_h8.hef 인스턴스 세그(COCO, 출력 4개, CPU 후처리 필요)   도로 세그가 아니라 seg_hef로 못 씀
#   yolov8s_pose_h8.hef 자세 추정                                        쓰지 않음
# 도로 세그(예: Hailo Model Zoo의 stdc1)나 깊이(scdepthv3) HEF는 파이에 없다.
def check_hailo_seg_outputs(output_names: Sequence[str], hef_path: str = "") -> None:
    """seg_hef는 출력 1개(클래스 로짓이나 클래스 번호 맵)인 시맨틱 세그 HEF만 받는다.

    yolov5n_seg처럼 출력이 여럿인 인스턴스 세그 HEF를 넣으면 첫 출력만 argmax해서 엉뚱한 마스크가 나오므로 막는다.
    """
    if len(output_names) != 1:
        raise ValueError(
            f"seg_hef는 출력이 1개인 시맨틱 세그 HEF여야 합니다: {hef_path} 출력 {len(output_names)}개 {list(output_names)}. "
            "yolov5n_seg 같은 인스턴스 세그 HEF는 도로 마스크를 내지 않습니다."
        )


class HailoBackend:
    def __init__(
        self,
        det_hef: str,
        seg_hef: str | None = None,
        score_thr: float = 0.4,
        road_classes: Sequence[int] = (0,),  # Cityscapes trainId 0 = road
        seg_every: int = 3,  # 세그는 N 프레임마다 한 번(도로는 천천히 변한다)
        timer: StageTimer | None = None,  # 단계별 시간 측정(기본 꺼짐)
        labels: dict | None = None,  # 남길 클래스(번호 → 이름). None이면 COCO_KEEP
    ):
        from hailo_platform import HailoSchedulingAlgorithm, VDevice  # type: ignore

        # HEF가 없으면 HailoRT 안쪽에서 알아보기 힘든 오류가 나므로 먼저 확인한다
        for p in (det_hef, seg_hef):
            if p and not os.path.isfile(p):
                raise FileNotFoundError(
                    f"HEF 파일이 없습니다: {p} (예: mkdir -p models && cp /usr/share/hailo-models/yolov8s_h8.hef models/)"
                )
        self.timer = timer or StageTimer(enabled=False)
        self.labels = labels
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.det = _HefModel(self.vdevice, det_hef)
        self.seg = _HefModel(self.vdevice, seg_hef) if seg_hef else None
        if self.seg is not None:
            check_hailo_seg_outputs(self.seg.output_names, seg_hef)
        self.score_thr = score_thr
        self.road_classes = tuple(road_classes)
        self.seg_every = max(1, seg_every)
        self._frame_idx = 0
        self._last_masks: list[np.ndarray | None] = []

    def chip_temperature(self) -> float | None:
        """Hailo-8 칩 온도(도, 센서 두 개 중 높은 값). 읽지 못하면 None.

        추론 중에도 제어 채널로 읽을 수 있다. 30분 연속 실행에서 방열 없이 120도까지 올라
        HailoRT가 클럭을 낮추고 스트림을 닫은 적이 있어, 장시간 실행 기록에 함께 남긴다.
        """
        try:
            dev = self.vdevice.get_physical_devices()[0]
            t = dev.control.get_chip_temperature()
            return round(float(max(t.ts0_temperature, t.ts1_temperature)), 1)
        except Exception:
            return None

    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]:
        # 카메라 N대 영상을 한 배치로 넣는다. frames는 BGR.
        tm = self.timer
        # ---- 전처리: BGR->RGB, 비율 유지 리사이즈 + 패딩 ----
        with tm.stage("preprocess"):
            boxes, metas = [], []
            for f in frames:
                lb, s, pad = letterbox(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), self.det.input_hw)
                boxes.append(lb)
                metas.append((s, pad))
            batch = np.stack(boxes)
        # ---- Hailo 추론(장치로 보내고 결과 받기까지) ----
        with tm.stage("hailo_infer"):
            out = self.det.infer(batch)
        # ---- NMS 출력 → 원본 영상 좌표의 Detection ----
        with tm.stage("nms_parse"):
            nms = out[self.det.output_names[0]]
            results = []
            for i, (s, pad) in enumerate(metas):
                dets = parse_hailo_nms(nms[i], self.det.input_hw, s, pad, self.score_thr, self.labels)
                results.append(FrameResult(detections=dets))

        if self.seg is not None:
            if self._frame_idx % self.seg_every == 0 or len(self._last_masks) != len(frames):
                with tm.stage("seg"):
                    th, tw = self.seg.input_hw
                    batch = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (tw, th)) for f in frames])
                    sout = self.seg.infer(batch)[self.seg.output_names[0]]
                    self._last_masks = [
                        seg_to_road_mask(sout[i], frames[i].shape[:2], self.road_classes) for i in range(len(frames))
                    ]
            for r, m in zip(results, self._last_masks):
                r.road_mask = m
        self._frame_idx += 1
        return results

    def close(self) -> None:
        self.det.close()
        if self.seg is not None:
            self.seg.close()


# ---------------------------------------------------------------------------
# DX-M1 (DEEPX): 모델 파일 확인
# ---------------------------------------------------------------------------
# 파이의 dx_engine 3.4.2에서 확인한 사실(2026-11-24, 드론 검출 .dxnn으로 시험)
#   - 입력: get_input_tensors_info()가 [1, H, W, 3] uint8(NHWC)로 보고했다. ONNX가 NCHW float32여도
#     DX-COM이 정규화를 모델 안에 넣고 입력을 NHWC uint8로 바꾼다. 그래서 형식은 늘 장치에 물어서 정한다.
#   - 출력: run()과 wait()가 get_output_tensors_info() 순서대로 numpy 배열 목록을 돌려준다(NCHW float32).
#   - 카메라 여러 대: run()에 배열 목록을 주면 배치로 보고 출력 버퍼를 요구한다. 그래서 카메라마다
#     run_async()를 보내고 wait()로 차례대로 받는다(동기 1개씩보다 약 2배 빨랐다).
DX_MODEL_HELP = (
    "파이 저장소의 models/ 폴더에 .dxnn 파일을 두고 설정 파일 npu.{key} 에 경로를 적으세요 "
    "(예: models/dxnn/yolov8-s_640x640_ppu.dxnn). DEEPX 공식 모델 저장소에서 받는 주소와 sha256은 "
    "out/measure/05_dxnn_sources.md에, 직접 만드는 방법은 out/measure/02_dxm1_backend.md 6장에 있습니다."
)


def _require_dx_model(path: str, key: str) -> None:
    """모델 파일이 없으면 무엇을 어디에 둘지 알려 주는 오류를 낸다.

    dx_engine을 불러오기 전에 확인하므로 맥처럼 dx_engine이 없는 곳에서도 같은 안내가 나온다.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DX-M1 모델 파일이 없습니다: {path}\n" + DX_MODEL_HELP.format(key=key))


# ---------------------------------------------------------------------------
# DX-M1: CPU 후처리 (NMS 전 원시 출력 → Detection)
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# 확률(0~1)인지 로짓인지 가를 때 봐 주는 양자화 오차 폭
_PROB_TOL = 1e-3


def _looks_like_logits(x: np.ndarray) -> bool:
    """값이 0~1 범위를 _PROB_TOL 넘게 벗어나면 시그모이드 전 로짓으로 본다."""
    return bool(x.size) and (float(x.min()) < -_PROB_TOL or float(x.max()) > 1.0 + _PROB_TOL)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int = 100) -> np.ndarray:
    """한 무리의 박스에 대한 탐욕 NMS. boxes는 (N,4) xyxy. 남길 번호를 점수 높은 순으로 돌려준다."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        # 가장 점수 높은 박스와 나머지의 IoU를 한꺼번에 계산
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float, max_det: int = 100
) -> np.ndarray:
    """클래스별 NMS. 클래스마다 박스를 멀리 떨어뜨려 놓고 한 번에 NMS를 돌린다
    (다른 클래스끼리는 겹쳐도 지우지 않는다)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    offset = classes.astype(np.float64)[:, None] * (float(np.abs(boxes).max()) + 1.0)
    return nms_numpy(boxes + offset, scores, iou_thr, max_det)


def decode_yolo_raw(
    out: np.ndarray,
    input_hw: tuple[int, int],
    score_thr: float,
    iou_thr: float = 0.45,
    fmt: str = "yolov8",
    keep_classes: Sequence[int] | None = None,
    max_det: int = 100,
) -> list[tuple[int, float, float, float, float, float]]:
    """YOLO 원시 출력 하나를 NMS까지 마친 (클래스, 점수, x1, y1, x2, y2) 목록으로 바꾼다.

    좌표는 모델 입력(레터박스된 영상) 픽셀 기준이다.
    out: (1, C, N) 또는 (1, N, C). C는 yolov8이면 4+클래스 수, yolov5면 5+클래스 수.
         상자 개수 N(640 입력이면 8400)이 C보다 훨씬 크므로 짧은 축을 C로 본다.
    fmt: "yolov8" = [cx, cy, w, h, 클래스 점수...], "yolov5" = [cx, cy, w, h, 물체 점수, 클래스 점수...]
    keep_classes: 남길 클래스 번호. None이면 모두.
    """
    if fmt not in ("yolov8", "yolov5"):
        raise ValueError(f"모르는 검출 출력 형식입니다: {fmt} (yolov8 또는 yolov5)")
    a = np.asarray(out, dtype=np.float32)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]  # 배치 축(1) 없애기
    if a.ndim != 2:
        raise ValueError(f"YOLO 원시 출력은 (1, C, N) 또는 (1, N, C)여야 합니다. 받은 모양: {np.shape(out)}")
    if a.shape[0] < a.shape[1]:
        a = a.T  # (C, N) → (N, C)
    off = 4 if fmt == "yolov8" else 5
    if a.shape[1] <= off:
        raise ValueError(f"{fmt} 출력의 채널 수({a.shape[1]})가 너무 작습니다. 받은 모양: {np.shape(out)}")

    # ---- 점수: 로짓(0~1 밖의 값)으로 나오면 시그모이드를 씌운다 ----
    # DEEPX 공식 yolov8-s_640x640.dxnn은 이미 시그모이드를 거친 확률을 주는데, 양자화 때문에
    # 최솟값이 -1.2e-7처럼 0보다 아주 조금 작게 나온다(2026-11-24 파이 실측). 0을 기준으로 가르면
    # 확률에 시그모이드를 한 번 더 씌워 모든 상자가 0.5 이상이 되므로 여유(_PROB_TOL)를 둔다.
    cls = a[:, off:]
    if _looks_like_logits(cls):
        cls = _sigmoid(cls)
    if fmt == "yolov5":
        obj = a[:, 4]
        if _looks_like_logits(obj):
            obj = _sigmoid(obj)
        cls = cls * obj[:, None]
    cls_id = np.argmax(cls, axis=1)
    score = cls[np.arange(len(cls)), cls_id]

    # ---- 점수와 클래스로 먼저 거른다(NMS에 들어갈 수를 줄여 CPU 절약) ----
    m = score >= score_thr
    if keep_classes is not None:
        m &= np.isin(cls_id, list(keep_classes))
    if not m.any():
        return []

    # ---- 상자: 0~1 정규화 좌표로 나오면 입력 픽셀로 늘린다 ----
    # 앵커 8400개의 중심은 영상 전체에 퍼져 있어서 픽셀 좌표라면 최댓값이 2를 넘는다
    xywh = a[:, :4]
    if float(np.abs(xywh[:, :2]).max()) <= 2.0:
        th, tw = input_hw
        xywh = xywh * np.array([tw, th, tw, th], dtype=np.float32)
    xywh, score, cls_id = xywh[m], score[m], cls_id[m]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    keep = batched_nms(boxes, score, cls_id, iou_thr, max_det)
    return [
        (int(cls_id[k]), float(score[k]), *(float(v) for v in boxes[k]))  # type: ignore[misc]
        for k in keep
    ]


def parse_dx_detections(
    outputs: Sequence[np.ndarray],
    output_names: Sequence[str],
    input_hw: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    score_thr: float,
    iou_thr: float = 0.45,
    fmt: str = "yolov8",
    labels: dict | None = None,
    max_det: int = 100,
) -> list[Detection]:
    """DX-M1 검출 모델 출력 → 원본 영상 좌표의 Detection.

    지금은 출력 텐서가 하나인 YOLO 원시 출력(NMS 전)만 해석한다. 출력이 여러 개인 모델
    (스케일별 reg/obj/cls로 나뉜 헤드, DFL 분포 그대로, 장치 후처리(PPU) 결과)은 형식이 모델마다
    달라 추측하지 않고 오류를 낸다. 그런 모델은 DXM1Backend.infer_raw()로 원시 출력을 받아 쓴다.
    """
    if len(outputs) != 1:
        desc = ", ".join(f"{n}{list(np.shape(o))}" for n, o in zip(output_names, outputs))
        raise ValueError(
            f"출력 텐서가 {len(outputs)}개라 YOLO 원시 출력으로 해석할 수 없습니다: {desc}. "
            "DXM1Backend.infer_raw()로 원시 출력을 받아 모델에 맞는 후처리를 쓰세요."
        )
    keep = COCO_KEEP if labels is None else labels
    rows = decode_yolo_raw(outputs[0], input_hw, score_thr, iou_thr, fmt, list(keep), max_det)
    dets = []
    for cls_id, score, x1, y1, x2, y2 in rows:
        # 레터박스를 되돌려 원본 영상 픽셀로 (parse_hailo_nms와 같은 식)
        dets.append(
            Detection(
                keep[cls_id],
                score,
                ((x1 - pad[0]) / scale, (y1 - pad[1]) / scale, (x2 - pad[0]) / scale, (y2 - pad[1]) / scale),
            )
        )
    return dets


# ---------------------------------------------------------------------------
# DX-M1: 장치 후처리(PPU) 모델 출력 해석
# ---------------------------------------------------------------------------
# DEEPX 공식 yolov8-s_640x640_ppu.dxnn(모델 저장소 2_4_0)으로 확인한 사실(2026-11-24, 파이)
#   - get_output_tensors_info()는 이름 BBOX, dtype 'BBOX'(numpy가 모르는 이름), shape [8400]을 보고한다.
#   - 실제 run()/wait()는 (1, N, 32) uint8 배열 하나를 준다. N은 장치 문턱값을 넘은 상자 수이고,
#     상자가 하나도 없으면(검은 영상) 모양이 (0, 0)이다.
#   - 32바이트는 DX-RT 헤더(/usr/local/include/dxrt/dxrt_cxx_api.h)의 DeviceBoundingBox_t와 같다.
#       float x, y, w, h | uint8 grid_y, grid_x, box_idx, layer_idx | float score | uint32 label | 빈칸 4
#   - YOLOv8 PPU는 x, y가 상자 중심, w, h가 폭과 높이이고 모두 모델 입력(레터박스 영상) 픽셀이다.
#     DEEPX dx_app v3.2.2의 YOLOv8PPUPostprocessor도 앵커나 stride 계산 없이 이 값을 그대로 쓴다.
#   - NMS는 장치에서 하지 않으므로 CPU에서 한다(dx_app도 CPU NMS).
#   - 장치 쪽 점수 문턱값은 0.001이다. 모델 안 compile_config의 ppu.conf_thres가 0.001이고, 실제 사진 두 장에서
#     돌려받은 상자 135개, 344개의 최저 점수도 0.0010이었다(2026-11-24 파이). 그래서 장치가 거의 모든 후보를
#     넘기고, 실제로 남는 상자는 설정의 score_thr(기본 0.4)가 정한다. score_thr를 0.001보다 낮춰도 효과가 없다.
PPU_DEVICE_SCORE_THR = 0.001
PPU_BBOX_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
        ("grid_y", "u1"), ("grid_x", "u1"), ("box_idx", "u1"), ("layer_idx", "u1"),
        ("score", "<f4"), ("label", "<u4"), ("pad", "V4"),
    ]
)
assert PPU_BBOX_DTYPE.itemsize == 32


def ppu_boxes(raw: np.ndarray) -> np.ndarray:
    """PPU 출력 하나를 DeviceBoundingBox_t 구조체 배열(1차원)로 바꾼다.

    raw: (1, N, 32) 또는 (N, 32) uint8, 빈 결과 (0, 0), 또는 이미 구조체 배열.
    """
    a = np.asarray(raw)
    if a.dtype.names:
        return a.reshape(-1)
    if a.size == 0:
        return np.zeros(0, dtype=PPU_BBOX_DTYPE)
    if a.dtype != np.uint8 or a.shape[-1] != PPU_BBOX_DTYPE.itemsize:
        raise ValueError(f"PPU 출력은 마지막 축이 32바이트인 uint8이어야 합니다. 받은 모양: {a.shape} {a.dtype}")
    return np.ascontiguousarray(a.reshape(-1, PPU_BBOX_DTYPE.itemsize)).view(PPU_BBOX_DTYPE).reshape(-1)


def parse_dx_ppu_detections(
    outputs: Sequence[np.ndarray],
    scale: float,
    pad: tuple[int, int],
    score_thr: float,
    iou_thr: float = 0.45,
    labels: dict | None = None,
    max_det: int = 100,
) -> list[Detection]:
    """YOLOv8 PPU 모델 출력 → 원본 영상 좌표의 Detection (점수 거르기, 클래스 거르기, 클래스별 NMS)."""
    if len(outputs) != 1:
        raise ValueError(f"PPU 검출 모델은 출력이 1개여야 합니다. 받은 개수: {len(outputs)}")
    b = ppu_boxes(outputs[0])
    keep = COCO_KEEP if labels is None else labels
    if len(b) == 0:
        return []
    score = b["score"].astype(np.float32)
    cls_id = b["label"].astype(np.int64)
    m = (score >= score_thr) & np.isin(cls_id, list(keep))
    if not m.any():
        return []
    b, score, cls_id = b[m], score[m], cls_id[m]
    boxes = np.stack(
        [b["x"] - b["w"] / 2, b["y"] - b["h"] / 2, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2], axis=1
    ).astype(np.float32)
    dets = []
    for k in batched_nms(boxes, score, cls_id, iou_thr, max_det):
        x1, y1, x2, y2 = (float(v) for v in boxes[k])
        # 레터박스를 되돌려 원본 영상 픽셀로 (parse_dx_detections와 같은 식)
        dets.append(
            Detection(
                keep[int(cls_id[k])],
                float(score[k]),
                ((x1 - pad[0]) / scale, (y1 - pad[1]) / scale, (x2 - pad[0]) / scale, (y2 - pad[1]) / scale),
            )
        )
    return dets


def dx_seg_to_road_mask(
    seg_out: np.ndarray, out_hw: tuple[int, int], road_classes: Sequence[int], channels_first: bool = True
) -> np.ndarray:
    """DX-M1 세그 출력 → 도로 마스크.

    - (1,C,H,W) 로짓: dx_engine이 출력을 NCHW로 돌려주므로(드론 모델로 확인) 기본은 채널 우선.
      NHWC로 나오는 모델이면 channels_first=False.
    - (1,H,W) 또는 (H,W): 이미 클래스 번호로 나온 출력으로 보고 그대로 쓴다.
      DEEPX 공식 bisenetv2_1024x2048.dxnn은 장치 안에서 argmax까지 해서 (1, 1024, 2048) uint16
      클래스 번호(Cityscapes 19 클래스, 0이 road)를 준다(2026-11-24 파이 실측).
    """
    seg = np.asarray(seg_out)
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]  # 배치 축만 없앤다(채널이 1개여도 축을 지키려고 squeeze는 쓰지 않음)
    if seg.ndim == 3 and np.issubdtype(seg.dtype, np.integer) and seg.shape[0] == 1:
        seg = seg[0]  # (1,H,W) 정수 클래스 맵
    size = (out_hw[1], out_hw[0])
    if seg.ndim == 2:
        # ---- 클래스 번호 맵: 먼저 원본 크기로 줄이고(최근접) 그다음 비교한다 ----
        # 1024x2048(200만 화소)에서 비교하면 파이5에서 수 ms가 더 든다. 최근접 축소는 클래스 번호를
        # 섞지 않으므로 "비교 후 축소"와 결과가 같다. cv2.resize가 받는 자료형(uint8, uint16, int16,
        # float32)이 아니면 int32 대신 float32로 바꿔 줄인다(클래스 번호는 작은 정수라 정확히 보존된다).
        cls = seg
        if cls.dtype not in (np.uint8, np.uint16, np.int16, np.float32):
            cls = cls.astype(np.float32)
        if cls.shape != (out_hw[0], out_hw[1]):
            cls = cv2.resize(cls, size, interpolation=cv2.INTER_NEAREST)
        rc = list(road_classes)
        if len(rc) == 1:
            return (cls == rc[0]).astype(np.uint8)
        return np.isin(cls, rc).astype(np.uint8)
    # ---- 로짓: 채널 축으로 argmax 뒤 비교, 그다음 원본 크기로 ----
    cls = np.argmax(seg, axis=0 if channels_first else -1)
    mask = np.isin(cls, list(road_classes)).astype(np.uint8)
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# DX-M1: 모델 하나 (InferenceEngine 감싸기)
# ---------------------------------------------------------------------------
class DxModel:
    """.dxnn 하나를 DX-M1에 올려 둔다. 입력 형식은 장치에 물어서 정한다.

    engine_factory: 시험용. 경로를 받아 InferenceEngine처럼 동작하는 객체를 돌려주는 함수.
                    None이면 dx_engine.InferenceEngine을 쓴다(파이 .venv에만 있음).
    """

    def __init__(self, path: str, key: str = "dx_det_model", engine_factory=None):
        _require_dx_model(path, key)
        if engine_factory is None:
            try:
                from dx_engine import InferenceEngine  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "dx_engine을 불러오지 못했습니다. 파이에서 ~/work/fsd_diff_npu/.venv 를 켜고 실행하세요 "
                    "(시스템 파이썬에는 dx_engine이 없습니다)."
                ) from e
            engine_factory = InferenceEngine
        self.path = path
        self.engine = engine_factory(path)

        # ---- 입력 형식(모양, 자료형, 축 순서) ----
        ins = self.engine.get_input_tensors_info()
        if len(ins) != 1:
            raise ValueError(f"입력이 {len(ins)}개인 모델은 아직 지원하지 않습니다: {path}")
        shape = [int(v) for v in ins[0]["shape"]]
        self.input_shape = shape
        self.input_dtype = np.dtype(ins[0].get("dtype") or np.uint8)
        if len(shape) == 4 and shape[3] in (1, 3):
            self.layout, self.input_hw = "NHWC", (shape[1], shape[2])
        elif len(shape) == 4 and shape[1] in (1, 3):
            self.layout, self.input_hw = "NCHW", (shape[2], shape[3])
        elif len(shape) == 3 and shape[2] in (1, 3):
            self.layout, self.input_hw = "HWC", (shape[0], shape[1])
        else:
            raise ValueError(f"입력 모양을 해석할 수 없습니다: {shape} ({path})")

        # ---- 출력 정보 ----
        outs = self.engine.get_output_tensors_info()
        self.output_names = [o["name"] for o in outs]
        self.output_shapes = [[int(v) for v in o["shape"]] for o in outs]
        # 장치 후처리(PPU) 모델은 출력이 이미 상자 목록이라 YOLO 원시 출력 해석을 쓰면 안 된다
        self.is_ppu = bool(self.engine.is_ppu()) if hasattr(self.engine, "is_ppu") else False

    def prepare(self, img: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 영상을 모델이 바라는 모양과 자료형으로 바꾼다. 크기는 미리 맞춰 둘 것."""
        if img.shape[:2] != tuple(self.input_hw):
            raise ValueError(f"입력 크기가 다릅니다: {img.shape[:2]} (모델 {self.input_hw})")
        x = img
        if self.layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))
        if self.input_dtype.kind == "f":
            # 정규화가 모델 밖에 남은 float 입력 모델: Ultralytics 관례대로 0~1로 (확인 필요)
            x = x.astype(self.input_dtype) / 255.0
        else:
            x = x.astype(self.input_dtype, copy=False)
        return np.ascontiguousarray(x.reshape(self.input_shape))

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        """한 장 동기 실행. 출력은 출력 텐서 순서대로의 배열 목록."""
        return list(self.engine.run(x))

    def run_many(self, xs: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
        """여러 장을 run_async로 한꺼번에 보내고 보낸 순서대로 받는다(NPU 코어 3개를 겹쳐 쓴다)."""
        ids = [self.engine.run_async(x) for x in xs]
        return [list(self.engine.wait(j)) for j in ids]

    def close(self) -> None:
        self.engine.dispose()


# ---------------------------------------------------------------------------
# DX-M1 백엔드 (HailoBackend와 같은 인터페이스)
# ---------------------------------------------------------------------------
class DXM1Backend:
    """DX-M1으로 검출(+세그)을 한다. infer()는 HailoBackend와 같은 FrameResult 목록을 돌려준다.

    - 검출 모델: NMS 전 YOLO 원시 출력(출력 1개)을 CPU에서 디코딩하고 NMS한다.
    - 세그 모델: N 프레임마다 실행하고 그 사이에는 최근 마스크를 쓴다(HailoBackend와 같음).
      segment()만 따로 부를 수 있어서, 검출은 Hailo, 세그는 DX-M1을 별도 스레드로 돌리는
      파이프라인 옵션(RealityViewPipeline(seg_backend=...))에도 쓴다. 이때 det_model은 None이어도 된다.
    - infer_raw(): 후처리 전 원시 출력. 출력 형식이 다른 모델(세그, 스케일별 헤드)을 붙일 때 쓴다.
    """

    def __init__(
        self,
        det_model: str | None,
        seg_model: str | None = None,
        score_thr: float = 0.4,
        iou_thr: float = 0.45,
        road_classes: Sequence[int] = (0,),
        seg_every: int = 3,
        timer: StageTimer | None = None,
        labels: dict | None = None,  # 남길 클래스(번호 → 이름). None이면 COCO_KEEP
        det_format: str = "yolov8",  # "yolov8" | "yolov5"
        input_rgb: bool = True,  # 모델이 RGB 입력을 바라면 True(Ultralytics 기본), BGR이면 False
        max_det: int = 100,
        seg_channels_first: bool = True,  # 세그 로짓이 (1,C,H,W)면 True. dx_engine 출력은 NCHW로 확인함
        engine_factory=None,  # 시험용 가짜 엔진. None이면 dx_engine
        depth_model: str | None = None,  # 단안 깊이 .dxnn(선택). 있으면 estimate_depth()를 쓸 수 있다
    ):
        if not det_model and not seg_model and not depth_model:
            raise ValueError(
                "DX-M1 백엔드에는 검출, 세그, 깊이 모델 중 하나는 있어야 합니다. " + DX_MODEL_HELP.format(key="dx_det_model")
            )
        # 엔진을 하나라도 열기 전에 두 파일을 모두 확인한다(하나만 열고 실패하면 장치를 놓지 못할 수 있음)
        if det_model:
            _require_dx_model(det_model, "dx_det_model")
        if seg_model:
            _require_dx_model(seg_model, "dx_seg_model")
        if depth_model:
            _require_dx_model(depth_model, "dx_depth_model")
        self.timer = timer or StageTimer(enabled=False)
        self.labels = labels
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.road_classes = tuple(road_classes)
        self.seg_every = max(1, seg_every)
        self.det_format = det_format
        self.input_rgb = input_rgb
        self.max_det = max_det
        self.seg_channels_first = seg_channels_first
        self.det: DxModel | None = None
        self.seg: DxModel | None = None
        self.depth: DxModel | None = None
        try:
            if det_model:
                self.det = DxModel(det_model, "dx_det_model", engine_factory)
            if seg_model:
                self.seg = DxModel(seg_model, "dx_seg_model", engine_factory)
            if depth_model:
                self.depth = DxModel(depth_model, "dx_depth_model", engine_factory)
        except Exception:
            self.close()
            raise
        self._frame_idx = 0
        self._last_masks: list[np.ndarray] = []

    # ---- 전처리 ---------------------------------------------------------
    def _color(self, f: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if self.input_rgb else f

    def _prepare_det(self, frames: Sequence[np.ndarray]):
        xs, metas = [], []
        for f in frames:
            lb, s, pad = letterbox(self._color(f), self.det.input_hw)
            xs.append(self.det.prepare(lb))
            metas.append((s, pad))
        return xs, metas

    def _prepare_seg(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        th, tw = self.seg.input_hw
        return [self.seg.prepare(cv2.resize(self._color(f), (tw, th))) for f in frames]

    def _prepare_depth(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        # DEEPX dx_app 깊이 예제와 같이 레터박스 없이 단순 리사이즈(SimpleResizePreprocessor).
        # 그래서 출력 맵을 원본 크기로 늘리기만 하면 화소 위치가 맞는다.
        th, tw = self.depth.input_hw
        return [self.depth.prepare(cv2.resize(self._color(f), (tw, th))) for f in frames]

    # ---- 원시 출력 ------------------------------------------------------
    def infer_raw(self, frames: Sequence[np.ndarray], which: str = "det") -> list[list[np.ndarray]]:
        """후처리 전 출력. 카메라마다 출력 텐서 목록 하나씩.

        which="det": 레터박스 입력(좌표를 되돌리려면 letterbox()의 배율과 패딩을 같이 써야 함)
        which="seg": 단순 리사이즈 입력
        """
        if which == "det":
            if self.det is None:
                raise RuntimeError("검출 모델(dx_det_model)이 없습니다")
            xs, _ = self._prepare_det(frames)
            return self.det.run_many(xs)
        if which == "seg":
            if self.seg is None:
                raise RuntimeError("세그 모델(dx_seg_model)이 없습니다")
            return self.seg.run_many(self._prepare_seg(frames))
        if which == "depth":
            if self.depth is None:
                raise RuntimeError("깊이 모델(dx_depth_model)이 없습니다")
            return self.depth.run_many(self._prepare_depth(frames))
        raise ValueError(f"which는 det, seg, depth 중 하나여야 합니다: {which}")

    # ---- 세그: 도로 마스크 ----------------------------------------------
    def segment(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        """카메라마다 원본 크기의 도로 마스크(uint8 0/1)."""
        if self.seg is None:
            raise RuntimeError("세그 모델(dx_seg_model)이 없습니다")
        outs = self.seg.run_many(self._prepare_seg(frames))
        return [
            dx_seg_to_road_mask(o[0], f.shape[:2], self.road_classes, self.seg_channels_first)
            for o, f in zip(outs, frames)
        ]

    # ---- 깊이 -----------------------------------------------------------
    def estimate_depth(self, frames: Sequence[np.ndarray], full_size: bool = True) -> list[np.ndarray]:
        """카메라마다 깊이 맵(float32, 모델 값 그대로). full_size=True면 원본 영상 크기로 늘린다.

        값의 뜻(깊이/역깊이, 미터 여부)은 realityview/depth.py 참고. 시간은 "depth" 단계로 잰다.
        """
        from .depth import depth_from_output, resize_depth

        if self.depth is None:
            raise RuntimeError("깊이 모델(dx_depth_model)이 없습니다")
        tm = self.timer
        with tm.stage("depth_pre"):
            xs = self._prepare_depth(frames)
        with tm.stage("depth"):
            outs = self.depth.run_many(xs)
        with tm.stage("depth_post"):
            maps = [depth_from_output(o[0]) for o in outs]
            if full_size:
                maps = [resize_depth(m, f.shape[:2]) for m, f in zip(maps, frames)]
        return maps

    # ---- 검출(+세그) -----------------------------------------------------
    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]:
        if self.det is None:
            raise RuntimeError("검출 모델(dx_det_model)이 없어 infer()를 쓸 수 없습니다. segment()만 쓸 수 있습니다.")
        if self.det.is_ppu and self.det_format != "yolov8":
            # PPU 출력 해석은 YOLOv8 PPU(중심, 폭, 높이가 픽셀로 나옴)만 확인했다. yolov5 PPU는
            # 앵커와 격자 계산이 따로 필요하므로(dx_app YOLOv5PPUPostprocessor) 추측하지 않는다.
            desc = ", ".join(f"{n}{s}" for n, s in zip(self.det.output_names, self.det.output_shapes))
            raise ValueError(
                f"PPU 모델은 dx_det_format: yolov8(YOLOv8 PPU)만 해석합니다: {desc}. "
                "다른 PPU 모델은 DXM1Backend.infer_raw()로 출력을 받아 쓰세요."
            )
        tm = self.timer
        # ---- 전처리: BGR->RGB, 레터박스, 모델 입력 모양으로 ----
        with tm.stage("preprocess"):
            xs, metas = self._prepare_det(frames)
        # ---- DX-M1 추론: 카메라마다 run_async, 모두 wait ----
        with tm.stage("dxm1_infer"):
            outs = self.det.run_many(xs)
        # ---- 디코딩 + CPU NMS → 원본 영상 좌표의 Detection ----
        with tm.stage("nms_parse"):
            results = []
            for o, (s, pad) in zip(outs, metas):
                if self.det.is_ppu:
                    # 장치 후처리(PPU) 모델: 상자 구조체 목록(DeviceBoundingBox_t)을 읽고 CPU NMS
                    dets = parse_dx_ppu_detections(
                        o, s, pad, self.score_thr, self.iou_thr, self.labels, self.max_det
                    )
                else:
                    # NMS 전 YOLO 원시 출력 (1, 84, 8400)
                    dets = parse_dx_detections(
                        o, self.det.output_names, self.det.input_hw, s, pad,
                        self.score_thr, self.iou_thr, self.det_format, self.labels, self.max_det,
                    )
                results.append(FrameResult(detections=dets))

        if self.seg is not None:
            if self._frame_idx % self.seg_every == 0 or len(self._last_masks) != len(frames):
                with tm.stage("seg"):
                    self._last_masks = self.segment(frames)
            for r, m in zip(results, self._last_masks):
                r.road_mask = m
        self._frame_idx += 1
        return results

    def close(self) -> None:
        for m in (self.det, self.seg, self.depth):
            if m is not None:
                m.close()
        self.det = self.seg = self.depth = None


# ---------------------------------------------------------------------------
# PC 개발용
# ---------------------------------------------------------------------------
class UltralyticsBackend:
    """PC에서 알고리즘을 개발할 때 쓰는 백엔드. `pip install ultralytics` 필요."""

    def __init__(self, model: str = "yolov8n.pt", score_thr: float = 0.4):
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(model)
        self.score_thr = score_thr

    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]:
        res = self.model.predict(list(frames), conf=self.score_thr, verbose=False)
        out = []
        for r in res:
            dets = []
            for (x1, y1, x2, y2), c, s in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
            ):
                label = COCO_KEEP.get(int(c))
                if label:
                    dets.append(Detection(label, float(s), (float(x1), float(y1), float(x2), float(y2))))
            out.append(FrameResult(detections=dets))
        return out

    def close(self) -> None:
        pass
