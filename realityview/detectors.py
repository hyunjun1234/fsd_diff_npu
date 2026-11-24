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
# - UltralyticsBackend: PC 개발용. ultralytics YOLO로 검출만 한다.
# - 시뮬레이션은 sim.SimBackend 참고.


class Backend(Protocol):
    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]: ...

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


def seg_to_road_mask(seg_out: np.ndarray, out_hw: tuple[int, int], road_classes: Sequence[int]) -> np.ndarray:
    """세그 출력(H,W,C 로짓 또는 H,W 클래스맵) → 원본 크기의 도로 마스크(uint8 0/1)."""
    seg = np.asarray(seg_out)
    seg = np.squeeze(seg)
    if seg.ndim == 3:
        cls = np.argmax(seg, axis=-1)
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
        self.score_thr = score_thr
        self.road_classes = tuple(road_classes)
        self.seg_every = max(1, seg_every)
        self._frame_idx = 0
        self._last_masks: list[np.ndarray | None] = []

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
