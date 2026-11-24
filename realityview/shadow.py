from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from .detcompare import match_detections
from .types import Detection, FrameResult

# 백엔드를 감싸는 두 가지 도구. 둘 다 run_demo.py 옵션을 켰을 때만 쓰고, 화면과 추적에는 늘 기본 결과만 간다.
#
# ScoreTap      : 백엔드를 낮은 점수 문턱(예: 0.3)으로 돌리고, 파이프라인에는 원래 문턱(score_thr, 기본 0.4) 이상만
#                 넘긴다. 문턱 아래 상자는 last_raw에 남아 트리거 클립의 "애매한 점수" 판정에만 쓴다.
#                 탐욕 NMS에서 점수 낮은 상자는 높은 상자를 지우지 못하므로, 거른 뒤 결과는 원래 문턱으로 돌린 것과 같다.
# ShadowBackend : 같은 프레임을 기본 백엔드(예: Hailo-8)와 섀도 백엔드(예: DX-M1)에서 동시에 돌리고,
#                 카메라마다 박스를 짝지어(IoU) 차이를 csv에 한 줄씩 남긴다. 돌려주는 값은 기본 백엔드 결과뿐이다.


# ---------------------------------------------------------------------------
# 낮은 점수 상자 엿보기
# ---------------------------------------------------------------------------
class ScoreTap:
    def __init__(self, inner, display_thr: float):
        self.__dict__["inner"] = inner  # __getattr__이 inner를 찾다가 되부르지 않게 먼저 넣는다
        self.display_thr = float(display_thr)
        self.last_raw: list[FrameResult] = []

    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]:
        raw = self.inner.infer(frames)
        self.last_raw = raw
        return [
            FrameResult(detections=[d for d in r.detections if d.score >= self.display_thr], road_mask=r.road_mask)
            for r in raw
        ]

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, name):
        # chip_temperature처럼 감싼 백엔드에만 있는 기능은 그대로 넘긴다(run_demo.py의 Hailo 온도 멈춤이 꺼지지 않게)
        return getattr(self.inner, name)


# ---------------------------------------------------------------------------
# 한 카메라 영상의 두 결과 비교 (순수 함수)
# ---------------------------------------------------------------------------
SHADOW_FIELDS = [
    "frame", "camera", "n_primary", "n_shadow", "matched", "shadow_missed", "shadow_extra",
    "iou_mean", "iou_min", "score_diff_mean", "score_diff_max_abs", "primary_ms", "shadow_ms",
]


def compare_dets(primary: Sequence[Detection], shadow: Sequence[Detection], iou_thr: float = 0.5) -> dict:
    """같은 영상의 두 검출 목록 비교.

    matched: 같은 클래스끼리 IoU iou_thr 이상으로 짝지은 수
    shadow_missed: 기본 모델은 찾았는데 섀도 모델이 놓친 수, shadow_extra: 섀도 모델만 찾은 수
    score_diff: 짝지은 상자의 (섀도 점수 - 기본 점수)
    """
    pairs = match_detections(primary, shadow, iou_thr)
    ious = [p[2] for p in pairs]
    diffs = [shadow[j].score - primary[i].score for i, j, _, _ in pairs]
    return {
        "n_primary": len(primary),
        "n_shadow": len(shadow),
        "matched": len(pairs),
        "shadow_missed": len(primary) - len(pairs),
        "shadow_extra": len(shadow) - len(pairs),
        "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
        "iou_min": round(float(np.min(ious)), 4) if ious else None,
        "score_diff_mean": round(float(np.mean(diffs)), 4) if diffs else None,
        "score_diff_max_abs": round(float(np.max(np.abs(diffs))), 4) if diffs else None,
    }


# ---------------------------------------------------------------------------
# 섀도 실행
# ---------------------------------------------------------------------------
class ShadowBackend:
    def __init__(
        self,
        primary,
        shadow,
        csv_path: str,
        camera_names: Sequence[str],
        iou_thr: float = 0.5,
    ):
        self.__dict__["primary"] = primary
        self.shadow = shadow
        self.camera_names = list(camera_names)
        self.iou_thr = float(iou_thr)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow")
        self._frame = 0
        self.last_rows: list[dict] = []
        # 전체 요약용 누계
        self.totals = {"frames": 0, "n_primary": 0, "n_shadow": 0, "matched": 0, "shadow_missed": 0, "shadow_extra": 0}
        self._ious: list[float] = []
        self._sdiffs: list[float] = []
        self._shadow_ms: list[float] = []
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self.csv_path = csv_path
        self._f = open(csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=SHADOW_FIELDS)
        self._w.writeheader()

    def _timed_shadow(self, frames):
        t = time.perf_counter()
        r = self.shadow.infer(frames)
        return r, (time.perf_counter() - t) * 1e3

    def infer(self, frames: Sequence[np.ndarray]) -> list[FrameResult]:
        # 섀도 추론을 먼저 다른 스레드로 보내고, 그동안 기본 추론을 한다(두 NPU가 같은 시간에 돈다)
        fut = self._pool.submit(self._timed_shadow, list(frames))
        t = time.perf_counter()
        prim = self.primary.infer(frames)
        prim_ms = (time.perf_counter() - t) * 1e3
        sh, sh_ms = fut.result()  # 섀도 쪽 오류는 여기서 그대로 올라온다
        rows = []
        for k, (a, b) in enumerate(zip(prim, sh)):
            name = self.camera_names[k] if k < len(self.camera_names) else str(k)
            c = compare_dets(a.detections, b.detections, self.iou_thr)
            row = {"frame": self._frame, "camera": name, **c, "primary_ms": round(prim_ms, 2), "shadow_ms": round(sh_ms, 2)}
            rows.append(row)
            for key in ("n_primary", "n_shadow", "matched", "shadow_missed", "shadow_extra"):
                self.totals[key] += c[key]
            pairs = match_detections(a.detections, b.detections, self.iou_thr)
            self._ious += [p[2] for p in pairs]
            self._sdiffs += [b.detections[j].score - a.detections[i].score for i, j, _, _ in pairs]
        self._w.writerows(rows)
        self._f.flush()
        self.totals["frames"] += 1
        self._shadow_ms.append(sh_ms)
        self.last_rows = rows
        self._frame += 1
        return prim  # 화면과 추적에는 기본 결과만

    def summary(self) -> dict:
        s = dict(self.totals)
        s["iou_mean"] = round(float(np.mean(self._ious)), 4) if self._ious else None
        s["iou_min"] = round(float(np.min(self._ious)), 4) if self._ious else None
        s["score_diff_mean"] = round(float(np.mean(self._sdiffs)), 4) if self._sdiffs else None
        s["shadow_ms_mean"] = round(float(np.mean(self._shadow_ms)), 2) if self._shadow_ms else None
        s["csv"] = self.csv_path
        return s

    def chip_temperature(self):
        """Hailo 온도. 기본 쪽이 Hailo가 아니면 섀도 쪽에서 찾는다(--hailo-stop-temp가 꺼지지 않게)."""
        for b in (self.primary, self.shadow):
            fn = getattr(b, "chip_temperature", None)
            if fn is not None:
                return fn()
        return None

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if not self._f.closed:
            self._f.close()
        try:
            self.primary.close()
        finally:
            self.shadow.close()

    def __getattr__(self, name):
        return getattr(self.primary, name)
