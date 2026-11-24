from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext

import numpy as np

# 단계별 처리 시간 측정. 기본은 꺼져 있고, 꺼져 있으면 아무것도 재지 않는다(오버헤드 거의 0).
#
# 쓰는 법
#   timer = StageTimer(enabled=True, warmup=10)
#   with timer.stage("capture"):
#       ...
#   timer.end_frame()          # 한 프레임이 끝날 때마다 부른다
#   timer.summary()            # 단계별 평균, p50, p95 (ms)


# ---------------------------------------------------------------------------
# 표에 찍을 단계 순서와 한글 이름
# ---------------------------------------------------------------------------
STAGE_ORDER = (
    "capture",       # 카메라 획득 (카메라 N대 읽기)
    "preprocess",    # 전처리 (BGR->RGB, 레터박스)
    "hailo_infer",   # Hailo 추론
    "dxm1_infer",    # DX-M1 추론(카메라마다 run_async 후 모두 wait)
    "nms_parse",     # NMS 출력 파싱
    "seg",           # 세그 추론(있을 때만)
    "seg_async",     # 별도 스레드 세그(검출과 다른 NPU, 끝난 프레임에만 기록)
    "detect_total",  # 검출 전체(위 세 단계 + 백엔드 기타)
    "ipm_project",   # 검출 박스 지면 투영
    "merge",         # 카메라 간 중복 병합
    "track",         # 추적
    "bev_warp",      # BEV 바닥 영상 합성(IPM remap)
    "render",        # 렌더링(BEV 그림, 카메라 주석, 대시보드)
    "frame_total",   # 한 프레임 전체(루프 한 바퀴)
)

STAGE_LABEL_KO = {
    "capture": "카메라 획득",
    "preprocess": "전처리",
    "hailo_infer": "Hailo 추론",
    "dxm1_infer": "DX-M1 추론",
    "nms_parse": "NMS 파싱",
    "seg": "세그 추론",
    "seg_async": "세그 추론(별도 스레드)",
    "detect_total": "검출 합계",
    "ipm_project": "지면 투영(IPM)",
    "merge": "병합",
    "track": "추적",
    "bev_warp": "BEV 합성",
    "render": "렌더링",
    "frame_total": "프레임 전체",
}


# ---------------------------------------------------------------------------
# 타이머
# ---------------------------------------------------------------------------
class StageTimer:
    """단계별 시간을 프레임 단위로 모은다.

    전처리와 NMS 파싱처럼 카메라마다 여러 번 불리는 단계는 한 프레임 안에서 더해서
    "프레임당 시간"으로 기록한다. 렌더링 스레드처럼 다른 스레드에서 잰 값도 같은 방식으로
    다음 end_frame() 때 그 프레임 몫으로 들어간다(그 프레임에 렌더링이 없었으면 표본이 빠진다).
    """

    def __init__(self, enabled: bool = False, warmup: int = 0):
        self.enabled = bool(enabled)
        self.warmup = max(0, int(warmup))  # 앞쪽 N 프레임은 통계에서 뺀다(첫 추론, 자동 노출 안정화)
        self.frames = 0  # end_frame()이 불린 횟수(워밍업 포함)
        self._current: dict[str, float] = {}
        self._samples: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    # ---- 재기 ----------------------------------------------------------
    def stage(self, name: str):
        """`with timer.stage("이름"):` 블록의 시간을 잰다. 꺼져 있으면 빈 컨텍스트."""
        if not self.enabled:
            return nullcontext()
        return self._measure(name)

    @contextmanager
    def _measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1e3)

    def add(self, name: str, ms: float) -> None:
        """이번 프레임의 `name` 단계에 ms를 더한다."""
        if not self.enabled:
            return
        with self._lock:
            self._current[name] = self._current.get(name, 0.0) + float(ms)

    def end_frame(self, total_ms: float | None = None) -> None:
        """한 프레임을 마감한다. total_ms를 주면 frame_total로 기록한다."""
        if not self.enabled:
            return
        with self._lock:
            if total_ms is not None:
                self._current["frame_total"] = float(total_ms)
            if self.frames >= self.warmup:
                for k, v in self._current.items():
                    self._samples.setdefault(k, []).append(v)
            self._current = {}
            self.frames += 1

    # ---- 결과 ----------------------------------------------------------
    def summary(self) -> dict[str, dict]:
        """단계별 {n, mean_ms, p50_ms, p95_ms}. STAGE_ORDER 순서, 모르는 이름은 뒤에."""
        with self._lock:
            items = {k: list(v) for k, v in self._samples.items()}
        names = [n for n in STAGE_ORDER if n in items] + sorted(n for n in items if n not in STAGE_ORDER)
        out = {}
        for n in names:
            a = np.asarray(items[n], dtype=np.float64)
            out[n] = {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)),
            }
        return out


def format_table(summary: dict[str, dict]) -> str:
    """요약을 3열 마크다운 표로 만든다(단계 | 평균 ms | p50 / p95 ms)."""
    lines = ["| 단계 | 평균 ms | p50 / p95 ms |", "|---|---|---|"]
    for name, s in summary.items():
        label = STAGE_LABEL_KO.get(name, name)
        lines.append(f"| {label} | {s['mean_ms']:.1f} | {s['p50_ms']:.1f} / {s['p95_ms']:.1f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 파이 상태(스로틀링, 온도)
# ---------------------------------------------------------------------------
def read_pi_health() -> dict:
    """vcgencmd로 throttled 값과 SoC 온도를 읽는다. 맥처럼 명령이 없으면 None."""
    out = {}
    for key, cmd in (("throttled", ["vcgencmd", "get_throttled"]), ("temp", ["vcgencmd", "measure_temp"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            out[key] = r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            out[key] = None
    return out
