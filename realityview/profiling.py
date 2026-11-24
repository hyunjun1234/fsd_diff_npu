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


# ---------------------------------------------------------------------------
# 메모리 사용량 (/proc 읽기, 리눅스 전용)
# ---------------------------------------------------------------------------
def read_proc_mem(status_path: str = "/proc/self/status", meminfo_path: str = "/proc/meminfo") -> dict:
    """이 프로세스의 RSS와 시스템 전체 가용 메모리(MB). 맥처럼 /proc이 없으면 값이 None.

    rss_mb: 이 프로세스가 실제로 차지한 메모리(VmRSS). 새는지 볼 때 이 값의 추세를 본다.
    mem_available_mb: 시스템 전체에서 더 쓸 수 있는 메모리(MemAvailable). 카메라, NPU 드라이버 버퍼처럼
    RSS에 잡히지 않는 메모리가 새면 이 값이 줄어든다.
    """
    out = {"rss_mb": None, "mem_available_mb": None}
    for path, key, field in ((status_path, "rss_mb", "VmRSS:"), (meminfo_path, "mem_available_mb", "MemAvailable:")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith(field):
                        out[key] = round(int(line.split()[1]) / 1024.0, 1)  # kB 단위를 MB로
                        break
        except OSError:
            pass
    return out


def _parse_vcgencmd(text: str | None) -> str | None:
    """'throttled=0x0' 이나 "temp=55.4'C" 에서 = 뒤의 값만 꺼낸다(0x0, 55.4)."""
    if not text or "=" not in text:
        return None
    return text.split("=", 1)[1].replace("'C", "").strip()


# ---------------------------------------------------------------------------
# 장시간 실행 상태 기록 (N초마다 csv 한 줄)
# ---------------------------------------------------------------------------
class HealthLogger:
    """장시간 실행(soak) 동안 일정 간격으로 FPS, 스로틀링, 온도, 메모리를 csv에 한 줄씩 남긴다.

    - fps는 직전 줄 이후 구간의 값이다(그 구간 프레임 수 / 구간 시간). 처음부터의 평균이 아니라서
      시간에 따라 떨어지는지 바로 보인다.
    - --profile의 StageTimer는 프레임마다 값을 쌓아 메모리가 늘어나므로, 메모리 누수를 볼 때는
      --profile 없이 이 기록만 켠다.
    - 줄마다 flush해서 도중에 끊겨도 앞 줄은 남는다.
    - clock, health_fn, mem_fn을 바꿔 끼울 수 있어 하드웨어 없이 시험한다.
    """

    COLUMNS = ("elapsed_s", "frames", "fps", "throttled", "soc_temp_c", "rss_mb", "mem_available_mb")

    def __init__(
        self,
        path: str,
        interval_s: float = 60.0,
        clock=time.monotonic,
        health_fn=None,
        mem_fn=None,
        extra_fn=None,
        extra_columns: tuple[str, ...] = (),
    ):
        self.path = path
        # 추가 열(예: NPU 온도). extra_fn()이 {열 이름: 값}을 돌려준다. 주지 않으면 기본 열만 쓴다
        self._extra_fn = extra_fn
        self.columns = self.COLUMNS + tuple(extra_columns)
        self.interval_s = float(interval_s)
        self._clock = clock
        self._health_fn = health_fn or read_pi_health
        self._mem_fn = mem_fn or read_proc_mem
        self.rows: list[dict] = []
        self._t0 = None  # 기록 시작 시각
        self._t_last = None  # 직전 줄 시각
        self._frames = 0  # 직전 줄 이후 프레임 수
        self._f = None

    def start(self) -> None:
        """기록을 시작한다. 시작 상태를 elapsed 0인 줄로 먼저 남긴다."""
        import csv
        import os

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._f, fieldnames=self.columns, extrasaction="ignore")
        self._csv.writeheader()
        self._t0 = self._t_last = self._clock()
        self._write(self._t0)

    def tick(self) -> dict | None:
        """프레임 하나가 끝날 때마다 부른다. 간격이 지났으면 한 줄을 쓰고 그 줄을 돌려준다(아니면 None)."""
        if self._f is None:
            return None
        self._frames += 1
        now = self._clock()
        if now - self._t_last >= self.interval_s:
            return self._write(now)
        return None

    def close(self) -> None:
        """끝낼 때 부른다. 마지막 줄 이후 남은 구간이 있으면 한 줄 더 쓰고 파일을 닫는다."""
        if self._f is None:
            return
        if self._frames > 0:
            self._write(self._clock())
        self._f.close()
        self._f = None

    def _write(self, now: float) -> dict:
        h = self._health_fn()
        m = self._mem_fn()
        span = now - self._t_last
        row = {
            "elapsed_s": round(now - self._t0, 1),
            "frames": self._frames,
            "fps": round(self._frames / span, 2) if span > 0 else None,
            "throttled": _parse_vcgencmd(h.get("throttled")),
            "soc_temp_c": _parse_vcgencmd(h.get("temp")),
            "rss_mb": m.get("rss_mb"),
            "mem_available_mb": m.get("mem_available_mb"),
        }
        if self._extra_fn is not None:
            row.update(self._extra_fn())
        self.rows.append(row)
        self._csv.writerow(row)
        self._f.flush()
        self._t_last, self._frames = now, 0
        return row


# ---------------------------------------------------------------------------
# 지연 시간 (센서 타임스탬프에서 화면 완성까지)
# ---------------------------------------------------------------------------
def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean()), 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "max_ms": round(float(a.max()), 2),
    }


class LatencyRecorder:
    """프레임마다 "센서가 찍은 시각"에서 "대시보드가 완성된 시각"까지를 잰다.

    - 시각은 모두 time.monotonic_ns()와 같은 시계(ns)로 받는다. Picamera2의 SensorTimestamp가 이 시계다
      (파이에서 CLOCK_MONOTONIC과 CLOCK_BOOTTIME 차이가 0인 것을 확인했다).
    - 카메라 N대를 차례로 읽으므로 카메라마다 따로 모은다. "oldest"는 그 프레임에 쓴 영상 중 가장 오래된 것
      기준이라 화면이 실제보다 늦은 정도의 상한이다.
    - capture_wait는 센서 시각에서 read()가 영상을 돌려준 시각까지다. 이 값이 크면 카메라 큐에 쌓인 묵은
      프레임을 받고 있다는 뜻이다.
    - seg_mask_age는 그 프레임에 쓴 세그 마스크의 원본 영상(카메라 0번 센서 시각)에서 화면 완성까지다.
    - 앞 warmup 프레임은 통계에서 뺀다. 타임스탬프를 모르는 카메라(파일, sim)는 기록하지 않는다.
    """

    def __init__(self, camera_names: list[str], warmup: int = 0):
        self.names = list(camera_names)
        self.warmup = max(0, int(warmup))
        self.frames = 0
        self.per_cam: dict[str, list[float]] = {n: [] for n in self.names}
        self.wait: dict[str, list[float]] = {n: [] for n in self.names}
        self.oldest: list[float] = []
        self.seg_mask_age: list[float] = []

    def add(
        self,
        sensor_ns: list[int | None],
        read_done_ns: list[int | None],
        done_ns: int,
        seg_mask_sensor_ns: int | None = None,
    ) -> None:
        """한 프레임 기록. sensor_ns, read_done_ns는 카메라 순서, done_ns는 대시보드 완성 시각."""
        measured = self.frames >= self.warmup
        self.frames += 1
        if not measured:
            return
        lats = []
        for name, ts, rd in zip(self.names, sensor_ns, read_done_ns):
            if ts is None:
                continue
            lat = (done_ns - ts) / 1e6
            self.per_cam[name].append(lat)
            lats.append(lat)
            if rd is not None:
                self.wait[name].append((rd - ts) / 1e6)
        if lats:
            self.oldest.append(max(lats))
        if seg_mask_sensor_ns is not None:
            self.seg_mask_age.append((done_ns - seg_mask_sensor_ns) / 1e6)

    def summary(self) -> dict:
        return {
            "frames_seen": self.frames,
            "warmup": self.warmup,
            "sensor_to_display_ms": {n: _stats(v) for n, v in self.per_cam.items()},
            "sensor_to_display_oldest_ms": _stats(self.oldest),
            "sensor_to_read_ms": {n: _stats(v) for n, v in self.wait.items()},
            "seg_mask_sensor_to_display_ms": _stats(self.seg_mask_age),
        }
