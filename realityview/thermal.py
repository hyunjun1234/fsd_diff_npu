from __future__ import annotations

import csv
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 단계 이름
# ---------------------------------------------------------------------------
# 숫자가 클수록 뜨겁다. 전체 단계는 센서별 단계 중 가장 높은 값이다.
NORMAL, CAUTION, WARNING, CRITICAL = 0, 1, 2, 3
LEVEL_NAMES = {NORMAL: "normal", CAUTION: "caution", WARNING: "warning", CRITICAL: "critical"}

# 과열로 스스로 끝낼 때 쓰는 종료 코드. systemd 서비스는 이 코드로 끝나면 다시 띄우지 않는다
# (RestartPreventExitStatus=75). 다시 띄우면 식지 않은 칩을 곧바로 다시 달구기 때문이다.
EXIT_THERMAL = 75


# ---------------------------------------------------------------------------
# 경계값 설정
# ---------------------------------------------------------------------------
@dataclass
class SensorLimits:
    """센서 하나의 단계 경계값(도).

    온도가 caution 이상이면 주의, warning 이상이면 경고, critical 이상이면 위험으로 올라간다.
    내려갈 때는 경계값보다 hysteresis 만큼 더 식어야 한 단계 내려간다(경계에서 오락가락하지 않게).
    해제 온도(경계 - hysteresis)는 부하 없을 때 온도보다 높아야 단계가 풀린다. 테스트가 이 조건을 지킨다.
    """

    caution: float
    warning: float
    critical: float
    hysteresis: float = 3.0

    def bounds(self) -> list[float]:
        return [self.caution, self.warning, self.critical]


# 기본값과 근거(측정 장치: 파이 5, 방열판과 팬 없는 Hailo-8, DX-M1, 2026-11 실측)
#   Hailo-8: 부하 없을 때 80.8~84도. 연속 실행 76초에 약 104도에서 HailoRT orange zone 경고(클럭 낮춤),
#            720초에 120도에서 red zone으로 스트림을 닫고 멈춤(out/measure/08_soak_latency.md).
#            run_demo 의 --hailo-stop-temp 권장값이 95도라 경고를 95도에, 위험을 104도보다 아래인 100도에 둔다.
#            주의 90도의 해제 온도 86도는 부하 없을 때 최고 84도보다 높다.
#   DX-M1:   방열 없이 부하 없을 때 72~75도, 12초 두 NPU 실행에 83도, 85도 이상은 위험으로 본다.
#            작업 규칙(80도를 넘으면 멈춤)에 맞춰 경고를 80도에 둔다. 주의 78도의 해제 온도 76도는 75도보다 높다.
#   파이 SoC: 펌웨어가 80도에서 클럭을 낮추기 시작하고 85도에서 더 낮춘다. 연속 실행 중 63.7~67.5도.
DEFAULT_LIMITS: dict[str, SensorLimits] = {
    "hailo": SensorLimits(caution=90.0, warning=95.0, critical=100.0, hysteresis=4.0),
    "dxm1": SensorLimits(caution=78.0, warning=80.0, critical=85.0, hysteresis=2.0),
    "soc": SensorLimits(caution=75.0, warning=80.0, critical=85.0, hysteresis=5.0),
}

# vcgencmd get_throttled 의 "지금" 비트(0~3). 지난 일을 뜻하는 16~19 비트는 보지 않는다.
#   0x1 저전압, 0x2 ARM 클럭 제한, 0x4 스로틀 중, 0x8 온도 소프트 제한
# 지금 비트가 하나라도 서 있으면 최소 주의 단계로 본다(전원 장치나 발열 문제 신호).
THROTTLED_NOW_MASK = 0xF


@dataclass
class ThermalConfig:
    """감시 주기, 경계값, 단계별 조치."""

    interval_s: float = 5.0  # 온도를 읽는 간격(초). Hailo가 4초에 약 1도씩 오르므로 5초면 충분히 촘촘하다
    limits: dict[str, SensorLimits] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    throttled_level: int = CAUTION  # throttled 지금 비트가 서 있을 때의 최소 단계
    # 단계별 조치. run_demo 의 메인 루프가 읽어서 적용한다
    caution_period_mult: int = 2  # 주의: 세그, 차선, 3D 바닥 주기를 이만큼 늘림
    warning_period_mult: int = 4  # 경고: 같은 주기를 이만큼 늘림
    warning_det_fps: float = 8.0  # 경고: 검출을 초당 이 횟수까지만(그 사이 프레임은 검출 건너뜀)
    critical_action: str = "exit"  # 위험: exit(깨끗하게 끝냄), camera_only(NPU 추론 멈추고 카메라와 녹화만), continue(기록만)


def parse_limits(text: str, base: dict[str, SensorLimits] | None = None) -> dict[str, SensorLimits]:
    """명령줄 경계값 문자열을 읽는다. 예: "hailo=90,95,100;dxm1=78,80,85" (뒤에 ,히스테리시스 를 붙여도 됨).

    적지 않은 센서는 base(기본 DEFAULT_LIMITS) 값을 그대로 쓴다.
    """
    out = dict(base or DEFAULT_LIMITS)
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, _, vals = part.partition("=")
        nums = [float(v) for v in vals.split(",")]
        if len(nums) not in (3, 4) or not (nums[0] <= nums[1] <= nums[2]):
            raise ValueError(f"경계값 형식이 틀렸습니다(주의<=경고<=위험): {part}")
        hyst = nums[3] if len(nums) == 4 else out.get(name.strip(), SensorLimits(0, 0, 0)).hysteresis
        out[name.strip()] = SensorLimits(nums[0], nums[1], nums[2], hyst)
    return out


# ---------------------------------------------------------------------------
# 센서별 단계 계산(히스테리시스)
# ---------------------------------------------------------------------------
def step_level(prev: int, temp: float | None, lim: SensorLimits) -> int:
    """직전 단계와 이번 온도로 새 단계를 정한다.

    올라갈 때: 넘은 경계 중 가장 높은 단계로 곧바로 간다(급하게 뜨거워지면 한 번에 위험까지).
    내려갈 때: 지금 단계의 경계보다 hysteresis 만큼 식어야 한 단계씩 내려간다.
    온도를 못 읽으면(None) 직전 단계를 그대로 둔다. 못 읽은 것을 "식었다"로 보지 않는다.
    """
    if temp is None:
        return prev
    b = lim.bounds()
    up = NORMAL
    for lv, edge in zip((CAUTION, WARNING, CRITICAL), b):
        if temp >= edge:
            up = lv
    if up >= prev:
        return up
    # 내려가는 중: 한 단계씩, 해제 온도 아래일 때만
    lv = prev
    while lv > up and temp < b[lv - 1] - lim.hysteresis:
        lv -= 1
    return lv


def throttled_now(value: str | int | None) -> int | None:
    """vcgencmd get_throttled 값(0x50005 같은 문자열이나 정수)에서 지금 비트만 꺼낸다."""
    if value is None:
        return None
    try:
        v = int(value, 0) if isinstance(value, str) else int(value)
    except ValueError:
        return None
    return v & THROTTLED_NOW_MASK


# ---------------------------------------------------------------------------
# 실제 온도 읽기(파이). 맥이나 장치가 없으면 None을 돌려준다
# ---------------------------------------------------------------------------
DX_NPU_LINE = re.compile(r"NPU\s+(\d+):.*?temperature\s+(-?\d+)")


def read_soc_temp() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled_raw() -> str | None:
    try:
        r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        s = r.stdout.strip()
        return s.split("=", 1)[1] if "=" in s else None
    except (OSError, subprocess.SubprocessError):
        return None


def read_dxm1_temp(timeout: float = 10.0) -> float | None:
    """dxrt-cli -s 의 코어별 온도 중 최고값. 명령이 없거나 출력에 온도가 없으면 None."""
    try:
        out = subprocess.run(["dxrt-cli", "-s"], capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    temps = [int(t) for _, t in DX_NPU_LINE.findall(out)]
    return float(max(temps)) if temps else None


def read_hailo_temp_standalone() -> float | None:
    """Hailo 장치를 잠깐 열어 칩 온도를 읽고 닫는다(녹화처럼 Hailo를 쓰지 않는 프로그램용).

    다른 프로그램이 Hailo를 쓰고 있으면 열리지 않아 None이 된다.
    """
    try:
        from hailo_platform import VDevice  # type: ignore
    except Exception:
        return None
    try:
        vdev = VDevice()
    except Exception:
        return None
    try:
        # 물리 장치 객체를 변수에 붙잡아 두어야 한다. 한 줄로 이어 쓰면 HailoRT가 장치를 바로 풀어
        # "The device in use has been released" 오류가 난다(파이에서 확인)
        devs = vdev.get_physical_devices()
        dev = devs[0]
        t = dev.control.get_chip_temperature()
        return round(float(max(t.ts0_temperature, t.ts1_temperature)), 1)
    except Exception:
        return None
    finally:
        try:
            vdev.release()
        except Exception:
            pass


def locked(fn, lock: threading.Lock):
    """같은 장치를 두 스레드(건강 기록, 과열 감시)가 읽을 때 한 번에 하나씩만 읽게 감싼다."""

    def wrapped():
        with lock:
            return fn()

    return wrapped


def default_readers(hailo_fn=None, dxm1: bool = True, hailo_standalone: bool = False) -> dict:
    """센서 이름 -> 읽기 함수. hailo_fn을 주면(Hailo 백엔드의 chip_temperature) 그것으로 읽는다."""
    r = {"soc": read_soc_temp}
    if hailo_fn is not None:
        r["hailo"] = hailo_fn
    elif hailo_standalone:
        r["hailo"] = read_hailo_temp_standalone
    if dxm1:
        r["dxm1"] = read_dxm1_temp
    return r


# ---------------------------------------------------------------------------
# 감시기: 읽기와 단계 계산(스레드). 조치는 하지 않는다
# ---------------------------------------------------------------------------
@dataclass
class ThermalState:
    level: int = NORMAL
    temps: dict = field(default_factory=dict)  # 센서 -> 마지막으로 읽은 온도(못 읽었으면 None)
    sensor_levels: dict = field(default_factory=dict)
    throttled: str | None = None
    cause: str = ""  # 지금 단계를 만든 센서(예: "hailo 96.1")
    changed_at: float = 0.0
    samples: int = 0

    @property
    def name(self) -> str:
        return LEVEL_NAMES[self.level]

    def label(self) -> str:
        """화면 표시용 한 줄. 예: "THERMAL caution hailo 91.2C" """
        return f"THERMAL {self.name}" + (f" {self.cause}C" if self.cause else "")


class ThermalMonitor:
    """센서를 주기적으로 읽어 단계를 계산한다.

    readers: 센서 이름 -> 온도(도)를 돌려주는 함수. 이름이 config.limits 에 있어야 단계에 반영된다.
    throttled_fn: vcgencmd get_throttled 값을 돌려주는 함수(없으면 None).
    sample() 을 직접 부르면 스레드 없이 한 번 읽는다(테스트용). start() 는 같은 일을 스레드에서 반복한다.
    단계가 바뀔 때마다 log 로 한 줄 남기고, csv_path 를 주면 매번 읽은 값을 한 줄씩 남긴다.
    """

    def __init__(self, config: ThermalConfig, readers: dict, throttled_fn=read_throttled_raw,
                 csv_path: str | None = None, log=print, clock=time.monotonic, on_change=None):
        self.cfg = config
        self.on_change = on_change  # 단계가 바뀔 때 감시 스레드에서 on_change(state) 를 부른다(이벤트 세우기 정도만)
        self.readers = dict(readers)
        self.throttled_fn = throttled_fn
        self.log = log
        self.clock = clock
        self._lock = threading.Lock()
        self._state = ThermalState(sensor_levels={k: NORMAL for k in self.readers})
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fail = {k: 0 for k in self.readers}  # 센서별 연속 읽기 실패 수
        self.transitions: list[dict] = []  # 단계가 바뀐 기록(요약 보고용)
        self.max_temps: dict = {}
        self._csv_f = None
        self._csv = None
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            self._csv_f = open(csv_path, "w", newline="")
            self._csv = csv.writer(self._csv_f)
            self._csv.writerow(["t_s", "wall_time", "level"] + [f"{k}_c" for k in self.readers] + ["throttled", "cause"])
        self._t0 = clock()

    # ---- 바깥에서 읽기(메인 루프) ----
    def state(self) -> ThermalState:
        with self._lock:
            s = self._state
            return ThermalState(s.level, dict(s.temps), dict(s.sensor_levels), s.throttled, s.cause, s.changed_at, s.samples)

    # ---- 한 번 읽고 단계 계산 ----
    def sample(self, temps: dict | None = None, throttled: str | int | None = "read") -> ThermalState:
        """센서를 한 번 읽어 단계를 갱신한다. temps 를 주면 읽지 않고 그 값을 쓴다(가짜 값 시험)."""
        if temps is None:
            temps = {}
            for k, fn in self.readers.items():
                try:
                    temps[k] = fn()
                except Exception:
                    temps[k] = None
        if throttled == "read":
            throttled = self.throttled_fn() if self.throttled_fn else None
        with self._lock:
            prev = self._state
            levels, cause_lv, cause = {}, NORMAL, ""
            for k, t in temps.items():
                lim = self.cfg.limits.get(k)
                if t is None:
                    self._fail[k] = self._fail.get(k, 0) + 1
                    if self._fail[k] in (3, 30):  # 연속 실패는 처음 3번째와 30번째에만 알린다(로그 넘침 방지)
                        self.log(f"[thermal] {k} 온도를 {self._fail[k]}번 연속 읽지 못했습니다(직전 단계 유지)")
                else:
                    self._fail[k] = 0
                    self.max_temps[k] = max(t, self.max_temps.get(k, t))
                lv = step_level(prev.sensor_levels.get(k, NORMAL), t, lim) if lim else NORMAL
                levels[k] = lv
                if lv > cause_lv:
                    cause_lv, cause = lv, f"{k} {t if t is not None else '?'}"
            now_bits = throttled_now(throttled)
            level = cause_lv
            if now_bits and self.cfg.throttled_level > level:
                level, cause = self.cfg.throttled_level, f"throttled {throttled}"
            now = self.clock()
            st = ThermalState(level, dict(temps), levels, None if throttled is None else str(throttled),
                              cause, prev.changed_at, prev.samples + 1)
            if level != prev.level:
                st.changed_at = now
                rec = {"t_s": round(now - self._t0, 1), "from": LEVEL_NAMES[prev.level], "to": LEVEL_NAMES[level],
                       "cause": cause, "temps": dict(temps)}
                self.transitions.append(rec)
                self.log(f"[thermal] 단계 {rec['from']} -> {rec['to']} ({cause}) 온도 {temps}")
            self._state = st
            changed = level != prev.level
            if self._csv is not None:
                self._csv.writerow([f"{now - self._t0:.1f}", f"{time.time():.3f}", LEVEL_NAMES[level]]
                                   + [temps.get(k) for k in self.readers] + [st.throttled, cause])
                self._csv_f.flush()
        if changed and self.on_change is not None:
            self.on_change(st)
        return st

    # ---- 스레드 ----
    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.cfg.interval_s)

    def start(self) -> "ThermalMonitor":
        self._thread = threading.Thread(target=self._run, name="thermal", daemon=True)
        self._thread.start()
        return self

    def close(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.cfg.interval_s + 15.0)  # dxrt-cli 한 번(최대 10초)이 끝날 시간까지 기다린다
        if self._csv_f is not None:
            self._csv_f.close()
            self._csv_f = None
        return self.summary()

    def summary(self) -> dict:
        s = self.state()
        return {"final_level": s.name, "samples": s.samples, "max_temps": dict(self.max_temps),
                "transitions": list(self.transitions)}


# ---------------------------------------------------------------------------
# 단계별 조치(run_demo 메인 루프에서 적용)
# ---------------------------------------------------------------------------
@dataclass
class ThermalActions:
    period_mult: int = 1  # 세그, 차선, 3D 바닥 주기 배수
    pause_dxm1: bool = False  # DX-M1 작업(세그 스레드, DX 검출) 멈춤
    det_fps_cap: float = 0.0  # 0이면 제한 없음
    stop_npu: bool = False  # NPU 추론 전부 멈춤(카메라와 녹화만)
    exit: bool = False  # 깨끗하게 끝냄


def actions_for(level: int, cfg: ThermalConfig) -> ThermalActions:
    if level <= NORMAL:
        return ThermalActions()
    if level == CAUTION:
        return ThermalActions(period_mult=cfg.caution_period_mult)
    if level == WARNING:
        return ThermalActions(period_mult=cfg.warning_period_mult, pause_dxm1=True, det_fps_cap=cfg.warning_det_fps)
    if cfg.critical_action == "exit":
        return ThermalActions(period_mult=cfg.warning_period_mult, pause_dxm1=True, stop_npu=True, exit=True)
    if cfg.critical_action == "continue":
        # 기록만: 경고 단계 조치를 그대로 유지한다(NPU를 쓰지 않는 녹화 도구의 기본값)
        return ThermalActions(period_mult=cfg.warning_period_mult, pause_dxm1=True, det_fps_cap=cfg.warning_det_fps)
    return ThermalActions(period_mult=cfg.warning_period_mult, pause_dxm1=True, stop_npu=True)


class ThermalGateBackend:
    """검출 백엔드를 감싸 경고 단계에서 검출 횟수를 줄이고, 위험 단계에서 검출을 멈춘다.

    --thermal 일 때만 끼운다. 건너뛴 프레임에는 지난 검출 결과를 새 FrameResult 로 복사해 돌려준다.
    (파이프라인이 결과 객체에 도로 마스크를 덧붙이므로 같은 객체를 다시 돌려주면 안 된다.)
    멈춘 동안에는 빈 결과를 돌려주고 NPU를 부르지 않는다. 그 밖의 속성(chip_temperature,
    set_frame_meta, last_rows 등)은 안쪽 백엔드로 그대로 넘긴다.
    """

    def __init__(self, inner, clock=time.monotonic):
        self.inner = inner
        self.clock = clock
        self.fps_cap = 0.0
        self.stopped = False
        self._last_t: float | None = None
        self._last: list | None = None
        self.skipped = 0
        self.ran = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, frames):
        from .types import FrameResult

        if self.stopped:
            self.skipped += 1
            return [FrameResult() for _ in frames]
        now = self.clock()
        if (self.fps_cap > 0 and self._last is not None and self._last_t is not None
                and now - self._last_t < 1.0 / self.fps_cap and len(self._last) == len(frames)):
            self.skipped += 1
            return [FrameResult(detections=list(r.detections), road_mask=r.road_mask) for r in self._last]
        res = self.inner.infer(frames)
        self._last, self._last_t = res, now
        self.ran += 1
        return res

    def close(self):
        return self.inner.close()


class ThermalController:
    """감시기 상태를 읽어 파이프라인과 백엔드에 조치를 적용한다. 메인 루프에서만 부른다."""

    def __init__(self, monitor: ThermalMonitor, pipe=None, gate: ThermalGateBackend | None = None,
                 det_on_dxm1: bool = False, log=print):
        self.mon = monitor
        self.pipe = pipe
        self.gate = gate
        self.det_on_dxm1 = det_on_dxm1  # --backend dxm1 이면 DX-M1 멈춤이 곧 검출 멈춤
        self.log = log
        self.level = NORMAL
        self.actions = ThermalActions()
        # 바꾸기 전 주기 값을 기억해 두었다가 정상으로 돌아오면 그대로 되돌린다
        self._base = {}
        if pipe is not None:
            for k in ("seg_every", "lanes_every", "ground_every"):
                if hasattr(pipe, k):
                    self._base[("pipe", k)] = getattr(pipe, k)
            be = getattr(pipe, "backend", None)
            inner = getattr(be, "inner", be)
            if inner is not None and hasattr(inner, "seg_every"):
                self._base[("backend", "seg_every")] = inner.seg_every  # HailoBackend 안쪽 세그 주기
                self._inner = inner

    def poll(self) -> ThermalActions:
        """단계가 바뀌었으면 조치를 적용하고, 지금 조치를 돌려준다."""
        st = self.mon.state()
        if st.level == self.level:
            return self.actions
        self.level = st.level
        a = actions_for(st.level, self.mon.cfg)
        self.actions = a
        for (where, k), base in self._base.items():
            obj = self.pipe if where == "pipe" else self._inner
            setattr(obj, k, max(1, int(base) * a.period_mult))
        if self.pipe is not None and hasattr(self.pipe, "seg_paused"):
            self.pipe.seg_paused = a.pause_dxm1
        if self.gate is not None:
            self.gate.fps_cap = a.det_fps_cap
            self.gate.stopped = a.stop_npu or (a.pause_dxm1 and self.det_on_dxm1)
        self.log(f"[thermal] 조치 적용: 단계 {st.name}, 주기 x{a.period_mult}, DX-M1 멈춤 {a.pause_dxm1}, "
                 f"검출 상한 {a.det_fps_cap or '없음'}, NPU 멈춤 {a.stop_npu}, 종료 {a.exit}")
        return a


def draw_label(img, state: ThermalState) -> None:
    """화면 오른쪽 위에 단계를 적는다. 정상은 회색, 주의 노랑, 경고 주황, 위험 빨강(BGR)."""
    import cv2

    color = {NORMAL: (160, 160, 160), CAUTION: (0, 220, 255), WARNING: (0, 140, 255), CRITICAL: (0, 0, 255)}[state.level]
    text = state.label()
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = max(0, img.shape[1] - w - 8)
    cv2.rectangle(img, (x - 4, 4), (x + w + 4, 10 + h), (0, 0, 0), -1)
    cv2.putText(img, text, (x, 6 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 스크립트 공통 옵션
# ---------------------------------------------------------------------------
def add_thermal_args(ap, critical_default: str = "exit") -> None:
    """run_demo.py 와 record_drive.py 가 같이 쓰는 과열 대응 옵션. --thermal 을 주지 않으면 아무 일도 하지 않는다."""
    g = ap.add_argument_group("과열 대응")
    g.add_argument("--thermal", action="store_true", help="Hailo, DX-M1, SoC 온도와 throttled 를 감시해 단계별로 부하를 낮춤")
    g.add_argument("--thermal-interval", type=float, default=5.0, help="온도를 읽는 간격(초). 기본 5")
    g.add_argument("--thermal-limits", default="",
                   help='경계값 덮어쓰기. 예: "hailo=90,95,100;dxm1=78,80,85;soc=75,80,85" (뒤에 ,히스테리시스)')
    g.add_argument("--thermal-csv", default=None, help="읽은 온도와 단계를 매번 남길 csv 경로")
    g.add_argument("--thermal-critical", default=critical_default, choices=("exit", "camera_only", "continue"),
                   help=f"위험 단계 조치. exit=깨끗하게 끝냄, camera_only=NPU 멈추고 카메라만, continue=기록만. 기본 {critical_default}")
    g.add_argument("--thermal-det-fps", type=float, default=8.0, help="경고 단계의 검출 횟수 상한(초당). 기본 8")


def config_from_args(args) -> ThermalConfig:
    return ThermalConfig(
        interval_s=float(args.thermal_interval),
        limits=parse_limits(args.thermal_limits) if args.thermal_limits else dict(DEFAULT_LIMITS),
        warning_det_fps=float(args.thermal_det_fps),
        critical_action=args.thermal_critical,
    )
