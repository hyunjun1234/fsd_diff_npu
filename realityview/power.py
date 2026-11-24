from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time

# ---------------------------------------------------------------------------
# 전원 꺼짐 신호 감시
# ---------------------------------------------------------------------------
# 시동(ACC)이 꺼지면 전원이 끊기기 전에 녹화를 닫고 디스크에 쓰고 끝내야 한다.
# 신호는 두 가지를 받는다.
#   1) 신호 파일: 이 경로에 파일이 생기면 꺼짐으로 본다(시험용, 또는 다른 프로그램이 알려 줄 때)
#   2) GPIO 입력: ACC 12V를 포토커플러로 받은 핀. 기본 배선(포토커플러 출력을 GND로 끌어내림 + 내부 풀업)에서는
#      ACC 켜짐 = 0(LOW), ACC 꺼짐 = 1(HIGH)이다. 배선이 반대면 acc_on_value=1 로 바꾼다.
# SIGTERM/SIGINT 는 각 스크립트의 신호 처리기가 받는다(여기서는 다루지 않는다).
# 실제 poweroff 는 하지 않는다. 끝난 뒤 부를 명령(예: "sudo poweroff")은 옵션 자리만 둔다.

RP1_LABEL = "pinctrl-rp1"  # 파이 5의 40핀 헤더 GPIO 칩 라벨. 커널에 따라 gpiochip0 또는 gpiochip4 라서 라벨로 찾는다


def find_header_chip() -> str | None:
    """40핀 헤더 GPIO 칩 경로(/dev/gpiochipN)를 라벨로 찾는다. gpiod가 없거나 못 찾으면 None."""
    try:
        import gpiod  # type: ignore
    except Exception:
        return None
    for name in sorted(os.listdir("/dev")):
        if not name.startswith("gpiochip"):
            continue
        path = "/dev/" + name
        try:
            with gpiod.Chip(path) as chip:
                if chip.get_info().label == RP1_LABEL:
                    return path
        except Exception:
            continue
    return None


class GpioLine:
    """libgpiod 2.x 로 입력 핀 하나를 연다(파이에 이미 있는 python3-gpiod / .venv 의 gpiod 2.2).

    bias 는 "pull_up", "pull_down", "disabled" 중 하나. read() 는 0 또는 1.
    """

    def __init__(self, line: int, chip: str | None = None, bias: str = "pull_up", consumer: str = "fsd-acc"):
        import gpiod  # type: ignore
        from gpiod.line import Bias, Direction, Value  # type: ignore

        self._active = Value.ACTIVE
        path = chip or find_header_chip()
        if path is None:
            raise RuntimeError(f"GPIO 칩({RP1_LABEL})을 찾지 못했습니다")
        b = {"pull_up": Bias.PULL_UP, "pull_down": Bias.PULL_DOWN, "disabled": Bias.DISABLED}[bias]
        self.line = int(line)
        self.chip = path
        self._req = gpiod.request_lines(
            path, consumer=consumer, config={self.line: gpiod.LineSettings(direction=Direction.INPUT, bias=b)}
        )

    def read(self) -> int:
        return 1 if self._req.get_value(self.line) == self._active else 0

    def close(self) -> None:
        try:
            self._req.release()
        except Exception:
            pass


class ShutdownWatcher:
    """신호 파일과 GPIO 를 주기적으로 보고, 꺼짐이 debounce_s 동안 이어지면 on_power_loss(이유)를 한 번 부른다.

    stop_file: 이 경로에 파일이 생기면 곧바로 꺼짐(디바운스 없음). 시작할 때 이미 있던 파일은
               지난 실행이 남긴 것으로 보고 지운다(남은 파일 때문에 다음 부팅 녹화가 바로 끝나지 않게).
    gpio_read: 0/1을 돌려주는 함수(GpioLine.read). acc_on_value 와 다른 값이 debounce_s 이상 이어지면 꺼짐.
               시동을 걸 때처럼 잠깐 전압이 떨어지는 것은 디바운스로 거른다.
    clock 과 gpio_read 를 바꿔 끼워 하드웨어 없이 시험한다.
    """

    def __init__(self, on_power_loss, stop_file: str | None = None, gpio_read=None, acc_on_value: int = 0,
                 debounce_s: float = 2.0, poll_s: float = 0.2, clock=time.monotonic, log=print):
        self.on_power_loss = on_power_loss
        self.stop_file = stop_file
        self.gpio_read = gpio_read
        self.acc_on_value = int(acc_on_value)
        self.debounce_s = float(debounce_s)
        self.poll_s = float(poll_s)
        self.clock = clock
        self.log = log
        self.reason: str | None = None
        self._off_since: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if stop_file and os.path.exists(stop_file):
            os.remove(stop_file)
            log(f"[power] 지난 실행이 남긴 신호 파일을 지웠습니다: {stop_file}")

    def check(self) -> str | None:
        """한 번 본다. 꺼짐을 처음 확정하면 이유를 돌려주고 on_power_loss 를 부른다."""
        if self.reason is not None:
            return None
        reason = None
        if self.stop_file and os.path.exists(self.stop_file):
            reason = f"stop_file:{self.stop_file}"
        elif self.gpio_read is not None:
            try:
                v = self.gpio_read()
            except Exception as e:  # 읽기 실패는 꺼짐으로 보지 않는다(잘못된 종료 방지)
                self.log(f"[power] GPIO 읽기 실패: {e}")
                v = self.acc_on_value
            now = self.clock()
            if v != self.acc_on_value:
                if self._off_since is None:
                    self._off_since = now
                elif now - self._off_since >= self.debounce_s:
                    reason = f"acc_off:gpio={v}:{self.debounce_s:g}s"
            else:
                self._off_since = None
        if reason is not None:
            self.reason = reason
            self.log(f"[power] 전원 꺼짐 신호: {reason}")
            self.on_power_loss(reason)
        return reason

    def _run(self) -> None:
        while not self._stop.is_set() and self.reason is None:
            self.check()
            self._stop.wait(self.poll_s)

    def start(self) -> "ShutdownWatcher":
        self._thread = threading.Thread(target=self._run, name="power-watch", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)


# ---------------------------------------------------------------------------
# 마무리: 디스크 동기화와 (옵션) 전원 끄기 명령
# ---------------------------------------------------------------------------
def finish(power_loss_reason: str | None, after_cmd: str | None = None, log=print) -> dict:
    """os.sync()로 쓰기 버퍼를 디스크에 내리고, 전원 꺼짐으로 끝났을 때만 after_cmd 를 부른다.

    after_cmd 는 기본 빈 값이다. 실제 차량에서는 여기서 "sudo poweroff" 를 부르게 설정한다
    (sudoers 에 poweroff 만 비밀번호 없이 허용하는 줄이 필요, docs/vehicle_power_and_autostart.md).
    SIGTERM 으로 끝날 때는 systemd 나 사람이 끄는 중이므로 부르지 않는다.
    """
    t0 = time.perf_counter()
    os.sync()
    sync_ms = (time.perf_counter() - t0) * 1e3
    log(f"[power] os.sync 완료 {sync_ms:.0f}ms")
    ran = None
    if power_loss_reason and after_cmd:
        log(f"[power] 전원 꺼짐 뒤 명령 실행: {after_cmd}")
        try:
            ran = subprocess.run(shlex.split(after_cmd), timeout=30).returncode
        except (OSError, subprocess.SubprocessError) as e:
            log(f"[power] 명령 실패: {e}")
            ran = -1
    return {"sync_ms": round(sync_ms, 1), "power_loss": power_loss_reason, "after_cmd_rc": ran}


def make_watcher(args, on_power_loss, log=print) -> ShutdownWatcher | None:
    """스크립트 공통: --stop-file, --acc-gpio 옵션으로 감시기를 만든다. 둘 다 없으면 None."""
    if not args.stop_file and args.acc_gpio is None:
        return None
    read = None
    if args.acc_gpio is not None:
        line = GpioLine(args.acc_gpio, chip=args.acc_chip, bias=args.acc_bias)
        log(f"[power] ACC 입력: {line.chip} 줄 {line.line}, 켜짐 값 {args.acc_on_value}, 지금 값 {line.read()}")
        read = line.read
    return ShutdownWatcher(on_power_loss, stop_file=args.stop_file, gpio_read=read, acc_on_value=args.acc_on_value,
                           debounce_s=args.acc_debounce, log=log)


def add_power_args(ap) -> None:
    """record_drive.py 와 run_demo.py 가 같이 쓰는 안전 종료 옵션."""
    g = ap.add_argument_group("안전 종료(시동 꺼짐)")
    g.add_argument("--stop-file", default=None, help="이 경로에 파일이 생기면 정리하고 끝냄(시험용). 예: out/STOP")
    g.add_argument("--acc-gpio", type=int, default=None, help="ACC 입력 GPIO 번호(BCM). 주지 않으면 GPIO를 보지 않음")
    g.add_argument("--acc-chip", default=None, help="GPIO 칩 경로. 기본은 라벨 pinctrl-rp1 로 찾음")
    g.add_argument("--acc-bias", default="pull_up", choices=("pull_up", "pull_down", "disabled"))
    g.add_argument("--acc-on-value", type=int, default=0, choices=(0, 1), help="ACC 켜짐일 때 핀 값. 기본 0(포토커플러가 끌어내림)")
    g.add_argument("--acc-debounce", type=float, default=2.0, help="ACC 꺼짐이 이 시간(초) 이어져야 꺼짐으로 봄. 기본 2")
    g.add_argument("--power-loss-cmd", default="", help='전원 꺼짐으로 끝난 뒤 부를 명령. 기본 없음. 실제 차량: "sudo poweroff"')
