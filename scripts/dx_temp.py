#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

# DX-M1 온도를 dxrt-cli -s 로 읽어 기록하고, 너무 뜨거우면 식을 때까지 기다린다.
#
# 예) 파이 저장소 폴더에서
#   # 온도만 기록
#   python3 scripts/dx_temp.py --label before_dual --log out/measure/03_dx_temp.log
#   # 85도를 넘으면 60초씩 쉬며 80도 아래가 될 때까지 최대 600초 기다린다(못 식으면 종료 코드 2)
#   python3 scripts/dx_temp.py --label guard --log out/measure/03_dx_temp.log --guard 85 --resume-below 80
#
# 결과는 화면과 --log 파일 끝에 한 줄씩 붙는다: 시각, 이름, 최고 온도, 코어별 온도, 파이 CPU 온도

NPU_LINE = re.compile(r"NPU\s+(\d+):.*?temperature\s+(-?\d+)")


# ---------------------------------------------------------------------------
# 온도 읽기
# ---------------------------------------------------------------------------
def read_dx_temps(timeout: float = 15.0) -> list[int]:
    """dxrt-cli -s 출력의 'NPU n: ... temperature 75'C' 줄에서 코어별 온도(도)를 읽는다."""
    out = subprocess.run(["dxrt-cli", "-s"], capture_output=True, text=True, timeout=timeout).stdout
    temps = [int(t) for _, t in NPU_LINE.findall(out)]
    if not temps:
        raise RuntimeError("dxrt-cli -s 출력에서 NPU 온도를 찾지 못했습니다:\n" + out[-500:])
    return temps


def read_cpu_temp() -> float | None:
    """파이 CPU 온도(도). 읽을 수 없으면 None."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return None


def log_line(label: str, log: str | None) -> int:
    """온도를 한 번 읽어 화면과 로그에 남기고 최고 온도를 돌려준다."""
    temps = read_dx_temps()
    cpu = read_cpu_temp()
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {label} dx_max={max(temps)} dx_cores={temps} cpu={cpu}"
    print(line, flush=True)
    if log:
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return max(temps)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="check", help="로그에 남길 이름(예: before_dual_stub)")
    ap.add_argument("--log", help="온도를 덧붙일 로그 파일. 예: out/measure/03_dx_temp.log")
    ap.add_argument("--guard", type=float, default=None, help="이 온도를 넘으면 식을 때까지 기다린다(예: 85)")
    ap.add_argument("--resume-below", type=float, default=80.0, help="기다리다가 이 온도 아래가 되면 끝낸다")
    ap.add_argument("--step", type=int, default=60, help="기다릴 때 한 번에 쉬는 초")
    ap.add_argument("--max-wait", type=int, default=600, help="최대로 기다리는 초")
    args = ap.parse_args()

    t = log_line(args.label, args.log)
    if args.guard is None or t <= args.guard:
        return 0
    # 너무 뜨겁다: step초씩 쉬면서 resume-below 아래로 내려올 때까지 기다린다
    waited = 0
    while waited < args.max_wait:
        time.sleep(args.step)
        waited += args.step
        t = log_line(f"{args.label}_wait{waited}s", args.log)
        if t < args.resume_below:
            return 0
    print(f"{args.max_wait}초를 기다려도 {args.resume_below}도 아래로 내려오지 않았습니다", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
