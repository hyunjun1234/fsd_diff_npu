#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

# Hailo-8 칩 온도를 읽어 기록하고, 너무 뜨거우면 식을 때까지 기다린다(scripts/dx_temp.py의 Hailo판).
# 방열판 없이 연속 실행 720초 만에 120도에 닿아 멈춘 적이 있어 실행 전에 확인한다(out/measure/08_soak_latency.md).
#
# 예) 파이 저장소 폴더에서 (장치를 잠깐 잡으므로 timeout을 붙인다. 다른 프로그램이 Hailo를 쓰는 중이면 실패한다)
#   timeout 700 python3 scripts/hailo_temp.py --label before --log out/measure/12_hailo_temp.log --guard 85 --resume-below 80
#
# 결과는 화면과 --log 파일 끝에 한 줄씩 붙는다: 시각, 이름, 칩 온도(센서 두 개 중 높은 값)


# ---------------------------------------------------------------------------
# 온도 읽기
# ---------------------------------------------------------------------------
def read_hailo_temp() -> float:
    """장치를 열어 칩 온도를 읽고 닫는다. VDevice 객체를 붙잡아 두어야 control을 쓸 수 있다."""
    from hailo_platform import VDevice  # type: ignore

    vdev = VDevice()
    try:
        dev = vdev.get_physical_devices()[0]
        t = dev.control.get_chip_temperature()
        return round(float(max(t.ts0_temperature, t.ts1_temperature)), 1)
    finally:
        vdev.release()


def log_line(label: str, log: str | None) -> float:
    t = read_hailo_temp()
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {label} hailo_temp={t}"
    print(line, flush=True)
    if log:
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return t


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="check", help="로그에 남길 이름")
    ap.add_argument("--log", help="온도를 덧붙일 로그 파일")
    ap.add_argument("--guard", type=float, default=None, help="이 온도를 넘으면 식을 때까지 기다린다(예: 85)")
    ap.add_argument("--resume-below", type=float, default=80.0, help="기다리다가 이 온도 아래가 되면 끝낸다")
    ap.add_argument("--step", type=int, default=60, help="기다릴 때 한 번에 쉬는 초")
    ap.add_argument("--max-wait", type=int, default=600, help="최대로 기다리는 초")
    args = ap.parse_args()

    t = log_line(args.label, args.log)
    if args.guard is None or t <= args.guard:
        return 0
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
