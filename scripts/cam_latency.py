#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402

# 카메라만 열어 "센서 시각 -> read()가 영상을 돌려준 시각"을 잰다(NPU를 쓰지 않음).
# --work-ms로 처리 시간을 흉내 내면, 루프가 카메라보다 느릴 때 묵은 프레임을 받는 몫을 NPU 없이 볼 수 있다.
# 카메라를 잡으므로 --frames로 끝나고, 바깥에서 timeout도 붙인다.
#
# 예) 파이 저장소 폴더에서
#   timeout -s INT -k 5 30 python3 scripts/cam_latency.py --config configs/two_cam_current.yaml --frames 150
#   timeout -s INT -k 5 30 python3 scripts/cam_latency.py --config configs/two_cam_current.yaml --frames 150 --work-ms 25 --queue off
# 결과는 화면에 JSON 한 줄, --out을 주면 그 파일에도 쓴다.


def stats(v: list[float]) -> dict:
    if not v:
        return {}
    a = np.asarray(v)
    return {"n": int(a.size), "mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam_current.yaml")
    ap.add_argument("--frames", type=int, default=150, help="이만큼 루프를 돌고 끝냄")
    ap.add_argument("--warmup", type=int, default=20, help="앞 N 루프는 통계에서 뺌")
    ap.add_argument("--work-ms", type=float, default=0.0, help="루프마다 이만큼 쉬어 처리 시간을 흉내 냄")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--buffers", type=int, default=None)
    ap.add_argument("--queue", choices=["on", "off"], default=None)
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    from realityview.sources import open_source

    cams, _, _ = load_config(args.config)
    ov = {}
    if args.fps is not None:
        ov["fps"] = args.fps
    if args.buffers is not None:
        ov["buffer_count"] = args.buffers
    if args.queue is not None:
        ov["queue"] = args.queue == "on"
    if args.flush:
        ov["flush"] = True
    srcs = []
    try:
        for c in cams:
            s = open_source(c, ov or None)
            s.want_timestamp = True
            srcs.append(s)
        wait = {c.name: [] for c in cams}
        t0 = None
        for n in range(args.frames):
            if n == args.warmup:
                t0 = time.perf_counter()
            for c, s in zip(cams, srcs):
                s.read()
                rd = time.monotonic_ns()
                if n >= args.warmup and s.last_timestamp_ns is not None:
                    wait[c.name].append((rd - s.last_timestamp_ns) / 1e6)
            if args.work_ms > 0:
                time.sleep(args.work_ms / 1e3)
        el = time.perf_counter() - t0 if t0 else 0.0
        rep = {"options": ov, "work_ms": args.work_ms, "loop_fps": round((args.frames - args.warmup) / el, 2) if el else None,
               "sensor_to_read_ms": {k: stats(v) for k, v in wait.items()}}
        text = json.dumps(rep, ensure_ascii=False)
        print(text)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text + "\n")
    finally:
        for s in srcs:
            s.close()


if __name__ == "__main__":
    main()
