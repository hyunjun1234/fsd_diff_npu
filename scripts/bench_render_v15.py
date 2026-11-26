#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.raster import c_available  # noqa: E402
from realityview.render_v15 import RendererV15, V15Config  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from bench_render3d import fake_tracks  # noqa: E402

# v15풍 화면(render_v15.py)만 따로 재는 벤치(카메라, NPU 없이 CPU만).
# 가짜 트랙 N개를 놓고, 시점이 멈춘 경우(자차 그림 재사용)와 orbit 시점(매 프레임 바닥 표와 자차를 다시 그림)을 잰다.
#
# 예)
#   python3 scripts/bench_render_v15.py --config configs/five_cam.yaml --tracks 0 10 20 --out out/measure/20_bench_v15.json


def bench(r: RendererV15, tracks, ground, cov, refresh: bool, reps: int) -> float:
    for k in range(3):
        r.render(tracks, ground, None, cov, refresh_ground=True, t=k * 0.05, ego_speed=12.0)
    t0 = time.perf_counter()
    for k in range(reps):
        r.render(tracks, ground, None, cov, refresh_ground=refresh, t=(k + 3) * 0.05, ego_speed=12.0)
    return (time.perf_counter() - t0) * 1e3 / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/five_cam.yaml")
    ap.add_argument("--tracks", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5], help="render_scale 값들")
    ap.add_argument("--views", nargs="+", default=["chase", "orbit"])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", help="결과 JSON 경로")
    args = ap.parse_args()

    _, grid, cfg = load_config(args.config)
    scale = float((cfg.get("scene") or {}).get("scale", 1.0))
    rng = np.random.default_rng(1)
    ground = rng.integers(0, 255, (grid.height, grid.width, 3), dtype=np.uint8)
    cov = np.ones((grid.height, grid.width), bool)
    rows = []
    for rs in args.scales:
        for view in args.views:
            for n in args.tracks:
                r = RendererV15(grid, V15Config(render_scale=rs, view=view), scale=scale)
                tr = fake_tracks(n, scale)
                full = bench(r, tr, ground, cov, True, args.reps)
                reuse = bench(r, tr, ground, cov, False, args.reps)
                rows.append({"render_scale": rs, "view": view, "tracks": n, "with_ground_ms": round(full, 2),
                             "reuse_ground_ms": round(reuse, 2)})
                print(f"scale={rs:g} view={view:6s} tracks={n:2d}  바닥 갱신 포함 {full:6.2f} ms  바닥 재사용 {reuse:6.2f} ms",
                      flush=True)
    rep = {"config": args.config, "machine": platform.machine(), "c_raster": c_available(),
           "grid_hw": [grid.height, grid.width], "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
