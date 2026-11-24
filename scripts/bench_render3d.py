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
from realityview.render3d import Renderer3D, View3DConfig  # noqa: E402
from realityview.tracker import Track  # noqa: E402

# 3D 시점 렌더러만 따로 재는 벤치(카메라, NPU 없이 CPU만 쓴다).
# 실제 방에는 사람이나 차가 없어 데모로는 물체 그리기 비용을 잴 수 없어서, 가짜 트랙 N개를 놓고 잰다.
#
# 예)
#   python3 scripts/bench_render3d.py --config configs/two_cam_current.yaml --tracks 0 10 20 \
#       --out out/measure/10_bench_render3d_pi.json
# 결과: 트랙 수와 render_scale마다 바닥 갱신 포함/제외 평균 ms를 출력하고 --out JSON에 남긴다.


# ---------------------------------------------------------------------------
# 가짜 트랙(차, 트럭, 사람을 섞어 자차 앞뒤 좌우에 흩어 놓는다)
# ---------------------------------------------------------------------------
def fake_tracks(n: int, scale: float, seed: int = 0) -> list[Track]:
    rng = np.random.default_rng(seed)
    labels = ["car", "car", "truck", "person"]
    sizes = {"car": (4.5, 1.9), "truck": (8.0, 2.5), "person": (0.6, 0.6)}
    out = []
    for i in range(n):
        lab = labels[i % len(labels)]
        xy = np.array([rng.uniform(-3, 25), rng.uniform(-9, 9)]) * scale
        vel = np.array([rng.uniform(-3, 3), rng.uniform(-1, 1)]) * scale
        L, W = sizes[lab]
        out.append(Track(i + 1, lab, xy, vel=vel, size=(L * scale, W * scale), min_speed=0.8 * scale))
    return out


# ---------------------------------------------------------------------------
# 재기
# ---------------------------------------------------------------------------
def bench(r: Renderer3D, tracks, ground, road, cov, refresh: bool, reps: int) -> float:
    """render() 한 번의 평균 ms. 처음 3번은 버린다(캐시, 메모리 준비)."""
    for _ in range(3):
        r.render(tracks, ground, road, cov, refresh_ground=True)
    t0 = time.perf_counter()
    for k in range(reps):
        r.render(tracks, ground, road, cov, refresh_ground=refresh, t=k * 0.05)
    return (time.perf_counter() - t0) * 1e3 / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam_current.yaml")
    ap.add_argument("--tracks", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5], help="render_scale 값들")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", help="결과 JSON 경로")
    args = ap.parse_args()

    _, grid, cfg = load_config(args.config)
    scale = float((cfg.get("scene") or {}).get("scale", 1.0))
    rng = np.random.default_rng(1)
    # 바닥 영상은 무늬가 있는 아무 영상, 도로 마스크는 절반쯤 덮이게(렌더링 비용이 커지는 쪽)
    ground = rng.integers(0, 255, (grid.height, grid.width, 3), dtype=np.uint8)
    road = (rng.random((grid.height, grid.width)) < 0.5).astype(np.uint8)
    cov = np.ones((grid.height, grid.width), bool)

    rows = []
    for rs in args.scales:
        for n in args.tracks:
            r = Renderer3D(grid, View3DConfig(render_scale=rs, interp=True), scale=scale)
            tr = fake_tracks(n, scale)
            full = bench(r, tr, ground, road, cov, True, args.reps)
            reuse = bench(r, tr, ground, road, cov, False, args.reps)
            rows.append({"render_scale": rs, "tracks": n, "with_ground_ms": round(full, 2), "reuse_ground_ms": round(reuse, 2)})
            print(f"render_scale={rs:g} tracks={n:2d}  바닥 갱신 포함 {full:6.2f} ms  바닥 재사용 {reuse:6.2f} ms", flush=True)
    rep = {"config": args.config, "machine": platform.machine(), "grid_hw": [grid.height, grid.width], "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
