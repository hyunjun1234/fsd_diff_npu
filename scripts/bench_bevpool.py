#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# FlashOcc BEVPool(깊이 분포 x 이미지 특징을 BEV 격자로 모으기)의 CPU 시간을 잰다.
# 카메라, NPU, 모델 파일을 쓰지 않는다. 깊이 확률과 특징은 무작위 값이다.
#
# 예) 파이 저장소 폴더에서, 스레드 1개와 4개를 따로 실행
#   timeout 900 python scripts/bench_bevpool.py --threads 1 --reps 30 --out out/measure/04_bevpool_cpu_pi5_t1.json
#   timeout 900 python scripts/bench_bevpool.py --threads 4 --reps 30 --out out/measure/04_bevpool_cpu_pi5_t4.json
# 결과는 --out에 JSON으로 남고, 요약은 화면에 찍힌다.


# ---------------------------------------------------------------------------
# 1. 인자 읽기와 스레드 수 고정 (numpy를 불러오기 전에 해야 효과가 있음)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BEVPool CPU benchmark")
    ap.add_argument("--threads", type=int, default=1, help="C(OpenMP)와 BLAS 스레드 수")
    ap.add_argument("--reps", type=int, default=30, help="구현마다 반복 횟수(20 이상 권장)")
    ap.add_argument("--warmup", type=int, default=3, help="재기 전에 버리는 반복 횟수")
    ap.add_argument("--cams", type=int, nargs="+", default=[4, 2], help="카메라 수 목록 (4, 2 중에서)")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--bins", type=str, default="1,45,0.5", help="깊이 구간 d_min,d_max,step (m)")
    ap.add_argument("--methods", type=str, default="reference,reduceat,bincount,c")
    ap.add_argument("--out", type=str, default="", help="JSON 저장 경로(비우면 저장 안 함)")
    return ap.parse_args()


ARGS = parse_args()
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = str(ARGS.threads)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview import bevpool as bp  # noqa: E402
from realityview.camera_model import Camera  # noqa: E402


# ---------------------------------------------------------------------------
# 2. 시험용 카메라 배치 (4단계 크기: 입력 256x704, 특징 1/16 = 16x44)
# ---------------------------------------------------------------------------
# 실제 차량 보정값이 아니라 nuScenes와 비슷하게 잡은 가상 배치다.
# 초점거리 560 px(256x704 입력 기준, 가로 화각 약 64도), 높이 1.6 m, 수평 시선.
IMG_HW = (256, 704)
FEAT_HW = (16, 44)


def make_rig(n_cams: int):
    K = [[560.0, 0, 351.5], [0, 560.0, 127.5], [0, 0, 1]]
    all_cams = {
        "front": Camera("front", IMG_HW[1], IMG_HW[0], K, [1.5, 0.0, 1.6], yaw_deg=0),
        "back": Camera("back", IMG_HW[1], IMG_HW[0], K, [-1.0, 0.0, 1.6], yaw_deg=180),
        "left": Camera("left", IMG_HW[1], IMG_HW[0], K, [0.5, 0.9, 1.6], yaw_deg=90),
        "right": Camera("right", IMG_HW[1], IMG_HW[0], K, [0.5, -0.9, 1.6], yaw_deg=-90),
    }
    names = {4: ["front", "back", "left", "right"], 2: ["front", "back"]}[n_cams]
    return bp.cameras_to_arrays([all_cams[n] for n in names], IMG_HW), names


# ---------------------------------------------------------------------------
# 3. 시간 재기
# ---------------------------------------------------------------------------
def time_fn(fn, reps: int, warmup: int):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(a.min()), "max_ms": float(a.max()), "reps": reps}


def run_condition(n_cams: int, bins: "bp.DepthBins", grid: "bp.BEVGrid", methods, backend):
    (K, R, t), names = make_rig(n_cams)
    C = ARGS.channels

    # 미리 계산은 카메라 설정이 바뀔 때만 하므로 따로 잰다.
    t0 = time.perf_counter()
    plan = bp.precompute_plan(K, R, t, grid, bins, FEAT_HW, IMG_HW)
    pre_ms = (time.perf_counter() - t0) * 1000.0

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(n_cams, bins.count) + FEAT_HW).astype(np.float32)
    depth = np.exp(logits)
    depth /= depth.sum(axis=1, keepdims=True)
    feat = rng.normal(size=(n_cams, C) + FEAT_HW).astype(np.float32)

    ref = bp.bev_pool_reference(depth, feat, K, R, t, grid, bins, IMG_HW)
    res = {"cams": names, "D": bins.count, "C": C, "precompute_ms": pre_ms,
           "n_total": plan.n_total, "n_kept": plan.n_kept, "n_cells": plan.n_cells,
           "XY": plan.X * plan.Y, "methods": {}}

    for m in methods:
        if m == "reference":
            fn = lambda: bp.bev_pool_reference(depth, feat, K, R, t, grid, bins, IMG_HW)  # noqa: E731
        elif m == "c":
            if backend is None:
                res["methods"]["c"] = {"skipped": "C backend not available"}
                continue
            fn = lambda: backend.pool(depth, feat, plan, nthreads=ARGS.threads)  # noqa: E731
        else:
            fn = lambda m=m: bp.bev_pool(depth, feat, plan, method=m)  # noqa: E731
        out = fn()
        r = time_fn(fn, ARGS.reps, ARGS.warmup)
        r["max_abs_diff_vs_reference"] = float(np.abs(out - ref).max())
        res["methods"][m] = r
        print(f"  cams={n_cams} D={bins.count} {m:9s} mean {r['mean_ms']:8.2f} ms  p95 {r['p95_ms']:8.2f} ms"
              f"  diff {r['max_abs_diff_vs_reference']:.2e}", flush=True)
    return res


# ---------------------------------------------------------------------------
# 4. 실행과 저장
# ---------------------------------------------------------------------------
def main() -> None:
    d_min, d_max, step = (float(x) for x in ARGS.bins.split(","))
    bins = bp.DepthBins(d_min, d_max, step)
    grid = bp.BEVGrid()  # 100x100, 0.8 m 셀, x/y -40~40 m, z -1~5.4 m
    methods = [m.strip() for m in ARGS.methods.split(",") if m.strip()]
    backend = bp.get_c_backend()

    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "numpy": np.__version__, "threads": ARGS.threads, "cpu_count": os.cpu_count(),
        "img_hw": IMG_HW, "feat_hw": FEAT_HW, "grid_xy": [grid.nx, grid.ny], "cell_m": grid.cell,
        "bins": [d_min, d_max, step],
        "c_backend": None if backend is None else {
            "lib": backend.path.name, "openmp": backend.openmp,
            "threads_used": backend.threads_used(ARGS.threads)},
    }
    print(json.dumps(info, ensure_ascii=False), flush=True)

    results = [run_condition(n, bins, grid, methods, backend) for n in ARGS.cams]
    for r in results:
        print(f"  cams={len(r['cams'])}: kept {r['n_kept']}/{r['n_total']} points, "
              f"{r['n_cells']}/{r['XY']} cells, precompute {r['precompute_ms']:.1f} ms", flush=True)

    if ARGS.out:
        os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
        with open(ARGS.out, "w") as f:
            json.dump({"info": info, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"saved {ARGS.out}")


if __name__ == "__main__":
    main()
