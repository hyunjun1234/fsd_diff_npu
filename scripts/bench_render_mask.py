#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview import render  # noqa: E402
from realityview.bev import IpmCompositor  # noqa: E402
from realityview.config import load_config  # noqa: E402

# 도로 마스크 칠하기와 BEV 합성의 시간을 카메라, NPU 없이 잰다.
#
# - "legacy_*"는 2단계 준비 때(커밋 bb2bc78)의 render.py, bev.py 식을 그대로 옮긴 기준 함수다.
#   옛 방식이 어느 단계에서 시간을 쓰는지 쪼개서 보고, 새 방식과 같은 입력으로 비교한다.
# - "current_*"는 지금 저장소의 realityview 함수를 그대로 부른다.
# - 영상은 무작위 값, 마스크는 화소 50%가 1인 무작위 마스크(--fraction)다. 실제 도로 모양과 다르다.
#
# 예) 맥 또는 파이 저장소 폴더에서(카메라와 NPU를 쓰지 않음)
#   python3 scripts/bench_render_mask.py --config configs/two_cam_current.yaml --reps 50 \
#       --out out/measure/03_bench_mask_before_pi.json


# ---------------------------------------------------------------------------
# 옛 방식(기준 함수): 커밋 bb2bc78의 식 그대로
# ---------------------------------------------------------------------------
def legacy_annotate_overlay(img: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """옛 annotate_camera의 마스크 덧칠 부분(검출 상자, 글자는 빼고)."""
    out = img.copy()
    m = road_mask.astype(bool)
    out[m] = (0.6 * out[m] + 0.4 * np.array([200, 120, 60])).astype(np.uint8)
    return out


def legacy_annotate_steps(img: np.ndarray, road_mask: np.ndarray, t: dict) -> None:
    """옛 덧칠을 한 줄씩 쪼개서 잰다(어느 줄이 느린지 보려고). t에 단계별 ms를 더한다."""
    t0 = time.perf_counter()
    out = img.copy()
    t1 = time.perf_counter()
    m = road_mask.astype(bool)  # uint8 → bool 변환
    t2 = time.perf_counter()
    sel = out[m]  # 불리언 색인 읽기(선택된 화소를 새 배열로 모음)
    t3 = time.perf_counter()
    val = (0.6 * sel + 0.4 * np.array([200, 120, 60])).astype(np.uint8)  # float64 계산 후 uint8
    t4 = time.perf_counter()
    out[m] = val  # 불리언 색인 쓰기
    t5 = time.perf_counter()
    for k, a, b in (("copy", t0, t1), ("to_bool", t1, t2), ("gather", t2, t3), ("float_math", t3, t4), ("scatter", t4, t5)):
        t[k] = t.get(k, 0.0) + (b - a) * 1e3


def legacy_warp(ipm: IpmCompositor, images: list[np.ndarray], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """옛 IpmCompositor.warp: 카메라마다 owner == i 비교와 불리언 색인 복사를 매번 한다."""
    out = None
    for i, (img, (m1, m2)) in enumerate(zip(images, ipm.maps)):
        if img is None:
            continue
        w = cv2.remap(img, m1, m2, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if out is None:
            out = np.zeros_like(w)
        sel = ipm.owner == i
        out[sel] = w[sel]
    if out is None:
        out = np.zeros((ipm.grid.height, ipm.grid.width, 3), np.uint8)
    return out


def legacy_bev_layers(grid, ground_img, road_mask, coverage) -> np.ndarray:
    """옛 render_bev의 바닥 영상 깔기와 도로 덧칠 부분(고리, 자차, 트랙 그리기는 뺌)."""
    H, W = grid.height, grid.width
    canvas = np.full((H, W, 3), render.BG, np.uint8)
    if ground_img is not None:
        g = cv2.cvtColor(ground_img, cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
        tint = np.array([1.0, 0.95, 0.9], np.float32)
        layer = (g * 0.45 * tint).clip(0, 255).astype(np.uint8)
        sel = coverage if coverage is not None else np.ones((H, W), bool)
        canvas[sel] = layer[sel]
    if road_mask is not None:
        m = road_mask.astype(bool)
        if coverage is not None:
            m &= coverage
        canvas[m] = (0.5 * canvas[m] + 0.5 * np.array(render.ROAD)).astype(np.uint8)
    return canvas


def legacy_render_bev(grid, ground_img, road_mask, coverage) -> np.ndarray:
    """옛 render_bev 전체(트랙 없음). 고리와 자차 그리기는 지금 render.py의 함수를 같이 쓴다."""
    canvas = legacy_bev_layers(grid, ground_img, road_mask, coverage)
    render._draw_rings(canvas, grid)
    render._draw_box(
        canvas, grid, np.array([render.EGO_SIZE[0] / 2 - render.EGO_REAR_OFFSET, 0.0]), render.EGO_SIZE, 0.0,
        render.EGO, fill=True,
    )
    return canvas


# ---------------------------------------------------------------------------
# 시간 재기 도우미
# ---------------------------------------------------------------------------
def timeit(fn, reps: int, warm: int = 3) -> dict:
    """fn()을 warm번 돌려 데운 뒤 reps번 재서 평균, 중앙값, p95(ms)를 돌려준다."""
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    a = np.array(xs)
    return {"mean_ms": round(float(a.mean()), 3), "p50_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam_current.yaml")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--fraction", type=float, default=0.5, help="무작위 마스크에서 1인 화소 비율")
    ap.add_argument("--out", help="결과 JSON 경로. 예: out/measure/03_bench_mask_mac.json")
    args = ap.parse_args()

    cams, grid, _ = load_config(args.config)
    ipm = IpmCompositor(cams, grid)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (c.height, c.width, 3), dtype=np.uint8) for c in cams]
    masks = [(rng.random((c.height, c.width)) < args.fraction).astype(np.uint8) for c in cams]
    cov = ipm.coverage()
    ground = ipm.warp(frames)
    road = ipm.warp(masks, interpolation=0)
    names = [c.name for c in cams]
    R = args.reps

    res: dict = {}
    # ---- 카메라 영상 덧칠: 카메라 전부(실제 렌더링과 같은 수) ----
    res["annotate_nomask_legacy"] = timeit(lambda: [f.copy() for f in frames], R)
    res["annotate_mask_legacy"] = timeit(lambda: [legacy_annotate_overlay(f, m) for f, m in zip(frames, masks)], R)
    res["annotate_nomask_current"] = timeit(
        lambda: [render.annotate_camera(f, [], n, None) for f, n in zip(frames, names)], R)
    res["annotate_mask_current"] = timeit(
        lambda: [render.annotate_camera(f, [], n, m) for f, m, n in zip(frames, masks, names)], R)
    steps: dict = {}
    for _ in range(R):
        for f, m in zip(frames, masks):
            legacy_annotate_steps(f, m, steps)
    res["annotate_mask_legacy_steps_mean_ms"] = {k: round(v / R, 3) for k, v in steps.items()}

    # ---- BEV 합성(IPM remap + 카메라별 담당 영역 복사) ----
    res["warp_ground_legacy"] = timeit(lambda: legacy_warp(ipm, frames), R)
    res["warp_ground_current"] = timeit(lambda: ipm.warp(frames), R)
    res["warp_mask_legacy"] = timeit(lambda: legacy_warp(ipm, masks, 0), R)
    res["warp_mask_current"] = timeit(lambda: ipm.warp(masks, interpolation=0), R)

    # ---- 탑뷰 그리기(바닥 영상 + 도로 덧칠 + 고리, 자차. 트랙 없음) ----
    res["render_bev_noroad_legacy"] = timeit(lambda: legacy_render_bev(grid, ground, None, cov), R)
    res["render_bev_road_legacy"] = timeit(lambda: legacy_render_bev(grid, ground, road, cov), R)
    res["render_bev_noroad_current"] = timeit(lambda: render.render_bev(grid, [], ground, None, cov), R)
    res["render_bev_road_current"] = timeit(lambda: render.render_bev(grid, [], ground, road, cov), R)

    # ---- 결과 차이(새 방식이 옛 방식과 화소 몇만큼 다른지) ----
    a = legacy_annotate_overlay(frames[0], masks[0]).astype(int)
    b = render.annotate_camera(frames[0], [], "", masks[0]).astype(int)  # 이름이 빈 글자라 덧칠만 남는다
    res["max_abs_diff_annotate"] = int(np.abs(a - b).max())
    res["max_abs_diff_bev_layers"] = int(np.abs(
        legacy_render_bev(grid, ground, road, cov).astype(int) - render.render_bev(grid, [], ground, road, cov).astype(int)
    ).max())
    res["max_abs_diff_warp_ground"] = int(np.abs(legacy_warp(ipm, frames).astype(int) - ipm.warp(frames).astype(int)).max())

    meta = {
        "config": args.config, "reps": R, "mask_fraction": args.fraction,
        "camera_hw": [[c.height, c.width] for c in cams], "bev_hw": [grid.height, grid.width],
        "machine": platform.machine(), "node": platform.node(), "opencv": cv2.__version__, "numpy": np.__version__,
        "cv2_threads": cv2.getNumThreads(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rep = {"meta": meta, "results": res}
    for k, v in res.items():
        print(f"{k:40s} {v}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"결과: {args.out}")


if __name__ == "__main__":
    main()
