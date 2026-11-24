#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview.bev import IpmCompositor  # noqa: E402
from realityview.config import load_config  # noqa: E402
from realityview.lanes import LaneDetector, LaneSmoother, box_masks  # noqa: E402
from realityview.sim import SimBackend, SimWorld, default_traffic_lights  # noqa: E402
from realityview.traffic_lights import TrafficLightReader  # noqa: E402

# 합성 장면에서 차선 곡선 오차와 신호등 색 정확도, CPU 시간을 잰다(카메라와 NPU를 쓰지 않는다).
#
# 예)
#   python scripts/eval_lanes_lights.py --config configs/two_cam_current.yaml --frames 100 \
#       --out out/measure/15_eval_mac.json
# 결과: 표준 출력에 요약, --out에 JSON


# ---------------------------------------------------------------------------
# 한 조건 평가
# ---------------------------------------------------------------------------
def evaluate(config: str, curvature: float, frames: int, near=(3.0, 15.0), with_objects: bool = True,
             mask_objects: bool = True) -> dict:
    cams, grid, _ = load_config(config)
    world = SimWorld(cams, objects=None if with_objects else [], curvature=curvature,
                     traffic_lights=default_traffic_lights())
    backend = SimBackend(world)
    ipm = IpmCompositor(cams, grid)
    det, sm = LaneDetector(grid, None, ipm.coverage()), LaneSmoother()
    reader = TrafficLightReader(cams)
    gt_lanes = world.gt_lanes()
    errs, found, style_ok, color_ok, side_ok = [], 0, 0, 0, 0
    raw_ok = vote_ok = n_light = 0
    t_lane, t_light = [], []
    for _ in range(frames):
        world.step(0.1)
        imgs = world.render()
        results = backend.infer(imgs)
        ground = ipm.warp(imgs)
        road = ipm.warp(world.gt_masks, interpolation=0)
        # ---- 차선: 검출 + 평활 시간, 정답과 비교 ----
        t0 = time.perf_counter()
        masks = box_masks([f.shape[:2] for f in imgs], [r.detections for r in results], ("traffic light",))
        occ = ipm.warp(masks, interpolation=0) if (mask_objects and masks is not None) else None
        lanes = sm.update(det.detect(ground, road, occ))
        t_lane.append((time.perf_counter() - t0) * 1e3)
        for gt in gt_lanes:
            ln = min(lanes, key=lambda l: abs(l.offset - gt["c"]), default=None)
            if ln is None or abs(ln.offset - gt["c"]) > 0.6:
                continue
            found += 1
            x0, x1 = max(near[0], ln.x_range[0]), min(near[1], ln.x_range[1])
            if x1 > x0:
                xs = np.linspace(x0, x1, 25)
                errs.append(float(np.mean(np.abs(ln.y_at(xs) - (gt["a"] * xs ** 2 + gt["c"])))))
            style_ok += ln.style == gt["style"]
            color_ok += ln.color == gt["color"]
            side_ok += ln.side == gt["side"]
        # ---- 신호등: 상자를 정답과 짝지어 이번 프레임 판별(raw)과 다수결(state)을 비교 ----
        t0 = time.perf_counter()
        lights = reader.update(imgs, [r.detections for r in results])
        t_light.append((time.perf_counter() - t0) * 1e3)
        for lt in lights:
            gts = world.gt_lights[lt.cam_index] if lt.cam_index < len(world.gt_lights) else []
            if not gts:
                continue
            c = np.array([(lt.box[0] + lt.box[2]) / 2, (lt.box[1] + lt.box[3]) / 2])
            box, state, _ = min(gts, key=lambda g: np.hypot((g[0][0] + g[0][2]) / 2 - c[0], (g[0][1] + g[0][3]) / 2 - c[1]))
            n_light += 1
            raw_ok += lt.raw_state == state
            vote_ok += lt.state == state
    n_gt = frames * len(gt_lanes)
    return {
        "config": config,
        "curvature": curvature,
        "frames": frames,
        "objects": with_objects,
        "mask_objects": mask_objects,
        "lane_found_rate": round(found / n_gt, 4),
        "lane_err_mean_m": round(float(np.mean(errs)), 4) if errs else None,
        "lane_err_p95_m": round(float(np.percentile(errs, 95)), 4) if errs else None,
        "lane_style_acc": round(style_ok / max(found, 1), 4),
        "lane_color_acc": round(color_ok / max(found, 1), 4),
        "lane_side_acc": round(side_ok / max(found, 1), 4),
        "light_samples": n_light,
        "light_raw_acc": round(raw_ok / max(n_light, 1), 4),
        "light_vote_acc": round(vote_ok / max(n_light, 1), 4),
        "lane_ms_mean": round(float(np.mean(t_lane[5:])), 3),
        "lane_ms_p95": round(float(np.percentile(t_lane[5:], 95)), 3),
        "light_ms_mean": round(float(np.mean(t_light[5:])), 3),
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="합성 장면 차선, 신호등 평가")
    ap.add_argument("--config", action="append", default=None, help="설정 파일(여러 번 가능)")
    ap.add_argument("--curvature", default="0,0.006,-0.006", help="쉼표로 구분한 도로 곡률 목록(1/m)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", help="결과 JSON 경로")
    ap.add_argument("--no-mask-objects", action="store_true", help="검출 상자 자리를 빼지 않고 잰다(비교용)")
    args = ap.parse_args()
    configs = args.config or ["configs/two_cam_current.yaml"]
    rows = []
    for cfg in configs:
        for c in (float(v) for v in args.curvature.split(",")):
            r = evaluate(cfg, c, args.frames, mask_objects=not args.no_mask_objects)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
