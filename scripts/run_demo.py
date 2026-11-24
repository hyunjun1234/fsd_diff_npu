#!/usr/bin/env python3
"""리얼리티 뷰 1단계 데모 실행.

예)
  # 하드웨어 없이 합성 장면으로 확인
  python scripts/run_demo.py --backend sim --frames 150 --save out/sim.mp4 --headless

  # 라즈베리파이 5 + Hailo-8 + CSI 카메라 2대
  python scripts/run_demo.py --config configs/two_cam.yaml --backend hailo

  # PC에서 녹화 영상으로 개발 (ultralytics 필요)
  python scripts/run_demo.py --backend ultralytics --video front=rec/front.mp4 --video rear=rec/rear.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.pipeline import RealityViewPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam.yaml")
    ap.add_argument("--backend", choices=["hailo", "ultralytics", "sim"], default="sim")
    ap.add_argument("--video", action="append", default=[], help="name=path 형식으로 카메라 입력을 파일로 대체")
    ap.add_argument("--model", default="yolov8n.pt", help="ultralytics 백엔드 모델")
    ap.add_argument("--frames", type=int, default=0, help="0이면 무한")
    ap.add_argument("--fps", type=float, default=10.0, help="sim 백엔드의 시뮬레이션 프레임레이트")
    ap.add_argument("--save", help="결과 동영상 저장 경로(.mp4)")
    ap.add_argument("--snapshot", help="마지막 프레임 이미지 저장 경로")
    ap.add_argument("--headless", action="store_true", help="창을 띄우지 않음")
    ap.add_argument("--no-ground", action="store_true", help="IPM 바닥 영상 표시 끔(CPU 절약)")
    args = ap.parse_args()

    cams, grid, cfg = load_config(args.config)
    npu = cfg.get("npu", {})

    world = None
    sources = []
    if args.backend == "sim":
        from realityview.sim import SimBackend, SimWorld

        world = SimWorld(cams)
        backend = SimBackend(world)
    else:
        from realityview.sources import open_source

        overrides = dict(v.split("=", 1) for v in args.video)
        sources = [
            open_source(c, {"type": "opencv", "uri": overrides[c.name]} if c.name in overrides else None) for c in cams
        ]
        if args.backend == "hailo":
            from realityview.detectors import HailoBackend

            backend = HailoBackend(
                npu["det_hef"],
                npu.get("seg_hef"),
                score_thr=float(npu.get("score_thr", 0.4)),
                road_classes=npu.get("road_classes", [0]),
                seg_every=int(npu.get("seg_every", 3)),
            )
        else:
            from realityview.detectors import UltralyticsBackend

            backend = UltralyticsBackend(args.model, float(npu.get("score_thr", 0.4)))

    pipe = RealityViewPipeline(cams, backend, grid, show_ground=not args.no_ground)
    writer = None
    n = 0
    last = time.perf_counter()
    try:
        while args.frames == 0 or n < args.frames:
            if world is not None:
                dt = 1.0 / args.fps
                world.step(dt)
                frames = world.render()
            else:
                frames = [s.read() for s in sources]
                if any(f is None for f in frames):
                    break
                now = time.perf_counter()
                dt, last = now - last, now
            out = pipe.step(frames, dt)
            if args.save:
                if writer is None:
                    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                    h, w = out.dashboard.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                writer.write(out.dashboard)
            if not args.headless:
                cv2.imshow("reality view", out.dashboard)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if n % 30 == 0:
                t = out.timings
                print(
                    f"frame {n}: tracks={len(out.tracks)} npu={t['npu_ms']:.1f}ms "
                    f"bev={t['bev_ms']:.1f}ms render={t['render_ms']:.1f}ms",
                    flush=True,
                )
            n += 1
    finally:
        if args.snapshot and n > 0:
            os.makedirs(os.path.dirname(args.snapshot) or ".", exist_ok=True)
            cv2.imwrite(args.snapshot, out.dashboard)
        if writer is not None:
            writer.release()
        for s in sources:
            s.close()
        pipe.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
