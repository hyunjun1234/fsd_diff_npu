#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.pipeline import RealityViewPipeline  # noqa: E402
from realityview.profiling import StageTimer, format_table, read_pi_health  # noqa: E402

# 리얼리티 뷰 1단계 데모 실행.
#
# 예)
#   # 하드웨어 없이 합성 장면으로 확인
#   python scripts/run_demo.py --backend sim --frames 150 --save out/sim.mp4 --headless
#
#   # 라즈베리파이 5 + Hailo-8 + 지금 꽂힌 CSI 카메라 2대, 단계별 시간 측정
#   # (카메라와 NPU를 잡으므로 timeout과 --frames를 꼭 붙인다)
#   timeout -s INT -k 10 180 python scripts/run_demo.py --config configs/two_cam_current.yaml \
#       --backend hailo --headless --frames 330 --profile --profile-out out/measure/01_profile_base.json
#
#   # PC에서 녹화 영상으로 개발 (ultralytics 필요)
#   python scripts/run_demo.py --backend ultralytics --video front=rec/front.mp4 --video rear=rec/rear.mp4


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
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
    # ---- 계획 3.3절 CPU 절약 대응(기본은 모두 꺼짐 = 기존 동작) ----
    ap.add_argument("--no-ground", action="store_true", help="IPM 바닥 영상 합성과 표시를 끔(CPU 절약)")
    ap.add_argument("--bev-res", type=float, default=None, help="BEV 격자 해상도(m/px) 덮어쓰기. 예: 0.1")
    ap.add_argument("--render-thread", action="store_true", help="렌더링을 별도 스레드에서 실행")
    # ---- 단계별 시간 측정(기본 꺼짐) ----
    ap.add_argument("--profile", action="store_true", help="단계별 처리 시간을 재서 표로 출력")
    ap.add_argument("--profile-out", help="측정 결과 JSON 경로(--profile일 때). 예: out/measure/01_profile.json")
    ap.add_argument("--warmup", type=int, default=30, help="통계에서 뺄 앞쪽 프레임 수(--profile일 때)")
    return ap


# ---------------------------------------------------------------------------
# 측정 결과 정리
# ---------------------------------------------------------------------------
def profile_report(args, timer: StageTimer, pipe, health_start: dict, loop: dict, extra: dict) -> dict:
    """측정값을 JSON으로 남길 사전 하나로 모은다.

    - loop_fps: 워밍업 뒤 루프가 돈 횟수 / 걸린 시간(벽시계). 카메라 획득까지 모두 포함한 값이다.
    - display_fps: 같은 시간 동안 완성된 대시보드 수 / 시간. 렌더링 스레드 모드에서는 loop_fps보다 낮을 수 있고,
      "화면 갱신" 목표 판정에는 이 값을 쓴다.
    """
    summary = timer.summary()
    elapsed = loop.get("elapsed_s") or 0.0
    measured = loop.get("measured_frames", 0)
    renders = loop.get("measured_renders", 0)
    return {
        "config": args.config,
        "backend": args.backend,
        "options": {
            "no_ground": args.no_ground,
            "bev_res": pipe.grid.res,
            "bev_size_hw": [pipe.grid.height, pipe.grid.width],
            "render_thread": args.render_thread,
            "warmup": timer.warmup,
        },
        "cameras": [{"name": c.name, "width": c.width, "height": c.height} for c in pipe.cameras],
        "frames_total": loop.get("frames", 0),
        "frames_measured": measured,
        "elapsed_s": round(elapsed, 3),
        "loop_fps": round(measured / elapsed, 2) if elapsed > 0 else None,
        "display_fps": round(renders / elapsed, 2) if elapsed > 0 else None,
        "render_dropped": pipe.render_dropped,
        "health_start": health_start,
        "health_end": read_pi_health(),
        "stages_ms": {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in summary.items()},
        "table_md": format_table(summary),
        **extra,
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    cams, grid, cfg = load_config(args.config)
    if args.bev_res:
        grid = dataclasses.replace(grid, res=float(args.bev_res))
    npu = cfg.get("npu", {})
    timer = StageTimer(enabled=args.profile, warmup=args.warmup)
    health_start = read_pi_health() if args.profile else {}

    # ---- 입력과 백엔드 준비 ----
    world = None
    sources = []
    if args.backend == "sim":
        from realityview.sim import SimBackend, SimWorld

        world = SimWorld(cams)
        backend = SimBackend(world)
    else:
        from realityview.sources import open_source

        overrides = dict(v.split("=", 1) for v in args.video)
        try:
            for c in cams:
                sources.append(open_source(c, {"type": "opencv", "uri": overrides[c.name]} if c.name in overrides else None))
        except Exception:
            # 두 번째 카메라에서 실패해도 먼저 연 카메라는 놓아 준다
            for s in sources:
                s.close()
            raise
        if args.backend == "hailo":
            from realityview.detectors import HailoBackend

            backend = HailoBackend(
                npu["det_hef"],
                npu.get("seg_hef"),
                score_thr=float(npu.get("score_thr", 0.4)),
                road_classes=npu.get("road_classes", [0]),
                seg_every=int(npu.get("seg_every", 3)),
                timer=timer,
            )
        else:
            from realityview.detectors import UltralyticsBackend

            backend = UltralyticsBackend(args.model, float(npu.get("score_thr", 0.4)))

    pipe = RealityViewPipeline(
        cams, backend, grid, show_ground=not args.no_ground, timer=timer, render_thread=args.render_thread
    )

    # ---- 메인 루프 ----
    writer = None
    out = None
    n = 0
    det_frames = 0  # 검출이 하나라도 있었던 프레임 수(확인용)
    det_total = 0
    loop = {"frames": 0, "measured_frames": 0, "measured_renders": 0, "elapsed_s": 0.0}
    t_measure0, renders0 = None, 0
    last = time.perf_counter()
    try:
        while args.frames == 0 or n < args.frames:
            t_loop = time.perf_counter()
            if n == timer.warmup:
                # 워밍업이 끝난 시점부터 FPS를 잰다
                t_measure0, renders0 = t_loop, pipe.render_count
            # 1) 카메라 획득. Picamera2는 다음 프레임이 나올 때까지 기다리므로 그 대기 시간도 들어간다
            with timer.stage("capture"):
                if world is not None:
                    dt = 1.0 / args.fps
                    world.step(dt)
                    frames = world.render()
                else:
                    frames = [s.read() for s in sources]
            if any(f is None for f in frames):
                break
            if world is None:
                now = time.perf_counter()
                dt, last = now - last, now
            # 2) 검출부터 렌더링까지
            out = pipe.step(frames, dt)
            ndet = sum(len(r.detections) for r in out.results)
            det_total += ndet
            det_frames += int(ndet > 0)
            # 3) 저장과 화면 표시
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
                    f"frame {n}: dets={ndet} tracks={len(out.tracks)} npu={t['npu_ms']:.1f}ms "
                    f"bev={t['bev_ms']:.1f}ms render={t['render_ms']:.1f}ms",
                    flush=True,
                )
            timer.end_frame((time.perf_counter() - t_loop) * 1e3)
            n += 1
    except KeyboardInterrupt:
        # timeout -s INT로 끊겨도 아래 finally에서 측정 결과를 남기고 장치를 놓는다
        print("중단 신호를 받아 정리합니다", flush=True)
    finally:
        loop["frames"] = n
        if t_measure0 is not None:
            loop["elapsed_s"] = time.perf_counter() - t_measure0
            loop["measured_frames"] = n - timer.warmup
            loop["measured_renders"] = pipe.render_count - renders0
        # 측정 결과는 장치를 닫기 전에 먼저 쓴다(닫다가 멈춰도 결과는 남게)
        if args.profile:
            rep = profile_report(
                args, timer, pipe, health_start, loop, {"det_frames": det_frames, "det_total": det_total}
            )
            print(rep["table_md"], flush=True)
            print(
                f"loop_fps={rep['loop_fps']} display_fps={rep['display_fps']} frames={n} "
                f"health_start={rep['health_start']} health_end={rep['health_end']}",
                flush=True,
            )
            if args.profile_out:
                os.makedirs(os.path.dirname(args.profile_out) or ".", exist_ok=True)
                with open(args.profile_out, "w", encoding="utf-8") as f:
                    json.dump(rep, f, ensure_ascii=False, indent=2)
        if args.snapshot and out is not None:
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
