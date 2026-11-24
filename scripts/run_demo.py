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
from realityview.profiling import (  # noqa: E402
    HealthLogger,
    LatencyRecorder,
    StageTimer,
    format_table,
    read_pi_health,
)

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
#   # DX-M1 단독 검출 (.dxnn을 models/에 두고 설정의 npu.dx_det_model에 적은 뒤)
#   timeout -s INT -k 10 180 python scripts/run_demo.py --config configs/two_cam_current.yaml \
#       --backend dxm1 --headless --frames 330 --profile --profile-out out/measure/02_profile_dxm1.json
#
#   # 두 NPU 동시: 검출은 Hailo-8, 도로 세그는 DX-M1 별도 스레드(npu.seg_every 프레임마다)
#   timeout -s INT -k 10 180 python scripts/run_demo.py --config configs/two_cam_current.yaml \
#       --backend hailo --seg-npu dxm1 --headless --frames 330 --profile --profile-out out/measure/02_profile_dual.json
#
#   # 30분 연속 실행(soak): 1분마다 FPS, 스로틀링, 온도, 메모리를 csv로. 누수를 보려면 --profile은 끈다
#   timeout -s INT -k 10 2000 python scripts/run_demo.py --config configs/two_cam_current.yaml \
#       --backend hailo --headless --duration 1800 --health-csv out/measure/08_soak.csv
#
#   # 결과 화면을 동영상으로 남기기(60초). --profile을 같이 주면 센서 시각에서 화면 완성까지의 지연도 잰다
#   timeout -s INT -k 10 120 python scripts/run_demo.py --config configs/two_cam_current.yaml \
#       --backend hailo --headless --duration 60 --record-video out/video/hailo_60s.mp4
#
#   # 3D 시점(자차 뒤쪽 위에서 내려다봄). both는 탑뷰와 3D를 나란히. 기본은 지금처럼 bev
#   python scripts/run_demo.py --backend sim --view 3d --view3d-interp --frames 150 --save out/sim3d.mp4 --headless
#
#   # 주행 녹화 회차를 그대로 재생하면서 트리거 클립을 모으고, DX-M1에서 섀도 회차 모델을 같이 돌려 비교
#   # (docs/drive_recording.md. DX-M1은 방열 전이라 30초 이내)
#   timeout -s INT -k 10 40 python scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo \
#       --replay out/drives/20261125_101500 --headless --duration 25 --trigger-clips --shadow \
#       --shadow-csv out/measure/shadow.csv --health-csv out/measure/shadow_health.csv --health-interval 10 --hailo-stop-temp 95
#
#   # PC에서 녹화 영상으로 개발 (ultralytics 필요)
#   python scripts/run_demo.py --backend ultralytics --video front=rec/front.mp4 --video rear=rec/rear.mp4


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam.yaml")
    ap.add_argument("--backend", choices=["hailo", "dxm1", "ultralytics", "sim"], default="sim")
    ap.add_argument(
        "--seg-npu",
        choices=["none", "dxm1"],
        default="none",
        help="도로 세그를 다른 NPU의 별도 스레드에서 실행(dxm1: 설정의 npu.dx_seg_model 사용). 기본 none",
    )
    ap.add_argument(
        "--seg-mask-test",
        choices=["none", "random50"],
        default="none",
        help="시험용: --seg-npu 세그는 그대로 돌리고 마스크만 화소 50%%가 1인 무작위 마스크로 바꿈(렌더링 비용 측정). 기본 none",
    )
    ap.add_argument(
        "--det-hef",
        default=None,
        help="Hailo 검출 HEF를 설정(npu.det_hef) 대신 이것으로. NMS 내장 HEF만. 예: /usr/share/hailo-models/yolov6n_h8.hef",
    )
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
    ap.add_argument("--warmup", type=int, default=30, help="통계에서 뺄 앞쪽 프레임 수(--profile일 때). 녹화도 이 뒤부터")
    # ---- 장시간 실행과 상태 기록(기본 꺼짐) ----
    ap.add_argument("--duration", type=float, default=0.0, help="이 시간(초)이 지나면 스스로 끝냄. 0이면 제한 없음")
    ap.add_argument("--health-csv", help="일정 간격으로 FPS, throttled, 온도, 메모리(RSS)를 남길 csv 경로")
    ap.add_argument("--health-interval", type=float, default=60.0, help="--health-csv 기록 간격(초). 기본 60")
    ap.add_argument(
        "--hailo-stop-temp",
        type=float,
        default=0.0,
        help="--health-csv 기록 때 읽은 Hailo-8 칩 온도가 이 값(도) 이상이면 스스로 끝냄. 0이면 끔. 예: 95",
    )
    # ---- 결과 화면 녹화(기본 꺼짐) ----
    ap.add_argument("--record-video", help="대시보드를 동영상으로 저장(.mp4). 예: out/video/hailo_60s.mp4")
    ap.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="녹화 파일 fps. 0이면 워밍업 동안 잰 루프 속도를 반올림해 쓴다(재생 속도는 실제 시간에 맞춤)",
    )
    ap.add_argument("--record-codec", default="avc1,mp4v", help="시도할 코덱 순서(쉼표 구분). 기본 avc1,mp4v")
    # ---- 화면 종류(기본 bev = 예전과 같은 탑뷰 대시보드) ----
    ap.add_argument(
        "--view",
        choices=["bev", "3d", "both"],
        default="bev",
        help="bev: 탑뷰(기본), 3d: 자차 뒤쪽 위 3D 시점을 크게, both: 탑뷰 + 3D 시점",
    )
    ap.add_argument("--view3d-size", default=None, help="3D 화면 크기 폭x높이. 기본 960x540(설정 view3d 항목이 있으면 그 값)")
    ap.add_argument("--view3d-scale", type=float, default=None, help="3D를 이 비율로 작게 그린 뒤 키움(0.1~1). 예: 0.5")
    ap.add_argument("--view3d-ground-every", type=int, default=1, help="3D 바닥 텍스처를 N 프레임마다 다시 만듦. 기본 1")
    ap.add_argument("--view3d-interp", action="store_true", help="3D 화면의 트랙 위치를 프레임 사이에서 부드럽게 보간")
    ap.add_argument(
        "--view3d-cam",
        default=None,
        help="가상 카메라 '뒤거리,높이,숙임각,수평화각'(실차 m, 도). 기본 7,4.5,18,70",
    )
    # ---- 추적기(기본 basic = 예전과 같음) ----
    ap.add_argument(
        "--tracker",
        choices=["basic", "robust"],
        default="basic",
        help="basic: 예전 추적기(기본), robust: 먼 물체 ID 쪼개짐을 고친 판(out/measure/11_tracker_latency.md)",
    )
    # ---- 저장 이미지 재생(기본 꺼짐) ----
    ap.add_argument(
        "--replay",
        default=None,
        help="카메라 대신 저장 영상을 넣음(끝나면 처음부터). 주행 녹화 회차나 세그먼트 폴더, 트리거 클립 폴더({카메라}.mjpeg), "
        "또는 '{카메라 이름}_*.png|jpg' 이미지 폴더. 예: out/drives/20261125_101500, out/rec/room_100",
    )
    # ---- 트리거 클립(기본 꺼짐, docs/drive_recording.md) ----
    ap.add_argument("--trigger-clips", action="store_true", help="애매한 점수, 새 ID와 바뀐 ID, 두 NPU 불일치 순간의 앞뒤 원본을 저장")
    ap.add_argument("--trigger-dir", default="out/triggers", help="트리거 클립 저장 폴더. 기본 out/triggers")
    ap.add_argument("--trigger-pre", type=float, default=2.0, help="트리거 앞쪽으로 남길 시간(초). 기본 2")
    ap.add_argument("--trigger-post", type=float, default=2.0, help="트리거 뒤쪽으로 남길 시간(초). 기본 2")
    ap.add_argument("--trigger-cooldown", type=float, default=10.0, help="같은 종류 트리거 사이 최소 간격(초). 기본 10")
    ap.add_argument("--trigger-score", default="0.3,0.5", help="애매한 점수 범위 '하한,상한'(하한 이상 상한 미만). 기본 0.3,0.5")
    ap.add_argument("--trigger-switch-dist", type=float, default=2.0, help="ID가 바뀐 것으로 볼 거리(m, 실차). 기본 2")
    ap.add_argument("--trigger-mismatch", type=int, default=1, help="--shadow에서 짝 못 찾은 상자가 이 수 이상이면 트리거. 기본 1")
    ap.add_argument("--trigger-buffer-mb", type=float, default=300.0, help="링 버퍼 최대 메모리(MB). 기본 300")
    # ---- 섀도 실행(기본 꺼짐): 같은 프레임을 다른 NPU의 다른 회차 모델로 돌려 차이를 csv로 ----
    ap.add_argument(
        "--shadow",
        action="store_true",
        help="--backend hailo면 DX-M1, dxm1이면 Hailo에서 설정 models.shadow_model_round 모델을 같이 돌림. 화면과 추적은 기본 모델만",
    )
    ap.add_argument("--shadow-csv", default="out/measure/shadow.csv", help="프레임별 비교 csv 경로. 기본 out/measure/shadow.csv")
    ap.add_argument("--shadow-iou", type=float, default=0.5, help="두 결과의 상자를 같은 물체로 볼 IoU. 기본 0.5")
    # ---- CSI 카메라 지연 시험 옵션(기본은 모두 설정 파일 값 그대로) ----
    ap.add_argument("--cam-fps", type=float, default=None, help="Picamera2 프레임레이트를 이 값으로 덮어씀. 예: 25")
    ap.add_argument("--cam-buffers", type=int, default=None, help="Picamera2 buffer_count. 기본은 Picamera2 값(6)")
    ap.add_argument(
        "--cam-queue",
        choices=["on", "off"],
        default=None,
        help="off: Picamera2가 끝난 프레임을 들고 있지 않고 늘 다음 프레임을 기다림. 기본은 Picamera2 값(on)",
    )
    ap.add_argument("--cam-flush", action="store_true", help="read()마다 부른 뒤에 노출을 시작한 프레임만 받음")
    return ap


def camera_override(args) -> dict:
    """--cam-* 옵션을 picamera2 소스 설정에 덮어쓸 값으로 바꾼다. 아무것도 안 주면 빈 사전(설정 그대로)."""
    ov = {}
    if args.cam_fps is not None:
        ov["fps"] = float(args.cam_fps)
    if args.cam_buffers is not None:
        ov["buffer_count"] = int(args.cam_buffers)
    if args.cam_queue is not None:
        ov["queue"] = args.cam_queue == "on"
    if args.cam_flush:
        ov["flush"] = True
    return ov


# ---------------------------------------------------------------------------
# 3D 시점 설정 만들기
# ---------------------------------------------------------------------------
def make_view3d_config(args, cfg: dict):
    """3D 시점 설정. 설정 파일의 view3d 항목(있으면)을 바탕으로 명령줄 옵션을 덮어쓴다."""
    from realityview.render3d import View3DConfig

    base = dict(cfg.get("view3d") or {})
    if args.view3d_size:
        w, h = args.view3d_size.lower().split("x")
        base["width"], base["height"] = int(w), int(h)
    if args.view3d_scale is not None:
        base["render_scale"] = float(args.view3d_scale)
    if args.view3d_interp:
        base["interp"] = True
    if args.view3d_cam:
        vals = [float(v) for v in args.view3d_cam.split(",")]
        if len(vals) != 4:
            raise SystemExit("--view3d-cam 은 '뒤거리,높이,숙임각,수평화각' 네 값입니다. 예: 7,4.5,18,70")
        base["cam_back"], base["cam_height"], base["pitch_deg"], base["hfov_deg"] = vals
    return View3DConfig(**base).checked()


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
            "seg_npu": args.seg_npu,
            "seg_mask_test": args.seg_mask_test,
            "view": args.view,
            "view3d": dataclasses.asdict(pipe.r3d.cfg) if pipe.r3d is not None else None,
            "view3d_ground_every": pipe.ground_every,
            "warmup": timer.warmup,
        },
        "seg_stats": pipe.seg_stats,
        # 탑뷰 도로 마스크 캐시: 마스크가 그대로라 합성을 건너뛴 수(hits)와 다시 합성한 수(misses)
        "road_cache": {"hits": pipe.road_cache_hits, "misses": pipe.road_cache_misses},
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
# DX-M1 백엔드 만들기
# ---------------------------------------------------------------------------
def make_dxm1_backend(npu: dict, timer: StageTimer, det: bool):
    """설정의 npu 항목으로 DXM1Backend를 만든다.

    det=True: --backend dxm1 (검출 + 설정에 dx_seg_model이 있으면 세그도 같은 백엔드에서)
    det=False: --seg-npu dxm1 (세그만, 파이프라인의 별도 스레드가 segment()를 부른다)
    모델 경로가 설정에 없거나 파일이 없으면 무엇을 어디에 둘지 알려 주는 오류가 난다.
    """
    from realityview.detectors import DX_MODEL_HELP, DXM1Backend

    key = "dx_det_model" if det else "dx_seg_model"
    if not npu.get(key):
        raise SystemExit(f"설정 파일의 npu.{key} 가 비어 있습니다.\n" + DX_MODEL_HELP.format(key=key))
    return DXM1Backend(
        npu["dx_det_model"] if det else None,
        npu.get("dx_seg_model") if det else npu["dx_seg_model"],
        score_thr=float(npu.get("score_thr", 0.4)),
        iou_thr=float(npu.get("dx_iou_thr", 0.45)),
        road_classes=npu.get("road_classes", [0]),
        seg_every=int(npu.get("seg_every", 3)),
        timer=timer,
        det_format=npu.get("dx_det_format", "yolov8"),
        input_rgb=bool(npu.get("dx_input_rgb", True)),
        seg_channels_first=bool(npu.get("dx_seg_channels_first", True)),
    )


# ---------------------------------------------------------------------------
# 녹화기 만들기
# ---------------------------------------------------------------------------
def make_recorder(args, frames_so_far: int, elapsed_s: float):
    """--record-video 녹화기를 만든다. fps를 정하지 않았으면 지금까지의 루프 속도를 반올림해 쓴다."""
    from realityview.recorder import VideoRecorder

    fps = float(args.record_fps)
    if fps <= 0:
        # 워밍업이 0이면 잰 값이 없으니 15로 둔다. 시간축은 녹화기가 실제 시각에 맞춘다
        fps = float(max(1, round(frames_so_far / elapsed_s))) if frames_so_far > 0 and elapsed_s > 0 else 15.0
    codecs = tuple(c.strip() for c in args.record_codec.split(",") if c.strip())
    print(f"녹화 시작: {args.record_video} fps={fps:g} 코덱 후보={codecs}", flush=True)
    return VideoRecorder(args.record_video, fps, codecs=codecs)


# ---------------------------------------------------------------------------
# 회차 모델, 트리거 점수 범위, 섀도 백엔드
# ---------------------------------------------------------------------------
def parse_score_range(ap, text: str) -> tuple[float, float]:
    """'0.3,0.5' 를 (0.3, 0.5)로. 하한 < 상한, 0~1 범위가 아니면 명령줄 오류."""
    try:
        lo, hi = (float(v) for v in text.split(","))
    except ValueError:
        ap.error(f"--trigger-score 는 '하한,상한' 형식입니다: {text}")
    if not (0.0 <= lo < hi <= 1.0):
        ap.error(f"--trigger-score 는 0 <= 하한 < 상한 <= 1 이어야 합니다: {text}")
    return lo, hi


def detector_npu(args, cfg: dict, npu: dict, trig_lo: float) -> tuple[dict, int | None]:
    """검출 백엔드에 넘길 npu 설정 사본과 기본 회차 번호.

    - 모델 경로: --det-hef(Hailo만) > models.det_model_round 회차 > npu.det_hef / npu.dx_det_model
    - --trigger-clips: 애매한 점수 하한까지 보려고 백엔드 점수 문턱을 min(score_thr, 하한)으로 낮춘다.
      화면과 추적에는 ScoreTap이 원래 문턱 이상만 넘기므로 결과는 그대로다.
    """
    from realityview.model_rounds import primary_det_model

    out = dict(npu)
    rnd = None
    if args.backend in ("hailo", "dxm1"):
        path, rnd = primary_det_model(cfg, args.backend)
        key = "det_hef" if args.backend == "hailo" else "dx_det_model"
        if path:
            out[key] = path
        if args.backend == "hailo" and args.det_hef:
            out[key], rnd = args.det_hef, None
        if args.trigger_clips:
            out["score_thr"] = min(float(npu.get("score_thr", 0.4)), float(trig_lo))
    return out, rnd


def make_shadow_backend(args, cfg: dict, npu: dict, primary, camera_names):
    """기본 백엔드의 반대쪽 NPU에 섀도 회차 모델을 올린다. 섀도 쪽 시간은 --profile 표에 섞이지 않게 따로 둔다."""
    from realityview.model_rounds import shadow_det_model
    from realityview.shadow import ShadowBackend

    other = "dxm1" if args.backend == "hailo" else "hailo"
    path, rnd = shadow_det_model(cfg, other)
    if not path:
        raise SystemExit(f"섀도 모델 경로가 없습니다. 설정에 models 항목이나 npu.{'dx_det_model' if other == 'dxm1' else 'det_hef'} 를 적으세요")
    quiet = StageTimer(enabled=False)
    if other == "dxm1":
        # 세그 모델은 올리지 않는다(검출 비교만)
        sh = make_dxm1_backend(dict(npu, dx_det_model=path, dx_seg_model=None), quiet, det=True)
    else:
        from realityview.detectors import HailoBackend

        sh = HailoBackend(path, None, score_thr=float(npu.get("score_thr", 0.4)), timer=quiet)
    sb = ShadowBackend(primary, sh, args.shadow_csv, camera_names, iou_thr=args.shadow_iou)
    sb.shadow_round, sb.shadow_path = rnd, path
    return sb


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if args.seg_mask_test != "none" and args.seg_npu == "none":
        # 바꿔 줄 세그 출력이 없으니 아무 효과가 없다. 조용히 넘어가지 않고 알려 준다
        ap.error("--seg-mask-test 는 --seg-npu dxm1 과 같이 써야 합니다")
    if args.hailo_stop_temp > 0 and not args.health_csv:
        # 온도는 --health-csv 기록 때만 읽으므로, 없으면 멈춤 조건이 조용히 꺼진다. 미리 알려 준다
        ap.error("--hailo-stop-temp 는 --health-csv 와 같이 써야 합니다")
    if args.shadow and args.backend not in ("hailo", "dxm1"):
        ap.error("--shadow 는 --backend hailo 나 dxm1 과 같이 써야 합니다(다른 NPU에서 섀도 모델을 돌림)")
    if args.shadow and args.seg_npu != "none":
        # DX-M1에 세그와 섀도 검출을 함께 올리면 방열 없는 지금 상태에서 너무 뜨거워지고 비교 조건도 달라진다
        ap.error("--shadow 와 --seg-npu 는 같이 쓸 수 없습니다")
    trig_lo, trig_hi = parse_score_range(ap, args.trigger_score)

    cams, grid, cfg = load_config(args.config)
    if args.bev_res:
        grid = dataclasses.replace(grid, res=float(args.bev_res))
    npu = cfg.get("npu", {})
    # 검출 백엔드에 넘길 npu 설정. 회차 모델(models.det_model_round)과 트리거용 낮은 점수 문턱을 여기서 반영한다
    det_npu, det_round = detector_npu(args, cfg, npu, trig_lo)
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
        from realityview.sources import open_replay_sources, open_source

        overrides = dict(v.split("=", 1) for v in args.video)
        cam_ov = camera_override(args)
        try:
            if args.replay:
                # 저장 영상 재생: 카메라를 열지 않는다. 주행 녹화(.mjpeg)와 이미지 폴더 모두 받는다
                sources.extend(open_replay_sources(args.replay, cams))
            for c in ([] if args.replay else cams):
                if c.name in overrides:
                    sources.append(open_source(c, {"type": "opencv", "uri": overrides[c.name]}))
                else:
                    # --cam-* 옵션은 CSI 카메라(picamera2)에만 덮어쓴다
                    is_pi = c.source.get("type") == "picamera2"
                    sources.append(open_source(c, cam_ov if (cam_ov and is_pi) else None))
        except Exception:
            # 두 번째 카메라에서 실패해도 먼저 연 카메라는 놓아 준다
            for s in sources:
                s.close()
            raise
        if args.backend == "hailo":
            from realityview.detectors import HailoBackend

            backend = HailoBackend(
                det_npu["det_hef"],
                npu.get("seg_hef"),
                score_thr=float(det_npu.get("score_thr", 0.4)),
                road_classes=npu.get("road_classes", [0]),
                seg_every=int(npu.get("seg_every", 3)),
                timer=timer,
            )
        elif args.backend == "dxm1":
            try:
                backend = make_dxm1_backend(det_npu, timer, det=True)
            except BaseException:
                # 모델 파일이 없어 실패해도 이미 연 카메라는 놓아 준다
                for s in sources:
                    s.close()
                raise
        else:
            from realityview.detectors import UltralyticsBackend

            backend = UltralyticsBackend(args.model, float(npu.get("score_thr", 0.4)))

    # ---- 두 NPU 동시 실행 옵션: 세그를 DX-M1 별도 스레드에서 ----
    seg_backend = None
    if args.seg_npu == "dxm1":
        try:
            seg_backend = make_dxm1_backend(npu, timer, det=False)
        except BaseException:
            # 세그 모델이 없으면 이미 연 카메라와 검출 NPU를 놓아 주고 오류를 그대로 보여 준다
            for s in sources:
                s.close()
            backend.close()
            raise
        if args.seg_mask_test == "random50":
            # NPU 추론은 그대로 두고 마스크만 무작위 50%로 바꾼다(렌더링 비용을 정해진 도로 비율로 재기)
            from realityview.sim import ForcedMaskSeg

            seg_backend = ForcedMaskSeg(seg_backend, fraction=0.5)

    # ---- 트리거용 낮은 점수 상자 엿보기와 섀도 실행(옵션을 켰을 때만 백엔드를 감싼다) ----
    tap = None
    shadow = None
    if args.trigger_clips and args.backend in ("hailo", "dxm1"):
        from realityview.shadow import ScoreTap

        # 백엔드는 낮은 문턱으로 돌고, 파이프라인에는 원래 문턱(npu.score_thr) 이상만 간다
        backend = tap = ScoreTap(backend, float(npu.get("score_thr", 0.4)))
    if args.shadow:
        try:
            shadow = make_shadow_backend(args, cfg, npu, backend, [c.name for c in cams])
        except BaseException:
            for s in sources:
                s.close()
            backend.close()
            raise
        backend = shadow
    if det_round is not None or shadow is not None:
        print(f"검출 모델 회차: 기본 {det_round} ({args.backend})"
              + (f", 섀도 {shadow.shadow_round} ({shadow.shadow_path})" if shadow is not None else ""), flush=True)

    pipe = RealityViewPipeline(
        cams,
        backend,
        grid,
        show_ground=not args.no_ground,
        timer=timer,
        render_thread=args.render_thread,
        seg_backend=seg_backend,
        seg_every=int(npu.get("seg_every", 3)),
        scale=float((cfg.get("scene") or {}).get("scale", 1.0)),  # 모형 설정은 scene.scale (없으면 실차 1.0)
        view=args.view,
        view3d=make_view3d_config(args, cfg) if args.view != "bev" else None,
        ground_every=args.view3d_ground_every,
        tracker=args.tracker,
    )

    # ---- 메인 루프 ----
    writer = None
    out = None
    n = 0
    det_frames = 0  # 검출이 하나라도 있었던 프레임 수(확인용)
    det_total = 0
    loop = {"frames": 0, "measured_frames": 0, "measured_renders": 0, "elapsed_s": 0.0}
    t_measure0, renders0 = None, 0
    # 지연 측정: 렌더링 스레드 모드에서는 step()이 돌려준 화면이 이번 프레임 것이 아니라서 재지 않는다
    latency = None
    sensor_hist: dict[int, int | None] = {}  # 프레임 번호 -> 카메라 0번 센서 시각(세그 마스크 나이 계산용)
    if args.profile and sources and not args.render_thread and all(hasattr(s, "want_timestamp") for s in sources):
        for s in sources:
            s.want_timestamp = True
        latency = LatencyRecorder([c.name for c in cams], warmup=args.warmup)
    recorder = None
    rec_stats = None
    clipper = None
    trig_stats = None
    cam_names = [c.name for c in cams]
    if args.trigger_clips:
        from realityview.triggers import TrackIdWatcher, TriggerClipper, ambiguous_event, mismatch_event

        clipper = TriggerClipper(
            args.trigger_dir, cam_names, pre_s=args.trigger_pre, post_s=args.trigger_post,
            cooldown_s=args.trigger_cooldown, max_buffer_mb=args.trigger_buffer_mb,
        )
        scale = float((cfg.get("scene") or {}).get("scale", 1.0))
        id_watch = TrackIdWatcher(switch_dist=args.trigger_switch_dist * scale)
        if shadow is None:
            print("트리거 클립: 두 NPU 불일치(npu_mismatch) 트리거는 --shadow 를 같이 줄 때만 동작합니다", flush=True)
    health = None
    if args.health_csv:
        # Hailo 백엔드면 칩 온도 열을 더한다(방열 없이 오래 돌리면 Hailo-8이 먼저 뜨거워진다)
        temp_fn = getattr(backend, "chip_temperature", None) if world is None else None
        health = HealthLogger(
            args.health_csv,
            args.health_interval,
            extra_fn=(lambda: {"hailo_temp_c": temp_fn()}) if temp_fn else None,
            extra_columns=("hailo_temp_c",) if temp_fn else (),
        )
    if health is not None:
        health.start()
    t_first = last = time.perf_counter()
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
                    frames, read_done = [], []
                    for s in sources:
                        frames.append(s.read())
                        read_done.append(time.monotonic_ns())  # 지연 측정용: 영상을 손에 넣은 시각
            if any(f is None for f in frames):
                break
            if world is None:
                now = time.perf_counter()
                dt, last = now - last, now
            # 2) 검출부터 렌더링까지
            out = pipe.step(frames, dt)
            done_ns = time.monotonic_ns()  # 대시보드 완성 시각(녹화, 화면 표시 전)
            # 지연 기록(--profile이고 센서 타임스탬프가 있을 때만)
            if latency is not None:
                sensor_ns = [getattr(s, "last_timestamp_ns", None) for s in sources]
                sensor_hist[n] = sensor_ns[0]
                sensor_hist.pop(n - 256, None)  # 세그 마스크를 찾는 데 필요한 만큼만 남긴다
                seg_ns = sensor_hist.get(pipe.last_mask_idx) if pipe.last_mask_idx is not None else None
                latency.add(sensor_ns, read_done, done_ns, seg_ns)
            # 트리거 클립: 원본을 링 버퍼에 넣고, 이번 프레임에서 나온 트리거를 넘긴다
            if clipper is not None:
                t_now = time.monotonic()
                clipper.push(n, frames, t_now)
                events = [ambiguous_event(n, tap.last_raw if tap is not None else out.results, cam_names, trig_lo, trig_hi)]
                events += id_watch.update(n, out.tracks)
                if shadow is not None:
                    events.append(mismatch_event(n, shadow.last_rows, args.trigger_mismatch))
                clipper.fire([e for e in events if e is not None], t_now)
            ndet = sum(len(r.detections) for r in out.results)
            det_total += ndet
            det_frames += int(ndet > 0)
            # 3) 저장과 화면 표시
            if args.record_video and n >= args.warmup:
                if recorder is None:
                    recorder = make_recorder(args, n, t_loop - t_first)
                recorder.submit(out.dashboard)
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
            row = health.tick() if health is not None else None
            hot = row.get("hailo_temp_c") if row else None
            if args.hailo_stop_temp > 0 and hot is not None and hot >= args.hailo_stop_temp:
                print(f"Hailo-8 온도 {hot}도가 --hailo-stop-temp {args.hailo_stop_temp:g}도 이상이라 끝냅니다", flush=True)
                break
            if args.duration > 0 and time.perf_counter() - t_first >= args.duration:
                print(f"--duration {args.duration:g}초가 지나 끝냅니다", flush=True)
                break
    except KeyboardInterrupt:
        # timeout -s INT로 끊겨도 아래 finally에서 측정 결과를 남기고 장치를 놓는다
        print("중단 신호를 받아 정리합니다", flush=True)
    finally:
        loop["frames"] = n
        if t_measure0 is not None:
            loop["elapsed_s"] = time.perf_counter() - t_measure0
            loop["measured_frames"] = n - timer.warmup
            loop["measured_renders"] = pipe.render_count - renders0
        if health is not None:
            health.close()
        if clipper is not None:
            # 모으던 클립까지 저장하고 쓰기 스레드를 끝낸다
            trig_stats = clipper.close()
            print(f"trigger_stats={json.dumps(trig_stats, ensure_ascii=False)}", flush=True)
        if shadow is not None:
            print(f"shadow_summary={json.dumps(shadow.summary(), ensure_ascii=False)}", flush=True)
        if recorder is not None:
            # 큐에 남은 영상을 다 쓰고 닫는다. 측정 JSON에 녹화 결과를 같이 남기려고 먼저 닫는다
            rec_stats = recorder.close()
            print(f"녹화: {rec_stats}", flush=True)
        # 측정 결과는 장치를 닫기 전에 먼저 쓴다(닫다가 멈춰도 결과는 남게)
        if args.profile:
            extra = {"det_frames": det_frames, "det_total": det_total, "record": rec_stats,
                     "triggers": trig_stats, "shadow": shadow.summary() if shadow is not None else None}
            if latency is not None:
                extra["latency"] = latency.summary()
            rep = profile_report(args, timer, pipe, health_start, loop, extra)
            if latency is not None:
                print(f"latency={json.dumps(extra['latency'], ensure_ascii=False)}", flush=True)
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
