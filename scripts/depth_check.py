from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.depth import (  # noqa: E402
    colorize_depth,
    compare_boxes_with_ipm,
    fit_depth_scale,
    ground_samples,
    guess_kind,
)
from realityview.profiling import StageTimer  # noqa: E402
from realityview.types import COCO_KEEP  # noqa: E402

# DX-M1 단안 깊이 모델을 카메라로 돌려 보는 도구.
#   - 카메라 N대에서 프레임을 받아 깊이 모델을 돌리고 NPU 시간과 FPS를 잰다.
#   - 마지막 프레임의 깊이 맵을 컬러로 칠해 카메라별로 저장한다(왼쪽 원본, 오른쪽 깊이, 가까울수록 밝음).
#   - 깊이인지 역깊이인지 추정(guess_kind)과, 지면 점으로 맞춘 축척, 검출 박스 아래 깊이와 IPM 거리 비교를
#     JSON에 남긴다. 보정 전에는 IPM 거리가 의미 없으므로 비교 값은 코드 확인용이다.
# DX-M1에 방열팬이 없으므로 --duration(기본 20초)을 넘기면 스스로 멈춘다. 바깥에도 timeout을 건다.
#
# 예(파이 저장소 폴더, .venv):
#   timeout -s INT -k 5 40 python3 scripts/depth_check.py --config configs/two_cam_current.yaml \
#       --depth models/dxnn/yolo26-depth-n_768x768.dxnn --frames 60 --out out/measure/09_depth_yolo26n
#   결과: out/measure/09_depth_yolo26n.json, out/measure/09_depth_yolo26n_<카메라>.jpg


# ---------------------------------------------------------------------------
# 명령행 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="카메라 설정 yaml")
    ap.add_argument("--depth", required=True, help="깊이 .dxnn 경로. 예: models/dxnn/yolo26-depth-n_768x768.dxnn")
    ap.add_argument("--kind", choices=["auto", "depth", "inverse"], default="auto", help="모델 값의 뜻. auto면 바닥 흐름으로 추정")
    ap.add_argument("--det", help="같이 돌릴 DX-M1 검출 .dxnn(박스 아래 깊이 비교용). 없으면 비교를 건너뜀")
    ap.add_argument("--det-score", type=float, default=0.25, help="비교용 검출 점수 문턱값(COCO 80 클래스)")
    ap.add_argument("--frames", type=int, default=60, help="잴 프레임 수(카메라 묶음 기준)")
    ap.add_argument("--warmup", type=int, default=5, help="시간 통계에서 뺄 앞쪽 프레임 수")
    ap.add_argument("--duration", type=float, default=20.0, help="이 시간(초)이 지나면 멈춤(DX-M1 방열 보호)")
    ap.add_argument("--images", nargs="*", default=[], help="카메라 대신 쓸 사진(카메라 순서대로). 맥 시험용")
    ap.add_argument("--out", required=True, help="결과 경로 앞부분. 예: out/measure/09_depth_yolo26n")
    return ap


# ---------------------------------------------------------------------------
# 입력: 카메라 또는 사진
# ---------------------------------------------------------------------------
def open_inputs(cams, images):
    if images:
        imgs = [cv2.imread(p) for p in images]
        if any(i is None for i in imgs):
            raise FileNotFoundError(f"사진을 읽지 못했습니다: {images}")
        return (lambda: [i.copy() for i in imgs]), (lambda: None)
    from realityview.sources import open_source

    srcs = [open_source(c) for c in cams]

    def read():
        fr = [s.read() for s in srcs]
        return None if any(f is None for f in fr) else fr

    def close():
        for s in srcs:
            s.close()

    return read, close


# ---------------------------------------------------------------------------
# 카메라 한 대 분석: kind 추정, 지면 축척, 박스 비교, 컬러 이미지
# ---------------------------------------------------------------------------
def analyze_camera(cam, frame, depth, kind_opt, dets, out_prefix):
    guess, rows = guess_kind(depth)
    kind = guess if kind_opt == "auto" else kind_opt
    rep = {"camera": cam.name, "kind_guess": guess, "kind_used": kind, "floor_rows_median": [round(v, 4) for v in rows]}
    rep["raw_min"], rep["raw_max"], rep["raw_mean"] = (round(float(f(depth)), 4) for f in (np.min, np.max, np.mean))
    # 지면 점으로 축척 맞추기(보정 전 카메라 값이라 결과 미터 값은 의미 없음, 식이 도는지만 본다)
    raw, z = ground_samples(cam, depth)
    rep["ground_points"] = int(len(raw))
    scale = None
    try:
        scale = fit_depth_scale(raw, z, kind)
        rep["scale"] = {"kind": scale.kind, "a": scale.a, "b": scale.b, "n": scale.n}
    except ValueError as e:
        rep["scale"] = {"error": str(e)}
    rep["boxes"] = compare_boxes_with_ipm(cam, depth, dets, scale) if dets is not None else None
    # 컬러 이미지: 왼쪽 원본, 오른쪽 깊이(가까울수록 밝음)
    color = colorize_depth(depth, kind)
    if dets:
        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in d.box)
            for im in (frame, color):
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    path = f"{out_prefix}_{cam.name}.jpg"
    cv2.imwrite(path, np.hstack([frame, color]))
    rep["image"] = path
    return rep


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    from realityview.detectors import DXM1Backend

    cams, _, _ = load_config(args.config)
    timer = StageTimer(enabled=True, warmup=args.warmup)
    # 비교용 검출은 COCO 80 클래스 모두(방 안에는 사람과 차가 없다). 이름은 COCO_KEEP에 있으면 그 이름을 쓴다
    labels = {i: COCO_KEEP.get(i, f"coco{i}") for i in range(80)}
    be = DXM1Backend(args.det, depth_model=args.depth, timer=timer, score_thr=args.det_score, labels=labels)
    read, close = open_inputs(cams, args.images)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n, t0, frames, depths, dets, t_meas = 0, time.perf_counter(), None, None, None, None
    try:
        while n < args.frames and time.perf_counter() - t0 < args.duration:
            fr = read()
            if fr is None:
                break
            ts = time.perf_counter()
            depths = be.estimate_depth(fr)
            timer.end_frame((time.perf_counter() - ts) * 1000.0)
            frames, n = fr, n + 1
            if n == max(args.warmup, 1):
                t_meas = time.perf_counter()
        elapsed = time.perf_counter() - t0
        if frames is not None and args.det:
            dets = [r.detections for r in be.infer(frames)]
        cam_npu_ms = be.depth.engine.get_npu_inference_time_mean() / 1000.0 if hasattr(be.depth.engine, "get_npu_inference_time_mean") else None
    finally:
        close()
    summ = timer.summary()
    meas_frames = n - max(args.warmup, 1) if t_meas is not None else 0
    rep = {
        "model": args.depth,
        "config": args.config,
        "frames": n,
        "elapsed_s": round(elapsed, 2),
        "loop_fps_after_warmup": round(meas_frames / (t0 + elapsed - t_meas), 2) if meas_frames > 0 else None,
        "stages_ms": summ,
        "npu_time_mean_ms_per_image": None if cam_npu_ms is None else round(cam_npu_ms, 2),
        "input_hw": list(be.depth.input_hw),
        "cameras": [],
    }
    if frames is not None:
        for i, (c, f, d) in enumerate(zip(cams, frames, depths)):
            rep["cameras"].append(analyze_camera(c, f.copy(), d, args.kind, dets[i] if dets else None, args.out))
    be.close()
    with open(args.out + ".json", "w") as fp:
        json.dump(rep, fp, indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("frames", "loop_fps_after_warmup", "npu_time_mean_ms_per_image")}, ensure_ascii=False))
    for c in rep["cameras"]:
        print(c["camera"], c["kind_guess"], c["floor_rows_median"], c["scale"], c["image"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
