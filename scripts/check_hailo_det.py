#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.detcheck import draw_detections, flip_consistency  # noqa: E402
from realityview.types import COCO_ALL  # noqa: E402

# 카메라마다 한 프레임을 찍어 HailoBackend에 넣고, 박스를 원본 영상에 그려 저장한다.
# parse_hailo_nms의 좌표 순서(ymin, xmin, ymax, xmax)와 0~1 정규화가 맞는지 보는 용도.
#
# 예) 파이에서 (카메라와 NPU를 잡으므로 timeout을 꼭 붙인다)
#   timeout -s INT 90 python3 scripts/check_hailo_det.py --config configs/two_cam_current.yaml
#   결과: out/measure/01_det_check_cam0.png, 01_det_check_cam1.png, 01_frame_cam*.jpg, 01_det_check.json


# ---------------------------------------------------------------------------
# 입력 영상 얻기
# ---------------------------------------------------------------------------
def grab_frame(cam, skip: int):
    """설정의 카메라를 열어 앞쪽 skip 프레임(자동 노출 안정화)을 버리고 한 장을 돌려준다."""
    from realityview.sources import open_source

    src = open_source(cam)
    try:
        frame = None
        for _ in range(max(1, skip + 1)):
            frame = src.read()
        info = {"sensor_config": _plain(getattr(src, "sensor_config", {}))}
        if hasattr(src, "metadata"):
            md = src.metadata()
            info["ScalerCrop"] = _plain(md.get("ScalerCrop"))
            info["ExposureTime"] = md.get("ExposureTime")
            info["AnalogueGain"] = md.get("AnalogueGain")
        return frame, info
    finally:
        src.close()


def _plain(v):
    """JSON에 쓸 수 있는 값으로 바꾼다(libcamera 객체, 튜플 등)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam_current.yaml")
    ap.add_argument("--image", action="append", default=[], help="name=path 형식으로 카메라 대신 이미지 파일 사용")
    ap.add_argument("--skip", type=int, default=20, help="카메라 시작 후 버릴 프레임 수(자동 노출 안정화)")
    ap.add_argument("--score", type=float, default=0.25, help="확인용 점수 문턱(낮게 두어 무엇이든 보이게)")
    ap.add_argument("--out-dir", default="out/measure")
    ap.add_argument("--prefix", default="01")
    args = ap.parse_args()

    cams, _, cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    images = dict(v.split("=", 1) for v in args.image)

    # ---- 1. 카메라마다 한 장 얻기(카메라를 먼저 다 놓아 준 뒤에 NPU를 연다) ----
    frames, report = [], {"config": args.config, "score_thr": args.score, "cameras": []}
    for i, cam in enumerate(cams):
        if cam.name in images:
            frame, info = cv2.imread(images[cam.name]), {"image": images[cam.name]}
            if frame is None:
                raise SystemExit(f"이미지를 읽지 못했습니다: {images[cam.name]}")
        else:
            frame, info = grab_frame(cam, args.skip)
        cv2.imwrite(os.path.join(args.out_dir, f"{args.prefix}_frame_cam{i}.jpg"), frame)
        frames.append(frame)
        report["cameras"].append({"index": i, "name": cam.name, "shape": list(frame.shape), **info})

    # ---- 2. 실제 HailoBackend 경로로 검출(원본과 좌우 반전본을 함께) ----
    from realityview.detectors import HailoBackend

    npu = cfg.get("npu", {})
    # 확인용이라 80개 클래스를 모두 남긴다(방 안이면 의자, 모니터 같은 물체라도 잡히게)
    backend = HailoBackend(npu["det_hef"], score_thr=args.score, labels=COCO_ALL)
    try:
        flipped = [cv2.flip(f, 1) for f in frames]
        res = backend.infer(frames + flipped)
    finally:
        backend.close()

    # ---- 3. 박스 그리기, 좌우 반전 일치도 계산 ----
    n = len(frames)
    for i, (cam, frame) in enumerate(zip(cams, frames)):
        dets, dets_f = res[i].detections, res[n + i].detections
        ious = flip_consistency(dets, dets_f, frame.shape[1])
        title = f"cam{i} {cam.name} dets={len(dets)}"
        out_png = os.path.join(args.out_dir, f"{args.prefix}_det_check_cam{i}.png")
        cv2.imwrite(out_png, draw_detections(frame, dets, title))
        h, w = frame.shape[:2]
        inside = all(-2 <= d.box[0] < d.box[2] <= w + 2 and -2 <= d.box[1] < d.box[3] <= h + 2 for d in dets)
        report["cameras"][i].update(
            {
                "png": out_png,
                "detections": [
                    {"label": d.label, "score": round(d.score, 3), "box": [round(v, 1) for v in d.box]} for d in dets
                ],
                "flip_pairs": len(ious),
                "flip_iou_mean": round(sum(ious) / len(ious), 3) if ious else None,
                "flip_iou_min": round(min(ious), 3) if ious else None,
                "boxes_inside_image": inside,
            }
        )
    out_json = os.path.join(args.out_dir, f"{args.prefix}_det_check.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
