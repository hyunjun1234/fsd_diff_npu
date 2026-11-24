#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.bev import project_detections  # noqa: E402
from realityview.config import load_config  # noqa: E402
from realityview.types import COCO_ALL  # noqa: E402

# 사진 한 장에서 물체를 검출해 바닥 위치(차량 좌표 X, Y)를 찍고, 줄자로 잰 위치와의 오차를 잰다.
# 모형 시험에서 장난감 차나 작은 물체를 놓고 거리 오차를 볼 때 쓴다. 위치 계산은 데모와 같은
# project_detections(설정의 scene.scale 적용)를 쓴다.
#
# 예) 파이 저장소 폴더, .venv 켠 뒤 (Hailo를 잡으므로 timeout을 붙인다)
#   timeout -s INT -k 5 60 python3 scripts/locate_objects.py --config configs/two_cam_model.yaml \
#       --camera front --image out/calib/toy_front/front_00000.png --all-classes --truth 0.60 0.05
#
# 결과: 화면과 --out JSON(기본 사진 옆 _locate.json), 사진 옆 _locate.png(상자와 X, Y 표시)


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", required=True, help="사진을 찍은 카메라 이름(front, rear)")
    ap.add_argument("--image", required=True, help="그 카메라로 찍은 사진(설정과 같은 해상도)")
    ap.add_argument("--backend", choices=["hailo", "dxm1"], default="hailo")
    ap.add_argument("--all-classes", action="store_true", help="COCO 80 클래스 모두(장난감은 car가 아닐 수 있음)")
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--truth", type=float, nargs=2, metavar=("X", "Y"), help="줄자로 잰 물체 중심 위치(m)")
    ap.add_argument("--out", help="결과 JSON 경로")
    return ap


def make_backend(kind: str, npu: dict, score_thr: float, labels):
    if kind == "hailo":
        from realityview.detectors import HailoBackend

        return HailoBackend(npu["det_hef"], None, score_thr=score_thr, labels=labels)
    from realityview.detectors import DXM1Backend

    return DXM1Backend(npu["dx_det_model"], None, score_thr=score_thr, labels=labels)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    cams, _, cfg = load_config(args.config)
    idx = next((i for i, c in enumerate(cams) if c.name == args.camera), None)
    if idx is None:
        raise SystemExit(f"설정에 없는 카메라: {args.camera}")
    cam = cams[idx]
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"사진을 읽지 못했습니다: {args.image}")
    if img.shape[:2] != (cam.height, cam.width):
        raise SystemExit(f"사진 크기 {img.shape[1]}x{img.shape[0]}가 설정 {cam.width}x{cam.height}와 다릅니다")
    scale = float((cfg.get("scene") or {}).get("scale", 1.0))

    # ---- 검출 ----
    backend = make_backend(args.backend, cfg.get("npu", {}), args.score_thr, COCO_ALL if args.all_classes else None)
    try:
        dets = backend.infer([img])[0].detections
    finally:
        backend.close()

    # ---- 바닥 위치: 데모와 같은 계산(접지점 투영, 차량류는 축척 맞춘 길이 절반만큼 안쪽으로) ----
    objs = project_detections(idx, cam, dets, max_range=40.0 * scale, scale=scale)
    rows = []
    vis = img.copy()
    for o in objs:
        d = next(d for d in dets if d.label == o.label and d.score == o.score)
        row = {"label": o.label, "score": round(o.score, 3), "X": round(float(o.xy[0]), 4),
               "Y": round(float(o.xy[1]), 4), "box": [round(v, 1) for v in d.box]}
        if args.truth:
            row["error_m"] = round(float(np.hypot(o.xy[0] - args.truth[0], o.xy[1] - args.truth[1])), 4)
        rows.append(row)
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis, f"{o.label} X={o.xy[0]:.3f} Y={o.xy[1]:.3f}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    print(f"검출 {len(dets)}개, 바닥에 놓인 것 {len(rows)}개 (scene.scale {scale})")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if args.truth and rows:
        best = min(rows, key=lambda r: r["error_m"])
        print(f"줄자 위치에 가장 가까운 검출: {best['label']} 오차 {best['error_m'] * 100:.1f}cm")

    base = os.path.splitext(args.image)[0]
    cv2.imwrite(base + "_locate.png", vis)
    out = args.out or base + "_locate.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"image": args.image, "camera": args.camera, "scale": scale, "truth": args.truth, "objects": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"저장: {out}, {base}_locate.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
