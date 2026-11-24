#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.sim import SimWorld, camera_timelines, per_camera  # noqa: E402

# 합성 4대 장면을 카메라별 영상 파일로 만든다(카메라 4대 입력과 시간 맞춤을 하드웨어 없이 시험하는 용도).
#
# 카메라 k는 n/fps + skew_ms[k] (+ 흔들림) 시각에 찍고, drop[k] 확률로 프레임을 빠뜨린다.
# 만든 폴더는 run_demo.py --replay 로 바로 넣는다(FrameSyncer가 시각으로 묶는다).
#
# 만드는 파일(--out 폴더)
#   cam{k}.mp4 : 카메라 k 영상(설정의 카메라 순서, 빠진 프레임은 없음)
#   cam{k}.csv : frame(파일 안 번호), sim_index(시뮬레이션 프레임 번호), sensor_ns(찍은 시각 ns)
#   gt.json    : boxes[카메라][파일 프레임] = [[라벨, x1, y1, x2, y2, 물체 번호, 다시 등장 횟수], ...]
#                truth[시뮬레이션 프레임] = [[물체 번호, 다시 등장 횟수, 라벨, x, y], ...] (기준 시각 n/fps)
#
# 예)
#   /usr/bin/python3 scripts/make_sim4.py --out out/sim4
#   /usr/bin/python3 scripts/run_demo.py --config configs/four_cam.yaml --backend gt --replay out/sim4 \
#       --headless --view both --tracker robust --frames 200 --save out/video/sim4_replay.mp4


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/four_cam.yaml")
    ap.add_argument("--out", default="out/sim4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=225, help="시뮬레이션 프레임 수(기준 카메라 프레임 수)")
    ap.add_argument("--skew-ms", default="0,0,20,50", help="카메라별 시각 어긋남(ms). 기본 좌 20, 우 50")
    ap.add_argument("--drop", default="0,0,0,0.05", help="카메라별 프레임 누락 확률. 기본 우측만 5%%")
    ap.add_argument("--jitter-ms", default="0,0,1,1", help="카메라별 시각 흔들림 표준편차(ms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-ns", type=int, default=1_000_000_000_000, help="0초에 해당하는 sensor_ns 값")
    return ap


# ---------------------------------------------------------------------------
# 만들기
# ---------------------------------------------------------------------------
def make(args) -> dict:
    cams, _, _ = load_config(args.config)
    n = len(cams)
    skew, drop, jit = per_camera(args.skew_ms, n), per_camera(args.drop, n), per_camera(args.jitter_ms, n)
    tls = camera_timelines(n, args.frames, args.fps, skew, drop, jit, seed=args.seed)
    world = SimWorld(cams)
    os.makedirs(args.out, exist_ok=True)
    boxes = []
    counts = []
    for k, cam in enumerate(cams):
        vw = cv2.VideoWriter(os.path.join(args.out, f"cam{k}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (cam.width, cam.height))
        if not vw.isOpened():
            raise IOError("mp4 쓰기를 열지 못했습니다(OpenCV mp4v 코덱 확인)")
        cam_boxes = []
        with open(os.path.join(args.out, f"cam{k}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "sim_index", "sensor_ns"])
            for j, (sim_n, t) in enumerate(tls[k]):
                world.set_time(t)
                img, dets, ids, _ = world.render_camera(k)
                vw.write(img)
                w.writerow([j, sim_n, args.base_ns + int(round(t * 1e9))])
                cam_boxes.append([[d.label, *[round(v, 2) for v in d.box], i, world.wraps[i]] for d, i in zip(dets, ids)])
        vw.release()
        boxes.append(cam_boxes)
        counts.append(len(tls[k]))
    # 기준 시각(n/fps)의 물체 위치: 화면 확인과 추적 평가의 정답
    truth = []
    for s in range(args.frames):
        world.set_time(s / args.fps)
        truth.append([[i, world.wraps[i], o.label, round(o.x, 3), round(o.y, 3)] for i, o in enumerate(world.objects)])
    info = {
        "config": args.config,
        "fps": args.fps,
        "frames": args.frames,
        "cameras": [c.name for c in cams],
        "skew_ms": skew,
        "drop": drop,
        "jitter_ms": jit,
        "seed": args.seed,
        "base_ns": args.base_ns,
        "frames_per_camera": counts,
    }
    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as f:
        json.dump({**info, "boxes": boxes, "truth": truth}, f)
    return info


def main() -> None:
    info = make(build_parser().parse_args())
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
