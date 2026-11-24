#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.sources import open_source  # noqa: E402

# 설정 파일의 카메라로 프레임을 찍어 파일로 남긴다. 두 가지에 쓴다.
#   1) 검출 비교용 녹화: 카메라 2대를 같은 순간에 100프레임 (scripts/compare_det.py 입력)
#   2) 보정용 사진: 카메라 1대를 몇 초 간격으로 (체커보드를 옮길 시간을 둔다)
# 설정 파일의 해상도와 센서 모드 그대로 찍으므로 보정값과 데모 영상의 화각이 같다.
#
# 예) 파이 저장소 폴더, .venv 켠 뒤 (카메라를 잡으므로 timeout을 붙인다)
#   timeout -s INT -k 5 120 python3 scripts/record_frames.py --config configs/two_cam_current.yaml \
#       --frames 100 --out out/rec/room_100
#   timeout -s INT -k 5 300 python3 scripts/record_frames.py --config configs/two_cam_model.yaml \
#       --camera front --frames 25 --interval 3 --out out/calib/front
#
# 결과: --out 폴더에 {카메라}_{번호 5자리}.png, timestamps.csv(프레임, 카메라, 시각, 파일), meta.json


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="카메라 설정 yaml")
    ap.add_argument("--frames", type=int, default=100, help="남길 프레임 수(카메라마다)")
    ap.add_argument("--out", required=True, help="저장 폴더. 예: out/rec/room_100")
    ap.add_argument("--camera", action="append", default=[], help="찍을 카메라 이름(여러 번 가능). 없으면 모두")
    ap.add_argument("--interval", type=float, default=0.0, help="저장 사이 간격(초). 보정 사진은 2~3초")
    ap.add_argument("--warmup", type=int, default=15, help="자동 노출이 자리 잡을 때까지 버릴 프레임 수")
    ap.add_argument("--ext", choices=["png", "jpg"], default="png", help="png는 손실 없음, jpg는 작음")
    return ap


# ---------------------------------------------------------------------------
# 파일 이름 규칙 (compare_det.py와 같이 쓴다)
# ---------------------------------------------------------------------------
def frame_name(camera: str, idx: int, ext: str = "png") -> str:
    return f"{camera}_{idx:05d}.{ext}"


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    cams, _, _ = load_config(args.config)
    if args.camera:
        names = set(args.camera)
        unknown = names - {c.name for c in cams}
        if unknown:
            print(f"설정에 없는 카메라: {sorted(unknown)}", file=sys.stderr)
            return 2
        cams = [c for c in cams if c.name in names]
    os.makedirs(args.out, exist_ok=True)

    sources = []
    try:
        for c in cams:
            sources.append(open_source(c))
        # ---- 버릴 프레임: 노출과 화이트밸런스가 자리 잡게 ----
        for _ in range(args.warmup):
            for s in sources:
                s.read()

        rows = []
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        t_start = time.monotonic()
        for i in range(args.frames):
            if args.interval > 0 and i > 0:
                # 체커보드를 옮길 시간. 그동안에도 읽어 버려서 오래된 프레임이 쌓이지 않게 한다
                t_next = t_start + i * args.interval
                while time.monotonic() < t_next:
                    for s in sources:
                        s.read()
            # ---- 카메라마다 한 장씩, 읽은 시각을 함께 남긴다 ----
            for c, s in zip(cams, sources):
                f = s.read()
                t_mono, t_wall = time.monotonic(), time.time()
                if f is None:
                    print(f"{c.name}: 프레임을 읽지 못했습니다(번호 {i})", file=sys.stderr)
                    continue
                name = frame_name(c.name, i, args.ext)
                cv2.imwrite(os.path.join(args.out, name), f)
                rows.append({"frame": i, "camera": c.name, "t_mono": f"{t_mono:.6f}", "t_wall": f"{t_wall:.6f}", "file": name})
            if args.interval > 0:
                print(f"{i + 1}/{args.frames} 저장", flush=True)
        elapsed = time.monotonic() - t_start
    finally:
        for s in sources:
            s.close()

    # ---- 타임스탬프와 촬영 조건 ----
    with open(os.path.join(args.out, "timestamps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "camera", "t_mono", "t_wall", "file"])
        w.writeheader()
        w.writerows(rows)
    meta = {
        "config": args.config,
        "cameras": [{"name": c.name, "width": c.width, "height": c.height, "source": c.source} for c in cams],
        "frames": args.frames,
        "saved": len(rows),
        "interval_s": args.interval,
        "elapsed_s": round(elapsed, 3),
        "capture_fps": round(args.frames / elapsed, 2) if elapsed > 0 else None,
        "started": started,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
