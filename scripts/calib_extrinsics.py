#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.calib import (  # noqa: E402
    apply_pose,
    extrinsics_yaml,
    find_corners,
    floor_board_xy,
    ground_errors,
    solve_extrinsics,
)
from realityview.config import load_config  # noqa: E402

# 바닥 기준점으로 카메라 외부 파라미터(위치, yaw, pitch, roll)를 구하거나, 이미 정한 값을 검증한다.
# 기준점은 두 가지 방법으로 줄 수 있다.
#   1) 바닥에 깐 체커보드 사진 (권장): 코너를 자동으로 찾고, 첫 코너의 위치와 보드 방향만 줄자로 잰다
#   2) CSV 파일: 헤더 u,v,X,Y (u, v는 픽셀, X, Y는 차량 좌표 m)
#
# 예) 모형: 보드(내부 코너 9x6, 칸 20mm) 첫 코너가 모형 앞 X=0.40m, Y=0.08m, 보드 가로가 -Y 방향(yaw -90도)
#   python3 scripts/calib_extrinsics.py --config configs/two_cam_model.yaml --camera front \
#       --board-image out/calib/front_floor/front_00000.png --cols 9 --rows 6 --square 0.020 \
#       --origin 0.40 0.08 --board-yaw -90 --out out/calib/front_extrinsics.yaml
#   # 보드를 다른 자리에 옮겨 찍고 검증만(설정 파일 값 그대로 오차만 잼)
#   python3 scripts/calib_extrinsics.py --config configs/two_cam_model.yaml --camera front --verify \
#       --board-image out/calib/front_check/front_00000.png --cols 9 --rows 6 --square 0.020 \
#       --origin 0.60 -0.10 --board-yaw -90
#
# 결과: 화면과 --out 파일에 extrinsics 항목, --board-image 옆에 _corners.png(0번 코너 표시)


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--points", help="CSV(u,v,X,Y) 기준점 파일")
    ap.add_argument("--board-image", help="바닥 체커보드 사진(설정과 같은 해상도)")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=0.020, help="칸 한 변(m)")
    ap.add_argument("--origin", type=float, nargs=2, metavar=("X", "Y"), help="0번 코너의 차량 좌표(m)")
    ap.add_argument("--board-yaw", type=float, default=0.0, help="보드 가로 방향이 차량 X축과 이루는 각(도, 좌회전 +)")
    ap.add_argument("--verify", action="store_true", help="풀지 않고 설정 파일 값으로 오차만 잰다")
    ap.add_argument("--max-ground-err", type=float, help="지면 역투영 오차 평균의 합격 기준(m). 주면 합격 여부를 찍는다")
    ap.add_argument("--out", help="extrinsics yaml 조각 저장 경로")
    return ap


# ---------------------------------------------------------------------------
# 기준점 모으기
# ---------------------------------------------------------------------------
def points_from_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    XY = np.array([[float(r["X"]), float(r["Y"])] for r in rows])
    return uv, XY


def points_from_board(args) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(args.board_image)
    if img is None:
        raise SystemExit(f"사진을 읽지 못했습니다: {args.board_image}")
    if args.origin is None:
        raise SystemExit("--board-image에는 --origin X Y(0번 코너의 차량 좌표)가 필요합니다")
    uv = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), args.cols, args.rows)
    if uv is None:
        raise SystemExit("보드 코너를 찾지 못했습니다. 보드 전체가 보이게, 반사 없이 다시 찍어 주세요.")
    XY = floor_board_xy(args.cols, args.rows, args.square, args.origin, args.board_yaw)
    # 0번 코너(빨강)와 1번 코너(초록)를 그려 둔다. 0번이 줄자로 잰 코너가 맞는지 눈으로 확인한다
    vis = img.copy()
    cv2.drawChessboardCorners(vis, (args.cols, args.rows), uv.reshape(-1, 1, 2).astype(np.float32), True)
    cv2.circle(vis, tuple(int(v) for v in uv[0]), 8, (0, 0, 255), 2)
    cv2.circle(vis, tuple(int(v) for v in uv[1]), 8, (0, 255, 0), 2)
    out = os.path.splitext(args.board_image)[0] + "_corners.png"
    cv2.imwrite(out, vis)
    print(f"코너 그림: {out} (빨강 = 0번 코너, 초록 = 1번 코너)")
    return uv, XY


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    cams, _, _ = load_config(args.config)
    cam = next((c for c in cams if c.name == args.camera), None)
    if cam is None:
        raise SystemExit(f"설정에 없는 카메라: {args.camera}")
    if args.board_image:
        uv, XY = points_from_board(args)
    elif args.points:
        uv, XY = points_from_csv(args.points)
    else:
        raise SystemExit("--board-image 또는 --points 중 하나가 필요합니다")

    if not args.verify:
        apply_pose(cam, solve_extrinsics(cam, uv, XY))
    e = ground_errors(cam, uv, XY)
    print(f"기준점 {e['n']}개 ({'검증만' if args.verify else '이 점으로 풀었음'})")
    print(f"재투영 오차: 평균 {e['reproj_px_mean']:.2f}px, 최대 {e['reproj_px_max']:.2f}px")
    print(f"지면 역투영 오차: 평균 {e['ground_m_mean'] * 100:.2f}cm, 최대 {e['ground_m_max'] * 100:.2f}cm"
          + (f", 지면과 만나지 않는 점 {e['invalid']}개" if e["invalid"] else ""))
    ok = True
    if args.max_ground_err is not None:
        ok = e["ground_m_mean"] <= args.max_ground_err
        print(f"합격 기준 평균 {args.max_ground_err * 100:.2f}cm 이하: {'합격' if ok else '불합격'}")
    text = extrinsics_yaml(cam)
    print(text)
    if args.out and not args.verify:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"저장: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
