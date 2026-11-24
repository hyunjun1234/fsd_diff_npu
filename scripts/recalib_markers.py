#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.calib import (  # noqa: E402
    ARUCO_DICT,
    _cam_with,
    aruco_available,
    detect_aruco,
    detect_circles,
    marker_errors,
    pose_delta,
    solve_pose_markers,
    update_extrinsics_text,
)
from realityview.config import load_config, scene_scale  # noqa: E402

# 바닥 마커 4개 이상으로 카메라 외부 값(기본은 각도만)을 빠르게 다시 구한다.
# 내부 보정과 외부 보정 초기값이 설정 파일에 이미 있어야 한다. 마커 인쇄와 놓는 법은
# docs/phase1_model_rig_calibration.md의 "빠른 재보정" 절에 있다.
#
# 마커 배치 CSV(헤더 id,X,Y): 마커 번호와 마커 중심의 차량 좌표(m, 줄자로 잼). 원 마커는 번호를 정해 적는다.
#
# 예) 확인만(설정은 그대로)
#   python3 scripts/recalib_markers.py --config configs/two_cam_current.yaml --camera front \
#       --image out/calib/front_markers/front_00000.png --layout out/calib/markers_front.csv
# 예) 결과를 확인한 뒤 설정에 쓰기(원래 파일은 out/calib/backup/ 에 사본을 남김)
#   python3 scripts/recalib_markers.py ... --write
#
# 결과: 화면에 이전 값과의 차이(각도, 위치)와 지면 역투영 오차, --json 을 주면 같은 내용을 json으로,
#       영상 옆에 _markers.png(찾은 마커 번호와 새 값으로 다시 그린 자리)


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--image", required=True, help="마커가 찍힌 영상(설정과 같은 해상도)")
    ap.add_argument("--layout", required=True, help="마커 배치 CSV(id,X,Y)")
    ap.add_argument("--kind", choices=["auto", "aruco", "circle"], default="auto",
                    help="마커 종류. auto는 ArUco가 있으면 ArUco, 없으면 원")
    ap.add_argument("--dict", default=ARUCO_DICT, help="ArUco 사전 이름")
    ap.add_argument("--gate-px", type=float, default=40.0, help="원 마커: 예상 자리와 이 거리(px) 안의 원만 짝짓는다")
    ap.add_argument("--solve-position", action="store_true",
                    help="위치까지 푼다(기본은 위치 고정, 각도만). 마커 6개 이상 권장")
    ap.add_argument("--max-ground-err", type=float, default=0.05,
                    help="새 값의 지면 역투영 오차 평균이 이 값(실차 m, 설정의 축척을 곱해 씀)보다 크면 --write를 거절")
    ap.add_argument("--max-change-deg", type=float, default=10.0,
                    help="각도가 이 값(도)보다 크게 바뀌면 --write를 거절(마커 번호나 배치 실수일 가능성이 큼)")
    ap.add_argument("--write", action="store_true", help="결과를 설정 파일에 쓴다(사람이 결과를 본 뒤에만 준다)")
    ap.add_argument("--backup-dir", default="out/calib/backup", help="--write 전에 원래 설정 사본을 둘 곳")
    ap.add_argument("--json", help="결과 json 저장 경로")
    return ap


# ---------------------------------------------------------------------------
# 입력 읽기
# ---------------------------------------------------------------------------
def read_layout(path: str) -> dict[int, tuple[float, float]]:
    with open(path, newline="") as f:
        return {int(r["id"]): (float(r["X"]), float(r["Y"])) for r in csv.DictReader(f)}


def draw_result(img: np.ndarray, cam_old, cam_new, layout, found: dict) -> np.ndarray:
    """찾은 마커(초록 번호), 이전 값으로 예상한 자리(빨강), 새 값으로 예상한 자리(파랑)를 그린다."""
    vis = img.copy()
    ids = list(layout)
    P = np.array([[*layout[i], 0.0] for i in ids])
    for cam, color in ((cam_old, (0, 0, 255)), (cam_new, (255, 128, 0))):
        uv, ok = cam.project(P)
        for (u, v), good in zip(uv, ok):
            if good and 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.drawMarker(vis, (int(u), int(v)), color, cv2.MARKER_CROSS, 14, 2)
    uv, _ = cam_new.project(np.array([[*layout[i], 0.0] for i in found]))
    for i, (u, v) in zip(found, uv):
        cv2.putText(vis, str(i), (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cams, _, cfg = load_config(args.config)
    cam = next((c for c in cams if c.name == args.camera), None)
    if cam is None:
        raise SystemExit(f"설정에 없는 카메라: {args.camera}")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"영상을 읽지 못했습니다: {args.image}")
    if img.shape[1] != cam.width or img.shape[0] != cam.height:
        raise SystemExit(f"영상 크기 {img.shape[1]}x{img.shape[0]}가 설정 {cam.width}x{cam.height}와 다릅니다")
    layout = read_layout(args.layout)
    kind = args.kind if args.kind != "auto" else ("aruco" if aruco_available() else "circle")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 1) 마커 찾기 ----
    t0 = time.perf_counter()
    det = detect_aruco(gray, cam, args.dict) if kind == "aruco" else detect_circles(gray, cam, layout, args.gate_px)
    found = sorted(i for i in det if i in layout)
    extra = sorted(i for i in det if i not in layout)
    print(f"마커 종류 {kind}, 찾은 마커 {found}" + (f", 배치에 없는 번호 {extra}(무시)" if extra else ""))
    if len(found) < 4:
        print("마커가 4개 이상 보여야 합니다. 마커를 카메라 쪽으로 당기거나 더 크게 인쇄해 주세요.")
        return 2
    norm = np.array([det[i] for i in found])
    XY = np.array([layout[i] for i in found])

    # ---- 2) 풀기와 비교 ----
    before = marker_errors(cam, norm, XY)
    pose = solve_pose_markers(cam, norm, XY, solve_position=args.solve_position)
    new = _cam_with(cam, pose["yaw"], pose["pitch"], pose["roll"], pose["position"])
    after = marker_errors(new, norm, XY)
    d = pose_delta(cam, pose)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"계산 시간 {ms:.0f}ms ({'각도와 위치' if args.solve_position else '각도만, 위치 고정'})")
    print(f"각도 차이(새 값 - 지금 값): yaw {d['yaw']:+.2f}도, pitch {d['pitch']:+.2f}도, roll {d['roll']:+.2f}도")
    print(f"위치 차이: [{', '.join(f'{v:+.3f}' for v in d['position'])}] m (크기 {d['position_m']:.3f} m)")
    for name, e in (("지금 값", before), ("새 값", after)):
        print(f"{name}: 재투영 평균 {e['reproj_px_mean']:.2f}px, 지면 역투영 평균 {e['ground_m_mean'] * 100:.1f}cm, "
              f"최대 {e['ground_m_max'] * 100:.1f}cm")
    vis = draw_result(img, cam, new, layout, found)
    out_png = os.path.splitext(args.image)[0] + "_markers.png"
    cv2.imwrite(out_png, vis)
    print(f"확인 그림: {out_png} (빨강 = 지금 값으로 예상한 자리, 파랑 = 새 값, 초록 글자 = 찾은 번호)")
    rep = {"camera": cam.name, "kind": kind, "found": found, "solve_position": args.solve_position,
           "delta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()},
           "new": {"position": np.asarray(pose["position"]).tolist(), "yaw": pose["yaw"], "pitch": pose["pitch"],
                   "roll": pose["roll"]},
           "before": before, "after": after, "ms": ms, "written": False}

    # ---- 3) 설정에 쓰기(--write 를 준 때만, 안전 조건을 모두 지날 때만) ----
    code = 0
    max_err = args.max_ground_err * scene_scale(cfg)  # 모형 설정이면 축척만큼 줄인 기준(축척 0.1이면 5mm)
    rep["max_ground_err_m"] = max_err
    if args.write:
        big = max(abs(d["yaw"]), abs(d["pitch"]), abs(d["roll"]))
        if not after["ground_m_mean"] <= max_err:
            print(f"쓰지 않음: 새 값의 지면 역투영 평균 {after['ground_m_mean']:.4f}m > 기준 {max_err:.4f}m")
            code = 1
        elif big > args.max_change_deg:
            print(f"쓰지 않음: 각도가 {big:.1f}도 바뀜(기준 {args.max_change_deg}도). 마커 번호와 배치 CSV를 확인해 주세요.")
            code = 1
        else:
            os.makedirs(args.backup_dir, exist_ok=True)
            bak = os.path.join(args.backup_dir, f"{os.path.basename(args.config)}.{time.strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(args.config, bak)
            with open(args.config, encoding="utf-8") as f:
                text = f.read()
            text = update_extrinsics_text(text, cam.name, pose, args.solve_position, scene_scale(cfg))
            with open(args.config, "w", encoding="utf-8") as f:
                f.write(text)
            rep["written"] = True
            print(f"설정에 씀: {args.config} (원래 파일 사본: {bak})")
    else:
        print("설정은 바꾸지 않았습니다. 결과가 맞으면 같은 명령에 --write 를 붙여 다시 실행합니다.")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    return code


if __name__ == "__main__":
    sys.exit(main())
