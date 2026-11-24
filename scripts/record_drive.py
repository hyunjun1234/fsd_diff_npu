#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.drive_rec import DriveRecorder, list_drives, write_json_atomic  # noqa: E402
from realityview.power import add_power_args, finish, make_watcher  # noqa: E402
from realityview.thermal import CRITICAL, EXIT_THERMAL, add_thermal_args, config_from_args  # noqa: E402

# 주행 녹화: NPU 추론 없이 설정 파일의 카메라 전부를 세그먼트 단위로 저장한다.
# 폴더 구조와 파일 형식은 realityview/drive_rec.py 맨 위 설명과 docs/drive_recording.md 를 본다.
#
# 예) 파이 저장소 폴더, .venv 켠 뒤 (카메라를 잡으므로 timeout을 붙인다)
#   # 한 시간 녹화(60초 세그먼트, 남은 디스크 5GB 아래면 멈춤)
#   timeout -s TERM -k 15 3700 python3 scripts/record_drive.py --config configs/two_cam_current.yaml \
#       --duration 3600
#   # 회차 목록과 용량
#   python3 scripts/record_drive.py --list
#
# 끝내기: Ctrl+C(SIGINT)나 SIGTERM(ACC가 꺼질 때 전원 장치나 systemd가 보내는 신호)을 받으면
# 지금 세그먼트를 닫고 drive.json을 마무리한 뒤 끝낸다. 결과는 --out 아래 회차 폴더에 생긴다.


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="카메라 설정 yaml. 카메라 수와 모드를 여기서 읽는다")
    ap.add_argument("--out", default="out/drives", help="회차 폴더를 만들 곳. 기본 out/drives")
    ap.add_argument("--name", default=None, help="회차 폴더 이름. 기본은 시작 시각 YYYYmmdd_HHMMSS")
    ap.add_argument("--segment", type=float, default=60.0, help="세그먼트 길이(초). 기본 60")
    ap.add_argument("--duration", type=float, default=0.0, help="이 시간(초)이 지나면 끝냄. 0이면 신호를 받을 때까지")
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="남은 디스크가 이 값(GB)보다 적으면 멈춤. 기본 5")
    ap.add_argument("--status-interval", type=float, default=60.0, help="프레임 누락 수와 온도를 남기는 간격(초). 기본 60")
    ap.add_argument("--quality", type=int, default=85, help="JPEG 품질(1~100). 기본 85")
    ap.add_argument("--cam-fps", type=float, default=None, help="Picamera2 프레임레이트를 이 값으로 덮어씀. 기본은 설정값")
    ap.add_argument("--list", action="store_true", help="녹화하지 않고 --out 아래 회차 목록과 용량만 보여 줌")
    # 녹화는 NPU를 쓰지 않으므로 위험 단계 기본 조치는 continue(기록만 남기고 녹화 유지)
    add_thermal_args(ap, critical_default="continue")
    add_power_args(ap)
    return ap


def print_list(root: str) -> None:
    rows = list_drives(root)
    if not rows:
        print(f"{root} 에 회차가 없습니다")
        return
    total = 0
    print(f"{'회차':<18}{'시작':<21}{'길이(초)':>9}{'세그':>6}{'프레임':>9}{'누락':>7}{'용량(GB)':>10}  끝난 이유")
    for r in rows:
        total += r["bytes"]
        el = f"{r['elapsed_s']:.0f}" if r["elapsed_s"] is not None else "-"
        segs = f"{r['segments_clean']}/{r['segments']}"
        print(
            f"{r['drive']:<18}{str(r['started'] or '-'):<21}{el:>9}{segs:>6}{r['frames']:>9}{r['gaps']:>7}"
            f"{r['bytes'] / 1e9:>10.2f}  {r['stop_reason'] or '-'}"
        )
    print(f"합계 {len(rows)}회차 {total / 1e9:.2f}GB (세그 = 깨끗하게 닫힌 수/전체 수)")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.list:
        print_list(args.out)
        return 0
    if not args.config:
        ap.error("--config 가 필요합니다(--list 일 때만 없어도 됨)")

    from realityview.sources import open_source

    cams, _, _ = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    sources = []
    try:
        for c in cams:
            ov = {"fps": float(args.cam_fps)} if (args.cam_fps and c.source.get("type") == "picamera2") else None
            sources.append(open_source(c, ov))
        if args.cam_fps:
            for c in cams:
                c.source["fps"] = float(args.cam_fps)  # 누락 계산의 한 주기도 같은 값으로
        rec = DriveRecorder(
            args.out, cams, sources, config_path=args.config, segment_s=args.segment, duration_s=args.duration,
            min_free_gb=args.min_free_gb, status_interval=args.status_interval, quality=args.quality, name=args.name,
        )

        # ---- 신호: 이벤트만 세우고, 실제 정리는 아래 close()에서 한다 ----
        def on_signal(signum, _frame):
            rec.request_stop(f"signal:{signal.Signals(signum).name}")

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        # ---- 안전 종료(옵션): 신호 파일이나 ACC GPIO 가 꺼짐을 알리면 녹화를 멈춘다 ----
        power_loss = {"reason": None}

        def on_power_loss(reason):
            power_loss["reason"] = reason
            rec.request_stop(f"power_loss:{reason}")

        watcher = make_watcher(args, on_power_loss)

        rec.start()

        # ---- 과열 감시(옵션): 녹화는 NPU를 쓰지 않으므로 기록이 주 목적. --thermal-critical exit 이면 멈춘다 ----
        # 회차 폴더가 rec.start()에서 생기므로 그 뒤에 만든다. csv 기본 위치는 회차 폴더의 thermal.csv
        monitor = None
        thermal_exit = {"hit": False}
        if args.thermal:
            from realityview.thermal import ThermalMonitor, default_readers

            def on_change(st):
                if st.level >= CRITICAL and args.thermal_critical == "exit":
                    thermal_exit["hit"] = True
                    rec.request_stop(f"thermal_critical:{st.cause}")

            monitor = ThermalMonitor(
                config_from_args(args), default_readers(hailo_standalone=True),
                csv_path=args.thermal_csv or os.path.join(rec.drive_dir, "thermal.csv"), on_change=on_change,
            ).start()
        if watcher is not None:
            watcher.start()
        print(f"녹화 시작: {rec.drive_dir} 카메라 {[c.name for c in cams]} 세그먼트 {args.segment:g}초", flush=True)
        try:
            rec.run_until_stopped()
        finally:
            meta = rec.close()
            if watcher is not None:
                watcher.close()
            if monitor is not None:
                meta["thermal"] = monitor.close()
            if watcher is not None or monitor is not None:
                write_json_atomic(os.path.join(rec.drive_dir, "drive.json"), meta)
            if watcher is not None or args.power_loss_cmd or str(rec.stop_reason).startswith("signal:"):
                # 녹화 파일은 close()에서 이미 fsync 했다. 나머지(로그, 메타데이터)까지 디스크로 내린다.
                # SIGTERM(systemd 정지, 전원 장치)으로 끝날 때도 한다. 전원 꺼짐 뒤 명령은 GPIO나 신호 파일일 때만 부른다
                meta["finish"] = finish(power_loss["reason"], args.power_loss_cmd or None)
        summary = {k: meta.get(k) for k in ("drive", "elapsed_s", "stop_reason", "segments_closed", "totals")}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        errors = [t for t in meta.get("totals") or [] if t.get("error")]
        if thermal_exit["hit"]:
            return EXIT_THERMAL
        return 1 if errors else 0
    finally:
        for s in sources:
            s.close()


if __name__ == "__main__":
    sys.exit(main())
