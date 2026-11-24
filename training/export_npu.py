#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    REPO_ROOT,
    RoundPaths,
    load_config,
    read_csv,
    resolve_path,
    update_meta,
    write_json,
)

# 회차 best.pt 를 ONNX 로 내보내고 Hailo-8(HEF), DX-M1(dxnn)로 바꾸는 명령을 만든다.
# 기본은 --dry-run 이라 보정 이미지 폴더와 DX-COM 설정 json 만 만들고 명령은 화면과 파일에 적기만 한다.
# 실제 실행은 학습 서버(x86_64, Hailo DFC 와 DX-COM 이 설치된 곳)에서 --run 으로 한다.
#
# 예) python3 -m training.export_npu --config training/configs/round.yaml          # 명령만 출력
#     python3 -m training.export_npu --config training/configs/round.yaml --run    # 서버에서 실행
#
# 결과: out/training/datasets/<회차>/calib/*.jpg (보정 이미지, 학습 프레임에서 장면별로 고름)
#       out/training/datasets/<회차>/export/dxcom_<모델>_<회차>.json, commands.sh
#       models/yolov8s_<회차>.onnx, .hef, .dxnn, .sha256 (--run 일 때)
#
# 이름 규칙: models/<model_stem>_<회차>.<확장자> (예: yolov8s_r001.hef). 파이는 설정에서 회차를 골라 쓴다.


# ---------------------------------------------------------------------------
# 보정 이미지: 이번 회차 학습 프레임에서 장면별로 고르게 (시험 세트는 이미 빠진 목록)
# ---------------------------------------------------------------------------
def prepare_calib(cfg: dict, n: int) -> tuple[str, int]:
    from training.select_frames import balanced_pick, load_lock
    from training.train_det import filter_test_leak, known_frame_times

    paths = RoundPaths(cfg)
    own_txt = os.path.join(paths.rd, "own_train.txt")
    if not os.path.exists(own_txt):
        raise FileNotFoundError(f"{own_txt} 가 없습니다. train_det 를 먼저 돌리세요.")
    own = [x.strip() for x in open(own_txt, encoding="utf-8") if x.strip()]
    own, _ = filter_test_leak(own, load_lock(paths), known_frame_times(cfg))
    own_ids = {os.path.splitext(os.path.basename(p))[0] for p in own}
    rows = [r for r in read_csv(paths.frames_csv) if r["frame_id"] in own_ids]
    picked = balanced_pick(rows, n, key="scene")
    out = os.path.join(paths.rd, "calib")
    if os.path.isdir(out):
        shutil.rmtree(out)  # out/training 아래 회차 폴더 안의 보정 사본만 지운다
    os.makedirs(out)
    for r in picked:
        shutil.copyfile(os.path.join(paths.images, r["frame_id"] + ".jpg"), os.path.join(out, r["frame_id"] + ".jpg"))
    return out, len(picked)


# ---------------------------------------------------------------------------
# DX-COM 설정 json
# ---------------------------------------------------------------------------
def dxcom_config(cfg: dict, calib_dir: str) -> dict:
    """DX-COM 입력 설정. DEEPX 예제(dx-all-suite 문서의 YOLO 예제)를 따랐으며 키 이름은 설치한 DX-COM 판에서 확인 필요.

    전처리는 Ultralytics 와 같게 맞춘다: 640 레터박스(회색 114), BGR 을 RGB 로, 0~1 로 나누기, HWC 를 CHW 로.
    PPU(장치 후처리)는 기본으로 넣지 않는다. 그러면 출력이 (1, 84, 8400) 하나가 되고, 파이의
    DXM1 검출기는 engine.is_ppu() 가 거짓일 때 이 원시 출력을 decode_yolo_raw 로 푼다(파이에서 확인 필요).
    """
    e = cfg["export"]
    sz = int(e["imgsz"])
    conf = {
        "inputs": {"images": [1, 3, sz, sz]},
        "calibration_num": int(e["dx_calib_n"]),
        "calibration_method": "ema",
        "default_loader": {
            "dataset_path": calib_dir,
            "file_extensions": ["jpeg", "jpg", "png", "JPEG"],
            "preprocessings": [
                {"resize": {"mode": "pad", "size": sz, "pad_location": "edge", "pad_value": [114, 114, 114]}},
                {"convertColor": {"form": "BGR2RGB"}},
                {"div": {"x": 255}},
                {"transpose": {"axis": [2, 0, 1]}},
                {"expandDim": {"axis": 0}},
            ],
        },
    }
    if e.get("dx_ppu"):
        # PPU 항목 형식은 DEEPX 예제에서 복사해 export.dx_ppu_json 파일로 준다(확인 필요)
        import json

        p = resolve_path(e.get("dx_ppu_json"))
        if not p or not os.path.exists(p):
            raise FileNotFoundError("dx_ppu: true 이면 export.dx_ppu_json 에 PPU 항목 json 파일을 주세요.")
        with open(p, encoding="utf-8") as f:
            conf["ppu"] = json.load(f)
    return conf


# ---------------------------------------------------------------------------
# 명령 만들기
# ---------------------------------------------------------------------------
def rel(p: str) -> str:
    """저장소 안 경로는 상대 경로로 적어 명령을 짧게."""
    r = os.path.relpath(p, REPO_ROOT)
    return r if not r.startswith("..") else p


def build_commands(cfg: dict, calib_dir: str, dx_json: str) -> list:
    """(설명, 명령) 목록. 명령은 저장소 맨 위 폴더에서 실행한다고 본다."""
    paths = RoundPaths(cfg)
    e = cfg["export"]
    rnd = cfg["round"]
    best = rel(paths.best_pt())
    onnx = rel(paths.npu_file("onnx"))
    hef = rel(paths.npu_file("hef"))
    dxnn = rel(paths.npu_file("dxnn"))
    best_onnx = os.path.splitext(best)[0] + ".onnx"
    dx_out = rel(os.path.join(paths.models, f"dx_{rnd}"))
    q = shlex.quote
    cmds = [
        ("ONNX 내보내기(Ultralytics, opset 11 은 Hailo 재학습 안내를 따름, 확인 필요)",
         f"yolo export model={q(best)} format=onnx imgsz={int(e['imgsz'])} opset={int(e['opset'])} simplify=True"
         f" && mv {q(best_onnx)} {q(onnx)}"),
        ("Hailo-8 HEF(Model Zoo v2.x 브랜치와 DFC 3.x 가상환경에서. 80 클래스 그대로라 zoo 의 yolov8s 설정 사용, 확인 필요)",
         f"hailomz compile {q(e['hailo_zoo_name'])} --ckpt {q(onnx)} --hw-arch {q(e['hailo_arch'])}"
         f" --calib-path {q(rel(calib_dir))} --classes 80 && mv {q(e['hailo_zoo_name'])}.hef {q(hef)}"),
        ("DX-M1 dxnn(DX-COM, x86_64. 옵션 이름 확인 필요)",
         f"mkdir -p {q(dx_out)} && dxcom -m {q(onnx)} -c {q(rel(dx_json))} -o {q(dx_out)}"
         f" && mv {q(dx_out)}/{q(os.path.splitext(os.path.basename(onnx))[0])}.dxnn {q(dxnn)}"),
        ("체크섬 기록",
         f"sha256sum {q(onnx)} {q(hef)} {q(dxnn)} > {q(os.path.splitext(onnx)[0])}.sha256"),
        ("파이로 보내기(대상 주소는 자리표시자. 파이의 models/ 에 회차 이름 그대로 쌓임)",
         f"rsync -av --progress {q(hef)} {q(dxnn)} {q(os.path.splitext(onnx)[0])}.sha256 {e['pi_target']}"),
    ]
    return cmds


def has_placeholder(cmd: str) -> bool:
    return "<PI_" in cmd


def export_round(cfg: dict, run: bool = False, prepare_calib_files: bool = True) -> dict:
    paths = RoundPaths(cfg)
    e = cfg["export"]
    out = os.path.join(paths.rd, "export")
    os.makedirs(out, exist_ok=True)
    calib_dir, n_calib = os.path.join(paths.rd, "calib"), 0
    if prepare_calib_files:
        calib_dir, n_calib = prepare_calib(cfg, int(e["calib_n"]))
    dx_json = os.path.join(out, f"dxcom_{e['model_stem']}_{cfg['round']}.json")
    write_json(dx_json, dxcom_config(cfg, calib_dir))
    cmds = build_commands(cfg, calib_dir, dx_json)
    sh = os.path.join(out, "commands.sh")
    with open(sh, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n# 저장소 맨 위 폴더에서 실행합니다. 단계마다 필요한 가상환경을 켜고 한 줄씩 실행해도 됩니다.\n")
        f.write("# Ultralytics 의 format=\"deepx\" 내보내기는 설치된 8.4.19 의 exporter 에 없어 쓰지 않았습니다(확인 필요).\n")
        f.write("set -euo pipefail\n")
        for desc, c in cmds:
            if has_placeholder(c):
                # <PI_HOST> 같은 자리표시자가 남은 줄은 셸이 < 를 입력 넘기기로 읽으므로 주석으로 둔다
                f.write(f"\n# {desc}\n# 설정 export.pi_target 의 주소를 바꾼 뒤 실행합니다.\n# {c}\n")
            else:
                f.write(f"\n# {desc}\n{c}\n")
    os.chmod(sh, 0o755)
    ran = []
    if run:
        for desc, c in cmds:
            if has_placeholder(c):
                print(f"건너뜀(주소 자리표시자): {desc}")
                continue
            print(f"실행: {desc}\n  {c}")
            subprocess.run(c, shell=True, check=True, cwd=REPO_ROOT)
            ran.append(desc)
    info = {"calib_dir": calib_dir, "calib_images": n_calib, "dx_json": dx_json, "commands": sh,
            "dry_run": not run, "ran": ran,
            "files": {k: paths.npu_file(k) for k in ("onnx", "hef", "dxnn")}}
    update_meta(paths, "export", info)
    info["cmds"] = cmds
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ONNX, HEF, dxnn 변환 명령")
    ap.add_argument("--config", default="training/configs/round.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True, help="명령만 출력(기본)")
    g.add_argument("--run", action="store_true", help="명령을 실제로 실행(학습 서버에서)")
    ap.add_argument("--no-calib", action="store_true", help="보정 이미지 폴더를 새로 만들지 않음")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    info = export_round(cfg, run=args.run, prepare_calib_files=not args.no_calib)
    print(f"보정 이미지 {info['calib_images']}장: {info['calib_dir']}")
    print(f"DX-COM 설정: {info['dx_json']}")
    for desc, c in info["cmds"]:
        print(f"\n# {desc}\n{c}")
    print(f"\n명령 모음: {info['commands']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
