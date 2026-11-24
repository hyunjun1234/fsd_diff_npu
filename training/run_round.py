#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import RoundPaths, load_config, read_csv  # noqa: E402

# 한 회차를 처음부터 끝까지 돌린다. 사람 검수(CVAT)가 가운데 있어 두 번에 나눠 실행한다.
#
# 예) 학습 서버에서
#   1부: 선별, 자동 라벨, CVAT 내보내기까지(기본 --until review_export)
#     python3 -m training.run_round --config training/configs/round.yaml
#   (CVAT 에서 검수하고 내보낸 zip 을 review/cvat_export.zip, 시험 세트는 test_set/review/cvat_export.zip 에 저장)
#   2부: 들여오기부터 변환 명령까지
#     python3 -m training.run_round --config training/configs/round.yaml --from review_import --until export
#
#   --fake 를 붙이면 가짜 라벨러, 가짜 학습기, 가짜 예측기로 흐름만 점검한다(GPU 없이).
#
# 결과 위치는 각 단계 모듈의 주석과 docs/training_pipeline.md 에 있다. 단계별 요약은 datasets/<회차>/meta.json.

STEPS = [
    "select",         # 프레임 선별(시험 세트가 없으면 먼저 정해 잠금)
    "autolabel",      # 자동 라벨(회차 프레임, 사람 라벨이 아직 없는 시험 프레임)
    "review_export",  # 검수 목록과 CVAT 업로드 zip
    "review_import",  # CVAT 내보내기 zip 들여오기(회차, 시험 세트)
    "label_qa",       # 사람 라벨과 자동 라벨 비교
    "merge",          # 최종 라벨 = 사람 라벨 우선
    "train_det",      # 데이터 섞기와 YOLOv8s 학습
    "seg_prepare",    # BiSeNetV2 도로 세그 데이터 준비
    "evaluate",       # 고정 시험 세트에서 직전 회차와 비교
    "export",         # ONNX, HEF, dxnn 명령(기본 dry-run)
]


def _test_needs_labels(paths: RoundPaths) -> bool:
    return any(r.get("label_source") != "human" for r in read_csv(paths.test_csv))


def run(cfg: dict, start: str = "select", until: str = "review_export", fake: bool = False,
        review_zip: str | None = None, test_zip: str | None = None, run_export: bool = False,
        run_seg: bool = False) -> dict:
    from training import autolabel, cvat_io, evaluate, export_npu, label_qa, select_frames, train_det, train_seg

    paths = RoundPaths(cfg)
    if fake:
        cfg["autolabel"]["primary"] = "fake"
        cfg["autolabel"]["secondary"] = "fake_secondary"
    steps = STEPS[STEPS.index(start): STEPS.index(until) + 1]
    done: dict = {}
    for step in steps:
        print(f"== {cfg['round']} {step}")
        if step == "select":
            done[step] = select_frames.select_round(cfg)
        elif step == "autolabel":
            done[step] = autolabel.autolabel_round(cfg)
            if _test_needs_labels(paths):
                # 시험 세트는 사람이 전부 라벨링하지만, 처음부터 칠하지 않게 자동 라벨로 미리 채워 둔다
                done["autolabel_test"] = autolabel.autolabel_round(cfg, test_set=True)
        elif step == "review_export":
            cvat_io.make_review_list(cfg)
            done[step] = cvat_io.export_for_cvat(cfg)
            if _test_needs_labels(paths):
                cvat_io.make_review_list(cfg, test_set=True)
                done["review_export_test"] = cvat_io.export_for_cvat(cfg, test_set=True)
            print(f"CVAT 검수 대기: {done[step]['images_zip']} 로 작업을 만들고 {done[step]['annotations_zip']} 를 불러온 뒤,"
                  f" 검수 결과를 {os.path.join(paths.review, 'cvat_export.zip')} 로 내보내세요.")
        elif step == "review_import":
            rz = review_zip or os.path.join(paths.review, "cvat_export.zip")
            if os.path.exists(rz):
                done[step] = cvat_io.import_from_cvat(cfg, rz)
            else:
                print(f"[경고] 검수 zip 이 없어 자동 라벨만 씁니다: {rz}", file=sys.stderr)
            tz = test_zip or os.path.join(paths.test_dir, "review", "cvat_export.zip")
            if os.path.exists(tz):
                done["review_import_test"] = cvat_io.import_from_cvat(cfg, tz, test_set=True)
        elif step == "label_qa":
            done[step] = label_qa.qa_round(cfg)
        elif step == "merge":
            done[step] = cvat_io.merge_round(cfg)
        elif step == "train_det":
            trainer = None
            if fake:
                from training.fakes import FakeTrainer

                trainer = FakeTrainer()
            done[step] = train_det.train_round(cfg, trainer)
        elif step == "seg_prepare":
            info = train_seg.prepare_seg(cfg)
            if run_seg:
                info["work_dir"] = train_seg.run_mmseg(cfg, info)
            done[step] = info
        elif step == "evaluate":
            models = None
            if fake:
                models = [(cfg["prev_round"], "fake"), (cfg["round"], "fake")]
            done[step] = evaluate.evaluate_round(cfg, models)
            for w in done[step]["warnings"]:
                print(f"[경고] {w}")
        elif step == "export":
            done[step] = export_npu.export_round(cfg, run=run_export)
            for desc, c in done[step]["cmds"]:
                print(f"# {desc}\n{c}")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="학습 한 회차 실행")
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--from", dest="start", choices=STEPS, default="select")
    ap.add_argument("--until", choices=STEPS, default="review_export")
    ap.add_argument("--review-zip", default=None, help="회차 검수 결과 zip(기본 datasets/<회차>/review/cvat_export.zip)")
    ap.add_argument("--test-zip", default=None, help="시험 세트 라벨 zip(기본 test_set/review/cvat_export.zip)")
    ap.add_argument("--fake", action="store_true", help="가짜 라벨러, 학습기, 예측기")
    ap.add_argument("--run-export", action="store_true", help="변환 명령을 실제로 실행(기본은 출력만)")
    ap.add_argument("--run-seg", action="store_true", help="mmsegmentation 학습까지 실행")
    args = ap.parse_args(argv)
    if STEPS.index(args.start) > STEPS.index(args.until):
        ap.error("--from 이 --until 보다 뒤입니다")
    cfg = load_config(args.config)
    run(cfg, args.start, args.until, args.fake, args.review_zip, args.test_zip, args.run_export, args.run_seg)
    print(f"회차 기록: {RoundPaths(cfg).meta_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
