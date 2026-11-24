from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from training import run_round
from training.common import REPO_ROOT, Box, RoundPaths, read_csv, read_json, write_csv
from training.cvat_io import make_review_list
from training.evaluate import evaluate_boxes, evaluate_round, regression_warnings, subsets
from training.export_npu import export_round
from training.fakes import FakePredictor
from training.select_frames import load_lock
from training.train_det import assert_no_test_leak, filter_test_leak, start_weights
from training.train_seg import convert_mask
from training.tests import synth
from training.tests.conftest import make_cfg


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
def test_map_perfect_and_partial():
    gt = {"a": [Box(2, 0, 0, 10, 10), Box(0, 20, 20, 30, 40)], "b": [Box(2, 50, 50, 70, 60)]}
    perfect = {k: [Box(b.cls, *b.xyxy(), score=0.9) for b in v] for k, v in gt.items()}
    r = evaluate_boxes(gt, perfect)
    assert r["map50"] == pytest.approx(1.0) and r["map50_95"] == pytest.approx(1.0)
    assert r["per_class"]["car"]["recall"] == 1.0
    # 차 하나를 놓치고, 엉뚱한 차를 하나 높은 점수로 낸 경우
    worse = {"a": [Box(2, 0, 0, 10, 10, 0.5), Box(0, 20, 20, 30, 40, 0.9), Box(2, 100, 100, 120, 120, 0.95)], "b": []}
    r = evaluate_boxes(gt, worse)
    assert r["per_class"]["car"]["recall"] == pytest.approx(0.5)
    assert r["per_class"]["person"]["ap50"] == pytest.approx(1.0)
    assert 0.0 < r["per_class"]["car"]["ap50"] < 0.5
    # 박스가 조금 어긋나면 mAP50 은 그대로, mAP50-95 는 떨어진다
    shifted = {k: [Box(b.cls, b.x1 + 1.5, b.y1, b.x2 + 1.5, b.y2, 0.9) for b in v] for k, v in gt.items()}
    r = evaluate_boxes(gt, shifted)
    assert r["map50"] == pytest.approx(1.0) and r["map50_95"] < 0.9


def test_regression_warning_rules():
    cfg = make_cfg("/tmp/unused")
    good = {"old": {"map50": 0.8, "map50_95": 0.5, "n_gt": 100,
                    "per_class": {"car": {"n_gt": 50, "recall": 0.9}, "person": {"n_gt": 5, "recall": 0.9}}}}
    bad = {"old": {"map50": 0.78, "map50_95": 0.495, "n_gt": 100,
                   "per_class": {"car": {"n_gt": 50, "recall": 0.8}, "person": {"n_gt": 5, "recall": 0.1}}}}
    w = regression_warnings(good, bad, cfg)
    assert any("map50 " in x for x in w)  # 0.02 하락은 허용폭 0.01 을 넘음
    assert not any("map50_95" in x for x in w)  # 0.005 하락은 허용
    assert any("car 재현율" in x for x in w)
    assert not any("person" in x for x in w)  # 박스 10개 미만 클래스는 판단하지 않음


def test_subsets_old_new():
    rows = [{"frame_id": "a", "added_round": "r001", "scene": "day"},
            {"frame_id": "b", "added_round": "r002", "scene": "night"}]
    s = subsets(rows, "r002")
    assert s["old"] == ["a"] and s["new"] == ["b"] and s["scene:night"] == ["b"]


def test_convert_mask_modes():
    m = np.array([[0, 1, 255]], np.uint8)
    assert convert_mask(m, "road_first").tolist() == [[1, 0, 255]]
    assert convert_mask(m, "cityscapes_keep").tolist() == [[255, 0, 255]]
    with pytest.raises(ValueError):
        convert_mask(m, "nope")


def test_review_random_sample(tmp_path):
    cfg = make_cfg(str(tmp_path))
    paths = RoundPaths(cfg)
    ids = [f"d__s__c__{i:06d}" for i in range(20)]
    write_csv(paths.frames_csv, [{"frame_id": f} for f in ids], ["frame_id"])
    write_csv(paths.suspects_csv, [{"frame_id": f, "reasons": "ambiguous:1" if i < 4 else ""}
                                   for i, f in enumerate(ids)], ["frame_id", "reasons"])
    rows = make_review_list(cfg)
    assert sum(r["why"].startswith("suspect") for r in rows) == 4
    assert sum(r["why"] == "random" for r in rows) == 8  # 나머지 16장의 50%


# ---------------------------------------------------------------------------
# 끝에서 끝: 가짜 라벨러, 가짜 학습기로 두 회차
# ---------------------------------------------------------------------------
def _accept_all(zip_in: str, zip_out: str) -> None:
    """사람이 자동 라벨을 그대로 승인한 검수 결과(CVAT 내보내기 zip)를 흉내 낸다."""
    import shutil

    os.makedirs(os.path.dirname(zip_out), exist_ok=True)
    shutil.copyfile(zip_in, zip_out)


@pytest.fixture(scope="module")
def two_rounds(world, tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("e2e"))
    synth.make_drive(os.path.join(world, "drives"), "d2", seed=7)
    common = {"work_root": os.path.join(tmp, "work"), "models_root": os.path.join(tmp, "models")}

    cfg1 = make_cfg(world, drives=["d1"])
    cfg1["paths"].update(common)
    out1 = run_round.run(cfg1, "select", "review_export", fake=True)
    p1 = RoundPaths(cfg1)
    _accept_all(out1["review_export"]["annotations_zip"], os.path.join(p1.review, "cvat_export.zip"))
    _accept_all(out1["review_export_test"]["annotations_zip"], os.path.join(p1.test_dir, "review", "cvat_export.zip"))
    out1.update(run_round.run(cfg1, "review_import", "export", fake=True))

    cfg2 = make_cfg(world, round_name="r002", drives=["d2"])
    cfg2["paths"].update(common)
    out2 = run_round.run(cfg2, "select", "review_export", fake=True)
    p2 = RoundPaths(cfg2)
    _accept_all(out2["review_export"]["annotations_zip"], os.path.join(p2.review, "cvat_export.zip"))
    out2.update(run_round.run(cfg2, "review_import", "export", fake=True))
    return cfg1, out1, cfg2, out2


def test_e2e_round1(two_rounds):
    cfg1, out1, _, _ = two_rounds
    p = RoundPaths(cfg1)
    # 시험 세트는 사람 라벨(검수 zip)로 바뀌었고 평가는 그 프레임만 쓴다
    test_rows = read_csv(p.test_csv)
    assert test_rows and all(r["label_source"] == "human" for r in test_rows)
    assert out1["evaluate"]["skipped_unlabeled"] == 0
    assert out1["evaluate"]["models"]["r001"]["all"]["map50"] == pytest.approx(1.0)
    assert out1["evaluate"]["warnings"] == []
    md = open(os.path.join(p.rd, "eval", "eval.md"), encoding="utf-8").read()
    assert md.startswith("이 문서는") and "| 지표 | r000 | r001 |" in md
    assert all(line.count("|") <= 4 for line in md.splitlines())  # 표는 3열 이하

    # 데이터 섞기: 첫 회차는 이전 회차가 없고 COCO 는 비율대로
    d = out1["train_det"]
    assert d["prev"] == 0 and d["coco"] == min(12, round(0.25 * d["own_train"]))
    assert d["start_weights"] == "yolov8s.pt"
    data = yaml.safe_load(open(os.path.join(p.rd, "data.yaml"), encoding="utf-8"))
    assert len(data["names"]) == 80 and data["names"][2] == "car"
    train = [x.strip() for x in open(data["train"], encoding="utf-8")]
    assert len(train) == d["own_train"] + d["coco"]
    assert_no_test_leak(train, load_lock(p))
    for img in train:  # Ultralytics 규칙: images 를 labels 로 바꾼 자리에 라벨이 있다
        lab = os.path.splitext(img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep))[0] + ".txt"
        assert os.path.exists(lab), lab
    best = json.load(open(p.best_pt(), encoding="utf-8"))
    assert best["fake"] and best["params"]["imgsz"] == 640 and best["params"]["batch"] == 16

    # 세그 데이터: 0 도로, 1 그 밖
    seg = out1["seg_prepare"]
    assert seg["train"] > 0 and seg["num_classes"] == 2
    ann_dir = os.path.join(p.rd, "seg", "ann_dir", "train")
    a = cv2.imread(os.path.join(ann_dir, sorted(os.listdir(ann_dir))[0]), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(a)) <= {0, 1, 255} and (a == 0).sum() > 1000

    # 변환: 기본은 명령만. 이름은 회차 규칙
    ex = out1["export"]
    assert ex["dry_run"] and ex["ran"] == []
    text = "\n".join(c for _, c in ex["cmds"])
    for name in ("yolov8s_r001.onnx", "yolov8s_r001.hef", "yolov8s_r001.dxnn"):
        assert name in text
    assert "hailomz compile yolov8s" in text and "--hw-arch hailo8" in text and "dxcom -m" in text
    assert "rsync -av" in text and "<PI_HOST>" in text
    calib = os.listdir(ex["calib_dir"])
    assert 0 < len(calib) <= 8
    lock = load_lock(p)
    assert not filter_test_leak([os.path.join(ex["calib_dir"], c) for c in calib], lock)[1]
    dx = read_json(ex["dx_json"])
    assert dx["inputs"]["images"] == [1, 3, 640, 640] and dx["default_loader"]["dataset_path"] == ex["calib_dir"]
    assert "ppu" not in dx
    assert os.access(ex["commands"], os.X_OK)
    sh = open(ex["commands"], encoding="utf-8").read()
    assert "\n# rsync -av" in sh and "\nrsync" not in sh  # 자리표시자 줄은 주석
    r = subprocess.run(["bash", "-n", ex["commands"]], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr  # 문법 검사만(실행하지 않음)

    meta = read_json(p.meta_json)
    for step in ("select", "autolabel", "review_import", "label_qa", "merge", "data", "train_det", "evaluate", "export"):
        assert step in meta["steps"], step


def test_e2e_round2_mixes_previous(two_rounds):
    cfg1, _, cfg2, out2 = two_rounds
    d = out2["train_det"]
    assert d["start_weights"] == RoundPaths(cfg1).best_pt()  # 직전 회차 best.pt 에서 시작
    prev_own = sum(1 for _ in open(os.path.join(RoundPaths(cfg1).rd, "own_train.txt"), encoding="utf-8"))
    assert d["prev"] == min(prev_own, round(0.5 * d["own_train"])) > 0  # prev_ratio 0.5
    train = [x.strip() for x in open(os.path.join(RoundPaths(cfg2).rd, "train.txt"), encoding="utf-8")]
    assert any(os.sep + "r001" + os.sep in t for t in train)
    # 시험 세트는 r001 에서 정한 것 하나뿐이고 r002 에서는 모두 예전 장면이다
    assert out2["select"]["test_set_created"] is None
    assert "old" in out2["evaluate"]["models"]["r002"]
    assert "new" not in out2["evaluate"]["models"]["r002"]


def test_regression_detected_with_bad_model(two_rounds):
    """직전 회차보다 사람을 못 잡는 모델이면 경고가 나온다."""
    _, _, cfg2, _ = two_rounds
    res = evaluate_round(cfg2, models=[("r001", "x"), ("r002", "y")],
                         predictors={"r001": FakePredictor(), "r002": FakePredictor(miss_classes=[0])})
    # 시험 세트의 사람 박스가 10개 미만이라 클래스 재현율 대신 평균 mAP 하락으로 잡힌다
    assert any(w.startswith("old map50 ") for w in res["warnings"])
    assert res["models"]["r002"]["old"]["per_class"]["person"]["recall"] == 0.0


def test_start_weights_requires_previous(tmp_path):
    cfg = make_cfg(str(tmp_path), round_name="r003")
    with pytest.raises(FileNotFoundError):
        start_weights(cfg)


def test_export_dx_ppu_requires_block(two_rounds):
    cfg1 = dict(two_rounds[0])
    cfg1["export"] = {**cfg1["export"], "dx_ppu": True, "dx_ppu_json": None}
    with pytest.raises(FileNotFoundError):
        export_round(cfg1, prepare_calib_files=False)


def test_cli_fake_help_runs():
    r = subprocess.run([sys.executable, "-m", "training.run_round", "--help"], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--until" in r.stdout


def test_modules_do_not_import_big_libraries():
    """패키지를 불러오기만 해서는 torch, ultralytics, groundingdino, sam2, mmengine 이 올라오지 않는다."""
    code = (
        "import sys\n"
        "import training.common, training.select_frames, training.autolabel, training.cvat_io, training.label_qa\n"
        "import training.train_det, training.train_seg, training.evaluate, training.export_npu, training.run_round\n"
        "import training.fakes\n"
        "print(','.join(m for m in ('torch', 'ultralytics', 'groundingdino', 'sam2', 'mmengine', 'mmseg') "
        "if m in sys.modules))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_split_own_by_segment(tmp_path):
    from training.train_det import split_own

    cfg = make_cfg(str(tmp_path))
    paths = RoundPaths(cfg)
    rows = []
    for seg, n in (("s1", 30), ("s2", 5), ("s3", 30), ("s4", 35)):
        rows += [{"frame_id": f"d__{seg}__c__{i:06d}", "source": "drive", "seg_key": f"d/{seg}", "t": i}
                 for i in range(n)]
    write_csv(paths.frames_csv, rows, ["frame_id", "source", "seg_key", "t"])
    write_csv(paths.sources_csv, [{"frame_id": r["frame_id"], "label_source": "auto"} for r in rows],
              ["frame_id", "label_source"])
    train, val = split_own(cfg)
    segs = lambda xs: {os.path.basename(x).split("__")[1] for x in xs}  # noqa: E731
    assert segs(val) == {"s2"}  # 목표 10장(10%)의 1.5배를 넘는 큰 세그먼트는 건너뛴다
    assert not segs(train) & segs(val) and len(train) + len(val) == 100

    # 세그먼트 하나뿐이면 시간 순 마지막 10% 가 검증
    write_csv(paths.frames_csv, rows[:30], ["frame_id", "source", "seg_key", "t"])
    train, val = split_own(cfg)
    assert len(val) == 3 and all(int(v[-10:-4]) >= 27 for v in val)


def test_window_mode_keeps_training_frames(world, tmp_path):
    """exclude=window 이면 시험 프레임 앞뒤만 빠지고 같은 세그먼트의 먼 프레임은 학습에 남는다."""
    cfg = make_cfg(world, test_set={"exclude": "window", "window_sec": 1.0, "n_frames": 6})
    cfg["paths"]["work_root"] = str(tmp_path / "work")
    cfg["paths"]["models_root"] = str(tmp_path / "models")
    out = run_round.run(cfg, "select", "train_det", fake=True)
    p = RoundPaths(cfg)
    lock = load_lock(p)
    assert lock.mode == "window" and not lock.segments
    d = out["train_det"]
    assert d["own_train"] > 0
    train = [x.strip() for x in open(os.path.join(p.rd, "train.txt"), encoding="utf-8")]
    from training.train_det import known_frame_times

    times = known_frame_times(cfg)
    assert_no_test_leak(train, lock, times)
    # 시험 프레임과 같은 세그먼트인 학습 프레임이 있고, 모두 1초 넘게 떨어져 있다
    test_by_seg = {}
    for f in lock.frames.values():
        test_by_seg.setdefault(f["seg_key"], []).append(float(f["t"]))
    shared = 0
    for img in train:
        fid = os.path.splitext(os.path.basename(img))[0]
        parts = fid.split("__")
        if len(parts) != 4:
            continue
        seg = parts[0] + "/" + parts[1]
        if seg in test_by_seg:
            shared += 1
            assert min(abs(times[fid] - t) for t in test_by_seg[seg]) > 1.0
    assert shared > 0
