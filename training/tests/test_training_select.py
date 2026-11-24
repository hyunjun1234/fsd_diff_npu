from __future__ import annotations

import ast
import os

import cv2
import pytest

from training.common import (
    KEEP_CLASSES,
    REPO_ROOT,
    RoundPaths,
    make_frame_id,
    parse_frame_id,
    read_csv,
    read_json,
)
from training.select_frames import (
    HoldoutLock,
    balanced_pick,
    discover_drive_streams,
    discover_trigger_streams,
    frame_features,
    iter_frames,
    load_lock,
    make_test_set,
    scan_all,
    select_round,
)
from training.tests import synth
from training.tests.conftest import make_cfg


# ---------------------------------------------------------------------------
# 규칙
# ---------------------------------------------------------------------------
def test_keep_classes_match_pi_decoder():
    """파이 후처리(realityview/types.py)의 COCO_KEEP 과 같은 번호여야 HEF, dxnn 을 바꿔 끼울 수 있다."""
    src = open(os.path.join(REPO_ROOT, "realityview", "types.py"), encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and n.targets[0].id == "COCO_KEEP")
    assert ast.literal_eval(node.value) == KEEP_CLASSES


def test_frame_id_roundtrip():
    fid = make_frame_id("d 2026/12", "seg__01", "front", 42)
    assert "__" not in fid.split("__")[0]
    info = parse_frame_id(fid)
    assert info["camera"] == "front" and info["idx"] == 42 and info["segment"] == "seg_01"
    assert parse_frame_id("000000000139") is None  # COCO 이미지는 규칙 밖


# ---------------------------------------------------------------------------
# 프레임 성질과 고르게 뽑기
# ---------------------------------------------------------------------------
def test_blur_and_brightness_features():
    tex = synth._texture(0)
    sharp = synth.draw_frame(tex, 0, 0, False, False)
    blurred = synth.draw_frame(tex, 0, 0, False, True)
    night = synth.draw_frame(tex, 0, 0, True, False)
    b1, br1, _ = frame_features(sharp)
    b2, _, _ = frame_features(blurred)
    _, br3, _ = frame_features(night)
    assert b1 > 60 > b2
    assert br1 > 60 > br3


def test_balanced_pick_fills_from_larger_group():
    cands = [{"scene": "day", "seg_key": "a/s", "camera": "c", "t": i} for i in range(10)]
    cands += [{"scene": "night", "seg_key": "a/n", "camera": "c", "t": i} for i in range(2)]
    out = balanced_pick(cands, 6)
    assert len(out) == 6
    assert sum(1 for c in out if c["scene"] == "night") == 2
    # 낮 프레임은 시간상 퍼져서 골라진다(처음과 끝 포함)
    ts = sorted(c["t"] for c in out if c["scene"] == "day")
    assert ts[0] == 0 and ts[-1] == 9


# ---------------------------------------------------------------------------
# 입력 찾기
# ---------------------------------------------------------------------------
def test_discover_streams_and_timestamps(world):
    cfg = make_cfg(world)
    streams = discover_drive_streams(cfg)
    assert len(streams) == 6  # 세그먼트 3 x 카메라 2
    s = streams[0]
    assert s.kind == "mjpeg" and s.times[1] == pytest.approx(0.1, abs=1e-6)
    assert streams[-1].kind == "video"  # 마지막 세그먼트는 avi
    got = list(iter_frames(s, 0.5, 30.0))
    assert [g[0] for g in got] == list(range(0, 40, 5))  # 10fps 에서 0.5초 간격
    trig = discover_trigger_streams(cfg)
    assert len(trig) == 1 and trig[0].trigger_kind == "ambiguous+track_new" and trig[0].trigger_score == 0.42
    assert trig[0].segment == "clip01" and trig[0].kind == "mjpeg"


def test_record_frames_image_layout(tmp_path):
    """record_frames.py 형식(<카메라>_00000.png + timestamps.csv)도 읽는다."""
    synth.write_segment(str(tmp_path / "drv" / "segA"), "front", n=10, video=False)
    cfg = make_cfg(str(tmp_path), drives=["drv"])
    cfg["paths"]["drives_root"] = str(tmp_path)
    streams = discover_drive_streams(cfg)
    assert len(streams) == 1 and streams[0].kind == "images" and len(streams[0].images) == 10
    assert len(list(iter_frames(streams[0], 0.5, 30.0))) == 2


def test_scan_drops_blur_and_duplicates(world):
    cfg = make_cfg(world)
    cands, stats = scan_all(discover_drive_streams(cfg), cfg)
    assert stats["blur_dropped"] == 2 * 3  # 5, 10, 15 번 프레임 x 카메라 2
    assert stats["dup_dropped"] >= 2 * 2  # 정차 구간(25~39)에서 첫 장 뒤로는 같은 프레임
    scenes = {c["scene"] for c in cands}
    assert scenes == {"day", "night"}
    assert all(c["scene"] == "night" for c in cands if c["segment"] == "seg_0001")


def test_scene_tags_override(world, tmp_path):
    tags = tmp_path / "tags.csv"
    tags.write_text("drive,segment,camera,start_s,end_s,scene\nd1,seg_0000,,0,100,rain\n")
    cfg = make_cfg(world, select={"scene_tags_csv": str(tags)})
    cands, _ = scan_all(discover_drive_streams(cfg), cfg)
    assert {c["scene"] for c in cands if c["segment"] == "seg_0000"} == {"rain"}


# ---------------------------------------------------------------------------
# 고정 시험 세트와 회차 선별
# ---------------------------------------------------------------------------
def test_test_set_lock_and_round_selection(world, tmp_path):
    cfg = make_cfg(world)
    cfg["paths"]["work_root"] = str(tmp_path / "work")
    info = select_round(cfg)
    paths = RoundPaths(cfg)
    lock = load_lock(paths)
    assert lock.exists and len(lock.segments) == 1
    test_rows = read_csv(paths.test_csv)
    assert 0 < len(test_rows) <= 6
    assert all(r["label_source"] == "none" and r["added_round"] == "r001" for r in test_rows)
    assert len(os.listdir(os.path.join(paths.test_dir, "images"))) == len(test_rows)

    rows = read_csv(paths.frames_csv)
    assert len(rows) == info["selected"] <= 20
    # 시험 세그먼트의 프레임은 학습 선별에 하나도 없다
    assert not any(r["seg_key"] in lock.segments for r in rows)
    # 트리거 클립이 먼저 들어가되 비율 상한(25%)을 넘지 않는다
    n_trig = sum(1 for r in rows if r["source"] == "trigger")
    assert 1 <= n_trig <= 5
    for r in rows:
        img = cv2.imread(os.path.join(paths.images, r["frame_id"] + ".jpg"))
        assert img is not None and img.shape[:2] == (240, 320)
    assert read_json(paths.meta_json)["steps"]["select"]["selected"] == len(rows)

    # 두 번 정하려 하면 막는다
    with pytest.raises(RuntimeError):
        make_test_set(cfg, [])
    # 손으로 목록을 고치면 잠금 검사에 걸린다
    with open(paths.test_csv, "a") as f:
        f.write("fake__x__y__000001,,,,,,,,r001,none\n")
    with pytest.raises(RuntimeError):
        load_lock(paths)


def test_window_lock_mode():
    lock = HoldoutLock({"exclude": "window", "window_sec": 2.0,
                        "frames": [{"frame_id": "a__s__c__000010", "seg_key": "a/s", "t": 10.0}]})
    assert lock.excludes("a__s__c__000010", "a/s", 10.0)
    assert lock.excludes("a__s__c__000011", "a/s", 11.5)
    assert not lock.excludes("a__s__c__000020", "a/s", 13.0)
    assert not lock.excludes("a__t__c__000010", "a/t", 10.0)
    assert lock.excludes("a__s__c__000099", "a/s", None)  # 시각을 모르면 보수적으로 뺀다


def test_second_round_skips_used_frames(world, tmp_path):
    work = str(tmp_path / "work")
    cfg1 = make_cfg(world)
    cfg1["paths"]["work_root"] = work
    select_round(cfg1)
    used = {r["frame_id"] for r in read_csv(RoundPaths(cfg1).frames_csv)}
    cfg2 = make_cfg(world, round_name="r002")
    cfg2["paths"]["work_root"] = work
    info = select_round(cfg2)
    rows2 = read_csv(RoundPaths(cfg2).frames_csv)
    assert not used & {r["frame_id"] for r in rows2}
    assert info["prev_round_excluded"] >= len(used) - 5  # 트리거 클립 일부는 다시 볼 수 있음
    assert info["test_set_created"] is None  # 시험 세트는 다시 만들지 않는다


def test_add_to_test_set_appends(world, tmp_path):
    synth.make_drive(str(tmp_path / "drives"), "d2", seed=3)
    cfg = make_cfg(world)
    cfg["paths"]["work_root"] = str(tmp_path / "work")
    cfg["paths"]["drives_root"] = str(tmp_path / "drives")
    cfg["drives"] = ["d2"]
    cands, _ = scan_all(discover_drive_streams(cfg), cfg)
    first = make_test_set(cfg, cands)
    cfg2 = make_cfg(world, round_name="r002")
    cfg2["paths"].update(cfg["paths"])
    synth.make_drive(str(tmp_path / "drives"), "d3", seed=5)
    cfg2["drives"] = ["d2", "d3"]
    cands2, _ = scan_all(discover_drive_streams(cfg2), cfg2)
    second = make_test_set(cfg2, cands2, append=True)
    assert second["total"] == first["total"] + second["added"]
    rows = read_csv(RoundPaths(cfg2).test_csv)
    assert {r["added_round"] for r in rows} == {"r001", "r002"}
    # 이미 시험 세그먼트를 낸 주행 세그먼트에서는 더 고르지 않는다
    seg_first = set(first["segments"])
    assert not any(r["seg_key"] in seg_first for r in rows if r["added_round"] == "r002")
