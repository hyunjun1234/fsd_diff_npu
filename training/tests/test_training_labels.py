from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
import pytest

from training.autolabel import (
    GroundingSamLabeler,
    autolabel_round,
    mask_to_box,
    phrase_to_coco,
    suspect_reasons,
    track_clip,
)
from training.common import MASK_ROAD, Box, RoundPaths, read_csv, read_json, read_yolo
from training.cvat_io import (
    cvat_rle_to_mask,
    export_for_cvat,
    import_from_cvat,
    make_review_list,
    mask_to_polygons,
    merge_round,
    polygons_to_mask,
    read_cvat_zip,
)
from training.label_qa import qa_round
from training.select_frames import select_round
from training.tests.conftest import make_cfg


@pytest.fixture(scope="module")
def labeled(world, tmp_path_factory):
    """선별과 자동 라벨까지 끝낸 회차 하나."""
    cfg = make_cfg(world)
    cfg["paths"]["work_root"] = str(tmp_path_factory.mktemp("lab_work"))
    select_round(cfg)
    autolabel_round(cfg)
    return cfg


# ---------------------------------------------------------------------------
# 자동 라벨
# ---------------------------------------------------------------------------
def test_autolabel_outputs_and_suspects(labeled):
    paths = RoundPaths(labeled)
    ids = [r["frame_id"] for r in read_csv(paths.frames_csv)]
    auto = paths.auto_dir("fake")
    for fid in ids:
        assert os.path.exists(os.path.join(auto, "labels", fid + ".txt"))
        assert os.path.exists(os.path.join(auto, "scores", fid + ".json"))
        m = cv2.imread(os.path.join(auto, "masks", fid + ".png"), cv2.IMREAD_GRAYSCALE)
        assert m is not None and set(np.unique(m)) <= {0, 1} and (m == MASK_ROAD).sum() > 1000
    # 차와 사람이 모두 라벨에 있고 COCO 번호(차 2, 사람 0)를 쓴다
    boxes = read_yolo(os.path.join(auto, "labels", ids[0] + ".txt"), 320, 240)
    assert sorted(b.cls for b in boxes) == [0, 2]
    # 비교 라벨러는 사람을 놓치므로 사람이 보이는 프레임은 모두 의심 프레임
    sus = read_csv(paths.suspects_csv)
    assert len(sus) == len(ids)
    assert all("disagree:1+0" in r["reasons"] for r in sus)
    assert read_json(paths.meta_json)["steps"]["autolabel"]["suspects"] == len(ids)


def test_suspect_rules():
    acfg = {"ambiguous": [0.3, 0.5], "match_iou": 0.5, "disagree_min": 1}
    a = [Box(2, 0, 0, 10, 10, 0.9)]
    r = suspect_reasons(a + [Box(0, 50, 50, 60, 70, 0.4)], a, [Box(2, 0, 0, 10, 10, 0.8)], acfg)
    assert r["ambiguous"] == 1 and r["reasons"] == "ambiguous:1"
    r = suspect_reasons(a, a, [Box(7, 0, 0, 10, 10, 0.8)], acfg)  # 차를 트럭으로: 클래스가 다르면 어긋남
    assert r["reasons"] == "disagree:1+1"
    r = suspect_reasons(a, a, None, acfg)
    assert r["reasons"] == ""


def test_phrase_mapping():
    assert phrase_to_coco("car") == 2
    assert phrase_to_coco("a truck") == 7
    assert phrase_to_coco("motorcycle bicycle") == 3
    assert phrase_to_coco("tree") is None


class _FakeApi:
    """OfficialApi 자리에 넣어 Grounding DINO + SAM 2 라벨러의 앞뒤 처리를 시험한다."""

    def __init__(self):
        self.calls = []

    def gdino_predict(self, path, caption):
        self.calls.append(caption)
        if caption.startswith("road"):
            return [(0, 120, 320, 240)], [0.6], ["road"]
        return [(10, 10, 50, 60), (100, 100, 140, 160), (0, 0, 5, 5)], [0.8, 0.33, 0.5], ["car", "person", "tree"]

    def sam2_points(self, rgb, pts):
        assert pts.shape == (3, 2) and pts[0, 1] > rgb.shape[0] * 0.9
        m = np.zeros(rgb.shape[:2], bool)
        m[150:] = True
        return m

    def sam2_boxes(self, rgb, boxes):
        out = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros(rgb.shape[:2], bool)
            m[int(y1):int(y2), int(x1):int(x2)] = True
            out.append(m)
        return out

    def sam2_track(self, frames_dir, first_boxes):
        out = {}
        for f in range(3):
            out[f] = {}
            for k, (x1, y1, x2, y2) in enumerate(first_boxes):
                m = np.zeros((240, 320), bool)
                m[int(y1):int(y2), int(x1) + 5 * f:int(x2) + 5 * f] = True
                out[f][k] = m
        return out


def test_grounding_sam_labeler_with_fake_api(tmp_path):
    cfg = make_cfg(str(tmp_path))
    img = np.zeros((240, 320, 3), np.uint8)
    p = str(tmp_path / "a.jpg")
    cv2.imwrite(p, img)
    api = _FakeApi()
    res = GroundingSamLabeler(cfg, api=api).label(img, p)
    assert [b.cls for b in res.boxes] == [2, 0]  # tree 는 버림
    assert res.road.dtype == np.uint8 and (res.road == MASK_ROAD).sum() == 90 * 320
    cfg["autolabel"]["road"] = "text"
    res = GroundingSamLabeler(cfg, api=api).label(img, p)
    assert api.calls[-1].startswith("road") and (res.road == MASK_ROAD).sum() == 120 * 320


def test_track_clip_with_fake_api(tmp_path):
    cfg = make_cfg(str(tmp_path))
    d = tmp_path / "clip"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"{i:05d}.jpg"), np.zeros((240, 320, 3), np.uint8))
    info = track_clip(cfg, str(d), str(tmp_path / "out"), api=_FakeApi(),
                      first_boxes=[Box(2, 10, 20, 50, 60, 0.9)])
    assert info == {"frames": 3, "objects": 1}
    b = read_yolo(str(tmp_path / "out" / "labels" / "00002.txt"), 320, 240)
    assert b[0].cls == 2 and b[0].x1 == pytest.approx(20, abs=0.01)
    assert mask_to_box(np.zeros((4, 4), bool)) is None


# ---------------------------------------------------------------------------
# CVAT 형식
# ---------------------------------------------------------------------------
def test_mask_polygon_roundtrip():
    m = np.zeros((240, 320), np.uint8)
    pts = np.array([[60, 239], [140, 130], [180, 130], [300, 239]], np.int32)
    cv2.fillPoly(m, [pts], 1)
    m[200:220, 150:200] = 0  # 도로 위 차 자리(구멍)
    outers, holes = mask_to_polygons(m)
    assert len(outers) == 1 and len(holes) == 1
    back = polygons_to_mask(outers, 320, 240, holes)
    a, b = m == 1, back == 1
    assert (a & b).sum() / (a | b).sum() > 0.97
    assert back[210, 175] == 0


def test_cvat_rle_decode():
    # 3x2 상자: 0 두 칸, 1 세 칸, 0 한 칸
    m = cvat_rle_to_mask("2,3,1", left=1, top=1, width=3, height=2, w=5, h=4)
    assert m[1, 1] == 0 and m[1, 2] == 0 and m[1, 3] == MASK_ROAD and m[2, 1] == MASK_ROAD and m[2, 3] == 0


@pytest.mark.parametrize("fmt", ["cvat", "yolo"])
def test_export_import_roundtrip(labeled, fmt):
    paths = RoundPaths(labeled)
    rows = make_review_list(labeled)
    info = export_for_cvat(labeled, fmt=fmt)
    assert info["frames"] == len(rows)
    with zipfile.ZipFile(info["images_zip"]) as z:
        assert sorted(z.namelist()) == sorted(r["frame_id"] + ".jpg" for r in rows)
    sizes = {r["frame_id"]: (320, 240) for r in rows}
    data = read_cvat_zip(info["annotations_zip"], sizes)
    fid = rows[0]["frame_id"]
    auto = read_yolo(os.path.join(paths.auto_dir("fake"), "labels", fid + ".txt"), 320, 240)
    got, mask = data[fid]
    assert sorted((b.cls, round(b.x1), round(b.y2)) for b in got) == sorted((b.cls, round(b.x1), round(b.y2)) for b in auto)
    if fmt == "cvat":
        assert mask is not None and (mask == MASK_ROAD).sum() > 1000
    else:
        assert mask is None


def test_review_list_has_random_sample(labeled):
    # 모든 프레임이 의심이면 무작위 표본은 비고, 의심 프레임은 빠짐없이 들어간다
    rows = make_review_list(labeled)
    assert len(rows) == len(read_csv(RoundPaths(labeled).frames_csv))
    assert all(r["why"].startswith("suspect:") for r in rows)


def test_review_import_qa_and_merge(labeled, tmp_path):
    """사람이 첫 프레임에 박스 하나를 더하고 둘째 프레임의 사람 박스를 지운 검수 결과를 흉내 낸다."""
    paths = RoundPaths(labeled)
    make_review_list(labeled)
    info = export_for_cvat(labeled, fmt="cvat")
    with zipfile.ZipFile(info["annotations_zip"]) as z:
        root = ET.fromstring(z.read("annotations.xml"))
    images = list(root.iter("image"))
    ET.SubElement(images[0], "box", label="truck", source="manual", occluded="0",
                  xtl="5", ytl="5", xbr="60", ybr="50", z_order="0")
    for b in list(images[1]):
        if b.get("label") == "person":
            images[1].remove(b)
    human = str(tmp_path / "cvat_export.zip")
    with zipfile.ZipFile(human, "w") as z:
        z.writestr("annotations.xml", ET.tostring(root, encoding="unicode"))
    got = import_from_cvat(labeled, human)
    assert got["frames"] == len(images) and got["unknown_frames"] == 0

    qa = qa_round(labeled)["all"]
    n_h = qa["human_boxes"]
    assert qa["miss_rate"] == pytest.approx(1 / n_h, abs=1e-3)  # 더한 트럭을 자동 라벨이 놓침
    assert qa["false_rate"] == pytest.approx(1 / qa["auto_boxes"], abs=1e-3)  # 지운 사람을 자동 라벨이 잡음
    assert qa["box_iou"] > 0.99 and qa["road_iou"] > 0.97  # 구멍(not_road)까지 되돌아와야 높다
    md = open(os.path.join(paths.rd, "qa", "label_qa.md"), encoding="utf-8").read()
    assert "| 지표 | 검수 전체 | 무작위 표본 |" in md

    c = merge_round(labeled)
    assert c["reviewed"] == len(images) and c["missing"] == 0
    first = os.path.splitext(images[0].get("name"))[0]
    assert 7 in [b.cls for b in read_yolo(os.path.join(paths.labels, first + ".txt"), 320, 240)]
    # 자동 라벨 폴더는 그대로다
    assert 7 not in [b.cls for b in read_yolo(os.path.join(paths.auto_dir("fake"), "labels", first + ".txt"), 320, 240)]
