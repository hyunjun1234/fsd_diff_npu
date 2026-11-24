#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    KEEP_CLASSES,
    MASK_BG,
    MASK_ROAD,
    NAME_TO_COCO,
    Box,
    RoundPaths,
    load_config,
    read_csv,
    read_yolo,
    update_meta,
    write_csv,
    write_yolo,
)

# 자동 라벨을 CVAT 에 올릴 zip 으로 내보내고, 검수 뒤 CVAT 에서 내려받은 zip 을 다시 읽는다.
#
# 예) 저장소 폴더에서
#   python3 -m training.cvat_io review-list --config training/configs/round.yaml
#   python3 -m training.cvat_io export --config training/configs/round.yaml
#   (CVAT 에서 검수 후 내보낸 zip 을 out/training/datasets/r001/review/cvat_export.zip 으로 저장)
#   python3 -m training.cvat_io import --config training/configs/round.yaml --zip out/training/datasets/r001/review/cvat_export.zip
#   python3 -m training.cvat_io merge --config training/configs/round.yaml
#   시험 세트는 같은 명령에 --test-set 을 붙인다.
#
# 결과: review/review_list.csv, review/review_images.zip(CVAT 작업 만들 때 올릴 이미지),
#       review/cvat_upload.zip(작업에 불러올 자동 라벨), review/reviewed/labels, masks(사람 라벨),
#       labels/, masks/, label_sources.csv(학습에 쓰는 최종 라벨과 출처)

# CVAT 프로젝트의 라벨 순서. YOLO 1.1 의 obj.names 와 같은 순서이고, 번호는 COCO 번호와 다르다.
CVAT_LABELS = [KEEP_CLASSES[k] for k in sorted(KEEP_CLASSES)]  # person, bicycle, car, motorcycle, bus, truck
# 도로 다각형 안에서 도로가 아닌 곳(도로 위 차 등). 들여올 때 도로를 칠한 뒤 이 다각형을 지운다.
HOLE_LABEL = "not_road"


# ---------------------------------------------------------------------------
# 검수 대상 목록: 의심 프레임 전부 + 나머지에서 무작위 표본
# ---------------------------------------------------------------------------
def make_review_list(cfg: dict, test_set: bool = False) -> list[dict]:
    """review_list.csv 를 쓴다. 시험 세트는 사람이 전부 라벨링해야 하므로 모든 프레임이 대상이다."""
    paths = RoundPaths(cfg)
    if test_set:
        rows = [{"frame_id": r["frame_id"], "why": "test_set"} for r in read_csv(paths.test_csv)]
        out = os.path.join(paths.test_dir, "review", "review_list.csv")
    else:
        sus = read_csv(paths.suspects_csv)
        all_ids = [r["frame_id"] for r in read_csv(paths.frames_csv)]
        flagged = {r["frame_id"]: r["reasons"] for r in sus if r.get("reasons")}
        rest = [f for f in all_ids if f not in flagged]
        rng = random.Random(int(cfg["review"].get("seed", 0)))
        k = int(round(len(rest) * float(cfg["review"]["random_ratio"])))
        sample = set(rng.sample(rest, min(k, len(rest)))) if rest else set()
        rows = [{"frame_id": f, "why": "suspect:" + flagged[f]} for f in all_ids if f in flagged]
        rows += [{"frame_id": f, "why": "random"} for f in all_ids if f in sample]
        out = os.path.join(paths.review, "review_list.csv")
    write_csv(out, rows, ["frame_id", "why"])
    return rows


# ---------------------------------------------------------------------------
# 도로 마스크 <-> CVAT 다각형
# ---------------------------------------------------------------------------
def mask_to_polygons(mask: np.ndarray, min_area: float = 200.0, eps_px: float = 2.0,
                     min_hole: float = 50.0) -> tuple[list, list]:
    """도로 화소(MASK_ROAD) 영역을 (바깥 윤곽 목록, 구멍 윤곽 목록)으로. 작은 조각은 버린다.

    CVAT 다각형은 구멍을 못 가지므로, 도로 위 차처럼 도로 안에 빠진 부분은 not_road 다각형으로 따로 내보낸다.
    """
    m = (mask == MASK_ROAD).astype(np.uint8)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outers, holes = [], []
    for i, c in enumerate(cs):
        is_hole = hier is not None and hier[0][i][3] != -1
        if cv2.contourArea(c) < (min_hole if is_hole else min_area):
            continue
        c = cv2.approxPolyDP(c, eps_px, True)
        if len(c) >= 3:
            (holes if is_hole else outers).append([(float(p[0][0]), float(p[0][1])) for p in c])
    return outers, holes


def polygons_to_mask(polys: list, w: int, h: int, holes=()) -> np.ndarray:
    """도로 다각형을 칠한 뒤 not_road 다각형을 지운다."""
    m = np.full((h, w), MASK_BG, np.uint8)
    for group, val in ((polys, MASK_ROAD), (holes, MASK_BG)):
        for poly in group:
            pts = np.round(np.array(poly, np.float64)).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts], val)
    return m


def cvat_rle_to_mask(rle: str, left: int, top: int, width: int, height: int, w: int, h: int) -> np.ndarray:
    """CVAT for images 1.1 의 <mask rle=...> 를 전체 영상 마스크로.

    rle 는 상자(left, top, width, height) 안을 행 순서로 훑은 길이 목록이고 0(빈칸)부터 번갈아 나온다고 보았다.
    CVAT 소스로 확인 필요. 브러시 대신 다각형 도구로 고치면 이 함수를 쓰지 않는다.
    """
    counts = [int(x) for x in rle.replace(" ", "").split(",") if x != ""]
    flat = np.zeros(width * height, np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    full = np.full((h, w), MASK_BG, np.uint8)
    sub = flat[: width * height].reshape(height, width)
    full[top:top + height, left:left + width][sub > 0] = MASK_ROAD
    return full


# ---------------------------------------------------------------------------
# 내보내기
# ---------------------------------------------------------------------------
def _frame_sources(cfg: dict, test_set: bool) -> tuple[str, str, str]:
    """(이미지 폴더, 자동 라벨 폴더, 검수 폴더)."""
    paths = RoundPaths(cfg)
    primary = cfg["autolabel"]["primary"]
    if test_set:
        return (os.path.join(paths.test_dir, "images"), paths.auto_dir(primary, paths.test_dir),
                os.path.join(paths.test_dir, "review"))
    return paths.images, paths.auto_dir(primary), paths.review


def export_for_cvat(cfg: dict, test_set: bool = False, prefill: bool = True, fmt: str | None = None) -> dict:
    """검수 대상 프레임의 이미지 zip 과 자동 라벨 zip 을 만든다.

    CVAT 순서: 작업 만들기에서 review_images.zip 을 올린 뒤, 작업 메뉴의 Upload annotations 에서
    cvat_upload.zip 을 형식(CVAT for images 1.1 또는 YOLO 1.1)을 골라 올린다.
    prefill=False 면 빈 라벨(사람이 처음부터 칠함)로 내보낸다.
    """
    fmt = fmt or cfg["review"]["format"]
    img_dir, auto_dir, review_dir = _frame_sources(cfg, test_set)
    rows = read_csv(os.path.join(review_dir, "review_list.csv")) or make_review_list(cfg, test_set)
    ids = [r["frame_id"] for r in rows]
    os.makedirs(review_dir, exist_ok=True)
    images_zip = os.path.join(review_dir, "review_images.zip")
    ann_zip = os.path.join(review_dir, "cvat_upload.zip")
    items = []
    with zipfile.ZipFile(images_zip, "w", zipfile.ZIP_STORED) as zi:
        for fid in ids:
            p = os.path.join(img_dir, fid + ".jpg")
            if not os.path.exists(p):
                continue
            zi.write(p, fid + ".jpg")
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            boxes = read_yolo(os.path.join(auto_dir, "labels", fid + ".txt"), w, h) if prefill else []
            mp = os.path.join(auto_dir, "masks", fid + ".png")
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if (prefill and os.path.exists(mp)) else None
            items.append((fid, w, h, boxes, mask))
    if fmt == "yolo":
        _write_yolo11_zip(ann_zip, items)
    else:
        _write_cvat_xml_zip(ann_zip, items, cfg["review"].get("road_label", "road"))
    return {"frames": len(items), "images_zip": images_zip, "annotations_zip": ann_zip, "format": fmt}


def _write_yolo11_zip(path: str, items: list) -> None:
    """CVAT YOLO 1.1 형식: obj.names, obj.data, train.txt, obj_train_data/<이름>.txt (번호는 CVAT_LABELS 순서)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CVAT_LABELS) + "\n")
        z.writestr("obj.data", f"classes = {len(CVAT_LABELS)}\ntrain = data/train.txt\nnames = data/obj.names\n"
                               "backup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{fid}.jpg\n" for fid, *_ in items))
        for fid, w, h, boxes, _m in items:
            lines = []
            for b in boxes:
                if b.cls not in KEEP_CLASSES:
                    continue
                k = CVAT_LABELS.index(KEEP_CLASSES[b.cls])
                lines.append(f"{k} {(b.x1 + b.x2) / 2 / w:.6f} {(b.y1 + b.y2) / 2 / h:.6f} "
                             f"{(b.x2 - b.x1) / w:.6f} {(b.y2 - b.y1) / h:.6f}")
            z.writestr(f"obj_train_data/{fid}.txt", "\n".join(lines) + ("\n" if lines else ""))


def _write_cvat_xml_zip(path: str, items: list, road_label: str) -> None:
    """CVAT for images 1.1 형식: annotations.xml 하나. 박스는 <box>, 도로는 <polygon>."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(ET.SubElement(root, "meta"), "task")
    labels = ET.SubElement(meta, "labels")
    for name in CVAT_LABELS + [road_label, HOLE_LABEL]:
        lab = ET.SubElement(labels, "label")
        ET.SubElement(lab, "name").text = name
        ET.SubElement(lab, "type").text = "polygon" if name in (road_label, HOLE_LABEL) else "rectangle"
    for i, (fid, w, h, boxes, mask) in enumerate(items):
        im = ET.SubElement(root, "image", id=str(i), name=fid + ".jpg", width=str(w), height=str(h))
        for b in boxes:
            if b.cls not in KEEP_CLASSES:
                continue
            ET.SubElement(im, "box", label=KEEP_CLASSES[b.cls], source="auto", occluded="0",
                          xtl=f"{b.x1:.2f}", ytl=f"{b.y1:.2f}", xbr=f"{b.x2:.2f}", ybr=f"{b.y2:.2f}", z_order="0")
        if mask is not None:
            outers, holes = mask_to_polygons(mask)
            for lab, z, group in ((road_label, "0", outers), (HOLE_LABEL, "1", holes)):
                for poly in group:
                    ET.SubElement(im, "polygon", label=lab, source="auto", occluded="0", z_order=z,
                                  points=";".join(f"{x:.2f},{y:.2f}" for x, y in poly))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations.xml", ET.tostring(root, encoding="unicode"))


# ---------------------------------------------------------------------------
# 들여오기: CVAT 에서 내보낸 zip(CVAT for images 1.1 또는 YOLO 1.1)
# ---------------------------------------------------------------------------
def read_cvat_zip(zip_path: str, sizes: dict, road_label: str = "road") -> dict:
    """{프레임 이름: (박스 목록, 도로 마스크 또는 None)}. sizes 는 {프레임 이름: (폭, 높이)}.

    zip 은 풀지 않고 필요한 파일만 이름으로 읽는다. 이미지 이름의 폴더 부분과 확장자는 떼어 낸다.
    """
    out: dict = {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        xml_name = next((n for n in names if os.path.basename(n) == "annotations.xml"), None)
        if xml_name:
            root = ET.fromstring(z.read(xml_name))
            for im in root.iter("image"):
                fid = os.path.splitext(os.path.basename(im.get("name", "")))[0]
                w, h = int(float(im.get("width", 0))), int(float(im.get("height", 0)))
                boxes, polys, holes, rle_masks = [], [], [], []
                for el in im:
                    lab = el.get("label", "")
                    if el.tag == "box" and lab in NAME_TO_COCO:
                        boxes.append(Box(NAME_TO_COCO[lab], float(el.get("xtl")), float(el.get("ytl")),
                                         float(el.get("xbr")), float(el.get("ybr"))))
                    elif el.tag == "polygon" and lab in (road_label, HOLE_LABEL):
                        pts = [tuple(map(float, p.split(","))) for p in el.get("points", "").split(";") if p]
                        (polys if lab == road_label else holes).append(pts)
                    elif el.tag == "mask" and lab == road_label:
                        rle_masks.append(cvat_rle_to_mask(el.get("rle", ""), int(float(el.get("left", 0))),
                                                          int(float(el.get("top", 0))), int(float(el.get("width", 0))),
                                                          int(float(el.get("height", 0))), w, h))
                mask = None
                if polys or rle_masks:
                    mask = polygons_to_mask(polys, w, h)
                    for m in rle_masks:
                        mask[m == MASK_ROAD] = MASK_ROAD
                    for hp in holes:
                        cv2.fillPoly(mask, [np.round(np.array(hp)).astype(np.int32).reshape(-1, 1, 2)], MASK_BG)
                out[fid] = (boxes, mask)
            return out
        # YOLO 1.1
        names_file = next((n for n in names if os.path.basename(n) == "obj.names"), None)
        if names_file is None:
            raise ValueError(f"CVAT for images 1.1 도 YOLO 1.1 도 아닙니다: {zip_path}")
        labels = [x.strip() for x in z.read(names_file).decode("utf-8").splitlines() if x.strip()]
        for n in names:
            if not n.endswith(".txt") or not os.path.basename(os.path.dirname(n)).startswith("obj_"):
                continue
            fid = os.path.splitext(os.path.basename(n))[0]
            if fid not in sizes:
                continue
            w, h = sizes[fid]
            boxes = []
            for line in z.read(n).decode("utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                name = labels[int(p[0])]
                if name not in NAME_TO_COCO:
                    continue
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                boxes.append(Box(NAME_TO_COCO[name], (cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h))
            out[fid] = (boxes, None)
    return out


def import_from_cvat(cfg: dict, zip_path: str, test_set: bool = False) -> dict:
    """사람 라벨을 review/reviewed/labels, masks 에 쓴다(자동 라벨 폴더는 건드리지 않음).

    CVAT 는 작업의 모든 이미지를 내보내므로 박스가 없는 이미지도 "물체 없음"이라는 사람 판단으로 본다.
    도로 다각형이 없는 이미지는 도로를 검수하지 않은 것으로 보고 마스크를 쓰지 않는다(자동 마스크 유지).
    """
    paths = RoundPaths(cfg)
    img_dir, _auto, review_dir = _frame_sources(cfg, test_set)
    out_dir = os.path.join(review_dir, "reviewed")
    sizes = {}
    for n in os.listdir(img_dir):
        fid, ext = os.path.splitext(n)
        if ext.lower() == ".jpg":
            img = cv2.imread(os.path.join(img_dir, n), cv2.IMREAD_GRAYSCALE)
            sizes[fid] = (img.shape[1], img.shape[0])
    data = read_cvat_zip(zip_path, sizes, cfg["review"].get("road_label", "road"))
    n_box = n_mask = 0
    unknown = []
    for fid, (boxes, mask) in data.items():
        if fid not in sizes:
            unknown.append(fid)
            continue
        w, h = sizes[fid]
        write_yolo(os.path.join(out_dir, "labels", fid + ".txt"), boxes, w, h)
        n_box += len(boxes)
        if mask is not None:
            os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "masks", fid + ".png"), mask)
            n_mask += 1
    if test_set:
        # 시험 세트: 사람 라벨을 test_set/labels 로 옮기고 목록의 라벨 출처를 human 으로 바꾼다
        rows = read_csv(paths.test_csv)
        for r in rows:
            if r["frame_id"] in data and r["frame_id"] in sizes:
                r["label_source"] = "human"
        from training.select_frames import TEST_FIELDS

        write_csv(paths.test_csv, rows, TEST_FIELDS)
        os.makedirs(os.path.join(paths.test_dir, "labels"), exist_ok=True)
        for fid in data:
            src = os.path.join(out_dir, "labels", fid + ".txt")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(paths.test_dir, "labels", fid + ".txt"))
    info = {"frames": len(data) - len(unknown), "boxes": n_box, "masks": n_mask, "unknown_frames": len(unknown)}
    if not test_set:
        update_meta(paths, "review_import", info)
    return info


# ---------------------------------------------------------------------------
# 합치기: 최종 라벨 = 사람 라벨이 있으면 그것, 없으면 자동 라벨
# ---------------------------------------------------------------------------
def merge_round(cfg: dict) -> dict:
    paths = RoundPaths(cfg)
    auto = paths.auto_dir(cfg["autolabel"]["primary"])
    rev = paths.reviewed
    os.makedirs(paths.labels, exist_ok=True)
    os.makedirs(paths.masks, exist_ok=True)
    rows = []
    counts = {"reviewed": 0, "auto": 0, "missing": 0}
    for r in read_csv(paths.frames_csv):
        fid = r["frame_id"]
        src = None
        for name, d in (("reviewed", rev), ("auto", auto)):
            p = os.path.join(d, "labels", fid + ".txt")
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(paths.labels, fid + ".txt"))
                src = name
                break
        msrc = None
        for name, d in (("reviewed", rev), ("auto", auto)):
            p = os.path.join(d, "masks", fid + ".png")
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(paths.masks, fid + ".png"))
                msrc = name
                break
        counts[src or "missing"] += 1
        rows.append({"frame_id": fid, "label_source": src or "missing", "mask_source": msrc or "missing"})
    write_csv(paths.sources_csv, rows, ["frame_id", "label_source", "mask_source"])
    update_meta(paths, "merge", counts)
    return counts


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CVAT 내보내기와 들여오기")
    ap.add_argument("action", choices=["review-list", "export", "import", "merge"])
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--zip", default=None, help="import 할 CVAT 내보내기 zip")
    ap.add_argument("--test-set", action="store_true", help="고정 시험 세트를 대상으로")
    ap.add_argument("--format", choices=["cvat", "yolo"], default=None)
    ap.add_argument("--empty", action="store_true", help="자동 라벨 없이 빈 작업으로 내보냄")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.action == "review-list":
        rows = make_review_list(cfg, args.test_set)
        print(f"검수 대상 {len(rows)}장")
    elif args.action == "export":
        make_review_list(cfg, args.test_set)
        info = export_for_cvat(cfg, args.test_set, prefill=not args.empty, fmt=args.format)
        print(f"CVAT 내보내기 {info['frames']}장: 이미지 {info['images_zip']}, 라벨 {info['annotations_zip']}")
    elif args.action == "import":
        if not args.zip:
            ap.error("--zip 이 필요합니다")
        info = import_from_cvat(cfg, args.zip, args.test_set)
        print(f"사람 라벨 {info['frames']}장, 박스 {info['boxes']}개, 도로 마스크 {info['masks']}장 "
              f"(모르는 이름 {info['unknown_frames']}장)")
    else:
        c = merge_round(cfg)
        print(f"최종 라벨: 사람 {c['reviewed']}장, 자동 {c['auto']}장, 없음 {c['missing']}장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
