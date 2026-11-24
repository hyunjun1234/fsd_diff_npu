#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

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
    list_images,
    load_config,
    match_boxes,
    read_csv,
    resolve_path,
    update_meta,
    write_csv,
    write_json,
    write_yolo,
)

# 선별한 프레임에 자동 라벨을 붙인다. 주 라벨러(Grounding DINO + SAM 2) 결과가 학습 라벨이 되고,
# 비교 라벨러(지금 쓰는 YOLOv8s) 결과와 어긋나거나 점수가 애매한 프레임을 의심 프레임으로 표시한다.
#
# 예) 학습 서버(WSL2, GPU)에서
#   python3 -m training.autolabel --config training/configs/round.yaml
#   python3 -m training.autolabel --config training/configs/round.yaml --test-set   # 시험 세트 미리 채우기용
#   python3 -m training.autolabel --track-dir out/triggers/clip01/front --out out/training/track/clip01
#
# 결과: out/training/datasets/<회차>/auto/<라벨러>/labels/*.txt (YOLO, COCO 번호)
#                                              /masks/*.png (0 도로 아님, 1 도로)
#                                              /scores/*.json (박스별 점수)
#       out/training/datasets/<회차>/auto/suspects.csv (의심 프레임과 이유)


# ---------------------------------------------------------------------------
# 라벨러 결과와 인터페이스
# ---------------------------------------------------------------------------
@dataclass
class LabelResult:
    boxes: list = field(default_factory=list)  # Box 목록(문턱 전 전부, 점수 포함)
    road: np.ndarray | None = None  # HxW uint8, MASK_BG / MASK_ROAD. 없으면 None
    object_masks: list | None = None  # 박스마다 HxW bool(SAM 2 박스 프롬프트, 선택)


class Labeler:
    """라벨러 공통 모양. label() 은 BGR 영상과 파일 경로를 받아 LabelResult 를 돌려준다."""

    name = "base"

    def label(self, img: np.ndarray, path: str | None = None) -> LabelResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 외부 모델 호출 (공식 저장소 API 이름을 따랐다. 바뀌면 이 구획만 고친다)
# ---------------------------------------------------------------------------
# Grounding DINO: github.com/IDEA-Research/GroundingDINO 의 groundingdino.util.inference
#   load_model(config, weights, device), load_image(path) -> (RGB 배열, 텐서),
#   predict(model, image, caption, box_threshold, text_threshold, device) -> (박스 cxcywh 0~1, 점수, 문구)
#   위 이름과 인자는 저장소 README 예제 기준이며, 설치할 커밋에서 같은지 확인 필요.
# SAM 2: github.com/facebookresearch/sam2
#   build_sam2(cfg, ckpt, device), SAM2ImagePredictor(model).set_image(RGB),
#   .predict(point_coords, point_labels, box, multimask_output) -> (masks, scores, logits)
#   build_sam2_video_predictor(cfg, ckpt, device), .init_state(video_path=폴더),
#   .add_new_points_or_box(inference_state, frame_idx, obj_id, box), .propagate_in_video(state)
#   README 기준. 상자 여러 개를 한 번에 줄 때 masks 모양(N,1,H,W)은 확인 필요.
class OfficialApi:
    def __init__(self, cfg: dict):
        self.cfg = cfg["autolabel"]
        self._gd = None
        self._sam = None
        self._sam_video = None

    # ---- Grounding DINO ----
    def gdino_predict(self, path: str, caption: str) -> tuple[list, list, list]:
        """이미지 파일 하나에 문장 프롬프트를 주고 (픽셀 xyxy 박스 목록, 점수 목록, 문구 목록)을 돌려준다."""
        from groundingdino.util.inference import load_image, load_model, predict  # 확인 필요

        g = self.cfg["grounding_dino"]
        if self._gd is None:
            self._gd = load_model(resolve_path(g["config"]), resolve_path(g["weights"]), device=g.get("device", "cuda"))
        image_source, image = load_image(path)
        boxes, logits, phrases = predict(
            model=self._gd, image=image, caption=caption,
            box_threshold=float(g["box_threshold"]), text_threshold=float(g["text_threshold"]),
            device=g.get("device", "cuda"),
        )
        h, w = image_source.shape[:2]
        out = []
        for cx, cy, bw, bh in boxes.cpu().numpy().tolist():  # 0~1 중심, 폭, 높이
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        return out, [float(x) for x in logits.cpu().numpy().tolist()], list(phrases)

    # ---- SAM 2 이미지 ----
    def _sam_image(self):
        if self._sam is None:
            from sam2.build_sam import build_sam2  # 확인 필요
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # 확인 필요

            s = self.cfg["sam2"]
            self._sam = SAM2ImagePredictor(build_sam2(s["config"], resolve_path(s["checkpoint"]), device=s.get("device", "cuda")))
        return self._sam

    def sam2_points(self, rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
        """양성 점 여러 개로 마스크 하나(HxW bool). 후보 3개 중 점수가 가장 높은 것."""
        import torch

        p = self._sam_image()
        with torch.inference_mode():
            p.set_image(rgb)
            masks, scores, _ = p.predict(point_coords=points.astype(np.float32),
                                         point_labels=np.ones(len(points), np.int32), multimask_output=True)
        return np.asarray(masks[int(np.argmax(scores))]) > 0

    def sam2_boxes(self, rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list:
        """박스마다 마스크 하나(HxW bool) 목록."""
        import torch

        if len(boxes_xyxy) == 0:
            return []
        p = self._sam_image()
        with torch.inference_mode():
            p.set_image(rgb)
            masks, _, _ = p.predict(point_coords=None, point_labels=None, box=boxes_xyxy.astype(np.float32),
                                    multimask_output=False)
        masks = np.asarray(masks)
        if masks.ndim == 4:  # (N, 1, H, W)
            masks = masks[:, 0]
        elif masks.ndim == 3 and len(boxes_xyxy) == 1:  # (1, H, W)
            pass
        return [m > 0 for m in masks]

    # ---- SAM 2 영상 (Grounded-SAM-2 방식 추적) ----
    def sam2_track(self, frames_dir: str, first_boxes: np.ndarray) -> dict:
        """frames_dir(00000.jpg 처럼 숫자 이름 JPEG)에 첫 프레임 박스를 주고 끝까지 마스크를 잇는다.

        돌려주는 값: {프레임 번호: {물체 번호: HxW bool}}
        """
        import torch
        from sam2.build_sam import build_sam2_video_predictor  # 확인 필요

        s = self.cfg["sam2"]
        if self._sam_video is None:
            self._sam_video = build_sam2_video_predictor(s["config"], resolve_path(s["checkpoint"]),
                                                         device=s.get("device", "cuda"))
        pred = self._sam_video
        out: dict = {}
        with torch.inference_mode():
            state = pred.init_state(video_path=frames_dir)
            for k, box in enumerate(first_boxes):
                pred.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=k, box=np.asarray(box, np.float32))
            for f_idx, obj_ids, mask_logits in pred.propagate_in_video(state):
                out[int(f_idx)] = {int(o): (mask_logits[i] > 0.0).cpu().numpy().squeeze() for i, o in enumerate(obj_ids)}
        return out


# ---------------------------------------------------------------------------
# 실제 라벨러
# ---------------------------------------------------------------------------
def phrase_to_coco(phrase: str) -> int | None:
    """Grounding DINO 문구("car", "a truck" 등)를 COCO 번호로. 모르는 문구는 None."""
    words = phrase.lower().replace(".", " ").split()
    for name in ("motorcycle", "bicycle", "person", "truck", "bus", "car"):
        if name in words:
            return NAME_TO_COCO[name]
    return None


class GroundingSamLabeler(Labeler):
    """Grounding DINO 박스 + SAM 2 도로 마스크(+ 선택: 박스별 물체 마스크)."""

    name = "grounding_dino"

    def __init__(self, cfg: dict, api: OfficialApi | None = None):
        self.cfg = cfg["autolabel"]
        self.api = api or OfficialApi(cfg)

    def label(self, img: np.ndarray, path: str | None = None) -> LabelResult:
        if path is None:
            raise ValueError("Grounding DINO 는 파일 경로로 읽습니다.")
        g = self.cfg["grounding_dino"]
        xyxy, scores, phrases = self.api.gdino_predict(path, g["caption"])
        boxes = []
        for bb, s, ph in zip(xyxy, scores, phrases):
            c = phrase_to_coco(ph)
            if c is not None:
                boxes.append(Box(c, *bb, score=s))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        road = None
        mode = self.cfg.get("road", "points")
        if mode == "points":
            pts = np.array([[x * w, y * h] for x, y in self.cfg["sam2"]["road_points"]], np.float32)
            road = self.api.sam2_points(rgb, pts)
        elif mode == "text":
            rb, rs, _ = self.api.gdino_predict(path, g.get("road_caption", "road ."))
            if rb:
                road = self.api.sam2_boxes(rgb, np.array([rb[int(np.argmax(rs))]]))[0]
        if road is not None:
            road = np.where(road, MASK_ROAD, MASK_BG).astype(np.uint8)
        obj = None
        if self.cfg["sam2"].get("object_masks"):
            obj = self.api.sam2_boxes(rgb, np.array([b.xyxy() for b in boxes]).reshape(-1, 4))
        return LabelResult(boxes=boxes, road=road, object_masks=obj)


class YoloLabeler(Labeler):
    """지금 파이에서 쓰는 YOLOv8s(COCO 사전학습)를 비교 라벨러로. Ultralytics 는 여기서만 불러온다."""

    name = "yolo"

    def __init__(self, cfg: dict):
        self.cfg = cfg["autolabel"]["yolo"]
        self._m = None

    def label(self, img: np.ndarray, path: str | None = None) -> LabelResult:
        if self._m is None:
            from ultralytics import YOLO

            self._m = YOLO(self.cfg["weights"])
        r = self._m.predict(source=img, conf=float(self.cfg["conf"]), imgsz=int(self.cfg.get("imgsz", 640)),
                            device=self.cfg.get("device", 0), verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        return LabelResult(boxes=[Box(int(c), *map(float, b), score=float(s))
                                  for b, s, c in zip(xyxy, conf, cls) if int(c) in KEEP_CLASSES])


def make_labeler(name: str | None, cfg: dict) -> Labeler | None:
    """설정 이름으로 라벨러를 만든다. fake 계열은 시험용(training/fakes.py)."""
    if not name:
        return None
    if name == "grounding_dino":
        return GroundingSamLabeler(cfg)
    if name == "yolo":
        return YoloLabeler(cfg)
    if name == "fake":
        from training.fakes import FakeColorLabeler

        return FakeColorLabeler("fake")
    if name == "fake_secondary":
        from training.fakes import FakeColorLabeler

        return FakeColorLabeler("fake_secondary", score=0.8, drop_classes=(0,), with_road=False)
    raise ValueError(f"모르는 라벨러: {name}")


# ---------------------------------------------------------------------------
# 저장과 의심 프레임 판정
# ---------------------------------------------------------------------------
def save_result(out_dir: str, fid: str, res: LabelResult, w: int, h: int, keep_score: float, min_px: float) -> list:
    """라벨 txt(문턱 넘은 박스), 점수 json(전부), 도로 마스크 png 를 쓴다. 남긴 박스 목록을 돌려준다."""
    kept = [b for b in res.boxes if b.score >= keep_score and (b.x2 - b.x1) >= min_px and (b.y2 - b.y1) >= min_px]
    write_yolo(os.path.join(out_dir, "labels", fid + ".txt"), kept, w, h)
    write_json(os.path.join(out_dir, "scores", fid + ".json"), {
        "width": w, "height": h,
        "boxes": [{"cls": b.cls, "name": KEEP_CLASSES.get(b.cls, str(b.cls)), "xyxy": [round(v, 2) for v in b.xyxy()],
                   "score": round(b.score, 4), "kept": b in kept} for b in res.boxes],
        "road_pixels": int((res.road == MASK_ROAD).sum()) if res.road is not None else None,
    })
    if res.road is not None:
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "masks", fid + ".png"), res.road.astype(np.uint8))
    if res.object_masks:
        # 물체 마스크는 한 장에 1부터 번호를 매겨 저장(0 은 배경). 순서는 scores json 의 boxes 와 같다.
        inst = np.zeros((h, w), np.uint16)
        for k, m in enumerate(res.object_masks, start=1):
            inst[np.asarray(m, bool)] = k
        os.makedirs(os.path.join(out_dir, "instances"), exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "instances", fid + ".png"), inst)
    return kept


def suspect_reasons(primary_all: list, primary_kept: list, secondary_kept: list | None, acfg: dict) -> dict:
    """의심 이유를 센다. 애매한 점수 박스 수, 두 라벨러에서 짝이 없는 박스 수."""
    lo, hi = acfg["ambiguous"]
    amb = sum(1 for b in primary_all if lo <= b.score < hi)
    miss_p = miss_s = 0
    if secondary_kept is not None:
        _, miss_p, miss_s = match_boxes(primary_kept, secondary_kept, float(acfg["match_iou"]))
        miss_p, miss_s = len(miss_p), len(miss_s)
    reasons = []
    if amb > 0:
        reasons.append(f"ambiguous:{amb}")
    if miss_p + miss_s >= int(acfg["disagree_min"]) and secondary_kept is not None:
        reasons.append(f"disagree:{miss_p}+{miss_s}")
    return {"ambiguous": amb, "only_primary": miss_p, "only_secondary": miss_s, "reasons": ";".join(reasons)}


def autolabel_dir(cfg: dict, image_dir: str, base_dir: str, primary: Labeler, secondary: Labeler | None,
                  suspects_csv: str | None = None, frame_ids: list | None = None) -> dict:
    """image_dir 의 이미지에 라벨을 붙여 base_dir/auto/<라벨러>/ 아래에 쓰고 의심 프레임 표를 만든다.

    frame_ids 를 주면 그 프레임만 본다(선별을 다시 돌려 폴더에 남은 옛 이미지는 건너뜀).
    """
    acfg = cfg["autolabel"]
    keep = float(acfg["keep_score"])
    min_px = float(acfg.get("min_box_px", 6))
    p_dir = os.path.join(base_dir, "auto", primary.name)
    s_dir = os.path.join(base_dir, "auto", secondary.name) if secondary else None
    rows = []
    n_boxes = 0
    files = list_images(image_dir)
    if frame_ids is not None:
        want = set(frame_ids)
        files = [p for p in files if os.path.splitext(os.path.basename(p))[0] in want]
    for path in files:
        fid = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        pr = primary.label(img, path)
        pk = save_result(p_dir, fid, pr, w, h, keep, min_px)
        n_boxes += len(pk)
        sk = None
        if secondary is not None:
            sr = secondary.label(img, path)
            # 비교 라벨러는 자기 문턱(YOLO conf)을 이미 썼으므로 크기 조건만 둔다
            sk = save_result(s_dir, fid, sr, w, h, 0.0, min_px)
        rows.append({"frame_id": fid, "n_primary": len(pk), "n_secondary": "" if sk is None else len(sk),
                     **suspect_reasons(pr.boxes, pk, sk, acfg)})
    out_csv = suspects_csv or os.path.join(base_dir, "auto", "suspects.csv")
    write_csv(out_csv, rows, ["frame_id", "n_primary", "n_secondary", "ambiguous", "only_primary",
                              "only_secondary", "reasons"])
    n_sus = sum(1 for r in rows if r["reasons"])
    return {"frames": len(rows), "boxes": n_boxes, "suspects": n_sus, "primary": primary.name,
            "secondary": secondary.name if secondary else None}


def autolabel_round(cfg: dict, primary: Labeler | None = None, secondary: Labeler | None = None,
                    test_set: bool = False) -> dict:
    paths = RoundPaths(cfg)
    primary = primary or make_labeler(cfg["autolabel"]["primary"], cfg)
    if secondary is None:
        secondary = make_labeler(cfg["autolabel"].get("secondary"), cfg)
    if test_set:
        ids = [r["frame_id"] for r in read_csv(paths.test_csv)]
        return autolabel_dir(cfg, os.path.join(paths.test_dir, "images"), paths.test_dir, primary, secondary,
                             frame_ids=ids)
    ids = [r["frame_id"] for r in read_csv(paths.frames_csv)]
    info = autolabel_dir(cfg, paths.images, paths.rd, primary, secondary, paths.suspects_csv, frame_ids=ids)
    update_meta(paths, "autolabel", info)
    return info


# ---------------------------------------------------------------------------
# 영상 추적 라벨 (Grounded-SAM-2 방식): 첫 프레임 박스를 SAM 2 로 끝까지 잇는다
# ---------------------------------------------------------------------------
def mask_to_box(m: np.ndarray) -> tuple | None:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def track_clip(cfg: dict, frames_dir: str, out_dir: str, api: OfficialApi | None = None,
               first_boxes: list | None = None) -> dict:
    """트리거 클립처럼 촘촘한 프레임 폴더를 추적 라벨로 채운다.

    frames_dir 는 SAM 2 영상 예측기 규칙대로 00000.jpg 처럼 숫자 이름 JPEG 여야 한다.
    first_boxes 가 없으면 첫 프레임에 Grounding DINO 를 돌려 시작 박스를 얻는다.
    결과: out_dir/labels/<프레임>.txt, out_dir/scores/<프레임>.json
    """
    api = api or OfficialApi(cfg)
    files = list_images(frames_dir)
    if not files:
        return {"frames": 0}
    if first_boxes is None:
        lab = GroundingSamLabeler({**cfg, "autolabel": {**cfg["autolabel"], "road": "none"}}, api=api)
        img0 = cv2.imread(files[0], cv2.IMREAD_COLOR)
        keep = float(cfg["autolabel"]["keep_score"])
        first_boxes = [b for b in lab.label(img0, files[0]).boxes if b.score >= keep]
    tracks = api.sam2_track(frames_dir, np.array([b.xyxy() for b in first_boxes]).reshape(-1, 4))
    n = 0
    for f_idx, path in enumerate(files):
        h, w = cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape[:2]
        boxes = []
        for k, m in (tracks.get(f_idx) or {}).items():
            bb = mask_to_box(m)
            if bb is not None and k < len(first_boxes):
                boxes.append(Box(first_boxes[k].cls, *bb, score=first_boxes[k].score))
        fid = os.path.splitext(os.path.basename(path))[0]
        write_yolo(os.path.join(out_dir, "labels", fid + ".txt"), boxes, w, h)
        write_json(os.path.join(out_dir, "scores", fid + ".json"),
                   {"width": w, "height": h, "boxes": [{"cls": b.cls, "track": i, "xyxy": list(b.xyxy()),
                                                         "score": b.score} for i, b in enumerate(boxes)]})
        n += 1
    return {"frames": n, "objects": len(first_boxes)}


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="자동 라벨링")
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--primary", default=None, help="설정의 autolabel.primary 대신 쓸 라벨러")
    ap.add_argument("--secondary", default=None, help="설정의 autolabel.secondary 대신(none 이면 비교 안 함)")
    ap.add_argument("--test-set", action="store_true", help="고정 시험 세트 이미지에 붙임(사람 검수 전 채우기용)")
    ap.add_argument("--track-dir", default=None, help="이 프레임 폴더를 영상 추적으로 라벨링")
    ap.add_argument("--out", default=None, help="--track-dir 결과 폴더")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.primary:
        cfg["autolabel"]["primary"] = args.primary
    if args.secondary:
        cfg["autolabel"]["secondary"] = None if args.secondary == "none" else args.secondary
    if args.track_dir:
        out = args.out or os.path.join(RoundPaths(cfg).rd, "track", os.path.basename(os.path.normpath(args.track_dir)))
        info = track_clip(cfg, args.track_dir, out)
        print(f"추적 라벨 {info}: {out}")
        return 0
    info = autolabel_round(cfg, test_set=args.test_set)
    print(f"자동 라벨: 프레임 {info['frames']}장, 박스 {info['boxes']}개, 의심 프레임 {info['suspects']}장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
