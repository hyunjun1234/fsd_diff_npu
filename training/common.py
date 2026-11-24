from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import yaml

# 저장소 맨 위 폴더. 설정의 상대 경로는 모두 이 폴더를 기준으로 푼다(어느 폴더에서 실행해도 같게).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 클래스 번호 규칙
# ---------------------------------------------------------------------------
# 파이의 검출 후처리(realityview/types.py 의 COCO_KEEP)는 COCO 80 클래스 번호를 그대로 읽는다.
# 재학습 모델도 80 클래스 머리를 유지해야 HEF, dxnn 을 바꿔 끼우기만 하면 되므로
# 라벨 txt 의 클래스 번호도 COCO 번호(사람 0, 자전거 1, 차 2, 오토바이 3, 버스 5, 트럭 7)를 쓴다.
# realityview 는 다른 작업이 고치는 중이라 import 하지 않고 값을 복사해 둔다(같은 값인지 테스트가 확인).
KEEP_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
NAME_TO_COCO = {v: k for k, v in KEEP_CLASSES.items()}

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)
assert len(COCO_NAMES) == 80

# 도로 마스크 png 의 화소 값(자동 라벨, 검수 결과 모두 같은 규칙)
MASK_BG = 0  # 도로 아님
MASK_ROAD = 1  # 도로
MASK_IGNORE = 255  # 모름(학습에서 빼는 화소)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# 설정 기본값
# ---------------------------------------------------------------------------
# training/configs/round.yaml 에 없는 항목은 이 값을 쓴다. 항목 설명은 round.yaml 주석에 있다.
DEFAULTS: dict = {
    "round": "r001",
    "prev_round": None,
    "paths": {
        "drives_root": "out/drives",
        "triggers_root": "out/triggers",
        "work_root": "out/training",
        "models_root": "models",
        "coco_root": None,
    },
    "drives": ["*"],
    "recording": {
        "video_exts": [".mp4", ".mkv", ".avi", ".h264", ".mjpeg", ".mjpg"],
        "image_pattern": r"^(?P<camera>.+?)_(?P<idx>\d{5,})$",
        "timestamp_csv": ["{camera}.csv", "{camera}_ts.csv", "{camera}_timestamps.csv", "timestamps.csv"],
        "frame_columns": ["frame", "frame_idx", "idx", "index"],
        "time_columns": ["sensor_ns", "t_sensor_ns", "sensor_ts_ns", "ts_ns", "t_sensor", "t_mono", "t"],
        "camera_column": "camera",
        "fps_fallback": 30.0,
    },
    "triggers": {
        "enabled": True,
        "json_glob": "**/*.json",
        "kind_keys": ["kind", "type", "trigger"],
        "score_keys": ["score"],
        "events_key": "events",
        "media_keys": ["video", "file", "frames_dir", "path", "dir"],
        "camera_keys": ["camera", "cam"],
        "every_sec": 0.5,
    },
    "select": {
        "every_sec": 1.0,
        "blur_min": 60.0,
        "blur_min_night": 15.0,
        "blur_width": 640,
        "dup_diff": 2.5,
        "dup_window": 20,
        "night_brightness": 60.0,
        "scene_tags_csv": None,
        "n_frames": 2000,
        "trigger_max_ratio": 0.3,
        "seed": 0,
        "jpeg_quality": 95,
    },
    "test_set": {
        "n_frames": 200,
        "exclude": "segment",
        "segment_ratio": 0.15,
        "window_sec": 10.0,
        "include_triggers": False,
    },
    "autolabel": {
        "primary": "grounding_dino",
        "secondary": "yolo",
        "road": "points",
        "keep_score": 0.35,
        "ambiguous": [0.3, 0.5],
        "match_iou": 0.5,
        "disagree_min": 1,
        "min_box_px": 6,
        "grounding_dino": {
            "config": "models/gdino/GroundingDINO_SwinT_OGC.py",
            "weights": "models/gdino/groundingdino_swint_ogc.pth",
            "caption": "car . person . truck . bus . bicycle . motorcycle .",
            "road_caption": "road .",
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "device": "cuda",
        },
        "sam2": {
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "models/sam2/sam2.1_hiera_large.pt",
            "device": "cuda",
            "road_points": [[0.5, 0.93], [0.3, 0.97], [0.7, 0.97]],
            "object_masks": False,
        },
        "yolo": {"weights": "yolov8s.pt", "conf": 0.25, "imgsz": 640, "device": 0},
    },
    "review": {"format": "cvat", "random_ratio": 0.1, "seed": 0, "road_label": "road"},
    "train_det": {
        "base_weights": "yolov8s.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 50,
        "patience": 20,
        "workers": 8,
        "cache": False,
        "device": 0,
        "val_ratio": 0.1,
        "prev_ratio": 0.5,
        "coco_ratio": 0.5,
        "coco_split": "train2017",
        "seed": 0,
    },
    "train_seg": {
        "label_mode": "road_first",
        "val_ratio": 0.1,
        "mmseg_root": "~/mmsegmentation",
        "base_config": "configs/bisenetv2/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py",
        "load_from": None,
        "crop": [512, 1024],
        "batch": 8,
        "iters": 20000,
    },
    "evaluate": {
        "map_tolerance": 0.01,
        "recall_tolerance": 0.03,
        "recall_conf": 0.4,
        "match_iou": 0.5,
        "min_gt": 10,
        "imgsz": 640,
        "measure_dir": "docs/measurements",
    },
    "export": {
        "model_stem": "yolov8s",
        "imgsz": 640,
        "opset": 11,
        "calib_n": 1024,
        "hailo_arch": "hailo8",
        "hailo_zoo_name": "yolov8s",
        "dx_calib_n": 100,
        "dx_ppu": False,
        "pi_target": "<PI_USER>@<PI_HOST>:<PI_REPO>/models/",
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    """base 를 복사해 over 의 값으로 덮는다. 둘 다 dict 인 항목은 안쪽까지 내려가며 합친다."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p: str | None) -> str | None:
    """설정의 경로를 절대 경로로. ~ 는 홈으로, 상대 경로는 저장소 맨 위 폴더 기준."""
    if p is None or p == "":
        return None
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """회차 설정 yaml 을 읽어 기본값과 합친다. path 가 None 이면 기본값만 쓴다."""
    user = {}
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    check_round_name(cfg["round"])
    if cfg.get("prev_round") is None:
        cfg["prev_round"] = prev_round_name(cfg["round"])
    return cfg


# ---------------------------------------------------------------------------
# 회차 이름 규칙: r + 숫자 3자리 (r000 은 처음 받은 COCO 사전학습 모델)
# ---------------------------------------------------------------------------
ROUND_RE = re.compile(r"^r(\d{3})$")


def check_round_name(name: str) -> int:
    m = ROUND_RE.match(str(name))
    if not m:
        raise ValueError(f"회차 이름은 r001 처럼 r 과 숫자 3자리여야 합니다: {name!r}")
    return int(m.group(1))


def round_name(i: int) -> str:
    return f"r{i:03d}"


def prev_round_name(name: str) -> str:
    """r001 의 직전은 r000(사전학습 모델), r005 의 직전은 r004."""
    i = check_round_name(name)
    return round_name(max(i - 1, 0))


# ---------------------------------------------------------------------------
# 폴더 배치
# ---------------------------------------------------------------------------
@dataclass
class RoundPaths:
    """한 회차가 읽고 쓰는 경로를 한곳에 모은다. 폴더 구조는 docs/training_pipeline.md 2장과 같다."""

    cfg: dict

    @property
    def round(self) -> str:
        return self.cfg["round"]

    # ---- 입력 ----
    @property
    def drives_root(self) -> str:
        return resolve_path(self.cfg["paths"]["drives_root"])

    @property
    def triggers_root(self) -> str:
        return resolve_path(self.cfg["paths"]["triggers_root"])

    @property
    def coco_root(self) -> str | None:
        return resolve_path(self.cfg["paths"].get("coco_root"))

    # ---- 작업 폴더 ----
    @property
    def work(self) -> str:
        return resolve_path(self.cfg["paths"]["work_root"])

    @property
    def datasets(self) -> str:
        return os.path.join(self.work, "datasets")

    def round_dir(self, name: str | None = None) -> str:
        return os.path.join(self.datasets, name or self.round)

    @property
    def rd(self) -> str:
        return self.round_dir()

    @property
    def images(self) -> str:  # 선별한 프레임. Ultralytics 가 images 를 labels 로 바꿔 라벨을 찾는다
        return os.path.join(self.rd, "images")

    @property
    def labels(self) -> str:  # 학습에 쓰는 최종 라벨(검수 결과 우선, 없으면 자동 라벨)
        return os.path.join(self.rd, "labels")

    @property
    def masks(self) -> str:  # 학습에 쓰는 최종 도로 마스크
        return os.path.join(self.rd, "masks")

    @property
    def frames_csv(self) -> str:
        return os.path.join(self.rd, "frames.csv")

    @property
    def meta_json(self) -> str:
        return os.path.join(self.rd, "meta.json")

    def auto_dir(self, labeler: str, base: str | None = None) -> str:
        return os.path.join(base or self.rd, "auto", labeler)

    @property
    def suspects_csv(self) -> str:
        return os.path.join(self.rd, "auto", "suspects.csv")

    @property
    def review(self) -> str:
        return os.path.join(self.rd, "review")

    @property
    def reviewed(self) -> str:  # CVAT 에서 돌아온 사람 라벨(자동 라벨과 섞지 않고 따로 둔다)
        return os.path.join(self.review, "reviewed")

    @property
    def sources_csv(self) -> str:
        return os.path.join(self.rd, "label_sources.csv")

    # ---- 고정 시험 세트 (회차와 상관없이 하나) ----
    @property
    def test_dir(self) -> str:
        return os.path.join(self.work, "test_set")

    @property
    def test_csv(self) -> str:
        return os.path.join(self.test_dir, "test_set.csv")

    @property
    def test_lock(self) -> str:
        return os.path.join(self.test_dir, "lock.json")

    # ---- 모델 ----
    @property
    def models(self) -> str:
        return resolve_path(self.cfg["paths"]["models_root"])

    def model_dir(self, name: str | None = None) -> str:
        return os.path.join(self.models, name or self.round)

    def best_pt(self, name: str | None = None) -> str:
        return os.path.join(self.model_dir(name), "best.pt")

    def npu_file(self, ext: str, name: str | None = None) -> str:
        """회차 모델 파일 이름 규칙: models/yolov8s_r001.hef, models/yolov8s_r001.dxnn, models/yolov8s_r001.onnx"""
        stem = self.cfg["export"]["model_stem"]
        return os.path.join(self.models, f"{stem}_{name or self.round}.{ext}")


# ---------------------------------------------------------------------------
# 프레임 이름 규칙
# ---------------------------------------------------------------------------
# 이미지, 라벨 txt, 마스크 png, 점수 json, CVAT 이미지 이름, 시험 세트 목록이 모두 같은 줄기 이름을 쓴다.
#   주행:   <주행>__<세그먼트>__<카메라>__<프레임 6자리>      예) d20261201a__seg0003__front__000450
#   트리거: trig__<클립>__<카메라>__<프레임 6자리>
# 구분자 "__" 는 이름 안에 나오지 않게 sanitize 가 바꾼다.
_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def sanitize(s: str) -> str:
    s = _SAFE_RE.sub("-", str(s)).strip("-")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "x"


def make_frame_id(drive: str, segment: str, camera: str, idx: int) -> str:
    return f"{sanitize(drive)}__{sanitize(segment)}__{sanitize(camera)}__{int(idx):06d}"


def parse_frame_id(fid: str) -> dict | None:
    """프레임 이름을 주행, 세그먼트, 카메라, 번호로 나눈다. 규칙에 안 맞으면(예: COCO 이미지) None."""
    parts = os.path.splitext(os.path.basename(fid))[0].split("__")
    if len(parts) != 4 or not parts[3].isdigit():
        return None
    return {"drive": parts[0], "segment": parts[1], "camera": parts[2], "idx": int(parts[3])}


def segment_key(drive: str, segment: str) -> str:
    """시험 세트 잠금과 학습, 검증 나누기에 쓰는 세그먼트 열쇠."""
    return f"{sanitize(drive)}/{sanitize(segment)}"


# ---------------------------------------------------------------------------
# 박스와 YOLO txt
# ---------------------------------------------------------------------------
@dataclass
class Box:
    """한 물체 박스. 좌표는 원본 이미지 픽셀(x1, y1, x2, y2), cls 는 COCO 번호."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    def xyxy(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


def write_yolo(path: str, boxes: Sequence[Box], w: int, h: int, with_score: bool = False) -> None:
    """YOLO 검출 txt: 한 줄에 '클래스 cx cy w h'(0~1 정규화). with_score 면 끝에 점수를 붙인다(예측 저장용)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    for b in boxes:
        x1, y1 = max(0.0, min(b.x1, w)), max(0.0, min(b.y1, h))
        x2, y2 = max(0.0, min(b.x2, w)), max(0.0, min(b.y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        row = f"{int(b.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        if with_score:
            row += f" {b.score:.5f}"
        lines.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def read_yolo(path: str, w: int, h: int) -> list[Box]:
    """write_yolo 의 반대. 파일이 없으면 빈 목록. 6번째 열이 있으면 점수로 읽는다."""
    out: list[Box] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            s = float(p[5]) if len(p) >= 6 else 1.0
            out.append(Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, s))
    return out


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    """박스 목록 둘 사이 IoU 표(len(a) x len(b))."""
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    A = np.array([x.xyxy() for x in a], dtype=np.float64)
    B = np.array([x.xyxy() for x in b], dtype=np.float64)
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_boxes(ref: Sequence[Box], other: Sequence[Box], iou_thr: float = 0.5) -> tuple[list, list, list]:
    """같은 클래스끼리 IoU 큰 순서로 하나씩 짝짓는다.

    돌려주는 값: (짝 목록 [(ref 번호, other 번호, IoU)], 짝 없는 ref 번호, 짝 없는 other 번호)
    """
    m = iou_matrix(ref, other)
    if m.size:
        same = np.array([[r.cls == o.cls for o in other] for r in ref], dtype=bool)
        m = np.where(same, m, 0.0)
    pairs = []
    used_r, used_o = set(), set()
    if m.size:
        order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
        for i, j in order:
            if m[i, j] < iou_thr:
                break
            if i in used_r or j in used_o:
                continue
            used_r.add(int(i))
            used_o.add(int(j))
            pairs.append((int(i), int(j), float(m[i, j])))
    miss_r = [i for i in range(len(ref)) if i not in used_r]
    miss_o = [j for j in range(len(other)) if j not in used_o]
    return pairs, miss_r, miss_o


# ---------------------------------------------------------------------------
# 파일 도우미
# ---------------------------------------------------------------------------
def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Iterable[dict], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(folder: str) -> list[str]:
    """폴더 안 이미지 파일 경로(이름순)."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTS))


def image_size(path: str) -> tuple[int, int]:
    """(폭, 높이). 이미지를 읽지 못하면 오류."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")
    return int(img.shape[1]), int(img.shape[0])


def update_meta(paths: RoundPaths, step: str, info: dict) -> dict:
    """회차 메타데이터(meta.json)의 steps 아래에 단계별 결과를 덮어쓴다."""
    meta = read_json(paths.meta_json, {}) or {}
    meta.setdefault("round", paths.round)
    meta.setdefault("prev_round", paths.cfg.get("prev_round"))
    meta.setdefault("steps", {})
    meta["steps"][step] = info
    meta["config"] = paths.cfg
    write_json(paths.meta_json, meta)
    return meta
