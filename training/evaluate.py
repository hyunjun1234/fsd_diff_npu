#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.common import (  # noqa: E402
    KEEP_CLASSES,
    Box,
    RoundPaths,
    check_round_name,
    iou_matrix,
    load_config,
    read_csv,
    read_yolo,
    resolve_path,
    update_meta,
    write_json,
    write_yolo,
)

# 고정 시험 세트에서 회차 모델끼리 mAP50, mAP50-95, 클래스별 재현율을 비교한다.
# 직전 회차보다 예전 장면(이전 회차에 시험 세트에 들어간 프레임) 성능이 떨어지면 경고한다.
# 두 모델을 같은 목록에서 매번 다시 돌려 비교한다(저장해 둔 옛 점수와 비교하지 않음).
#
# 예) 학습 서버에서
#   python3 -m training.evaluate --config training/configs/round.yaml
#   python3 -m training.evaluate --config training/configs/round.yaml --model r001_hailo=out/pi_preds/r001_hailo
#     (--model 이름=경로: .pt 는 Ultralytics 로 돌리고, 폴더는 미리 저장한 예측 txt(클래스 cx cy w h 점수)로 읽음)
#
# 결과: out/training/datasets/<회차>/eval/eval.json, eval.md(docs/measurements/ 에 붙일 표), pred_<모델>/

# mAP 는 COCO 방식(IoU 0.50~0.95 10단계, 재현율 101점 보간)으로 직접 계산한다. Ultralytics val 과
# 보간 방식이 조금 달라 소수 셋째 자리에서 차이가 날 수 있다(확인 필요). 대신 NPU 예측 txt 도 같은 식으로 잴 수 있다.
IOU_THRS = np.linspace(0.5, 0.95, 10)


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    p = precision.copy()
    for i in range(len(p) - 2, -1, -1):  # 오른쪽에서부터 최대값으로 채운 포락선
        p[i] = max(p[i], p[i + 1])
    out = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(recall, r, side="left")
        out += p[k] if k < len(p) else 0.0
    return out / 101.0


def evaluate_boxes(gt: dict, pred: dict, recall_conf: float = 0.4, recall_iou: float = 0.5) -> dict:
    """gt, pred: {프레임: [Box]}. 클래스별 AP50, AP50-95, 재현율과 평균을 돌려준다."""
    per = {}
    for c, name in KEEP_CLASSES.items():
        g = {f: [b for b in bs if b.cls == c] for f, bs in gt.items()}
        n_gt = sum(len(v) for v in g.values())
        if n_gt == 0:
            continue
        preds = sorted(((b.score, f, b) for f, bs in pred.items() if f in gt for b in bs if b.cls == c),
                       key=lambda x: -x[0])
        aps = []
        for t in IOU_THRS:
            used = {f: np.zeros(len(v), bool) for f, v in g.items()}
            tp = np.zeros(len(preds))
            for i, (_s, f, b) in enumerate(preds):
                if not g[f]:
                    continue
                ious = iou_matrix([b], g[f])[0]
                ious[used[f]] = -1.0
                j = int(np.argmax(ious))
                if ious[j] >= t:
                    used[f][j] = True
                    tp[i] = 1
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1 - tp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            aps.append(_ap_101(rec, prec))
        # 재현율: 파이 문턱(recall_conf) 이상 예측만 놓고 IoU recall_iou 로 짝지은 비율
        hit = 0
        for f, gb in g.items():
            pb = [b for b in pred.get(f, []) if b.cls == c and b.score >= recall_conf]
            if not gb or not pb:
                continue
            m = iou_matrix(gb, pb)
            while m.size and m.max() >= recall_iou:
                i, j = np.unravel_index(np.argmax(m), m.shape)
                hit += 1
                m[i, :] = -1
                m[:, j] = -1
        per[name] = {"n_gt": n_gt, "ap50": float(aps[0]), "ap50_95": float(np.mean(aps)), "recall": hit / n_gt}
    out = {"per_class": per}
    out["map50"] = float(np.mean([v["ap50"] for v in per.values()])) if per else None
    out["map50_95"] = float(np.mean([v["ap50_95"] for v in per.values()])) if per else None
    out["n_gt"] = sum(v["n_gt"] for v in per.values())
    out["frames"] = len(gt)
    return out


# ---------------------------------------------------------------------------
# 예측기: .pt(Ultralytics), 예측 txt 폴더, 가짜
# ---------------------------------------------------------------------------
class UltralyticsPredictor:
    def __init__(self, weights: str, imgsz: int = 640, device=0):
        self.weights, self.imgsz, self.device = weights, imgsz, device
        self._m = None

    def predict(self, img: np.ndarray, path: str | None = None) -> list:
        if self._m is None:
            from ultralytics import YOLO  # 서버에서만

            self._m = YOLO(self.weights)
        r = self._m.predict(source=img, conf=0.001, iou=0.7, imgsz=self.imgsz, max_det=300,
                            device=self.device, verbose=False)[0]
        xyxy, conf, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        return [Box(int(c), *map(float, b), score=float(s)) for b, s, c in zip(xyxy, conf, cls) if int(c) in KEEP_CLASSES]


class DirPredictor:
    """미리 저장한 예측 txt 폴더(파이 NPU 결과 등). 파일 이름은 시험 세트 프레임 이름과 같아야 한다."""

    def __init__(self, folder: str):
        self.folder = folder

    def predict(self, img: np.ndarray, path: str | None = None) -> list:
        fid = os.path.splitext(os.path.basename(path))[0]
        h, w = img.shape[:2]
        return read_yolo(os.path.join(self.folder, fid + ".txt"), w, h)


def make_predictor(spec: str, cfg: dict):
    if spec == "fake":
        from training.fakes import FakePredictor

        return FakePredictor()
    if spec.startswith("fake_miss:"):
        from training.fakes import FakePredictor

        return FakePredictor(miss_classes=[int(x) for x in spec.split(":", 1)[1].split(",")])
    p = resolve_path(spec) if (os.sep in spec or os.path.exists(resolve_path(spec))) else spec
    if os.path.isdir(p):
        return DirPredictor(p)
    return UltralyticsPredictor(p, int(cfg["evaluate"]["imgsz"]), cfg["train_det"]["device"])


def default_models(cfg: dict) -> list:
    """직전 회차와 이번 회차 모델. 직전이 r000 이면 base_weights(처음 받은 yolov8s.pt)."""
    paths = RoundPaths(cfg)
    prev = cfg["prev_round"]
    if prev == "r000":
        base = cfg["train_det"]["base_weights"]
        local = os.path.join(paths.models, os.path.basename(base))
        prev_spec = local if os.path.exists(local) else base
    else:
        prev_spec = paths.best_pt(prev)
    return [(prev, prev_spec), (cfg["round"], paths.best_pt())]


# ---------------------------------------------------------------------------
# 시험 세트 읽기와 부분 집합
# ---------------------------------------------------------------------------
def load_test_gt(cfg: dict) -> tuple[dict, list, int]:
    """(사람 라벨 정답, 목록 행, 사람 라벨이 없어 뺀 수). 자동 라벨만 있는 프레임은 평가에 쓰지 않는다."""
    paths = RoundPaths(cfg)
    rows = read_csv(paths.test_csv)
    gt, used, skipped = {}, [], 0
    for r in rows:
        ip = os.path.join(paths.test_dir, "images", r["frame_id"] + ".jpg")
        lp = os.path.join(paths.test_dir, "labels", r["frame_id"] + ".txt")
        if r.get("label_source") != "human" or not os.path.exists(lp) or not os.path.exists(ip):
            skipped += 1
            continue
        h, w = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).shape[:2]
        gt[r["frame_id"]] = [b for b in read_yolo(lp, w, h) if b.cls in KEEP_CLASSES]
        used.append(r)
    return gt, used, skipped


def subsets(rows: list, current_round: str) -> dict:
    """all, old(이번 회차 전에 들어간 시험 프레임), new(이번 회차에 들어간 것), scene:<이름>."""
    me = check_round_name(current_round)
    out = {"all": [r["frame_id"] for r in rows]}
    old = [r["frame_id"] for r in rows if check_round_name(r.get("added_round") or current_round) < me]
    new = [r["frame_id"] for r in rows if check_round_name(r.get("added_round") or current_round) >= me]
    if old:
        out["old"] = old
    if new:
        out["new"] = new
    for r in rows:
        out.setdefault("scene:" + r.get("scene", ""), []).append(r["frame_id"])
    return out


# ---------------------------------------------------------------------------
# 회차 비교
# ---------------------------------------------------------------------------
def run_predictions(cfg: dict, name: str, predictor, fids: list, out_dir: str) -> dict:
    paths = RoundPaths(cfg)
    preds = {}
    for fid in fids:
        ip = os.path.join(paths.test_dir, "images", fid + ".jpg")
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        boxes = predictor.predict(img, ip)
        preds[fid] = boxes
        write_yolo(os.path.join(out_dir, f"pred_{name}", fid + ".txt"), boxes, img.shape[1], img.shape[0], with_score=True)
    return preds


def regression_warnings(prev: dict, cur: dict, cfg: dict) -> list:
    """예전 장면(없으면 전체)과 장면별로 직전 회차보다 떨어진 항목을 문장으로 돌려준다."""
    e = cfg["evaluate"]
    tol_m, tol_r, min_gt = float(e["map_tolerance"]), float(e["recall_tolerance"]), int(e["min_gt"])
    warns = []
    keys = ["old"] if "old" in cur else ["all"]
    keys += [k for k in cur if k.startswith("scene:")]
    for k in keys:
        a, b = prev.get(k), cur.get(k)
        if not a or not b or a["map50"] is None or b["map50"] is None:
            continue
        if k.startswith("scene:") and b["n_gt"] < min_gt:
            continue
        for m in ("map50", "map50_95"):
            if b[m] < a[m] - tol_m:
                warns.append(f"{k} {m} {a[m]:.3f} 에서 {b[m]:.3f} 로 떨어짐")
        if k in ("old", "all"):
            for cname, v in b["per_class"].items():
                pv = a["per_class"].get(cname)
                if pv and v["n_gt"] >= min_gt and v["recall"] < pv["recall"] - tol_r:
                    warns.append(f"{k} {cname} 재현율 {pv['recall']:.3f} 에서 {v['recall']:.3f} 로 떨어짐")
    return warns


def _f(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def to_markdown(res: dict, a: str, b: str) -> str:
    """docs/measurements/ 에 붙일 표. 표는 3열(지표, 직전, 이번)로 맞춘다."""
    ra, rb = res["models"][a], res["models"][b]
    lines = [
        f"이 문서는 고정 시험 세트에서 {a} 와 {b} 검출 모델을 비교한 결과입니다. "
        f"사람 라벨이 있는 {ra['all']['frames']}장(박스 {ra['all']['n_gt']}개)으로 쟀고, "
        f"사람 라벨이 없어 뺀 프레임은 {res['skipped_unlabeled']}장입니다. 재현율은 점수 "
        f"{res['recall_conf']} 이상, IoU 0.5 기준입니다.",
        "",
        f"| 지표 | {a} | {b} |",
        "|---|---|---|",
    ]
    for k in [k for k in ("all", "old", "new") if k in rb] + sorted(k for k in rb if k.startswith("scene:")):
        names = {"all": "전체", "old": "예전 장면", "new": "새 장면"}
        label = names[k] if k in names else "장면 " + k.split(":", 1)[1]
        lines.append(f"| {label} mAP50 | {_f(ra.get(k, {}).get('map50'))} | {_f(rb[k]['map50'])} |")
        lines.append(f"| {label} mAP50-95 | {_f(ra.get(k, {}).get('map50_95'))} | {_f(rb[k]['map50_95'])} |")
    lines += ["", "클래스별 재현율(전체)입니다.", "", f"| 클래스 | {a} | {b} |", "|---|---|---|"]
    for cname, v in rb["all"]["per_class"].items():
        pv = ra["all"]["per_class"].get(cname, {})
        lines.append(f"| {cname} ({v['n_gt']}개) | {_f(pv.get('recall'))} | {_f(v['recall'])} |")
    lines.append("")
    if res["warnings"]:
        lines.append("직전 회차보다 떨어진 항목이 있습니다. 이전 회차와 COCO 섞는 비율(prev_ratio, coco_ratio)을 "
                     "늘려 다시 학습하는 것을 검토합니다.")
        lines.append("")
        lines += [f"{i}. {w}" for i, w in enumerate(res["warnings"], 1)]
    else:
        lines.append("직전 회차보다 허용 범위를 넘게 떨어진 항목은 없습니다.")
    return "\n".join(lines) + "\n"


def evaluate_round(cfg: dict, models: list | None = None, predictors: dict | None = None) -> dict:
    """models: [(이름, 경로나 fake 지정)]. 앞의 두 개를 직전, 이번으로 보고 경고를 낸다."""
    paths = RoundPaths(cfg)
    models = models or default_models(cfg)
    gt, rows, skipped = load_test_gt(cfg)
    if not gt:
        raise RuntimeError("사람 라벨이 붙은 시험 프레임이 없습니다. 시험 세트를 CVAT 에서 라벨링하고 "
                           "cvat_io import --test-set 으로 들여오세요.")
    subs = subsets(rows, cfg["round"])
    out_dir = os.path.join(paths.rd, "eval")
    e = cfg["evaluate"]
    res = {"models": {}, "skipped_unlabeled": skipped, "recall_conf": float(e["recall_conf"])}
    for name, spec in models:
        pr = (predictors or {}).get(name) or make_predictor(spec, cfg)
        preds = run_predictions(cfg, name, pr, list(gt), out_dir)
        res["models"][name] = {k: evaluate_boxes({f: gt[f] for f in ids}, preds, float(e["recall_conf"]),
                                                 float(e["match_iou"])) for k, ids in subs.items()}
    names = [m[0] for m in models]
    res["warnings"] = regression_warnings(res["models"][names[0]], res["models"][names[1]], cfg) if len(names) >= 2 else []
    write_json(os.path.join(out_dir, "eval.json"), res)
    if len(names) >= 2:
        with open(os.path.join(out_dir, "eval.md"), "w", encoding="utf-8") as f:
            f.write(to_markdown(res, names[0], names[1]))
    update_meta(paths, "evaluate", {"models": names, "warnings": res["warnings"],
                                    "map50": {n: res["models"][n]["all"]["map50"] for n in names}})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="고정 시험 세트에서 회차 모델 비교")
    ap.add_argument("--config", default="training/configs/round.yaml")
    ap.add_argument("--model", action="append", default=[], help="이름=경로(.pt 또는 예측 txt 폴더). 처음 둘이 직전, 이번")
    ap.add_argument("--fail-on-regression", action="store_true", help="떨어진 항목이 있으면 종료 코드 3")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    models = [tuple(m.split("=", 1)) for m in args.model] or None
    res = evaluate_round(cfg, models)
    out = os.path.join(RoundPaths(cfg).rd, "eval")
    for n, v in res["models"].items():
        print(f"{n}: mAP50 {_f(v['all']['map50'])}, mAP50-95 {_f(v['all']['map50_95'])}")
    for w in res["warnings"]:
        print(f"[경고] {w}")
    print(f"결과: {out}/eval.md")
    return 3 if (args.fail_on_regression and res["warnings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
