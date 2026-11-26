#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview import layout_sim as ls  # noqa: E402
from realityview.bev import BevGrid  # noqa: E402

# 카메라를 몇 대, 어디에, 어떤 렌즈로 달아야 하는지 시뮬레이션으로 비교한다(realityview/layout_sim.py).
#
# 하위 명령
#   layouts  : 비교할 배치 목록과 카메라별 화각, 예상 FPS를 찍는다.
#   coverage : 물체를 바닥 격자의 칸마다 세워 보고 몇 대가 보는지, 한 프레임에 검출될 확률을 낸다(정지 시야 지도).
#              --figure를 주면 배치별 지도를 그림으로 저장한다.
#   sweep    : 좌우 카메라의 자리, 렌즈, 방향, 숙인 각을 바꿔 가며 가까운 곳 시야와 장면 지표를 낸다.
#   run      : 장면(추월, 정체, 마주 오는 차, 교차로, 차 한 바퀴 돌기, 먼 앞차, 대기열)을 돌려
#              화면에 나온 비율, 1초 넘게 끊긴 횟수, ID 바뀜, 중복 표시를 낸다.
#              추적은 실제 파이프라인과 같은 함수(project_detections, merge_cross_camera, BevTracker)를 쓴다.
#   report   : run이 남긴 JSON을 표로 찍는다.
#
# FPS는 --fps에 숫자를 주면 모든 배치에 같은 값을 쓰고, model이면 카메라 수에 따른 처리량 모형(FpsModel)을 쓴다.
# 검출 모형(flat, easy, base, hard)의 크기별 검출률은 실측이 아니라 가정이라 여러 개를 함께 돌려 본다.
#
# 예)
#   /usr/bin/python3 scripts/sim_camera_count.py layouts
#   /usr/bin/python3 scripts/sim_camera_count.py coverage --out out/measure/19_coverage.json \
#       --figure docs/images/camera_count_coverage.png --figure-layouts c4_fender,c5_fender86,c5_side160,c7_tesla
#   /usr/bin/python3 scripts/sim_camera_count.py sweep --out out/measure/19_side_sweep.json
#   /usr/bin/python3 scripts/sim_camera_count.py run --fps model --out out/measure/19_run_model.json
#   /usr/bin/python3 scripts/sim_camera_count.py run --fps 13 --det base --seeds 2 --contact drop \
#       --layouts c4_fender,c5_side160,c7_tesla --out out/measure/19_run_drop.json
#   /usr/bin/python3 scripts/sim_camera_count.py report --in out/measure/19_run_model.json


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
def _split(text: str, known, what: str) -> list[str]:
    keys = list(known) if text == "all" else [k for k in text.split(",") if k]
    for k in keys:
        if k not in known:
            raise SystemExit(f"모르는 {what}: {k} (가능한 값: {', '.join(known)})")
    return keys


def _fps_model(args) -> ls.FpsModel:
    return ls.FpsModel(decode_ms=args.decode_ms, seg=args.seg, ground=not args.no_ground, resize_ms=args.resize_ms)


def _truth_opts(args) -> dict:
    return {"ego_body": not args.no_ego_body, "occlusion": args.occlusion, "skew": not args.no_skew,
            "rule": args.rule, "contact": args.contact}


def _args_dict(args) -> dict:
    return {k: v for k, v in vars(args).items() if k != "func"}


def _save(path: str | None, data: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"저장: {path}")


def _layout_info(layout: ls.Layout, fm: ls.FpsModel) -> dict:
    cams = []
    for spec, cam in zip(layout.specs, layout.cameras()):
        lens = ls.LENSES[spec.lens]
        cams.append({"name": spec.name, "lens": spec.lens, "mount": spec.mount, "link": spec.link,
                     "position": [round(float(v), 3) for v in cam.position], "yaw": cam.yaw_deg, "pitch": cam.pitch_deg,
                     "hfov": round(lens.hfov_deg, 1), "vfov": round(lens.vfov_deg, 1), "h_clip": spec.clip()})
    return {"key": layout.key, "title": layout.title, "n": layout.n, "n_usb": layout.n_usb,
            "fps_model": round(fm.fps(layout.n, layout.n_usb), 2), "cams": cams}


def _fmt(v, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------
def cmd_layouts(args) -> None:
    fm = _fps_model(args)
    layouts = ls.make_layouts(args.side_pitch, args.rear_lens)
    for key in _split(args.layouts, layouts, "배치"):
        info = _layout_info(layouts[key], fm)
        print(f"{key}: {info['title']} | {info['n']}대(USB {info['n_usb']}) | 모형 FPS {info['fps_model']}")
        for c in info["cams"]:
            clip = f", 창틀 한계 {c['h_clip'][0]:+.0f}도 기준 좌우 {c['h_clip'][1]:.0f}도" if c["h_clip"] else ""
            print(f"    {c['name']:<11} {c['lens']:<16} {c['mount']:<10} 위치 {c['position']} yaw {c['yaw']:+.0f} "
                  f"pitch {c['pitch']:.0f} 화각 {c['hfov']}x{c['vfov']}도 {c['link']}{clip}")


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def _coverage_task(task: tuple) -> dict:
    key, side_pitch, rear_lens, label, det_key, step, opts = task
    layout = ls.make_layouts(side_pitch, rear_lens)[key]
    cov = ls.coverage_map(layout, label, ls.DET_MODELS[det_key], step, ego_body=opts["ego_body"], rule=opts["rule"],
                          contact=opts["contact"])
    return {"layout": key, "label": label, **ls.coverage_summary(cov)}


def cmd_coverage(args) -> None:
    layouts = ls.make_layouts(args.side_pitch, args.rear_lens)
    keys = _split(args.layouts, layouts, "배치")
    labels = _split(args.labels, ls.SIZES, "물체")
    opts = _truth_opts(args)
    tasks = [(k, args.side_pitch, args.rear_lens, lab, args.det, args.step, opts) for k in keys for lab in labels]
    with Pool(args.jobs) as pool:
        rows = pool.map(_coverage_task, tasks)
    print("| 배치 | 물체 | 보이는 칸 | 가까운 곳(5m) | 앞옆 | 옆 | 뒤옆 | 평균 검출 확률 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['layout']} | {r['label']} | {_fmt(r['seen_ratio'])} | {_fmt(r['seen_near'])} | "
              f"{_fmt(r['seen_front_side'])} | {_fmt(r['seen_side'])} | {_fmt(r['seen_rear_side'])} | {_fmt(r['p_mean'])} |")
    _save(args.out, {"args": _args_dict(args), "rows": rows})
    if args.figure:
        fig_keys = _split(args.figure_layouts, layouts, "배치") if args.figure_layouts else keys
        draw_coverage_figure([layouts[k] for k in fig_keys], args.figure, args.figure_label, ls.DET_MODELS[args.det], opts)


# 시야 지도 그림 ------------------------------------------------------------
_COL_BLIND = (70, 70, 215)  # BGR: 아무 카메라도 못 보는 칸
_COL_ONE = (200, 225, 170)  # 한 대가 봄
_COL_MANY = (120, 185, 70)  # 두 대 이상이 봄
_COL_OUT = (235, 235, 235)  # 자차와 겹쳐 물체를 둘 수 없는 칸


def _panel(layout: ls.Layout, label: str, det: ls.DetModel, opts: dict, cell: int = 8, step: float = 0.5) -> np.ndarray:
    grid = BevGrid()
    cov = ls.coverage_map(layout, label, det, step, x_range=(grid.x_min, grid.x_max), y_range=(grid.y_min, grid.y_max),
                          ego_body=opts["ego_body"], rule=opts["rule"], contact=opts["contact"])
    xs, ys = cov["xs"], cov["ys"]
    h, w = len(xs) * cell, len(ys) * cell  # 위쪽이 차 앞, 왼쪽이 차 왼쪽
    img = np.full((h, w, 3), 255, np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round((grid.y_max - y) / step * cell)), int(round((grid.x_max - x) / step * cell))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            n = int(cov["n_cam"][i, j])
            col = _COL_OUT if not cov["free"][i, j] else (_COL_BLIND if n == 0 else (_COL_ONE if n == 1 else _COL_MANY))
            u, v = px(x + step / 2, y + step / 2)
            cv2.rectangle(img, (u, v), (u + cell - 1, v + cell - 1), col, -1)
    for r in (5.0, 10.0):  # 차체에서 5m, 10m 떨어진 선
        a, b = px(ls.EGO_X_FRONT + r, ls.EGO_HALF_W + r), px(ls.EGO_X_REAR - r, -ls.EGO_HALF_W - r)
        cv2.rectangle(img, a, b, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(img, px(ls.EGO_X_FRONT, ls.EGO_HALF_W), px(ls.EGO_X_REAR, -ls.EGO_HALF_W), (40, 40, 40), -1)
    for cam in layout.cameras():  # 카메라 자리와 보는 방향
        c = px(float(cam.position[0]), float(cam.position[1]))
        yaw = np.radians(cam.yaw_deg)
        tip = px(float(cam.position[0] + 1.6 * np.cos(yaw)), float(cam.position[1] + 1.6 * np.sin(yaw)))
        cv2.arrowedLine(img, c, tip, (0, 140, 255), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(img, c, 4, (0, 140, 255), -1, cv2.LINE_AA)
    return img


def draw_coverage_figure(layouts: list[ls.Layout], path: str, label: str, det: ls.DetModel, opts: dict) -> None:
    """배치별 시야 지도를 옆으로 이어 붙인 그림. 빨강은 사각, 연두는 한 대, 초록은 두 대 이상이 보는 칸이다."""
    pad, head = 14, 46
    panels = [_panel(lay, label, det, opts) for lay in layouts]
    h = max(p.shape[0] for p in panels) + head + pad
    w = sum(p.shape[1] for p in panels) + pad * (len(panels) + 1)
    canvas = np.full((h, w, 3), 255, np.uint8)
    x = pad
    for lay, p in zip(layouts, panels):
        canvas[head:head + p.shape[0], x:x + p.shape[1]] = p
        cv2.rectangle(canvas, (x - 1, head - 1), (x + p.shape[1], head + p.shape[0]), (120, 120, 120), 1)
        cv2.putText(canvas, f"{lay.n} cameras", (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, lay.key, (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        x += p.shape[1] + pad
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, canvas)
    print(f"그림 저장: {path} ({canvas.shape[1]}x{canvas.shape[0]})")


# ---------------------------------------------------------------------------
# run: 장면 돌리기
# ---------------------------------------------------------------------------
def _make_layout(spec: tuple, side_pitch: float, rear_lens: str) -> ls.Layout:
    """("key",)면 make_layouts의 배치, ("side", key, lens, mount, yaw, pitch, main)이면 좌우 카메라를 바꾼 배치."""
    if spec[0] == "side":
        _, key, lens, mount, yaw, pitch, main = spec
        return ls.side_layout(key, key, lens, mount, yaw, pitch, main=main, rear_lens=rear_lens)
    return ls.make_layouts(side_pitch, rear_lens)[spec[0]]


def _run_task(task: tuple) -> tuple[list[dict], float]:
    spec, scen_key, det_keys, seeds, fps, opts, side_pitch, rear_lens, tracker, vehicle_center = task
    t0 = time.time()
    layout = _make_layout(spec, side_pitch, rear_lens)
    scen = ls.make_scenarios()[scen_key]
    truth = ls.build_truth(layout, scen, fps, **opts)
    out = []
    for dk in det_keys:
        items = [ls.run_once(layout, scen, ls.DET_MODELS[dk], fps, seed, truth=truth, tracker_mode=tracker,
                             vehicle_center=vehicle_center) for seed in seeds]
        per_seed = [{k: it[k] for k in ("total", "shown", "gaps_over_1s", "id_switches", "dup_s", "false_s", "minutes")}
                    for it in items]
        out.append({"layout": layout.key, "scenario": scen_key, "det": dk, "fps": round(fps, 2), "seeds": len(seeds),
                    **ls.merge_summaries(items), "per_seed": per_seed})
    return out, time.time() - t0


def _layout_fps(layout: ls.Layout, args) -> float:
    if args.fps == "model":
        return _fps_model(args).fps(layout.n, layout.n_usb)
    return float(args.fps)


def run_batch(specs: list[tuple], scen_keys: list[str], det_keys: list[str], args) -> list[dict]:
    opts = _truth_opts(args)
    seeds = list(range(args.seeds))
    scens = ls.make_scenarios()
    tasks = []
    for spec in specs:
        layout = _make_layout(spec, args.side_pitch, args.rear_lens)
        fps = _layout_fps(layout, args)
        for sk in scen_keys:
            cost = layout.n * len(scens[sk].actors) * scens[sk].duration * fps  # 오래 걸리는 것부터 돌린다
            tasks.append((cost, (spec, sk, det_keys, seeds, fps, opts, args.side_pitch, args.rear_lens, args.tracker,
                                 args.vehicle_center)))
    tasks.sort(key=lambda t: -t[0])
    rows, t0 = [], time.time()
    with Pool(args.jobs) as pool:
        for i, (out, sec) in enumerate(pool.imap_unordered(_run_task, [t[1] for t in tasks]), 1):
            rows += out
            if not args.quiet:
                r = out[0]
                print(f"[{i}/{len(tasks)}] {r['layout']} {r['scenario']} {sec:.0f}s (누적 {time.time() - t0:.0f}s)",
                      file=sys.stderr, flush=True)
    order = {(_make_layout(s, args.side_pitch, args.rear_lens).key): i for i, s in enumerate(specs)}
    rows.sort(key=lambda r: (order[r["layout"]], scen_keys.index(r["scenario"]), det_keys.index(r["det"])))
    return rows


def cmd_run(args) -> None:
    layouts = ls.make_layouts(args.side_pitch, args.rear_lens)
    keys = _split(args.layouts, layouts, "배치")
    scen_keys = _split(args.scenarios, ls.make_scenarios(), "장면")
    det_keys = _split(args.det, ls.DET_MODELS, "검출 모형")
    rows = run_batch([(k,) for k in keys], scen_keys, det_keys, args)
    fm = _fps_model(args)
    data = {"args": _args_dict(args), "layouts": {k: _layout_info(layouts[k], fm) for k in keys}, "rows": rows}
    _save(args.out, data)
    print_report(data, None)


# ---------------------------------------------------------------------------
# report: 표
# ---------------------------------------------------------------------------
def _pool_rows(rows: list[dict]) -> dict:
    """여러 줄(검출 모형, 장면)을 합친다. 횟수는 더하고 비율은 합친 횟수로 다시 계산한다."""
    keys = ("total", "shown", "visible", "far_total", "far_shown", "id_switches", "dup_s", "false_s", "minutes",
            "gaps_over_1s", "gaps_over_3s", "gap_sum_s")
    out = {k: sum(r[k] for r in rows) for k in keys}
    out["gap_max_s"] = max(r["gap_max_s"] for r in rows)
    mins = out["minutes"] or 1e-9
    out["shown_ratio"] = out["shown"] / out["total"] if out["total"] else None
    out["visible_ratio"] = out["visible"] / out["total"] if out["total"] else None
    out["far_ratio"] = out["far_shown"] / out["far_total"] if out["far_total"] else None
    out["gaps_over_1s_per_min"] = out["gaps_over_1s"] / mins
    out["id_switches_per_min"] = out["id_switches"] / mins
    out["dup_s_per_min"] = out["dup_s"] / mins
    out["false_s_per_min"] = out["false_s"] / mins
    return out


def print_report(data: dict, det: str | None) -> None:
    rows = data["rows"] if det is None else [r for r in data["rows"] if r["det"] == det]
    lay_keys = list(dict.fromkeys(r["layout"] for r in rows))
    scen_keys = list(dict.fromkeys(r["scenario"] for r in rows))
    dets = list(dict.fromkeys(r["det"] for r in rows))
    by = {}
    for r in rows:
        by.setdefault((r["layout"], r["scenario"]), []).append(r)
    fps = {r["layout"]: r["fps"] for r in rows}
    print(f"\n검출 모형: {', '.join(dets)} (합침). 장면: {', '.join(scen_keys)}")
    print("\n화면에 나온 비율(탑뷰 범위 안의 물체, 카메라에 안 보이는 것 포함)")
    print("| 배치 | FPS | " + " | ".join(scen_keys) + " | 전체 |")
    print("|---|---|" + "---|" * (len(scen_keys) + 1))
    for lk in lay_keys:
        cells = [_fmt(_pool_rows(by[(lk, sk)])["shown_ratio"]) for sk in scen_keys]
        allr = _pool_rows([r for sk in scen_keys for r in by[(lk, sk)]])
        print(f"| {lk} | {fps[lk]:.1f} | " + " | ".join(cells) + f" | {_fmt(allr['shown_ratio'])} |")
    print("\n1초 넘게 화면에서 사라진 횟수(분당)")
    print("| 배치 | " + " | ".join(scen_keys) + " | 전체 |")
    print("|---|" + "---|" * (len(scen_keys) + 1))
    for lk in lay_keys:
        cells = [_fmt(_pool_rows(by[(lk, sk)])["gaps_over_1s_per_min"], 2) for sk in scen_keys]
        allr = _pool_rows([r for sk in scen_keys for r in by[(lk, sk)]])
        print(f"| {lk} | " + " | ".join(cells) + f" | {_fmt(allr['gaps_over_1s_per_min'], 2)} |")
    print("\n전체 장면 합: 먼 앞쪽(25~40m) 표시 비율, ID 바뀜(분당 횟수), 중복 표시와 헛 표시(분당 초)")
    print("| 배치 | 카메라가 본 비율 | 먼 앞쪽 | ID 바뀜 | 중복 | 헛 표시 | 가장 긴 끊김(초) |")
    print("|---|---|---|---|---|---|---|")
    for lk in lay_keys:
        a = _pool_rows([r for sk in scen_keys for r in by[(lk, sk)]])
        print(f"| {lk} | {_fmt(a['visible_ratio'])} | {_fmt(a['far_ratio'])} | {_fmt(a['id_switches_per_min'], 1)} | "
              f"{_fmt(a['dup_s_per_min'], 1)} | {_fmt(a['false_s_per_min'], 1)} | {_fmt(a['gap_max_s'], 1)} |")


def cmd_report(args) -> None:
    with open(args.inp, encoding="utf-8") as f:
        data = json.load(f)
    if args.layouts != "all":
        keep = set(args.layouts.split(","))
        data["rows"] = [r for r in data["rows"] if r["layout"] in keep]
    if args.scenarios != "all":
        keep = set(args.scenarios.split(","))
        data["rows"] = [r for r in data["rows"] if r["scenario"] in keep]
    print_report(data, args.det)


# ---------------------------------------------------------------------------
# sweep: 좌우 카메라 자리, 렌즈, 방향
# ---------------------------------------------------------------------------
def _sweep_static(task: tuple) -> dict:
    spec, rear_lens, det_key, opts, step = task
    layout = _make_layout(spec, 20.0, rear_lens)
    grid = BevGrid()
    out = {"layout": layout.key}
    for label in ("person", "car"):
        cov = ls.coverage_map(layout, label, ls.DET_MODELS[det_key], step, x_range=(grid.x_min, grid.x_max),
                              y_range=(grid.y_min, grid.y_max), ego_body=opts["ego_body"], rule=opts["rule"],
                              contact=opts["contact"])
        s = ls.coverage_summary(cov)
        out[f"{label}_near"] = s["seen_near"]
        out[f"{label}_seen"] = s["seen_ratio"]
        out[f"{label}_p"] = s["p_mean"]
    return out


def cmd_sweep(args) -> None:
    mounts = _split(args.mounts, ls.MOUNTS, "자리")
    lenses = _split(args.lenses, ls.LENSES, "렌즈")
    yaws = [float(v) for v in args.yaws.split(",")]
    pitches = [float(v) for v in args.pitches.split(",")]
    scen_keys = _split(args.scenarios, ls.make_scenarios(), "장면")
    det_keys = _split(args.det, ls.DET_MODELS, "검출 모형")
    main = not args.no_main
    specs, meta = [], {}
    for m in mounts:
        for lens in lenses:
            for yaw in yaws:
                for pitch in pitches:
                    key = f"{m}|{lens}|{yaw:g}|{pitch:g}"
                    specs.append(("side", key, lens, m, yaw, pitch, main))
                    meta[key] = {"mount": m, "lens": lens, "yaw": yaw, "pitch": pitch}
    opts = _truth_opts(args)
    with Pool(args.jobs) as pool:
        static = {r["layout"]: r for r in pool.map(_sweep_static, [(s, args.rear_lens, det_keys[0], opts, args.step) for s in specs])}
    rows = run_batch(specs, scen_keys, det_keys, args) if scen_keys else []
    by: dict = {}
    for r in rows:
        by.setdefault((r["layout"], r["scenario"]), []).append(r)
    table = []
    for key, m in meta.items():
        row = {**m, **{k: v for k, v in static[key].items() if k != "layout"}}
        for sk in scen_keys:
            a = _pool_rows(by[(key, sk)])
            row[f"{sk}_shown"] = a["shown_ratio"]
            row[f"{sk}_gaps"] = a["gaps_over_1s_per_min"]
        if scen_keys:
            a = _pool_rows(rows_for(by, key, scen_keys))
            row.update(shown=a["shown_ratio"], gaps=a["gaps_over_1s_per_min"], switches=a["id_switches_per_min"],
                       dup=a["dup_s_per_min"])
        table.append(row)
    table.sort(key=lambda r: (-(r.get("shown") or 0.0), r.get("gaps") or 0.0))
    head = ["자리", "렌즈", "yaw", "숙임", "사람 가까이", "차 가까이"] + [f"{sk} 표시" for sk in scen_keys] + \
           (["전체 표시", "끊김/분", "ID 바뀜/분", "중복 초/분"] if scen_keys else [])
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in table[: args.top] if args.top else table:
        cells = [r["mount"], r["lens"], f"{r['yaw']:g}", f"{r['pitch']:g}", _fmt(r["person_near"]), _fmt(r["car_near"])]
        cells += [_fmt(r[f"{sk}_shown"]) for sk in scen_keys]
        if scen_keys:
            cells += [_fmt(r["shown"]), _fmt(r["gaps"], 2), _fmt(r["switches"], 1), _fmt(r["dup"], 1)]
        print("| " + " | ".join(cells) + " |")
    _save(args.out, {"args": _args_dict(args), "rows": table})


def rows_for(by: dict, key: str, scen_keys: list[str]) -> list[dict]:
    return [r for sk in scen_keys for r in by[(key, sk)]]


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--layouts", default="all", help="배치 이름(쉼표로 구분). all이면 전부")
    ap.add_argument("--side-pitch", type=float, default=20.0, help="정측면을 보는 좌우 카메라를 아래로 숙인 각(도)")
    ap.add_argument("--rear-lens", default="imx291_160", help="후방 카메라 렌즈(LENSES의 이름)")
    ap.add_argument("--rule", default="partial", choices=["partial", "full"],
                    help="정답 상자 규칙. full은 SimWorld처럼 상자 전체가 화각 안에 있어야 한다")
    ap.add_argument("--contact", default="keep", choices=["keep", "drop"], help="drop이면 접지점이 잘린 상자를 뺀다")
    ap.add_argument("--no-ego-body", action="store_true", help="자차 차체가 가리는 것을 끈다")
    ap.add_argument("--occlusion", default="exact", choices=["exact", "box", "none"],
                    help="물체끼리 가리는 것을 보는 방법. exact는 시선 단위 계산, box는 영상 상자 겹침 어림, none은 끔")
    ap.add_argument("--no-skew", action="store_true", help="카메라별 촬영 시각 어긋남을 끈다")
    ap.add_argument("--decode-ms", type=float, default=0.0, help="처리량 모형: USB 카메라 한 프레임의 MJPEG 해제 시간(ms)")
    ap.add_argument("--seg", action="store_true", help="처리량 모형: DX-M1 도로 세그를 함께 돌리는 경우")
    ap.add_argument("--no-ground", action="store_true", help="처리량 모형: 탑뷰 바닥 영상을 끈 경우(run_demo.py --no-ground)")
    ap.add_argument("--resize-ms", type=float, default=0.0, help="처리량 모형: USB 카메라 한 대의 영상을 줄이는 시간(ms)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None, help="결과 JSON 파일")


def _add_run(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--scenarios", default="all", help="장면 이름(쉼표로 구분). all이면 전부")
    ap.add_argument("--det", default="easy,base,hard", help="검출 모형(쉼표로 구분): flat, easy, base, hard")
    ap.add_argument("--seeds", type=int, default=3, help="되풀이 횟수(seed 0부터)")
    ap.add_argument("--fps", default="model", help="숫자면 모든 배치에 같은 FPS, model이면 카메라 수에 따른 모형 값")
    ap.add_argument("--tracker", default="robust", choices=["basic", "robust"])
    ap.add_argument("--vehicle-center", default="ray", choices=["ray", "box"])
    ap.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="카메라 배치별 시야와 추적 지표 시뮬레이션")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("layouts", help="배치 목록")
    _add_common(p)
    p.set_defaults(func=cmd_layouts)
    p = sub.add_parser("coverage", help="정지 시야 지도")
    _add_common(p)
    p.add_argument("--labels", default="car,person", help="세워 볼 물체(쉼표로 구분)")
    p.add_argument("--det", default="base")
    p.add_argument("--step", type=float, default=0.5, help="격자 간격(m)")
    p.add_argument("--figure", default=None, help="시야 지도 그림 파일")
    p.add_argument("--figure-layouts", default=None, help="그림에 넣을 배치(쉼표로 구분). 없으면 --layouts와 같다")
    p.add_argument("--figure-label", default="person", help="그림에 쓸 물체")
    p.set_defaults(func=cmd_coverage)
    p = sub.add_parser("run", help="장면 돌리기")
    _add_common(p)
    _add_run(p)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("sweep", help="좌우 카메라 자리, 렌즈, 방향 훑기")
    _add_common(p)
    _add_run(p)
    p.add_argument("--mounts", default="bpillar,mirror,fender,side_glass")
    p.add_argument("--lenses", default="starvis2_86,imx291_120,imx291_160")
    p.add_argument("--yaws", default="60,75,90,105,120,135", help="왼쪽 카메라가 보는 방향(도). 90이 정측면, 클수록 뒤쪽")
    p.add_argument("--pitches", default="10,20,30")
    p.add_argument("--no-main", action="store_true", help="전방 메인 카메라를 뺀 4대 구성으로 훑는다")
    p.add_argument("--top", type=int, default=0, help="표에 찍을 줄 수(0이면 전부)")
    p.add_argument("--step", type=float, default=1.0, help="정지 시야 지도의 격자 간격(m)")
    p.set_defaults(func=cmd_sweep, scenarios="walk,overtake,queue", det="base", seeds=1, fps="13")
    p = sub.add_parser("report", help="run 결과 JSON을 표로")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--det", default=None, help="이 검출 모형만(없으면 전부 합침)")
    p.add_argument("--layouts", default="all")
    p.add_argument("--scenarios", default="all")
    p.set_defaults(func=cmd_report)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
