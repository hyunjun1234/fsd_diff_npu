#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from realityview.config import load_config  # noqa: E402
from realityview.pipeline import RealityViewPipeline  # noqa: E402
from realityview.multicam import FrameSyncer, SyncedSources  # noqa: E402
from realityview.sim import (  # noqa: E402
    LANE_W,
    SimBackend,
    SimObject,
    SimWorld,
    camera_timelines,
    default_objects,
    per_camera,
)

# 시뮬레이션 정답 ID와 추적기 출력을 비교해 추적 품질을 숫자로 낸다.
#
# 지표(정답 물체는 "어느 카메라든 정답 박스가 있는 프레임"만 센다)
#   id_switches      : 같은 정답 물체에 붙은 트랙 ID가 바뀐 횟수
#   tracks_per_object: 정답 물체 하나에 한 번이라도 붙은 트랙 ID 수(평균, 최대). 1이 이상적
#   miss_ratio       : 정답 물체가 보이는데 붙은 트랙이 없는 프레임 비율
#   dup_frames       : 정답 물체 근처에 짝이 없는 트랙이 하나 더 떠 있는 경우의 수(화면에서 "한 차에 박스 두 개")
#   false_tracks     : 어느 정답 물체 근처에도 없는 화면 트랙 수(프레임 합)
#
# 짝짓기는 프레임마다 일대일(가까운 쌍부터)이고, 허용 거리는 멀수록 커진다(먼 물체는 투영 오차가 크다).
#
# 예)
#   /usr/bin/python3 scripts/eval_tracking.py --config configs/two_cam.yaml --fps 15 --frames 225
#   /usr/bin/python3 scripts/eval_tracking.py --config configs/four_cam.yaml --tracker robust --seeds 0,1,2
#   # 카메라별 시각 어긋남(좌우 카메라 +50ms)을 넣고 FrameSyncer로 묶어 평가, 위치 보정 켜기
#   /usr/bin/python3 scripts/eval_tracking.py --config configs/four_cam.yaml --tracker robust \
#       --cam-skew-ms 0,0,50,50 --sync-tol-ms 25 --skew-comp
# 결과는 화면에 JSON으로 찍고, --out을 주면 그 파일에도 쓴다.


# ---------------------------------------------------------------------------
# 짝짓기 기준
# ---------------------------------------------------------------------------
def match_radius(xy: np.ndarray, scale: float = 1.0) -> float:
    """정답과 트랙을 같은 물체로 볼 최대 거리(m). 가까우면 3m, 멀면 거리의 20%."""
    rng = float(np.hypot(xy[0], xy[1]))
    return max(3.0 * scale, 0.2 * rng)


def _group(label: str) -> str:
    return "vehicle" if label in ("car", "truck", "bus") else label


def match_frame(gts: list[tuple], tracks: list, scale: float = 1.0) -> tuple[dict, list]:
    """한 프레임의 정답과 트랙을 일대일로 짝짓는다.

    gts: (정답 키, 라벨, xy) 목록. tracks: Track 목록.
    돌려줌: ({정답 키: 트랙 ID}, 짝이 없는 트랙 목록)
    """
    pairs = []
    for gi, (_, glab, gxy) in enumerate(gts):
        r = match_radius(gxy, scale)
        for ti, t in enumerate(tracks):
            if _group(t.label) != _group(glab):
                continue
            d = float(np.linalg.norm(np.asarray(t.xy) - gxy))
            if d < r:
                pairs.append((d, gi, ti))
    pairs.sort()
    used_g, used_t, out = set(), set(), {}
    for _, gi, ti in pairs:
        if gi in used_g or ti in used_t:
            continue
        used_g.add(gi)
        used_t.add(ti)
        out[gts[gi][0]] = tracks[ti].tid
    rest = [t for ti, t in enumerate(tracks) if ti not in used_t]
    return out, rest


# ---------------------------------------------------------------------------
# 지표 누적
# ---------------------------------------------------------------------------
class TrackingMetrics:
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.gt_frames = 0  # (정답 물체, 프레임) 쌍의 수 = 놓침 비율의 분모
        self.missed = 0
        self.switches = 0
        self.dup = 0
        self.false_tracks = 0
        self._last: dict = {}  # 정답 키 -> 마지막으로 붙은 트랙 ID
        self._ids: dict = {}  # 정답 키 -> 붙은 적 있는 트랙 ID 집합
        self.events: list[tuple] = []  # (프레임, 정답 키, 옛 ID, 새 ID) 확인용

    def add(self, frame: int, gts: list[tuple], tracks: list) -> None:
        assign, rest = match_frame(gts, tracks, self.scale)
        for key, label, xy in gts:
            self.gt_frames += 1
            tid = assign.get(key)
            if tid is None:
                self.missed += 1
            else:
                prev = self._last.get(key)
                if prev is not None and prev != tid:
                    self.switches += 1
                    self.events.append((frame, key, prev, tid))
                self._last[key] = tid
                self._ids.setdefault(key, set()).add(tid)
        # 남은 트랙: 정답 근처면 중복 표시, 아니면 헛트랙
        for t in rest:
            near = any(
                _group(t.label) == _group(lab) and np.linalg.norm(np.asarray(t.xy) - xy) < match_radius(xy, self.scale)
                for _, lab, xy in gts
            )
            if near:
                self.dup += 1
            else:
                self.false_tracks += 1

    def summary(self) -> dict:
        counts = [len(v) for v in self._ids.values()]
        return {
            "objects": len(self._ids),
            "gt_frames": self.gt_frames,
            "id_switches": self.switches,
            "tracks_per_object_mean": round(float(np.mean(counts)), 3) if counts else None,
            "tracks_per_object_max": int(max(counts)) if counts else None,
            "miss_ratio": round(self.missed / self.gt_frames, 4) if self.gt_frames else None,
            "dup_frames": self.dup,
            "false_tracks": self.false_tracks,
        }


# ---------------------------------------------------------------------------
# 시뮬레이션 실행
# ---------------------------------------------------------------------------
def visible_gt(world: SimWorld) -> list[tuple]:
    """이번 프레임에 어느 카메라든 정답 박스가 있는 물체. 키는 (물체 번호, 다시 등장한 횟수)."""
    seen = sorted({i for ids in world.gt_ids for i in ids})
    out = []
    for i in seen:
        o = world.objects[i]
        out.append(((i, world.wraps[i]), o.label, np.array([o.x, o.y])))
    return out


def run_sim(config: str, fps: float, frames: int, seed: int, tracker: str, warmup: int = 0,
            vehicle_center: str = "ray") -> TrackingMetrics:
    cams, grid, cfg = load_config(config)
    scale = float((cfg.get("scene") or {}).get("scale", 1.0))
    world = SimWorld(cams)
    pipe = RealityViewPipeline(
        cams, SimBackend(world, seed=seed), grid, show_ground=False, show_cameras=False, scale=scale, tracker=tracker,
        vehicle_center=vehicle_center,
    )
    m = TrackingMetrics(scale)
    dt = 1.0 / fps
    try:
        for n in range(frames):
            world.step(dt)
            out = pipe.step(world.render(), dt)
            if n >= warmup:
                m.add(n, visible_gt(world), out.tracks)
    finally:
        pipe.close()
    return m


# ---------------------------------------------------------------------------
# 카메라마다 찍은 시각이 다른 시뮬레이션 실행(FrameSyncer로 묶음)
# ---------------------------------------------------------------------------
class _ListStream:
    """미리 만든 (시각 ns, 정답) 목록을 재생 소스처럼 읽는다. 영상은 늘 같은 검은 영상(평가에는 상자만 쓴다)."""

    def __init__(self, entries: list, blank: np.ndarray):
        self.entries = entries
        self.blank = blank
        self.pos = 0
        self.recorded_timestamp_ns = None
        self.recorded_frame = None

    def next_ns(self):
        return self.entries[self.pos][0] if self.pos < len(self.entries) else None

    def read(self):
        if self.pos >= len(self.entries):
            return None
        self.recorded_timestamp_ns = self.entries[self.pos][0]
        self.recorded_frame = self.pos
        self.pos += 1
        return self.blank


class _GtStub:
    """SimBackend가 읽는 정답 자리(gt, gt_masks)만 가진 가짜 세계."""

    def __init__(self, n: int):
        self.gt = [[] for _ in range(n)]
        self.gt_masks = [None] * n


def run_sim_sync(config: str, fps: float, frames: int, seed: int, tracker: str, warmup: int, skew_ms: list[float],
                 drop: list[float], jitter_ms: list[float], tol_ms: float, on_miss: str, skew_comp: bool,
                 oncoming: bool = False, vehicle_center: str = "ray"):
    """카메라 k를 n/fps + skew_ms[k] 시각에 찍고(흔들림, 누락 포함) FrameSyncer로 묶어 추적을 평가한다.

    정답 위치는 기준 카메라(0번) 시각의 물체 위치다. 그래서 어긋난 카메라의 물체는 그만큼 다른 자리에서
    측정되고, 보정(skew_comp)을 켜면 트랙 속도로 기준 시각 자리로 옮긴다.
    돌려줌: (TrackingMetrics, FrameSyncer 요약)
    """
    cams, grid, cfg = load_config(config)
    scale = float((cfg.get("scene") or {}).get("scale", 1.0))
    n = len(cams)
    objs = default_objects()
    if oncoming:
        # 시각 어긋남이 크게 드러나는 경우: 왼쪽 차로로 마주 오는 차(상대 속도 25 m/s = 90 km/h).
        # 50ms 어긋나면 1.25m 다른 자리에서 측정된다
        objs.append(SimObject("car", 40.0, LANE_W, -25.0, 0.0, 4.5, 1.9, 1.5, (90, 160, 90)))
    world = SimWorld(cams, objects=objs)
    # 1) 카메라별 촬영 시각과 그 시각의 정답 상자(바닥 무늬는 그리지 않아 빠르다)
    tls = camera_timelines(n, frames, fps, skew_ms, drop, jitter_ms, seed=1000 + seed)
    blanks = [np.zeros((c.height, c.width, 3), np.uint8) for c in cams]
    streams = []
    for k in range(n):
        entries = []
        for _, t in tls[k]:
            world.set_time(t)
            _, dets, ids, _ = world.render_camera(k, ground=False)
            entries.append((int(round(t * 1e9)), dets, ids))
        streams.append(_ListStream(entries, blanks[k]))
    # 2) 묶어서 파이프라인에 넣는다
    syncer = FrameSyncer([(c.width, c.height) for c in cams], [c.name for c in cams], tolerance_ms=tol_ms, on_miss=on_miss)
    reader = SyncedSources(streams, syncer)
    stub = _GtStub(n)
    pipe = RealityViewPipeline(cams, SimBackend(stub, seed=seed), grid, show_ground=False, show_cameras=False,
                               scale=scale, tracker=tracker, skew_comp=skew_comp,
                               vehicle_center=vehicle_center)
    m = TrackingMetrics(scale)
    prev_ns, fi = None, 0
    try:
        while reader.read() is not None:
            b = reader.last
            used = [streams[k].entries[b.meta[k]] if b.meta[k] is not None else None for k in range(n)]
            stub.gt = [u[1] if u else [] for u in used]
            dt = (b.ref_ns - prev_ns) / 1e9 if prev_ns is not None else 1.0 / fps
            prev_ns = b.ref_ns
            out = pipe.step(b.frames, dt, b.offsets_s)
            # 정답: 쓴 영상 중 어디든 보인 물체의 기준 시각 위치
            world.set_time(b.ref_ns / 1e9)
            seen = sorted({i for u in used if u for i in u[2]})
            gts = [((i, world.wraps[i]), world.objects[i].label, np.array([world.objects[i].x, world.objects[i].y]))
                   for i in seen]
            if fi >= warmup:
                m.add(fi, gts, out.tracks)
            fi += 1
    finally:
        pipe.close()
    return m, syncer.summary()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_cam.yaml")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=225)
    ap.add_argument("--seeds", default="0", help="쉼표 구분 잡음 seed 목록. 지표는 seed별 값과 합계를 낸다")
    ap.add_argument("--tracker", choices=["basic", "robust"], default="basic", help="basic: 기존(기본), robust: 개선판")
    ap.add_argument("--warmup", type=int, default=0, help="앞 N 프레임은 지표에서 뺌")
    ap.add_argument("--events", action="store_true", help="ID 바뀜 사건(프레임, 정답, 옛 ID, 새 ID)도 찍음")
    ap.add_argument("--out", help="결과 JSON 경로. 예: out/measure/11_track_two_cam_basic.json")
    # ---- 카메라별 시각 어긋남 평가(하나라도 주면 FrameSyncer로 묶어 평가. 기본은 예전 방식) ----
    ap.add_argument("--cam-skew-ms", default=None, help="카메라별 시각 어긋남(ms, 쉼표 구분). 예: 0,0,50,50")
    ap.add_argument("--cam-drop", default=None, help="카메라별 프레임 누락 확률. 예: 0,0,0,0.05")
    ap.add_argument("--cam-jitter-ms", default=None, help="카메라별 시각 흔들림 표준편차(ms). 예: 0,0,1,1")
    ap.add_argument("--sync-tol-ms", type=float, default=25.0, help="묶음 허용 오차(ms). 기본 25")
    ap.add_argument("--sync-miss", choices=["hold", "blank"], default="hold", help="허용 오차를 넘으면 이전 프레임(hold) 또는 빈 칸(blank)")
    ap.add_argument("--skew-comp", action="store_true", help="묶음의 시각 차이만큼 물체 위치를 트랙 속도로 옮김")
    ap.add_argument("--oncoming", action="store_true", help="시각 어긋남 평가에 마주 오는 차(상대 25 m/s) 한 대를 더함")
    ap.add_argument("--vehicle-center", choices=["ray", "box"], default="ray",
                    help="차 중심 추정. ray: 시선 방향 길이 절반(기본, 예전과 같음), box: 보는 방향에 따라 길이나 폭 절반")
    args = ap.parse_args()
    sync_mode = any(v is not None for v in (args.cam_skew_ms, args.cam_drop, args.cam_jitter_ms)) or args.skew_comp \
        or args.oncoming
    sync_summ = {}

    per_seed = {}
    total = {"id_switches": 0, "gt_frames": 0, "missed": 0, "dup_frames": 0, "false_tracks": 0, "max_tracks": 0}
    tpo = []
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        if sync_mode:
            n = len(load_config(args.config)[0])
            m, sync_summ[s] = run_sim_sync(
                args.config, args.fps, args.frames, s, args.tracker, args.warmup,
                per_camera(args.cam_skew_ms, n), per_camera(args.cam_drop, n), per_camera(args.cam_jitter_ms, n),
                args.sync_tol_ms, args.sync_miss, args.skew_comp, args.oncoming, args.vehicle_center,
            )
        else:
            m = run_sim(args.config, args.fps, args.frames, s, args.tracker, args.warmup, args.vehicle_center)
        summ = m.summary()
        if args.events:
            summ["events"] = m.events
        per_seed[s] = summ
        total["id_switches"] += m.switches
        total["gt_frames"] += m.gt_frames
        total["missed"] += m.missed
        total["dup_frames"] += m.dup
        total["false_tracks"] += m.false_tracks
        total["max_tracks"] = max(total["max_tracks"], summ["tracks_per_object_max"] or 0)
        tpo.append(summ["tracks_per_object_mean"] or 0.0)
    total["miss_ratio"] = round(total["missed"] / total["gt_frames"], 4) if total["gt_frames"] else None
    total["tracks_per_object_mean"] = round(float(np.mean(tpo)), 3) if tpo else None
    rep = {
        "config": args.config,
        "fps": args.fps,
        "frames": args.frames,
        "tracker": args.tracker,
        "vehicle_center": args.vehicle_center,
        "total": total,
        "per_seed": per_seed,
    }
    if sync_mode:
        rep["sync"] = {"cam_skew_ms": args.cam_skew_ms, "cam_drop": args.cam_drop, "cam_jitter_ms": args.cam_jitter_ms,
                       "tol_ms": args.sync_tol_ms, "miss": args.sync_miss, "skew_comp": args.skew_comp,
                       "oncoming": args.oncoming,
                       "per_seed": sync_summ}
    text = json.dumps(rep, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
