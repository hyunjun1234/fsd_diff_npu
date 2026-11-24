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
from realityview.sim import SimBackend, SimWorld  # noqa: E402

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


def run_sim(config: str, fps: float, frames: int, seed: int, tracker: str, warmup: int = 0) -> TrackingMetrics:
    cams, grid, cfg = load_config(config)
    scale = float((cfg.get("scene") or {}).get("scale", 1.0))
    world = SimWorld(cams)
    pipe = RealityViewPipeline(
        cams, SimBackend(world, seed=seed), grid, show_ground=False, show_cameras=False, scale=scale, tracker=tracker
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
    args = ap.parse_args()

    per_seed = {}
    total = {"id_switches": 0, "gt_frames": 0, "missed": 0, "dup_frames": 0, "false_tracks": 0, "max_tracks": 0}
    tpo = []
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        m = run_sim(args.config, args.fps, args.frames, s, args.tracker, args.warmup)
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
        "total": total,
        "per_seed": per_seed,
    }
    text = json.dumps(rep, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
