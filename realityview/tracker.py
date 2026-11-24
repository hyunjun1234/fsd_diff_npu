from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import GroundObject

# BEV 평면 위의 간단한 다중 객체 추적(최근접 매칭 + 등속 모델 + 지수 평활).
#
# 방식은 두 가지다.
#   basic (기본): 예전과 똑같다. 짝짓기 거리 2.5m 고정, 속도는 "이번 측정 - 지난 위치"를 dt로 나눈 값의 반을 섞는다.
#   robust      : 먼 물체에서 한 물체가 트랙 여러 개로 쪼개지던 문제(3D 시뮬레이션 영상의 ID 32, 37)를 고친 판.
#
# 쪼개짐 원인(sim, configs/two_cam.yaml, 15 fps에서 확인)
#   1) 먼 물체는 박스 아래 끝 1~2화소 잡음이 지면에서 시선 방향으로 수 m 오차가 된다(25m에서 화소당 약 1.5m).
#   2) basic의 속도 갱신은 그 오차를 dt(1/15초)로 나눠 섞으므로 한 번 튄 측정이 수십 m/s 속도가 된다.
#      다음 예측이 고정 짝짓기 거리 2.5m 밖으로 날아가 새 트랙이 생기고, 옛 트랙은 몇 프레임 더 남아
#      서로 측정을 번갈아 받으며 둘 다 화면에 나온다.
# robust의 고침
#   a) 속도는 예측 오차에 작은 비율(beta)만 반영하는 알파-베타 필터로 갱신하고 최대 속도로 자른다.
#   b) 짝짓기 거리를 시선 방향(원점에서 물체 쪽)으로만 거리에 비례해 넓힌다. 옆 방향은 그대로 두어
#      나란히 달리는 옆 차선 차와 바뀌지 않게 한다.
#   c) 첫 짝짓기에서 남은 측정과 트랙은 거리를 두 배로 늘려 한 번 더 짝짓는다(잠깐 놓친 트랙 되살리기).
#   d) 그래도 남은 측정이 이미 있는 트랙 바로 곁이면 새 트랙을 만들지 않는다(같은 물체의 중복 측정).
#   e) 짝짓기 거리 안에 겹친 트랙 두 개는 오래된 쪽만 남긴다(카메라 겹침 영역에서 병합을 빠져나간 경우).


# ---------------------------------------------------------------------------
# 트랙 하나
# ---------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    label: str
    xy: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    size: tuple[float, float] = (1.0, 1.0)
    hits: int = 1
    misses: int = 0
    score: float = 0.0
    min_speed: float = 0.8  # 이 속도(m/s) 이상일 때만 진행 방향을 믿는다. 모형이면 축척만큼 작게

    @property
    def heading(self) -> float | None:
        """속도가 충분하면 진행 방향(rad, 차량 X축 기준), 아니면 None."""
        if np.linalg.norm(self.vel) < self.min_speed:
            return None
        return float(np.arctan2(self.vel[1], self.vel[0]))


# ---------------------------------------------------------------------------
# 추적기
# ---------------------------------------------------------------------------
class BevTracker:
    def __init__(
        self,
        gate: float = 2.5,
        max_misses: int = 5,
        min_hits: int = 2,
        alpha: float = 0.5,
        scale: float = 1.0,
        mode: str = "basic",
        beta: float = 0.1,  # robust: 예측 오차 중 속도에 반영할 비율(작을수록 튀는 측정에 둔하다)
        range_gate: float = 0.15,  # robust: 시선 방향 짝짓기 거리 = max(gate, 거리 x 이 값)
        max_speed: float = 60.0,  # robust: 상대 속도 상한(m/s, 실차 기준). 넘으면 이 크기로 자른다
    ):
        if mode not in ("basic", "robust"):
            raise ValueError(f"mode는 basic 또는 robust여야 합니다: {mode}")
        # scale: 장면 축척. 모형(예: 0.15)이면 짝짓기 거리(gate)와 방향 판단 속도를 같은 비율로 줄인다
        self.scale = scale
        self.gate = gate * scale
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.alpha = alpha
        self.mode = mode
        self.beta = beta
        self.range_gate = range_gate  # 거리에 곱하는 비율이라 축척과 무관하다(거리 자체가 이미 축척만큼 작다)
        self.max_speed = max_speed * scale
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, objs: list[GroundObject], dt: float) -> list[Track]:
        if self.mode == "robust":
            return self._update_robust(objs, dt)
        return self._update_basic(objs, dt)

    # ---- basic: 예전 동작 그대로 ----------------------------------------
    def _update_basic(self, objs: list[GroundObject], dt: float) -> list[Track]:
        dt = max(dt, 1e-3)
        preds = [t.xy + t.vel * dt for t in self.tracks]
        pairs = []
        for ti, p in enumerate(preds):
            for oi, o in enumerate(objs):
                d = float(np.linalg.norm(p - o.xy))
                if d < self.gate and _compatible(self.tracks[ti].label, o.label):
                    pairs.append((d, ti, oi))
        pairs.sort()
        used_t, used_o = set(), set()
        for d, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            t, o = self.tracks[ti], objs[oi]
            new_xy = self.alpha * o.xy + (1 - self.alpha) * preds[ti]
            t.vel = 0.5 * t.vel + 0.5 * (o.xy - t.xy) / dt
            t.xy = new_xy
            t.size = o.size
            t.score = o.score
            t.label = o.label
            t.hits += 1
            t.misses = 0
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.xy = preds[ti]
                t.misses += 1
        for oi, o in enumerate(objs):
            if oi not in used_o:
                self._spawn(o)
        return self._finish()

    # ---- robust: 먼 물체 쪼개짐 개선판 ------------------------------------
    def _gate_dist(self, pred: np.ndarray, meas: np.ndarray, widen: float = 1.0) -> float:
        """예측과 측정 사이의 '정규화 거리'. 1보다 작으면 짝지을 수 있다.

        측정점 방향(원점 -> 측정)을 시선 방향으로 보고, 차이를 시선 방향과 옆 방향으로 나눈다.
        시선 방향 허용치는 거리에 비례해 커지고(먼 물체는 그쪽 오차가 크다), 옆 방향은 gate 그대로다.
        카메라가 원점 근처(수 m 안)에 있어 원점 기준 방향으로 시선 방향을 어림한다.
        """
        diff = meas - pred
        rng = float(np.linalg.norm(meas))
        if rng < 1e-6:
            return float(np.linalg.norm(diff)) / (self.gate * widen)
        u = meas / rng  # 시선 방향 단위 벡터
        along = float(diff @ u)
        across = float(diff[0] * -u[1] + diff[1] * u[0])
        g_along = max(self.gate, self.range_gate * rng) * widen
        g_across = self.gate * widen
        return float(np.hypot(along / g_along, across / g_across))

    def _associate(self, preds, objs, cand_t, cand_o, widen: float) -> list[tuple[int, int]]:
        """후보 트랙과 측정을 정규화 거리가 작은 쌍부터 일대일로 짝짓는다."""
        pairs = []
        for ti in cand_t:
            for oi in cand_o:
                if not _compatible(self.tracks[ti].label, objs[oi].label):
                    continue
                d = self._gate_dist(preds[ti], objs[oi].xy, widen)
                if d < 1.0:
                    pairs.append((d, ti, oi))
        pairs.sort()
        used_t, used_o, out = set(), set(), []
        for _, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            out.append((ti, oi))
        return out

    def _update_robust(self, objs: list[GroundObject], dt: float) -> list[Track]:
        dt = max(dt, 1e-3)
        preds = [t.xy + t.vel * dt for t in self.tracks]
        all_t = list(range(len(self.tracks)))
        all_o = list(range(len(objs)))
        # 1차: 보통 거리. 2차: 남은 것끼리 두 배 거리(c)
        matches = self._associate(preds, objs, all_t, all_o, widen=1.0)
        left_t = [ti for ti in all_t if ti not in {m[0] for m in matches}]
        left_o = [oi for oi in all_o if oi not in {m[1] for m in matches}]
        matches += self._associate(preds, objs, left_t, left_o, widen=2.0)
        used_t = {m[0] for m in matches}
        used_o = {m[1] for m in matches}
        for ti, oi in matches:
            t, o = self.tracks[ti], objs[oi]
            resid = o.xy - preds[ti]  # 예측 오차
            t.xy = preds[ti] + self.alpha * resid
            # (a) 알파-베타 필터: 예측 오차의 beta만 속도에 더한다. 튄 측정 하나가 속도를 망치지 않게
            t.vel = t.vel + self.beta * resid / dt
            spd = float(np.linalg.norm(t.vel))
            if spd > self.max_speed:
                t.vel = t.vel * (self.max_speed / spd)
            t.size = o.size
            t.score = o.score
            t.label = o.label
            t.hits += 1
            t.misses = 0
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.xy = preds[ti]
                t.misses += 1
        # (d) 남은 측정이 이미 있는 트랙(이번에 짝지은 것, 또는 확정된 것) 곁이면 새로 만들지 않는다
        guards = [
            t.xy for ti, t in enumerate(self.tracks) if ti in used_t or t.hits >= self.min_hits
        ]
        guard_labels = [
            t.label for ti, t in enumerate(self.tracks) if ti in used_t or t.hits >= self.min_hits
        ]
        for oi, o in enumerate(objs):
            if oi in used_o:
                continue
            dup = any(
                _compatible(lab, o.label) and self._gate_dist(g, o.xy) < 1.0 for g, lab in zip(guards, guard_labels)
            )
            if not dup:
                self._spawn(o)
        self._merge_overlapping()
        return self._finish()

    def _merge_overlapping(self) -> None:
        """(e) 같은 물체에 붙은 트랙 두 개가 짝짓기 거리 안에 겹쳐 있으면 오래된 쪽 하나만 남긴다.

        카메라 4대 sim에서 두 카메라가 같은 차를 서로 다른 자리로 투영해 병합(merge_cross_camera)을
        빠져나가면, 측정 두 개가 트랙 두 개를 따로 붙잡고 계속 같이 산다. 그 경우를 정리한다.
        옆 차선(3.5m) 차는 옆 방향 거리 gate(2.5m) 밖이라 합쳐지지 않는다.
        """
        order = sorted(range(len(self.tracks)), key=lambda i: (-self.tracks[i].hits, self.tracks[i].tid))
        keep: list[Track] = []
        for i in order:
            t = self.tracks[i]
            if any(_compatible(k.label, t.label) and self._gate_dist(k.xy, t.xy) < 1.0 for k in keep):
                continue
            keep.append(t)
        if len(keep) != len(self.tracks):
            ids = {id(k) for k in keep}
            self.tracks = [t for t in self.tracks if id(t) in ids]  # 원래 순서는 그대로

    # ---- 공통 ------------------------------------------------------------
    def _spawn(self, o: GroundObject) -> None:
        self.tracks.append(Track(self._next_id, o.label, o.xy.copy(), size=o.size, score=o.score, min_speed=0.8 * self.scale))
        self._next_id += 1

    def _finish(self) -> list[Track]:
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        # 화면에는 확정되었고 최근에 다시 관측된 트랙만 보낸다(가려진 직후 1프레임은 유지)
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses <= 1]


def _compatible(a: str, b: str) -> bool:
    veh = {"car", "truck", "bus"}
    return a == b or (a in veh and b in veh)
