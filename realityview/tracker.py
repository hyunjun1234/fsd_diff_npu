"""BEV 평면 위의 간단한 다중 객체 추적(최근접 매칭 + 등속 모델 + 지수 평활)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import GroundObject


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


class BevTracker:
    def __init__(
        self, gate: float = 2.5, max_misses: int = 5, min_hits: int = 2, alpha: float = 0.5, scale: float = 1.0
    ):
        # scale: 장면 축척. 모형(예: 0.15)이면 짝짓기 거리(gate)와 방향 판단 속도를 같은 비율로 줄인다
        self.scale = scale
        self.gate = gate * scale
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.alpha = alpha
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, objs: list[GroundObject], dt: float) -> list[Track]:
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
                self.tracks.append(
                    Track(self._next_id, o.label, o.xy.copy(), size=o.size, score=o.score, min_speed=0.8 * self.scale)
                )
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        # 화면에는 확정되었고 최근에 다시 관측된 트랙만 보낸다(가려진 직후 1프레임은 유지)
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses <= 1]


def _compatible(a: str, b: str) -> bool:
    veh = {"car", "truck", "bus"}
    return a == b or (a in veh and b in veh)
