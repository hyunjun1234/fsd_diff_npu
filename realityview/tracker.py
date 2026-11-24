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
#
# 빠른 물체 개선(마주 오는 차, 상대 25 m/s. out/measure/17_tracking_fast.md)
#   원인: 빠른 트랙이 느린 옆 트랙 곁을 지나는 순간 (e)가 "겹친 트랙"으로 보고 빠른 트랙을 지웠고,
#         새로 만든 트랙은 속도 0에서 아주 천천히(beta 0.1) 따라가서 다음 측정이 짝짓기 거리 밖에 떨어졌다.
#   고침(robust 판에만 들어간다. basic은 그대로)
#   f) 트랙마다 칼만 필터(위치, 속도 상태와 공분산)를 둔다. 측정 잡음은 시선 방향으로 거리 제곱에 비례해 커진다
#      (sim 실측: 시선 방향 표준편차 10m 0.7m, 25m 2.2m, 35m 3.8m, 옆 방향 0.3~0.5m).
#   g) 짝짓기 기준은 공분산으로 잰 마할라노비스 거리(카이제곱 문턱)이고, 옆 방향은 gate(2.5m)로 한 번 더 막는다.
#      그래서 처음 보는 트랙(속도 불확실)이나 잠깐 놓친 트랙은 문턱이 저절로 넓고, 오래 본 트랙은 좁다.
#   h) 일대일 짝짓기는 가까운 쌍부터가 아니라 헝가리안(전체 비용 최소) 방식이다. scipy 없이 직접 구현했다.
#   i) (e)는 두 트랙 모두 속도가 자리 잡았고 속도 차이가 merge_dv 넘게 나면 합치지 않는다(서로 스쳐 지나가는 차).


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
    # robust 칼만 필터의 상태 공분산(4x4, 순서 x, y, vx, vy). basic은 쓰지 않아 None
    cov: np.ndarray | None = None

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
        meas_sigma: float = 0.5,  # robust: 측정 잡음 표준편차의 바닥값(m, 옆 방향과 가까운 시선 방향)
        range_noise: float = 0.003,  # robust: 시선 방향 잡음 = hypot(meas_sigma, 이 값 x 거리^2)
        accel_sigma: float = 4.0,  # robust: 등속 모델이 놓치는 가속도 표준편차(m/s^2). 클수록 예측을 덜 믿는다
        init_speed_sigma: float = 15.0,  # robust: 새 트랙 앞뒤(X) 속도의 처음 불확실성(m/s). 마주 오는 차(25 m/s)도 들어오게
        init_lat_speed_sigma: float = 3.0,  # robust: 새 트랙 옆(Y) 속도의 처음 불확실성(m/s). 차는 대개 도로를 따라 움직인다
        coast_misses: int = 10,  # robust: 속도가 자리 잡은 트랙은 이만큼 연속으로 놓쳐도 예측으로 남긴다(카메라 사이 사각지대)
        gate_chi2: float = 13.8,  # robust: 마할라노비스 거리 제곱 문턱(자유도 2, 99.9%)
        merge_dv: float = 5.0,  # robust: 이보다 속도가 다른 두 트랙은 겹쳐도 합치지 않는다(m/s)
        settled_hits: int = 5,  # robust: 이만큼 짝지어진 트랙의 속도는 자리 잡았다고 본다
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
        self.meas_sigma = meas_sigma * scale
        # 거리 제곱 항: 거리가 축척만큼 작아지면 잡음도 축척만큼 작아지도록 1/scale을 곱해 둔다
        self.range_noise = range_noise / scale
        self.accel_sigma = accel_sigma * scale
        self.init_speed_sigma = init_speed_sigma * scale
        self.init_lat_speed_sigma = init_lat_speed_sigma * scale
        self.coast_misses = coast_misses
        self.gate_chi2 = gate_chi2
        self.merge_dv = merge_dv * scale
        self.settled_hits = settled_hits
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

    # ---- robust 칼만 필터: 측정 잡음, 예측, 갱신 ---------------------------
    def _meas_cov(self, xy: np.ndarray) -> np.ndarray:
        """측정 하나의 잡음 공분산(2x2). 시선 방향은 거리 제곱에 비례해 커지고 옆 방향은 meas_sigma 그대로다.

        박스 아래 끝 화소 잡음이 지면에서 시선 방향 오차가 되는데, 그 크기가 대략 거리^2 / 카메라 높이에 비례한다.
        카메라가 원점 근처에 있어 원점에서 측정점 쪽을 시선 방향으로 어림한다(_gate_dist와 같은 가정).
        """
        rng = float(np.linalg.norm(xy))
        s0 = self.meas_sigma
        if rng < 1e-6:
            return np.eye(2) * s0 * s0
        u = xy / rng
        n = np.array([-u[1], u[0]])
        s_along = float(np.hypot(s0, self.range_noise * rng * rng))
        return s_along * s_along * np.outer(u, u) + s0 * s0 * np.outer(n, n)

    def _kf_predict(self, t: Track, dt: float) -> None:
        """등속 모델로 dt초 뒤 상태와 공분산을 만든다. 공분산은 가속도 잡음만큼 커진다."""
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.accel_sigma**2
        a, b, c = dt**4 / 4.0, dt**3 / 2.0, dt**2  # 가속도 잡음이 위치, 위치-속도, 속도에 주는 몫
        Q = q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        t.xy = t.xy + t.vel * dt
        t.cov = F @ t.cov @ F.T + Q

    def _innovation(self, t: Track, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(예측 오차 y = 측정 - 예측 위치, 그 공분산 S = 위치 공분산 + 측정 잡음)."""
        return z - t.xy, t.cov[:2, :2] + self._meas_cov(z)

    def _kf_update(self, t: Track, z: np.ndarray) -> None:
        """측정 z로 위치와 속도를 고친다. 칼만 이득은 공분산이 큰(덜 믿는) 쪽을 더 많이 움직인다."""
        y, S = self._innovation(t, z)
        K = t.cov[:, :2] @ np.linalg.inv(S)  # 4x2
        x = np.concatenate([t.xy, t.vel]) + K @ y
        IKH = np.eye(4)
        IKH[:, :2] -= K
        # 조셉 형태: 수치 오차가 쌓여도 공분산이 대칭, 양의 정부호로 남는다
        R = self._meas_cov(z)
        t.cov = IKH @ t.cov @ IKH.T + K @ R @ K.T
        t.xy, t.vel = x[:2], x[2:]
        spd = float(np.linalg.norm(t.vel))
        if spd > self.max_speed:
            t.vel = t.vel * (self.max_speed / spd)

    def _init_cov(self, z: np.ndarray) -> np.ndarray:
        """새 트랙 공분산: 위치는 측정 잡음만큼, 속도는 앞뒤 init_speed_sigma, 옆 init_lat_speed_sigma만큼 모른다.

        옆 속도를 작게 두는 까닭: 처음 두 측정 사이 옆으로 1m 튄 잡음이 옆 속도 10 m/s로 바뀌어
        다음 예측이 옆 차선으로 날아가는 것을 막는다(sim에서 뒤 카메라가 처음 본 마주 오는 차).
        """
        P = np.zeros((4, 4))
        P[:2, :2] = self._meas_cov(z)
        P[2, 2] = self.init_speed_sigma**2
        P[3, 3] = self.init_lat_speed_sigma**2
        return P

    # ---- robust 짝짓기 -------------------------------------------------------
    def _pair_cost(self, t: Track, o: GroundObject, widen: float, R: np.ndarray | None = None) -> float | None:
        """트랙 t와 측정 o를 짝지을 비용. 문턱 밖이면 None.

        비용은 마할라노비스 거리 제곱 + ln det(S). 뒤 항은 불확실한 트랙이 넓은 문턱으로 아무 측정이나
        가져가지 않게 하는 몫이다(확률 밀도의 음의 로그와 같은 꼴).
        문턱은 두 가지를 모두 넘어야 한다.
          1) 마할라노비스 거리 제곱 < gate_chi2 x widen^2
          2) 옆 방향(시선에 수직) 차이 < gate x widen  - 나란히 달리는 옆 차선 차(3.5m)와 바뀌지 않게
        R: 이 측정의 잡음 공분산을 미리 구해 두었으면 넘긴다(짝 후보마다 다시 구하지 않게).
        파이 CPU에서 물체가 20개쯤이면 짝 후보가 수백 개라, 2x2 역행렬과 행렬식을 numpy 대신 식으로 바로 푼다.
        """
        if not _compatible(t.label, o.label):
            return None
        if R is None:
            R = self._meas_cov(o.xy)
        yx = float(o.xy[0] - t.xy[0])
        yy = float(o.xy[1] - t.xy[1])
        P = t.cov
        a = float(P[0, 0] + R[0, 0])
        b = float(P[0, 1] + R[0, 1])
        d = float(P[1, 1] + R[1, 1])
        lim = self.gate_chi2 * widen * widen
        # 빠른 거르기: y'S^-1 y >= |y|^2 / (S의 가장 큰 고유값) >= |y|^2 / trace(S). 이것만으로 문턱 밖이면 끝
        if (yx * yx + yy * yy) >= lim * (a + d):
            return None
        det = a * d - b * b
        if det <= 1e-12:
            return None
        d2 = (d * yx * yx - 2.0 * b * yx * yy + a * yy * yy) / det
        if d2 >= lim:
            return None
        rng = float(np.hypot(o.xy[0], o.xy[1]))
        if rng > 1e-6:
            across = abs((yx * -float(o.xy[1]) + yy * float(o.xy[0])) / rng)
            if across >= self.gate * widen:
                return None
        return d2 + float(np.log(det))

    def _associate_kf(self, objs, cand_t, cand_o, widen: float) -> list[tuple[int, int]]:
        """후보 트랙과 측정을 헝가리안 방식으로 일대일 짝짓는다(비용 합이 가장 작은 조합)."""
        if not cand_t or not cand_o:
            return []
        INF = 1e9
        Rs = {oi: self._meas_cov(objs[oi].xy) for oi in cand_o}
        C = np.full((len(cand_t), len(cand_o)), INF)
        for a, ti in enumerate(cand_t):
            for b, oi in enumerate(cand_o):
                c = self._pair_cost(self.tracks[ti], objs[oi], widen, Rs[oi])
                if c is not None:
                    C[a, b] = c
        out = []
        # 문턱 안 짝이 하나도 없는 행과 열은 빼고 푼다(헝가리안은 크기의 세제곱으로 느려진다)
        rows = [a for a in range(C.shape[0]) if (C[a] < INF).any()]
        cols = [b for b in range(C.shape[1]) if (C[:, b] < INF).any()]
        if not rows or not cols:
            return []
        sub = C[np.ix_(rows, cols)]
        for a, b in hungarian(sub):
            if sub[a, b] < INF:  # 문턱 밖 칸으로 억지로 짝지어진 것은 버린다
                out.append((cand_t[rows[a]], cand_o[cols[b]]))
        return out

    def _update_robust(self, objs: list[GroundObject], dt: float) -> list[Track]:
        dt = max(dt, 1e-3)
        # 0) 모든 트랙을 이번 시각으로 예측한다(놓친 트랙도 예측 위치로 옮겨 두고 공분산은 커진다)
        for t in self.tracks:
            if t.cov is None:
                t.cov = self._init_cov(t.xy)
            self._kf_predict(t, dt)
        all_t = list(range(len(self.tracks)))
        all_o = list(range(len(objs)))
        # 1차: 보통 문턱. 2차: 남은 것끼리 두 배 문턱(c)
        matches = self._associate_kf(objs, all_t, all_o, widen=1.0)
        left_t = [ti for ti in all_t if ti not in {m[0] for m in matches}]
        left_o = [oi for oi in all_o if oi not in {m[1] for m in matches}]
        matches += self._associate_kf(objs, left_t, left_o, widen=2.0)
        used_t = {m[0] for m in matches}
        used_o = {m[1] for m in matches}
        for ti, oi in matches:
            t, o = self.tracks[ti], objs[oi]
            self._kf_update(t, o.xy)
            t.size = o.size
            t.score = o.score
            t.label = o.label
            t.hits += 1
            t.misses = 0
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1  # 위치는 이미 예측 위치다
        # (d) 남은 측정이 이미 있는 트랙(이번에 짝지은 것, 또는 확정된 것) 곁이면 새로 만들지 않는다
        guards = [t for ti, t in enumerate(self.tracks) if ti in used_t or t.hits >= self.min_hits]
        for oi, o in enumerate(objs):
            if oi in used_o:
                continue
            dup = any(_compatible(g.label, o.label) and self._gate_dist(g.xy, o.xy) < 1.0 for g in guards)
            if not dup:
                self._spawn(o)
                self.tracks[-1].cov = self._init_cov(o.xy)
        self._merge_overlapping()
        return self._finish()

    def _merge_overlapping(self) -> None:
        """(e) 같은 물체에 붙은 트랙 두 개가 짝짓기 거리 안에 겹쳐 있으면 오래된 쪽 하나만 남긴다.

        카메라 4대 sim에서 두 카메라가 같은 차를 서로 다른 자리로 투영해 병합(merge_cross_camera)을
        빠져나가면, 측정 두 개가 트랙 두 개를 따로 붙잡고 계속 같이 산다. 그 경우를 정리한다.
        옆 차선(3.5m) 차는 옆 방향 거리 gate(2.5m) 밖이라 합쳐지지 않는다.
        (i) 두 트랙 모두 속도가 자리 잡았고(settled_hits 이상) 속도 차이가 merge_dv보다 크면 서로 다른 물체가
            스쳐 지나가는 중이라 합치지 않는다. 막 생긴 트랙은 속도를 아직 모르므로 예전처럼 합친다.
        """
        order = sorted(range(len(self.tracks)), key=lambda i: (-self.tracks[i].hits, self.tracks[i].tid))
        keep: list[Track] = []
        for i in order:
            t = self.tracks[i]
            if any(self._same_object(k, t) for k in keep):
                continue
            keep.append(t)
        if len(keep) != len(self.tracks):
            ids = {id(k) for k in keep}
            self.tracks = [t for t in self.tracks if id(t) in ids]  # 원래 순서는 그대로

    def _same_object(self, a: Track, b: Track) -> bool:
        if not _compatible(a.label, b.label) or self._gate_dist(a.xy, b.xy) >= 1.0:
            return False
        settled = a.hits >= self.settled_hits and b.hits >= self.settled_hits
        if not settled:
            return True
        dv = a.vel - b.vel
        if float(np.linalg.norm(dv)) <= self.merge_dv:
            return True
        # 속도 차이가 속도 불확실성에 견줘서도 뚜렷해야(카이제곱 문턱) 다른 물체로 본다.
        # 먼 물체는 시선 방향 잡음이 커서 같은 물체의 두 트랙도 속도가 5 m/s 넘게 다를 수 있다
        if a.cov is None or b.cov is None:
            return False
        Sv = a.cov[2:, 2:] + b.cov[2:, 2:]
        try:
            d2 = float(dv @ np.linalg.inv(Sv) @ dv)
        except np.linalg.LinAlgError:
            return True
        return d2 < self.gate_chi2

    # ---- 공통 ------------------------------------------------------------
    def _spawn(self, o: GroundObject) -> None:
        self.tracks.append(Track(self._next_id, o.label, o.xy.copy(), size=o.size, score=o.score, min_speed=0.8 * self.scale))
        self._next_id += 1

    def _finish(self) -> list[Track]:
        if self.mode == "robust":
            # 속도가 자리 잡은 트랙은 더 오래 예측으로 남긴다. 앞 카메라에서 뒤 카메라로 넘어가는 사이
            # 어느 카메라에도 안 보이는 구간(15fps에서 7프레임 안팎)을 마주 오는 차가 건너갈 수 있게
            self.tracks = [
                t for t in self.tracks
                if t.misses <= (self.coast_misses if t.hits >= self.settled_hits else self.max_misses)
            ]
        else:
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        # 화면에는 확정되었고 최근에 다시 관측된 트랙만 보낸다(가려진 직후 1프레임은 유지)
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses <= 1]


def _compatible(a: str, b: str) -> bool:
    veh = {"car", "truck", "bus"}
    return a == b or (a in veh and b in veh)


# ---------------------------------------------------------------------------
# 헝가리안 방식 일대일 짝짓기(scipy 없이)
# ---------------------------------------------------------------------------
def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """비용 행렬(행 n x 열 m)에서 합이 가장 작은 일대일 짝 목록 [(행, 열), ...]을 돌려준다.

    짝 수는 min(n, m)개다. 파이 .venv에는 scipy가 없어 직접 구현했다(최단 증가 경로 방식, O(n^2 m)).
    행이 열보다 많으면 뒤집어 풀고 결과를 다시 뒤집는다.
    """
    C = np.asarray(cost, dtype=np.float64)
    if C.size == 0:
        return []
    if C.shape[0] > C.shape[1]:
        return [(r, c) for c, r in hungarian(C.T)]
    n, m = C.shape
    INF = float("inf")
    # u, v: 행과 열의 잠재값. p[j]: 열 j에 붙은 행(1부터 센다, 0은 비어 있음). way: 경로 되짚기용
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # 찾은 증가 경로를 따라 짝을 바꾼다
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)
