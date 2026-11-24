from __future__ import annotations

import dataclasses
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .bev import BevGrid

# IPM 바닥 영상(탑뷰)에서 흰색과 노란색 차선을 찾는 고전 방식 차선 검출.
#
# 파이에 차선 모델 HEF(ufld 계열)가 없어서(2026-11-25 /usr/share/hailo-models, models/ 확인) CPU로 한다.
# 한 번의 검출 순서
#   1) 탑뷰 바닥 영상을 분석용 작은 격자(기본 0.1m 칸)로 줄인다(파이 CPU 절약)
#   2) 색 임계값: 밝고 채도 낮은 화소는 흰색, 색상이 노란 범위이고 채도가 높은 화소는 노란색 후보
#   3) 폭 거르기(top-hat)와 기울기: 차선은 좁고 밝은 띠라 "주변보다 밝고(top-hat), 좌우로 밝기가 크게
#      바뀌는(Sobel) 자리" 근처만 남긴다. 넓은 보도나 밝은 벽 면은 여기서 빠진다
#   4) 카메라가 보는 셀(coverage)을 조금 깎아 경계선을 지우고, 도로 마스크가 있으면 조금 넓혀 그 안만 남긴다
#   5) 자차 가까운 띠의 열 히스토그램 봉우리를 차선 출발점으로 잡고, 앞뒤로 슬라이딩 윈도우를 옮기며 화소를 모은다
#   6) 모은 화소로 Y = a X^2 + b X + c (차량 좌표, m) 2차 곡선을 맞춘다
#   7) 곡선을 따라 칸마다 화소가 있는 비율로 실선/점선, 노란 화소 비율로 색을 정한다
# 그다음 LaneSmoother가 프레임 사이에서 곡선을 이어 붙이고 평활한다(깜빡임, 튐 줄이기).
#
# 좌표계는 다른 모듈과 같다: X 전방, Y 좌측(m). 탑뷰 영상의 열이 작을수록 Y가 크다(왼쪽).
# 모든 길이 설정은 실차 m이고, 장면 축척(scale)을 곱해 쓴다(모형에서도 같은 동작).


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class LaneConfig:
    """차선 검출 설정. 길이는 실차 m 기준(검출기가 scale을 곱한다), 색 값은 OpenCV HSV(H 0~180)."""

    cell_m: float = 0.10  # 분석 격자 칸 크기. 탑뷰보다 크면 그 배수로 줄여서 분석한다
    max_x: float = 20.0  # 이 거리(앞뒤)보다 먼 셀은 쓰지 않는다. 먼 IPM 영상은 늘어져 차선이 번진다
    white_v: int = 170  # 흰색: 밝기 하한
    white_s: int = 70  # 흰색: 채도 상한
    yellow_h: tuple = (12, 38)  # 노란색: 색상 범위
    yellow_s: int = 70  # 노란색: 채도 하한
    yellow_v: int = 100  # 노란색: 밝기 하한
    tophat_m: float = 0.6  # top-hat 가로 커널 폭. 차선 폭(0.15m)보다 넉넉히 넓게
    tophat_thr: int = 25  # 주변보다 이만큼 밝아야 차선 후보
    grad_thr: float = 30.0  # 좌우 밝기 기울기(Sobel) 하한. 차선 가장자리에서 크다
    line_w_m: float = 0.15  # 차선 폭(기울기 자리를 이 폭만큼 넓혀 차선 가운데까지 덮는다)
    edge_erode_m: float = 0.4  # 카메라가 보는 영역 경계를 이만큼 깎는다(경계선이 차선으로 잡히지 않게)
    road_dilate_m: float = 0.3  # 도로 마스크를 이만큼 넓힌다(도로 가장자리 차선이 반쯤 잘리지 않게)
    base_x: tuple = (-8.0, 8.0)  # 출발점 히스토그램을 셀 X 범위(자차 앞뒤 가까운 띠)
    min_sep_m: float = 1.2  # 출발점 봉우리 사이 최소 간격
    min_base_px: int = 6  # 봉우리 높이 하한(열 하나에 모인 차선 화소 수)
    window_m: float = 1.0  # 슬라이딩 윈도우 길이(X 방향)
    margin_m: float = 0.5  # 윈도우 반폭(Y 방향)
    min_pix: int = 3  # 윈도우 중심을 옮길 최소 화소 수
    min_support_m: float = 3.0  # 화소가 있는 구간이 이 길이보다 짧으면 버린다
    min_points: int = 15  # 곡선을 맞출 최소 화소 수
    dedup_m: float = 0.6  # X=0 횡위치가 이보다 가까운 두 곡선은 같은 차선으로 보고 하나만 남긴다
    solid_fill: float = 0.75  # 곡선을 따라 화소가 있는 칸의 비율이 이 값 이상이면 실선
    style_min_m: float = 9.0  # 카메라가 보는 구간이 이보다 짧으면 실선/점선을 정하지 않는다(점선 한 주기 약 9m)
    yellow_frac: float = 0.4  # 노란 화소 비율이 이 값 이상이면 노란 차선
    ego_max_m: float = 2.6  # 자차에서 이 거리 안의 가장 가까운 좌우 차선을 내 차선 경계로 본다
    mask_objects: bool = True  # 검출 상자 안 화소가 펴진 자리는 차선 화소에서 뺀다(차체가 IPM에 늘어져 생기는 가짜 선)
    # 프레임 간 평활(LaneSmoother)
    smooth_alpha: float = 0.35  # 새 곡선 반영 비율(0~1). 작을수록 부드럽고 늦다
    hold_frames: int = 5  # 못 찾아도 이 프레임 수까지는 지난 곡선을 보여 준다
    vote_frames: int = 9  # 실선/점선, 색을 다수결로 정할 최근 프레임 수
    match_m: float = 0.7  # 이전 곡선과 같은 차선으로 볼 X=0 횡위치 차이


# 차선 색 이름 -> 화면 색(BGR)
LANE_BGR = {"white": (235, 235, 235), "yellow": (40, 200, 245)}


# ---------------------------------------------------------------------------
# 결과: 차량 좌표계 차선 곡선 한 개
# ---------------------------------------------------------------------------
@dataclass
class LaneCurve:
    """차선 곡선 하나. Y = a X^2 + b X + c (m, 차량 좌표)."""

    coeffs: np.ndarray  # (a, b, c)
    x_range: tuple[float, float]  # 화소가 있던 X 구간(이 안에서만 그린다)
    side: str = "other"  # "left"(내 차선 왼쪽 경계), "right"(오른쪽 경계), "other"
    style: str = "solid"  # "solid"(실선), "dashed"(점선). 검출 직후에는 "unsure"(보인 구간이 짧음)도 있다
    color: str = "white"  # "white" 또는 "yellow"
    fill: float = 1.0  # 곡선을 따라 화소가 있던 칸의 비율(실선/점선 판단 근거)
    n_points: int = 0  # 맞춤에 쓴 화소 수

    def y_at(self, x) -> np.ndarray:
        a, b, c = self.coeffs
        x = np.asarray(x, dtype=np.float64)
        return a * x * x + b * x + c

    @property
    def offset(self) -> float:
        """X=0(뒷바퀴 축)에서의 횡위치 Y(m). +면 왼쪽."""
        return float(self.coeffs[2])

    def points(self, step: float, x_min: float | None = None, x_max: float | None = None) -> np.ndarray:
        """x_range(필요하면 더 좁게) 안에서 step 간격으로 뽑은 곡선 위 점 (N,2: X, Y)."""
        x0 = self.x_range[0] if x_min is None else max(self.x_range[0], x_min)
        x1 = self.x_range[1] if x_max is None else min(self.x_range[1], x_max)
        if x1 <= x0:
            return np.zeros((0, 2))
        n = max(2, int(np.ceil((x1 - x0) / max(step, 1e-6))) + 1)
        xs = np.linspace(x0, x1, n)
        return np.stack([xs, self.y_at(xs)], axis=1)

    def copy(self) -> "LaneCurve":
        return dataclasses.replace(self, coeffs=np.array(self.coeffs, dtype=np.float64))


def assign_sides(lanes: list[LaneCurve], ego_max: float) -> list[LaneCurve]:
    """X=0 횡위치로 내 차선 왼쪽(+Y 중 가장 가까운 것), 오른쪽(-Y 중 가장 가까운 것)을 정한다(그 자리에서 고침).

    자차에서 ego_max보다 먼 차선은 내 차선 경계가 아니라 "other"로 둔다.
    """
    for ln in lanes:
        ln.side = "other"
    left = [ln for ln in lanes if 0.0 < ln.offset <= ego_max]
    right = [ln for ln in lanes if -ego_max <= ln.offset < 0.0]
    if left:
        min(left, key=lambda ln: ln.offset).side = "left"
    if right:
        max(right, key=lambda ln: ln.offset).side = "right"
    return lanes


# ---------------------------------------------------------------------------
# 검출기
# ---------------------------------------------------------------------------
class LaneDetector:
    """탑뷰 바닥 영상 한 장에서 차선 곡선 목록을 찾는다.

    grid: 바닥 영상이 놓인 탑뷰 격자
    coverage: 카메라가 보는 셀(bool, 탑뷰 크기). 없으면 영상 전체를 본다
    scale: 장면 축척. 모든 길이 설정에 곱한다
    """

    def __init__(self, grid: BevGrid, cfg: LaneConfig | None = None, coverage: np.ndarray | None = None,
                 scale: float = 1.0):
        self.grid = grid
        self.cfg = cfg or LaneConfig()
        self.scale = float(scale)
        c, s = self.cfg, self.scale
        # ---- 분석 격자: 탑뷰를 k배 줄인다(칸 크기가 cell_m에 가장 가깝게, 최소 1배) ----
        self.k = max(1, int(round(c.cell_m * s / grid.res)))
        self.w = max(8, grid.width // self.k)
        self.h = max(8, grid.height // self.k)
        self.cell_x = (grid.x_max - grid.x_min) / self.h  # 행 한 칸의 X 길이(m)
        self.cell_y = (grid.y_max - grid.y_min) / self.w  # 열 한 칸의 Y 길이(m)
        self.row_x = grid.x_max - (np.arange(self.h) + 0.5) * self.cell_x  # 행마다 X
        self.col_y = grid.y_max - (np.arange(self.w) + 0.5) * self.cell_y  # 열마다 Y
        px = lambda m: max(1, int(round(m * s / self.cell_y)))  # noqa: E731 - 실차 m -> 분석 칸 수
        self._px = px
        self._tophat_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px(c.tophat_m / 2) + 1, 1))
        self._line_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px(c.line_w_m) + 1, 1))
        self._road_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px(c.road_dilate_m) + 1,) * 2)
        # ---- 쓸 수 있는 셀: 카메라가 보고(경계는 깎음), 너무 멀지 않은 곳 ----
        valid = np.ones((self.h, self.w), np.uint8)
        if coverage is not None:
            cov = cv2.resize(np.asarray(coverage, np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px(c.edge_erode_m) + 1,) * 2)
            valid = cv2.erode(cov, ek, borderValue=0)
        valid[np.abs(self.row_x) > c.max_x * s] = 0
        self.valid = valid
        self.last_mask: np.ndarray | None = None  # 마지막 차선 화소 마스크(확인용, 0/1 흰색=1 노란색=2)

    # ---- 1) ~ 4) 차선 화소 마스크 ----------------------------------------
    def lane_mask(self, ground_bgr: np.ndarray, road_mask: np.ndarray | None = None,
                  occluder: np.ndarray | None = None) -> np.ndarray:
        """분석 격자 크기의 차선 화소 마스크(uint8). 0 = 아님, 1 = 흰색, 2 = 노란색.

        occluder: 탑뷰 크기 마스크(0이 아니면 뺄 자리). 검출 상자 안 화소를 펴 놓은 자리를 준다
        """
        c = self.cfg
        small = ground_bgr if self.k == 1 and ground_bgr.shape[:2] == (self.h, self.w) else \
            cv2.resize(ground_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        # 색 임계값
        white = (V >= c.white_v) & (S <= c.white_s)
        yellow = (H >= c.yellow_h[0]) & (H <= c.yellow_h[1]) & (S >= c.yellow_s) & (V >= c.yellow_v)
        # 폭 거르기: 밝기에서 "가로로 넓은 밝은 면"을 뺀 값(top-hat). 노란 선은 채도도 같이 본다
        th_v = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, self._tophat_k)
        th_s = cv2.morphologyEx(S, cv2.MORPH_TOPHAT, self._tophat_k)
        # 기울기: 좌우 방향 Sobel 크기. 차선 양쪽 가장자리에서 크고, 선 폭만큼 넓혀 가운데까지 덮는다
        gx = np.abs(cv2.Sobel(V, cv2.CV_16S, 1, 0, ksize=3)) + np.abs(cv2.Sobel(S, cv2.CV_16S, 1, 0, ksize=3))
        edge = cv2.dilate((gx >= c.grad_thr).astype(np.uint8), self._line_k)
        keep = edge.astype(bool) & (self.valid > 0)
        white &= keep & (th_v >= c.tophat_thr)
        yellow &= keep & ((th_v >= c.tophat_thr) | (th_s >= c.tophat_thr))
        if road_mask is not None:
            rm = cv2.resize(np.asarray(road_mask, np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            rm = cv2.dilate((rm > 0).astype(np.uint8), self._road_k)
            white &= rm > 0
            yellow &= rm > 0
        if occluder is not None:
            oc = cv2.resize(np.asarray(occluder, np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) == 0
            white &= oc
            yellow &= oc
        out = np.zeros((self.h, self.w), np.uint8)
        out[white] = 1
        out[yellow] = 2  # 둘 다 해당하면 노란색으로 본다(밝은 노란 선은 흰색 조건도 맞을 수 있다)
        return out

    # ---- 5) 출발점 -------------------------------------------------------
    def _bases(self, mask: np.ndarray) -> list[int]:
        """자차 앞뒤 가까운 띠의 열 히스토그램 봉우리(열 번호). 큰 봉우리부터, min_sep 안의 작은 것은 버린다."""
        c, s = self.cfg, self.scale
        rows = (self.row_x >= c.base_x[0] * s) & (self.row_x <= c.base_x[1] * s)
        hist = np.count_nonzero(mask[rows] > 0, axis=0).astype(np.float32)
        hist = cv2.GaussianBlur(hist.reshape(1, -1), (5, 1), 0).ravel()
        sep = self._px(c.min_sep_m)
        bases: list[int] = []
        for col in np.argsort(-hist):
            if hist[col] < c.min_base_px:
                break
            if all(abs(int(col) - b) >= sep for b in bases):
                bases.append(int(col))
        return bases

    # ---- 5) ~ 7) 한 장 검출 ----------------------------------------------
    def detect(self, ground_bgr: np.ndarray | None, road_mask: np.ndarray | None = None,
               occluder: np.ndarray | None = None) -> list[LaneCurve]:
        """바닥 영상 한 장의 차선 곡선 목록(평활 전). 바닥 영상이 없으면 빈 목록."""
        if ground_bgr is None:
            return []
        c, s = self.cfg, self.scale
        mask = self.lane_mask(ground_bgr, road_mask, occluder)
        self.last_mask = mask
        rr, cc = np.nonzero(mask)
        if rr.size < c.min_points:
            return []
        xs, ys = self.row_x[rr], self.col_y[cc]
        is_yel = mask[rr, cc] == 2
        win = c.window_m * s
        margin = c.margin_m * s
        # 화소를 X 윈도우 번호로 미리 나눈다(윈도우마다 전체 화소를 훑지 않게)
        x_lo = -c.max_x * s
        n_win = int(np.ceil(2 * c.max_x * s / win))
        wid = np.clip(((xs - x_lo) / win).astype(np.int64), 0, n_win - 1)
        order = np.argsort(wid, kind="stable")
        starts = np.searchsorted(wid[order], np.arange(n_win + 1))
        zero_w = int(np.clip((0.0 - x_lo) / win, 0, n_win - 1))  # X=0이 든 윈도우

        lanes: list[LaneCurve] = []
        for b in self._bases(mask):
            y0 = float(self.col_y[b])
            picked: list[np.ndarray] = []
            # 앞(+X)과 뒤(-X)로 나눠 X=0에서 출발한다. 사각지대(차 옆)에서는 화소가 없어도 중심을 그대로 들고 간다
            for direction in (1, -1):
                yc = y0
                rng = range(zero_w, n_win) if direction > 0 else range(zero_w - 1, -1, -1)
                for w_i in rng:
                    idx = order[starts[w_i]:starts[w_i + 1]]
                    if idx.size == 0:
                        continue
                    sel = idx[np.abs(ys[idx] - yc) <= margin]
                    if sel.size >= c.min_pix:
                        yc = float(np.mean(ys[sel]))
                        picked.append(sel)
            if not picked:
                continue
            sel = np.concatenate(picked)
            if sel.size < c.min_points:
                continue
            px_x, px_y = xs[sel], ys[sel]
            x0, x1 = float(px_x.min()), float(px_x.max())
            if x1 - x0 < c.min_support_m * s:
                continue
            # 6) 2차 곡선 맞춤. 구간이 짧으면 곡률이 튀므로 1차로 맞춘다
            deg = 2 if (x1 - x0) >= 6.0 * s else 1
            coef = np.polyfit(px_x, px_y, deg)
            coef = np.concatenate([np.zeros(3 - len(coef)), coef])
            # 곡선에서 너무 먼 화소(옆 차선을 잘못 모은 것)를 빼고 한 번 더 맞춘다
            resid = np.abs(np.polyval(coef, px_x) - px_y)
            good = resid <= max(margin * 0.6, 2 * self.cell_y)
            if good.sum() >= c.min_points and not good.all():
                coef = np.polyfit(px_x[good], px_y[good], deg)
                coef = np.concatenate([np.zeros(3 - len(coef)), coef])
                sel = sel[good]
                x0, x1 = float(xs[sel].min()), float(xs[sel].max())
            ln = LaneCurve(np.asarray(coef, np.float64), (x0, x1), n_points=int(sel.size))
            # 7) 실선/점선, 색
            ln.fill, seen_m = self._fill_ratio(ln, xs[sel], win)
            if seen_m < c.style_min_m * s:
                ln.style = "unsure"  # 앞뒤 차가 가려 보인 구간이 짧으면 점선 하나가 실선처럼 보인다
            else:
                ln.style = "solid" if ln.fill >= c.solid_fill else "dashed"
            ln.color = "yellow" if float(np.mean(is_yel[sel])) >= c.yellow_frac else "white"
            lanes.append(ln)
        lanes = self._dedup(lanes)
        return assign_sides(lanes, c.ego_max_m * s)

    def _fill_ratio(self, ln: LaneCurve, px_x: np.ndarray, win: float) -> tuple[float, float]:
        # (가려서 뺀 칸도 "화소 없음"으로 세므로 물체가 실선을 가리면 점선 쪽으로 기운다. 평활의 다수결이 이를 덮는다)
        """곡선 구간을 윈도우 길이로 나눠, 카메라가 보는 칸 중 화소가 있는 칸의 비율과 보는 칸의 길이 합(m)."""
        edges = np.arange(ln.x_range[0], ln.x_range[1] + win, win)
        if len(edges) < 2:
            return 1.0, 0.0
        centers = (edges[:-1] + edges[1:]) / 2
        # 칸 가운데가 카메라가 보는 셀인지(사각지대 칸은 비율 계산에서 뺀다)
        rows = np.clip(((self.grid.x_max - centers) / self.cell_x).astype(int), 0, self.h - 1)
        cols = np.clip(((self.grid.y_max - ln.y_at(centers)) / self.cell_y).astype(int), 0, self.w - 1)
        seen = self.valid[rows, cols] > 0
        if not seen.any():
            return 1.0, 0.0
        cnt, _ = np.histogram(px_x, bins=edges)
        return float(np.mean(cnt[seen] >= 1)), float(seen.sum() * win)

    def _dedup(self, lanes: list[LaneCurve]) -> list[LaneCurve]:
        """X=0 횡위치가 dedup_m보다 가까운 곡선은 화소가 많은 쪽 하나만 남긴다."""
        d = self.cfg.dedup_m * self.scale
        kept: list[LaneCurve] = []
        for ln in sorted(lanes, key=lambda l: -l.n_points):
            if all(abs(ln.offset - k.offset) >= d for k in kept):
                kept.append(ln)
        return sorted(kept, key=lambda l: -l.offset)  # 왼쪽(+Y)부터


# ---------------------------------------------------------------------------
# 프레임 간 평활
# ---------------------------------------------------------------------------
@dataclass
class _LaneTrack:
    curve: LaneCurve
    styles: deque = field(default_factory=deque)
    colors: deque = field(default_factory=deque)
    miss: int = 0
    hits: int = 1


class LaneSmoother:
    """프레임마다 찾은 차선을 이전 차선과 이어 붙여 곡선은 지수 평활, 실선/점선과 색은 다수결로 정한다.

    - 이어 붙이기: X=0 횡위치가 match_m 안에서 가장 가까운 것끼리(가까운 짝부터 하나씩)
    - 못 찾은 차선은 hold_frames 프레임까지 지난 곡선을 그대로 보여 주고, 그 뒤에 지운다
    - 돌려주는 곡선은 사본이라 받는 쪽이 고쳐도 된다(렌더링 스레드에 넘겨도 안전)
    """

    def __init__(self, cfg: LaneConfig | None = None, scale: float = 1.0):
        self.cfg = cfg or LaneConfig()
        self.scale = float(scale)
        self.tracks: list[_LaneTrack] = []

    def update(self, lanes: list[LaneCurve]) -> list[LaneCurve]:
        c = self.cfg
        gate = c.match_m * self.scale
        pairs = sorted(
            ((abs(t.curve.offset - ln.offset), i, j) for i, t in enumerate(self.tracks) for j, ln in enumerate(lanes)),
            key=lambda p: p[0],
        )
        used_t, used_l = set(), set()
        for d, i, j in pairs:
            if d > gate or i in used_t or j in used_l:
                continue
            used_t.add(i)
            used_l.add(j)
            t, ln = self.tracks[i], lanes[j]
            a = c.smooth_alpha
            t.curve.coeffs = (1 - a) * t.curve.coeffs + a * np.asarray(ln.coeffs, np.float64)
            t.curve.x_range = tuple((1 - a) * np.array(t.curve.x_range) + a * np.array(ln.x_range))
            t.curve.fill = (1 - a) * t.curve.fill + a * ln.fill
            t.curve.n_points = ln.n_points
            self._vote(t, ln)
            t.miss = 0
            t.hits += 1
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.miss += 1
        for j, ln in enumerate(lanes):
            if j not in used_l:
                t = _LaneTrack(ln.copy())
                self._vote(t, ln)
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.miss <= c.hold_frames]
        out = [t.curve.copy() for t in self.tracks]
        out.sort(key=lambda l: -l.offset)
        return assign_sides(out, c.ego_max_m * self.scale)

    def _vote(self, t: _LaneTrack, ln: LaneCurve) -> None:
        """최근 vote_frames 번의 실선/점선과 색으로 다수결. 실선/점선을 정하지 못한("unsure") 검출은 표에 넣지 않는다.

        아직 한 번도 정하지 못했으면 실선으로 보여 준다(넘어가도 된다는 뜻인 점선보다 조심스러운 쪽).
        """
        n = max(1, self.cfg.vote_frames)
        if ln.style != "unsure":
            t.styles.append(ln.style)
        t.colors.append(ln.color)
        while len(t.styles) > n:
            t.styles.popleft()
        while len(t.colors) > n:
            t.colors.popleft()
        t.curve.style = _majority(t.styles) if t.styles else "solid"
        t.curve.color = _majority(t.colors)


def _majority(items) -> str:
    """가장 많이 나온 값. 같으면 가장 최근에 나온 쪽."""
    cnt = Counter(items)
    best = max(cnt.values())
    for v in reversed(items):
        if cnt[v] == best:
            return v
    return items[-1]


def box_masks(frames_hw: list[tuple[int, int]], dets_per_cam, skip_labels=()) -> list[np.ndarray] | None:
    """카메라마다 검출 상자 자리를 1로 칠한 마스크. 상자가 하나도 없으면 None(합성을 건너뛰게)."""
    masks, any_box = [], False
    for (h, w), dets in zip(frames_hw, dets_per_cam):
        m = np.zeros((h, w), np.uint8)
        for d in dets:
            if d.label in skip_labels:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in d.box)
            cv2.rectangle(m, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), 1, -1)
            any_box = True
        masks.append(m)
    return masks if any_box else None


def snapshot_lanes(lanes: list[LaneCurve] | None) -> list[LaneCurve] | None:
    """렌더링 스레드로 넘길 사본."""
    return None if lanes is None else [ln.copy() for ln in lanes]
