from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import Detection, FrameResult

# 카메라 4대 구성용 입력과 시간 맞춤.
#   UsbCameraSource  : USB UVC 카메라(OpenCV V4L2). 받은 시각을 time.monotonic_ns()로 붙인다.
#                      동영상 파일을 넣으면 "가짜 USB 카메라"로도 쓴다(type: video).
#   QuadSplitSource  : 쿼드 키트처럼 2x2로 합쳐진 한 프레임을 카메라 4개로 자른다. 카메라별 프레임 지연 보정.
#   FrameSyncer      : 카메라마다 시각이 다를 때 기준 카메라 시각에 가장 가까운 프레임끼리 묶는다.
#   SyncedSources    : 소스 목록을 FrameSyncer로 묶어 한 번에 읽는다(run_demo.py --sync).
#   compensate_offsets: 묶음 안 카메라 시각 차이만큼 지면 물체 위치를 트랙 속도로 옮긴다(옵션).
#   TimedVideoSource, GtReplayBackend: 합성 4대 장면 파일(out/sim4)을 재생하고 정답 상자를 검출처럼 돌려준다.
#
# 시계 기준: Picamera2의 SensorTimestamp는 time.monotonic_ns()와 같은 시계다(sources.py 주석).
# 그래서 USB 카메라도 time.monotonic_ns()로 받은 시각을 적으면 CSI 카메라와 같은 기준이 된다.
# 다만 받은 시각은 노출 시각보다 전송 시간만큼 늦으므로 timestamp_offset_ms로 그만큼 빼 줄 수 있다.


# ---------------------------------------------------------------------------
# 공통: 소스에서 이번 프레임 시각 꺼내기
# ---------------------------------------------------------------------------
def frame_stamp_ns(src) -> int:
    """소스가 마지막으로 돌려준 프레임의 시각(ns).

    센서 시각(last_timestamp_ns), 녹화 시각(recorded_timestamp_ns) 순서로 찾고, 둘 다 없으면 지금 시각을 쓴다.
    """
    for key in ("last_timestamp_ns", "recorded_timestamp_ns"):
        v = getattr(src, key, None)
        if v is not None:
            return int(v)
    return time.monotonic_ns()


def _resize(f: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (f.shape[1], f.shape[0]) != size:
        f = cv2.resize(f, size)
    return f


# ---------------------------------------------------------------------------
# USB UVC 카메라 (OpenCV V4L2), 가짜 장치로 동영상 파일
# ---------------------------------------------------------------------------
def _default_capture(device):
    """장치 번호나 /dev/video* 경로는 V4L2로, 그 밖(동영상 파일)은 OpenCV 기본으로 연다."""
    if isinstance(device, int) or (isinstance(device, str) and device.startswith("/dev/")):
        return cv2.VideoCapture(device, cv2.CAP_V4L2)
    return cv2.VideoCapture(device)


class UsbCameraSource:
    """USB 카메라 한 대. 별도 스레드가 계속 읽고 가장 최근 프레임 한 장만 들고 있다.

    - 받은 순간 time.monotonic_ns()를 적는다. 메인 루프에서 적으면 앞 카메라(CSI)를 기다린 시간까지
      섞이므로 스레드에서 적는다.
    - read()는 아직 안 준 새 프레임이 올 때까지 기다린다(최대 wait_s초). 못 받으면 None.
    - timestamp_offset_ms: 받은 시각에서 뺄 값(전송 지연 추정치). 기본 0.
    - pace_fps: 동영상 파일을 가짜 카메라로 쓸 때 이 속도보다 빨리 읽지 않는다. 진짜 장치는 장치가 속도를 정한다.
    - loop: 파일 끝에서 처음으로 돌아간다(가짜 카메라용).
    - capture_factory: 시험에서 가짜 VideoCapture를 넣는 자리. 기본은 _default_capture.
    - threaded=False: 스레드 없이 read()에서 바로 읽는다(시험용).
    """

    def __init__(
        self,
        device,
        width: int,
        height: int,
        fps: float | None = None,
        timestamp_offset_ms: float = 0.0,
        pace_fps: float | None = None,
        loop: bool = False,
        capture_factory=None,
        threaded: bool = True,
        wait_s: float = 2.0,
        fourcc: str | None = None,
    ):
        self.device = device
        self.size = (int(width), int(height))
        self.offset_ns = int(round(float(timestamp_offset_ms) * 1e6))
        self.pace = float(pace_fps) if pace_fps else None
        self.loop = bool(loop)
        self.wait_s = float(wait_s)
        # run_demo.py 지연 측정과 같은 이름. USB는 늘 받은 시각을 채운다
        self.want_timestamp = False
        self.last_timestamp_ns: int | None = None
        self.frames_read = 0  # 장치에서 받은 프레임 수(read()로 주지 못하고 덮인 것 포함)
        self.frames_given = 0  # read()로 넘긴 프레임 수
        self.cap = (capture_factory or _default_capture)(device)
        if not self.cap.isOpened():
            raise IOError(f"USB 카메라를 열지 못했습니다: {device}")
        is_device = isinstance(device, int) or str(device).startswith("/dev/")
        if is_device:
            # 해상도와 속도를 장치에 요청한다. 장치가 지원하지 않는 값이면 가까운 값으로 바뀐다
            if fourcc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, float(fps))
            # 드라이버 버퍼를 줄여 묵은 프레임을 덜 받는다(지원하지 않는 장치는 무시)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Condition()
        self._latest: tuple[np.ndarray, int, int] | None = None  # (영상, 시각, 번호)
        self._given_seq = 0
        self._stop = False
        self._ended = False
        self._next_t = 0.0
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name=f"usb-{device}", daemon=True)
            self._thread.start()

    # ---- 장치에서 한 장 받기 -----------------------------------------------
    def _grab(self) -> tuple[np.ndarray, int] | None:
        if self.pace:
            # 가짜 카메라: 정해진 간격보다 빨리 내보내지 않는다
            now = time.monotonic()
            if now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t = max(now, self._next_t) + 1.0 / self.pace
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if not ok or f is None:
            return None
        ts = time.monotonic_ns() - self.offset_ns
        self.frames_read += 1
        return f, ts

    def _run(self) -> None:
        seq = 0
        while not self._stop:
            got = self._grab()
            with self._lock:
                if got is None:
                    self._ended = True
                    self._lock.notify_all()
                    return
                seq += 1
                self._latest = (got[0], got[1], seq)
                self._lock.notify_all()

    # ---- 바깥에서 부르는 함수 ----------------------------------------------
    def read(self) -> np.ndarray | None:
        if self._thread is None:
            got = self._grab()
            if got is None:
                return None
            f, ts = got
        else:
            with self._lock:
                deadline = time.monotonic() + self.wait_s
                while not self._ended and (self._latest is None or self._latest[2] <= self._given_seq):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    self._lock.wait(left)
                if self._latest is None or self._latest[2] <= self._given_seq:
                    return None  # 끝났고 새 프레임도 없음
                f, ts, self._given_seq = self._latest
        self.last_timestamp_ns = int(ts)
        self.frames_given += 1
        return _resize(f, self.size)

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()


# ---------------------------------------------------------------------------
# 쿼드 키트: 2x2 한 프레임을 카메라 4개로
# ---------------------------------------------------------------------------
def split_quad(frame: np.ndarray) -> list[np.ndarray]:
    """2x2로 합쳐진 영상을 [왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래] 네 장으로 자른다(복사 없이 보기만)."""
    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    return [frame[:h2, :w2], frame[:h2, w2 : 2 * w2], frame[h2 : 2 * h2, :w2], frame[h2 : 2 * h2, w2 : 2 * w2]]


class QuadSplitSource:
    """쿼드 키트 한 대(합쳐진 프레임 하나)를 여러 카메라가 나눠 쓰는 중심 객체.

    카메라마다 view(칸 번호, 지연 프레임 수)로 QuadTileSource를 받아 쓴다. 어느 칸이든 이미 받은 프레임을
    다시 달라고 하면 그때 새 합쳐진 프레임을 하나 읽는다. 그래서 한 루프에서 네 칸을 차례로 읽으면
    합쳐진 프레임 한 장으로 네 칸이 나온다.

    프레임 지연(lag): Arducam 설명대로 2번과 3번 칸이 한 프레임 늦다면 lag=1. 합쳐진 프레임 N의 그 칸은
    실제로는 프레임 N-1 때 찍힌 것이다(칸 순서와 지연 방향은 실제 키트로 확인 필요).
      align=True : 늦지 않은 칸을 그만큼 묵혀서(이전 합쳐진 프레임에서 꺼내) 네 칸이 같은 순간을 보이게 한다.
                   시각은 모두 프레임 N-최대지연 의 시각이다.
      align=False: 칸은 이번 프레임 그대로 쓰고, 시각만 lag x 프레임 간격만큼 앞당겨 적는다
                   (FrameSyncer가 차이를 기록하고, 추적기 위치 보정 옵션이 쓸 수 있게).
    """

    def __init__(self, inner, align: bool = True, frame_period_ms: float | None = None):
        self.inner = inner
        # CSI(Picamera2)는 이 값을 켜야 센서 시각을 꺼낸다. 칸 소스만 켜면 키트 장치는 읽은 시각으로 남는다
        if hasattr(inner, "want_timestamp"):
            inner.want_timestamp = True
        self.align = bool(align)
        self.period_ns = int(frame_period_ms * 1e6) if frame_period_ms else None
        self.lags: dict[int, int] = {}
        self.seq = 0  # 지금까지 읽은 합쳐진 프레임 수
        self._hist: deque = deque(maxlen=1)  # (합쳐진 영상, 시각) 최근 것이 끝
        self._closed = False

    def view(self, tile: int, lag_frames: int = 0, size: tuple[int, int] | None = None) -> "QuadTileSource":
        tile = int(tile)
        if not 0 <= tile <= 3:
            raise ValueError(f"쿼드 칸 번호는 0~3입니다: {tile}")
        if tile in self.lags:
            raise ValueError(f"쿼드 칸 {tile}을 두 카메라가 같이 쓸 수 없습니다")
        self.lags[tile] = max(0, int(lag_frames))
        self._hist = deque(self._hist, maxlen=max(self.lags.values()) + 2)
        return QuadTileSource(self, tile, size)

    def _advance(self) -> None:
        f = self.inner.read()
        if f is None:
            raise EOFError
        ts = frame_stamp_ns(self.inner)
        self._hist.append((f, ts))
        self.seq += 1

    def _period_ns(self) -> int:
        if self.period_ns:
            return self.period_ns
        if len(self._hist) >= 2:
            return max(1, self._hist[-1][1] - self._hist[-2][1])
        return 0

    def tile(self, tile: int, consumer_seq: int) -> tuple[np.ndarray, int, int] | None:
        """칸 하나를 (영상, 시각, 합쳐진 프레임 번호)로 돌려준다. 이 칸이 이미 받은 번호면 새로 읽는다."""
        if consumer_seq >= self.seq:
            try:
                self._advance()
            except EOFError:
                return None
        lag = self.lags.get(tile, 0)
        big = max(self.lags.values()) if self.lags else 0
        if self.align:
            back = big - lag  # 늦지 않은 칸일수록 더 옛 프레임에서 꺼낸다
            k = max(0, len(self._hist) - 1 - back)
            f = self._hist[k][0]
            # 네 칸 모두 "가장 늦은 칸이 찍힌 순간" = 합쳐진 프레임 (N - 최대지연)의 시각
            j = max(0, len(self._hist) - 1 - big)
            ts = self._hist[j][1]
        else:
            f, ts = self._hist[-1]
            ts = ts - lag * self._period_ns()
        return split_quad(f)[tile], int(ts), self.seq

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.inner.close()


class QuadTileSource:
    """QuadSplitSource의 칸 하나를 카메라처럼 read()로 준다."""

    def __init__(self, hub: QuadSplitSource, tile: int, size: tuple[int, int] | None):
        self.hub = hub
        self.tile_index = tile
        self.size = size
        self.want_timestamp = False
        self.last_timestamp_ns: int | None = None
        self._seq = 0

    def read(self) -> np.ndarray | None:
        got = self.hub.tile(self.tile_index, self._seq)
        if got is None:
            return None
        f, ts, self._seq = got
        self.last_timestamp_ns = ts
        f = np.ascontiguousarray(f)
        return _resize(f, self.size) if self.size else f

    def close(self) -> None:
        # 칸 하나를 닫아도 다른 칸이 쓰는 장치는 그대로 둔다. 마지막 정리는 hub.close()
        self.hub.close()


# ---------------------------------------------------------------------------
# 시간 맞춤: 기준 카메라 시각에 가장 가까운 프레임끼리 묶기
# ---------------------------------------------------------------------------
@dataclass
class SyncBundle:
    """한 번 묶은 결과. 목록은 모두 카메라 순서다."""

    ref_ns: int
    frames: list
    stamps_ns: list  # 쓴 프레임의 시각. 빈 칸이면 None
    offsets_ms: list  # 쓴 프레임 시각 - 기준 시각(ms). 빈 칸이면 None
    status: list  # "ok"(허용 오차 안), "hold"(이전 프레임 재사용), "blank"(빈 칸)
    meta: list = field(default_factory=list)  # 프레임에 붙은 부가 정보(예: 파일 안 프레임 번호)

    @property
    def offsets_s(self) -> list[float]:
        """추적기 위치 보정용 초 단위 차이. 빈 칸은 0."""
        return [0.0 if o is None else o / 1e3 for o in self.offsets_ms]

    @property
    def spread_ms(self) -> float:
        """묶음 안 카메라 시각의 최대 - 최소(ms). 빈 칸은 뺀다."""
        v = [o for o in self.offsets_ms if o is not None]
        return float(max(v) - min(v)) if v else 0.0


class FrameSyncer:
    """카메라마다 들어온 (영상, 시각)을 모아 두었다가 기준 카메라 시각에 맞춰 묶는다.

    - push(카메라, 영상, 시각, meta): 받은 프레임을 그 카메라 대기열(keep장)에 넣는다.
    - bundle(ref_ns): 카메라마다 |시각 - ref_ns|가 가장 작은 프레임을 고른다.
        허용 오차(tolerance_ms) 안이면 "ok". 넘으면 on_miss에 따라
        "hold" = 가진 프레임(대기열과 바로 앞 묶음 프레임) 중 가장 가까운 것을 다시 쓴다(하나도 없으면 빈 칸),
        "blank" = 그 카메라 크기의 검은 영상(빈 칸)을 넣는다.
    - 묶을 때마다 카메라별 시각 차이를 통계로 쌓고, csv_path를 주면 묶음마다 한 줄씩 쓴다.
    """

    def __init__(
        self,
        sizes: list[tuple[int, int]],
        names: list[str] | None = None,
        ref: int = 0,
        tolerance_ms: float = 25.0,
        on_miss: str = "hold",
        keep: int = 6,
        csv_path: str | None = None,
    ):
        if on_miss not in ("hold", "blank"):
            raise ValueError(f"on_miss는 hold 또는 blank여야 합니다: {on_miss}")
        self.n = len(sizes)
        self.sizes = [(int(w), int(h)) for w, h in sizes]
        self.names = list(names) if names else [f"cam{i}" for i in range(self.n)]
        self.ref = int(ref)
        self.tol_ns = int(float(tolerance_ms) * 1e6)
        self.tolerance_ms = float(tolerance_ms)
        self.on_miss = on_miss
        self._q = [deque(maxlen=max(1, int(keep))) for _ in range(self.n)]
        self._prev: list[tuple | None] = [None] * self.n  # 카메라별 바로 앞 묶음에서 쓴 (영상, 시각, meta)
        self._blank = [np.zeros((h, w, 3), np.uint8) for w, h in self.sizes]
        self.count = 0
        self._abs: list[list[float]] = [[] for _ in range(self.n)]  # 카메라별 |차이| ms
        self._status = [{"ok": 0, "hold": 0, "blank": 0} for _ in range(self.n)]
        self._spread: list[float] = []
        self._csv = None
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            self._csv_f = open(csv_path, "w", newline="")
            self._csv = csv.writer(self._csv_f)
            head = ["bundle", "ref_ns", "spread_ms"]
            for nm in self.names:
                head += [f"{nm}_offset_ms", f"{nm}_status"]
            self._csv.writerow(head)

    def push(self, cam: int, frame: np.ndarray, ts_ns: int, meta=None) -> None:
        self._q[cam].append((frame, int(ts_ns), meta))

    def bundle(self, ref_ns: int) -> SyncBundle:
        ref_ns = int(ref_ns)
        frames, stamps, offs, status, metas = [], [], [], [], []
        for i in range(self.n):
            best = min(self._q[i], key=lambda e: abs(e[1] - ref_ns), default=None)
            if best is not None and abs(best[1] - ref_ns) <= self.tol_ns:
                pick, st = best, "ok"
            elif self.on_miss == "hold" and (best is not None or self._prev[i] is not None):
                # 허용 오차 밖: 가진 것 중 가장 가까운 프레임을 다시 쓴다(대개 직전 묶음 프레임).
                # 직전 프레임만 고집하면 새 프레임이 조금 앞서 도착하는 카메라(받은 시각을 쓰는 USB)에서
                # 같은 옛 프레임이 계속 쓰여 1초 넘게 묵을 수 있어(파이 실측) 대기열과 함께 비교한다
                cands = [e for e in (best, self._prev[i]) if e is not None]
                pick, st = min(cands, key=lambda e: abs(e[1] - ref_ns)), "hold"
            else:
                pick, st = None, "blank"
            if pick is None:
                frames.append(self._blank[i])
                stamps.append(None)
                offs.append(None)
                metas.append(None)
            else:
                self._prev[i] = pick
                frames.append(pick[0])
                stamps.append(pick[1])
                offs.append((pick[1] - ref_ns) / 1e6)
                metas.append(pick[2])
                self._abs[i].append(abs(offs[-1]))
            status.append(st)
            self._status[i][st] += 1
        b = SyncBundle(ref_ns, frames, stamps, offs, status, metas)
        self._spread.append(b.spread_ms)
        if self._csv is not None:
            row = [self.count, ref_ns, f"{b.spread_ms:.3f}"]
            for o, s in zip(offs, status):
                row += ["" if o is None else f"{o:.3f}", s]
            self._csv.writerow(row)
        self.count += 1
        return b

    def summary(self) -> dict:
        """카메라별 시각 차이 통계(ms)와 ok/hold/blank 횟수."""
        cams = {}
        for i, nm in enumerate(self.names):
            a = self._abs[i]
            cams[nm] = {
                "abs_offset_ms_mean": round(float(np.mean(a)), 3) if a else None,
                "abs_offset_ms_p95": round(float(np.percentile(a, 95)), 3) if a else None,
                "abs_offset_ms_max": round(float(max(a)), 3) if a else None,
                **self._status[i],
            }
        sp = self._spread
        return {
            "bundles": self.count,
            "ref": self.names[self.ref],
            "tolerance_ms": self.tolerance_ms,
            "on_miss": self.on_miss,
            "spread_ms_mean": round(float(np.mean(sp)), 3) if sp else None,
            "spread_ms_p95": round(float(np.percentile(sp, 95)), 3) if sp else None,
            "spread_ms_max": round(float(max(sp)), 3) if sp else None,
            "cameras": cams,
        }

    def close(self) -> None:
        if self._csv is not None:
            self._csv_f.close()
            self._csv = None


class SyncedSources:
    """소스 목록을 FrameSyncer로 묶어 read() 한 번에 카메라 전체 영상을 준다.

    - 기준 소스를 먼저 한 장 읽고 그 시각을 기준으로 삼는다.
    - 다른 소스가 next_ns()를 가진 재생 소스면 "기준 시각 + 허용 오차"까지의 프레임을 모두 넣는다
      (파일 재생은 기다릴 필요가 없어서 시간 순서대로 꺼내 쓴다).
    - 실시간 소스(카메라)는 한 장씩 읽어 넣는다. 끝난 소스(None)는 다시 부르지 않는다.
    - 기준 소스가 끝나면 None.
    """

    def __init__(self, sources: list, syncer: FrameSyncer):
        self.sources = sources
        self.syncer = syncer
        self.last: SyncBundle | None = None
        self._ended = [False] * len(sources)

    @staticmethod
    def _meta(src):
        return getattr(src, "recorded_frame", None)

    def read(self) -> list[np.ndarray] | None:
        sy = self.syncer
        ref_src = self.sources[sy.ref]
        f = ref_src.read()
        if f is None:
            return None
        ref_ns = frame_stamp_ns(ref_src)
        sy.push(sy.ref, f, ref_ns, self._meta(ref_src))
        for i, s in enumerate(self.sources):
            if i == sy.ref or self._ended[i]:
                continue
            if hasattr(s, "next_ns"):
                while True:
                    nx = s.next_ns()
                    if nx is None or nx > ref_ns + sy.tol_ns:
                        break
                    g = s.read()
                    if g is None:
                        break
                    sy.push(i, g, frame_stamp_ns(s), self._meta(s))
            else:
                g = s.read()
                if g is None:
                    self._ended[i] = True
                    continue
                sy.push(i, g, frame_stamp_ns(s), self._meta(s))
        self.last = sy.bundle(ref_ns)
        return self.last.frames


# ---------------------------------------------------------------------------
# 시각 차이만큼 지면 물체 위치 옮기기(옵션)
# ---------------------------------------------------------------------------
def _compatible(a: str, b: str) -> bool:
    veh = {"car", "truck", "bus"}
    return a == b or (a in veh and b in veh)


def compensate_offsets(objs: list, tracks: list, offsets_s: list[float], dt: float, gate: float,
                       range_gate: float = 0.15, min_offset_s: float = 1e-3) -> list:
    """카메라 시각이 기준보다 offset만큼 다른 물체를 기준 시각 위치로 옮긴다.

    물체 o(카메라 i, 시각 기준+off)가 기준 시각에 있었을 위치 = o.xy - 속도 x off.
    속도는 가장 가까운 확정 트랙(라벨이 맞고 짝짓기 거리 안)의 상대 속도를 쓴다. 트랙 예측 위치는
    "지난 위치 + 속도 x (dt + off)"로 물체 시각에 맞춰 비교한다. 맞는 트랙이 없으면 그대로 둔다.
    objs의 xy를 새 배열로 바꾼 같은 목록을 돌려준다.
    """
    for o in objs:
        off = offsets_s[o.cam_index] if o.cam_index < len(offsets_s) else 0.0
        if abs(off) < min_offset_s:
            continue
        best, best_d = None, None
        for t in tracks:
            if t.hits < 2 or not _compatible(t.label, o.label):
                continue
            pred = t.xy + t.vel * (dt + off)
            d = float(np.linalg.norm(pred - o.xy))
            lim = max(gate, range_gate * float(np.linalg.norm(pred)))
            if d < lim and (best_d is None or d < best_d):
                best, best_d = t, d
        if best is not None:
            o.xy = o.xy - best.vel * off
    return objs


# ---------------------------------------------------------------------------
# 합성 4대 장면 파일(out/sim4) 재생
# ---------------------------------------------------------------------------
# 폴더 구성(scripts/make_sim4.py가 만든다)
#   cam{k}.mp4 : 카메라 k의 영상(설정의 카메라 순서). 빠진 프레임은 아예 없다
#   cam{k}.csv : 영상 프레임마다 한 줄. frame(파일 안 번호), sim_index(시뮬레이션 프레임 번호), sensor_ns
#   gt.json    : 정답 상자(카메라별, 파일 프레임별)와 기준 시각의 물체 위치
def timed_video_paths(folder: str, k: int) -> tuple[str, str]:
    return os.path.join(folder, f"cam{k}.mp4"), os.path.join(folder, f"cam{k}.csv")


def has_timed_videos(folder: str, n: int) -> bool:
    return all(os.path.isfile(p) for k in range(n) for p in timed_video_paths(folder, k))


class TimedVideoSource:
    """mp4 + 시각 csv를 차례로 읽는다. 끝나면 None(시각이 되돌아가지 않게 반복하지 않는다).

    MjpegReplaySource와 같이 녹화 시각은 recorded_timestamp_ns, 파일 안 번호는 recorded_frame 에 둔다.
    next_ns()로 다음 프레임 시각을 미리 볼 수 있어 SyncedSources가 시간 순서대로 꺼낸다.
    """

    def __init__(self, video: str, csv_path: str, width: int, height: int):
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise IOError(f"동영상을 열지 못했습니다: {video}")
        with open(csv_path, newline="") as f:
            self.rows = [(int(r["frame"]), int(r["sensor_ns"])) for r in csv.DictReader(f)]
        self.size = (int(width), int(height))
        self.pos = 0
        self.want_timestamp = False
        self.last_timestamp_ns: int | None = None
        self.recorded_timestamp_ns: int | None = None
        self.recorded_frame: int | None = None

    def next_ns(self) -> int | None:
        return self.rows[self.pos][1] if self.pos < len(self.rows) else None

    def read(self) -> np.ndarray | None:
        if self.pos >= len(self.rows):
            return None
        ok, f = self.cap.read()
        if not ok:
            return None
        self.recorded_frame, self.recorded_timestamp_ns = self.rows[self.pos]
        self.pos += 1
        return _resize(f, self.size)

    def close(self) -> None:
        self.cap.release()


def open_timed_videos(folder: str, cams) -> list[TimedVideoSource]:
    out = []
    for k, c in enumerate(cams):
        v, p = timed_video_paths(folder, k)
        out.append(TimedVideoSource(v, p, c.width, c.height))
    return out


class GtReplayBackend:
    """out/sim4/gt.json 의 정답 상자를 검출 결과처럼 돌려준다(카메라 없이 4대 화면과 추적 확인용).

    파이프라인은 영상만 넘기므로, 각 카메라 영상이 파일의 몇 번째 프레임인지는 run_demo.py가
    set_frame_meta()로 먼저 알려 준다. 모르는 칸(빈 칸)은 검출 없음.
    """

    def __init__(self, gt_path: str, noise_px: float = 1.0, seed: int = 0):
        import json

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        self.boxes = gt["boxes"]  # [카메라][파일 프레임] = [[라벨, x1, y1, x2, y2, 물체, 다시 등장 횟수], ...]
        self.noise_px = float(noise_px)
        self.rng = np.random.default_rng(seed)
        self.meta: list = []

    def set_frame_meta(self, meta: list) -> None:
        self.meta = list(meta)

    def infer(self, frames) -> list[FrameResult]:
        out = []
        for i, _ in enumerate(frames):
            k = self.meta[i] if i < len(self.meta) else None
            dets = []
            if k is not None and i < len(self.boxes) and 0 <= k < len(self.boxes[i]):
                for lab, x1, y1, x2, y2, *_ in self.boxes[i][k]:
                    b = np.array([x1, y1, x2, y2], float) + self.rng.normal(0, self.noise_px, 4)
                    dets.append(Detection(lab, 0.9, tuple(b.tolist())))
            out.append(FrameResult(detections=dets))
        return out

    def close(self) -> None:
        pass
