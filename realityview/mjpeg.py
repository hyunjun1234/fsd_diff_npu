from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np

# 카메라 한 대의 영상을 "JPEG를 이어 붙인 파일(.mjpeg) + 프레임마다 한 줄인 csv"로 저장하고 다시 읽는다.
# 주행 녹화(realityview/drive_rec.py)와 트리거 클립(realityview/triggers.py)이 같은 형식을 쓰므로
# 둘 다 run_demo.py --replay 로 바로 넣을 수 있다.
#
# 이 형식을 고른 이유
#   - 덧붙이기만 하는 파일이라 전원이 갑자기 꺼져도 그때까지 쓴 프레임은 남는다(MP4는 닫지 못하면 못 읽는다).
#   - 프레임마다 JPEG 한 장이라 아무 프레임이나 바로 꺼내 라벨링할 수 있다.
#   - ffmpeg도 읽는다: ffmpeg -f mjpeg -i front.mjpeg ...
#   - 파이 5에는 H.264 하드웨어 인코더가 없어 카메라 4대를 소프트웨어 H.264로 누르기에는 CPU가 아깝다.
#
# csv 열(CSV_FIELDS)
#   frame      : 이 파일 안의 프레임 번호(0부터)
#   index      : 녹화 전체에서 이 카메라의 프레임 번호(트리거 클립은 run_demo.py 프레임 번호)
#   sensor_ns  : 센서 타임스탬프(ns, time.monotonic_ns()와 같은 시계). 센서 값이 없으면 읽은 시각
#   wall_time  : 읽은 순간의 벽시계(초, time.time())
#   offset     : .mjpeg 파일 안에서 이 JPEG가 시작하는 바이트 위치
#   size       : JPEG 바이트 수
#   gap        : 바로 앞 프레임 이후 빠진 프레임 수(센서 시각 간격으로 추정, 첫 프레임은 0)

CSV_FIELDS = ["frame", "index", "sensor_ns", "wall_time", "offset", "size", "gap"]


# ---------------------------------------------------------------------------
# 빠진 프레임 수 추정 (순수 함수)
# ---------------------------------------------------------------------------
def frames_missing(prev_ns: int | None, cur_ns: int, period_ns: float) -> int:
    """앞 프레임과 이번 프레임의 센서 시각 간격으로 사이에 빠진 프레임 수를 센다.

    간격이 한 주기의 1.5배를 넘으면 빠진 것으로 본다. 예: 주기 33.3ms에 간격 100ms면 2장 빠짐.
    """
    if prev_ns is None or period_ns <= 0:
        return 0
    n = int(round((cur_ns - prev_ns) / period_ns)) - 1
    return max(0, n)


# ---------------------------------------------------------------------------
# 쓰기
# ---------------------------------------------------------------------------
class MjpegWriter:
    """카메라 한 대 분량의 {이름}.mjpeg 와 {이름}.csv 를 쓴다.

    flush_every 프레임마다 두 파일을 비워(flush) 전원이 꺼져도 잃는 양을 줄인다.
    close()는 fsync까지 해서 디스크에 확실히 남긴다.
    """

    def __init__(self, folder: str, name: str, quality: int = 85, flush_every: int = 15):
        os.makedirs(folder, exist_ok=True)
        self.folder, self.name = folder, name
        self.video_path = os.path.join(folder, f"{name}.mjpeg")
        self.csv_path = os.path.join(folder, f"{name}.csv")
        self.quality = int(quality)
        self.flush_every = max(1, int(flush_every))
        self._vf = open(self.video_path, "wb")
        self._cf = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._cf)
        self._csv.writerow(CSV_FIELDS)
        self._cf.flush()
        self.frames = 0  # 쓴 프레임 수
        self.gaps = 0  # 센서 시각으로 추정한 빠진 프레임 수 합
        self.bytes = 0
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.closed = False

    def write(self, frame: np.ndarray, index: int, sensor_ns: int, wall_time: float, gap: int = 0) -> int:
        """BGR 영상 한 장을 JPEG로 눌러 덧붙인다. 쓴 바이트 수를 돌려준다."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise IOError(f"JPEG 인코딩 실패: {self.video_path}")
        return self.write_jpeg(buf.tobytes(), index, sensor_ns, wall_time, gap)

    def write_jpeg(self, data: bytes, index: int, sensor_ns: int, wall_time: float, gap: int = 0) -> int:
        """이미 누른 JPEG 바이트를 덧붙인다."""
        offset = self.bytes
        self._vf.write(data)
        self.bytes += len(data)
        self._csv.writerow([self.frames, int(index), int(sensor_ns), f"{wall_time:.6f}", offset, len(data), int(gap)])
        self.frames += 1
        self.gaps += int(gap)
        if self.first_ns is None:
            self.first_ns = int(sensor_ns)
        self.last_ns = int(sensor_ns)
        if self.frames % self.flush_every == 0:
            self.flush()
        return len(data)

    def flush(self) -> None:
        self._vf.flush()
        self._cf.flush()

    def close(self) -> dict:
        """파일을 닫고 요약을 돌려준다. 두 번 불러도 된다."""
        if not self.closed:
            self.closed = True
            for f in (self._vf, self._cf):
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
                f.close()
        return self.stats()

    def stats(self) -> dict:
        dur = (self.last_ns - self.first_ns) / 1e9 if self.frames > 1 else 0.0
        return {
            "camera": self.name,
            "frames": self.frames,
            "gaps": self.gaps,
            "bytes": self.bytes,
            "duration_s": round(dur, 3),
            "fps": round((self.frames - 1) / dur, 2) if dur > 0 else None,
        }


# ---------------------------------------------------------------------------
# 읽기: csv 색인
# ---------------------------------------------------------------------------
@dataclass
class FrameRef:
    """저장된 프레임 하나의 위치. path 파일의 offset에서 size 바이트가 JPEG 한 장이다."""

    path: str
    offset: int
    size: int
    sensor_ns: int
    index: int


def read_index(folder: str, name: str) -> list[FrameRef]:
    """{이름}.csv 를 읽어 프레임 위치 목록을 만든다.

    전원이 꺼져 csv 마지막 줄이 잘렸거나, csv에는 있는데 .mjpeg가 그만큼 써지지 않은 줄은 버린다.
    """
    video = os.path.join(folder, f"{name}.mjpeg")
    path = os.path.join(folder, f"{name}.csv")
    size_on_disk = os.path.getsize(video) if os.path.exists(video) else 0
    refs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                off, sz = int(row["offset"]), int(row["size"])
                ref = FrameRef(video, off, sz, int(row["sensor_ns"]), int(row["index"]))
            except (TypeError, ValueError, KeyError):
                continue  # 잘린 줄
            if off + sz > size_on_disk:
                continue  # 영상 쪽이 덜 써진 프레임
            refs.append(ref)
    return refs


def has_mjpeg(folder: str, name: str) -> bool:
    return os.path.isfile(os.path.join(folder, f"{name}.mjpeg")) and os.path.isfile(os.path.join(folder, f"{name}.csv"))


def segment_dirs(folder: str) -> list[str]:
    """재생할 폴더 목록. 회차 폴더(seg_* 하위 폴더가 있음)를 주면 세그먼트를 이름 순서로, 아니면 그 폴더 하나."""
    subs = sorted(
        os.path.join(folder, d) for d in os.listdir(folder) if d.startswith("seg_") and os.path.isdir(os.path.join(folder, d))
    )
    return subs or [folder]


# ---------------------------------------------------------------------------
# 읽기: 카메라 사이 시각 맞추기 (순수 함수)
# ---------------------------------------------------------------------------
def nearest_indices(ref_ns: np.ndarray, other_ns: np.ndarray) -> np.ndarray:
    """ref_ns의 각 시각에 가장 가까운 other_ns의 위치. 두 배열 모두 오름차순이어야 한다.

    카메라마다 따로 도는 스레드로 녹화하므로 카메라별 프레임 수가 다르다. 번호로 짝지으면 시간이
    점점 어긋나므로, 기준 카메라 프레임마다 시각이 가장 가까운 다른 카메라 프레임을 고른다.
    """
    other_ns = np.asarray(other_ns, dtype=np.int64)
    ref_ns = np.asarray(ref_ns, dtype=np.int64)
    if len(other_ns) == 0:
        raise ValueError("짝지을 프레임이 없습니다")
    j = np.searchsorted(other_ns, ref_ns)
    j = np.clip(j, 1, len(other_ns) - 1) if len(other_ns) > 1 else np.zeros_like(j)
    if len(other_ns) > 1:
        left = other_ns[j - 1]
        right = other_ns[j]
        j = np.where(np.abs(ref_ns - left) <= np.abs(right - ref_ns), j - 1, j)
    return j.astype(np.int64)


def synced_refs(folder: str, names: list[str]) -> tuple[list[list[FrameRef]], float]:
    """카메라 이름 목록의 프레임을 기준 카메라(첫 번째) 시각에 맞춰 같은 길이 목록으로 만든다.

    돌려주는 값: (카메라별 FrameRef 목록, 짝지은 프레임 사이 최대 시각 차이 ms)
    세그먼트가 여러 개면 이어 붙인 뒤 맞춘다.
    """
    per_cam: list[list[FrameRef]] = [[] for _ in names]
    for d in segment_dirs(folder):
        if not all(has_mjpeg(d, n) for n in names):
            continue  # 어떤 카메라 파일이 없는 세그먼트(녹화가 막 시작되다 끊긴 경우 등)는 건너뛴다
        for k, n in enumerate(names):
            per_cam[k] += read_index(d, n)
    if not per_cam or not per_cam[0]:
        raise FileNotFoundError(f"{folder} 에 {names} 의 .mjpeg/.csv 프레임이 없습니다")
    for k, refs in enumerate(per_cam):
        if not refs:
            raise FileNotFoundError(f"{folder} 에 {names[k]} 프레임이 없습니다")
        refs.sort(key=lambda r: r.sensor_ns)
    ref_ns = np.array([r.sensor_ns for r in per_cam[0]], dtype=np.int64)
    out = [per_cam[0]]
    skew = 0.0
    for refs in per_cam[1:]:
        ns = np.array([r.sensor_ns for r in refs], dtype=np.int64)
        j = nearest_indices(ref_ns, ns)
        skew = max(skew, float(np.max(np.abs(ns[j] - ref_ns))) / 1e6)
        out.append([refs[int(i)] for i in j])
    return out, skew


# ---------------------------------------------------------------------------
# 읽기: 카메라처럼 한 장씩 돌려주는 소스
# ---------------------------------------------------------------------------
class MjpegReplaySource:
    """저장된 .mjpeg 프레임을 카메라처럼 read()로 한 장씩 돌려준다. 끝나면 처음부터 다시(loop=True).

    녹화 당시 센서 시각은 recorded_timestamp_ns 에 둔다. last_timestamp_ns 는 늘 None이다.
    run_demo.py --profile 의 지연 측정은 last_timestamp_ns를 지금 시각과 빼므로, 며칠 전 녹화 시각을 넣으면
    엉뚱한 값이 나오기 때문이다.
    """

    def __init__(self, refs: list[FrameRef], width: int, height: int, loop: bool = True):
        if not refs:
            raise ValueError("재생할 프레임이 없습니다")
        self.refs = refs
        self.size = (int(width), int(height))
        self.loop = loop
        self.pos = 0
        self.want_timestamp = False
        self.last_timestamp_ns: int | None = None
        self.recorded_timestamp_ns: int | None = None
        self.recorded_index: int | None = None
        self._files: dict[str, object] = {}

    def _bytes(self, ref: FrameRef) -> bytes:
        f = self._files.get(ref.path)
        if f is None:
            f = self._files[ref.path] = open(ref.path, "rb")
        f.seek(ref.offset)
        return f.read(ref.size)

    def read(self) -> np.ndarray | None:
        if self.pos >= len(self.refs):
            if not self.loop:
                return None
            self.pos = 0
        ref = self.refs[self.pos]
        self.pos += 1
        img = cv2.imdecode(np.frombuffer(self._bytes(ref), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"JPEG를 풀지 못했습니다: {ref.path} 위치 {ref.offset}")
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        self.recorded_timestamp_ns = ref.sensor_ns
        self.recorded_index = ref.index
        return img

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


def open_mjpeg_sources(folder: str, cams, loop: bool = True) -> list[MjpegReplaySource]:
    """설정의 카메라 목록(이름, 폭, 높이)에 맞춰 시각을 맞춘 재생 소스를 만든다."""
    refs, _ = synced_refs(folder, [c.name for c in cams])
    return [MjpegReplaySource(r, c.width, c.height, loop) for r, c in zip(refs, cams)]
