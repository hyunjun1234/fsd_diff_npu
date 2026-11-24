from __future__ import annotations

import os
import queue
import threading
import time

import cv2
import numpy as np

# 데모 대시보드를 동영상 파일로 남기는 녹화기.
#
# - 인코딩은 별도 스레드에서 한다. 메인 루프는 영상을 큐에 넣기만 하므로 덜 느려진다.
#   큐가 차 있으면(인코딩이 못 따라가면) 그 영상은 버리고 dropped로 센다.
# - 동영상 파일의 fps는 열 때 정해진다. 루프 속도는 매번 다르므로, 들어온 시각을 보고
#   정해진 fps 시간축에 맞춰 같은 영상을 한 번 더 쓰거나 건너뛴다. 그래서 재생 속도가 실제 시간과 같다.
# - 코덱은 파이의 OpenCV(시스템 패키지, FFmpeg libx264)에서 열리는 avc1(H.264)을 먼저 쓰고,
#   열리지 않으면 mp4v(MPEG-4 Part 2)로 바꾼다. 둘 다 안 되면 오류를 낸다.

DEFAULT_CODECS = ("avc1", "mp4v")


# ---------------------------------------------------------------------------
# 시간축 맞추기 (순수 함수, 시험용으로 분리)
# ---------------------------------------------------------------------------
def frames_due(elapsed_s: float, fps: float, written: int) -> int:
    """녹화 시작 뒤 elapsed_s초가 지난 시점에 이 영상을 몇 번 써야 하는지.

    fps 시간축에서 elapsed_s까지 있어야 할 장 수(첫 장 포함)에서 이미 쓴 장 수를 뺀다.
    0이면 이번 영상은 건너뛰고(루프가 녹화 fps보다 빠를 때), 2 이상이면 같은 영상을 반복한다
    (루프가 잠깐 느려졌을 때).
    """
    target = int(elapsed_s * fps + 1e-9) + 1
    return max(0, target - written)


# ---------------------------------------------------------------------------
# 녹화기
# ---------------------------------------------------------------------------
class VideoRecorder:
    def __init__(
        self,
        path: str,
        fps: float,
        codecs: tuple[str, ...] = DEFAULT_CODECS,
        queue_size: int = 2,
        writer_factory=None,
        clock=time.monotonic,
    ):
        self.path = path
        self.fps = float(fps)
        self.codecs = tuple(codecs)
        self.codec = None  # 실제로 열린 코덱
        self.size = None  # (폭, 높이)
        self.submitted = 0  # 넣으려고 한 영상 수
        self.dropped = 0  # 큐가 차서 버린 영상 수
        self.written = 0  # 파일에 쓴 장 수(반복 포함)
        self._factory = writer_factory or cv2.VideoWriter
        self._clock = clock
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._writer = None
        self._t0 = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)
        self._started = False

    # ---- 열기 -----------------------------------------------------------
    def _open(self, w: int, h: int) -> None:
        """첫 영상 크기로 파일을 연다. 코덱을 차례로 시도하고 모두 실패하면 오류."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tried = []
        for cc in self.codecs:
            wr = self._factory(self.path, cv2.VideoWriter_fourcc(*cc), self.fps, (w, h))
            if wr.isOpened():
                self._writer, self.codec, self.size = wr, cc, (w, h)
                return
            wr.release()
            tried.append(cc)
        raise RuntimeError(f"동영상 파일을 열지 못했습니다: {self.path} (시도한 코덱 {tried}, 크기 {w}x{h})")

    # ---- 넣기 (메인 루프) -------------------------------------------------
    def submit(self, frame: np.ndarray) -> None:
        """대시보드 한 장을 넣는다. 첫 장에서 파일을 열고 인코딩 스레드를 시작한다."""
        if self._error is not None:
            raise RuntimeError("녹화 스레드가 멈췄습니다") from self._error
        now = self._clock()
        if not self._started:
            h, w = frame.shape[:2]
            self._open(w, h)  # 여기서 실패하면 바로 오류가 메인 루프로 올라간다
            self._t0 = now
            self._started = True
            self._thread.start()
        self.submitted += 1
        try:
            # 메인 루프가 다음 영상을 덮어쓸 수 있으므로 복사본을 넣는다
            self._q.put_nowait((now - self._t0, frame.copy()))
        except queue.Full:
            self.dropped += 1

    # ---- 쓰기 (인코딩 스레드) ---------------------------------------------
    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    return
                t, frame = item
                for _ in range(frames_due(t, self.fps, self.written)):
                    self._writer.write(frame)
                    self.written += 1
        except BaseException as e:  # 메인 루프의 다음 submit()에서 다시 낸다
            self._error = e

    # ---- 닫기 -----------------------------------------------------------
    def close(self, timeout: float = 30.0) -> dict:
        """큐에 남은 영상을 다 쓰고 파일을 닫는다. 결과 요약을 돌려준다."""
        if self._started:
            self._q.put(None)
            self._thread.join(timeout)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.stats()

    def stats(self) -> dict:
        return {
            "path": self.path,
            "codec": self.codec,
            "fps": self.fps,
            "size_wh": list(self.size) if self.size else None,
            "submitted": self.submitted,
            "dropped": self.dropped,
            "written": self.written,
            "duration_s": round(self.written / self.fps, 2) if self.fps > 0 else None,
            "error": repr(self._error) if self._error else None,
        }
