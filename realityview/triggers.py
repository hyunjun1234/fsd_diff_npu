from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .mjpeg import MjpegWriter

# 트리거 클립: 실시간 실행 중 모델이 약해 보이는 순간의 앞뒤 원본 프레임을 따로 저장한다(테슬라 트리거를 작게 흉내).
#
# 트리거 종류
#   ambiguous     : 검출 점수가 애매한 상자(기본 0.3 이상 0.5 미만)가 나온 프레임
#   track_new     : 새 추적 ID가 생긴 순간(근처에서 사라진 ID가 없을 때)
#   track_switch  : 어떤 ID가 사라진 같은 프레임에 그 근처에서 새 ID가 생긴 순간(ID가 바뀐 것으로 본다)
#   npu_mismatch  : --shadow로 두 NPU를 같이 돌릴 때 짝을 못 찾은 상자가 정한 수 이상인 프레임
#
# 동작
#   - 원본 프레임을 링 버퍼에 pre_s초만큼(그리고 max_buffer_mb 안에서) 들고 있다.
#   - 트리거가 나오면 버퍼 내용을 앞부분으로 삼고 post_s초 동안 프레임을 더 모은 뒤, 쓰기 스레드가 저장한다.
#   - 모으는 동안 나온 다른 트리거는 새 클립을 만들지 않고 지금 클립의 events에 더한다.
#   - 같은 종류 트리거는 cooldown_s초 안에 다시 클립을 만들지 않는다(suppressed로 센다).
#
# 저장: <out_dir>/<YYYYmmdd_HHMMSS>_f<프레임 6자리>_<종류>/
#   {카메라}.mjpeg, {카메라}.csv   주행 녹화와 같은 형식이라 run_demo.py --replay 로 바로 넣을 수 있다
#   trigger.json                  트리거 목록(종류, 점수, 프레임 번호, 자세한 값), 클립 프레임 범위, 설정값


# ---------------------------------------------------------------------------
# 트리거 사건
# ---------------------------------------------------------------------------
@dataclass
class TriggerEvent:
    kind: str
    score: float
    frame: int
    detail: dict = field(default_factory=dict)


def ambiguous_event(frame: int, results, camera_names: Sequence[str], lo: float, hi: float) -> TriggerEvent | None:
    """점수가 [lo, hi) 안인 상자가 있으면 사건 하나. score는 그 상자들 중 가장 낮은 점수."""
    boxes = []
    for k, r in enumerate(results):
        for d in r.detections:
            if lo <= d.score < hi:
                boxes.append({"camera": camera_names[k], "label": d.label, "score": round(float(d.score), 4),
                              "box": [round(float(v), 1) for v in d.box]})
    if not boxes:
        return None
    return TriggerEvent("ambiguous", min(b["score"] for b in boxes), frame, {"boxes": boxes[:20], "count": len(boxes)})


class TrackIdWatcher:
    """프레임마다 트랙 목록을 받아 새 ID와 바뀐 ID를 찾는다.

    switch_dist: 사라진 ID와 새 ID 사이 거리(m, 차량 좌표)가 이 안이면 ID가 바뀐 것으로 본다.
    """

    def __init__(self, switch_dist: float = 2.0):
        self.switch_dist = float(switch_dist)
        self.seen: set[int] = set()
        self.prev: dict[int, np.ndarray] = {}

    def update(self, frame: int, tracks) -> list[TriggerEvent]:
        cur = {t.tid: np.asarray(t.xy, dtype=float) for t in tracks}
        lost = {tid: xy for tid, xy in self.prev.items() if tid not in cur}
        events = []
        for tid, xy in cur.items():
            if tid in self.seen:
                continue
            near = [(float(np.linalg.norm(xy - lxy)), ltid) for ltid, lxy in lost.items()]
            near = [n for n in near if n[0] <= self.switch_dist]
            label = next((t.label for t in tracks if t.tid == tid), "")
            if near:
                d, old = min(near)
                events.append(TriggerEvent("track_switch", round(d, 3), frame, {"old_id": old, "new_id": tid, "label": label}))
            else:
                events.append(TriggerEvent("track_new", 0.0, frame, {"new_id": tid, "label": label,
                                                                     "xy": [round(float(v), 2) for v in xy]}))
        self.seen.update(cur)
        self.prev = cur
        return events


def mismatch_event(frame: int, rows: Sequence[dict], min_unmatched: int = 1) -> TriggerEvent | None:
    """섀도 비교 줄(카메라별)에서 짝 못 찾은 상자 수 합이 min_unmatched 이상이면 사건 하나. score = 그 수."""
    n = sum(int(r.get("shadow_missed", 0)) + int(r.get("shadow_extra", 0)) for r in rows)
    if n < max(1, int(min_unmatched)):
        return None
    cams = [{k: r.get(k) for k in ("camera", "n_primary", "n_shadow", "matched", "shadow_missed", "shadow_extra", "iou_min")}
            for r in rows]
    return TriggerEvent("npu_mismatch", float(n), frame, {"cameras": cams})


# ---------------------------------------------------------------------------
# 링 버퍼와 클립 저장
# ---------------------------------------------------------------------------
@dataclass
class _Clip:
    name: str
    end_t: float
    events: list[TriggerEvent]
    frames: list  # (frame, t, t_ns, [카메라별 영상], 벽시계)
    trigger_frame: int


class TriggerClipper:
    def __init__(
        self,
        out_dir: str,
        camera_names: Sequence[str],
        pre_s: float = 2.0,
        post_s: float = 2.0,
        cooldown_s: float = 10.0,
        max_buffer_mb: float = 300.0,
        quality: int = 85,
        max_pending: int = 2,
        log=print,
    ):
        self.out_dir = out_dir
        self.camera_names = list(camera_names)
        self.pre_s, self.post_s, self.cooldown_s = float(pre_s), float(post_s), float(cooldown_s)
        self.max_bytes = float(max_buffer_mb) * 1e6
        self.quality = int(quality)
        self._log = log
        self._ring: deque = deque()  # (frame, t, t_ns, frames, 바이트)
        self._ring_bytes = 0
        self._active: _Clip | None = None
        self._last_fire: dict[str, float] = {}
        self.stats = {"fired": {}, "suppressed": {}, "merged": 0, "clips_written": 0, "clips_dropped": 0,
                      "ring_trimmed_by_memory": 0}
        self.written: list[str] = []
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self._thread = threading.Thread(target=self._write_loop, name="trigger-writer", daemon=True)
        self._thread.start()

    # ---- 프레임 넣기 (메인 루프, 프레임마다) -------------------------------
    def push(self, frame: int, frames: Sequence[np.ndarray], t: float) -> None:
        """원본 프레임을 링 버퍼에 넣는다. 영상은 복사하지 않는다(카메라 소스가 매번 새 배열을 준다)."""
        nbytes = int(sum(f.nbytes for f in frames))
        item = (frame, t, int(t * 1e9), list(frames), nbytes, time.time())  # 끝 값은 벽시계(csv wall_time)
        self._ring.append(item)
        self._ring_bytes += nbytes
        # 오래된 것과 메모리 한도를 넘는 것을 버린다(지금 프레임 하나는 남긴다)
        while len(self._ring) > 1 and (self._ring[0][1] < t - self.pre_s or self._ring_bytes > self.max_bytes):
            if self._ring_bytes > self.max_bytes and self._ring[0][1] >= t - self.pre_s:
                self.stats["ring_trimmed_by_memory"] += 1
            old = self._ring.popleft()
            self._ring_bytes -= old[4]
        if self._active is not None:
            if not self._active.frames or self._active.frames[-1][0] != frame:
                self._active.frames.append(item[:4] + item[5:])
            if t >= self._active.end_t - 1e-6:  # 부동소수 끝자리 오차로 한 장 더 들어가지 않게
                self._finish_active()

    # ---- 트리거 받기 ------------------------------------------------------
    def fire(self, events: Sequence[TriggerEvent], t: float) -> None:
        for ev in events:
            if ev is None:
                continue
            last = self._last_fire.get(ev.kind)
            if last is not None and t - last < self.cooldown_s:
                self.stats["suppressed"][ev.kind] = self.stats["suppressed"].get(ev.kind, 0) + 1
                continue
            self._last_fire[ev.kind] = t
            self.stats["fired"][ev.kind] = self.stats["fired"].get(ev.kind, 0) + 1
            if self._active is not None:
                self._active.events.append(ev)  # 모으는 중인 클립에 합친다
                self.stats["merged"] += 1
                continue
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"{stamp}_f{ev.frame:06d}_{ev.kind}"
            self._active = _Clip(name, t + self.post_s, [ev], [it[:4] + it[5:] for it in self._ring], ev.frame)
            if self.post_s <= 0:
                self._finish_active()

    def _finish_active(self) -> None:
        clip, self._active = self._active, None
        try:
            self._q.put_nowait(clip)
        except queue.Full:
            # 쓰기가 밀리면 메모리를 지키려고 이 클립은 버린다
            self.stats["clips_dropped"] += 1
            self._log(f"트리거 클립 저장이 밀려 버림: {clip.name}")

    # ---- 저장 (쓰기 스레드) -----------------------------------------------
    def _write_loop(self) -> None:
        while True:
            clip = self._q.get()
            if clip is None:
                return
            try:
                self._write(clip)
            except Exception as e:  # 저장 실패가 실시간 실행을 멈추지 않게 기록만 한다
                self._log(f"트리거 클립 저장 실패 {clip.name}: {e!r}")

    def _write(self, clip: _Clip) -> None:
        folder = os.path.join(self.out_dir, clip.name)
        writers = [MjpegWriter(folder, n, quality=self.quality) for n in self.camera_names]
        try:
            for frame, _, t_ns, frames, wall in clip.frames:
                for w, f in zip(writers, frames):
                    # 카메라 모두 같은 시각을 적어 --replay 때 같은 프레임끼리 짝지어지게 한다
                    w.write(f, frame, t_ns, wall)
        finally:
            for w in writers:
                w.close()
        meta = {
            "clip": clip.name,
            "trigger_frame": clip.trigger_frame,
            "first_frame": clip.frames[0][0] if clip.frames else None,
            "last_frame": clip.frames[-1][0] if clip.frames else None,
            "frames": len(clip.frames),
            "cameras": self.camera_names,
            "pre_s": self.pre_s,
            "post_s": self.post_s,
            "cooldown_s": self.cooldown_s,
            "events": [asdict(e) for e in clip.events],
        }
        with open(os.path.join(folder, "trigger.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.stats["clips_written"] += 1
        self.written.append(folder)
        self._log(f"트리거 클립 저장: {folder} ({len(clip.frames)}프레임, {[e.kind for e in clip.events]})")

    # ---- 끝내기 ---------------------------------------------------------
    def close(self, timeout: float = 30.0) -> dict:
        """모으던 클립은 그때까지의 프레임으로 저장하고, 쓰기 스레드가 끝날 때까지 기다린다."""
        if self._active is not None:
            clip, self._active = self._active, None
            self._q.put(clip)  # 마지막 클립은 기다려서라도 넣는다
        self._q.put(None)
        self._thread.join(timeout)
        self._ring.clear()
        self._ring_bytes = 0
        return dict(self.stats, clips=self.written)
