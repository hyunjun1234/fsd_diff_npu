from __future__ import annotations

import csv
import json
import os
import platform
import queue
import shutil
import threading
import time

from .mjpeg import MjpegWriter, frames_missing

# 주행 녹화: NPU 추론 없이 카메라 전부의 원본을 세그먼트(기본 60초) 단위로 저장한다.
#
# 폴더 구조 (회차 = 시동을 켜고 녹화를 시작해서 끝낼 때까지 한 번)
#   out/drives/<회차 이름 YYYYmmdd_HHMMSS>/
#       drive.json        회차 메타데이터: 설정 파일 사본(본문), 카메라 모드, 시작/끝 시각, 파이 상태, 끝난 이유
#       config.yaml       녹화에 쓴 설정 파일 그대로 복사
#       status.csv        status_interval(기본 60초)마다 프레임 수, 누락 수, 온도, 남은 디스크
#       seg_0000/         녹화 시작 뒤 0~60초
#           front.mjpeg   카메라별 영상(JPEG 이어 붙임) + front.csv(프레임별 센서 시각, 위치)
#           rear.mjpeg, rear.csv ...
#           segment.json  세그먼트를 깨끗하게 닫았을 때만 생김(카메라별 프레임 수, 누락 수, 크기)
#       seg_0001/ ...
#
# 카메라마다 스레드 하나가 "읽기, JPEG 누르기, 쓰기"를 한다. 카메라 수는 설정에서 읽으므로 4대도 그대로 돈다.
# 프레임이 어느 세그먼트에 들어갈지는 센서 시각으로 정해서(시작 시각 + k x 세그먼트 길이) 카메라끼리 경계가 맞다.


# ---------------------------------------------------------------------------
# 작은 도구 (순수 함수, 시험용으로 분리)
# ---------------------------------------------------------------------------
def drive_name(t: float | None = None) -> str:
    """회차 폴더 이름. 시작 시각(현지 시각) YYYYmmdd_HHMMSS."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t if t is not None else time.time()))


def segment_name(k: int) -> str:
    return f"seg_{k:04d}"


def segment_of(sensor_ns: int, t0_ns: int, segment_s: float) -> int:
    """센서 시각이 몇 번째 세그먼트에 들어가는지. 시작보다 앞선 프레임은 0번."""
    if segment_s <= 0:
        return 0
    return max(0, int((sensor_ns - t0_ns) // int(segment_s * 1e9)))


def disk_free_gb(path: str) -> float:
    """path가 있는 디스크의 남은 공간(GB, 10^9 바이트)."""
    return shutil.disk_usage(path).free / 1e9


def read_soc_temp() -> float | None:
    """파이 SoC 온도(도). 맥처럼 없으면 None."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled() -> str | None:
    """vcgencmd get_throttled 값(예: 0x0). 없으면 None."""
    import subprocess

    try:
        r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        s = r.stdout.strip()
        return s.split("=", 1)[1] if "=" in s else None
    except (OSError, subprocess.SubprocessError):
        return None


def pi_state(out_dir: str) -> dict:
    """회차 메타데이터에 남길 파이 상태. 읽지 못한 값은 None."""
    st = {
        "host": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "soc_temp_c": read_soc_temp(),
        "throttled": read_throttled(),
        "disk_free_gb": round(disk_free_gb(out_dir), 2),
    }
    try:
        with open("/proc/uptime") as f:
            st["uptime_s"] = float(f.read().split()[0])
    except (OSError, ValueError):
        st["uptime_s"] = None
    try:
        import picamera2  # type: ignore

        st["picamera2"] = getattr(picamera2, "__version__", None) or os.path.dirname(picamera2.__file__)
    except Exception:
        st["picamera2"] = None
    return st


def camera_mode(cam, src) -> dict:
    """카메라 하나의 녹화 모드: 설정의 크기와 소스 항목, 실제로 잡힌 센서 모드와 센서 이름(Picamera2일 때)."""
    m = {"name": cam.name, "width": cam.width, "height": cam.height, "source": dict(cam.source)}
    sc = getattr(src, "sensor_config", None)
    if sc:
        m["sensor_config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sc.items()}
    picam = getattr(src, "cam", None)
    try:
        props = getattr(picam, "camera_properties", None) or {}
        m["sensor_model"] = props.get("Model")
    except Exception:
        m["sensor_model"] = None
    return m


def write_json_atomic(path: str, obj: dict) -> None:
    """임시 파일에 쓴 뒤 이름을 바꿔, 쓰는 도중 꺼져도 예전 내용이나 새 내용 중 하나는 온전히 남게 한다."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 카메라 한 대 녹화 스레드
# ---------------------------------------------------------------------------
class _CamWorker:
    """카메라 한 대를 맡는 스레드 둘.

    - 읽기 스레드: 카메라에서 영상과 센서 시각을 받아 큐에 넣기만 한다. 디스크가 잠깐 멈춰도 카메라를 놓치지 않게
      여기서는 디스크를 건드리지 않는다. 큐가 가득 차면 그 영상은 버리고 queue_drops로 센다.
    - 쓰기 스레드: 큐에서 꺼내 JPEG로 누르고 세그먼트 파일에 쓴다. 세그먼트가 바뀌면 새 파일을 바로 열고,
      지난 파일의 닫기(fsync 포함)는 따로 띄운 닫기 스레드가 한다. 파이 SD 카드에서 fsync가 1초 넘게 걸려
      그동안 카메라 프레임 33장을 놓친 적이 있어서 이렇게 나눴다.
    """

    def __init__(self, rec: "DriveRecorder", k: int, cam, src, queue_frames: int = 60):
        self.rec, self.k, self.cam, self.src = rec, k, cam, src
        fps = float(cam.source.get("fps", 15) or 15)
        self.period_ns = 1e9 / fps  # 빠진 프레임을 셀 때 쓰는 한 주기
        self.frames = 0  # 파일에 쓴 프레임 수(전체)
        self.gaps = 0  # 센서 시각으로 추정한 빠진 프레임 수(전체). 큐에서 버린 것도 여기 들어간다
        self.queue_drops = 0  # 큐가 가득 차 버린 영상 수(gaps의 일부)
        self.writer: MjpegWriter | None = None
        self.seg = -1
        self.prev_ns: int | None = None
        self.error: BaseException | None = None
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_frames)))
        self._closers: list[threading.Thread] = []
        self.thread = threading.Thread(target=self._read_loop, name=f"cap-{cam.name}", daemon=True)
        self.wthread = threading.Thread(target=self._write_loop, name=f"rec-{cam.name}", daemon=True)
        # 센서 시각을 받는다(Picamera2는 capture_request 하나에서 영상과 SensorTimestamp를 같이 꺼낸다)
        if hasattr(src, "want_timestamp"):
            src.want_timestamp = True

    def start(self) -> None:
        self.wthread.start()
        self.thread.start()

    def join(self, timeout: float) -> None:
        self.thread.join(timeout)
        self.wthread.join(timeout)
        for t in self._closers:
            t.join(timeout)

    def alive(self) -> bool:
        return self.thread.is_alive() or self.wthread.is_alive() or any(t.is_alive() for t in self._closers)

    # ---- 읽기 스레드 ------------------------------------------------------
    def _read_loop(self) -> None:
        rec = self.rec
        try:
            while not rec.stop_event.is_set():
                f = self.src.read()
                wall = time.time()
                if f is None:
                    raise IOError(f"{self.cam.name}: 프레임을 읽지 못했습니다")
                ts = getattr(self.src, "last_timestamp_ns", None)
                ns = int(ts) if ts is not None else time.monotonic_ns()
                try:
                    self._q.put_nowait((f, ns, wall))
                except queue.Full:
                    self.queue_drops += 1  # 다음 프레임의 센서 시각 간격에서 누락으로 함께 잡힌다
        except BaseException as e:  # 녹화 전체를 멈추고 이유를 남긴다
            self.error = e
            rec.request_stop(f"camera_error:{self.cam.name}:{e}")
        finally:
            self._q.put(None)  # 쓰기 스레드에 끝을 알린다(큐가 차 있으면 자리가 날 때까지 기다림)

    # ---- 쓰기 스레드 ------------------------------------------------------
    def _write_loop(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    return
                f, ns, wall = item
                gap = frames_missing(self.prev_ns, ns, self.period_ns)
                self.prev_ns = ns
                seg = segment_of(ns, self.rec.t0_ns, self.rec.segment_s)
                if seg != self.seg:
                    self._rotate(seg)
                self.writer.write(f, self.frames, ns, wall, gap)
                self.frames += 1
                self.gaps += gap
        except BaseException as e:
            self.error = self.error or e
            self.rec.request_stop(f"write_error:{self.cam.name}:{e}")
            # 읽기 스레드가 큐에 막혀 멈추지 않게 남은 것을 비운다
            while True:
                try:
                    if self._q.get_nowait() is None:
                        break
                except queue.Empty:
                    break
        finally:
            self._close_async(self.writer, self.seg)
            self.writer = None

    def _rotate(self, seg: int) -> None:
        old, old_seg = self.writer, self.seg
        folder = os.path.join(self.rec.drive_dir, segment_name(seg))
        self.writer = MjpegWriter(folder, self.cam.name, quality=self.rec.quality)
        self.seg = seg
        self._close_async(old, old_seg)

    def _close_async(self, writer: MjpegWriter | None, seg: int) -> None:
        """지난 세그먼트 파일을 다른 스레드에서 닫는다(fsync가 쓰기를 막지 않게)."""
        if writer is None:
            return

        def run():
            try:
                self.rec._segment_closed(seg, writer.close())
            except BaseException as e:
                self.error = self.error or e

        t = threading.Thread(target=run, name=f"close-{self.cam.name}-{seg}", daemon=True)
        self._closers = [c for c in self._closers if c.is_alive()] + [t]
        t.start()


# ---------------------------------------------------------------------------
# 주행 녹화기
# ---------------------------------------------------------------------------
class DriveRecorder:
    """카메라 소스 목록을 받아 회차 폴더에 세그먼트 단위로 녹화한다.

    사용 순서: start() 다음 run_until_stopped() (또는 직접 poll()을 부르며 기다림) 다음 close().
    다른 스레드나 시그널 처리기에서 request_stop()을 부르면 모든 카메라가 지금 세그먼트를 닫고 끝난다.
    clock, free_fn, health_fn을 바꿔 끼울 수 있어 하드웨어 없이 시험한다.
    """

    def __init__(
        self,
        out_root: str,
        cams,
        sources,
        config_path: str | None = None,
        segment_s: float = 60.0,
        duration_s: float = 0.0,  # 0이면 멈춤 신호나 디스크 부족까지 계속
        min_free_gb: float = 5.0,
        status_interval: float = 60.0,
        quality: int = 85,
        name: str | None = None,
        queue_frames: int = 60,  # 카메라마다 쓰기를 기다릴 수 있는 영상 수(768x576이면 장당 1.3MB)
        free_fn=None,
        health_fn=None,
        log=print,
    ):
        if len(cams) != len(sources):
            raise ValueError("카메라 수와 소스 수가 다릅니다")
        self.cams, self.sources = list(cams), list(sources)
        self.segment_s = float(segment_s)
        self.duration_s = float(duration_s)
        self.min_free_gb = float(min_free_gb)
        self.status_interval = float(status_interval)
        self.quality = int(quality)
        self.queue_frames = int(queue_frames)
        self.config_path = config_path
        self.name = name or drive_name()
        self.drive_dir = os.path.join(out_root, self.name)
        self._free = free_fn or (lambda: disk_free_gb(self.drive_dir))
        self._health = health_fn or (lambda: {"soc_temp_c": read_soc_temp(), "throttled": read_throttled()})
        self._log = log
        self.stop_event = threading.Event()
        self.stop_reason: str | None = None
        self._lock = threading.Lock()
        self._seg_stats: dict[int, list[dict]] = {}  # 세그먼트 번호 -> 닫힌 카메라 요약 목록
        self.segments_closed = 0
        self.meta: dict = {}
        self.t0_ns = 0
        self._workers: list[_CamWorker] = []

    # ---- 멈춤 ------------------------------------------------------------
    def request_stop(self, reason: str) -> None:
        """멈춤 요청. 처음 들어온 이유만 남긴다. 시그널 처리기에서 불러도 된다(이벤트만 세움)."""
        if self.stop_reason is None:
            self.stop_reason = reason
        self.stop_event.set()

    # ---- 세그먼트 닫힘 기록 (카메라 스레드에서 불림) ---------------------
    def _segment_closed(self, seg: int, stats: dict) -> None:
        with self._lock:
            lst = self._seg_stats.setdefault(seg, [])
            lst.append(stats)
            if len(lst) == len(self.cams):
                # 모든 카메라가 이 세그먼트를 닫았다: 깨끗하게 끝난 세그먼트 표시
                self._write_segment_json(seg)

    def _write_segment_json(self, seg: int) -> None:
        """segment.json을 쓴다. 이 파일이 있으면 그 세그먼트의 파일은 모두 닫히고 fsync된 것이다.

        마지막 세그먼트는 어떤 카메라가 경계를 넘기 전에 멈출 수 있다. 그 카메라는 missing_cameras에 적는다.
        """
        lst = self._seg_stats.get(seg, [])
        have = {s["camera"] for s in lst}
        write_json_atomic(
            os.path.join(self.drive_dir, segment_name(seg), "segment.json"),
            {"segment": seg, "segment_s": self.segment_s, "closed_cleanly": True,
             "cameras": sorted(lst, key=lambda s: s["camera"]),
             "missing_cameras": [c.name for c in self.cams if c.name not in have]},
        )
        self.segments_closed += 1

    # ---- 시작 -----------------------------------------------------------
    def start(self) -> None:
        os.makedirs(self.drive_dir, exist_ok=False)  # 같은 이름 회차를 덮어쓰지 않는다
        free = self._free()
        config_text = None
        if self.config_path:
            shutil.copyfile(self.config_path, os.path.join(self.drive_dir, "config.yaml"))
            with open(self.config_path, encoding="utf-8") as f:
                config_text = f.read()
        self.t0_ns = time.monotonic_ns()
        self.meta = {
            "drive": self.name,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_epoch": time.time(),
            "t0_monotonic_ns": self.t0_ns,
            "config_path": self.config_path,
            "config_text": config_text,
            "cameras": [camera_mode(c, s) for c, s in zip(self.cams, self.sources)],
            "segment_s": self.segment_s,
            "jpeg_quality": self.quality,
            "min_free_gb": self.min_free_gb,
            "pi_state_start": pi_state(self.drive_dir),
            "ended": None,
            "stop_reason": None,
        }
        write_json_atomic(os.path.join(self.drive_dir, "drive.json"), self.meta)
        if free < self.min_free_gb:
            # 시작부터 공간이 모자라면 한 장도 쓰지 않고 끝낸다
            self.request_stop(f"disk_low:{free:.2f}GB<{self.min_free_gb:g}GB")
            return
        self._status_f = open(os.path.join(self.drive_dir, "status.csv"), "w", newline="")
        self._status = csv.writer(self._status_f)
        cols = ["elapsed_s", "wall_time", "soc_temp_c", "throttled", "disk_free_gb", "segments_closed"]
        for c in self.cams:
            cols += [f"{c.name}_frames", f"{c.name}_gaps", f"{c.name}_gaps_interval", f"{c.name}_queue_drops"]
        self._status.writerow(cols)
        self._status_f.flush()
        self._last_gaps = [0] * len(self.cams)
        self._workers = [
            _CamWorker(self, k, c, s, self.queue_frames) for k, (c, s) in enumerate(zip(self.cams, self.sources))
        ]
        for w in self._workers:
            w.start()

    # ---- 상태 기록 ------------------------------------------------------
    def write_status(self) -> dict:
        """지금까지의 프레임 수와 누락 수(전체, 직전 기록 이후), 온도, 남은 디스크를 한 줄 남긴다."""
        h = self._health() or {}
        free = self._free()
        elapsed = (time.monotonic_ns() - self.t0_ns) / 1e9
        row = [f"{elapsed:.1f}", f"{time.time():.3f}", h.get("soc_temp_c"), h.get("throttled"), f"{free:.2f}", self.segments_closed]
        parts = []
        for k, w in enumerate(self._workers):
            d = w.gaps - self._last_gaps[k]
            self._last_gaps[k] = w.gaps
            row += [w.frames, w.gaps, d, w.queue_drops]
            parts.append(f"{w.cam.name} {w.frames}장 누락 {w.gaps}(+{d}) 큐 {w._q.qsize()}")
        self._status.writerow(row)
        self._status_f.flush()
        self._log(f"[{elapsed:6.1f}s] " + ", ".join(parts) + f", SoC {h.get('soc_temp_c')}도, 남은 디스크 {free:.1f}GB")
        return {"elapsed_s": elapsed, "free_gb": free}

    # ---- 기다리기 -------------------------------------------------------
    def run_until_stopped(self, poll_s: float = 0.5, disk_check_s: float = 2.0) -> None:
        """멈춤 신호, 녹화 시간 끝, 디스크 부족, 카메라 오류 중 하나가 생길 때까지 기다린다."""
        t_start = time.monotonic()
        t_status = t_start
        t_disk = t_start
        while not self.stop_event.is_set():
            self.stop_event.wait(poll_s)
            now = time.monotonic()
            if self.duration_s > 0 and now - t_start >= self.duration_s:
                self.request_stop("duration")
            if now - t_disk >= disk_check_s:
                t_disk = now
                free = self._free()
                if free < self.min_free_gb:
                    self.request_stop(f"disk_low:{free:.2f}GB<{self.min_free_gb:g}GB")
            if self.status_interval > 0 and now - t_status >= self.status_interval:
                t_status = now
                self.write_status()

    # ---- 끝내기 ---------------------------------------------------------
    def close(self, join_timeout: float = 10.0) -> dict:
        """카메라 스레드가 지금 세그먼트를 닫을 때까지 기다리고, 회차 메타데이터를 마무리한다."""
        self.request_stop(self.stop_reason or "close")
        for w in self._workers:
            w.join(join_timeout)
        with self._lock:
            # 끝 무렵 일부 카메라만 들어간 세그먼트도 닫혔으면 표시한다(스레드가 아직 살아 있으면 남겨 둔다)
            if all(not w.alive() for w in self._workers):
                for seg, lst in sorted(self._seg_stats.items()):
                    if len(lst) < len(self.cams):
                        self._write_segment_json(seg)
        if self._workers:
            self.write_status()
            self._status_f.close()
        self.meta.update(
            {
                "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s": round((time.monotonic_ns() - self.t0_ns) / 1e9, 3),
                "stop_reason": self.stop_reason,
                "segments_closed": self.segments_closed,
                "totals": [
                    {"camera": w.cam.name, "frames": w.frames, "gaps": w.gaps, "queue_drops": w.queue_drops,
                     "error": repr(w.error) if w.error else None, "thread_alive": w.alive()}
                    for w in self._workers
                ],
                "pi_state_end": pi_state(self.drive_dir),
            }
        )
        write_json_atomic(os.path.join(self.drive_dir, "drive.json"), self.meta)
        return self.meta


# ---------------------------------------------------------------------------
# 회차 목록과 용량
# ---------------------------------------------------------------------------
def dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def list_drives(root: str) -> list[dict]:
    """회차 폴더마다 시작 시각, 길이, 세그먼트 수, 프레임 수, 용량, 끝난 이유를 모은다."""
    rows = []
    if not os.path.isdir(root):
        return rows
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if d.startswith(".") or not os.path.isdir(p):
            continue  # 점으로 시작하는 폴더는 치워 둔 것으로 보고 뺀다
        meta = {}
        try:
            with open(os.path.join(p, "drive.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
        segs = [s for s in os.listdir(p) if s.startswith("seg_")]
        clean = sum(os.path.exists(os.path.join(p, s, "segment.json")) for s in segs)
        rows.append(
            {
                "drive": d,
                "started": meta.get("started"),
                "elapsed_s": meta.get("elapsed_s"),
                "segments": len(segs),
                "segments_clean": clean,
                "frames": sum(t.get("frames", 0) for t in meta.get("totals") or []),
                "gaps": sum(t.get("gaps", 0) for t in meta.get("totals") or []),
                "bytes": dir_bytes(p),
                "stop_reason": meta.get("stop_reason"),
            }
        )
    return rows
