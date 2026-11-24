from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np

from .bev import BevGrid, IpmCompositor, merge_cross_camera, project_detections
from .camera_model import Camera
from .detectors import Backend, SegBackend
from .profiling import StageTimer
from .render import annotate_camera, compose_dashboard, render_bev
from .tracker import BevTracker, Track
from .types import FrameResult

# 한 프레임 처리 순서
#   카메라 프레임 -> 검출/세그(NPU) -> 지면 투영, 병합, 추적, 바닥 영상 합성(CPU) -> 렌더링
# 렌더링은 기본으로 같은 스레드에서 하고, render_thread=True면 별도 스레드에서 한다.
# seg_backend를 주면 도로 세그를 다른 NPU에서 별도 스레드로 N 프레임마다 돌리고, 늦으면 최근 마스크를 쓴다.


# ---------------------------------------------------------------------------
# 결과 묶음
# ---------------------------------------------------------------------------
@dataclass
class StepOutput:
    dashboard: np.ndarray  # 왼쪽 탑뷰 + 오른쪽 카메라 영상(렌더링 스레드 모드에서는 가장 최근 완성본)
    bev: np.ndarray  # 탑뷰만
    tracks: list[Track]
    results: list[FrameResult]
    timings: dict  # 이번 프레임의 대략적인 구간 시간(ms). 자세한 통계는 StageTimer로 본다


@dataclass
class _RenderJob:
    """렌더링에 필요한 것을 한 프레임분 모아 둔 것. 렌더링 스레드로 넘길 때 쓴다."""

    tracks: list[Track]
    ground: np.ndarray | None
    road: np.ndarray | None
    frames: list[np.ndarray]
    results: list[FrameResult]
    fps: float | None


def _snapshot_tracks(tracks: list[Track]) -> list[Track]:
    """트랙 복사본. 추적기는 다음 update()에서 같은 Track 객체를 고치므로,
    다른 스레드가 그리는 동안 값이 바뀌지 않게 복사해서 넘긴다."""
    return [dataclasses.replace(t, xy=np.array(t.xy, copy=True), vel=np.array(t.vel, copy=True)) for t in tracks]


# ---------------------------------------------------------------------------
# 렌더링 스레드 (계획 3.3절 대응: 렌더링을 별도 스레드로 분리)
# ---------------------------------------------------------------------------
class _RenderWorker:
    """가장 최근 작업 하나만 그리는 렌더링 스레드.

    - 메인 루프는 submit()으로 작업을 넘기고 기다리지 않는다.
    - 아직 그리지 못한 작업이 있는데 새 작업이 오면 옛 작업은 버린다(dropped). 화면은 늘 최신 프레임.
    - OpenCV와 numpy의 큰 연산은 GIL을 놓고 돌기 때문에 파이썬 스레드로도 CPU 코어를 나눠 쓸 수 있다.
    """

    def __init__(self, render_fn, timer: StageTimer):
        self._render_fn = render_fn
        self._timer = timer
        self._cond = threading.Condition()
        self._job: _RenderJob | None = None
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._stop = False
        self.done = 0  # 완성한 대시보드 수
        self.dropped = 0  # 그리기 전에 새 작업에 밀려 버린 수
        # daemon: 혹시 렌더링이 멈춰도 프로세스 종료를 막지 않게
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)
        self._thread.start()

    def submit(self, job: _RenderJob) -> None:
        with self._cond:
            if self._job is not None:
                self.dropped += 1
            self._job = job
            self._cond.notify()

    def latest(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._cond:
            return self._latest

    def set_latest(self, out: tuple[np.ndarray, np.ndarray]) -> None:
        """첫 프레임처럼 메인 스레드가 직접 그린 결과를 최신 결과로 등록한다."""
        with self._cond:
            self._latest = out
            self.done += 1

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job, self._job = self._job, None
            t0 = time.perf_counter()
            out = self._render_fn(job)
            # 렌더링 시간은 다른 스레드에서 쟀으므로 add()로 넣는다(다음 end_frame 때 집계)
            self._timer.add("render", (time.perf_counter() - t0) * 1e3)
            with self._cond:
                self._latest = out
                self.done += 1

    def close(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout)


# ---------------------------------------------------------------------------
# 세그 스레드 (계획 4.3절: 검출은 Hailo, 세그는 DX-M1을 N 프레임마다, 늦으면 최근 마스크 재사용)
# ---------------------------------------------------------------------------
class _SegWorker:
    """다른 NPU에서 도로 세그를 돌리는 스레드. 구조는 _RenderWorker와 같다.

    - 메인 루프는 N 프레임마다 submit()으로 카메라 영상을 넘기고 기다리지 않는다.
      세그는 검출과 같은 시간에 다른 NPU에서 돈다.
    - 아직 시작하지 못한 작업이 있는데 새 작업이 오면 옛 작업은 버린다(dropped).
    - 메인 루프는 latest()로 가장 최근에 끝난 마스크를 가져다 쓴다. 세그가 늦으면
      몇 프레임 전 마스크가 그대로 쓰인다(age로 몇 프레임 묵었는지 본다).
    - 세그 함수가 예외를 내면 저장해 두었다가 메인 루프에서 다시 낸다(조용히 멈추지 않게).
    """

    def __init__(self, seg_fn, timer: StageTimer):
        self._seg_fn = seg_fn  # frames → 카메라별 도로 마스크 목록
        self._timer = timer
        self._cond = threading.Condition()
        self._job: tuple[int, list[np.ndarray]] | None = None
        self._latest: tuple[int, list[np.ndarray]] | None = None  # (영상을 넘긴 프레임 번호, 마스크)
        self._busy = False
        self._stop = False
        self.error: BaseException | None = None
        self.submitted = 0  # 넘긴 작업 수
        self.done = 0  # 끝낸 작업 수
        self.dropped = 0  # 시작 전에 새 작업에 밀려 버린 수
        self._thread = threading.Thread(target=self._run, name="seg", daemon=True)
        self._thread.start()

    def submit(self, frame_idx: int, frames: list[np.ndarray]) -> None:
        # 카메라 소스가 버퍼를 다시 쓸 수 있으므로 복사해서 넘긴다(768x576 두 장에 1ms 안쪽)
        job = (frame_idx, [np.array(f, copy=True) for f in frames])
        with self._cond:
            if self._job is not None:
                self.dropped += 1
            self._job = job
            self.submitted += 1
            self._cond.notify()

    def latest(self) -> tuple[int, list[np.ndarray]] | None:
        with self._cond:
            if self.error is not None:
                raise RuntimeError(f"세그 스레드가 멈췄습니다: {self.error!r}") from self.error
            return self._latest

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._busy or self._job is not None

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                (idx, frames), self._job = self._job, None
                self._busy = True
            t0 = time.perf_counter()
            try:
                masks = list(self._seg_fn(frames))
            except BaseException as e:  # noqa: BLE001 - 메인 루프에 넘기려고 무엇이든 잡는다
                with self._cond:
                    self.error = e
                    self._busy = False
                return
            self._timer.add("seg_async", (time.perf_counter() - t0) * 1e3)
            with self._cond:
                self._latest = (idx, masks)
                self.done += 1
                self._busy = False
                self._cond.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        """맡은 작업이 모두 끝날 때까지 기다린다(시험과 종료용). 끝났으면 True."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while (self._busy or self._job is not None) and self.error is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
            return True

    def close(self, timeout: float = 5.0) -> bool:
        """스레드를 멈추고 기다린다. 제때 끝났으면 True(그래야 NPU 엔진을 닫아도 안전하다)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
class RealityViewPipeline:
    def __init__(
        self,
        cameras: list[Camera],
        backend: Backend,
        grid: BevGrid | None = None,
        show_ground: bool = True,  # False면 바닥 영상 IPM 합성을 건너뛴다(CPU 절약)
        show_cameras: bool = True,  # False면 대시보드 오른쪽 카메라 영상을 그리지 않는다
        timer: StageTimer | None = None,  # 단계별 시간 측정(기본 꺼짐)
        render_thread: bool = False,  # True면 렌더링을 별도 스레드에서 한다(기본은 같은 스레드)
        seg_backend: SegBackend | None = None,  # 주면 도로 세그를 이 백엔드로 별도 스레드에서 돌린다(기본 없음)
        seg_every: int = 3,  # seg_backend가 있을 때 세그를 몇 프레임마다 넘길지
    ):
        self.cameras = cameras
        self.backend = backend
        self.grid = grid or BevGrid()
        self.ipm = IpmCompositor(cameras, self.grid)
        self.tracker = BevTracker()
        self.show_ground = show_ground
        self.show_cameras = show_cameras
        self.timer = timer or StageTimer(enabled=False)
        self._fps = None
        self._last_step_t = None
        self._sync_renders = 0
        self._worker = _RenderWorker(self._render, self.timer) if render_thread else None
        # 두 NPU 동시 실행 옵션: 검출은 backend(예: Hailo), 세그는 seg_backend(예: DX-M1)
        self.seg_backend = seg_backend
        self.seg_every = max(1, int(seg_every))
        self._seg = _SegWorker(seg_backend.segment, self.timer) if seg_backend is not None else None
        self._frame_idx = 0
        self._seg_ages: list[int] = []  # 프레임마다 쓴 마스크가 몇 프레임 묵었는지(마스크가 있을 때만)
        self._seg_missing = 0  # 아직 첫 마스크가 없어 마스크 없이 지나간 프레임 수

    # ---- 화면 갱신 횟수 -------------------------------------------------
    @property
    def render_count(self) -> int:
        """지금까지 완성한 대시보드 수. 렌더링 스레드 모드에서는 루프 횟수보다 적을 수 있다."""
        return self._worker.done if self._worker is not None else self._sync_renders

    @property
    def render_dropped(self) -> int:
        """렌더링 스레드가 그리기 전에 새 프레임에 밀려 버린 작업 수."""
        return self._worker.dropped if self._worker is not None else 0

    @property
    def seg_stats(self) -> dict | None:
        """별도 스레드 세그의 통계. seg_backend가 없으면 None.

        submitted/done/dropped: 넘긴, 끝낸, 시작 전에 밀려 버린 작업 수
        mask_age_mean/max: 쓴 마스크가 평균, 최대 몇 프레임 묵었는지(0이면 같은 프레임 영상의 마스크)
        frames_without_mask: 첫 마스크가 나오기 전이라 마스크 없이 지나간 프레임 수
        """
        if self._seg is None:
            return None
        ages = self._seg_ages
        return {
            "seg_every": self.seg_every,
            "submitted": self._seg.submitted,
            "done": self._seg.done,
            "dropped": self._seg.dropped,
            "mask_age_mean": float(np.mean(ages)) if ages else None,
            "mask_age_max": int(max(ages)) if ages else None,
            "frames_without_mask": self._seg_missing,
        }

    # ---- 한 프레임 처리 -------------------------------------------------
    def step(self, frames: list[np.ndarray], dt: float) -> StepOutput:
        tm = self.timer
        t0 = time.perf_counter()
        idx = self._frame_idx
        self._frame_idx += 1

        # 0) 별도 스레드 세그: N 프레임마다 영상을 넘긴다. 아래 검출과 같은 시간에 다른 NPU에서 돈다
        if self._seg is not None and idx % self.seg_every == 0:
            self._seg.submit(idx, frames)

        # 1) 검출(+세그). 전처리, NPU 추론, NMS 파싱의 세부 시간은 백엔드가 같은 타이머에 적는다
        with tm.stage("detect_total"):
            results = self.backend.infer(frames)
        t1 = time.perf_counter()

        # 1-1) 별도 스레드 세그의 가장 최근 마스크를 붙인다. 이번 프레임 것이 아직이면 이전 마스크를 쓴다
        if self._seg is not None:
            self._attach_async_masks(idx, results)

        # 2) 검출 박스의 접지점을 지면(차량 좌표)으로 투영
        with tm.stage("ipm_project"):
            objs = []
            for i, (cam, r) in enumerate(zip(self.cameras, results)):
                objs += project_detections(i, cam, r.detections)

        # 3) 카메라 경계에서 겹친 같은 물체 합치기
        with tm.stage("merge"):
            objs = merge_cross_camera(objs)

        # 4) 추적(ID 유지, 속도 추정)
        with tm.stage("track"):
            tracks = self.tracker.update(objs, dt)

        # 5) 바닥 영상과 도로 마스크를 탑뷰로 합성(IPM remap)
        with tm.stage("bev_warp"):
            ground = self.ipm.warp(frames) if self.show_ground else None
            road = None
            if any(r.road_mask is not None for r in results):
                masks = [
                    r.road_mask if r.road_mask is not None else np.zeros(f.shape[:2], np.uint8)
                    for r, f in zip(results, frames)
                ]
                road = self.ipm.warp(masks, interpolation=0)
        t2 = time.perf_counter()

        # 화면에 찍는 FPS는 step() 호출 간격(카메라 획득까지 포함한 실제 갱신 주기)으로 구한다
        if self._last_step_t is not None:
            period = max(t0 - self._last_step_t, 1e-6)
            self._fps = 1.0 / period if self._fps is None else 0.9 * self._fps + 0.1 / period
        self._last_step_t = t0

        # 6) 렌더링(탑뷰, 카메라 주석, 대시보드 합치기)
        job = _RenderJob(tracks, ground, road, frames, results, self._fps)
        if self._worker is None:
            with tm.stage("render"):
                dash, bev = self._render(job)
            self._sync_renders += 1
        else:
            job.tracks = _snapshot_tracks(tracks)
            latest = self._worker.latest()
            if latest is None:
                # 첫 프레임은 보여 줄 결과가 없으므로 직접 그린다(호출자가 None을 받지 않게)
                with tm.stage("render"):
                    latest = self._render(job)
                self._worker.set_latest(latest)
            else:
                self._worker.submit(job)
            dash, bev = latest
        t3 = time.perf_counter()

        timings = {"npu_ms": (t1 - t0) * 1e3, "bev_ms": (t2 - t1) * 1e3, "render_ms": (t3 - t2) * 1e3}
        return StepOutput(dash, bev, tracks, results, timings)

    def _attach_async_masks(self, idx: int, results: list[FrameResult]) -> None:
        latest = self._seg.latest()  # 세그 스레드가 예외로 멈췄으면 여기서 다시 낸다
        if latest is None or len(latest[1]) != len(results):
            self._seg_missing += 1
            return
        mask_idx, masks = latest
        # 세그 NPU를 따로 고른 것이므로 검출 백엔드가 만든 마스크가 있어도 이 마스크로 바꾼다
        for r, m in zip(results, masks):
            r.road_mask = m
        self._seg_ages.append(idx - mask_idx)

    def _render(self, job: _RenderJob) -> tuple[np.ndarray, np.ndarray]:
        """탑뷰와 대시보드를 그린다. 같은 스레드와 렌더링 스레드가 함께 쓴다."""
        bev = render_bev(self.grid, job.tracks, job.ground, job.road, self.ipm.coverage())
        views = []
        if self.show_cameras:
            views = [
                annotate_camera(f, r.detections, c.name, r.road_mask)
                for f, r, c in zip(job.frames, job.results, self.cameras)
            ]
        dash = compose_dashboard(bev, views, job.fps)
        return dash, bev

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
        if self._seg is not None:
            # 세그 스레드가 NPU를 쓰는 중에 엔진을 닫지 않도록 먼저 멈추고 기다린다
            if self._seg.close(timeout=5.0):
                self.seg_backend.close()
            else:
                print("경고: 세그 스레드가 5초 안에 끝나지 않아 세그 엔진을 닫지 않았습니다", flush=True)
        self.backend.close()
