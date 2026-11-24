from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np

from .bev import BevGrid, IpmCompositor, merge_cross_camera, project_detections
from .camera_model import Camera
from .detectors import Backend
from .profiling import StageTimer
from .render import annotate_camera, compose_dashboard, render_bev
from .tracker import BevTracker, Track
from .types import FrameResult

# 한 프레임 처리 순서
#   카메라 프레임 -> 검출/세그(NPU) -> 지면 투영, 병합, 추적, 바닥 영상 합성(CPU) -> 렌더링
# 렌더링은 기본으로 같은 스레드에서 하고, render_thread=True면 별도 스레드에서 한다.


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

    # ---- 화면 갱신 횟수 -------------------------------------------------
    @property
    def render_count(self) -> int:
        """지금까지 완성한 대시보드 수. 렌더링 스레드 모드에서는 루프 횟수보다 적을 수 있다."""
        return self._worker.done if self._worker is not None else self._sync_renders

    @property
    def render_dropped(self) -> int:
        """렌더링 스레드가 그리기 전에 새 프레임에 밀려 버린 작업 수."""
        return self._worker.dropped if self._worker is not None else 0

    # ---- 한 프레임 처리 -------------------------------------------------
    def step(self, frames: list[np.ndarray], dt: float) -> StepOutput:
        tm = self.timer
        t0 = time.perf_counter()

        # 1) 검출(+세그). 전처리, NPU 추론, NMS 파싱의 세부 시간은 백엔드가 같은 타이머에 적는다
        with tm.stage("detect_total"):
            results = self.backend.infer(frames)
        t1 = time.perf_counter()

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
        self.backend.close()
