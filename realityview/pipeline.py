from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np

from .bev import BevGrid, IpmCompositor, merge_cross_camera, project_detections
from .camera_model import Camera
from .detectors import Backend, SegBackend
from .lanes import LaneConfig, LaneCurve, LaneDetector, LaneSmoother, box_masks, snapshot_lanes
from .profiling import StageTimer
from .render import annotate_camera, compose_dashboard, compose_dashboard_3d, render_bev
from .render3d import Renderer3D, View3DConfig
from .tracker import BevTracker, Track
from .traffic_lights import AERIAL_LABELS, TrafficLight, TrafficLightConfig, TrafficLightReader
from .types import FrameResult

# 한 프레임 처리 순서
#   카메라 프레임 -> 검출/세그(NPU) -> 지면 투영, 병합, 추적, 바닥 영상 합성(CPU) -> 렌더링
# 렌더링은 기본으로 같은 스레드에서 하고, render_thread=True면 별도 스레드에서 한다.
# seg_backend를 주면 도로 세그를 다른 NPU에서 별도 스레드로 N 프레임마다 돌리고, 늦으면 최근 마스크를 쓴다.
# view로 화면을 고른다: "bev"(기본, 탑뷰), "3d"(자차 뒤쪽 위 시점), "both"(탑뷰와 3D를 나란히).
# lanes를 주면 탑뷰 바닥 영상에서 차선을 찾고(lanes.py), traffic_lights를 주면 신호등 상자의 색을 판별한다
# (traffic_lights.py). 둘 다 기본은 꺼져 있다. 신호등 같은 공중 물체(AERIAL_LABELS)는 늘 지면 투영에서 뺀다.


# ---------------------------------------------------------------------------
# 결과 묶음
# ---------------------------------------------------------------------------
@dataclass
class StepOutput:
    dashboard: np.ndarray  # 화면 전체(렌더링 스레드 모드에서는 가장 최근 완성본). 배치는 view에 따라 다르다
    bev: np.ndarray | None  # 탑뷰만. view="3d"이면 탑뷰를 그리지 않으므로 None
    tracks: list[Track]
    results: list[FrameResult]
    timings: dict  # 이번 프레임의 대략적인 구간 시간(ms). 자세한 통계는 StageTimer로 본다
    view3d: np.ndarray | None = None  # 3D 시점 화면만(view가 "3d", "both"일 때)
    lanes: list[LaneCurve] | None = None  # 차선 곡선(평활 후). lanes 옵션이 꺼져 있으면 None
    lights: list[TrafficLight] | None = None  # 신호등 상태. traffic_lights 옵션이 꺼져 있으면 None


@dataclass
class _RenderJob:
    """렌더링에 필요한 것을 한 프레임분 모아 둔 것. 렌더링 스레드로 넘길 때 쓴다."""

    tracks: list[Track]
    ground: np.ndarray | None
    road: np.ndarray | None
    frames: list[np.ndarray]
    results: list[FrameResult]
    fps: float | None
    t: float = 0.0  # step()에 넘긴 dt를 더한 시각(초). 3D 트랙 보간에 쓴다
    refresh_ground: bool = True  # 3D 바닥을 이번에 다시 만들지(ground_every 프레임마다)
    lanes: list[LaneCurve] | None = None
    lights: list[TrafficLight] | None = None


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
        self._latest: tuple | None = None  # (대시보드, 탑뷰, 3D 화면)
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
                # 버리는 작업이 3D 바닥을 다시 만들 차례였으면 그 일을 새 작업이 이어받는다.
                # 안 그러면 바닥 갱신이 통째로 빠져 다음 갱신 차례까지 옛 바닥이 남는다.
                # (렌더링 스레드 모드에서는 바닥 영상을 매 프레임 합성하므로 새 작업에도 바닥 영상이 있다)
                if self._job.refresh_ground and not job.refresh_ground:
                    job.refresh_ground = True
            self._job = job
            self._cond.notify()

    def latest(self) -> tuple | None:
        with self._cond:
            return self._latest

    def set_latest(self, out: tuple) -> None:
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
        scale: float = 1.0,  # 장면 축척. 책상 위 모형이면 실차 대비 비율(설정 scene.scale), 1.0이면 실차
        view: str = "bev",  # 화면 종류: "bev"(기본), "3d", "both"
        view3d: View3DConfig | None = None,  # 3D 시점 설정(없으면 기본값)
        ground_every: int = 1,  # 3D 바닥(IPM 텍스처)을 몇 프레임마다 다시 만들지. 1이면 매 프레임
        tracker: str = "basic",  # 추적기 방식: "basic"(기본, 예전과 같음), "robust"(먼 물체 ID 쪼개짐 개선판)
        vehicle_center: str = "ray",  # 차 중심 추정: "ray"(기본, 예전과 같음, 시선 방향 길이 절반), "box"(보는 방향에 따라 길이나 폭 절반)
        skew_comp: bool = False,  # True면 step()에 준 카메라별 시각 차이만큼 물체 위치를 트랙 속도로 옮긴다(기본 꺼짐)
        lanes: LaneConfig | None = None,  # 주면 차선 검출을 켠다(기본 꺼짐)
        lanes_every: int = 1,  # 차선 검출을 몇 프레임마다 할지. 그 사이는 지난 결과를 쓴다
        traffic_lights: TrafficLightConfig | None = None,  # 주면 신호등 색 판별을 켠다(기본 꺼짐)
    ):
        self.cameras = cameras
        self.scale = float(scale)
        self.backend = backend
        self.grid = grid or BevGrid()
        self.ipm = IpmCompositor(cameras, self.grid)
        self.tracker = BevTracker(scale=self.scale, mode=tracker)
        self.skew_comp = bool(skew_comp)
        self.vehicle_center = vehicle_center
        self.show_ground = show_ground
        self.show_cameras = show_cameras
        self.timer = timer or StageTimer(enabled=False)
        if view not in ("bev", "3d", "both"):
            raise ValueError(f"view는 bev, 3d, both 중 하나여야 합니다: {view}")
        self.view = view
        self.ground_every = max(1, int(ground_every))
        # 3D 렌더러는 트랙 보간 상태와 바닥 캐시를 들고 있어서 렌더링을 하는 쪽(같은 스레드 또는 렌더링 스레드) 하나만 쓴다
        self.r3d = Renderer3D(self.grid, view3d, scale=self.scale, timer=self.timer) if view != "bev" else None
        self._time = 0.0  # dt 누적(초)
        self._fps = None
        self._last_step_t = None
        self._sync_renders = 0
        self._worker = _RenderWorker(self._render, self.timer) if render_thread else None
        # 두 NPU 동시 실행 옵션: 검출은 backend(예: Hailo), 세그는 seg_backend(예: DX-M1)
        self.seg_backend = seg_backend
        self.seg_every = max(1, int(seg_every))
        # 과열 대응(--thermal)이 경고 단계에서 True로 바꾸면 세그 스레드에 새 영상을 넘기지 않는다(지난 마스크 유지)
        self.seg_paused = False
        self._seg = _SegWorker(seg_backend.segment, self.timer) if seg_backend is not None else None
        self._frame_idx = 0
        # 차선과 신호등(옵션)
        self.lane_detector = LaneDetector(self.grid, lanes, self.ipm.coverage(), self.scale) if lanes else None
        self.lane_smoother = LaneSmoother(lanes, self.scale) if lanes else None
        self.lanes_every = max(1, int(lanes_every))
        self._lanes: list[LaneCurve] | None = [] if lanes else None
        self.light_reader = TrafficLightReader(cameras, traffic_lights, self.scale) if traffic_lights else None
        # 차선과 신호등 통계(측정 기록용): 차선 검출 횟수, 찾은 곡선 수 합, 내 차선 양쪽을 찾은 횟수, 신호등 색별 수
        self._lane_runs = 0
        self._lane_sum = 0
        self._lane_ego = 0
        self._light_counts: dict[str, int] = {}
        self._seg_ages: list[int] = []  # 프레임마다 쓴 마스크가 몇 프레임 묵었는지(마스크가 있을 때만)
        self._seg_missing = 0  # 아직 첫 마스크가 없어 마스크 없이 지나간 프레임 수
        self._seg_last_idx = None  # 마지막으로 도로 비율을 센 마스크의 프레임 번호
        self._seg_fracs: list[float] = []  # 새 마스크마다 도로 화소 비율(카메라 평균)
        # 바로 앞 step()에서 쓴 세그 마스크가 몇 번째 프레임 영상으로 만든 것인지(없으면 None).
        # 지연 측정에서 그 프레임의 센서 시각을 찾아 "마스크가 몇 ms 묵었는지"를 계산할 때 쓴다
        self.last_mask_idx: int | None = None
        # 탑뷰 도로 마스크 캐시: (카메라별 원본 마스크 객체 목록, 합성한 탑뷰 마스크)
        self._road_cache: tuple[list, np.ndarray] | None = None
        self.road_cache_hits = 0  # 마스크가 그대로라 합성을 건너뛴 프레임 수
        self.road_cache_misses = 0  # 새 마스크라 다시 합성한 프레임 수

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
        mask_fraction_mean: 받은 마스크의 도로 화소 비율 평균(0~1). 렌더링 비용 비교에 쓴다
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
            "mask_age_p95": float(np.percentile(ages, 95)) if ages else None,
            "frames_without_mask": self._seg_missing,
            "mask_fraction_mean": float(np.mean(self._seg_fracs)) if self._seg_fracs else None,
        }

    @property
    def lane_light_stats(self) -> dict | None:
        """차선과 신호등 통계. 둘 다 꺼져 있으면 None.

        lane_runs: 차선 검출을 한 횟수, lanes_mean: 한 번에 남은(평활 후) 곡선 수 평균,
        ego_both_frac: 내 차선 왼쪽과 오른쪽 경계를 모두 찾은 비율, light_states: 신호등 색별 누적 수(다수결 후)
        """
        if self.lane_detector is None and self.light_reader is None:
            return None
        return {
            "lane_runs": self._lane_runs,
            "lanes_mean": round(self._lane_sum / self._lane_runs, 3) if self._lane_runs else None,
            "ego_both_frac": round(self._lane_ego / self._lane_runs, 3) if self._lane_runs else None,
            "light_states": dict(self._light_counts),
        }

    # ---- 한 프레임 처리 -------------------------------------------------
    def step(self, frames: list[np.ndarray], dt: float, cam_offsets_s: list[float] | None = None) -> StepOutput:
        """한 프레임을 처리한다.

        cam_offsets_s: 카메라별 "이 영상의 시각 - 기준 시각"(초). FrameSyncer 묶음의 offsets_s.
          skew_comp=True일 때만 쓰고, 없거나 꺼져 있으면 예전과 똑같이 처리한다.
        """
        tm = self.timer
        t0 = time.perf_counter()
        idx = self._frame_idx
        self._frame_idx += 1

        # 0) 별도 스레드 세그: N 프레임마다 영상을 넘긴다. 아래 검출과 같은 시간에 다른 NPU에서 돈다
        if self._seg is not None and not self.seg_paused and idx % self.seg_every == 0:
            self._seg.submit(idx, frames)

        # 1) 검출(+세그). 전처리, NPU 추론, NMS 파싱의 세부 시간은 백엔드가 같은 타이머에 적는다
        with tm.stage("detect_total"):
            results = self.backend.infer(frames)
        t1 = time.perf_counter()

        # 1-1) 별도 스레드 세그의 가장 최근 마스크를 붙인다. 이번 프레임 것이 아직이면 이전 마스크를 쓴다
        if self._seg is not None:
            self._attach_async_masks(idx, results)

        # 1-2) 신호등 색(옵션): 상자를 잘라 HSV로 판별하고 프레임 간 다수결
        lights = None
        if self.light_reader is not None:
            with tm.stage("traffic_lights"):
                lights = self.light_reader.update(frames, [r.detections for r in results])
            for lt in lights:
                self._light_counts[lt.state] = self._light_counts.get(lt.state, 0) + 1

        # 2) 검출 박스의 접지점을 지면(차량 좌표)으로 투영. 신호등은 공중에 있어 바닥 투영이 틀리므로 뺀다
        with tm.stage("ipm_project"):
            objs = []
            for i, (cam, r) in enumerate(zip(self.cameras, results)):
                dets = [d for d in r.detections if d.label not in AERIAL_LABELS]
                objs += project_detections(i, cam, dets, max_range=40.0 * self.scale, scale=self.scale,
                                           vehicle_center=self.vehicle_center)

        # 2-1) 시각 맞춤 보정(옵션): 기준보다 앞뒤로 찍힌 카메라의 물체를 기준 시각 위치로 옮긴다
        if self.skew_comp and cam_offsets_s is not None:
            from .multicam import compensate_offsets

            with tm.stage("skew_comp"):
                objs = compensate_offsets(objs, self.tracker.tracks, cam_offsets_s, float(dt), self.tracker.gate)

        # 3) 카메라 경계에서 겹친 같은 물체 합치기
        with tm.stage("merge"):
            objs = merge_cross_camera(objs, scale=self.scale)

        # 4) 추적(ID 유지, 속도 추정)
        with tm.stage("track"):
            tracks = self.tracker.update(objs, dt)

        # 5) 바닥 영상과 도로 마스크를 탑뷰로 합성(IPM remap)
        #    3D만 그릴 때 ground_every > 1이면 3D 바닥을 다시 만드는 프레임에만 합성한다(그 사이는 렌더러가 지난 바닥을 쓴다)
        refresh = idx % self.ground_every == 0
        #    렌더링 스레드 모드에서는 버린 작업의 갱신을 다음 작업이 이어받을 수 있게 늘 합성한다
        need_ground = self.show_ground and (self.view != "3d" or refresh or self._worker is not None)
        #    차선 검출 차례이면 화면에 바닥을 안 깔아도(--no-ground) 바닥 영상을 합성한다
        lane_turn = self.lane_detector is not None and idx % self.lanes_every == 0
        with tm.stage("bev_warp"):
            ground = self.ipm.warp(frames) if (need_ground or lane_turn) else None
            road = self._warp_road(results, frames)
        # 5-1) 차선(옵션): 차례인 프레임에만 찾고 평활한다. 그 사이 프레임은 지난 결과를 그대로 쓴다
        if lane_turn:
            with tm.stage("lanes"):
                occ = None
                if self.lane_detector.cfg.mask_objects:
                    # 검출 상자 안 화소(차체)가 IPM에서 바닥으로 길게 늘어져 차선처럼 보이는 것을 뺀다.
                    # 상자가 없는 프레임은 합성을 건너뛴다(추가 시간 0)
                    masks = box_masks([f.shape[:2] for f in frames], [r.detections for r in results], AERIAL_LABELS)
                    occ = self.ipm.warp(masks, interpolation=0) if masks is not None else None
                self._lanes = self.lane_smoother.update(self.lane_detector.detect(ground, road, occ))
            self._lane_runs += 1
            self._lane_sum += len(self._lanes)
            self._lane_ego += int({"left", "right"} <= {ln.side for ln in self._lanes})
        if not need_ground:
            ground = None  # 차선 때문에 합성한 바닥은 화면에 깔지 않는다(예전 화면 그대로)
        self._time += float(dt)
        t2 = time.perf_counter()

        # 화면에 찍는 FPS는 step() 호출 간격(카메라 획득까지 포함한 실제 갱신 주기)으로 구한다
        if self._last_step_t is not None:
            period = max(t0 - self._last_step_t, 1e-6)
            self._fps = 1.0 / period if self._fps is None else 0.9 * self._fps + 0.1 / period
        self._last_step_t = t0

        # 6) 렌더링(탑뷰, 카메라 주석, 대시보드 합치기)
        job = _RenderJob(tracks, ground, road, frames, results, self._fps, t=self._time, refresh_ground=refresh,
                         lanes=self._lanes, lights=lights)
        if self._worker is None:
            with tm.stage("render"):
                dash, bev, v3d = self._render(job)
            self._sync_renders += 1
        else:
            job.tracks = _snapshot_tracks(tracks)
            job.lanes = snapshot_lanes(self._lanes)  # 다음 프레임 평활이 같은 곡선을 고치므로 사본을 넘긴다
            latest = self._worker.latest()
            if latest is None:
                # 첫 프레임은 보여 줄 결과가 없으므로 직접 그린다(호출자가 None을 받지 않게)
                with tm.stage("render"):
                    latest = self._render(job)
                self._worker.set_latest(latest)
            else:
                self._worker.submit(job)
            dash, bev, v3d = latest
        t3 = time.perf_counter()

        timings = {"npu_ms": (t1 - t0) * 1e3, "bev_ms": (t2 - t1) * 1e3, "render_ms": (t3 - t2) * 1e3}
        return StepOutput(dash, bev, tracks, results, timings, v3d, lanes=self._lanes, lights=lights)

    def _warp_road(self, results: list[FrameResult], frames: list[np.ndarray]) -> np.ndarray | None:
        """카메라별 도로 마스크를 탑뷰 한 장으로 합성한다. 마스크가 지난 프레임과 같으면 지난 결과를 다시 쓴다.

        세그는 N 프레임마다만 돌고 그 사이에는 같은 마스크 배열이 그대로 붙어 온다(세그 스레드, HailoBackend 모두).
        그래서 카메라마다 "같은 배열 객체인가(is)"로 비교해 모두 같으면 remap을 건너뛴다.
        - id() 숫자만 비교하면 옛 배열이 지워진 자리에 새 배열이 같은 번호로 생길 수 있어서,
          옛 마스크 객체를 직접 들고 있다가 is로 비교한다.
        - 마스크를 그 자리에서 고치는 백엔드라면 이 캐시가 틀린 그림을 줄 수 있다. 지금 백엔드는
          모두 마스크를 새 배열로 만들고 고치지 않는다(sim은 매 프레임 새 배열이라 캐시가 늘 빗나간다).
        - 돌려주는 탑뷰 마스크는 여러 프레임이 같이 쓰므로 받는 쪽은 읽기만 한다(render_bev는 고치지 않는다).
        """
        keys = [r.road_mask for r in results]
        if all(k is None for k in keys):
            self._road_cache = None
            return None
        cache = self._road_cache
        if cache is not None and len(cache[0]) == len(keys) and all(a is b for a, b in zip(cache[0], keys)):
            self.road_cache_hits += 1
            return cache[1]
        # 마스크가 없는 카메라는 빈 마스크로 채워 합성한다
        masks = [k if k is not None else np.zeros(f.shape[:2], np.uint8) for k, f in zip(keys, frames)]
        road = self.ipm.warp(masks, interpolation=0)
        self._road_cache = (keys, road)
        self.road_cache_misses += 1
        return road

    def _attach_async_masks(self, idx: int, results: list[FrameResult]) -> None:
        latest = self._seg.latest()  # 세그 스레드가 예외로 멈췄으면 여기서 다시 낸다
        if latest is None or len(latest[1]) != len(results):
            self._seg_missing += 1
            self.last_mask_idx = None
            return
        mask_idx, masks = latest
        self.last_mask_idx = mask_idx
        # 세그 NPU를 따로 고른 것이므로 검출 백엔드가 만든 마스크가 있어도 이 마스크로 바꾼다
        for r, m in zip(results, masks):
            r.road_mask = m
        self._seg_ages.append(idx - mask_idx)
        # 새 마스크가 들어왔을 때만 도로 비율(1인 화소 비율)을 센다. 렌더링 비용이 이 비율에 따라 달라서 기록해 둔다
        if mask_idx != self._seg_last_idx:
            self._seg_last_idx = mask_idx
            self._seg_fracs.append(float(np.mean([np.count_nonzero(m) / max(m.size, 1) for m in masks])))

    def _render(self, job: _RenderJob) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """화면을 그린다. (대시보드, 탑뷰 또는 None, 3D 화면 또는 None). 같은 스레드와 렌더링 스레드가 함께 쓴다.

        - bev: 왼쪽 탑뷰 + 오른쪽 카메라 영상(예전과 똑같다)
        - 3d: 3D 화면을 크게, 오른쪽에 카메라 영상을 작게
        - both: 왼쪽 탑뷰(작게) + 가운데 3D 화면 + 오른쪽 카메라 영상
        """
        bev = None
        if self.view in ("bev", "both"):
            bev = render_bev(self.grid, job.tracks, job.ground, job.road, self.ipm.coverage(), scale=self.scale,
                             lanes=job.lanes, lights=job.lights)
        v3d = None
        if self.r3d is not None:
            # 렌더링 스레드에서 불려도 timer.stage는 잠금으로 더하므로 안전하다(그 프레임 몫으로 들어간다)
            with self.timer.stage("render3d"):
                v3d = self.r3d.render(
                    job.tracks, job.ground, job.road, self.ipm.coverage(), refresh_ground=job.refresh_ground, t=job.t,
                    lanes=job.lanes, lights=job.lights,
                )
        views = []
        if self.show_cameras:
            views = [
                annotate_camera(f, r.detections, c.name, r.road_mask)
                for f, r, c in zip(job.frames, job.results, self.cameras)
            ]
        if v3d is None:
            dash = compose_dashboard(bev, views, job.fps)
        else:
            dash = compose_dashboard_3d(v3d, views, bev=bev, fps=job.fps)
        return dash, bev, v3d

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
