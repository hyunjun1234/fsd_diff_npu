"""카메라 프레임 → 검출/세그(NPU) → 지면 투영/합성/추적(CPU) → 렌더링."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .bev import BevGrid, IpmCompositor, merge_cross_camera, project_detections
from .camera_model import Camera
from .detectors import Backend
from .render import annotate_camera, compose_dashboard, render_bev
from .tracker import BevTracker, Track
from .types import FrameResult


@dataclass
class StepOutput:
    dashboard: np.ndarray
    bev: np.ndarray
    tracks: list[Track]
    results: list[FrameResult]
    timings: dict


class RealityViewPipeline:
    def __init__(
        self,
        cameras: list[Camera],
        backend: Backend,
        grid: BevGrid | None = None,
        show_ground: bool = True,
        show_cameras: bool = True,
    ):
        self.cameras = cameras
        self.backend = backend
        self.grid = grid or BevGrid()
        self.ipm = IpmCompositor(cameras, self.grid)
        self.tracker = BevTracker()
        self.show_ground = show_ground
        self.show_cameras = show_cameras
        self._fps = None

    def step(self, frames: list[np.ndarray], dt: float) -> StepOutput:
        t0 = time.perf_counter()
        results = self.backend.infer(frames)
        t1 = time.perf_counter()

        objs = []
        for i, (cam, r) in enumerate(zip(self.cameras, results)):
            objs += project_detections(i, cam, r.detections)
        objs = merge_cross_camera(objs)
        tracks = self.tracker.update(objs, dt)

        ground = self.ipm.warp(frames) if self.show_ground else None
        road = None
        if any(r.road_mask is not None for r in results):
            masks = [r.road_mask if r.road_mask is not None else np.zeros(f.shape[:2], np.uint8)
                     for r, f in zip(results, frames)]
            road = self.ipm.warp(masks, interpolation=0)
        t2 = time.perf_counter()

        bev = render_bev(self.grid, tracks, ground, road, self.ipm.coverage())
        views = []
        if self.show_cameras:
            views = [annotate_camera(f, r.detections, c.name, r.road_mask) for f, r, c in zip(frames, results, self.cameras)]
        t3 = time.perf_counter()
        total = t3 - t0
        self._fps = 1.0 / total if self._fps is None else 0.9 * self._fps + 0.1 / max(total, 1e-6)
        dash = compose_dashboard(bev, views, self._fps)
        timings = {"npu_ms": (t1 - t0) * 1e3, "bev_ms": (t2 - t1) * 1e3, "render_ms": (t3 - t2) * 1e3}
        return StepOutput(dash, bev, tracks, results, timings)

    def close(self) -> None:
        self.backend.close()
