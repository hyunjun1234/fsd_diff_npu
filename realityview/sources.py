"""카메라 입력. CSI 카메라는 Picamera2, USB 카메라/동영상 파일은 OpenCV로 읽는다."""
from __future__ import annotations

import cv2
import numpy as np

from .camera_model import Camera


class PiCameraSource:
    def __init__(self, index: int, width: int, height: int, fps: float = 15.0):
        from picamera2 import Picamera2  # type: ignore

        self.cam = Picamera2(index)
        cfg = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},  # Picamera2의 RGB888은 메모리상 BGR 순서
            controls={"FrameRate": fps},
        )
        self.cam.configure(cfg)
        self.cam.start()

    def read(self) -> np.ndarray | None:
        return self.cam.capture_array("main")

    def close(self) -> None:
        self.cam.stop()


class OpenCvSource:
    def __init__(self, uri, width: int, height: int, loop: bool = True):
        self.cap = cv2.VideoCapture(uri)
        self.size = (width, height)
        self.loop = loop and isinstance(uri, str)
        if isinstance(uri, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if not ok:
            return None
        if (f.shape[1], f.shape[0]) != self.size:
            f = cv2.resize(f, self.size)
        return f

    def close(self) -> None:
        self.cap.release()


def open_source(cam: Camera, override: dict | None = None):
    src = dict(cam.source)
    if override:
        src.update(override)
    kind = src.get("type", "opencv")
    if kind == "picamera2":
        return PiCameraSource(int(src.get("index", 0)), cam.width, cam.height, float(src.get("fps", 15)))
    uri = src.get("uri", 0)
    return OpenCvSource(uri, cam.width, cam.height)
