from __future__ import annotations

import cv2
import numpy as np

from .camera_model import Camera

# 카메라 입력. CSI 카메라는 Picamera2, USB 카메라와 동영상 파일은 OpenCV로 읽는다.


# ---------------------------------------------------------------------------
# Picamera2 설정 만들기 (하드웨어 없이 시험할 수 있게 순수 함수로 분리)
# ---------------------------------------------------------------------------
def picamera_config_kwargs(
    width: int,
    height: int,
    fps: float,
    sensor_size: tuple[int, int] | None = None,
) -> dict:
    """Picamera2.create_video_configuration()에 넘길 인자를 만든다.

    - main: 파이프라인이 받는 영상. RGB888은 메모리에 B, G, R 순서로 놓여 OpenCV의 BGR과 같다.
    - sensor: 센서 모드를 직접 고른다. 고르지 않으면 Picamera2가 출력 크기에 가까운 모드를 고르는데,
      imx708의 1536x864나 ov5647의 1920x1080처럼 가운데만 잘라 쓰는 모드가 뽑히면 화각이 좁아져
      설정 파일의 fx와 맞지 않게 된다. 그래서 설정에 sensor_size를 적어 화각이 정해진 모드를 쓴다.
    """
    kw = {
        "main": {"size": (int(width), int(height)), "format": "RGB888"},
        "controls": {"FrameRate": float(fps)},
    }
    if sensor_size:
        kw["sensor"] = {"output_size": (int(sensor_size[0]), int(sensor_size[1]))}
    return kw


def rotation_flips(rotation: int) -> tuple[bool, bool]:
    """설정의 rotation(도)을 (좌우 뒤집기, 상하 뒤집기)로 바꾼다.

    180도 회전은 좌우와 상하를 함께 뒤집은 것과 같다. libcamera는 센서가 보고한 장착 회전
    (Picamera2의 'Rotation' 값)을 이미 보정해서 영상을 내보내므로, 여기의 rotation은
    "그 보정 뒤에도 영상이 뒤집혀 보일 때" 더 돌릴 양이다. 실제 영상을 보고 정한다.
    """
    rotation = int(rotation) % 360
    if rotation == 0:
        return False, False
    if rotation == 180:
        return True, True
    raise ValueError(f"rotation은 0 또는 180만 지원합니다: {rotation}")


# ---------------------------------------------------------------------------
# CSI 카메라 (Picamera2)
# ---------------------------------------------------------------------------
class PiCameraSource:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: float = 15.0,
        sensor_size: tuple[int, int] | None = None,
        rotation: int = 0,
    ):
        from picamera2 import Picamera2  # type: ignore

        self.size = (int(width), int(height))
        self._closed = False
        # 지연 측정용(기본 꺼짐). True로 켜면 read()가 영상과 같은 요청에서 센서 타임스탬프를 함께 꺼낸다.
        # capture_metadata()를 따로 부르면 다른 프레임의 값이 나오므로 capture_request() 하나에서 둘 다 받는다.
        self.want_timestamp = False
        # 마지막 read()가 돌려준 프레임의 SensorTimestamp(ns, time.monotonic_ns()와 같은 시계). 모르면 None
        self.last_timestamp_ns: int | None = None
        self.cam = Picamera2(index)
        try:
            kw = picamera_config_kwargs(width, height, fps, sensor_size)
            hflip, vflip = rotation_flips(rotation)
            if hflip or vflip:
                from libcamera import Transform  # type: ignore

                kw["transform"] = Transform(hflip=int(hflip), vflip=int(vflip))
            cfg = self.cam.create_video_configuration(**kw)
            self.cam.configure(cfg)
            # 실제로 잡힌 센서 모드를 남겨 둔다(보정값과 맞는 모드인지 확인용)
            applied = getattr(self.cam, "camera_config", None) or cfg
            self.sensor_config = dict(applied.get("sensor", {}) or {}) if isinstance(applied, dict) else {}
            self.cam.start()
        except Exception:
            # 설정 도중 실패해도 카메라를 놓아 주어야 다음 실행에서 "busy"가 나지 않는다
            self.cam.close()
            self._closed = True
            raise

    def read(self) -> np.ndarray | None:
        if self.want_timestamp:
            f = self._read_with_timestamp()
        else:
            f = self.cam.capture_array("main")
        if f is None:
            return None
        # 일부 형식은 4채널(XRGB)로 오므로 3채널로 맞춘다
        if f.ndim == 3 and f.shape[2] == 4:
            f = f[:, :, :3]
        if (f.shape[1], f.shape[0]) != self.size:
            f = cv2.resize(f, self.size)
        return f

    def _read_with_timestamp(self) -> np.ndarray | None:
        """영상과 메타데이터를 같은 요청에서 받는다. 요청은 꼭 release()해야 카메라 버퍼가 돌아간다."""
        req = self.cam.capture_request()
        try:
            f = req.make_array("main")
            ts = req.get_metadata().get("SensorTimestamp")
        finally:
            req.release()
        self.last_timestamp_ns = int(ts) if ts is not None else None
        return f

    def metadata(self) -> dict:
        """현재 프레임의 메타데이터(ScalerCrop, 노출 등). 확인용."""
        return dict(self.cam.capture_metadata())

    def close(self) -> None:
        # stop()만 부르면 카메라 장치가 프로세스가 끝날 때까지 잡혀 있다. close()까지 불러 놓아 준다.
        if self._closed:
            return
        self._closed = True
        try:
            self.cam.stop()
        finally:
            self.cam.close()


# ---------------------------------------------------------------------------
# USB 카메라, 동영상 파일 (OpenCV)
# ---------------------------------------------------------------------------
class OpenCvSource:
    def __init__(self, uri, width: int, height: int, loop: bool = True):
        self.cap = cv2.VideoCapture(uri)
        self.size = (width, height)
        self.loop = loop and isinstance(uri, str)
        # 파일과 USB 카메라에는 센서 타임스탬프가 없다. 지연 측정에서 "모름"으로 처리한다
        self.want_timestamp = False
        self.last_timestamp_ns: int | None = None
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


# ---------------------------------------------------------------------------
# 설정에서 입력 장치 열기
# ---------------------------------------------------------------------------
def open_source(cam: Camera, override: dict | None = None):
    src = dict(cam.source)
    if override:
        src.update(override)
    kind = src.get("type", "opencv")
    if kind == "picamera2":
        return PiCameraSource(
            int(src.get("index", 0)),
            cam.width,
            cam.height,
            float(src.get("fps", 15)),
            sensor_size=src.get("sensor_size"),
            rotation=int(src.get("rotation", 0)),
        )
    uri = src.get("uri", 0)
    return OpenCvSource(uri, cam.width, cam.height)
