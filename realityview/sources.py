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
    buffer_count: int | None = None,
    queue: bool | None = None,
) -> dict:
    """Picamera2.create_video_configuration()에 넘길 인자를 만든다.

    - buffer_count, queue: 지연 시험용 옵션. None이면 넣지 않아 Picamera2 기본값(동영상 설정은 버퍼 6개,
      queue=True)을 그대로 쓴다. queue=True면 Picamera2가 "가장 최근에 끝난 프레임 1장"을 들고 있다가
      바로 돌려주고(최대 한 프레임 묵음), False면 늘 다음 프레임이 끝날 때까지 기다린다(설치된 0.3.39 소스 확인).

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
    if buffer_count is not None:
        kw["buffer_count"] = int(buffer_count)
    if queue is not None:
        kw["queue"] = bool(queue)
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
        buffer_count: int | None = None,
        queue: bool | None = None,
        flush: bool = False,
    ):
        from picamera2 import Picamera2  # type: ignore

        self.size = (int(width), int(height))
        # 지연 시험용(기본 꺼짐). True면 read()가 "부른 뒤에 노출을 시작한 프레임"만 받는다
        # (Picamera2 capture_request(flush=True)). 묵은 프레임은 없지만 매번 다음 프레임을 기다린다
        self.flush = bool(flush)
        self._closed = False
        # 지연 측정용(기본 꺼짐). True로 켜면 read()가 영상과 같은 요청에서 센서 타임스탬프를 함께 꺼낸다.
        # capture_metadata()를 따로 부르면 다른 프레임의 값이 나오므로 capture_request() 하나에서 둘 다 받는다.
        self.want_timestamp = False
        # 마지막 read()가 돌려준 프레임의 SensorTimestamp(ns, time.monotonic_ns()와 같은 시계). 모르면 None
        self.last_timestamp_ns: int | None = None
        self.cam = Picamera2(index)
        try:
            kw = picamera_config_kwargs(width, height, fps, sensor_size, buffer_count, queue)
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
        if self.want_timestamp or self.flush:
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
        # flush를 켰을 때만 인자를 넘긴다(가짜 카메라와 예전 Picamera2에서도 기본 경로가 그대로 돌게)
        req = self.cam.capture_request(flush=True) if self.flush else self.cam.capture_request()
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
# 저장 이미지 재생 (scripts/record_frames.py 결과 폴더나 직접 모은 사진)
# ---------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def replay_files(folder: str, name: str) -> list[str]:
    """폴더에서 '{카메라 이름}_' 으로 시작하는 이미지 파일을 이름 순서로 찾는다.

    record_frames.py는 front_00000.png 처럼 저장하므로 그 폴더를 그대로 넣으면 된다.
    사진 한 장만 있어도 된다(같은 장을 되풀이해 준다).
    """
    import os

    pre = f"{name}_"
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.startswith(pre) and f.lower().endswith(IMAGE_EXTS)
    ]
    return files


class ReplaySource:
    """저장 이미지를 카메라처럼 한 장씩 돌려준다. 끝까지 가면 처음부터 다시(loop=True).

    카메라 앞에 사람이나 차가 없어도 그런 장면의 사진으로 검출부터 3D 화면까지 확인할 때 쓴다.
    센서 타임스탬프가 없으므로 지연 측정에서는 "모름"으로 처리된다.
    """

    def __init__(self, folder: str, name: str, width: int, height: int, loop: bool = True):
        self.files = replay_files(folder, name)
        if not self.files:
            raise FileNotFoundError(f"{folder} 에 '{name}_' 로 시작하는 이미지가 없습니다")
        self.size = (int(width), int(height))
        self.loop = loop
        self.pos = 0
        self.want_timestamp = False
        self.last_timestamp_ns: int | None = None
        self._cache: dict[int, np.ndarray] = {}  # 몇 장 안 되는 사진은 한 번만 읽는다

    def read(self) -> np.ndarray | None:
        if self.pos >= len(self.files):
            if not self.loop:
                return None
            self.pos = 0
        k = self.pos
        self.pos += 1
        f = self._cache.get(k)
        if f is None:
            f = cv2.imread(self.files[k], cv2.IMREAD_COLOR)
            if f is None:
                raise IOError(f"이미지를 읽지 못했습니다: {self.files[k]}")
            if (f.shape[1], f.shape[0]) != self.size:
                f = cv2.resize(f, self.size)
            if len(self.files) <= 64:
                self._cache[k] = f
        # 받는 쪽이 고쳐도 다음 장에 번지지 않게 복사본을 준다
        return f.copy()

    def close(self) -> None:
        self._cache.clear()


def open_replay_sources(folder: str, cams: list[Camera]) -> list:
    """--replay 폴더에 맞는 재생 소스를 카메라마다 만든다.

    - 주행 녹화 회차 폴더(seg_* 하위 폴더)나 세그먼트 폴더, 트리거 클립 폴더처럼 {이름}.mjpeg 와 {이름}.csv 가
      있으면 녹화 시각으로 카메라를 짝지어 읽는다(realityview/mjpeg.py).
    - 그렇지 않으면 예전처럼 '{이름}_*.png|jpg' 이미지를 차례로 읽는다.
    """
    import os

    from .mjpeg import has_mjpeg, open_mjpeg_sources, segment_dirs
    from .multicam import has_timed_videos, open_timed_videos

    if has_timed_videos(folder, len(cams)):
        # 합성 4대 장면(scripts/make_sim4.py): cam{번호}.mp4 + cam{번호}.csv. 카메라마다 시각과 프레임 수가 달라
        # run_demo.py가 --sync(FrameSyncer)로 묶어 읽는다
        return open_timed_videos(folder, cams)
    if any(all(has_mjpeg(d, c.name) for c in cams) for d in segment_dirs(folder) if os.path.isdir(d)):
        return open_mjpeg_sources(folder, cams)
    return [ReplaySource(folder, c.name, c.width, c.height) for c in cams]


# ---------------------------------------------------------------------------
# 설정에서 입력 장치 열기
# ---------------------------------------------------------------------------
def open_source(cam: Camera, override: dict | None = None, hubs: dict | None = None):
    """설정의 source 항목으로 입력 장치를 연다.

    type 값
      picamera2 : CSI 카메라(Picamera2)
      opencv    : 예전 방식 OpenCV 입력(장치 번호나 파일). 시각을 붙이지 않는다
      usb       : USB UVC 카메라(realityview/multicam.py UsbCameraSource). device는 번호나
                  /dev/v4l/by-id/... 경로. 받은 시각을 time.monotonic_ns()로 붙인다.
                  capture: [가로, 세로]를 적으면 그 크기로 받아 설정 크기로 줄인다
      video     : 동영상 파일을 가짜 USB 카메라로(uri, fps 속도로 내보내고 끝나면 처음부터)
      quad      : 쿼드 키트 칸 하나(group, tile, lag_frames, inner). 같은 group의 카메라는 hubs로 장치를 나눠 쓴다
    """
    src = dict(cam.source)
    if override:
        src.update(override)
    kind = src.get("type", "opencv")
    if kind in ("usb", "video"):
        from .multicam import UsbCameraSource

        if kind == "usb":
            return UsbCameraSource(
                src.get("device", 0),
                cam.width,
                cam.height,
                fps=src.get("fps"),
                timestamp_offset_ms=float(src.get("timestamp_offset_ms", 0.0)),
                fourcc=src.get("fourcc"),
                capture_size=src.get("capture"),
            )
        return UsbCameraSource(
            src["uri"], cam.width, cam.height, pace_fps=float(src.get("fps", 15)), loop=bool(src.get("loop", True)),
            timestamp_offset_ms=float(src.get("timestamp_offset_ms", 0.0)),
        )
    if kind == "quad":
        import dataclasses

        from .multicam import QuadSplitSource

        hubs = {} if hubs is None else hubs
        group = src.get("group", "quad0")
        if group not in hubs:
            # 같은 group의 첫 카메라가 키트 장치를 연다. 합쳐진 영상 크기 기본값은 칸 크기의 2배
            inner = dict(src.get("inner") or {"type": "picamera2", "index": 0})
            iw = int(inner.pop("width", 2 * cam.width))
            ih = int(inner.pop("height", 2 * cam.height))
            inner_cam = dataclasses.replace(cam, name=f"{group}_raw", width=iw, height=ih, source=inner)
            hubs[group] = QuadSplitSource(
                open_source(inner_cam), align=bool(src.get("align", True)), frame_period_ms=src.get("frame_period_ms")
            )
        return hubs[group].view(int(src.get("tile", 0)), int(src.get("lag_frames", 0)), (cam.width, cam.height))
    if kind == "picamera2":
        return PiCameraSource(
            int(src.get("index", 0)),
            cam.width,
            cam.height,
            float(src.get("fps", 15)),
            sensor_size=src.get("sensor_size"),
            rotation=int(src.get("rotation", 0)),
            buffer_count=src.get("buffer_count"),
            queue=src.get("queue"),
            flush=bool(src.get("flush", False)),
        )
    uri = src.get("uri", 0)
    return OpenCvSource(uri, cam.width, cam.height)
