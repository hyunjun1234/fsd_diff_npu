from __future__ import annotations

import csv
import os

import cv2
import numpy as np

# 시험용 작은 합성 주행 영상을 만든다. 파이 녹화 형식(세그먼트 폴더에 카메라별 영상과 타임스탬프 csv)을 흉내 낸다.
#   배경: 초록 잡음 무늬(자차가 움직이면 옆으로 흐름), 아래 절반: 회색 도로
#   차: 파란 사각형, 사람: 빨간 사각형(가짜 라벨러가 색으로 찾음)
#   밤: 전체 밝기를 0.3배, 흐림: 큰 가우시안 흐림, 정차: 배경과 물체를 멈춤(거의 같은 프레임)

W, H = 320, 240


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = np.zeros((H, W * 4, 3), np.int16)
    tex[:] = (40, 100, 40)
    tex += rng.integers(-20, 21, size=tex.shape, dtype=np.int16)
    road_noise = rng.integers(-10, 11, size=(H // 2, W * 4, 1), dtype=np.int16)
    tex[H // 2:] = 110 + road_noise  # 회색은 세 채널이 같은 잡음
    for x in range(0, W * 4, 40):  # 차선 무늬
        tex[H - 30:H - 26, x:x + 20] = 230
    return tex


def draw_frame(tex: np.ndarray, k: int, offset: int, night: bool, blur: bool, objects: bool = True) -> np.ndarray:
    img = tex[:, offset:offset + W].copy()
    if objects:
        x = 40 + (k * 3) % 160
        img[140:180, x:x + 50] = (200, 30, 30)  # 차(파랑)
        img[120:170, 250:266] = (30, 30, 200)  # 사람(빨강)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if night:
        img = (img.astype(np.float32) * 0.3).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 0)
    return img


def write_segment(seg_dir: str, camera: str, n: int = 40, fps: float = 10.0, night: bool = False,
                  blur_frames=(), static_from: int | None = None, seed: int = 0, video: bool = True,
                  objects: bool = True, fmt: str = "mjpeg") -> None:
    """카메라 하나 분량의 세그먼트를 쓴다.

    fmt="mjpeg": 파이 녹화 형식(<camera>.mjpeg 에 JPEG 이어 붙임 + <camera>.csv 의
                 frame, index, sensor_ns, wall_time, offset, size, gap 열. realityview/mjpeg.py 와 같은 열)
    fmt="avi":   <camera>.avi(MJPG) + <camera>.csv(frame, t_sensor_ns)
    video=False: record_frames.py 형식(<camera>_00000.png 와 timestamps.csv)
    """
    os.makedirs(seg_dir, exist_ok=True)
    tex = _texture(seed)
    frames = []
    for k in range(n):
        kk = k if static_from is None or k < static_from else static_from
        frames.append(draw_frame(tex, kk, (kk * 4) % (W * 3), night, k in set(blur_frames), objects))
    # 파이의 sensor_ns 는 부팅 뒤 monotonic 시계라 차와 함께 켜지면 값이 작다(여기서는 부팅 45초 뒤)
    t0 = 45_000_000_000 + seed * 10**9
    if video and fmt == "mjpeg":
        off = 0
        with open(os.path.join(seg_dir, camera + ".mjpeg"), "wb") as fm, \
                open(os.path.join(seg_dir, camera + ".csv"), "w", newline="") as fc:
            w = csv.writer(fc)
            w.writerow(["frame", "index", "sensor_ns", "wall_time", "offset", "size", "gap"])
            for k, f in enumerate(frames):
                data = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                fm.write(data)
                w.writerow([k, 1000 + k, t0 + int(k * 1e9 / fps), f"{1.7e9 + k / fps:.6f}", off, len(data), 0])
                off += len(data)
    elif video:
        vw = cv2.VideoWriter(os.path.join(seg_dir, camera + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        assert vw.isOpened(), "MJPG 영상 쓰기를 열지 못했습니다"
        for f in frames:
            vw.write(f)
        vw.release()
        with open(os.path.join(seg_dir, camera + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_sensor_ns"])
            for k in range(n):
                w.writerow([k, t0 + int(k * 1e9 / fps)])
    else:
        rows = []
        for k, f in enumerate(frames):
            name = f"{camera}_{k:05d}.png"
            cv2.imwrite(os.path.join(seg_dir, name), f)
            rows.append([k, camera, k / fps, "", name])
        p = os.path.join(seg_dir, "timestamps.csv")
        new = not os.path.exists(p)
        with open(p, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["frame", "camera", "t_mono", "t_wall", "file"])
            w.writerows(rows)


def make_drive(root: str, drive: str, seed: int = 0) -> None:
    """세그먼트 3개짜리 주행: 낮, 밤, 낮(흐린 프레임과 정차 구간 포함). 카메라 front, rear.

    앞의 두 세그먼트는 파이 녹화 형식(mjpeg), 마지막은 avi 로 써서 두 읽기 길을 모두 시험한다.
    """
    for cam_i, cam in enumerate(("front", "rear")):
        write_segment(os.path.join(root, drive, "seg_0000"), cam, seed=seed * 10 + cam_i)
        write_segment(os.path.join(root, drive, "seg_0001"), cam, night=True, seed=seed * 10 + cam_i + 2)
        write_segment(os.path.join(root, drive, "seg_0002"), cam, blur_frames=(5, 10, 15), static_from=25,
                      seed=seed * 10 + cam_i + 4, fmt="avi")


def make_trigger(root: str, clip: str, kind: str = "ambiguous", score: float = 0.42, seed: int = 9) -> None:
    """파이 트리거 클립 형식: <실행 이름>/<클립>/trigger.json(events 목록) + front.mjpeg, front.csv"""
    import json

    d = os.path.join(root, "run1", clip)
    write_segment(d, "front", n=20, seed=seed)
    with open(os.path.join(d, "trigger.json"), "w") as f:
        json.dump({"clip": clip, "trigger_frame": 1010, "first_frame": 1000, "last_frame": 1019, "frames": 20,
                   "events": [{"kind": kind, "score": score, "frame": 1010, "detail": {}},
                              {"kind": "track_new", "score": 1.0, "frame": 1012, "detail": {}}]}, f)


def make_coco(root: str, n: int = 12) -> None:
    """COCO YOLO 형식 흉내: images/train2017/*.jpg, labels/train2017/*.txt"""
    os.makedirs(os.path.join(root, "images", "train2017"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels", "train2017"), exist_ok=True)
    for i in range(n):
        cv2.imwrite(os.path.join(root, "images", "train2017", f"{i:012d}.jpg"), np.full((64, 64, 3), 128, np.uint8))
        with open(os.path.join(root, "labels", "train2017", f"{i:012d}.txt"), "w") as f:
            f.write("2 0.5 0.5 0.2 0.2\n")
