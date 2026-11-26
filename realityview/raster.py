"""삼각형 z-버퍼 그리기(C 구현을 ctypes로 부르고, 안 되면 OpenCV 화가 알고리즘으로 대신한다).

render_v15.py가 쓴다. C 소스(_raster.c)는 처음 쓸 때 build/ 아래에 빌드한다(bevpool.py와 같은 방식).
환경변수 REALITYVIEW_RASTER_NO_C=1이면 C를 쓰지 않는다(시험용).
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_C_SOURCE = _HERE / "_raster.c"
_BUILD_DIR = _HERE.parent / "build"
_CACHE: dict = {}

_f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")


def _lib_path() -> Path:
    return _BUILD_DIR / f"_raster-{sys.platform}-{platform.machine().lower() or 'unknown'}.so"


def _compile(out: Path) -> bool:
    cc = os.environ.get("CC") or "cc"
    tmp = out.with_suffix(f".tmp{os.getpid()}.so")
    try:
        r = subprocess.run([cc, "-O3", "-std=c99", "-shared", "-fPIC", str(_C_SOURCE), "-o", str(tmp), "-lm"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, out)
    return True


def _load():
    if os.environ.get("REALITYVIEW_RASTER_NO_C") == "1":
        return None
    try:
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    path = _lib_path()
    if not path.exists() or path.stat().st_mtime < _C_SOURCE.stat().st_mtime:
        if not _compile(path):
            return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        if not _compile(path):
            return None
        lib = ctypes.CDLL(str(path))
    lib.raster_tris.restype = ctypes.c_int
    lib.raster_tris.argtypes = [_f32, _f32, ctypes.c_int, _i32, ctypes.c_int, _u8, ctypes.c_void_p,
                                _u8, _f32, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.raster_tris_vc.restype = ctypes.c_int
    lib.raster_tris_vc.argtypes = [_f32, _f32, ctypes.c_int, _i32, ctypes.c_int, _u8,
                                   _u8, _f32, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    return lib


def c_available() -> bool:
    if "lib" not in _CACHE:
        _CACHE["lib"] = _load()
    return _CACHE["lib"] is not None


def raster(img: np.ndarray, zbuf: np.ndarray, uv: np.ndarray, iz: np.ndarray, tris: np.ndarray, col: np.ndarray,
           alpha: np.ndarray | None = None, mask: np.ndarray | None = None) -> None:
    """삼각형을 img와 zbuf에 그 자리에서 그린다.

    uv (N,2) 화소 좌표, iz (N,) 깊이의 역수(클수록 가까움), tris (M,3) int, col (M,3) BGR uint8,
    alpha (M,) uint8(없으면 모두 불투명). mask를 주면 칠한 화소에 255를 쓴다.
    zbuf는 float32 (H,W)이고 비어 있으면 0이다.
    """
    if len(tris) == 0:
        return
    if c_available():
        lib = _CACHE["lib"]
        H, W = img.shape[:2]
        a = None if alpha is None else np.ascontiguousarray(alpha, np.uint8)
        lib.raster_tris(np.ascontiguousarray(uv, np.float32), np.ascontiguousarray(iz, np.float32), len(uv),
                        np.ascontiguousarray(tris, np.int32), len(tris), np.ascontiguousarray(col, np.uint8),
                        None if a is None else a.ctypes.data, img, zbuf,
                        None if mask is None else mask.ctypes.data, W, H)
        return
    _raster_numpy(img, zbuf, uv, iz, tris, col, alpha, mask)


def raster_vc(img: np.ndarray, zbuf: np.ndarray, uv: np.ndarray, iz: np.ndarray, tris: np.ndarray,
              col_vc: np.ndarray, mask: np.ndarray | None = None) -> None:
    """raster()와 같지만 삼각형 꼭짓점마다 색(M,3,3)을 주고 사이를 보간한다(불투명만).

    C가 없으면 꼭짓점 색의 평균으로 raster()를 부른다(면마다 한 색).
    """
    if len(tris) == 0:
        return
    if c_available():
        H, W = img.shape[:2]
        _CACHE["lib"].raster_tris_vc(np.ascontiguousarray(uv, np.float32), np.ascontiguousarray(iz, np.float32),
                                     len(uv), np.ascontiguousarray(tris, np.int32), len(tris),
                                     np.ascontiguousarray(col_vc, np.uint8), img, zbuf,
                                     None if mask is None else mask.ctypes.data, W, H)
        return
    _raster_numpy(img, zbuf, uv, iz, tris, col_vc.astype(np.float32).mean(axis=1).astype(np.uint8), None, mask)


def _raster_numpy(img, zbuf, uv, iz, tris, col, alpha, mask) -> None:
    """C가 없을 때: 면을 먼 것부터 OpenCV로 칠한다(z-버퍼 대신 면 중심 깊이 정렬). 결과가 C와 조금 다를 수 있다."""
    depth = iz[tris].mean(axis=1)
    order = np.argsort(depth)  # 1/z 작은 것(먼 것)부터
    pts = np.round(uv).astype(np.int32)
    for t in order:
        p = pts[tris[t]]
        c = tuple(int(v) for v in col[t])
        a = 255 if alpha is None else int(alpha[t])
        if a >= 255:
            cv2.fillConvexPoly(img, p, c)
            cv2.fillConvexPoly(zbuf, p, float(depth[t]))
            if mask is not None:
                cv2.fillConvexPoly(mask, p, 255)
        else:
            x0, y0 = np.maximum(p.min(0), 0)
            x1, y1 = np.minimum(p.max(0) + 1, [img.shape[1], img.shape[0]])
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            lay = roi.copy()
            cv2.fillConvexPoly(lay, p - [x0, y0], c)
            cv2.addWeighted(lay, a / 255.0, roi, 1 - a / 255.0, 0, dst=roi)
