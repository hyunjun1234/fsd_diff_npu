from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# =====================================================================
# 1. 설정: BEV 격자와 깊이 구간
# =====================================================================
# 좌표계는 저장소의 다른 코드(camera_model.py)와 같습니다.
#   차량 좌표: X 전방, Y 왼쪽, Z 위 (단위 m)
#   카메라 좌표: x 오른쪽, y 아래, z 광축 방향
# 출력 BEV 특징은 (C, X, Y) 순서입니다. X 방향 셀 번호 ix, Y 방향 셀 번호 iy.


@dataclass(frozen=True)
class BEVGrid:
    """BEV 격자 범위와 셀 크기.

    기본값은 FlashOcc 범위(x, y는 -40~40 m, z는 -1~5.4 m)를 0.8 m 셀로 나눈 100x100 격자입니다.
    z 범위 밖의 점(땅속이나 너무 높은 점)도 버립니다.
    """

    x_min: float = -40.0
    x_max: float = 40.0
    y_min: float = -40.0
    y_max: float = 40.0
    z_min: float = -1.0
    z_max: float = 5.4
    cell: float = 0.8

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.cell))


@dataclass(frozen=True)
class DepthBins:
    """깊이 구간. 기본값은 BEVDet/FlashOcc와 같은 1 m 부터 45 m 직전까지 0.5 m 간격(88개)입니다."""

    d_min: float = 1.0
    d_max: float = 45.0
    step: float = 0.5

    @property
    def count(self) -> int:
        return int(round((self.d_max - self.d_min) / self.step))

    def values(self) -> np.ndarray:
        # 구간 시작값을 그 구간의 깊이로 씁니다(BEVDet의 arange(d_min, d_max, step)과 같음).
        return self.d_min + self.step * np.arange(self.count, dtype=np.float64)


# =====================================================================
# 2. 기하: frustum 점을 차량 좌표로 옮기기
# =====================================================================


def frustum_points_ego(
    K: np.ndarray,
    R_cam2ego: np.ndarray,
    t_cam2ego: np.ndarray,
    depths: np.ndarray,
    feat_hw: Tuple[int, int],
    img_hw: Tuple[int, int],
) -> np.ndarray:
    """카메라마다 (깊이, 특징 행, 특징 열) 격자점의 차량 좌표를 구합니다.

    K          : (N, 3, 3) 입력 이미지(img_hw) 해상도 기준 내부 파라미터
    R_cam2ego  : (N, 3, 3) 카메라 좌표를 차량 좌표로 돌리는 회전
    t_cam2ego  : (N, 3)    차량 좌표에서 카메라 위치
    depths     : (D,)      깊이 값 (m)
    feat_hw    : 특징맵 크기 (H, W), 예: 256x704 입력의 1/16이면 (16, 44)
    img_hw     : 입력 이미지 크기 (H_img, W_img)

    반환: (N, D, H, W, 3) float64 차량 좌표
    """
    K = np.asarray(K, dtype=np.float64).reshape(-1, 3, 3)
    R = np.asarray(R_cam2ego, dtype=np.float64).reshape(-1, 3, 3)
    t = np.asarray(t_cam2ego, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(depths, dtype=np.float64).ravel()
    H, W = feat_hw
    H_img, W_img = img_hw

    # 특징맵의 각 칸이 입력 이미지에서 어느 픽셀에 해당하는지(BEVDet처럼 양 끝을 맞춘 등간격).
    u = np.linspace(0.0, W_img - 1.0, W)
    v = np.linspace(0.0, H_img - 1.0, H)
    dd, vv, uu = np.meshgrid(d, v, u, indexing="ij")  # 각각 (D, H, W)

    # 픽셀 (u, v)와 깊이 d를 동차 좌표 (u*d, v*d, d)로 만든 뒤 K^-1을 곱하면 카메라 좌표입니다.
    pix = np.stack([uu * dd, vv * dd, dd], axis=-1)  # (D, H, W, 3)
    K_inv = np.linalg.inv(K)  # (N, 3, 3)
    # 카메라 좌표 -> 차량 좌표: p_ego = R (K^-1 pix) + t. 두 행렬을 먼저 곱해 한 번에 계산합니다.
    M = R @ K_inv  # (N, 3, 3)
    pts = np.einsum("nij,dhwj->ndhwi", M, pix) + t[:, None, None, None, :]
    return pts


def cell_index(pts: np.ndarray, grid: BEVGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """차량 좌표 점을 BEV 셀 번호로 바꿉니다.

    반환: (ix, iy, keep). keep은 x, y, z가 모두 격자 안인 점만 True.
    음수 쪽 경계에서 점이 0번 셀로 잘못 들어가지 않도록 정수 변환 전에 floor를 합니다.
    """
    ix = np.floor((pts[..., 0] - grid.x_min) / grid.cell).astype(np.int64)
    iy = np.floor((pts[..., 1] - grid.y_min) / grid.cell).astype(np.int64)
    z = pts[..., 2]
    keep = (
        (ix >= 0)
        & (ix < grid.nx)
        & (iy >= 0)
        & (iy < grid.ny)
        & (z >= grid.z_min)
        & (z < grid.z_max)
    )
    return ix, iy, keep


def cameras_to_arrays(cameras: Sequence, img_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """realityview.camera_model.Camera 목록을 (K, R_cam2ego, t_cam2ego) 배열로 바꿉니다.

    카메라 원본 해상도에서 img_hw로 단순 크기 변경(자르기 없음)했다고 보고 K를 비율만큼 줄입니다.
    어안(fisheye) 왜곡은 다루지 않습니다. 왜곡을 편 이미지를 넣는다고 가정합니다.
    """
    H_img, W_img = img_hw
    Ks, Rs, ts = [], [], []
    for cam in cameras:
        sx = W_img / float(cam.width)
        sy = H_img / float(cam.height)
        K = np.array(cam.K, dtype=np.float64).copy()
        K[0, :] *= sx
        K[1, :] *= sy
        Ks.append(K)
        Rs.append(cam.R_vc)
        ts.append(cam.position)
    return np.stack(Ks), np.stack(Rs), np.stack(ts)


# =====================================================================
# 3. 미리 계산(precompute): 카메라 설정이 고정이면 한 번만
# =====================================================================


@dataclass
class BEVPoolPlan:
    """매 프레임 누적에 필요한 인덱스 묶음. 모두 C 연속 int32입니다.

    depth_idx : 격자 안 점마다 depth (N, D, H, W)를 1차원으로 편 배열에서의 위치
    feat_idx  : 같은 점이 쓰는 특징 픽셀 번호 (n * H * W + h * W + w)
    cell      : 같은 점의 BEV 평면 번호 (ix * Y + iy). 이 값으로 정렬되어 있습니다.
    iv_start, iv_len, iv_cell : 점이 하나 이상 있는 셀마다 시작 위치, 점 개수, 셀 번호
    """

    n_cams: int
    D: int
    H: int
    W: int
    X: int
    Y: int
    depth_idx: np.ndarray
    feat_idx: np.ndarray
    cell: np.ndarray
    iv_start: np.ndarray
    iv_len: np.ndarray
    iv_cell: np.ndarray
    n_total: int  # 버리기 전 전체 frustum 점 수 (N * D * H * W)

    @property
    def n_kept(self) -> int:
        return int(self.depth_idx.size)

    @property
    def n_cells(self) -> int:
        return int(self.iv_cell.size)


def precompute_plan(
    K: np.ndarray,
    R_cam2ego: np.ndarray,
    t_cam2ego: np.ndarray,
    grid: BEVGrid = BEVGrid(),
    bins: DepthBins = DepthBins(),
    feat_hw: Tuple[int, int] = (16, 44),
    img_hw: Tuple[int, int] = (256, 704),
) -> BEVPoolPlan:
    """frustum 점마다 BEV 셀을 계산하고, 격자 밖 점을 버린 뒤 셀 순서로 정렬해 둡니다."""
    depths = bins.values()
    pts = frustum_points_ego(K, R_cam2ego, t_cam2ego, depths, feat_hw, img_hw)
    N, D, H, W, _ = pts.shape
    ix, iy, keep = cell_index(pts, grid)

    # 점마다 (n, d, h, w) 번호를 만들어 depth/feat 위치를 계산합니다.
    n_i, d_i, h_i, w_i = np.indices((N, D, H, W))
    depth_flat = ((n_i * D + d_i) * H + h_i) * W + w_i
    feat_flat = (n_i * H + h_i) * W + w_i
    cell_flat = ix * grid.ny + iy

    depth_idx = depth_flat[keep]
    feat_idx = feat_flat[keep]
    cell = cell_flat[keep]

    # 같은 셀의 점이 붙어 있도록 정렬(안정 정렬이라 결과가 매번 같음).
    order = np.argsort(cell, kind="stable")
    depth_idx, feat_idx, cell = depth_idx[order], feat_idx[order], cell[order]

    # 셀마다 구간(시작, 길이)을 만듭니다. 점이 없는 셀은 구간에 아예 들어가지 않습니다.
    iv_cell, iv_start, iv_len = np.unique(cell, return_index=True, return_counts=True)

    def i32(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.int32)

    if N * D * H * W >= 2**31:
        raise ValueError("frustum too large for int32 indices")

    return BEVPoolPlan(
        n_cams=N,
        D=D,
        H=H,
        W=W,
        X=grid.nx,
        Y=grid.ny,
        depth_idx=i32(depth_idx),
        feat_idx=i32(feat_idx),
        cell=i32(cell),
        iv_start=i32(iv_start),
        iv_len=i32(iv_len),
        iv_cell=i32(iv_cell),
        n_total=N * D * H * W,
    )


def _check_inputs(depth: np.ndarray, feat: np.ndarray, plan: BEVPoolPlan) -> int:
    """입력 모양이 계획과 맞는지 확인하고 채널 수를 돌려줍니다."""
    if depth.shape != (plan.n_cams, plan.D, plan.H, plan.W):
        raise ValueError(f"depth shape {depth.shape} != plan {(plan.n_cams, plan.D, plan.H, plan.W)}")
    if feat.ndim != 4 or feat.shape[0] != plan.n_cams or feat.shape[2:] != (plan.H, plan.W):
        raise ValueError(f"feat shape {feat.shape} does not match plan")
    return int(feat.shape[1])


# =====================================================================
# 4. 기준 구현(정답용): 정렬이나 구간 없이 점마다 그대로 더하기
# =====================================================================


def bev_pool_reference(
    depth: np.ndarray,
    feat: np.ndarray,
    K: np.ndarray,
    R_cam2ego: np.ndarray,
    t_cam2ego: np.ndarray,
    grid: BEVGrid = BEVGrid(),
    bins: DepthBins = DepthBins(),
    img_hw: Tuple[int, int] = (256, 704),
) -> np.ndarray:
    """정답 확인용. 계획(plan)을 쓰지 않고 기하부터 다시 계산해 float64로 더합니다.

    빠른 구현과 정렬, 구간 코드를 공유하지 않으므로 그쪽 버그를 잡을 수 있습니다. 느립니다.
    """
    depth = np.asarray(depth, dtype=np.float64)
    feat = np.asarray(feat, dtype=np.float64)
    N, D, H, W = depth.shape
    C = feat.shape[1]
    pts = frustum_points_ego(K, R_cam2ego, t_cam2ego, bins.values(), (H, W), img_hw)
    ix, iy, keep = cell_index(pts, grid)

    out = np.zeros((C, grid.nx, grid.ny), dtype=np.float64)
    n_i, d_i, h_i, w_i = np.nonzero(keep)
    w = depth[n_i, d_i, h_i, w_i]  # (P,)
    f = feat[n_i, :, h_i, w_i]  # (P, C)
    # np.add.at은 같은 셀 번호가 여러 번 나와도 모두 더해 줍니다.
    np.add.at(out, (slice(None), ix[keep], iy[keep]), (f * w[:, None]).T)
    return out.astype(np.float32)


# =====================================================================
# 5. 빠른 구현 (NumPy 두 가지)
# =====================================================================


def bev_pool_reduceat(depth: np.ndarray, feat: np.ndarray, plan: BEVPoolPlan) -> np.ndarray:
    """정렬된 점을 (점, 채널) 행렬로 만든 뒤 셀 구간별로 np.add.reduceat 합니다."""
    C = _check_inputs(depth, feat, plan)
    w = depth.reshape(-1)[plan.depth_idx]  # 점마다 깊이 확률 (P,)
    # 특징을 (픽셀, 채널)로 바꿔 한 픽셀의 채널이 붙어 있게 합니다.
    ft = np.ascontiguousarray(feat.transpose(0, 2, 3, 1)).reshape(-1, C)
    contrib = ft[plan.feat_idx]  # (P, C) 복사본
    contrib *= w[:, None]
    out = np.zeros((C, plan.X * plan.Y), dtype=np.float32)
    if plan.n_cells:
        # iv_start는 비어 있지 않은 구간의 시작만 담고 있어 reduceat의 빈 구간 문제가 없습니다.
        sums = np.add.reduceat(contrib, plan.iv_start, axis=0)  # (U, C)
        out[:, plan.iv_cell] = sums.T
    return out.reshape(C, plan.X, plan.Y)


def bev_pool_bincount(depth: np.ndarray, feat: np.ndarray, plan: BEVPoolPlan) -> np.ndarray:
    """채널마다 np.bincount(셀 번호, 가중치=깊이 x 특징)로 더합니다."""
    C = _check_inputs(depth, feat, plan)
    w = depth.reshape(-1)[plan.depth_idx]  # (P,)
    fc = np.ascontiguousarray(feat.transpose(1, 0, 2, 3)).reshape(C, -1)  # (C, 픽셀)
    XY = plan.X * plan.Y
    out = np.empty((C, XY), dtype=np.float32)
    for c in range(C):
        # bincount는 같은 셀 번호의 가중치를 float64로 더해 줍니다.
        out[c] = np.bincount(plan.cell, weights=fc[c][plan.feat_idx] * w, minlength=XY)
    return out.reshape(C, plan.X, plan.Y)


# =====================================================================
# 6. 빠른 구현 (C, ctypes)
# =====================================================================
# 소스는 realityview/_bevpool.c, 빌드 결과는 저장소의 build/ 아래에 둡니다(.gitignore 대상).
# 맥에서 만든 파일이 rsync로 파이에 넘어가도 섞이지 않도록 파일 이름에 OS와 CPU 종류를 넣습니다.
# 컴파일러가 없거나 빌드, 로드가 실패하면 None을 돌려주고 NumPy 경로를 씁니다.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_C_SOURCE = Path(__file__).resolve().parent / "_bevpool.c"
_DEFAULT_BUILD_DIR = _REPO_ROOT / "build"

_i32p = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


class CBackend:
    """빌드된 C 라이브러리를 감싼 객체."""

    def __init__(self, lib: ctypes.CDLL, path: Path, openmp: bool):
        self.lib = lib
        self.path = path
        self.openmp = openmp

    def threads_used(self, nthreads: int) -> int:
        return int(self.lib.bevpool_threads_used(int(nthreads)))

    def pool(self, depth: np.ndarray, feat: np.ndarray, plan: BEVPoolPlan, nthreads: int = 1) -> np.ndarray:
        C = _check_inputs(depth, feat, plan)
        d = np.ascontiguousarray(depth, dtype=np.float32).reshape(-1)
        ft = np.ascontiguousarray(feat.transpose(0, 2, 3, 1), dtype=np.float32).reshape(-1)
        XY = plan.X * plan.Y
        out = np.zeros((C, XY), dtype=np.float32)  # 점이 없는 셀은 0으로 남습니다.
        self.lib.bevpool_accumulate(
            d, ft, C,
            plan.depth_idx, plan.feat_idx,
            plan.iv_start, plan.iv_len, plan.iv_cell,
            plan.n_cells, XY, out, int(nthreads),
        )
        return out.reshape(C, plan.X, plan.Y)


def _lib_name() -> str:
    machine = platform.machine().lower() or "unknown"
    return f"_bevpool-{sys.platform}-{machine}.so"


def _compile(cc: str, out_path: Path, openmp: bool) -> bool:
    """C 소스를 공유 라이브러리로 빌드합니다. 성공하면 True."""
    tmp = out_path.with_suffix(f".tmp{os.getpid()}.so")
    cmd = [cc, "-O3", "-std=c99", "-shared", "-fPIC"]
    if openmp:
        cmd.append("-fopenmp")
    cmd += [str(_C_SOURCE), "-o", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)  # 실패한 산출물은 남기지 않습니다.
        return False
    os.replace(tmp, out_path)  # 다 만든 뒤 이름을 바꿔서 반쯤 쓴 파일을 읽지 않게 합니다.
    return True


def _load(path: Path) -> Optional[CBackend]:
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        return None
    lib.bevpool_has_openmp.restype = ctypes.c_int
    lib.bevpool_has_openmp.argtypes = []
    lib.bevpool_threads_used.restype = ctypes.c_int
    lib.bevpool_threads_used.argtypes = [ctypes.c_int]
    lib.bevpool_accumulate.restype = None
    lib.bevpool_accumulate.argtypes = [
        _f32p, _f32p, ctypes.c_int,
        _i32p, _i32p,
        _i32p, _i32p, _i32p,
        ctypes.c_int, ctypes.c_int, _f32p, ctypes.c_int,
    ]
    return CBackend(lib, path, bool(lib.bevpool_has_openmp()))


def load_c_backend(
    cc: Optional[str] = None,
    build_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Optional[CBackend]:
    """C 라이브러리를 (필요하면 빌드해서) 불러옵니다. 실패하면 None.

    cc를 주지 않으면 환경변수 CC, 없으면 cc를 씁니다.
    -fopenmp로 먼저 빌드해 보고, 안 되면(맥의 기본 clang 등) OpenMP 없이 다시 빌드합니다.
    """
    if not _C_SOURCE.exists():
        return None
    cc = cc or os.environ.get("CC") or "cc"
    build_dir = Path(build_dir) if build_dir is not None else _DEFAULT_BUILD_DIR
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    lib_path = build_dir / _lib_name()

    stale = (not lib_path.exists()) or lib_path.stat().st_mtime < _C_SOURCE.stat().st_mtime
    if force_rebuild or stale:
        if not (_compile(cc, lib_path, openmp=True) or _compile(cc, lib_path, openmp=False)):
            return None
    backend = _load(lib_path)
    if backend is None and not force_rebuild:
        # 이전에 만든 파일이 깨졌거나 다른 장비용이면 한 번만 다시 빌드해 봅니다.
        if _compile(cc, lib_path, openmp=True) or _compile(cc, lib_path, openmp=False):
            backend = _load(lib_path)
    return backend


_C_CACHE: dict = {}


def get_c_backend() -> Optional[CBackend]:
    """프로세스 안에서 한 번만 빌드, 로드합니다. 환경변수 REALITYVIEW_BEVPOOL_NO_C=1이면 끕니다."""
    if os.environ.get("REALITYVIEW_BEVPOOL_NO_C") == "1":
        return None
    if "backend" not in _C_CACHE:
        _C_CACHE["backend"] = load_c_backend()
    return _C_CACHE["backend"]


# =====================================================================
# 7. 공통 진입점
# =====================================================================

METHODS = ("c", "bincount", "reduceat")


def resolve_method(method: str = "auto") -> str:
    """"auto"면 C가 되면 C, 아니면 NumPy bincount를 고릅니다."""
    if method == "auto":
        return "c" if get_c_backend() is not None else "bincount"
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, choose from {METHODS} or 'auto'")
    return method


def bev_pool(
    depth: np.ndarray,
    feat: np.ndarray,
    plan: BEVPoolPlan,
    method: str = "auto",
    nthreads: int = 1,
) -> np.ndarray:
    """매 프레임 부르는 함수. depth (N, D, H, W), feat (N, C, H, W)를 (C, X, Y) BEV 특징으로 모읍니다."""
    m = resolve_method(method)
    if m == "c":
        backend = get_c_backend()
        if backend is None:
            raise RuntimeError("C backend not available (compiler missing or build failed)")
        return backend.pool(depth, feat, plan, nthreads=nthreads)
    if m == "reduceat":
        return bev_pool_reduceat(depth, feat, plan)
    return bev_pool_bincount(depth, feat, plan)
