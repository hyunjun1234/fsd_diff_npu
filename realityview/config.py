from __future__ import annotations

import copy

import yaml

from .bev import BevGrid
from .camera_model import Camera, camera_from_dict

# ---------------------------------------------------------------------------
# 실차 단위 값을 모형 축척으로 줄이기
# ---------------------------------------------------------------------------
# 모형 설정은 테슬라 배치를 실차 m 그대로 적고 scene.scale 하나로 줄여 쓴다.
#   cameras[].extrinsics.position_car : 실차 m 위치. position = position_car x scale
#   bev_car                            : 실차 m BEV 범위와 셀 크기. bev = bev_car x scale
# 각도(yaw, pitch, roll)와 내부 파라미터(화각)는 축척과 상관없으므로 건드리지 않는다.
#
# 외부 보정(scripts/calib_extrinsics.py)이 내놓는 position은 이미 모형 m로 잰 값이다.
# 그래서 position이 있으면 그 값을 그대로 쓰고 position_car는 무시한다(두 번 줄이지 않음).
# BEV는 bev와 bev_car를 함께 적으면 어느 쪽인지 모호하므로 오류로 멈춘다.
BEV_KEYS = ("x_min", "x_max", "y_min", "y_max", "res")


def scene_scale(cfg: dict) -> float:
    """설정의 장면 축척. scene.scale이 없으면 실차(1.0)."""
    s = float((cfg.get("scene") or {}).get("scale", 1.0))
    if not s > 0:
        raise ValueError(f"scene.scale은 0보다 커야 합니다: {s}")
    return s


def _scaled(v, s: float) -> float:
    """실차 값 x 축척. 0.1 곱셈의 끝자리 오차(1.2000000000000002 같은 값)는 반올림해 없앤다."""
    return round(float(v) * s, 9)


def apply_scene_scale(cfg: dict) -> dict:
    """실차 단위 키를 모형 m로 바꾼 새 설정을 돌려준다(원본 dict는 그대로 둔다)."""
    cfg = copy.deepcopy(cfg)
    s = scene_scale(cfg)

    # ---- BEV 범위: bev_car가 있으면 축척을 곱해 bev로 만든다 ----
    if "bev_car" in cfg:
        if "bev" in cfg:
            raise ValueError("bev와 bev_car를 함께 적을 수 없습니다. 하나만 남기세요")
        car = cfg.pop("bev_car") or {}
        unknown = set(car) - set(BEV_KEYS)
        if unknown:
            raise ValueError(f"bev_car에 모르는 키가 있습니다: {sorted(unknown)}")
        cfg["bev"] = {k: _scaled(v, s) for k, v in car.items()}

    # ---- 카메라 위치: position이 없을 때만 position_car x scale ----
    for cam in cfg.get("cameras", []):
        ext = cam.get("extrinsics") or {}
        car_pos = ext.pop("position_car", None)
        if "position" not in ext:
            if car_pos is None:
                raise ValueError(f"카메라 {cam.get('name')}: extrinsics에 position이나 position_car가 필요합니다")
            if len(car_pos) != 3:
                raise ValueError(f"카메라 {cam.get('name')}: position_car는 x, y, z 세 값이어야 합니다")
            ext["position"] = [_scaled(v, s) for v in car_pos]
        cam["extrinsics"] = ext
    return cfg


# ---------------------------------------------------------------------------
# 설정 파일 읽기
# ---------------------------------------------------------------------------
def load_config(path: str) -> tuple[list[Camera], BevGrid, dict]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 실차 단위 키가 없는 설정(two_cam.yaml 등)은 내용이 바뀌지 않는다
    cfg = apply_scene_scale(cfg)
    cams = [camera_from_dict(c) for c in cfg["cameras"]]
    grid = BevGrid(**cfg.get("bev", {}))
    return cams, grid, cfg
