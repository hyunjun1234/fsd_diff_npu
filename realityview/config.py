from __future__ import annotations

import yaml

from .bev import BevGrid
from .camera_model import Camera, camera_from_dict


def load_config(path: str) -> tuple[list[Camera], BevGrid, dict]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cams = [camera_from_dict(c) for c in cfg["cameras"]]
    grid = BevGrid(**cfg.get("bev", {}))
    return cams, grid, cfg
