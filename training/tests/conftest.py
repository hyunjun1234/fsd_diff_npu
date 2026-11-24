from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.common import load_config  # noqa: E402
from training.tests import synth  # noqa: E402


# ---------------------------------------------------------------------------
# 공용 준비물: 임시 폴더 안에 합성 주행, 트리거 클립, COCO 흉내를 만들고 그곳을 가리키는 설정
# ---------------------------------------------------------------------------
def make_cfg(tmp: str, round_name: str = "r001", drives=("d1",), **over) -> dict:
    o = {
        "round": round_name,
        "paths": {
            "drives_root": os.path.join(tmp, "drives"),
            "triggers_root": os.path.join(tmp, "triggers"),
            "work_root": os.path.join(tmp, "work"),
            "models_root": os.path.join(tmp, "models"),
            "coco_root": os.path.join(tmp, "coco"),
        },
        "drives": list(drives),
        "select": {"every_sec": 0.5, "n_frames": 20, "trigger_max_ratio": 0.25},
        "test_set": {"n_frames": 6, "segment_ratio": 0.34},
        "autolabel": {"primary": "fake", "secondary": "fake_secondary"},
        "review": {"random_ratio": 0.5},
        "train_det": {"prev_ratio": 0.5, "coco_ratio": 0.25},
        "export": {"calib_n": 8, "dx_calib_n": 4},
    }
    for k, v in over.items():
        if isinstance(v, dict):
            o.setdefault(k, {}).update(v)
        else:
            o[k] = v
    return load_config(None, o)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("train_world"))
    synth.make_drive(os.path.join(tmp, "drives"), "d1", seed=1)
    synth.make_trigger(os.path.join(tmp, "triggers"), "clip01")
    synth.make_coco(os.path.join(tmp, "coco"))
    return tmp
