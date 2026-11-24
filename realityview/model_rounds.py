from __future__ import annotations

import os

# 회차 모델 고르기. 추가 학습을 한 번 돌릴 때마다(한 회차) 검출 모델을 Hailo용 HEF와 DX-M1용 .dxnn으로 만들어
# 파이의 models/ 아래에 쌓고, 설정 파일의 models 항목에서 어떤 회차를 쓸지 고른다.
#
# 파일 이름 규칙: models/<det_name>_r<회차 세 자리>.<hef|dxnn>
#   예) models/yolov8s_r001.hef, models/yolov8s_r001.dxnn
# 규칙보다 먼저 있던 파일은 models.rounds 에 회차 번호로 직접 등록한다(0회차 = 공개 사전학습 모델).
#
# 설정 예 (configs/two_cam_current.yaml)
#   models:
#     det_name: yolov8s
#     det_model_round: 0        # 화면과 추적에 쓰는 회차
#     shadow_model_round: 0     # --shadow 때 다른 NPU에서 같이 돌려 비교할 회차
#     rounds:
#       0: {hef: models/yolov8s_h8.hef, dxnn: models/dxnn/yolov8-s_640x640_ppu.dxnn}
#
# models 항목이 없는 설정은 예전처럼 npu.det_hef, npu.dx_det_model 을 그대로 쓴다.

KINDS = ("hef", "dxnn")


def round_file_name(det_name: str, rnd: int, kind: str) -> str:
    """회차 모델 파일 이름. round_file_name('yolov8s', 1, 'hef') == 'yolov8s_r001.hef'."""
    if kind not in KINDS:
        raise ValueError(f"kind는 {KINDS} 중 하나여야 합니다: {kind}")
    if int(rnd) < 0:
        raise ValueError(f"회차는 0 이상이어야 합니다: {rnd}")
    return f"{det_name}_r{int(rnd):03d}.{kind}"


def resolve_round_model(cfg: dict, rnd: int, kind: str) -> str:
    """회차 번호와 종류(hef, dxnn)로 모델 파일 경로를 정한다.

    models.rounds 에 그 회차가 등록되어 있으면 그 경로, 없으면 이름 규칙(models.dir/<det_name>_rNNN.<kind>).
    """
    models = cfg.get("models") or {}
    rounds = models.get("rounds") or {}
    entry = rounds.get(int(rnd)) if int(rnd) in rounds else rounds.get(str(int(rnd)))
    if entry and entry.get(kind):
        return str(entry[kind])
    return os.path.join(str(models.get("dir", "models")), round_file_name(str(models.get("det_name", "yolov8s")), rnd, kind))


def backend_kind(backend: str) -> str:
    """run_demo.py 백엔드 이름을 모델 파일 종류로."""
    if backend == "hailo":
        return "hef"
    if backend == "dxm1":
        return "dxnn"
    raise ValueError(f"회차 모델은 hailo, dxm1 백엔드에만 있습니다: {backend}")


def primary_det_model(cfg: dict, backend: str) -> tuple[str | None, int | None]:
    """기본(화면, 추적) 검출 모델 경로와 회차. models.det_model_round 가 없으면 (None, None) = 예전 설정 그대로."""
    models = cfg.get("models") or {}
    rnd = models.get("det_model_round")
    if rnd is None:
        return None, None
    return resolve_round_model(cfg, int(rnd), backend_kind(backend)), int(rnd)


def shadow_det_model(cfg: dict, shadow_backend: str) -> tuple[str | None, int | None]:
    """섀도 실행 모델 경로와 회차. shadow_model_round 가 없으면 det_model_round, 그것도 없으면 0회차.

    models 항목이 없는 설정은 npu.det_hef 나 npu.dx_det_model 을 그대로 쓰고 회차는 None이다.
    """
    models = cfg.get("models") or {}
    if not models:
        npu = cfg.get("npu") or {}
        return npu.get("det_hef" if shadow_backend == "hailo" else "dx_det_model"), None
    rnd = models.get("shadow_model_round", models.get("det_model_round", 0))
    rnd = 0 if rnd is None else int(rnd)
    return resolve_round_model(cfg, rnd, backend_kind(shadow_backend)), rnd
