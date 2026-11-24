#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

# DX-M1(dx_engine) 실제 API 확인용 스크립트. 파이의 .venv에서 실행한다.
#
# 예)
#   # 모델 없이: dx_engine import, 장치 수, 온도, 런타임 버전만 확인
#   timeout 30 python3 scripts/probe_dxm1.py --out out/measure/02_dx_probe_nomodel.json
#
#   # 모델이 있으면: 입출력 텐서 정보, run, run_async + wait, 콜백, 반복 시간까지 확인
#   timeout 60 python3 scripts/probe_dxm1.py --model models/yolov8s.dxnn --loops 100 \
#       --out out/measure/02_dx_probe.json
#
# 결과는 화면에 찍고, --out을 주면 같은 내용을 JSON으로 남긴다.


# ---------------------------------------------------------------------------
# 명령줄 옵션
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help=".dxnn 경로. 없으면 장치 정보만 본다")
    ap.add_argument("--loops", type=int, default=50, help="시간 측정 반복 수(모델이 있을 때)")
    ap.add_argument("--out", help="결과 JSON 경로. 예: out/measure/02_dx_probe.json")
    return ap


# ---------------------------------------------------------------------------
# 장치 정보 (모델 없이 가능한 범위)
# ---------------------------------------------------------------------------
def probe_device() -> dict:
    import dx_engine
    from dx_engine import Configuration, DeviceStatus

    info: dict = {"dx_engine_version": getattr(dx_engine, "__version__", None)}
    cfg = Configuration()
    info["runtime_version"] = cfg.get_version()
    info["driver_version"] = cfg.get_driver_version()
    info["pcie_driver_version"] = cfg.get_pcie_driver_version()
    n = DeviceStatus.get_device_count()
    info["device_count"] = n
    devs = []
    for i in range(n):
        s = DeviceStatus.get_current_status(i)
        # DX-M1 안에는 NPU 코어가 3개 있어 온도, 클럭을 코어마다 따로 읽는다
        devs.append(
            {
                "id": s.get_id(),
                "temperature_c": [s.get_temperature(c) for c in range(3)],
                "npu_clock_mhz": [s.get_npu_clock(c) for c in range(3)],
                "memory_free_bytes": s.get_memory_free(),
            }
        )
    info["devices"] = devs
    return info


# ---------------------------------------------------------------------------
# 모델 확인 (입출력 형식, 동기/비동기 실행, 시간)
# ---------------------------------------------------------------------------
def _tensor_info(lst: list[dict]) -> list[dict]:
    """텐서 정보 사전의 dtype(numpy 타입)을 JSON에 쓸 수 있게 문자열로 바꾼다."""
    return [{k: (np.dtype(v).name if k == "dtype" and v is not None else v) for k, v in d.items()} for d in lst]


def _describe(outs) -> list[dict]:
    return [{"shape": list(np.asarray(o).shape), "dtype": np.asarray(o).dtype.name} for o in outs]


def probe_model(path: str, loops: int) -> dict:
    from dx_engine import InferenceEngine

    rep: dict = {"model": path}
    ie = InferenceEngine(path)
    try:
        ins = ie.get_input_tensors_info()
        outs = ie.get_output_tensors_info()
        rep["inputs"] = _tensor_info(ins)
        rep["outputs"] = _tensor_info(outs)
        rep["input_size_bytes"] = ie.get_input_size()
        rep["output_tensor_sizes"] = ie.get_output_tensor_sizes()
        rep["compile_type"] = ie.get_compile_type()
        rep["model_version"] = ie.get_model_version()
        rep["is_ppu"] = ie.is_ppu()

        # ---- 입력: 보고된 shape과 dtype 그대로 만든 무작위 영상 ----
        shape = [int(v) for v in ins[0]["shape"]]
        dtype = ins[0]["dtype"] or np.uint8
        x = np.random.randint(0, 255, size=shape).astype(dtype)

        # ---- 동기 실행 ----
        y = ie.run(x)
        rep["run_output"] = _describe(y)

        # ---- 비동기 실행 + wait ----
        jid = ie.run_async(x, user_arg="probe")
        y2 = ie.wait(jid)
        rep["run_async_wait_output"] = _describe(y2)

        # ---- 콜백 방식: 콜백 인자 (출력 목록, user_arg) 확인 ----
        got = {}
        ev = threading.Event()

        def cb(outputs, user_arg):
            got["n"] = len(outputs)
            got["user_arg"] = repr(user_arg)
            got["shapes"] = [list(np.asarray(o).shape) for o in outputs]
            ev.set()
            return 0

        ie.register_callback(cb)
        jid = ie.run_async(x, user_arg=("cb", 1))
        ev.wait(10)
        rep["callback"] = got or "10초 안에 콜백이 오지 않음"
        # wait()는 콜백을 등록한 뒤에도 같은 job을 정리할 수 있는지 확인
        try:
            ie.wait(jid)
            rep["wait_after_callback"] = "ok"
        except Exception as e:  # noqa: BLE001 - 확인용이므로 무엇이든 기록
            rep["wait_after_callback"] = f"{type(e).__name__}: {e}"
        ie.register_callback(None)

        # ---- 반복 시간: 동기 1개씩, 비동기 4개씩 겹쳐 보내기 ----
        t0 = time.perf_counter()
        for _ in range(loops):
            ie.run(x)
        sync_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        done = 0
        while done < loops:
            ids = [ie.run_async(x) for _ in range(min(4, loops - done))]
            for j in ids:
                ie.wait(j)
            done += len(ids)
        async_s = time.perf_counter() - t0
        rep["sync_fps"] = round(loops / sync_s, 2)
        rep["async4_fps"] = round(loops / async_s, 2)
        rep["latency_mean_us"] = ie.get_latency_mean()
        rep["npu_time_mean_us"] = ie.get_npu_inference_time_mean()
    finally:
        ie.dispose()
    return rep


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    rep = {"device": probe_device()}
    if args.model:
        if not os.path.isfile(args.model):
            print(f"모델 파일이 없습니다: {args.model}", file=sys.stderr)
            return 2
        rep["model"] = probe_model(args.model, args.loops)
    text = json.dumps(rep, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
