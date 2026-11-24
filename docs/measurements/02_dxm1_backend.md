# 2단계 준비: DX-M1 백엔드 추가 기록

이 문서는 구현 계획 4.3절(2단계)을 준비하면서 파이의 dx_engine API를 확인하고, DX-M1 백엔드와 두 NPU 동시 실행 옵션을 코드에 넣은 기록입니다. 작업일은 2026-11-24이고, 파이에 YOLOv8s .dxnn이 아직 없어서 실제 검출 정확도와 YOLOv8s FPS는 재지 못했습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| dx_engine API 확인 | 완료 | 파이 .venv의 dx-engine 3.4.2, 드론 검출 .dxnn으로 실행 |
| DXM1Backend | 추가 | 가짜 엔진 시험 통과, 진짜 엔진에서 원시 출력까지 확인 |
| 두 NPU 동시 실행 옵션 | 추가 | 기본은 꺼짐, `--seg-npu dxm1`로 켬 |
| 시험 | 맥 70 통과, 2 건너뜀 / 파이 72 통과 | 맥은 /usr/bin/python3, 파이는 .venv |
| 샘플 .dxnn FPS | 389.94 FPS | 드론 모델 tiny_p3p5_relu, 640 입력, run_model 10초 |
| YOLOv8s .dxnn FPS | 확인 필요 | 모델 파일 없음 |
| 세그 .dxnn 출력 모양 | 확인 필요 | 모델 파일 없음 |

## 1. 확인한 dx_engine API

파이의 `.venv/lib/python3.13/site-packages/dx_engine/`의 파이썬 소스를 읽고, 파이에 있던 드론 검출 모델(~/DaeDron/runs/npu/deepx/tiny_p3p5_relu.dxnn)로 직접 돌려 확인했습니다. 원본 wheel은 /usr/share/libdxrt-bin/python/에 파이썬 버전별(cp38부터 cp314까지)로 있습니다.

| 항목 | 확인 결과 | 근거 |
|---|---|---|
| 엔진 만들기 | `InferenceEngine(model_path, inference_option=None)`, `with` 문 지원, `dispose()`로 해제 | 소스 |
| 옵션 | `InferenceOption`: `devices`, `bound_option`(NPU 코어 0, 1, 2 묶음), `buffer_count`, `use_ort` | 소스 |
| 입력 정보 | `get_input_tensors_info()`가 name, shape, dtype, elem_size 사전 목록을 돌려줌 | 실행 |
| 입력 형식 | [1, 640, 640, 3] uint8 (NHWC) | 실행 |
| 입력 크기 | `get_input_size()` 1228800바이트, `get_input_tensor_sizes()` | 실행 |
| 동기 실행 | `run(x)`가 출력 배열 목록을 돌려줌 | 실행 |
| 비동기 실행 | `run_async(x, user_arg)`가 job 번호를 주고 `wait(job)`이 출력 목록을 줌 | 실행 |
| 콜백 | `register_callback(fn)`, `fn(outputs, user_arg)` 형태로 불림 | 실행 |
| 콜백 스레드 | 런타임 작업 스레드에서 GIL을 잡고 부름 | 바인딩 소스(py_inference_engine.cpp) |
| 출력 형식 | `get_output_tensors_info()` 순서의 float32 배열, NCHW (예: [1, 4, 80, 80]) | 실행 |
| 배치 | `run()`에 배열 목록을 주면 배치로 보고 출력 버퍼를 직접 달라고 함 | 소스 |
| 입력 검사 | 입력 dtype이 모델과 다르면 ValueError | 소스 |
| 시간 통계 | `get_latency_mean()`, `get_npu_inference_time_mean()` (마이크로초) | 실행 |
| 장치 정보 | `DeviceStatus.get_device_count()` 1, 코어 3개의 온도와 클럭, `Configuration().get_version()` 3.4.2 | 모델 없이 실행 |

이 모델의 ONNX 입력은 NCHW float32였는데 컴파일된 .dxnn은 NHWC uint8을 받습니다. 정규화가 모델 안으로 들어간 것으로 보입니다. 그래서 백엔드는 입력 모양과 dtype을 늘 장치에 물어서 정합니다. 다른 .dxnn도 같은지는 확인 필요입니다.

비동기 실행 확인 결과는 out/measure/02_dx_probe_sample.json, 모델 없이 확인한 장치 정보는 02_dx_probe_nomodel.json에 있습니다. 다시 확인하는 명령은 다음과 같습니다(파이 저장소 폴더, .venv 켠 뒤).

```bash
timeout 30 python3 scripts/probe_dxm1.py --out out/measure/02_dx_probe_nomodel.json
timeout 60 python3 scripts/probe_dxm1.py --model models/yolov8s.dxnn --loops 100 --out out/measure/02_dx_probe_yolov8s.json
```

결과는 화면과 --out 경로의 JSON에 남습니다.

### 1.1 예제와 도구

- /usr/share/libdxrt-bin에는 wheel, dxrt.service, SanityCheckForDebian.sh만 있고 예제 코드와 샘플 .dxnn은 없습니다.
- `dpkg -L libdxrt-bin dxrt-driver-dkms`의 파일 217개 중 이름에 example, sample, .dxnn이 들어간 것은 없습니다.
- /usr/local/bin에 run_model, parse_model, dxbenchmark, dxrt-cli, dxcli, dxtop이 있습니다.
- 파이에 있는 .dxnn은 ~/DaeDron, ~/DaeDron_gpu 아래의 드론 검출 모델뿐입니다. 출력이 스케일별 reg, obj, cls 9개로 나뉜 구조라 YOLOv8 출력과 다릅니다.

## 2. 샘플 모델 FPS

YOLOv8s가 아니라 드론 검출 모델로 잰 값입니다. 장치는 파이5와 DX-M1(PCIe Gen3 x1), 날짜는 2026-11-24, 입력은 무작위 값입니다.

| 모델과 방식 | 결과 | 명령 |
|---|---|---|
| tiny_p3p5_relu, 최대 처리량 | 389.94 FPS | `run_model -b -t 10 -w 20` |
| tiny_p3p5_relu, 한 장씩 | 95.5 FPS, NPU 5.03ms, 지연 10.1ms | `run_model -s -l 200 -w 20 -v` |
| nano_p3p5_relu, 최대 처리량 | 451.64 FPS | `run_model -b -t 10 -w 20` |
| tiny, 파이썬 run() 반복 | 97.77 FPS | probe_dxm1.py |
| tiny, 파이썬 run_async 4개씩 | 202.71 FPS | probe_dxm1.py |

원본 로그는 out/measure/02_dx_run_model.log, 02_dx_parse_model.log에 있습니다. 카메라마다 run_async를 보내고 wait로 받는 방식이 한 장씩보다 약 2배 빨라서 백엔드도 이 방식을 씁니다.

온도: DX-M1 코어 온도가 유휴 72도(00_env_check)에서 10초 최대 처리량 벤치 두 번과 200회 한 장씩 실행 뒤 88~89도까지 올랐습니다. 몇 분 쉬고 75도, 아래 동시 실행 시험 뒤 78도였습니다. 방열판 장착 여부는 확인 필요입니다. 그래서 이번에는 긴 벤치를 더 하지 않았습니다.

## 3. 코드 변경

### 3.1 DXM1Backend (realityview/detectors.py)

- HailoBackend와 같이 `infer(frames)`가 FrameResult 목록을 돌려줍니다.
- 처리 순서: BGR을 RGB로 바꾸고 레터박스(`preprocess`), 카메라마다 run_async 후 wait(`dxm1_infer`), CPU 디코딩과 클래스별 NMS(`nms_parse`). 괄호 안은 1단계 StageTimer의 단계 이름입니다.
- 검출 출력: 출력 텐서가 하나인 YOLO 원시 출력만 해석합니다. (1, 84, 8400)과 (1, 8400, 84) 둘 다 받고, 0~1 정규화 좌표와 시그모이드 전 값도 알아서 맞춥니다. yolov5 형식(물체 점수 포함)은 설정 `dx_det_format: yolov5`로 고릅니다.
- 출력이 여러 개인 모델(드론 모델 같은 스케일별 헤드)과 장치 후처리(PPU) 모델은 추측하지 않고 텐서 이름과 모양을 적은 오류를 냅니다. 이런 모델은 `infer_raw(frames, "det")`로 원시 출력을 받아 씁니다.
- 세그: `segment(frames)`가 카메라별 도로 마스크를 돌려줍니다. 출력은 NCHW 로짓으로 보고(설정 `dx_seg_channels_first: true`), 클래스 번호로 나오는 출력도 받습니다. 원시 출력은 `infer_raw(frames, "seg")`입니다.
- 모델 파일이 없으면 dx_engine을 부르기 전에 FileNotFoundError를 내고, 파이 models/에 두고 설정 `npu.dx_det_model` 또는 `npu.dx_seg_model`에 적으라고 안내합니다. 파이에서 `--backend dxm1`을 모델 없이 실행해 이 안내가 나오고 프로세스가 끝나는 것을 확인했습니다(out/measure/02_run_demo_dxm1_nomodel.log).
- dx_engine이 없으면(시스템 파이썬) .venv를 켜라는 ImportError를 냅니다.

### 3.2 두 NPU 동시 실행 (realityview/pipeline.py)

- `RealityViewPipeline(seg_backend=..., seg_every=3)`을 주면 세그 스레드가 생깁니다. 주지 않으면(기본) 예전과 똑같이 동작합니다.
- 메인 루프는 N 프레임마다 영상 복사본을 세그 스레드에 넘기고 기다리지 않습니다. 세그는 검출(Hailo)과 같은 시간에 DX-M1에서 돕니다.
- 각 프레임은 가장 최근에 끝난 마스크를 씁니다. 세그가 늦으면 이전 마스크를 다시 쓰고, 시작 전에 밀린 작업은 버립니다.
- 통계 `seg_stats`: 넘긴 수, 끝낸 수, 버린 수, 마스크가 몇 프레임 묵었는지, 첫 마스크 전 프레임 수. `--profile-out` JSON에도 들어갑니다.
- 세그 스레드에서 오류가 나면 다음 프레임에서 메인 루프가 같은 오류를 냅니다. 종료 때는 세그 스레드를 먼저 멈춘 뒤 DX-M1 엔진을 닫습니다.

### 3.3 설정과 실행 옵션

- 세 설정 파일의 npu 항목에 `dx_det_model`(기본 models/yolov8s.dxnn), `dx_det_format`, `dx_seg_model`(기본 null), `dx_input_rgb`, `dx_seg_channels_first`, `dx_iou_thr`을 더했습니다.
- run_demo.py: `--backend dxm1`(DX-M1 단독 검출), `--seg-npu dxm1`(세그만 DX-M1 별도 스레드). 기본값은 예전 그대로(sim, none)입니다.
- profiling.py: 단계 이름 `dxm1_infer`(DX-M1 추론), `seg_async`(세그 추론, 별도 스레드)를 더했습니다.
- scripts/probe_dxm1.py: 1장의 API 확인 스크립트입니다.

## 4. 시험

| 장소 | 결과 | 명령 |
|---|---|---|
| 맥 | 70 통과, 2 건너뜀 | `/usr/bin/python3 -m pytest -q` |
| 파이 | 72 통과 | `RV_DX_TEST_MODEL=$HOME/DaeDron/runs/npu/deepx/tiny_p3p5_relu.dxnn timeout 120 python3 -m pytest -q` |

맥에서 건너뛴 2개는 진짜 dx_engine이 있어야 하는 시험입니다. 새 시험(tests/test_dxm1.py)은 dx_engine 3.4.2를 흉내 낸 가짜 엔진으로 다음을 봅니다.

- 출력 해석: 두 가지 축 순서, 정규화 좌표, 로짓, yolov5 물체 점수, 레터박스 되돌리기
- CPU NMS: 겹친 같은 클래스는 지우고 다른 클래스는 남기기, 최대 개수
- 입력 준비: NHWC uint8, NCHW float32, RGB와 BGR 순서, 가짜 엔진의 dtype 검사 통과
- PPU 모델: infer()는 오류, infer_raw()는 원시 출력
- 동시 실행 스케줄: 세그가 막혀 있어도 메인 루프가 진행, 3프레임마다 넘기기, 밀린 작업 버리기, 최근 마스크 재사용과 묵은 정도, 오류 전달, 종료 순서
- 파이 전용: dx_engine import, 장치 수와 온도 읽기, 드론 모델로 카메라 2대 원시 출력 모양 확인

## 5. 두 NPU 동시 실행 맛보기

세그 .dxnn이 없어서 드론 모델을 세그 자리에 넣어 시간만 쟀습니다. 마스크 내용은 의미가 없습니다. 조건은 카메라 2대(configs/two_cam_current.yaml), 230프레임 중 200프레임 측정, 창 없음, 2026-11-24입니다. 설정은 dx_seg_model만 바꾼 사본(out/tmp/two_cam_dxseg_stub.yaml)을 썼습니다.

| 단계 | Hailo만 ms | Hailo + DX-M1 세그 ms |
|---|---|---|
| Hailo 추론 | 14.0 | 14.9 |
| 세그 추론(별도 스레드) | 없음 | 27.8 |
| BEV 합성 | 22.9 | 26.9 |
| 렌더링 | 28.5 | 43.6 |
| 프레임 전체 | 70.9 | 92.1 |
| 루프 FPS | 14.11 | 10.86 |

- 세그는 77번 넘겨 77번 끝났고 버린 작업은 0, 마스크는 평균 2.0프레임, 최대 3프레임 묵었습니다. 스로틀링은 두 실행 모두 0x0입니다.
- Hailo 추론 시간은 거의 그대로라 두 NPU가 서로를 막지는 않았습니다.
- 늘어난 시간은 렌더링 15.1ms, BEV 합성 4.0ms입니다. 둘 다 도로 마스크가 생겨서 하는 CPU 일(마스크 칠하기, 마스크 IPM 합성)이 주된 원인으로 보입니다. 세그 스레드의 CPU 전처리가 CPU를 나눠 쓴 몫도 있을 수 있고, 둘의 비율은 확인 필요입니다.
- 근거로 파이에서 마스크 칠하기만 따로 쟀습니다. 무작위 50% 마스크에서 카메라 2대 영상 칠하기가 39.3ms(마스크 없으면 0.5ms), 탑뷰 칠하기가 8.9ms 더 들었습니다(out/measure/02_render_mask_cost.log). 실제 실행의 마스크는 드론 모델 출력이라 도로 비율을 재지 않았고, 그래서 증가폭이 이 값과 다릅니다.
- render.py는 이번 작업에서 고치지 않았습니다.
- 그래서 계획의 "두 NPU를 함께 쓸 때 FPS가 1단계보다 낮아지지 않음"은 아직 달성하지 못했습니다. 렌더링 쪽 마스크 칠하기를 고친 뒤 진짜 세그 모델로 다시 재야 합니다.

원본: out/measure/02_profile_hailo_only.json, 02_profile_dual_stub.json과 같은 이름의 .log

## 6. .dxnn 얻는 방법

### 6.1 Ultralytics로 만들기 (x86-64 리눅스)

구현 계획 3장과 참고 자료 P7(https://docs.ultralytics.com/integrations/deepx)에 적힌 내용입니다. 내보내기는 x86-64 리눅스에서만 되고, INT8만 지원하며, 검출, 세그멘테이션, 시맨틱 세그멘테이션, 깊이 추정을 지원합니다. 시맨틱 세그멘테이션과 깊이는 YOLO26에서만 됩니다.

x86-64 리눅스 컴파일 서버에서 실행합니다.

```bash
python3 -m venv ~/dx_export && . ~/dx_export/bin/activate
pip install ultralytics
python3 -c 'from ultralytics import YOLO; YOLO("yolov8s.pt").export(format="deepx", imgsz=640)'
```

- 결과 파일 이름과 폴더 구조는 확인 필요입니다. 나온 .dxnn을 파이로 옮깁니다.
- INT8 보정 이미지 지정 옵션(예: data 인자)의 이름과 기본값은 확인 필요입니다. 보정 이미지는 실제 장착 카메라 영상과 비슷해야 합니다(P7).
- 내보낸 모델의 출력이 NMS 전 (1, 84, 8400) 하나인지 확인 필요입니다. 아니면 `infer_raw()`로 받아 맞는 후처리를 더해야 합니다.

```bash
scp <내보낸 파일>.dxnn pi5:~/work/fsd_diff_npu/models/yolov8s.dxnn
```

파이에서 출력 모양 확인과 데모 실행(파이 저장소 폴더, .venv 켠 뒤):

```bash
timeout 60 python3 scripts/probe_dxm1.py --model models/yolov8s.dxnn --loops 100 --out out/measure/02_dx_probe_yolov8s.json
timeout 60 run_model -m models/yolov8s.dxnn -b -t 10 -w 20 | tee out/measure/02_dx_run_model_yolov8s.log
timeout -s INT -k 10 180 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend dxm1 --headless --frames 330 --profile --profile-out out/measure/02_profile_dxm1.json
```

결과는 각각 out/measure/의 JSON과 로그에 남습니다.

### 6.2 DEEPX 공식 모델 저장소

미리 컴파일된 YOLOv8s, YOLOv5s, 세그 모델의 정확한 이름과 받는 주소는 확인 필요입니다. 저장소 문서에는 DEEPX 모델 저장소 주소가 없고, 이번 작업에서는 웹 문서 읽기 권한이 없어 공식 페이지를 열지 못했습니다. 이름과 주소를 지어내지 않으려고 비워 둡니다. 사람이 DEEPX 개발자 사이트(https://developer.deepx.ai)에서 모델 저장소 페이지를 찾아 이름, 주소, 입력 형식, PPU 여부를 알려 주면 이 장을 채웁니다.

### 6.3 세그 모델

계획대로 도로 세그를 DX-M1에 맡기려면 시맨틱 세그 .dxnn이 필요합니다. P7에 따르면 Ultralytics에서는 YOLO26으로만 만들 수 있습니다. 모델 이름, 학습 데이터(Cityscapes 여부), 도로 클래스 번호, 출력 모양은 확인 필요입니다. 받으면 `dx_seg_model`에 경로를, `road_classes`에 도로 클래스 번호를 적고 다음으로 확인합니다.

```bash
timeout 60 python3 scripts/probe_dxm1.py --model models/road_seg.dxnn --loops 50 --out out/measure/02_dx_probe_seg.json
timeout -s INT -k 10 180 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --seg-npu dxm1 --headless --frames 330 --profile --profile-out out/measure/02_profile_dual.json
```

## 7. 사람이 할 일

1. 모델 내려받기 허락: Ultralytics 가중치(yolov8s.pt 또는 YOLO26 세그)와 DEEPX 모델 저장소 파일을 받아도 되는지 정해 주세요.
2. 컴파일 서버: x86-64 리눅스 PC나 서버에 접속할 수 있게 해 주세요. 6.1절 명령을 그곳에서 돌립니다.
3. 웹 문서 확인: DEEPX 모델 저장소 페이지 주소를 알려 주거나, 에이전트가 공식 문서를 읽도록 WebFetch를 허용해 주세요.
4. DX-M1 방열: 10초 벤치에서 89도까지 올랐습니다. 방열판이나 팬이 붙어 있는지 확인해 주세요.
5. 렌더링 개선 결정: 두 NPU 맛보기 실행에서 마스크가 생기자 렌더링이 15.1ms, BEV 합성이 4.0ms 늘었습니다(카메라 2대, 200프레임, 2026-11-24). 마스크 칠하기를 출력 화소가 반올림 차이(1 이하)만큼 달라지는 OpenCV 방식으로 바꿔도 되는지 정해 주세요.

## 8. 확인 필요 목록

- YOLOv8s .dxnn의 출력 모양, 입력 색 순서, FPS
- 세그 .dxnn의 출력 모양(NCHW 로짓인지, 클래스 번호인지)과 도로 클래스 번호
- float 입력 .dxnn의 정규화 방식(지금은 0~1로 나눔)
- DEEPX 모델 저장소의 모델 이름과 주소
- DX-M1 방열판 장착 여부와 오래 돌릴 때의 온도
- 같은 녹화 영상 100프레임으로 두 NPU의 검출 결과 비교(계획 4.3절 완료 기준, 모델이 생긴 뒤)
