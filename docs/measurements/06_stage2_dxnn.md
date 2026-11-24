# 2단계: 실제 .dxnn 모델로 DX-M1 검출, 세그, 동시 실행 측정 기록

이 문서는 DEEPX 공식 모델 저장소의 .dxnn으로 DX-M1 백엔드의 후처리를 실제 출력 형식에 맞추고, 같은 녹화 영상에서 Hailo-8과 검출 결과를 비교하고, Hailo 검출과 DX-M1 세그를 동시에 돌려 FPS를 잰 기록입니다. 측정일은 2026-11-24, 장치는 파이5, Hailo-8(yolov8s_h8.hef), DX-M1(DX-RT 3.4.2)입니다. 모델 출처는 05_dxnn_sources.md에 있습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| DX-M1 검출 모델 | yolov8-s_640x640_ppu.dxnn | 장치 후처리(PPU), 원시 출력 모델보다 3.6배 빠름 |
| DX-M1 세그 모델 | bisenetv2_1024x2048.dxnn | Cityscapes 19 클래스, 0이 road |
| 녹화 100프레임 검출 FPS | Hailo 55.6, DX-M1 PPU 41.1 | 카메라 2대 묶음, infer()만 잰 값 |
| 검출 위치 차이 | IoU 평균 0.973, 중심 1.38px | COCO 80 클래스, 점수 0.25, 같은 클래스 IoU 0.5 이상 짝 |
| 동시 실행 FPS | 19.59 | Hailo 검출 + DX-M1 세그 3프레임마다, 200프레임 |
| 같은 세션 Hailo 단독 | 26.16 | 같은 조건에서 세그만 뺌 |
| 시험 | 맥 116 통과, 4 건너뜀 / 파이 120 통과 | 맥 /usr/bin/python3, 파이 .venv |
| DX-M1 최고 온도 | 83도 | 동시 실행 약 14초 직후 |

계획 4.3절 완료 기준에 대한 판정은 다음과 같습니다. 동시 실행 19.59 FPS는 1단계 기록 14.2 FPS보다 높습니다. 같은 세션의 Hailo 단독 26.16 FPS(03 문서 기록 26.2와 같음)보다는 25% 낮습니다. 녹화 100프레임 비교표는 3장에 있습니다. 직접 도로를 칠한 20프레임의 도로 마스크 IoU는 방 안에 도로가 없어 재지 못했습니다.

## 1. 모델 입출력 형식 (probe_dxm1.py 실측)

`scripts/probe_dxm1.py` 에 실제 사진을 넣는 `--image` 옵션을 더하고, PPU 모델의 'BBOX' 자료형에서 멈추던 곳을 고친 뒤 확인했습니다. 다섯 모델 모두 입력은 NHWC uint8이었습니다.

| 모델 | 입력 | 출력 |
|---|---|---|
| yolov8-s_640x640 | [1,640,640,3] uint8 | [1,84,8400] float32, 픽셀 좌표, 점수는 이미 0~1 |
| yolov8-s_640x640_ppu | [1,640,640,3] uint8 | (1,N,32) uint8 상자 구조체, 상자 없으면 (0,0) |
| bisenetv2_1024x2048 | [1,1024,2048,3] uint8 | [1,1024,2048] uint16 클래스 번호 |
| segformer_mit-b0_512x1024 | [1,512,1024,3] uint8 | [1,19,128,256] float32 로짓 |
| stdc2-seg50_512x1024 | [1,512,1024,3] uint8 | [1,19,512,1024] float32 로짓 3개 |

속도는 같은 입력을 반복해 넣은 값입니다. "한 장씩"은 run() 반복, "4개 겹침"은 run_async 4개를 보내고 wait한 값입니다.

| 모델 | 한 장씩 / 4개 겹침 FPS | 지연 / NPU ms |
|---|---|---|
| yolov8-s_640x640 | 22.0 / 33.2 | 54.9 / 6.8 |
| yolov8-s_640x640_ppu | 64.9 / 107.3 | 19.0 / 15.4 |
| bisenetv2_1024x2048 | 13.0 / 19.4 | 98.9 / 74.1 |
| segformer_mit-b0_512x1024 | 0.43 / 0.43 | 2331 / 5.3 |
| stdc2-seg50_512x1024 | 1.6 / 3.0 | 687.5 / 71.4 |

- 원시 출력 YOLOv8s는 NPU 작업 뒤에 CPU 작업(parse_model의 cpu_0, 6개 출력을 output0 하나로 합침)이 붙어 있어 지연 55ms 중 NPU는 7ms입니다. PPU 모델은 CPU 작업이 없습니다.
- SegFormer와 STDC2도 CPU 작업이 대부분이라 세그 자리에서 뺐습니다. BiSeNetV2는 NPU 작업 하나로 끝나고 argmax까지 장치에서 합니다.
- PPU 상자 32바이트는 DX-RT 헤더(/usr/local/include/dxrt/dxrt_cxx_api.h)의 DeviceBoundingBox_t(x, y, w, h, grid_y, grid_x, box_idx, layer_idx, score, label, 빈칸 4)와 같습니다. DEEPX dx_app v3.2.2의 YOLOv8PPUPostprocessor처럼 x, y를 상자 중심, w, h를 폭과 높이(모델 입력 픽셀)로 읽고, label은 COCO 번호입니다. 이 해석은 3장의 정지 영상에서 원시 출력 모델과 같은 상자가 나오는 것으로 확인했습니다.
- 도로 클래스 번호 0은 dx_app v3.2.2의 `common/utility/labels.py` Cityscapes 목록 첫 항목(road)을 따랐습니다. BiSeNetV2가 Cityscapes로 학습한 것은 입력 크기(1024x2048)와 출력 값 범위(0~18)로 본 추정이며 학습 데이터 문서는 확인 필요입니다.
- 원본: out/measure/06_dx_probe_*.json

## 2. 후처리를 실제 형식에 맞춘 내용 (realityview/detectors.py)

| 곳 | 예전 | 지금 |
|---|---|---|
| 원시 출력 점수 | 0 미만이 하나라도 있으면 시그모이드 | 0~1을 0.001 넘게 벗어날 때만 시그모이드 |
| PPU 모델 | infer()가 오류 | 상자 구조체를 읽고 점수, 클래스로 거른 뒤 CPU NMS |
| 세그 클래스 번호 출력 | 200만 화소에서 비교 후 축소 | 원본 크기로 먼저 줄이고(최근접) 비교 |

- 원시 출력 YOLOv8s의 점수 최솟값이 -1.19e-7이었습니다. 예전 판정으로는 확률에 시그모이드가 한 번 더 씌워져 점수 0인 앵커 8400개가 모두 0.5가 되고 문턱값 0.4를 넘습니다.
- PPU 모델은 yolov8 형식만 해석합니다. yolov5 PPU는 앵커 계산이 따로 필요해서 예전처럼 infer_raw()를 쓰라는 오류를 냅니다.
- 세그 마스크 만들기는 카메라 2대 기준 파이에서 28.8ms에서 1.7ms로 줄었습니다(무작위 클래스 맵, 30회 평균). 최근접 축소는 클래스 번호를 섞지 않아 결과가 예전과 같고, 시험으로 확인했습니다.

## 3. 같은 녹화로 두 NPU 검출 비교

녹화는 `scripts/record_frames.py` 로 했습니다. 설정 configs/two_cam_current.yaml 그대로(전방 768x576, 후방 768x432), 앞 15프레임을 버린 뒤 100프레임씩, 22.9 프레임/초로 찍혔습니다. 파이의 out/rec/room_100/ 에 PNG 200장과 timestamps.csv가 있습니다.

비교는 `scripts/compare_det.py` 로 했습니다. 영상을 모두 메모리에 올린 뒤 한 검출기씩 차례로 돌렸고(동시에 돌리지 않음), 앞 5번 호출은 시간에서 뺐습니다. Hailo를 기준으로 같은 클래스끼리 IoU가 큰 순서로 짝을 지었고 IoU 0.5 미만은 짝이 아닙니다. DX-M1 쪽 NMS는 CPU에서 IoU 0.45로 하고, Hailo HEF 안의 NMS 값은 바꿀 수 없어 그대로 두었습니다(값은 확인 필요).

### 3.1 FPS (카메라 2대 묶음, 100회)

| 검출기 | 프레임/초 | 평균 / p95 ms |
|---|---|---|
| Hailo-8 yolov8s_h8.hef | 55.6 | 18.0 / 18.1 |
| DX-M1 yolov8-s PPU | 41.1 | 24.4 / 24.9 |
| DX-M1 yolov8-s 원시 출력 | 11.4 | 87.6 / 89.1 |

### 3.2 검출 수와 위치 차이

방 안에 사람과 차가 없어서, 데모와 같은 조건(주행 클래스 6개, 점수 0.4)에서는 세 검출기 모두 200장에서 0개였습니다. 그래서 COCO 80 클래스, 점수 0.25로 다시 비교했습니다. 잡힌 물체는 의자, TV, 노트북, 꽃병입니다.

| 비교 | DX-M1 PPU | DX-M1 원시 출력 |
|---|---|---|
| 영상당 검출 수(Hailo 5.08) | 4.88 | 4.78 |
| 짝지은 수 / Hailo 검출 수 | 840 / 1016 | 831 / 1016 |
| IoU 평균 (최소) | 0.973 (0.887) | 0.981 (0.901) |
| 중심 거리 평균 (최대) px | 1.38 (11.76) | 1.13 (9.22) |

- 카메라별로 보면 전방은 Hailo 652개, PPU 676개, 후방은 Hailo 364개, PPU 300개였습니다. 짝을 못 지은 것은 대부분 점수 0.25 근처의 의자였습니다.
- 노트북, TV, 꽃병은 세 검출기 모두 100프레임 내내 잡았습니다.
- 원본: out/measure/06_compare_rec_all.json, 06_compare_rec_keep.json

### 3.3 정지 영상

1단계에서 찍은 01_frame_cam0.jpg, 01_frame_cam1.jpg 두 장에서도 같은 방법으로 비교했습니다(COCO 80 클래스, 점수 0.25). Hailo 13개, PPU 11개, 원시 출력 11개였고 PPU는 9쌍이 IoU 평균 0.966, 중심 거리 평균 2.11px로 맞았습니다. PPU와 원시 출력 모델은 같은 물체에 1~2px 안쪽의 같은 상자를 냈습니다. 원본: out/measure/06_compare_still.json(상자 좌표 포함)

## 4. Hailo 검출 + DX-M1 세그 동시 실행

조건은 configs/two_cam_current.yaml(dx_seg_model에 BiSeNetV2), 카메라 2대, 230프레임 중 앞 30프레임을 뺀 200프레임, 창 없음, 렌더링 같은 스레드입니다. 두 실행 모두 스로틀링 0x0이었습니다.

| 단계 | Hailo 단독 ms | Hailo + DX-M1 세그 ms |
|---|---|---|
| 전처리 | 3.6 | 4.4 |
| Hailo 추론 | 14.1 | 16.2 |
| 세그 추론(별도 스레드) | 없음 | 151.4 |
| BEV 합성 | 9.5 | 12.5 |
| 렌더링 | 9.3 | 15.4 |
| 프레임 전체 | 38.2 | 51.0 |
| 루프 FPS | 26.16 | 19.59 |

- 기록값과 비교하면 1단계 14.2 FPS보다 38% 높고, Hailo 단독 26.2 FPS(03 문서)보다 25% 낮습니다.
- 세그는 77번 넘겨 76번 끝났고 버린 작업은 0이었습니다. 한 번에 카메라 2대를 처리하는 데 151ms가 들어 3프레임 간격(약 150ms)과 거의 같아서, 마스크는 평균 4.8프레임, 최대 6프레임 묵었습니다. 첫 마스크 전 4프레임은 마스크 없이 지나갔습니다.
- 세그는 한 번(카메라 2장)에 151ms, 장당 76ms였습니다. 단독 확인(1장)에서는 4개를 겹쳐 보내면 19.4 FPS, 장당 52ms였으므로 동시 실행에서 더 느렸습니다. 2장만 겹치는 탓인지, Hailo와 PCIe를 나눠 쓰는 탓인지는 확인 필요입니다.
- 렌더링 6.1ms, BEV 합성 3.0ms, Hailo 추론 2.1ms가 늘었습니다. 마스크 덧칠과 세그 스레드의 CPU 전처리(2048x1024로 키우기, 카메라 2대 5.2ms)가 CPU를 나눠 쓴 몫으로 보이며, 나눠 잰 값은 확인 필요입니다.
- 도로 비율은 평균 3.9%였습니다. 방 안이라 이 값은 의미가 없습니다.
- 원본: out/measure/06_profile_hailo_only.json, 06_profile_dual.json과 같은 이름의 .log

## 5. DX-M1 온도

DX-M1을 쓰는 실행마다 앞뒤로 `scripts/dx_temp.py` 로 기록했습니다(85도를 넘으면 80도 아래까지 기다림). 85도를 넘은 적은 없어 기다린 적도 없습니다.

| 시각 | 시점 | DX-M1 최고 |
|---|---|---|
| 18:41:29 | 모델 확인 전 | 73도 |
| 18:46:34 | 세그 모델 확인 뒤 | 78도 |
| 18:53:36 | 녹화 비교 전 | 75도 |
| 18:54:09 | 녹화 비교 뒤(DX-M1 약 15초) | 79도 |
| 18:54:39 | 동시 실행 전 | 77도 |
| 18:54:53 | 동시 실행 뒤(약 14초) | 83도 |
| 19:00:51 | 파이 시험 뒤 | 74도 |

BiSeNetV2를 3프레임마다 돌린 약 14초 사이에 6도가 올랐습니다. 몇 분 이상 동시 실행하면 85도를 넘을 수 있어, 방열판이나 팬 확인이 먼저입니다. 원본: out/measure/06_dx_temp.log

## 6. 모형 설정 실행 확인

configs/two_cam_model.yaml(전방 어안, BEV 앞뒤 2m, 셀 1cm, scene.scale 0.15)로 Hailo 데모를 90프레임 돌려 30.0 FPS(카메라 30 fps에 묶임), BEV 합성 5.9ms, 렌더링 4.1ms를 확인했습니다. 지금은 카메라가 모형 설정처럼 달려 있지 않고 두 대 모두 방 앞쪽을 보고 있어 탑뷰 내용은 의미가 없습니다. 원본: out/measure/06_profile_model_rig.json, 06_model_rig_snapshot.png

## 7. 시험

| 장소 | 결과 | 명령 |
|---|---|---|
| 맥 | 116 통과, 4 건너뜀 | `/usr/bin/python3 -m pytest -q` |
| 파이 | 120 통과 | `RV_DX_TEST_MODEL=$HOME/work/fsd_diff_npu/models/dxnn/yolov8-s_640x640_ppu.dxnn timeout 300 python3 -m pytest -q -p no:warnings` |

맥에서 건너뛴 4개는 진짜 dx_engine과 models/dxnn 모델이 있어야 하는 시험입니다. 새 시험은 다음을 봅니다.

- tests/test_dxm1.py: PPU 32바이트 구조체 읽기, 상자 0개((0,0) 출력), 점수와 클래스 거르기와 NMS, 레터박스 되돌리기, 백엔드의 PPU 경로, yolov5 PPU 거절, -1.2e-7 점수에서 시그모이드를 다시 씌우지 않음, uint16 클래스 맵 마스크가 예전 방식과 같음, 설정 파일의 모델 경로, 파이 전용 실제 PPU와 BiSeNetV2 실행
- tests/test_model_rig.py: 모형 설정 읽기, 어안 화각 상한(왜곡으로 접히는 곳 포함), 화각 밖 바닥 셀을 IPM에 쓰지 않음, 어안 지면 왕복, 칸 20mm 보드 합성 코너로 어안 내부 보정 복원, 그린 보드의 코너 찾기, 바닥 보드로 외부 보정 복원, 어안 영상에 그린 바닥 보드에서 코너 순서와 floor_board_xy 순서가 같음, 장면 축척, 모형 설정으로 파이프라인 한 단계, 검출 비교 계산

원본: out/measure/06_pytest_mac.log, 06_pytest_pi.log

## 8. 다시 재는 명령

파이 저장소 폴더에서 .venv를 켠 뒤 실행합니다. 결과는 out/rec/, out/measure/에 생깁니다.

```bash
timeout 90 python3 scripts/probe_dxm1.py --model models/dxnn/yolov8-s_640x640_ppu.dxnn --image out/measure/01_frame_cam1.jpg --loops 30 --out out/measure/06_dx_probe_yolov8-s_640x640_ppu.json
timeout -s INT -k 5 120 python3 scripts/record_frames.py --config configs/two_cam_current.yaml --frames 100 --out out/rec/room_100
timeout -s INT -k 10 300 python3 scripts/compare_det.py --config configs/two_cam_current.yaml --rec out/rec/room_100 --dx models/dxnn/yolov8-s_640x640_ppu.dxnn --dx models/dxnn/yolov8-s_640x640.dxnn --all-classes --score-thr 0.25 --out out/measure/06_compare_rec_all.json
python3 scripts/dx_temp.py --label pre --log out/measure/06_dx_temp.log --guard 85 --resume-below 80
timeout -s INT -k 10 150 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --seg-npu dxm1 --headless --frames 230 --profile --profile-out out/measure/06_profile_dual.json
python3 scripts/dx_temp.py --label post --log out/measure/06_dx_temp.log
```

## 9. 확인 필요

- Hailo HEF 안의 NMS IoU 값(DX-M1 쪽 CPU NMS 0.45와 같은지)
- 사람과 차가 있는 장면에서 두 NPU 비교(이번 비교는 의자 등 실내 물체)
- 도로가 있는 장면에서 BiSeNetV2 마스크 품질과 직접 칠한 20프레임 IoU
- 동시 실행에서 늘어난 CPU 시간의 원인별 몫
- 몇 분 이상 동시 실행할 때 DX-M1 온도
- BiSeNetV2 학습 데이터(Cityscapes로 추정)
