# NPU 모델 선택지 넓히기: DX-M1 단안 깊이, Hailo 대안 HEF, 배치 비교

이 문서는 2단계 끝(구현 계획 4.3절 마지막 문단)의 단안 깊이 모델을 DX-M1에 올려 보고, 파이에 있는 Hailo HEF와 두 NPU의 배치를 비교한 측정 기록입니다. 측정일은 2026-11-24 밤, 장치는 파이 5, Hailo-8(HailoRT 4.23.0), DX-M1(DX-RT 3.4.2)이고 카메라는 configs/two_cam_current.yaml의 2대(전방 ov5647 768x576, 후방 imx708 768x432)입니다. 카메라 보정 전이라 거리 값은 판정에 쓰지 않았습니다. 두 NPU 모두 방열이 없어 DX-M1은 한 번에 30초 이내, Hailo는 60초 이내로 돌렸습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| 고른 깊이 모델 | yolo26-depth-n_768x768 | 장면이 가장 그럴듯함, 카메라 2대 17.4 FPS |
| 깊이 NPU 시간 | 장당 25.4ms | 카메라 2대, 100프레임, run_async 2개 겹침 |
| 가벼운 대안 | fastdepth_224x224 | 장당 2.6ms, 경계가 흐림 |
| Hailo 도로 세그, 깊이 HEF | 파이에 없음 | /usr/share/hailo-models 의 Hailo-8용 4개 확인 |
| Hailo 검출 대안 | yolov6n_h8 | 데모 29.9 FPS(카메라 30 fps에 묶임), yolov8s는 25.0 |
| 지금 배치 | 18.6 FPS | Hailo 검출 + DX-M1 BiSeNetV2 세그, 15초 |
| 반대 배치 대용 | 16.2 FPS | DX-M1 검출 + Hailo에 yolov5n_seg 부하, 도로 마스크 없음 |
| DX-M1 PPU 장치 문턱값 | 0.001 | 모델 설정값과 실제 최저 점수가 같음 |
| 시험 | 맥 162 통과, 4 건너뜀 / 파이 166 통과 | 맥 /usr/bin/python3, 파이 .venv |

## 1. DX-M1 단안 깊이

### 1.1 받은 모델과 입출력

DEEPX 모델 저장소 2_4_0 목록의 "Depth Estimation" 분류에서 다섯 개를 받았습니다. 주소와 sha256은 05_dxnn_sources.md 6장에 있습니다. 목록에 YOLO26 깊이 모델(n, s, m, l, x)이 있어 계획 4.3절에 적은 "YOLO26 깊이 모델"을 그대로 시험할 수 있었습니다.

`scripts/probe_dxm1.py` 로 확인한 입출력은 다음과 같습니다. `parse_model` 로 보면 scdepthv3와 fastdepth는 NPU 작업 하나로 끝나고, yolo26-depth n과 s는 NPU 작업 뒤에 CPU 작업(cpu_0) 하나가, depthanythingv2-vits는 앞뒤에 CPU 작업이 붙습니다.

| 모델 | 입력 | 출력 |
|---|---|---|
| yolo26-depth-n_768x768 | [1,768,768,3] uint8 | depth [1,1,768,768] float32 |
| yolo26-depth-s_768x768 | [1,768,768,3] uint8 | depth [1,1,768,768] float32 |
| scdepthv3_256x320 | [1,256,320,3] uint8 | [1,1,256,320] float32 |
| fastdepth_224x224 | [1,224,224,3] uint8 | [1,1,224,224] float32 |
| depthanythingv2-vits_224x224 | [1,3,224,224] float32 | [1,224,224] float32 |

depthanythingv2-vits는 입력이 float32 NCHW라 정규화(평균, 표준편차)가 모델 밖에 남아 있고 그 값을 확인하지 않아 더 재지 않았습니다. probe_dxm1.py는 이 모양에서 멈추던 것을 고쳤지만(NCHW 입력, `--resize` 옵션) 이 모델로 다시 돌리지는 않았습니다.

DEEPX dx_app v3.2.2의 깊이 예제(yolo26_depth_n, scdepthv3, fastdepth_1)는 모두 단순 리사이즈 전처리(SimpleResizePreprocessor)와 FastDepthPostprocessor를 씁니다. 이 후처리는 출력의 최솟값과 최댓값으로 0~1에 펴서 보여 주기만 하므로, 값이 미터인지, 깊이인지 역깊이인지는 DEEPX 코드로는 알 수 없습니다. 그래서 백엔드도 원시 값을 그대로 돌려주고, 값의 뜻은 `kind` 옵션(depth 또는 inverse)으로 정합니다.

### 1.2 속도

한 장씩 재기는 probe_dxm1.py로 같은 사진(01_frame_cam0.jpg, 레터박스 입력)을 20번 넣은 값입니다. "4개 겹침"은 run_async 4개를 보내고 wait한 값입니다.

| 모델 | 한 장씩 / 4개 겹침 FPS | 지연 / NPU ms |
|---|---|---|
| yolo26-depth-n_768x768 | 30.5 / 63.8 | 36.5 / 23.1 |
| yolo26-depth-s_768x768 | 24.7 / 50.9 | 44.3 / 30.9 |
| scdepthv3_256x320 | 76.2 / 109.6 | 19.5 / 6.3 |
| fastdepth_224x224 | 148.8 / 195.7 | 10.6 / 2.2 |

카메라 2대 실행은 새 도구 `scripts/depth_check.py` 로 카메라에서 받은 프레임 100개(앞 5개는 통계에서 뺌)를 깊이 모델에만 넣은 값입니다. 깊이 단계는 카메라 2장을 run_async로 함께 보내고 둘 다 받을 때까지의 시간입니다.

| 모델 | 루프 FPS | 깊이 단계 ms (2장) / NPU ms (장당) |
|---|---|---|
| yolo26-depth-n | 17.4 | 50.8 / 25.4 |
| scdepthv3 | 30.0 (카메라에 묶임) | 23.1 / 6.8 |
| fastdepth | 30.0 (카메라에 묶임) | 12.7 / 2.6 |

yolo26-depth-n은 2장을 함께 보내도 50.8ms로 장당 NPU 시간 25.4ms의 약 2배라, 두 장이 거의 겹치지 않았습니다. 전처리(768x768로 줄이기, 3.1ms)는 이 값에 들어 있지 않습니다. 검출(Hailo)과 같은 프레임마다 돌리면 루프가 20 FPS 아래로 묶이므로, 세그처럼 별도 스레드에서 몇 프레임마다 돌려야 합니다.

### 1.3 눈으로 본 결과

카메라별 컬러 깊이 이미지는 out/measure/09_depth_<모델>_<카메라>.jpg 에 있습니다(왼쪽 원본, 오른쪽 깊이, 가까울수록 밝은 노랑, 멀수록 검정). yolo26-depth-n 이미지에는 DX-M1 YOLOv8s PPU 검출 박스(COCO 80 클래스, 점수 0.25)를 함께 그렸습니다.

- yolo26-depth-n: 전방에서 앞의 의자와 책상 모서리가 가장 밝고, 모니터와 뒤쪽 벽이 어둡습니다. 후방에서도 의자 등받이와 꽃병이 밝고 오른쪽 벽 모서리가 가장 어둡습니다. 물체 경계가 뚜렷하고 방 구조와 맞습니다. 값 범위는 전방 0.72~4.48, 후방 0.69~3.14이고 큰 값이 먼 곳이라 kind는 depth입니다. 방 크기(몇 m)와 비슷한 범위라 미터일 가능성이 있지만 확인 필요입니다.
- fastdepth: 큰 흐름(아래쪽 가깝고 모니터 쪽 멂)은 맞지만 경계가 번지고 천장 일부가 가깝게 칠해졌습니다. kind는 depth로 추정했습니다.
- scdepthv3: 값이 모두 음수(-6.5~-3.3)입니다. 전방은 역깊이로 보면 아래쪽 책상이 가깝게 칠해져 그럴듯하지만, 후방은 바닥 흐름 추정이 depth로 나와 두 카메라에서 방향이 엇갈렸습니다. 출력이 로그 값인지 등 뜻을 알 수 없어 확인 필요로 둡니다.

`guess_kind()` 는 영상 아래쪽 가운데 띠의 값이 위로 갈수록 커지는지로 깊이와 역깊이를 추정합니다. 방 안 장면은 아래쪽이 바닥이 아니라 책상과 의자라 이 추정이 흔들릴 수 있어, 실제 실행에서는 설정으로 kind를 정하고 추정은 확인용으로만 씁니다.

### 1.4 지면 평면 거리와 비교하는 코드 (보정 후 실행)

새 모듈 `realityview/depth.py` 에 다음을 넣었습니다.

- `box_bottom_depth()`: 검출 박스 아래 가장자리에서 박스 높이의 2%만큼 안으로 들어간 띠(높이 8%, 가운데 절반 폭)의 중앙값. 가장자리 화소는 바닥과 물체가 섞이기 때문입니다.
- `ground_depth()`: 픽셀이 가리키는 지면 점(IPM)을 카메라 좌표로 옮겨 광축 방향 거리 z와 직선거리를 냅니다. 깊이 모델은 보통 z를 내므로 비교는 z로 합니다.
- `compare_boxes_with_ipm()`: 검출마다 박스 아래 가운데 점의 IPM z, 같은 자리의 깊이 값, 축척을 맞춘 미터 값과 그 차이를 표로 냅니다.
- 상대 깊이의 축척 맞추기 `fit_depth_scale()`: 영상 아래쪽 격자점 중 지면과 만나는 점(`ground_samples()`, 도로 마스크를 주면 그 안의 점만)의 모델 값과 IPM z를 짝지어 맞춥니다. kind가 depth면 비율의 중앙값 하나(축척만), inverse면 역깊이 공간에서 축척과 이동(1/z = a x 값 + b)을 Theil-Sen으로 잡은 뒤 튀는 점을 빼고 최소제곱으로 다듬습니다.

시험(tests/test_depth.py)은 높이 1.5m 카메라의 합성 평지에서 모르는 축척 2.5를 되찾는지, 역깊이에 축척과 이동을 붙이고 점의 10%를 튀게 해도 되찾는지, 평지 위 박스에서 깊이 거리와 IPM 거리 차이가 10% 안인지를 봅니다.

이번 실행에서도 코드가 끝까지 도는지는 확인했습니다. yolo26-depth-n에서 지면 점 672개(전방), 528개(후방)로 축척 2.05, 1.68이 나왔고, 박스 10개(전방 7, 후방 3)의 비교 행이 out/measure/09_depth_yolo26n.json 에 있습니다. 카메라 위치와 내부 파라미터가 보정 전 가정값이고 아래쪽이 실제 바닥도 아니라서 이 숫자는 의미가 없습니다. 보정 뒤 경사로나 턱이 있는 모형 장면에서 다시 재야 계획의 질문(평지 가정이 틀어지는 곳에서 오차가 줄어드는지)에 답할 수 있습니다.

## 2. Hailo-8 대안 HEF

### 2.1 파이에 있는 HEF

/usr/share/hailo-models 에는 HEF 18개가 있고 그중 Hailo-8용(h8)은 4개입니다. 나머지는 Hailo-8L(h8l)과 Hailo-10(h10)용입니다. 계획에 적은 stdc1(도로 세그)이나 scdepthv3(깊이) HEF는 없습니다. 각 HEF를 `hailortcli run -t 8` 로 8초씩 돌렸습니다(입력은 hailortcli가 만든 값).

| HEF | 용도와 출력 | FPS |
|---|---|---|
| yolov8s_h8 | 검출, NMS 내장(점수 0.2, IoU 0.70) | 283.7 |
| yolov6n_h8 | 검출, NMS 내장(점수 0.2, IoU 0.65) | 549.1 |
| yolov5n_seg_h8 | COCO 인스턴스 세그, 출력 4개(CPU 후처리 필요) | 120.4 |
| yolov8s_pose_h8 | 사람 자세, 출력 9개 | 219.1 |

06 문서에서 확인 필요로 남긴 Hailo HEF 안의 NMS 값은 `hailortcli parse-hef` 로 확인했습니다. yolov8s_h8은 점수 0.2, IoU 0.70이고, DX-M1 쪽 CPU NMS(IoU 0.45)와 다릅니다. 두 NPU 비교에서 Hailo가 겹친 상자를 조금 더 남기는 쪽입니다.

yolov5n_seg는 COCO 물체의 인스턴스 마스크라 도로 마스크를 내지 않습니다. 그래서 seg_hef에 넣지 못하게, 출력이 1개가 아닌 HEF를 넣으면 `HailoBackend` 가 이유를 적은 오류를 내게 했습니다(`check_hailo_seg_outputs`). yolov8s_pose도 이 프로젝트에 쓸 곳이 없습니다.

### 2.2 검출 HEF 고르기

검출 HEF는 설정의 npu.det_hef 대신 `run_demo.py --det-hef` 로 바꿀 수 있게 했습니다(기본은 설정 그대로). 같은 세션에서 Hailo 단독 데모를 12초씩 돌린 값입니다(앞 30프레임 뺌, 창 없음).

| 검출 HEF | 루프 FPS | Hailo 추론 / 프레임 전체 ms |
|---|---|---|
| yolov8s_h8 | 25.0 | 13.9 / 39.9 |
| yolov6n_h8 | 29.9 | 7.3 / 33.3 |

yolov6n은 Hailo 추론이 절반이고 루프가 카메라 30 fps에 닿았습니다. 출력이 같은 NMS 형식이라 같은 코드로 끝까지 돌았지만, 이 실행에서 검출이 0개여서 상자 해석이 맞는지는 확인 필요입니다. 두 HEF의 검출 품질 비교(같은 녹화로 IoU, 검출 수)는 하지 않았으므로 확인 필요입니다.

### 2.3 Hailo 온도

hailortcli 벤치 네 개를 사이 확인 없이 이어서 돌린 것은 실수였습니다. 첫 실행 전 80.6도에서 시작해 둘째 앞 94.0도, 셋째 앞 99.6도, 넷째 앞 99.0도, 끝난 뒤 107.4도였습니다. 실행 전 85도 확인 규칙을 지키지 못했습니다. 넷째(yolov8s_pose)의 FPS가 8초 안에서 232에서 219로 떨어졌는데 열 보호로 클럭이 내려간 것으로 보이며 확인 필요입니다. 그 뒤 데모 실행은 모두 85도 이하가 될 때까지 60초 단위로 기다린 뒤 시작했습니다(out/measure/09_hailo_temp.log).

## 3. 두 NPU 배치 비교

계획대로라면 반대 배치는 "세그를 Hailo, 검출을 DX-M1"이지만, 파이에 도로 세그 HEF가 없고 이번에 허락받은 다운로드는 DEEPX 모델뿐이라 Hailo 세그 HEF를 받지 않았습니다. 그래서 반대 배치는 대용으로 쟀습니다. DX-M1이 YOLOv8s PPU로 검출하고(세그 없음, 설정 사본 out/tmp/two_cam_dxdet_only.yaml), 같은 시간에 Hailo가 hailortcli로 yolov5n_seg를 돌려(117.6 FPS, 15초 중 10초) PCIe와 CPU 부하만 비슷하게 만든 것입니다. 이 대용 실행은 도로 마스크를 만들지 않습니다.

| 배치 (같은 세션, 15초 이내) | 루프 FPS | 검출 추론 / 프레임 전체 ms |
|---|---|---|
| Hailo 검출 단독 | 25.0 | 13.9 / 39.9 |
| 지금: Hailo 검출 + DX-M1 세그 | 18.6 | 15.5 / 53.6 |
| 반대 대용: DX-M1 검출 + Hailo 부하 | 16.2 | 22.9 / 61.7 |

지금 배치에서 세그는 별도 스레드에서 한 번(카메라 2장)에 156ms였습니다. 06 문서의 같은 배치 19.6 FPS(200프레임)보다 조금 낮은 까닭은 확인 필요입니다. 조건 차이 하나는 검출 수입니다. 지금 배치 실행에서는 267프레임에 검출 531개(지면 투영 2.4ms)가 있었고, Hailo 단독, 반대 대용, yolov6n 실행에서는 0개였습니다. 이 차이는 지금 배치에 불리한 쪽이라 아래 추천은 바뀌지 않습니다. 반대 대용은 세그 마스크 덧칠이 없는데도 2.4 FPS 낮았습니다. DX-M1 검출(카메라 2장 22.9ms)이 Hailo 검출(15.5ms)보다 느린 것이 가장 큰 차이입니다. 진짜 반대 배치는 여기에 Hailo 세그 후처리와 마스크 덧칠이 더해지므로 더 느려질 것으로 봅니다(확인 필요).

## 4. DX-M1 YOLOv8s PPU 장치 문턱값

장치 쪽 점수 문턱값은 0.001입니다. 근거는 세 가지입니다.

1. yolov8-s_640x640_ppu.dxnn 안의 compile_config에 `"ppu": {"type": 1, "num_classes": 80, ..., "conf_thres": 0.001}` 이 있습니다.
2. 같은 파일의 PPU 설정 블록(20바이트) 안에 float 0.001이 들어 있습니다.
3. 실제 사진 두 장(01_frame_cam0.jpg, 01_frame_cam1.jpg)에서 장치가 돌려준 상자가 344개, 135개였고 최저 점수가 모두 0.0010이었습니다(out/measure/09_ppu_thr.json).

그래서 실제로 남는 상자는 설정의 score_thr가 정합니다. 효과는 max(0.001, score_thr)이고, 기본 0.4나 비교에 쓴 0.25에서는 score_thr 그대로입니다. score_thr를 0.001보다 낮춰도 더 나오지 않습니다. 장치가 후보를 수백 개 넘기지만 CPU에서 점수로 거른 뒤 NMS를 하므로 NMS 파싱은 카메라 2장에 0.5ms였습니다(반대 대용 실행). 06 문서 3장에서 Hailo와 짝을 못 지은 점수 0.25 근처의 의자는 장치 문턱값 때문이 아닙니다. detectors.py에 이 값을 `PPU_DEVICE_SCORE_THR` 상수와 주석으로 남겼습니다.

## 5. 모델별 정리

NPU 시간은 장당 값입니다. 전체 FPS는 카메라 2대 루프이고, 온도는 그 실행 전후의 칩 최고 온도입니다.

| 모델 (NPU) | NPU 시간 | 전체 FPS / 온도 |
|---|---|---|
| yolov8s_h8 검출 (Hailo) | 7ms (카메라 2장 13.9ms) | 25.0 / Hailo 84.8에서 87.2도 |
| yolov6n_h8 검출 (Hailo) | 3.7ms (카메라 2장 7.3ms) | 29.9 / Hailo 83.8에서 84.7도 |
| YOLOv8s PPU 검출 (DX-M1) | 15.4ms (06 문서) | 16.2 / DX-M1 78에서 82도 |
| BiSeNetV2 세그 (DX-M1) | 74.1ms (06 문서) | 18.6 / DX-M1 77에서 84도 |
| yolo26-depth-n 깊이 (DX-M1) | 25.4ms | 17.4 / DX-M1 77에서 80도 |
| scdepthv3 깊이 (DX-M1) | 6.8ms | 30.0 / DX-M1 79에서 80도 |
| fastdepth 깊이 (DX-M1) | 2.6ms | 30.0 / DX-M1 78에서 79도 |

Hailo 검출 NPU 시간은 데모의 "Hailo 추론"(전송 포함)을 장 수로 나눈 값이라 hailortcli 처리량(yolov8s 3.5ms, yolov6n 1.8ms)보다 깁니다. BiSeNetV2 줄의 FPS는 Hailo 검출과 함께 돌린 지금 배치 값이고, YOLOv8s PPU 줄의 FPS는 Hailo에 yolov5n_seg 부하를 건 반대 대용 실행 값입니다. 깊이 모델은 깊이만 돌린 값입니다. 각 실행은 8~20초라 온도는 짧은 실행 기준입니다.

## 6. 다음 단계 추천

검출은 Hailo-8, 도로 세그는 DX-M1에 두는 지금 배치를 유지하길 권합니다. 반대 대용 배치가 2.4 FPS 낮았고, Hailo에는 쓸 수 있는 도로 세그 HEF가 없습니다.

깊이는 yolo26-depth-n을 DX-M1에 두고, 세그처럼 별도 스레드에서 몇 프레임마다(예: 3~6프레임) 돌리는 쪽을 권합니다. 세그(장당 74ms)와 깊이(장당 25ms)를 함께 돌리면 DX-M1의 일이 지금보다 약 1.3배가 되고, DX-M1은 이번에도 15초 동시 실행에 77도에서 84도로 올랐습니다. 그래서 깊이 스레드는 DX-M1 방열팬을 단 뒤에 붙이고, 그 전에는 fastdepth(장당 2.6ms)를 가벼운 대안으로 둡니다.

Hailo 검출은 정확도 확인 전까지 yolov8s_h8을 유지합니다. FPS가 더 필요하면 yolov6n_h8로 바꿀 수 있지만 먼저 같은 녹화에서 검출 품질을 비교해야 합니다.

## 7. 시험

| 장소 | 결과 | 명령 |
|---|---|---|
| 맥 | 162 통과, 4 건너뜀 | `/usr/bin/python3 -m pytest -q` |
| 파이 | 166 통과 | `RV_DX_TEST_MODEL=$HOME/work/fsd_diff_npu/models/dxnn/yolov8-s_640x640_ppu.dxnn timeout 300 python3 -m pytest -q -p no:warnings` |

새 시험 tests/test_depth.py(21개)는 깊이 출력 모양 정리, 원본 크기 늘리기, 컬러에서 가까운 곳이 밝은지, 깊이와 역깊이 추정, 박스 아래 띠 값, 합성 평지에서 축척 되찾기(튀는 점 포함), 도로 마스크로 지면 점 거르기, 깊이와 IPM 비교, 가짜 엔진으로 DXM1Backend 깊이 경로(단순 리사이즈 입력, 원본 크기 출력, 모델 없을 때 오류, 닫기), Hailo seg HEF 출력 수 확인, run_demo `--det-hef` 기본값, probe_dxm1의 NCHW 입력, depth_check.py를 사진으로 끝까지 돌리기를 봅니다. 원본: out/measure/09_pytest_mac.log, 09_pytest_pi.log

## 8. 다시 재는 명령

파이 저장소 폴더에서 .venv를 켠 뒤 실행합니다. 결과는 out/measure/ 에 생깁니다. DX-M1을 쓰는 줄 앞뒤로 `python3 scripts/dx_temp.py --label 이름 --log out/measure/09_dx_temp.log --guard 79 --resume-below 79` 를 붙입니다.

```bash
timeout 30 python3 scripts/probe_dxm1.py --model models/dxnn/yolo26-depth-n_768x768.dxnn --image out/measure/01_frame_cam0.jpg --resize --loops 20 --out out/measure/09_dx_probe_yolo26-depth-n_768x768.json
timeout -s INT -k 5 45 python3 scripts/depth_check.py --config configs/two_cam_current.yaml --depth models/dxnn/yolo26-depth-n_768x768.dxnn --det models/dxnn/yolov8-s_640x640_ppu.dxnn --frames 100 --duration 20 --out out/measure/09_depth_yolo26n
timeout -s INT -k 5 20 hailortcli run /usr/share/hailo-models/yolov6n_h8.hef -t 8
timeout -s INT -k 10 40 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --det-hef /usr/share/hailo-models/yolov6n_h8.hef --headless --duration 12 --warmup 30 --profile --profile-out out/measure/09_hailo_det_yolov6n_h8.json --health-csv out/measure/09_hailo_det_yolov6n_h8.csv --health-interval 10 --hailo-stop-temp 95
timeout -s INT -k 10 40 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --seg-npu dxm1 --headless --duration 15 --warmup 30 --profile --profile-out out/measure/09_place_current.json --health-csv out/measure/09_place_current.csv --health-interval 10 --hailo-stop-temp 95
```

## 9. 확인 필요

- yolo26-depth와 fastdepth 출력이 미터인지, scdepthv3의 음수 출력의 뜻
- depthanythingv2의 입력 정규화 값
- 보정 뒤 경사로나 턱이 있는 장면에서 깊이 모델 거리와 IPM 거리의 오차
- yolov6n_h8의 상자 해석(검출이 있는 장면에서)과 yolov8s_h8과의 검출 품질 차이
- 진짜 반대 배치(Hailo 도로 세그 HEF)의 FPS. stdc1 같은 HEF를 받을지는 사람이 정합니다
- hailortcli 벤치 넷째 실행의 FPS 하락이 열 보호 때문인지
