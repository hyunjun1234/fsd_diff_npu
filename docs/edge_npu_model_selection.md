# 엣지 NPU 구성과 모델 선정: 라즈베리파이 5와 Hailo-8, DEEPX DX-M1

이 문서는 라즈베리파이 5와 M.2 NPU, 카메라 2~4대로 테슬라 FSD의 리얼리티 뷰와 비슷한 화면을 만들 때 어떤 하드웨어와 모델을 쓸지 정리한 추천안입니다. 여기서 말하는 화면은 주변 차량, 보행자, 차선, 주행 가능 영역을 위에서 내려다본 모습이나 3D로 보여 주는 시각화입니다.

출처 번호가 없는 수치는 공개 사양을 바탕으로 한 추정치이며, 실제 보드에서 측정해 고쳐야 합니다.

## 1. 결론

| 항목 | 추천 |
|---|---|
| 1단계 구현 방식 | 방식 B입니다. 카메라별로 검출과 세그멘테이션을 한 뒤 BEV로 합칩니다. 지면이 평평하다고 가정하고 미리 보정한 호모그래피를 씁니다. |
| 2단계 구현 방식 | FlashOcc 계열 occupancy 모델을 나눠서 실행합니다. 이미지 백본과 BEV 인코더는 NPU에서, view transform은 CPU에서 돌립니다. |
| NPU | Hailo-8(26 TOPS, Raspberry Pi AI HAT+)을 기본으로 하고, DEEPX DX-M1은 2단계에서 비교합니다. |
| 카메라 | 1단계는 2대(전방과 후방), 2단계는 4대(전후좌우)입니다. 4대를 쓸 때는 동기화되는 멀티카메라 어댑터를 씁니다. |
| 피해야 할 것 | BEVFormer처럼 deformable attention을 쓰는 모델을 NPU에 통째로 올리는 것입니다. grid_sample을 지원하지 않습니다. |

두 NPU 모두 CNN은 잘 돌리지만, BEV 모델의 핵심인 기하 변환 연산(grid_sample, scatter와 BEVPool, 인덱스가 바뀌는 gather)은 사실상 지원하지 않습니다. 그래서 CNN은 NPU에, 기하 변환은 CPU에 맡기는 구조 말고는 현실적인 방법이 없습니다. 이 구조에 가장 잘 맞는 것이 방식 B와, LSS 계열인 FlashOcc를 나눠 실행하는 방법입니다.

## 2. 하드웨어 비교

### 2.1 NPU

| 항목 | Hailo-8 (AI HAT+ 26T) | Hailo-8L (AI HAT+ 13T, AI Kit) | DEEPX DX-M1 |
|---|---|---|---|
| 연산 성능 | 26 TOPS INT8 | 13 TOPS INT8 | 25 TOPS INT8 [3][4][5] |
| 전력 | 약 2.5W | 약 1.5W | 2~5W [4], HAT 기준 최대 2.5~3W [5] |
| 보드 DRAM | 없습니다. 칩 안의 SRAM을 쓰고 큰 모델은 context switch로 처리합니다. | 없습니다. | 있습니다. 모듈마다 달라서 LPDDR5 4GB [3][4]도 있고 Sixfab HAT+는 LPDDR4X 1GB [5]입니다. |
| 모듈 인터페이스 | M.2 PCIe Gen3 4레인 | Hailo-8과 같습니다. | M.2 PCIe Gen3 4레인, 1레인으로도 동작 [3][4] |
| 라즈베리파이 5에서의 실제 연결 | PCIe 1레인. 기본은 Gen2이고 Gen3로 올릴 수 있습니다. | Hailo-8과 같습니다. | PCIe Gen3 1레인 [5] |
| 정밀도 | INT8, 일부 레이어는 16bit | Hailo-8과 같습니다. | INT8 (IQ8) [3] |
| 컴파일러와 런타임 | Dataflow Compiler(DFC)로 HEF를 만들고 HailoRT로 실행 | Hailo-8과 같습니다. | DX-COM으로 DXNN을 만들고 DX-RT로 실행 [5] |
| 라즈베리파이 지원 | 공식 AI HAT+, rpicam-apps와 hailo-apps 연동, Model Zoo에 모델이 많습니다. | Hailo-8과 같습니다. | Sixfab AI HAT+(90달러, 25 TOPS) 같은 다른 회사 제품 [5] |

Hailo-8은 외부 DRAM이 없어서 여러 모델을 한꺼번에 올리거나 큰 모델을 돌리면 context switch 때문에 처리량이 떨어집니다. 방식 B처럼 작은 YOLO 하나를 카메라 수만큼 묶어 돌리는 데는 문제가 없습니다.

DX-M1은 DRAM이 있어 큰 백본이나 여러 모델에 유리합니다. 그 대신 라즈베리파이용 지원이 Hailo보다 늦게 나왔고 연산자 지원 문서가 덜 공개되어 있어서, 직접 컴파일해 보아야 합니다.

라즈베리파이 5의 PCIe는 1레인이므로 Gen3 기준 대역폭이 약 1GB/s입니다. 카메라 4대의 입력(256x704x3 INT8 4장, 프레임당 약 2.1MB)을 초당 10프레임으로 보내도 약 21MB/s여서 여기서 막히지는 않습니다. 시간이 걸리는 곳은 CPU의 전처리와 BEV 합성입니다.

라즈베리파이 5에 NPU HAT를 붙이면 13~15W를 쓰므로 27W 공식 어댑터가 필요합니다. [5]

### 2.2 카메라

라즈베리파이 5에는 MIPI CSI 포트가 2개 있어서 카메라 2대는 그대로 연결하면 됩니다.

4대를 쓰는 방법은 두 가지입니다. 하나는 센서 4개의 영상을 CSI 프레임 하나로 합쳐 주는 쿼드 카메라 어댑터이고, 하드웨어 동기화를 지원하는 제품이 좋습니다. 다른 하나는 CSI 2대에 USB 2대를 더하는 것입니다. BEV 합성에서는 카메라 사이의 시간 동기가 중요하므로 쿼드 카메라 어댑터를 권합니다.

화각은 120도 이상의 광각을 권합니다. 4대로 360도를 덮으려면 수평 화각이 90도 이상이어야 합니다. 광각 렌즈는 왜곡이 커서 OpenCV fisheye calibration 같은 보정을 반드시 해야 합니다.

NPU 입력은 640x640(방식 B)이나 256x704(BEV 계열, nuScenes에서 흔히 쓰는 크기) 정도로 줄이므로 센서는 1~2MP면 충분합니다.

카메라의 위치와 자세(외부 파라미터)는 바닥에 체커보드나 ArUco 마커를 깔고 한 번에 보정합니다. 방식 B든 BEV 모델이든 이 값이 품질을 좌우합니다.

## 3. 컴파일러의 연산자 지원

BEV와 occupancy 모델에 필요한 연산자를 기준으로 정리했습니다.

| 연산자 또는 블록 | 쓰이는 곳 | Hailo DFC | DEEPX DX-COM |
|---|---|---|---|
| Conv, DWConv, BN, ReLU와 SiLU, Add와 Concat, Pool | CNN 백본, BEV 인코더, 검출과 세그멘테이션 헤드 | 지원 [2] | 지원 (CNN이 주력) |
| Resize(nearest, bilinear), Upsample | FPN, 세그멘테이션 디코더 | 지원. grid_sample 대신 쓰라고 권하기도 합니다. [1] | 지원 |
| Softmax, 작은 MatMul | depth 분포, 어텐션 | 일부 지원 | 일부 지원. 트랜스포머 디코더는 지원하지 않는다는 언급이 있습니다. [5] |
| GridSample (grid_sample) | BEVFormer, Fast-BEV의 일부 구현, deformable attention | 지원하지 않고 지원 계획도 없습니다. [1] | 확인 필요. 지원하지 않는다고 보고 설계합니다. |
| Deformable attention | BEVFormer, 여러 DETR 계열 | 지원하지 않습니다. 커뮤니티에도 CPU로 넘겼다는 사례뿐입니다. [1] | 확인 필요. 지원하지 않는다고 보고 설계합니다. |
| GatherElements, ScatterND, 인덱스가 바뀌는 인덱싱 | BEVPool, voxel pooling, LUT 투영 | 지원하지 않습니다. grid_sample을 gather로 풀어 써도 파서 오류가 납니다. [6] | 확인 필요. CPU에서 처리한다고 보고 설계합니다. |
| 5차원 텐서의 Slice, Reshape, Transpose | 3D voxel 처리 | 4번 축 slice처럼 지원하지 않는 경우가 있습니다. [6] | 확인 필요 |
| 3D Conv | 일부 occupancy 헤드 | 일부 지원 | 확인 필요 |
| Channel-to-Height (Reshape) | FlashOcc 출력 | 마지막 출력의 reshape이므로 CPU에서 처리하면 상관없습니다. | Hailo와 같습니다. |

두 NPU 모두 BEV 공간으로 옮기는 단계(view transform)를 실행할 수 없다고 보고 설계해야 합니다. 이 단계는 연산량이 작아서 CPU로 충분합니다. 예를 들어 카메라 4대, 특징맵 16x44, depth bin 59개, 채널 64개인 LSS 방식 BEVPool은 4x59x16x44x64, 약 1,060만 번의 곱셈과 덧셈입니다. 라즈베리파이 5의 Cortex-A76 4코어에서 NumPy나 C로 수 ms면 될 것으로 보이며 측정이 필요합니다. BEVPIPE 논문[7]도 인텔 내장 GPU라는 다른 하드웨어에서 같은 결론을 냈습니다. dense 연산은 가속기에 두고 scatter 계열의 기하 연산은 가속기 밖에서 돌리는 분할 실행입니다.

DEEPX의 지원 여부는 DX-COM에 아래 세 가지 ONNX를 넣어 컴파일해 보고 결과를 기록해 확인합니다.

1. ResNet18에 FPN을 붙인 백본 (성공 예상)
2. FlashOcc에서 BEV 인코더와 occupancy 헤드만 잘라낸 ONNX (성공 예상)
3. grid_sample 하나와 ScatterND 하나만 넣은 최소 모델 (실패하거나 CPU로 넘어갈 것으로 예상)

Hailo도 같은 세 파일을 `hailo parser onnx --hw-arch hailo8 model.onnx`로 확인해 위 표를 실측 결과로 고칩니다.

## 4. 모델 방식 비교

### 방식 A-1: Fast-BEV

카메라별 2D 백본을 거친 이미지 특징을 미리 계산해 둔 LUT로 3D voxel에 투영합니다. depth는 추정하지 않고 균일하다고 가정합니다. 그다음 voxel을 BEV로 접어 2D conv를 거치고 3D 검출 헤드로 보냅니다.

depth 네트워크가 없어 가볍고, 투영이 고정된 LUT여서 카메라 보정만 고정되면 결과가 일정합니다. 백본과 BEV 인코더, 헤드는 NPU에서 돌릴 수 있습니다. LUT 투영은 gather 연산이어서 CPU로 빼야 하는데, 인덱스가 고정된 gather는 CPU에서 매우 빠릅니다.

출력이 3D 박스 위주여서 리얼리티 뷰의 주행 가능 영역이나 정적 장애물을 표현하기에는 부족합니다. depth를 쓰지 않아 먼 거리의 정확도도 낮습니다.

### 방식 A-2: FlashOcc (BEVDet 계열에 Channel-to-Height를 더한 구조)

카메라별 2D 백본을 거친 뒤 depth 분포를 예측하고(LSS), BEVPool로 BEV 특징을 만듭니다. 이어서 2D conv BEV 인코더를 거치고, 채널을 높이 방향으로 reshape해 3D occupancy를 출력합니다. Occ3D 기준으로 200x200x16, 17개 클래스입니다. 3D conv 없이 2D conv만 쓰는 것이 핵심입니다. [8]

3D conv와 트랜스포머가 없어서 NPU에 올리기 쉬운 occupancy 모델 가운데 가장 단순합니다. 출력이 occupancy이므로 리얼리티 뷰처럼 차량뿐 아니라 연석, 벽, 주행 가능 영역까지 그릴 수 있습니다. 논문에 따르면 Occ3D-nuScenes에서 R50 기준 mIoU가 약 31~32입니다. [8]

실행은 네 부분으로 나뉩니다. 이미지 백본과 depth 헤드는 NPU, BEVPool은 CPU, BEV 인코더와 occupancy 헤드는 NPU, 마지막 channel-to-height reshape은 CPU입니다. NPU 모델 2개와 CPU 단계 1개를 번갈아 실행하게 됩니다.

nuScenes는 카메라가 6대이고 여기서는 2~4대를 쓰므로 반드시 다시 학습해야 합니다. 라즈베리파이 5에서 실시간에 가깝게 돌리려면 백본을 ResNet18이나 EfficientNet-lite 수준으로 낮추고 입력을 256x704 이하로 줄여야 합니다.

### 방식 B: 카메라별 검출과 세그멘테이션 뒤 BEV 합성

카메라마다 YOLO 계열로 차량, 보행자, 이륜차를 검출하고 도로와 차선을 세그멘테이션합니다. 그 결과를 지면이 평평하다는 가정과 카메라 보정값으로 IPM(역원근 변환)해 위에서 본 그림 한 장에 합칩니다. 객체는 박스 아래쪽이 땅에 닿는 점을 BEV 좌표로 바꿔 3D 아이콘으로 표시합니다.

두 NPU 모두 Model Zoo에 바로 쓸 수 있는 모델이 있고 연산자 문제가 없습니다. 학습 없이 곧바로 데모를 만들 수 있습니다. 지도 위에 차량 아이콘, 차선, 주행 영역을 그리는 화면 구성도 리얼리티 뷰와 거의 같습니다.

지면이 평평하다고 가정하기 때문에 경사로, 먼 거리, 가려진 객체에서는 거리 오차가 큽니다. 카메라 경계에서 같은 객체가 두 번 잡힐 수 있어서 BEV 위에서 NMS나 추적이 필요합니다. 높이나 정적 장애물의 모양 같은 3D 정보는 얻지 못합니다.

Depth Anything 소형 모델 같은 단안 depth 모델로 거리를 보정할 수 있지만 NPU 부하가 커지므로 2단계 이후에 검토합니다.

### 비교표

| 기준 | Fast-BEV | FlashOcc | 방식 B |
|---|---|---|---|
| NPU 연산자 호환 | 중간(투영을 CPU로 분리) | 중간(BEVPool을 CPU로 분리) | 높음 |
| 재학습 | 필요 | 필요 | 필요 없음(미세조정은 선택) |
| 출력 정보 | 3D 박스 | 3D occupancy | 2D 검출과 세그멘테이션을 평면에 투영 |
| 리얼리티 뷰와의 유사도 | 중간 | 높음 | 중간보다 높음(보기에는 충분) |
| 예상 FPS (라즈베리파이 5와 Hailo-8, 카메라 4대, 추정) | 5~10 | 3~8 | 10~20 |
| 구현 난이도 | 중간보다 높음 | 높음 | 낮음 |
| 데이터 | nuScenes 재가공 | nuScenes와 Occ3D 재가공 | COCO, BDD100K 사전학습 모델 그대로 |

FPS는 공개된 Model Zoo의 단일 모델 성능과 위 연산량으로 추정한 값입니다. 1단계가 끝나면 측정해서 고칩니다.

## 5. 추천안과 단계별 계획

### 1단계 (2~4주): 방식 B로 동작하는 리얼리티 뷰 데모

하드웨어는 라즈베리파이 5 8GB, Raspberry Pi AI HAT+(Hailo-8, 26 TOPS), 광각 카메라 2대(전방과 후방), 27W 전원, 액티브 쿨러입니다.

모델은 YOLOv8s-seg를 쓰거나, YOLOv8n에 가벼운 도로 세그멘테이션 모델을 더합니다. Model Zoo가 제공하는 HEF를 그대로 씁니다.

카메라 영상을 NPU에서 묶어 추론하고, CPU에서 IPM 합성과 객체의 BEV 좌표 변환, SORT 계열의 간단한 추적을 합니다. 화면은 Open3D, three.js, PyGame 가운데 하나로 위에서 본 그림이나 3D로 그립니다.

성공 기준은 카메라 2대에서 10FPS 이상, 5m 안의 차량 위치 오차 0.5m 이하입니다.

### 2단계 (4~8주): FlashOcc 경량화와 분할 실행

nuScenes의 카메라 6대 중 4대만 쓰도록 데이터 로더를 바꿉니다. CAM_FRONT, CAM_BACK, CAM_FRONT_LEFT, CAM_FRONT_RIGHT를 쓰거나 측면 카메라로 바꿉니다. 백본은 ResNet18, 입력은 256x704, BEV는 100x100으로 줄여 다시 학습합니다.

ONNX는 세 부분으로 내보냅니다. 첫째는 백본과 depth 헤드, 둘째는 BEV 인코더와 occupancy 헤드이고, 셋째인 BEVPool은 C나 NumPy로 구현합니다. INT8 양자화에 쓸 보정 데이터는 실제로 장착한 카메라의 영상으로 모읍니다.

이 단계에서 Hailo-8과 DX-M1에 같은 ONNX를 컴파일해 정확도(mIoU가 얼마나 떨어지는지), FPS, 전력을 비교합니다. DX-M1은 DRAM이 있어 두 모델을 함께 올리기 좋으므로 2단계에서는 결과가 뒤집힐 수 있습니다.

### 3단계 (선택): 자체 카메라 배치에 맞춘 데이터 수집과 미세조정

카메라 높이와 화각이 nuScenes와 다르면 성능이 크게 떨어집니다. 실제 차량이나 로봇에서 영상을 모으고, 1단계 방식 B의 결과를 임시 정답(의사 라벨)으로 삼아 미세조정하는 방법을 검토합니다.

## 6. 확인이 필요한 항목

- DX-COM 최신 버전의 연산자 지원 목록. 3장의 최소 모델 테스트로 확인합니다.
- Hailo DFC 최신 버전에서 GatherElements, ScatterND 지원이 달라졌는지
- 쿼드 카메라 어댑터의 동기화 정확도와 프레임레이트
- 라즈베리파이 5 CPU의 BEVPool과 IPM 처리 시간
- DX-M1 모듈별 메모리 차이. 출처에 따라 LPDDR5 4GB와 LPDDR4X 1GB로 다릅니다.

## 참고 자료

1. Hailo Community, "Support for GridSample operation": https://community.hailo.ai/t/support-for-gridsample-operation/13957
2. Hailo Community, "Supported operators" (2024, Hailo-8L 기준): https://community.hailo.ai/t/supported-operators/5046
3. Radxa AICore DX-M1/DX-M1M 문서: https://docs.radxa.com/en/aicore/dx-m1
4. DFRobot DX-M1 AI Accelerator (DFR1252): https://wiki.dfrobot.com/dfr1252/
5. CNX Software, "Sixfab AI HAT+ for Raspberry Pi 5 integrates DEEPX DX-M1": https://www.cnx-software.com/2026/06/01/sixfab-ai-hat-for-raspberry-pi-5-integrates-deepx-dx-m1-ai-accelerator/
6. Hailo Community, "grid_sample problem": https://community.hailo.ai/t/grid-sample-problem/17078
7. BEVPIPE, "The Operator Mismatch Problem: Deploying BEV Perception with Portable GPU Compute": https://arxiv.org/pdf/2610.11504
8. FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin: https://arxiv.org/abs/2311.12058
