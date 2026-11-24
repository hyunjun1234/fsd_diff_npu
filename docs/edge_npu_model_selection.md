# 엣지 NPU 구성과 모델 선정: 라즈베리파이 5 + Hailo-8 / DEEPX DX-M1

> 목표: 테슬라 FSD의 "리얼리티 뷰"(주변 차량, 보행자, 차선, 주행 가능 영역을 탑뷰/3D로 보여주는 시각화)와 비슷한 화면을 라즈베리파이 5와 M.2 NPU, 카메라 2~4대로 구현한다.
> 작성일: 2026-11-24. 수치 중 출처를 표시하지 않은 항목은 일반적인 공개 사양 기준의 추정치이며, 실제 보드에서 측정해 갱신해야 한다.

## 1. 결론 요약

| 항목 | 추천 |
|---|---|
| 1단계 구현 방식 | **방식 B: 카메라별 검출+세그멘테이션 후 BEV 합성** (지면 평면 가정 + 사전 보정된 호모그래피) |
| 2단계 구현 방식 | **FlashOcc 계열 occupancy를 "분할 실행"으로** (NPU: 이미지 백본/BEV 인코더, CPU: view transform) |
| NPU | **Hailo-8 (26 TOPS, Raspberry Pi AI HAT+)을 기본**, DEEPX DX-M1은 2단계에서 비교 대상 |
| 카메라 | 1단계 2대(전방+후방), 2단계 4대(전/후/좌/우). 4대는 동기화된 멀티카메라 어댑터 사용 |
| 하지 말 것 | BEVFormer 등 deformable attention 기반 모델을 NPU에 통째로 올리는 시도 (grid_sample 미지원) |

이유는 한 줄로 이렇다. 두 NPU 모두 **CNN은 잘 돌리지만 BEV 모델의 핵심인 기하 변환 연산(grid_sample, scatter/BEVPool, 가변 인덱스 gather)은 사실상 지원하지 않는다.** 따라서 "CNN은 NPU, 기하 변환은 CPU"로 나누는 구조가 현실적인 유일한 길이고, 그 구조에 가장 자연스럽게 맞는 것이 방식 B와 FlashOcc(LSS 계열) 분할 실행이다.

## 2. 하드웨어 비교

### 2.1 NPU

| 항목 | Hailo-8 (AI HAT+ 26T) | Hailo-8L (AI HAT+ 13T / AI Kit) | DEEPX DX-M1 |
|---|---|---|---|
| 연산 성능 | 26 TOPS INT8 | 13 TOPS INT8 | 25 TOPS INT8 [3][4][5] |
| 전력 | 약 2.5W 수준 | 약 1.5W 수준 | 2~5W [4], HAT 기준 피크 2.5~3W [5] |
| 온보드 DRAM | 없음 (온칩 SRAM, 큰 모델은 context switch) | 없음 | 있음. 모듈별 상이: 4GB LPDDR5 [3][4] / Sixfab HAT+ 1GB LPDDR4X [5] |
| 모듈 인터페이스 | M.2 PCIe Gen3 x4 | 동일 | M.2 PCIe Gen3 x4, x1 호환 [3][4] |
| 파이5에서 실제 링크 | PCIe x1 (Gen2 기본, Gen3 강제 가능) | 동일 | PCIe Gen3 x1 [5] |
| 정밀도 | INT8 (일부 16bit 레이어) | 동일 | INT8 (IQ8) [3] |
| 컴파일러/런타임 | Dataflow Compiler(DFC) → HEF, HailoRT | 동일 | DX-COM → DXNN, DX-RT [5] |
| 라즈베리파이 생태계 | 공식 AI HAT+, rpicam-apps/hailo-apps 연동, 모델 주(Model Zoo) 다수 | 동일 | Sixfab AI HAT+ ($90, 25T) 등 서드파티 [5] |

**해석**

- Hailo-8은 외부 DRAM이 없어 여러 모델을 동시에 올리거나 큰 모델을 돌리면 context switch로 처리량이 떨어진다. 방식 B처럼 "작은 YOLO 하나를 카메라 수만큼 배치로" 돌리는 데는 문제가 없다.
- DX-M1은 DRAM이 있어 큰 백본이나 다중 모델에 유리하다. 다만 라즈베리파이용 생태계가 Hailo보다 새롭고, 연산자 지원 문서가 공개적으로 덜 정리되어 있어 직접 컴파일 테스트가 필요하다.
- 파이5의 PCIe는 x1이므로 대역폭은 Gen3 기준 약 1GB/s다. 4대 카메라 입력(256x704x3 INT8 x4 = 약 2.1MB/프레임)을 10FPS로 보내도 약 21MB/s라 병목이 아니다. 병목은 CPU 쪽 전처리와 BEV 합성이다.
- 전원: 파이5 + NPU HAT는 13~15W를 쓰므로 27W 공식 어댑터가 필요하다 [5].

### 2.2 카메라

- 파이5는 MIPI CSI 포트가 2개다. 2대 구성은 그대로 연결하면 된다.
- 4대는 (a) 동기화된 쿼드 카메라 어댑터(4개 센서를 하나의 CSI 프레임으로 합쳐 주는 방식, 하드웨어 동기화 지원 제품 권장), (b) CSI 2대 + USB 2대 중에 고른다. BEV 합성은 카메라 간 시각 동기가 중요하므로 (a)를 권장한다.
- 화각: 120도 이상 광각을 권장한다. 4대로 360도를 덮으려면 수평 FOV 90도 이상이 필요하다. 광각 렌즈는 왜곡이 크므로 보정(OpenCV fisheye calibration)이 필수다.
- 해상도: NPU 입력은 640x640(방식 B) 또는 256x704(BEV 계열, nuScenes 관례) 정도로 축소하므로 센서는 1~2MP면 충분하다.
- 외부 파라미터(카메라 위치, 자세)는 바닥에 체커보드/ArUco 마커를 깔고 한 번에 보정한다. 방식 B와 BEV 모델 모두 이 값이 품질을 좌우한다.

## 3. 컴파일러 연산자 지원 확인 결과

BEV/occupancy 모델에 필요한 연산자를 기준으로 정리했다.

| 연산자 / 블록 | 쓰이는 곳 | Hailo DFC | DEEPX DX-COM |
|---|---|---|---|
| Conv, DWConv, BN, ReLU/SiLU, Add/Concat, Pool | CNN 백본, BEV 인코더, 검출/세그 헤드 | 지원 [2] | 지원 (CNN 주력) |
| Resize (nearest/bilinear), Upsample | FPN, 세그 디코더 | 지원 (grid_sample 대체로도 권장됨 [1]) | 지원 |
| Softmax, MatMul (소규모) | depth 분포, 어텐션 | 제한적 지원 | 제한적 지원, 트랜스포머 디코더 미지원 언급 [5] |
| **GridSample (grid_sample)** | BEVFormer, Fast-BEV 일부 구현, deformable attention | **미지원, 지원 계획 없음** [1] | 공개 문서에서 지원 확인 불가 → 미지원으로 가정 |
| **Deformable attention** | BEVFormer, 다수 DETR 계열 | 미지원. 커뮤니티에서도 CPU로 오프로딩했다는 사례뿐 [1] | 미지원으로 가정 |
| **GatherElements, ScatterND, 가변 인덱싱** | BEVPool, voxel pooling, LUT 투영 | **미지원**. grid_sample을 gather로 풀어 써도 파서 에러 [6] | 확인 필요 (CPU 처리 전제로 설계) |
| 5D 텐서 Slice/Reshape/Transpose | 3D voxel 처리 | 축 4 slice 등 미지원 사례 [6] | 확인 필요 |
| 3D Conv | 일부 occupancy 헤드 | 제한적 | 확인 필요 |
| Channel-to-Height (Reshape) | FlashOcc 출력 | 최종 출력 reshape이므로 CPU에서 처리하면 무관 | 동일 |

정리하면, **두 NPU 모두 "BEV 공간으로 옮기는 단계(view transform)"를 실행할 수 없다고 보고 설계해야 한다.** 다행히 이 단계는 연산량이 작아 CPU로 충분하다. 예를 들어 카메라 4대, 특징맵 16x44, depth bin 59, 채널 64의 LSS 방식 BEVPool은 4x59x16x44x64 = 약 1,060만 회 곱셈-누산으로, 파이5의 Cortex-A76 4코어에서 NumPy/C로 수 ms 수준이다(측정 필요). BEVPIPE 논문[7]도 다른 하드웨어(인텔 iGPU)에서 같은 결론, 즉 dense 연산은 가속기에 두고 scatter 계열 기하 연산은 가속기 밖에서 실행하는 "분할 실행"을 택했다.

**DEEPX 확인 방법 (작업 항목)**: DX-COM에 아래 3개 ONNX를 넣어 컴파일 결과를 기록한다.
1. ResNet18 + FPN 백본 (기대: 성공)
2. FlashOcc의 BEV 인코더 + occupancy 헤드만 잘라낸 ONNX (기대: 성공)
3. grid_sample 1개와 ScatterND 1개만 있는 최소 모델 (기대: 실패 또는 CPU fallback)
Hailo도 같은 3개를 `hailo parser onnx --hw-arch hailo8 model.onnx` 로 확인해 표를 실측값으로 갱신한다.

## 4. 모델 방식 비교

### 방식 A-1: Fast-BEV

- 구조: 카메라별 2D 백본 → 미리 계산한 LUT로 이미지 특징을 3D voxel에 투영(depth 추정 없음, 균일 분포 가정) → voxel을 BEV로 접고 2D conv → 3D 검출 헤드.
- 장점: depth 네트워크가 없어 가볍고, 투영이 고정 LUT라 카메라 보정만 고정되면 결정적이다.
- NPU 적합성: 백본과 BEV 인코더/헤드는 NPU 가능. **LUT 투영은 가변 gather라 CPU로 빼야 한다.** 고정 인덱스 gather는 CPU에서 매우 빠르다.
- 단점: 출력이 3D 박스 중심이라 리얼리티 뷰의 "주행 가능 영역/정적 장애물" 표현이 약하다. depth를 쓰지 않아 원거리 정확도가 낮다.

### 방식 A-2: FlashOcc (BEVDet 계열 + Channel-to-Height)

- 구조: 카메라별 2D 백본 → depth 분포 예측(LSS) → BEVPool로 BEV 특징 생성 → 2D conv BEV 인코더 → 채널을 높이 방향으로 reshape해 3D occupancy(Occ3D 기준 200x200x16, 17클래스) 출력. 3D conv 없이 2D conv만 쓰는 것이 핵심이다[8].
- 장점: 3D conv와 트랜스포머가 없어 **NPU 친화적인 occupancy 모델 중 가장 단순**하다. 출력이 occupancy라 리얼리티 뷰처럼 차량뿐 아니라 연석, 벽, 주행 가능 영역까지 그릴 수 있다. Occ3D-nuScenes에서 mIoU 약 31~32 수준(논문, R50 기준)[8].
- NPU 적합성: 이미지 백본+depth 헤드(NPU) → BEVPool(CPU) → BEV 인코더+occ 헤드(NPU) → channel-to-height reshape(CPU). **NPU 모델 2개 + CPU 단계 1개의 분할 실행**이 된다.
- 단점: nuScenes는 카메라 6대, 우리 구성은 2~4대라 **재학습이 필수**다. 백본을 ResNet18/EfficientNet-lite 급으로 낮추고 입력을 256x704 이하로 줄여야 파이5에서 실시간에 가깝다.

### 방식 B: 카메라별 검출+세그멘테이션 후 BEV 합성

- 구조: 카메라마다 YOLO 계열 검출(차량/보행자/이륜차) + 세그멘테이션(도로, 차선) → 지면 평면 가정 + 카메라 보정값으로 IPM(역원근 변환) → 탑뷰 한 장에 합성 → 객체는 박스 하단 접지점을 BEV 좌표로 변환해 3D 아이콘으로 표시.
- 장점: **두 NPU 모두 모델 주에 바로 쓸 수 있는 모델이 있고 연산자 문제가 없다.** 학습 없이 당장 데모 가능. 시각화 구조(지도 위에 차량 아이콘, 차선, 주행 영역)는 리얼리티 뷰와 거의 같다.
- 단점: 평평한 지면 가정이라 경사로, 원거리, 가려진 객체에서 거리 오차가 크다. 카메라 경계에서 같은 객체가 중복될 수 있어 BEV 상에서 NMS/트래킹이 필요하다. 진짜 3D 이해(높이, 정적 장애물 형태)는 없다.
- 보완: 단안 depth 모델(예: Depth Anything 소형)로 거리 보정 가능하나 NPU 부하가 커지므로 2단계 이후 검토.

### 비교표

| 기준 | Fast-BEV | FlashOcc | 방식 B (검출+세그 → BEV 합성) |
|---|---|---|---|
| NPU 연산자 호환 | 중 (투영 CPU 분리) | 중 (BEVPool CPU 분리) | **상** |
| 재학습 필요 | 필수 | 필수 | 불필요 (미세조정 선택) |
| 출력 풍부도 | 3D 박스 | **3D occupancy** | 2D 검출+세그를 평면 투영 |
| 리얼리티 뷰 유사도 | 중 | **상** | 중상 (시각적으로는 충분) |
| 예상 FPS (파이5 + Hailo-8, 4캠, 추정) | 5~10 | 3~8 | **10~20** |
| 구현 난이도 | 중상 | 상 | **하** |
| 데이터 | nuScenes 재가공 | nuScenes + Occ3D 재가공 | COCO/BDD100K 사전학습 그대로 |

FPS는 공개 모델 주의 단일 모델 성능과 위 연산량으로 추정한 값이며, 1단계 완료 후 실측해 갱신한다.

## 5. 추천안과 단계별 계획

### 1단계 (2~4주): 방식 B로 동작하는 리얼리티 뷰 데모

- 하드웨어: 파이5 8GB + Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS) + 카메라 2대(전/후, 광각) + 27W 전원 + 액티브 쿨러.
- 모델: YOLOv8s-seg 또는 YOLOv8n + 경량 도로 세그(모델 주 제공 HEF 사용).
- 파이프라인: 카메라 → NPU 배치 추론 → CPU에서 IPM 합성, 객체 BEV 좌표 변환, 간단한 트래킹(SORT 계열) → Open3D/three.js/PyGame으로 탑뷰 또는 3D 렌더링.
- 성공 기준: 2캠 10FPS 이상, 5m 이내 차량 위치 오차 0.5m 이하.

### 2단계 (4~8주): FlashOcc 경량화 + 분할 실행

- nuScenes 6캠 중 4대(CAM_FRONT, CAM_BACK, CAM_FRONT_LEFT/RIGHT 또는 측면 재배치)만 쓰도록 데이터 로더를 바꾸고, 백본 ResNet18, 입력 256x704, BEV 100x100으로 줄여 재학습.
- ONNX를 세 부분으로 export: (1) 백본+depth 헤드, (2) BEV 인코더+occ 헤드, (3) BEVPool은 C/NumPy 구현.
- INT8 양자화 보정 데이터는 실제 장착 카메라 영상으로 수집.
- 이 단계에서 Hailo-8과 DX-M1을 같은 ONNX로 컴파일해 정확도(mIoU 감소폭), FPS, 전력을 비교한다. DX-M1은 DRAM이 있어 두 모델을 동시에 올리기 유리하므로 2단계에서 역전될 가능성이 있다.

### 3단계 (선택): 자체 카메라 배치에 맞춘 데이터 수집과 미세조정

- nuScenes와 카메라 높이, 화각이 다르면 성능이 크게 떨어진다. 실제 차량/로봇에서 영상을 모아 1단계 방식 B 결과를 의사 라벨로 써서 미세조정하는 방법을 검토한다.

## 6. 남은 확인 항목

- [ ] DX-COM 최신 버전의 연산자 지원 목록 확보 (공개 문서에서 찾지 못함, 3장 최소 모델 테스트로 대체)
- [ ] Hailo DFC 최신 버전에서 GatherElements, ScatterND 지원 변화 여부
- [ ] 쿼드 카메라 어댑터 동기화 정확도와 프레임레이트 실측
- [ ] 파이5 CPU에서 BEVPool/IPM 처리 시간 실측
- [ ] DX-M1 모듈별 메모리 사양 차이 확인 (출처마다 4GB LPDDR5 / 1GB LPDDR4X로 다름)

## 참고 자료

1. Hailo Community, "Support for GridSample operation": https://community.hailo.ai/t/support-for-gridsample-operation/13957
2. Hailo Community, "Supported operators" (2024, Hailo-8L 기준): https://community.hailo.ai/t/supported-operators/5046
3. Radxa AICore DX-M1/DX-M1M 문서: https://docs.radxa.com/en/aicore/dx-m1
4. DFRobot DX-M1 AI Accelerator (DFR1252): https://wiki.dfrobot.com/dfr1252/
5. CNX Software, "Sixfab AI HAT+ for Raspberry Pi 5 integrates DEEPX DX-M1": https://www.cnx-software.com/2026/06/01/sixfab-ai-hat-for-raspberry-pi-5-integrates-deepx-dx-m1-ai-accelerator/
6. Hailo Community, "grid_sample problem": https://community.hailo.ai/t/grid-sample-problem/17078
7. BEVPIPE, "The Operator Mismatch Problem: Deploying BEV Perception with Portable GPU Compute": https://arxiv.org/pdf/2610.11504
8. FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin: https://arxiv.org/abs/2311.12058
