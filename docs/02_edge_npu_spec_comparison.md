# 엣지 NPU 사양 비교: Hailo-8, 8L, 10H와 DEEPX DX-M1

이 문서는 라즈베리파이 5에 붙일 수 있는 NPU의 사양을 모아, 테슬라 AI4에서 리얼리티 뷰가 쓰는 연산량과 비교한 자료입니다.

테슬라 쪽 수치는 [FSD v15 리얼리티 뷰 기술 조사](01_fsd_v15_reality_view_research.md)에 있습니다. 연산자 지원 여부(grid_sample, scatter 등)와 BEV 모델 방식 비교, 단계별 계획은 [엣지 NPU 구성과 모델 선정](edge_npu_model_selection.md)에서 다루므로 여기서는 되풀이하지 않습니다. 이 문서는 Hailo-10H(AI HAT+ 2)를 더하고, 리얼리티 뷰 v15처럼 영상을 생성하는 방식에 각 NPU가 맞는지를 봅니다.

수치마다 출처 번호 `[N번호]`와 구분을 적었습니다. "공식"은 제조사나 라즈베리파이가 밝힌 값, "3자"는 리뷰어와 판매사와 유통사의 값, "추정"은 이 문서에서 계산하거나 짐작한 값입니다.

## 1. 요약

1단계와 2단계(검출, 세그멘테이션, BEV, 점유)에는 Hailo-8(AI HAT+ 26 TOPS)이 무난합니다. 라즈베리파이 공식 지원과 제공 모델이 가장 잘 갖춰져 있습니다. 모델 선정 문서의 결론과 같습니다.

3단계(리얼리티 뷰 방식의 생성 렌더러)를 시도한다면 Hailo-10H(AI HAT+ 2)가 맞습니다. 메모리 8GB를 보드에 갖췄고, 라즈베리파이 공식 HAT 가운데 LLM과 VLM(약 1B~7B)을 지원하는 제품은 이것뿐입니다. 확산 모델 지원은 확인 필요입니다.

DEEPX DX-M1은 INT8 기준 25 TOPS에 LPDDR5 4GB가 있어 큰 CNN 백본이나 여러 모델을 함께 올리기에 좋습니다. 트랜스포머와 생성 모델 지원은 확인 필요입니다.

테슬라의 리얼리티 뷰는 AI4 한 칩 NPU의 약 6분의 1만 씁니다. TOPS만 보면 엣지 NPU와 자릿수가 같을 수 있지만 메모리 대역폭, 입력 영상의 양, 모델 크기(약 1B)에서 차이가 큽니다. 3장에 정리했습니다.

## 2. 사양 비교

| 항목 | Hailo-8L | Hailo-8 | Hailo-10H | DX-M1 |
|---|---|---|---|---|
| 제품 | AI HAT+ 13T | AI HAT+ 26T | AI HAT+ 2 | M.2 모듈 |
| INT8 TOPS | 13 | 26 | 20 | 25 |
| 보드 메모리 | 없음 | 없음 | 8GB | 4GB |
| 전력 | 최대 6.6W | 최대 8.25W | 일반 2.5W | 2~5W |
| 가격 | 약 70달러 | 약 110달러 | 200달러 | 85~139달러 |
| 양자화 | INT8 | INT8 | INT4, INT8 | INT8 |
| LLM, VLM | 미지원 | 미지원 | 지원 | 확인 필요 |
| 확산 모델 | 해당 없음 | 해당 없음 | 확인 필요 | 확인 필요 |

제품별 설명과 출처는 아래와 같습니다.

### 2.1 Hailo-8L과 Hailo-8

- 연산 성능: Hailo-8L은 13 TOPS, Hailo-8은 26 TOPS이며 모두 INT8 기준입니다. (공식 [N1])
- 메모리: 보드에 메모리가 없어 라즈베리파이의 메모리를 씁니다. (공식 [N1])
- 연결: PCIe(FPC 리본 케이블)이며 라즈베리파이 쪽은 1레인입니다.
- 전력: 모듈 기준 Hailo-8L 최대 6.6W, Hailo-8 최대 8.25W (3자 [N5])
- 가격: Hailo-8L 약 70달러, Hailo-8 약 110달러 (3자 [N5])
- 지원 모델: 검출, 세그멘테이션, 자세 추정 같은 CNN (공식 [N1])
- 개발 도구: Dataflow Compiler로 HEF 파일을 만들고 HailoRT로 실행합니다. (공식 [N3], 3자 [N5])
- 카메라 소프트웨어: rpicam-apps, Picamera2와 연동됩니다. (공식 [N1])
- 성능 측정 예(배치 8): Hailo-8L은 YOLOv8m 87 FPS, YOLOv11m 58 FPS. Hailo-8은 YOLOv8m 140 FPS, YOLOv11m 101 FPS (3자 [N5])
- 주의할 점: 크거나 여러 개인 모델은 전환(context switch) 때문에 처리량이 떨어집니다.

### 2.2 Hailo-10H

- 연산 성능: 40 TOPS INT4 (공식 [N2][N3]), 20 TOPS INT8 (3자 [N5])
- 메모리: 8GB (공식 [N2]). 종류는 공식 표기가 없고, 모듈 기준으로는 LPDDR4 또는 4X입니다. (3자 [N5])
- 연결: PCIe Gen3 1레인, FPC (3자 [N4])
- 전력: 모듈 일반 2.5W (3자 [N5])
- 가격: 200달러 (공식 [N2])
- 지원 모델: CNN과 LLM, VLM. 출시 때 모델은 DeepSeek-R1-Distill 1.5B, Llama 3.2 1B, Qwen2.5 1.5B 등입니다. 약 1B~7B까지 (공식 [N3]), 문서에는 약 6B까지로 적혀 있습니다. (공식 [N1])
- 개발 도구: Hailo-8과 같은 도구에 hailo-ollama와 LoRA 어댑터 컴파일이 더해집니다. (공식 [N3])
- 카메라 소프트웨어: rpicam-apps, Picamera2와 연동됩니다. (공식 [N3])
- 성능 측정 예: 영상 인식은 26 TOPS 제품과 대체로 같은 수준입니다. (공식 [N3]) qwen2 1.5B 질의 응답은 13.6초이고 라즈베리파이 CPU는 22.9초입니다. (3자 [N4])
- 주의할 점: 출시 초기 소프트웨어가 덜 다듬어져 리뷰어가 "HailoRT not ready" 오류를 겪었습니다. (3자 [N4]) Hailo-8용 HEF는 쓸 수 없어 다시 컴파일해야 합니다. (공식 [N3])

### 2.3 DEEPX DX-M1

- 연산 성능: 25 TOPS INT8 (3자 [N6][N7][N8])
- 메모리: LPDDR5 4GB (3자 [N6][N7]). Radxa DX-M1M 2242는 LPDDR4X 1GB로 소개되어 출처마다 다릅니다. (3자 [N8])
- 연결: M.2 HAT+를 거칩니다. 모듈은 Gen3 4레인이고 1레인으로도 동작합니다. (3자 [N6][N8])
- 전력: 2~5W (3자 [N6][N9]), 일반 3W (3자 [N8])
- 가격: DFRobot 139달러 (3자 [N9]), Radxa DX-M1M 85~98달러 (3자 [N8])
- 지원 모델: 검출, 분류, 세그멘테이션, 자세 추정, 노이즈 제거 같은 CNN (3자 [N8])
- 개발 도구: DXNN SDK. DX-COM(컴파일), DX-RT(실행), DX-STREAM(GStreamer) (3자 [N8])
- 양자화: INT8. 제조사는 "IQ8" 기술로 FP32 수준의 정확도를 낸다고 주장합니다. (3자 [N7])
- 카메라 소프트웨어: 연동이 없어 직접 구성해야 합니다. (추정)
- 성능 측정 예: 공개된 FPS는 확인 필요입니다.
- 주의할 점: 모듈마다 메모리와 크기가 다릅니다. 연산자 지원 목록은 확인 필요입니다.

### 2.4 해석

Hailo-10H의 40 TOPS는 INT4 기준입니다. INT8로 맞추면 약 20 TOPS여서 Hailo-8(26 TOPS)보다 낮습니다. 영상 인식 성능이 26 TOPS 제품과 비슷하다는 공식 설명과도 맞습니다. Hailo-10H를 고를 이유는 TOPS보다 8GB 메모리에 있습니다. 파라미터가 수억에서 수십억 개인 모델을 NPU 쪽에 올려 둘 수 있어서 생성형 렌더링을 실험하기에 좋습니다.

라즈베리파이 5의 PCIe는 1레인이어서 Gen3 기준 한 방향 약 1GB/s입니다(PCIe 규격으로 계산한 추정). 카메라 영상을 보내기에는 충분합니다. 다만 메모리가 없는 Hailo-8에서 큰 모델의 가중치를 계속 주고받으면 여기서 막힙니다.

## 3. 테슬라 AI4의 리얼리티 뷰와 비교

| 항목 | 테슬라 AI4 | Hailo-10H | Hailo-8 | DX-M1 |
|---|---|---|---|---|
| 렌더러의 NPU 몫 | 한 칩의 1/6 | 전체 | 전체 | 전체 |
| 그 몫의 TOPS | 약 8~40 | 20 | 26 | 25 |
| 메모리 | 칩마다 16GB | 8GB | 없음 | 4GB |
| 1B 모델 적재 | 가능 | 가능 | 어려움 | 가능 |
| 카메라 | 7~8대 | 2~4대 | 2~4대 | 2~4대 |

엣지 쪽 세 열은 라즈베리파이 5에 해당 NPU를 붙인 구성입니다. 항목별 설명은 다음과 같습니다.

- 렌더러의 NPU 몫: 테슬라는 한 칩의 TRIP 3개 중 0.5개를 씁니다. (공식, 머스크)
- TOPS: AI4의 공식 TOPS가 없습니다. 매체 추정인 칩당 50~250을 넣으면 약 8~40 TOPS입니다. (추정) Hailo-10H는 INT8 20, INT4 40이고 나머지는 INT8 기준입니다.
- 렌더러 모델 크기: 테슬라는 약 1B 파라미터입니다. (추정, 보도) Hailo-10H는 8GB이므로 INT4 1B(약 0.5GB)를 올릴 수 있습니다. Hailo-8은 보드 메모리가 없어 어렵습니다. DX-M1은 4GB이므로 INT8 1B(약 1GB)를 올릴 수 있지만 연산자 지원이 문제입니다. (추정)
- 메모리: AI4는 GDDR6입니다. (3자) Hailo-10H의 메모리 종류는 확인 필요이고 DX-M1은 LPDDR5입니다.
- 카메라 입력: 테슬라는 약 5MP 카메라 7~8대입니다. (3자) 라즈베리파이 5는 CSI 포트가 2개여서 4대를 쓰려면 어댑터가 필요합니다.
- 출력: 테슬라는 8프레임 클립을 36fps로, 화면 해상도로 만듭니다. Hailo-10H에서의 목표는 낮은 해상도(256~512픽셀)에서 초당 몇 프레임입니다. (추정) Hailo-8은 1단계와 2단계에 쓰고, DX-M1은 확인 필요입니다.

TOPS의 자릿수만 보면 엣지 NPU 하나가 리얼리티 뷰 렌더러의 몫과 크게 다르지 않을 수 있습니다. 그러나 AI4의 TOPS가 공식 수치가 아니어서 이 비교는 정확하지 않습니다.

실제 차이는 메모리 대역폭과, 트랜스포머와 확산 모델에 필요한 연산자의 지원에서 납니다. 테슬라는 자체 컴파일러로 diffusion transformer를 TRIP에 맞췄습니다. Hailo와 DEEPX는 확산 모델 지원을 밝히지 않았습니다. 그래서 3단계의 목표는 1B 렌더러를 그대로 만드는 것이 아니고, 파라미터가 수천만에서 수억 개인 작은 조건부 생성 모델을 낮은 해상도로 돌리는 것으로 잡는 편이 현실적입니다. 이 문단은 이 문서의 판단입니다.

## 4. 단계별 추천 구성

모델 선정 문서의 세 단계에 하드웨어를 맞추면 다음과 같습니다.

| 단계 | 내용 | 추천 NPU |
|---|---|---|
| 1단계 | 검출 뒤 BEV 합성 | Hailo-8 |
| 2단계 | FlashOcc 분할 실행 | Hailo-8, DX-M1 비교 |
| 3단계 | 작은 생성 렌더러 | Hailo-10H |

1단계는 카메라별로 검출과 세그멘테이션을 한 뒤 IPM으로 BEV에 합칩니다. Hailo-8(AI HAT+ 26T, 약 110달러)은 제공되는 HEF 모델을 바로 쓸 수 있고 rpicam-apps 연동을 공식 지원합니다.

2단계는 FlashOcc를 나눠 실행해 점유를 구합니다. 두 모델을 함께 올릴 때는 DX-M1의 4GB 메모리가 유리할 수 있어 Hailo-8과 비교합니다.

3단계는 인코더 특징으로 작은 생성 렌더러를 돌려 가상 시점 영상을 만듭니다. Hailo-10H(AI HAT+ 2, 200달러)는 8GB 메모리가 있고 트랜스포머(LLM, VLM)를 돌린 실적이 있는 유일한 라즈베리파이 공식 HAT입니다. 확산 블록이 컴파일되지 않으면 렌더러를 CNN 기반의 적은 단계 생성기(가벼운 U-Net 등)로 바꿔 Hailo-8이나 DX-M1에서 시도할 수 있습니다.

## 5. 확인이 필요한 항목

- Hailo-10H의 Dataflow Compiler가 작은 DiT 블록(어텐션과 adaLN)과 가벼운 U-Net을 컴파일하는지
- Hailo-10H에서 INT4, INT8 생성 모델의 프레임당 지연 시간(ms)
- AI HAT+ 2의 8GB 메모리 종류와 대역폭
- DEEPX DX-COM의 트랜스포머 연산자 지원 여부(deepx.ai 공식 데이터시트)
- 라즈베리파이 5와 NPU를 함께 쓸 때의 소비 전력
- DX-M1 모듈별 메모리 차이(LPDDR5 4GB, LPDDR4X 1GB)

## 6. 출처 목록

- N1: [AI HATs, Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) (공식)
- N2: [Raspberry Pi AI HAT+ 2 제품 페이지](https://www.raspberrypi.com/products/ai-hat-plus-2/) (공식)
- N3: [Introducing the Raspberry Pi AI HAT+ 2, Raspberry Pi News](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/) (공식)
- N4: [Raspberry Pi AI HAT+ 2 targets generative AI, CNX Software](https://www.cnx-software.com/2026/01/15/raspberry-pi-ai-hat-2-targets-generative-ai-llm-vlm-with-hailo-10h-accelerator/), [Raspberry Pi AI HAT+ 2 review, Tom's Hardware](https://www.tomshardware.com/uk/raspberry-pi/raspberry-pi-ai-hat-plus-2-review) (리뷰)
- N5: [Hailo-8 vs Hailo-8L vs Hailo-10H, Electromaker](https://www.electromaker.io/blog/article/hailo-8-vs-hailo-8l-vs-hailo-10h-which-ai-accelerator-should-you-buy) (3자, Hailo 데이터시트 인용)
- N6: [DX-M1 AI Accelerator (DFR1252), DFRobot Wiki](https://wiki.dfrobot.com/dfr1252/) (판매사)
- N7: [Radxa AICore DX-M1/DX-M1M, Radxa Docs](https://docs.radxa.com/en/aicore/dx-m1) (판매사)
- N8: [Radxa AICore DX-M1M M.2 2242, CNX Software](https://www.cnx-software.com/2026/03/21/radxa-aicore-dx-m1m-m-2-2242-low-power-ai-module-delivers-25-tops-of-edge-ai-performance-for-just-3w-of-power/) (리뷰)
- N9: [DFRobot DX-M1 25 TOPS M.2 AI accelerator, Wccftech](https://wccftech.com/dfrobot-dx-m1-25-tops-m2-ai-accelerator-module-raspberry-pi-pc/), [DEEPX DX-M1, DigiKey](https://www.digikey.com/en/product-highlight/d/deepx/dx-m1-m2-ai-acceleration-module) (3자, 유통사)
