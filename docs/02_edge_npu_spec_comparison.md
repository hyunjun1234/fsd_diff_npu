# 엣지 NPU 사양 비교: Hailo-8 / 8L / 10H, DEEPX DX-M1 vs 테슬라 AI4

- 작성일: 2026-11-24
- 목적: 라즈베리파이 5에 붙일 수 있는 NPU의 공식 사양을 한곳에 모으고, 테슬라 AI4에서 리얼리티 뷰가 쓰는 연산 예산과 비교합니다.
- 함께 볼 문서:
  - [01. FSD v15 리얼리티 뷰 기술 조사](01_fsd_v15_reality_view_research.md): 테슬라 쪽 수치
  - [엣지 NPU 구성과 모델 선정](edge_npu_model_selection.md): 연산자 지원(grid_sample, scatter 등), BEV 모델 방식 비교, 단계별 계획

> 이 문서는 위 모델 선정 문서가 다루지 않은 **Hailo-10H(AI HAT+ 2)**를 추가하고, **생성형 렌더링(리얼리티 뷰 v15 방식)** 관점에서 각 NPU를 평가합니다. 연산자 지원과 BEV 방식 비교는 중복하지 않습니다.
>
> 신뢰도 표시: 🟢 제조사·라즈베리파이 공식, 🟡 리뷰어·유통사·3자, 🔴 본 문서의 계산·추정. 출처 번호 `[N#]`는 맨 아래에 있습니다.

---

## 1. 요약

| 질문 | 답 |
|---|---|
| 1~2단계(검출·세그·BEV/occupancy)에 가장 무난한 NPU | **Hailo-8 (AI HAT+ 26 TOPS)**. 공식 생태계와 모델 주가 가장 성숙합니다. 기존 모델 선정 문서의 결론과 같습니다. |
| 3단계(리얼리티 뷰식 생성 렌더러)를 시도할 만한 NPU | **Hailo-10H (AI HAT+ 2)**. 8GB 온보드 메모리가 있고, 공식적으로 LLM·VLM(약 1~7B)을 지원하는 유일한 파이 공식 HAT입니다. 다만 확산(diffusion) 모델 지원은 공식 확인되지 않았습니다. |
| DEEPX DX-M1의 위치 | 25 TOPS INT8에 4GB LPDDR5가 있어 큰 CNN 백본이나 여러 모델을 동시에 올리기 유리합니다. 트랜스포머·생성 모델 지원은 공개 자료로 확인되지 않습니다. |
| 테슬라와의 격차 | 리얼리티 뷰는 AI4 한 칩 NPU의 약 1/6만 씁니다. TOPS만 보면 엣지 NPU와 같은 자릿수일 수 있지만, 메모리 대역폭·입력 대역폭·모델 규모(약 1B) 차이가 큽니다. 3장 참고. |

---

## 2. 사양 비교표

| 항목 | Hailo-8L (AI HAT+ 13T) | Hailo-8 (AI HAT+ 26T) | Hailo-10H (AI HAT+ 2) | DEEPX DX-M1 (M.2 모듈) |
|---|---|---|---|---|
| 연산 성능 | 13 TOPS INT8 🟢 [N1] | 26 TOPS INT8 🟢 [N1] | **40 TOPS INT4** 🟢 [N2][N3] / 20 TOPS INT8 🟡 [N5] | 25 TOPS INT8 🟢/🟡 [N6][N7][N8] |
| 온보드 메모리 | 없음(파이 메모리 사용) 🟢 [N1] | 없음 🟢 [N1] | **8GB** 🟢 [N2]. 종류는 공식 미표기, 모듈 기준 LPDDR4/4X 🟡 [N5] | 4GB LPDDR5 🟡 [N6][N7]. 1GB LPDDR4X 변형(Radxa DX-M1M 2242)도 있어 출처 충돌 🟡 [N8] |
| 파이5 연결 | PCIe(FPC 리본), 파이5 쪽은 x1 | 동일 | PCIe Gen3 x1 FPC 🟡 [N4] | M.2 HAT+ 경유, 모듈은 Gen3 x4이며 x1 호환 🟡 [N6][N8] |
| 전력 | 모듈 최대 6.6W 🟡 [N5] | 모듈 최대 8.25W 🟡 [N5] | 모듈 2.5W 전형 🟡 [N5] | 2~5W 🟡 [N6][N9], 3W 전형 🟡 [N8] |
| 가격 | 약 $70 🟡 [N5] | 약 $110 🟡 [N5] | **$200**(현재 공식가) 🟢 [N2]. 출시 때는 $130 🟢 [N3] | $139(DFRobot) 🟡 [N9], $85~98(Radxa DX-M1M) 🟡 [N8] |
| 공식 지원 모델 | CNN(검출·세그·포즈) 🟢 [N1] | 동일 🟢 [N1] | CNN + **LLM·VLM**. 출시 모델은 DeepSeek-R1-Distill 1.5B, Llama 3.2 1B, Qwen2.5 1.5B 등. 약 1~7B까지 🟢 [N3], 문서상 약 6B까지 🟢 [N1] | 검출, 분류, 세그, 포즈, 디노이징 등 CNN 🟡 [N8] |
| 생성(확산) 모델 | 해당 없음 | 해당 없음 | **공식 언급 없음** 🔴 | 언급 없음 🔴 |
| 툴체인 | Dataflow Compiler → HEF, HailoRT 🟢 [N3][N5] | 동일 | 동일 + hailo-ollama, LoRA 어댑터 컴파일 🟢 [N3] | DXNN SDK: DX-COM(컴파일), DX-RT(런타임), DX-STREAM(GStreamer) 🟡 [N8] |
| 양자화 | INT8 | INT8 | INT4/INT8 | INT8 "IQ8"(FP32 수준 정확도라는 제조사 주장) 🟡 [N7] |
| 파이 카메라 스택 연동 | rpicam-apps, Picamera2 🟢 [N1] | 동일 | 동일 🟢 [N3] | 없음(직접 구성) 🔴 |
| 벤치마크 예시 | YOLOv8m 87 FPS, YOLOv11m 58 FPS (배치 8) 🟡 [N5] | YOLOv8m 140 FPS, YOLOv11m 101 FPS (배치 8) 🟡 [N5] | 비전은 "26 TOPS 제품과 대체로 동급" 🟢 [N3]. qwen2 1.5B 질의 13.6초(파이 CPU 22.9초) 🟡 [N4] | 공개 FPS 없음 |
| 주의점 | 대형·다중 모델은 컨텍스트 스위치로 처리량 감소 | 동일 | 출시 초기 소프트웨어가 미완(리뷰어가 "HailoRT not ready" 오류 경험) 🟡 [N4]. Hailo-8용 HEF 호환 안 됨, 재컴파일 필요 🟢 [N3] | 모듈별 메모리·폼팩터가 다름. 연산자 지원 공개 문서 부족 |

**해석 메모**

- Hailo-10H의 40 TOPS는 INT4 기준입니다. INT8로 비교하면 약 20 TOPS로, Hailo-8(26 TOPS)보다 낮습니다. 비전 성능이 "26 TOPS 제품과 비슷하다"는 공식 설명과도 맞습니다. 🟢/🟡
- 즉 Hailo-10H를 고르는 이유는 **TOPS가 아니라 8GB 메모리**입니다. 수억~수십억 파라미터 모델을 NPU 쪽에 올려 둘 수 있다는 점이 생성형 렌더링 실험에 의미가 있습니다.
- 파이5의 PCIe는 x1이라 Gen3 기준 단방향 약 1GB/s입니다(PCIe 규격 계산 🔴). 카메라 입력을 보내는 데는 충분하지만, 메모리가 없는 Hailo-8에서 큰 모델의 가중치를 계속 주고받으면 병목이 됩니다.

---

## 3. 테슬라 AI4 리얼리티 뷰 예산과 비교

01 문서의 테슬라 수치를 기준으로 비교합니다.

| 항목 | 테슬라 AI4 (리얼리티 뷰) | 파이5 + Hailo-10H | 파이5 + Hailo-8 | 파이5 + DX-M1 |
|---|---|---|---|---|
| 렌더러가 쓰는 NPU 몫 | 한 칩의 TRIP 3개 중 0.5개(약 1/6) 🟢 머스크 | NPU 전체 | NPU 전체 | NPU 전체 |
| 그 몫의 TOPS | AI4 공식 TOPS 없음. 매체 추정 칩당 50~250을 쓰면 **약 8~40 TOPS** 🔴 | 20 (INT8) / 40 (INT4) | 26 (INT8) | 25 (INT8) |
| 렌더러 모델 크기 | 약 1B 파라미터 🔴 보도 | 8GB면 INT4 1B(약 0.5GB)는 메모리상 적재 가능 🔴 | 온보드 메모리 없음, 1B는 비현실적 🔴 | 4GB면 INT8 1B(약 1GB) 적재 가능, 단 연산자 지원이 관건 🔴 |
| 메모리 | GDDR6, SoC당 16GB 🟡 | 8GB(종류 미공개) | 없음 | 4GB LPDDR5 |
| 카메라 입력 | 약 5MP × 7~8대 🟡 | 파이5 CSI 2포트, 4대는 어댑터 필요 | 동일 | 동일 |
| 출력 | 8프레임 @ 36fps, UI 해상도 🟢/🟡 | 목표: 저해상도(예: 256~512px) 수 fps 🔴 | 해당 없음(1~2단계용) | 미정 |

**정리** 🔴

- TOPS 자릿수만 보면 엣지 NPU 하나가 리얼리티 뷰 렌더러가 쓰는 몫과 크게 다르지 않을 수 있습니다. 하지만 AI4의 TOPS 자체가 공식 수치가 아니라서 이 비교는 매우 거칠다는 점을 감안해야 합니다.
- 실제 격차는 **메모리 대역폭**과 **트랜스포머·확산 연산자 지원**에서 생깁니다. 테슬라는 자체 컴파일러로 diffusion transformer를 TRIP에 맞췄지만, Hailo와 DEEPX는 공개적으로 확산 모델 지원을 밝히지 않았습니다.
- 따라서 3단계 목표는 "1B 렌더러 재현"이 아니라 **수천만~수억 파라미터의 소형 조건부 생성 모델을 저해상도로 돌리는 것**이 현실적입니다.

---

## 4. 단계별 추천 구성

기존 모델 선정 문서의 1~3단계 계획에 하드웨어를 맞추면 아래와 같습니다.

| 단계 | 내용 | 추천 NPU | 이유 |
|---|---|---|---|
| 1단계 | 카메라별 검출·세그 → IPM으로 BEV 합성 | Hailo-8 (AI HAT+ 26T, 약 $110) | 모델 주 HEF를 바로 쓸 수 있고, rpicam-apps 연동이 공식 지원됩니다. |
| 2단계 | FlashOcc 분할 실행(occupancy) | Hailo-8 vs DX-M1 비교 | 두 모델을 동시에 올릴 때 DX-M1의 4GB 메모리가 유리할 수 있습니다. |
| 3단계 | UVE식 인코더 특징 → 소형 생성 렌더러(가상 시점 영상) | **Hailo-10H (AI HAT+ 2, $200)** 를 우선 검토 | 8GB 메모리와 트랜스포머(LLM/VLM) 실행 경험이 있는 유일한 파이 공식 HAT입니다. 확산 블록이 컴파일되지 않으면 렌더러를 CNN 기반 few-step 생성기(예: U-Net 경량판)로 바꿔 Hailo-8/DX-M1에서도 시도할 수 있습니다. |

---

## 5. 직접 확인해야 할 항목

- [ ] Hailo-10H Dataflow Compiler에서 소형 DiT 블록(어텐션 + adaLN)과 경량 U-Net이 컴파일되는지
- [ ] Hailo-10H에서 INT4/INT8 생성 모델의 실측 지연(프레임당 ms)
- [ ] AI HAT+ 2 8GB 메모리의 종류와 대역폭(공식 미공개)
- [ ] DEEPX DX-COM의 트랜스포머 연산자 지원 여부(deepx.ai 공식 데이터시트 확인)
- [ ] 파이5 + NPU 조합의 실측 소비전력(공식 수치 없음)
- [ ] DX-M1 모듈별 메모리 사양 차이(4GB LPDDR5 / 1GB LPDDR4X)

---

## 6. 출처 목록

| 번호 | 출처 | 비고 |
|---|---|---|
| N1 | [AI HATs – Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) | 공식. 페이지 후반부는 요약본으로 확인 |
| N2 | [Raspberry Pi AI HAT+ 2 제품 페이지](https://www.raspberrypi.com/products/ai-hat-plus-2/) | 공식 |
| N3 | [Introducing the Raspberry Pi AI HAT+ 2 – Raspberry Pi News](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/) (2026-01-15) | 공식 |
| N4 | [Raspberry Pi AI HAT+ 2 targets generative AI – CNX Software](https://www.cnx-software.com/2026/01/15/raspberry-pi-ai-hat-2-targets-generative-ai-llm-vlm-with-hailo-10h-accelerator/), [Raspberry Pi AI HAT+ 2 review – Tom's Hardware](https://www.tomshardware.com/uk/raspberry-pi/raspberry-pi-ai-hat-plus-2-review) | 리뷰어 |
| N5 | [Hailo-8 vs Hailo-8L vs Hailo-10H – Electromaker](https://www.electromaker.io/blog/article/hailo-8-vs-hailo-8l-vs-hailo-10h-which-ai-accelerator-should-you-buy) (2026-09-11) | 3자, Hailo 데이터시트 인용 |
| N6 | [DX-M1 AI Accelerator (DFR1252) – DFRobot Wiki](https://wiki.dfrobot.com/dfr1252/) | 판매사 |
| N7 | [Radxa AICore DX-M1/DX-M1M – Radxa Docs](https://docs.radxa.com/en/aicore/dx-m1) | 판매사 |
| N8 | [Radxa AICore DX-M1M M.2 2242 – CNX Software](https://www.cnx-software.com/2026/03/21/radxa-aicore-dx-m1m-m-2-2242-low-power-ai-module-delivers-25-tops-of-edge-ai-performance-for-just-3w-of-power/) (2026-03-21) | 리뷰어 |
| N9 | [DFRobot DX-M1 25 TOPS M.2 AI accelerator – Wccftech](https://wccftech.com/dfrobot-dx-m1-25-tops-m2-ai-accelerator-module-raspberry-pi-pc/), [DEEPX DX-M1 – DigiKey](https://www.digikey.com/en/product-highlight/d/deepx/dx-m1-m2-ai-acceleration-module) | 3자, 유통사 |
