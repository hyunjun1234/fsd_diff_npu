# 테슬라 FSD v15 리얼리티 뷰(Reality View) 기술 조사

- 작성일: 2026-11-24
- 범위: FSD v15 리얼리티 뷰의 파이프라인, 사용 모델, 차량 컴퓨터(HW3 / HW4(AI4) / AI5) 구조와 성능, 카메라 스펙
- 목적: 라즈베리파이 + NPU(Hailo, DEEPX 등) 엣지 장치에서 유사 기능을 구현하기 위한 기초 자료

> **읽는 법**
> 각 수치 옆에 신뢰도 표시를 달았습니다.
> - 🟢 **공식**: 테슬라 릴리스 노트, 테슬라/일론 머스크/아쇽 엘루스와미(테슬라 AI 부사장)의 발언. 다만 원문을 직접 읽지 못하고 언론 인용으로 확인한 경우가 대부분이라 "공식(언론 인용)"으로 적었습니다.
> - 🟡 **3자 분석**: 발표 자료 재구성, 펌웨어 분석(green 등), 티어다운, 전문 매체 분석.
> - 🔴 **추정/추측**: 기자나 블로거의 추측, 출처가 없는 수치, 서로 충돌하는 수치.
>
> 출처 번호 `[S#]`는 문서 맨 아래 [출처 목록](#9-출처-목록)에 있습니다.
> FSD v15는 2026-11-21에 막 배포를 시작해서, 테슬라가 공개한 기술 정보가 아직 매우 적습니다. 숫자는 앞으로 바뀔 수 있습니다.

---

## 1. 요약

| 항목 | 내용 | 신뢰도 |
|---|---|---|
| 첫 배포 | 2026-11-21 얼리 액세스, 버전 **FSD v15.0.1 / 펌웨어 2026.38.6.11** | 🟢 [S2][S3] |
| 정식(와이드) 배포 | 2026년 12월, "2026 홀리데이 업데이트"와 함께 예정 | 🟢 머스크 [S3][S7] |
| 리얼리티 뷰 정의 | "임의의 가상 카메라 시점으로 계속 스트리밍되는 실시간 생성형 주행 시각화" | 🟢 릴리스 노트 [S3] |
| 핵심 변화 | 기존의 "명시적 모델 출력(차량 박스, 차선 등)" 기반 렌더링을 **하나의 엔드투엔드 시각화로 대체**, 연산량은 오히려 감소 | 🟢 릴리스 노트 [S3], 엘루스와미 [S4] |
| 렌더러 모델 | 차량 내 모델은 **"conditional mean-flow 모델"**, 입력은 테슬라 **Universal Vision Encoder(UVE)** 특징 시퀀스 | 🟢 머스크 [S6] |
| 렌더러 크기 | 약 **10억(1B) 파라미터** | 🔴 기자 서술 [S1][S6] |
| 출력 | 한 사이클에 **8프레임 @ 36fps** 짧은 클립 생성, UVE는 **18Hz**로 처리, 4사이클에 걸쳐 분산 | 🟢/🟡 머스크 발언을 기자가 정리 [S1][S6] |
| 사용 칩 자원 | AI4 컴퓨터의 두 칩 중 한 칩에서, **TRIP 가속기 3개 중 절반 1개 분량** | 🟢 머스크(언론 인용) [S1][S5] |
| 지원 하드웨어 | **HW4(AI4) 전용**, HW3는 v14 Lite 분기 유지 | 🟢 머스크(실적 발표, 언론 인용) [S8], 🟡 다수 매체 [S3][S9] |
| 주행 모델 | v14 대비 **파라미터 약 10배(약 1B → 10B)** | 🟢 머스크 "10x" [S9], 🔴 구체 수치는 기자 서술 |
| 공유 인코더 | FSD·로보택시·옵티머스가 공유하는 비전 인코더, 수백만 개 인터넷 영상으로 학습, 치명적 개입률 25배 개선 | 🟢 릴리스 노트 [S2][S3] |

---

## 2. 리얼리티 뷰란 무엇인가

### 2.1 공식 설명

- 릴리스 노트 원문(일부) 🟢 [S3]:
  - "Introduced Tesla Reality View, replacing all of the explicit model signals into a single end-to-end visualization ... using less total compute."
  - "New real-time generative driving visualization that continuously streams a view from an arbitrary virtual camera."
  - 사용자는 "FSD가 무엇을 하려는지 보고 제어(see and control)"할 수 있습니다.
  - 화면 구성: 전체 화면 시각화 / 지도 / 분할 화면, 그리고 "quiet mode" 선택이 가능합니다.
  - 다가오는 회전 안내가 시각화 화면 안에 직접 표시됩니다. [S6]
- 일부 매체는 같은 항목을 "Tesla Augmented Reality"로 인용했습니다. 테슬라 X 공지에서는 "Tesla Reality View"라는 이름을 씁니다. [S2][S6]
- 아쇽 엘루스와미(X, 2026-11-21) 🟢 [S4]:
  - 이 화면은 "차량을 제어하는 **같은 엔드투엔드 주행 모델**이 생성"하며, 주변 세계의 사실적인 모습을 렌더링합니다.
  - "화살표는 FSD가 가려는 회전, 핀은 주차하려는 자리를 보여줍니다."
  - 2020년 와이어프레임 시각화와 비교하며 "오른쪽(새 화면)이 예전 왼쪽보다 연산을 덜 쓴다"고 했습니다.
- 일론 머스크 🟢 [S5]: "게임 엔진 없이(no gaming engine) 실시간 AI 영상 생성"이라고 설명했습니다(언론 의역).

### 2.2 어떻게 보이는가 (사용자·매체 관찰)

- 차 위쪽 가상 카메라에서 본 사실적인 3D 장면입니다. 도로, 주변 차량, 큰 회전 화살표가 장면 위에 직접 그려집니다. 🟡 [S1][S4]
- Not a Tesla App의 분석: 탐지 결과를 그리는 방식이 아니라 "카메라 영상을 3D 형상에 매핑한 것에 가깝다"고 봅니다. 반사는 그냥 텍스처로 보이고, 횡단보도·신호등이 '탐지된 것'인지 구분하기 어렵습니다. 🟡 [S1]
- 알려진 불만: 지연이 있습니다. 시점을 드래그할 때 입력 지연이 있고, 화면이 실제 앞유리보다 "한순간 늦게" 보입니다. 🟡 [S6]
  - 머스크는 "연산 제약으로 지연이 다소 높다", "새 UVE 특징을 diffusion transformer 레이어 사이에 끼워 넣어 지연을 약 35% 줄일 방법이 있다"고 답했습니다. 🟢 [S6]

### 2.3 렌더링 방식: 공식 vs 추측

| 주장 | 내용 | 신뢰도 |
|---|---|---|
| conditional mean-flow 모델 | 머스크: 차량 내 버전은 데이터 센터의 "~픽셀 단위 정확도의 월드 모델"의 **훨씬 작은 버전**이며, UVE 시퀀스 기반 conditional mean-flow 모델 | 🟢 [S6] |
| diffusion transformer 구조 | 머스크가 "diffusion transformer layers"를 언급 | 🟢 [S6] |
| 이전 프레임 기반 생성 | 이전 리얼리티 뷰 프레임에 노이즈를 더한 뒤 다음 프레임으로 정제 | 🟡 기자 해설 [S1] |
| 약 1B 파라미터 렌더러, 3B 행동 모델과 UVE 공유 | 행동 출력 모델이 3B라는 수치는 출처 없음 | 🔴 [S1][S6] |
| 가우시안 스플래팅 | 블로그 추측. 테슬라는 기법 이름을 밝힌 적이 없음 | 🔴 [S10] |
| 언리얼 엔진 | 티저 시점 기자 추측. 머스크의 "게임 엔진 없음" 발언과 충돌 | 🔴 [S6] |

참고로, 한 유튜버(vogel)가 테슬라 6개 카메라 대시캠 약 13분 분량으로 가우시안 스플랫 장면을 재구성했습니다. 이 과정에서 카메라 간 동기 오차가 **최대 8.7ms**로 측정됐습니다. 테슬라 자체 정보는 아니지만 엣지 재현 시 참고가 됩니다. 🟡 [S1]

---

## 3. 파이프라인: 카메라에서 화면까지

### 3.1 기존 방식 (FSD v11~v14 시각화)

```
8개 카메라 ─► 백본(RegNet/HydraNet 계열) ─► 다중 카메라 융합(BEV, 3D)
          ─► 명시적 출력 헤드: 차량/보행자 박스, 차선, 점유(occupancy), 신호등 …
          ─► UI 렌더러(사전 제작 3D 에셋을 위치에 배치)  ─► 화면
```

- 2021~2022 AI Day에서 테슬라가 공개한 구조(HydraNet, BEV 융합, Occupancy Network)가 기반입니다. 이 부분은 본 조사에서 원문을 다시 확인하지 못해 일반적인 공개 지식으로만 적었습니다. 🟡
- 화면의 차량·차선은 "탐지 결과 → 에셋"이라서 탐지되지 않은 것은 보이지 않습니다.

### 3.2 v15 리얼리티 뷰 방식

```
카메라(AI4: 약 5MP × 7~8대)
   │
   ▼
Universal Vision Encoder (UVE, FSD·로보택시·옵티머스 공용, 18Hz 처리)
   │                                   │
   ▼                                   ▼
엔드투엔드 주행 모델(v15, 약 10B)    리얼리티 뷰 렌더러
 → 조향·가감속, 경로 의도           (conditional mean-flow / diffusion transformer, 약 1B)
                                     입력: UVE 특징 시퀀스 + 가상 카메라 포즈 + 이전 프레임
                                     출력: 8프레임 @ 36fps 클립, UI 해상도
   │                                   │
   └──── 의도(경로 화살표, 주차 핀) ───┴─► 화면 합성
```

- 핵심은 **BEV/점유 같은 명시적 중간 표현을 그리지 않고**, 인코더의 잠재 특징에서 바로 영상을 생성한다는 점입니다. 🟢 [S3][S4][S6]
- 테슬라의 ICCV 2025 발표(엘루스와미, 2025-10)에 따르면, 같은 엔드투엔드 모델에 프롬프트를 주어 3D 점유, 객체, 신호등, 도로 경계, 자연어 설명 같은 **중간 출력**을 뽑을 수 있습니다. 또 테슬라식 "빠른 생성형 가우시안 스플래팅"을 디버깅용으로 씁니다. 🟡 발표 요약 [S11][S12]
  - 따라서 렌더러 내부에서 3D 구조(점유·가우시안)를 암묵적으로 쓰는지는 **공개되지 않았습니다**. 🔴
- 데이터 센터의 "뉴럴 월드 시뮬레이터"는 행동 조건부로 **8개 5MP 카메라 영상을 1분 이상, 일관되게** 생성합니다. 플릿 데이터는 "하루 500년치 주행"이 쌓인다고 합니다. 차량 내 리얼리티 뷰는 이 모델의 축소판입니다. 🟢/🟡 [S6][S11][S12]

---

## 4. 사용 모델 정리

| 모델 | 역할 | 크기 / 구조 | 실행 위치 | 신뢰도 |
|---|---|---|---|---|
| Universal Vision Encoder (UVE) | 카메라 → 잠재 특징. FSD·로보택시·옵티머스 공용, 수백만 인터넷 영상으로 학습 | 미공개 | 차량 AI4 | 🟢 [S2][S3][S6] |
| FSD v15 엔드투엔드 주행 모델 | 픽셀 → 조향·가속, 필요 시 추론 토큰 | v14 대비 10배, 약 10B (v14 약 1B) | 차량 AI4 | 🟢 "10x" [S9], 🔴 절대 수치 |
| 리얼리티 뷰 렌더러 | UVE 특징 → 가상 시점 영상 | conditional mean-flow, diffusion transformer, 약 1B | AI4의 한 칩, TRIP 1/2개 | 🟢 구조 [S6], 🔴 크기 |
| 뉴럴 월드 시뮬레이터 | 행동 조건부 멀티카메라 영상 생성, 평가·RL·합성 데이터 | "~픽셀 단위 정확" 대형 모델 | 데이터 센터 | 🟢 [S6], 🟡 [S11] |
| FSD v14 Lite | HW3용 증류(distilled) 모델 | 미공개 | HW3 | 🟡 [S9][S8] |

> 매체 간 충돌: 일부(KuCoin/CoinDesk)는 "1B → 10B"를 리얼리티 뷰와 같은 문맥으로 썼습니다. 하지만 이 수치는 **주행 네트워크** 이야기이고, 리얼리티 뷰 렌더러는 약 1B로 보도됐습니다. [S5][S1]

---

## 5. 차량 컴퓨터(칩셋) 구조와 성능

### 5.1 세대별 비교

| 항목 | HW3 (FSD Chip, 2019) | HW4 / AI4 (2023) | AI5 (개발 중) |
|---|---|---|---|
| 공정·파운드리 | 삼성 14nm 🟢(2019 발표, [S13]) | 삼성 7nm급 🟡 [S14][S15] | TSMC + 삼성(테일러) 이원화 🟡 [S15][S16]; 삼성 2nm 보도 🔴 [S14] |
| 다이 / 트랜지스터 | 260mm², 60억 개 🟢 [S13] | 미공개 | 미공개 |
| SoC 구성 | 보드당 2칩(이중화, 각자 DRAM·플래시) 🟢 [S13] | 보드당 2칩(메인 + 보조) 🟡 [S14] | **단일 SoC**로 전환 🟢 머스크(인용) [S16] |
| CPU | Cortex-A72 12코어(4코어 × 3), 2.2GHz 🟢 [S13] | 미공개 (인포테인먼트는 AMD 별도) | 미공개 |
| GPU | Mali-G71 MP12, 1GHz, 600 GFLOPS 🟢 [S13] | 미공개 | 미공개 |
| NPU | 칩당 2개. 각 96×96 MAC(9,216개), SRAM 32MiB, 2GHz 🟢 [S13] | 칩당 **TRIP 3개**("트리플 코어") 🟢/🟡 [S1][S5] | 미공개 |
| INT8 TOPS | NPU당 36.86, 칩당 73.7, **보드 144** 🟢 [S13][S17] | **공식 없음.** 매체별 100~150(이중 SoC 추정) / 243 / 300~500 / 약 500으로 제각각 🔴 [S14][S15][S16][S17] | 목표 2,000~2,500 🔴; "AI4 대비 40배(일부 시나리오)" 🟢 머스크(인용) [S16]; 다른 보도는 3~5배, 약 12배 🔴 [S6][S17] |
| 메모리 | LPDDR4-4266, 128bit, 최대 8GiB, 68GB/s 🟢 [S13] | GDDR6(삼성·마이크론) 🟡 [S14], SoC당 16GB(총 32GB) 🟡 [S17] | AI4 대비 9배, 144GB, SK하이닉스 🔴 [S16] |
| 전력 | 컴퓨터 72W(칩당 36W, NPU 15W) 🟢 [S13] | 최대 160W 🔴 [S16] | 최대 800W 🔴 [S16] |
| ISP | 24bit, 초당 10억 픽셀, HDR 센서 8개 🟢 [S13] | 미공개 | 미공개 |
| 일정 | 양산 중 | 양산 중, AI4+(SoC당 32GB, 연산·대역폭 약 10%↑) 2027 중반 🟡 [S17] | 테이프아웃 2026-04-15 🟢 [S16]; 양산 2027 중반~말 🔴 [S15][S16][S17] |

**HW4 TOPS에 대해**: 테슬라는 AI4의 TOPS를 공식 발표한 적이 없습니다. 매체마다 수치가 2~5배 차이 나므로, 설계 판단에는 쓰지 않는 것을 권합니다. 🔴

### 5.2 어느 칩에서 어떤 모델을 돌리는가

| 하드웨어 | FSD 버전 | 실행 모델 | 리얼리티 뷰 | 신뢰도 |
|---|---|---|---|---|
| HW3 | v14 Lite(v14.3 Lite) 분기에 머무름 | 증류된 v14 주행 모델 | **미지원** | 🟢 머스크(인용) [S8], 🟡 [S3][S9] |
| HW4 / AI4 | v15 | UVE + 약 10B 주행 모델 + 약 1B 리얼리티 뷰 렌더러 | 지원. 2칩 중 한 칩의 TRIP 3개 중 절반 1개 분량 사용 | 🟢 [S1][S5][S8] |
| AI5 | — | 머스크: "옵티머스로 간다. 차량은 AI4에 머문다. 초인적 주행에는 그걸로 충분하다" | 리얼리티 뷰 성능 향상이 기대된다는 언급만 있음 | 🟢 [S6], 🟡 [S1] |
| 로보택시 | 초기 v15 빌드(지오펜스 내 비감독) | — | — | 🟡 [S7][S8] |

**추론(🔴, 본 문서의 해석)**: 리얼리티 뷰가 TRIP 3개 중 0.5개, 즉 **한 칩 NPU의 약 1/6**만 쓴다고 하면, AI4 한 칩 TOPS를 매체 추정치 중간값으로 잡아도 수십 TOPS 수준입니다. 이는 Hailo-8(26 TOPS)이나 DEEPX DX-M1(25 TOPS)과 같은 자릿수입니다. 다만 AI4는 GDDR6 대역폭과 대용량 메모리가 있어서, 1B 파라미터 생성 모델을 실제로 돌리는 조건은 엣지 NPU와 크게 다릅니다(7장 참고).

---

## 6. 카메라 스펙

### 6.1 HW3 vs HW4

| 항목 | HW3 | HW4 (AI4) | 신뢰도 |
|---|---|---|---|
| 외부 카메라 수 | 8개(전방 3: 메인·내로우·와이드) | 7~8개(전방 활성 2: 메인·와이드). 앞유리 3번째 자리는 미사용 | 🟡 [S18][S19] |
| 추가 카메라 | — | 전방 범퍼 카메라: 사이버트럭(2023)부터, 모델 3는 2025-10 생산분부터. 모델·시기별로 다름 | 🟡 [S19] |
| 센서 | 온세미(Aptina) AR0136AT | 소니 IMX963 (전기차 매체 표에는 "IMX490(추정)"으로 충돌) | 🟡 펌웨어 분석 [S18][S20] |
| 해상도 | 1.2MP (1280 × 960) | 약 5MP (2896 × 1876, 약 5.4MP) | 🟡 [S18][S15] |
| 컬러 필터 | RCCC | RGGB (컬러) | 🟡 [S18] |
| 다이내믹 레인지 | 약 110dB | 120dB 이상(단일 노출) | 🟡 [S18] |
| 인터페이스 | FPD-Link III(추정) | GMSL2 또는 MIPI A-PHY(추정) | 🔴 [S18] |
| 렌즈 | 일반 | 붉은빛 IR컷·눈부심 방지 코팅 | 🟡 [S18][S19] |
| 히터 | 수동(폐열) | 능동 히터 | 🟡 [S18] |
| 프레임레이트 | 공식 없음. 리얼리티 뷰 출력이 36fps라는 점과 커뮤니티 통설(카메라 36Hz)이 일치하지만 본 조사에서 1차 출처는 확인하지 못함 | 동일 | 🔴 |
| 카메라별 화각·거리 | 공식 자료 미확인 | 후방 카메라가 더 넓은 어안 화각 | 🟡 [S21] |
| 차세대 | — | 펌웨어에 "IMX00N" 문자열 발견, 사양 미공개·양산 미확인 | 🟡 [S18][S20] |

### 6.2 레이더

- HW3 시기(2021-05 모델 3/Y, 2022-02 S/X)부터 전방 레이더를 제거했습니다(테슬라 비전). 🟡 [S15]
- 사이버트럭과 일부 HW4 S/X에는 2023년 말부터 HD 이미징 레이더가 들어갔습니다. 다만 FSD가 이를 쓰는지는 확인되지 않았습니다. 🟡 [S15]

---

## 7. 엣지 NPU 구현을 위한 시사점 (초안)

이 장은 최종 목표(라즈베리파이 + NPU)로 이어지는 **본 문서의 해석**입니다. 🔴 수치는 별도 검증이 필요합니다.

1. **규모 차이**: 테슬라 렌더러는 약 1B 파라미터 생성 모델을 수십 TOPS급 NPU와 GDDR6 메모리 위에서 돌립니다. 라즈베리파이 5 + Hailo-8(26 TOPS)이나 DEEPX DX-M1(25 TOPS) 계열은 연산 자릿수는 비슷하지만 메모리 대역폭과 용량이 훨씬 작습니다. 1B 확산 모델을 그대로 올리는 것은 현실적이지 않습니다. (Hailo-10H는 LPDDR4를 붙인 생성형 AI용 제품이라 후보로 검토할 만합니다.)
2. **현실적인 단계별 경로**:
   - 1단계: 다중 카메라(2~4대) → 경량 BEV/점유 네트워크(NPU) → 명시적 3D 렌더링(Pi GPU/CPU). "기존 방식(3.1)"을 재현합니다.
   - 2단계: 카메라 영상을 3D 형상에 투영하는 텍스처 매핑 렌더링. Not a Tesla App이 관찰한 "영상을 3D에 매핑한 것 같은" 느낌을 생성 모델 없이 근사합니다.
   - 3단계: 저해상도 잠재 공간에서 소형 생성 모델(수천만 파라미터급, few-step consistency/mean-flow 계열)로 시점 변환. 테슬라 구조(인코더 특징 → 조건부 1~few-step 생성)를 축소 재현합니다.
3. **설계 팁**: 테슬라처럼 인코더를 주행·시각화가 **공유**하게 하고, 생성은 저주기(18Hz 인코더 / 36fps 출력)로 나누어 여러 사이클에 분산하는 구조가 엣지에도 유리합니다.
4. **카메라 동기화**: 3D 재구성을 하려면 카메라 간 동기 오차를 수 ms 이내로 관리해야 합니다(테슬라 대시캠 실측 최대 8.7ms [S1]).

---

## 8. 확인하지 못한 것 / 후속 조사 항목

- 테슬라 릴리스 노트 원문, 머스크·엘루스와미 X 원문 직접 확인. 현재는 모두 언론 인용입니다.
- AI4 칩 다이 사진, TRIP 구조, 공식 TOPS.
- 카메라별 화각, 최대 거리, 프레임레이트의 1차 출처.
- 리얼리티 뷰가 사용하는 입력 카메라 목록과 출력 해상도(픽셀).
- ICCV 2025 엘루스와미 발표 영상 원본 확인(https://www.youtube.com/watch?v=wHK8GMc9O5A).
- 엣지 NPU(Hailo-8/8L/10H, DEEPX DX-M1 등) 공식 사양 비교. 다음 문서에서 다룰 예정입니다.

---

## 9. 출처 목록

| 번호 | 출처 | 날짜 |
|---|---|---|
| S1 | [How Tesla's FSD V15 Reality View Actually Works [VIDEO] – Not a Tesla App](https://www.notateslaapp.com/news/5068/how-teslas-fsd-v15-reality-view-actually-works-video) | 2026-11 |
| S2 | [Tesla launches FSD v15 … – Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/) | 2026-11-21 |
| S3 | [Tesla FSD v15 (2026.38.6.11): Rollout Begins, Official Release Notes – Tesla Oracle](https://www.teslaoracle.com/2026/11/22/tesla-fsd-v15-2026-38-6-11-rollout-begins-official-release-notes-and-new-features/) | 2026-11-22 |
| S4 | [Tesla Reality View takes FSD visualizations to an entirely new level – Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-reality-view/) | 2026-11 |
| S5 | [Tesla FSD v15 uses half of the AI4 chip's TRIP accelerator for real-time AI video – KuCoin News(CoinDesk 전재)](https://www.kucoin.com/news/flash/tesla-fsd-v15-uses-half-of-ai4-chip-s-trip-accelerator-for-real-time-ai-video) | 2026-11 |
| S6 | [Tesla FSD v15's 'Reality View' has one complaint and Musk has the solution – Teslarati](https://www.teslarati.com/tesla-fsd-v15s-reality-drive-one-complaint-musk-solution/) | 2026-11 |
| S7 | [Tesla FSD v15 to go wide release with the 2026 Holiday Update – Tesla Oracle](https://www.teslaoracle.com/2026/11/21/tesla-fsd-v15-to-go-wide-release-with-the-2026-holiday-update-and-mind-bending-reality-renders-in-december-says-musk/) | 2026-11-21 |
| S8 | [Tesla starts FSD v15 rollout to early testers and leaves HW3 cars out – Startup Fortune](https://startupfortune.com/tesla-starts-fsd-v15-rollout-to-early-testers-and-leaves-hw3-cars-out/) | 2026-11-22 |
| S9 | [Tesla VP Talks About What's Coming in FSD V15 – Not a Tesla App](https://www.notateslaapp.com/news/4679/tesla-ai-chief-talks-about-whats-coming-in-fsd-v15) | 2026-09-11 |
| S10 | [Gaussian Splatting explained: the tech likely behind Tesla's Reality View – astig.ph](https://astig.ph/gaussian-splatting-explained-tesla-reality-view/) | 2026-11 |
| S11 | [Tesla ICCV 2025 Foundational Model for FSD 요약 스레드 – Bradford Ferguson (Thread Reader)](https://threadreaderapp.com/thread/1982161471644545387) | 2025-10-25 |
| S12 | [Tesla Unveils "World Simulator" – Tiger Brokers / Deep News](https://www.itiger.com/news/1146546403) | 2025-10-27 |
| S13 | [FSD Chip – WikiChip](https://en.wikichip.org/wiki/FSD_Chip) (테슬라 2019 Autonomy Day, Hot Chips 31 자료 정리) | — |
| S14 | [Tesla's HW4 – Behind the Chip (Gidion)](https://behindthechip.substack.com/p/teslas-hw4) | 2026-08-19 |
| S15 | [Tesla FSD Hardware: HW1 to HW5 – TeslAnt](https://teslant.com/en/guides/tesla-fsd-hardware-generations-reference-2026) | 2026-06-23 |
| S16 | [Tesla AI5 Chip: Specs, Manufacturer, AI4 vs AI5 – AutoAkhbar](https://autoakhbar.com/tesla-ai5-chip-specifications-manufacturer-vs-ai4/) | 2026-04-18 |
| S17 | [How Tesla's FSD Hardware Has Changed Over the Years – Not a Tesla App](https://www.notateslaapp.com/news/3521/how-teslas-fsd-hardware-has-changed-over-the-years) | 2026-05-27 |
| S18 | [Tesla new camera upgrade … – Electrek](https://electrek.co/2025/12/02/tesla-new-camera-upgrade-casting-more-doubt-full-self-driving-promises/) (green 펌웨어 분석 인용) | 2025-12-02 |
| S19 | [Tesla Hardware 4 rolling out to new vehicles – AutoPilot Review](https://www.autopilotreview.com/tesla-hardware-4-rolling-out-to-new-vehicles/) | 2026-11 |
| S20 | [Tesla Preparing Upgraded Exterior Cameras as New IMX00N Code Appears – Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-preparing-upgraded-exterior-cameras-as-new-imx00n-code-appears-in-firmware/) | 2025-12-02 |
| S21 | [Tesla's HW3 and HW4 Cameras: Comparing the Differences – Not a Tesla App](https://www.notateslaapp.com/news/1564/teslas-hw3-and-hw4-cameras-comparing-the-differences-in-quality-and-hardware) | 2023-09-02 |
