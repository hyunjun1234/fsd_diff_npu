# 테슬라 FSD v15 리얼리티 뷰 기술 조사

이 문서는 테슬라 FSD v15에 들어간 리얼리티 뷰(Reality View)가 어떤 과정을 거쳐 화면을 만드는지, 어떤 모델과 칩과 카메라를 쓰는지 정리한 조사 자료입니다. 라즈베리파이에 NPU(Hailo, DEEPX 등)를 붙여 비슷한 기능을 만들 때 기준으로 삼으려고 작성했습니다.

FSD v15는 2026-11-21에 배포를 시작했기 때문에 테슬라가 공개한 기술 정보가 아직 적고, 여기 적힌 숫자는 앞으로 바뀔 수 있습니다.

## 수치의 구분

모든 수치에는 출처 번호 `[S번호]`와 함께 아래 세 가지 구분 중 하나를 적었습니다. 출처 목록은 문서 끝에 있습니다.

- 공식: 테슬라 릴리스 노트, 테슬라와 일론 머스크, 아쇽 엘루스와미(테슬라 AI 부사장)의 발언입니다. 대부분 언론이 인용한 내용이며 원문 대조는 확인 필요입니다.
- 3자: 발표 자료를 정리한 글, 펌웨어 분석, 분해 분석, 전문 매체의 분석입니다.
- 추정: 기자나 블로거의 추측, 출처가 없는 수치, 매체끼리 서로 다른 수치입니다.

## 1. 요약

| 항목 | 내용 | 구분 |
|---|---|---|
| 첫 배포 | 2026-11-21, 얼리 액세스 | 공식 [S2][S3] |
| 버전 | FSD v15.0.1 (2026.38.6.11) | 공식 [S3] |
| 전체 배포 | 2026년 12월 예정 | 공식 [S3][S7] |
| 지원 하드웨어 | HW4(AI4) 전용 | 공식 [S8] |
| 렌더러 크기 | 약 1B 파라미터 | 추정 [S1][S6] |
| 주행 모델 크기 | 약 3B 또는 약 10B | 3B는 머스크 발언 인용, 09 문서 |
| 출력 | 8프레임 클립, 36fps | 공식 [S1][S6] |
| 쓰는 칩 자원 | TRIP 3개 중 절반 1개 | 공식 [S1][S5] |

릴리스 노트는 리얼리티 뷰를 임의의 가상 카메라 시점으로 계속 스트리밍되는 실시간 생성형 주행 시각화라고 정의합니다. [S3] 차량 박스나 차선 같은 명시적 모델 출력으로 그리던 화면을 하나의 엔드투엔드 시각화로 바꿨고, 연산량은 오히려 줄었습니다. (공식, 릴리스 노트 [S3], 엘루스와미 [S4])

차량 안의 렌더러는 conditional mean-flow 모델이고, 입력은 테슬라 Universal Vision Encoder(UVE)의 특징 시퀀스입니다. (공식, 머스크 [S6]) 한 번에 8프레임짜리 짧은 클립을 36fps로 만들며, UVE는 18Hz로 처리하고 작업을 4사이클에 나눕니다. 이 내용은 머스크의 발언을 기자가 정리한 것입니다. [S1][S6] 렌더러는 AI4 컴퓨터의 두 칩 중 한 칩에서 TRIP 가속기 3개 가운데 절반 1개 분량을 씁니다. (머스크, 언론 인용 [S1][S5])

HW3는 v14 Lite에 머뭅니다. (머스크의 실적 발표 발언, 언론 인용 [S8], 3자 [S3][S9]) 전체 배포는 2026 홀리데이 업데이트와 함께 나올 예정입니다. (머스크 [S3][S7]) 주행 모델이 약 1B에서 10B로 커졌다는 절대 수치는 추정이고, 머스크는 약 3B라고 했습니다([09 문서](09_tesla_3b_1b_models.md)).

UVE는 FSD, 로보택시, 옵티머스가 함께 쓰는 비전 인코더입니다. 인터넷 영상 수백만 개로 학습했고 치명적 개입률이 25배 좋아졌다고 합니다. (공식, 릴리스 노트 [S2][S3])

## 2. 리얼리티 뷰란 무엇인가

### 2.1 테슬라의 설명

릴리스 노트는 리얼리티 뷰를 이렇게 소개합니다. [S3]

> Introduced Tesla Reality View, replacing all of the explicit model signals into a single end-to-end visualization ... using less total compute.
>
> New real-time generative driving visualization that continuously streams a view from an arbitrary virtual camera.

운전자는 FSD가 무엇을 하려는지 화면에서 보고 조작할 수 있습니다. 화면은 전체 화면 시각화, 지도, 분할 화면 중에서 고르고 quiet mode도 선택할 수 있습니다. 다가오는 회전 안내는 시각화 화면 안에 바로 표시됩니다. [S3][S6]

아쇽 엘루스와미는 2026-11-21에 X에서 다음과 같이 설명했습니다. [S4]

- 이 화면은 차량을 제어하는 것과 같은 엔드투엔드 주행 모델이 만들며, 주변 세계를 사실적으로 그립니다.
- 화살표는 FSD가 가려는 회전을, 핀은 주차하려는 자리를 나타냅니다.
- 2020년의 와이어프레임 화면과 비교하면 새 화면이 연산을 덜 씁니다.

머스크는 게임 엔진 없이 AI가 실시간으로 영상을 생성한다고 설명했습니다. 이 표현은 언론이 옮긴 것입니다. [S5]

### 2.2 화면에서 보이는 모습

화면은 차 위쪽의 가상 카메라에서 내려다본 사실적인 3D 장면입니다. 도로와 주변 차량이 보이고, 큰 회전 화살표가 장면 위에 그려집니다. (3자 [S1][S4])

Not a Tesla App은 이 화면이 탐지 결과를 그린 것이 아니고 카메라 영상을 3D 형상에 입힌 것에 가깝다고 분석했습니다. 반사는 그냥 무늬처럼 보이고, 횡단보도나 신호등은 차가 인식한 것인지 화면만으로는 알기 어렵다고 합니다. (3자 [S1])

지연이 있다는 불만이 있습니다. 손가락으로 시점을 돌릴 때 반응이 늦고, 화면이 실제 앞유리 풍경보다 조금 늦게 따라옵니다. (3자 [S6]) 머스크는 연산 제약 때문에 지연이 다소 크다고 인정했고, 새 UVE 특징을 diffusion transformer 레이어 사이에 끼워 넣어 지연을 약 35% 줄일 방법이 있다고 답했습니다. (공식 [S6])

### 2.3 렌더링 방식에 대한 주장

공식으로 나온 내용은 두 가지입니다. 머스크는 차량 안의 모델이 데이터 센터에 있는 "거의 픽셀 단위로 정확한 월드 모델"을 크게 줄인 것이고, UVE 시퀀스를 입력으로 받는 conditional mean-flow 모델이라고 했습니다. 또 "diffusion transformer layers"를 언급했습니다. [S6]

3자 해설로는 이전 리얼리티 뷰 프레임에 노이즈를 더한 뒤 다음 프레임으로 다듬는다는 기자의 설명이 있습니다. [S1]

나머지는 추정입니다.

- 약 1B 렌더러가 3B 행동 모델과 UVE를 공유한다는 보도가 있습니다. 머스크가 X에서 주행 동작을 내는 모델이 약 3B라고 밝힌 것으로 여러 매체가 옮겼고, 10B라는 수치와 맞지 않습니다. 자세한 내용은 [09 문서](09_tesla_3b_1b_models.md) 2장에 있습니다. [S1][S6]
- 가우시안 스플래팅을 쓴다는 것은 블로그의 추측입니다. 테슬라는 기법 이름을 밝히지 않았습니다. [S10]
- 언리얼 엔진을 쓴다는 것은 티저 때 나온 기자의 추측이며, 게임 엔진을 쓰지 않는다는 머스크의 말과 맞지 않습니다. [S6]

## 3. 카메라에서 화면까지의 처리 과정

### 3.1 이전 방식 (FSD v11부터 v14까지)

```
카메라 8대 -> 백본(RegNet, HydraNet 계열) -> 여러 카메라 융합(BEV, 3D)
          -> 명시적 출력: 차량과 보행자 박스, 차선, 점유(occupancy), 신호등
          -> UI 렌더러(미리 만든 3D 모델을 해당 위치에 배치) -> 화면
```

이 구조는 테슬라가 2021년과 2022년 AI Day에서 공개한 HydraNet, BEV 융합, Occupancy Network를 바탕으로 합니다. 세부 내용의 원문 대조는 확인 필요입니다. 화면의 차량과 차선은 탐지 결과를 3D 모델로 바꿔 놓은 것이어서, 탐지되지 않은 것은 화면에 나오지 않습니다.

### 3.2 v15 리얼리티 뷰

```
카메라(AI4: 약 5MP, 7~8대)
   |
   v
Universal Vision Encoder (UVE, 18Hz 처리)
   |                                   |
   v                                   v
엔드투엔드 주행 모델(약 3B)           리얼리티 뷰 렌더러(약 1B)
 출력: 조향, 가감속, 경로 의도         입력: UVE 특징 시퀀스, 가상 카메라 위치, 이전 프레임
                                      출력: 8프레임 클립(36fps), 화면 해상도
   |                                   |
   +---- 의도(경로 화살표, 주차 핀) ----+--> 화면 합성
```

v15는 BEV나 점유 같은 중간 결과를 그리지 않고, 인코더가 만든 특징에서 바로 영상을 생성합니다. (공식 [S3][S4][S6])

엘루스와미의 ICCV 2025 발표(2025년 10월)에 따르면 같은 엔드투엔드 모델에 요청을 주어 3D 점유, 객체, 신호등, 도로 경계, 자연어 설명 같은 중간 출력을 뽑을 수 있습니다. 테슬라가 만든 빠른 생성형 가우시안 스플래팅은 디버깅에 씁니다. (3자, 발표 요약 [S11][S12]) 렌더러가 내부에서 점유나 가우시안 같은 3D 구조를 쓰는지는 공개되지 않았습니다.

데이터 센터의 뉴럴 월드 시뮬레이터는 주행 동작을 조건으로 받아 5MP 카메라 8대의 영상을 1분 넘게 일관되게 만들어 냅니다. 차량들이 모으는 주행 데이터는 하루에 500년 분량이라고 합니다. 차 안의 리얼리티 뷰는 이 모델을 줄인 것입니다. (공식 [S6], 3자 [S11][S12])

## 4. 사용 모델

| 모델 | 크기 | 실행 위치 |
|---|---|---|
| UVE | 미공개 | 차량 AI4 |
| v15 주행 모델 | 약 3B | 차량 AI4 |
| 리얼리티 뷰 렌더러 | 약 1B | AI4의 한 칩 |
| 뉴럴 월드 시뮬레이터 | 미공개 | 데이터 센터 |
| FSD v14 Lite | 미공개 | HW3 |

Universal Vision Encoder(UVE)는 카메라 영상을 특징으로 바꿉니다. FSD, 로보택시, 옵티머스가 함께 쓰며 인터넷 영상 수백만 개로 학습했습니다. (공식 [S2][S3][S6])

FSD v15 엔드투엔드 주행 모델은 영상에서 조향과 가속을 바로 출력하고, 필요하면 추론 토큰도 만듭니다. 크기는 머스크가 약 3B라고 밝힌 것으로 보도됐습니다([09 문서](09_tesla_3b_1b_models.md) 2장). 이와 별도로 v14의 10배, 약 10B라는 보도도 있는데 10배는 엘루스와미의 발언이고 [S9] 10B는 출처가 없어, 두 수치가 어떻게 맞는지는 확인 필요입니다.

리얼리티 뷰 렌더러는 UVE 특징으로 가상 시점 영상을 만듭니다. conditional mean-flow, diffusion transformer 구조이며 AI4의 한 칩에서 TRIP 절반 1개 분량을 씁니다. 구조는 공식이고 [S6] 크기는 추정입니다.

뉴럴 월드 시뮬레이터는 주행 동작을 조건으로 여러 카메라 영상을 생성하는 대형 모델이며 평가, 강화학습, 합성 데이터에 씁니다. 세부는 공개되지 않았습니다. (공식 [S6], 3자 [S11])

FSD v14 Lite는 HW3용으로 줄인(증류한) 모델입니다. (3자 [S8][S9])

## 5. 차량 컴퓨터의 구조와 성능

### 5.1 세대별 비교

| 항목 | HW3 | HW4 (AI4) | AI5 |
|---|---|---|---|
| 출시 | 2019년 | 2023년 | 개발 중 |
| 공정 | 삼성 14nm | 삼성 7nm급 | 확인 필요 |
| 보드의 칩 수 | 2개 | 2개 | 1개 |
| NPU | 칩마다 2개 | 칩마다 TRIP 3개 | 미공개 |
| INT8 TOPS | 보드 144 | 공식 수치 없음 | 목표 2,000~2,500 |
| 메모리 | 최대 8GiB | 32GB | 144GB |
| 전력 | 72W | 최대 160W | 최대 800W |

표의 값 가운데 HW3는 공식, HW4는 3자 분석, AI5는 대부분 추정입니다. 세대별 설명은 아래와 같습니다.

#### HW3 (FSD Chip)

모든 수치는 테슬라가 2019년에 발표한 내용입니다. [S13]

- 공정과 크기: 삼성 14nm, 다이 260mm², 트랜지스터 60억 개
- 칩 구성: 보드에 칩 2개를 두어 이중화했고, 칩마다 DRAM과 플래시를 따로 둡니다.
- CPU와 GPU: Cortex-A72 12코어(4코어 3묶음) 2.2GHz, Mali-G71 MP12 1GHz(600 GFLOPS)
- NPU: 칩마다 2개. NPU 하나는 96x96 MAC 배열(9,216개), SRAM 32MiB, 2GHz
- INT8 TOPS: NPU 하나에 36.86, 칩 하나에 73.7, 보드 전체 144 [S13][S17]
- 메모리: LPDDR4-4266, 128bit, 최대 8GiB, 68GB/s
- 전력: 컴퓨터 전체 72W, 칩 하나 36W, NPU 15W
- ISP: 24bit, 초당 10억 픽셀, HDR 센서 8개

#### HW4 (AI4)

- 공정: 삼성 7nm급 (3자 [S14][S15])
- 칩 구성: 보드에 칩 2개(주 제어기와 보조 제어기) (3자 [S14])
- NPU: 칩마다 TRIP 가속기 3개 (머스크의 발언과 3자 [S1][S5])
- INT8 TOPS: 공식 수치가 없습니다. 매체에 따라 100~150(칩 2개 합산 추정), 243, 300~500, 약 500으로 다릅니다. (추정 [S14][S15][S16][S17])
- 메모리: GDDR6, 삼성과 마이크론 공급 (3자 [S14]). 칩마다 16GB, 합계 32GB (3자 [S17])
- 전력: 최대 160W (추정 [S16])
- 다이 크기, CPU, GPU, ISP는 공개되지 않았습니다. 인포테인먼트용 AMD 프로세서는 별도입니다.
- AI4+는 칩마다 32GB이고 연산과 대역폭이 약 10% 늘며 2027년 중반으로 알려져 있습니다. (3자 [S17])

#### AI5

- 제조사: TSMC와 삼성 테일러 공장 병행 (3자 [S15][S16]). 삼성 2nm라는 보도는 추정입니다. [S14]
- 칩 구성: 칩 1개로 바뀝니다. (머스크, 언론 인용 [S16])
- INT8 TOPS: 목표 2,000~2,500 (추정). 머스크는 일부 상황에서 AI4의 40배라고 했습니다. [S16] 3~5배, 약 12배라는 보도도 있습니다. (추정 [S6][S17])
- 메모리: AI4의 9배, 144GB, SK하이닉스 (추정 [S16])
- 전력: 최대 800W (추정 [S16])
- 일정: 테이프아웃 2026-04-15 (공식 [S16]). 양산은 2027년 중반에서 말 (추정 [S15][S16][S17])

테슬라는 AI4의 TOPS를 발표한 적이 없습니다. 매체마다 수치가 2배에서 5배까지 차이 나므로 설계 판단의 근거로 쓰기는 어렵습니다.

### 5.2 어느 칩에서 어떤 모델을 돌리는가

| 하드웨어 | FSD 버전 | 리얼리티 뷰 |
|---|---|---|
| HW3 | v14 Lite | 지원 안 함 |
| HW4 (AI4) | v15 | 지원 |
| AI5 | 해당 없음 | 해당 없음 |

HW3는 v14 Lite(v14.3 Lite)에 머물며 줄인 v14 주행 모델을 돌립니다. (머스크, 언론 인용 [S8], 3자 [S3][S9])

HW4(AI4)는 UVE, 약 3B 주행 모델, 약 1B 리얼리티 뷰 렌더러를 돌립니다. 렌더러는 칩 2개 중 한 칩에서 TRIP 3개 가운데 절반 1개 분량을 씁니다. (공식 [S1][S5][S8])

머스크는 AI5가 옵티머스에 들어가고 차량은 AI4에 머문다고 했습니다. 초인적 주행에는 AI4로 충분하다는 설명입니다. 리얼리티 뷰에 대해서는 AI5에서 성능이 좋아질 것이라는 언급만 있습니다. (공식 [S6], 3자 [S1])

아래는 이 문서의 해석입니다. 리얼리티 뷰가 TRIP 3개 중 0.5개, 곧 한 칩 NPU의 약 6분의 1만 쓴다면, AI4 한 칩의 TOPS를 매체 추정치의 중간쯤으로 잡아도 수십 TOPS 수준입니다. Hailo-8(26 TOPS)이나 DEEPX DX-M1(25 TOPS)과 자릿수가 같습니다. 그러나 AI4에는 GDDR6의 넓은 대역폭과 큰 메모리가 있어서 1B 파라미터 생성 모델을 돌리는 조건은 엣지 NPU와 많이 다릅니다. 자세한 비교는 [엣지 NPU 사양 비교](02_edge_npu_spec_comparison.md)에 있습니다.

## 6. 카메라

| 항목 | HW3 | HW4 (AI4) |
|---|---|---|
| 외부 카메라 | 8대 | 7~8대 |
| 전방 카메라 | 3대 | 2대 |
| 센서 | AR0136AT | IMX963 |
| 해상도 | 1280x960 | 2896x1876 |
| 화소 | 1.2MP | 약 5.4MP |
| 컬러 필터 | RCCC | RGGB |
| 다이내믹 레인지 | 약 110dB | 120dB 이상 |
| 히터 | 수동 | 능동 |

표의 값은 모두 3자 분석입니다. [S15][S18][S19] 항목별로 덧붙일 내용은 다음과 같습니다.

- 전방 카메라: HW3는 메인, 내로우, 와이드 3대입니다. HW4는 메인과 와이드 2대만 쓰고 앞유리의 세 번째 자리는 비어 있습니다. [S18][S19]
- 추가 카메라: HW4에는 전방 범퍼 카메라가 있습니다. 사이버트럭(2023년)부터 들어갔고 모델 3는 2025년 10월 생산분부터이며, 차종과 시기에 따라 다릅니다. [S19]
- 센서: HW3는 온세미(Aptina) 제품입니다. HW4의 IMX963은 소니 제품이며 펌웨어 분석에서 나온 이름입니다. 같은 기사의 표에는 "IMX490(추정)"으로 적혀 있어 서로 다릅니다. [S18][S20]
- 다이내믹 레인지: HW4의 120dB 이상은 단일 노출 기준입니다. [S18]
- 인터페이스: HW3는 FPD-Link III, HW4는 GMSL2 또는 MIPI A-PHY로 추정됩니다. [S18]
- 렌즈: HW4는 붉은빛이 도는 적외선 차단, 눈부심 방지 코팅을 씁니다. [S18][S19]
- 히터: HW3는 폐열을 이용하고 HW4는 히터가 따로 있습니다. [S18]
- 프레임레이트: 확인 필요입니다. 리얼리티 뷰 출력이 36fps이고 커뮤니티에서도 카메라를 36Hz로 봅니다. (추정)
- 화각과 거리: HW4 후방 카메라의 화각이 더 넓습니다(어안에 가까움). [S21] 카메라별 수치는 확인 필요입니다.
- 다음 세대: 펌웨어에서 "IMX00N"이라는 이름이 발견됐습니다. 사양은 공개되지 않았고 양산 차량 적용 여부는 확인 필요입니다. [S18][S20]

## 7. 엣지 NPU로 만들 때 생각할 점

이 장은 이 문서의 해석이며 수치는 따로 검증해야 합니다.

테슬라의 렌더러는 약 1B 파라미터 생성 모델을 수십 TOPS급 NPU와 GDDR6 메모리 위에서 돌립니다. 라즈베리파이 5에 Hailo-8(26 TOPS)이나 DEEPX DX-M1(25 TOPS)을 붙이면 연산량의 자릿수는 비슷하지만 메모리 대역폭과 용량이 훨씬 작습니다. 1B 확산 모델을 그대로 올리기는 어렵습니다. Hailo-10H는 메모리를 따로 갖춘 생성형 AI용 제품이어서 후보로 볼 만합니다.

현실적인 순서는 세 단계입니다.

1. 카메라 2~4대의 영상을 가벼운 BEV 또는 점유 네트워크(NPU)로 처리하고, 결과를 3D로 그립니다. 3.1절의 이전 방식을 따라 하는 단계입니다.
2. 카메라 영상을 3D 형상에 입혀서 그립니다. 생성 모델 없이도 리얼리티 뷰와 비슷한 느낌을 낼 수 있습니다.
3. 낮은 해상도의 잠재 공간에서 작은 생성 모델(수천만 파라미터 규모, 적은 단계로 생성하는 consistency 또는 mean-flow 계열)로 시점을 바꿉니다. 테슬라의 구조를 작게 줄여 따라 하는 단계입니다.

테슬라처럼 인코더 하나를 주행과 시각화가 함께 쓰게 하고, 생성 작업을 인코더 18Hz, 출력 36fps처럼 낮은 주기로 여러 사이클에 나누면 엣지 장치에서도 유리합니다. 3D로 재구성하려면 카메라 사이의 시간 차이를 수 ms 안으로 관리해야 합니다. 테슬라 대시캠 영상에서는 카메라 사이의 시간 차이가 최대 8.7ms로 측정된 적이 있습니다. (3자 [S1])

## 8. 확인이 필요한 항목

- 테슬라 릴리스 노트 원문, 머스크와 엘루스와미의 X 게시글 원문
- AI4 칩의 다이 사진, TRIP 구조, 공식 TOPS
- 카메라별 화각, 최대 거리, 프레임레이트의 1차 출처
- 리얼리티 뷰가 입력으로 쓰는 카메라 목록과 출력 해상도(픽셀)
- ICCV 2025 엘루스와미 발표 영상 원본 (https://www.youtube.com/watch?v=wHK8GMc9O5A)

## 9. 출처 목록

- S1: [How Tesla's FSD V15 Reality View Actually Works, Not a Tesla App](https://www.notateslaapp.com/news/5068/how-teslas-fsd-v15-reality-view-actually-works-video)
- S2: [Tesla launches FSD v15, Electrek](https://electrek.co/2026/11/21/tesla-fsd-v15-rollout-25x-safety-reality-view/)
- S3: [Tesla FSD v15 (2026.38.6.11): Rollout Begins, Official Release Notes, Tesla Oracle](https://www.teslaoracle.com/2026/11/22/tesla-fsd-v15-2026-38-6-11-rollout-begins-official-release-notes-and-new-features/)
- S4: [Tesla Reality View takes FSD visualizations to an entirely new level, Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-reality-view/)
- S5: [Tesla FSD v15 uses half of the AI4 chip's TRIP accelerator for real-time AI video, KuCoin News(CoinDesk 전재)](https://www.kucoin.com/news/flash/tesla-fsd-v15-uses-half-of-ai4-chip-s-trip-accelerator-for-real-time-ai-video)
- S6: [Tesla FSD v15's 'Reality View' has one complaint and Musk has the solution, Teslarati](https://www.teslarati.com/tesla-fsd-v15s-reality-drive-one-complaint-musk-solution/)
- S7: [Tesla FSD v15 to go wide release with the 2026 Holiday Update, Tesla Oracle](https://www.teslaoracle.com/2026/11/21/tesla-fsd-v15-to-go-wide-release-with-the-2026-holiday-update-and-mind-bending-reality-renders-in-december-says-musk/)
- S8: [Tesla starts FSD v15 rollout to early testers and leaves HW3 cars out, Startup Fortune](https://startupfortune.com/tesla-starts-fsd-v15-rollout-to-early-testers-and-leaves-hw3-cars-out/)
- S9: [Tesla VP Talks About What's Coming in FSD V15, Not a Tesla App](https://www.notateslaapp.com/news/4679/tesla-ai-chief-talks-about-whats-coming-in-fsd-v15)
- S10: [Gaussian Splatting explained: the tech likely behind Tesla's Reality View, astig.ph](https://astig.ph/gaussian-splatting-explained-tesla-reality-view/)
- S11: [Tesla ICCV 2025 Foundational Model for FSD 요약, Bradford Ferguson (Thread Reader)](https://threadreaderapp.com/thread/1982161471644545387)
- S12: [Tesla Unveils "World Simulator", Tiger Brokers / Deep News](https://www.itiger.com/news/1146546403)
- S13: [FSD Chip, WikiChip](https://en.wikichip.org/wiki/FSD_Chip) (테슬라 2019 Autonomy Day, Hot Chips 31 자료 정리)
- S14: [Tesla's HW4, Behind the Chip (Gidion)](https://behindthechip.substack.com/p/teslas-hw4)
- S15: [Tesla FSD Hardware: HW1 to HW5, TeslAnt](https://teslant.com/en/guides/tesla-fsd-hardware-generations-reference-2026)
- S16: [Tesla AI5 Chip: Specs, Manufacturer, AI4 vs AI5, AutoAkhbar](https://autoakhbar.com/tesla-ai5-chip-specifications-manufacturer-vs-ai4/)
- S17: [How Tesla's FSD Hardware Has Changed Over the Years, Not a Tesla App](https://www.notateslaapp.com/news/3521/how-teslas-fsd-hardware-has-changed-over-the-years)
- S18: [Tesla new camera upgrade, Electrek](https://electrek.co/2025/12/02/tesla-new-camera-upgrade-casting-more-doubt-full-self-driving-promises/) (green의 펌웨어 분석 인용)
- S19: [Tesla Hardware 4 rolling out to new vehicles, AutoPilot Review](https://www.autopilotreview.com/tesla-hardware-4-rolling-out-to-new-vehicles/)
- S20: [Tesla Preparing Upgraded Exterior Cameras as New IMX00N Code Appears, Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-preparing-upgraded-exterior-cameras-as-new-imx00n-code-appears-in-firmware/)
- S21: [Tesla's HW3 and HW4 Cameras: Comparing the Differences, Not a Tesla App](https://www.notateslaapp.com/news/1564/teslas-hw3-and-hw4-cameras-comparing-the-differences-in-quality-and-hardware)
