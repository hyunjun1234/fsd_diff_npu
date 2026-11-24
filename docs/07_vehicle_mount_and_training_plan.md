# 07. 차량 장착과 학습 순환 계획

이 문서는 카메라를 실제 차량에 달아 실시간으로 돌리기 전에 준비할 내용을 정리한 계획서입니다. 타공 없이 카메라를 붙이는 방법, 배선과 전원과 발열, 국내 법규, 테슬라가 모델을 학습시키는 방식, 공개 모델을 우리 영상으로 추가 학습하고 NPU에 다시 올리는 순환까지 순서대로 다룹니다. 카메라 제품 선정과 배선 방식 비교는 [06 문서](06_pi5_camera_selection.md)에 있으므로 여기서는 반복하지 않습니다.

## 1. 네 질문에 대한 답과 근거

### 1.1 타공 없이 카메라를 어떻게 붙이나

처음에는 카메라 4대를 모두 유리 안쪽에 붙이는 것을 권합니다. 전방은 앞유리 상단 룸미러 뒤(블랙박스 자리), 후방은 뒷유리 상단 안쪽, 좌우는 2열 옆유리 안쪽입니다. 이렇게 하면 방수가 필요 없고 차 밖에 아무것도 나오지 않아 법규 문제도 가장 적습니다. 좌우를 차 밖에 다는 것은 유리 안쪽 구성으로 검증을 마친 뒤 필요할 때만 합니다.

차 밖에 다는 경우 자석 거치는 차체 패널이 강철일 때만 붙습니다. 차종에 따라 알루미늄이나 수지 패널이 있어 내 차의 해당 부위 재질은 확인 필요입니다. 흡착 거치는 촬영용 제품이 있습니다. SmallRig 4인치 흡착 마운트는 1/4인치 나사 구멍이 있고 하중 50kg, 흡착 24시간 유지로 표기돼 있지만 [V1], 주행 속도 한계와 낙하 방지 줄에 대한 안내는 제품 페이지에 없어 확인 필요입니다.

외부 부착에 걸리는 국내 법규는 3장에 정리했습니다. 요약하면 자동차관리법은 "자동차에 부착물을 추가하는 것"을 튜닝으로 정의하고 [L1], 성능 기준은 차체 외형이 돌출되어 안전운행에 위험을 주면 안 된다고 정합니다 [L3].

### 1.2 테슬라는 실시간으로 학습하나

아닙니다. 확인한 자료 가운데 차량이 주행 중에 스스로 모델 가중치를 바꾼다고 한 곳은 없습니다. 차량은 추론만 하고, 학습은 데이터 센터에서 합니다. 흐름은 다음과 같습니다.

1. 섀도 모드: 오토파일럿이 차를 제어하지 않을 때도 계산을 병렬로 돌리고, 예측이 운전자의 행동과 어긋나면 카메라와 주행 값을 짧은 스냅샷으로 기록해 올립니다 [T1].
2. 트리거와 캠페인: 엔지니어가 특정 장면을 찾는 검출기를 만들어 차량에 내려보내면 차량이 그 장면을 찾아 올립니다 [T1]. 테슬라 특허 분석에서는 이 트리거 분류기가 본 주행 시스템과 따로 돌며, 몇 분 안에 새 트리거를 배포해 수집을 시작할 수 있다고 설명합니다 [T2]. 이 내용은 특허 기준이라 실제 운영과 같은지는 확인 필요입니다.
3. 오프라인 자동 라벨링: 올라온 클립은 서버에서 라벨링합니다. 서버의 오프라인 모델은 과거 프레임뿐 아니라 미래 프레임까지 볼 수 있어 차 안의 실시간 모델보다 정확한 정답을 만들 수 있고, 같은 장소를 여러 차가 지나간 영상을 합쳐 도로 표면을 재구성합니다 [T3]. 여러 차량의 데이터를 합친 3D 모델로 새 영상에 라벨을 붙이는 특허도 있습니다 [T4].
4. 시뮬레이션: 라벨링이 어려운 장면이나 드문 장면은 시뮬레이션으로 보충합니다 [T3]. v15에서는 데이터 센터의 뉴럴 월드 시뮬레이터를 평가와 강화학습에 씁니다([01 문서](01_fsd_v15_reality_view_research.md) 3.2절).
5. 학습과 배포: 학습은 대형 클러스터에서 합니다. 위키백과 정리에 따르면 Dojo 이전에 A100 5,760개 GPU 클러스터를 썼고 자동 라벨링 전용 1,752개 GPU 클러스터가 따로 있었으며, 학습 데이터 예로 36fps 10초 클립 100만 개(1.5PB)가 언급됩니다 [T5]. 학습된 모델은 소프트웨어 업데이트로 차량에 배포됩니다.

FSD v12부터는 영상에서 바로 조향과 가속을 내는 엔드투엔드 신경망을 씁니다 [T6]. 그래서 지금은 "사람이 어떻게 운전했는가"가 담긴 주행 클립 자체가 학습 데이터입니다.

### 1.3 공개 모델에 추가 학습해서 써도 되나

됩니다. 지금 파이에서 돌리는 YOLOv8s 검출과 BiSeNetV2 세그는 COCO, Cityscapes 같은 공개 데이터로 학습된 모델이고, 우리 차량 영상으로 미세조정한 뒤 다시 NPU용으로 변환하면 됩니다. 두 NPU 회사 모두 이 흐름을 공식으로 안내합니다. Hailo Model Zoo는 모델마다 재학습 안내(도커 환경, 학습, 내보내기)를 제공하고 [H1], DEEPX는 ONNX와 설정 파일과 보정 이미지로 .dxnn을 만드는 흐름을 안내합니다 [D1].

주의할 점이 세 가지 있습니다. 첫째, 학습과 변환은 파이에서 할 수 없고 x86 리눅스 PC가 필요합니다(6장). 둘째, Ultralytics YOLO는 AGPL-3.0이라 결과물을 공개하지 않는 상업 용도에는 별도 라이선스가 필요합니다 [U1]. 개인 연구 프로젝트에서는 문제 되지 않습니다. 셋째, FlashOcc처럼 3D 점유를 내는 모델은 nuScenes의 라이다 기반 정답으로 학습되고 깊이 감독도 라이다를 씁니다 [B1]. 라이다가 없는 우리 차에서는 같은 품질의 정답을 만들기 어려워, 추가 학습은 2D 검출과 세그부터 하는 것이 맞습니다.

### 1.4 학습이 끝난 모델을 계속 학습시킬 수 있나

됩니다. 지난번 학습 결과(best.pt)를 시작점으로 새 데이터를 더해 다시 학습하면 됩니다. 다만 두 가지 제약이 있습니다.

새 데이터로만 학습하면 예전에 잘하던 것을 잊는 문제가 생깁니다. 연속 학습 분야에서는 이를 파국적 망각이라 부르고, 이전 데이터를 일부 섞어 다시 보여 주는 재현(replay)이 대표적인 완화 방법입니다 [C1]. 그래서 매 회차마다 이전 데이터 일부를 섞고, 고정된 시험 세트로 예전 성능이 떨어지지 않았는지 확인합니다.

또 Hailo-8과 DX-M1은 추론 전용입니다. Hailo는 런타임을 "컴파일된 모델의 실시간 추론"용으로, 컴파일러를 "오프라인 컴파일"용으로 설명합니다 [H2]. 그래서 학습할 때마다 서버에서 INT8 양자화와 컴파일을 다시 하고 파이에 새 모델 파일을 올려야 합니다. 차에서 바로 배우는 구조는 테슬라도 쓰지 않습니다.

Ultralytics에서 `resume`은 끊긴 학습을 옵티마이저 상태와 에폭까지 그대로 이어 가는 기능이고, 새 데이터로 계속 학습할 때는 이전 가중치를 시작점으로 새 학습을 여는 쪽입니다 [U2][U3].

## 2. 차량 장착

### 2.1 자리별 장착 방법

| 자리 | 장착 위치 | 고정 방법 |
|---|---|---|
| 전방 | 앞유리 상단, 룸미러 뒤 | 패널 마운트와 유리용 양면테이프 |
| 후방 | 뒷유리 상단 안쪽 | 패널 마운트와 양면테이프 |
| 좌 | 2열 옆유리 앞쪽 안쪽 | 흡착 거치대 |
| 우 | 2열 옆유리 앞쪽 안쪽 | 흡착 거치대 |

Camera Module 3에는 Adafruit 패널 마운트 키트(M2 나사 포함)를 쓰면 기판을 판에 고정할 수 있고 [K1], 그 판을 유리에 붙입니다. 좌우는 2열 유리에 흡착 거치대로 붙입니다. 이 유리는 대부분 내려가는 창이므로 시험 중에는 창문 잠금 버튼으로 조작을 막아 둡니다.

유리 안쪽에 달 때 생기는 문제와 대책은 이렇습니다. 대시보드가 앞유리에 비치므로 렌즈 둘레에 검은 차광 후드를 둘러 반사를 줄입니다. 측후면 유리 틴팅은 빛을 줄여 야간 화질이 떨어지므로, 좌우와 후방은 저조도 센서(06 문서의 IMX462 계열)가 더 중요해집니다. 앞유리 위쪽은 와이퍼가 닦지 않는 영역이 있어 비 오는 날 전방 카메라 화면이 가려지는지 첫 시험에서 확인합니다.

카메라를 붙인 뒤에는 차에서 외부 보정을 다시 해야 합니다. 위치가 테슬라 자리와 달라진 만큼 `configs`의 카메라 좌표와 각도를 실측값으로 바꾸고, 보정 절차는 [모형 보정 문서](phase1_model_rig_calibration.md)를 차량 크기로 늘려 그대로 씁니다.

### 2.2 배선

Pi를 운전석과 조수석 사이 콘솔이나 조수석 아래에 두면 배선은 2~4m입니다(06 문서 추정). 전방과 후방 CSI 카메라는 CSI-HDMI 확장으로 늘립니다(06 문서). 문틈이나 트림 틈으로 넣기 쉬운 납작한 HDMI 케이블로는 RS PRO 5m 플랫 케이블(4K 60Hz 5m까지 지원 표기)이 국내에서 재고가 있습니다 [K2]. 좌우 USB 카메라도 납작한 USB 연장선을 쓰면 문 고무 실링 사이로 지나가게 할 수 있지만, 문을 닫을 때 눌려 끊기지 않는지 확인해야 합니다.

케이블은 A필러 트림과 천장 내장재 틈으로 밀어 넣는 방식이 일반적입니다. A필러 안에는 커튼 에어백이 있는 차가 많아, 케이블이 에어백 전개를 막지 않게 하는 방법은 차종 정비 지침 확인 필요입니다. 확실하지 않으면 A필러를 피하고 천장 가장자리와 문 실링을 따라 돌립니다.

## 3. 법규와 안전

아래 조문은 국가법령정보센터에서 직접 확인했습니다.

자동차관리법 제2조 제11호는 "자동차의 튜닝"을 구조나 장치를 바꾸거나 "자동차에 부착물을 추가하는 것"으로 정의하고 [L1], 제34조 제1항은 국토교통부령으로 정한 항목을 튜닝할 때 시장, 군수, 구청장의 승인을 받도록 합니다 [L1]. 카메라를 일시적으로 붙이는 것이 승인 대상 항목에 들어가는지는 확인 필요입니다. 블랙박스처럼 실내 유리에 붙이는 방식이 가장 안전한 이유가 이것입니다.

자동차 및 자동차부품의 성능과 기준에 관한 규칙 제19조 제6항은 차체 외형이 날카롭게 각이 지거나 돌출되어 안전운행에 위험을 줄 우려가 있으면 안 된다고 정합니다 [L3]. 차 밖 흡착 카메라는 여기에 걸릴 수 있습니다.

도로교통법 제39조 제4항은 운전 중 실은 화물이 떨어지지 않게 확실히 고정하도록 합니다 [L2]. 카메라는 화물이 아니지만, 고속에서 떨어진 부착물은 뒤차에 위험하므로 차 밖에 단다면 낙하 방지 줄을 별도로 다는 것을 원칙으로 합니다.

도로교통법 제49조 제1항 제11호는 운전 중 영상표시장치로 영상을 보는 것을 금지하고, 예외로 지리안내나 교통정보안내 영상, 차의 좌우나 전후방을 볼 수 있게 돕는 영상을 둡니다 [L2]. 같은 항 제11호의2는 운전 중 영상표시장치 조작을 금지합니다 [L2]. 리얼리티 뷰 화면이 이 예외에 해당하는지는 확인 필요입니다. 그래서 차량 시험은 운전자가 화면을 보지 않고, 동승자가 노트북으로 실행과 확인을 맡는 방식으로 합니다. 같은 항 제3호의 창유리 가시광선 투과율 기준도 있으므로 [L2] 앞유리와 운전석 옆유리 시야를 가리는 위치에는 카메라를 두지 않습니다.

## 4. 전원

### 4.1 공급 방식

라즈베리파이 5의 권장 전원은 5V 5A입니다 [P1]. 파이 5는 USB PD로 5V 5A를 요청하고, PD가 없는 전원이면 5V 3A로 간주해 USB 포트 전체 전류를 1.6A 대신 600mA로 제한합니다 [P1]. 이러면 USB SSD와 USB 카메라가 제대로 동작하지 않을 수 있고, 5V 5A가 아니면 USB 부팅도 꺼집니다 [P1].

차량 12V를 5V 5A로 바꾸는 방법은 두 가지입니다. 첫째는 PD 5V 5A를 지원하는 변환기를 쓰는 것이고, 둘째는 PD가 없는 강압 컨버터를 쓰고 EEPROM의 `PSU_MAX_CURRENT=5000`으로 5A를 가정하게 하는 것입니다 [P1]. 국내에서는 오토크루 차량용 12V/24V to 5V 5A 강압 모듈이 17,270원(부가세 포함)입니다 [K3]. 출력 단자가 USB-C인지, 파이 5의 5V 핀으로 넣어야 하는지는 상세 이미지에서 확인 필요입니다.

강압 컨버터를 쓸 때는 전선이 굵고 짧아야 합니다. Pololu 포럼 사례에서 브레드보드용 점퍼선을 거쳐 파이 5에 전원을 넣자 부하에서 4.7V까지 떨어졌고, 전선을 보드에 직접 납땜하자 4.96V 아래로 내려가지 않았습니다 [P2].

### 4.2 시동과 함께 켜고 끄기

시동을 끄면 파이가 갑자기 꺼지며 저장 중이던 영상과 SD 카드가 손상될 수 있습니다(공식 수치는 찾지 못해 확인 필요). 그래서 시동(ACC) 신호를 감지해 안전하게 종료하는 회로를 둡니다. 완제품으로는 CarPiHAT PRO 5가 있으며, 파이 5에 5A까지 공급하고 12V 입력을 감시해 전원이 빠지면 정상 종료하는 기능을 갖췄습니다 [P3]. 가격은 116.80파운드이고 품절 표시가 있었습니다 [P3]. 국내 판매처는 찾지 못해 확인 필요입니다. 직접 만든다면 ACC 12V를 포토커플러로 GPIO에 넣고, 꺼짐을 감지하면 `shutdown`을 실행하며, 그동안 버틸 작은 UPS나 슈퍼커패시터를 둡니다.

시거잭은 대부분 ACC 전원이지만 차종에 따라 상시 전원인 차도 있어 확인 필요입니다. 퓨즈박스에서 전원을 따는 블랙박스 상시전원 케이블 방식도 있으나, 첫 시험에서는 시거잭으로 시작하고 배터리 방전을 막기 위해 주차 중에는 장비를 끕니다.

### 4.3 소비전력

Hailo-8 칩은 일반 소비전력 2.5W [N1], DX-M1 M.2 모듈은 최대 5W입니다 [N2]. 파이 5와 USB 카메라 2대, USB SSD를 더한 차량 구성의 실제 소비전력은 측정 필요이며, 차에 달기 전에 책상에서 USB 전력계로 잽니다.

## 5. 발열과 진동

### 5.1 발열

라즈베리파이 5의 동작 온도는 0~70도입니다 [P4]. DX-M1은 상용 등급 0~70도, 산업용 -40~85도 [N2], Hailo-8 M.2 모듈은 -40~85도입니다 [N1]. 우리 DX-M1이 어느 등급인지는 확인 필요입니다.

여름 주차 차량은 이 범위를 쉽게 넘습니다. 미국 애리조나주립대 시험에서 햇볕 아래 주차한 차는 한 시간 만에 실내 평균 46.7도, 대시보드 평균 69.4도였습니다(화씨 116도, 157도) [E1]. 스탠퍼드 시험에서는 바깥 기온과 관계없이 한 시간 안에 실내가 평균 약 22도 올랐고 상승분의 80%가 처음 30분에 일어났습니다 [E2].

여기서 나오는 원칙은 네 가지입니다. 파이와 NPU를 대시보드나 뒷선반처럼 햇볕을 받는 곳에 두지 않고, 콘솔이나 좌석 아래 그늘에 둡니다. 여름에는 에어컨으로 실내를 식힌 뒤 켭니다. 주차할 때는 장비를 꺼내 두거나 최소한 꺼 둡니다. 그리고 지금 방열 없이 과열되는 두 NPU에 방열판과 팬을 다는 일(진행 상황 문서의 사용자 할 일 1, 2번)이 차량 시험의 선행 조건입니다. 실행 중에는 기존 `--hailo-stop-temp`와 `scripts/dx_temp.py`로 온도를 기록하고 넘으면 멈춥니다.

### 5.2 진동

차량 진동에 대한 별도 측정이나 규격 확인은 아직 하지 않았으므로, 아래는 일반적인 대책입니다. SD 카드 대신 USB SSD에 시스템과 녹화 파일을 둡니다(M.2 슬롯 두 개는 NPU가 쓰고 있습니다). 듀얼 M.2 HAT의 고정 나사와 스페이서를 확인하고 풀림 방지제를 씁니다. HDMI와 USB 커넥터는 케이블 무게가 커넥터에 걸리지 않게 근처에서 케이블을 묶어 고정합니다. 파이는 케이스에 넣어 벨크로나 고무 받침으로 바닥에 고정해 급제동 때 날아가지 않게 합니다.

## 6. 학습 서버

Hailo Dataflow Compiler는 Ubuntu 22.04나 24.04 64비트(WSL2 가능)와 NVIDIA GPU(compute capability 6.1~8.9, VRAM 8GB 이상, 16GB 권장)를 요구합니다 [H3]. GPU가 없어도 돌지만 최적화 수준이 0으로 떨어져 실사용 모델에는 권하지 않습니다 [H3]. compute capability 8.9는 RTX 40 계열까지이고, 블랙웰 세대(10.0, 12.0)는 지원하지 않는다고 명시돼 있습니다 [H3]. 그래서 RTX 50 계열 PC는 Hailo 변환에 쓸 수 없습니다. 또 Model Zoo의 master 브랜치는 Hailo-10과 15 전용이고, Hailo-8은 v2.x 브랜치와 DFC 3.x를 써야 합니다 [H1].

DEEPX DX-COM은 x86_64 전용이고 Ubuntu 20.04~26.04 등을 지원하며 GPU는 필수가 아닙니다 [D1][D2].

두 컴파일러가 모두 x86 전용이므로 Jetson Orin 같은 ARM 보드로 바꿔도 Hailo와 DX-M1 모델 변환은 여전히 x86 PC에서 해야 합니다(DX-COM은 x86_64 전용 확인 [D2], DFC의 ARM 지원 여부는 확인 필요). 학습 서버는 RTX 40 계열 이하 GPU가 달린 x86 Ubuntu PC 한 대로 학습, 자동 라벨링, 두 NPU 변환을 모두 하는 구성이 가장 단순합니다. 클라우드 GPU를 빌릴 때도 같은 조건(compute capability 8.9 이하)을 확인합니다.

## 7. 데이터 수집

수집은 실시간 시연보다 먼저 합니다. 첫 차량 주행에서는 NPU 추론 없이 카메라 4대 원본과 타임스탬프만 녹화하고(`scripts/record_drive.py`, 사용법은 docs/drive_recording.md), 이 영상으로 보정, 동기화, 라벨링, 학습을 모두 차 밖에서 진행합니다. 실시간 실행은 녹화 영상으로 `--replay` 결과가 만족스러운 뒤에 합니다.

장면은 낮, 밤, 비, 터널 출입, 역광, 주차장, 교차로를 고르게 모읍니다. 모든 회차의 영상 가운데 일부(예를 들어 장면 종류별로 고른 200프레임)는 학습에 절대 쓰지 않는 고정 시험 세트로 따로 둡니다. 이 세트가 회차마다 성능을 비교하는 기준이 됩니다.

테슬라의 트리거를 작게 흉내 낼 수 있습니다. 파이에서 실시간 실행 중에 검출 점수가 애매한 프레임, 추적 ID가 바뀌는 순간, 두 NPU 결과가 크게 어긋나는 프레임(지금도 두 NPU로 같은 YOLOv8s를 돌려 비교하고 있습니다)을 앞뒤 몇 초와 함께 따로 저장하면, 모델이 약한 장면만 골라 모을 수 있습니다.

## 8. 자동 라벨링

자동 라벨링은 가능합니다. 테슬라처럼 차에서가 아니라 학습 서버에서 큰 모델로 정답 후보를 만들고, 사람이 검수하는 방식입니다. 2D 정답은 자동으로 상당 부분 만들 수 있고, 라이다 없는 3D 정답은 어렵고 검수 부담이 큽니다.

### 8.1 무엇으로 무엇을 붙이나

| 정답 | 도구 | 라이선스 |
|---|---|---|
| 차, 사람 박스 | Grounding DINO | Apache-2.0 |
| 도로, 물체 마스크 | SAM 2 | Apache-2.0 |
| 영상 내 추적 | Grounded-SAM-2 | Apache-2.0 외 |
| 상대 깊이 | Depth Anything V2 Small | Apache-2.0 |
| 정지 장면 3D | COLMAP | BSD |

Grounding DINO는 이미지와 문장("car . person . truck .")을 함께 받아 박스를 냅니다 [A1]. 학습 코드는 공개되지 않아 [A1] 우리 데이터로 고칠 수는 없고 라벨러로만 씁니다. SAM 2는 이미지와 영상에서 점이나 박스를 주면 마스크를 만들고, 영상에서는 그 마스크를 다음 프레임으로 이어 갑니다 [A2]. Grounded-SAM-2는 이 둘을 묶어 문장으로 지정한 물체를 영상 전체에서 추적합니다 [A3]. autodistill은 이런 큰 모델로 이미지에 라벨을 붙이고 그 데이터로 YOLOv8 같은 작은 모델을 학습시키는 과정을 명령 한 줄로 묶어 둔 도구입니다 [A4].

깊이는 Depth Anything V2를 씁니다. Small(24.8M 파라미터)만 Apache-2.0이고 Base, Large는 비상업 라이선스(CC-BY-NC-4.0)입니다 [A5]. 기본 모델은 상대 깊이이고 미터 단위 모델은 따로 있습니다 [A5].

### 8.2 라이다 없이 3D 정답을 만드는 방법

세 가지를 조합합니다. 셋 다 실험 단계이며 정확도는 우리 데이터로 측정해야 합니다.

1. 지면 평면 투영: 보정된 카메라 높이와 각도로 박스 아래 끝을 바닥에 투영해 거리를 냅니다. 지금 1단계 데모가 쓰는 방식이고, 모형 보정 후 거리 오차를 잴 예정입니다.
2. 여러 프레임 재구성: 차가 움직이며 찍은 영상에서 COLMAP으로 정지한 배경(도로, 연석, 건물)의 3D를 만들고 [A6], SAM 2 마스크로 움직이는 물체를 빼냅니다. 테슬라가 미래 프레임까지 쓰는 오프라인 라벨링, 여러 번 지나간 영상을 합치는 재구성과 같은 원리입니다 [T3][T4].
3. 깊이 모델 보정: Depth Anything V2의 상대 깊이를 1번의 지면 거리나 2번의 재구성 점으로 축척을 맞춰 미터 단위로 바꿉니다.

이렇게 만든 3D 정답은 FlashOcc 재학습에 쓸 만큼 정확할지 확인 필요입니다. 그래서 계획은 2D 검출과 세그 미세조정을 먼저 하고, 3D는 거리 오차 측정과 비교 실험으로만 다룹니다.

### 8.3 검수

검수는 CVAT 커뮤니티판(자체 설치, 무료)으로 합니다. YOLO 형식을 내보낼 수 있고 Nuclio 서버리스 함수로 자동 라벨 모델을 연결할 수 있습니다 [A7]. 자동 라벨을 CVAT에 불러와 사람이 고치는 방식입니다.

전부 볼 수는 없으므로 두 가지를 봅니다. 하나는 점수가 애매한 박스와 두 라벨러 결과가 어긋나는 프레임처럼 자동으로 걸러 낸 의심 프레임이고, 다른 하나는 무작위 표본입니다. 처음 회차에는 사람이 직접 칠한 200프레임과 자동 라벨을 비교해 박스 IoU와 놓친 비율을 재고, 그 결과로 검수 비율을 정합니다. 7장에서 말한 고정 시험 세트는 자동 라벨이 아니라 사람이 직접 라벨링합니다.

## 9. 추가 학습과 NPU 배포

### 9.1 학습

| 모델 | 시작점 | 학습 데이터 |
|---|---|---|
| YOLOv8s 검출 | COCO 사전학습 | 우리 영상 자동 라벨 + COCO 일부 |
| BiSeNetV2 도로 세그 | 공개 사전학습 | 우리 영상 SAM 2 도로 마스크 |
| 깊이 (선택) | yolo26-depth-n 등 | 8.2절 깊이 정답 |

YOLO는 Ultralytics로 `model=yolov8s.pt`에서 시작해 학습합니다 [U3]. COCO 데이터 일부를 섞는 이유는 1.4절의 망각을 막기 위해서입니다.

### 9.2 NPU 변환

Hailo는 ONNX를 parse, optimize(보정 이미지로 INT8 양자화), compile 순서로 HEF로 만듭니다 [H1]. Model Zoo에 있는 구조를 재학습했다면 `hailomz` 명령을 쓰고, 그 밖의 모델은 Dataflow Compiler를 직접 써야 합니다 [H3]. DEEPX는 ONNX, JSON 설정, 보정 이미지로 .dxnn을 만듭니다 [D1]. 두 경우 모두 보정 이미지는 우리 차량 영상에서 장면 종류별로 고릅니다.

변환할 때마다 서버의 FP32 모델과 NPU의 INT8 모델을 고정 시험 세트에서 비교해 정확도 하락을 기록합니다. Hailo 포럼에는 사용자 YOLOv8을 변환할 때 보정 데이터 정규화 문제로 양자화 오류가 났고 출력층 범위를 지정해 해결한 사례가 있습니다 [H4].

### 9.3 배포와 되돌리기

모델 파일 이름에 회차 번호를 붙여 파이의 `models/` 아래에 쌓고, 설정 파일에서 어떤 회차를 쓸지 고릅니다. 새 모델은 먼저 테슬라의 섀도 모드처럼 기존 모델과 함께 돌려 결과만 기록합니다. 마침 NPU가 두 개라 한쪽에 기존 모델, 다른 쪽에 새 모델을 올려 같은 프레임을 비교할 수 있습니다. 문제가 없으면 기본값을 바꾸고, 문제가 생기면 설정만 이전 회차로 돌립니다.

## 10. 이어서 학습하는 순환

한 회차는 아래 순서로 돕니다.

1. 수집: 주행 녹화와 트리거 클립을 서버로 옮깁니다(7장).
2. 라벨링: 자동 라벨 후 의심 프레임과 표본을 검수합니다(8장).
3. 학습: 직전 회차 best.pt에서 시작하고, 이전 회차 데이터 일부와 COCO 일부를 섞습니다(9.1절).
4. 평가: 고정 시험 세트 전체와 이번 회차 새 장면에서 비교합니다. 예전 장면 성능이 직전 회차보다 떨어지면 섞는 비율을 늘려 다시 학습합니다.
5. 변환: Hailo HEF와 DEEPX .dxnn을 만들고 INT8 하락폭을 기록합니다(9.2절).
6. 섀도 실행과 교체: 두 NPU에 나눠 비교한 뒤 교체합니다(9.3절).

회차마다 데이터 수, 검수 시간, 시험 세트 점수, INT8 하락폭을 `docs/measurements/`에 남겨, 다음 회차에 무엇을 더 모을지 정하는 근거로 씁니다.

## 11. 준비 순서

1. 책상에서: 두 NPU 방열 장착, USB 전력계로 소비전력 측정, 12V 강압 컨버터와 `PSU_MAX_CURRENT` 설정 시험, ACC 꺼짐을 흉내 낸 안전 종료 시험.
2. 학습 서버 마련: RTX 40 계열 이하 GPU가 달린 x86 Ubuntu PC를 정하고, 지금 쓰는 YOLOv8s를 그대로 HEF와 .dxnn으로 다시 만들어 파이 결과와 같은지 확인합니다.
3. 정차 상태 장착: 주차장에서 카메라 4대를 유리 안쪽에 붙이고 배선한 뒤 외부 보정을 합니다.
4. 녹화 전용 주행: 동승자가 조작하고 추론 없이 녹화만 합니다.
5. 차 밖 처리: 녹화 영상으로 `--replay` 실행, 자동 라벨링, 첫 미세조정 회차를 돌립니다.
6. 실시간 주행: 동승자 조작으로 실시간 화면을 확인하고 트리거 클립을 모읍니다.
7. 이후 10장의 순환을 반복합니다.

## 12. 장비와 판매처

06 문서에 있는 카메라와 확장 모듈은 빼고, 이 문서에서 새로 필요한 것만 적었습니다.

| 제품 | 가격 | 판매처 |
|---|---|---|
| 카메라 모듈 3 패널 마운트 키트 | 9,900원 | 디바이스마트 [K1] |
| RS PRO 플랫 HDMI 5m | 36,820원 | RS코리아 [K2] |
| 오토크루 12V/24V to 5V 5A 강압 모듈 | 17,270원 | 디바이스마트 [K3] |
| SmallRig 4인치 흡착 마운트 | 확인 필요 | SmallRig 공식몰 [V1] |
| CarPiHAT PRO 5 | 116.80파운드 | The Pi Hut [P3] |

국내 가격은 부가세 포함입니다. RS코리아 가격은 "참고 가격"으로 표시돼 있습니다 [K2]. SmallRig 흡착 마운트와 CarPiHAT PRO 5는 국내 판매처를 찾지 못했습니다. 유리용 양면테이프, 차광 후드, 납작한 USB 연장선, 낙하 방지 줄, USB 전력계는 제품을 아직 정하지 않았습니다.

## 출처

- A1: [IDEA-Research/GroundingDINO, GitHub](https://github.com/IDEA-Research/GroundingDINO)
- A2: [facebookresearch/sam2, GitHub](https://github.com/facebookresearch/sam2)
- A3: [IDEA-Research/Grounded-SAM-2, GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- A4: [autodistill, GitHub](https://github.com/autodistill/autodistill)
- A5: [Depth-Anything-V2, GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- A6: [COLMAP, GitHub](https://github.com/colmap/colmap)
- A7: [CVAT, GitHub](https://github.com/cvat-ai/cvat)
- B1: [BEVDet, GitHub](https://github.com/HuangJunJie2017/BEVDet)
- C1: [Parisi 외, Continual Lifelong Learning with Neural Networks: A Review, arXiv 1802.07569](https://arxiv.org/abs/1802.07569)
- D1: [DEEPX dx-all-suite, Running Your First NPU Model](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/03_Running_Your_First_NPU_Model.md)
- D2: [DEEPX dx-compiler, GitHub](https://github.com/DEEPX-AI/dx-compiler)
- E1: [Study: Hot cars can hit deadly temperatures in as little as one hour, ASU News](https://news.asu.edu/20180516-discoveries-asu-study-hot-cars-can-hit-deadly-temperatures-within-one-hour)
- E2: [Parked cars get dangerously hot, even on cool days, Stanford Medicine](https://med.stanford.edu/news/all-news/2005/07/parked-cars-get-dangerously-hot-even-on-cool-days-stanford-study-finds.html)
- H1: [Hailo Model Zoo, GitHub](https://github.com/hailo-ai/hailo_model_zoo)
- H2: [Hailo AI Software Suite](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/)
- H3: [Hailo Model Zoo, Getting Started](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst)
- H4: [Model training GPU, Hailo Community](https://community.hailo.ai/t/model-training-gpu/13623)
- K1: [디바이스마트, Panel Mount Kit for Raspberry Pi Camera Module 3](https://www.devicemart.co.kr/goods/view?no=15014487)
- K2: [RS코리아, RS PRO 5m 플랫 HDMI 케이블 286-3325](https://kr.rs-online.com/web/p/hdmi-cables/2863325)
- K3: [디바이스마트, 오토크루 차량용 12V/24V to 5V 5A 강압 모듈](https://www.devicemart.co.kr/goods/view?no=15600349)
- L1: [자동차관리법, 국가법령정보센터](https://www.law.go.kr/법령/자동차관리법)
- L2: [도로교통법, 국가법령정보센터](https://www.law.go.kr/법령/도로교통법)
- L3: [자동차 및 자동차부품의 성능과 기준에 관한 규칙, 국가법령정보센터](https://www.law.go.kr/법령/자동차및자동차부품의성능과기준에관한규칙)
- N1: [Hailo-8 M.2 모듈](https://hailo.ai/products/ai-accelerators/hailo-8-m2-ai-acceleration-module/), [Hailo-8 칩](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/)
- N2: [DEEPX DX-M1](https://deepx.ai/products/dx-m1/)
- P1: [USB Power Delivery on Raspberry Pi 5, Raspberry Pi 백서 RP-009856](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-009856-WP-1-USB%20Power%20delivery%20on%20Raspberry%20Pi%205.pdf)
- P2: [Low voltage on Pi 5 with Pololu D36V50F5, Pololu 포럼](https://forum.pololu.com/t/low-voltage-on-pi-5-with-pololu-d36v50f5-step-down-voltage-regulator/29212)
- P3: [CarPiHAT PRO 5, The Pi Hut](https://thepihut.com/products/carpihat-pro-5-car-interface-dac-for-raspberry-pi-5)
- P4: [Raspberry Pi 5 product brief](https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008348-DS-10-raspberry-pi-5-product-brief.pdf)
- T1: [Tesla's Autopilot Depends on a Deluge of Data, IEEE Spectrum](https://spectrum.ieee.org/tesla-autopilot-data-deluge)
- T2: [How Tesla Continuously and Automatically Improves Autopilot, Code Compass (특허 EP3850549A1 해설)](https://codecompass00.substack.com/p/tesla-data-engine-trigger-classifiers)
- T3: [Tesla AI Day 2021 Review Part 2: Training Data, Towards Data Science](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0)
- T4: [How Tesla Will Automate Data Labeling for FSD, Not a Tesla App (특허 WO2024073033A1 해설)](https://notateslaapp.com/news/2455/how-tesla-will-automate-data-labeling-for-fsd)
- T5: [Tesla Dojo, 위키백과](https://en.wikipedia.org/wiki/Tesla_Dojo)
- T6: [Tesla Autopilot, 위키백과](https://en.wikipedia.org/wiki/Tesla_Autopilot)
- U1: [ultralytics/ultralytics, GitHub](https://github.com/ultralytics/ultralytics)
- U2: [Ultralytics Train 문서](https://docs.ultralytics.com/modes/train/)
- U3: [Ultralytics Model Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
- V1: [SmallRig 4인치 흡착 마운트 4122C](https://www.smallrig.com/4-Suction-Cup-Camera-Mounting-Support-for-Vehicle-Shooting-4122C.html)
