# 진행 상황

이 문서는 라즈베리파이 5, Hailo-8, DX-M1 구현 작업의 현재 진행 상황입니다. 다른 사람이나 다른 세션이 이 문서만 읽고 작업을 이어받을 수 있게, 실행이 끝날 때마다 갱신해 main에 올립니다. 단계와 완료 기준은 [구현 계획](03_pi5_hailo8_dxm1_implementation_plan.md)을 따릅니다.

## 지금 위치

1단계는 코드와 측정을 마쳤고 사람이 하는 보정만 남았습니다. 2단계는 DEEPX 공식 .dxnn(YOLOv8s PPU, BiSeNetV2 세그)으로 검출 비교와 동시 실행까지 마쳤고, 도로 IoU 측정만 남았습니다. 카메라 위치 기준값은 테슬라 모델 Y 배치로 바꿨습니다(전방 ov5647은 앞유리 상단 전방 광각 자리, 후방 imx708은 뒷번호판 위). 지금은 사용자의 모형 보정을 기다립니다. 4단계 준비로 BEVPool CPU 구현을 마쳤습니다.

| 단계 | 상태 |
|---|---|
| 0단계 장비 확인 | 끝남 (DX-M1 단독 FPS, 30분 동시 시험 남음) |
| 1단계 Hailo 실기 | 보정과 거리 오차 측정 남음 |
| 2단계 DX-M1 추가 | 도로 IoU 측정 남음 |
| 4단계 준비 | BEVPool CPU 구현 끝남 |

## 장비 구성

라즈베리파이 5(Debian 13, 64비트)에 Seeed 듀얼 M.2 HAT로 Hailo-8과 DX-M1 4GB를 함께 달았고, 두 NPU 모두 PCIe Gen3 x1로 인식됩니다. 런타임은 HailoRT 4.23.0, DXRT 3.4.2이고, `dx_engine`은 파이 저장소의 `.venv`에만 있습니다.

카메라는 계획의 Camera Module 3 Wide 2대가 아니라 지금 달린 카메라를 계속 씁니다. 0번은 ov5647(전방, 사용자 확인 기준 130도 광각, IR 컷, 야간 가능)이고 1번은 일반 Camera Module 3(imx708, 후방)입니다. 설정은 [configs/two_cam_current.yaml](../configs/two_cam_current.yaml)이고, 내부 파라미터는 보정 전 근사값입니다. 보정과 시험은 실제 차량이 아니라 모형으로 먼저 하고, 카메라 위치는 테슬라 모델 Y에서 같은 역할의 카메라 자리([04 문서](04_tesla_camera_layout.md))를 기준으로 둡니다. 모형은 그 배치를 축척으로 줄인 설정([configs/two_cam_model.yaml](../configs/two_cam_model.yaml))을 쓰고, 보정 절차는 [모형 보정 문서](phase1_model_rig_calibration.md)에 있습니다.

DX-M1은 부하가 없을 때도 약 72도이고, 짧은 벤치를 여러 번 돌린 뒤 89도까지 오른 적이 있습니다. DX-M1을 오래 돌리기 전에 방열 상태 확인이 필요합니다.

## 작업 방식

맥북의 감독 세션이 목표를 작업 파일로 써서 `~/Desktop/App/fsd_agent.py`(Claude Agent SDK, 사용자 API 크레딧 사용)에 넘기고, 결과를 검토한 뒤 커밋해 main에 올립니다. 넘긴 작업 파일은 [agent_tasks](agent_tasks/)에 있습니다. 실행 명령은 아래와 같고, 로그는 맥의 `out/agent_logs/`에 남습니다.

```bash
source ~/.venvs/claude-agent/bin/activate
cd ~/Desktop/App/fsd_diff_npu && python ~/Desktop/App/fsd_agent.py --task-file docs/agent_tasks/06_dxnn_stage2.txt --budget 15
```

파이는 맥북 ssh 설정의 `pi5`로 키 인증 접속하고, 파이의 저장소는 `~/work/fsd_diff_npu`, 가상환경은 그 안의 `.venv`입니다. 맥에서 고친 코드는 rsync로 파이에 보냅니다(파이 쪽은 git이 아닙니다). 맥의 테스트는 `/usr/bin/python3 -m pytest -q`로 돌립니다.

크레딧이 떨어져 `fsd_agent.py`가 멈추면, 아래 "다음에 할 일"의 작업 파일을 Claude Code에서 직접 읽혀 이어 가면 됩니다.

## 측정 결과 요약

모두 파이 5, 카메라 2대, 창 없는 실행 기준입니다. 자세한 값과 조건은 [measurements](measurements/)에 있습니다.

| 항목 | 값 |
|---|---|
| Hailo-8 YOLOv8s 단독 (hailortcli) | 284 FPS |
| 1단계 데모 기본 설정 | 14.2 FPS |
| 마스크 렌더링 개선 후 Hailo 단독 | 26.2 FPS |
| 개선 후 Hailo 검출 + DX-M1 세그 동시 | 21.4 FPS |
| BEVPool C 구현, 카메라 4대 (스레드 1개) | 7.8ms |
| Hailo 검출 + DX-M1 BiSeNetV2 세그 동시 | 19.6 FPS |
| 두 NPU YOLOv8s 박스 일치 (IoU 평균) | 0.973 |

1단계 기본 설정에서 병목은 NPU(약 18ms)가 아니라 CPU의 BEV 합성(23ms)과 렌더링(28ms)이었습니다. 도로 마스크 덧칠을 OpenCV로 바꿔 카메라 2대 기준 39.5ms에서 5.8ms로 줄였습니다. BEVPool은 입력 256x704, D=88, C=64, BEV 100x100 조건이고 NumPy로는 약 110ms라 C 구현(ctypes, OpenMP)을 씁니다. 21.4 FPS는 세그 자리에 드론 모델을 넣어 잰 값이고, 실제 BiSeNetV2 세그로는 19.6 FPS입니다. 세그는 한 번(카메라 2장)에 151ms가 걸려 3프레임마다 돌립니다. 검출 비교는 방 안 녹화 100프레임에서 COCO 80 클래스, 점수 0.25로 했습니다(사람과 차가 없어서).

## 다음에 할 일

1. 사용자가 모형 보정을 마치면, 보정값을 넣고 거리 오차와 도로 IoU(직접 칠한 20프레임)를 잽니다.
2. 3단계(카메라 4대)와 연산자 확인, FlashOcc 재학습은 쿼드 카메라와 x86 GPU 서버가 준비된 뒤에 합니다.

## 사용자가 해야 할 일

1. DX-M1 방열판이나 팬 확인. 동시 실행 14초 만에 77도에서 83도로 올랐습니다.
2. ov5647 렌즈의 130도가 수평 화각인지 대각 화각인지 확인. 지금은 대각으로 가정했습니다.
3. 모형 축척 결정(기본 1/10, 모델 Y 길이 4.79m 기준)과 카메라 장착. 전방 카메라가 거의 수평이라 모형 앞쪽 틀은 낮게 두어야 합니다. 이어서 [모형 보정 문서](phase1_model_rig_calibration.md) 순서대로 체커보드 촬영, 외부 보정, 장난감 차 거리 오차 측정.
4. 후방 Camera Module 3는 자동 초점이므로 보정과 시험 때 초점 고정 확인.
5. 동시 실행 판정 기준 결정: 1단계 기록(14.2 FPS) 기준이면 통과, 같은 코드의 Hailo 단독(26.2 FPS) 기준이면 25% 낮습니다.
6. Hailo DFC와 DX-COM 컴파일, FlashOcc 학습이 필요해지면 x86 리눅스 GPU 서버 연결.

## 알려진 문제

- `fsd_agent.py`가 큰 이미지를 읽으면 SDK 메시지 한도(1MB)를 넘어 멈추던 문제는 `max_buffer_size`를 32MB로 올려 고쳤습니다. 그래서 작업 파일에는 이미지를 줄인 사본을 읽으라고 적습니다.
- DEEPX 최신 모델 목록(2_5_0)은 DX-RT 3.5.0 이상이 필요해, 파이의 DX-RT 3.4.2에 맞는 2_4_0 목록의 모델을 씁니다.
- 맥 numpy에서 `camera_model.py`의 행렬 곱 경고가 나오지만 테스트 결과에는 영향이 없습니다.

## 크레딧 사용 기록

| 실행 | 추정 비용 |
|---|---|
| 01 파이 점검 | 2.00달러 |
| 02 1단계 (중간에 끊김) | 미집계 |
| 02b 1단계 이어서 | 4.96달러 |
| 03 DX-M1 백엔드 | 6.88달러 |
| 04 마스크 렌더링 | 5.73달러 |
| 05 BEVPool CPU | 3.23달러 |
| 06 2단계 .dxnn과 모형 보정 | 11.86달러 |
| 07 테슬라 배치 기준 설정 | 5.99달러 |

합계는 40.65달러에 끊긴 실행 한 번이 더해집니다. 실제 금액은 Console의 Cost 페이지가 기준입니다.
