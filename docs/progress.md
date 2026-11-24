# 진행 상황

이 문서는 라즈베리파이 5, Hailo-8, DX-M1 구현 작업의 현재 진행 상황입니다. 다른 사람이나 다른 세션이 이 문서만 읽고 작업을 이어받을 수 있게, 실행이 끝날 때마다 갱신해 main에 올립니다. 단계와 완료 기준은 [구현 계획](03_pi5_hailo8_dxm1_implementation_plan.md)을 따릅니다.

## 지금 위치

1단계는 코드와 측정을 마쳤고 사람이 하는 보정만 남았습니다. 2단계는 DX-M1 백엔드와 두 NPU 동시 실행 구조까지 만들었고, 실제 .dxnn 모델로 검출 비교를 하는 일을 진행 중입니다. 4단계 준비로 BEVPool CPU 구현을 마쳤습니다.

| 단계 | 상태 |
|---|---|
| 0단계 장비 확인 | 끝남 (DX-M1 단독 FPS, 30분 동시 시험 남음) |
| 1단계 Hailo 실기 | 보정과 거리 오차 측정 남음 |
| 2단계 DX-M1 추가 | 모델 받아 검출 비교 진행 중 |
| 4단계 준비 | BEVPool CPU 구현 끝남 |

## 장비 구성

라즈베리파이 5(Debian 13, 64비트)에 Seeed 듀얼 M.2 HAT로 Hailo-8과 DX-M1 4GB를 함께 달았고, 두 NPU 모두 PCIe Gen3 x1로 인식됩니다. 런타임은 HailoRT 4.23.0, DXRT 3.4.2이고, `dx_engine`은 파이 저장소의 `.venv`에만 있습니다.

카메라는 계획의 Camera Module 3 Wide 2대가 아니라 지금 달린 카메라를 계속 씁니다. 0번은 ov5647(전방, 사용자 확인 기준 130도 광각, IR 컷, 야간 가능)이고 1번은 일반 Camera Module 3(imx708, 후방)입니다. 설정은 [configs/two_cam_current.yaml](../configs/two_cam_current.yaml)이고, 내부 파라미터는 보정 전 근사값입니다. 보정과 시험은 실제 차량이 아니라 모형으로 먼저 합니다.

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

1단계 기본 설정에서 병목은 NPU(약 18ms)가 아니라 CPU의 BEV 합성(23ms)과 렌더링(28ms)이었습니다. 도로 마스크 덧칠을 OpenCV로 바꿔 카메라 2대 기준 39.5ms에서 5.8ms로 줄였습니다. BEVPool은 입력 256x704, D=88, C=64, BEV 100x100 조건이고 NumPy로는 약 110ms라 C 구현(ctypes, OpenMP)을 씁니다. 동시 실행 값은 세그 자리에 파이에 있던 드론 검출 .dxnn을 대신 넣어 잰 것이라, 실제 세그 모델로 다시 재야 합니다.

## 다음에 할 일

1. 2단계 마무리([agent_tasks/06_dxnn_stage2.txt](agent_tasks/06_dxnn_stage2.txt)): DEEPX 공식 저장소에서 YOLOv8s와 세그 .dxnn을 파이로 받고(사용자 허락 받음), 같은 녹화로 두 NPU 검출을 비교하고, 모형용 설정과 어안 보정, 모형 보정 절차 문서를 만듭니다.
2. 연산자 확인(계획 4.5절)과 FlashOcc 재학습은 x86 GPU 서버가 연결된 뒤에 합니다.

## 사용자가 해야 할 일

1. 모형에서 카메라별 체커보드 촬영, 장착 높이와 각도 실측, 거리 오차 측정. 절차는 2단계 마무리 작업에서 문서로 만듭니다.
2. DX-M1 방열판이나 팬이 붙어 있는지 확인.
3. Hailo DFC와 DX-COM 컴파일, FlashOcc 학습이 필요해지면 x86 리눅스 GPU 서버 연결.

## 알려진 문제

- `fsd_agent.py`가 큰 이미지를 읽으면 SDK 메시지 한도(1MB)를 넘어 멈추던 문제는 `max_buffer_size`를 32MB로 올려 고쳤습니다. 그래서 작업 파일에는 이미지를 줄인 사본을 읽으라고 적습니다.
- 에이전트 실행에서 웹 페이지 읽기가 거부되어, DEEPX 모델 이름과 주소는 파이에서 공식 주소를 직접 조회하는 방식으로 찾습니다.
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

합계는 22.80달러에 끊긴 실행 한 번이 더해집니다. 실제 금액은 Console의 Cost 페이지가 기준입니다.
