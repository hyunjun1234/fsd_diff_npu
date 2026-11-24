# 주행 녹화, 트리거 클립, 섀도 실행

이 문서는 차량 주행 데이터를 파이에서 모으고 학습 서버로 옮기는 방법과, 새 회차 모델을 기존 모델과 나란히 돌려 비교하는 방법을 정리한 안내서입니다. 배경과 전체 순환은 docs/07_vehicle_mount_and_training_plan.md 7장, 9.3절, 10장을 봅니다.

모든 명령은 파이 저장소 폴더(`~/work/fsd_diff_npu`)에서 `.venv`를 켠 뒤 실행합니다.

```bash
cd ~/work/fsd_diff_npu && . .venv/bin/activate
```

## 1. 폴더 구조

녹화를 한 번 시작해서 끝낼 때까지를 회차 하나로 보고, 시작 시각을 폴더 이름으로 씁니다.

```
out/drives/<YYYYmmdd_HHMMSS>/
    drive.json        설정 파일 본문, 카메라 모드(센서 이름, 센서 출력 크기), 시작과 끝 시각, 파이 상태, 끝난 이유
    config.yaml       녹화에 쓴 설정 파일 사본
    status.csv        1분마다 카메라별 프레임 수, 누락 수, 큐 버림 수, SoC 온도, throttled, 남은 디스크
    seg_0000/         녹화 시작 뒤 0~60초
        front.mjpeg   JPEG를 이어 붙인 영상
        front.csv     프레임마다 한 줄: 번호, 센서 시각(ns), 벽시계, 파일 안 위치와 크기, 앞 프레임 이후 누락 수
        rear.mjpeg, rear.csv
        segment.json  모든 카메라 파일을 닫고 fsync한 뒤에만 생깁니다
    seg_0001/ ...
out/triggers/<실행 이름>/<YYYYmmdd_HHMMSS>_f<프레임>_<종류>/
    front.mjpeg, front.csv, rear.mjpeg, rear.csv, trigger.json
```

카메라 수와 이름은 설정 파일의 `cameras` 항목에서 읽습니다. 4대 설정을 주면 세그먼트마다 파일이 4쌍 생깁니다.

영상은 카메라별 MJPEG(JPEG 이어 붙이기)입니다. 덧붙이기만 하는 파일이라 전원이 갑자기 꺼져도 그때까지 쓴 프레임은 읽을 수 있고, 프레임마다 독립된 JPEG라 라벨링할 프레임을 바로 꺼낼 수 있습니다. 파이 5에는 H.264 하드웨어 인코더가 없어 H.264는 쓰지 않습니다. `ffmpeg -f mjpeg -i front.mjpeg` 로 다른 형식으로 바꿀 수 있습니다.

`segment.json`이 없는 세그먼트는 닫히기 전에 끊긴 것입니다. 이런 세그먼트도 csv와 mjpeg 길이가 맞는 프레임까지는 읽힙니다.

## 2. 주행 녹화

NPU 추론 없이 카메라 원본만 저장합니다.

```bash
timeout -s TERM -k 15 3700 python3 scripts/record_drive.py --config configs/two_cam_current.yaml --duration 3600
```

결과는 `out/drives/<시작 시각>/`에 생깁니다.

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--segment` | 60 | 세그먼트 길이(초) |
| `--duration` | 0 | 이 시간(초)이 지나면 끝냄. 0이면 신호를 받을 때까지 |
| `--min-free-gb` | 5 | 남은 디스크가 이보다 적으면 멈춤 |
| `--status-interval` | 60 | 누락 수와 온도를 남기는 간격(초) |
| `--quality` | 85 | JPEG 품질 |
| `--cam-fps` | 설정값 | 카메라 프레임레이트 덮어쓰기 |

Ctrl+C(SIGINT)나 SIGTERM을 받으면 지금 세그먼트를 닫고 `drive.json`에 끝난 이유를 적은 뒤 끝납니다. ACC가 꺼질 때 전원 장치나 systemd가 SIGTERM을 보내고 몇 초 기다려 주게 설정하면 마지막 세그먼트까지 깨끗하게 남습니다. 디스크 남은 공간은 2초마다 보고, 시작할 때부터 모자라면 한 장도 쓰지 않고 끝냅니다.

카메라마다 읽기 스레드와 쓰기 스레드를 따로 둡니다. 디스크 쓰기가 잠깐 멈춰도 읽기 스레드는 큐(카메라당 60장)에 넣고 계속 읽으므로 프레임을 놓치지 않습니다. 큐가 가득 차서 버린 수는 `status.csv`의 `queue_drops`에, 센서 시각 간격으로 추정한 누락 수는 `gaps`에 남습니다.

용량은 카메라 2대(768x576, 768x432), 30fps, JPEG 품질 85, 실내 정지 장면에서 한 시간에 약 7.1GB였습니다. 도로 장면은 영상이 복잡해 더 커질 수 있고 카메라 4대는 두 배 안팎으로 보지만 둘 다 확인 필요입니다. 측정 조건과 계산은 out/measure/12_drive_collect.md 에 있습니다.

## 3. 녹화 확인과 재생

회차 목록과 용량:

```bash
python3 scripts/record_drive.py --list
du -sh out/drives/*
```

`--list`는 회차마다 시작 시각, 길이, 깨끗하게 닫힌 세그먼트 수와 전체 세그먼트 수, 프레임 수, 누락 수, 용량, 끝난 이유를 한 줄씩 보여 줍니다. 이름이 점으로 시작하는 폴더는 치워 둔 것으로 보고 뺍니다.

녹화한 회차나 세그먼트, 트리거 클립은 그대로 `--replay`에 넣습니다.

```bash
timeout -s INT -k 10 60 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo \
    --replay out/drives/<회차> --headless --duration 50 \
    --health-csv out/measure/replay_health.csv --health-interval 10 --hailo-stop-temp 95
```

회차 폴더를 주면 세그먼트를 이름 순서로 이어서 읽습니다. 카메라마다 프레임 수가 조금씩 다르므로 첫 번째 카메라 프레임마다 센서 시각이 가장 가까운 다른 카메라 프레임을 짝지어 넣습니다. 녹화 당시 시각은 지연 측정에 쓰지 않습니다(`--profile`의 지연 값은 재생에서 "모름"으로 나옵니다). 예전 이미지 폴더(`{카메라}_00000.png`)도 그대로 받습니다.

## 4. 학습 서버로 옮기기

학습 서버 주소와 경로는 아래 자리표시자를 바꿔 씁니다. 서버에 SSH 키 접속이 되어 있어야 합니다.

```bash
# 회차 하나
rsync -av --partial out/drives/<회차>/ <사용자>@<서버 주소>:/data/fsd_drives/<회차>/
# 모든 회차와 트리거 클립
rsync -av --partial out/drives/ <사용자>@<서버 주소>:/data/fsd_drives/
rsync -av --partial out/triggers/ <사용자>@<서버 주소>:/data/fsd_triggers/
```

결과는 서버의 `/data/fsd_drives/`, `/data/fsd_triggers/` 아래에 같은 폴더 구조로 생깁니다. 옮긴 뒤 서버 쪽 용량이 파이와 같은지 `du -sb`로 비교하고, 같을 때만 파이의 회차 폴더를 지웁니다. 파이에서 지울 때는 `out/drives/` 아래 회차 폴더만 지웁니다.

## 5. 트리거 클립

실시간 실행 중 모델이 약해 보이는 순간의 앞뒤 원본을 따로 저장합니다. 기본은 꺼져 있고 `--trigger-clips`로 켭니다.

```bash
timeout -s INT -k 10 70 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo \
    --headless --duration 60 --trigger-clips --trigger-dir out/triggers/<실행 이름> \
    --health-csv out/measure/trig_health.csv --health-interval 10 --hailo-stop-temp 95
```

결과는 `--trigger-dir` 아래에 클립마다 폴더 하나로 생깁니다.

| 종류 | 조건 | trigger.json의 score |
|---|---|---|
| ambiguous | 점수 0.3 이상 0.5 미만인 상자 | 그 상자들 중 가장 낮은 점수 |
| track_new | 새 추적 ID가 생김 | 0 |
| track_switch | 사라진 ID 근처(2m 안)에서 새 ID가 생김 | 두 위치 사이 거리(m) |
| npu_mismatch | `--shadow`에서 짝 못 찾은 상자가 1개 이상 | 짝 못 찾은 상자 수 |

`npu_mismatch`는 두 NPU를 같이 돌려야 하므로 `--shadow`를 함께 줄 때만 나옵니다. 점수 범위는 `--trigger-score 0.3,0.5`, 거리는 `--trigger-switch-dist`, 상자 수는 `--trigger-mismatch`로 바꿉니다.

설정의 `npu.score_thr`(0.4)보다 낮은 상자도 보려고, 옵션을 켜면 검출 백엔드는 점수 하한(0.3)까지 내보내고 화면과 추적에는 0.4 이상만 넘깁니다. 그래서 화면과 추적 결과는 옵션을 켜지 않았을 때와 같습니다. 지금 Hailo HEF(yolov8s_h8.hef)는 NMS 안에서 약 0.2 아래 상자를 내보내지 않으므로 하한을 0.2보다 낮춰도 효과가 없습니다(사람 사진 한 장에서 본 최저 점수 0.207, 정확한 문턱은 확인 필요). DX-M1 PPU 모델은 장치 문턱이 0.001입니다.

링 버퍼는 앞쪽 `--trigger-pre`초(기본 2)를 들고 있고, 트리거 뒤 `--trigger-post`초(기본 2)를 더 모아 쓰기 스레드가 저장합니다. 모으는 동안 나온 다른 트리거는 같은 클립의 `events`에 더합니다. 같은 종류는 `--trigger-cooldown`초(기본 10) 안에 다시 클립을 만들지 않고 실행이 끝날 때 `suppressed`로 수만 보여 줍니다. 버퍼는 `--trigger-buffer-mb`(기본 300MB)를 넘지 않게 오래된 프레임부터 버립니다. 카메라 2대, 약 26fps, 2초면 약 120MB를 씁니다. 저장이 밀려 대기 중인 클립이 2개를 넘으면 새 클립은 버리고 `clips_dropped`로 셉니다.

## 6. 섀도 실행

같은 프레임을 기본 NPU와 반대쪽 NPU에서 다른 회차 모델로 돌려 결과 차이를 프레임마다 csv로 남깁니다. 화면과 추적에는 기본 모델 결과만 씁니다.

```bash
python3 scripts/dx_temp.py --label before_shadow --log out/measure/dx_temp.log --guard 80 --resume-below 79
timeout -s INT -k 10 40 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo \
    --headless --duration 25 --shadow --shadow-csv out/measure/shadow.csv \
    --health-csv out/measure/shadow_health.csv --health-interval 10 --hailo-stop-temp 95
python3 scripts/dx_temp.py --label after_shadow --log out/measure/dx_temp.log
```

`--backend hailo`면 DX-M1에서, `--backend dxm1`이면 Hailo-8에서 섀도 모델을 돌립니다. `--backend dxm1` 쪽 조합은 파이에서 아직 돌려 보지 않아 확인 필요입니다. 설정에 `models` 항목이 없으면 섀도 모델로 `npu.det_hef`나 `npu.dx_det_model`을 씁니다. 끝날 때 `shadow_summary=` 줄에 전체 요약이 나옵니다. csv 열은 아래와 같습니다.

| 열 | 뜻 |
|---|---|
| frame, camera | 프레임 번호, 카메라 이름 |
| n_primary, n_shadow | 기본 모델과 섀도 모델의 상자 수 |
| matched | 같은 클래스, IoU 0.5 이상으로 짝지은 수(`--shadow-iou`) |
| shadow_missed | 기본 모델은 찾았는데 섀도 모델이 놓친 수 |
| shadow_extra | 섀도 모델만 찾은 수 |
| iou_mean, iou_min | 짝지은 상자의 IoU |
| score_diff_mean, score_diff_max_abs | 짝지은 상자의 (섀도 점수 - 기본 점수) |
| primary_ms, shadow_ms | 두 추론 시간(ms) |

두 추론은 동시에 돌고 프레임마다 둘 다 끝나기를 기다리므로, 섀도 모델이 느리면 화면도 그만큼 느려집니다. `--shadow`와 `--seg-npu`는 같이 쓸 수 없습니다.

DX-M1과 Hailo-8에 방열이 없는 동안에는 DX-M1을 쓰는 실행을 한 번에 30초 이내로 하고, 앞뒤로 `scripts/dx_temp.py`를 돌립니다. Hailo 온도는 `scripts/hailo_temp.py --guard 85 --resume-below 80`으로 실행 전에 확인합니다.

## 7. 회차 모델 파일 규칙

추가 학습을 한 번 돌릴 때마다 검출 모델을 HEF와 .dxnn으로 만들어 파이의 `models/`에 아래 이름으로 둡니다. 회차 번호는 세 자리입니다.

```
models/yolov8s_r001.hef
models/yolov8s_r001.dxnn
```

어떤 회차를 쓸지는 설정 파일의 `models` 항목에서 고릅니다.

```yaml
models:
  dir: models
  det_name: yolov8s
  det_model_round: 0        # 화면과 추적에 쓰는 회차
  shadow_model_round: 0     # --shadow 때 반대쪽 NPU에서 돌릴 회차
  rounds:
    0:
      hef: models/yolov8s_h8.hef
      dxnn: models/dxnn/yolov8-s_640x640_ppu.dxnn
```

0회차는 이름 규칙 이전의 공개 사전학습 모델이라 `rounds`에 경로를 직접 적었습니다. `rounds`에 없는 회차는 이름 규칙으로 파일을 찾습니다. 모델 경로를 고르는 순서는 `--det-hef`(Hailo만), `models.det_model_round`, `npu.det_hef` 또는 `npu.dx_det_model`입니다. `models` 항목이 없는 설정은 예전처럼 `npu` 항목을 씁니다.

새 회차를 들이는 순서는 다음과 같습니다.

1. 새 파일을 `models/yolov8s_r00N.hef`, `.dxnn`으로 복사합니다.
2. `shadow_model_round: N`으로 바꾸고 `--shadow`로 짧게 여러 번 돌려 csv를 봅니다.
3. 문제가 없으면 `det_model_round: N`으로 바꿉니다.
4. 문제가 생기면 `det_model_round`를 이전 회차로 되돌립니다.

섀도 실행은 회차가 다른 두 모델을 다른 NPU에서 비교하므로, 차이에는 모델 차이와 NPU 변환(INT8) 차이가 섞입니다. 같은 회차를 두 NPU에 올려 본 0회차 결과를 기준선으로 삼아 비교합니다.
