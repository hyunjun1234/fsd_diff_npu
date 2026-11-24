# 추적기 ID 쪼개짐, 카메라 큐 지연, 저장 이미지 재생

이 문서는 3D 시점 화면 작업에서 남은 두 문제(추적기 ID 쪼개짐, 카메라 큐 지연)를 고치거나 재 보고, 사람이 없는 방에서도 사람 표시를 볼 수 있게 저장 이미지 재생을 더한 기록입니다. 측정일은 2026-11-24 밤입니다. 추적 지표는 맥 시뮬레이션, 지연과 재생은 파이 5, Hailo-8(yolov8s_h8.hef), configs/two_cam_current.yaml(전방 ov5647 768x576, 후방 imx708 768x432, 30fps 요청)으로 쟀습니다. 카메라 보정 전이라 거리 값은 판정에 쓰지 않았고, DX-M1은 쓰지 않았습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| ID 바뀜 (카메라 2대) | 381회에서 48회 | sim 15fps, 225프레임 x seed 5개 |
| ID 바뀜 (카메라 4대) | 487회에서 51회 | 같은 조건, configs/four_cam.yaml |
| 지연 기준 | 84.1 / 97.4ms, 26.1 FPS | 파이, Hailo 단독, 평균 / p95 |
| 지연 `--cam-fps 20` | 74.0 / 76.3ms, 20.0 FPS | 같은 조건 |
| 지연 `--cam-queue off` | 67.9 / 69.6ms, 15.0 FPS | 같은 조건 |
| 사람 표시 재생 | 프레임당 검출 6개, 트랙 6개 | 파이, Hailo, `--replay`, 예제 사진 |
| 시험 | 맥 200 통과, 4 건너뜀 / 파이 203 통과, 1 건너뜀 | 맥 /usr/bin/python3, 파이 .venv |

추적기는 `--tracker robust`로 고쳤고 기본값(basic)은 그대로 두었습니다. 카메라 옵션은 지연을 줄이는 만큼 FPS도 떨어져서 기본값을 바꾸지 않았고 추천 설정도 정하지 않았습니다.

## 1. 추적기 ID 쪼개짐

### 1.1 원인

3D 시뮬레이션 영상과 같은 설정(configs/two_cam.yaml, 15fps, 잡음 seed 0)으로 재현했습니다. 겹쳐 보인 차는 오른쪽 차선 앞차(정답 물체 1번)이고, 108번째 프레임부터 거리 20~34m에서 트랙 3, 32, 35, 37, 46, 52, 55, 63, 69, 71이 차례로 붙었습니다. 이 차는 전방 카메라에만 보이므로 두 카메라 병합 문제는 아닙니다.

먼 물체는 박스 아래 끝 1~2화소 잡음이 지면에서 시선 방향으로 수 m 오차가 됩니다(two_cam.yaml 전방 카메라 기준 25m에서 화소당 약 1.5m). 기존 추적기는 속도를 "측정 - 지난 위치"를 dt(1/15초)로 나눈 값의 반으로 갱신해서, 3m 튄 측정 하나가 약 22m/s 속도가 됩니다. 다음 예측이 고정 짝짓기 거리 2.5m 밖으로 날아가 새 트랙이 생기고, 옛 트랙은 최대 5프레임 남아 두 트랙이 측정을 번갈아 받으며 둘 다 화면에 나왔습니다.

### 1.2 고친 내용

`BevTracker(mode="robust")`와 `run_demo.py --tracker robust`로 넣었습니다. 바꾼 점은 다섯 가지입니다.

1. 속도는 예측 오차의 0.1만 더하는 알파-베타 필터로 갱신하고 60m/s로 자릅니다.
2. 짝짓기 거리를 시선 방향으로만 거리 x 0.15까지 넓히고 옆 방향은 2.5m 그대로 둡니다. 나란히 가는 옆 차선 차(3.5m)와 바뀌지 않게 하려는 것입니다.
3. 첫 짝짓기에서 남은 것끼리 두 배 거리로 한 번 더 짝짓습니다.
4. 남은 측정이 이미 있는 트랙 곁이면 새 트랙을 만들지 않습니다.
5. 짝짓기 거리 안에 겹친 트랙 두 개는 오래된 쪽만 남깁니다. 카메라 4대에서 두 카메라가 같은 차를 다른 자리로 투영해 병합을 빠져나간 경우를 위한 것입니다.

길이 값은 모두 `scene.scale`을 곱해 모형 설정에서도 같은 비율로 동작합니다.

### 1.3 지표

`scripts/eval_tracking.py`가 sim 정답과 화면 트랙을 프레임마다 일대일로 짝짓습니다. 허용 거리는 max(3m, 거리 x 0.2)이고, 정답은 어느 카메라든 정답 박스가 있는 프레임만 셉니다. 반대편으로 다시 등장한 물체는 다른 물체로 봅니다. 중복 표시는 정답 물체 곁에 짝이 없는 트랙이 하나 더 떠 있는 경우의 수입니다. 아래 값은 225프레임 실행을 seed 0~4로 5번 돌린 합계이고, 트랙 수만 seed별 평균의 평균과 최대입니다.

카메라 2대(configs/two_cam.yaml), 15fps:

| 지표 | basic | robust |
|---|---|---|
| ID 바뀜 | 381 | 48 |
| 물체당 트랙 수 평균 (최대) | 7.25 (26) | 1.88 (5) |
| 놓친 프레임 비율 | 4.96% | 2.59% |
| 중복 표시 | 265 | 21 |
| 헛트랙 | 83 | 63 |

카메라 4대(configs/four_cam.yaml), 15fps:

| 지표 | basic | robust |
|---|---|---|
| ID 바뀜 | 487 | 51 |
| 물체당 트랙 수 평균 (최대) | 8.35 (22) | 1.65 (4) |
| 놓친 프레임 비율 | 3.73% | 1.93% |
| 중복 표시 | 904 | 96 |
| 헛트랙 | 150 | 95 |

10fps에서도 같은 경향입니다. 카메라 2대는 ID 바뀜 420회에서 52회, 중복 표시 254에서 12, 카메라 4대는 ID 바뀜 499회에서 86회, 중복 표시 1163에서 159였습니다. 카메라 4대의 중복 표시는 다섯 번째 고침(겹친 트랙 정리)을 넣기 전 robust에서 318이었습니다. 원자료는 out/measure/11_track_{two_cam,four_cam}_fps{15,10}_{basic,robust}.json입니다.

눈으로도 확인했습니다. 137번째 프레임에서 basic은 앞차 위에 ID 35, 37 박스 두 개가 겹치고(out/measure/11_sim_f137_basic.png), robust는 ID 3 하나만 있습니다(11_sim_f137_robust.png). robust 영상은 out/video/11_render3d_sim_robust.mp4(225장, 15fps)입니다.

남은 ID 바뀜의 대부분은 보행자(자차 대비 12m/s로 지나감)와 카메라 경계를 넘는 물체로 보이고, 사건별 분석은 확인 필요입니다. 시뮬레이션의 잡음 모형(박스 좌표 정규분포 2화소, 5% 놓침)에서 얻은 값이라 실제 검출기에서의 효과도 확인 필요입니다.

## 2. 카메라 큐 지연

### 2.1 Picamera2 동작

파이에 깔린 Picamera2 0.3.39 소스를 보면, 동영상 설정(queue=True, 버퍼 6개)에서 Picamera2는 끝난 프레임을 한 장만 들고 있다가 `capture_array`나 `capture_request`에 바로 돌려줍니다. 큐에 여러 장이 쌓이지는 않고, 받는 프레임은 최대 한 프레임(30fps에서 33ms) 묵은 것입니다. queue=False면 늘 다음 프레임이 끝날 때까지 기다리고, `capture_request(flush=True)`는 부른 뒤에 노출을 시작한 프레임을 기다립니다.

### 2.2 카메라만 잰 값 (NPU 없음)

`scripts/cam_latency.py`로 카메라만 열어 센서 시각에서 `read()`가 돌려줄 때까지를 쟀습니다(130루프, 원자료 out/measure/11_cam_latency.log). 처리가 없으면 전방 23.7ms, 후방 22.3ms였고, 이것이 노출 이후 전송과 ISP에 드는 바닥값입니다. 처리 35ms를 흉내 내 루프를 27.4 FPS로 늦추면 전방 39.9ms, 후방 39.0ms로 데모와 같아졌습니다. 그래서 약 40ms 중 23ms 안팎은 바닥값이고, 나머지 16ms 안팎이 들고 있던 프레임이 묵은 몫입니다.

buffer_count 2, 3은 차이가 없었습니다. queue off는 바닥값까지 내려가지만 카메라 두 대를 차례로 기다려 루프가 15.0 FPS로 떨어졌고, flush는 7.5 FPS였습니다. 프레임레이트를 루프보다 조금 낮추면 루프가 전방 카메라에 맞춰 돌아 전방은 바닥값이 되지만, 후방은 두 센서의 위상에 따라 25ms(24fps 실행)도 58ms(25fps 실행)도 나왔습니다.

### 2.3 파이 Hailo 단독 20초 전후

명령은 `--backend hailo --headless --duration 20 --warmup 30 --profile`이고 화면은 기본 탑뷰입니다. 지연은 그 프레임에 쓴 영상 중 오래된 쪽 카메라 기준입니다.

| 실행 | FPS | 지연 평균 / p95 (ms) |
|---|---|---|
| 기준 1회째 | 26.1 | 84.1 / 97.4 |
| 기준 2회째 | 26.0 | 80.8 / 95.8 |
| `--cam-fps 24` | 23.9 | 81.2 / 86.3 |
| `--cam-fps 20` | 20.0 | 74.0 / 76.3 |
| `--cam-queue off` | 15.0 | 67.9 / 69.6 |

센서에서 `read()`까지는 기준에서 전방 44.0ms, 후방 39.7ms였고, `--cam-fps 20`에서 25.9, 21.9ms, queue off에서 23.9, 21.4ms로 바닥값까지 내려갔습니다. 기준 두 번의 평균이 3.4ms 차이 나므로 그보다 작은 차이는 판정하지 않았습니다.

정리하면 묵은 프레임 몫(약 16ms)은 줄일 수 있지만, 줄이는 방법은 모두 루프가 카메라를 기다리게 만드는 것이라 FPS가 같이 떨어집니다. `--cam-fps 20`은 평균 10ms, p95 21ms를 줄이고 FPS를 23% 잃었습니다. 10 FPS 목표는 넘지만 FPS를 내주고 얻는 이득이라 기본값은 바꾸지 않았고, 지연이 FPS보다 중요할 때 고를 옵션으로만 둡니다. 원자료는 out/measure/11_lat_{base,base2,fps24,fps20,queue_off}.json, .csv, .log입니다.

### 2.4 온도

Hailo 칩 온도가 85도 이하가 될 때까지 60초 단위로 기다린 뒤 실행했고, 실행 직전 81.3~85.0도, 실행 직후 최고 88.2도였습니다. 95도 멈춤에는 닿지 않았습니다(out/measure/11_lat_runs.log, 11_replay_runs.log).

## 3. 저장 이미지 재생

`run_demo.py --replay 폴더`를 더했습니다. 폴더에서 `{카메라 이름}_`으로 시작하는 png, jpg를 이름 순서로 읽어 카메라 대신 넣고, 끝나면 처음부터 다시 읽습니다. scripts/record_frames.py 결과 폴더(out/rec/room_100)를 그대로 넣을 수 있고 사진 한 장만 있어도 됩니다. 카메라를 열지 않으므로 지연 측정 값은 없습니다.

방 녹화와 01_frame_cam*.jpg에는 사람이 없어서(01_det_check.json에 의자, TV, 노트북, 꽃병만 있음) 사람 표시는 예제 사진으로 확인했습니다. 맥에 있는 ultralytics 예제 사진 bus.jpg(버스와 보행자)를 전방 크기 768x576에 맞춰 회색으로 채우고, zidane.jpg(사람 두 명)를 후방 768x432로 줄여 out/replay/person/에 두었습니다. 예제 사진이라 3D 화면의 위치와 거리는 의미가 없고 클래스 표시만 봅니다.

| 실행 | 결과 | 파일 |
|---|---|---|
| 사람 사진, `--view 3d` | 검출 6개, 트랙 6개, 19.8 FPS 표시 | out/measure/11_replay_person_3d.png |
| 사람 사진, `--view both` | 검출 6개 | out/measure/11_replay_person_both.png |
| 방 녹화 100장, `--view 3d` | 검출 0개 | out/measure/11_replay_room_3d.png |

사람 사진 실행에서는 사람이 주황색 10각 기둥, 버스가 박스로 3D 화면에 그려졌습니다. 후방 사진 속 사람은 자차 바로 뒤 큰 기둥으로 보이는데, 사진이 실제 장착 시점이 아니어서 생긴 위치이고 렌더러 문제는 아닙니다. 세 실행 모두 `--tracker robust`를 썼습니다.

## 4. 남은 한계

robust를 기본값으로 바꾸는 일은 실제 카메라 장면에서 확인한 뒤로 미뤘습니다. 실제 장면 확인 전에는 기존 동작을 유지하는 편이 안전하다고 보았습니다.

카메라 4대 sim의 남은 중복 표시(96)는 측면 카메라가 차를 옆에서 볼 때도 투영점을 시선 방향으로 차 길이의 절반만큼 미는 `project_detections`의 가정 때문으로 보이고, 확인 필요입니다.

카메라 두 대는 동기 신호 없이 따로 돌아서, 프레임레이트를 맞춰도 후방 카메라의 묵음은 두 센서의 위상에 따라 달라집니다.

## 5. 다시 재는 명령

맥 저장소 폴더에서 실행합니다. 결과는 out/measure/11_track_*.json에 생깁니다.

```bash
/usr/bin/python3 scripts/eval_tracking.py --config configs/two_cam.yaml --fps 15 --frames 225 --seeds 0,1,2,3,4 --tracker basic --out out/measure/11_track_two_cam_fps15_basic.json
/usr/bin/python3 scripts/eval_tracking.py --config configs/four_cam.yaml --fps 15 --frames 225 --seeds 0,1,2,3,4 --tracker robust --out out/measure/11_track_four_cam_fps15_robust.json
/usr/bin/python3 scripts/run_demo.py --backend sim --view 3d --view3d-interp --fps 15 --frames 225 --headless --tracker robust --save out/video/11_render3d_sim_robust.mp4
```

파이 저장소 폴더에서 .venv를 켠 뒤 실행합니다. 각 Hailo 실행 전에 10 문서 5장의 온도 읽기 줄로 85도 이하인지 확인합니다. 결과는 out/measure/ 에 생깁니다.

```bash
timeout -s INT -k 5 30 python3 scripts/cam_latency.py --config configs/two_cam_current.yaml --frames 150 --work-ms 35
timeout -s INT -k 10 50 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --headless --duration 20 --warmup 30 --profile --profile-out out/measure/11_lat_fps20.json --health-csv out/measure/11_lat_fps20.csv --health-interval 10 --hailo-stop-temp 95 --cam-fps 20
timeout -s INT -k 10 50 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --replay out/replay/person --frames 40 --headless --tracker robust --view 3d --view3d-interp --profile --warmup 5 --profile-out out/measure/11_replay_person_3d.json --health-csv out/measure/11_replay_person_3d.csv --health-interval 10 --hailo-stop-temp 95 --snapshot out/measure/11_replay_person_3d.png
```
