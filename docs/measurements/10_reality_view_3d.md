# 3D 시점 렌더러 (리얼리티 뷰처럼 자차 뒤쪽 위에서 보기)

이 문서는 자차 뒤쪽 위에서 내려다보는 3D 시점 렌더러(`realityview/render3d.py`, `--view 3d`)를 만들고 맥 시뮬레이션과 파이 5 실제 카메라로 확인한 기록입니다. 측정일은 2026-11-24 밤이고, 파이 측정은 파이 5, Hailo-8(HailoRT 4.23.0, yolov8s_h8.hef), configs/two_cam_current.yaml의 카메라 2대(전방 ov5647 768x576, 후방 imx708 768x432)로 했습니다. 카메라 보정 전이라 거리 값은 판정에 쓰지 않았습니다. DX-M1은 쓰지 않았습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| 3D 시점 FPS | 20.8 | 파이, Hailo, `--view 3d --view3d-interp`, 20초 |
| 탑뷰 + 3D FPS | 18.0 | 파이, `--view both`, 20초 |
| 녹화하면서 3D FPS | 19.3 | 파이, `--record-video`, 17초 |
| 빠른 옵션 3D FPS | 28.3 | 파이, `--view3d-ground-every 3 --view3d-scale 0.5` |
| 기준(지금 탑뷰) FPS | 26.1 | 파이, `--view bev`, 같은 시간대 |
| 3D 그리기 시간 | 프레임당 11.1ms | 파이, 물체 0개(방에 사람과 차가 없음) |
| 물체 20개 3D 그리기 | 23.4ms | 파이 CPU 벤치, 바닥 갱신 포함 |
| 시험 | 맥 184 통과, 4 건너뜀 / 파이 187 통과, 1 건너뜀 | 맥 /usr/bin/python3, 파이 .venv |

목표인 "파이 5 CPU에서 카메라 2대 데모와 함께 10 FPS 이상"은 기본 설정(`--view 3d`)에서 10초 단위 기록 모두 20.7 FPS 이상으로 넘었습니다(out/measure/10_pi_3d.csv). 저해상도 렌더와 바닥 갱신 주기 옵션은 목표를 넘었으므로 기본값은 꺼 두었습니다.

## 1. 만든 것

`Renderer3D`는 핀홀 가상 카메라를 원점(뒷바퀴 축) 뒤 7m, 높이 4.5m에 두고 18도 내려다보며 수평 화각은 70도입니다. 이 네 값은 `--view3d-cam 뒤거리,높이,숙임각,화각`으로 바꿉니다. 길이는 모두 `scene.scale`을 곱해 쓰므로 모형 설정에서도 같은 화면이 나옵니다(시험 `test_scale_invariance`).

바닥은 탑뷰 크기의 바닥 그림(IPM 바닥 영상, 도로 마스크, 5m 격자, 거리 고리)을 만든 뒤, 화면 화소마다 시선과 지면의 교점을 미리 계산한 표로 `cv2.remap` 해서 폅니다. `cv2.warpPerspective`를 쓰지 않은 이유는 지평선 위 화소가 카메라 뒤쪽 바닥을 거울처럼 끌어오기 때문입니다. 지평선 위는 하늘 배경을 깔고, 표는 지평선 아래 줄만 만듭니다.

물체는 가상 카메라와의 거리로 정렬해 먼 것부터 그립니다(화가 알고리즘). 차, 트럭, 버스, 이륜차는 박스, 사람은 10각 기둥으로 그리고, 볼록 기둥이라 카메라를 향한 면만 칠하면 면끼리 겹치지 않습니다. 반투명(기본 불투명도 0.6)은 물체를 둘러싼 사각형 안에서만 섞어서 물체 하나에 약 0.5ms가 듭니다(파이 벤치). 꼭짓점 하나라도 가상 카메라 앞 0.3m보다 가까우면 그 물체는 그리지 않습니다. 그래서 카메라보다 뒤(원점 뒤 7m보다 뒤)에 있는 차는 3D 화면에 나오지 않고 탑뷰에만 나옵니다.

자차는 몸통 박스와 실내 박스 두 개로 그립니다. 트랙 ID 글자, 상대 속도 화살표(1초 뒤 위치까지, 최대 8m), 거리 고리는 옵션으로 끌 수 있습니다. `--view3d-interp`는 트랙 표시 위치를 `1 - exp(-dt / 0.15초)` 비율로 새 위치에 다가가게 해서 튐을 줄입니다. 시간은 step()에 넘긴 dt를 더한 값이라 시뮬레이션에서도 결과가 같습니다.

`--render-thread`와 `--view3d-ground-every`를 같이 쓰면, 렌더링 스레드가 바닥 갱신 차례의 작업을 버릴 때 다음 작업이 갱신을 이어받습니다. 이 모드에서는 IPM 바닥 합성을 매 프레임 하므로 BEV 합성 시간은 줄지 않고 3D 바닥 갱신만 줄어듭니다.

대시보드는 `--view 3d`에서 3D 화면 960x540 오른쪽에 카메라 영상을 폭 320으로 쌓아 1280x540이고, `--view both`는 왼쪽에 탑뷰를 높이 540으로 줄여 붙여 1604x540입니다. 크기가 프레임마다 같고 짝수라 avc1 녹화가 열립니다.

## 2. 맥 시뮬레이션 확인

`--backend sim`으로 60번째 프레임을 그려 눈으로 확인했습니다. 탑뷰와 3D 화면에서 왼쪽 차선의 차(ID 3)와 보행자는 왼쪽에, 오른쪽 차선 트럭(ID 14)은 오른쪽에 있어 좌우가 맞습니다. 차선 점선과 도로 가장자리가 원근으로 펴져 보입니다.

| 파일 | 내용 |
|---|---|
| out/measure/10_render3d_sim_3d.png | `--view 3d`, 60번째 프레임 |
| out/measure/10_render3d_sim_both.png | `--view both`, 같은 프레임 |
| out/measure/10_render3d_sim_3d_half.png | `--view3d-scale 0.5` |
| out/video/10_render3d_sim.mp4 | 15.0초, 225장, 15 fps, 1280x540, mp4v |

시뮬레이션 영상은 `--save --fps 15 --frames 225`로 만들었습니다. `--record-video`는 벽시계로 시간축을 맞추는데 맥 시뮬레이션 루프는 실제 시간과 다르게 돌아서, 시뮬레이션 시간과 길이가 정확히 맞는 `--save`를 썼습니다. 영상 중간에 같은 차에 트랙 두 개(ID 32, 37)가 겹쳐 보이는 구간이 있는데, 추적기(tracker.py)가 한 차를 둘로 나눈 것이고 렌더러 문제는 아닙니다.

## 3. 파이 측정

각 실행은 `--duration 20 --warmup 30 --profile --health-interval 10 --hailo-stop-temp 95`이고, 측정 프레임은 워밍업 30장을 뺀 나머지입니다. 실행 장면은 사무실이라 검출이 0개였고(det_total 0), 그래서 파이 실행의 3D 시간은 바닥과 자차만 그린 값입니다. 물체 수에 따른 비용은 아래 벤치로 따로 쟀습니다. 도로 세그 HEF가 없어(npu.seg_hef null) 파이에서는 도로 칠하기가 꺼져 있었고, 도로 칠하기는 시뮬레이션과 벤치로만 확인했습니다.

| 실행 | 화면 FPS | 렌더링 / 3D / 3D 바닥 (ms) |
|---|---|---|
| bev (기준) | 26.1 | 9.4 / - / - |
| 3d | 20.8 | 18.9 / 11.1 / 9.5 |
| both | 18.0 | 26.5 / 11.1 / 9.4 |
| 3d + 녹화 | 19.3 | 18.7 / 10.9 / 9.4 |
| 3d + 빠른 옵션 | 28.3 | 12.4 / 4.5 / 7.6 |

렌더링 시간에는 3D 그리기, 카메라 영상 주석, 대시보드 합치기가 모두 들어 있습니다. 3D 바닥 시간은 바닥을 다시 만든 프레임만의 평균이라, 빠른 옵션(3프레임마다)에서는 3D 시간 평균보다 큽니다. 빠른 옵션에서는 바닥을 다시 만들지 않는 프레임에 IPM 합성도 건너뛰어 BEV 합성이 9.3ms에서 평균 3.1ms로 줄었습니다. 이때 FPS는 카메라 30 fps에 가까워 카메라 대기에 묶인 것으로 보입니다(확인 필요). 원자료는 out/measure/10_pi_{bev,3d,both,3d_rec,3d_fast}.json, .csv, .log입니다.

### 3.1 물체 수에 따른 3D 그리기 시간 (CPU 벤치)

`scripts/bench_render3d.py`로 파이에서 가짜 트랙(차, 트럭, 사람)을 놓고 `Renderer3D.render()`만 쟀습니다. 바닥 영상은 무작위 무늬, 도로 마스크는 절반을 덮게 했습니다. 각 값은 60번 평균입니다(out/measure/10_bench_render3d_pi.json).

| 트랙 수 | 바닥 갱신 포함 (ms) | 바닥 재사용 (ms) |
|---|---|---|
| 0 | 13.2 | 0.9 |
| 10 | 18.1 | 5.6 |
| 20 | 23.4 | 10.9 |

render_scale 0.5에서는 같은 순서로 11.7 / 15.7 / 20.1ms(바닥 갱신 포함)였습니다. 물체 그리기는 파이썬 반복 비용이 커서 해상도를 줄여도 거의 줄지 않습니다. 20개일 때 데모 전체를 어림하면 3d 실행의 프레임 47.8ms에 12ms쯤 더해 약 60ms, 16 FPS 안팎이고, 실측은 확인 필요입니다. 그래도 10 FPS 아래로 내려가면 `--view3d-ground-every 3`부터 씁니다.

### 3.2 파이 실제 카메라 영상

out/video/10_render3d_pi.mp4는 `--view 3d --view3d-interp --record-video`로 남긴 15.57초(327장, 21 fps, avc1, 1280x540) 영상입니다. 방 바닥이 IPM으로 펴져 3D 바닥에 깔리고 자차 모형이 보입니다. 방에 사람과 차가 없어 3D 박스는 나오지 않습니다. 녹화 중 버린 영상은 0장이었습니다.

### 3.3 온도

Hailo 칩 온도는 실행 전마다 확인했고 모두 85도 이하에서 시작했습니다. 실행 전 78.6~83.6도, 실행 뒤 83.6~88.2도였고, 실행 중 최고는 빠른 옵션 실행의 88.6도(health csv)라 멈춤 기준 95도에 닿지 않았습니다. 처음 만든 온도 확인 스크립트가 `Device().control` 한 줄 호출 때문에 값을 읽지 못해 10분 동안 빈 값으로 기다린 기록이 out/measure/10_hailo_temp.log 앞쪽에 남아 있고, 그 줄들은 무효입니다. 파이 SoC는 56~64도, throttled는 모든 실행에서 0x0이었습니다.

## 4. 남은 한계

카메라보다 뒤(원점 뒤 약 7m 너머)의 차는 3D 화면에 나오지 않습니다. 가상 카메라를 더 뒤로 빼면 보이지만 앞쪽이 작아집니다.

자차 오도메트리가 없어 물체는 자차와 평행한 박스로 그립니다. 속도 화살표는 상대 속도입니다.

먼 바닥은 탑뷰 셀 5cm 텍스처를 크게 줄여 그려서 무늬가 깨져 보일 수 있습니다. 바닥 텍스처는 탑뷰 범위(앞 25m, 좌우 12m) 밖에서 끊기고 그 너머는 배경색입니다.

`--view3d-interp`는 지수 평활이라 빠른 물체는 표시 위치가 대략 "속도 x 0.15초"만큼 뒤처집니다. 10m/s 상대 속도면 약 1.5m입니다.

파이 실측은 물체가 없는 장면이었습니다. 사람이나 차가 있는 장면에서 데모 전체 FPS는 확인 필요입니다.

## 5. 다시 재는 명령

파이 저장소 폴더에서 .venv를 켠 뒤 실행합니다. 결과는 out/measure/ 와 out/video/ 에 생깁니다. 각 줄 앞에 아래 첫 줄로 Hailo 온도를 읽어 85도 이하인지 확인합니다. `Device().control...` 처럼 한 줄로 부르면 장치가 먼저 놓여 값을 읽지 못하므로 변수에 담아 씁니다.

```bash
python3 -c "from hailo_platform import Device; d=Device(); t=d.control.get_chip_temperature(); print(round(max(t.ts0_temperature, t.ts1_temperature), 1))"
timeout -s INT -k 10 50 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --view 3d --view3d-interp --headless --duration 20 --warmup 30 --profile --profile-out out/measure/10_pi_3d.json --health-csv out/measure/10_pi_3d.csv --health-interval 10 --hailo-stop-temp 95
timeout -s INT -k 10 50 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --view 3d --view3d-interp --headless --duration 17 --warmup 30 --profile --profile-out out/measure/10_pi_3d_rec.json --health-csv out/measure/10_pi_3d_rec.csv --health-interval 10 --hailo-stop-temp 95 --record-video out/video/10_render3d_pi.mp4
timeout 120 python3 scripts/bench_render3d.py --out out/measure/10_bench_render3d_pi.json
```

맥 시뮬레이션 영상은 저장소 폴더에서 아래 명령으로 만들고, 결과는 out/video/10_render3d_sim.mp4에 생깁니다.

```bash
/usr/bin/python3 scripts/run_demo.py --backend sim --view 3d --view3d-interp --fps 15 --frames 225 --headless --save out/video/10_render3d_sim.mp4
```
