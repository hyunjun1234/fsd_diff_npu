# 리얼리티 뷰 1단계 데모: 카메라별 검출과 세그멘테이션 뒤 BEV 합성

이 문서는 리얼리티 뷰 1단계 데모 코드의 구조와 설치, 실행, 카메라 보정 방법을 설명하는 안내서입니다. 이 코드는 [엣지 NPU 구성과 모델 선정](edge_npu_model_selection.md)의 1단계(방식 B)를 구현한 것입니다. 카메라 영상마다 NPU에서 객체를 검출하고(도로 세그멘테이션은 선택), CPU에서 지면이 평평하다고 가정해 위에서 본 그림(BEV)에 합쳐 리얼리티 뷰처럼 그립니다.

![카메라 2대 시뮬레이션](images/phase1_sim_two_cam.png)

왼쪽은 위에서 본 그림입니다. 흰 차가 내 차이고 원은 5m 간격입니다. 오른쪽은 카메라별 검출 결과입니다. 카메라 4대 구성은 [이 그림](images/phase1_sim_four_cam.png)처럼 좌우 사각지대까지 덮습니다.

## 구조

```
카메라 N대 --> NPU (Hailo-8)                    --> CPU                                    --> 화면
              - YOLOv8 검출 (NMS 내장 HEF)          - 접지점을 지면 좌표로 변환 (광선과 평면의 교차)
              - 도로 세그 HEF (선택, N프레임마다)     - 카메라 사이의 중복 병합, BEV 추적
                                                   - IPM LUT로 바닥 영상과 도로 마스크 합성
```

- `realityview/camera_model.py`: 카메라 모델(핀홀, 어안), 픽셀과 지면 사이의 변환, PnP 결과를 자세로 변환
- `realityview/bev.py`: BEV 격자, IPM 합성(`cv2.remap` LUT), 검출 결과를 지면 객체로 변환, 중복 병합
- `realityview/detectors.py`: `HailoBackend`(라즈베리파이), `UltralyticsBackend`(PC 개발용)
- `realityview/tracker.py`: BEV 평면에서 가장 가까운 것끼리 맞추는 추적
- `realityview/render.py`: 위에서 본 그림 렌더링과 대시보드
- `realityview/render3d.py`: 자차 뒤쪽 위에서 본 3D 시점 렌더링(`--view 3d`)
- `realityview/sim.py`: 하드웨어 없이 확인하는 합성 장면과 가짜 검출기
- `realityview/multicam.py`: USB 카메라, 쿼드 키트 칸 나누기, 카메라 간 시간 맞춤(`FrameSyncer`)
- `scripts/run_demo.py`: 실행 스크립트
- `scripts/calib_intrinsics.py`, `scripts/calib_extrinsics.py`: 카메라 보정
- `configs/two_cam.yaml`, `configs/four_cam.yaml`: 카메라 배치 예시

## PC에서 바로 해 보기 (하드웨어 없음)

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/run_demo.py --backend sim --frames 150 --save out/sim.mp4 --headless
python scripts/run_demo.py --config configs/four_cam.yaml --backend sim --frames 150 --save out/sim4.mp4 --headless
```

녹화한 영상이 있으면 PC에서 실제 검출기로 확인할 수 있습니다. `pip install ultralytics`가 필요합니다.

```bash
python scripts/run_demo.py --backend ultralytics --video front=rec/front.mp4 --video rear=rec/rear.mp4
```

## 3D 시점으로 보기

`--view 3d`를 주면 자차 뒤쪽 위의 가상 카메라에서 내려다본 화면을 크게 그리고, 카메라 영상은 오른쪽에 작게 둡니다. `--view both`는 왼쪽에 탑뷰를 더합니다. 옵션을 주지 않으면 지금처럼 탑뷰(`bev`)입니다. 코드는 `realityview/render3d.py`에 있습니다.

```bash
python scripts/run_demo.py --backend sim --view 3d --view3d-interp --fps 15 --frames 225 --save out/video/sim3d.mp4 --headless
```

결과 영상은 `out/video/sim3d.mp4`에 생깁니다. 바닥은 IPM 바닥 영상과 도로 마스크를 원근으로 편 것이고, 차와 트럭은 반투명 박스, 사람은 세로 원기둥으로 그립니다. 먼 물체부터 그려서 가까운 물체가 앞을 가립니다. 박스 위 숫자는 트랙 ID, 바닥의 화살표는 1초 뒤 위치까지의 상대 속도입니다.

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--view3d-cam` | 가상 카메라 뒤 거리, 높이, 숙임각, 수평 화각(실차 m, 도) | 7,4.5,18,70 |
| `--view3d-size` | 3D 화면 크기 | 960x540 |
| `--view3d-interp` | 트랙 표시 위치를 프레임 사이에서 부드럽게 보간 | 꺼짐 |
| `--view3d-scale` | 이 비율로 작게 그린 뒤 키움(0.1~1) | 1 |
| `--view3d-ground-every` | 바닥 텍스처를 N 프레임마다 다시 만듦 | 1 |

화면이 느리면 `--view3d-ground-every 3 --view3d-scale 0.5`를 먼저 씁니다. 바닥을 다시 만드는 일이 3D 그리기 시간의 대부분이라 갱신 주기를 늘리는 쪽이 효과가 큽니다. 모형 설정(`scene.scale`)에서는 가상 카메라 위치와 물체 크기에 같은 축척을 곱하므로 실차 설정과 같은 화면이 나옵니다. 설정 파일에 `view3d:` 항목을 두면 `View3DConfig`의 기본값을 바꿀 수 있고, 명령줄 옵션이 그 위에 덮어씁니다.

3D 그리기만 따로 재려면 `python3 scripts/bench_render3d.py --out out/measure/bench_render3d.json`을 실행합니다. 카메라와 NPU 없이 가짜 트랙 0, 10, 20개로 재고, 결과는 `--out` 경로에 생깁니다.

## 카메라 4대 구성

카메라마다 설정의 `source.type`으로 입력을 고릅니다. 코드는 `realityview/multicam.py`에 있습니다.

CSI 카메라는 지금처럼 `{type: picamera2, index: 0}`을 씁니다. 파이 5의 CSI 포트는 2개라 3번째, 4번째 카메라는 USB나 쿼드 키트로 채웁니다.

USB 카메라는 `{type: usb, device: /dev/v4l/by-id/<장치 이름>, fps: 30, fourcc: MJPG}`처럼 적습니다. 파이의 `/dev/video0`~`/dev/video15`는 CSI 장치라서 번호 대신 `ls /dev/v4l/by-id/`로 찾은 경로를 씁니다. 프레임마다 받은 시각(`time.monotonic_ns()`, CSI 센서 시각과 같은 시계)이 붙고, 전송 지연만큼 `timestamp_offset_ms`로 뺄 수 있습니다. 카메라 없이 시험할 때는 `{type: video, uri: out/sim4/cam2.mp4, fps: 30}`으로 동영상 파일을 가짜 USB 카메라로 씁니다. CSI 2대와 가짜 측면 2대를 섞은 예가 `configs/four_cam_mixed.yaml`입니다.

쿼드 키트는 네 카메라가 같은 `group`을 쓰고 칸 번호(`tile`, 왼쪽 위부터 0~3)와 지연 프레임 수(`lag_frames`)를 적습니다. 합쳐진 영상을 읽는 장치는 같은 group의 첫 카메라에 `inner`로 한 번만 적습니다.

```yaml
source: {type: quad, group: kit0, tile: 2, lag_frames: 1,
         inner: {type: picamera2, index: 0, fps: 30, width: 2304, height: 1296}}
```

`align: true`(기본)면 늦지 않은 칸을 지연만큼 묵혀 네 칸이 같은 순간을 보이게 하고, `false`면 시각만 앞당겨 적습니다. 칸 순서와 2번, 3번이 한 프레임 늦는 방향은 실제 키트로 확인 필요입니다.

카메라마다 시각이 다르면 `--sync`를 줍니다. 기준 카메라(`--sync-ref`, 기본 0) 시각에 가장 가까운 프레임끼리 묶고, 허용 오차(`--sync-tol-ms`, 기본 25)를 넘으면 `--sync-miss hold`(가장 가까운 프레임 재사용, 기본) 또는 `blank`(빈 칸)로 채웁니다. `--sync-csv`를 주면 묶음마다 카메라별 시각 차이가 그 csv에 남고, `--skew-comp`는 그 차이만큼 물체 위치를 트랙 속도로 옮겨 추적기에 넣습니다.

```bash
python scripts/make_sim4.py --out out/sim4
python scripts/run_demo.py --config configs/four_cam.yaml --backend gt --replay out/sim4 --headless \
    --view both --tracker robust --frames 200 --save out/video/sim4_replay.mp4
```

첫 명령은 카메라별 시각 어긋남과 프레임 누락을 넣은 합성 4대 영상(`out/sim4/cam0~3.mp4`, 시각 csv, 정답 `gt.json`)을 만들고, 둘째 명령은 그 영상을 시각으로 묶어 재생한 결과를 `out/video/sim4_replay.mp4`에 저장합니다. 시각 어긋남별 추적 지표는 `scripts/eval_tracking.py --config configs/four_cam.yaml --tracker robust --cam-skew-ms 0,0,50,50`로 냅니다.

## 라즈베리파이 5와 Hailo-8에 설치하기

1. Raspberry Pi OS(64bit)를 최신으로 업데이트하고 AI HAT+(26 TOPS)를 장착합니다. 전원은 27W 공식 어댑터를 쓰고 쿨러를 꼭 답니다.
2. `sudo apt install hailo-all python3-picamera2`를 실행하고 재부팅합니다. `hailortcli fw-control identify`로 Hailo-8이 보이는지 확인합니다.
3. PCIe Gen3를 쓰려면 `/boot/firmware/config.txt`에 `dtparam=pciex1_gen=3`을 넣습니다.
4. 가상환경은 시스템 패키지가 보이도록 만듭니다. `python3 -m venv --system-site-packages .venv && . .venv/bin/activate && pip install pyyaml`
5. Hailo Model Zoo에서 Hailo-8용 `yolov8s.hef`(또는 `yolov8m.hef`)를 받아 `models/yolov8s_h8.hef`로 둡니다. Hailo-8L용 HEF는 Hailo-8에서 돌지 않으므로 칩 종류를 맞춥니다.
6. 도로 세그멘테이션을 쓰려면 Model Zoo의 Cityscapes 시맨틱 세그멘테이션 HEF를 `npu.seg_hef`에 지정합니다. 출력이 클래스 로짓이든 클래스 맵이든 처리하며, 도로 클래스 번호는 `road_classes`로 지정합니다. 이 단계는 선택입니다.
7. `python scripts/run_demo.py --config configs/two_cam.yaml --backend hailo`로 실행합니다.

## 카메라 보정

보정은 결과 품질에 가장 큰 영향을 줍니다.

내부 파라미터는 체커보드로 구합니다. 내부 코너 9x6, 한 칸 2.5cm 같은 체커보드를 데모 해상도(768x432)로 20장 이상 여러 각도에서 찍은 뒤 아래 명령을 실행합니다. 광각 모듈은 대개 `fisheye` 모델이 더 잘 맞습니다. RMS 1px 이하를 목표로 합니다.

```bash
python scripts/calib_intrinsics.py --images calib/front/*.jpg --model fisheye
```

외부 파라미터는 바닥의 점으로 구합니다. 차 주변 바닥에 마스킹 테이프로 점을 6개 이상 찍고, 각 점의 차량 좌표를 줄자로 잽니다. 좌표는 X가 전방, Y가 좌측이고 원점은 뒷바퀴 축 가운데의 바닥입니다. 각 점의 픽셀 좌표와 함께 `u,v,X,Y` 형식의 CSV로 적은 뒤 아래 명령을 실행합니다. 출력되는 지면 역투영 오차가 10cm 안이면 충분합니다.

```bash
python scripts/calib_extrinsics.py --config configs/two_cam.yaml --camera front --points calib/front_points.csv
```

## 시뮬레이션으로 확인한 내용

단위 테스트는 24개입니다. 투영과 역투영의 왕복(핀홀, 어안, 여러 방향), PnP 자세 복원, IPM 담당 카메라, 접지점 거리, Hailo NMS 출력의 좌표 복원, 세그멘테이션 마스크, 추적, 전체 처리 과정을 검사합니다.

합성 장면에서는 15m 안에 있는 차량의 추적 위치가 정답과 1m 안으로 맞습니다. 이 기준은 테스트로 고정해 두었습니다.

클라우드 x86 CPU에서 잰 CPU 단계의 시간은 카메라 2대일 때 BEV 합성 약 19ms와 렌더링 약 40ms, 4대일 때 약 33ms와 75ms입니다. 라즈베리파이 5에서는 몇 배 느릴 것으로 보이며 측정이 필요합니다. 느리면 `--no-ground`로 바닥 영상 합성을 끄거나 `bev.res`를 0.1로 키웁니다.

## 알려진 한계와 다음 작업

지면이 평평하다고 가정하기 때문에 경사로와 과속방지턱에서는 거리 오차가 커집니다. 2단계에서 FlashOcc로 넘어가는 이유입니다.

내 차의 이동 정보(오도메트리)가 없어 상대 좌표만 알 수 있습니다. 그래서 다른 차량은 내 차와 평행하게 그립니다. IMU나 바퀴 속도를 넣으면 멈춰 있는 물체를 고정된 좌표로 추적하고 방향도 추정할 수 있습니다.

같은 이유로, 내 차가 빠르게 움직이면 멈춰 있는 물체도 빠르게 움직이는 것처럼 보여 추적 ID가 바뀔 수 있습니다. 오도메트리 보정이 다음 개선 항목입니다.

`HailoBackend`와 `PiCameraSource`는 HailoRT와 Picamera2의 공개 API에 맞춰 작성했고, 실제 보드에서의 동작은 확인 필요입니다. 처음 실행할 때 HEF 출력의 이름이나 형식이 다르면 `detectors.py`의 `parse_hailo_nms`만 맞추면 됩니다.

차선 인식은 아직 없고 도로 영역 세그멘테이션만 있습니다. UFLD 계열 같은 차선 전용 모델을 Hailo용으로 컴파일해 넣는 것이 다음 후보입니다.
