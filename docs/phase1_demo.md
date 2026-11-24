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

| 파일 | 역할 |
|---|---|
| `realityview/camera_model.py` | 카메라 모델(핀홀, 어안), 픽셀과 지면 사이의 변환, PnP 결과를 자세로 변환 |
| `realityview/bev.py` | BEV 격자, IPM 합성(`cv2.remap` LUT), 검출 결과를 지면 객체로 변환, 중복 병합 |
| `realityview/detectors.py` | `HailoBackend`(라즈베리파이), `UltralyticsBackend`(PC 개발용) |
| `realityview/tracker.py` | BEV 평면에서 가장 가까운 것끼리 맞추는 추적 |
| `realityview/render.py` | 위에서 본 그림 렌더링과 대시보드 |
| `realityview/sim.py` | 하드웨어 없이 확인하는 합성 장면과 가짜 검출기 |
| `scripts/run_demo.py` | 실행 스크립트 |
| `scripts/calib_intrinsics.py`, `scripts/calib_extrinsics.py` | 카메라 보정 |
| `configs/two_cam.yaml`, `configs/four_cam.yaml` | 카메라 배치 예시 |

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
