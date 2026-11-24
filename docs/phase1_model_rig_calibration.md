# 모형으로 카메라 보정하고 거리 오차 재기

이 문서는 실제 차 대신 책상이나 바닥 위 모형에 카메라 2대를 달고, 내부 보정, 외부 보정, 장난감 차 거리 오차 측정까지 사람이 순서대로 따라 하는 절차입니다. 설정 파일은 configs/two_cam_model.yaml이고, 명령은 모두 파이 저장소 폴더(~/work/fsd_diff_npu)에서 .venv를 켠 뒤 실행합니다.

```bash
cd ~/work/fsd_diff_npu && . .venv/bin/activate
```

카메라는 지금 달린 것을 그대로 씁니다. 전방은 ov5647 130도 광각 렌즈라 어안 모델(OpenCV fisheye)로, 후방은 일반 Camera Module 3(imx708)라 핀홀 모델로 보정합니다. 카메라는 테슬라 모델 Y에서 같은 역할의 카메라 자리([04 문서](04_tesla_camera_layout.md))를 모형 축척으로 줄인 곳에 답니다.

## 1. 준비물

- 모형 틀: 축척 1/10이면 길이 48cm, 폭 19cm 정도의 판이나 틀. 전방 카메라를 14.5cm, 후방 카메라를 10cm 높이에 고정할 수 있어야 합니다. 전방 카메라는 거의 수평이라 그 앞쪽 틀(앞범퍼 자리까지)이 높으면 바닥을 가립니다. 앞쪽은 낮거나 비워 두고, 보닛이 있는 닫힌 차체로 되는지는 확인 필요입니다.
- 평평한 책상이나 바닥. 축척 1/10이면 모형 뒷바퀴 축에서 앞 2.5m, 뒤 1.5m, 좌우 1.2m가 BEV 범위입니다.
- 체커보드 2장: 하나는 내부 보정용으로 단단한 판(폼보드, 클립보드)에 붙이고, 하나는 바닥에 깝니다.
- 쇠자(30cm 이상)와 줄자(mm 눈금), 마스킹 테이프, 네임펜
- 장난감 차 2~3대 또는 컵, 상자 같은 작은 물체

## 2. 체커보드 인쇄

가로 10칸, 세로 7칸, 칸 한 변 20mm인 보드를 씁니다. 내부 코너는 가로 9개, 세로 6개이고 인쇄 크기는 200mm x 140mm라 A4 한 장에 들어갑니다. 아래 명령은 600dpi 기준 이미지를 만듭니다(결과: out/calib/board_9x6_20mm.png).

```bash
mkdir -p out/calib && python3 -c "
import numpy as np, cv2
px = 472  # 20mm at 600dpi
b = np.full((9 * px, 12 * px), 255, np.uint8)
for r in range(7):
    for c in range(10):
        if (r + c) % 2 == 0:
            b[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
cv2.imwrite('out/calib/board_9x6_20mm.png', b)
"
```

"실제 크기" 또는 100%로 인쇄합니다(페이지 맞춤 끄기). 인쇄한 뒤 쇠자로 가로 10칸 길이를 재서 10으로 나눈 값을 칸 크기로 씁니다. 예를 들어 199mm면 `--square 0.0199`입니다. 칸은 15~25mm 사이면 같은 절차로 쓸 수 있고, 그때는 아래 명령의 `--square`만 바꿉니다.

## 3. 카메라 달기와 설정 값 고치기

### 3.1 축척 정하기

configs/two_cam_model.yaml의 `scene.scale`이 실차 대비 모형 비율입니다. 기본값은 0.1(1/10)이고, 모형 차체 길이 / 4.79m(모델 Y 길이)로 정합니다. 설정에는 카메라 위치(`position_car`)와 BEV 범위(`bev_car`)가 실차 m로 적혀 있고, 프로그램이 이 값에 축척을 곱해 씁니다. 각도와 렌즈 화각은 축척과 상관없이 그대로입니다.

### 3.2 원점 표시

원점은 실차 설정과 같은 기준인 모형 뒷바퀴 축 가운데 바닥입니다. 축척 1/10이면 뒤범퍼에서 앞으로 10.1cm인 자리입니다. 원점에서 앞쪽이 +X, 왼쪽이 +Y, 위가 +Z입니다. 책상 위에 테이프로 원점과 X축, Y축을 표시해 두면 이후 측정이 쉽습니다.

### 3.3 카메라 달기

테슬라 좌표(04 문서의 추정치)에 축척을 곱한 자리에 렌즈 중심을 맞춥니다. 아래 표는 축척 1/10일 때 값이고, 다른 축척이면 실차 좌표에 그 축척을 곱합니다.

| 카메라 | 렌즈 중심 자리 (원점 기준) | 방향 |
|---|---|---|
| 전방 (ov5647, 테슬라 전방 광각 자리, 앞유리 상단) | 앞 18.0cm, 오른쪽 0.6cm, 높이 14.5cm (실차 1.80, -0.06, 1.45m) | 정면, 아래로 3도 |
| 후방 (imx708, 테슬라 후방 카메라 자리, 뒷번호판 위) | 뒤 9.5cm, 가운데, 높이 10.0cm (실차 -0.95, 0, 1.00m) | 뒤쪽, 아래로 20도 |

전방은 거의 수평으로 달아서 화면 위쪽 절반에는 바닥이 없고, 바닥은 카메라 앞 약 16cm부터 보입니다. 아래로 3도는 04 문서의 3D 도면에 넣은 추정값이고, 후방 20도도 04 문서가 추정이라고 적은 값이라 둘 다 외부 보정으로 실제 값을 구합니다. 좌우 기울기는 0도로 맞춥니다.

### 3.4 잰 값 넣기

표대로 달았으면 설정을 고치지 않고 다음 장으로 갑니다. 줄자로 잰 렌즈 중심이 표와 1cm 넘게 다르면 그 카메라의 `position_car` 줄을 지우고 `position: [x, y, z]`를 모형 m로 적습니다. `position`이 있으면 `position_car`는 쓰지 않으므로 축척이 두 번 곱해지지 않습니다.

## 4. 내부 보정

### 4.1 촬영

카메라를 테슬라 자리에 단다는 가정은 위치와 각도에 대한 것이고, 렌즈 왜곡 보정(내부 보정)은 이와 별개로 카메라마다 꼭 해야 합니다.

카메라마다 3초 간격으로 25장을 찍습니다. 찍히는 동안 보드를 옮겨 가며 다음을 지킵니다.

- 보드 전체가 화면 안에 들어오고, 거리는 15~40cm로 바꿔 가며 찍습니다.
- 화면 네 구석과 가장자리에 보드가 오는 사진을 여러 장 넣습니다. 전방 어안 렌즈는 가장자리 사진이 특히 중요합니다.
- 보드를 좌우, 위아래로 30~45도 기울인 사진도 넣습니다.
- 흔들리지 않게 잠깐 멈춰 주고, 조명 반사가 보드에 생기지 않게 합니다.

```bash
timeout -s INT -k 5 150 python3 scripts/record_frames.py --config configs/two_cam_model.yaml --camera front --frames 25 --interval 3 --out out/calib/front
timeout -s INT -k 5 150 python3 scripts/record_frames.py --config configs/two_cam_model.yaml --camera rear --frames 25 --interval 3 --out out/calib/rear
```

결과는 out/calib/front/, out/calib/rear/ 에 front_00000.png 같은 이름으로 생깁니다. 설정 파일과 같은 해상도와 센서 모드로 찍히므로 보정값을 그대로 쓸 수 있습니다.

후방 Camera Module 3는 자동 초점 렌즈라 초점 위치에 따라 fx가 조금 바뀝니다. 지금 코드는 초점을 고정하지 않으므로, 보정과 시험 때 초점이 같은지는 확인 필요입니다.

### 4.2 계산

```bash
python3 scripts/calib_intrinsics.py --images 'out/calib/front/front_*.png' --cols 9 --rows 6 --square 0.020 --model fisheye --out out/calib/front_intrinsics.yaml
python3 scripts/calib_intrinsics.py --images 'out/calib/rear/rear_*.png' --cols 9 --rows 6 --square 0.020 --model pinhole --out out/calib/rear_intrinsics.yaml
```

결과는 화면과 out/calib/front_intrinsics.yaml(설정에 붙여 넣을 글), 같은 이름의 .json(사진별 오차)에 남습니다.

- 코너를 찾은 사진이 10장보다 적으면 멈춥니다. 보드를 더 크게, 더 밝게 찍어 다시 합니다.
- 어안 보정에서 조건이 나쁜 사진은 스크립트가 알아서 빼고 이름을 알려 줍니다.
- "오차가 큰 사진"에 나온 사진을 지우고 다시 계산하면 RMS가 내려가는 경우가 많습니다. 지울 때는 out/calib/ 아래에서만 지웁니다.
- RMS가 1px 이하이면 나온 `intrinsics` 글을 configs/two_cam_model.yaml의 해당 카메라 항목에 붙여 넣습니다. 전방의 `max_fov_deg: 140.0` 줄은 남겨 둡니다.

## 5. 외부 보정

### 5.1 바닥 보드 놓기

1. 바닥용 보드를 전방 카메라 앞에 평평하게 깝니다. 축척 1/10이면 보드 가로가 오른쪽(-Y)을 향하게 두고 0번 코너를 원점 앞 50cm, 왼쪽 8cm에 둡니다. 이때 검은 칸은 앞범퍼 자리(원점 앞 38cm)보다 앞에 있습니다. 카메라가 낮고 거의 수평이라 보드를 놓을 수 있는 폭이 좁습니다. 보드가 이보다 5cm 넘게 가까우면 아래가 화면 밖으로 나가고, 멀면 납작하게 찍혀 코너 순서가 틀릴 수 있습니다.
2. 보드의 0번 코너(검은 칸과 흰 칸이 만나는 안쪽 코너 중 하나) 위치를 원점에서 줄자로 재서 X, Y를 적습니다. 예: 앞으로 50cm, 왼쪽 8cm면 X=0.50, Y=0.08입니다.
3. 0번 코너에서 1번 코너로 가는 방향(보드 가로 방향)이 모형 X축과 이루는 각을 정합니다. 보드 가로가 오른쪽(-Y)을 향하면 -90, 앞(+X)을 향하면 0입니다.

어느 코너가 0번인지는 처음 한 번 계산을 돌린 뒤 생기는 _corners.png에서 확인합니다(빨간 원이 0번, 초록 원이 1번). 생각한 코너와 다르면 X, Y와 각도를 그 코너 기준으로 다시 재서 돌립니다. 재투영 오차가 몇 화소 넘게 크게 나오면 코너 순서가 틀렸을 가능성이 크니 보드를 카메라 쪽으로 5cm 당겨 다시 찍습니다.

### 5.2 촬영과 계산

```bash
timeout -s INT -k 5 60 python3 scripts/record_frames.py --config configs/two_cam_model.yaml --camera front --frames 1 --out out/calib/front_floor
python3 scripts/calib_extrinsics.py --config configs/two_cam_model.yaml --camera front --board-image out/calib/front_floor/front_00000.png --cols 9 --rows 6 --square 0.020 --origin 0.50 0.08 --board-yaw -90 --out out/calib/front_extrinsics.yaml
```

결과는 화면과 out/calib/front_extrinsics.yaml에 남고, 코너 그림은 out/calib/front_floor/front_00000_corners.png에 생깁니다. 나온 `extrinsics` 글을 설정 파일 전방 카메라 항목에 붙여 넣습니다.

후방도 같게 합니다. 0번 코너를 원점 뒤 40cm(카메라 뒤 약 30cm)에 두면 X가 음수입니다(예: X=-0.40). 후방 카메라는 화각이 좁아서 보드 가운데가 카메라 정면에 와야 코너를 찾습니다(가로가 왼쪽을 향하게 두면 Y=-0.08). 명령의 `--camera`, 폴더 이름, `--origin`, `--board-yaw`만 바꿉니다.

```bash
timeout -s INT -k 5 60 python3 scripts/record_frames.py --config configs/two_cam_model.yaml --camera rear --frames 1 --out out/calib/rear_floor
python3 scripts/calib_extrinsics.py --config configs/two_cam_model.yaml --camera rear --board-image out/calib/rear_floor/rear_00000.png --cols 9 --rows 6 --square 0.020 --origin -0.40 -0.08 --board-yaw 90 --out out/calib/rear_extrinsics.yaml
```

### 5.3 검증

보드를 처음과 다른 자리(옆으로 옮기거나 돌려서)로 옮겨 위치를 다시 재고, 풀지 않고 오차만 봅니다. 아래 예는 보드를 돌려 가로가 앞(+X)을 향하게 둔 것입니다(`--board-yaw 0`). `--max-ground-err`에는 7장의 지면 역투영 기준을 m로 줍니다(축척 1/10이면 0.010).

```bash
timeout -s INT -k 5 60 python3 scripts/record_frames.py --config configs/two_cam_model.yaml --camera front --frames 1 --out out/calib/front_check
python3 scripts/calib_extrinsics.py --config configs/two_cam_model.yaml --camera front --verify --board-image out/calib/front_check/front_00000.png --cols 9 --rows 6 --square 0.020 --origin 0.40 0.0 --board-yaw 0 --max-ground-err 0.010
```

화면에 지면 역투영 오차 평균, 최대와 합격 여부가 나옵니다. 후방도 같은 방법으로 확인합니다. 예를 들어 보드 가로가 뒤(-X)를 향하게 돌려 0번 코너를 X=-0.30, Y=0에 두면 `--origin -0.30 0.0 --board-yaw 180`입니다.

## 6. 장난감 차로 거리 오차 재기

1. 장난감 차를 전방 카메라 앞에 놓고, 차 바닥 가운데 위치를 원점에서 줄자로 잽니다(예: X=0.60, Y=0.05).
2. 한 장 찍고 위치를 계산합니다. 장난감은 car로 잡히지 않을 수 있어 `--all-classes`로 모든 클래스를 봅니다.

```bash
timeout -s INT -k 5 60 python3 scripts/record_frames.py --config configs/two_cam_model.yaml --camera front --frames 1 --out out/calib/toy_front_060
timeout -s INT -k 5 60 python3 scripts/locate_objects.py --config configs/two_cam_model.yaml --camera front --image out/calib/toy_front_060/front_00000.png --all-classes --truth 0.60 0.05
```

결과는 화면과 out/calib/toy_front_060/front_00000_locate.json, 상자와 X, Y를 그린 _locate.png에 남습니다. "줄자 위치에 가장 가까운 검출"의 오차를 기록합니다.

3. 축척 1/10이면 차를 X 0.45m, 0.55m, 0.65m, 0.75m, 옆으로 몇 cm씩 옮겨 가며 위치마다 반복합니다. 모두 앞범퍼 자리(0.38m) 앞이고 7장의 재는 범위(앞범퍼에서 약 0.5m) 안입니다. 폴더 이름에 거리를 넣어 섞이지 않게 합니다. 후방도 같은 방법으로 X를 음수로 놓고 잽니다.
4. 검출이 안 되면 더 큰 장난감이나 컵, 병처럼 COCO 클래스에 있는 물체로 바꿉니다. 차량류(car, truck, bus)는 상자 아래 끝에서 길이 절반(실차 4.5m x scene.scale)만큼 안쪽을 중심으로 보므로, 장난감 길이가 이 값과 크게 다르면 오차가 그만큼 생깁니다. 이때는 차 바닥 가운데 대신 카메라에 가까운 바퀴 끝 위치를 함께 적어 둡니다.

## 7. 완료 기준

실차 기준(구현 계획 4.2절)을 모형 크기에 맞게 비례로 줄입니다. 비율 s는 설정의 `scene.scale`입니다.

- 내부 보정: 카메라마다 RMS 1px 이하(실차와 같음, 화소 단위라 줄이지 않음)
- 외부 보정: 검증 보드의 지면 역투영 오차 평균 10cm x s 이하
- 물체 위치: 5m x s 안에 놓은 장난감의 위치 오차 0.5m x s 이하
- 속도: 카메라 2대 10 FPS 이상(실차와 같음)

| 축척 s | 지면 역투영 오차 평균 | 물체 위치 오차 (재는 범위) |
|---|---|---|
| 0.1 (기본값) | 1.0cm 이하 | 5.0cm 이하 (0.50m 안) |
| 0.125 | 1.3cm 이하 | 6.3cm 이하 (0.63m 안) |
| 0.15 | 1.5cm 이하 | 7.5cm 이하 (0.75m 안) |

위 표는 실차 기준을 산술로 줄인 값입니다. 모형에서 이 값이 실제로 달성 가능한지는 처음 측정 뒤에 다시 정합니다.

## 8. 결과 정리

보정 사진, 결과 yaml, 오차 기록은 out/calib/ 에 둡니다(저장소에 올라가지 않음). 고친 configs/two_cam_model.yaml은 저장소에 올리고, 측정한 오차 값은 out/measure/ 에 문서로 남깁니다.
