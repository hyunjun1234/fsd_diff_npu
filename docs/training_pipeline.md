# 학습 파이프라인

이 문서는 학습 서버에서 주행 녹화로 한 회차를 돌리는 방법입니다. 프레임 선별, 자동 라벨, CVAT 검수, YOLOv8s 학습, 고정 시험 세트 평가, Hailo-8과 DX-M1 변환, 파이 배포와 되돌리기까지 다룹니다. 계획의 근거는 docs/07_vehicle_mount_and_training_plan.md 6~10장에 있습니다.

학습 서버는 Windows 데스크탑(RTX 4070 12GB)의 WSL2 Ubuntu 22.04이고, 이 문서는 WSL2 안에서 `nvidia-smi`가 GPU를 보여 주는 상태에서 시작합니다. WSL2와 CUDA 설치 순서는 [07 문서 6장](07_vehicle_mount_and_training_plan.md)에 있습니다. 그 순서대로 CUDA 12.5 툴킷을 설치했다면, 아래 cu121 PyTorch로 Grounding DINO 확장을 빌드할 때 부 버전이 달라 경고가 나올 수 있습니다(동작 여부 확인 필요). 명령은 모두 저장소 맨 위 폴더에서 실행합니다.

## 1. 한 회차 흐름

1. 파이의 주행 녹화와 트리거 클립을 서버로 옮깁니다.
2. 프레임을 고릅니다. 처음 한 번은 고정 시험 세트를 정해 잠급니다.
3. Grounding DINO와 SAM 2로 자동 라벨을 붙이고, 비교 라벨러(YOLOv8s)와 어긋나는 프레임을 의심 프레임으로 표시합니다.
4. 의심 프레임 전부와 나머지의 무작위 표본을 CVAT에서 검수합니다. 시험 세트는 전부 사람이 라벨링합니다.
5. 검수 결과를 들여와 자동 라벨 품질을 재고, 최종 라벨을 만듭니다.
6. 직전 회차 best.pt에서 시작해 이번 회차 데이터, 이전 회차 일부, COCO 일부를 섞어 학습합니다.
7. 고정 시험 세트에서 직전 회차와 비교하고, 예전 장면 성능이 떨어지면 경고를 봅니다.
8. ONNX, HEF, dxnn을 만들어 파이로 보냅니다.

`training/run_round.py`가 2~8을 차례로 부릅니다. 4번 검수에서 사람이 필요하므로 두 번에 나눠 실행합니다.

## 2. 폴더 배치

```
out/drives/<주행>/seg_0000/<카메라>.mjpeg, <카메라>.csv       파이에서 옮긴 녹화
out/triggers/<실행>/<클립>/trigger.json, <카메라>.mjpeg       파이에서 옮긴 트리거 클립
out/training/test_set/                                        고정 시험 세트(회차와 상관없이 하나)
    images/  labels/  test_set.csv  lock.json  review/
out/training/datasets/r001/                                   회차 폴더
    images/  frames.csv  meta.json
    auto/<라벨러>/labels, masks, scores   auto/suspects.csv
    review/review_list.csv, review_images.zip, cvat_upload.zip, cvat_export.zip, reviewed/
    labels/  masks/  label_sources.csv                        학습에 쓰는 최종 라벨
    own_train.txt  train.txt  val.txt  data.yaml
    seg/img_dir, seg/ann_dir   qa/   eval/   calib/   export/
models/r001/best.pt, train_info.json, det/                    학습 결과
models/yolov8s_r001.onnx, .hef, .dxnn, .sha256                변환 결과
```

out/과 models/는 git에 올라가지 않습니다. 다른 디스크를 쓰려면 `training/configs/round.yaml`의 `paths`를 바꿉니다.

프레임 이름은 `<주행>__<세그먼트>__<카메라>__<프레임 6자리>`이고, 트리거 클립은 주행 자리에 `trig`가 들어갑니다. 이미지, 라벨, 마스크, 점수, CVAT 이미지 이름이 모두 이 이름을 씁니다.

라벨 txt는 YOLO 형식이고 클래스 번호는 COCO 번호(사람 0, 자전거 1, 차 2, 오토바이 3, 버스 5, 트럭 7)입니다. 파이 후처리가 COCO 번호를 읽으므로 재학습 모델도 80 클래스 머리를 유지합니다. 도로 마스크 png는 0이 도로 아님, 1이 도로, 255가 모름입니다.

녹화 형식은 파이의 주행 녹화와 트리거 클립(docs/drive_recording.md)과 같습니다. `.mjpeg`는 JPEG를 이어 붙인 파일이고, 같은 이름의 csv에 프레임마다 바이트 위치(offset, size)와 센서 시각(sensor_ns)이 있어 필요한 프레임만 바로 꺼냅니다. 트리거 종류와 점수는 trigger.json의 events 목록에서 읽습니다. mp4, avi 같은 영상과 record_frames.py 형식(`<카메라>_00000.png`와 timestamps.csv)도 읽습니다. 형식이 바뀌면 `round.yaml`의 `recording`과 `triggers` 항목만 고칩니다.

## 3. 서버 설치

주 가상환경 하나에 자동 라벨과 검출 학습을 넣고, 세그, Hailo, DEEPX는 각각 따로 둡니다. 판 요구가 서로 달라 한 환경에 넣으면 충돌합니다.

```
sudo apt install -y python3.10-venv git rsync
python3 -m venv ~/venv-train && . ~/venv-train/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r training/requirements.txt
mkdir -p ~/src && cd ~/src
git clone https://github.com/IDEA-Research/GroundingDINO && pip install -e GroundingDINO
git clone https://github.com/facebookresearch/sam2 && pip install -e sam2
cd -
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -m pytest -q training
```

마지막 줄이 `True NVIDIA GeForce RTX 4070`과 시험 통과를 보이면 됩니다. Grounding DINO는 CUDA 확장을 빌드하므로 cu121에 맞는 nvcc가 필요합니다.

모델 가중치는 아래 자리에 둡니다. 받는 주소는 각 저장소 README에 있습니다.

| 파일 | 자리 | 설정 항목 |
|---|---|---|
| Grounding DINO SwinT 설정 | models/gdino/GroundingDINO_SwinT_OGC.py | autolabel.grounding_dino.config |
| Grounding DINO 가중치 | models/gdino/groundingdino_swint_ogc.pth | autolabel.grounding_dino.weights |
| SAM 2.1 Large | models/sam2/sam2.1_hiera_large.pt | autolabel.sam2.checkpoint |
| YOLOv8s COCO | models/yolov8s.pt | train_det.base_weights |

SAM 2 설정 이름(`configs/sam2.1/sam2.1_hiera_l.yaml`)은 sam2 패키지 안의 이름이라 경로가 아닙니다. models/yolov8s.pt가 없으면 Ultralytics가 처음 실행할 때 받습니다.

세그 환경과 Hailo, DEEPX 환경은 `training/requirements.txt` 아래쪽 주석을 따릅니다. Hailo DFC는 파이의 HailoRT 판에, DX-COM은 파이의 DX-RT 판에 맞춥니다(확인 필요).

## 4. 녹화 옮기기

서버에서 파이의 녹화를 가져옵니다. 주소는 자리표시자입니다.

```
rsync -av --partial <PI_USER>@<PI_HOST>:<PI_REPO>/out/drives/ out/drives/
rsync -av --partial <PI_USER>@<PI_HOST>:<PI_REPO>/out/triggers/ out/triggers/
```

결과는 서버의 out/drives/, out/triggers/에 생깁니다. docs/drive_recording.md의 예처럼 파이에서 /data/fsd_drives/, /data/fsd_triggers/로 보냈다면 회차 설정의 `paths.drives_root`, `paths.triggers_root`를 그 경로로 바꿉니다. 옮긴 뒤 `du -sb`로 양쪽 크기를 비교하고, 같을 때만 파이 쪽을 지웁니다.

## 5. 회차 설정

`training/configs/round.yaml`을 회차마다 복사해 씁니다.

```
cp training/configs/round.yaml training/configs/r001.yaml
```

바꿀 곳은 `round`(r001, r002처럼 r과 숫자 3자리)와 `drives`(이번 회차에 새로 쓸 주행 폴더 이름)입니다. 같은 주행을 다시 적어도 이전 회차가 이미 쓴 프레임은 빠집니다. COCO를 섞으려면 `paths.coco_root`에 COCO의 YOLO 형식 폴더(images/train2017, labels/train2017)를 적습니다. 비워 두면 COCO 없이 학습하고 경고만 냅니다.

장면을 손으로 붙이려면 `select.scene_tags_csv`에 csv를 줍니다. 열은 drive, segment, camera, start_s, end_s, scene이고, camera를 비우면 모든 카메라에 붙습니다. 태그가 없는 프레임은 밝기로 day와 night를 나눕니다.

## 6. 1부: 선별, 자동 라벨, CVAT 내보내기

```
python3 -m training.run_round --config training/configs/r001.yaml
```

결과는 out/training/datasets/r001/ 아래에 생기고, 마지막 줄에 CVAT에 올릴 zip 경로가 나옵니다. 단계별 수는 meta.json에 남습니다.

선별 규칙은 다음과 같습니다. 영상에서 `every_sec`(기본 1초)마다 한 장을 보고, 라플라시안 분산이 `blur_min`(밤은 `blur_min_night`)보다 작으면 흐린 프레임으로 버립니다. 같은 카메라에서 최근에 고른 프레임과 32x18 축소 영상 차이가 `dup_diff`보다 작으면 정차 중 같은 장면으로 보고 버립니다. 트리거 클립 프레임을 `trigger_max_ratio`까지 먼저 넣고, 나머지는 장면별로 같은 몫을 나눠 시간상 고르게 뽑습니다. 문턱값은 합성 영상으로만 맞췄으므로 첫 주행 녹화로 조정합니다(확인 필요).

고정 시험 세트가 없으면 선별 전에 먼저 정해 out/training/test_set/lock.json으로 잠급니다. 기본(`exclude: segment`)은 세그먼트의 15%를 장면별로 골라 통째로 시험용으로 떼어 두고, 그 세그먼트는 어느 회차에서도 학습, 검증, 보정 이미지에 들어가지 않습니다. 1초 옆 프레임은 거의 같은 장면이라 프레임만 빼면 시험 점수가 부풀기 때문입니다. 학습 목록을 만들 때마다 같은 검사를 한 번 더 하므로 잠금을 우회해 섞일 수 없습니다. 나중 주행으로 시험 세트를 늘리려면 아래처럼 덧붙입니다. 이미 들어간 프레임은 빠지지 않습니다.

```
python3 -m training.select_frames --config training/configs/r003.yaml --add-to-test-set
```

덧붙이기는 그 회차의 선별(run_round 1부)보다 먼저 실행합니다. 선별을 먼저 하면 새 시험 세그먼트의 프레임이 이미 그 회차 학습 프레임에 들어가 있어, 평가의 새 장면 값이 부풀기 때문입니다.

`exclude: window`를 고르면 세그먼트를 통째로 빼지 않고 시험 프레임 앞뒤 `window_sec`초만 학습에서 뺍니다. 녹화 세그먼트가 적은 첫 회차에 학습 데이터를 더 남기고 싶을 때 씁니다.

단계를 따로 돌릴 수도 있습니다.

```
python3 -m training.select_frames --config training/configs/r001.yaml
python3 -m training.autolabel --config training/configs/r001.yaml
python3 -m training.cvat_io export --config training/configs/r001.yaml
```

자동 라벨은 주 라벨러(`autolabel.primary`, 기본 Grounding DINO)의 박스 가운데 점수 `keep_score`(0.35) 이상을 라벨로 남깁니다. 도로 마스크는 기본으로 영상 아래쪽 세 점을 SAM 2에 주어 만들고(`road: points`), 보닛이 보이는 카메라는 `sam2.road_points`를 위로 올립니다. `road: text`로 바꾸면 Grounding DINO에 "road ."를 물어 나온 박스를 SAM 2에 줍니다. 의심 프레임은 점수가 `ambiguous`(0.3~0.5) 구간인 박스가 있거나, 비교 라벨러와 같은 클래스끼리 IoU 0.5로 짝지었을 때 짝 없는 박스가 있는 프레임입니다.

트리거 클립처럼 촘촘한 프레임은 영상 추적 라벨(Grounded-SAM-2 방식)로 채울 수 있습니다. 프레임 폴더 이름은 SAM 2 규칙대로 00000.jpg 꼴이어야 합니다.

```
python3 -m training.autolabel --config training/configs/r001.yaml --track-dir out/tmp/clip01_front --out out/training/datasets/r001/track/clip01_front
```

결과는 --out 폴더의 labels/, scores/에 생깁니다. mjpeg 클립은 아래처럼 숫자 이름 JPEG로 풀어서 넣습니다(ffmpeg 필요).

```
mkdir -p out/tmp/clip01_front && ffmpeg -f mjpeg -i out/triggers/<실행>/<클립>/front.mjpeg -q:v 2 -start_number 0 out/tmp/clip01_front/%05d.jpg
```

## 7. CVAT 검수

CVAT 커뮤니티판을 WSL2의 Docker로 띄웁니다. 판은 설치할 때 고르고 기록합니다(확인 필요).

```
git clone https://github.com/cvat-ai/cvat ~/src/cvat && cd ~/src/cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Windows 브라우저에서 http://localhost:8080 으로 들어갑니다.

프로젝트를 하나 만들고 라벨을 아래처럼 둡니다. 이름이 다르면 들여올 때 버려집니다.

| 라벨 | 도구 | 쓰임 |
|---|---|---|
| person, bicycle, car, motorcycle, bus, truck | 사각형 | 검출 |
| road | 다각형 | 도로 영역 |
| not_road | 다각형 | 도로 안의 빈 곳(도로 위 차 등) |

회차마다 작업 하나를 만듭니다.

1. 작업 만들기에서 `review/review_images.zip`을 올립니다.
2. 작업의 Actions 메뉴에서 Upload annotations를 고르고, 형식 CVAT for images 1.1로 `review/cvat_upload.zip`을 올립니다. `review.format: yolo`로 내보냈다면 YOLO 1.1을 고르며, 이때는 도로가 빠집니다.
3. 박스를 고칩니다. 화면에 보이는 물체는 작아도 모두 칠하고, 절반 넘게 가려진 물체도 보이는 부분만 칠합니다. 물체가 없는 프레임은 빈 채로 둡니다. 빈 프레임도 "물체 없음"이라는 사람 판단으로 들여옵니다.
4. 도로는 road 다각형의 꼭짓점을 고치고, 도로 위 차나 사람 자리는 not_road 다각형으로 둡니다. road 다각형이 하나도 없는 프레임은 도로를 검수하지 않은 것으로 보고 자동 마스크를 그대로 씁니다.
5. Actions 메뉴의 Export task dataset에서 형식 CVAT for images 1.1, 이미지 저장 없이 내보내고, 받은 zip을 `out/training/datasets/r001/review/cvat_export.zip`으로 둡니다.

시험 세트도 같은 방법으로 작업을 하나 따로 만듭니다. 파일은 out/training/test_set/review/ 아래에 있고, 내보낸 zip은 `out/training/test_set/review/cvat_export.zip`으로 둡니다. 시험 세트는 자동 라벨로 미리 채워 두지만 모든 프레임을 사람이 확인해야 합니다. 들여온 프레임만 test_set.csv의 label_source가 human이 되고, 평가는 human 프레임만 씁니다.

처음 회차에는 검수 비율을 높게(`review.random_ratio: 0.3` 정도) 두고, 8장의 자동 라벨 품질 표를 본 뒤 다음 회차 비율을 정합니다. 의심 프레임은 어려운 것만 모은 것이라 전체 품질은 무작위 표본 열로 판단합니다.

## 8. 2부: 들여오기부터 평가까지

```
python3 -m training.run_round --config training/configs/r001.yaml --from review_import --until evaluate
```

단계별 결과 위치는 다음과 같습니다.

| 단계 | 결과 | 볼 것 |
|---|---|---|
| review_import | review/reviewed/ | 들여온 프레임 수, 모르는 이름 수 |
| label_qa | qa/label_qa.md | 박스 IoU, 놓친 비율, 잘못 잡은 비율 |
| merge | labels/, masks/, label_sources.csv | 사람 라벨 수와 자동 라벨 수 |
| train_det | data.yaml, models/r001/best.pt | 섞은 수, 시작 가중치 |
| seg_prepare | seg/img_dir, seg/ann_dir | 세그 학습, 검증 장수 |
| evaluate | eval/eval.md | 경고 줄 |

학습만 다시 하려면 아래를 씁니다. 결과는 models/r001/에 생깁니다.

```
python3 -m training.train_det --config training/configs/r001.yaml
```

학습 목록은 이번 회차 학습 프레임 전부에, 이전 회차 학습 프레임을 그 수의 `prev_ratio`(0.5)배, COCO 이미지를 `coco_ratio`(0.5)배만큼 무작위로 더합니다. 검증은 세그먼트 단위로 나눕니다. 시작 가중치는 직전 회차의 models/<직전>/best.pt이고, 첫 회차는 yolov8s.pt입니다. 기본값은 imgsz 640(파이 HEF, dxnn 입력과 같음), batch 16, epochs 50, patience 20, workers 8이고 근거는 training/train_det.py 주석에 있습니다. 첫 학습 로그의 GPU_mem 열이 8GB 아래면 batch를 32로 올려도 됩니다.

우리 라벨에는 6 클래스만 있어 신호등 같은 나머지 COCO 클래스는 우리 영상에서 배경으로 배우게 됩니다. COCO를 섞는 이유 중 하나이고, 나머지 클래스 성능 변화는 확인 필요입니다.

세그 학습은 세그 환경에서 따로 돌립니다. mmsegmentation의 BiSeNetV2 Cityscapes 설정을 읽어 클래스 수와 경로만 바꿉니다. 이 부분의 API 이름은 설치한 판에서 확인 필요입니다.

```
. ~/venv-seg/bin/activate
python3 -m training.train_seg --config training/configs/r001.yaml --run
```

결과는 models/r001/seg/에 생깁니다. 기본 라벨은 0 도로, 1 그 밖(`label_mode: road_first`)이라 파이의 `road_classes=(0,)`을 바꾸지 않아도 되지만 출력 클래스 수가 19에서 2로 바뀝니다. 세그 모델의 ONNX, HEF, dxnn 변환 명령은 아직 없습니다.

평가는 직전 회차 모델과 이번 회차 모델을 같은 시험 세트에서 매번 다시 돌려 비교합니다. 예전 장면(이전 회차에 시험 세트에 들어간 프레임)에서 mAP50이나 mAP50-95가 0.01 넘게, 박스 10개 이상인 클래스의 재현율이 0.03 넘게 떨어지면 경고합니다. 경고가 나오면 `prev_ratio`와 `coco_ratio`를 올려 train_det부터 다시 돌립니다. eval/eval.md는 docs/measurements/에 그대로 붙일 수 있는 표입니다.

다른 모델이나 파이 NPU 예측도 같은 식으로 잴 수 있습니다. 폴더를 주면 `<프레임 이름>.txt`(클래스 cx cy w h 점수)를 예측으로 읽습니다.

```
python3 -m training.evaluate --config training/configs/r001.yaml --model r001=models/r001/best.pt --model r001_hailo=out/pi_preds/r001_hailo
```

## 9. 변환과 배포

```
python3 -m training.run_round --config training/configs/r001.yaml --from export --until export
```

기본은 명령을 출력만 합니다. 보정 이미지(out/training/datasets/r001/calib/, 학습 프레임에서 장면별로 1024장)와 DX-COM 설정(export/dxcom_yolov8s_r001.json)은 이때 만들어지고, 명령은 export/commands.sh에 모입니다. 명령은 다섯 줄입니다.

1. `yolo export ... format=onnx opset=11`로 models/yolov8s_r001.onnx를 만듭니다.
2. Hailo 환경에서 `hailomz compile yolov8s --ckpt ... --hw-arch hailo8 --calib-path ... --classes 80`으로 models/yolov8s_r001.hef를 만듭니다. Hailo-8은 Model Zoo v2.x 브랜치와 DFC 3.x를 씁니다.
3. DEEPX 환경에서 `dxcom -m ... -c ... -o ...`로 models/yolov8s_r001.dxnn을 만듭니다.
4. 세 파일의 sha256을 models/yolov8s_r001.sha256에 적습니다.
5. rsync로 파이의 models/에 보냅니다. `export.pi_target`의 주소를 바꾸기 전에는 commands.sh에서 주석으로 남습니다.

각 환경을 켠 상태에서 한 줄씩 실행하거나, 세 도구가 한 환경에 있으면 `--run`으로 한 번에 실행합니다.

```
bash out/training/datasets/r001/export/commands.sh
```

DX-COM 설정은 PPU(장치 후처리) 없이 만듭니다. 그러면 출력이 (1, 84, 8400) 하나가 되고, 파이의 DX-M1 검출기는 이 원시 출력을 CPU에서 풉니다. 지금 파이가 쓰는 DEEPX 공식 모델은 PPU 판이므로 출력 형식이 달라집니다. 새 dxnn이 파이에서 바르게 풀리는지는 확인 필요입니다. Ultralytics의 `format="deepx"` 내보내기는 Ultralytics 8.4.19의 exporter에 없어 쓰지 않았습니다.

변환할 때마다 서버의 FP32 결과와 파이 NPU의 INT8 결과를 같은 시험 세트에서 비교해 하락폭을 기록합니다. 파이 예측을 프레임 이름의 txt로 남기는 도구는 아직 없습니다. Hailo 변환에서 양자화 오류가 나면 Hailo 포럼의 출력층 범위 지정 사례를 먼저 봅니다(07 문서 [H4]).

파이에서는 새 회차 모델을 섀도 실행으로 기존 모델과 함께 돌려 차이를 본 뒤 기본값을 바꿉니다. 회차를 고르는 설정 이름은 파이 쪽 문서(docs/drive_recording.md)를 따릅니다.

## 10. 되돌리기

파일 이름에 회차가 붙어 덮어쓰지 않으므로, 파이에서 문제가 생기면 설정의 회차만 이전 값으로 돌립니다. 이전 회차의 hef, dxnn은 지우지 않고 둡니다.

학습 쪽에서 나쁜 회차가 나오면 다음 회차 설정에 `prev_round`를 직접 적어 그 회차를 건너뜁니다. 예를 들어 r003이 나빴다면 r004 설정에 `prev_round: r002`를 적습니다. 이전 회차 데이터 섞기는 r003 데이터도 그대로 가져가므로, r003 라벨이 원인이면 `out/training/datasets/r003/own_train.txt`의 이름을 바꿔 섞기에서 뺍니다.

고정 시험 세트는 어떤 경우에도 고치거나 지우지 않습니다. test_set.csv의 프레임 목록이 lock.json과 다르면 모든 단계가 멈춥니다. 시험 세트 라벨을 고칠 때는 CVAT에서 고친 뒤 `cvat_io import --test-set`으로 다시 들여오고, 바뀐 라벨로 모든 회차를 다시 평가합니다.

## 11. 확인 필요 목록

1. 파이 녹화 파일 이름, 타임스탬프 csv 열, 트리거 json 키(`round.yaml`의 recording, triggers)
2. Grounding DINO와 SAM 2 함수 이름과 인자(training/autolabel.py의 OfficialApi 한곳에 모음)
3. CVAT 판과 마스크 RLE 형식(다각형만 쓰면 영향 없음)
4. hailomz compile 옵션과 출력 파일 이름, DFC와 파이 HailoRT 판 맞춤
5. dxcom 옵션과 설정 json 키, PPU 없는 dxnn의 파이 동작
6. mmsegmentation 학습 설정과 세그 모델 변환
7. 흐림, 중복 문턱값과 batch 크기를 실제 데이터와 GPU로 조정
