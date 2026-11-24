# 학습 파이프라인 준비

이 문서는 차량 장착 전에 학습 서버에서 돌릴 데이터 준비와 학습 파이프라인(training/)을 만들고 맥에서 시험한 기록입니다. 작업일은 2026-11-25이고, 장비는 맥북(/usr/bin/python3 3.9.6, numpy 2.0.2, OpenCV 4.13.0)입니다. GPU가 필요한 단계(Grounding DINO, SAM 2, YOLOv8s 학습, mmsegmentation, Hailo DFC, DX-COM)는 실행하지 않았고, 합성 영상과 가짜 라벨러, 가짜 학습기, 가짜 예측기로 흐름만 확인했습니다. 파이와 NPU는 쓰지 않았습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| 맥 시험 전체 | 251 통과, 4 건너뜀 | /usr/bin/python3 -m pytest -q, 같은 시기 다른 작업의 시험 포함 |
| training 시험 | 37 통과 | 합성 영상, 가짜 모델 |
| 가짜 회차 명령줄 실행 | 1부, 2부 모두 종료 코드 0 | run_round --fake, 합성 주행 1개 |
| 파이 녹화 형식 읽기 | 세그먼트, 트리거 클립 모두 읽음 | realityview/mjpeg.py 의 MjpegWriter 로 만든 파일 |
| 큰 라이브러리 지연 import | torch, ultralytics 등 안 올라옴 | 모든 training 모듈 import 후 sys.modules 검사 |
| 파이 시험 | 돌리지 않음 | 작업 지시상 생략 |

로그는 out/measure/13_pytest_mac.log, 13_fake_round.log, 13_pi_format_check.log에 있습니다.

## 1. 만든 것

새 코드는 모두 training/ 아래에 있습니다. 사용법은 docs/training_pipeline.md에 정리했습니다.

| 파일 | 하는 일 |
|---|---|
| common.py | 설정 기본값, 폴더 배치, 프레임 이름 규칙, YOLO txt, 박스 짝짓기 |
| configs/round.yaml | 회차 설정 예시 |
| select_frames.py | 간격 뽑기, 흐림과 중복 제외, 장면별 고르게, 트리거 우선, 시험 세트 잠금 |
| autolabel.py | 라벨러 인터페이스, Grounding DINO + SAM 2, 영상 추적, 의심 프레임 |
| cvat_io.py | 검수 목록, CVAT 내보내기와 들여오기, 최종 라벨 합치기 |
| label_qa.py | 사람 라벨과 자동 라벨 비교(박스 IoU, 놓친 비율, 잘못 잡은 비율) |
| train_det.py | 데이터 섞기, 시험 세트 누설 검사, Ultralytics 학습 |
| train_seg.py | BiSeNetV2 도로 세그 데이터 준비, mmsegmentation 설정 |
| evaluate.py | 고정 시험 세트 mAP50, mAP50-95, 클래스별 재현율, 하락 경고, md 표 |
| export_npu.py | 보정 이미지, DX-COM json, ONNX, hailomz, dxcom, rsync 명령(기본 출력만) |
| run_round.py | 한 회차 실행(검수 앞뒤 두 번에 나눔) |
| fakes.py, tests/ | 가짜 모델, 합성 영상, 시험 |
| requirements.txt | 서버 주 환경 판 고정, 별도 환경 안내 |

## 2. 가짜 회차 결과

합성 주행 1개(세그먼트 3개 x 카메라 2대, 10fps, 4초씩, 낮 2개와 밤 1개, 흐린 프레임과 정차 구간 포함)와 트리거 클립 1개, COCO 흉내 12장으로 r001을 명령줄에서 끝까지 돌렸습니다. 수치는 흐름 확인용이며 실제 데이터 성능과 관계가 없습니다.

| 단계 | 결과 | 비고 |
|---|---|---|
| 선별 | 본 프레임 52장, 흐림 6, 중복 19, 시험 세트 제외 6, 선택 20 | 트리거 3장, 낮 18, 밤 2 |
| 시험 세트 | 6장, 세그먼트 1개 통째 | 처음 실행 때 잠금 |
| 자동 라벨 | 20장, 박스 40개, 의심 20장 | 비교 라벨러가 사람을 일부러 놓침 |
| 라벨 품질 | 박스 IoU 1.000, 도로 IoU 0.988 | 자동 라벨을 그대로 승인한 검수 |
| 데이터 섞기 | 이번 회차 18, 검증 2, COCO 9 | coco_ratio 0.5 |
| 평가 | r000, r001 모두 mAP50 1.000, 경고 없음 | 가짜 예측기 |
| 변환 | 보정 8장, 명령 5줄 | 실행하지 않음, bash -n 통과 |

도로 IoU가 1이 아닌 이유는 마스크를 CVAT 다각형으로 바꿀 때 꼭짓점을 2화소 오차로 줄이기 때문입니다. 처음에는 바깥 윤곽만 내보내 도로 위 차 자리가 도로로 메워졌고(IoU 0.925), 그 자리를 not_road 다각형으로 따로 내보내게 고쳤습니다.

시험으로 확인한 규칙은 다음과 같습니다. 시험 세트 세그먼트는 학습, 검증, 보정 이미지 어디에도 들어가지 않고, 목록을 손으로 고치면 모든 단계가 멈춥니다. 둘째 회차는 직전 회차 best.pt에서 시작하고 이전 회차 프레임을 prev_ratio만큼 섞으며, 이미 쓴 프레임은 다시 고르지 않습니다. 사람을 못 잡는 모델을 넣으면 예전 장면 mAP50 하락 경고가 나옵니다. 학습 목록의 모든 이미지는 Ultralytics 규칙(images를 labels로 바꾼 자리)에 라벨이 있습니다.

## 3. 파이 녹화 형식 맞춤

같은 시기에 다른 작업이 파이 쪽 녹화 도구를 만들어, 그 형식에 맞춰 읽기를 고쳤습니다. 세그먼트는 `out/drives/<시작 시각>/seg_0000/<카메라>.mjpeg`와 csv(frame, index, sensor_ns, wall_time, offset, size, gap)이고, 트리거 클립은 `out/triggers/<실행>/<클립>/trigger.json`의 events 목록에 종류와 점수가 있습니다. 파이 코드의 MjpegWriter로 15fps 30프레임을 써서 읽혀 보니, 1초 간격으로 0번과 15번 프레임을 골랐고 시각은 0.0초, 1.0초였습니다. sensor_ns가 부팅 45초 뒤 값(4.5e10)인 경우와 큰 값(1.7e18)인 경우 모두 같았습니다. 처음에는 값 크기(1e12 초과)로 나노초를 판단해서 부팅 뒤 약 16분 안에 찍은 녹화의 시각이 10억 배로 읽히는 결함이 있었고, 열 이름(ns로 끝남)으로 단위를 정하게 고쳤습니다. 합성 영상의 sensor_ns도 부팅 직후 값으로 바꿔 시험이 이 경우를 잡게 했습니다. 트리거 클립은 종류 ambiguous, 점수 0.37로 읽혔습니다. 파이 코드는 고치지 않았고 이 확인에서 읽기만 했습니다.

## 4. 판단한 선택

1. 재학습 모델도 COCO 80 클래스 머리를 유지하고 라벨에 COCO 번호를 씁니다. 파이 후처리(COCO_KEEP)와 zoo의 yolov8s HEF 설정을 바꾸지 않고 파일만 바꿔 끼우기 위해서입니다. 우리 라벨에 없는 74 클래스는 우리 영상에서 배경으로 배우게 되는 부작용이 있습니다.
2. 시험 세트는 프레임이 아니라 세그먼트 단위로 떼어 둡니다(기본 15%). 1초 옆 프레임이 학습에 들어가면 시험 점수가 부풀기 때문입니다. 시험 세트는 사람 라벨이 붙은 프레임만 평가에 씁니다. 앞뒤 몇 초만 빼는 `exclude: window`도 넣었는데, 처음에는 학습 목록 검사에서 시각을 몰라 시험 프레임이 있는 세그먼트를 통째로 빼는 바람에 학습 목록이 비었습니다. 모든 회차 frames.csv에서 시각을 찾게 고치고 끝에서 끝 시험을 더했습니다.
3. DX-COM 설정에는 PPU를 넣지 않았습니다. 형식을 확인하지 못했기 때문이고, 출력이 (1, 84, 8400) 하나가 되어 파이의 원시 출력 해석 경로를 타게 됩니다. 이 경로를 재학습 모델로 파이에서 돌려 본 적은 없습니다(확인 필요).
4. 세그 라벨은 0 도로, 1 그 밖으로 두었습니다. 파이의 road_classes=(0,)을 그대로 쓸 수 있습니다.
5. 변환은 기본으로 명령만 출력하고, rsync 줄은 주소 자리표시자가 남아 있으면 commands.sh에서 주석으로 둡니다.
6. Ultralytics의 `format="deepx"`는 맥에 설치된 8.4.19의 exporter에 없어 쓰지 않았습니다.
7. 밤 영상은 라플라시안 분산이 작아(합성 영상에서 낮 약 900, 밤 약 20) 밤 흐림 문턱(blur_min_night 15)을 따로 두었습니다. 실제 밤 영상으로 조정해야 합니다.

## 5. 사람이 할 일

학습 서버(WSL2 Ubuntu 22.04, CUDA 잡힌 상태)에서 아래를 차례로 합니다. 자세한 설명은 docs/training_pipeline.md 3~9장에 있습니다.

```
sudo apt install -y python3.10-venv git rsync
python3 -m venv ~/venv-train && . ~/venv-train/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r training/requirements.txt
python3 -m pytest -q training
```

그다음 Grounding DINO와 SAM 2를 공식 저장소에서 설치하고 가중치를 models/gdino/, models/sam2/에 둡니다. 녹화를 out/drives/로 옮긴 뒤 아래로 1부를 돌리고 CVAT 검수 후 2부를 돌립니다.

```
cp training/configs/round.yaml training/configs/r001.yaml
python3 -m training.run_round --config training/configs/r001.yaml
python3 -m training.run_round --config training/configs/r001.yaml --from review_import --until export
```

결과는 out/training/datasets/r001/와 models/r001/에 생깁니다. 첫 회차 전에는 지금 쓰는 yolov8s를 같은 export 명령으로 HEF와 dxnn으로 다시 만들어 파이 결과와 같은지 보는 것이 좋습니다(07 문서 11장 2번).

## 6. 확인 필요

1. Grounding DINO, SAM 2 함수 이름과 인자. training/autolabel.py의 OfficialApi 한곳에 모았습니다.
2. hailomz compile 옵션과 출력 파일 이름, DFC 판과 파이 HailoRT 판 맞춤.
3. dxcom 옵션, 설정 json 키, PPU 없는 dxnn의 파이 동작.
4. mmsegmentation 1.2.2 설정 키와 mmcv 판, 세그 모델 변환 명령(아직 없음).
5. requirements.txt에서 "확인 필요"로 적은 판(onnx, supervision, transformers 등).
6. RTX 4070에서 YOLOv8s 640 배치 16의 GPU 메모리(4~6GB로 예상, 실측 전).
7. 흐림, 중복, 밤 밝기 문턱값을 실제 주행 녹화로 조정.
8. CVAT 마스크 RLE 형식(다각형만 쓰면 영향 없음).
9. 파이 NPU 예측을 프레임 이름의 txt로 남기는 도구(INT8 하락폭 측정용)는 아직 없습니다.
