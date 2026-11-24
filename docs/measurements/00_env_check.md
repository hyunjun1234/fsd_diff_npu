# 0단계 장비 상태 점검 기록

이 문서는 라즈베리파이 5(접속 이름 pi5)와 Hailo-8, DX-M1, 카메라의 상태를 2026-11-24에 점검한 기록입니다. 코드는 고치지 않았고, 모든 파이 명령은 맥에서 `ssh pi5`로 실행했습니다.

원본 출력은 같은 폴더의 로그 파일에 있습니다.

| 파일 | 내용 |
|---|---|
| 00_env_raw_dx_cam.log | DEEPX 런타임, 카메라 목록 |
| 00_env_raw_hailo_run.log | Hailo FPS 측정, HailoBackend 시험 |
| 00_env_raw_extra.log | run_demo 도움말, 홈 폴더 HEF 파싱, 측정 후 온도 |
| 00_env_pytest_pi.log | 파이 pytest 결과 |
| 00_env_pytest_mac.log | 맥 pytest 결과(맥에만 있음) |

## 요약

| 항목 | 판정 | 핵심 |
|---|---|---|
| 1. OS | 통과 | 64비트, Gen3 설정 있음 |
| 2. PCIe | 통과 | 두 NPU 모두 8.0 GT/s x1 |
| 3. Hailo 런타임 | 통과 | 4.23.0, yolov8s_h8.hef 있음 |
| 4. DEEPX 런타임 | 확인 필요 | 동작함, 유휴 온도 72도 |
| 5. 카메라 | 확인 필요 | 계획과 다른 센서 |
| 6. 상태 | 통과 | throttled 0x0 |
| 7. 저장소 시험 | 통과 | 양쪽 24개 통과 |
| 8. 데모 준비도 | 확인 필요 | HEF 경로, 카메라 설정 |

## 1. OS

실행한 명령: `uname -a`, `cat /etc/os-release`, `getconf LONG_BIT`, `uname -r`, `python3 --version`, `grep -n -i pci /boot/firmware/config.txt`

| 항목 | 값 |
|---|---|
| OS | Debian GNU/Linux 13 (trixie), 13.7 |
| 커널 | 6.18.60+rpt-rpi-2712, aarch64 |
| LONG_BIT | 64 |
| 파이썬 | 3.13.5 (시스템, .venv 같음) |
| config.txt | 61행 dtparam=pciex1, 62행 dtparam=pciex1_gen=3 |

판정: 통과. .venv는 `--system-site-packages`로 만들어져 시스템 패키지(hailo_platform, picamera2)를 그대로 봅니다.

## 2. PCIe

실행한 명령: `lspci -nn`, `lspci -t`, `lspci -vv -s <장치>`, `cat /sys/bus/pci/devices/<장치>/current_link_speed`, `dmesg | grep -i -E "hailo|dx|pcie|aer"`

한 부팅에서 ASM2806 스위치(1b21:2806), Hailo-8(1e60:2864, 0001:03:00.0), DEEPX DX_M1(1ff4:0000, 0001:06:00.0)이 모두 보였습니다. 커널 드라이버는 Hailo가 hailo(hailo_pci), DX-M1이 dx_dma_pcie(dx_dma)입니다.

`lspci -vv`는 sudo 없이 LnkSta를 보여 주지 않아서, 링크 속도는 sysfs에서 읽었습니다.

| 장치 | 현재 링크 | 최대 링크 |
|---|---|---|
| 스위치 위쪽(01:00.0) | 8.0 GT/s x1 | 8.0 GT/s x8 |
| Hailo-8(03:00.0) | 8.0 GT/s x1 | 8.0 GT/s x4 |
| DX-M1(06:00.0) | 8.0 GT/s x1 | 8.0 GT/s x4 |

스위치의 02:02.0, 02:06.0 포트는 2.5 GT/s로 보이지만 아래에 장치가 없는 빈 포트입니다. dmesg는 sudo 없이 읽혔고, `link up, 8.0 GT/s PCIe x1`, Hailo 펌웨어 로드 성공, DX-M1 Probe Done이 있었습니다. AER 오류 줄은 없었고 sysfs의 aer_dev_correctable, fatal, nonfatal 값도 모두 0이었습니다(측정 시점 기준, 30분 동시 부하 시험은 하지 않음).

판정: 통과.

## 3. Hailo 런타임

실행한 명령: `hailortcli --version`, `timeout 20 hailortcli fw-control identify`, `python3 -c "import hailo_platform"`(시스템, .venv), `ls /usr/share/hailo-models`, `find ~ -maxdepth 7 -name "*.hef"`, `hailortcli parse-hef <파일>`

- HailoRT CLI 4.23.0, 펌웨어 4.23.0, Board Name Hailo-8, 아키텍처 HAILO8입니다.
- `import hailo_platform`은 시스템 파이썬과 .venv 모두 4.23.0으로 됩니다(경로 /usr/lib/python3/dist-packages).
- 설치 패키지: hailo-all 5.1.1, hailort 4.23.0, hailort-pcie-driver 4.23.0, hailo-models 1.0.0-2, hailo-tappas-core 5.1.0.

/usr/share/hailo-models 안의 HEF 중 Hailo-8에서 쓸 수 있는 것은 아래와 같습니다. 이름 끝이 h8l인 파일은 Hailo-8L용, h10인 파일은 Hailo-15H용입니다.

| HEF | 입력 | 출력 |
|---|---|---|
| yolov8s_h8.hef | UINT8 NHWC 640x640x3 | FLOAT32 NMS BY CLASS, 80클래스 |
| yolov6n_h8.hef | UINT8 NHWC 640x640x3 | FLOAT32 NMS BY CLASS, 80클래스 |
| yolov5n_seg_h8.hef | UINT8 NHWC 640x640x3 | 원시 출력 4개(UINT8, UINT16) |
| yolov8s_pose_h8.hef | UINT8 NHWC 640x640x3 | 원시 출력 9개(UINT8) |

yolov8s_h8.hef의 출력 이름은 `yolov8s/yolov8_nms_postprocess`이고 단일 컨텍스트입니다. h8l용 yolov8s_h8l.hef도 있지만 3개 컨텍스트라 Hailo-8에서는 yolov8s_h8.hef를 씁니다.

홈 폴더는 깊이 4에서 HEF가 0개였고, 깊이 5에서 15개, 깊이 6에서 27개가 나왔습니다. 모두 ~/DaeDron, ~/DaeDron_gpu 아래의 드론 검출 모델(HAILO8용, NMS 없는 원시 출력 conv5x)이라 이 프로젝트의 차량 검출에는 쓰지 않습니다.

추가 측정(2026-11-24, 실제 영상 없음):

- `timeout 40 hailortcli run /usr/share/hailo-models/yolov8s_h8.hef --time-to-run 10` 결과 2842프레임, 284.03 FPS입니다. 0단계 완료 기준 중 Hailo-8 YOLOv8s 단독 FPS 기록에 해당합니다.
- 저장소의 HailoBackend에 yolov8s_h8.hef를 넣고 무작위 768x432 영상 2장을 5회 넣었을 때 한 번에 17.7ms(전처리 포함, 첫 회 포함)였고 오류가 없었습니다.

판정: 통과.

## 4. DEEPX 런타임

실행한 명령: `timeout 20 dxcli -s`, `dpkg -l | grep -i dx`, `lsmod | grep -i dx`, `ls -l /dev/dxrt*`, `python3 -c "import dx_engine"`(시스템, .venv), `find ~ /usr/share /opt -maxdepth 5 -name "*.dxnn"`

- DXRT v3.4.2, RT 드라이버 v2.6.0, PCIe 드라이버 v2.5.0, 펌웨어 v2.7.4, 메모리 LPDDR5 3.92GiB, PCIe Gen3 X1입니다.
- 패키지: dxrt-driver-dkms 2.6.0-2, libdxrt-bin 3.4.2. 라이브러리는 /usr/local/lib/libdxrt.so.3.4.2입니다.
- 커널 모듈: dxrt_driver, dx_dma가 올라와 있고 /dev/dxrt0이 있습니다.
- `import dx_engine`은 .venv에서만 됩니다(dx-engine 3.4.2, InferenceEngine 불러오기 성공). 시스템 파이썬에서는 ModuleNotFoundError입니다.
- .dxnn 파일은 ~/DaeDron/runs/npu/deepx, ~/DaeDron_gpu/runs/npu/deepx, ~/DaeDron/runs/npu_v100/deepx에만 있고 모두 드론 검출 모델입니다. YOLOv8s .dxnn은 없습니다.

`dxcli -s`가 보고한 NPU 온도가 유휴 상태에서 72도, Hailo 측정 뒤 73도였습니다. 같은 때 파이 SoC는 56도였습니다. 방열판 장착 여부와 센서 기준은 미확인입니다.

판정: 확인 필요. 런타임은 동작하지만 유휴 온도가 높고, 데모는 .venv에서 실행해야 합니다.

## 5. 카메라

실행한 명령: `timeout 20 rpicam-hello --list-cameras`, `python3 -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"`. 촬영은 하지 않았습니다.

| 번호 | 센서 | 모드 |
|---|---|---|
| 0 | ov5647 (CSI, i2c@88000) | 640x480, 1296x972, 1920x1080, 2592x1944 |
| 1 | imx708 (CSI, i2c@80000) | 1536x864, 2304x1296, 4608x2592 |

카메라 2대가 인식되었습니다. Picamera2는 1번 imx708의 Rotation을 180으로 보고합니다.

계획(2.3절)과 configs/two_cam.yaml은 Camera Module 3 Wide 2대를 가정합니다. 설정의 fx 311, 폭 768은 수평 화각 약 102도에 해당합니다. 실제 0번은 ov5647(Camera Module 1 계열)이고, 1번은 센서 이름이 imx708_wide가 아니라 imx708로 나와 일반 Camera Module 3일 가능성이 있습니다(미확인).

판정: 확인 필요. 1단계 거리 정확도 기준을 맞추려면 카메라 구성을 계획과 맞추거나 보정 값을 실제 렌즈로 다시 구해야 합니다.

## 6. 상태

실행한 명령: `vcgencmd get_throttled`, `vcgencmd measure_temp`, `free -h`, `df -h /`, `nproc`

| 항목 | 값 |
|---|---|
| throttled | 0x0 (점검 시작, Hailo 측정 후 모두) |
| SoC 온도 | 56.0도, Hailo 측정 후 56.5도 |
| 메모리 | 4.0Gi 중 가용 2.5Gi, 스왑 7.0Gi |
| 루트 디스크 | 57G 중 28G 남음(50% 사용) |
| CPU 코어 | 4 |

판정: 통과. 메모리가 4GB 모델이므로 큰 모델을 CPU에서 돌리는 일은 피합니다.

## 7. 저장소 시험

실행한 명령:

```bash
rsync -av -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' --exclude .git --exclude models --exclude out --exclude '.venv*' ./ pi5:~/work/fsd_diff_npu/
ssh -o BatchMode=yes -o ConnectTimeout=10 pi5 'cd ~/work/fsd_diff_npu && . ~/work/fsd_diff_npu/.venv/bin/activate && timeout 300 python3 -m pytest -q'
/usr/bin/python3 -m pytest -q
```

| 장비 | 파이썬 | 결과 |
|---|---|---|
| 파이 .venv | 3.13.5 | 24 passed (6.13s) |
| 맥 | 3.9.6 (/usr/bin/python3) | 24 passed, 경고 9개 (1.71s) |

맥의 기본 `python3`(3.12.16 에이전트 가상환경)에는 pytest가 없어 `python3 -m pytest -q`가 바로 실패했고, pytest가 있는 /usr/bin/python3로 돌렸습니다. 맥 경고 9개는 camera_model.py 85행, 103행의 matmul RuntimeWarning(0으로 나누기, overflow)이고 test_ipm_coverage_and_owner, test_sim_pipeline_end_to_end_positions에서 나왔습니다. 파이에서는 같은 경고가 출력되지 않았습니다.

파이 .venv에는 requirements.txt의 패키지(numpy 2.2.4, opencv 4.10.0, PyYAML 6.0.2, pytest 9.2.0)가 이미 있어 `pip install`은 하지 않았습니다. 설치하면 opencv-python-headless가 시스템 opencv를 가릴 수 있습니다. `pip check`는 hailort 4.23.0이 numpy<2, contextlib2, future를 요구한다고 알리지만, 위 3장의 추론 시험은 numpy 2.2.4로 오류 없이 끝났습니다.

판정: 통과.

## 8. 데모 준비도

실행한 명령: `timeout 30 python3 scripts/run_demo.py --help`(파이 .venv, 정상 출력), 코드 읽기(realityview/detectors.py, realityview/sources.py, scripts/run_demo.py, configs/two_cam.yaml)

맞는 것:

- HailoBackend가 쓰는 hailo_platform API(VDevice, HailoSchedulingAlgorithm, ConfigureParams, InferVStreams)는 HailoRT 4.23.0에서 실제로 동작했습니다.
- yolov8s_h8.hef 출력은 프레임마다 80개 클래스 목록이고 각 항목이 (n, 5) 배열이라, parse_hailo_nms가 가정한 구조와 같습니다.
- PiCameraSource가 쓰는 Picamera2 API(create_video_configuration, RGB888, FrameRate, capture_array)는 picamera2 0.3.39에 있는 API입니다.

맞지 않거나 확인이 필요한 곳:

1. configs/two_cam.yaml의 det_hef가 `models/yolov8s_h8.hef`인데 파이 저장소에 models/ 폴더가 없습니다. 이대로 `--backend hailo`를 실행하면 HEF 열기에서 실패합니다.
2. 카메라 설정이 실제 장비와 다릅니다. index 0(front)은 ov5647, index 1(rear)은 imx708이고, 내부 파라미터 fx 311은 102도 광각 기준입니다. 보정 전에는 거리 값을 믿을 수 없습니다.
3. imx708의 Rotation 180을 PiCameraSource가 처리하지 않습니다. 영상이 뒤집혀 나오는지는 촬영해 봐야 압니다(미확인).
4. parse_hailo_nms의 좌표 순서(ymin, xmin, ymax, xmax)와 0~1 정규화는 무작위 입력이라 검출이 0개여서 확인하지 못했습니다. 실제 영상으로 박스 위치를 봐야 합니다.
5. PiCameraSource.close()는 stop()만 부르고 close()를 부르지 않아 카메라가 프로세스 종료 때까지 잡혀 있을 수 있습니다.
6. run_demo.py는 `--frames` 기본값이 0(무한)이고 `--headless`가 없으면 창을 띄웁니다. SSH에서 돌릴 때는 `--headless --frames N`과 `timeout`을 붙여야 합니다.
7. 시스템 파이썬에는 dx_engine이 없으므로 2단계 DX-M1 백엔드는 .venv에서 실행해야 합니다.

판정: 확인 필요. 1번은 파일 복사로 바로 풀리고, 2번과 3번은 1단계 보정 작업에서 다룹니다.

## 0단계 완료 기준 대비

| 기준 | 상태 |
|---|---|
| 두 NPU가 한 부팅에서 lspci에 보임 | 충족 |
| 각자 확인 명령에 응답 | 충족 |
| Hailo-8 YOLOv8s 단독 FPS | 284.03 FPS (hailortcli 10초) |
| DX-M1 YOLOv8s 단독 FPS | 미측정, .dxnn 없음 |
| 두 NPU 동시 30분, throttled 0x0, PCIe 오류 없음 | 미시험 |
