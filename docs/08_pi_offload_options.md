# 파이 한 대가 모자랄 때 처리를 나누는 방법

이 문서는 카메라 5대와 v15풍 3D 화면을 라즈베리파이 5 한 대로 처리하기 어려울 때, 지연을 거의 늘리지 않고 일을 나눠 맡길 방법을 비교하고 하나를 추천하는 자료입니다.

수치 뒤의 `[번호]`는 맨 아래 출처이고, 저장소 안의 측정은 문서 링크로 적었습니다. "추정"은 이 문서에서 계산한 값이라 실측이 필요합니다.

## 1. 결론

두 번째 라즈베리파이 5를 렌더링 전용으로 두고, 두 파이를 랜선 한 가닥으로 바로 잇는 방법을 추천합니다. 첫 번째 파이(이하 파이 A)는 지금처럼 카메라, NPU 검출, 추적, 탑뷰 바닥 합성까지 하고, 그 결과(물체 목록, 차선, 신호등, 바닥 영상)만 두 번째 파이(이하 파이 B)에 보냅니다. 파이 B는 v15풍 화면을 그려 차 안 모니터에 띄웁니다.

이렇게 하면 파이 A의 루프에서 화면 그리기(v15 chase 기준 약 42ms, 추정)가 빠져 5대 구성의 초당 처리 장수가 6~10장에서 약 13장으로 오르고(추정), 랜선 전송으로 더해지는 지연은 1~2ms 안쪽으로 봅니다(추정, 4장). 비용은 파이 5 한 대와 전원, 케이블 정도이고 코드는 지금의 렌더링 스레드 경계를 네트워크로 옮기는 정도라 다시 짤 부분이 적습니다.

NPU를 더 다는 것은 지금 상황에 도움이 되지 않습니다. 병목이 NPU가 아니라 CPU이기 때문입니다(2장). 다른 보드로 바꾸는 것은 파이 2대 구성으로도 부족할 때의 다음 단계로 둡니다.

## 2. 지금 어디서 시간이 드는지

1단계 측정에서 NPU 추론은 약 18ms였고, 시간을 더 쓴 쪽은 CPU의 BEV 합성(23ms)과 렌더링(28ms)이었습니다([진행 상황](progress.md) 측정 요약). Hailo-8은 YOLOv8s를 단독으로 초당 284장 처리합니다([00번 측정](measurements/00_env_check.md)). 카메라 5대를 초당 10장씩 돌려도 초당 50장이라 Hailo-8 한 개의 약 18%만 씁니다(추정).

반면 CPU 쪽은 카메라 한 대마다 루프가 약 12ms 늘고(검출만 켠 탑뷰 화면, 2대와 4대 실측을 이은 직선), USB 카메라 3대의 MJPEG 해제도 CPU가 맡습니다([19번 측정](measurements/19_camera_count_sim.md) 7장). 여기에 v15풍 화면은 파이에서 물체 20개 기준 chase 약 42ms, orbit 약 69ms로 어림됩니다([v15풍 화면 문서](v15_render.md), 클라우드 측정값에 파이 배율 1.9를 곱한 값이라 확인 필요). 5대 구성에서 루프 약 75ms에 화면 42ms가 더해지면 초당 약 8장이 되고, 이것이 지금 예상하는 6~10장의 근거입니다.

따라서 덜어 내야 할 것은 CPU 시간이고, 그중 가장 크고 떼어 내기 쉬운 덩어리가 화면 그리기입니다.

## 3. 후보 비교

| 방법 | 추가 비용 | 효과 | 판단 |
|---|---|---|---|
| 같은 파이 GPU로 그리기 | 없음 | 확인 필요 | 장기 과제 |
| 파이 2대, 렌더링 분리 | 약 15만 원 | 큼 | 추천 |
| 파이 2대, 카메라 분할 | 파이와 NPU 1개 | 큼 | 복잡함 |
| NPU 추가 | 약 15~30만 원 | 거의 없음 | 제외 |
| Jetson Orin Nano Super | 약 249달러 | 큼 | 다음 단계 |
| RK3588 보드 | 90~129달러 | 중간 | 보류 |
| 맥북, 휴대폰 | 없음 | 큼 | 차량용 부적합 |

비용 중 원화 값은 파이 5 8GB 본체, 방열 케이스, 12V-5V 변환기, 랜선을 합친 어림이고 판매처 가격은 확인 필요입니다. 각 방법의 근거는 아래와 같습니다.

### 3.1 같은 파이의 GPU로 그리기

파이 5의 GPU(VideoCore VII)는 OpenGL ES 3.1과 Vulkan 1.2 드라이버를 갖췄습니다 [1]. 지금 v15 렌더러는 CPU에서 C로 z-버퍼를 돌리므로, 삼각형 그리기와 바닥 텍스처를 GPU로 옮기면 하드웨어를 더하지 않고 CPU 시간을 줄일 수 있습니다. 다만 렌더러를 거의 새로 짜야 하고, 파이 GPU에서 이 화면이 몇 ms에 그려지는지는 확인 필요입니다. GPU가 렌더링을 맡아도 BEV 합성과 MJPEG 해제는 그대로 CPU에 남습니다. 파이 2대 구성에서도 파이 B의 렌더러를 GPU로 옮기면 그대로 쓸 수 있으므로, 순서상 2대 구성 뒤에 하는 것이 낫습니다.

### 3.2 파이 2대, 렌더링 분리 (추천)

파이 A의 렌더링 스레드는 이미 한 프레임분 입력을 `_RenderJob` 하나로 모아 넘기고, 밀린 작업은 버리고 최신 것만 그립니다(`realityview/pipeline.py`). 이 묶음을 그대로 직렬화해 UDP로 보내면 파이 B가 같은 `RendererV15.render()`를 부릅니다. 두 파이가 일을 이어 달리기처럼 나눠 하므로 초당 장수는 둘 중 느린 쪽(파이 A 약 75ms)에 맞춰지고, 지연은 한 대에서 차례로 하던 것과 거의 같습니다.

파이 A는 USB 포트 4개를 카메라 3대와 녹화 SSD가 이미 다 쓰고 있습니다([19번 측정](measurements/19_camera_count_sim.md) 7장). 이더넷 포트는 비어 있어 연결에 쓰기 좋습니다. 파이 B는 HDMI로 화면만 내보내면 되므로 NPU나 카메라가 필요 없습니다.

전력은 파이 5가 네 코어 모두 부하일 때 약 8.8W였다는 측정이 있어 [2], 파이 B를 더해도 차량 12V 쪽에서 10W 안팎이 늘어납니다(추정). 파이 B 전원은 파이 A와 같은 방식(12V-5V 5A 변환기, ACC 연동)을 하나 더 두면 됩니다([차량 전원과 자동 시작 문서](vehicle_power_and_autostart.md)).

### 3.3 파이 2대, 카메라 분할

카메라를 3대와 2대로 나눠 파이마다 NPU를 하나씩 붙이는 방법입니다. CPU가 두 배라 효과는 크지만, 두 파이의 시계를 맞추고(카메라 사이 시각 맞춤은 지금 25ms 허용으로 한 파이 안에서만 합니다) 카메라 간 물체 병합과 추적을 한쪽으로 모아야 합니다. 병합은 지금도 6대 이상에서 문제가 되는 부분이라([진행 상황](progress.md) 다음에 할 일 6번) 네트워크를 사이에 두면 더 어려워집니다. 파이 B에 NPU를 하나 더 사야 하는 점도 부담입니다. 렌더링 분리로도 모자랄 때 고려합니다.

### 3.4 NPU 추가

Hailo-8을 하나 더 달거나 Hailo-10H로 바꾸는 방법입니다. 2장에서 본 대로 지금 NPU는 여유가 크므로 FPS에 거의 도움이 되지 않습니다. 파이 5의 PCIe는 1레인이라 NPU를 여럿 달면 스위치(예: Seeed 듀얼 M.2 HAT의 ASM2806)를 거쳐 그 1레인을 나눠 씁니다 [3]. PCIe Gen2와 Gen3에서 Hailo-8 다중 스트림 처리량이 약 두 배 차이가 난 측정도 있어 [4], 대역이 이미 빠듯한 연결에 장치를 더 붙이는 셈입니다. 지금 HAT에는 Hailo-8과 DX-M1이 꽂혀 있어 빈 슬롯도 없습니다. 두 NPU 모두 방열 없이 과열된 기록이 있으니([진행 상황](progress.md) 사용자가 해야 할 일 1, 2번), NPU 쪽은 개수보다 방열이 먼저입니다.

### 3.5 Jetson Orin Nano Super로 바꾸기

개발자 키트가 249달러이고 INT8 67 TOPS(희소 기준, 밀집 33 TOPS), CUDA 코어 1,024개 GPU, 6코어 Cortex-A78AE, LPDDR5 8GB(102GB/s)를 갖췄습니다 [5][6]. 전력 모드는 7W, 15W, 25W입니다 [5]. GPU가 있어 렌더링과 BEV 합성을 모두 GPU로 옮길 수 있으므로 한 대로 5대 구성을 감당할 가능성이 가장 큽니다.

대신 지금 쌓은 Hailo HEF와 DEEPX .dxnn 모델을 TensorRT로 다시 변환해야 하고, CSI 커넥터는 2개라 [5] 나머지는 USB 카메라로 받아야 합니다. MJPEG 하드웨어 해제 지원은 출처에서 확인하지 못해 확인 필요입니다. 25W 모드는 벽 전원 기준 실제 소비가 더 크다는 지적이 있고 [6], 차량 전원과 방열을 다시 설계해야 합니다. 파이 2대 구성으로도 목표에 못 미칠 때 옮겨 갈 다음 단계로 둡니다.

### 3.6 RK3588 보드 (Orange Pi 5 Plus 등)

Orange Pi 5 Plus는 4GB 89.90달러, 8GB 109달러, 16GB 129달러이고, 6 TOPS NPU, Mali-G610 GPU(OpenGL ES 3.2, Vulkan 1.2), 2.5GbE 두 포트를 갖췄습니다 [7]. RK3588은 Cortex-A76 4개와 A55 4개로 파이 5보다 CPU 코어가 많아 [7] 파이 B 자리에 쓰면 렌더링 여유가 더 있습니다. 다만 NPU는 Hailo-8보다 작고 툴체인(RKNN)이 달라, 주 처리 보드로 바꾸면 모델 변환을 다시 해야 합니다. MJPEG 하드웨어 해제는 출처에서 확인하지 못해 확인 필요입니다. 파이 B 자리에 같은 파이 5를 두는 쪽이 소프트웨어 관리가 쉬워 보류합니다.

### 3.7 맥북이나 휴대폰

연산력은 충분하지만 차량용으로는 맞지 않습니다. 시동과 함께 켜지고 꺼지는 동작을 맞추기 어렵고, 여름 차 안 온도와 노트북 덮개를 닫은 상태의 방열이 문제이며, 휴대폰은 유선으로 이을 표준 경로가 없어 Wi-Fi에 기대야 하는데 무선은 지연이 고르지 않습니다(정량값 확인 필요). 개발 중 화면을 보는 용도로만 씁니다.

## 4. 두 파이를 잇는 방법과 지연

| 연결 | 쓸 수 있나 | 판단 |
|---|---|---|
| 기가비트 이더넷 직결 | 가능 | 추천 |
| USB-C 장치 모드 | 가능 | 전원 포트와 겹침 |
| PCIe | 일반 제품 없음 | 제외 |
| Wi-Fi | 가능 | 지연이 고르지 않음 |

이더넷 직결은 스위치 없이 두 파이를 랜선으로 바로 잇고 고정 IP를 줍니다. 1Gbps에서 1,500바이트 패킷 하나를 내보내는 데 12마이크로초가 걸리므로, 물체 20개와 차선, 신호등 정보(수 KB)는 전송 시간을 무시할 수 있습니다(추정). 가장 큰 짐은 바닥 영상입니다. 지금 5대 설정의 BEV는 40m x 24m를 5cm 칸으로 나눠 800x480이고(`configs/five_cam_carnival.yaml`), 원본 RGB면 약 1.15MB라 1Gbps에서 약 9ms가 걸립니다(추정). JPEG로 줄이면 전송은 1ms 안쪽이 되지만 파이 A의 압축 시간이 더해지고(확인 필요), 바닥 갱신을 몇 프레임마다로 줄이는 지금의 `--view3d-ground-every` 방식과 같이 쓰면 평균 부담이 더 줄어듭니다. 운영체제를 거치는 왕복 시간은 `ping`으로 실측해 이 문서에 적습니다(확인 필요).

USB-C 장치 모드는 라즈베리파이 OS가 파이 5의 보드 위 USB-C 포트를 USB 네트워크로 쓰는 기능을 공식 지원합니다 [8]. 그런데 이 포트는 파이 5의 전원 입력이기도 해서 [8] 차량에서 전원과 데이터를 함께 넣는 배선이 까다롭고, 속도도 출처에서 확인하지 못했습니다(확인 필요). 파이 5의 PCIe 단자는 NPU처럼 장치를 붙이는 용도라, 두 파이를 PCIe로 바로 잇는 일반 제품은 찾지 못했습니다.

## 5. 해 볼 순서

1. 파이 A에서 `--render-thread`를 켜고 5대 구성의 초당 장수와 렌더링 시간을 잽니다. 이 값이 목표(4대 기준 8 FPS, [구현 계획](03_pi5_hailo8_dxm1_implementation_plan.md) 4.4절)를 넘고 화면도 끊기지 않으면 두 번째 파이는 필요 없습니다.
2. 모자라면 파이 B를 들이고, `_RenderJob`을 보내는 송신부와 받는 수신부를 만들어 랜선으로 잇습니다. 바닥 영상은 원본과 JPEG 두 방식을 모두 재 봅니다.
3. 두 파이 사이 `ping` 왕복과 센서에서 화면까지 지연을 잽니다(지금 기준은 카메라 2대, Hailo 단독 평균 81ms, [진행 상황](progress.md) 측정 요약).
4. 그래도 모자라면 파이 B의 렌더러를 GPU(OpenGL ES)로 옮기거나, Jetson Orin Nano Super로 옮기는 것을 검토합니다.

## 출처

[1] Igalia, "Raspberry Pi 5 Announced", https://www.igalia.com/2023/09/28/Raspberry-Pi-5-Announced.html

[2] raspberry.tips, "Raspberry Pi power consumption, all models compared", https://raspberry.tips/en/faq/raspberry-pi-power-consumption-update-2026-all-models-compared

[3] CNX Software, "PCIe3.0 to Dual M.2 HAT+ for Raspberry Pi 5 features ASMedia ASM2806 PCIe 3.0 switch", https://www.cnx-software.com/2025/01/21/pcie3-0-to-dual-m-2-hat-for-raspberry-pi-5-features-asmedia-asm2806-pcie-3-0-switch/

[4] Seeed Studio Wiki, "Benchmark of multistream inference on Raspberry Pi 5 with Hailo-8", https://wiki.seeedstudio.com/benchmark_of_multistream_inference_on_raspberrypi5_with_hailo8/

[5] CNX Software, "$249 NVIDIA Jetson Orin Nano Super Developer Kit", https://www.cnx-software.com/2024/12/18/249-nvidia-jetson-orin-nano-super-developer-kit-targets-generative-ai-applications-at-the-edge/

[6] Hackster.io, "NVIDIA Unveils the Jetson Orin Nano Super", https://www.hackster.io/news/nvidia-unveils-the-jetson-orin-nano-super-delivers-a-1-7x-gen-ai-speed-boost-for-all-3602c2e98c40

[7] CNX Software, "Orange Pi 5 Plus SBC switches to Rockchip RK3588 SoC", https://www.cnx-software.com/2023/05/10/orange-pi-5-plus-sbc-switches-to-rockchip-rk3588-soc-brings-dual-hdmi-2-1-dual-2-5gbe-m-2-pcie-sockets/

[8] Raspberry Pi, "USB gadget mode in Raspberry Pi OS: SSH over USB", https://www.raspberrypi.com/news/usb-gadget-mode-in-raspberry-pi-os-ssh-over-usb/
