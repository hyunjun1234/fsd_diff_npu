# 06. 테슬라 카메라 자리별 라즈베리파이 5 카메라 선정

이 문서는 테슬라 모델 Y의 카메라 자리마다 화각과 사용 조건을 기준으로, 라즈베리파이 5에 붙일 수 있는 카메라 제품을 고른 자료입니다. 테슬라 쪽 화각과 위치는 [04 문서](04_tesla_camera_layout.md)의 값을 그대로 씁니다.

## 고르는 기준

NPU 입력이 640x640 이하로 줄어들기 때문에 화소 수는 1~2MP면 충분합니다. 대신 세 가지를 봤습니다. 첫째는 수평 화각이 테슬라의 같은 자리와 얼마나 가까운지, 둘째는 터널 출구나 해를 마주 보는 장면을 위한 HDR, 셋째는 야간 저조도 성능입니다. 여기에 라즈베리파이 5에서 별도 드라이버 없이 libcamera로 바로 잡히는지를 더했습니다.

테슬라 HW4 카메라 센서(소니 IMX963으로 알려짐)는 일반 판매 모듈을 찾지 못했습니다. 차량용 HDR 센서 IMX490 모듈은 GMSL2 연결이라 파이 5에는 변환 보드가 필요해 이번 후보에서 뺐습니다.

## 후보 제품

| 제품 | 수평 화각 | 강점 |
|---|---|---|
| Camera Module 3 | 66도 | HDR, 자동초점 |
| Camera Module 3 Wide | 102도 | HDR, 자동초점 |
| Arducam IMX462 141도 | 141도 | 저조도 |
| Arducam IMX296 M12 | 45도 | 글로벌 셔터 |
| Arducam IMX708 쿼드 키트 | 100도 | CSI 하나에 4대 |

Camera Module 3 계열은 소니 IMX708 센서이고, HDR 모드에서 최대 3MP로 출력합니다 [C1]. 공식 페이지는 대각 화각만 적어 두었는데 일반형이 75도, 와이드가 120도입니다 [C1]. 수평 화각은 일반형 66도(디바이스마트 표기 [K3]), 와이드 102도(같은 센서와 렌즈의 Arducam 제품 표기 [C3])입니다. 적외선 차단 필터가 없는 NoIR 버전도 있어서 적외선 조명과 함께 쓰면 야간 촬영이 됩니다 [C1].

Arducam IMX462 141도 모델은 화각이 대각 161도, 수평 141.4도, 수직 81.7도이고 1920x1080입니다 [C2]. 소니 STARVIS 저조도 센서라 밤에 강합니다. 제조사는 파이 5를 지원하고 추가 드라이버가 필요 없다고 적었습니다 [C2]. 다만 같은 페이지에 적외선 차단 필터가 있다는 말과 근적외선을 감지한다는 말이 함께 있고, 최대 프레임레이트도 46fps와 30fps가 섞여 있어 실물 확인 필요입니다. HDR 표기는 없습니다.

Arducam IMX296 M12 모델은 1440x1080 글로벌 셔터 카메라이고 화각은 수평 45도입니다 [C4]. 빠르게 움직이는 장면에서 롤링 셔터 기울어짐이 없지만 HDR은 없습니다. 라즈베리파이 공식 글로벌 셔터 카메라도 같은 센서이며 렌즈를 따로 사는 C/CS 마운트 제품입니다.

## 저조도 성능 비교

| 제품 | 화소 크기 | 조리개 |
|---|---|---|
| Camera Module 3 | 1.4um | F1.8 |
| Camera Module 3 Wide | 1.4um | F2.2 |
| IMX708 쿼드 키트 | 1.4um | F2.2 |
| Arducam IMX462 141도 | 2.9um | 확인 필요 |

화소 크기와 조리개는 라즈베리파이 카메라 문서 [C8], Arducam 제품 페이지 [C2][C5]의 값입니다. IMX462는 화소 면적이 IMX708의 약 4.3배라 한 화소가 받는 빛이 훨씬 많고, 제조사는 0.001lux에서도 찍힌다고 적었습니다 [C2]. 렌즈 조리개는 표기가 없어 확인 필요입니다. 반대로 IMX708 계열은 저조도에 강한 센서가 아닙니다. 특히 Wide와 쿼드 키트는 F2.2라 일반형(F1.8)보다 빛을 약 1.5배 적게 받습니다.

IMX708의 2304x1296 모드는 화소 4개를 묶어 읽는 방식이라 묶은 화소 하나의 크기가 2.8um에 해당합니다. NPU 입력이 640픽셀 안팎이라 이 모드나 더 작은 모드를 쓰게 되므로 실제 차이는 표의 화소 크기 차이보다 작을 것으로 보지만, 실측은 확인 필요입니다. 대신 IMX462는 HDR 표기가 없어서 터널 출구나 마주 오는 전조등처럼 밝기 차이가 큰 장면에는 약합니다. 밤과 역광 가운데 무엇을 우선하느냐에 따라 자리마다 고른 제품이 달라진 이유가 이것입니다.

## 테슬라 자리별 추천

| 테슬라 자리 | 테슬라 화각 | 추천 제품 |
|---|---|---|
| 전방 광각 | 120도 | Camera Module 3 Wide |
| 전방 메인 | 50도 | Camera Module 3 |
| 전방 범퍼 | 140도 | Arducam IMX462 141도 |
| B필러 좌우 | 90도 | Camera Module 3 Wide |
| 펜더 좌우 | 80도 | Camera Module 3 Wide |
| 후방 | 130도 | Arducam IMX462 141도 |

전방 광각은 앞유리 안쪽이라 역광과 터널 출구를 자주 마주치므로 HDR이 되는 Camera Module 3 Wide가 맞습니다. 수평 102도로 테슬라의 120도보다 좁지만 가장 가깝고, 같은 화각대에서 HDR이 되는 제품은 이것뿐이었습니다. 지금 달린 ov5647 130도는 화각은 넓지만 HDR이 없고, 130도가 수평인지 대각인지도 확인 필요입니다.

전방 메인 50도에는 Camera Module 3 일반형(66도)을 권합니다. 화각이 조금 넓지만 HDR이 있습니다. 움직임 번짐이 문제가 되면 IMX296 M12(45도)로 바꿀 수 있는데, 이때는 HDR을 포기해야 합니다.

전방 범퍼와 후방은 바닥 가까이 넓게 봐야 하고 밤에 주차할 때 많이 쓰는 자리입니다. 수평 141도에 저조도 센서인 Arducam IMX462가 화각과 조건 모두 맞습니다. 지금 후방에 달린 Camera Module 3 일반형은 66도라 테슬라 후방 130도의 절반 정도만 봅니다. 후방을 바꾼다면 이 제품이 첫 후보입니다.

B필러와 펜더는 측면이라 해를 정면으로 받는 일이 적지만, 낮과 밤이 섞인 도로를 계속 봐야 하므로 HDR이 되는 Camera Module 3 Wide로 통일하는 편이 보정과 운영이 쉽습니다. 3단계 계획의 쿼드 키트도 같은 IMX708 광각(수평 100도)이라 이 자리와 맞습니다.

## 3대 이상 연결할 때

파이 5의 카메라 포트는 2개라 3대 이상은 다른 방법이 필요합니다.

Arducam IMX708 쿼드 키트는 Camarray HAT 하나로 4대를 CSI 포트 하나에 묶습니다 [C5]. 4대를 켜면 한 장에 2x2로 합쳐 나오며 카메라당 1152x648이 되고, 프레임레이트는 단독일 때의 절반입니다 [C5]. 0번과 1번, 2번과 3번이 각각 동기화되고 2번과 3번은 한 프레임 늦습니다 [C5]. 제조사 표의 파이 5 지원 칸은 비어 있어 공식 지원은 확인 필요입니다. Arducam 포럼에는 파이 5에서 `dtoverlay=imx708,cam0` 설정으로 이 키트가 잡혔다는 사용자 사례가 있고, Arducam 직원은 IMX708 키트에는 Arducam 전용 libcamera가 필요 없다고 답했습니다 [C6]. 4대 모두 같은 화각이라 전방 메인이나 범퍼처럼 화각이 다른 자리에는 쓸 수 없습니다.

Arducam 멀티 카메라 어댑터(V2.2)는 여러 대를 붙여도 한 번에 한 대씩 번갈아 켜는 방식이라 [C7] 동시에 360도를 봐야 하는 이 프로젝트에는 맞지 않습니다. 파이 5에는 별도 변환 케이블도 필요하고 Camera Module 3 호환도 확인되지 않았습니다 [C7].

나머지 방법은 CSI 2대에 USB 카메라를 더하는 구성입니다. USB 쪽은 하드웨어 동기화가 없으므로 타임스탬프로 프레임을 묶어야 합니다(03 문서 2.3절).

모형 시험에서는 CSI 0번에 쿼드 키트(좌우와 후방 등 같은 화각 자리), CSI 1번에 전방 카메라 한 대를 두는 구성이 가장 간단합니다. 실제 차량에서는 배선 길이 때문에 구성이 달라지며, 다음 장에 정리했습니다.

## 차량에 달 때의 배선

파이를 운전석과 조수석 사이 콘솔에 둔다고 하면, 04 문서의 좌표와 내장재 안쪽으로 돌아가는 길이를 더해 배선 길이는 앞유리 약 2m, B필러 약 2m, 뒷유리 약 3m, 앞펜더 약 3~4m로 봅니다. 모두 추정이며 실제 차에서 재야 합니다. 파이 5에 기본으로 쓰는 CSI 리본 케이블은 판매 제품이 300mm 수준이라 [K6] 이 거리를 그대로 감당하지 못합니다.

| 방식 | 최대 거리 | 파이 5 |
|---|---|---|
| CSI-HDMI 확장 | 10m | 지원 |
| GMSL2 확장 키트 | 15m | 지원 |
| USB 카메라 | 케이블 규격에 따름 | 지원 |

CSI-HDMI 확장 모듈(Arducam B0091)은 CSI 신호를 HDMI 케이블로 옮겨 10m까지 늘립니다 [C9][C10]. Camera Module 3에서 동작한다고 적혀 있고 [C10], 고품질 HDMI 케이블을 쓰라는 조건이 붙어 있습니다 [C9]. 파이 5에는 22핀 변환 케이블이 따로 필요합니다 [C10]. 값이 싸고 지금 쓰는 CSI 카메라를 그대로 쓸 수 있지만, HDMI 커넥터는 차량 진동에 맞게 만든 것이 아니라 커넥터 고정이 필요합니다. Arducam IMX462 141도 모델과의 조합은 확인 필요입니다.

GMSL2 확장 키트는 차량용 동축 케이블(FAKRA)로 최대 15m까지 보내며 파이 5 호환으로 표기돼 있습니다 [C11]. 차량용으로 만든 방식이라 가장 튼튼하지만, Arducam 키트에 들어가는 카메라는 IMX219와 IMX477뿐이라 HDR이나 저조도 센서를 고를 수 없습니다 [C11][C12]. Camera Module 3를 GMSL2로 잇는 다른 회사 제품(V-Link)도 있지만 크라우드펀딩 제품이고 별도 드라이버가 필요해 [C13] 지금은 후보에서 뺐습니다.

USB 카메라는 파이 5의 CSI 포트 2개를 넘는 3번째, 4번째 카메라에 씁니다. USB 2.0은 수동 케이블로 5m 안팎까지 쓸 수 있고 USB 3.0은 그보다 짧다고 알려져 있으나 확인 필요입니다. 저조도 후보로 Arducam IMX462 USB 3.0 모듈(수평 85도, F1.0, 1080p 60fps)이 있습니다 [C14]. 다만 USB 2.0 포트에 꽂으면 720p 10fps로 떨어집니다 [C14]. 차 밖에 달 경우에는 방수 하우징과 적외선 LED가 달린 Arducam STARVIS 2 USB 카메라(IP65, 수평 86도, USB 2.0)가 맞습니다 [C15]. USB 카메라는 하드웨어 동기화가 없어서 타임스탬프로 프레임을 묶어야 합니다.

쿼드 키트는 카메라 4대가 HAT 하나에 짧은 리본으로 연결되는 구조라, 차 앞뒤와 양옆에 카메라를 흩어 놓는 데는 맞지 않습니다. 카메라마다 CSI-HDMI 확장을 붙이는 방법은 확인 필요라서 쿼드 키트는 모형 시험용으로만 권합니다.

차 밖(펜더, 범퍼, 번호판 위)에 다는 카메라는 방수가 필요합니다. 처음 차량 시험에서는 앞유리, 뒷유리, 양쪽 옆유리 안쪽에 붙이면 방수 문제 없이 시작할 수 있고, 위치가 테슬라와 조금 달라지는 만큼 설정 파일의 카메라 좌표를 실제 값으로 바꾸면 됩니다.

정리하면 차량용 구성은 CSI 2대를 CSI-HDMI 확장으로 늘려 전방(Camera Module 3 Wide)과 후방(IMX462 141도)에 두고, 좌우는 USB 카메라 2대로 채우는 방식입니다. 좌우 카메라는 밤 주행이 중요하면 IMX462 USB 3.0 모듈, 차 밖에 달아야 하면 STARVIS 2 IP65 모델을 고릅니다.

## 국내 가격

| 제품 | 국내 가격 | 판매처 |
|---|---|---|
| Camera Module 3 | 43,780원 | 디바이스마트 [K3] |
| Camera Module 3 Wide | 56,100원 | 디바이스마트 [K1] |
| Camera Module 3 NoIR Wide | 56,540원 | 디바이스마트 [K2] |
| 공식 글로벌 셔터 카메라 | 148,500원 | 가치창조기술 [K4] |
| Arducam IMX462 92도 | 93,500원 | 가치창조기술 [K5] |
| Arducam IMX462 141도 | 확인 필요 | 해외 39.99달러 [C2] |
| Arducam IMX708 쿼드 키트 | 확인 필요 | 해외 199.99달러 [C5] |
| CSI-HDMI 확장 모듈 1세트 | 22,000원 | 디바이스마트 [K7] |
| 파이 5용 카메라 케이블 300mm | 2,970원 | 디바이스마트 [K6] |
| STARVIS 2 USB IP65 카메라 | 확인 필요 | 해외 59.99달러 [C15] |
| IMX462 USB 3.0 모듈 | 확인 필요 | 가격 표기 없음 [C14] |

국내 가격은 모두 부가세 포함입니다. 디바이스마트 상품 페이지에는 품절 문구가 함께 보여 재고는 주문 전에 확인 필요입니다. 가치창조기술의 IMX462는 화각 92도의 다른 모델(Pivariety 방식)이라 141도 모델과 같은 제품이 아닙니다 [K5]. 141도 모델과 쿼드 키트는 국내 판매처를 찾지 못해 Arducam 공식몰 가격만 적었고, 해외 배송비와 관세는 확인 필요입니다.

## 출처

- C1: [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/)
- C2: [Arducam 2MP IMX462 141°(H) Wide-Angle M12 Lens](https://www.arducam.com/product/2mp-imx462-color-ultra-low-light-starvis-camera-module-with-141h-wide-angle-m12-lens-for-raspberry-pi/)
- C3: [Arducam 12MP IMX708 102 Degree Wide-Angle HDR Camera Module](https://www.arducam.com/product/presalearducam-12mp-imx708-102-degree-wide-angle-fixed-focus-hdr-high-snr-camera-module-for-raspberry-pi/) (단종 제품, 화각 표기만 인용)
- C4: [Arducam 1.58MP IMX296 Color Global Shutter Camera Module with M12 Lens](https://www.arducam.com/product/1-58mp-imx296-color-global-shutter-camera-module-with-m12-lens-for-raspberry-pi/)
- C5: [Arducam 12MP IMX708 Quad-Camera Kit](https://www.arducam.com/arducam-12mp-imx708-quad-camera-kit-wide-angle-stereo-synchronized-camera-module-for-raspberry-pi.html)
- C6: [Camarray Hat wont connect to Raspberry Pi 5, Arducam 포럼](https://forum.arducam.com/t/camarray-hat-wont-connect-to-raspberry-pi-5/8281)
- C7: [Multi Camera Adapter Module V2.2, The Pi Hut](https://thepihut.com/products/multi-camera-adapter-module-v2-2-for-raspberry-pi)
- C8: [Raspberry Pi 카메라 문서](https://www.raspberrypi.com/documentation/accessories/camera.html)
- C9: [Arducam CSI-to-HDMI Adapter Set](https://www.arducam.com/product/camera-csi-to-hdmi-adapter-set-2/)
- C10: [Raspberry Pi Camera HDMI Cable Extension, The Pi Hut](https://thepihut.com/products/raspberry-pi-camera-hdmi-cable-extension)
- C11: [GMSL2 8MP Camera Extension Kit for Raspberry Pi, The Pi Hut](https://thepihut.com/products/gmsl2-8mp-camera-extension-kit-for-raspberry-pi)
- C12: [Arducam GMSL Camera for Raspberry Pi 소개](https://docs.arducam.com/GMSL-Camera-Solution/GMSL-Camera-for-Raspberry-Pi/Introduction/)
- C13: [V-Link leverages GMSL2 to extend Raspberry Pi camera, CNX Software](https://www.cnx-software.com/2025/01/29/v-link-gmsl2-extend-raspberry-pi-camera-with-up-to-15-meter-cable/)
- C14: [Arducam 2MP IMX462 USB 3.0 Camera Module](https://www.arducam.com/product/arducam-2mp-imx462-manual-focus-usb-3-0-camera-module/)
- C15: [Arducam True Day and Night Vision USB2 UVC Camera, IP65](https://www.arducam.com/arducam-true-day-night-vision-usb2-uvc-camera-with-ip65-waterproof-housing.html)
- K1: [디바이스마트, 카메라모듈 3 와이드](https://www.devicemart.co.kr/goods/view?no=14933041)
- K2: [디바이스마트, 카메라모듈 3 NoIR 와이드](https://www.devicemart.co.kr/goods/view?no=14944173)
- K3: [디바이스마트, 카메라모듈 3](https://www.devicemart.co.kr/goods/view?no=14917048)
- K4: [가치창조기술, 라즈베리 글로벌셔터 카메라 C/CS 마운트](https://vctec.co.kr/product/x/21663/)
- K5: [가치창조기술, 아두캠 저조도 카메라 모듈 IMX462](https://vctec.co.kr/product/x/19478/)
- K6: [디바이스마트, 라즈베리파이5 FPC 카메라 케이블 300mm](https://www.devicemart.co.kr/goods/view?no=15285261)
- K7: [디바이스마트, Arducam CSI-HDMI 케이블 확장 모듈 B0091](https://www.devicemart.co.kr/goods/view?no=12231955)
