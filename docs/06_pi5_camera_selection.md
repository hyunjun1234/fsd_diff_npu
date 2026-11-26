# 06. 테슬라 카메라 자리별 라즈베리파이 5 카메라 선정

이 문서는 테슬라 모델 Y의 카메라 자리마다 화각과 사용 조건을 기준으로, 라즈베리파이 5에 붙일 수 있는 카메라 제품을 고른 자료입니다. 테슬라 쪽 화각과 위치는 [04 문서](04_tesla_camera_layout.md)의 값을 그대로 씁니다.

## 고르는 기준

NPU 입력이 640x640 이하로 줄어들기 때문에 화소 수는 1~2MP면 충분합니다. 대신 세 가지를 봤습니다. 첫째는 수평 화각이 테슬라의 같은 자리와 얼마나 가까운지, 둘째는 터널 출구나 해를 마주 보는 장면을 위한 HDR, 셋째는 야간 저조도 성능입니다. 여기에 라즈베리파이 5에서 별도 드라이버 없이 바로 잡히는지(CSI 카메라는 libcamera, USB 카메라는 UVC)를 더했습니다. 좌우와 후방의 화각은 테슬라 화각에 맞추는 대신 [19번 측정](measurements/19_camera_count_sim.md)의 시뮬레이션으로 정했습니다.

테슬라 HW4 카메라 센서(소니 IMX963으로 알려짐)는 일반 판매 모듈을 찾지 못했습니다. 차량용 HDR 센서 IMX490 모듈은 GMSL2 연결이라 파이 5에는 변환 보드가 필요해 이번 후보에서 뺐습니다.

## 후보 제품

| 제품 | 수평 화각 | 강점 |
|---|---|---|
| Camera Module 3 | 66도 | HDR, 자동초점 |
| Camera Module 3 Wide | 102도 | HDR, 자동초점 |
| Arducam IMX462 141도 | 141도 | 저조도 |
| Arducam IMX296 M12 | 45도 | 글로벌 셔터 |
| Arducam IMX708 쿼드 키트 | 100도 | CSI 하나에 4대 |
| Arducam IMX291 160도 USB | 약 139도 | 저조도, 넓은 화각 |
| Arducam IMX291 120도 방수 USB | 약 105도 | 저조도, 방수 금속 케이스 |

Camera Module 3 계열은 소니 IMX708 센서이고, HDR 모드에서 최대 3MP로 출력합니다 [C1]. 공식 페이지는 대각 화각만 적어 두었는데 일반형이 75도, 와이드가 120도입니다 [C1]. 수평 화각은 일반형 66도(디바이스마트 표기 [K3]), 와이드 102도(같은 센서와 렌즈의 Arducam 제품 표기 [C3])입니다. 적외선 차단 필터가 없는 NoIR 버전도 있어서 적외선 조명과 함께 쓰면 야간 촬영이 됩니다 [C1].

Arducam IMX462 141도 모델은 화각이 대각 161도, 수평 141.4도, 수직 81.7도이고 1920x1080입니다 [C2]. 소니 STARVIS 저조도 센서라 밤에 강합니다. 제조사는 파이 5를 지원하고 추가 드라이버가 필요 없다고 적었습니다 [C2]. 다만 같은 페이지에 적외선 차단 필터가 있다는 말과 근적외선을 감지한다는 말이 함께 있고, 최대 프레임레이트도 46fps와 30fps가 섞여 있어 실물 확인 필요입니다. HDR 표기는 없습니다.

Arducam IMX291 USB 카메라는 소니 STARVIS 저조도 센서(1920x1080)를 쓰고, MJPG로 1920x1080 30fps까지 보냅니다 [C16]. 160도 모델은 화각을 대각으로만 적어 두어 [C16], 표의 수평 화각은 등거리 어안 렌즈로 보고 옮긴 값이라 확인 필요입니다. 금속 케이스에 든 모델(UB020201)과 기판만 있는 모델(B0202)이 있고, 둘 다 방수 표기는 없으며 소비 전류는 최대 300mA [C16], 동작 온도는 -20~70도입니다 [C19]. 120도 모델(B0448)은 방수 금속 케이스에 들어 있지만 제조사는 IP 등급을 적지 않았고 [C17], 대각 120도를 등거리 어안으로 옮기면 수평 화각은 약 105도로 어림됩니다(확인 필요).

Arducam IMX296 M12 모델은 1440x1080 글로벌 셔터 카메라이고 화각은 수평 45도입니다 [C4]. 빠르게 움직이는 장면에서 롤링 셔터 기울어짐이 없지만 HDR은 없습니다. 라즈베리파이 공식 글로벌 셔터 카메라도 같은 센서이며 렌즈를 따로 사는 C/CS 마운트 제품입니다.

## 저조도 성능 비교

| 제품 | 화소 크기 | 조리개 |
|---|---|---|
| Camera Module 3 | 1.4um | F1.8 |
| Camera Module 3 Wide | 1.4um | F2.2 |
| IMX708 쿼드 키트 | 1.4um | F2.2 |
| Arducam IMX462 141도 | 2.9um | 확인 필요 |

화소 크기와 조리개는 라즈베리파이 카메라 문서 [C8], Arducam 제품 페이지 [C2][C5]의 값입니다. IMX462는 화소 면적이 IMX708의 약 4.3배라 한 화소가 받는 빛이 훨씬 많고, 제조사는 0.001lux에서도 찍힌다고 적었습니다 [C2]. 렌즈 조리개는 표기가 없어 확인 필요입니다. 반대로 IMX708 계열은 저조도에 강한 센서가 아닙니다. 특히 Wide와 쿼드 키트는 F2.2라 일반형(F1.8)보다 빛을 약 1.5배 적게 받습니다.

IMX708의 2304x1296 모드는 화소 4개를 묶어 읽는 방식이라 묶은 화소 하나의 크기가 2.8um에 해당합니다. NPU 입력이 640픽셀 안팎이라 이 모드나 더 작은 모드를 쓰게 되므로 실제 차이는 표의 화소 크기 차이보다 작을 것으로 보지만, 실측은 확인 필요입니다. 대신 IMX462는 HDR 표기가 없어서 터널 출구나 마주 오는 전조등처럼 밝기 차이가 큰 장면에는 약합니다. 앞유리의 두 대는 역광을 자주 마주치므로 HDR이 되는 IMX708 계열로, 옆과 뒤의 세 대는 IMX462와 같은 소니 STARVIS 저조도 센서인 IMX291 [C16]로 고른 이유가 이것입니다. IMX291 제품의 화소 크기와 조리개는 표기를 찾지 못해 확인 필요입니다.

## 테슬라 자리별 추천

| 자리 | 테슬라 화각 | 추천 제품 |
|---|---|---|
| 전방 광각 | 120도 | Camera Module 3 Wide |
| 전방 메인 | 50도 | Camera Module 3 |
| 좌우(B필러 자리, 정측면) | 90도 | Arducam IMX291 160도 USB |
| 후방(번호판 위) | 130도 | Arducam IMX291 160도 USB(방수 처리 필요) |
| 전방 범퍼 | 140도 | 5대 구성에서는 뺌 |
| 펜더 좌우 | 80도 | 5대 구성에서는 뺌 |

카메라는 5대로 정했습니다. 지금의 2대와 4대에서 8대까지의 배치를 같은 장면에 넣어 비교한 시뮬레이션([19번 측정](measurements/19_camera_count_sim.md))에서, 이 5대는 차를 한 바퀴 도는 사람을 끊김 없이 같은 ID로 따라갔고 물체가 화면에 나온 비율(표시 비율)은 8대보다 1.2%p 낮았습니다. 범퍼와 펜더를 더하면 처리량이 떨어지고 USB 허브가 필요해, 이 둘은 필요해질 때 더합니다. 다만 이 결과는 보닛에 발이 가려진 사람도 검출된다는 것과 어안 렌즈 가장자리의 사람도 가운데만큼 검출된다는 두 가정에 기대며, 어느 하나가 무너지면 5대는 이 기준을 넘지 못했습니다(19번 측정 8장). 그래서 카메라를 받으면 이 두 가지와 좌우 렌즈의 실제 화각을 먼저 확인합니다.

전방 광각은 앞유리 안쪽이라 역광과 터널 출구를 자주 마주치므로 HDR이 되는 Camera Module 3 Wide가 맞습니다. 수평 102도로 테슬라의 120도보다 좁지만 가장 가깝고, 같은 화각대에서 HDR이 되는 제품은 이것뿐이었습니다. 지금 달린 ov5647 130도는 화각은 넓지만 HDR이 없고, 130도가 수평인지 대각인지도 확인 필요입니다. 시뮬레이션에서는 전방 광각을 ov5647로 두어도 결과가 거의 같아서(아홉 장면 표시 비율 0.915와 0.913, 19번 측정 4.3절), 새로 사는 순서는 뒤로 미뤄도 됩니다.

전방 메인 50도에는 Camera Module 3 일반형(66도)을 권합니다. 화각이 조금 넓지만 HDR이 있습니다. 움직임 번짐이 문제가 되면 IMX296 M12(45도)로 바꿀 수 있는데, 이때는 HDR을 포기해야 합니다. 전방 메인은 3D 화면에 나오는 25~40m 앞을 맡고, 시뮬레이션에서 이 거리의 표시 비율을 0.39에서 0.50으로 올렸습니다(19번 측정 4.3절).

좌우는 테슬라처럼 B필러와 펜더에 두 대씩 다는 대신, B필러 자리에서 옆을 정면으로 보는 대각 160도 카메라 한 대씩으로 정했습니다. 좌우 렌즈의 수평 화각이 130도보다 좁으면 차 옆을 지나는 사람의 ID가 바뀌거나(126도) 끊겼고(122도), 좌우 카메라를 펜더에서 뒤쪽 옆을 보게 달면 사람이 문 옆에서 끊겼습니다(19번 측정 4.1절과 5장). 옆을 보는 카메라는 해를 정면으로 받는 일이 적어 HDR보다 저조도가 중요하므로 저조도 센서인 IMX291이 맞습니다. 이 제품은 방수 표기가 없어 옆유리 안쪽에 붙이는 것을 권합니다. 카니발에서는 3열 옆유리 안쪽에 달고, 그 자리로 다시 돌린 시뮬레이션에서도 기준을 통과했습니다([20번 측정](measurements/20_carnival_mount.md) 6장). 차 밖 B필러에 달려면 따로 방수 처리가 필요합니다.

후방은 테슬라처럼 차 밖 번호판 위에 달고, 좌우와 같은 IMX291 160도 USB 카메라를 씁니다. 시뮬레이션에서 120도 방수 모델(B0448)은 차 뒤 모서리를 도는 사람의 ID가 바뀌었고, 뒷유리 안쪽은 차 바로 뒤에 선 사람의 발이 트렁크에 가려 거리를 실제보다 멀리 잡았습니다([19번 측정](measurements/19_camera_count_sim.md) 6장). 160도급이면서 방수 표기가 있는 USB 카메라는 찾지 못했고, Arducam의 USB 2.0 카메라 안내 표에서 방수 케이스로 적힌 IMX291 모델은 120도 모델뿐입니다 [C18]. 그래서 후방은 금속 케이스 모델(UB020201)을 비를 덜 맞는 자리(카니발에서는 번호판 홈의 위쪽 턱 아래, 20번 측정 5장)에 달고, 케이블이 나오는 곳과 케이스 이음새, 렌즈 나사를 실리콘으로 막습니다. 렌즈 앞에 평평한 창을 덧대면 160도 시야의 가장자리가 잘릴 수 있습니다. 이렇게 막은 카메라가 비와 세차를 견디는지는 확인 필요입니다.

범퍼와 펜더를 늘린다면 앞범퍼 카메라가 먼저입니다. 시뮬레이션에서는 범퍼에도 후방과 같은 IMX291 160도를 넣었고, 차 밖이라 후방처럼 방수 처리가 필요합니다. 5대에서는 앞범퍼 앞 2.4m 안의 바닥이 앞유리 카메라에 보이지 않아, 이 안에 선 사람이 실제보다 1.2~2.3m 멀리 그려지고 키가 작은 아이는 잘 잡히지 않습니다. 범퍼 카메라는 이곳을 메우고, 교차로에서 옆 차로에 선 차 너머의 보행자도 일부 봅니다. 그다음은 앞펜더에서 뒤쪽 옆을 보는 2대(STARVIS 2 IP65, 수평 86도 [C15])로, 정차 중 차 사이로 뒤에서 다가오는 오토바이가 1~2초 늦게 나타나는 것을 메웁니다. 다만 실제 영상에서 어안 렌즈 가장자리의 검출이 크게 떨어지면 앞펜더를 먼저 더합니다. 어느 쪽이든 카메라 간 병합을 먼저 고쳐야 효과가 납니다([19번 측정](measurements/19_camera_count_sim.md) 4.3절, 4.4절, 8장).

## 3대 이상 연결할 때

파이 5의 카메라 포트는 2개라 3대 이상은 다른 방법이 필요합니다.

Arducam IMX708 쿼드 키트는 Camarray HAT 하나로 4대를 CSI 포트 하나에 묶습니다 [C5]. 4대를 켜면 한 장에 2x2로 합쳐 나오며 카메라당 1152x648이 되고, 프레임레이트는 단독일 때의 절반입니다 [C5]. 0번과 1번, 2번과 3번이 각각 동기화되고 2번과 3번은 한 프레임 늦습니다 [C5]. 제조사 표의 파이 5 지원 칸은 비어 있어 공식 지원은 확인 필요입니다. Arducam 포럼에는 파이 5에서 `dtoverlay=imx708,cam0` 설정으로 이 키트가 잡혔다는 사용자 사례가 있고, Arducam 직원은 IMX708 키트에는 Arducam 전용 libcamera가 필요 없다고 답했습니다 [C6]. 4대 모두 같은 화각이라 전방 메인이나 범퍼처럼 화각이 다른 자리에는 쓸 수 없습니다.

Arducam 멀티 카메라 어댑터(V2.2)는 여러 대를 붙여도 한 번에 한 대씩 번갈아 켜는 방식이라 [C7] 동시에 360도를 봐야 하는 이 프로젝트에는 맞지 않습니다. 파이 5에는 별도 변환 케이블도 필요하고 Camera Module 3 호환도 확인되지 않았습니다 [C7].

나머지 방법은 CSI 2대에 USB 카메라를 더하는 구성이고, 5대 구성이 이 방식입니다. USB 쪽은 하드웨어 동기화가 없으므로 타임스탬프로 프레임을 묶어야 합니다(03 문서 2.3절). 녹화용 SSD가 USB 포트 하나를 쓰므로 허브 없이 달 수 있는 USB 카메라는 3대입니다(19번 측정 7장).

모형 시험에서는 CSI 0번에 쿼드 키트(좌우와 후방 등 같은 화각 자리), CSI 1번에 전방 카메라 한 대를 두는 구성이 가장 간단합니다. 실제 차량에서는 배선 길이 때문에 구성이 달라지며, 다음 장에 정리했습니다.

## 차량에 달 때의 배선

파이는 조수석 아래 바닥에 둡니다. 카니발 3D 모델에서 [07 문서](07_vehicle_mount_and_training_plan.md) 2.3절의 배선 경로를 따라 잰 길이에 여유 15%를 더하면 앞유리 약 3.7m, 3열 옆유리 우측 약 3.0m와 좌측 약 3.8m, 뒷번호판 위 약 6.1m입니다([20번 측정](measurements/20_carnival_mount.md) 7장). 범퍼와 펜더 자리는 아직 재지 않았습니다. 모두 어림값이며 실제 차에서 다시 잽니다. 파이 5에 기본으로 쓰는 CSI 리본 케이블은 판매 제품이 300mm 수준이라 [K6] 이 거리를 그대로 감당하지 못합니다.

| 방식 | 최대 거리 | 파이 5 |
|---|---|---|
| CSI-HDMI 확장 | 10m | 지원 |
| GMSL2 확장 키트 | 15m | 지원 |
| USB 카메라 | 케이블 규격에 따름 | 지원 |

CSI-HDMI 확장 모듈(Arducam B0091)은 CSI 신호를 HDMI 케이블로 옮겨 10m까지 늘립니다 [C9][C10]. Camera Module 3에서 동작한다고 적혀 있고 [C10], 고품질 HDMI 케이블(제조사는 HDMI 2.1 케이블을 권함)을 쓰라는 조건이 붙어 있습니다 [C9]. 세트에는 HDMI 케이블이 들어 있지 않아 따로 준비합니다 [C9][C10]. 파이 5에는 22핀 변환 케이블이 따로 필요합니다 [C10]. 값이 싸고 지금 쓰는 CSI 카메라를 그대로 쓸 수 있지만, HDMI 커넥터는 차량 진동에 맞게 만든 것이 아니라 커넥터 고정이 필요합니다.

GMSL2 확장 키트는 차량용 동축 케이블(FAKRA)로 최대 15m까지 보내며 파이 5 호환으로 표기돼 있습니다 [C11]. 차량용으로 만든 방식이라 가장 튼튼하지만, Arducam 키트에 들어가는 카메라는 IMX219와 IMX477뿐이라 HDR이나 저조도 센서를 고를 수 없습니다 [C11][C12]. Camera Module 3를 GMSL2로 잇는 다른 회사 제품(V-Link)도 있지만 크라우드펀딩 제품이고 별도 드라이버가 필요해 [C13] 지금은 후보에서 뺐습니다.

USB 카메라는 파이 5의 CSI 포트 2개를 넘는 카메라에 씁니다(5대 구성에서는 좌우와 후방 3대). USB 2.0은 수동 케이블로 5m 안팎까지 쓸 수 있고 USB 3.0은 그보다 짧다고 알려져 있으나 확인 필요입니다. 저조도 USB 카메라로 Arducam IMX462 USB 3.0 모듈(수평 85도, F1.0, 1080p 60fps)도 있지만 [C14], 좌우나 후방에 쓰기에는 화각이 좁고 USB 2.0 포트에 꽂으면 720p 10fps로 떨어집니다 [C14]. 차 밖에 다는 방수 제품으로는 적외선 LED가 달린 Arducam STARVIS 2 USB 카메라(IP65, 수평 86도, USB 2.0)가 있지만 [C15], 좌우나 후방 한 대로 쓰기에는 화각이 좁습니다(19번 측정). USB 카메라는 하드웨어 동기화가 없어서 타임스탬프로 프레임을 묶어야 합니다.

쿼드 키트는 카메라 4대가 HAT 하나에 짧은 리본으로 연결되는 구조라, 차 앞뒤와 양옆에 카메라를 흩어 놓는 데는 맞지 않습니다. 카메라마다 CSI-HDMI 확장을 붙이는 방법은 확인 필요라서 쿼드 키트는 모형 시험용으로만 권합니다.

차 밖(펜더, 범퍼, 번호판 위)에 다는 카메라는 방수가 필요합니다. 처음 차량 시험에서는 전방 2대를 앞유리, 좌우 2대를 3열 옆유리 안쪽에 붙이면 방수 문제 없이 시작할 수 있습니다. 카니발에 맞춘 자리는 `configs/five_cam_carnival.yaml`에 적었고, 붙인 뒤 보정한 값으로 바꿉니다. 후방은 뒷유리 안쪽에 달면 차 바로 뒤에 선 사람의 거리를 멀리 잡으므로(19번 측정 6장) 처음부터 번호판 위에 방수 처리를 해서 답니다.

정리하면 차량용 구성은 5대입니다. 앞유리의 전방 광각과 전방 메인은 CSI-HDMI 확장과 HDMI 5m 케이블로 조수석 아래까지 잇고, 좌우 2대는 USB 카메라에 3m USB 2.0 연장 케이블을 이어 씁니다. 카메라 기본 케이블 1m(금속 케이스 모델 UB020201 기준 [C19], 기판 모델 B0202는 확인 필요)에 연장 3m를 더하면 4m로, 좌측에 필요한 약 3.8m보다 길고 앞에서 본 USB 2.0 수동 케이블 길이(5m 안팎) 안에 듭니다. 후방은 번호판에서 테일게이트 안과 D필러, 오른쪽 문턱을 지나 조수석 아래까지 약 6.1m라, 수동 연장 대신 카메라 케이블 1m 뒤에 5m USB 2.0 리피터 연장 케이블 두 개를 잇습니다 [K11]. 하나만 이으면 6m라 모자랍니다. 리피터는 안에 허브 칩이 든 케이블이라 리눅스가 그 뒤의 장치에 포트당 100mA만 배정할 수 있고, 그러면 이 카메라가 동작하지 않을 수 있어 5V 입력 단자가 있는 제품을 고르고 두 개 모두 전원을 넣습니다(07 문서 2.3절).

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
| Arducam IMX291 160도 USB(B0202) | 90,640원 | 디바이스마트 [K8] |
| Arducam IMX291 160도 USB 금속 케이스(UB020201) | 확인 필요 | 해외 60.99달러 [C19] |
| Arducam IMX291 120도 방수 USB(B0448) | 96,800원 | 디바이스마트 [K9] |
| USB 2.0 연장 케이블 3m | 1,430원 | 디바이스마트 [K10] |
| USB 2.0 리피터 연장 케이블 5m(5V 입력 단자) | 13,000원 | 컴퓨존 [K11] |
| HDMI 2.1 케이블 5m | 5,550원 | 컴퓨존 [K12] |
| STARVIS 2 USB IP65 카메라 | 확인 필요 | 해외 59.99달러 [C15] |
| IMX462 USB 3.0 모듈 | 확인 필요 | 가격 표기 없음 [C14] |

5대 구성에서 사는 것은 IMX291 160도 USB(좌우용 B0202 2대, 후방용 UB020201 1대), CSI-HDMI 확장 모듈 2세트, 카메라 케이블, HDMI 5m 케이블 2개, USB 연장 케이블 2개, 후방용 리피터 케이블 2개입니다. 전방 메인은 지금 후방에 단 Camera Module 3를 옮겨 쓰고, Camera Module 3 Wide는 지금의 ov5647을 전방 광각으로 쓰다가 나중에 사도 됩니다. 나머지는 비교용이며, 120도 방수 모델(B0448)은 시뮬레이션에서 후방 기준을 넘지 못해 사지 않습니다.

국내 가격은 모두 부가세 포함입니다. 리피터는 판매가 칸에 9,500원과 13,000원이 함께 보여 옵션 목록에 적힌 13,000원을 적었고, 전원 어댑터는 들어 있지 않습니다. 차에 붙이는 데 드는 테이프, 실리콘, 브래킷, 공구는 [07 문서](07_vehicle_mount_and_training_plan.md) 12장에 있습니다. 디바이스마트 상품 페이지에는 품절 문구가 함께 보여 재고는 주문 전에 확인 필요입니다. 가치창조기술의 IMX462는 화각 92도의 다른 모델(Pivariety 방식)이라 141도 모델과 같은 제품이 아닙니다 [K5]. 141도 모델과 쿼드 키트는 국내 판매처를 찾지 못해 Arducam 공식몰 가격만 적었고, 해외 배송비와 관세는 확인 필요입니다. 디바이스마트의 IMX291 160도(B0202)는 해외 주문 상품으로 표시돼 있어 받기까지 2주 안팎이 걸리고 [K8], 기판만 있는 모델이라 차에 달 때는 케이스나 고정판이 따로 필요합니다.

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
- C16: [Arducam 1080P Low Light WDR USB Camera Module with Metal Case, IMX291 160 Degree](https://www.arducam.com/b020201-arducam-1080p-low-light-wdr-usb-camera-module-with-metal-case-2mp-1-2-8-cmos-imx291-160-degree-ultra-wide-angle-mini-uvc-webcam-board-with-microphones.html)
- C17: [Arducam 2MP IMX291 Low Light 120° Wide Angle USB Camera Module with Waterproof Metal Case](https://www.arducam.com/2mp-imx291-low-light-120-wide-angle-usb-camera-module-with-waterproof-metal-case-for-windows-linux-android-and-mac-os.html)
- C18: [Arducam USB 2.0 UVC 카메라 사양과 선택 안내](https://docs.arducam.com/UVC-Camera/USB2-UVC-Camera-Kit/Specs-and-Selection-Guide/)
- C19: [UCTRONICS, Arducam IMX291 160도 USB 카메라 금속 케이스 UB020201](https://www.uctronics.com/arducam-1080p-low-light-wdr-usb-camera-module-with-metal-case-2mp-1-2-8-cmos-imx291-160-degree-ultra-wide-angle-mini-uvc-webcam-board-with-microphone-3-3ft-1m-cable-for-windows-linux-mac-os-2244-ub020201.html)
- K1: [디바이스마트, 카메라모듈 3 와이드](https://www.devicemart.co.kr/goods/view?no=14933041)
- K2: [디바이스마트, 카메라모듈 3 NoIR 와이드](https://www.devicemart.co.kr/goods/view?no=14944173)
- K3: [디바이스마트, 카메라모듈 3](https://www.devicemart.co.kr/goods/view?no=14917048)
- K4: [가치창조기술, 라즈베리 글로벌셔터 카메라 C/CS 마운트](https://vctec.co.kr/product/x/21663/)
- K5: [가치창조기술, 아두캠 저조도 카메라 모듈 IMX462](https://vctec.co.kr/product/x/19478/)
- K6: [디바이스마트, 라즈베리파이5 FPC 카메라 케이블 300mm](https://www.devicemart.co.kr/goods/view?no=15285261)
- K7: [디바이스마트, Arducam CSI-HDMI 케이블 확장 모듈 B0091](https://www.devicemart.co.kr/goods/view?no=12231955)
- K8: [디바이스마트, IMX291 160도 Fisheye USB 카메라 모듈 B0202](https://www.devicemart.co.kr/goods/view?no=15946158)
- K9: [디바이스마트, IMX291 120도 광각 USB 카메라 모듈 방수 금속 케이스 B0448](https://www.devicemart.co.kr/goods/view?no=15946282)
- K10: [디바이스마트, USB-A 2.0 M/F 연장 케이블 MBF-UF230 3m](https://www.devicemart.co.kr/goods/view?no=12235985)
- K11: [컴퓨존, 강원전자 USB 2.0 리피터 연장 케이블 CBL-203-5P 5m](https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=932980)
- K12: [컴퓨존, 넥시 HDMI v2.1 8K 케이블 NX-HDMI21, 5m 옵션](https://www.compuzone.co.kr/product/product_detail.htm?ProductNo=1211225)
