# 04. 테슬라 카메라 배치와 시야

이 문서는 테슬라 모델 Y 외부 카메라의 장착 위치와 각 카메라의 방향, 화각, 최대 거리를 정리한 자료입니다. 같은 내용을 돌려 볼 수 있는 3D 도면이 [viz/tesla_camera_layout_3d.html](../viz/tesla_camera_layout_3d.html)에 있습니다. 리얼리티 뷰가 HW4 전용이므로 HW4 배치를 기준으로 하고, HW3와 다른 점을 함께 적었습니다.

## 좌표계

원점은 뒷바퀴 축 가운데의 지면이고, x는 전방, y는 왼쪽, z는 위쪽이며 단위는 m입니다. 저장소 configs 파일과 같습니다. 차체 치수는 휠베이스 2.890m, 앞 오버행 0.893m, 뒤 오버행 1.011m, 전폭 1.920m(미러 제외), 전고 1.621m입니다 [S2].

## 카메라별 제원

장착 위치는 매뉴얼에 문장으로만 나와 있어서 [S1] 좌표는 모두 추정치입니다. 방향은 정면을 0도로 하고 왼쪽으로 돌수록 커집니다.

| 카메라 | 장착 위치 | 좌표 x, y, z (추정) | 방향 | 수평 화각 | 최대 거리 | 탑재 |
|---|---|---|---|---|---|---|
| 전방 협각 | 앞유리 상단 | 1.80, 0.06, 1.45 | 정면 | 35도 (확인 필요) | 250m [S3] | HW3 |
| 전방 메인 | 앞유리 상단 | 1.80, 0.00, 1.45 | 정면 | 50도 (확인 필요) | 150m [S3] | HW3, HW4 |
| 전방 광각 | 앞유리 상단 | 1.80, -0.06, 1.45 | 정면 | 120도 어안 (확인 필요) | 60m [S3] | HW3, HW4 |
| 전방 범퍼 | 앞범퍼 그릴 위 [S1] | 3.72, 0.00, 0.55 | 정면, 아래로 12도 (추정) | 확인 필요 (도면 140도) | 확인 필요 (도면 20m) | HW4 일부 |
| B필러 좌우 | 양쪽 문 기둥 [S1] | 1.20, ±0.93, 1.15 | 전방 측면 60도 (추정) | 90도 (확인 필요) | 80m [S5] | HW3, HW4 |
| 펜더 좌우 | 앞펜더 방향지시등 [S1] | 2.25, ±0.97, 0.92 | 후방 쪽 약 45도 [S4] | 확인 필요 (도면 80도) | 100m [S5] | HW3, HW4 |
| 후방 | 뒷번호판 위 [S1] | -0.95, 0.00, 1.00 | 뒤쪽, 아래로 20도 (추정) | 확인 필요 (도면 130도) | 50m [S4][S5] | HW3, HW4 |

전방 35도, 50도, 120도와 B필러 90도는 테슬라 오토파일럿 소개 그림의 값으로 널리 인용되지만 원문은 확인 필요입니다. B필러와 펜더 거리는 포럼 출처라 신뢰도가 낮습니다 [S5]. 수직 화각은 공개 값이 없어 센서 가로세로비(HW3 1280×960, HW4 2896×1876 [R1])로 계산했습니다.

## HW3와 HW4의 차이

HW3는 앞유리에 협각, 메인, 광각 3대가 있고, HW4는 메인과 광각 2대만 쓰며 세 번째 자리는 비어 있습니다 [R1][S6]. HW4 모델 Y는 앞범퍼 그릴 위에 카메라가 하나 더 있으며 [S1], 모델과 생산 시기에 따라 없는 차도 있습니다 [R2]. HW4 카메라는 화각이 더 넓다고 보도됐지만 수치는 공개되지 않았습니다 [S6].

## 도면 보는 법

기본값인 "거리 축약"은 2.56m 이후 거리를 1.6×√거리로 줄여 그리고, 바닥 링의 숫자는 실제 거리입니다. "실제 거리"를 누르면 1:1로 바뀝니다. 시야 광선은 땅에 닿으면 멈추므로 바닥에 찍히는 범위가 보입니다. 주황색 "추정" 꼬리표가 붙은 값은 공식 자료가 없는 값입니다.

## 출처

| 번호 | 출처 |
|---|---|
| S1 | [Tesla 모델 Y 소유자 매뉴얼, 카메라 위치](https://www.tesla.com/ownersmanual/modely/en_gb/GUID-682FF4A7-D083-4C95-925A-5EE3752F4865.html) |
| S2 | [Tesla 모델 Y 소유자 매뉴얼, 치수와 중량](https://www.tesla.com/ownersmanual/modely/en_gb/GUID-1E76B638-7B12-4D9A-8767-94B7F1E92A0E.html) |
| S3 | [Here's what Tesla's Autopilot 2.0 can see with its 8 cameras, Electrek](https://electrek.co/2017/05/16/tesla-autopilot-2-0-can-see/) |
| S4 | [Tesla Guide: Number of Cameras, Their Locations, Uses and How To View Them, Not a Tesla App](https://www.notateslaapp.com/news/1452/tesla-guide-number-of-cameras-their-locations-uses-and-how-to-view-them) |
| S5 | [AP2.0 Cameras: Capabilities and Limitations, Tesla Motors Club 포럼](https://teslamotorsclub.com/tmc/threads/ap2-0-cameras-capabilities-and-limitations.86430) |
| S6 | [Tesla's New Hardware 4 May Help Make Unprotected Turns Safer, InsideEVs](https://insideevs.com/news/658708/tesla-new-hardware-4-improvements-unprotected-turn-safety/) |
| R1 | 이 저장소의 [01 문서](01_fsd_v15_reality_view_research.md) 6장 |
| R2 | 같은 문서의 출처 S19 |
