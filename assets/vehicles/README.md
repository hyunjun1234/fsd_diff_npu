# 차량 모델 출처

이 문서는 `assets/vehicles/` 안 차량 모델 파일의 출처와 라이선스를 적은 것입니다.

`ego_carnival_2011.npz`는 v15풍 화면(`--view v15`)의 자차 모델입니다. 원본 3D 모델을 `scripts/convert_vehicle_model.py`로 면 수를 줄이고 도장을 흰색으로 바꿔 만들었습니다. 변환 방법은 [docs/v15_render.md](../../docs/v15_render.md) 4장에 있습니다.

원작자 표기(CC-BY-4.0이 요구하는 문구를 그대로 옮김):

This work is based on "Kia Carnival 2011" (https://sketchfab.com/3d-models/kia-carnival-2011-855c464351f244cbaafa8269a5d472df) by Doz3 (https://sketchfab.com/Doz3) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

원본에서 바꾼 점은 엠블럼과 실내 질감 부품 삭제, 면 수 축소(153,373개에서 11,561개와 6,378개), 차체 색을 흰색으로 변경, 크기를 실제 길이 5.13m에 맞춤, 좌표를 차량 좌표(X 전방, Y 좌측, Z 위, 원점은 뒷바퀴 축 가운데 바닥)로 바꾼 것입니다.
