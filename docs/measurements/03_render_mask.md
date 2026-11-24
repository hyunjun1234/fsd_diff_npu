# 2단계: 도로 마스크 렌더링 속도 개선 기록

이 문서는 두 NPU를 함께 쓸 때 FPS가 떨어지던 원인(도로 마스크 덧칠과 BEV 합성)을 숫자로 확인하고, OpenCV 방식으로 고친 뒤 파이에서 전후를 잰 기록입니다. 작업일은 2026-11-24입니다. 세그 자리에는 아직 진짜 세그 .dxnn이 없어 드론 검출 모델을 넣었습니다.

## 요약

| 항목 | 결과 | 조건 |
|---|---|---|
| Hailo 단독 FPS | 14.18 에서 26.15 | 파이5, 카메라 2대, 200프레임 |
| Hailo + DX-M1 세그 FPS | 10.60 에서 21.44 | 같은 조건, 세그 자리에 드론 모델 |
| 같은 조건, 마스크 무작위 50% | 7.76 에서 20.05 | --seg-mask-test random50 |
| 화소 차이 | 최대 1 | 마스크 자리만, 마스크 밖과 바닥 영상은 같음 |
| 시험 | 맥 84 통과, 2 건너뜀 / 파이 86 통과 | 맥 /usr/bin/python3, 파이 .venv |
| DX-M1 최고 온도 | 75도 | 이번 작업의 모든 DX-M1 실행 |

판정: 계획 4.3절의 "두 NPU를 함께 쓸 때의 FPS가 1단계보다 낮아지지 않습니다"는 비교 기준에 따라 결과가 다릅니다. 1단계 기록값(14.11 FPS, 02 문서. 이번에 옛 코드로 다시 잰 값 14.18 FPS)과 비교하면 동시 실행이 21.44 FPS, 무작위 50% 마스크에서 20.05 FPS라 낮아지지 않았습니다. 같은 새 코드의 Hailo 단독(26.15 FPS)과 비교하면 동시 실행이 18%(드론 모델 마스크), 23%(무작위 50%) 낮습니다. 어느 쪽으로 판정할지는 사람이 정해 주세요. 진짜 세그 모델로 잰 값은 확인 필요입니다.

## 1. 시간이 나간 곳

카메라와 NPU 없이 마스크 경로만 따로 쟀습니다(scripts/bench_render_mask.py). 영상은 무작위 값, 마스크는 화소 50%가 1인 무작위 마스크, 카메라 2대(768x576, 768x432), 탑뷰 800x480, 30회 평균입니다.

### 1.1 옛 카메라 영상 덧칠의 단계별 시간 (카메라 2대 합계)

| 단계 | 파이5 ms | 맥 ms |
|---|---|---|
| 불리언 색인 읽기 out[m] | 15.5 | 7.6 |
| float64 계산과 uint8 변환 | 7.5 | 2.0 |
| 불리언 색인 쓰기 out[m] = | 15.6 | 7.7 |
| uint8 에서 bool 변환, 복사 | 1.2 | 0.05 |

네 줄을 더하면 1.2절의 덧칠 시간(파이 39.5ms)과 거의 같습니다. 덧칠 시간의 대부분은 계산이 아니라 선택된 화소를 모으고 다시 흩뿌리는 numpy 불리언 색인이었습니다.

### 1.2 함수별 전후 (파이5)

| 함수 | 전 ms | 후 ms |
|---|---|---|
| 카메라 2대 덧칠, 마스크 있음 | 39.5 | 5.8 |
| 카메라 2대 덧칠, 마스크 없음 | 0.4 | 0.4 |
| 바닥 영상 IPM 합성 | 22.0 | 8.9 |
| 마스크 IPM 합성 | 3.0 | 2.3 |
| 탑뷰 그리기, 도로 없음 | 23.2 | 5.4 |
| 탑뷰 그리기, 도로 50% | 32.6 | 7.5 |

### 1.3 함수별 전후 (맥)

| 함수 | 전 ms | 후 ms |
|---|---|---|
| 카메라 2대 덧칠, 마스크 있음 | 18.1 | 0.47 |
| 바닥 영상 IPM 합성 | 8.5 | 1.2 |
| 탑뷰 그리기, 도로 50% | 13.9 | 1.6 |

맥 sim 실행(카메라 2대, 130프레임 중 100프레임, 창 없음)에서는 렌더링이 27.7ms에서 2.7ms, BEV 합성이 9.1ms에서 1.5ms로 줄었습니다. 맥 sim은 합성 장면 그리기(카메라 획득 단계 60ms)가 대부분이라 FPS(10.18 에서 15.50)는 참고만 합니다.

- 바닥 영상 IPM 합성(22ms)과 탑뷰의 바닥 영상 깔기(23ms)도 같은 불리언 색인 때문에 느렸습니다. 이 두 곳은 마스크가 없어도 매 프레임 돌아서 Hailo 단독 FPS도 같이 올랐습니다.
- 원본: out/measure/03_bench_mask_before_pi.json, 03_bench_mask_after_pi.json, 03_bench_mask_before_mac.json, 03_bench_mask_after_mac.json

## 2. 고친 내용

| 곳 | 예전 | 지금 |
|---|---|---|
| 카메라 영상 덧칠 | numpy 불리언 색인과 float64 계산 | cv2.addWeighted로 전체를 섞고 cv2.copyTo로 마스크 자리만 복사 |
| 탑뷰 도로 덧칠 | numpy 불리언 색인과 float64 계산 | addWeighted와 copyTo(도로이면서 카메라가 보는 셀만) |
| 탑뷰 바닥 영상 | 화소마다 float32 계산 | 회색값 256개 색조 표(cv2.LUT), 결과가 예전과 똑같음 |
| IPM 합성 | 매번 owner == i 비교와 불리언 색인 | 카메라별 담당 셀 마스크를 미리 만들고 cv2.copyTo |
| 탑뷰 도로 마스크 | 매 프레임 다시 합성 | 마스크 배열이 지난 프레임과 같으면 지난 결과 재사용 |

- 화소 차이: OpenCV는 반올림, 예전 식은 버림이라 마스크 자리 화소가 최대 1 다릅니다. 마스크 밖 화소, 바닥 영상, IPM 합성 결과는 예전과 똑같습니다(시험으로 확인).
- 캐시: 세그는 3프레임마다 돌고 그 사이에는 같은 마스크 배열이 붙어 옵니다. 카메라마다 배열 객체가 같은지(is)로 비교해 같으면 합성을 건너뜁니다. 동시 실행 229프레임 중 152프레임이 재사용, 77프레임이 다시 합성이었습니다. 카메라 영상은 매 프레임 바뀌어서 덧칠 자체는 캐시하지 않았습니다.
- 작게 칠한 뒤 확대하는 방식은 마스크 경계에서 차이가 1을 넘을 수 있어 쓰지 않았습니다.

## 3. 파이 전후 측정

조건: 파이5, Hailo-8 yolov8s_h8.hef, 카메라 2대(configs/two_cam_current.yaml, 768x576과 768x432), 230프레임 중 앞 30프레임을 뺀 200프레임, 창 없음, 렌더링 같은 스레드, 시나리오마다 실행 1회, 2026-11-24 18:23 이전(DX-M1 실행의 시각은 4장 표). 세그 실행은 out/tmp/two_cam_dxseg_stub.yaml(세그 자리에 ~/DaeDron/runs/npu/deepx/tiny_p3p5_relu.dxnn, 3프레임마다)을 썼습니다. 스로틀링은 모든 실행에서 0x0입니다.

### 3.1 Hailo 단독

| 단계 | 전 ms | 후 ms |
|---|---|---|
| Hailo 추론 | 14.0 | 13.9 |
| BEV 합성 | 22.9 | 9.5 |
| 렌더링 | 28.3 | 9.4 |
| 프레임 전체 | 70.5 | 38.2 |
| 루프 FPS | 14.18 | 26.15 |

### 3.2 Hailo + DX-M1 세그(드론 모델 출력 마스크, 도로 비율 11%)

| 단계 | 전 ms | 후 ms |
|---|---|---|
| Hailo 추론 | 15.1 | 14.8 |
| 세그 추론(별도 스레드) | 27.9 | 27.7 |
| BEV 합성 | 26.5 | 10.7 |
| 렌더링 | 45.1 | 14.2 |
| 프레임 전체 | 94.3 | 46.6 |
| 루프 FPS | 10.60 | 21.44 |

### 3.3 Hailo + DX-M1 세그(마스크 무작위 50%로 강제)

| 단계 | 전 ms | 후 ms |
|---|---|---|
| Hailo 추론 | 15.2 | 15.1 |
| 세그 추론(별도 스레드) | 27.6 | 28.1 |
| BEV 합성 | 26.9 | 10.8 |
| 렌더링 | 79.8 | 17.1 |
| 프레임 전체 | 128.8 | 49.9 |
| 루프 FPS | 7.76 | 20.05 |

- 세그는 네 번의 동시 실행 모두 77번 넘겨 77번 끝났고 버린 작업은 0, 마스크는 평균 2.0프레임 묵었습니다.
- 도로 비율은 새 통계 seg_stats.mask_fraction_mean 값입니다. 드론 모델 출력은 의미 없는 마스크라 실제 도로 비율과 다릅니다.
- 같은 새 코드에서 동시 실행이 Hailo 단독보다 8.4ms(드론 모델 마스크) 또는 11.6ms(무작위 50%) 느립니다. 늘어난 곳은 렌더링 4.8ms(드론 모델 마스크) 또는 7.7ms(무작위 50%), BEV 합성 1.2ms, 전처리 1.2ms, Hailo 추론 0.9ms입니다. 덧칠 외의 증가는 세그 스레드가 CPU를 나눠 쓴 몫으로 보이며 비율은 확인 필요입니다.
- 원본: out/measure/03_profile_before_hailo.json, 03_profile_after_hailo.json, 03_profile_before_dual_stub.json, 03_profile_after_dual_stub.json, 03_profile_before_dual_r50.json, 03_profile_after_dual_r50.json과 같은 이름의 .log

## 4. DX-M1 온도 기록

DX-M1을 쓰는 실행마다 앞뒤로 dxrt-cli -s를 읽었습니다(scripts/dx_temp.py, 85도를 넘으면 60초씩 쉬며 80도 아래까지 최대 10분 기다림). 85도를 넘은 적이 없어 기다린 적은 없습니다.

| 시각 | 시점 | DX-M1 최고 |
|---|---|---|
| 18:17:51 | 전, 드론 모델 마스크 실행 앞 | 72도 |
| 18:18:16 | 전, 드론 모델 마스크 실행 뒤 | 74도 |
| 18:18:16 | 전, 무작위 50% 실행 앞 | 74도 |
| 18:18:48 | 전, 무작위 50% 실행 뒤 | 75도 |
| 18:22:10 | 후, 드론 모델 마스크 실행 앞 | 73도 |
| 18:22:24 | 후, 드론 모델 마스크 실행 뒤 | 74도 |
| 18:22:24 | 후, 무작위 50% 실행 앞 | 74도 |
| 18:22:38 | 후, 무작위 50% 실행 뒤 | 75도 |
| 18:22:52 | 파이 시험 앞 | 74도 |
| 18:23:10 | 파이 시험 뒤 | 74도 |

한 번 실행이 15초에서 30초로 짧고 세그가 3프레임마다만 돌아서 온도는 3도 안쪽으로만 올랐습니다. 오래 돌릴 때의 온도는 확인 필요입니다. 원본: out/measure/03_dx_temp.log

## 5. 시험

| 장소 | 결과 | 명령 |
|---|---|---|
| 맥 | 84 통과, 2 건너뜀 | `/usr/bin/python3 -m pytest -q` |
| 파이 | 86 통과 | `RV_DX_TEST_MODEL=$HOME/DaeDron/runs/npu/deepx/tiny_p3p5_relu.dxnn timeout 180 python3 -m pytest -q` |

새 시험(tests/test_render_mask.py)은 예전 식을 기준 함수로 두고 다음을 봅니다.

- 0~255 값을 모두 담은 영상에서 덧칠 차이 1 이하, 마스크 밖은 정확히 같음
- 마스크 없음, 전부 0, 전부 1, bool, 255 값, int32 마스크
- 바닥 영상 색조 표가 예전과 화소까지 같음, coverage가 bool, uint8, 없음인 경우
- IPM 합성이 예전과 똑같음, coverage가 읽기 전용
- 도로 마스크 캐시: 같은 배열이면 재사용, 한 카메라만 바뀌어도 다시 합성, 마스크가 사라지면 비움, sim은 늘 다시 합성
- 무작위 50% 감싸개와 --seg-mask-test 옵션

맥 결과는 out/measure/03_pytest_mac.log, 파이 결과는 03_pytest_pi.log에 있습니다.

## 6. 다시 재는 명령

파이 저장소 폴더에서 .venv를 켠 뒤 실행합니다. 결과는 out/measure/에 생깁니다.

```bash
timeout 120 python3 scripts/bench_render_mask.py --config configs/two_cam_current.yaml --reps 30 --out out/measure/03_bench_mask_after_pi.json
timeout -s INT -k 10 120 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --headless --frames 230 --profile --profile-out out/measure/03_profile_after_hailo.json
python3 scripts/dx_temp.py --label pre --log out/measure/03_dx_temp.log --guard 85 --resume-below 80
timeout -s INT -k 10 120 python3 scripts/run_demo.py --config out/tmp/two_cam_dxseg_stub.yaml --backend hailo --seg-npu dxm1 --seg-mask-test random50 --headless --frames 230 --profile --profile-out out/measure/03_profile_after_dual_r50.json
python3 scripts/dx_temp.py --label post --log out/measure/03_dx_temp.log
```

## 7. 확인 필요

- 진짜 세그 .dxnn으로 동시 실행 FPS(세그 후처리 CPU 시간이 드론 모델과 다를 수 있음)
- 동시 실행과 Hailo 단독의 남은 차이(8.4ms에서 11.6ms) 중 세그 스레드의 CPU 사용 몫
- 몇 분 이상 동시 실행할 때 DX-M1 온도
