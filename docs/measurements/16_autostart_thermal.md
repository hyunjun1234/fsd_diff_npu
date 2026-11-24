# 16. 자동 시작, 안전 종료, 과열 대응 측정

이 문서는 과열 단계 감시(`--thermal`), 안전 종료(SIGTERM, 신호 파일), systemd 사용자 서비스 파일을 파이 5에서 시험한 결과입니다. 측정일은 2026-11-25, 장치는 파이 5(Debian 13), 방열판과 팬이 없는 Hailo-8과 DX-M1, 카메라 2대(configs/two_cam_current.yaml)입니다. 사용법은 docs/vehicle_power_and_autostart.md 에 있습니다.

## 1. 요약

| 항목 | 결과 | 근거 파일 |
|---|---|---|
| 테스트(맥 Python 3.9) | 325 통과, 4 건너뜀 | 16_pytest_mac.log |
| 테스트(파이 .venv 3.13) | 328 통과, 1 건너뜀 | 16_pytest_pi.log |
| 서비스 파일 문법 | systemd-analyze verify --user 출력 없음(통과) | 이 문서 5장 |
| 기본 경계값 20초 데모 | 정상 단계 유지, Hailo 79.7도에서 84.3도 | 16_thermal_demo.csv |
| 낮춘 경계값 데모 | 주의 다음 경고로 전환, 검출 상한 동작 | 16_thermal_demo_low.csv |
| DX-M1 추론 중 온도 읽기 | 읽힘, 경고에서 세그 멈춤 | 16_thermal_dual_low.csv |
| 녹화 중 SIGTERM | 세그먼트 3/3 깨끗하게 닫힘 | 16_rec_verify.txt |
| 녹화 중 신호 파일 | 세그먼트 3/3, 전원 끄기 자리 명령 실행 | 16_rec_verify.txt |

## 2. 기본 경계값으로 20초 데모

실행 전 Hailo-8 79.7도, DX-M1 75도였습니다. Hailo 검출만 켜고 온도를 2초마다 읽었습니다.

```bash
timeout -s INT -k 10 40 python3 scripts/run_demo.py --config configs/two_cam_current.yaml --backend hailo --headless \
    --duration 20 --thermal --thermal-interval 2 --thermal-csv out/measure/16_thermal_demo.csv \
    --health-csv out/measure/16_demo_health.csv --health-interval 10 --hailo-stop-temp 95
```

Hailo-8은 2초에 약 0.5도씩 올라 20초 뒤 84.3도였고, 주의 경계 90도에 닿지 않아 정상 단계로 끝났습니다. 데모 시작 전 Hailo가 이미 80도 안팎이라 주의 단계에 들어가는 것을 기대했지만 20초로는 부족했습니다. FPS는 10초 구간 두 번에 25.41, 25.59였습니다. 감시 스레드를 켜지 않은 같은 날 비교 실행은 하지 않아, 감시에 따른 FPS 차이는 확인 필요입니다(이전 측정 Hailo 단독 26.2 FPS는 다른 날 값입니다). SoC는 최고 62.8도, throttled는 0x0이었습니다.

## 3. 경계값을 낮춰 단계 전환 확인

기본값을 바꾸지 않고, 실행 직전 Hailo 온도(81.2도)에 1도와 2.5도를 더한 값을 주의, 경고 경계로 준 별도 실행입니다(`--thermal-limits "hailo=82.2,83.7,100"`, `--view 3d`, 나머지 옵션은 2장과 같음).

| 시각(초) | 단계 | Hailo 온도 |
|---|---|---|
| 0.1 | normal | 81.1 |
| 4.2 | caution | 82.4 |
| 8.3 | warning | 83.7 |
| 18.5 | warning | 83.7 |

주의에서 세그, 차선, 3D 바닥 주기가 2배, 경고에서 4배가 되고 검출이 초당 8회로 제한된 것이 로그에 남았습니다. 20초 동안 실제 검출은 278번, 건너뛴 프레임은 262번이었습니다. 경고 단계에 들어간 뒤 Hailo 온도는 83.5~83.7도에 머물러, 그 전 8초 동안 2.6도 오른 것과 달리 더 오르지 않았습니다. 화면 오른쪽 위에 `THERMAL warning hailo 83.7C`가 표시된 것을 스냅샷(16_demo_low_snap.jpg)으로 확인했습니다. 20초 안에 해제 온도까지 식지는 않아, 실제 장치에서 단계가 내려가는 것은 보지 못했습니다. 내려가는 동작과 히스테리시스는 가짜 온도 테스트(tests/test_autostart_thermal.py)로만 확인했습니다.

## 4. 두 NPU 실행에서 DX-M1 경고

Hailo 검출과 DX-M1 BiSeNetV2 세그를 같이 15초 돌리고, DX-M1 경계만 `dxm1=76,78,90`으로 낮췄습니다. 실행 전 DX-M1 74도, 끝난 뒤 77도였습니다(16_temps.log).

| 시각(초) | 단계 | DX-M1 온도 |
|---|---|---|
| 0.1 | normal | 75 |
| 2.2 | caution | 77 |
| 4.2 | warning | 78 |
| 14.5 | warning | 77 |

DX-M1이 추론하는 중에도 `dxrt-cli -s`로 온도가 읽혔습니다. 경고 단계에서 세그 스레드로 영상 넘기기를 멈추자 DX-M1은 77도로 내려와 그대로 있었습니다.

## 5. 서비스 파일 확인

파이에서 아래 명령의 출력이 없고 종료 코드가 0이었습니다. 서비스를 설치하거나 켜지는 않았습니다. 이 파이는 이미 `Linger=yes`입니다.

```bash
systemd-analyze verify --user deploy/systemd/fsd-record.service deploy/systemd/fsd-demo.service
```

## 6. 녹화 중 안전 종료

세그먼트를 5초로 두고 세 번째 세그먼트 중간에 끝냈습니다. 결과 폴더는 파이의 out/drives_test/ 입니다.

| 회차 | 끝난 이유 | 세그먼트(닫힘/전체) |
|---|---|---|
| t16_sigterm | signal:SIGTERM | 3/3 |
| t16_stopfile | power_loss:stop_file:out/tmp/STOP16 | 3/3 |
| t16_sigterm2 | signal:SIGTERM(Hailo 온도 읽기 수정 뒤) | 3/3 |

세 회차 모두 마지막 세그먼트에 `segment.json`이 있고, 카메라마다 csv 줄 수와 영상 안 JPEG 수가 같으며, 마지막 JPEG의 끝 위치가 파일 크기와 같고 디코드되었습니다(16_rec_verify.txt). `os.sync()`는 31~54ms였습니다. 신호 파일 시험에서는 시작 전에 일부러 남겨 둔 신호 파일을 지운 뒤 녹화를 시작했고, 12초 뒤 새로 만든 파일로 끝났으며, `--power-loss-cmd`로 준 `touch out/tmp/poweroff_called_16`이 실행되었습니다. SIGTERM 시험은 `timeout`의 종료 코드 124였고 `-k 15`의 SIGKILL까지 가지 않았습니다.

녹화 중 카메라별 누락 추정은 0~3장이었습니다(t16_sigterm 0과 3, t16_stopfile 1과 1, t16_sigterm2 2와 2). Hailo 온도를 읽지 못하던 첫 회차에도 누락이 있어 온도 읽기 탓으로 보지는 않지만, 원인은 확인 필요입니다.

첫 두 녹화에서는 Hailo 온도 읽기가 실패했습니다. 장치를 잠깐 열어 읽을 때 물리 장치 객체를 변수로 붙잡지 않으면 HailoRT가 바로 장치를 풀어 "The device in use has been released" 오류가 났습니다. 고친 뒤 t16_sigterm2에서는 2초마다 81.4~81.5도가 기록되었습니다.

## 7. 남은 확인

ACC GPIO는 배선이 없어 실제 핀 입력으로는 시험하지 못했습니다(가짜 입력으로 디바운스만 테스트). GPIO17이 비어 있고 라벨 `pinctrl-rp1` 칩이 `/dev/gpiochip0`인 것만 확인했습니다. 데모 서비스의 화면 환경 변수, 실제 부팅 뒤 자동 시작, 전원 회로가 버티는 시간도 확인 필요입니다.
