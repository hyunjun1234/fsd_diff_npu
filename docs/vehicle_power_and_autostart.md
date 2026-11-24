# 차량 전원, 자동 시작, 과열 대응

이 문서는 파이를 차에 달았을 때 시동과 함께 녹화나 데모가 저절로 시작되고, 시동을 끄면 파일을 깨끗하게 닫고 끝나며, 칩이 뜨거워지면 단계별로 부하를 낮추도록 설정하는 방법을 정리한 안내서입니다. 전원 회로의 배경은 docs/07_vehicle_mount_and_training_plan.md 4장, 발열 배경은 5.1절을 봅니다.

모든 명령은 파이 저장소 폴더에서 실행합니다.

```bash
cd ~/work/fsd_diff_npu && . .venv/bin/activate
```

## 1. 서비스 파일

저장소의 `deploy/systemd/` 에 systemd 사용자 서비스 두 개가 있습니다. 둘 다 카메라를 잡으므로 `Conflicts=`로 서로 동시에 돌지 않게 했습니다.

| 파일 | 하는 일 | 끝내는 신호 |
|---|---|---|
| fsd-record.service | 카메라 원본 녹화(NPU 안 씀) | SIGTERM |
| fsd-demo.service | Hailo-8 검출과 3D 화면 | SIGINT |

`run_demo.py`는 SIGINT를 받아야 장치를 놓고 끝나므로 데모 서비스는 `KillSignal=SIGINT`를 씁니다. 두 서비스 모두 `--thermal`(과열 단계 대응)과 `--stop-file`(시험용 신호 파일)을 켜 두었고, 저장소가 `~/work/fsd_diff_npu`, 가상환경이 그 안의 `.venv`라고 가정합니다. 경로가 다르면 파일의 `WorkingDirectory`와 `ExecStart`를 고칩니다.

과열로 스스로 끝날 때는 종료 코드 75를 내고, 서비스는 `RestartPreventExitStatus=75`로 이때 다시 띄우지 않습니다. 식지 않은 칩을 곧바로 다시 달구지 않기 위해서입니다. 그 밖의 오류로 죽으면 15초 뒤 다시 띄우고, 10분 동안 5번 넘게 죽으면 멈춥니다.

데모 서비스는 창을 띄우므로 데스크톱 자동 로그인이 필요합니다. 파일에 적은 `WAYLAND_DISPLAY=wayland-0`, `DISPLAY=:0`이 지금 파이 데스크톱과 맞는지는 확인 필요입니다. 화면 없이 돌릴 때는 `ExecStart` 끝에 `--headless`를 더합니다.

## 2. 설치와 켜고 끄기

사용자 서비스라서 설치에는 sudo가 필요 없습니다. 서비스 파일을 사용자 설정 폴더에 연결하고 읽어 들입니다.

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/work/fsd_diff_npu/deploy/systemd/fsd-record.service ~/.config/systemd/user/
ln -sf ~/work/fsd_diff_npu/deploy/systemd/fsd-demo.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

부팅 때 자동으로 시작할 서비스 하나만 켭니다. 녹화를 예로 듭니다.

```bash
systemctl --user enable --now fsd-record.service
```

로그인하지 않아도 사용자 서비스가 부팅 때 돌려면 linger가 켜져 있어야 합니다. 지금 파이는 `loginctl show-user $USER -p Linger`가 `Linger=yes`라 이미 켜져 있습니다. 새 파이에서는 사람이 한 번 실행합니다.

```bash
sudo loginctl enable-linger $USER
```

자주 쓰는 명령은 아래와 같습니다.

```bash
systemctl --user status fsd-record.service       # 상태
systemctl --user stop fsd-record.service         # 멈춤(SIGTERM, 세그먼트를 닫고 끝남)
systemctl --user disable fsd-record.service      # 부팅 때 자동 시작 끔
journalctl --user -u fsd-record.service -f       # 실시간 로그
journalctl --user -u fsd-record.service -b       # 이번 부팅 이후 로그
```

녹화 결과는 `out/drives/<시작 시각>/`에, 그 회차의 온도 기록은 같은 폴더의 `thermal.csv`에 생깁니다. 데모 서비스의 온도와 건강 기록은 `out/service/demo_thermal.csv`, `out/service/demo_health.csv`에 생깁니다.

서비스 파일을 고친 뒤에는 문법을 먼저 확인합니다. 아무것도 출력하지 않으면 통과입니다.

```bash
systemd-analyze verify --user deploy/systemd/fsd-record.service deploy/systemd/fsd-demo.service
```

## 3. 안전 종료

### 3.1 받는 신호

`record_drive.py`와 `run_demo.py`는 같은 옵션으로 전원 꺼짐 신호를 받습니다(`realityview/power.py`).

| 신호 | 옵션 | 동작 |
|---|---|---|
| SIGTERM | 없음(녹화 도구 기본) | 세그먼트를 닫고 os.sync |
| 신호 파일 | `--stop-file out/STOP_RECORD` | 파일이 생기면 바로 꺼짐으로 봄 |
| ACC GPIO | `--acc-gpio 17` | 꺼짐이 2초 이어지면 꺼짐으로 봄 |

녹화 도구는 어떤 신호로 끝나든 카메라마다 지금 세그먼트를 닫고(fsync), `segment.json`과 `drive.json`을 쓴 뒤 `os.sync()`로 나머지 파일까지 디스크에 내립니다. 신호 파일은 시작할 때 이미 있으면 지난 실행이 남긴 것으로 보고 지웁니다. 그래서 남은 파일 때문에 다음 부팅 녹화가 곧바로 끝나지는 않습니다.

GPIO 감시는 `--acc-gpio`를 줄 때만 켜집니다. 배선하지 않은 핀은 풀업 때문에 꺼짐으로 읽히므로, 배선 전에는 이 옵션을 넣지 않습니다. 시동을 걸 때 잠깐 전압이 빠지는 것은 `--acc-debounce`(기본 2초)로 거릅니다. GPIO 읽기가 실패하면 꺼짐으로 보지 않습니다. GPIO는 파이에 이미 있는 libgpiod 2.x 파이썬 모듈(`gpiod`, .venv 에서 2.2.0)로 읽고, 40핀 헤더 칩은 라벨 `pinctrl-rp1`로 찾습니다(지금 파이는 `/dev/gpiochip0`).

### 3.2 전원 끄기 자리

실제 `poweroff`는 코드가 스스로 부르지 않습니다. `--power-loss-cmd`에 명령을 주면 GPIO나 신호 파일로 끝났을 때만, 파일을 닫고 `os.sync()`를 마친 다음 그 명령을 부릅니다. SIGTERM으로 끝날 때는 systemd나 사람이 이미 끄는 중이므로 부르지 않습니다.

차에서 쓰려면 먼저 사람이 poweroff만 비밀번호 없이 허용하는 sudoers 파일을 만듭니다.

```bash
echo "$USER ALL=(root) NOPASSWD: /usr/sbin/poweroff" | sudo tee /etc/sudoers.d/fsd-poweroff
sudo chmod 440 /etc/sudoers.d/fsd-poweroff
sudo visudo -c
```

그다음 서비스 파일 `ExecStart` 끝에 `--acc-gpio 17 --power-loss-cmd "sudo -n /usr/sbin/poweroff"`를 더하고 `systemctl --user daemon-reload`, `systemctl --user restart fsd-record.service`를 실행합니다.

### 3.3 배선(확인 필요)

07 문서 4.2절의 방식대로 ACC 12V를 포토커플러로 받아 GPIO에 넣습니다. 아래 핀과 부품값은 아직 실물로 확인하지 않았습니다.

| 연결 | 파이 쪽 | 차 쪽 |
|---|---|---|
| 포토커플러 입력 | - | ACC 12V, 저항 거쳐 LED, 차량 GND |
| 포토커플러 출력 | GPIO17(헤더 11번 핀), GND(9번 핀) | - |
| 파이 전원 | 5V 5A 입력 | 상시 전원 또는 UPS 거친 12V |

포토커플러(PC817 같은 부품) LED 쪽 직렬 저항은 12V에서 LED 전류 5~10mA가 되게 고릅니다(약 1.5~2.2k옴, 부품 데이터시트로 확인 필요). 출력 트랜지스터의 컬렉터를 GPIO17, 이미터를 파이 GND에 잇고 내부 풀업을 씁니다. 그러면 ACC 켜짐에서 핀이 0, 꺼짐에서 1이 되고, 이것이 기본값 `--acc-on-value 0`과 맞습니다. 반대로 배선했다면 `--acc-on-value 1`을 줍니다.

ACC 전원만으로 파이를 켜면 시동을 끄는 순간 전원도 같이 끊겨 정리할 시간이 없습니다. 파이 전원은 상시 전원에서 받거나 UPS 또는 슈퍼커패시터를 거치게 하고, ACC는 신호로만 씁니다. 아래 시험에서 `os.sync()`는 31~54ms였고 SIGKILL까지 가지 않고 끝났지만, 신호부터 종료까지 걸린 전체 시간은 재지 않아 확인 필요입니다. 서비스는 넉넉히 40초(`TimeoutStopSec`)를 기다리고, UPS도 그 이상 버티는 것을 고릅니다.

### 3.4 시험

배선 없이 신호 파일과 SIGTERM으로 시험합니다. 세그먼트를 5초로 짧게 해서 세그먼트 중간에 끊기게 합니다.

```bash
# SIGTERM: 13초 뒤 timeout 이 SIGTERM 을 보낸다
timeout -s TERM -k 15 13 python3 scripts/record_drive.py --config configs/two_cam_current.yaml \
    --out out/drives_test --name t_sigterm --segment 5 --thermal --thermal-interval 2
# 신호 파일: 12초 뒤 파일을 만든다. 끝난 뒤 out/tmp/poweroff_called 가 생기면 전원 끄기 자리까지 온 것
(sleep 12; touch out/tmp/STOP) &
timeout -s TERM -k 15 40 python3 scripts/record_drive.py --config configs/two_cam_current.yaml \
    --out out/drives_test --name t_stopfile --segment 5 --stop-file out/tmp/STOP \
    --power-loss-cmd "touch out/tmp/poweroff_called"
python3 scripts/record_drive.py --list --out out/drives_test
```

결과는 `out/drives_test/<이름>/`에 생깁니다. `--list`의 세그 열이 `3/3`처럼 두 숫자가 같고, 끝난 이유가 `signal:SIGTERM`, `power_loss:stop_file:...`이면 통과입니다. 데모도 같은 `--stop-file`로 시험할 수 있고, 끝난 뒤 `[power] os.sync 완료` 줄이 나옵니다.

## 4. 과열 대응

### 4.1 무엇을 읽나

`--thermal`을 주면 감시 스레드가 5초(`--thermal-interval`)마다 아래 값을 읽습니다(`realityview/thermal.py`).

| 값 | 읽는 방법 | 비고 |
|---|---|---|
| Hailo-8 칩 온도 | Hailo 백엔드의 칩 온도 함수 | 녹화 도구는 장치를 잠깐 열어 읽음 |
| DX-M1 온도 | `dxrt-cli -s` 코어 최고값 | DX-M1 추론 중에도 읽힘 |
| SoC 온도와 throttled | thermal_zone0, `vcgencmd get_throttled` | throttled 지금 비트면 최소 주의 |

못 읽은 센서는 식은 것으로 보지 않고 직전 단계를 유지하며, 연속으로 실패하면 로그에 한 줄 남깁니다. 다른 프로그램이 Hailo를 쓰는 중이면 녹화 도구 쪽 Hailo 온도는 비어 있습니다.

### 4.2 단계와 기준값

센서마다 단계를 정하고, 전체 단계는 그중 가장 높은 값입니다. 온도가 경계값 이상이면 바로 그 단계로 올라가고, 경계값보다 히스테리시스만큼 더 식어야 내려갑니다.

| 센서 | 주의, 경고, 위험(도) | 히스테리시스 |
|---|---|---|
| Hailo-8 | 90, 95, 100 | 4도 |
| DX-M1 | 78, 80, 85 | 2도 |
| 파이 SoC | 75, 80, 85 | 5도 |

근거는 다음과 같습니다(방열판과 팬이 없는 파이 5, Hailo-8, DX-M1 실측). Hailo-8은 부하 없이 80~84도이고, 연속 실행 중 약 104도에서 HailoRT가 경고하며 클럭을 낮췄고 120도에서 스트림을 닫고 멈췄습니다. 그래서 위험을 104도보다 아래인 100도에, 경고를 기존 `--hailo-stop-temp` 권장값인 95도에 두었습니다. DX-M1은 부하 없이 72~75도이고 85도 이상을 위험으로 보며, 작업 규칙인 "80도를 넘으면 멈춤"에 맞춰 경고를 80도에 두었습니다. 파이 SoC는 펌웨어가 80도부터 클럭을 낮춥니다. 해제 온도(경계값 빼기 히스테리시스)는 세 센서 모두 부하 없을 때 온도보다 높게 잡아, 부하를 줄이면 단계가 풀립니다.

경계값은 `--thermal-limits "hailo=90,95,100;dxm1=78,80,85,2"`처럼 바꿉니다(네 번째 숫자는 히스테리시스, 생략 가능). 방열판과 팬을 단 뒤에는 부하 없을 때 온도를 다시 재고 이 값을 고칩니다.

### 4.3 단계별 조치

| 단계 | 데모(run_demo.py) | 녹화(record_drive.py) |
|---|---|---|
| 주의 | 세그, 차선, 3D 바닥 주기 2배 | 기록만 |
| 경고 | 주기 4배, DX-M1 작업 멈춤, 검출 초당 8회 | 기록만 |
| 위험 | 깨끗하게 끝냄(종료 코드 75) | 녹화 유지 |

데모의 위험 조치는 `--thermal-critical`로 바꿉니다. `camera_only`는 NPU 추론을 멈추고 카메라 화면만 계속 그리고, `continue`는 경고 조치를 유지합니다. 녹화 도구는 NPU를 쓰지 않으므로 기본이 `continue`이고, `--thermal-critical exit`를 주면 위험 단계에서 녹화를 닫고 종료 코드 75로 끝냅니다. `--backend dxm1`이면 경고 단계의 DX-M1 멈춤이 곧 검출 멈춤입니다. 검출 상한은 `--thermal-det-fps`로 바꾸고, 건너뛴 프레임에는 지난 검출 결과를 다시 씁니다.

단계가 바뀌면 로그에 `[thermal] 단계 normal -> caution (hailo 90.4)` 같은 줄과 적용한 조치가 남고, 데모 화면 오른쪽 위에 `THERMAL caution hailo 90.4C`처럼 표시됩니다(정상 회색, 주의 노랑, 경고 주황, 위험 빨강). `--thermal-csv`를 주면 매번 읽은 온도와 단계가 한 줄씩 남습니다. 끝날 때 `thermal_summary=` 줄에 최고 온도와 단계 전환 목록이 나오고, 녹화 도구는 같은 내용을 `drive.json`의 `thermal`에 남깁니다.

깊이 모델은 아직 `run_demo.py` 파이프라인에 없어서 주기를 늘리는 대상에 들어 있지 않습니다. 깊이를 파이프라인에 넣을 때 같은 방식으로 주기 배수를 걸어야 합니다.
