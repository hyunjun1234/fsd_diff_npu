# DX-M1 .dxnn 모델 출처 기록

이 문서는 2단계에서 파이로 내려받은 DEEPX 공식 .dxnn 모델의 주소, 크기, sha256과 버전을 고른 근거를 적은 기록입니다. 내려받은 날은 2026-11-24이고, 파일은 파이 저장소의 models/dxnn/ 에 있습니다.

## 1. 주소를 찾은 경로

DEEPX가 운영하는 GitHub 조직 DEEPX-AI의 dx_app 저장소가 모델 목록 파일을 함께 배포합니다. 파이에서 curl로 다음 순서로 읽었습니다.

1. `https://api.github.com/orgs/DEEPX-AI/repos` 에서 dx_app, dx_rt, dx-modelzoo 저장소를 찾았습니다.
2. dx_app의 `setup_sample_models.sh` 는 `scripts/download_models.py` 를 부르고, 이 스크립트는 같은 폴더의 `modelzoo_manifest.json` 에 적힌 주소에서 받습니다.
3. 모델 주소는 모두 DEEPX 서버 `https://sdk.deepx.ai/modelzoo/` 아래에 있습니다.

## 2. 버전을 고른 근거

파이의 dx_engine과 DX-RT는 3.4.2입니다(00_env_check). 버전은 다음 근거로 맞췄습니다.

- dx_app main(v3.3.0)의 목록은 2_5_0 폴더를 가리킵니다. 같은 저장소 RELEASE_NOTES.md는 "Model Zoo 2_5_0 .dxnn files are container format v9, which needs DX-RT 3.5.0 or later"라고 적고 있어 쓰지 않았습니다.
- DX-RT 3.4.2와 dx_app v3.2.2는 둘 다 2026-08-14에 나왔습니다(dx_rt, dx_app RELEASE_NOTES.md). v3.2.2 태그의 `scripts/modelzoo_manifest.json` 은 2_4_0 폴더를 가리킵니다.
- 받은 파일을 `parse_model` 로 열어 보니 모두 .dxnn 형식 v8, DX-COM v2.4.0-rc.4였습니다. DX-RT RELEASE_NOTES에 따르면 v8 형식은 3.x에서 읽고 v9 파서는 3.5.0에서 들어왔습니다. 실제로 다섯 파일 모두 파이의 dx_engine 3.4.2에서 열리고 돌았습니다.

목록 파일 사본은 파이의 out/tmp/dxzoo/ 에 있습니다(modelzoo_manifest.json은 main, manifest_v3.2.2.json은 v3.2.2 태그).

## 3. 받은 파일

다섯 파일의 주소는 모두 `https://sdk.deepx.ai/modelzoo/dxnn/2_4_0/` 뒤에 파일 이름을 붙인 것입니다. 크기는 바이트입니다.

| 파일 | 크기 | sha256 |
|---|---|---|
| yolov8-s_640x640.dxnn | 24048217 | 8a475b29006f6e345a335cc48460afc0b0050e0bf94473525982e562ba227925 |
| yolov8-s_640x640_ppu.dxnn | 27994893 | c0d7f30ad542ae7219c56d211a6543708d2df49d68f09cea1d25862d3ddcf68a |
| bisenetv2_1024x2048.dxnn | 15659227 | 1586ee2bd6ceb17b93d1f3218abd3d16dd70753a988c1d8811d8398fa8870a80 |
| segformer_mit-b0_512x1024.dxnn | 15593929 | 37ec2bbd979902c5787830d569e81f85355b7f5edc30f275b7fd855d8052ffe6 |
| stdc2-seg50_512x1024.dxnn | 43877963 | 3a94aef92f5f57eedc28340a3a13787ec6ad424747f58d7fd8a8ad41dc049218 |

목록 파일에 적힌 이름은 차례대로 YoloV8S, YoloV8S_PPU, BiSeNetV2, segformer_b0_512x1024, stdc2-model_maxmIOU50-1입니다.

## 4. 쓰는 모델과 빼 둔 모델

설정(configs/two_cam_current.yaml, two_cam_model.yaml)에는 검출 yolov8-s_640x640_ppu.dxnn, 세그 bisenetv2_1024x2048.dxnn을 적었습니다. yolov8-s_640x640.dxnn은 비교용으로 둡니다.

SegFormer-B0와 STDC2-Seg50은 받아서 재 봤지만 쓰지 않습니다. 파이에서 한 장씩 처리량이 SegFormer 0.43 FPS(지연 2.3초), STDC2 1.6 FPS(출력이 19x512x1024 float 3개)였습니다. 둘 다 NPU 시간은 5ms, 71ms인데 나머지가 CPU 작업이었습니다. 자세한 값은 06_stage2_dxnn.md 1장에 있습니다.

## 5. 다시 받는 명령

파이 저장소 폴더에서 실행합니다. 결과는 models/dxnn/ 에 생깁니다.

```bash
mkdir -p models/dxnn && cd models/dxnn
for f in yolov8-s_640x640.dxnn yolov8-s_640x640_ppu.dxnn bisenetv2_1024x2048.dxnn; do
  curl -sS -f -o $f https://sdk.deepx.ai/modelzoo/dxnn/2_4_0/$f
done
sha256sum *.dxnn
```

나온 sha256이 3장 표와 같은지 확인합니다. DX-RT를 3.5.0 이상으로 올리면 2_5_0 폴더의 v9 모델로 바꿀 수 있지만, 출력 형식이 같은지는 확인 필요입니다.
