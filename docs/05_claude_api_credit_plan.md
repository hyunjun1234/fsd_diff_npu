# Claude API 크레딧 사용 계획

이 문서는 Claude Console(Anthropic API)의 200달러 조직 크레딧을 이 프로젝트에 어디에, 어떤 모델로, 얼마씩 쓸지 정한 계획과 Console에서 시작하는 방법을 적은 안내서입니다. 구현 단계 이름과 완료 기준은 [라즈베리파이 5, Hailo-8, DX-M1 구현 계획](03_pi5_hailo8_dxm1_implementation_plan.md)을 따릅니다.

## 1. 어디에 쓰는가

이 크레딧은 API(Batch API 포함), Agent SDK, Playground, 관리형 에이전트(Managed Agents)에만 쓸 수 있습니다(크레딧 안내 기준). Claude Code 사용은 대상이 아니므로 계획에 넣지 않았습니다.

크레딧은 사람이 손으로 하면 오래 걸리지만 NPU에서 돌릴 필요는 없는 일에 씁니다. 파이5 위에서 실시간으로 도는 경로(검출, 세그, BEV 합성, 렌더링)에는 넣지 않습니다. 오프라인으로 PC에서 돌리고 결과만 저장소의 데이터나 기록으로 남깁니다.

| 용도 | 쓰는 방식 |
|---|---|
| 자체 영상 검수와 태깅 | Batch API |
| 결과 화면 비교 판정 | Batch API |
| 컴파일과 실행 로그 분석 | API |
| 자료 읽기 | API |
| 연산자 확인 자동 반복 | Agent SDK |
| ONNX 분할과 BEVPool 시험 | 관리형 에이전트 |
| NPU 소프트웨어 변경 확인 | 관리형 에이전트 |
| 프롬프트 다듬기 | Playground |

용도별로 구현 계획의 어디에 쓰이고 무엇을 하는지는 다음과 같습니다.

- 자체 영상 검수와 태깅: 2단계 검출 비교, 3단계 INT8 보정 이미지 고르기, 5단계 의사 라벨 걸러내기에 씁니다. 카메라 프레임과 NPU 검출 결과를 함께 보내 놓친 물체, 잘못 잡은 물체를 표시하고 장면 조건(밝기, 역광, 비, 경사로, 주차장 등)을 태그로 붙입니다. 사람은 표시된 프레임만 다시 봅니다.
- 결과 화면 비교 판정: 3단계의 "한 바퀴 도는 사람이 같은 ID로 추적되는지"와 4단계의 "occupancy와 지면 투영 결과를 나란히 놓고 확인"에 씁니다. 카메라 영상과 BEV 화면을 한 장으로 붙여 보내고, 두 결과가 크게 어긋나는 프레임을 골라 이유를 적게 합니다.
- 컴파일과 실행 로그 분석: 0단계 드라이버 설치와 2단계 이후의 실행 오류에 씁니다. Hailo DFC, DX-COM 컴파일 로그나 `dmesg`를 넣고 지원되지 않은 연산자와 그 위치, 바꿀 수 있는 대체 연산을 정리하게 합니다.
- 자료 읽기: 4단계 FlashOcc 경량화와 DX-RT, HailoRT 문서에 씁니다. 논문과 SDK PDF를 넣고 필요한 부분(입력 크기, BEVPool 구현, 양자화 옵션)을 찾아 묻습니다.
- 연산자 확인 자동 반복: 4.5절 연산자 확인에 쓰며 PC에서 실행합니다. 에이전트가 최소 ONNX 모델을 만들고, PC에 설치된 Hailo DFC와 DX-COM으로 컴파일하고, 로그를 읽고, 실패한 연산을 바꿔 다시 컴파일하는 일을 반복한 뒤 모델 선정 문서 3장 표에 들어갈 결과를 정리합니다.
- ONNX 분할과 BEVPool 시험: 4단계 FlashOcc 분할 실행에 씁니다. Anthropic이 운영하는 샌드박스에 FlashOcc 코드를 받아 CPU만으로 ResNet18 구성의 ONNX를 앞뒤로 나눠 내보내고, BEVPool NumPy 구현이 원래 결과와 같은지 무작위 입력으로 확인합니다. 학습은 하지 않습니다.
- NPU 소프트웨어 변경 확인: 문서의 "확인 필요" 항목에 쓰며 주기적으로 실행합니다. 주 1회 Hailo, DEEPX, Ultralytics의 릴리스 노트를 웹 검색으로 읽고, 연산자 지원이나 Raspberry Pi OS 지원이 바뀌었으면 요약을 남깁니다.
- 프롬프트 다듬기: 위 Batch 작업 전체에 씁니다. 프레임 몇 장으로 프롬프트와 모델, effort를 고른 뒤 코드로 옮깁니다.

연산자 확인을 Agent SDK로 PC에서 돌리는 이유는 Hailo DFC와 DX-COM이 PC에 설치된 컴파일러여서 에이전트가 그 자리에서 직접 실행해야 하기 때문입니다. 관리형 에이전트는 Anthropic 쪽 샌드박스에서 돌기 때문에 이 컴파일러가 필요 없는 일만 맡깁니다.

픽셀 단위 마스크나 정밀한 3D 박스를 Claude가 직접 만들게 하는 일은 넣지 않았습니다. 도로 마스크 IoU를 잴 20프레임은 지금 계획대로 사람이 칠하고, Claude는 그 앞뒤의 고르기와 검수만 맡깁니다.

## 2. 모델과 가격

가격은 Anthropic 공식 가격표 기준이며 100만 토큰당 달러입니다 [C1]. Batch API로 보내면 입력과 출력 모두 반값이고, 대부분 1시간 안에 끝나며 24시간 안에 끝나지 않은 요청은 청구되지 않습니다 [C1][C2].

| 모델 | 입력 | 출력 | Batch 입력 | Batch 출력 |
|---|---|---|---|---|
| Haiku 5.5 | 0.10 | 0.50 | 0.05 | 0.25 |
| Sonnet 5.5 | 2 | 10 | 1 | 5 |
| Opus 5.5 | 4 | 20 | 2 | 10 |

모델 ID와 이 프로젝트에서 맡길 일은 다음과 같습니다.

- Claude Haiku 5.5(`claude-haiku-5-5`): 대량 프레임 1차 검수와 태깅
- Claude Sonnet 5.5(`claude-sonnet-5-5`): Haiku가 애매하다고 한 프레임 재확인, 결과 화면 비교
- Claude Opus 5.5(`claude-opus-5-5`): 컴파일 로그 분석, 논문과 SDK 문서 읽기

Haiku 5.5 가격은 프롬프트가 10만 토큰 이하일 때 값이고, 넘으면 입력 0.50, 출력 2.50으로 오릅니다 [C1]. 프레임 한 장씩 보내는 용도에서는 넘을 일이 없습니다. Claude Fable 5.2(입력 8, 출력 40)는 이 예산에서는 쓰지 않습니다 [C1].

사고(thinking) 토큰은 출력 토큰으로 청구됩니다. Opus 5.5와 Haiku 5.5의 effort 기본값은 `medium`이므로, 대량 작업은 `low`로 명시해 출력량을 줄입니다(확인 필요: 실제 감소 폭은 처음 100건의 `usage` 값으로 잽니다).

### 2.1 이미지 한 장 비용

이미지는 28x28 픽셀 단위로 나뉘고 `ceil(가로/28) x ceil(세로/28)`개의 토큰이 듭니다 [C3]. 4.7 이후 모델은 긴 변 2,576px, 4,784토큰까지 그대로 처리합니다 [C3]. Camera Module 3 Wide 프레임을 1280x720으로 줄여 보내면 46 x 26 = 1,196토큰입니다.

프롬프트와 검출 결과 텍스트를 약 400토큰, 응답(사고 포함)을 약 400토큰으로 가정하면 프레임 1,000장의 Batch 비용은 다음과 같습니다. 400토큰은 가정값이므로 처음 시험한 결과로 바꿉니다.

| 모델 | 입력 1.6M 토큰 | 출력 0.4M 토큰 | 1,000장 합계 |
|---|---|---|---|
| Haiku 5.5 (Batch) | 0.08달러 | 0.10달러 | 약 0.18달러 |
| Sonnet 5.5 (Batch) | 1.60달러 | 2.00달러 | 약 3.60달러 |
| Opus 5.5 (Batch) | 3.20달러 | 4.00달러 | 약 7.20달러 |

카메라 4대를 1초에 1장씩 30분 녹화하면 7,200장입니다. 전부 Haiku로 보면 약 1.3달러, 전부 Sonnet으로 보면 약 26달러입니다.

### 2.2 로그와 문서 한 번 비용

컴파일 로그 5만 토큰을 넣고 5천 토큰 답을 받으면 Opus 5.5 기준 입력 0.20달러, 출력 0.10달러로 약 0.30달러입니다(가격표로 계산 [C1]). 같은 문서에 여러 번 질문할 때는 프롬프트 캐시를 켭니다. Opus 5.5의 캐시 읽기는 100만 토큰당 0.20달러여서 두 번째 질문부터 입력 비용이 약 20분의 1로 줄어듭니다 [C1].

## 3. 200달러 예산안

| 항목 | 금액 | 모델 |
|---|---|---|
| 자체 영상 검수와 태깅 | 45달러 | Haiku 5.5, Sonnet 5.5 |
| 결과 화면 비교 판정 | 25달러 | Sonnet 5.5 |
| 로그 분석과 자료 읽기 | 25달러 | Opus 5.5 |
| 연산자 확인 자동 반복 | 40달러 | Sonnet 5.5 |
| ONNX 분할과 BEVPool 시험 | 20달러 | Sonnet 5.5 |
| NPU 소프트웨어 변경 확인 | 10달러 | Haiku 5.5 |
| 프롬프트 다듬기 | 10달러 | 세 모델 |
| 예비 | 25달러 | 해당 없음 |
| 합계 | 200달러 | 해당 없음 |

항목별로 대략 할 수 있는 양은 다음과 같습니다.

- 자체 영상 검수와 태깅(Batch API): Haiku 5.5로 전체를 보고 일부를 Sonnet 5.5로 다시 봅니다. Haiku로 5만 장(약 9달러)과 Sonnet 재확인 1만 장(약 36달러)입니다.
- 결과 화면 비교 판정(Batch API): 합친 이미지 약 4,600장입니다. 합친 이미지는 토큰이 더 들어 장당 비용을 1.5배로 잡았습니다.
- 로그 분석과 자료 읽기(API): 0.30달러짜리 분석과 질문을 합쳐 약 80번입니다.
- 연산자 확인 자동 반복(Agent SDK): 한 번 실행에 상한 5달러를 걸고 약 8번입니다.
- ONNX 분할과 BEVPool 시험(관리형 에이전트): 세션당 상한 5달러로 약 4번입니다.
- NPU 소프트웨어 변경 확인(관리형 에이전트): 주 1회 실행합니다. 검색 10번 안팎이면 한 번에 0.1~0.2달러 수준입니다. (추정)
- 프롬프트 다듬기(Playground): 용도별로 프레임 수십 장을 시험합니다.
- 예비: 5단계 데이터가 늘어날 때 씁니다.

순서는 구현 단계에 맞춥니다. 0단계에서는 로그 분석과 자료 읽기, 0~3단계와 함께 진행하는 4.5절에서는 Agent SDK 연산자 확인을 먼저 씁니다. 2단계부터 영상 검수, 3단계와 4단계에서 화면 비교 판정과 관리형 에이전트 ONNX 분할 시험을 씁니다.

에이전트 방식은 한 번 실행에 모델을 여러 번 부르기 때문에 단순 API 호출보다 비용이 빨리 늘어납니다. Claude Code 공식 문서에 기업 사용자 평균이 활동한 날 하루 약 13달러로 나와 있는데 [C5], Agent SDK는 Claude Code와 같은 에이전트 구조를 라이브러리로 쓰는 것이어서 [C8] 비슷한 규모로 쓰일 수 있습니다. 그래서 Agent SDK에는 실행마다 `max_budget_usd`를, 관리형 에이전트에는 세션 지출 상한(budget)을 반드시 겁니다. Agent SDK의 비용 상한은 프로그램이 계산한 추정치로 멈추는 방식이라 조금 넘을 수 있으므로 여유를 두고 잡습니다 [C9]. 관리형 에이전트는 토큰 비용에 더해 세션이 실제로 일하는 시간 1시간당 0.08달러가 붙고, 웹 검색은 1,000번에 10달러입니다 [C1]. 관리형 에이전트는 아직 베타입니다 [C10].

## 4. Console에서 시작하기

### 4.1 크레딧과 워크스페이스

먼저 [Billing](https://platform.claude.com/settings/billing) 페이지에서 200달러 크레딧이 조직에 들어와 있는지 확인합니다 [C1]. 크레딧 만료일이 있는지는 공개 문서에서 찾지 못했습니다(확인 필요). Billing 페이지나 크레딧을 받은 안내 메일에서 확인해 두고, 만료일이 있으면 예산안의 순서를 앞당깁니다.

다음으로 이 프로젝트 전용 워크스페이스를 만듭니다. [Workspaces](https://platform.claude.com/settings/workspaces) 페이지에서 Create workspace를 누르면 되고, 조직 관리자만 만들 수 있습니다 [C4]. 기본 워크스페이스(Default Workspace)에 건 지출 한도에는 Playground와 관리형 에이전트 세션 사용량이 잡히지 않을 수 있습니다 [C4]. 이 사용량도 조직 전체 한도에는 잡히므로, 조직 한도를 반드시 걸고 API 키와 관리형 에이전트는 따로 만든 워크스페이스에 둡니다.

### 4.2 지출 한도와 알림

[Spend limits](https://platform.claude.com/settings/spend-limits) 페이지에서 아래 한도와 알림을 설정합니다 [C4][C6].

- 조직 월 지출 한도: Monthly spend limit의 Set limit에서 정합니다. 크레딧을 넘겨 카드 결제가 되지 않도록 낮게 겁니다. 처음 달은 100달러로 시작합니다.
- 워크스페이스 월 지출 한도: Workspace limits에서 워크스페이스를 고른 뒤 Set limit에서 정합니다. 한 달에 쓸 예산(예: 50달러)을 넣습니다.
- 이메일 알림: 같은 화면의 Email notifications에서 Add notification을 누르고 한도의 50%와 80%에 겁니다.

한도에 닿으면 요청이 HTTP 400 `invalid_request_error`로 거절되며, 오류 메시지에 다시 쓸 수 있는 시점이 나옵니다 [C6]. 배치 작업 중간에 멈출 수 있으므로 큰 배치를 보내기 전에 남은 금액을 확인합니다. 조직의 사용 등급(tier)과 분당 한도는 [Rate limits](https://platform.claude.com/usage/limits) 페이지에서 볼 수 있습니다 [C6]. 가장 낮은 Start 등급도 Sonnet 5.5와 Haiku 5.5가 분당 4,000요청이어서 [C6] 이 프로젝트 규모에서는 문제가 되지 않습니다.

### 4.3 API 키 발급과 보관

[API keys](https://platform.claude.com/settings/keys) 페이지에서 위에서 만든 워크스페이스를 고르고 키를 만들고 [C7], 바로 PC의 환경 변수에 넣습니다. 키에 만료일을 정할 수 있으니 [C7] 짧게 잡고 필요할 때 다시 만듭니다.

```bash
# ~/.bashrc 또는 ~/.zshrc (저장소 밖)
export ANTHROPIC_API_KEY="발급받은 키"
```

키는 저장소에 넣지 않습니다. 저장소의 `.gitignore`에 `.env`를 넣어 두었으므로 `.env` 파일을 쓰더라도 커밋되지 않습니다. 라즈베리파이에는 키를 두지 않습니다. 이 계획의 API 작업은 모두 PC에서 합니다.

### 4.4 사용량 확인

[Usage](https://platform.claude.com/usage) 페이지에서 모델별 토큰 사용량과 캐시 적중률을, [Cost](https://platform.claude.com/cost) 페이지에서 금액을 워크스페이스별로 봅니다 [C5][C6]. 코드에서는 응답의 `usage` 필드(입력, 출력, 캐시 토큰 수)를 기록해 두면 2장의 가정값을 실측값으로 바꿀 수 있습니다.

## 5. 프로젝트 코드에서 호출하기

PC에 공식 Python SDK를 설치합니다 [C7]. 파이5용 `requirements.txt`에는 넣지 않습니다.

```bash
pip install anthropic
```

### 5.1 로그 한 건 분석 (Opus 5.5)

```python
import anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경 변수를 읽습니다

log = open("out/dfc_compile.log", encoding="utf-8").read()
resp = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    output_config={"effort": "medium"},
    messages=[{
        "role": "user",
        "content": "Hailo DFC 컴파일 로그입니다. 지원되지 않은 연산자, 그 연산자가 있는 레이어 이름, "
                   "바꿀 수 있는 대체 연산을 표로 정리해 주세요.\n\n" + log,
    }],
)
print(next(b.text for b in resp.content if b.type == "text"))
print(resp.usage)
```

### 5.2 프레임 검수 배치 (Haiku 5.5)

프레임 이미지와 NPU 검출 결과를 함께 보내고, 답은 JSON 스키마로 받습니다. 결과는 요청 순서와 상관없이 오므로 `custom_id`로 프레임을 찾습니다 [C2].

```python
import base64, json
from pathlib import Path
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

SCHEMA = {
    "type": "object",
    "properties": {
        "missed_objects": {"type": "array", "items": {"type": "string"}},
        "false_detections": {"type": "array", "items": {"type": "integer"}},
        "scene_tags": {"type": "array", "items": {"type": "string"}},
        "needs_human_review": {"type": "boolean"},
    },
    "required": ["missed_objects", "false_detections", "scene_tags", "needs_human_review"],
    "additionalProperties": False,
}

def build_request(img_path: Path, detections: list) -> Request:
    data = base64.standard_b64encode(img_path.read_bytes()).decode()
    prompt = (
        "차량 카메라 프레임과 NPU 검출 결과(번호, 클래스, 박스)입니다.\n"
        f"{json.dumps(detections, ensure_ascii=False)}\n"
        "놓친 물체, 잘못 잡은 검출 번호, 장면 조건 태그를 적고, 판단이 애매하면 needs_human_review를 true로 하세요."
    )
    return Request(
        custom_id=img_path.stem,
        params=MessageCreateParamsNonStreaming(
            model="claude-haiku-5-5",
            max_tokens=2000,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": prompt},
            ]}],
        ),
    )

client = anthropic.Anthropic()
# frames: [(Path, detections), ...] 는 녹화 영상에서 뽑은 프레임과 검출 결과
batch = client.messages.batches.create(requests=[build_request(p, d) for p, d in frames])
print(batch.id)
```

끝난 뒤에는 `client.messages.batches.retrieve(batch.id).processing_status`가 `ended`인지 보고 `client.messages.batches.results(batch.id)`로 결과를 읽습니다 [C2]. 한 배치에 최대 10만 요청, 256MB까지 넣을 수 있습니다 [C2]. `needs_human_review`가 true인 프레임만 Sonnet 5.5로 한 번 더 보내고, 그래도 애매하면 사람이 봅니다.

처음에는 100장으로 시험해서 사람이 직접 본 결과와 얼마나 맞는지, 장당 실제 비용이 얼마인지 잽니다. Haiku의 판정이 사람과 많이 다르면 1차 검수 모델을 Sonnet으로 올리고 예산표의 장수를 그에 맞춰 줄입니다.

### 5.3 연산자 확인 에이전트 (Agent SDK)

Agent SDK는 Hailo DFC와 DX-COM이 설치된 x86-64 리눅스 PC에서 돌립니다. 인증은 같은 `ANTHROPIC_API_KEY`를 씁니다 [C8].

```bash
python -m venv .venv-agent && . .venv-agent/bin/activate
pip install claude-agent-sdk
```

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

TASK = (
    "docs/edge_npu_model_selection.md 3장의 시험을 합니다. out/opcheck/ 아래에 "
    "grid_sample만 있는 모델, ScatterND만 있는 모델, ResNet18+FPN 백본을 ONNX로 만들고 "
    "hailo parser와 DX-COM으로 각각 컴파일하세요. 실패하면 로그에서 원인 연산을 찾아 "
    "대체 연산으로 바꾼 버전도 시험하고, 결과를 out/opcheck/result.md 표로 정리하세요. "
    "저장소의 기존 파일은 고치지 마세요."
)

async def main():
    options = ClaudeAgentOptions(
        model="claude-sonnet-5-5",
        cwd="/home/사용자/fsd_diff_npu",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=60,
        max_budget_usd=5.0,
    )
    async for msg in query(prompt=TASK, options=options):
        if isinstance(msg, ResultMessage):
            print(msg.result)
            print("추정 비용(USD):", msg.total_cost_usd)

asyncio.run(main())
```

`allowed_tools`에 `Bash`가 들어가 에이전트가 PC에서 명령을 직접 실행합니다. 처음에는 별도 사용자 계정이나 컨테이너에서 돌리고, `out/` 아래 결과만 확인한 뒤 필요한 것만 저장소로 옮깁니다. `total_cost_usd`는 추정치이므로 실제 금액은 Console의 Cost 페이지에서 확인합니다 [C9].

### 5.4 관리형 에이전트

관리형 에이전트는 Console에서 에이전트(모델, 시스템 프롬프트, 도구)를 한 번 만들고, 일을 시킬 때마다 세션을 엽니다. 세션마다 Anthropic이 운영하는 컨테이너가 붙어 그 안에서 명령과 파일 작업을 합니다 [C10]. 이 프로젝트에서는 두 가지로 씁니다.

ONNX 분할 시험 에이전트는 네트워크를 패키지 저장소와 GitHub 정도로 제한한 환경에서 돌리고, 세션을 만들 때 지출 상한 5달러를 겁니다. 결과물(분할한 ONNX 구조 요약, BEVPool 비교 결과)은 세션이 끝나기 전에 파일로 받아 둡니다.

NPU 소프트웨어 변경 확인 에이전트는 웹 검색 도구만 켜고, 검색 대상 도메인을 Hailo, DEEPX, Ultralytics 사이트와 GitHub로 제한한 뒤 주 1회 실행합니다. 실행할 때마다 검색 횟수만큼 비용이 붙으므로 시스템 프롬프트에 검색 횟수 상한을 적어 둡니다.

세션 진행 상황은 Console의 워크스페이스 화면에서 세션별로 볼 수 있습니다 [C10].

## 참고 자료

- C1: Anthropic, Pricing. https://platform.claude.com/docs/en/about-claude/pricing
- C2: Anthropic, Batch processing. https://platform.claude.com/docs/en/build-with-claude/batch-processing
- C3: Anthropic, Vision (Image limits and costs). https://platform.claude.com/docs/en/build-with-claude/vision
- C4: Anthropic, Workspaces. https://platform.claude.com/docs/en/manage-claude/workspaces
- C5: Anthropic, Claude Code costs. https://code.claude.com/docs/en/costs
- C6: Anthropic, Rate limits (Spend limits 포함). https://platform.claude.com/docs/en/api/rate-limits
- C7: Anthropic, Get started with Claude. https://platform.claude.com/docs/en/get-started
- C8: Anthropic, Agent SDK overview. https://code.claude.com/docs/en/agent-sdk/overview
- C9: Anthropic, Agent SDK cost tracking. https://code.claude.com/docs/en/agent-sdk/cost-tracking
- C10: Anthropic, Claude Managed Agents overview. https://platform.claude.com/docs/en/managed-agents/overview
