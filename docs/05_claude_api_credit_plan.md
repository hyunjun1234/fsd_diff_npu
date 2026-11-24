# Claude API 크레딧 사용 계획

이 문서는 Claude Console(Anthropic API)의 200달러 조직 크레딧을 이 프로젝트에 어디에, 어떤 모델로, 얼마씩 쓸지 정한 계획과 Console에서 시작하는 방법을 적은 안내서입니다. 구현 단계 이름과 완료 기준은 [라즈베리파이 5, Hailo-8, DX-M1 구현 계획](03_pi5_hailo8_dxm1_implementation_plan.md)을 따릅니다.

## 1. 어디에 쓰는가

이 크레딧은 API(Batch API 포함), Agent SDK, Playground, 관리형 에이전트(Managed Agents)에만 쓸 수 있습니다(크레딧 안내 기준). Claude Code 사용은 대상이 아니므로 계획에 넣지 않았습니다.

이 프로젝트는 Claude Opus 5.5를 기본 모델로 쓰고, Opus로 풀리지 않는 작업만 Claude Fable 5.2로 올립니다. 크레딧의 대부분은 맥북에서 Agent SDK로 돌리는 코드 작업에 씁니다. 파이5 위에서 실시간으로 도는 경로(검출, 세그, BEV 합성, 렌더링)에는 넣지 않습니다.

| 용도 | 쓰는 방식 | 모델 |
|---|---|---|
| 구현 단계 코드 작업 | Agent SDK (맥북) | Opus 5.5, 필요할 때 Fable 5.2 |
| 연산자 확인 자동 반복 | Agent SDK (x86-64 리눅스 PC) | Opus 5.5 |
| 컴파일과 실행 로그 분석, 자료 읽기 | API | Opus 5.5 |
| 자체 영상 검수와 결과 화면 비교 | Batch API | Haiku 5.5, 일부 Opus 5.5 재확인 |

용도별로 구현 계획의 어디에 쓰이고 무엇을 하는지는 다음과 같습니다.

- 구현 단계 코드 작업: 2단계 DX-M1 백엔드(`realityview/detectors.py`), 3단계 쿼드 카메라 소스, 4단계 ONNX 분할과 BEVPool NumPy 구현, 각 단계 측정 스크립트를 에이전트가 저장소 안에서 작성하고 시험합니다. 커밋은 사람이 결과를 보고 직접 합니다.
- 연산자 확인 자동 반복: 4.5절 연산자 확인에 씁니다. 에이전트가 최소 ONNX 모델을 만들고, Hailo DFC와 DX-COM으로 컴파일하고, 로그를 읽고, 실패한 연산을 바꿔 다시 컴파일한 뒤 모델 선정 문서 3장 표에 들어갈 결과를 정리합니다.
- 컴파일과 실행 로그 분석, 자료 읽기: 0단계 드라이버 설치와 2단계 이후 실행 오류 로그, FlashOcc 논문과 DX-RT, HailoRT 문서를 넣고 원인과 필요한 부분을 정리하게 합니다.
- 자체 영상 검수와 결과 화면 비교: 2단계 검출 비교, 3단계 INT8 보정 이미지 고르기와 ID 추적 확인, 4단계 occupancy와 지면 투영 비교, 5단계 의사 라벨 걸러내기에 씁니다. 사람은 표시된 프레임만 다시 봅니다.

DX-COM 내보내기는 x86-64 리눅스에서만 되고([구현 계획](03_pi5_hailo8_dxm1_implementation_plan.md) 3장), Hailo DFC도 리눅스용으로 배포되어 맥에서는 돌리기 어렵습니다(확인 필요). 그래서 맥북의 에이전트는 코드 작성, ONNX 내보내기와 분할, CPU 시험까지 맡고, 컴파일이 필요한 연산자 확인은 컴파일러가 설치된 리눅스 PC에서 같은 방식으로 돌립니다.

픽셀 단위 마스크나 정밀한 3D 박스를 Claude가 직접 만들게 하는 일은 넣지 않았습니다. 도로 마스크 IoU를 잴 20프레임은 지금 계획대로 사람이 칠하고, Claude는 그 앞뒤의 고르기와 검수만 맡깁니다.

## 2. 모델과 가격

가격은 Anthropic 공식 가격표 기준이며 100만 토큰당 달러입니다 [C1]. Batch API로 보내면 입력과 출력 모두 반값이고, 대부분 1시간 안에 끝나며 24시간 안에 끝나지 않은 요청은 청구되지 않습니다 [C1][C2].

| 모델 | 모델 ID | 입력 | 출력 | Batch 입력 | Batch 출력 |
|---|---|---|---|---|---|
| Opus 5.5 | `claude-opus-5-5` | 4 | 20 | 2 | 10 |
| Fable 5.2 | `claude-fable-5-2` | 8 | 40 | 4 | 20 |
| Haiku 5.5 | `claude-haiku-5-5` | 0.10 | 0.50 | 0.05 | 0.25 |

Haiku 5.5 가격은 프롬프트가 10만 토큰 이하일 때 값이고, 넘으면 입력 0.50, 출력 2.50으로 오릅니다 [C1]. 사고(thinking) 토큰은 출력 토큰으로 청구됩니다. Opus 5.5의 effort 기본값은 `medium`이므로 코드 작업에서는 `high`로 지정합니다.

### 2.1 Fable 5.2로 올리는 기준

Fable 5.2는 Opus 5.5보다 토큰 가격이 두 배여서 [C1] 아래 경우에만 씁니다.

- 같은 문제를 Opus 5.5로 두 번 고쳤는데도 완료 기준을 넘지 못할 때. 예를 들어 BEVPool NumPy 구현이 원본 결과와 맞지 않거나, 외부 보정 지면 역투영 오차가 10cm 안으로 들어오지 않는 경우입니다.
- 여러 파일에 걸친 큰 설계 작업. 4단계 FlashOcc를 카메라 4대 구성으로 바꾸는 작업이나 두 NPU 동시 실행 구조가 여기에 해당합니다.
- 몇 시간 동안 혼자 이어서 해야 하는 긴 작업.

Fable 5.2는 사고(adaptive thinking)가 항상 켜져 있고 깊이는 effort로만 조절합니다 [C11]. 그래서 짧은 질문에 쓰면 비용이 크게 늘어나므로, 위 기준에 해당하는 작업에만 씁니다.

### 2.2 프레임 한 장 비용

이미지는 28x28 픽셀 단위로 나뉘고 `ceil(가로/28) x ceil(세로/28)`개의 토큰이 듭니다 [C3]. Camera Module 3 Wide 프레임을 1280x720으로 줄여 보내면 46 x 26 = 1,196토큰입니다.

판정 기준을 명확히 적은 같은 프롬프트로 프레임을 판정했을 때 네 모델의 답이 같았고, 장당 비용은 아래와 같았습니다(이 프로젝트의 Playground 시험 결과, Batch 아님). Batch로 보내면 반값입니다 [C1].

| 모델 | 장당 비용 | Batch 1,000장 |
|---|---|---|
| Fable 5.2 | 0.0088달러 | 약 4.4달러 |
| Opus 5.5 | 0.0049달러 | 약 2.5달러 |
| Sonnet 5.5 | 0.0016달러 | 약 0.8달러 |
| Haiku 5.5 | 0.00022달러 | 약 0.11달러 |

판정이 같았으므로 대량 판정은 Haiku 5.5로 하고, Opus 5.5는 무작위로 고른 일부(약 5%)와 Haiku가 애매하다고 한 프레임만 다시 봅니다. 카메라 4대를 1초에 1장씩 30분 녹화하면 7,200장이고, Haiku Batch로 약 0.8달러입니다.

### 2.3 로그와 문서 한 번 비용

컴파일 로그 5만 토큰을 넣고 5천 토큰 답을 받으면 Opus 5.5 기준 입력 0.20달러, 출력 0.10달러로 약 0.30달러입니다(가격표로 계산 [C1]). 같은 문서에 여러 번 질문할 때는 프롬프트 캐시를 켭니다. Opus 5.5의 캐시 읽기는 100만 토큰당 0.20달러여서 두 번째 질문부터 입력 비용이 약 20분의 1로 줄어듭니다 [C1].

## 3. 200달러 예산안

| 항목 | 금액 | 모델 |
|---|---|---|
| 구현 단계 코드 작업과 연산자 확인 (Agent SDK) | 120달러 | Opus 5.5 |
| 어려운 작업 (Agent SDK) | 30달러 | Fable 5.2 |
| 영상 검수와 결과 화면 비교 (Batch API) | 10달러 | Haiku 5.5, 일부 Opus 5.5 |
| 로그 분석과 자료 읽기 (API) | 10달러 | Opus 5.5 |
| 예비 | 30달러 | 해당 없음 |
| 합계 | 200달러 | 해당 없음 |

Playground 시험 등에 이미 쓴 금액은 Console의 [Cost](https://platform.claude.com/cost) 페이지에서 확인해 예비에서 뺍니다.

에이전트 한 번 실행에 드는 비용은 작업 크기에 따라 차이가 커서 미리 정하지 않았습니다(확인 필요). 참고로 Claude Code 공식 문서에는 기업 사용자 평균이 활동한 날 하루 약 13달러로 나와 있고 [C5], Agent SDK는 Claude Code와 같은 에이전트 구조를 라이브러리로 쓰는 것이어서 [C8] 비슷한 규모로 쓰일 수 있습니다. 이 기준이면 120달러는 Opus로 열흘 남짓 작업할 수 있는 양입니다. 처음 세 번 실행한 비용을 보고 실행당 상한을 다시 정합니다.

Agent SDK에는 실행마다 `max_budget_usd`를 겁니다. 이 값은 한 번의 실행(한 연결) 전체에 걸리는 상한이며, 한 연결 안에서 요청을 여러 번 보내면 결과에 오는 `total_cost_usd`는 그 요청만의 값이 아니라 누적값입니다 [C9]. 상한은 프로그램이 계산한 추정치로 멈추는 방식이라 조금 넘을 수 있으므로 여유를 두고 잡습니다 [C9].

순서는 구현 단계에 맞춥니다. 0단계에서는 로그 분석과 자료 읽기, 4.5절에서는 연산자 확인, 2단계부터 각 단계 코드 작업과 영상 검수를 씁니다. 예비 금액은 5단계 데이터가 늘어날 때나, 관리형 에이전트로 NPU 소프트웨어 릴리스 노트를 주기적으로 확인할 때 씁니다. 관리형 에이전트는 토큰 비용에 더해 세션이 실제로 일하는 시간 1시간당 0.08달러가 붙고, 웹 검색은 1,000번에 10달러이며 [C1], 아직 베타입니다 [C10].

## 4. Console에서 시작하기

### 4.1 크레딧과 워크스페이스

먼저 [Billing](https://platform.claude.com/settings/billing) 페이지에서 200달러 크레딧이 조직에 들어와 있는지 확인합니다 [C1]. 크레딧 만료일이 있는지는 공개 문서에서 찾지 못했습니다(확인 필요). Billing 페이지나 크레딧을 받은 안내 메일에서 확인해 두고, 만료일이 있으면 예산안의 순서를 앞당깁니다.

다음으로 이 프로젝트 전용 워크스페이스를 만듭니다. [Workspaces](https://platform.claude.com/settings/workspaces) 페이지에서 Create workspace를 누르면 되고, 조직 관리자만 만들 수 있습니다 [C4]. 기본 워크스페이스(Default Workspace)에 건 지출 한도에는 Playground와 관리형 에이전트 세션 사용량이 잡히지 않을 수 있습니다 [C4]. 이 사용량도 조직 전체 한도에는 잡히므로, 조직 한도를 반드시 걸고 API 키와 관리형 에이전트는 따로 만든 워크스페이스에 둡니다.

### 4.2 지출 한도와 알림

[Spend limits](https://platform.claude.com/settings/spend-limits) 페이지에서 아래 한도와 알림을 설정합니다 [C4][C6].

- 조직 월 지출 한도: Monthly spend limit의 Set limit에서 정합니다. 크레딧을 넘겨 카드 결제가 되지 않도록 크레딧 금액인 200달러로 겁니다.
- 워크스페이스 월 지출 한도: Workspace limits에서 워크스페이스를 고른 뒤 Set limit에서 정합니다. 조직 한도보다 조금 낮은 180달러를 넣습니다.
- 이메일 알림: 같은 화면의 Email notifications에서 Add notification을 누르고 160달러(조직 한도의 80%)에 겁니다.

한도에 닿으면 요청이 HTTP 400 `invalid_request_error`로 거절되며, 오류 메시지에 다시 쓸 수 있는 시점이 나옵니다 [C6]. 배치 작업 중간에 멈출 수 있으므로 큰 배치를 보내기 전에 남은 금액을 확인합니다. 조직의 사용 등급(tier)과 분당 한도는 [Rate limits](https://platform.claude.com/usage/limits) 페이지에서 볼 수 있습니다 [C6]. 가장 낮은 Start 등급도 Opus 5.5와 Haiku 5.5가 분당 4,000요청이어서 [C6] 이 프로젝트 규모에서는 문제가 되지 않습니다.

### 4.3 API 키 발급과 보관

[API keys](https://platform.claude.com/settings/keys) 페이지에서 위에서 만든 워크스페이스를 고르고 키를 만들고 [C7], 바로 맥북(그리고 연산자 확인용 리눅스 PC)의 환경 변수에 넣습니다. 키에 만료일을 정할 수 있으니 [C7] 짧게 잡고 필요할 때 다시 만듭니다.

```bash
# ~/.bashrc 또는 ~/.zshrc (저장소 밖)
export ANTHROPIC_API_KEY="발급받은 키"
```

키는 저장소에 넣지 않습니다. 저장소의 `.gitignore`에 `.env`를 넣어 두었으므로 `.env` 파일을 쓰더라도 커밋되지 않습니다. 라즈베리파이에는 키를 두지 않습니다. 이 계획의 API 작업은 모두 맥북과 리눅스 PC에서 합니다.

### 4.4 사용량 확인

[Usage](https://platform.claude.com/usage) 페이지에서 모델별 토큰 사용량과 캐시 적중률을, [Cost](https://platform.claude.com/cost) 페이지에서 금액을 워크스페이스별로 봅니다 [C5][C6]. 코드에서는 응답의 `usage` 필드(입력, 출력, 캐시 토큰 수)를 기록해 두면 2장의 가정값을 실측값으로 바꿀 수 있습니다.

## 5. 프로젝트 코드에서 호출하기

맥북에 공식 Python SDK를 설치합니다 [C7]. 파이5용 `requirements.txt`에는 넣지 않습니다.

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

끝난 뒤에는 `client.messages.batches.retrieve(batch.id).processing_status`가 `ended`인지 보고 `client.messages.batches.results(batch.id)`로 결과를 읽습니다 [C2]. 한 배치에 최대 10만 요청, 256MB까지 넣을 수 있습니다 [C2]. `needs_human_review`가 true인 프레임과 무작위로 고른 일부만 Opus 5.5로 한 번 더 보내고, 그래도 애매하면 사람이 봅니다.

처음에는 100장으로 시험해서 사람이 직접 본 결과와 얼마나 맞는지, 장당 실제 비용이 얼마인지 잽니다. Haiku의 판정이 사람과 많이 다르면 1차 검수 모델을 Opus 5.5로 올립니다. 그래도 Batch 1,000장에 약 2.5달러여서 예산 안에 들어옵니다.

### 5.3 Agent SDK 에이전트 (Opus 5.5)

Agent SDK는 Python 3.10 이상에서 `pip install claude-agent-sdk`로 설치하고, 인증은 같은 `ANTHROPIC_API_KEY`를 씁니다. SDK는 `.env` 파일을 읽지 않으므로 키는 셸 환경 변수로 넣습니다 [C12]. 아래는 연산자 확인 작업을 한 번 실행하는 예입니다. 맥북의 코드 작업도 `TASK`와 `cwd`만 바꿔 같은 방식으로 돌립니다.

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
        model="claude-opus-5-5",      # 어려운 작업은 "claude-fable-5-2"
        effort="high",
        cwd="/home/사용자/fsd_diff_npu",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=60,
        max_budget_usd=5.0,           # 이번 실행 전체의 상한(추정치 기준)
    )
    async for msg in query(prompt=TASK, options=options):
        if isinstance(msg, ResultMessage):
            print(msg.result)
            print("추정 비용(USD):", msg.total_cost_usd)

asyncio.run(main())
```

`allowed_tools`에 `Bash`가 들어가 에이전트가 명령을 묻지 않고 직접 실행합니다. `disallowed_tools`에 `Bash(rm -rf *)` 같은 규칙을 넣어도 명령 앞부분 글자가 같을 때만 막히고, `/bin/rm -rf`나 `bash -c '...'` 같은 다른 형태는 막지 못합니다 [C13]. 처음에는 별도 사용자 계정이나 컨테이너에서 돌리고, 결과를 확인한 뒤 필요한 것만 저장소로 옮깁니다. `total_cost_usd`는 추정치이므로 실제 금액은 Console의 Cost 페이지에서 확인합니다 [C9].

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
- C11: Anthropic, What's new in Claude Fable 5.2. https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2
- C12: Anthropic, Agent SDK quickstart. https://code.claude.com/docs/en/agent-sdk/quickstart
- C13: Anthropic, Claude Code permissions. https://code.claude.com/docs/en/permissions
