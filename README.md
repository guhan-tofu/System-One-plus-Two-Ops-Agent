# Sentinel

An operations agent for inbound support work (tickets, alerts, emails). It decides
what to do with each item quickly and cheaply, drafts the reply, takes approved
actions, **checks its own claims before anything goes out**, and hands the item to a
human whenever it isn't confident.

It combines two kinds of model, and keeps code in charge of both:

| | Role | Used for |
|---|---|---|
| **System One: Jev** (official TypeSafe `typesafe-ai/jev` via Vercel AI Gateway) | Fast, typed judgments with probabilities | Triage, model routing, "is this tool call safe?", "is this reply supported?" |
| **System Two: OpenAI** | Generation and reasoning | Drafting the reply and proposing tool calls |
| **Code** | The final authority | State, redaction, policy, running tools, audit |

> Jev decides which path, the LLM does the work, code pulls the trigger.

## How it works

Every item goes through the same pipeline. Each stage is written to an audit log as
it happens, together with the model ID that made the decision.

```
ingest -> build_state -> triage -> [enrich] -> route_model -> generate
       -> guard -> execute -> verify -> decide
```

1. **build_state**: builds the minimal state that models see. Emails, phone numbers
   and card numbers are redacted. Customer IDs, names and contact details stay in code.
2. **triage** (Jev, one request): category, urgency, path (`lookup` / `generate` /
   `human`) and whether the message is abusive. Deterministic thresholds in
   `policies/thresholds.yaml` decide what happens next: low confidence, path
   `human`, or abuse all go to a human.
3. **enrich**: when triage asks for data, read-only tools (e.g. `lookup_customer`) add
   an allowlisted set of account fields to the state.
4. **route_model** (Jev): picks the `fast` or `strong` OpenAI tier. If Jev is unsure,
   policy defaults to `strong`.
5. **generate** (OpenAI, structured output): writes the reply plus any tool calls it
   depends on (e.g. `issue_refund`). The model only *proposes* these calls.
6. **guard**: every proposed call goes through `policies/tools.yaml` first:
   - `read_only` tools run without a guard;
   - `write` tools (e.g. refunds up to £100 in GBP) also need Jev's "safe to run"
     at 0.9 or above;
   - `destructive` tools (e.g. closing an account) **always** need a human;
   - unknown tools, invalid arguments or a missing customer are blocked.

   Jev is only asked when policy allows the call, so it can never override policy. If
   any call needs approval, nothing runs.
7. **execute**: code runs the guarded calls. The customer comes from the work item,
   never from the model. Execution stops at the first failure.
8. **verify**: a failed tool always escalates. Otherwise Jev checks that every claim in
   the draft is supported by the tool results or account data.
9. **decide**: the outcome is one of:
   - `ready_to_send`: verified, and any actions were taken;
   - `awaiting_approval`: held tool calls wait for a human;
   - `escalated`: a human handles it.

   Anything that needs a human lands in the **review queue**.

Failures never fall back to silent defaults. If Jev is down (with retries, a rate
limiter and a circuit breaker in front of it), items either go to the `llm_fallback`
provider (OpenAI emulating Jev) or to the review queue, depending on
`JEV_ON_FAILURE`.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env        # then fill in the values below
uv run pytest               # offline test suite (no real API calls)
```

Minimum `.env`:

```sh
AI_GATEWAY_API_KEY=...        # Vercel AI Gateway key (for Jev)
OPENAI_API_KEY=...
OPENAI_MODEL_FAST=...         # e.g. a small current model, pinned to a dated ID
OPENAI_MODEL_STRONG=...       # e.g. a capable current model
SENTINEL_API_TOKEN=...        # only for `sentinel serve`; a long random value
```

Keys are read only through `config.Settings`. They are never logged, and never go
into prompts, tests or commits. If your environment injects credentials through a
proxy, set the key variables to any non-empty placeholder. `.env.example` lists
every setting with a comment.

## Using it

### Command line

```sh
uv run sentinel probe                          # one live Jev call with fake data; prints the response shape
uv run sentinel run-file samples/ticket.json   # run one item and print the decision trace (--json for all of it)
uv run sentinel eval                           # evaluate Jev providers on labeled data (see Evals)
uv run sentinel serve                          # run the HTTP API on 127.0.0.1:8000
```

`run-file` prints every stage with the model that answered, its confidence,
latency and cost, followed by the outcome, any tool calls and the draft.

### HTTP API

Every endpoint except `/healthz` requires `Authorization: Bearer $SENTINEL_API_TOKEN`.
The server refuses to start without a token.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/items` | Submit a work item. Returns `202`; processing happens in the background. |
| `GET` | `/items/{id}` | Status (`queued`, `processing`, `ready_to_send`, `awaiting_approval`, `escalated`, `failed`, `rejected`, `resolved`) and the latest outcome with its full trace. |
| `GET` | `/review?kind=approval\|review` | Pending review queue. |
| `GET` | `/review/{id}` | One review entry: reasons, draft and proposed tool calls. |
| `POST` | `/review/{id}/approve` | `{"by": "...", "note": "..."}`. For held tool calls: policy is re-checked, the calls run, then the draft is verified. For escalations: marks the item resolved. |
| `POST` | `/review/{id}/reject` | Nothing runs; the item is marked rejected. |
| `GET` | `/healthz` | Liveness check. |

```sh
curl -X POST localhost:8000/items \
  -H "Authorization: Bearer $SENTINEL_API_TOKEN" -H 'content-type: application/json' \
  -d @samples/ticket.json
```

A work item looks like:

```json
{"id": "t-1", "source": "ticket", "subject": "Charged twice",
 "body": "I was charged twice this month...", "customer_id": "cus_1001"}
```

`POST /items` is rate limited (`API_MAX_ITEMS_PER_MINUTE`, default 60) and item bodies
are capped at 20,000 characters.

### Docker

```sh
docker build -t sentinel .
docker run -p 8000:8000 --env-file .env -v sentinel-data:/data sentinel
```

The container runs as a non-root user and stores its SQLite database in `/data`.
Secrets are only supplied at runtime.

## Configuration

**Policies** (YAML, deterministic, always win over model output):

| File | Contents |
|---|---|
| `policies/thresholds.yaml` | Confidence cutoffs: triage category/path, abuse, model routing, guard, verify |
| `policies/tools.yaml` | Each tool's risk level and approval rules (e.g. refund auto-limit £100, GBP only) |
| `policies/model_tiers.yaml` | Token prices per model for cost accounting (empty until you add current prices) |

**Main settings** (environment or `.env`):

| Setting | Default | Purpose |
|---|---|---|
| `JEV_PROVIDER` | `vercel` | `vercel` (official Jev), `llm_fallback` (OpenAI emulation), or `typesafe` (direct API stub) |
| `JEV_GATEWAY_MODEL` | `typesafe-ai/jev` | Jev model on the gateway |
| `JEV_ON_FAILURE` | `human` | When Jev fails: `human` (escalate) or `llm_fallback` (retry on OpenAI emulation) |
| `JEV_MAX_RPS` | `2` | Client-side cap on Jev requests per second (the gateway free tier throttles at about 3) |
| `JEV_BREAKER_FAILURES` / `JEV_BREAKER_RESET_S` | `5` / `30` | Circuit breaker: failures before failing fast, and the cool-down |
| `OPENAI_MODEL_FAST` / `OPENAI_MODEL_STRONG` | none | Models behind the two tiers |
| `DATABASE_URL` | `sqlite:///./sentinel.db` | Audit log, review queue and items |

## Evals

`evals/datasets/triage.jsonl` holds 67 hand-labeled fake tickets (category, urgency,
path, abusive). This command:

```sh
uv run sentinel eval -p vercel -p llm_fallback --rate 1
```

compares providers side by side and writes a markdown report to `evals/reports/`.
The report covers:
- accuracy, Brier score, and 10-bin calibration with ECE per question;
- p50/p95 latency, cost per item and errors per provider;
- what the triage policy would do: how many items proceed, whether items that need
  a human reach one, and how many are escalated unnecessarily.

Tune thresholds from these reports, not by feel. `--rate` keeps under the gateway's
free-tier limit.

## Safety properties (and the tests that hold them)

- **Destructive tools never run without human approval**, even when Jev scores them
  as safe (`tests/integration/test_pipeline_actions.py`).
- **A reply that claims success after a failed tool is escalated, never sent**
  (same file).
- **Approval can't unblock a call that policy blocks** (`tests/integration/test_api.py`).
- **Prompt injection:** even with a fully hijacked drafting model, instructions in a
  ticket can't do any of the following (`tests/integration/test_prompt_injection.py`):
  close an account; refund above the limit; refund another customer's charge; smuggle
  a `customer_id` argument; or get an unsupported reply sent. Live tests confirm that
  injected text doesn't change Jev's routing (`test_live_injection.py`).
- **Every decision is audited** with the model ID, the gateway's generation ID,
  latency and cost. If an audit write fails, the pipeline stops.

## Development

```sh
uv run pytest                                # offline tests (HTTP mocked with respx)
uv run pytest --run-live -m live             # real Jev / OpenAI calls (needs keys)
uv run ruff check . && uv run mypy src       # lint and strict type check
```

Conventions (from `CLAUDE.md`):
- new behavior needs tests;
- all external calls go through `jev/` providers or `llm/openai_client.py`;
- one judgment per Jev question;
- deterministic policy beats model output.

`PLAN.md` has the full design and the build phases.

### Layout

```
src/sentinel/
  agent/     pipeline stages: state, triage, routing, generate, guard, verify, pipeline
  jev/       Jev providers (vercel, llm_fallback), models, questions, confidence, resilience
  llm/       OpenAI client (tiered generation, structured output) and pricing
  tools/     tool registry and mock built-in tools (lookup_customer, issue_refund, close_account)
  policy/    YAML policy engine (thresholds, tool rules)
  storage/   SQLAlchemy: audit log, review queue, items
  api/       FastAPI app and service layer
  evals/     dataset loader, metrics, runner, report
  cli.py     `sentinel` command
policies/    thresholds.yaml, tools.yaml, model_tiers.yaml
samples/     example work items (fake data)
evals/       datasets/ (committed) and reports/ (git-ignored)
```

## Current limitations

- **Replies aren't sent anywhere yet:** `ready_to_send` is the final state.
- **Tools are mocks** against an in-memory store. Real integrations would replace
  `tools/builtin/`.
- **Background processing uses in-process tasks**, so items being processed are not
  resumed after a restart.
- **No product-knowledge source:** how-to questions usually escalate, because verify
  has nothing to check product claims against.
- **The gateway returns no versioned Jev model ID**, so audits record the requested
  model with `model_verified=false`, plus the generation ID.
