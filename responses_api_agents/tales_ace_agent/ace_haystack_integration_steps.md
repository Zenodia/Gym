# ACE + Haystack Memory Agent — NeMo Gym Integration

Integrates `ace_relay_trace.py` (NeMo Relay instrumented ACE agent, `/home/ubuntu/dspy/`) into
NeMo Gym as a first-class agent harness. The agent owns the full DSPy + Haystack structured memory
loop and evolving ACE playbook, calling the TALES resources server via `ServerClient` for
environment interaction.

```
responses_api_agents/tales_ace_agent/    ← this agent
resources_servers/tales/                 ← existing TALES env server (ALFWorld, TextWorld, …)
```

---

## Architecture Overview

```
ng_run / ng_test
│
├── TALESResourcesServer  (resources_servers/tales/app.py)
│     /reset → gym.make("tales/alfworld") + env.reset()
│     /step  → env.step(action) → (obs, reward, done, info)
│     expose_admissible_commands: true  ← REQUIRED for ACE
│
└── TalesAceAgent  (responses_api_agents/tales_ace_agent/app.py)
      /run  → full episode loop
        ├── asyncio.to_thread(_sync_memory_step)   DSPy MemoryUpdate + Haystack embed/recall
        ├── asyncio.to_thread(_reason_traced)      NvidiaChatGenerator + nemo_relay spans
        ├── POST /step  →  ServerClient             env transition
        └── asyncio.to_thread(_sync_reflect)       DSPy Reflect + playbook curate
```

**No model server.** The agent calls NVIDIA NIM directly. `/v1/responses` returns HTTP 501.

**Playbook persistence.** `self._playbook` (a `Playbook` instance) lives on the agent object and
survives across `run()` calls, enabling cross-episode strategy learning.

### Component origin

| File | Origin | Note |
|---|---|---|
| `app.py` | Ported inline from `ace_relay_trace.py`, `rung2_haystack_memory.py`, `rung3_ace_alfworld.py` | No cross-module imports — avoids module-level `load_dotenv` / `dspy.configure` side effects |
| `configs/tales_ace_agent.yaml` | New | Wires TALES server (`expose_admissible_commands: true`) + this agent |
| `data/example.jsonl` | New | 3 ALFWorld tasks pointing to `tales_ace_agent` |
| `requirements.txt` | New | Pins dspy, haystack, nvidia-haystack, nemo_relay wheel, tale-suite |
| `tests/test_app.py` | New | 34 offline unit + integration tests; all heavy deps mocked |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.12 | Required by NeMo Gym |
| `uv` | Package manager used by `ng_test` / `ng_run` |
| Java 11+ | Required by some tale-suite environments (e.g. Jericho) |
| `NVIDIA_API_KEY` | NIM API key for all LLM + embedding calls |
| NeMo Relay wheel | `nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl` |

---

## Step-by-Step Environment Setup

### 1. Clone / navigate to the Gym repository

```bash
cd /home/ubuntu/Gym
```

### 2. Create and activate the main Gym venv

```bash
uv venv                          # creates .venv/
uv sync --extra dev --group docs
source .venv/bin/activate
```

Verify `ng_test` is available:

```bash
which ng_test   # should print /home/ubuntu/Gym/.venv/bin/ng_test
```

> **Note:** `ng_test` / `ng_run` live inside the venv. Always activate the venv first
> or prefix commands with `.venv/bin/`.

### 3. Install pre-commit hooks

```bash
pre-commit install
```

> First run may fail as hooks auto-modify files (ruff format, add-verified-flag). Stage the
> changes and commit again.

### 4. Install Java (required for Jericho / some tale-suite envs)

```bash
sudo apt-get install -y default-jdk
java -version   # verify
```

### 5. Set up NeMo Relay source and wheel

The `nemo_relay` wheel is a **path-based distribution**: it installs only a `.pth` file that
adds `/home/ubuntu/NeMo-Relay/python` to `sys.path`. The actual Python source must exist at
that path — the wheel itself contains no Python files.

**Step 5a — Create the source directory the .pth expects:**
```bash
mkdir -p /home/ubuntu/NeMo-Relay/python
cp -r /home/ubuntu/dspy/.venv/lib/python3.12/site-packages/nemo_relay \
      /home/ubuntu/NeMo-Relay/python/
```

Verify:
```bash
python3 -c "
import sys
sys.path.insert(0, '/home/ubuntu/NeMo-Relay/python')
import nemo_relay
print('nemo_relay OK')
"
```

**Step 5b — Copy the wheel to a stable path** (away from volatile `/ephemeral/cache/`):
```bash
cp /ephemeral/cache/uv/sdists-v9/editable/ffca03370d02b0c8/JZHaYRC7MMdzzz7r/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl \
   /home/ubuntu/Gym/responses_api_agents/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
```

`requirements.txt` already references this stable path:
```
../nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
```

> **Why both steps?** `ng_run` / `ng_test` create isolated venvs and run `uv pip install -r requirements.txt`
> which installs the wheel (the `.pth` file) but cannot install the actual Python source —
> that must be at `/home/ubuntu/NeMo-Relay/python` before any venv is created.
>
> **Root cause:** The wheel was built as a development path stub against a local checkout at
> `/home/ubuntu/NeMo-Relay/`. When that checkout is absent (e.g. fresh machine, after reboot),
> `import nemo_relay` fails with `ModuleNotFoundError` even though dist-info shows it as installed.

### 6. Set the NVIDIA API key

The agent reads `NVIDIA_API_KEY` from the environment or from a `.env` file in its directory.

**Option A — `.env` file (recommended for local dev):**
```bash
cp /home/ubuntu/dspy/.env /home/ubuntu/Gym/responses_api_agents/tales_ace_agent/.env
# or create manually:
echo "NVIDIA_API_KEY=nvapi-<your-key>" > responses_api_agents/tales_ace_agent/.env
```

> **Security:** Never commit `.env`. Add it to `.gitignore` if not already excluded.

**Option B — shell env var:**
```bash
export NVIDIA_API_KEY=nvapi-<your-key>
```

### 7. Verify the wheel and rung scripts exist

```bash
ls responses_api_agents/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
ls responses_api_agents/tales_ace_agent/.env
```

---

## Running the Tests

`ng_test` creates an isolated venv for the agent, installs `requirements.txt`, and runs `pytest`.
All 34 tests run offline (heavy deps mocked — no API calls, no game engine needed).

```bash
source .venv/bin/activate
ng_test "+entrypoint=responses_api_agents/tales_ace_agent"
```

Expected output:

```
collected 34 items
...
============================== 34 passed in 3.23s ==============================
```

To run tests directly inside the agent venv (faster iteration after first setup):

```bash
cd responses_api_agents/tales_ace_agent
source .venv/bin/activate
pytest -v
```

---

## Running a Live Episode

Both servers must be running simultaneously. Use two terminals or a `ng_run` config that starts both.

### Terminal 1 — TALES resources server

```bash
source .venv/bin/activate
ng_run "+config_paths=[responses_api_agents/tales_ace_agent/configs/tales_ace_agent.yaml]" \
       "+servers=[tales]"
```

> The `tales_ace_agent.yaml` config sets `expose_admissible_commands: true` and `max_episode_steps: 50`.
> This overrides the default `tales.yaml` (which has `expose_admissible_commands: false`).

### Terminal 2 — TalesAceAgent server

```bash
source .venv/bin/activate
ng_run "+config_paths=[responses_api_agents/tales_ace_agent/configs/tales_ace_agent.yaml]" \
       "+servers=[tales_ace_agent]"
```

### Trigger a run (Terminal 3)

```bash
source .venv/bin/activate
ng_run "+config_paths=[responses_api_agents/tales_ace_agent/configs/tales_ace_agent.yaml]" \
       "+run_dataset=example"
```

Or POST directly:

```bash
curl -X POST http://localhost:<agent_port>/run \
  -H "Content-Type: application/json" \
  -d '{
    "framework": "alfworld",
    "task_no": 0,
    "split": "train",
    "seed": 1236,
    "responses_create_params": {"input": []}
  }'
```

### Stopping all servers

Kill all `ng_run`, server, and Ray processes:

```bash
pkill -f "ng_run\|tales_ace_agent\|tales.*app\.py\|uvicorn.*tales\|ray"
sleep 2
pkill -9 -f "ng_run\|tales_ace_agent\|tales.*app\.py\|uvicorn.*tales\|ray" 2>/dev/null
```

Verify clean:

```bash
pgrep -a -f "ng_run\|tales_ace\|uvicorn.*tales\|ray" || echo "all clear"
```

---

## Configuration Reference

### `configs/tales_ace_agent.yaml`

```yaml
tales:
  resources_servers:
    tales:
      entrypoint: app.py
      expose_admissible_commands: true   # REQUIRED — ACE reads admissible_commands each step
      framework: alfworld
      task_no: 0
      seed: 0
      split: train
      max_episode_steps: 50

tales_ace_agent:
  responses_api_agents:
    tales_ace_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: tales
      max_steps: 50                      # episode step cap; truncated=True if reached
      trace_dir: traces                  # ATIF + ATOF output directory
      datasets:
      - name: example
        type: example
        jsonl_fpath: responses_api_agents/tales_ace_agent/data/example.jsonl
        num_repeats: 1
        license: Apache 2.0
```

Key config fields:

| Field | Default | Effect |
|---|---|---|
| `expose_admissible_commands` | **must be `true`** | TALES server strips admissible_commands from info when false; agent cannot function |
| `max_steps` | `50` | Episode step cap. Matches `max_episode_steps` in server config |
| `trace_dir` | `traces` | Root for trace output. Subdirs `atif/` and `atof/` created on startup |

---

## Models Used

All calls go to NVIDIA NIM (`https://integrate.api.nvidia.com/v1`).

| Role | Model | Used for |
|---|---|---|
| Reasoner | `nvidia/nemotron-3-ultra-550b-a55b` | Tool-call action selection per step (Haystack NvidiaChatGenerator) |
| Summarizer / Reflector | `nvidia/llama-3.3-nemotron-super-49b-v1` | DSPy MemoryUpdate + Reflect (DSPy LM) |
| Embedder | `nvidia/nv-embedqa-e5-v5` | Document + query embedding for Haystack InMemoryDocumentStore |

---

## NeMo Relay Tracing

The agent emits structured trace events via `nemo_relay` v0.4.0.

### Output files

```
traces/
  atif/
    episode-<uuid8>.json     # one structured trajectory per episode (ATIF v1.7)
  atof/
    events.jsonl             # append-mode NDJSON event stream across all runs
```

### ATOF — raw event stream

Registered once in `model_post_init`. Appends every lifecycle event to
`traces/atof/events.jsonl`. Use for grep-based debugging, dashboards, and
timeline reconstruction.

### ATIF — per-episode trajectory

Created, registered, exported, and deregistered per `run()` call. Each file
is a self-contained replay: scope hierarchy, mark events, and LLM call spans.

### Traced events

**Episode scope events**

| Event | Fired | Key fields |
|---|---|---|
| `episode_goal` | start of episode | `goal`, `common_sense_hint`, `framework`, `task_no`, `split` |
| `playbook_injected` | after goal | `bullet_count`, `playbook` (rendered text) |
| `agent_step` | after every `env.step()` | `step`, `observation`, `action_taken`, `progress`, `plan`, `recalled_facts`, `dead_ends`, `result_obs`, `reward`, `is_done` |
| `episode_outcome` | end of episode | `label`, `goal`, `outcome`, `steps`, `total_reward`, `trajectory_tail` |
| `reflection_summary` | after reflector | `goal`, `outcome`, `steps_taken`, `last_observation`, `helpful_strategy_ids`, `harmful_strategy_ids`, `next_best_strategies`, `playbook_size_before`, `current_playbook` |
| `playbook_updated` | after curate | `playbook_size_after`, `bullets_delta`, `playbook_snapshot` |

**LLM lifecycle events** (via `nemo_relay.llm.call` / `call_end`)

| Provider | Fired when |
|---|---|
| `memory-updater` | Before/after every DSPy `MemoryUpdate` predict call |
| `reasoner` | Before/after every `NvidiaChatGenerator.run()` attempt |
| `reflector` | Before/after every DSPy `Reflect` predict attempt |

**RPM / failure events**

| Event | Key fields |
|---|---|
| `llm_rate_limit_hit` | `provider`, `model`, `step`, `attempt`, `wait_seconds`, `error` |
| `reasoner_fallback` | `step`, `fallback_action`, `last_error` |

---

## Rate Limiting

Two layers run together:

| Layer | Mechanism | Where |
|---|---|---|
| **Proactive** | `_throttle()` — sliding 28-RPM window, fires before each NVIDIA API call | `_sync_memory_step`, `_reason_traced`, `_sync_reflect` |
| **Reactive** | RPM retry loop — catches 429 / "rate limit" / "too many requests" / "rpm" in exception string | `_reason_traced`, `_sync_reflect` |

Retry parameters:

```python
_RPM_WAIT_BASE    = 15.0   # seconds (conservative floor)
_RPM_MAX_RETRIES  = 5
_RPM_WAIT_CAP     = 60.0   # doubles each attempt up to this cap
```

On `_reason_traced` exhaustion: heuristic fallback action chosen (prefers unexplored `go to` /
`open` actions); `reasoner_fallback` event fired. On `_sync_reflect` exhaustion: `None` returned;
playbook update skipped for that episode.

---

## ACE Playbook — Cross-Episode Learning

The `Playbook` object lives on the agent instance (`self._playbook`) and persists across all
`run()` calls in a session.

**Per-episode cycle:**

1. **Inject** — rendered playbook prepended to reasoner prompt before each step.
2. **Reflect** — after episode ends, `Reflect` DSPy signature extracts:
   - `helpful_ids`: strategy IDs that helped this episode
   - `harmful_ids`: strategy IDs that misled / wasted steps
   - `new_strategies`: up to 3 new general imperative sentences
3. **Curate** (`_curate`) — applies ops deterministically:
   - Bumps `helpful` / `harmful` counters on existing items
   - Adds new strategies (deduped by Jaccard similarity > 0.6; max cap 12)
   - Prunes items where `harmful - helpful >= 2` and `helpful == 0`

**Playbook render format (injected into reasoner):**
```
[1] Always check the target container before searching nearby locations.
[2] For multi-object tasks, track placed count to know when task is done.
```

---

## Memory System (DSPy + Haystack)

Per-episode, per-step cycle:

1. **MemoryUpdate** (DSPy `Predict`) — extracts `new_fact`, `dead_end`, `progress`, `plan` from current observation.
2. **Grounding** — `ground_fact()` rejects facts whose named objects don't appear verbatim in the observation.
3. **Embed + store** — grounded facts embedded via `NvidiaDocumentEmbedder` and written to `InMemoryDocumentStore`.
4. **Recall** — current goal + plan + observation embedded via `NvidiaTextEmbedder`; top-10 nearest facts retrieved via `InMemoryEmbeddingRetriever`.
5. **Dead-end tracking** — verified noops (`is_noop()`) added to `mem.dead_ends`; filtered from `offered` actions before reasoner call.

---

## Dataset Format

`data/example.jsonl` — one JSON object per line:

```jsonc
{
  "framework": "alfworld",       // alfworld | textworld | textworld_express | scienceworld | jericho
  "task_no": 0,                  // index into framework's train/test task list
  "split": "train",              // "train" or "test"
  "seed": 1236,
  "responses_create_params": {
    "input": [{"role": "system", "content": "You are an ALFWorld agent. ..."}]
  },
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "tales_ace_agent"
  }
}
```

---

## File Tree

```
responses_api_agents/tales_ace_agent/
├── __init__.py
├── app.py                         # ~460 lines — full agent implementation
├── ace_haystack_integration_steps.md  # this file
├── configs/
│   └── tales_ace_agent.yaml       # Hydra config (TALES server + agent)
├── data/
│   └── example.jsonl              # 3 ALFWorld example tasks
├── requirements.txt               # dspy, haystack, nvidia-haystack, nemo_relay wheel
├── tests/
│   ├── __init__.py
│   └── test_app.py                # 34 unit + integration tests (offline, all mocked)
└── .env                           # NVIDIA_API_KEY  ← DO NOT COMMIT

responses_api_agents/
└── nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl   # stable wheel copy
```

---

## Common Issues

### `ng_test: command not found`

The Gym venv is not activated. Fix:

```bash
source /home/ubuntu/Gym/.venv/bin/activate
```

### `pre-commit: command not found`

Same cause. Pre-commit is installed inside the venv, not system-wide.

### `ModuleNotFoundError: No module named 'nemo_relay'`

Two distinct causes — check both:

**Cause A — wheel not at stable path:**
```bash
ls /home/ubuntu/Gym/responses_api_agents/nemo_relay-0.4.0-cp311-abi3-linux_x86_64.whl
```
If absent, copy from the ephemeral cache (see Step 5b above).

**Cause B — `.pth` source directory missing** (most common after reboot or on a fresh machine):
```bash
ls /home/ubuntu/NeMo-Relay/python/nemo_relay/__init__.py
```
If absent, run Step 5a above. The wheel only installs a `.pth` file pointing to
`/home/ubuntu/NeMo-Relay/python`; that directory must contain the actual source.

**Quick verify after fix:**
```bash
source responses_api_agents/tales_ace_agent/.venv/bin/activate
python -c "import nemo_relay; print('OK')"
```
If still failing: delete the agent venv and let `ng_run` recreate it:
```bash
rm -rf responses_api_agents/tales_ace_agent/.venv
```

### `admissible_commands` empty / agent always picks `look`

TALES server running with default `expose_admissible_commands: false`. Must use
`tales_ace_agent.yaml` config which sets it to `true`.

### 429 / rate limit errors

Expected behavior. The reactive retry loop handles them automatically (fires
`llm_rate_limit_hit` event, waits up to 60s, retries up to 5 times). If they
persist, lower the `_RPM_CAP` constant in `app.py` (line ~68) below 28.

### `load_dotenv` not loading key

`.env` must be in `responses_api_agents/tales_ace_agent/`. The agent loads it
relative to `__file__`, not the working directory.

---

## Source Files (original, pre-integration)

| File | Location | Role |
|---|---|---|
| `ace_relay_trace.py` | `/home/ubuntu/dspy/` | Entry point — NeMo Relay instrumented ACE agent; source of `_LLMSpan`, `_reason_traced`, `TracedMemoryAgent`, `run_episode_traced` |
| `rung2_haystack_memory.py` | `/home/ubuntu/dspy/` | DSPy + Haystack memory agent — `StructuredMemoryAgent`, `MemoryState`, `MemoryUpdate`, helpers |
| `rung3_ace_alfworld.py` | `/home/ubuntu/dspy/` | ACE optimizer layer — `Playbook`, `Reflect`, `curate`, `run_episode` |

All three are ported **inline** into `app.py`. The originals are not imported because both
`rung2_haystack_memory.py` and `rung3_ace_alfworld.py` have module-level side effects
(`load_dotenv("/home/ubuntu/dspy/.env")` with hard-coded absolute path, `dspy.configure(lm=...)`)
that would fire at import time and break the Gym server process.
