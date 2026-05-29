# Design Doc — SambaNova RDU Benchmarking (Phase 0: SambaStack Hosted Endpoint)

**Status:** Proposed
**Author:** andy.chen@sambanovasystems.com
**Date:** 2026-05-28
**Scope:** Phase 0 — wire SambaNova RDU into InferenceX as an OpenAI-compatible **remote endpoint** backend, using the **SambaStack hosted** deployment, with the API credentials supplied via the runner host environment (**Option B**).

---

## 1. Background

InferenceX benchmarks LLM inference *throughput and efficiency per accelerator* across hardware (NVIDIA B200/B300/H100/H200/GB200, AMD MI300X/MI325X/MI355X) and software stacks. Every existing runner follows one pattern:

```
salloc/enroot/docker  →  WE launch the inference server  →  benchmark localhost  →  read GPU power via nvidia-smi / amd-smi
```

We want to add **SambaNova RDU**. The long-term goal is **chip-comparable** results — per-RDU throughput and tokens-per-Megawatt, directly comparable to B200/MI355X. This doc covers only **Phase 0**, which validates the end-to-end pipeline against the hosted endpoint and produces endpoint-level metrics. Per-RDU normalization and power (the two headline metrics) are explicitly **out of scope for Phase 0** and tracked in §10–§11.

## 2. What SambaStack Hosted Is

A remote, **OpenAI-compatible HTTPS inference endpoint** that SambaNova operates. We do **not** launch a server.

- **Endpoint:** `https://<API_DOMAIN>/v1/chat/completions` (`<API_DOMAIN>` + API key are admin-provided per deployment; *not* the public `api.sambanova.ai`, which is SambaCloud).
- **Auth:** `Authorization: Bearer $SAMBANOVA_API_KEY`.
- **OpenAI-compatible:** `/v1/chat/completions`, `/v1/responses`, streaming with `stream_options.include_usage`.
- **Rich `usage` object** (per response): `prompt_tokens`, `completion_tokens`, `total_tokens`, `time_to_first_token`, `total_latency`, `completion_tokens_per_sec`, `start_time`/`end_time`.
- **Models are fixed "bundles"** declared in `sambastack.yaml`. The bundle template bakes in **sequence/context length, batch size, RDU count, tensor parallelism, and precision** — not customer-tunable without SambaNova support. One bundle per node.
  - First target: **`gpt-oss-120b`** (InferenceX prefix `gptoss`), bundle context 8/32/64/128K, **batch size 2–8**.

Docs: <https://docs.sambanova.ai/docs/en/v1.1.1/sambastack/getting-started/hosted>

## 3. Why This Doesn't Fit the Existing Model

| InferenceX assumption | Hosted RDU reality |
|---|---|
| `runs-on:` a GPU runner inside the cluster | Runner needs only **network + the API key** — no GPU, not even inside SambaNova |
| `launch_<type>.sh` does `salloc`→`enroot`→serve→exec recipe | Launcher just runs the benchmark client against a remote URL |
| `tp` / `ep` / `dp-attn` sweep knobs | **Inert** — parallelism is fixed in the bundle template |
| Concurrency swept into the thousands | Bundle **max batch size 2–8**; above that we measure SambaNova's queueing, not steady-state |
| `tput_per_gpu = total_tput / TP` (`process_result.py:120`) | We don't know the true RDU count → divisor is an **assumption** |
| `aggregate_power.py` → tokens/MW | **No power signal** (no SMI on/near host) → power keys silently omitted |

**Good news:** the benchmark client already supports remote endpoints. `benchmark_serving.py` accepts `--base-url` (`api_url = base_url + endpoint`) and `backend_request_func.py` has an `openai-chat` backend that POSTs to a `…/chat/completions` URL with `Authorization: Bearer ${OPENAI_API_KEY}`. So the client is reusable as-is; the integration work is plumbing (a runner type, a launcher, a recipe, a config entry) — not a new benchmark client.

## 4. Design (Option B — credentials via runner host env)

### 4.1 Credential handling

The `rdu` runner is a box we control, so we bake the credentials into its **host environment** (mirrors how cluster credentials are ambient on existing runner hosts):

- `SAMBANOVA_API_KEY` and `API_DOMAIN` are set in the runner's environment (e.g. the Actions runner `.env`).
- The launcher reads them directly; **no GitHub Actions secret is added, and `benchmark-tmpl.yml` is not modified.**

> **Alternative (Option A, not chosen):** store them as GitHub Actions secrets and add one line to the `benchmark-tmpl.yml` env block to inject them. Rejected for Phase 0 to keep the blast radius zero on the shared template. Revisit if the runner becomes shared/multi-tenant or if secret rotation via GitHub is preferred.

### 4.2 Components

1. **Runner type** — new `rdu:` key in `.github/configs/runners.yaml` listing node `rdu_0`.
2. **Self-hosted runner** — a network-only box registered as **name `rdu_0`, label `rdu`**, with the two env vars set. (The dispatch is name-based: `bash ./runners/launch_${RUNNER_NAME%%_*}.sh`, so the name prefix `rdu` must match the launcher.)
3. **Launcher** — `runners/launch_rdu.sh`: no `salloc`/`enroot`; exports `OPENAI_API_KEY` + `BASE_URL` and execs the recipe.
4. **Recipe** — `benchmarks/single_node/gptoss_<precision>_rdu.sh`: calls `benchmark_serving.py` directly against the remote endpoint (no server launch, no GPU monitor), saving the result JSON in the schema `process_result.py` expects.
5. **Master-config entry** — `gptoss-<precision>-rdu-sambastack` in `.github/configs/nvidia-master.yaml`, `runner: rdu`, `tp:` set to the (to-be-confirmed) RDU count, concurrency capped to the bundle batch size.
6. **Trigger** — append a `perf-changelog.yaml` entry.

### 4.3 Why call `benchmark_serving.py` directly (not `run_benchmark_serving`)

`run_benchmark_serving` in `benchmarks/benchmark_lib.sh:356` hardcodes `--base-url "http://0.0.0.0:$port"` (assumes a local server). Rather than modify a helper shared by every GPU recipe, the RDU recipe invokes `benchmark_serving.py` directly with `--base-url "$BASE_URL"`. Because both paths use `benchmark_serving.py --save-result`, the saved JSON schema (`total_token_throughput`, `output_throughput`, TTFT/TPOT, …) is identical, so `process_result.py` works unchanged.

> **Alternative:** add an optional `--base-url` override to `run_benchmark_serving` so the RDU recipe matches GPU recipes. Lower duplication, but touches shared code used by all recipes. Deferred.

## 5. Draft Artifacts

> Illustrative drafts. Placeholders (`<…>`) — notably the RDU count and precision label — must be confirmed with SambaNova before publishing results.

**`.github/configs/runners.yaml`** (append):
```yaml
rdu:
  - 'rdu_0'   # network-only host; SAMBANOVA_API_KEY + API_DOMAIN in its env
```

**`runners/launch_rdu.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail

# Option B: credentials come from the runner host environment.
: "${SAMBANOVA_API_KEY:?must be set in the rdu runner host environment}"
: "${API_DOMAIN:?must be set in the rdu runner host environment}"

MODEL_CODE="${EXP_NAME%%_*}"
BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_rdu.sh"

# benchmark_serving's openai-chat backend reads OPENAI_API_KEY.
export OPENAI_API_KEY="$SAMBANOVA_API_KEY"
export BASE_URL="https://${API_DOMAIN}/v1"

set -x
bash "$BENCH_SCRIPT"
```

**`benchmarks/single_node/gptoss_<precision>_rdu.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail

# Remote endpoint: no server to launch, no GPU monitor. Health-check only.
for i in $(seq 1 30); do
  if curl -sf "${BASE_URL}/models" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/dev/null 2>&1 \
     || curl -sf -X POST "${BASE_URL}/chat/completions" \
          -H "Authorization: Bearer ${OPENAI_API_KEY}" -H "Content-Type: application/json" \
          -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" >/dev/null 2>&1; then
    echo "endpoint reachable"; break
  fi
  echo "waiting for endpoint... ($i)"; sleep 2
done

cd utils/bench_serving
python3 benchmark_serving.py \
  --backend openai-chat \
  --base-url "$BASE_URL" \
  --endpoint /chat/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --max-concurrency "$CONC" \
  --num-prompts "$(( CONC * 8 ))" \
  --use-chat-template \
  --save-result \
  --result-filename "${GITHUB_WORKSPACE}/${RESULT_FILENAME}.json"
```
> Match exact flag names to this repo's `benchmark_serving.py` before merging.

**`.github/configs/nvidia-master.yaml`** (or a new vendor master file), append:
```yaml
gptoss-<precision>-rdu-sambastack:
  image: "n/a-hosted-endpoint"   # not used; serving is remote/managed
  model: gpt-oss-120b            # the bundle's API model id
  model-prefix: gptoss
  runner: rdu
  precision: <precision>         # informational; SambaNova-managed. CONFIRM.
  framework: sambastack
  multinode: false
  disagg: false
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      # batch size 2-8 ceiling: keep concurrency in-bundle, else we measure queueing.
      - { tp: <RDU_COUNT>, conc-start: 1, conc-end: 8 }   # CONFIRM RDU_COUNT
```

**`perf-changelog.yaml`** (append at the END, preserve whitespace, end with newline):
```yaml
- config-keys:
    - gptoss-<precision>-rdu-sambastack
  description:
    - "Phase 0: add gpt-oss-120b on SambaNova RDU via SambaStack hosted endpoint (endpoint-level metrics; per-RDU/power pending)"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

## 6. How to Run

**One-time setup**
1. Deploy `gpt-oss-120b` on SambaStack; confirm `curl https://$API_DOMAIN/v1/chat/completions …` works **from the runner box**.
2. Register the self-hosted runner: name `rdu_0`, label `rdu`, with `SAMBANOVA_API_KEY` + `API_DOMAIN` in its env.
3. Land the artifacts in §5 on a branch.

**Smoke test (no changelog touch)** — `e2e-tests.yml` dispatch:
```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=<rdu-branch>' \
  -f 'inputs[test-name]=gptoss RDU sambastack smoke' \
  -f 'inputs[generate-cli-command]=test-config --config-keys gptoss-*-rdu-* --config-files .github/configs/nvidia-master.yaml --conc 4'
```

**Production run** — append the `perf-changelog.yaml` entry in a PR and apply `full-sweep-enabled` (or push to main).

**Monitor + fetch**
```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
jq -r '.[] | [.hw,"\(.isl)/\(.osl)",(.tput_per_gpu|round),(.mean_ttft|round),(.mean_tpot|round)] | @tsv' \
  ./results/agg_bmk.json | column -t
```

## 7. Results: What You Get

| Metric | Phase 0 (hosted) | Notes |
|---|---|---|
| `total_token_throughput`, `output_throughput` | ✅ Real | Aggregate endpoint throughput (tok/s) |
| `mean_ttft` / `p99_ttft` | ✅ Real | **Includes network latency** runner→endpoint; co-locate the box or TTFT is inflated vs on-cluster runs |
| `mean_tpot`, `mean_e2el` | ✅ Real | Per-output-token & end-to-end latency |
| per-request `completion_tokens_per_sec` | ✅ Real | From the API `usage` object |
| `tput_per_gpu`, `output_tput_per_gpu` | ⚠️ Estimate | `total / TP`; only correct if `TP` = true RDU count (§10) |
| `avg_power_w`, `joules_per_*token` (tokens/MW) | ❌ Absent | No `gpu_metrics.csv` → `aggregate_power.py` skips |

## 8. Caveats & Limitations

- **Concurrency ceiling:** meaningful operating points are `conc ≤ bundle batch size` (2–8 for gpt-oss). The Pareto curve is short/low-batch vs a GPU swept to thousands.
- **Network latency in TTFT:** the runner is off-host; TTFT/E2EL include internet/RTT. Co-locate the runner near the endpoint for representative latency.
- **Per-RDU is an estimate** until the RDU count is confirmed (§10).
- **No power metric** on hosted at all.
- **Not chip-comparable / likely `Unofficial`:** until §10–§11 are resolved, results are endpoint-level for the *deployed service*, not the silicon. Label accordingly per the README's official/unofficial policy.

## 9. Coverage vs. Existing InferenceX (GPU) Benchmarks

Assuming the `TP`/RDU-count divisor is accurate (so per-RDU throughput is valid), this is what the hosted-endpoint path still does **not** capture relative to a GPU benchmark. The summary: hosted gives **one opaque operating point of a black-box stack**, where the GPU benchmarks give **a swept frontier on a known, version-pinned stack**.

### 9.1 Sweep dimensions frozen to a single point (largest gap)
A bundle fixes the entire serving configuration. The GPU benchmarks vary each of these to map the throughput/latency frontier; on hosted RDU we get exactly one config per model and lose every axis:

| Dimension | GPU benchmarks | Hosted RDU |
|---|---|---|
| **Concurrency / batch** | Swept to thousands → full Pareto curve | Capped at bundle batch size (2–8) → short low-batch segment; throughput-max end unreachable |
| **Parallelism (TP/EP/dp-attn)** | Swept to find optimum per workload | Fixed in bundle; one topology only |
| **Precision (fp4/fp8/…)** | Compared on the same chip | Whatever SambaNova deployed; not sweepable |
| **Speculative decoding / MTP** | STP vs MTP (`*_mtp.sh`, EAGLE/NEXTN) + acceptance rate | Invisible, untoggleable |
| **Disaggregated prefill/decode** | `sglang-disagg`/dynamo P/D splits measured | Internal/opaque; only aggregate visible |
| **Multi-node** | Supported | N/A |

### 9.2 Hardware/software telemetry
- **Power / energy** — `avg_power_w`, `joules_per_*token`, tokens/MW: no path on hosted (no SMI on/near host).
- **Other GPU telemetry** — the GPU path's `gpu_metrics.csv` also captures `utilization.gpu`, `utilization.memory`, `clocks.current.sm/.memory`, `temperature.gpu`. All absent — it is not only power that is lost.
- **Kernel profiling / traces** (`profile.yml` + profiler storage) — impossible without access to the server process.

### 9.3 Software-version attribution (cuts at the InferenceX mission)
InferenceX's premise is tracking the software stack improving over time: every config pins an `image:` tag and `perf-changelog.yaml` attributes deltas to a known version bump. A hosted endpoint has **no image tag and no visible stack version** — we cannot pin, reproduce, or attribute a perf change to a specific SambaNova software release. The "live indicator of software progress" degrades to "whatever the endpoint does today."

### 9.4 Control / reproducibility / fidelity
- **Warmup & prefix caching** are server-side and uncontrolled (can inflate results).
- **Isolation** — GPU jobs get a dedicated node via `salloc`; a hosted bundle is more opaque, and SambaCloud is fully shared/multi-tenant.
- **TTFT fidelity** — network RTT is included in TTFT/E2EL (see §8).

### 9.5 What is *not* missing
- **ISL/OSL coverage** — 1k1k and 8k1k both work via the random dataset (context length ≤ bundle limit).
- **Accuracy evals are feasible** — `run_eval` (`benchmark_lib.sh`) uses lm-eval `local-chat-completions` with a `base_url`, so it can target the RDU endpoint. Caveat: it validates *SambaNova's deployed precision*, not a quant we chose — but it is a real, runnable accuracy number (unlike power, which has no path on hosted).

## 10. Open Questions / Dependencies on SambaNova

1. **RDUs per node / per bundle** for the gpt-oss-120b deployment (validate the assumed "8 SN40L per node"). Required to make `tput_per_gpu` real.
2. **Per-RDU power telemetry** — does SambaNova's tooling expose it (likely only via **on-prem**)? Required for tokens/MW.
3. **Higher-batch gpt-oss bundle** — so the throughput-ceiling comparison reflects the silicon, not a low-batch service config.
4. **Precision label** for the deployed gpt-oss-120b bundle.

## 11. Phase 1 (Future Work — chip-comparable)

- Set `tp` (the per-GPU divisor) to the **confirmed** RDU count → real per-RDU throughput.
- Add an RDU power source. `aggregate_power.py`'s column detection is generic (any header containing `power`, minus `limit/cap/max/min`); a SambaNova power CSV may ingest with little change. Likely requires **on-prem** (co-located runner reading node telemetry).
- Reconsider Option A (GitHub secrets) if the runner becomes shared.
- Re-evaluate `Unofficial` labeling once normalization + power are in place.

## 12. Decision Log

- **Hosted, not SambaCloud:** per project direction; endpoint is admin-provided per SambaStack deployment.
- **Option B (host-env credentials):** chosen for Phase 0 to avoid touching the shared `benchmark-tmpl.yml`/GitHub secrets; revisit if multi-tenant.
- **Direct `benchmark_serving.py` call:** chosen over editing `run_benchmark_serving` to keep blast radius off shared GPU recipes; result schema is identical so `process_result.py` is unaffected.
- **First model `gpt-oss-120b`:** maps to existing `gptoss` prefix; directly comparable to existing gptoss B200/MI355X rows.
