# Design Doc — SambaNova RDU Benchmarking (Option B: GPU-Parity, CI-Managed Bundle Lifecycle)

**Status:** Proposed
**Author:** andy.chen@sambanovasystems.com
**Date:** 2026-05-29
**Supersedes:** the manual-deploy approach in `SAMBANOVA_RDU_DESIGN.md` (Phase 0). That doc remains valid as a zero-infra SambaCloud *pipeline smoke test*; this doc is the real integration once a **dedicated SambaStack hosted environment exists**.

---

## 1. Goal

Benchmark SambaNova RDU in InferenceX **as close to the existing GPU benchmark flow as possible**:

- **Experiment config tracked in git** — which bundle, its version, and its deployment spec are all version-controlled, reviewed, and reproducible (the same guarantee the GPU `image:` tag gives).
- **The GitHub Action handles setup *and* teardown** of the bundle from the git-pinned config — no manual `kubectl` in the loop.
- **Chip-comparable** results: per-RDU throughput now; tokens/MW deferred to a later phase (see §10).

A **dedicated SambaStack hosted environment is available**, so we control the cluster (RDU count known, version pinnable, runner co-locatable). First model: **`gpt-oss-120b`** (InferenceX prefix `gptoss`).

## 2. Background (how the GPU flow works)

For an existing GPU config, a benchmark job:

```
master-config entry (image + model + recipe params + search space)   ← git
  └─ perf-changelog.yaml entry triggers the sweep                      ← git
       └─ run-sweep.yml → matrix → benchmark-tmpl.yml (runs-on: <runner>)
            └─ launch_<hw>.sh:  salloc node → import image → SERVE → benchmark → scancel (teardown)
                 └─ process_result → collect-results → ingest → dashboard
```

Two properties we want to preserve: the **`image:` tag pins the exact software version** (CI enforces it by pulling that image), and the **launcher does setup + teardown per job** on an isolated node.

## 3. The 1:1 Mapping to the GPU Flow

| GPU flow | RDU equivalent |
|---|---|
| `image:` in master config = pinned software version | `image:` holds the **bundle image/chart digest**; same field, same "bump ⇒ re-benchmark" semantics |
| recipe + serve flags in git | **committed bundle manifest** `benchmarks/sambastack/<model>_<precision>.yaml`, templated from `$IMAGE` |
| `perf-changelog.yaml` triggers the sweep | identical — a manifest/`image:` edit + changelog entry triggers it |
| launcher `salloc` a dedicated node | launcher `kubectl apply -f <manifest>` (idempotent) |
| launcher starts the server (`vllm/sglang serve`) | bundle pods load the model onto RDUs; launcher polls `/v1/models` until healthy |
| benchmark localhost | `benchmark_serving --backend openai-chat --base-url https://$API_DOMAIN/v1` |
| launcher `scancel` (teardown) | launcher `kubectl delete -f <manifest>` |
| squash import cached via flock + reuse | **idempotent `kubectl apply`**: if the pinned bundle is already healthy, it is a no-op |

The result: the Action deploys, verifies, benchmarks, and tears down — driven entirely by the git-pinned config, just like a GPU recipe.

## 4. Experiment Tracking in Git (provenance)

Three layers give the same "git provably produced this result" guarantee the GPU rows have:

1. **`image:` = bundle version pin.** The master-config entry's `image:` is the bundle image/chart digest. The manifest is **templated from it** (single source of truth). A version bump is an `image:` edit + changelog entry — *identical workflow to a GPU image bump*.
2. **Committed manifest.** `benchmarks/sambastack/<model>_<precision>.yaml` is the exact `bundleSpecs`/`bundleDeploymentSpecs` (batch ceiling, replicas, precision), reviewable and diffable.
3. **Runtime capture + assert.** The recipe records the *actually-deployed* bundle (`/v1/models` + a read-only `kubectl get` of the deployment) and **fails the run if it does not match the pinned config** — killing silent drift between "what git says" and "what is serving."
   - **Caveat (verified against the repo):** `meta_env.json` is written by `benchmark_lib.sh` with a *fixed* field set; there is no per-recipe hook to add custom fields, and `process_result.py` builds the agg row from required env vars (`RUNNER_TYPE`, `TP`, …), not arbitrary meta. So to make the captured bundle identity *flow into `agg_bmk.json`*, we must either (a) extend `benchmark_lib.sh`/`process_result.py` to carry an extra field, or (b) upload the captured identity as a small side artifact. The assert-and-fail check itself needs neither and works today. Pick (a) if bundle identity must appear on the dashboard row.

## 5. CI-Managed Setup / Teardown

- **Setup (per job, in the launcher):** `kubectl apply` the manifest (templated from `$IMAGE`) → poll `/v1/models` until healthy. Idempotent.
- **Serialization:** register **exactly one `rdu` runner**. A self-hosted runner runs one job at a time, so all `runs-on: rdu` matrix jobs serialize → each benchmark hits the bundle alone (clean per-RDU numbers), and **only the first job actually loads the model** (idempotent apply makes the rest no-ops) — mirroring how only the first GPU job imports the squash file.
- **Teardown — the one justified divergence from GPU.** GPU tears down per job because re-setup is cheap and node-isolated. **The RDU bundle cold-loads in ~40 min** (measured), so per-job (or even per-sweep) teardown is out — it would pay 40 min repeatedly. On a **dedicated env** the right model is a **persistent bundle**:
  - Every job runs idempotent `kubectl apply` of the git-pinned manifest. It is a **fast no-op when unchanged**, and only triggers the ~40-min reload when `image:`/the manifest changed in git — i.e. exactly when a fresh benchmark is intended. (Closer to the GPU squash cache than per-job teardown ever was: apply = cached no-op; reload = re-import on a new image.)
  - **No automatic teardown.** The bundle stays warm across sweeps. Free the node manually (`kubectl delete`, or a tiny dispatch workflow) only when you're done with RDU.
  - First job after a bump sits ~40 min "loading," paid once; the launcher's health-check wait is set to 60 min to cover it.

## 6. Sweep Model

Split InferenceX's sweep axes by what the endpoint lets us control:

- **Client-side (swept natively, exactly like GPU):**
  - **Concurrency** → `conc-start`/`conc-end` in the search-space. A single bundle does dynamic batching up to its max batch size, so sweeping concurrency `1→maxBS` traces a real (short) throughput/latency curve. **Cap `conc-end` at the bundle batch ceiling** — note this is **seq-length-dependent** for gpt-oss (per the docs: 8K/32K → BS 8, 64K → BS 4, 128K → BS 2). The §8 1k1k/8k1k scenarios sit in the 8K tier so `conc-end: 8` is correct *for those rows*; longer-context scenarios need a lower cap. Beyond the ceiling you measure SambaNova's queue, not throughput.
  - **ISL/OSL** → multiple `fixed-seq-len` scenarios (1k1k, 8k1k), within the bundle's context length. (There is no native 1K bundle — the smallest gpt-oss bundle is 8K — but ISL/OSL is client-side, so 1024-token requests against an 8K bundle are fine.)
- **Bundle-side (one master-config entry per deployed bundle):**
  - **Higher batch ceiling / precision / parallelism / spec-decoding** are baked into the bundle template. To sweep them, deploy a different bundle (a different committed manifest) and add a **second config-key**. Such variants come from either a **self-service custom bundle** (the docs allow custom bundles "so long as they fit in DDR memory") or a SambaNova-built template — not necessarily SambaNova-only.

## 7. Credentials

Match the repo's existing split, which is verifiable in the GPU path: **substrate access is ambient on the runner host** (every launcher calls `salloc`/`srun` with no secret — the host has SLURM submit rights), while **the service token the benchmark needs is a GitHub secret** (`benchmark-tmpl.yml` injects `HF_TOKEN: ${{ secrets.HF_TOKEN }}`). RDU maps onto that line 1:1:

| RDU credential | GPU analog | Where it lives |
|---|---|---|
| **`SAMBANOVA_API_KEY`** (auth to the inference service) | `HF_TOKEN` (auth to HuggingFace) — GitHub secret | **GitHub secret**, injected in `benchmark-tmpl.yml` next to `HF_TOKEN` |
| **kubeconfig** (reach/command the cluster) | SLURM submit rights — ambient | **Ambient on the `rdu` runner host** (`KUBECONFIG` in the runner's `.env`) |
| **`API_DOMAIN`** (endpoint address) | partition names / mount paths — hardcoded in the launcher | Launcher/config or a GitHub **Variable** (not sensitive) |

So:
1. Add `SAMBANOVA_API_KEY` under **Settings → Secrets and variables → Actions**, and one line in `benchmark-tmpl.yml`:
   ```yaml
   env:
     HF_TOKEN: ${{ secrets.HF_TOKEN }}
     SAMBANOVA_API_KEY: ${{ secrets.SAMBANOVA_API_KEY }}   # ← mirrors HF_TOKEN
   ```
2. Place the SambaNova-provided kubeconfig on the `rdu` runner host and set `KUBECONFIG` in the runner's `.env` (mirrors SLURM access being ambient — the repo has no precedent for a cluster-access credential as a GitHub secret).
3. `API_DOMAIN` as a GitHub Actions Variable (or in the launcher), since it is endpoint config, not a secret.

## 8. File Set & Draft Artifacts

> Drafts. Confirmed: RDU count = 16 (`tp: 16`); `benchmark_serving` flags verified against this repo's client. Remaining placeholders (`<…>`): precision label, bundle image digest, bundle template name, and the optional `kubectl get` provenance path.

**`.github/configs/runners.yaml`** (append):
```yaml
rdu:
  - 'rdu_0'   # single runner ⇒ serialized sweep; kubeconfig + API_DOMAIN on its host (API key comes from the secret)
```

**`runners/launch_rdu.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail
: "${SAMBANOVA_API_KEY:?injected from secrets.SAMBANOVA_API_KEY via benchmark-tmpl.yml}"
: "${API_DOMAIN:?GitHub Actions Variable (or set on the runner host)}"
: "${KUBECONFIG:?deploy-capable kubeconfig, ambient on the rdu runner host}"

MODEL_CODE="${EXP_NAME%%_*}"
BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_rdu.sh"
MANIFEST="benchmarks/sambastack/${MODEL_CODE}_${PRECISION}.yaml"

export OPENAI_API_KEY="$SAMBANOVA_API_KEY"
export BASE_URL="https://${API_DOMAIN}/v1"

set -x
# setup: idempotent apply of the git-pinned bundle, version from $IMAGE.
# No-op when unchanged; triggers a ~40 min reload only when image/manifest changed.
IMAGE="$IMAGE" envsubst < "$MANIFEST" | kubectl apply -f -

# wait healthy — must exceed the ~40 min cold model-load time. 360 * 10s = 60 min.
for i in $(seq 1 360); do
  if curl -sf "${BASE_URL}/models" -H "Authorization: Bearer ${OPENAI_API_KEY}" | grep -q "\"${MODEL}\""; then
    echo "bundle healthy"; break
  fi
  [ "$i" = 360 ] && { echo "ERROR: bundle never healthy after 60 min"; exit 1; }
  echo "waiting for bundle... ($i)"; sleep 10
done
# No teardown here: the bundle is persistent (dedicated env). It is torn down only
# manually when freeing the node, or replaced automatically by the apply above on an
# image/manifest bump. (benchmark-tmpl's container/slurm cleanup steps are no-ops here.)

# benchmark
bash "$BENCH_SCRIPT"
```
> Teardown is intentionally absent: with a 40-min cold load on a dedicated env, the bundle stays persistent across sweeps. A separate `kubectl delete` (manual or a tiny dispatch workflow) frees the node when you're done with RDU.

**`benchmarks/single_node/gptoss_<precision>_rdu.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail

# provenance: capture actually-deployed bundle and assert it matches the pin
DEPLOYED=$(kubectl get <bundleDeploymentSpec/path> -o jsonpath='{...template,version,replicas...}')
echo "[provenance] deployed bundle: $DEPLOYED  expected image: $IMAGE"
# TODO: assert DEPLOYED matches $IMAGE/manifest; exit 1 on drift.
# TODO: persist DEPLOYED into meta so process_result records it in the agg row.

cd utils/bench_serving
# All flags below verified against utils/bench_serving/benchmark_serving.py.
# openai-chat backend reads OPENAI_API_KEY and requires the URL to end in chat/completions.
python3 benchmark_serving.py \
  --backend openai-chat \
  --base-url "$BASE_URL" --endpoint /chat/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$ISL" --random-output-len "$OSL" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --max-concurrency "$CONC" --num-prompts "$(( CONC * 8 ))" \
  --request-rate inf \
  --num-warmups "$(( CONC * 2 ))" \
  --use-chat-template \
  --save-result --result-filename "${GITHUB_WORKSPACE}/${RESULT_FILENAME}.json"
```
> `--request-rate inf` + `--num-warmups` match `run_benchmark_serving` in `benchmark_lib.sh`. `--ignore-eos` is used on GPU to guarantee OSL; on a hosted endpoint it may be ignored server-side — confirm whether SambaNova honors it before relying on it.

**`benchmarks/sambastack/gptoss_<precision>.yaml`** (templated bundle manifest):
```yaml
# applied via: IMAGE=<digest> envsubst < this | kubectl apply -f -
# NOTE: docs use bundle *template* names like `cd-dyt-gpt-oss-120b-8-32-64-128k`,
# which differ from the served model-id (`gpt-oss-120b`) used in the API `model` field.
bundles:
  bundleSpecs:
    - name: cd-dyt-gpt-oss-120b-8-32-64-128k   # bundle template name (CONFIRM exact)
      image: ${IMAGE}                           # single source of truth = master-config `image:`
  bundleDeploymentSpecs:
    - bundle: cd-dyt-gpt-oss-120b-8-32-64-128k
      minReplicas: 1            # docs field is minReplicas (+ groups / qosList); one bundle per node
      # qosList / context lengths / batch sizes per the chosen bundle template
```

**`.github/configs/nvidia-master.yaml`** (or a vendor master file), append:
```yaml
gptoss-<precision>-rdu-sambastack:
  image: "<bundle-image-or-chart-digest>"   # version pin; templated into the manifest
  model: gpt-oss-120b                        # served model-id the bundle answers to
  model-prefix: gptoss
  runner: rdu
  precision: <precision>                     # informational; bundle-managed. CONFIRM.
  framework: sambastack
  multinode: false
  disagg: false
  scenarios:
    fixed-seq-len:
    - { isl: 1024, osl: 1024, search-space: [ { tp: 16, conc-start: 1, conc-end: 8 } ] }
    - { isl: 8192, osl: 1024, search-space: [ { tp: 16, conc-start: 1, conc-end: 8 } ] }
```

**`perf-changelog.yaml`** (append at END, preserve whitespace, end with newline):
```yaml
- config-keys:
    - gptoss-<precision>-rdu-sambastack
  description:
    - "Add gpt-oss-120b on SambaNova RDU via dedicated SambaStack (CI-managed bundle lifecycle)"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

## 9. How to Run

> Operator runbook (fork setup, runner registration, the 4 TODOs, smoke test) lives in `benchmarks/sambastack/README.md`.

**One-time:** stand up the dedicated SambaStack env (with a high-limit QoS tier in the manifest, and the benchmark key mapped to it); add the `SAMBANOVA_API_KEY` GitHub secret + one line in `benchmark-tmpl.yml` (§7); register the **single** `rdu` runner (name `rdu_0`, label `rdu`) with the kubeconfig + `API_DOMAIN` on its host; commit the artifacts in §8.

**Smoke test (no changelog touch):**
```bash
gh api -X POST /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' -f 'inputs[ref]=<rdu-branch>' -f 'inputs[test-name]=gptoss RDU smoke' \
  -f 'inputs[generate-cli-command]=test-config --config-keys gptoss-*-rdu-* --config-files .github/configs/nvidia-master.yaml --conc 4'
```

**Production:** append the `perf-changelog.yaml` entry in a PR, apply `full-sweep-enabled`. The first serialized job applies the bundle (a no-op if already warm; ~40 min if the `image:` changed), then jobs sweep concurrency × seq-len against the persistent bundle. No teardown — the bundle stays warm.

**Fetch:** `gh run download <RUN_ID> -n results_bmk -D ./results` → `agg_bmk.json` (per-RDU throughput, TTFT/TPOT; captured bundle identity only if §4 option (a) is implemented).

> **No-container note:** the GPU launchers run a server inside an enroot/Docker container; the `rdu` launcher runs none. `benchmark-tmpl.yml`'s pre/post "Resource cleanup" steps are guarded by `command -v docker` / `command -v squeue`, so on a network-only `rdu` runner they are harmless no-ops. The recipe runs directly on the runner host (which has `kubectl`, `curl`, `python3`).

## 10. What This Captures vs. GPU

| Metric / property | This design |
|---|---|
| Per-RDU throughput (`tput_per_gpu = total / TP`, TP = 16) | ✅ Real |
| Latency (TTFT/TPOT/E2EL) | ✅ Real, faithful (co-located runner) |
| Version pinning / attribution | ✅ Restored (`image:` digest + committed manifest + runtime assert) |
| Isolation / reproducibility | ✅ (single runner serialization, one bundle per node) |
| Concurrency sweep | ✅ (capped at bundle batch ceiling) |
| Batch / precision / parallelism sweep | ⚠️ One config-key per deployed bundle; variants via self-service custom bundles or SambaNova-built templates |
| **tokens/MW (power)** | ❌ **Accepted limitation** — no power access today. Needs a node BMC/IPMI/Redfish or PDU side-channel (Phase C); `aggregate_power.py` ingests a generic power CSV, so the collector is the only new piece. Energy axis stays blank until then |
| Per-RDU util/clocks/temp; kernel profiling | ❌ Not exposed by the SambaStack stack |

## 11. Open Questions / To Confirm

**Resolved:**
1. **Cold model-load time ≈ 40 min** (measured) → persistent bundle, no auto-teardown, idempotent apply, 60-min health-check wait (§5).
2. **RDU count = 16** → `tp: 16` (per-RDU throughput is real, not an estimate).
3. **Rate limit / QoS — operator-controlled.** We hold kubectl, so we set a high/unlimited benchmark tier via `qosList` in the committed manifest. *Action:* ensure the benchmark API key maps to that high-limit usergroup (set in-manifest so it's git-tracked).
4. **Bundle addressing via `model` field — moot for this flow.** One bundle is deployed at a time (one config-key → one manifest), so `model: gpt-oss-120b` is unambiguous. Only revisit if multiple same-model bundles ever run concurrently.
5. **Provenance read-back — auto-handled.** The CI applies the git manifest itself, so the deployed bundle *is* the pinned one by construction; the `kubectl get` read-back/assert is optional hardening, not a blocker.

**Still open / accepted:**
6. **Power — no access (accepted limitation), and it's deployment-gated.** No `tokens/MW`. Per-RDU throughput + latency are unaffected. On **hosted**, power is only possible if SambaNova exposes a **network power API** (Case A, §12); host-level paths (local SMI, BMC/IPMI, PDU) require **on-prem or bare-metal**. Energy axis stays blank until one of those exists; results aren't fully chip-comparable on energy — label accordingly.
7. **Precision label** for the deployed gpt-oss bundle (informational field in the master config).

## 12. Phase C (chip-comparable + energy) — power is deployment-gated

Adding `tokens/MW` is one well-defined integration plus a hard access constraint.

### 12.1 The integration target (fixed, vendor-agnostic)
`aggregate_power.py` (invoked by `process_result.py`) ingests a CSV and needs only **(a)** a timestamp column (epoch / ISO-8601 / `YYYY/MM/DD HH:MM:SS`) and **(b)** a column whose name contains `power` in watts (it ignores `limit/cap/max/min`). It sums power across accelerators per sample, averages over the benchmark window, and patches `avg_power_w` / `joules_per_output_token` / `joules_per_total_token` into the agg row. So the only new code is a **background collector** that writes an `nvidia-smi`-shaped CSV — one row per RDU per sample, `timestamp,index,power_w`, 16 indices — started before `benchmark_serving` and stopped after (mirroring `start_gpu_monitor`/`stop_gpu_monitor`):

```bash
start_rdu_power_monitor() {                      # mirrors start_gpu_monitor
  echo "timestamp,index,power_w" > "$GPU_METRICS_CSV"
  ( while true; do
      ts=$(date +%s.%N)
      curl -sf "$SN_POWER_API" -H "Authorization: Bearer $SAMBANOVA_API_KEY" \
        | jq -r --arg ts "$ts" '.rdus[] | "\($ts),\(.index),\(.power_w)"'
      sleep 1
    done ) >> "$GPU_METRICS_CSV" &
  GPU_MONITOR_PID=$!
}
```
`aggregate_power.py` then needs **no change**. (A Prometheus exporter works too: query `/api/v1/query_range` over the run window and materialize the same CSV.)

### 12.2 The hard constraint: where the power data can come from depends on the deployment
Measuring power means *something* reads the hardware. Who can do that is set by the hosted/on-prem control boundary:

- **Hosted** = SambaNova manages infrastructure/Kubernetes. Your kubeconfig is scoped to *deploying bundles* + the HTTP endpoint. You have **no** node OS/SSH access, **no** BMC/IPMI, and **cannot** run privileged host-device workloads (a power-reading DaemonSet/sidecar).
- **On-prem** = customer full control of the K8s cluster and the SambaRack — host shell, privileged DaemonSets, BMC/IPMI/PDU all available.

Three sourcing cases, and which deployment each requires:

| Power source | Hosted | On-prem | Bare-metal direct-runtime |
|---|---|---|---|
| **Case A** — SambaNova exposes a network power API / Prometheus exporter (you poll it from the runner) | ✅ only if SambaNova exposes it | ✅ | ✅ |
| **Case B** — host-local SMI tool (via SSH or a privileged DaemonSet) | ❌ no node/privileged access | ✅ | ✅ |
| **BMC/IPMI/Redfish or metered PDU** node power | ❌ not your datacenter | ✅ | ✅ |
| **Local SMI as a background monitor** (full GPU parity) | ❌ | ⚠️ if the runtime runs on a node you control | ✅ |

**Rule:** on **hosted**, power is possible **only via Case A** (a SambaNova-provided network API) — every path where *we* read the hardware (Case B, BMC, PDU, local SMI) requires **on-prem or bare-metal**. This is why on-prem is the answer for chip-comparable *energy*: power measurement needs hardware access, which hosted does not grant.

### 12.3 Granularity to document
- A **per-RDU** power API → true per-accelerator energy (divide by 16 naturally) — best.
- A **node/chassis** reading (BMC/PDU) → node-level energy ÷ 16, i.e. includes host overhead; closer to wall power. Still a valid `tokens/MW`, just a different (and arguably fairer) granularity — state which is used.

### 12.4 Bare-metal direct-runtime (the cleanest path if power is host-local)
If power lives only in a host-local SMI tool, the strongest move is to drop SambaStack/K8s and run the SambaNova runtime **directly on a SambaRack node that is the GitHub runner host** — full GPU parity: launcher starts the runtime locally (like `vllm/sglang serve`), `start_rdu_power_monitor` runs the local SMI, benchmark hits localhost. This also removes the 40-min bundle reload and unlocks the *other* missing telemetry (utilization/clocks/temp, profiling). Cost: bare-metal access + the runtime distribution. Track as a separate variant if pursued.

## 13. Decision Log

- **Option B on a dedicated env (not SambaCloud):** the dedicated env exists, giving RDU count, version pinning, isolation, and a co-located runner.
- **GPU-parity via CI-managed bundle lifecycle:** the launcher does `kubectl apply`/wait/`delete`, mirroring `salloc`/serve/`scancel`; config is git-pinned via `image:` + committed manifest.
- **Single `rdu` runner:** serializes the sweep (clean per-RDU numbers) and makes idempotent apply load the model once.
- **Persistent bundle, no auto-teardown (driven by the 40-min cold load):** each job does an idempotent `kubectl apply` — a no-op unless `image:`/manifest changed, in which case it pays the ~40-min reload once. The bundle stays warm across sweeps; teardown is manual only. Chosen over per-job/per-sweep teardown because re-loading 16 RDUs costs 40 min.
- **`tp: 16`:** confirmed RDU count, so `tput_per_gpu` is real.
- **Hybrid credentials, matching the repo's existing split:** `SAMBANOVA_API_KEY` is a GitHub secret injected via `benchmark-tmpl.yml` (mirrors `HF_TOKEN`); the kubeconfig is ambient on the runner host (mirrors SLURM submit rights, which the repo never stores as a secret); `API_DOMAIN` is a Variable/launcher config (mirrors hardcoded partition names).
- **Power deferred to Phase C, and deployment-gated:** the integration is trivial (a CSV-writing collector → `aggregate_power.py` unchanged), but the *data* requires hardware access. On **hosted** that means a SambaNova-provided **network power API** (Case A); host-local SMI / BMC / PDU all require **on-prem or bare-metal**. See §12.
