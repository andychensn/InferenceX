# SambaNova RDU (SambaStack hosted) — runbook

Run InferenceX benchmarks on SambaNova RDU through the SambaStack **hosted**
OpenAI-compatible endpoint. Full design + rationale: `SAMBANOVA_RDU_OPTION_B_DESIGN.md`
(repo root).

## How it works (vs the GPU flow)

The model is served by a SambaStack **bundle** on a Kubernetes cluster and reached
over HTTPS — there is no local server. So the `rdu` launcher does **not**
`salloc`/serve; instead it `kubectl apply`s a git-pinned bundle manifest, waits for
the endpoint to be healthy, and drives load with `benchmark_serving`'s `openai-chat`
backend. Mapping to the GPU flow:

| GPU | RDU |
|---|---|
| `image:` tag (software version pin) | `image:` = bundle image/chart **digest**, templated into the manifest |
| launcher `salloc` + serve | launcher `kubectl apply` (idempotent) + wait healthy |
| benchmark localhost | `benchmark_serving --backend openai-chat --base-url https://$API_DOMAIN/v1` |
| launcher `scancel` (teardown) | **none** — bundle is persistent (≈40-min cold load); `kubectl delete` manually |

Files: `runners/launch_rdu.sh`, `benchmarks/single_node/gptoss_<precision>_rdu.sh`,
`benchmarks/sambastack/<model>_<precision>.yaml` (this dir),
`.github/configs/sambanova-master.yaml`, and the `rdu` entry in
`.github/configs/runners.yaml`.

## This is a fork deployment (Unofficial)

`SemiAnalysisAI/InferenceX` is the official repo; results from a fork are
**Unofficial** per the project README. The fork has its own Actions, runner, and
secrets — set them up under your account/org.

### One-time setup on the fork
1. **Enable Actions** (forks disable workflows by default): Settings → Actions.
2. **Register the runner**: a network-only box (no GPU/RDU needed) with egress to
   the endpoint and a deploy-capable `kubeconfig`. Register it with **name `rdu_0`**,
   **label `rdu`**, and set `KUBECONFIG` + `API_DOMAIN` in the runner's `.env`.
   Co-locate it near the cluster for faithful TTFT.
3. **Credentials**: add secret `SAMBANOVA_API_KEY` and variable `API_DOMAIN`
   (Settings → Secrets and variables → Actions). The API key is injected into the
   job via `benchmark-tmpl.yml` (mirrors `HF_TOKEN`); kubeconfig stays on the host.
4. **High-limit QoS**: configure a high/unlimited tier via `qosList` in the bundle
   manifest and map the benchmark API key to it (rate limit must be ≥ max concurrency).

### Fill these TODOs before the first run
| TODO | Where |
|---|---|
| Bundle image/chart **digest** | `.github/configs/sambanova-master.yaml` → `image:` |
| Bundle **template name** (e.g. `cd-dyt-gpt-oss-120b-8-32-64-128k`) | `benchmarks/sambastack/gptoss_fp8.yaml` |
| **`API_DOMAIN`** | fork Actions Variable |
| **Precision** (assumed `fp8`) | confirm; rename file + entry if different |

## Run

**Smoke test (one point, no changelog dependency):**
```bash
gh api -X POST /repos/<owner>/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref=main -f 'inputs[ref]=add-rdu-gptoss-fp8' -f 'inputs[test-name]=gptoss RDU smoke' \
  -f 'inputs[generate-cli-command]=test-config --config-keys gptoss-fp8-rdu-sambastack --config-files .github/configs/sambanova-master.yaml --conc 4'
```

**Full sweep:** the `perf-changelog.yaml` entry triggers `run-sweep.yml` on push to
the fork's `main` (or on a labeled PR). The first serialized job applies the bundle
(no-op if warm; ~40 min if `image:` changed), then jobs sweep concurrency × seq-len.

**Validate locally first:**
```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/sambanova-master.yaml --single-node
```

## What you get / don't

- ✅ Per-RDU throughput (`tput_per_gpu = total / 16`), TTFT/TPOT/E2EL, tokens/sec.
- ⚠️ Concurrency sweep capped at the bundle batch ceiling (8 at the 8K/32K seq tier).
- ❌ **No `tokens/MW`** — power is not exposed on hosted. It is deployment-gated:
  hosted needs a SambaNova **network power API** (Case A); host-local SMI / BMC / PDU
  require on-prem or bare-metal. The power collector is stubbed in the recipe. See
  design doc §12.
