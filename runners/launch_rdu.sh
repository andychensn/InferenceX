#!/usr/bin/env bash
#
# Launcher for SambaNova RDU (SambaStack hosted, OpenAI-compatible endpoint).
#
# Unlike the GPU launchers, this does NOT salloc/enroot/serve a model. The model
# is served by a SambaStack "bundle" deployed on a Kubernetes cluster and reached
# over HTTPS. This launcher:
#   1. idempotently applies the git-pinned bundle manifest (no-op unless the
#      image/manifest changed in git, in which case it triggers a ~40 min reload),
#   2. waits until the endpoint is healthy (cold load is ~40 min — wait up to 60),
#   3. runs the benchmark recipe against the endpoint.
# There is NO teardown: on a dedicated env the bundle is persistent and stays warm
# across sweeps; free the node manually (kubectl delete) when done with RDU.
#
# Credentials (see SAMBANOVA_RDU_OPTION_B_DESIGN.md §7):
#   SAMBANOVA_API_KEY  - GitHub secret, injected via benchmark-tmpl.yml (mirrors HF_TOKEN)
#   API_DOMAIN         - GitHub Actions Variable, or set on the runner host
#   KUBECONFIG         - deploy-capable kubeconfig, ambient on the rdu runner host
set -euo pipefail

: "${SAMBANOVA_API_KEY:?injected from secrets.SAMBANOVA_API_KEY via benchmark-tmpl.yml}"
: "${API_DOMAIN:?GitHub Actions Variable, or set on the rdu runner host}"
: "${KUBECONFIG:?deploy-capable kubeconfig, ambient on the rdu runner host}"

MODEL_CODE="${EXP_NAME%%_*}"
BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_rdu.sh"
MANIFEST="benchmarks/sambastack/${MODEL_CODE}_${PRECISION}.yaml"

# benchmark_serving's openai-chat backend reads OPENAI_API_KEY.
export OPENAI_API_KEY="$SAMBANOVA_API_KEY"
export BASE_URL="https://${API_DOMAIN}/v1"

set -x

# --- setup: idempotent apply of the git-pinned bundle, version templated from $IMAGE ---
# No-op when unchanged; triggers a ~40 min reload only when image/manifest changed.
IMAGE="$IMAGE" envsubst < "$MANIFEST" | kubectl apply -f -

# --- wait healthy: must exceed the ~40 min cold model-load time (360 * 10s = 60 min) ---
for i in $(seq 1 360); do
    if curl -sf "${BASE_URL}/models" -H "Authorization: Bearer ${OPENAI_API_KEY}" \
         | grep -q "\"${MODEL}\""; then
        echo "[rdu] bundle healthy"
        break
    fi
    if [ "$i" = 360 ]; then
        echo "[rdu] ERROR: bundle never became healthy after 60 min" >&2
        exit 1
    fi
    echo "[rdu] waiting for bundle... ($i)"
    sleep 10
done

# --- benchmark (recipe hits the endpoint; no GPU monitor; Phase-C power stub inside) ---
bash "$BENCH_SCRIPT"

# No teardown by design — see header. benchmark-tmpl.yml's docker/slurm cleanup
# steps are guarded by `command -v docker`/`squeue` and are harmless no-ops here.
