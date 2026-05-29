#!/usr/bin/env bash
#
# gpt-oss-120b on SambaNova RDU (SambaStack hosted, OpenAI-compatible endpoint).
#
# NOTE: precision in the filename is "fp8" — this is a GUESS for the deployed
# SambaStack bundle's precision. CONFIRM with SambaNova and rename this file +
# the master-config entry + the manifest if it differs.
#
# Runs against a remote endpoint (no local server, no GPU monitor). The launcher
# (runners/launch_rdu.sh) has already deployed the bundle and exported BASE_URL +
# OPENAI_API_KEY. We just drive load with benchmark_serving's openai-chat backend.
set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    ISL \
    OSL \
    CONC \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

: "${BASE_URL:?set by launch_rdu.sh}"
: "${OPENAI_API_KEY:?set by launch_rdu.sh}"

# --- Phase C (power) stub -------------------------------------------------------
# tokens/MW requires a power CSV (timestamp,index,power_w) over the benchmark
# window; aggregate_power.py then patches avg_power_w/joules_per_*token with NO
# change. On SambaStack HOSTED this is only possible if SambaNova exposes a
# network power API (Case A) — host-local SMI/BMC/PDU need on-prem/bare-metal.
# See SAMBANOVA_RDU_OPTION_B_DESIGN.md §12. Until an API exists, power is omitted.
#
# if [[ -n "${SN_POWER_API:-}" ]]; then start_rdu_power_monitor; fi   # TODO(Phase C)

# --- benchmark ------------------------------------------------------------------
# Flags verified against utils/bench_serving/benchmark_serving.py. The openai-chat
# backend reads OPENAI_API_KEY and requires the URL to end in chat/completions.
# --request-rate inf + --num-warmups mirror run_benchmark_serving in benchmark_lib.sh.
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
    --request-rate inf \
    --num-warmups "$(( CONC * 2 ))" \
    --use-chat-template \
    --save-result \
    --result-filename "${GITHUB_WORKSPACE}/${RESULT_FILENAME}.json"

# if [[ -n "${SN_POWER_API:-}" ]]; then stop_rdu_power_monitor; fi    # TODO(Phase C)
