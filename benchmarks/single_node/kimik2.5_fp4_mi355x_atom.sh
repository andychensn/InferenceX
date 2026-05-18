#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ "$CONC" -ge 1280 ]]; then
  USE_TBO=1
  DP_ATTENTION=true
else
  USE_TBO=0
  DP_ATTENTION=false
fi
RESULT_FILENAME="${RESULT_FILENAME/dpafalse/dpa${DP_ATTENTION}}"
RESULT_FILENAME="${RESULT_FILENAME/dpatrue/dpa${DP_ATTENTION}}"
export USE_TBO DP_ATTENTION RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION, USE_TBO: $USE_TBO"
echo "Result file: ${RESULT_FILENAME}.json"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export OMP_NUM_THREADS=1

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=""
else
    CALCULATED_MAX_MODEL_LEN=" --max-model-len 10240 "
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

SERVER_ARGS=()
if [[ "$USE_TBO" == "1" ]]; then
  SERVER_ARGS+=(--enable-tbo --enable-dp-attention)
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor
MEM_FRAC_STATIC=${MEM_FRAC_STATIC:-0.7}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}

set -x

python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$MEM_FRAC_STATIC" \
    --no-enable_prefix_caching \
    --trust-remote-code \
    "${SERVER_ARGS[@]}" \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

export PYTHONDONTWRITEBYTECODE=1
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" 
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
