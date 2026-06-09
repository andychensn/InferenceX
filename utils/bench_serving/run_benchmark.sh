#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_benchmark.sh — Run benchmark_serving.py against any OpenAI-compatible
# endpoint without GitHub Actions.
#
# Usage:
#   ./run_benchmark.sh --endpoint http://10.10.0.157:8196 --model gpt-oss-120b
#
# All flags are optional if defaults suit you. Run with --help for full list.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/tmp/bench_venv"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"

# Defaults (mirror workflow defaults)
ENDPOINT=""
API_KEY=""
MODEL=""
TOKENIZER=""
SEQ_CONFIG="1k1k"
CONCURRENCY="1,4,16,32"
NUM_PROMPTS="200"

usage() {
  cat <<EOF
Usage: $0 --endpoint URL --model MODEL_ID [OPTIONS]

Required:
  --endpoint URL          API base URL (e.g. http://10.10.0.157:8196)
  --model MODEL_ID        Model id to benchmark (e.g. gpt-oss-120b)

Optional:
  --api-key KEY           Bearer token (omit if endpoint needs no auth)
  --tokenizer HF_REPO     HuggingFace tokenizer repo (defaults to --model value;
                          override when the model id is not a public HF repo,
                          e.g. openai/gpt-oss-120b)
  --seq-config CONFIG     1k1k | 8k1k | both  (default: 1k1k)
  --concurrency LEVELS    Comma-separated concurrency levels (default: 1,4,16,32)
  --num-prompts N         Prompts per run (default: 200)
  --results-dir DIR       Where to write JSON results (default: ./benchmark_results)
  --help                  Show this help
EOF
  exit 0
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)    ENDPOINT="$2";    shift 2 ;;
    --model)       MODEL="$2";       shift 2 ;;
    --api-key)     API_KEY="$2";     shift 2 ;;
    --tokenizer)   TOKENIZER="$2";   shift 2 ;;
    --seq-config)  SEQ_CONFIG="$2";  shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --help)        usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

[[ -z "$ENDPOINT" ]] && { echo "Error: --endpoint is required"; usage; }
[[ -z "$MODEL" ]]    && { echo "Error: --model is required"; usage; }

ENDPOINT="${ENDPOINT%/}"
TOKENIZER="${TOKENIZER:-$MODEL}"

# ---------------------------------------------------------------------------
# Python / venv setup
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -q aiohttp transformers datasets tqdm numpy huggingface_hub
  echo "Dependencies installed."
fi
PYTHON="$VENV_DIR/bin/python3"

# ---------------------------------------------------------------------------
# Validate endpoint is reachable
# ---------------------------------------------------------------------------
echo "Checking endpoint $ENDPOINT/v1/chat/completions ..."
HTTP=$(curl -sS -o /dev/null -w "%{http_code}" \
  -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
  --max-time 10) || true
if [[ "$HTTP" != "200" ]]; then
  echo "Error: endpoint returned HTTP $HTTP — check --endpoint, --model, and --api-key"
  exit 1
fi
echo "Endpoint OK (HTTP $HTTP)"

# ---------------------------------------------------------------------------
# Build sequence config list
# ---------------------------------------------------------------------------
declare -a SEQ_KEYS ISL_VALS OSL_VALS
if [[ "$SEQ_CONFIG" == "1k1k" || "$SEQ_CONFIG" == "both" ]]; then
  SEQ_KEYS+=("1k1k"); ISL_VALS+=("1024"); OSL_VALS+=("1024")
fi
if [[ "$SEQ_CONFIG" == "8k1k" || "$SEQ_CONFIG" == "both" ]]; then
  SEQ_KEYS+=("8k1k"); ISL_VALS+=("8192"); OSL_VALS+=("1024")
fi

IFS=',' read -ra CONC_LEVELS <<< "$CONCURRENCY"
MODEL_SLUG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/__*/_/g;s/^_//;s/_$//')
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Run benchmark matrix
# ---------------------------------------------------------------------------
TOTAL=0; PASSED=0; FAILED_RUNS=""

for idx in "${!SEQ_KEYS[@]}"; do
  SEQ="${SEQ_KEYS[$idx]}"
  ISL="${ISL_VALS[$idx]}"
  OSL="${OSL_VALS[$idx]}"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Sequence config: $SEQ  (ISL=$ISL, OSL=$OSL)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for CONC in "${CONC_LEVELS[@]}"; do
    CONC="${CONC// /}"
    RESULT_FILE="$RESULTS_DIR/${MODEL_SLUG}_${SEQ}_conc${CONC}.json"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "→ concurrency=$CONC  isl=$ISL  osl=$OSL"

    OPENAI_API_KEY="${API_KEY:-none}" \
    "$PYTHON" "$SCRIPT_DIR/benchmark_serving.py" \
      --backend openai-chat \
      --base-url "$ENDPOINT" \
      --endpoint /v1/chat/completions \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --trust-remote-code \
      --dataset-name random \
      --random-input-len "$ISL" \
      --random-output-len "$OSL" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate inf \
      --max-concurrency "$CONC" \
      --save-result \
      --result-dir "$RESULTS_DIR" \
      --result-filename "$(basename "$RESULT_FILE")" \
      && PASSED=$((PASSED + 1)) || {
        echo "Warning: run failed — seq=$SEQ conc=$CONC"
        FAILED_RUNS="$FAILED_RUNS seq=${SEQ}/conc=${CONC}"
      }
  done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Completed: $PASSED/$TOTAL runs passed"
[[ -n "$FAILED_RUNS" ]] && echo "Failed:$FAILED_RUNS"
echo "Results in: $RESULTS_DIR"

if [[ "$PASSED" -eq 0 && "$TOTAL" -gt 0 ]]; then
  echo "Error: all runs failed."
  exit 1
fi
