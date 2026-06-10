#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_benchmark.sh — Run benchmark_serving.py against any OpenAI-compatible
# endpoint without GitHub Actions.
#
# Usage:
#   ./run_benchmark.sh --endpoint http://<host>:<port> --model <model-id>
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
SEQ_CONFIG="1024:1024"
CONCURRENCY="1,4,8,16,32"
NUM_PROMPTS="200"

usage() {
  cat <<EOF
Usage: $0 --endpoint URL --model MODEL_ID [OPTIONS]

Required:
  --endpoint URL          API base URL (e.g. http://myserver:8000)
  --model MODEL_ID        Model id to benchmark (e.g. gpt-oss-120b)

Optional:
  --api-key KEY           Bearer token (omit if endpoint needs no auth)
  --tokenizer HF_REPO     HuggingFace tokenizer repo (defaults to --model value;
                          override when the model id is not a public HF repo,
                          e.g. openai/gpt-oss-120b)
  --seq-config CONFIG     Comma-separated ISL:OSL pairs, e.g. 1024:1024,8192:1024
                          (default: 1024:1024)
  --concurrency LEVELS    Comma-separated concurrency levels (default: 1,4,8,16,32)
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
    --num-prompts)      NUM_PROMPTS="$2";      shift 2 ;;
    --results-dir)      RESULTS_DIR="$2";      shift 2 ;;
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
# Pin to a 3.11+ interpreter — system `python3` on RHEL 8 is 3.6, too old for
# modern aiohttp/transformers wheels. Prefer 3.12, fall back to 3.11.
if command -v python3.12 >/dev/null 2>&1; then
  VENV_PY=$(command -v python3.12)
elif command -v python3.11 >/dev/null 2>&1; then
  VENV_PY=$(command -v python3.11)
else
  echo "Error: need python3.11 or python3.12 in PATH to build the venv"
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  echo "Creating venv at $VENV_DIR using $VENV_PY ..."
  "$VENV_PY" -m venv "$VENV_DIR"
  # Upgrade pip first — old pip in fresh venvs sometimes fails on modern wheels
  "$VENV_DIR/bin/pip" install -q --upgrade pip setuptools wheel
  "$VENV_DIR/bin/pip" install -q aiohttp transformers datasets tqdm numpy huggingface_hub
  echo "Dependencies installed."
fi
PYTHON="$VENV_DIR/bin/python3"

# ---------------------------------------------------------------------------
# Build sequence config list from ISL:OSL pairs
# ---------------------------------------------------------------------------
declare -a SEQ_KEYS ISL_VALS OSL_VALS
IFS=',' read -ra SEQ_PAIRS <<< "$SEQ_CONFIG"
for PAIR in "${SEQ_PAIRS[@]}"; do
  PAIR="${PAIR// /}"
  if [[ "$PAIR" != *":"* ]]; then
    echo "Error: --seq-config entries must be in ISL:OSL format (e.g. 1024:1024), got: '$PAIR'"
    exit 1
  fi
  ISL="${PAIR%%:*}"
  OSL="${PAIR##*:}"
  SEQ_KEYS+=("${ISL}:${OSL}"); ISL_VALS+=("$ISL"); OSL_VALS+=("$OSL")
done

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
    SEQ_SLUG="${SEQ//:/_}"
    RESULT_FILE="$RESULTS_DIR/${MODEL_SLUG}_${SEQ_SLUG}_conc${CONC}.json"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "→ concurrency=$CONC  isl=$ISL  osl=$OSL"

    OPENAI_API_KEY="${API_KEY:-none}" \
    "$PYTHON" "$SCRIPT_DIR/benchmark_serving.py" \
      --backend openai \
      --base-url "$ENDPOINT" \
      --endpoint /v1/completions \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --trust-remote-code \
      --dataset-name random \
      --random-input-len "$ISL" \
      --random-output-len "$OSL" \
      --num-prompts "$NUM_PROMPTS" \
      --request-rate inf \
      --max-concurrency "$CONC" \
      --ignore-eos \
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
