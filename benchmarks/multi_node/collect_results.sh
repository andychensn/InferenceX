#!/usr/bin/env bash
# Collect and summarize benchmark results from a manual disagg run.
#
# Usage: bash collect_results.sh <log_dir>
#   e.g.: bash collect_results.sh logs/manual-1779883634
#
# Processes all result JSONs and prints a summary table + saves CSV.
set -euo pipefail

LOG_DIR="$(cd "${1:?Usage: $0 <log_dir>}" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULT_DIR=$(mktemp -d)
trap 'rm -rf "$RESULT_DIR"' EXIT

echo "Looking for result JSONs..."

# Results may be in the repo root (--result-dir /workspace/)
for f in "$REPO_ROOT"/*.json; do
    [[ -f "$f" ]] || continue
    if grep -q "total_token_throughput\|output_throughput" "$f" 2>/dev/null; then
        cp "$f" "$RESULT_DIR/"
        echo "  Found: $(basename "$f")"
    fi
done

# Results may also be in the log dir
find "$LOG_DIR" -name '*.json' -print0 2>/dev/null | while IFS= read -r -d '' f; do
    if grep -q "total_token_throughput\|output_throughput" "$f" 2>/dev/null; then
        cp "$f" "$RESULT_DIR/"
        echo "  Found: $(basename "$f")"
    fi
done

RESULT_COUNT=$(find "$RESULT_DIR" -name '*.json' | wc -l)
if [[ "$RESULT_COUNT" -eq 0 ]]; then
    echo "No benchmark result JSONs found."
    exit 1
fi
echo ""
echo "Found $RESULT_COUNT result file(s). Processing..."
echo ""

# Process each result with process_result.py
export RUNNER_TYPE="mi300x"
export FRAMEWORK="sglang-disagg"
export PRECISION="fp8"
export SPEC_DECODING="none"
export MODEL_PREFIX="qwen3.5"
export IMAGE="sglang-rocm-mi300x-rdma:latest"
export DISAGG="true"
export IS_MULTINODE="true"
export PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-1}"
export PREFILL_TP="${PREFILL_TP:-8}"
export PREFILL_EP="${PREFILL_EP:-1}"
export PREFILL_DP_ATTN="${PREFILL_DP_ATTN:-false}"
export DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-1}"
export DECODE_TP="${DECODE_TP:-8}"
export DECODE_EP="${DECODE_EP:-1}"
export DECODE_DP_ATTN="${DECODE_DP_ATTN:-false}"

cd "$RESULT_DIR"

for f in *.json; do
    [[ "$f" == agg_* ]] && continue
    basename="${f%.json}"

    prefill_gpus=$(echo "$basename" | grep -oP '(?<=ctx_)\d+' || echo "8")
    decode_gpus=$(echo "$basename" | grep -oP '(?<=gen_)\d+' || echo "8")

    export RESULT_FILENAME="$basename"
    export ISL="${ISL:-1024}"
    export OSL="${OSL:-1024}"
    export PREFILL_GPUS="${prefill_gpus:-8}"
    export DECODE_GPUS="${decode_gpus:-8}"

    python3 "$REPO_ROOT/utils/process_result.py" 2>/dev/null || \
        echo "  Warning: failed to process $f"
done

# Determine output CSV path (next to the log dir)
CSV_OUT="${LOG_DIR}/results.csv"

# Generate CSV from the processed agg_*.json files
python3 - "$RESULT_DIR" "$CSV_OUT" <<'PYEOF'
import json, csv, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
csv_path = sys.argv[2]

results = []
for p in results_dir.rglob("agg_*.json"):
    try:
        r = json.loads(p.read_text())
        if "is_multinode" in r:
            results.append(r)
    except Exception:
        pass

if not results:
    print("No processed results found for CSV.")
    sys.exit(0)

is_mn = any(r.get("is_multinode") for r in results)

if is_mn:
    headers = [
        "model", "hw", "framework", "precision", "isl", "osl",
        "prefill_tp", "prefill_ep", "prefill_dp_attention", "prefill_num_workers", "num_prefill_gpu",
        "decode_tp", "decode_ep", "decode_dp_attention", "decode_num_workers", "num_decode_gpu",
        "conc",
        "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "p99.9_ttft_ms",
        "median_tpot_ms",
        "median_intvty", "p90_intvty", "p99_intvty", "p99.9_intvty",
        "median_e2el_s", "p90_e2el_s", "p99_e2el_s", "p99.9_e2el_s",
        "tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu",
    ]
    def row(r):
        return [
            r.get("model", ""), r.get("hw", ""), r.get("framework", ""),
            r.get("precision", ""), r.get("isl", ""), r.get("osl", ""),
            r.get("prefill_tp", ""), r.get("prefill_ep", ""),
            r.get("prefill_dp_attention", ""), r.get("prefill_num_workers", ""),
            r.get("num_prefill_gpu", ""),
            r.get("decode_tp", ""), r.get("decode_ep", ""),
            r.get("decode_dp_attention", ""), r.get("decode_num_workers", ""),
            r.get("num_decode_gpu", ""),
            r.get("conc", ""),
            f"{r.get('median_ttft', 0) * 1000:.2f}",
            f"{r.get('p90_ttft', 0) * 1000:.2f}",
            f"{r.get('p99_ttft', 0) * 1000:.2f}",
            f"{r.get('p99.9_ttft', 0) * 1000:.2f}",
            f"{r.get('median_tpot', 0) * 1000:.2f}",
            f"{r.get('median_intvty', 0):.2f}",
            f"{r.get('p90_intvty', 0):.2f}",
            f"{r.get('p99_intvty', 0):.2f}",
            f"{r.get('p99.9_intvty', 0):.2f}",
            f"{r.get('median_e2el', 0):.2f}",
            f"{r.get('p90_e2el', 0):.2f}",
            f"{r.get('p99_e2el', 0):.2f}",
            f"{r.get('p99.9_e2el', 0):.2f}",
            f"{r.get('tput_per_gpu', 0):.2f}",
            f"{r.get('output_tput_per_gpu', 0):.2f}",
            f"{r.get('input_tput_per_gpu', 0):.2f}",
        ]
else:
    headers = [
        "model", "hw", "framework", "precision", "isl", "osl", "tp", "ep", "dp_attention",
        "conc",
        "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "p99.9_ttft_ms",
        "median_tpot_ms",
        "median_intvty", "p90_intvty", "p99_intvty", "p99.9_intvty",
        "median_e2el_s", "p90_e2el_s", "p99_e2el_s", "p99.9_e2el_s",
        "tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu",
    ]
    def row(r):
        return [
            r.get("model", ""), r.get("hw", ""), r.get("framework", ""),
            r.get("precision", ""), r.get("isl", ""), r.get("osl", ""),
            r.get("tp", ""), r.get("ep", ""), r.get("dp_attention", ""),
            r.get("conc", ""),
            f"{r.get('median_ttft', 0) * 1000:.2f}",
            f"{r.get('p90_ttft', 0) * 1000:.2f}",
            f"{r.get('p99_ttft', 0) * 1000:.2f}",
            f"{r.get('p99.9_ttft', 0) * 1000:.2f}",
            f"{r.get('median_tpot', 0) * 1000:.2f}",
            f"{r.get('median_intvty', 0):.2f}",
            f"{r.get('p90_intvty', 0):.2f}",
            f"{r.get('p99_intvty', 0):.2f}",
            f"{r.get('p99.9_intvty', 0):.2f}",
            f"{r.get('median_e2el', 0):.2f}",
            f"{r.get('p90_e2el', 0):.2f}",
            f"{r.get('p99_e2el', 0):.2f}",
            f"{r.get('p99.9_e2el', 0):.2f}",
            f"{r.get('tput_per_gpu', 0):.2f}",
            f"{r.get('output_tput_per_gpu', 0):.2f}",
            f"{r.get('input_tput_per_gpu', 0):.2f}",
        ]

results.sort(key=lambda r: r.get("conc", 0))

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(headers)
    for r in results:
        w.writerow(row(r))

print(f"CSV saved to: {csv_path}")

# Also print a simple table to stdout
col_widths = [max(len(str(headers[i])), *(len(str(row(r)[i])) for r in results)) for i in range(len(headers))]
fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
print()
print(fmt.format(*headers))
print("-+-".join("-" * w for w in col_widths))
for r in results:
    print(fmt.format(*row(r)))
PYEOF

echo ""
echo "========================================="
echo "  Results saved to: $CSV_OUT"
echo "========================================="
