#!/usr/bin/env bash
# =============================================================================
# runD2.sh — Manual decode node 2 launcher (no Slurm)
#
# Run this on DECODE NODE 2. It starts a decode sglang server, waits for the
# router on the prefill node to come up, then waits for it to shut down
# (signaling end of benchmark) before cleaning up.
#
# Usage:
#   1. Edit the configuration block below (must match runP.sh and runD1.sh)
#   2. Run this script on decode node 2
# =============================================================================
set -euo pipefail

# =============================================================================
# >>> CONFIGURATION — edit these values for your cluster <<<
# =============================================================================

# Node IPs (must be reachable from all 3 nodes)
NODE_P_IP="10.21.9.8"       # This (prefill) node's IP
NODE_D1_IP="10.21.9.29"      # Decode node 1 IP
NODE_D2_IP="10.21.9.34"      # Decode node 2 IP

# Model
MODEL_REPO="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8"
MODEL_SNAPSHOT_HASH="ea5b4f81096f3901c91dea97f81324302495781d"
MODEL_NAME="Qwen3.5-397B-A17B-FP8"

# Container image
IMAGE="sglang-rocm-mi300x-rdma:latest"  # Built from Dockerfile.mi300x-rdma

# Hardware
GPUS_PER_NODE=8

# Parallelism (matches amd-master.yaml qwen3.5-fp8-mi3*x-sglang-disagg)
PREFILL_TP_SIZE=8
PREFILL_ENABLE_EP=false
PREFILL_ENABLE_DP=false
DECODE_TP_SIZE=8
DECODE_ENABLE_EP=false
DECODE_ENABLE_DP=false
DECODE_MTP_SIZE=0

# Benchmark (needed by server.sh even on decode nodes)
ISL=1024
OSL=1024
CONCURRENCIES="8x16x32x64x128x256x512"
REQUEST_RATE="inf"
RANDOM_RANGE_RATIO=1
NUM_PROMPTS_MULTIPLIER=10

# Disagg topology: 1 prefill worker, 2 decode workers
xP=1
yD=1

# Debug: set to 1 to print commands without running them
DRY_RUN=0

IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"

# Sync barrier port (change if 5000 is already in use on your nodes)
SYNC_BARRIER_PORT=5050

# =============================================================================
# End of configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="manual-$(date +%s)"
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

DOCKER_MOUNT_PATH="/workspace"
SGLANG_WS_PATH="${DOCKER_MOUNT_PATH}/benchmarks/multi_node/amd_utils"
if [[ "$yD" -eq 1 ]]; then
    IPADDRS="${NODE_P_IP},${NODE_D1_IP}"
elif [[ "$yD" -eq 2 ]]; then
    IPADDRS="${NODE_P_IP},${NODE_D1_IP},${NODE_D2_IP}"
else
    echo "ERROR: unsupported yD=$yD (expected 1 or 2)" >&2; exit 1
fi
DOCKER_CONT_NAME="sglang-disagg-decode2-${RUN_ID}"

# MoRI conn.py patch overlay
EXTRA_DOCKER_MOUNTS=""
_MORI_PATCH_FILE="${REPO_ROOT}/benchmarks/multi_node/amd_utils/patches/mori_conn.py"
_MORI_PATCH_TARGET="/sgl-workspace/sglang/python/sglang/srt/disaggregation/mori/conn.py"
if [[ -f "$_MORI_PATCH_FILE" ]]; then
    EXTRA_DOCKER_MOUNTS="-v ${_MORI_PATCH_FILE}:${_MORI_PATCH_TARGET}:ro"
    echo "[runD2] Auto-applied MoRI conn.py overlay"
fi

# NODE_RANK=2: second decode node in a 1P+2D topology
NODE_RANK=2

echo "============================================="
echo "  Decode Node 2 — run_id=${RUN_ID}"
echo "  This node IP : ${NODE_D2_IP}"
echo "  Prefill IP   : ${NODE_P_IP}"
echo "  Model        : ${MODEL_REPO} -> /models/${MODEL_NAME}"
echo "  Image        : ${IMAGE}"
echo "  NODE_RANK    : ${NODE_RANK}"
echo "  DRY_RUN      : ${DRY_RUN}"
echo "  Logs         : ${LOG_DIR}"
echo "============================================="

docker run --rm \
    --init \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 128G \
    -v "${MODEL_REPO}:/models/${MODEL_NAME}-repo" \
    -v "${REPO_ROOT}:${DOCKER_MOUNT_PATH}" \
    -v "${LOG_DIR}:/run_logs/slurm_job-${RUN_ID}" \
    ${EXTRA_DOCKER_MOUNTS} \
    -e SLURM_JOB_ID="${RUN_ID}" \
    -e NODE_RANK="${NODE_RANK}" \
    -e NODE0_ADDR="${NODE_P_IP}" \
    -e MODEL_DIR=/models \
    -e MODEL_NAME="${MODEL_NAME}" \
    -e SGLANG_WS_PATH="${SGLANG_WS_PATH}" \
    -e GPUS_PER_NODE="${GPUS_PER_NODE}" \
    -e xP="${xP}" \
    -e yD="${yD}" \
    -e IPADDRS="${IPADDRS}" \
    -e PREFILL_TP_SIZE="${PREFILL_TP_SIZE}" \
    -e PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP}" \
    -e PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP}" \
    -e DECODE_TP_SIZE="${DECODE_TP_SIZE}" \
    -e DECODE_ENABLE_EP="${DECODE_ENABLE_EP}" \
    -e DECODE_ENABLE_DP="${DECODE_ENABLE_DP}" \
    -e DECODE_MTP_SIZE="${DECODE_MTP_SIZE}" \
    -e ENABLE_DISAGG_DECODE_PARALLELISM_FLAGS=false \
    -e BENCH_INPUT_LEN="${ISL}" \
    -e BENCH_OUTPUT_LEN="${OSL}" \
    -e BENCH_RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO}" \
    -e BENCH_NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER}" \
    -e BENCH_MAX_CONCURRENCY="${CONCURRENCIES}" \
    -e BENCH_REQUEST_RATE="${REQUEST_RATE}" \
    -e DRY_RUN="${DRY_RUN}" \
    -e TQDM_MININTERVAL=20 \
    -e SYNC_BARRIER_PORT="${SYNC_BARRIER_PORT}" \
    ${IBDEVICES:+-e IBDEVICES="${IBDEVICES}"} \
    ${MORI_RDMA_TC:+-e MORI_RDMA_TC="${MORI_RDMA_TC}"} \
    --name "${DOCKER_CONT_NAME}" \
    "${IMAGE}" bash -lc "
        set -o pipefail
        ln -sfn /models/${MODEL_NAME}-repo/snapshots/${MODEL_SNAPSHOT_HASH} /models/${MODEL_NAME}
        mkdir -p /run_logs/slurm_job-${RUN_ID}
        ${SGLANG_WS_PATH}/server.sh 2>&1 | tee /run_logs/slurm_job-${RUN_ID}/server_decode2.log
    "

echo "Decode node 2 finished. Logs in ${LOG_DIR}"
