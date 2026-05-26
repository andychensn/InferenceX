#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 NVFP4 on B200 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - vLLM GPU KV only.
#   cpu     - vLLM native simple CPU offload.
#   lmcache - LMCache MP server + vLLM LMCacheMPConnector.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
PREFIX_CACHE_ARGS=()
LMCACHE_PID=""

cleanup_lmcache_server() {
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
}

trap cleanup_lmcache_server EXIT

wait_for_lmcache_ready() {
    { set +x; } 2>/dev/null
    local attempts="${LMCACHE_READY_ATTEMPTS:-120}"
    local tail_pid=""

    while [ ! -f "$LMCACHE_LOG" ]; do
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before creating log file. Exiting." >&2
            exit 1
        fi
        sleep 1
    done

    tail -f -n +1 "$LMCACHE_LOG" &
    tail_pid=$!

    for ((i = 1; i <= attempts; i++)); do
        if curl --output /dev/null --silent --fail "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            return 0
        fi
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before becoming healthy. Log follows:" >&2
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done

    echo "Timed out waiting for LMCache server healthcheck. Log follows:" >&2
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
    cat "$LMCACHE_LOG" >&2 || true
    exit 1
}

write_lmcache_cuda_mp_patch() {
    local patch_dir="$1"
    mkdir -p "$patch_dir"
    cat > "$patch_dir/sitecustomize.py" <<'PY'
"""Runtime compatibility for LMCache MP on CUDA Kimi MLA KV caches."""

import os
import threading

if os.environ.get("LMCACHE_CUDA_DEMAND_PINNED_ALLOCATOR") == "1":
    import builtins
    import sys

    _orig_import = builtins.__import__

    def _patch_lazy_memory_allocator(_lazy_memory_allocator) -> None:
        _LazyMemoryAllocator = _lazy_memory_allocator.LazyMemoryAllocator

        if getattr(_LazyMemoryAllocator, "_agentic_cuda_demand_patch", False):
            return

        _orig_init = _LazyMemoryAllocator.__init__
        _orig_allocate = _LazyMemoryAllocator.allocate
        _orig_batched_allocate = _LazyMemoryAllocator.batched_allocate

        def _expand_to(self, target_size: int) -> None:
            target_size = min(
                self._final_size,
                _lazy_memory_allocator.align_to(target_size, self.PIN_CHUNK_SIZE),
            )
            lock = self._agentic_cuda_demand_expand_lock
            with lock:
                if target_size <= self._curr_size:
                    return

                start_size = self._curr_size
                while self._curr_size < target_size:
                    commit_start = self._curr_size
                    commit_target = min(target_size, self._curr_size + self.COMMIT_SIZE)
                    while self._curr_size < commit_target:
                        self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
                        self._curr_size += self.PIN_CHUNK_SIZE
                    self._commit_expansion(self._curr_size - commit_start)

                self._log_expansion_progress(self._curr_size - start_size)

        def _retry_with_demand_expansion(self, allocate_once):
            obj = allocate_once()
            step_gb = float(os.environ.get("LMCACHE_CUDA_DEMAND_PINNED_STEP_GB", "64"))
            step_bytes = max(self.COMMIT_SIZE, int(step_gb * (1024**3)))

            while obj is None and self._curr_size < self._final_size:
                _expand_to(self, self._curr_size + step_bytes)
                obj = allocate_once()

            return obj

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._agentic_cuda_demand_expand_lock = threading.Lock()

            # LMCache MP's upstream LazyMemoryAllocator currently expands to
            # the final pinned size in a background thread. On CUDA Kimi TP4,
            # vLLM reaches KV-cache registration only after that 1.5 TB pool
            # is fully pinned, and the server-side IPC open path can stall
            # before acknowledging register_kv_caches. Keep the same final
            # capacity, but pin/commit extra host memory only when L1
            # allocations actually need it.
            self._stop_expand.set()
            self._expand_thread.join()
            _lazy_memory_allocator.logger.info(
                "Agentic CUDA patch: using demand-driven LMCache pinned "
                "memory expansion; final capacity remains %s MB",
                self._final_size >> 20,
            )

        def _patched_allocate(
            self,
            shapes,
            dtypes,
            fmt=_lazy_memory_allocator.MemoryFormat.UNDEFINED,
            allocator_type=None,
        ):
            return _retry_with_demand_expansion(
                self,
                lambda: _orig_allocate(self, shapes, dtypes, fmt, allocator_type),
            )

        def _patched_batched_allocate(
            self,
            shapes,
            dtypes,
            batch_size,
            fmt=_lazy_memory_allocator.MemoryFormat.UNDEFINED,
            allocator_type=None,
        ):
            return _retry_with_demand_expansion(
                self,
                lambda: _orig_batched_allocate(
                    self, shapes, dtypes, batch_size, fmt, allocator_type
                ),
            )

        _LazyMemoryAllocator.__init__ = _patched_init
        _LazyMemoryAllocator.allocate = _patched_allocate
        _LazyMemoryAllocator.batched_allocate = _patched_batched_allocate
        _LazyMemoryAllocator._agentic_cuda_demand_patch = True

    def _patch_l1_memory_manager(_memory_manager) -> None:
        _L1MemoryManager = getattr(_memory_manager, "L1MemoryManager", None)
        _LazyMemoryAllocator = getattr(_memory_manager, "LazyMemoryAllocator", None)
        if _L1MemoryManager is None or _LazyMemoryAllocator is None:
            return
        if getattr(_L1MemoryManager, "_agentic_cuda_final_capacity_patch", False):
            return

        _orig_get_memory_usage = _L1MemoryManager.get_memory_usage

        def _patched_get_memory_usage(self):
            allocator = getattr(self, "_allocator", None)
            if isinstance(allocator, _LazyMemoryAllocator):
                address_manager = allocator.get_address_manager()
                used_size = (
                    address_manager.get_heap_size() - address_manager.get_free_size()
                )
                return used_size, allocator._final_size
            return _orig_get_memory_usage(self)

        _L1MemoryManager.get_memory_usage = _patched_get_memory_usage
        _L1MemoryManager._agentic_cuda_final_capacity_patch = True

    def _maybe_patch_lazy_memory_allocator() -> None:
        module = sys.modules.get("lmcache.v1.lazy_memory_allocator")
        if module is not None and hasattr(module, "LazyMemoryAllocator"):
            _patch_lazy_memory_allocator(module)

    def _maybe_patch_l1_memory_manager() -> None:
        module = sys.modules.get("lmcache.v1.distributed.memory_manager")
        if module is not None and hasattr(module, "L1MemoryManager"):
            _patch_l1_memory_manager(module)

    def _agentic_cuda_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        if name == "lmcache.v1.lazy_memory_allocator" or (
            name.startswith("lmcache") and "lmcache.v1.lazy_memory_allocator" in sys.modules
        ):
            _maybe_patch_lazy_memory_allocator()
        if name == "lmcache.v1.distributed.memory_manager" or (
            name.startswith("lmcache")
            and "lmcache.v1.distributed.memory_manager" in sys.modules
        ):
            _maybe_patch_l1_memory_manager()
        return module

    builtins.__import__ = _agentic_cuda_import
    _maybe_patch_lazy_memory_allocator()
    _maybe_patch_l1_memory_manager()
PY
}

case "$OFFLOADING" in
    none)
        ;;
    cpu)
        # B200 DGXC nodes have ~2.7 TiB host DRAM; reserve 2.5 TB for the
        # simple offload connector and leave ~200 GB headroom for worker
        # RSS + page cache. Eager mode (the shortcut form default) is
        # intentional here per user request — Kimi FP4 on B200 has cleared
        # the full eager sweep before.
        #(srok), internal node limitation
        #TOTAL_CPU_DRAM_GB=2500
        TOTAL_CPU_DRAM_GB=1500
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    lmcache)
        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        agentic_pip_install --quiet --no-cache-dir lmcache
        LMCACHE_CUDA_PATCH_DIR="$RESULT_DIR/lmcache_cuda_patch"
        write_lmcache_cuda_mp_patch "$LMCACHE_CUDA_PATCH_DIR"
        export LMCACHE_CUDA_DEMAND_PINNED_ALLOCATOR=1
        export PYTHONPATH="$LMCACHE_CUDA_PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

        # Keep the semantic CPU KV pool at 2.5 TB for every TP shape. MP mode
        # owns that pool in the external LMCache server instead of passing
        # --kv-offloading-size through vLLM's integrated LMCache convenience
        # path, which divides the value by TP and then hits a large single-shot
        # cudaHostAlloc in LMCache 0.4.5's single-process local CPU backend.
        #(srok), internal node limitation
        #TOTAL_CPU_DRAM_GB=2500
        TOTAL_CPU_DRAM_GB=1500
        LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
        LMCACHE_PORT="${LMCACHE_PORT:-5555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
        # LMCacheMPConnector builds its ZMQ endpoint by concatenating
        # lmcache.mp.host and lmcache.mp.port, and its default host already
        # includes the tcp:// scheme. Keep the server bind host raw, but pass
        # a ZMQ-style host string to the connector.
        LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
        # Initial allocation is deliberately small; --l1-size-gb above is the
        # actual pool capacity and grows lazily as the run fills the cache.
        LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$TP}"
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

        echo "Starting LMCache MP server..."
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "$LMCACHE_MAX_WORKERS"
            --eviction-policy LRU
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
        printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!
        echo "LMCache server PID: $LMCACHE_PID"
        wait_for_lmcache_ready

        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu, lmcache)" >&2
        exit 1
        ;;
esac

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
# Disable vLLM v0.21+ CUDA-graph memory estimator. Its pre-reservation
# eats ~32% of HBM upfront which, combined with FP4 weights at TP=4
# (~62 GB/GPU), leaves no room for KV blocks -- _check_enough_kv_cache_memory
# trips before the engine starts. Our --gpu-memory-utilization=0.90 already
# leaves ~18 GB/GPU slack outside vLLM's budget, which is the same safety
# net the estimator provides, so disabling it is redundant rather than
# unsafe.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    --gpu-memory-utilization 0.90
    --max-num-seqs "$CONC"
    --reasoning-parser kimi_k2
    --tool-call-parser kimi_k2
    --compilation_config.pass_config.fuse_allreduce_rms true
    --kv-cache-dtype fp8
    --max-cudagraph-capture-size 2048
    --stream-interval 20
    --trust-remote-code
    "${PREFIX_CACHE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
