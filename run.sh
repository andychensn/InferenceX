export MODEL="/dev/shm/DeepSeek-V4-Pro/"
export TP=8
export CONC=2
export DP_ATTENTION=false
export ISL=8192
export OSL=1024
export RANDOM_RANGE_RATIO=1.0
export RESULT_FILENAME="dsv4_fp4_mi355x_sglang_tp16_conc16_dp_attention_isl8192_osl1024_random_range_ratio0.8"
export EP_SIZE=1
rm /workspace/profiles/*
#export EVAL_ONLY=true
#export RUN_EVAL=true

export SGLANG_DEBUG_DSV4_ATTN=1
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/profiles}"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
pkill -9 tail
pkill -9 python
pkill -9 sglang

bash benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh
