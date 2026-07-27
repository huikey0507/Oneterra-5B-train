#!/usr/bin/env bash
#
# nccl_smoke_test.sh - 4 节点 NCCL 连通冒烟（不算训练，通常 1~3 分钟跑完）
# 用于排队昂贵前，先验证多机通信是否正常，避免等一天才发现网卡/P2P 配错。
#
# 用法（与 run3.sh 相同的环境变量）：
#   WORLD_SIZE=4 RANK=0 MASTER_ADDR=10.x.x.x MASTER_PORT=29500 TQ_GPU_NUM=8 \
#     bash runs/nccl_smoke_test.sh

set -euo pipefail

log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [$0]"

setup_cluster_nccl() {
    local user_ifname="${NCCL_SOCKET_IFNAME:-}"
    if [[ -n "$user_ifname" && "$user_ifname" != ^* ]]; then
        export NCCL_SOCKET_IFNAME="$user_ifname"
        echo -e "$log_format [NCCL] user NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
        return 0
    fi
    if [[ -n "${MASTER_ADDR:-}" && "${MASTER_ADDR}" != "localhost" && "${MASTER_ADDR}" != "127.0.0.1" ]]; then
        local _iface
        _iface=$(ip -o route get "${MASTER_ADDR}" 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") print $(i + 1)}' | head -1)
        if [[ -n "$_iface" ]]; then
            export NCCL_SOCKET_IFNAME="${_iface}"
            echo -e "$log_format [NCCL] auto NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} (route to MASTER_ADDR=${MASTER_ADDR})"
            return 0
        fi
    fi
    export NCCL_SOCKET_IFNAME="${user_ifname:-^docker0,lo,veth,virbr,cali,flannel,cni,tunl,dummy}"
    echo -e "$log_format [NCCL] WARNING: fallback NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
}

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-600}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-600}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

nproc_per_node="${TQ_GPU_NUM:-${GPU_PER_NODE:-8}}"
nnodes="${WORLD_SIZE:-4}"
node_rank="${RANK:-0}"
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-29500}"
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"

setup_cluster_nccl

echo -e "$log_format NCCL smoke: nnodes=$nnodes nproc_per_node=$nproc_per_node node_rank=$node_rank master=$master_addr:$master_port"
echo -e "$log_format hostname=$(hostname) ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo N/A)"

root_dir=$(realpath "$(dirname "$0")/..")
test_py="${root_dir}/runs/_nccl_smoke_test.py"
if [[ ! -f "$test_py" ]]; then
    echo -e "$log_format ERROR: missing $test_py" >&2
    exit 1
fi

torchrun \
    --nproc_per_node="$nproc_per_node" \
    --nnodes="$nnodes" \
    --node_rank="$node_rank" \
    --master_addr="$master_addr" \
    --master_port="$master_port" \
    "$test_py"

echo -e "$log_format NCCL smoke PASSED on node_rank=$node_rank"
