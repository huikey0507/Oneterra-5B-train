#!/usr/bin/env bash
# set -x
#
# run2.sh - 第三阶段多节点训练启动脚本
# 集群需提供环境变量：TQ_GPU_NUM（每节点GPU数）、WORLD_SIZE（节点数）、RANK（当前节点秩）、MASTER_ADDR、MASTER_PORT
# 示例：bash runs/run2.sh --modes train --config xsam/configs/.../s3_mixed_fineture_base/xxx.py --suffix v1

#######################################################################
#                          PART 1  Logging                             #
#######################################################################
log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [$0]"

#######################################################################
#                          PART 2  Project Config                      #
#######################################################################
root_dir=${root_dir:-$(realpath $(dirname $0)/../)}
code_name="xsam"
code_dir="$root_dir"
data_dir="$root_dir/datas"
init_dir="$root_dir/inits"
work_dir="${WORK_DIR:-$root_dir/wkdrs}"
if [[ "$work_dir" == "$root_dir/wkdrs" || "$work_dir" == "$root_dir/wkdrs/" ]]; then
    :
elif [[ -z "${WORK_DIR:-}" ]]; then
    work_dir="$root_dir/wkdrs"
fi
export ROOT_DIR="$root_dir/"
export DATA_DIR="$data_dir/"
export INIT_DIR="$init_dir/"
export WORK_DIR="$work_dir/"
export LMUData="$data_dir/LMUData"
export HF_HOME="$init_dir/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export XTUNER_DATASET_TIMEOUT=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_NET_GDR_LEVEL=2
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=72000
export DIST_TIMEOUT=72000
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-72000}"

# 多节点 NCCL 配置（跨节点通信）
# ---------------------------------------------------------------------------
# 若报错 "Connection closed by remote peer <IP>" / "remote process exited or network error"：
# （训练跑一段时间后出现，多为某一节点进程退出，常见为 OOM）
# 1) <IP> 对应的是「先挂掉的那台节点」。请到该节点上检查：
#    - OOM：dmesg | grep -i oom；nvidia-smi 看是否某卡 OOM
#    - 是否被调度器杀掉、磁盘满、网络中断
# 2) 网络：确认 NCCL_SOCKET_IFNAME 为实际通信网卡（如 eth0、bond0），
#    可通过 ip route get 10.107.254.x 查看出口接口，再 export NCCL_SOCKET_IFNAME=eth0
# 3) 可临时开启 NCCL_DEBUG=INFO 复现一次，便于定位是哪次 collective 与哪台节点断连
# 4) 若集群仅用 TCP（无 InfiniBand），可设 NCCL_IB_DISABLE=1 再试，避免 IB 路径超时
# ---------------------------------------------------------------------------
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE=0
export NCCL_BLOCKING_WAIT=0
export NCCL_TREE_THRESHOLD=0
# 使用 InfiniBand 时，可适当增大超时避免瞬断误报（单位秒，默认 18）
[ -z "${NCCL_IB_TIMEOUT:-}" ] && export NCCL_IB_TIMEOUT=23

# ProcessGroupNCCL watchdog：多节点 64 卡时易误报 480s 挂起，增大心跳超时或关闭监控
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
# 若仍报 watchdog hang，可取消下行注释以关闭 watchdog：export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
# 多节点断连时，让所有 rank 在同一 collective 上一起失败，便于定位（可选）
# export TORCH_NCCL_BLOCKING_WAIT=1

#######################################################################
#                          PART 3  Run Config                          #
#######################################################################
default_modes=("train" "segeval" "vlmeval" "visualize")

modes=()
config_file=""
suffix=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --modes|-m)
            shift
            if [[ -z "${1:-}" || "$1" == -* ]]; then
                echo "Error: --modes requires a value (comma-separated or space-separated)."
                exit 1
            fi
            if [[ $1 == *","* ]]; then
                IFS=',' read -ra modes <<< "$1"
            else
                modes+=("$1")
            fi
            ;;
        --config|-c)
            shift
            config_file="$1"
            ;;
        --suffix|-s)
            shift
            suffix="$1"
            ;;
        --work-dir|-w)
            shift
            work_dir="$1"
            ;;
        --help|-h)
            echo "Usage: $0 [--modes MODE1,MODE2,...] --config CONFIG_FILE [--work-dir WORK_DIR] [--suffix SUFFIX] [--help]"
            echo "Multi-node training. Cluster must set: TQ_GPU_NUM (or GPU_PER_NODE), WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT."
            echo "Arguments:"
            echo "  --modes, -m          Modes to run (default: train,segeval,vlmeval,visualize)"
            echo "  --config, -c        Config file path (REQUIRED)"
            echo "  --work-dir, -w      Work directory (optional)"
            echo "  --suffix, -s        Suffix for work directory (optional)"
            echo "  --help, -h          Show this help"
            echo "Example (multi-node):"
            echo "  WORLD_SIZE=4 RANK=0 MASTER_ADDR=master.example.com MASTER_PORT=29500 TQ_GPU_NUM=8 \\"
            echo "    bash runs/run2.sh --modes train --config xsam/configs/.../s3_mixed_fineture_base/xxx.py --suffix v1"
            exit 0
            ;;
        *)
            modes+=("$1")
            ;;
    esac
    shift
done

if [ -z "$config_file" ]; then
    echo "Error: --config/-c is required."
    exit 1
fi

if [ -n "$config_file" ]; then
    prefix=$(echo "$config_file" | grep -o 's[0-9]_[^/]*' | head -1)
    [ -z "$prefix" ] && prefix="s3_mixed_finetune"
else
    prefix="s3_mixed_finetune"
fi

[ ${#modes[@]} -eq 0 ] && modes=("${default_modes[@]}")
model_name=$(basename "$config_file" .py)

if [[ "$config_file" == *"llava"* ]]; then
    vlm_name="llava-phi3-siglip2-ft"
else
    vlm_name="xsam-phi3-siglip2-sam-l-mft"
fi

if [[ "$work_dir" == "$root_dir/wkdrs" || "$work_dir" == "$root_dir/wkdrs/" || \
      "$work_dir" == "$root_dir/checkpoints" || "$work_dir" == "$root_dir/checkpoints/" ]]; then
    suffix_norm=""
    if [[ -n "$suffix" ]]; then
        [[ "$suffix" == -* ]] && suffix_norm="$suffix" || suffix_norm="_$suffix"
    fi
    work_dir="$work_dir/$prefix/$model_name$suffix_norm"
    echo -e "$log_format Work directory: $work_dir"
fi

ckpt_file="$work_dir/pytorch_model.bin"

valid_modes=("train" "segeval" "vlmeval" "visualize" "demo")
for mode in "${modes[@]}"; do
    valid=0
    for valid_mode in "${valid_modes[@]}"; do
        [ "$mode" = "$valid_mode" ] && { valid=1; break; }
    done
    if [ $valid -eq 0 ]; then
        echo "Error: Invalid mode '$mode'. Valid: ${valid_modes[*]}"
        exit 1
    fi
done

echo -e "$log_format Running modes: ${modes[*]}"

# 多节点：每节点 GPU 数（优先 TQ_GPU_NUM，与集群约定一致）
nproc_per_node="${TQ_GPU_NUM:-${GPU_PER_NODE:-}}"
if [[ -z "$nproc_per_node" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        nproc_per_node=$(nvidia-smi -L | wc -l | tr -d ' ')
        [[ -z "$nproc_per_node" || "$nproc_per_node" -lt 1 ]] && nproc_per_node=1
    else
        nproc_per_node=1
    fi
    echo -e "$log_format Auto-detected nproc_per_node: $nproc_per_node"
else
    echo -e "$log_format Using nproc_per_node (TQ_GPU_NUM/GPU_PER_NODE): $nproc_per_node"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpu_count=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
    if [[ "$visible_gpu_count" -gt 0 && "$visible_gpu_count" -lt "$nproc_per_node" ]]; then
        echo -e "$log_format WARNING: CUDA_VISIBLE_DEVICES has $visible_gpu_count GPU(s), adjusting nproc_per_node"
        nproc_per_node=$visible_gpu_count
    fi
fi

# 多节点 rank/通信（由集群注入）
nnodes="${WORLD_SIZE:-1}"
node_rank="${RANK:-0}"
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-29500}"
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"

echo -e "$log_format Multi-node: nnodes=$nnodes, nproc_per_node=$nproc_per_node, node_rank=$node_rank, master=$master_addr:$master_port"
# 若报错里出现 "remote peer <IP>"，则 <IP> 为异常退出的节点；可在各节点查本机 IP 对照（便于排查 OOM/网络）
echo -e "$log_format Current node hostname=$(hostname), primary_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'N/A')"

# 代码目录：code_dir 为外层 xsam（config 里 code_dir + "xsam/configs/..." 会拼成 xsam/xsam/configs/...）
code_dir="$root_dir/$code_name"
xsam_code_root="$root_dir/$code_name/$code_name"
export CODE_DIR="$code_dir/"

# 在项目根下解析 config 为绝对路径，避免受当前目录影响
if [[ "$config_file" == /* ]]; then
    config_abs="$config_file"
else
    config_abs="$root_dir/$config_file"
fi
if [ ! -f "$config_abs" ]; then
    # 用户传 configs/...（无 xsam 前缀）-> 实际在 xsam/xsam/configs/...
    if [[ "$config_file" == configs/* ]]; then
        alt_abs="$xsam_code_root/$config_file"
        [ -f "$alt_abs" ] && config_abs="$alt_abs"
    fi
fi
if [ ! -f "$config_abs" ]; then
    # 兼容用户传 xsam/configs/... 而实际为 xsam/xsam/configs/...
    if [[ "$config_file" == "$code_name/"* ]]; then
        alt_abs="$xsam_code_root/${config_file#$code_name/}"
        [ -f "$alt_abs" ] && config_abs="$alt_abs"
    fi
fi
if [ ! -f "$config_abs" ]; then
    echo -e "$log_format Config not found: $config_file (tried $config_abs)" >&2
    exit 1
fi
echo -e "$log_format Using config: $config_abs"

# Run（始终在项目根目录执行，不 cd 到 xsam）
for mode in "${modes[@]}"; do
    cd "$root_dir"
    echo -e "$log_format Mode: $mode."
    time=$(date "+%Y%m%d-%H%M%S")
    if [ "$mode" = "train" ] && [ ! -d "$work_dir" ] && [ "$node_rank" = "0" ]; then
        mkdir -p "$work_dir"
        cp -rf $(realpath $0) "$code_dir" "$work_dir"
        find "$work_dir/$code_name" -name "*.crc" -type f -delete
        find "$work_dir/$code_name" -type f -exec chmod 664 {} +
        find "$work_dir/$code_name" -type d -exec chmod 775 {} +
    fi
    if [ "$mode" = "train" ] && [ -d "$work_dir/$code_name" ] && [ "$node_rank" = "0" ]; then
        cp $(realpath $0) "$work_dir"
    fi
    echo -e "$log_format work from root_dir: $root_dir (pwd: $(pwd))"

    trained_flag=0
    if [ "$mode" = "train" ]; then
        echo -e "$log_format Training $model_name (multi-node)."
        echo -e "$log_format NCCL_TIMEOUT=${NCCL_TIMEOUT}s, TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT}s"
        torchrun_args="--nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port"
        echo -e "$log_format torchrun $torchrun_args"
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun $torchrun_args \
            "$xsam_code_root/tools/train.py" \
            "$config_abs" \
            --work-dir "$work_dir" \
            --resume auto \
            --launcher pytorch \
            --deepspeed deepspeed_zero2 \
            --seed 1024 | { [ "$node_rank" = "0" ] && tee "$work_dir/${mode}-${time}.log" || cat; }
    fi
    [ -f "$ckpt_file" ] && trained_flag=1

    if [ "$mode" = "segeval" ] && [ $trained_flag = 1 ]; then
        echo -e "$log_format Evaluating Seg: $model_name."
        [ "$node_rank" -ne 0 ] && sleep 60
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port \
            "$xsam_code_root/tools/eval.py" \
            "$config_abs" \
            --launcher pytorch \
            --work-dir "$work_dir" \
            --seed 0 \
            --pth_model latest | { [ "$node_rank" = "0" ] && tee "$work_dir/${mode}-${time}.log" || cat; }
    fi

    if [ "$mode" = "vlmeval" ] && [ $trained_flag = 1 ]; then
        if [ "$node_rank" = "0" ] && [ ! -d "$work_dir/xtuner_model" ]; then
            echo -e "$log_format Converting $model_name to HF format."
            PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH \
                python "$xsam_code_root/tools/model_tools/pth_to_hf.py" "$config_abs" "$work_dir"
        fi
        rm -rf "$init_dir/$vlm_name"
        ln -sfn "$work_dir/xtuner_model" "$init_dir/$vlm_name"
        while [ ! -d "$work_dir/xtuner_model" ]; do
            echo -e "$log_format Waiting for HF conversion."
            sleep 5
        done
        if [ -d "$work_dir/xtuner_model" ]; then
            echo -e "$log_format Evaluating VLM: $model_name."
            [ "$node_rank" -ne 0 ] && sleep 30
            PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                torchrun --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port \
                "$xsam_code_root/evaluation/vlmeval/run.py" \
                --data MME MMBench_DEV_EN SEEDBench_IMG POPE AI2D_TEST \
                --model "$vlm_name" \
                --work-dir "$work_dir/vlmeval_results" | { [ "$node_rank" = "0" ] && tee "$work_dir/${mode}-${time}.log" || cat; }
        fi
    fi

    if [ "$mode" = "visualize" ] && [ $trained_flag = 1 ] && [ "$node_rank" = "0" ]; then
        echo -e "$log_format Visualizing $model_name."
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python "$xsam_code_root/tools/visualize.py" "$config_abs" --work-dir "$work_dir" --seed 0 --pth_model latest | tee "$work_dir/${mode}-${time}.log"
    fi

    if [ "$mode" = "demo" ] && [ $trained_flag = 1 ] && [ "$node_rank" = "0" ]; then
        echo -e "$log_format Demoing $model_name."
        mkdir -p "$work_dir/app_logs"
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python "$xsam_code_root/demo/app.py" "$config_abs" --work-dir "$work_dir" --log-dir "$work_dir/app_logs" --pth_model latest --seed 0 --port 7862 | tee "$work_dir/${mode}-${time}.log"
    fi
    rm -rf /tmp/xsam_cache 2>/dev/null
done
