#!/usr/bin/env bash
# set -x

#######################################################################
#                          PART 1  Logging                             #
#######################################################################
# Log format
log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [$0]"

#######################################################################
#                          PART 2  Project Config                      #
#######################################################################
# Directory
root_dir=${root_dir:-$(realpath $(dirname $0)/../)}
code_name="xsam"
code_dir="$root_dir/$code_name"
data_dir="$root_dir/datas"
init_dir="$root_dir/inits"
work_dir="${WORK_DIR:-$root_dir/wkdrs}"
# 如果 WORK_DIR 环境变量未设置，使用默认值 wkdrs
if [[ "$work_dir" == "$root_dir/wkdrs" || "$work_dir" == "$root_dir/wkdrs/" ]]; then
    # 保持原值，后续会构建子目录
    :
elif [[ -z "${WORK_DIR:-}" ]]; then
    # 如果用户没有通过环境变量或参数指定，使用默认的 wkdrs
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
export NCCL_TIMEOUT=72000  # 20小时超时（秒）
export DIST_TIMEOUT=72000  # PyTorch分布式超时时间（秒）
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-72000}"  # torchrun rendezvous超时（秒）

# NCCL基础配置（将在后面根据节点数调整）
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

#######################################################################
#                          PART 3  Run Config                          #
#######################################################################
# Default modes
default_modes=("train" "segeval" "vlmeval" "visualize")

# Parse command line arguments
modes=()
config_file=""
suffix=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --modes|-m)
            shift
            # Parse modes from comma-separated string or space-separated arguments
            if [[ -z "${1:-}" || "$1" == -* ]]; then
                echo "Error: --modes requires a value (comma-separated or space-separated)."
                exit 1
            fi
            if [[ $1 == *","* ]]; then
                IFS=',' read -ra modes <<< "$1"
            else
                # If no comma, treat the next argument as a single mode
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
            echo "Available modes: train, segeval, vlmeval, visualize, demo"
            echo "Arguments:"
            echo "  --modes, -m          Specify modes to run (comma-separated or space-separated)"
            echo "  --config, -c         Specify config file path (REQUIRED)"
            echo "  --work-dir, -w       Specify work directory path (optional)"
            echo "  --suffix, -s         Specify suffix for work directory (optional)"
            echo "  --help, -h           Show this help message"
            echo "Examples:"
            echo "  $0 --config path/to/config.py                    # Run all modes with specified config"
            echo "  $0 --config config.py --modes train             # Run only training"
            echo "  $0 --config config.py --modes train,segeval     # Run training and segmentation evaluation"
            echo "  $0 --config config.py --work-dir /path/to/work   # Run with custom work directory"
            echo "  $0 --config config.py --suffix test             # Run with suffix 'test'"
            echo "  $0 --config config.py --modes demo --work-dir /path/to/work  # Launch local Gradio demo (requires checkpoint in work-dir)"
            exit 0
            ;;
        *)
            # If no recognized flag, treat as mode
            modes+=("$1")
            ;;
    esac
    shift
done

# Validate config_file is provided
if [ -z "$config_file" ]; then
    echo "Error: --config/-c parameter is required. Please specify a config file."
    echo "Usage: $0 [--modes MODE1,MODE2,...] --config CONFIG_FILE [--work-dir WORK_DIR] [--suffix SUFFIX] [--help]"
    exit 1
fi

# Extract prefix from config file path
if [ -n "$config_file" ]; then
    # Extract the stage name (s1, s2, s3, etc.) from config file path
    prefix=$(echo "$config_file" | grep -o 's[0-9]_[^/]*' | head -1)
    if [ -z "$prefix" ]; then
        # Fallback to default if no stage found in path
        prefix="s3_mixed_finetune"
    fi
else
    prefix="s3_mixed_finetune"
fi

# If no modes specified, use defaults
if [ ${#modes[@]} -eq 0 ]; then
    modes=("${default_modes[@]}")
fi
model_name=$(basename "$config_file" .py)

# Set vlm_name based on config_file content
if [[ "$config_file" == *"llava"* ]]; then
    vlm_name="llava-phi3-siglip2-ft"
else
    vlm_name="xsam-phi3-siglip2-sam-l-mft"
fi

# 如果 work_dir 是默认的 wkdrs 或 checkpoints 目录，自动构建子目录结构
# 如果用户通过 --work-dir 指定了具体路径，则不自动构建子目录
if [[ "$work_dir" == "$root_dir/wkdrs" || "$work_dir" == "$root_dir/wkdrs/" || \
      "$work_dir" == "$root_dir/checkpoints" || "$work_dir" == "$root_dir/checkpoints/" ]]; then
    suffix_norm=""
    if [[ -n "$suffix" ]]; then
        if [[ "$suffix" == -* ]]; then
            suffix_norm="$suffix"
        else
            suffix_norm="_$suffix"
        fi
    fi
    work_dir="$work_dir/$prefix/$model_name$suffix_norm"
    echo -e "$log_format Work directory will be: $work_dir"
fi

ckpt_file="$work_dir/pytorch_model.bin"

# Validate modes
valid_modes=("train" "segeval" "vlmeval" "visualize" "demo")
for mode in "${modes[@]}"; do
    valid=0
    for valid_mode in "${valid_modes[@]}"; do
        if [ "$mode" = "$valid_mode" ]; then
            valid=1
            break
        fi
    done
    if [ $valid -eq 0 ]; then
        echo "Error: Invalid mode '$mode'. Valid modes are: ${valid_modes[*]}"
        exit 1
    fi
done

echo -e "$log_format Running modes: ${modes[*]}"

# 检测GPU配置
gpu_per_node="${GPU_PER_NODE:-}"
gpu_per_node_set_explicitly=0
if [[ -z "$gpu_per_node" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_per_node=$(nvidia-smi -L | wc -l | tr -d ' ')
        [[ -z "$gpu_per_node" || "$gpu_per_node" -lt 1 ]] && gpu_per_node=1
    else
        gpu_per_node=1
    fi
    echo -e "$log_format Auto-detected GPU count: $gpu_per_node"
else
    gpu_per_node_set_explicitly=1
    echo -e "$log_format Using GPU_PER_NODE environment variable: $gpu_per_node"
fi

# 检查 CUDA_VISIBLE_DEVICES 是否已设置，如果设置了，确保 gpu_per_node 不超过可见GPU数量
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # 计算 CUDA_VISIBLE_DEVICES 中的GPU数量
    visible_gpu_count=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
    echo -e "$log_format CUDA_VISIBLE_DEVICES is set to: ${CUDA_VISIBLE_DEVICES} (contains $visible_gpu_count GPU(s))"
    if [[ "$visible_gpu_count" -gt 0 && "$visible_gpu_count" -lt "$gpu_per_node" ]]; then
        echo -e "$log_format WARNING: CUDA_VISIBLE_DEVICES contains $visible_gpu_count GPU(s), but gpu_per_node=$gpu_per_node"
        echo -e "$log_format Adjusting gpu_per_node from $gpu_per_node to $visible_gpu_count"
        gpu_per_node=$visible_gpu_count
    fi
else
    # 如果用户显式设置了 GPU_PER_NODE，自动设置 CUDA_VISIBLE_DEVICES 为前 N 个GPU
    if [[ "$gpu_per_node_set_explicitly" -eq 1 ]]; then
        # 生成 CUDA_VISIBLE_DEVICES 字符串：0,1,2,...,gpu_per_node-1
        cuda_visible_devices_list=()
        for ((i=0; i<gpu_per_node; i++)); do
            cuda_visible_devices_list+=("$i")
        done
        export CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${cuda_visible_devices_list[*]}")
        echo -e "$log_format Auto-setting CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES} (to match GPU_PER_NODE=$gpu_per_node)"
    else
        echo -e "$log_format CUDA_VISIBLE_DEVICES is not set, will use all available GPUs"
        # 如果未设置 CUDA_VISIBLE_DEVICES，明确取消设置以确保使用所有GPU
        unset CUDA_VISIBLE_DEVICES
    fi
fi

echo -e "$log_format Final GPU configuration: gpu_per_node=$gpu_per_node"

# 单节点配置（固定设置）
nnodes=1
master_port="${MASTER_PORT:-29500}"
node_rank=0
master_addr="localhost"
export MASTER_ADDR="localhost"

# 单节点训练：优先使用本地通信，避免网络socket问题
export NCCL_IB_DISABLE=1  # 单节点禁用InfiniBand
export NCCL_P2P_DISABLE=0  # 启用P2P通信（NVLink）
export NCCL_SHM_DISABLE=0  # 启用共享内存（单节点最佳选择）
unset NCCL_SOCKET_IFNAME  # 单节点不需要网络接口
export NCCL_BLOCKING_WAIT=1
export NCCL_TREE_THRESHOLD=0

echo -e "$log_format Single-node training: Using local communication (SHM + NVLink)"
echo -e "$log_format Training configuration: nnodes=$nnodes, gpu_per_node=$gpu_per_node, master_addr=$master_addr:$master_port"

# Run
for mode in "${modes[@]}"
do
    cd $root_dir
    echo -e "$log_format Mode: $mode."
    time=$(date "+%Y%m%d-%H%M%S")
    if [ $mode = "train" ] && [ ! -d "$work_dir" ]; then
        mkdir -p $work_dir
        cp -rf $(realpath $0) $code_dir $work_dir
        find "$work_dir/$code_name" -name "*.crc" -type f -delete
        find "$work_dir/$code_name" -type f -exec chmod 664 {} +
        find "$work_dir/$code_name" -type d -exec chmod 775 {} +
    fi
    if [ -d "$work_dir/$code_name" ]; then
        code_dir="$work_dir/$code_name"
        cp $(realpath $0) $work_dir
    fi
    cd $code_dir
    export CODE_DIR="$code_dir/"
    echo -e "$log_format code_dir: $code_dir"

    # 处理配置文件路径：支持多种路径格式
    # 如果配置文件路径以 code_name/ 开头，去掉该前缀
    if [[ "$config_file" == "$code_name/"* ]]; then
        config_file="${config_file#$code_name/}"
    fi
    # 如果配置文件路径以 ./code_name/ 开头，去掉该前缀
    if [[ "$config_file" == "./$code_name/"* ]]; then
        config_file="${config_file#./$code_name/}"
    fi
    # 如果配置文件是绝对路径，检查是否存在
    if [[ "$config_file" == /* ]]; then
        if [ ! -f "$config_file" ]; then
            echo -e "$log_format Config file not found: $config_file" >&2
            exit 1
        fi
    else
        # 相对路径：先尝试直接使用，如果不存在则尝试去掉 code_name/ 前缀
        if [ ! -f "$config_file" ]; then
            # 如果路径中包含 code_name/，尝试去掉
            if [[ "$config_file" == *"$code_name/"* ]]; then
                temp_config="${config_file#*$code_name/}"
                if [ -f "$temp_config" ]; then
                    config_file="$temp_config"
                fi
            fi
        fi
        # 最终检查
        if [ ! -f "$config_file" ]; then
            echo -e "$log_format Config file not found: $config_file (searched in: $(pwd))" >&2
            echo -e "$log_format Please check the config file path." >&2
            exit 1
        fi
    fi
    echo -e "$log_format Using config file: $config_file"
    
    # mode: train
    trained_flag=0
    if [ $mode = "train" ]; then
        echo -e "$log_format Training $model_name."
        echo -e "$log_format NCCL_TIMEOUT=${NCCL_TIMEOUT}s, DIST_TIMEOUT=${DIST_TIMEOUT}s"
        # 单节点训练配置
        torchrun_args="--nnodes=1 --nproc_per_node=$gpu_per_node --master_addr=$master_addr --master_port=$master_port"
        echo -e "$log_format Single-node training configuration:"
        echo -e "$log_format   - nnodes: 1"
        echo -e "$log_format   - nproc_per_node: $gpu_per_node"
        echo -e "$log_format   - master_addr: $master_addr"
        echo -e "$log_format   - master_port: $master_port"
        echo -e "$log_format torchrun command: torchrun $torchrun_args"
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun $torchrun_args \
            $code_dir/xsam/tools/train.py \
            $config_file \
            --work-dir $work_dir \
            --resume auto \
            --launcher pytorch \
            --deepspeed deepspeed_zero2 \
            --seed 1024 | tee $work_dir/${mode}-${time}.log
    fi
    # Check if training completed successfully
    if [ -f $ckpt_file ]; then
        trained_flag=1
    fi
    # mode: segeval
    if [ $mode = "segeval" ] && [ $trained_flag = 1 ]; then
        echo -e "$log_format Evaluating Seg: $model_name."
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --master_addr=$master_addr --master_port=$master_port --nproc_per_node=$gpu_per_node \
            $code_dir/xsam/tools/eval.py \
            $config_file \
            --launcher pytorch \
            --work-dir $work_dir \
            --seed 0 \
            --pth_model latest | tee $work_dir/${mode}-${time}.log
    fi
    # mode: vlmeval
    if [ $mode = "vlmeval" ] && [ $trained_flag = 1 ]; then
        if [ ! -d "$work_dir/xtuner_model" ]; then
            echo -e "$log_format Converting $model_name to HF format."
            PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH \
                python $code_dir/xsam/tools/model_tools/pth_to_hf.py \
                $code_dir/$config_file \
                $work_dir
        fi
        # Remove existing target and create/refresh symlink safely
        rm -rf "$init_dir/$vlm_name"
        ln -sfn "$work_dir/xtuner_model" "$init_dir/$vlm_name"
        while [ ! -d "$work_dir/xtuner_model" ]; do
            echo -e "$log_format Waiting for $model_name to be converted to HF format."
            sleep 5
        done
        if [ -d "$work_dir/xtuner_model" ]; then
            echo -e "$log_format Evaluating VLM: $model_name."
            PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                torchrun --master_addr=$master_addr --master_port=$master_port --nproc_per_node=$gpu_per_node \
                $code_dir/xsam/evaluation/vlmeval/run.py \
                --data MME MMBench_DEV_EN SEEDBench_IMG POPE AI2D_TEST \
                --model $vlm_name \
                --work-dir $work_dir/vlmeval_results | tee "$work_dir/${mode}-${time}.log"
        fi
    fi
    # mode: visualize
    if [ $mode = "visualize" ] && [ $trained_flag = 1 ]; then
        echo -e "$log_format Visualizing $model_name."
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python $code_dir/xsam/tools/visualize.py \
            $config_file \
            --work-dir $work_dir \
            --seed 0 \
            --pth_model latest | tee $work_dir/${mode}-${time}.log
    fi
    # mode: demo
    if [ $mode = "demo" ] && [ $trained_flag = 1 ]; then
        echo -e "$log_format Demoing $model_name."
        mkdir -p "$work_dir/app_logs"
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python $code_dir/xsam/demo/app.py \
            $config_file \
            --work-dir $work_dir \
            --log-dir $work_dir/app_logs \
            --pth_model latest \
            --seed 0 \
            --port 7862 | tee $work_dir/${mode}-${time}.log
    fi
    rm -rf /tmp/xsam_cache > /dev/null 2>&1
done