#!/bin/bash
# X-SAM Stage 3 恢复训练脚本
# 用于从中断的checkpoint继续训练

set -e

# 配置路径
X_SAM_DIR="/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM"
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py"
DEEPSPEED_CONFIG="deepspeed_configs/ds_config_zero2.json"

# 默认checkpoint路径
DEFAULT_CHECKPOINT="${X_SAM_DIR}/wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_2000.pth"
LAST_CHECKPOINT_FILE="${X_SAM_DIR}/wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/last_checkpoint"

echo "=========================================="
echo "X-SAM Stage 3 恢复训练脚本"
echo "=========================================="
echo ""

# 切换到项目目录
cd "${X_SAM_DIR}"

# 查找最新的checkpoint
RESUME_CHECKPOINT=""

# 首先尝试读取last_checkpoint文件
if [ -f "${LAST_CHECKPOINT_FILE}" ]; then
    RESUME_CHECKPOINT=$(cat "${LAST_CHECKPOINT_FILE}" | tr -d '\n')
    # 如果是绝对路径，转换为相对路径
    if [[ "${RESUME_CHECKPOINT}" == "${X_SAM_DIR}"* ]]; then
        RESUME_CHECKPOINT=$(echo "${RESUME_CHECKPOINT}" | sed "s|${X_SAM_DIR}/||")
    fi
    echo "📋 从last_checkpoint文件读取: ${RESUME_CHECKPOINT}"
fi

# 如果last_checkpoint不存在或路径无效，使用默认路径
if [ -z "${RESUME_CHECKPOINT}" ] || [ ! -d "${X_SAM_DIR}/${RESUME_CHECKPOINT}" ]; then
    if [ -d "${DEFAULT_CHECKPOINT}" ]; then
        RESUME_CHECKPOINT="wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_2000.pth"
        echo "📋 使用默认checkpoint: ${RESUME_CHECKPOINT}"
    else
        echo "❌ 错误: 未找到可用的checkpoint"
        echo "   请检查以下路径:"
        echo "   - ${DEFAULT_CHECKPOINT}"
        echo "   - ${LAST_CHECKPOINT_FILE}"
        exit 1
    fi
fi

# 检查checkpoint是否存在
if [ ! -d "${X_SAM_DIR}/${RESUME_CHECKPOINT}" ]; then
    echo "❌ 错误: Checkpoint不存在: ${X_SAM_DIR}/${RESUME_CHECKPOINT}"
    exit 1
fi

echo "✅ Checkpoint存在: ${X_SAM_DIR}/${RESUME_CHECKPOINT}"
echo ""

# 检查配置文件
if [ ! -f "${X_SAM_DIR}/${CONFIG_FILE}" ]; then
    echo "❌ 错误: 配置文件不存在: ${X_SAM_DIR}/${CONFIG_FILE}"
    exit 1
fi

echo "✅ 配置文件存在: ${CONFIG_FILE}"
echo "📂 工作目录: ${X_SAM_DIR}"
echo "🔄 恢复checkpoint: ${RESUME_CHECKPOINT}"
echo ""

# 设置环境变量
export ROOT_DIR="${X_SAM_DIR}/"
export DATA_DIR="${X_SAM_DIR}/datas/"
export INIT_DIR="${X_SAM_DIR}/inits/"
export WORK_DIR="${X_SAM_DIR}/wkdrs/"
export LMUData="${X_SAM_DIR}/datas/LMUData"
export HF_HOME="${X_SAM_DIR}/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export XTUNER_DATASET_TIMEOUT=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=2
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=36000  # 10小时超时（秒）
export DIST_TIMEOUT=36000  # PyTorch分布式超时时间（秒）
# NCCL 优化环境变量
export NCCL_IB_DISABLE=0  # 启用 InfiniBand（如果可用）
export NCCL_IB_GID_INDEX=3  # InfiniBand GID索引
export NCCL_SOCKET_IFNAME=^docker0,lo  # 排除docker和loopback接口
export NCCL_DEBUG=WARN  # 减少NCCL日志输出（可选：INFO用于调试）
export NCCL_P2P_DISABLE=0  # 启用P2P通信
export NCCL_SHM_DISABLE=0  # 启用共享内存
export NCCL_TREE_THRESHOLD=0  # 使用tree算法（适用于多GPU）

# 确定工作目录
WORK_DIR_PATH="${X_SAM_DIR}/wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3"
CODE_DIR="${WORK_DIR_PATH}/xsam"

# 检查code_dir是否存在，如果不存在则使用原始路径
if [ ! -d "${CODE_DIR}" ]; then
    CODE_DIR="${X_SAM_DIR}/xsam/xsam"
fi

# 设置CODE_DIR环境变量（配置文件需要这个来解析评估图片路径）
export CODE_DIR="${CODE_DIR}/"

# 获取GPU数量（默认使用所有可用GPU）
GPU_PER_NODE=${GPU_PER_NODE:-$(nvidia-smi -L | wc -l)}
if [ -z "$GPU_PER_NODE" ] || [ "$GPU_PER_NODE" -eq 0 ]; then
    GPU_PER_NODE=1
fi

# 构建训练命令
echo "🚀 启动恢复训练..."
echo "GPU数量: ${GPU_PER_NODE}"
echo "工作目录: ${WORK_DIR_PATH}"
echo "代码目录: ${CODE_DIR}"
echo ""

# 确定训练脚本路径
if [ -f "${CODE_DIR}/xsam/tools/train.py" ]; then
    TRAIN_SCRIPT="${CODE_DIR}/xsam/tools/train.py"
elif [ -f "${CODE_DIR}/tools/train.py" ]; then
    TRAIN_SCRIPT="${CODE_DIR}/tools/train.py"
elif [ -f "${X_SAM_DIR}/xsam/xsam/tools/train.py" ]; then
    TRAIN_SCRIPT="${X_SAM_DIR}/xsam/xsam/tools/train.py"
else
    echo "❌ 错误: 找不到训练脚本 train.py"
    echo "   尝试的路径:"
    echo "   - ${CODE_DIR}/xsam/tools/train.py"
    echo "   - ${CODE_DIR}/tools/train.py"
    echo "   - ${X_SAM_DIR}/xsam/xsam/tools/train.py"
    exit 1
fi

# 切换到代码目录
cd "${CODE_DIR}"

# 设置PYTHONPATH
export PYTHONPATH="$(realpath ${CODE_DIR}):${PYTHONPATH}"

# 使用torchrun启动训练
PYTHONPATH="${PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun \
    --nproc_per_node=${GPU_PER_NODE} \
    "${TRAIN_SCRIPT}" \
    "${X_SAM_DIR}/${CONFIG_FILE}" \
    --work-dir "${WORK_DIR_PATH}" \
    --launcher pytorch \
    --deepspeed deepspeed_zero2 \
    --resume "${X_SAM_DIR}/${RESUME_CHECKPOINT}" \
    --seed 1024

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练完成！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

