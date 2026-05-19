#!/bin/bash
# X-SAM Stage 3 混合微调训练脚本（带优化）
# 包含官方配置优化点：SourceGroupedSampler、NCCL优化等

set -e

# 配置路径
X_SAM_DIR="/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM"
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat.py"
WORK_DIR_PATH="${X_SAM_DIR}/work_dirs/s3_mixed_finetune_geochat_optim"

echo "=========================================="
echo "X-SAM Stage 3 混合微调训练（优化版）"
echo "=========================================="
echo ""

# 切换到项目目录
cd "${X_SAM_DIR}"

# 检查配置文件
if [ ! -f "${X_SAM_DIR}/${CONFIG_FILE}" ]; then
    echo "❌ 错误: 配置文件不存在: ${X_SAM_DIR}/${CONFIG_FILE}"
    exit 1
fi

echo "✅ 配置文件存在: ${CONFIG_FILE}"
echo "📂 工作目录: ${WORK_DIR_PATH}"
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
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# NCCL 优化环境变量（官方配置优化点）
export NCCL_ALGO=Tree
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1200000  # 20分钟超时（毫秒）
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_DISABLE=0  # 启用 InfiniBand（如果可用）
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export DIST_TIMEOUT=1200  # PyTorch分布式超时时间（秒）

# 确定代码目录（使用绝对路径）
CODE_DIR="${X_SAM_DIR}/xsam/xsam"
CODE_DIR_ABS="$(cd "${CODE_DIR}" && pwd)"

# 设置CODE_DIR环境变量（配置文件需要这个来解析评估图片路径）
export CODE_DIR="${CODE_DIR_ABS}/"

# 获取GPU数量（优先使用环境变量，否则使用4个GPU）
if [ -z "$GPU_PER_NODE" ]; then
    # 默认使用4个GPU进行训练
    GPU_PER_NODE=4
    echo "🔍 使用默认GPU数量: ${GPU_PER_NODE}"
else
    echo "🔍 使用环境变量指定的GPU数量: ${GPU_PER_NODE}"
fi

# 验证GPU数量
if [ "$GPU_PER_NODE" -lt 1 ] || [ "$GPU_PER_NODE" -gt 8 ]; then
    echo "⚠️  警告: GPU数量 ${GPU_PER_NODE} 看起来不合理，请检查"
fi

# 显示实际可用的GPU信息
echo "🔍 系统GPU信息:"
nvidia-smi -L 2>/dev/null || echo "   ⚠️  无法执行 nvidia-smi，请确保NVIDIA驱动已安装"
echo ""

# 构建训练命令
echo "🚀 启动训练..."
echo "📊 GPU配置:"
echo "   - 使用GPU数量: ${GPU_PER_NODE}"
echo "   - 工作目录: ${WORK_DIR_PATH}"
echo "   - 代码目录: ${CODE_DIR_ABS}"
echo ""
echo "💡 提示: 如需强制使用4个GPU，请在脚本中设置 GPU_PER_NODE=4"
echo ""

# 确定训练脚本路径
if [ -f "${CODE_DIR_ABS}/tools/train.py" ]; then
    TRAIN_SCRIPT="${CODE_DIR_ABS}/tools/train.py"
elif [ -f "${X_SAM_DIR}/xsam/xsam/tools/train.py" ]; then
    TRAIN_SCRIPT="${X_SAM_DIR}/xsam/xsam/tools/train.py"
else
    echo "❌ 错误: 找不到训练脚本 train.py"
    echo "   尝试的路径:"
    echo "   - ${CODE_DIR_ABS}/tools/train.py"
    echo "   - ${X_SAM_DIR}/xsam/xsam/tools/train.py"
    exit 1
fi

# 设置PYTHONPATH - xsam模块的根目录是 /mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/xsam/xsam/
# 所以PYTHONPATH应该包含这个目录，这样import xsam才能找到xsam.utils等模块
if [ -z "${PYTHONPATH}" ]; then
    export PYTHONPATH="${CODE_DIR_ABS}"
else
    export PYTHONPATH="${CODE_DIR_ABS}:${PYTHONPATH}"
fi

# 切换到代码目录
cd "${CODE_DIR_ABS}"

# 调试信息：打印PYTHONPATH
echo "🔍 调试信息:"
echo "   CODE_DIR_ABS: ${CODE_DIR_ABS}"
echo "   PYTHONPATH: ${PYTHONPATH}"
echo "   TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
echo ""

# 验证xsam模块是否可以导入（使用环境变量PYTHONPATH）
echo "🔍 验证xsam模块导入..."
if PYTHONPATH="${PYTHONPATH}" python3 -c "from xsam.utils.logging import print_log" 2>/dev/null; then
    echo "✅ xsam模块导入成功"
else
    echo "⚠️  警告: xsam模块导入验证失败，但将继续尝试运行训练"
    echo "   如果训练失败，请检查PYTHONPATH设置: ${PYTHONPATH}"
fi
echo ""

# 使用torchrun启动训练（带优化参数）
echo "📋 训练参数:"
echo "   - GPU数量: ${GPU_PER_NODE} (通过 --nproc_per_node 设置)"
echo "   - batch_size: 2 (通过 --cfg-options 设置)"
echo "   - accumulative_counts: 2 (通过 --cfg-options 设置)"
echo "   - 总batch_size: $((2 * ${GPU_PER_NODE} * 2)) (batch_size × GPU数量 × accumulative_counts)"
echo "   - sampler: SourceGroupedSampler (已在配置文件中设置)"
echo "   - bypass_duplicate: True (已在配置文件中设置)"
echo "   - NCCL_TIMEOUT: 1200000ms (20分钟)"
echo ""

PYTHONPATH="${PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun \
    --nproc_per_node=${GPU_PER_NODE} \
    "${TRAIN_SCRIPT}" \
    "${X_SAM_DIR}/${CONFIG_FILE}" \
    --work-dir "${WORK_DIR_PATH}" \
    --launcher pytorch \
    --deepspeed deepspeed_zero2 \
    --seed 1024 \
    --cfg-options \
        train_dataloader.batch_size=2 \
        optim_wrapper.accumulative_counts=2

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

