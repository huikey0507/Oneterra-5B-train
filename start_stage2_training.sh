#!/bin/bash
# X-SAM 第二阶段训练启动脚本（使用SkyScript和SAR数据）
#
# 使用方法：
#   1. 直接运行: bash start_stage2_training.sh
#   2. 通过环境变量设置外部数据路径（可选）:
#      export SKYSCRIPT_IMAGE_DIR="/path/to/skyscript/images"
#      export SAR_TRAIN_JSON="/path/to/sar/train.json"
#      export SAR_VAL_JSON="/path/to/sar/val.json"
#      export SAR_IMAGE_DIR="/path/to/sar/images"
#      bash start_stage2_training.sh
#   3. 或者直接修改脚本中的默认路径

set -e

# 获取脚本所在目录作为项目根目录（自动检测，无需手动修改）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# 配置路径（相对路径，相对于项目根目录）
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s2_align_pretrain/xsam_s2_align_pretrain_skyscript_sar.py"
DEEPSPEED_CONFIG="xsam/xsam/configs/deepspeed/deepspeed_zero2.json"

# 数据路径（相对路径，相对于项目根目录）
SKYSCRIPT_JSON_FILE="datas/img_conv_data/skyscript/skyscript.json"

# 外部数据路径（可通过环境变量覆盖，或直接修改下面的默认路径）
# 如果数据在项目目录外，请设置环境变量或修改下面的路径
#SKYSCRIPT_IMAGE_DIR="${SKYSCRIPT_IMAGE_DIR:-/mnt/si001883vtjl/yangsen/datasets}"
#SAR_TRAIN_JSON="${SAR_TRAIN_JSON:-/mnt/si001883vtjl/yangsen/datasets/sar_total/pretraining/train.json}"
#SAR_VAL_JSON="${SAR_VAL_JSON:-/mnt/si001883vtjl/yangsen/datasets/sar_total/pretraining/val.json}"
#SAR_IMAGE_DIR="${SAR_IMAGE_DIR:-/mnt/si001883vtjl/yangsen/datasets}"

SKYSCRIPT_IMAGE_DIR="${SKYSCRIPT_IMAGE_DIR:-/mnt_llm_A100_V1/yangsen/datasets}"
SAR_TRAIN_JSON="${SAR_TRAIN_JSON:-/mnt_llm_A100_V1/yangsen/datasets/sar_total/pretraining/train.json}"
SAR_VAL_JSON="${SAR_VAL_JSON:-/mnt_llm_A100_V1/yangsen/datasets/sar_total/pretraining/val.json}"
SAR_IMAGE_DIR="${SAR_IMAGE_DIR:-/mnt_llm_A100_V1/yangsen/datasets}"

# Stage 1权重路径（相对路径）
S1_CHECKPOINT="checkpoints/s1_seg_finetune/pytorch_model.bin"

echo "=========================================="
echo "X-SAM 第二阶段训练启动脚本"
echo "=========================================="
echo ""

# 步骤1: 检查Stage 1权重
echo "步骤1: 检查Stage 1权重..."
S1_CHECKPOINT_ABS="${PROJECT_DIR}/${S1_CHECKPOINT}"
if [ ! -f "${S1_CHECKPOINT_ABS}" ]; then
    echo "❌ 错误: Stage 1权重不存在: ${S1_CHECKPOINT_ABS}"
    echo "请先完成Stage 1训练，或修改配置文件中的s1_pretrained_pth路径"
    exit 1
fi
echo "✅ Stage 1权重存在: ${S1_CHECKPOINT_ABS}"
echo ""

# 步骤2: 检查数据文件
echo "步骤2: 检查数据文件..."
SKYSCRIPT_JSON_ABS="${PROJECT_DIR}/${SKYSCRIPT_JSON_FILE}"
if [ ! -f "${SKYSCRIPT_JSON_ABS}" ]; then
    echo "❌ 错误: SkyScript JSON文件不存在: ${SKYSCRIPT_JSON_ABS}"
    exit 1
fi
echo "✅ SkyScript JSON文件存在: ${SKYSCRIPT_JSON_ABS}"

if [ ! -d "${SKYSCRIPT_IMAGE_DIR}" ]; then
    echo "❌ 错误: SkyScript图像目录不存在: ${SKYSCRIPT_IMAGE_DIR}"
    echo "提示: 可以通过环境变量 SKYSCRIPT_IMAGE_DIR 设置路径"
    exit 1
fi
echo "✅ SkyScript图像目录存在: ${SKYSCRIPT_IMAGE_DIR}"

if [ ! -f "${SAR_TRAIN_JSON}" ]; then
    echo "❌ 错误: SAR训练JSON文件不存在: ${SAR_TRAIN_JSON}"
    echo "提示: 可以通过环境变量 SAR_TRAIN_JSON 设置路径"
    exit 1
fi
echo "✅ SAR训练JSON文件存在: ${SAR_TRAIN_JSON}"

if [ ! -f "${SAR_VAL_JSON}" ]; then
    echo "⚠️  警告: SAR验证JSON文件不存在: ${SAR_VAL_JSON}"
    echo "提示: 可以通过环境变量 SAR_VAL_JSON 设置路径"
    echo "验证时将无法使用SAR验证集"
else
    echo "✅ SAR验证JSON文件存在: ${SAR_VAL_JSON}"
fi

if [ ! -d "${SAR_IMAGE_DIR}" ]; then
    echo "❌ 错误: SAR图像目录不存在: ${SAR_IMAGE_DIR}"
    echo "提示: 可以通过环境变量 SAR_IMAGE_DIR 设置路径"
    exit 1
fi
echo "✅ SAR图像目录存在: ${SAR_IMAGE_DIR}"
echo ""

# 步骤3: 检查配置文件
echo "步骤3: 检查配置文件..."
CONFIG_FILE_ABS="${PROJECT_DIR}/${CONFIG_FILE}"
if [ ! -f "${CONFIG_FILE_ABS}" ]; then
    echo "❌ 错误: 配置文件不存在: ${CONFIG_FILE_ABS}"
    exit 1
fi
echo "✅ 配置文件存在: ${CONFIG_FILE_ABS}"
echo ""

# 步骤4: 检查DeepSpeed配置
echo "步骤4: 检查DeepSpeed配置..."
DEEPSPEED_CONFIG_ABS="${PROJECT_DIR}/${DEEPSPEED_CONFIG}"
if [ ! -f "${DEEPSPEED_CONFIG_ABS}" ]; then
    echo "⚠️  警告: DeepSpeed配置文件不存在: ${DEEPSPEED_CONFIG_ABS}"
    echo "将不使用DeepSpeed训练"
    USE_DEEPSPEED=false
else
    echo "✅ DeepSpeed配置文件存在: ${DEEPSPEED_CONFIG_ABS}"
    USE_DEEPSPEED=true
fi
echo ""

# 步骤5: 检测GPU数量
echo "步骤5: 检测GPU数量..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
    echo "✅ 检测到 ${GPU_COUNT} 个GPU"
    
    # 计算有效batch_size和预计时间
    EFFECTIVE_BATCH_SIZE=$((4 * 4 * GPU_COUNT))
    echo "   有效batch_size: ${EFFECTIVE_BATCH_SIZE}"
    echo ""
else
    echo "⚠️  无法检测GPU数量，将使用默认配置"
    echo ""
fi

# 步骤6: 显示训练信息
echo "=========================================="
echo "训练配置信息"
echo "=========================================="
echo "项目目录: ${PROJECT_DIR}"
echo "配置文件: ${CONFIG_FILE_ABS}"
echo "SkyScript数据: ${SKYSCRIPT_JSON_ABS}"
echo "SkyScript图像目录: ${SKYSCRIPT_IMAGE_DIR}"
echo "SAR训练数据: ${SAR_TRAIN_JSON}"
echo "SAR验证数据: ${SAR_VAL_JSON}"
echo "SAR图像目录: ${SAR_IMAGE_DIR}"
echo "Stage 1权重: ${S1_CHECKPOINT_ABS}"
if [ "$USE_DEEPSPEED" = true ]; then
    echo "DeepSpeed: 启用"
else
    echo "DeepSpeed: 未启用"
fi
echo "=========================================="
echo ""

# 步骤7: 启动训练（前台运行）
echo "步骤7: 启动训练..."
echo "=========================================="
echo "开始训练（前台运行模式）..."
echo "=========================================="
echo ""

cd ${PROJECT_DIR}

# 将数据路径传入训练配置（xsam_s2_align_pretrain_skyscript_sar.py 会读取这些环境变量）
export YANGSEN_DATASETS_ROOT="${SAR_IMAGE_DIR}"
export SAR_DATA_ROOT="$(dirname "${SAR_TRAIN_JSON}")"
export SAR_TRAIN_JSON="${SAR_TRAIN_JSON}"
export SAR_VAL_JSON="${SAR_VAL_JSON}"
export SAR_IMAGE_DIR="${SAR_IMAGE_DIR}"

# 设置分布式训练端口（避免端口冲突）
# 如果端口被占用，可以修改这个值或通过环境变量 MASTER_PORT 覆盖
MASTER_PORT="${MASTER_PORT:-29501}"
export MASTER_PORT

echo "使用分布式训练端口: ${MASTER_PORT}"
echo "提示: 如果端口冲突，可以通过环境变量设置: export MASTER_PORT=29502"
echo ""

# 生成日志文件名
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")
LOG_FILE="stage2_training_${TIMESTAMP}.log"
LOG_FILE_ABS="${PROJECT_DIR}/${LOG_FILE}"

echo "日志将同时输出到终端和文件: ${LOG_FILE_ABS}"
echo "配置文件: ${CONFIG_FILE_ABS}"
echo ""

bash runs/run.sh --modes train \
    --config ${CONFIG_FILE_ABS} 2>&1 | tee ${LOG_FILE_ABS}

