#!/bin/bash
# 交互式多任务测试Demo启动脚本

# 设置项目根目录
cd "$(dirname "$0")"

# 配置文件路径（根据你的实际配置调整）
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat.py"

# 模型检查点路径（请根据你的实际训练结果修改）
# 方式1: 使用根目录下的pytorch_model.bin（推荐，最简单）
PTH_MODEL="./wkdrs/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat/pytorch_model.bin"

# 方式2: 使用DeepSpeed checkpoint目录（会自动查找mp_rank_00_model_states.pt）
# PTH_MODEL="./wkdrs/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat/iter_28904.pth"

# 输出目录
OUTPUT_DIR="./demo_outputs"

# 设置Python路径
export PYTHONPATH="$(realpath xsam):$PYTHONPATH"

# 运行demo
python xsam/xsam/tools/interactive_multi_task_demo.py \
    "$CONFIG_FILE" \
    --pth_model "$PTH_MODEL" \
    --output_dir "$OUTPUT_DIR"

