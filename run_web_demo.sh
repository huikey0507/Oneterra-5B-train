#!/bin/bash
# 网页交互式多任务测试Demo启动脚本

# 设置项目根目录
cd "$(dirname "$0")"

# 配置文件路径
CONFIG_FILE="xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat.py"

# 模型检查点路径（请根据你的实际训练结果修改）
# 方式1: 使用根目录下的pytorch_model.bin（推荐，最简单）
PTH_MODEL="./wkdrs/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat/pytorch_model.bin"

# 方式2: 使用DeepSpeed checkpoint目录（会自动查找mp_rank_00_model_states.pt）
# PTH_MODEL="./wkdrs/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat/iter_28904.pth"

# 输出目录
OUTPUT_DIR="./demo_outputs"

# 端口（可以修改）
PORT=7860

# 是否启用Gradio公网分享链接（设置为true可生成公网访问链接）
ENABLE_SHARE=true

# 设置Python路径
export PYTHONPATH="$(realpath xsam):$PYTHONPATH"

# 设置PyTorch CUDA内存分配配置，减少内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行web demo
if [ "$ENABLE_SHARE" = "true" ]; then
    python xsam/xsam/tools/web_demo.py \
        "$CONFIG_FILE" \
        --pth_model "$PTH_MODEL" \
        --log-dir "$OUTPUT_DIR" \
        --port "$PORT" \
        --host "0.0.0.0" \
        --share
else
    python xsam/xsam/tools/web_demo.py \
        "$CONFIG_FILE" \
        --pth_model "$PTH_MODEL" \
        --log-dir "$OUTPUT_DIR" \
        --port "$PORT" \
        --host "0.0.0.0"
fi

