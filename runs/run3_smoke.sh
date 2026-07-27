#!/usr/bin/env bash
#
# run3_smoke.sh - 4 节点训练冒烟：NCCL 同 run3 + 只跑少量 iter，验证能训、别单卡假死
#
# 注意：首次仍会加载完整数据集（可能 30~90 分钟），但训练环只跑 SMOKE_MAX_ITERS 步后退出。
# 建议流程：先 nccl_smoke_test.sh（几分钟）-> 再本脚本（数小时）-> 再 run3.sh 全量
#
# 用法：
#   SMOKE_MAX_ITERS=200 WORLD_SIZE=4 RANK=0 MASTER_ADDR=... TQ_GPU_NUM=8 \
#     bash runs/run3_smoke.sh --config xsam/xsam/configs/.../xsam_finetune_v31_16A100.py

export SMOKE_MAX_ITERS="${SMOKE_MAX_ITERS:-200}"
export RUN3_SMOKE=1

_script_dir=$(dirname "$0")
exec bash "${_script_dir}/run3.sh" "$@"
