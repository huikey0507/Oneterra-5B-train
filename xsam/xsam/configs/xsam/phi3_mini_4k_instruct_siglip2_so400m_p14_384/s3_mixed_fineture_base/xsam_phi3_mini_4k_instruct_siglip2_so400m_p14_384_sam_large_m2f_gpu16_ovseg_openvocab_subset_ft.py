from os import getenv

from mmengine.config import read_base

# 必须用 read_base + 相对 import* 继承含 torch/xsam 顶层 import 的 base（lazy 链）。
# 勿用 _base_ 列表：子配置会被判为非 lazy，递归加载 base 时在 mmengine 中会触发 RuntimeError。
with read_base():
    from .xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all import *  # noqa: F401, F403

# 输出目录（可用环境变量 WORK_DIR 覆盖）
work_dir = "/mnt_llm_A100_V1/shui/LAE/OneTerra-train/checkpoints/xsam_s3_ovseg_openvocab_subset_ft"
# S1/S2 加载逻辑继续沿用 base 配置；这里额外指定“已训练完 S3”的起点权重。
# 优先读取环境变量 PREV_S3_CKPT；未设置时走默认路径。
prev_s3_ckpt = getenv(
    "PREV_S3_CKPT",
    "/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs/s3_mixed_fineture_base/"
    "xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/pytorch_model.bin",
)

load_from = prev_s3_ckpt

# False: 仅加载模型权重继续微调（推荐）
# True: 连优化器/调度器状态一并恢复（用于中断续训）
resume = False

# OVSeg open-vocab finetune (subset category sampling enabled).
# 41类数据集下，sample_num=32 能稳定产生子集采样并保留足够上下文类别。
pano_ovseg_dataset = dict(
    use_variant_cat=True,
    use_full_cat=True,  # True: 约50%全类 + 约50%子集；False: 100%子集采样
    sample_num=32,
    repeats_scale=3,
)

# 仅保留 OVSeg 任务进行继续微调，其他训练超参沿用 base 配置。
combined_train_dataset = dict(
    type="ConcatDataset",
    oversample_ratio=0.0,
    datasets=[pano_ovseg_dataset],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    persistent_workers=False,  # 禁用persistent_workers以避免共享内存不足问题
    dataset=combined_train_dataset,
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=xsam_collate_fn),
)