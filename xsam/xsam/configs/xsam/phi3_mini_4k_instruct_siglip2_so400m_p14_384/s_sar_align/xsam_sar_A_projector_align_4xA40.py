"""Stage A: SAR language alignment (projectors + router).

Train: sar_visual_projector, sar_seg_projector, modality_router
Freeze: LLM / SigLIP / SAM / optical projectors (+ no decoder)
Data:
  - SAR: pretrain imgconv + SFT VQA/指令 (真正更新 SAR projector)
  - Optical: SkyScript 小比例 replay (几乎只训 router；光学能力靠冻 S3 保真)
Route: train hard by modality label; infer hard by router threshold; loss = loss_llm + 0.1 * loss_gate
"""
from os import getenv

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import ConcatDataset, ImageConvDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import image_conv_map_fn, template_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners import TrainLoop
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.sam import SamModel

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
base_root = "/mnt_llm_A100_V1/"
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", base_root + "shui/LAE/OneTerra-train/wkdrs_sar_align_A")
checkpoint_dir = base_root + "shui/LAE/OneTerra-train/checkpoints/"

llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip2-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"

s1_pretrained_pth = checkpoint_dir + "s1_seg_finetune/pytorch_model.bin"
s2_pretrained_pth = checkpoint_dir + "xsam_s2_align_pretrain_skyscript_sar/iter_35874.pth"
# DeepSpeed load_from 需要 ZeRO tag 目录（iter_*.pth/），不能直接吃 pytorch_model.bin。
# 若 PREV_S3_CKPT 指向 .bin，则改走模型 init 的 guess_load（s2 槽位）。
prev_s3_ckpt = getenv(
    "PREV_S3_CKPT",
    checkpoint_dir + "s3_mixed_fineture_v3.2/iter_70504.pth",
)
# 勿 import os.path：module 会进 cfg 导致 deepcopy 失败
if prev_s3_ckpt.endswith(".bin"):
    s2_pretrained_pth = prev_s3_ckpt
    load_from = None
else:
    load_from = prev_s3_ckpt

skyscript_data_path = data_dir + "img_conv_data/skyscript/skyscript.json"
skyscript_image_folder = data_dir + "img_conv_data/skyscript/"
sar_image_folder = base_root + "yangsen/datasets"
sar_pretrain_data_path = base_root + "yangsen/datasets/sar_total/pretraining/train.json"
sar_sft_data_path = base_root + "yangsen/datasets/sar_total/sft/train.json"

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

batch_size = 1
accumulative_counts = 8
dataloader_num_workers = 2
max_epochs = 1
optim_type = AdamW
lr = 5e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1
warmup_ratio = 0.03
save_steps = 1000
save_total_limit = 4
logging_interval = 20

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=SiglipImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    freeze_segmentor_encoder=True,
    use_dual_encoder=True,
    use_sar_adapters=True,
    freeze_optical_adapters=True,
    sar_gate_loss_weight=0.1,
    sar_infer_threshold=0.5,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    connector_type="conv",
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype=torch.bfloat16,
    ),
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        drop_decoder=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
# 光学：冻路径无梯度，只给 router 对比；量不必大
skyscript_imgconv_dataset = dict(
    type=ImageConvDataset,
    data_path=skyscript_data_path,
    tokenizer=tokenizer,
    image_folder=skyscript_image_folder,
    task_name="imgconv",
    data_name="skyscript_imgconv",
    modality=0,
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    dataset_map_fn=image_conv_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    preprocess_text_data=False,
    repeats_scale=0.15,  # ~759k * 0.15 ≈ 114k（router 对比）
)

# SAR 对齐 caption/pretrain：主监督之一
sar_pretrain_imgconv_dataset = dict(
    type=ImageConvDataset,
    data_path=sar_pretrain_data_path,
    tokenizer=tokenizer,
    image_folder=sar_image_folder,
    task_name="imgconv",
    data_name="sar_pretrain_imgconv",
    modality=1,
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    dataset_map_fn=image_conv_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    preprocess_text_data=False,
    repeats_scale=1.0,  # ~389k
)

# SAR SFT 问答/指令：补问答形态，真正更新 SAR projector
sar_sft_imgconv_dataset = dict(
    type=ImageConvDataset,
    data_path=sar_sft_data_path,
    tokenizer=tokenizer,
    image_folder=sar_image_folder,
    task_name="imgconv",
    data_name="sar_sft_imgconv",
    modality=1,
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    dataset_map_fn=image_conv_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    preprocess_text_data=True,
    repeats_scale=0.25,  # ~1.01M * 0.25 ≈ 252k
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=dict(
        type=ConcatDataset,
        oversample_ratio=0.0,
        datasets=[
            sar_pretrain_imgconv_dataset,
            sar_sft_imgconv_dataset,
            skyscript_imgconv_dataset,
        ],
    ),
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=xsam_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["visual_projector", "seg_projector", "sar_visual_projector", "sar_seg_projector", "modality_router"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(type=PTCheckpointHook, clean_pth=False),
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

visualizer = None
log_level = "INFO"
# load_from 已在上方按 PREV_S3_CKPT 类型设定（ZeRO dir / .bin）
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)
