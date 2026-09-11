"""Stage B: SAR panoptic alignment via sar_seg_connector.

Train: sar_seg_connector (+ modality_router)
Freeze: LLM/SigLIP/SAM encoder, Mask2Former decoder/pixel_decoder,
        optical adapters, SAR projectors (from Stage A)
Data: SAR-ovseg (no synonym) vs OPT-ovseg (with synonym).
Raw sizes ~3.6k vs ~111k; repeats_scale tuned so effective SAR:OPT ≈ 3~4:1.
No genseg.
"""
from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import ConcatDataset, OVSegDataset, SarOVSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import dataset_map_fn_factory, template_map_fn_factory
from xsam.dataset.map_fns.dataset_map_fns import ov_seg_map_fn
from xsam.dataset.process_fns.postprocess_fns import generic_seg_postprocess_fn
from xsam.dataset.processors import SamImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, DistLossReduceHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners.loops import TrainLoop
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
base_root = "/mnt_llm_A100_V1/"
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", base_root + "shui/LAE/OneTerra-train/wkdrs_sar_align_B")
checkpoint_dir = base_root + "shui/LAE/OneTerra-train/checkpoints/"

llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

s1_pretrained_pth = checkpoint_dir + "s1_seg_finetune/pytorch_model.bin"
# Stage B 必须先吃满 S3 光学底座（含 decoder/connector），再叠 Stage A 的 SAR adapter。
# 不能把 Stage A 的 ZeRO（往往只有 sar_* + router 十余个 tensor）直接当 load_from。
s3_pretrained_pth = getenv(
    "PREV_S3_CKPT",
    checkpoint_dir + "s3_mixed_fineture_v3.2/pytorch_model.bin",
)
# 若给的是 ZeRO tag 目录，改用同目录下合并好的 bin（若存在），否则要求用户给 .bin。
if s3_pretrained_pth.endswith(".pth") and not s3_pretrained_pth.endswith(".bin"):
    # DeepSpeed tag dir: prefer sibling pytorch_model.bin under the same experiment folder
    # e.g. .../s3_mixed_fineture_v3.2/iter_70504.pth -> .../s3_mixed_fineture_v3.2/pytorch_model.bin
    _parts = s3_pretrained_pth.rstrip("/").split("/")
    _parent = "/".join(_parts[:-1]) if _parts else ""
    _bin_candidate = _parent + "/pytorch_model.bin"
    s3_pretrained_pth = _bin_candidate

s2_pretrained_pth = s3_pretrained_pth
sar_adapter_pth = getenv(
    "PREV_SAR_A_CKPT",
    base_root + "shui/LAE/OneTerra-train/wkdrs_sar_align_A/iter_91000.pth",
)
load_from = None

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

batch_size = 1
accumulative_counts = 8
dataloader_num_workers = 2
max_epochs = 1
optim_type = AdamW
lr = 5e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.03
save_steps = 500
save_total_limit = 6
logging_interval = 20

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=SiglipProcessor.from_pretrained,
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
    freeze_segmentor_decoder=True,  # keep optical M2F path identical to S3/Stage A
    use_dual_encoder=True,
    use_activation_checkpointing=True,
    use_sar_adapters=True,
    freeze_optical_adapters=True,
    freeze_sar_projectors=True,  # Stage A already aligned language side
    sar_gate_loss_weight=0.1,
    sar_infer_threshold=0.5,
    connector_type="conv",
    cond_type=cond_type,
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    sar_adapter_pth=sar_adapter_pth,
    tokenizer=tokenizer,
    postprocess_fn=generic_seg_postprocess_fn,
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
            attn_implementation="eager",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=False,
        open_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
pano_data_root = data_dir + "pano/"
sar_root = base_root + "shui/oneterra_data/sarseg/pansar2/"

train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update(
    {
        "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},
        "do_crop": True,
        "crop_size": {"height": 1024, "width": 1024},
    }
)

# OPT-ovseg with synonyms — capability to preserve
opt_ovseg_train = dict(
    type=OVSegDataset,
    data_path=pano_data_root + "annotations_train.json",
    image_folder=pano_data_root + "train/images",
    panseg_map_folder=pano_data_root + "train/panoptic_labels",
    tokenizer=tokenizer,
    task_name="ovseg",
    data_name="pano_ovseg_train",
    modality=0,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=ov_seg_map_fn, cond_type=cond_type),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    use_cat_synonym=True,
    cat_synonym_keep_prob=0.5,
    sample_num=32,
    variant_subset_prob=0.7,
    # 111066 * 0.08 ≈ 8885 effective (optical replay, anti-forgetting)
    repeats_scale=0.15,
)

# SAR-ovseg without synonyms — main SAR supervision
sar_ovseg_train = dict(
    type=SarOVSegDataset,
    data_path=sar_root + "annotations/panoptic_train2017.json",
    image_folder=sar_root + "train2017",
    panseg_map_folder=sar_root + "annotations/panoptic_train2017",
    tokenizer=tokenizer,
    task_name="ovseg",
    data_name="sar_pano_ovseg_train",
    modality=1,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=ov_seg_map_fn, cond_type=cond_type),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    use_cat_synonym=False,
    # 3642 * 8 ≈ 29136 effective → SAR:OPT ≈ 3.3:1
    repeats_scale=8.0,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=dict(
        type=ConcatDataset,
        oversample_ratio=0.0,
        datasets=[sar_ovseg_train, opt_ovseg_train],
    ),
    sampler=dict(
        type=SourceGroupedSampler,
        length_property="source_length",
        mega_batch_mult=1,
        per_device_batch_size=batch_size * accumulative_counts,
    ),
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
        module_names=["seg_connector", "sar_seg_connector", "modality_router", "llm_projector"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(type=DistLossReduceHook),
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
# S3 全量经 s2_pretrained_pth 加载；Stage A SAR adapter 经 sar_adapter_pth 叠加；勿用 Stage A ZeRO 做 load_from
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)
