from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset import (
    ConcatDataset,
    GenSegDataset,
    ImgConvDataset,
    OVSegDataset,
    RefSegDataset,
    ReasonSegDataset,
)
from xsam.dataset.pano_seg_dataset import PanoSegDataset
from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.dataset.map_fns import dataset_map_fn_factory, template_map_fn_factory
from xsam.dataset.map_fns.dataset_map_fns import image_conv_map_fn, generic_seg_map_fn, ov_seg_map_fn, refer_seg_map_fn, reason_seg_map_fn
from xsam.dataset.process_fns.postprocess_fns import generic_seg_postprocess_fn, ov_seg_postprocess_fn, refer_seg_postprocess_fn, reason_seg_postprocess_fn
from xsam.dataset.process_fns import process_map_fn_factory
from xsam.dataset.processors import SamImageProcessor
from xsam.dataset.samplers import SourceGroupedSampler
from xsam.engine.hooks import DatasetInfoHook, DistLossReduceHook, EvaluateChatHook, ModelInfoHook, PTCheckpointHook
from xsam.engine.runners.loops import TrainLoop
from peft import LoraConfig
from xsam.model import XSamModel
from xsam.model.segmentors import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel
from xsam.utils.visualize import Visualizer


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
base_root = "/mnt_llm_A100_V1/"
code_dir = getenv("CODE_DIR", "./xsam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", base_root + "shui/LAE/OneTerra-train/checkpoints/xsam_sota_ultimate")
checkpoint_dir = base_root + "/shui/LAE/OneTerra-train/checkpoints/"

# Model Paths
llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# 与 ovseg_subset_ft_flat 相同：构建阶段先加载 S1/S2 底座
s1_pretrained_pth = checkpoint_dir + "s1_seg_finetune/pytorch_model.bin"
s2_pretrained_pth = (
    checkpoint_dir + "xsam_s2_align_pretrain_skyscript_sar/iter_35874.pth"
)

# LoRA 与 V1 (mixed_finetune_all) 保持一致，确保 checkpoint 可完整加载
llm_lora_config = dict(
    type=LoraConfig,
    r=16,
    lora_alpha=32,
    target_modules=[
        "self_attn.qkv_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

# 4×A40：与 ovseg 一致 micro_batch=1，避免 SAM 1024² + dual_encoder OOM
batch_size = 1
accumulative_counts = 8
dataloader_num_workers = 4 
max_epochs = 2
optim_type = AdamW
lr = 2e-5            # 在冻结主干的情况下，用保守学习率微调 LoRA 和对齐层
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.03
save_steps = 500
save_total_limit = 3
logging_interval = 10

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"
ignore_label = 255

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
    freeze_llm=False,
    freeze_visual_encoder=True,       # 🔒 锁定，极大节约显存
    freeze_segmentor_encoder=True,    # 🔒 锁定，维持底层泛化边界能力
    use_dual_encoder=True,
    use_vision_sampler=True,
    use_activation_checkpointing=True,
    connector_type="conv",
    cond_type=cond_type,
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat="extra_pixel_values",
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    postprocess_fn=generic_seg_postprocess_fn,
    llm_lora=llm_lora_config,
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
        reinit_decoder=False, # 继承V1权重
        open_cls=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
oneterra_data_root = base_root + "shui/oneterra_data/"
yangsen_data_root = base_root + "yangsen/datasets/"
fitrs_imgconv_data_path = oneterra_data_root + "imgconv/FIT-RS/raw_data/train_data_of_each_individual_task/"
fitrs_imgconv_image_folder = oneterra_data_root + "imgconv/FIT-RS/raw_data/imgv2_split_512_100_vaild"
optical_caption_data_root = oneterra_data_root + "imgconv/image_caption/"

train_extra_image_processor = deepcopy(extra_image_processor)
train_extra_image_processor.update({
    "size": {"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},
    "do_crop": True,
    "crop_size": {"height": 1024, "width": 1024},
})


# =====================================================================
# 🔴 核心修复：Pano 任务（完美对齐 OVSeg 必备参数）
# =====================================================================
pano_ovseg_dataset = dict(
    type=OVSegDataset,
    data_path=data_dir + "pano/annotations_train_mini_20k.json", # 读取抽样文件
    image_folder=data_dir + "pano/train/images",
    panseg_map_folder=data_dir + "pano/train/panoptic_labels",
    tokenizer=tokenizer,
    task_name="ovseg",
    data_name="pano_ovseg_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=ov_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    # 🚨 下面三个参数是抢救 OVSeg 的绝对核心：
    use_variant_cat=True,
    use_full_cat=True,
    sample_num=32,
    repeats_scale=2.0,  # 翻倍权重进行抢救洗白
)

pano_genseg_dataset = dict(
    type=PanoSegDataset,
    data_path=data_dir + "pano/annotations_train_mini_20k.json", # 共享文件
    image_folder=data_dir + "pano/train/images",
    panseg_map_folder=data_dir + "pano/train/panoptic_labels",
    tokenizer=tokenizer,
    task_name="genseg",
    data_name="pano_panoptic_genseg_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=generic_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    use_variant_cat=True,
    pad_image_to_square=False,
    repeats_scale=1.0,
)

# =====================================================================
# 🟡 模态特征与高质量图文 (SAR + GeoChat PRO)
# =====================================================================
sar_total_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=yangsen_data_root + "sar_total/sft/train_mini_10k.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=yangsen_data_root,
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="sar_total_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=1.0,
)

geochat_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=data_dir + "img_conv_data/geochat/geochat_mini_5k_PRO.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=data_dir + "img_conv_data/geochat/images",
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="geochat_imgconv",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=2.0, # 长文本大幅提权
)

# =====================================================================
# 🟢 评测打榜：Caption 与 VQA 
# =====================================================================
ucm_captions_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=optical_caption_data_root + "UCM-Captions/dataset_qwenvl.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=optical_caption_data_root + "UCM-Captions/imgs",
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="imgconv_UCM-Captions",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=4.0, # 全量且极大提权
)

nwpu_captions_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=optical_caption_data_root + "NWPU-Captions/nwpu_mini_10k.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=optical_caption_data_root + "NWPU-Captions/",
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="imgconv_NWPU-Captions",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=2.0,
)

# hrben_imgconv_dataset = dict(
#     type=ImgConvDataset,
#     data_path=oneterra_data_root + "imgconv/FIT-RS/FIT-RSFG/FIT-RSFG-Bench/hrben_mini_10k.json",
#     tokenizer=tokenizer,
#     cond_type=cond_type,
#     special_tokens=special_tokens,
#    image_folder=oneterra_data_root + "imgconv/FIT-RS/FIT-RSFG/FIT-RSFG-Bench/images", # 假设你的抽样在此同级
#     image_processor=image_processor,
#     extra_image_processor=train_extra_image_processor,
#     task_name="imgconv",
#     data_name="imgconv_hrben",
#     dataset_map_fn=dict(
#         type=dataset_map_fn_factory,
#         fn=image_conv_map_fn,
#     ),
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     pixel_values_ndim=2,
#     is_multimodal=True,
#     exclude_pure_text=True,
#     pad_image_to_square=False,
#     preprocess_text_data=True,
#     repeats_scale=2.0,
# )

rsvqa_lr_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=oneterra_data_root + "imgconv/VQA/RSVQA-LR/train_mini_10k.json",
    tokenizer=tokenizer,
    cond_type=cond_type,
    special_tokens=special_tokens,
    image_folder=oneterra_data_root + "imgconv/VQA/RSVQA-LR/Images_LR",
    image_processor=image_processor,
    extra_image_processor=train_extra_image_processor,
    task_name="imgconv",
    data_name="imgconv_RSVQA_LR",
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=image_conv_map_fn,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pixel_values_ndim=2,
    is_multimodal=True,
    exclude_pure_text=True,
    pad_image_to_square=False,
    preprocess_text_data=True,
    repeats_scale=2.0,
)

# =====================================================================
# 🟣 评测打榜：RefSeg 与 ReaSeg
# =====================================================================
remotesam_refseg_dataset = dict(
    type=RefSegDataset,
    # remotesam_sample/ 内含 instances.json + refs(unc).p；dataset=fast 表示 data_root 即 DATA_DIR
    data_root=data_dir + "ref_seg_data/remotesam_sample",
    image_folder=data_dir + "ref_seg_data/images/remotesam_images",
    dataset="fast",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="remotesam_train_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refer_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refer_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    repeats_scale=2.0,
)

fast_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=oneterra_data_root + "refseg/FAST/fast_sample",
    image_folder=oneterra_data_root + "refseg/FAST/images",
    dataset="fast",
    data_split="train",  
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refseg_fast_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refer_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refer_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    repeats_scale=2.0,
)

risbench_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=oneterra_data_root + "refseg/RISBench/risbench_sample",
    image_folder=oneterra_data_root + "refseg/RISBench/RISBench_dataset/img_rgb",
    dataset="fast",
    data_split="train",  
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refseg_risbench_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refer_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refer_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    repeats_scale=2.0,
)

earthreason_reaseg_dataset = dict(
    type=ReasonSegDataset,
    data_root=oneterra_data_root + "reasonseg/EarthReason_convert",
    image_folder=oneterra_data_root + "reasonseg/EarthReason_convert/train",  
    explain_path=oneterra_data_root + "reasonseg/EarthReason_convert/explanatory/train.json",
    data_split="train",  
    tokenizer=tokenizer,
    task_name="reaseg",
    data_name="reaseg_earthreason_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=reason_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=reason_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    use_threads=True,
    repeats_scale=3.0, # 全量并提权
)

diy1_reaseg_dataset = dict(
    type=ReasonSegDataset,
    data_root=oneterra_data_root + "reasonseg/diy1",
    # 🚨 注意：这里指向了我们使用 dir_pairs 生成的软链接抽样目录
    image_folder=oneterra_data_root + "reasonseg/diy1/train_mini_10k",  
    data_split="train",  
    tokenizer=tokenizer,
    task_name="reaseg",
    data_name="reaseg_diy1_train",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=train_extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=reason_seg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=reason_seg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
    use_threads=True,
    repeats_scale=2.0,
)

# =====================================================================
# ⚫ FIT-RS 极限缩水背景板 (6 个)
# =====================================================================
fitrs_complexcompre_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "complexcompre_mini_3k.json",
    tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder, image_processor=image_processor,
    extra_image_processor=train_extra_image_processor, task_name="imgconv",
    data_name="fitrs_complexcompre_imgconv",
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length, pixel_values_ndim=2, is_multimodal=True,
    exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)

fitrs_vqa_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "vqa_mini_3k.json",
    tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder, image_processor=image_processor,
    extra_image_processor=train_extra_image_processor, task_name="imgconv",
    data_name="fitrs_vqa_imgconv",
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length, pixel_values_ndim=2, is_multimodal=True,
    exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)

fitrs_imagecaption_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "imagecaption_mini_2k.json",
    tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder, image_processor=image_processor,
    extra_image_processor=train_extra_image_processor, task_name="imgconv",
    data_name="fitrs_imagecaption_imgconv",
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length, pixel_values_ndim=2, is_multimodal=True,
    exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)

fitrs_imageclassification_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "imageclassification_mini_2k.json",
    tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder, image_processor=image_processor,
    extra_image_processor=train_extra_image_processor, task_name="imgconv",
    data_name="fitrs_imageclassification_imgconv",
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length, pixel_values_ndim=2, is_multimodal=True,
    exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)

fitrs_multiturn_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "multiturn_mini_2k.json",
    tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder, image_processor=image_processor,
    extra_image_processor=train_extra_image_processor, task_name="imgconv",
    data_name="fitrs_multiturn_imgconv",
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length, pixel_values_ndim=2, is_multimodal=True,
    exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)

fitrs_regioncaption_imgconv_dataset = dict(
    type=ImgConvDataset,
    data_path=fitrs_imgconv_data_path + "regioncaption_mini_2k.json",
    tokenizer=tokenizer, cond_type=cond_type, special_tokens=special_tokens,
    image_folder=fitrs_imgconv_image_folder, image_processor=image_processor,
    extra_image_processor=train_extra_image_processor, task_name="imgconv",
    data_name="fitrs_regioncaption_imgconv",
    dataset_map_fn=dict(type=dataset_map_fn_factory, fn=image_conv_map_fn),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length, pixel_values_ndim=2, is_multimodal=True,
    exclude_pure_text=True, pad_image_to_square=False, preprocess_text_data=True,
    repeats_scale=1.0,
)

# =====================================================================
# 封装装车
# =====================================================================
combined_train_dataset = dict(
    type=ConcatDataset,
    oversample_ratio=0.0,  
    datasets=[
        geochat_imgconv_dataset,
        sar_total_imgconv_dataset,
        ucm_captions_imgconv_dataset,
        nwpu_captions_imgconv_dataset,
       # hrben_imgconv_dataset,
        rsvqa_lr_imgconv_dataset,
        fitrs_complexcompre_imgconv_dataset,
        fitrs_imagecaption_imgconv_dataset,
        fitrs_imageclassification_imgconv_dataset,
        fitrs_multiturn_imgconv_dataset,
        fitrs_regioncaption_imgconv_dataset,
        fitrs_vqa_imgconv_dataset,
        pano_genseg_dataset,
        pano_ovseg_dataset,
        fast_refseg_dataset,
        remotesam_refseg_dataset,
        risbench_refseg_dataset,
        earthreason_reaseg_dataset,
        diy1_reaseg_dataset,
    ],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    persistent_workers=False,
    dataset=combined_train_dataset,
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
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "visual_encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
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
default_hooks = dict(
    dist_loss_reduce=dict(type=DistLossReduceHook),
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

log_level = "INFO"

# 与 ovseg_subset_ft_flat 相同：Runner 启动后再 load_from 覆盖 V1 全任务权重
prev_s3_ckpt = getenv(
    "PREV_S3_CKPT",
    "/mnt_llm_A100_V1/shui/LAE/OneTerra-train/wkdrs_01/s3_mixed_fineture_base/"
    "xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_all_v1/iter_61176.pth",
)
load_from = prev_s3_ckpt
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)