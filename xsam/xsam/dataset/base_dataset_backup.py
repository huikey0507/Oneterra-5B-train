import copy
import os

import torch
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from PIL import Image
from torch.utils.data import Dataset
from xtuner.dataset.utils import expand2square
from xtuner.registry import BUILDER, MAP_FUNC

from xsam.utils.logging import print_log

from ..utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_TASKS,
)
from .utils.catalog import MetadataCatalog
from .utils.encode import encode_fn

SPECIAL_TOKENS = [DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN, DEFAULT_SEG_TOKEN, DEFAULT_CLS_TOKEN]
TASK_MODALITY_LENGTH = {k: int(i * 512) for i, k in enumerate(DEFAULT_TASKS)}

debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
debug_iter = 200


def _resolve_data_path(data_path):
    """
    解析数据文件路径，如果文件不存在，尝试多个可能的位置
    
    Args:
        data_path: 数据文件路径（可能是绝对路径或相对路径）
    
    Returns:
        实际存在的数据文件路径，如果找不到则返回原始路径
    """
    # 如果已经是绝对路径且存在，直接返回
    if os.path.isabs(data_path):
        if os.path.exists(data_path):
            return data_path
        # 如果绝对路径不存在，返回原始路径（让调用者处理错误）
        return data_path
    
    # 如果是相对路径，先尝试直接使用（相对于当前工作目录）
    if os.path.exists(data_path):
        return os.path.abspath(data_path)
    
    # 如果路径以 ../ 开头，尝试从多个可能的基础路径解析
    if data_path.startswith("../"):
        from os import getenv
        import os.path as osp
        
        # 尝试从环境变量获取基础路径（仅使用相对路径，避免绝对路径）
        oneterra_data_dir = getenv("ONETERRA_DATA_DIR", None)
        if oneterra_data_dir:
            # 如果环境变量设置了，尝试从那里解析
            # 例如：data_path = "../../../../oneterra_data/imgconv/..."
            if "oneterra_data" in data_path:
                parts = data_path.split("oneterra_data/")
                if len(parts) > 1:
                    relative_part = parts[1]
                    combined_path = osp.join(oneterra_data_dir, relative_part)
                    if os.path.exists(combined_path):
                        return os.path.abspath(combined_path)
        # yangsen：仅当环境变量为相对路径时才使用，避免 /yangsen/datasets/ 等绝对路径
        yangsen_data_dir = getenv("YANGSEN_DATA_DIR", None)
        if yangsen_data_dir and not os.path.isabs(yangsen_data_dir) and "yangsen/datasets/" in data_path:
            parts = data_path.split("yangsen/datasets/")
            if len(parts) > 1:
                # parts[1] 即 sar_total/sft/train.json 等
                combined_path = osp.join(yangsen_data_dir.rstrip("/"), parts[1])
                if os.path.exists(combined_path):
                    return os.path.abspath(combined_path)
        
        # 尝试已知的绝对路径
        known_paths = [
            "/mnt_llm_A100_V1/shui/oneterra_data",
        ]
        
        for base_path in known_paths:
            if os.path.exists(base_path):
                # 提取相对路径中 oneterra_data 之后的部分
                if "oneterra_data" in data_path:
                    parts = data_path.split("oneterra_data/")
                    if len(parts) > 1:
                        relative_part = parts[1]
                        combined_path = osp.join(base_path, relative_part)
                        if os.path.exists(combined_path):
                            return os.path.abspath(combined_path)
        
        # 尝试从当前工作目录解析（原始行为）
        abs_path = os.path.abspath(data_path)
        if os.path.exists(abs_path):
            return abs_path
    
    # 对于其他相对路径，尝试从环境变量获取基础路径
    from os import getenv
    possible_base_paths = [
        getenv("DATA_DIR", None),
        getenv("ONETERRA_DATA_DIR", None),
        getenv("CODE_DIR", None),
    ]
    
    # 过滤掉 None 值
    possible_base_paths = [p for p in possible_base_paths if p]
    
    # 尝试组合基础路径和数据路径
    for base_path in possible_base_paths:
        if os.path.exists(base_path):
            combined_path = os.path.join(base_path, data_path)
            if os.path.exists(combined_path):
                return os.path.abspath(combined_path)
    
    # 如果都找不到，返回原始路径（让调用者处理错误）
    return data_path


def _resolve_image_path(image_folder, image_file):
    """
    解析图像文件路径，如果文件不存在，尝试去掉重复前缀再查找
    
    Args:
        image_folder: 图像文件夹路径
        image_file: 图像文件名或路径（可能是绝对路径）
    
    Returns:
        实际存在的图像文件路径
    """
    # 如果 image_file 已经是绝对路径，直接使用
    if os.path.isabs(image_file):
        if os.path.exists(image_file):
            return image_file
        # 如果绝对路径不存在，尝试使用相对路径
        image_file = os.path.basename(image_file)
    
    # 使用相对路径组合
    image_path = os.path.join(image_folder, image_file)
    if os.path.exists(image_path):
        return image_path
    
    # 尝试去掉重复前缀（例如：iSAIDPoca_iSAIDPoca_xxx -> iSAIDPoca_xxx）
    file_name_base = os.path.splitext(image_file)[0]
    file_ext = os.path.splitext(image_file)[1]
    parts = file_name_base.split('_')
    
    if len(parts) >= 2 and parts[0] == parts[1]:
        # 去掉重复的前缀
        simplified_name = '_'.join(parts[1:]) + file_ext
        simplified_path = os.path.join(image_folder, simplified_name)
        if os.path.exists(simplified_path):
            return simplified_path
    
    # 如果还是找不到，返回原始路径（让调用者处理错误）
    return image_path


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_folder,
        gt_image_folder=None,
        image_processor=None,
        tokenizer=None,
        task_name="seg",
        data_name="",
        data_mode="train",
        use_random_cat=False,
        special_tokens=None,
        cond_type="phrase",
        extra_image_processor=None,
        preprocess_fn=None,
        postprocess_fn=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        task_length=None,
        pad_image_to_square=False,
        output_ids_with_output=True,
        ignore_label=255,
        sample_num=134,
        repeats_scale=1.0,
        **kwargs,
    ):
        super().__init__()

        assert task_name in DEFAULT_TASKS, f"Invalid dataset type: {task_name}"
        assert data_mode in ["train", "eval", "infer"], f"Invalid dataset mode: {data_mode}"
        assert cond_type in ["phrase", "cls", "all"], f"Invalid cond_type: {cond_type}"
        self.task_name = task_name
        self.data_name = data_name
        self.data_mode = data_mode
        self.use_random_cat = use_random_cat
        # 解析数据路径，尝试找到实际存在的文件
        self.data_path = _resolve_data_path(data_path) if data_path else data_path
        self.image_folder = image_folder
        self.gt_image_folder = gt_image_folder
        self.pad_image_to_square = pad_image_to_square
        self.max_length = max_length
        self.task_length = TASK_MODALITY_LENGTH[task_name] if task_length is None else task_length
        self.ignore_label = ignore_label
        self.sample_num = sample_num
        self.output_ids_with_output = output_ids_with_output
        self.cond_type = cond_type
        self.repeats_scale = repeats_scale
        self.repeats = 1.0

        if isinstance(tokenizer, dict) or isinstance(tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            tokenizer = BUILDER.build(tokenizer)

        if isinstance(dataset_map_fn, str):
            map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(dataset_map_fn)
            if map_fn_obj is not None:
                dataset_map_fn = map_fn_obj
            else:
                raise TypeError(
                    "dataset_map_fn must be a function or a "
                    "registered function's string in MAP_FUNC, "
                    f"but got a string of '{dataset_map_fn}'"
                )
        elif (
            isinstance(dataset_map_fn, dict)
            or isinstance(dataset_map_fn, Config)
            or isinstance(dataset_map_fn, ConfigDict)
        ):
            dataset_map_fn = BUILDER.build(dataset_map_fn)

        if (
            isinstance(template_map_fn, dict)
            or isinstance(template_map_fn, Config)
            or isinstance(template_map_fn, ConfigDict)
        ):
            template_map_fn = BUILDER.build(template_map_fn)

        if (
            isinstance(postprocess_fn, dict)
            or isinstance(postprocess_fn, Config)
            or isinstance(postprocess_fn, ConfigDict)
        ):
            postprocess_fn = BUILDER.build(postprocess_fn)

        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.tokenizer = tokenizer

        if special_tokens is not None:
            assert all(
                token in SPECIAL_TOKENS for token in special_tokens
            ), f"special_tokens must be a subset of {SPECIAL_TOKENS}"
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

            self.seg_token_idx = -1
            self.cls_token_idx = -1
            self.pstart_token_idx = -1
            self.pend_token_idx = -1

            if DEFAULT_SEG_TOKEN in special_tokens:
                self.seg_token_idx = self.tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_CLS_TOKEN in special_tokens:
                self.cls_token_idx = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_PSTART_TOKEN in special_tokens:
                self.pstart_token_idx = self.tokenizer(DEFAULT_PSTART_TOKEN, add_special_tokens=False)["input_ids"][0]
            if DEFAULT_PEND_TOKEN in special_tokens:
                self.pend_token_idx = self.tokenizer(DEFAULT_PEND_TOKEN, add_special_tokens=False)["input_ids"][0]

        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if (
            isinstance(extra_image_processor, dict)
            or isinstance(extra_image_processor, Config)
            or isinstance(extra_image_processor, ConfigDict)
        ):
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        else:
            self.extra_image_processor = extra_image_processor

        self.custom_init(**kwargs)
        self.woann_cnt = 0
        print_log(f"Loading {self.data_name} dataset from {self.data_path}...", logger="current")
        self.data = self.load_ann_data()
        if self.woann_cnt > 0:
            print_log(f"Filtered {self.woann_cnt} images without annotations of {self.data_name}.", logger="current")

    def __len__(self):
        return int(len(self.data) * self.repeats)

    @property
    def repeats(self):
        return self._repeats * self.repeats_scale

    @property
    def modality_length(self):
        return [self.task_length] * int(len(self.data) * self.repeats)

    @property
    def source_length(self):
        return int(len(self.data) * self.repeats)

    @property
    def metadata(self):
        return self._metadata

    @repeats.setter
    def repeats(self, value=1.0):
        self._repeats = value

    def custom_init(self, **kwargs):
        pass

    def _set_metadata(self, **kwargs):
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _get_input_ids(self, data_dict, with_image_token=True):
        if self.tokenizer is None:
            return data_dict

        conversation = None
        if self.dataset_map_fn is not None:
            data_dict = self.dataset_map_fn(data_dict, self.output_ids_with_output)
            conversation = copy.deepcopy(data_dict.get("conversation", None))
            # print_log(f"After dataset_map_fn, conversation: {conversation}", logger="current")
        
        if self.template_map_fn is not None:
            data_dict = self.template_map_fn(data_dict)
            # print_log(f"After template_map_fn, conversation: {conversation}", logger="current")
        
        if self.tokenizer is not None:
            data_dict = encode_fn(
                data_dict, self.tokenizer, self.max_length, self.output_ids_with_output, with_image_token
            )
        
        if conversation is not None:
            data_dict["conversation"] = conversation
        # print_log(f"After encoding, data_dict keys: {list(data_dict.keys())}", logger="current")
        return data_dict

    def _get_cond_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        cond_ids = [-1] * len(input_ids)
        pstart_idx = [i for i, x in enumerate(input_ids) if x == self.pstart_token_idx]
        pend_idx = [i for i, x in enumerate(input_ids) if x == self.pend_token_idx]
        cls_idx = [i for i, x in enumerate(input_ids) if x == self.cls_token_idx]

        if len(pstart_idx) == 0 and len(pend_idx) == 0 and len(cls_idx) == 0:
            return data_dict

        if self.cond_type in ["phrase", "all"]:
            for i, (ps, pe) in enumerate(zip(pstart_idx, pend_idx)):
                cond_ids[ps : pe + 1] = [i] * (pe - ps + 1)
        if self.cond_type in ["cls", "all"]:
            for i, ci in enumerate(cls_idx):
                cond_ids[ci] = i

        data_dict["cond_ids"] = cond_ids
        return data_dict

    def _get_seg_ids(self, data_dict):
        if self.tokenizer is None:
            return data_dict

        input_ids = data_dict["input_ids"]
        seg_ids = [-1] * len(input_ids)

        seg_idx = [i for i, x in enumerate(input_ids) if x == self.seg_token_idx]
        for i, idx in enumerate(seg_idx):
            seg_ids[idx] = i

        data_dict["seg_ids"] = seg_ids
        return data_dict

    def load_ann_data(self):
        data = self._load_ann_data()
        if debug_mode:
            data = data[:debug_iter] + data[-debug_iter:]
        self.data_length = len(data)
        return data

    def _load_ann_data(self):
        pass

    def _decode_mask(self):
        pass
    
    def __getitem__(self, index):
        index = index % self.data_length
        max_skip = 32  # 遇到损坏/截断图像时最多尝试后续 32 个样本，避免单张坏图导致训练中断
        last_error = None
        
        for attempt in range(max_skip):
            idx = (index + attempt) % self.data_length
            data_dict = copy.deepcopy(self.data[idx])
            
            if data_dict.get("image_file", None) is not None:
                image_file = data_dict["image_file"]
                # 使用辅助函数解析图像路径（处理重复前缀问题）
                image_path = _resolve_image_path(self.image_folder, image_file)
                
                try:
                    pil_image = Image.open(image_path).convert("RGB")
                except OSError as e:
                    last_error = e
                    print_log(
                        f"Skipping corrupted/truncated image (attempt {attempt + 1}/{max_skip}): {image_path}: {e}",
                        level="WARNING",
                    )
                    continue  # 尝试下一个样本
                
                # ========== 图像处理（与 X-SAM 一致：s1 等阶段可无 image_processor）==========
                if self.image_processor is not None:
                    image = pil_image
                    actual_image_processor = self.image_processor
                    if hasattr(self.image_processor, "image_processor"):
                        inner = self.image_processor.image_processor
                        if inner is not None:
                            actual_image_processor = inner

                    if self.pad_image_to_square:
                        if hasattr(actual_image_processor, "image_mean"):
                            image_mean = actual_image_processor.image_mean
                        elif hasattr(self.image_processor, "image_mean"):
                            image_mean = self.image_processor.image_mean
                        else:
                            image_mean = [0.5, 0.5, 0.5]
                        image = expand2square(pil_image, tuple(int(x * 255) for x in image_mean))

                    if hasattr(actual_image_processor, "preprocess"):
                        image = actual_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                    else:
                        image = actual_image_processor(image, return_tensors="pt")["pixel_values"][0]
                    data_dict["pixel_values"] = image

                # 额外图像处理器（mask 相关）
                if self.extra_image_processor is not None:
                    data_dict.update(self._decode_mask(data_dict))
                    seg_output = self.extra_image_processor.preprocess(
                        pil_image, data_dict["mask_labels"], return_tensors="pt"
                    )
                    data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
                    data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
                    data_dict["mask_labels"] = seg_output.get("mask_labels", None)
                    data_dict["task_name"] = self.task_name
                # ========== 图像处理结束 ==========
                
                # 编码和返回（成功加载图像的分支）
                data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
                data_dict.update(self._get_cond_ids(data_dict))
                data_dict.update(self._get_seg_ids(data_dict))
                return data_dict  # 成功，直接返回
            
            else:
                # 没有 image_file 的情况，跳出循环用默认空图
                break
        
        # ========== 所有尝试都失败，或没有 image_file，返回空图 ==========
        if self.image_processor is not None:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict["pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        if self.extra_image_processor is not None:
            if hasattr(self.extra_image_processor, "crop_size"):
                crop_size = self.extra_image_processor.crop_size
            else:
                crop_size = self.extra_image_processor.size
            data_dict["seg_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            data_dict["image_info"] = {"image_file": None}
            data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
            data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
            data_dict["mask_labels"] = torch.zeros(0, crop_size["height"], crop_size["width"])
            data_dict["class_labels"] = torch.zeros(0)
            data_dict["task_name"] = self.task_name
        
        data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
        data_dict.update(self._get_cond_ids(data_dict))
        data_dict.update(self._get_seg_ids(data_dict))
        
        # 如果是因为错误导致的退出，抛出异常
        if last_error is not None:
            raise RuntimeError(
                f"Unable to load valid image after {max_skip} attempts. "
                f"Last error: {last_error}"
            ) from last_error
        
        return data_dict

    # def __getitem__(self, index):
    #     index = index % self.data_length
    #     max_skip = 32  # 遇到损坏/截断图像时最多尝试后续 32 个样本，避免单张坏图导致训练中断
    #     last_error = None
    #     for attempt in range(max_skip):
    #         idx = (index + attempt) % self.data_length
    #         data_dict = copy.deepcopy(self.data[idx])
    #         if data_dict.get("image_file", None) is not None:
    #             image_file = data_dict["image_file"]
    #             # 使用辅助函数解析图像路径（处理重复前缀问题）
    #             image_path = _resolve_image_path(self.image_folder, image_file)
    #             try:
    #                 pil_image = Image.open(image_path).convert("RGB")
    #             except OSError as e:
    #                 last_error = e
    #                 print_log(
    #                     f"Skipping corrupted/truncated image (attempt {attempt + 1}/{max_skip}): {image_path}: {e}",
    #                     level="WARNING",
    #                 )
    #                 continue
    #             if self.image_processor is not None:
    #             image = pil_image
    #             # 如果 image_processor 是 SiglipProcessor，需要使用内部的 image_processor 属性
    #             actual_image_processor = self.image_processor
    #             if hasattr(self.image_processor, 'image_processor'):
    #                 actual_image_processor = self.image_processor.image_processor
                
    #             if self.pad_image_to_square:
    #                 # 获取 image_mean，优先使用 actual_image_processor 的，否则使用外层的
    #                 if hasattr(actual_image_processor, 'image_mean'):
    #                     image_mean = actual_image_processor.image_mean
    #                 elif hasattr(self.image_processor, 'image_mean'):
    #                     image_mean = self.image_processor.image_mean
    #                 else:
    #                     # 默认值（SigLIP 的标准均值）
    #                     image_mean = [0.5, 0.5, 0.5]
    #                 image = expand2square(pil_image, tuple(int(x * 255) for x in image_mean))
                
    #             # 使用实际的图像处理器处理图像
    #             if hasattr(actual_image_processor, 'preprocess'):
    #                 image = actual_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    #             else:
    #                 image = actual_image_processor(image, return_tensors="pt")["pixel_values"][0]
    #             data_dict["pixel_values"] = image
    #         if self.extra_image_processor is not None:
    #             data_dict.update(self._decode_mask(data_dict))
    #             seg_output = self.extra_image_processor.preprocess(
    #                 pil_image, data_dict["mask_labels"], return_tensors="pt"
    #             )
    #             data_dict["seg_pixel_values"] = seg_output["pixel_values"][0]
    #             data_dict["scaled_size"] = tuple(seg_output["scaled_sizes"][0].tolist())
    #             data_dict["mask_labels"] = seg_output.get("mask_labels", None)
    #             data_dict["task_name"] = self.task_name
    #         data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
    #         data_dict.update(self._get_cond_ids(data_dict))
    #         data_dict.update(self._get_seg_ids(data_dict))
    #         return data_dict
    #     else:
    #         if hasattr(self.image_processor, "crop_size"):
    #             crop_size = self.image_processor.crop_size
    #         else:
    #             crop_size = self.image_processor.size
    #         data_dict["pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
    #         if self.extra_image_processor is not None:
    #             if hasattr(self.extra_image_processor, "crop_size"):
    #                 crop_size = self.extra_image_processor.crop_size
    #             else:
    #                 crop_size = self.extra_image_processor.size
    #             data_dict["seg_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
    #             data_dict["image_info"] = {"image_file": None}
    #             data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
    #             data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
    #             data_dict["mask_labels"] = torch.zeros(0, crop_size["height"], crop_size["width"])
    #             data_dict["class_labels"] = torch.zeros(0)
    #             data_dict["task_name"] = self.task_name
    #         data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
    #         data_dict.update(self._get_cond_ids(data_dict))
    #         data_dict.update(self._get_seg_ids(data_dict))
    #         return data_dict
    #     if last_error is not None:
    #         raise last_error
    #     raise RuntimeError(
    #         "Unable to load any image after 32 attempts (all corrupted or truncated). "
    #         "Please check your image data."
    #     )
