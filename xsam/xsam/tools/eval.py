#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import re
import sys
import traceback
import warnings
from typing import Dict, Optional, Tuple

# 添加项目根目录到Python路径
# 获取当前文件的目录，然后向上找到项目根目录（包含xsam目录的目录）
current_dir = osp.dirname(osp.abspath(__file__))
# eval.py 在 xsam/xsam/tools/ 下，需要向上3级到项目根目录
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
# 添加项目根目录和xsam目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 同时添加xsam目录（因为xsam模块在xsam/xsam/下）
xsam_dir = osp.join(project_root, "xsam")
if xsam_dir not in sys.path:
    sys.path.insert(0, xsam_dir)

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.dataset.collate_fns import xsam_collate_fn
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.constants import DEFAULT_SEG_TOKEN
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.misc import data_dict_to_device
from xsam.utils.utils import register_function

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for evaluation",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrase_ids = input_ids[ps + 1 : pe - 1]
        if (phrase_ids < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_ids).strip()
        phrases.append(phrase)
    return phrases


def get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = llm_generation_output.split(".")
    sents = [part.strip() for part in parts if DEFAULT_SEG_TOKEN not in part]
    caption = ". ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
    save_llm_output: bool = False,
    llm_outputs_list: Optional[list] = None,
) -> Tuple[bool, Optional[torch.Tensor], Optional[str], Optional[list]]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model
        save_llm_output: Whether to save LLM output
        llm_outputs_list: List to store LLM outputs

    Returns:
        Tuple of (success status, segmentation outputs, llm_generation_output, llm_generation_output_list)
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "seg_pixel_values": data["data_dict"].get("seg_pixel_values", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])

    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)

    llm_generation_output = ""
    with torch.no_grad():
        llm_outputs, seg_outputs = model(
            data_dict,
            data_samples,
            mode=mode,
            generation_config=generation_config,
            stopping_criteria=stop_criteria,
            metadata=metadata,
            do_postprocess=True,
            do_loss=False,
        )

    # Extract LLM generation output
    llm_generation_output_list = []
    if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        llm_generation_output_list = model.tokenizer.batch_decode(llm_outputs.sequences)
        llm_generation_output = llm_generation_output_list[0] if llm_generation_output_list else ""
        
        # Save LLM output if requested
        # 为batch中的每个样本都保存输出（支持batch_size > 1的情况）
        if save_llm_output and llm_outputs_list is not None:
            num_outputs = len(llm_generation_output_list)
            num_images = len(image_files) if isinstance(image_files, list) else 1
            # 确保每个样本都有对应的输出
            for i in range(max(num_outputs, num_images)):
                output_idx = min(i, len(llm_generation_output_list) - 1)
                image_file = image_files[i] if isinstance(image_files, list) and i < len(image_files) else (image_files[0] if image_files else "")
                llm_outputs_list.append({
                    "image_file": image_file,
                    "question": llm_question_input,  # 同一个batch的问题可能相同
                    "answer": llm_generation_output_list[output_idx] if llm_generation_output_list else "",
                })
    else:
        llm_generation_output = ""

    # 对于imgconv任务，seg_outputs为None是正常的（这是对话任务，没有分割输出）
    if seg_outputs is None:
        # 检查是否是imgconv任务
        if "imgconv" in data_name:
            # imgconv任务没有分割输出是正常的，返回成功
            return True, None, llm_generation_output, llm_generation_output_list
        else:
            # 其他任务如果seg_outputs为None则失败
            print_log(
                rf"Failed to get segmentation outputs: {image_files}, "
                rf"llm question_input: {repr(llm_question_input)}, "
                rf"llm generation_output: {repr(llm_generation_output)}",
                logger="current",
            )
            return False, None, llm_generation_output, llm_generation_output_list

    if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        gcg_phrases = [
            get_gcg_phrases(output_ids, model.tokenizer, model.pstart_token_idx, model.pend_token_idx)
            for output_ids in llm_outputs.sequences
        ]
        gcg_captions = [get_gcg_caption(output) for output in llm_generation_output_list]
        for i, segmentation_output in enumerate(seg_outputs):
            segmentation_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

    return True, seg_outputs, llm_generation_output, llm_generation_output_list


def evaluate_dataset(
    model,
    dataset,
    evaluator,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    output_dir: Optional[str] = None,
    visualizer: Optional[object] = None,
    save_visualizations: bool = True,
    max_vis_samples: Optional[int] = None,
) -> None:
    """Evaluate model on a single dataset."""
    data_name = evaluator.data_name
    metadata = dataset.metadata
    output_ids_with_output = dataset.output_ids_with_output
    mode = "tensor" if output_ids_with_output else "predict"

    # Setup dataloader
    sampler = DistributedSampler(dataset=dataset, rank=rank, num_replicas=world_size, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4, sampler=sampler, shuffle=False, collate_fn=xsam_collate_fn
    )

    # Create directories for saving results
    if output_dir is not None and rank == 0:
        vis_dir = osp.join(output_dir, "visualizations", data_name)
        llm_output_dir = osp.join(output_dir, "llm_outputs")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(llm_output_dir, exist_ok=True)

    # Evaluation loop
    failed_cnt = 0
    evaluator.reset()
    llm_outputs_list = []
    vis_count = 0
    processed_count = 0  # Track total processed samples
    
    print_log(f"Evaluating {data_name}...", logger="current")
    if max_vis_samples is not None:
        print_log(f"Will evaluate and visualize first {max_vis_samples} samples only (to save time)", logger="current")
    else:
        print_log(f"Will evaluate and visualize ALL samples in the test set", logger="current")
    print_log(f"Will save visualizations: {save_visualizations}", logger="current")

    for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Evaluating {data_name}", disable=rank != 0)):
        # Check before processing to avoid unnecessary work (only if max_vis_samples is set)
        if max_vis_samples is not None and processed_count >= max_vis_samples:
            print_log(f"Reached max_vis_samples ({max_vis_samples}), stopping evaluation", logger="current")
            break
            
        success, seg_outputs, llm_output, llm_output_list = process_batch(
            model, data, data_name, metadata, generation_config, stop_criteria, mode,
            save_llm_output=True, llm_outputs_list=llm_outputs_list
        )
        if not success:
            failed_cnt += 1
            continue

        image_infos = data["data_samples"].metainfo["image_infos"]
        # Count actual number of samples in this batch
        num_samples_in_batch = len(image_infos) if isinstance(image_infos, list) else 1
        
        # Check if adding this batch would exceed the limit (only if max_vis_samples is set)
        if max_vis_samples is not None and processed_count + num_samples_in_batch > max_vis_samples:
            # Only process the samples that fit within the limit
            remaining = max_vis_samples - processed_count
            if remaining > 0:
                image_infos = image_infos[:remaining]
                if isinstance(seg_outputs, list):
                    seg_outputs = seg_outputs[:remaining]
                if llm_output_list:
                    llm_output_list = llm_output_list[:remaining]
                evaluator.process(image_infos, seg_outputs)
                processed_count += remaining
            break
        
        # 对于imgconv任务，即使seg_outputs为None也要处理
        if "imgconv" in data_name:
            # imgconv任务：将问题和答案传递给evaluator
            # 直接从process_batch返回的llm_output_list中获取答案
            if llm_output_list is None:
                llm_output_list = []
            for i, img_info in enumerate(image_infos):
                if isinstance(img_info, dict):
                    # 从当前batch的llm_output_list获取对应的预测答案
                    if i < len(llm_output_list):
                        pred_answer = llm_output_list[i]
                        # 从input_ids中提取问题
                        if data["data_dict"].get("input_ids") is not None:
                            _input_ids = data["data_dict"]["input_ids"]
                            if isinstance(_input_ids, list) and i < len(_input_ids):
                                question = model.tokenizer.decode(_input_ids[i][_input_ids[i] > 0])
                            else:
                                question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                        else:
                            question = ""
                        if hasattr(evaluator, 'add_prediction'):
                            evaluator.add_prediction(pred_answer, question, img_info.get("file_name", ""))
                    # 处理ground truth（从image_info的conversations中提取）
                    evaluator.process([img_info], None)
        else:
            # 其他任务正常处理
            evaluator.process(image_infos, seg_outputs)
        
        processed_count += num_samples_in_batch

        # Save visualizations and print LLM outputs
        # 对于imgconv任务，即使seg_outputs为None也要可视化
        should_visualize = (
            rank == 0 and 
            save_visualizations and 
            (max_vis_samples is None or vis_count < max_vis_samples) and
            (seg_outputs is not None or "imgconv" in data_name)
        )
        if should_visualize:
            try:
                # Print LLM output
                if llm_output:
                    print_log(f"\n{'='*80}", logger="current")
                    print_log(f"Sample {batch_idx + 1} - {data_name}", logger="current")
                    print_log(f"Image: {image_infos[0].get('file_name', 'unknown') if image_infos else 'unknown'}", logger="current")
                    # 从data中获取问题，而不是从llm_outputs_list
                    if data["data_dict"].get("input_ids") is not None:
                        _input_ids = data["data_dict"]["input_ids"]
                        if isinstance(_input_ids, (list, torch.Tensor)) and len(_input_ids) > 0:
                            if isinstance(_input_ids, torch.Tensor):
                                question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                            else:
                                question = model.tokenizer.decode(_input_ids[0][_input_ids[0] > 0])
                        else:
                            question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                    else:
                        question = "N/A"
                    print_log(f"LLM Question: {question}", logger="current")
                    print_log(f"LLM Answer: {llm_output}", logger="current")
                    print_log(f"{'='*80}\n", logger="current")

                # Debug: Check conditions for visualization
                if batch_idx < 3:  # Only log first 3 samples to avoid too much output
                    print_log(f"DEBUG: batch_idx={batch_idx}, visualizer={visualizer is not None}, seg_outputs_len={len(seg_outputs) if seg_outputs else 0}", logger="current")
                    if hasattr(dataset, "image_folder"):
                        print_log(f"DEBUG: dataset.image_folder={dataset.image_folder}", logger="current")
                    else:
                        print_log(f"DEBUG: dataset does not have image_folder attribute", logger="current")

                # Save visualization if visualizer is available
                # 对于imgconv任务，即使seg_outputs为None也要可视化
                if visualizer is not None and (len(seg_outputs) > 0 if seg_outputs is not None else "imgconv" in data_name):
                    try:
                        # Get original image
                        # First try to get full path from image_infos
                        image_file = image_infos[0].get("file_name", "")
                        
                        # If image_file is just a filename (not a full path), combine with dataset.image_folder
                        if image_file and not osp.isabs(image_file) and hasattr(dataset, "image_folder") and dataset.image_folder:
                            image_file = osp.join(dataset.image_folder, image_file)
                        
                        if batch_idx < 3:
                            print_log(f"DEBUG: image_file from image_infos: {image_file}, exists: {osp.exists(image_file) if image_file else False}", logger="current")
                        
                        if image_file and osp.exists(image_file):
                            sample_image = np.array(Image.open(image_file).convert("RGB"))
                        else:
                            # Try to get from data_samples
                            if hasattr(data["data_samples"], "image_files") and data["data_samples"].image_files:
                                image_file = data["data_samples"].image_files[0]
                                # If it's a relative path, combine with dataset.image_folder
                                if image_file and not osp.isabs(image_file) and hasattr(dataset, "image_folder") and dataset.image_folder:
                                    image_file = osp.join(dataset.image_folder, image_file)
                                if batch_idx < 3:
                                    print_log(f"DEBUG: image_file from data_samples: {image_file}, exists: {osp.exists(image_file) if image_file else False}", logger="current")
                                if osp.exists(image_file):
                                    sample_image = np.array(Image.open(image_file).convert("RGB"))
                                else:
                                    sample_image = None
                            else:
                                sample_image = None
                        
                        if batch_idx < 3:
                            print_log(f"DEBUG: sample_image is None: {sample_image is None}", logger="current")

                        # 对于imgconv任务，即使seg_outputs为None也要可视化
                        if sample_image is not None and (len(seg_outputs) > 0 if seg_outputs is not None else "imgconv" in data_name):
                            # 获取原始图片文件名用于命名可视化文件
                            original_image_file = ""
                            if image_infos and len(image_infos) > 0:
                                img_info = image_infos[0]
                                if isinstance(img_info, dict):
                                    original_image_file = img_info.get("file_name", "")
                                elif hasattr(img_info, "file_name"):
                                    original_image_file = img_info.file_name
                            
                            # 如果没有从image_infos获取到，尝试从data_samples获取
                            if not original_image_file and hasattr(data["data_samples"], "image_files") and data["data_samples"].image_files:
                                original_image_file = data["data_samples"].image_files[0]
                            
                            # 生成可视化文件名：使用原始图片文件名，如果没有则使用sample_编号
                            if original_image_file:
                                # 提取文件名（不含路径）并替换扩展名为.png
                                base_name = osp.splitext(osp.basename(original_image_file))[0]
                                vis_filename = f"{base_name}.png"
                            else:
                                vis_filename = f"sample_{batch_idx:05d}.png"
                            
                            vis_output_file = osp.join(vis_dir, vis_filename)
                            
                            # Get question text for refseg and imgconv tasks
                            question = None
                            answer = None
                            
                            if "imgconv" in data_name:
                                # 对于imgconv任务，从当前batch的llm_output_list获取答案，从data中获取问题
                                if llm_output_list and len(llm_output_list) > 0:
                                    answer = llm_output_list[0]  # 使用当前batch的第一个输出
                                else:
                                    answer = llm_output if llm_output else ""
                                # 从input_ids中提取问题
                                if data["data_dict"].get("input_ids") is not None:
                                    _input_ids = data["data_dict"]["input_ids"]
                                    if isinstance(_input_ids, (list, torch.Tensor)) and len(_input_ids) > 0:
                                        if isinstance(_input_ids, torch.Tensor):
                                            question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                                        else:
                                            question = model.tokenizer.decode(_input_ids[0][_input_ids[0] > 0])
                                    else:
                                        question = model.tokenizer.decode(_input_ids[_input_ids > 0])
                                else:
                                    question = ""
                                
                                # Draw predictions for imgconv task
                                try:
                                    visualizer.draw_predictions(
                                        sample_image,
                                        data_name=data_name,
                                        output_file=vis_output_file,
                                        question=question,
                                        answer=answer,
                                    )
                                    vis_count += 1
                                    if batch_idx < 3:
                                        print_log(f"Successfully saved imgconv visualization to {vis_output_file}", logger="current")
                                except Exception as e:
                                    print_log(f"Error saving imgconv visualization for sample {batch_idx}: {e}", logger="current")
                            else:
                                # 其他任务：处理seg_outputs
                                # Get the first segmentation output
                                seg_output = seg_outputs[0]
                                
                                # Convert to dict if it's an object
                                if isinstance(seg_output, dict):
                                    vis_kwargs = seg_output.copy()
                                elif hasattr(seg_output, "__dict__"):
                                    vis_kwargs = seg_output.__dict__.copy()
                                elif hasattr(seg_output, "to_dict"):
                                    vis_kwargs = seg_output.to_dict()
                                else:
                                    vis_kwargs = {}
                                
                                if batch_idx < 3:
                                    print_log(f"DEBUG: seg_output type: {type(seg_output)}, vis_kwargs keys: {list(vis_kwargs.keys())}", logger="current")
                                
                                # Get phrases if available, default to empty list if None
                                phrases = vis_kwargs.get("gcg_phrases", None)
                                if phrases is None:
                                    phrases = []
                                
                                # Ensure segments_info has 'isthing' key for each segment
                                # _PanopticPrediction.semantic_masks() and instance_masks() require 'isthing' key
                                if "segments_info" in vis_kwargs and isinstance(vis_kwargs["segments_info"], list):
                                    segments_info = vis_kwargs["segments_info"]
                                    # Check if metadata has thing_dataset_id_to_contiguous_id
                                    thing_contiguous_ids = set()
                                    if metadata is not None and hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
                                        thing_contiguous_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())
                                    
                                    # Ensure each segment info dict has 'isthing' key
                                    for seg_info in segments_info:
                                        if isinstance(seg_info, dict) and "isthing" not in seg_info:
                                            # Try to determine isthing from category_id
                                            category_id = seg_info.get("category_id", None)
                                            if category_id is not None and thing_contiguous_ids:
                                                seg_info["isthing"] = category_id in thing_contiguous_ids
                                            else:
                                                # Default to False (stuff) if cannot determine
                                                seg_info["isthing"] = False
                                
                                # Ensure segmentation is torch.Tensor (visualizer expects tensor, not numpy array)
                                # _PanopticPrediction.__init__ calls torch.unique(segmentation) which requires a Tensor
                                if "segmentation" in vis_kwargs:
                                    seg = vis_kwargs["segmentation"]
                                    # Convert to tensor if needed
                                    if isinstance(seg, np.ndarray):
                                        # Convert numpy array to tensor, ensuring correct dtype for segmentation masks
                                        seg_tensor = torch.from_numpy(seg.copy()).cpu()  # Use .copy() to avoid memory sharing issues
                                        # Ensure integer dtype for segmentation masks
                                        if seg_tensor.dtype not in (torch.int32, torch.int64, torch.long):
                                            seg_tensor = seg_tensor.long()
                                        vis_kwargs["segmentation"] = seg_tensor
                                    elif torch.is_tensor(seg):
                                        # Ensure it's on CPU and has correct dtype
                                        seg_tensor = seg.cpu()
                                        if seg_tensor.dtype not in (torch.int32, torch.int64, torch.long):
                                            seg_tensor = seg_tensor.long()
                                        vis_kwargs["segmentation"] = seg_tensor
                                    else:
                                        # Convert other types to tensor
                                        seg_tensor = torch.tensor(seg, dtype=torch.long).cpu()
                                        vis_kwargs["segmentation"] = seg_tensor
                                    
                                    if batch_idx < 3:
                                        print_log(f"DEBUG: segmentation type: {type(vis_kwargs['segmentation'])}, shape: {vis_kwargs['segmentation'].shape}, dtype: {vis_kwargs['segmentation'].dtype}, is_tensor: {torch.is_tensor(vis_kwargs['segmentation'])}", logger="current")
                                
                                # Remove non-visualization keys
                                vis_kwargs.pop("gcg_phrases", None)
                                vis_kwargs.pop("gcg_caption", None)
                                
                                if batch_idx < 3:
                                    print_log(f"DEBUG: About to call draw_predictions, output_file: {vis_output_file}", logger="current")
                                
                                # Get question text for refseg and reaseg tasks
                                if "refseg" in data_name or "reaseg" in data_name:
                                    # Get question from image_infos (phrases or sampled_sents)
                                    if image_infos and len(image_infos) > 0:
                                        img_info = image_infos[0]
                                        # Try to get phrases or sampled_sents from image_info
                                        if "phrases" in img_info and img_info["phrases"]:
                                            # phrases might be a list, get the first one
                                            phrases_list = img_info["phrases"]
                                            if isinstance(phrases_list, list) and len(phrases_list) > 0:
                                                question = phrases_list[0]
                                            elif isinstance(phrases_list, str):
                                                question = phrases_list
                                        elif "sampled_sents" in img_info and img_info["sampled_sents"]:
                                            # sampled_sents might be a list, get the first one
                                            sents_list = img_info["sampled_sents"]
                                            if isinstance(sents_list, list) and len(sents_list) > 0:
                                                question = sents_list[0]
                                            elif isinstance(sents_list, str):
                                                question = sents_list
                                    if batch_idx < 3:
                                        print_log(f"DEBUG: refseg question from image_infos: {question}", logger="current")
                                
                                # Draw predictions for other tasks
                                try:
                                    visualizer.draw_predictions(
                                        sample_image,
                                        data_name=data_name,
                                        output_file=vis_output_file,
                                        phrases=phrases,
                                        question=question,
                                        **vis_kwargs,
                                    )
                                    vis_count += 1
                                    if batch_idx < 3 or vis_count % 10 == 0:
                                        print_log(f"Successfully saved visualization {vis_count} to {vis_output_file}", logger="current")
                                except Exception as vis_e:
                                    print_log(f"Error in draw_predictions for sample {batch_idx}: {vis_e}", logger="current")
                                    print_log(f"  seg_output keys: {list(vis_kwargs.keys())}", logger="current")
                                    import traceback
                                    print_log(f"  Traceback: {traceback.format_exc()}", logger="current")
                        else:
                            if batch_idx < 3:
                                if sample_image is None:
                                    print_log(f"DEBUG: Skipping visualization - sample_image is None", logger="current")
                                if len(seg_outputs) == 0:
                                    print_log(f"DEBUG: Skipping visualization - seg_outputs is empty", logger="current")
                    except Exception as e:
                        print_log(f"Error saving visualization for sample {batch_idx}: {e}", logger="current")
            except Exception as e:
                print_log(f"Error processing visualization for sample {batch_idx}: {e}", logger="current")

    # Save LLM outputs to JSON file
    if rank == 0 and output_dir is not None and llm_outputs_list:
        llm_output_file = osp.join(llm_output_dir, f"{data_name}_llm_outputs.json")
        with open(llm_output_file, "w", encoding="utf-8") as f:
            json.dump(llm_outputs_list, f, indent=2, ensure_ascii=False)
        print_log(f"Saved {len(llm_outputs_list)} LLM outputs to {llm_output_file}", logger="current")

    print_log(f"Processed {processed_count} samples for {data_name}", logger="current")
    print_log(f"Failed number of {data_name}: {failed_cnt}", logger="current")
    print_log(f"Evaluating {data_name} done!", logger="current")
    if rank == 0 and save_visualizations:
        print_log(f"Saved {vis_count} visualizations to {vis_dir}", logger="current")
    
    # 执行评估并获取结果
    eval_results = evaluator.evaluate()
    
    # 保存评估结果到txt文件
    if rank == 0 and output_dir is not None and eval_results is not None:
        results_txt_file = osp.join(output_dir, "evaluation_results", f"{data_name}_results.txt")
        os.makedirs(osp.dirname(results_txt_file), exist_ok=True)
        
        with open(results_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results for {data_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Number of samples: {processed_count}\n")
            f.write(f"Failed samples: {failed_cnt}\n\n")
            
            # 根据评估器类型格式化结果
            if isinstance(eval_results, dict):
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                    elif isinstance(value, (list, tuple)):
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            elif isinstance(eval_results, str):
                # 如果是字符串（表格格式），直接写入
                f.write(eval_results)
            elif hasattr(eval_results, '__str__'):
                f.write(str(eval_results))
            else:
                f.write(f"Results: {eval_results}\n")
        
        print_log(f"Evaluation results saved to: {results_txt_file}", logger="current")


def main():
    """Main evaluation function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)

    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        else:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")
    
    # Handle DeepSpeed checkpoint directory (iter_*.pth)
    if args.pth_model and osp.isdir(args.pth_model):
        # Check if it's a DeepSpeed checkpoint directory
        model_states_file = osp.join(args.pth_model, "mp_rank_00_model_states.pt")
        pytorch_model_file = osp.join(args.pth_model, "pytorch_model.bin")
        
        if osp.exists(model_states_file):
            # Use the model states file for DeepSpeed checkpoints
            args.pth_model = model_states_file
            print_log(f"Detected DeepSpeed checkpoint, using: {args.pth_model}", logger="current")
        elif osp.exists(pytorch_model_file):
            # Use pytorch_model.bin if it exists in the directory
            args.pth_model = pytorch_model_file
            print_log(f"Using pytorch_model.bin from checkpoint directory: {args.pth_model}", logger="current")
        else:
            # Try to use the directory itself (guess_load_checkpoint might handle it)
            print_log(f"Using checkpoint directory: {args.pth_model}", logger="current")
            print_log("Note: If loading fails, try specifying the full path to mp_rank_00_model_states.pt", logger="current")

    # Build and setup model
    model = BUILDER.build(cfg.model)
    if "llm" in cfg.model:
        model.llm.to(cfg.model.llm.torch_dtype)
    model.eval()
    model = model.to(get_device())
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank]).module

    load_checkpoint(model, args.pth_model)
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Setup visualizer if available in config
    visualizer = None
    if hasattr(cfg, "visualizer") and cfg.visualizer is not None:
        try:
            from xsam.utils.visualize import Visualizer
            visualizer = BUILDER.build(cfg.visualizer)
            print_log("Visualizer initialized successfully", logger="current")
        except Exception as e:
            print_log(f"Warning: Could not initialize visualizer: {e}", logger="current")
            visualizer = None

    # Evaluate on all datasets
    # 支持通过环境变量过滤任务
    eval_only_task = os.environ.get("EVAL_ONLY_TASK", None)
    if eval_only_task:
        print_log(f"只评估任务: {eval_only_task}", logger="current")
        cfg.val_datasets = [d for d in cfg.val_datasets if d.get('task_name') == eval_only_task]
        cfg.val_evaluators = [e for e in cfg.val_evaluators if eval_only_task in e.get('data_name', '')]
        print_log(f"过滤后: {len(cfg.val_datasets)} 个数据集, {len(cfg.val_evaluators)} 个评估器", logger="current")
    
    assert len(cfg.val_datasets) == len(
        cfg.val_evaluators
    ), f"len(cfg.val_datasets) = {len(cfg.val_datasets)}, len(cfg.val_evaluators) = {len(cfg.val_evaluators)}"
    print_log(f"Evaluating {len(cfg.val_datasets)} datasets...", logger="current")
    print_log(f"Results will be saved to: {args.work_dir}", logger="current")
    
    for dataset_cfg, evaluator_cfg in zip(cfg.val_datasets, cfg.val_evaluators):
        dataset = BUILDER.build(dataset_cfg)
        model.postprocess_fn = dataset.postprocess_fn

        evaluator = BUILDER.build(evaluator_cfg)
        evaluator.metadata = dataset.metadata
        evaluator.output_dir = osp.join(args.work_dir, "pred_data", evaluator.data_name)

        try:
            evaluate_dataset(
                model, dataset, evaluator, rank, world_size, generation_config, stop_criteria,
                output_dir=args.work_dir,
                visualizer=visualizer,
                save_visualizations=True,
                max_vis_samples=None,  # Process all samples in the test set
            )
        except Exception as e:
            print_log(f"Error evaluating {evaluator.data_name}: {e}\n{traceback.format_exc()}", logger="current")
            continue
    
    print_log(f"\n{'='*80}", logger="current")
    print_log("Evaluation completed!", logger="current")
    print_log(f"Results saved to: {args.work_dir}", logger="current")
    print_log(f"  - Predictions: {args.work_dir}/pred_data/", logger="current")
    print_log(f"  - Visualizations: {args.work_dir}/visualizations/", logger="current")
    print_log(f"  - LLM Outputs: {args.work_dir}/llm_outputs/", logger="current")
    print_log(f"  - Evaluation Results (TXT): {args.work_dir}/evaluation_results/", logger="current")
    print_log(f"{'='*80}\n", logger="current")


if __name__ == "__main__":
    main()
