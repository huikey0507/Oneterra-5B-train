#!/usr/bin/env python
"""
交互式多任务测试Demo - 支持4个任务
1. imgconv - 图像对话（GeoChat）
2. genseg - 通用分割
3. ovseg - 开放词汇分割
4. refseg - 指代分割

使用方法:
    python xsam/xsam/tools/interactive_multi_task_demo.py \
        xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat.py \
        --pth_model ./wkdrs/s3_mixed_finetune_geochat/xxx/iter_xxx.pth
"""

import argparse
import os
import os.path as osp
import sys
import traceback
import warnings
from typing import Optional

# 添加项目根目录到Python路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
xsam_dir = osp.join(project_root, "xsam")
if xsam_dir not in sys.path:
    sys.path.insert(0, xsam_dir)

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.dist import setup_distributed
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

# 导入XSamDemo
from xsam.demo.demo import XSamDemo

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

# Task descriptions (English)
TASK_DESCRIPTIONS = {
    "imgconv": {
        "name": "Image Conversation",
        "description": "Answer questions about the image (English prompts)",
        "prompt_example": "Please describe this image in detail.",
        "prompt_hint": "Enter your question in English, e.g., 'What is in this image?', 'Please describe the main content of the image'"
    },
    "genseg": {
        "name": "Generic Segmentation",
        "description": "Segment objects using SOTA dataset categories (no prompt needed, uses all SOTA categories by default)",
        "prompt_example": "(No prompt needed - will use all SOTA dataset categories by default)",
        "prompt_hint": "Generic segmentation uses all predefined SOTA categories by default. You can leave the prompt empty, or optionally specify: ins: category1, category2, ...; sem: category1, category2, ..."
    },
    "ovseg": {
        "name": "Open-Vocabulary Segmentation",
        "description": "Segment objects using open vocabulary (panoptic segmentation with thing and stuff classes)",
        "prompt_example": "thing: person, car; stuff: tree, building",
        "prompt_hint": "Enter categories in format: 'thing: category1, category2; stuff: category3, category4' or simply 'category1, category2, category3' (will try to determine thing/stuff from dataset)"
    },
    "refseg": {
        "name": "Referring Segmentation",
        "description": "Segment objects by referring expressions",
        "prompt_example": "the white tshirt kid",
        "prompt_hint": "Enter a referring expression, e.g., 'the red car on the left', 'the person wearing blue shirt'"
    }
}

SUPPORTED_TASKS = list(TASK_DESCRIPTIONS.keys())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="交互式多任务测试Demo - 支持4个任务测试"
    )
    parser.add_argument("config", help="config file name or path")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_outputs",
        help="directory to save output images",
    )
    return parser.parse_args()


def handle_checkpoint_path(pth_model: Optional[str], work_dir: Optional[str] = None) -> Optional[str]:
    """Handle checkpoint path, including DeepSpeed checkpoints."""
    if pth_model is None:
        return None
    
    if pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint
        if work_dir and osp.exists(osp.join(work_dir, "pytorch_model.bin")):
            pth_model = osp.join(work_dir, "pytorch_model.bin")
        elif work_dir:
            pth_model = find_latest_checkpoint(work_dir)
        else:
            raise ValueError("work_dir must be specified when using 'latest' checkpoint")
        print_log(f"Found latest checkpoint: {pth_model}", logger="current")
    
    # Handle DeepSpeed checkpoint directory (iter_*.pth)
    if osp.isdir(pth_model):
        model_states_file = osp.join(pth_model, "mp_rank_00_model_states.pt")
        pytorch_model_file = osp.join(pth_model, "pytorch_model.bin")
        
        if osp.exists(model_states_file):
            pth_model = model_states_file
            print_log(f"Detected DeepSpeed checkpoint, using: {pth_model}", logger="current")
        elif osp.exists(pytorch_model_file):
            pth_model = pytorch_model_file
            print_log(f"Using pytorch_model.bin from checkpoint directory: {pth_model}", logger="current")
        else:
            print_log(f"Using checkpoint directory: {pth_model}", logger="current")
            print_log("Note: If loading fails, try specifying the full path to mp_rank_00_model_states.pt", logger="current")
    
    return pth_model


def resolve_image_path(image_path: str, project_root: str) -> Optional[str]:
    """解析图片路径，支持绝对路径和相对路径。"""
    # 如果是绝对路径且存在，直接返回
    if osp.isabs(image_path) and osp.exists(image_path):
        return osp.abspath(image_path)
    
    # 尝试相对于当前工作目录
    if osp.exists(image_path):
        return osp.abspath(image_path)
    
    # 尝试相对于项目根目录
    project_path = osp.join(project_root, image_path)
    if osp.exists(project_path):
        return osp.abspath(project_path)
    
    # 尝试相对于脚本所在目录
    script_dir = osp.dirname(osp.abspath(__file__))
    script_path = osp.join(script_dir, image_path)
    if osp.exists(script_path):
        return osp.abspath(script_path)
    
    return None


def print_task_menu():
    """打印任务选择菜单"""
    print("\n" + "=" * 80)
    print("请选择要测试的任务:")
    print("=" * 80)
    for i, task in enumerate(SUPPORTED_TASKS, 1):
        task_info = TASK_DESCRIPTIONS[task]
        print(f"  {i}. {task_info['name']}")
        print(f"     描述: {task_info['description']}")
        print(f"     示例: {task_info['prompt_example']}")
    print("=" * 80)


def save_result(image, task_name: str, output_dir: str, suffix: str = ""):
    """保存结果图片"""
    os.makedirs(output_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{timestamp}{suffix}.png"
    filepath = osp.join(output_dir, filename)
    
    # 确保是PIL Image
    if not isinstance(image, Image.Image):
        if hasattr(image, 'get_image'):
            image = image.get_image()
        else:
            import numpy as np
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                print(f"⚠️  无法保存结果，未知的图像类型: {type(image)}")
                return None
    
    image.save(filepath)
    print(f"💾 结果已保存到: {filepath}")
    return filepath


def interactive_demo(demo: XSamDemo, output_dir: str):
    """交互式测试循环"""
    current_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
    
    print("\n" + "=" * 80)
    print("X-SAM 多任务交互式测试Demo")
    print("=" * 80)
    print("支持的任务:")
    for task in SUPPORTED_TASKS:
        print(f"  - {TASK_DESCRIPTIONS[task]['name']}")
    print("\n使用说明:")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'menu' 显示任务菜单")
    print("  - 输入 'new' 切换新图片")
    print("=" * 80 + "\n")
    
    current_image = None
    current_image_path = None
    current_task = None
    
    while True:
        try:
            # 选择任务
            if current_task is None:
                print_task_menu()
                task_input = input("\n请输入任务编号 (1-4) 或任务名称: ").strip()
                
                if task_input.lower() in ['quit', 'exit', 'q']:
                    print("\n再见！")
                    break
                elif task_input.lower() == 'menu':
                    continue
                
                # 解析任务选择
                if task_input.isdigit():
                    task_idx = int(task_input) - 1
                    if 0 <= task_idx < len(SUPPORTED_TASKS):
                        current_task = SUPPORTED_TASKS[task_idx]
                    else:
                        print(f"❌ 无效的任务编号，请输入 1-{len(SUPPORTED_TASKS)}")
                        continue
                elif task_input.lower() in [t.lower() for t in SUPPORTED_TASKS]:
                    current_task = task_input.lower()
                    # 找到匹配的任务名
                    for task in SUPPORTED_TASKS:
                        if task.lower() == task_input.lower():
                            current_task = task
                            break
                else:
                    print(f"❌ 无效的任务名称，请选择: {', '.join(SUPPORTED_TASKS)}")
                    continue
                
                task_info = TASK_DESCRIPTIONS[current_task]
                print(f"\n✅ 已选择任务: {task_info['name']}")
                print(f"📋 任务描述: {task_info['description']}")
                print(f"💡 提示: {task_info['prompt_hint']}")
                print(f"📝 示例: {task_info['prompt_example']}\n")
            
            # 获取图片路径
            if current_image is None:
                image_path = input("\n请输入图片路径 (或输入 'quit' 退出, 'task' 切换任务): ").strip()
                if image_path.lower() in ['quit', 'exit', 'q']:
                    print("\n再见！")
                    break
                elif image_path.lower() in ['task', 't']:
                    current_task = None
                    continue
                
                # 解析图片路径
                resolved_path = resolve_image_path(image_path, project_root)
                if resolved_path is None:
                    print(f"❌ 错误: 图片文件不存在: {image_path}")
                    print(f"   提示: 请检查路径是否正确（支持绝对路径和相对路径）")
                    continue
                
                try:
                    current_image = Image.open(resolved_path).convert("RGB")
                    current_image_path = resolved_path
                    print(f"✅ 成功加载图片: {resolved_path}")
                    print(f"   图片尺寸: {current_image.size}")
                except Exception as e:
                    print(f"❌ 错误: 无法加载图片: {e}")
                    current_image = None
                    continue
            
            # 获取prompt
            task_info = TASK_DESCRIPTIONS[current_task]
            prompt = input(f"\n请输入prompt (输入 'new' 切换新图片, 'task' 切换任务, 'quit' 退出):\n提示: {task_info['prompt_hint']}\n> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            elif prompt.lower() in ['new', 'new_image', 'n']:
                current_image = None
                current_image_path = None
                print("已清除当前图片，请重新输入图片路径")
                continue
            elif prompt.lower() in ['task', 't']:
                current_task = None
                current_image = None
                current_image_path = None
                print("已清除当前任务和图片")
                continue
            elif not prompt:
                print("⚠️  prompt不能为空，请重新输入")
                continue
            
            # 运行推理
            print(f"\n{'='*80}")
            print(f"🖼️  图片: {current_image_path}")
            print(f"🎯 任务: {task_info['name']}")
            print(f"❓ Prompt: {prompt}")
            print("🤔 正在处理...")
            print(f"{'='*80}\n")
            
            try:
                llm_input, generation_output, visualized_image = demo.run_on_image(
                    image=current_image,
                    prompt=prompt,
                    task_name=current_task,
                    vprompt_masks=None
                )
                
                # 显示结果
                if generation_output and generation_output.strip():
                    print(f"\n💬 模型回答:\n{generation_output}\n")
                else:
                    print("⚠️  模型未生成文本回答（这对于分割任务是正常的）\n")
                
                # 显示可视化结果
                if visualized_image is not None:
                    print("✅ 已生成分割结果")
                    # 保存结果
                    save_result(visualized_image, current_task, output_dir)
                else:
                    if current_task == "imgconv":
                        print("ℹ️  对话任务通常不生成可视化结果\n")
                    else:
                        print("⚠️  未生成可视化结果（可能是分割失败或任务不支持可视化）\n")
                
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                import traceback
                traceback.print_exc()
                print()
                
        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，退出...")
            break
        except EOFError:
            print("\n\n检测到 EOF，退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            print()


def main():
    """Main function for interactive multi-task demo."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)
    
    if world_size > 1:
        print_log("Warning: Interactive demo mode is designed for single GPU. Using rank 0 only.", logger="current")
    
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
        set_random_seed(args.seed)
        print_log(f"Set the random seed to {args.seed}.", logger="current")
    
    # Handle checkpoint path
    args.pth_model = handle_checkpoint_path(args.pth_model)
    
    if args.pth_model is None:
        raise ValueError("--pth_model must be specified")
    
    # Create demo instance
    print_log("=" * 80, logger="current")
    print_log("初始化模型...", logger="current")
    print_log("=" * 80, logger="current")
    
    try:
        demo = XSamDemo(
            cfg=cfg,
            pth_model=args.pth_model,
            output_ids_with_output=False,
        )
        print_log("✅ 模型初始化成功！", logger="current")
    except Exception as e:
        print_log(f"❌ 模型初始化失败: {e}", logger="current")
        traceback.print_exc()
        return
    
    # Start interactive demo
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        interactive_demo(demo, args.output_dir)
    else:
        print_log("Non-zero rank, skipping interactive demo", logger="current")


if __name__ == "__main__":
    main()

