#!/usr/bin/env python
"""
交互式对话脚本 - 可以给一张图问一个问题
使用方法:
    python xsam/xsam/tools/interactive_chat.py \
        xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py \
        --pth_model ./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_46000.pth
"""

import argparse
import os.path as osp
import sys
import traceback
import warnings
from typing import Optional

# 添加项目根目录到Python路径（必须在导入xsam模块之前）
# 获取当前文件的目录，然后向上找到项目根目录（包含xsam目录的目录）
current_dir = osp.dirname(osp.abspath(__file__))
# interactive_chat.py 在 xsam/xsam/tools/ 下，需要向上3级到项目根目录
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
# 添加项目根目录和xsam目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 同时添加xsam目录（因为xsam模块在xsam/xsam/下）
xsam_dir = osp.join(project_root, "xsam")
if xsam_dir not in sys.path:
    sys.path.insert(0, xsam_dir)

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for X-SAM model - 可以给一张图问一个问题"
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
        "--task_name",
        type=str,
        default="imgconv",
        help="task name for conversation (default: imgconv for VQA/chat)",
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
    """解析图片路径，支持绝对路径和相对路径。
    
    Args:
        image_path: 用户输入的图片路径
        project_root: 项目根目录路径
        
    Returns:
        解析后的绝对路径，如果不存在则返回None
    """
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


def interactive_chat(demo: XSamDemo, task_name: str = "imgconv"):
    """Interactive chat loop."""
    # 获取项目根目录（用于解析相对路径）
    current_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
    
    print("\n" + "=" * 80)
    print("X-SAM 交互式对话界面")
    print("=" * 80)
    print("使用说明:")
    print("  - 输入图片路径（支持绝对路径或相对路径），然后输入问题")
    print("  - 相对路径相对于当前工作目录或项目根目录")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'new' 或 'new_image' 切换新图片")
    print("=" * 80 + "\n")
    
    current_image = None
    current_image_path = None
    
    while True:
        try:
            # 获取图片路径
            if current_image is None:
                image_path = input("\n请输入图片路径 (或输入 'quit' 退出): ").strip()
                if image_path.lower() in ['quit', 'exit', 'q']:
                    print("\n再见！")
                    break
                
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
                except Exception as e:
                    print(f"❌ 错误: 无法加载图片: {e}")
                    current_image = None
                    continue
            
            # 获取问题
            question = input(f"\n请输入您的问题 (输入 'new' 切换新图片, 'quit' 退出): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            elif question.lower() in ['new', 'new_image', 'n']:
                current_image = None
                current_image_path = None
                print("已清除当前图片，请重新输入图片路径")
                continue
            elif not question:
                print("⚠️  问题不能为空，请重新输入")
                continue
            
            # 处理问题
            print(f"\n🖼️  图片: {current_image_path}")
            print(f"❓ 问题: {question}")
            print("🤔 正在思考...")
            
            try:
                llm_input, generation_output, visualized_image = demo.run_on_image(
                    image=current_image,
                    prompt=question,
                    task_name=task_name,
                    vprompt_masks=None
                )
                
                if generation_output and generation_output.strip():
                    # 直接显示模型生成的原始输出（已经去除了输入问题部分）
                    print(f"\n💬 回答: {generation_output}\n")
                    
                    # 对于对话任务，可视化是可选的
                    if visualized_image is not None:
                        print("✅ 已生成可视化结果（如果适用）\n")
                else:
                    print("⚠️  模型未生成回答")
                    print("   提示: 这可能是由于生成配置、停止条件或模型行为导致的。")
                    print("   建议: 检查生成配置（max_new_tokens, temperature等）或尝试不同的问题。\n")
                    
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                # 对于对话任务，可视化错误可以忽略
                if "visualization" in str(e).lower() or "NoneType" in str(e):
                    print("⚠️  注意: 这是对话任务，不需要可视化输出，可以忽略此错误\n")
                else:
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
    """Main function for interactive chat."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)
    
    if world_size > 1:
        print_log("Warning: Interactive chat mode is designed for single GPU. Using rank 0 only.", logger="current")
    
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
            output_ids_with_output=False,  # 对于对话任务，不需要输出ID
        )
        print_log("✅ 模型初始化成功！", logger="current")
    except Exception as e:
        print_log(f"❌ 模型初始化失败: {e}", logger="current")
        traceback.print_exc()
        return
    
    # Start interactive chat
    if rank == 0:
        interactive_chat(demo, task_name=args.task_name)
    else:
        print_log("Non-zero rank, skipping interactive chat", logger="current")


if __name__ == "__main__":
    main()

