#!/usr/bin/env python
"""
网页交互式多任务测试Demo - 支持4个任务
1. imgconv - 图像对话（GeoChat）
2. genseg - 通用分割
3. ovseg - 开放词汇分割
4. refseg - 指代分割

使用方法:
    python xsam/xsam/tools/web_demo.py \
        xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune_geochat/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune_geochat.py \
        --pth_model ./wkdrs/s3_mixed_finetune_geochat/xxx/pytorch_model.bin
"""

import argparse
import datetime
import os
import os.path as osp
import time
import traceback
import warnings

import cv2
import gradio as gr
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.tools.utils import set_model_resource

from xsam.dataset.utils.coco import COCO_INSTANCE_CATEGORIES, COCO_SEMANTIC_CATEGORIES
from xsam.demo.demo import XSamDemo
from xsam.utils.checkpoint import load_checkpoint
from xsam.utils.config import setup_model_config
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

this_dir = osp.dirname(osp.abspath(__file__))

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

# 自定义CSS样式
custom_css = """
/* 全局样式 */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* 主容器 */
.main {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    margin: 15px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

/* 主标题样式 */
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 3rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    position: relative;
    z-index: 1;
}

.main-header h2 {
    font-size: 1.2rem;
    font-weight: 400;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

.running-info {
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    background: #f0f7ff;
}

.input-section, .output-section {
    display: flex;
    flex-direction: column;
}

.input-section > div, .output-section > div {
    flex-grow: 1;
}

.task-description {
    border-left: 4px solid #007bff;
    padding-left: 10px;
    background: #f8f9fa;
    border-radius: 4px;
    margin-top: 10px;
}

.image-upload, .image-upload > div, .image-upload canvas, .image-upload img {
    width: 100% !important;
    height: 500px !important;
    max-width: 100% !important;
    min-height: 400px !important;
    object-fit: contain !important;
    display: block !important;
}
"""

# Task descriptions (English)
TASK_DESCRIPTION = {
    "imgconv": "Image Conversation - Answer questions about the image (English prompts)",
    "genseg": "Generic Segmentation - Segment objects using SOTA dataset categories (no prompt needed, uses all SOTA categories by default)",
    "ovseg": "Open-Vocabulary Segmentation - Segment objects using open vocabulary (comma-separated categories)",
    "refseg": "Referring Segmentation - Segment objects by referring expressions",
}

# Task examples and hints (English)
TASK_EXAMPLES = {
    "imgconv": {
        "prompt_example": "Please describe this image in detail.",
        "prompt_hint": "Enter your question in English, e.g., 'What is in this image?', 'Please describe the main content of the image'"
    },
    "genseg": {
        "prompt_example": "(No prompt needed - will use all SOTA dataset categories by default)",
        "prompt_hint": "Generic segmentation uses all predefined SOTA categories by default. You can leave the prompt empty, or optionally specify: ins: category1, category2, ...; sem: category1, category2, ..."
    },
    "ovseg": {
        "prompt_example": "thing: person, car; stuff: tree, building",
        "prompt_hint": "Enter categories in format: 'thing: category1, category2; stuff: category3, category4' or simply 'category1, category2, category3' (will try to determine thing/stuff from dataset)"
    },
    "refseg": {
        "prompt_example": "the white tshirt kid",
        "prompt_hint": "Enter a referring expression, e.g., 'the red car on the left', 'the person wearing blue shirt'"
    }
}

SUPPORTED_TASKS = ["imgconv", "genseg", "ovseg", "refseg"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="X-SAM 网页交互式多任务测试Demo")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and visualizations")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint or 'latest' to use the latest checkpoint in work_dir",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--log-dir", type=str, default="./demo_outputs", help="directory to save logs and outputs")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument("--port", type=int, default=7860, help="port for gradio server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host for gradio server")
    parser.add_argument("--share", action="store_true", help="share gradio app")
    return parser.parse_args()


def handle_checkpoint_path(pth_model, work_dir=None):
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
    
    return pth_model


class GradioApp:
    def __init__(self, demo: XSamDemo, log_dir: str):
        self.demo = demo
        self.log_dir = log_dir

    def gradio_predict_with_progress(self, data, prompt, task_name="imgconv", score_thr=0.5, progress=gr.Progress()):
        """预测函数，带进度条和错误处理"""
        if data is None:
            return "❌ 请先上传图片", "", "", None

        try:
            progress(0.1, desc="🔍 初始化中...")

            # Validate inputs
            if not prompt or prompt.strip() == "":
                if task_name not in ["imgconv", "genseg"]:  # imgconv and genseg can work without prompt
                    return "❌ Please enter a prompt", "", "", None

            # 日志设置
            day_timestamp = datetime.datetime.now().strftime("%Y%m%d")
            file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            day_log_dir = osp.join(self.log_dir, day_timestamp)
            log_file = osp.join(day_log_dir, f"{day_timestamp}.log")
            img_log_dir = osp.join(day_log_dir, "images")
            out_log_dir = osp.join(day_log_dir, "outputs")

            os.makedirs(day_log_dir, exist_ok=True)
            os.makedirs(img_log_dir, exist_ok=True)
            os.makedirs(out_log_dir, exist_ok=True)

            progress(0.3, desc="🖼️ 处理图片...")

            # 转换图片格式
            vprompt_masks = None
            if isinstance(data, Image.Image):
                pil_image = data.convert("RGB")
                array_image = np.array(pil_image)
            elif isinstance(data, np.ndarray):
                pil_image = Image.fromarray(data).convert("RGB")
                array_image = data
            elif isinstance(data, dict):
                pil_image = data["background"].convert("RGB")
                array_image = np.array(pil_image)
                vprompt_masks = [np.array(layer)[..., -1] for layer in data["layers"]]
                vprompt_masks = [mask for mask in vprompt_masks if mask.sum() > 0]
                vprompt_masks = None if len(vprompt_masks) == 0 else vprompt_masks
            else:
                raise ValueError(f"不支持的图片类型: {type(data)}")

            progress(0.5, desc="🔎 运行模型推理...")

            # 运行模型推理
            start_time = time.time()

            llm_input, llm_output, seg_output = self.demo.run_on_image(
                pil_image, prompt, task_name, vprompt_masks=vprompt_masks
            )

            llm_success = llm_output is not None and llm_output.strip() != ""
            seg_success = seg_output is not None

            inference_time = time.time() - start_time

            progress(0.9, desc="💾 保存结果...")

            # 保存输入图片和输出图片
            cv2.imwrite(f"{img_log_dir}/{file_timestamp}.png", cv2.cvtColor(array_image, cv2.COLOR_RGB2BGR))
            if seg_success:
                cv2.imwrite(f"{out_log_dir}/{file_timestamp}.png", cv2.cvtColor(seg_output, cv2.COLOR_RGB2BGR))

            # 记录日志
            if not osp.exists(log_file):
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("timestamp\timage\tprompt\ttask_name\tinference_time\tllm_success\tseg_success\n")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{file_timestamp}\t{file_timestamp}.png\t{prompt}\t{task_name}\t{inference_time:.3f}\t{llm_success}\t{seg_success}\n"
                )

            progress(1.0, desc="✅ 完成！")

            if llm_success or seg_success:
                status_message = f"✅ 处理完成！耗时 {inference_time:.2f}秒"
            else:
                status_message = f"⚠️ 处理完成但未生成结果，耗时 {inference_time:.2f}秒"

            return (
                status_message,
                llm_input if llm_input else "",
                llm_output if llm_output else "",
                (gr.update(value=seg_output, height=seg_output.shape[0] + 10) if seg_output is not None else None),
            )

        except Exception as e:
            error_msg = f"❌ 错误: {str(e)}"
            print(f"Error in gradio_predict: {traceback.format_exc()}")
            return error_msg, "", "", None

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="OneTerra-panoearth-V1 多任务测试Demo", css=custom_css, theme=gr.themes.Soft()) as app:
            # 标题
            gr.HTML(
                """
                <div class="main-header">
                    <h1>✨ OneTerra-panoearth-V1 多任务测试Demo</h1>
                    <h2>支持4个任务：图像对话 | 通用分割 | 开放词汇分割 | 指代分割</h2>
                </div>
                """
            )

            # 主界面
            with gr.Row(elem_classes="main-row"):
                with gr.Column(scale=5, elem_classes="input-section"):
                    # 图片上传
                    image_input = gr.Image(
                        type="pil",
                        label="📸 上传图片",
                        elem_classes="image-upload",
                        sources=["upload", "webcam", "clipboard"],
                    )

                    # 任务选择
                    task_name = gr.Dropdown(
                        choices=SUPPORTED_TASKS,
                        value="imgconv",
                        label="🎯 选择任务",
                        info="选择要测试的任务类型",
                    )

                    # 任务描述
                    task_description = gr.Textbox(
                        value=TASK_DESCRIPTION["imgconv"],
                        label="📋 任务描述",
                        interactive=False,
                        lines=2,
                        elem_classes="task-description",
                    )

                    # Prompt input
                    text_input = gr.Textbox(
                        lines=3,
                        label="🤔 Enter Prompt (Optional)",
                        placeholder="Enter prompt based on selected task. For genseg, you can leave it empty to use all SOTA categories by default.",
                        value="",
                    )

                    # Prompt提示
                    prompt_hint = gr.Textbox(
                        value=TASK_EXAMPLES["imgconv"]["prompt_hint"],
                        label="💡 Prompt提示",
                        interactive=False,
                        lines=2,
                    )

                    # 操作按钮
                    with gr.Row():
                        submit_btn = gr.Button("🚀 运行推理", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")

                with gr.Column(scale=6, elem_classes="output-section"):
                    # 状态显示
                    status_display = gr.Textbox(
                        value="🟢 准备就绪 - 请上传图片并输入prompt",
                        label="ℹ️ 运行状态",
                        interactive=False,
                        lines=1,
                        elem_classes="running-info",
                    )

                    # 对话结果
                    with gr.Group():
                        gr.HTML("<h3 style='margin: 0 0 15px 0;'>🤖 对话结果</h3>")
                        llm_input = gr.Textbox(
                            value="",
                            label="📝 模型输入",
                            placeholder="模型接收到的输入将显示在这里",
                            lines=2,
                            interactive=False,
                        )
                        llm_output = gr.Textbox(
                            value="",
                            label="💬 模型回答",
                            placeholder="模型的回答将显示在这里",
                            lines=4,
                            interactive=False,
                        )

                    # 分割结果
                    with gr.Group():
                        seg_output = gr.Image(
                            type="pil",
                            label="🎨 分割结果",
                            height=500,
                        )

            # 使用说明
            with gr.Accordion("📖 使用说明", open=False):
                gr.Markdown("""
                ### 支持的任务类型：
                
                1. **图像对话 (imgconv)**: 回答关于图像的问题
                   - 示例: "请详细描述这张图片。"
                
                2. **通用分割 (genseg)**: 根据类别名称分割物体
                   - 格式: `ins: 类别1, 类别2, ...; sem: 类别1, 类别2, ...`
                   - 示例: `ins: person, car; sem: sky, road`
                
                3. **开放词汇分割 (ovseg)**: 使用开放词汇进行分割
                   - 格式: 类别名称，用逗号分隔
                   - 示例: `person, car, tree, building`
                
                4. **指代分割 (refseg)**: 根据指代表达式分割物体
                   - 格式: 自然语言描述
                   - 示例: `the white tshirt kid`
                
                ### 使用步骤：
                1. 上传一张图片
                2. 选择要测试的任务类型
                3. 根据任务类型输入相应的prompt
                4. 点击"运行推理"按钮
                5. 查看结果（对话结果和分割结果）
                """)

            # 事件处理
            submit_btn.click(
                fn=self.gradio_predict_with_progress,
                inputs=[image_input, text_input, task_name],
                outputs=[status_display, llm_input, llm_output, seg_output],
                show_progress=True,
            )

            clear_btn.click(
                fn=lambda: [None, "imgconv", "", "", "", None, "🟢 已清空，请重新输入"],
                outputs=[image_input, task_name, text_input, llm_input, llm_output, seg_output, status_display],
            )

            # 任务切换时更新描述和提示
            def update_task_info(task):
                return (
                    TASK_DESCRIPTION.get(task, ""),
                    TASK_EXAMPLES.get(task, {}).get("prompt_hint", ""),
                    TASK_EXAMPLES.get(task, {}).get("prompt_example", ""),
                )

            task_name.change(
                fn=update_task_info,
                inputs=[task_name],
                outputs=[task_description, prompt_hint, text_input],
            )

            # 图片上传时更新状态
            image_input.change(
                fn=lambda img: "📸 图片已上传！请选择任务并输入prompt，然后点击'运行推理'" if img is not None else "🟢 准备就绪 - 请上传图片并输入prompt",
                inputs=[image_input],
                outputs=[status_display],
            )

        return app


def setup_cfg(args):
    """Setup configuration from arguments."""
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
    if args.seed is not None:
        set_random_seed(args.seed)
        print_log(f"Set the random seed to {args.seed}.", logger="current")
    register_function(cfg._cfg_dict)

    # Handle checkpoint path
    args.pth_model = handle_checkpoint_path(args.pth_model, args.work_dir)

    return args, cfg


def main():
    """Main function for X-SAM Gradio demo."""
    args = parse_args()

    # Setup configuration
    args, cfg = setup_cfg(args)

    if args.pth_model is None:
        raise ValueError("--pth_model must be specified")

    # Create demo instance
    print_log("=" * 80, logger="current")
    print_log("初始化模型...", logger="current")
    print_log("=" * 80, logger="current")

    try:
        demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)
        print_log("✅ 模型初始化成功！", logger="current")
    except Exception as e:
        print_log(f"❌ 模型初始化失败: {e}", logger="current")
        traceback.print_exc()
        return

    # Create Gradio app
    os.makedirs(args.log_dir, exist_ok=True)
    gradio_app = GradioApp(demo, args.log_dir)
    app = gradio_app.create_interface()

    # Launch the app
    print_log(f"启动Gradio服务器: {args.host}:{args.port}", logger="current")
    if args.share:
        print_log("已启用分享链接，可以通过公网访问", logger="current")
    # 启用队列功能，增加推理时间限制（默认30秒，启用队列后无限制）
    app.queue()
    app.launch(
        show_error=True,
        share=args.share,
        server_port=args.port,
        server_name=args.host,
    )


if __name__ == "__main__":
    main()

