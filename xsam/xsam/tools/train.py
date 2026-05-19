import argparse
import json
import logging
import os
import os.path as osp
import sys
import warnings
from functools import partial
from datetime import timedelta
import torch
import torch.distributed as dist

# 在导入其他模块之前，确保 xsam 模块可以被找到
# 添加项目根目录到Python路径
# 获取当前文件的目录，然后向上找到项目根目录（包含xsam目录的目录）
_current_file = osp.abspath(__file__)
_current_dir = osp.dirname(_current_file)  # tools/
# train.py 在 xsam/xsam/tools/ 下，需要向上3级到项目根目录
_project_root = osp.dirname(osp.dirname(osp.dirname(_current_dir)))  # 项目根目录
# 添加项目根目录和xsam目录到Python路径
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# 同时添加xsam目录（因为xsam模块在xsam/xsam/下）
_xsam_dir = osp.join(_project_root, "xsam")
if _xsam_dir not in sys.path:
    sys.path.insert(0, _xsam_dir)

# 在导入 torch 之前就设置环境变量，确保超时在torchrun初始化时就被读取
# 这是最关键的：必须在torch.distributed初始化之前设置
_timeout_seconds = int(os.environ.get("NCCL_TIMEOUT", os.environ.get("DIST_TIMEOUT", 72000)))
_timeout_ms = _timeout_seconds * 1000
_timeout = timedelta(seconds=_timeout_seconds)
# 设置所有相关的环境变量（必须在导入torch之前）
os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = str(_timeout_ms)
os.environ["NCCL_TIMEOUT"] = str(_timeout_seconds)
os.environ["DIST_TIMEOUT"] = str(_timeout_seconds)
# 确保 PyTorch 使用正确的超时值（以秒为单位，但传递给 init_process_group 需要 timedelta）
print(f"[X-SAM] Setting NCCL timeout to {_timeout_seconds} seconds ({_timeout_seconds/3600:.1f} hours)", file=sys.stderr)
print(f"[X-SAM] Setting TORCH_DISTRIBUTED_DEFAULT_TIMEOUT to {_timeout_ms} ms", file=sys.stderr)

# 关键修复：如果分布式已经初始化（torchrun可能已经初始化），直接设置超时
if dist.is_initialized():
    try:
        pg = dist.distributed_c10d._get_default_group()
        local_rank = os.environ.get("LOCAL_RANK", "0")
        
        # 尝试多种方式修改超时
        old_timeout = None
        if hasattr(pg, '_timeout'):
            old_timeout = pg._timeout
            pg._timeout = _timeout
        if hasattr(pg, 'timeout'):
            old_timeout = getattr(pg, 'timeout', None)
            pg.timeout = _timeout
        
        # 尝试直接修改ProcessGroupNCCL的内部超时（毫秒）
        try:
            # ProcessGroupNCCL内部使用毫秒存储超时
            if hasattr(pg, '_default_pg_timeout'):
                pg._default_pg_timeout = _timeout_ms
            # 尝试修改所有NCCL通信器的超时
            if hasattr(pg, '_pg'):
                for comm_key, comm in pg._pg.items():
                    if hasattr(comm, 'timeout'):
                        comm.timeout = _timeout_ms
                    # 尝试修改NCCL通信器的内部超时
                    if hasattr(comm, '_timeout'):
                        comm._timeout = _timeout_ms
        except Exception as inner_e:
            if local_rank == "0":
                print(f"[X-SAM] Could not set internal NCCL timeout: {inner_e}", file=sys.stderr)
        
        if local_rank == "0":
            print(f"[X-SAM] Distributed already initialized, updating timeout from {old_timeout} to {_timeout} ({_timeout_seconds}s, {_timeout_ms}ms)", file=sys.stderr)
            # 验证超时是否真的被设置
            current_timeout = getattr(pg, '_timeout', None)
            print(f"[X-SAM] Current ProcessGroup._timeout: {current_timeout}", file=sys.stderr)
    except Exception as e:
        local_rank = os.environ.get("LOCAL_RANK", "0")
        if local_rank == "0":
            print(f"[X-SAM] Warning: Could not update timeout for existing ProcessGroup: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

# Patch init_process_group 以确保超时被设置
if not hasattr(dist, '_xsam_train_patched'):
    _original_init_process_group = dist.init_process_group
    
    def _patched_init_process_group_train(*args_patch, **kwargs_patch):
        # 强制设置超时，即使已经存在也要确保使用我们的值
        kwargs_patch['timeout'] = _timeout
        # 添加调试信息
        local_rank = os.environ.get("LOCAL_RANK", "0")
        if local_rank == "0":
            print(f"[X-SAM] init_process_group called with timeout={_timeout} ({_timeout_seconds}s)", file=sys.stderr)
            print(f"[X-SAM] Force setting timeout to {_timeout_seconds} seconds", file=sys.stderr)
        result = _original_init_process_group(*args_patch, **kwargs_patch)
        
        # 初始化后立即设置超时
        try:
            if dist.is_initialized():
                pg = dist.distributed_c10d._get_default_group()
                # 尝试多种方式设置超时
                if hasattr(pg, '_timeout'):
                    pg._timeout = _timeout
                if hasattr(pg, 'timeout'):
                    pg.timeout = _timeout
                # 尝试设置NCCL内部超时（使用毫秒）
                try:
                    # 通过反射访问内部属性
                    if hasattr(pg, '_pg'):
                        for comm_key, comm in pg._pg.items():
                            if hasattr(comm, 'timeout'):
                                comm.timeout = _timeout_ms  # NCCL使用毫秒
                            # 尝试修改NCCL通信器的内部超时
                            if hasattr(comm, '_timeout'):
                                comm._timeout = _timeout_ms
                except Exception as inner_e:
                    if local_rank == "0":
                        print(f"[X-SAM] Could not set NCCL comm timeout: {inner_e}", file=sys.stderr)
                if local_rank == "0":
                    print(f"[X-SAM] Set ProcessGroup timeout to {_timeout_seconds}s after init", file=sys.stderr)
        except Exception as e:
            if local_rank == "0":
                print(f"[X-SAM] Warning: Could not set ProcessGroup timeout: {e}", file=sys.stderr)
        
        # 验证超时是否生效
        if hasattr(dist, 'get_backend'):
            try:
                backend = dist.get_backend()
                if local_rank == "0":
                    print(f"[X-SAM] Distributed backend initialized: {backend}", file=sys.stderr)
            except:
                pass
        return result
    
    dist.init_process_group = _patched_init_process_group_train
    dist._xsam_train_patched = True

from mmengine.config import Config, DictAction
from mmengine.config.lazy import LazyObject
from mmengine.model import BaseModel
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from transformers.models.auto.modeling_auto import _BaseAutoModelClass
from xtuner.configs import cfgs_name_path
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.model.modules import dispatch_modules
from xtuner.model.modules.dispatch import SUPPORT_FLASH2
from xtuner.model.utils import LoadWoInit, find_all_linear_names, traverse_dict
from xtuner.registry import BUILDER
from xtuner.tools.utils import auto_dtype_of_deepspeed_config, get_seed_from_checkpoint, set_model_resource

from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

set_default_logging_format()
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="the path to the .json file for deepspeed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="specify checkpoint path to be resumed from.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the training")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    return args


def check_cfg(cfg, args):
    if getattr(cfg, "use_varlen_attn", False) and cfg.train_dataloader.batch_size > 1:
        raise NotImplementedError(
            f"If utilizing varlen attention, the batch size should be"
            f" set to 1, but got {cfg.train_dataloader.batch_size}"
        )

    if getattr(cfg, "use_varlen_attn", False):
        sequence_parallel = getattr(cfg, "sequence_parallel", 1)
        max_length = getattr(cfg.train_dataloader.dataset, "max_length", None)
        if max_length is not None:
            assert max_length % sequence_parallel == 0, (
                "When using varlen attention, `max_length` should be evenly "
                "divided by sequence parallel world size, but got "
                f"max_length = {max_length} and sequence_parallel = "
                f"{sequence_parallel}"
            )

    if getattr(cfg, "sequence_parallel_size", 1) > 1:
        assert SUPPORT_FLASH2, "`flash_attn` is required if you want to use " "sequence parallel."
        attn_implementation = getattr(cfg.model.llm, "attn_implementation", None)
        assert attn_implementation is None or attn_implementation == "flash_attention_2", (
            "If you want to use sequence parallel, please set "
            "attn_implementation to `flash_attention_2` or do not "
            f"set this attribute. Got `{attn_implementation}` ."
        )

    if getattr(cfg, "use_varlen_attn", False):
        assert SUPPORT_FLASH2, "`flash_attn` is required if you set " "`use_varlen_attn` to True."
        attn_implementation = getattr(cfg.model.llm, "attn_implementation", None)
        assert attn_implementation is None or attn_implementation == "flash_attention_2", (
            "If you want to set `use_varlen_attn` to True, please set"
            " attn_implementation to `flash_attention_2` or do not "
            f"set this attribute. Got `{attn_implementation}` ."
        )

    if args.deepspeed is None:
        assert getattr(cfg, "sequence_parallel_size", 1) == 1, (
            "Sequence parallel training without DeepSpeed lacks validation."
            "Please use DeepSpeed to optimize the training phase by "
            "`--deepspeed deepspeed_zero1 (deepspeed_zero2 or "
            "deepspeed_zero3)`."
        )


def main():
    args = parse_args()

    # 检查 CUDA 可用性（在初始化分布式训练之前）
    if args.deepspeed or args.launcher == "pytorch":
        if not torch.cuda.is_available():
            error_msg = (
                "错误: 未检测到 CUDA GPU。\n"
                "请检查:\n"
                "1. 容器是否使用 --gpus 参数启动（例如: docker run --gpus all ...）\n"
                "2. nvidia-docker 或 nvidia-container-toolkit 是否已正确安装\n"
                "3. 运行 'nvidia-smi' 检查 GPU 是否可见\n"
                "4. 运行 'python -c \"import torch; print(torch.cuda.is_available())\"' 检查 PyTorch 是否能检测到 GPU\n"
            )
            print_log(error_msg, logger="current", level=logging.ERROR)
            sys.exit(1)
        else:
            print_log(
                f"检测到 {torch.cuda.device_count()} 个 CUDA GPU",
                logger="current",
                level=logging.INFO,
            )

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    check_cfg(cfg, args)

    if args.resume == "auto":
        from mmengine.runner import find_latest_checkpoint

        args.resume = find_latest_checkpoint(args.work_dir)

        print_log(f"Auto resumed from the latest checkpoint {args.resume}.", logger="current")

    if cfg.get("framework", "mmengine").lower() == "huggingface":
        # set default training_args
        if cfg.get("training_args", None) is None:
            cfg.training_args = dict(type=TrainingArguments)
        if args.seed is not None:
            cfg.training_args.seed = args.seed
        # set work_dir
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.training_args.output_dir = args.work_dir
        elif cfg.training_args.get("output_dir", None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.training_args.output_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
        # enable deepspeed
        if args.deepspeed:
            if not osp.isfile(args.deepspeed):
                try:
                    args.deepspeed = cfgs_name_path[args.deepspeed]
                except KeyError:
                    raise FileNotFoundError(f"Cannot find {args.deepspeed}")
            cfg.training_args.deepspeed = args.deepspeed
        if cfg.training_args.get("deepspeed"):
            device_map = None
        else:
            # Data Parallel
            device_map = {"": int(os.environ.get("LOCAL_RANK", args.local_rank))}
        # build training_args
        training_args = BUILDER.build(cfg.training_args)
        # build model
        if issubclass(cfg.model.type, _BaseAutoModelClass):
            with LoadWoInit():
                cfg.model.device_map = device_map
                traverse_dict(cfg.model)
            model = BUILDER.build(cfg.model)
            model.config.use_cache = False
            dispatch_modules(model)
            if cfg.get("lora", None):
                lora = BUILDER.build(cfg.lora)
                model = prepare_model_for_kbit_training(model)
                if lora.target_modules is None:
                    modules = find_all_linear_names(model)
                    lora.target_modules = modules
                model = get_peft_model(model, lora)
        elif issubclass(cfg.model.type, BaseModel):
            model = BUILDER.build(cfg.model)
        else:
            raise ValueError(f"Unsupported model type: {cfg.model.type}")

        # build dataset
        train_dataset = BUILDER.build(cfg.train_dataset)
        if cfg.get("data_collator", None) is not None:
            data_collator = BUILDER.build(cfg.data_collator)
        else:
            data_collator = partial(default_collate_fn, return_hf_format=True)
        # build trainer
        trainer = cfg.trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        # training
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    else:
        if args.seed is not None and args.resume is None:
            # Use args.seed
            cfg.merge_from_dict(dict(randomness=dict(seed=args.seed)))
            print_log(
                f"Set the random seed to {args.seed}.",
                logger="current",
                level=logging.INFO,
            )
        elif args.resume is not None:
            # Use resumed seed
            from mmengine.fileio import PetrelBackend, get_file_backend
            from xtuner.utils.fileio import patch_fileio

            backend = get_file_backend(args.resume)
            if isinstance(backend, PetrelBackend):
                with patch_fileio():
                    resumed_seed = get_seed_from_checkpoint(args.resume)
            else:
                resumed_seed = get_seed_from_checkpoint(args.resume)
            cfg.merge_from_dict(dict(randomness=dict(seed=resumed_seed)))
            if args.seed is not None and args.seed != resumed_seed:
                print_log(
                    (
                        f"The value of random seed in resume checkpoint "
                        f'"{args.resume}" is different from the value in '
                        f"arguments. The resumed seed is {resumed_seed}, while "
                        f"the input argument seed is {args.seed}. Using the "
                        f"resumed seed {resumed_seed}."
                    ),
                    logger="current",
                    level=logging.WARNING,
                )
            else:
                print_log(
                    f"Set the random seed to {resumed_seed}.",
                    logger="current",
                    level=logging.INFO,
                )

        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args.local_rank)
        cfg.launcher = args.launcher
        # work_dir is determined in this priority:
        # CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get("work_dir", None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

        if args.deepspeed:
            try:
                import deepspeed
            except ImportError:
                raise ImportError("deepspeed is not installed properly, please check.")
            if digit_version(deepspeed.__version__) < digit_version("0.12.3"):
                raise RuntimeError(
                    "Please upgrade your DeepSpeed version " "by using the command pip install " "`deepspeed>=0.12.3`"
                )
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == "DeepSpeedOptimWrapper":
                print_log(
                    "Deepspeed training is already enabled in your config.",
                    logger="current",
                    level=logging.WARNING,
                )
            else:
                if not osp.isfile(args.deepspeed):
                    try:
                        args.deepspeed = cfgs_name_path[args.deepspeed]
                    except KeyError:
                        raise FileNotFoundError(f"Cannot find {args.deepspeed}")
                with open(args.deepspeed) as f:
                    ds_cfg = json.load(f)

                ds_grad_accum = ds_cfg.get("gradient_accumulation_steps", "auto")
                mm_grad_accum = cfg.optim_wrapper.get("accumulative_counts", 1)
                if ds_grad_accum != "auto" and ds_grad_accum != mm_grad_accum:
                    print_log(
                        (
                            "Mismatch on gradient_accumulation_steps: "
                            f"MMEngine {mm_grad_accum}, "
                            f"Deepspeed {ds_grad_accum}. "
                            f"Set to {mm_grad_accum}"
                        ),
                        logger="current",
                        level=logging.WARNING,
                    )
                grad_accum = mm_grad_accum

                ds_train_bs = ds_cfg.get("train_micro_batch_size_per_gpu", "auto")
                mm_train_bs = cfg.train_dataloader.batch_size
                if ds_train_bs != "auto" and ds_train_bs != mm_train_bs:
                    print_log(
                        (
                            "Mismatch on train_micro_batch_size_per_gpu: "
                            f"MMEngine {mm_train_bs}, Deepspeed {ds_train_bs}. "
                            f"Set to {mm_train_bs}"
                        ),
                        logger="current",
                        level=logging.WARNING,
                    )
                train_bs = cfg.train_dataloader.batch_size

                ds_grad_clip = ds_cfg.get("gradient_clipping", "auto")
                clip_grad = cfg.optim_wrapper.get("clip_grad", None)
                paramwise_cfg = cfg.optim_wrapper.get("paramwise_cfg", None)
                if clip_grad and clip_grad.get("max_norm", None) is not None:
                    mm_max_norm = cfg.optim_wrapper.clip_grad.max_norm
                else:
                    mm_max_norm = 1.0
                if ds_grad_clip != "auto" and ds_grad_clip != mm_max_norm:
                    print_log(
                        (
                            "Mismatch on gradient_clipping: "
                            f"MMEngine {mm_max_norm}, Deepspeed {ds_grad_clip}. "
                            f"Set to {mm_max_norm}"
                        ),
                        logger="current",
                        level=logging.WARNING,
                    )
                grad_clip = mm_max_norm
                # 尝试使用 xtuner 的 auto_dtype_of_deepspeed_config，如果失败则手动处理
                try:
                    ds_cfg = auto_dtype_of_deepspeed_config(ds_cfg)
                except (AttributeError, NameError) as e:
                    # 修复 xtuner 中 get_torch_device() 返回 torch.cpu 模块的问题
                    print_log(
                        f"Warning: auto_dtype_of_deepspeed_config failed ({e}), using manual dtype configuration.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    if ds_cfg.get('fp16') and not ds_cfg.get('bf16'):
                        if ds_cfg.get('fp16').get('enabled') == 'auto':
                            ds_cfg['fp16']['enabled'] = torch.cuda.is_available()
                    elif not ds_cfg.get('fp16') and ds_cfg.get('bf16'):
                        if ds_cfg.get('bf16').get('enabled') == 'auto':
                            ds_cfg['bf16']['enabled'] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    elif ds_cfg.get('fp16') and ds_cfg.get('bf16'):
                        if ds_cfg.get('fp16').get('enabled') == 'auto':
                            ds_cfg['fp16']['enabled'] = torch.cuda.is_available()
                        if ds_cfg.get('bf16').get('enabled') == 'auto':
                            ds_cfg['bf16']['enabled'] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        if (ds_cfg['fp16']['enabled'] is True and ds_cfg['bf16']['enabled'] is True):
                            ds_cfg['fp16']['enabled'] = False
                            ds_cfg['bf16']['enabled'] = True
                exclude_frozen_parameters = (
                    True if digit_version(deepspeed.__version__) >= digit_version("0.10.1") else None
                )
                strategy = dict(
                    type=LazyObject("xtuner.engine", "DeepSpeedStrategy"),
                    config=ds_cfg,
                    gradient_accumulation_steps=grad_accum,
                    train_micro_batch_size_per_gpu=train_bs,
                    gradient_clipping=grad_clip,
                    exclude_frozen_parameters=exclude_frozen_parameters,
                    sequence_parallel_size=getattr(cfg, "sequence_parallel_size", 1),
                )
                cfg.__setitem__("strategy", strategy)
                optim_wrapper = dict(
                    type="DeepSpeedOptimWrapper", optimizer=cfg.optim_wrapper.optimizer, paramwise_cfg=paramwise_cfg
                )
                cfg.__setitem__("optim_wrapper", optim_wrapper)
                cfg.runner_type = "FlexibleRunner"

        # resume is determined in this priority: resume from > auto_resume
        if args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if "runner_type" not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # 关键修复：在Runner初始化后，再次确保超时设置生效
        if dist.is_initialized():
            try:
                pg = dist.distributed_c10d._get_default_group()
                if hasattr(pg, '_timeout'):
                    old_timeout = pg._timeout
                    pg._timeout = _timeout
                    local_rank = os.environ.get("LOCAL_RANK", "0")
                    if local_rank == "0":
                        print(f"[X-SAM] Final timeout update: {old_timeout} -> {_timeout} ({_timeout_seconds}s)", file=sys.stderr)
                        # 验证超时是否真的被设置
                        current_timeout = pg._timeout
                        print(f"[X-SAM] Current ProcessGroup timeout: {current_timeout} (expected: {_timeout})", file=sys.stderr)
            except Exception as e:
                local_rank = os.environ.get("LOCAL_RANK", "0")
                if local_rank == "0":
                    print(f"[X-SAM] Warning: Final timeout update failed: {e}", file=sys.stderr)

        # start training
        runner.train()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(f"\n[rank {local_rank}] FAILED with exception:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
