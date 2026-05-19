"""在 after_train_iter 里对 loss_dict 做 all-reduce(mean)，使日志打印 64 卡平均 loss 而非单卡."""
import torch
from mmengine.dist import all_reduce, get_world_size
from mmengine.hooks import Hook


class DistLossReduceHook(Hook):
    """对 train_step 返回的 loss_dict 做跨卡 mean all-reduce，再交给后续 log。

    这样 LoggerHook 打印的是全局平均 loss，而不是 rank0 单卡 loss。
    priority=0 保证在其它写 log 的 hook 之前执行。
    """

    priority = 0

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if outputs is None or not isinstance(outputs, dict):
            return
        if get_world_size() <= 1:
            return
        for key, value in list(outputs.items()):
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                t = value.detach().float().clone()
                all_reduce(t, op="mean")
                outputs[key] = t.to(value.device)
