"""Minimal multi-node NCCL all_reduce smoke test."""

import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    hostname = socket.gethostname()
    device = torch.device(f"cuda:{local_rank}")
    x = torch.tensor([float(rank + 1)], device=device, dtype=torch.float32)

    t0 = time.time()
    for step in range(20):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    expected = world_size * (world_size + 1) / 2.0
    ok = abs(x.item() - expected) < 1e-3

    if rank == 0:
        print(
            f"[NCCL_SMOKE] world_size={world_size} all_reduce_ok={ok} "
            f"value={x.item():.1f} expected={expected:.1f} elapsed={elapsed:.2f}s",
            flush=True,
        )
    if not ok:
        raise RuntimeError(f"rank {rank} ({hostname}): all_reduce mismatch {x.item()} != {expected}")

    dist.barrier()
    if rank == 0:
        print("[NCCL_SMOKE] barrier OK - multi-node NCCL looks healthy", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
