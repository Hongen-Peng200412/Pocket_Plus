from __future__ import annotations

import os
from multiprocessing import Queue

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.wrappers.voxel_point_stage1_diagnostics import (
    CpcDiagnosticsConfig,
    CpcValidationDiagnostics,
)


def _config() -> CpcDiagnosticsConfig:
    """
    构造 CPU DDP diagnostics 测试配置。
    """
    return CpcDiagnosticsConfig(
        enabled=True,
        num_bins=4,
        write_local_artifacts=False,
        log_wandb_curves=False,
        wandb_curve_every_n_validation=1,
        output_subdir="validation_diagnostics",
    )


def _ddp_worker(rank: int, world_size: int, port: int, queue: Queue) -> None:
    """
    在 CPU/Gloo rank 中执行 diagnostics all-reduce 逻辑。
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["USE_LIBUV"] = "0"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        diagnostics = CpcValidationDiagnostics(
            config=_config(),
            class_names=["background", "foreground"],
            candidate_class_ids=[1],
            adaptive_expand_factor=[1.0],
            max_candidate_voxels_per_class=[4],
        )
        if rank == 0:
            logits = torch.logit(torch.tensor([[[[[0.9, 0.1]]]]]), eps=1e-6)
            target = torch.tensor([[[[1, 0]]]])
        else:
            logits = torch.logit(torch.tensor([[[[[0.2, 0.8]]]]]), eps=1e-6)
            target = torch.tensor([[[[0, 1]]]])
        valid_mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
        diagnostics.update_uncapped_best(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
            allow_cache_update=True,
        )

        def _sum(tensor: torch.Tensor) -> torch.Tensor:
            reduced = tensor.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            return reduced

        payload = diagnostics.compute_payload(sync_fn=_sum)
        queue.put(float(payload.scalars["val_uncapped_best/global/best_F1"].item()))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_cpc_diagnostics_cpu_gloo_all_reduce_is_symmetric() -> None:
    """
    验证本地 CPU/Gloo 多进程下 diagnostics 所有 rank 对称聚合。
    """
    ctx = mp.get_context("spawn")
    queue: Queue = ctx.Queue()
    port = 29537
    mp.spawn(_ddp_worker, args=(2, port, queue), nprocs=2, join=True)
    values = [queue.get(timeout=10.0), queue.get(timeout=10.0)]

    assert values == [1.0, 1.0]
