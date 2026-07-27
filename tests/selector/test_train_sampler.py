"""Selector 按 PDB 分组 batch sampler 的确定性与 I/O 局部性测试. """

from __future__ import annotations

from types import SimpleNamespace

from src.selector.train import PdbGroupedBatchSampler


def test_pdb_grouped_batch_sampler_never_crosses_pdb_and_is_reproducible() -> None:
    """同一 batch 只含一个 PDB, 且固定 seed/epoch 可逐元素复现. """

    records = [
        SimpleNamespace(split="train", pdb_id=pdb_id)
        for pdb_id in ("a", "a", "a", "b", "b", "c")
    ]
    sampler = PdbGroupedBatchSampler(records, batch_size=2, seed=19)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    second = list(sampler)

    assert first == second
    assert len(sampler) == 4
    assert sorted(index for batch in first for index in batch) == list(range(len(records)))
    assert all(len({records[index].pdb_id for index in batch}) == 1 for batch in first)
