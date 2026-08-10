from __future__ import annotations

from src.datasets.stage1_batch_sampler import Stage1PdbBatchSampler
from src.datasets.stage1_requests import ResolvedStage1Crop


class _StaticRequestSource:
    """提供不访问磁盘的 Stage1 请求序列，专门验证 batch 契约."""

    def __init__(
        self,
        requests: tuple[ResolvedStage1Crop, ...],
        seed: int = 19,
    ) -> None:
        self.requests = requests
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.requests)


def _request(pdb_id: str, role: str, request_id: int) -> ResolvedStage1Crop:
    return ResolvedStage1Crop(
        pdb_id=pdb_id,
        box_start_zyx=(request_id, 0, 0),
        require_targets=True,
        role=role,
        occurrence_id=request_id,
    )


def test_sampler_keeps_pdb_blocks_and_balances_each_batch_fragment() -> None:
    requests = tuple(
        [
            _request("pdb_a", "center", 0),
            _request("pdb_a", "bias", 1),
            _request("pdb_a", "bias", 2),
            _request("pdb_a", "context", 3),
            _request("pdb_a", "context", 4),
            _request("pdb_b", "bias", 5),
            _request("pdb_b", "context", 6),
            _request("pdb_b", "context", 7),
        ]
    )
    source = _StaticRequestSource(requests)
    sampler = Stage1PdbBatchSampler(source, batch_size=4)

    batches = list(sampler)
    repeated = list(
        Stage1PdbBatchSampler(
            _StaticRequestSource(requests),
            batch_size=4,
        )
    )

    assert batches == repeated
    assert [len(batch) for batch in batches] == [4, 4]
    ordered = [index for batch in batches for index in batch]
    assert sorted(ordered) == list(range(8))
    assert [requests[index].pdb_id for index in ordered] == [
        "pdb_a",
        "pdb_a",
        "pdb_a",
        "pdb_a",
        "pdb_a",
        "pdb_b",
        "pdb_b",
        "pdb_b",
    ]

    first_a_fragment = batches[0]
    final_a_fragment = batches[1][:1]
    b_fragment = batches[1][1:]
    assert sum(requests[index].role != "context" for index in first_a_fragment) == 2
    assert [requests[index].role for index in first_a_fragment] == [
        "context",
        "center",
        "context",
        "bias",
    ]
    assert sum(requests[index].role != "context" for index in final_a_fragment) == 1
    assert sum(requests[index].role != "context" for index in b_fragment) == 1


def test_sampler_drops_incomplete_ddp_tail_without_padding() -> None:
    requests = tuple(
        _request(
            "single_pdb",
            "bias" if index % 2 == 0 else "context",
            index,
        )
        for index in range(22)
    )
    source_rank0 = _StaticRequestSource(requests, seed=31)
    source_rank1 = _StaticRequestSource(requests, seed=31)
    rank0 = Stage1PdbBatchSampler(
        source_rank0,
        batch_size=4,
        num_replicas=2,
        rank=0,
    )
    rank1 = Stage1PdbBatchSampler(
        source_rank1,
        batch_size=4,
        num_replicas=2,
        rank=1,
    )

    rank0_batches = list(rank0)
    rank1_batches = list(rank1)
    expected = rank0._ordered_indices()[:16]
    reconstructed: list[int] = []
    for step in range(len(rank0_batches)):
        reconstructed.extend(rank0_batches[step])
        reconstructed.extend(rank1_batches[step])

    assert len(rank0) == len(rank1) == 2
    assert all(len(batch) == 4 for batch in rank0_batches + rank1_batches)
    assert reconstructed == expected
    assert len(set(reconstructed)) == 16
    assert set(reconstructed).isdisjoint(set(rank0._ordered_indices()[16:]))
