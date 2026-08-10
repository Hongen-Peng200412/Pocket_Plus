from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.datasets.stage1_requests import Stage1TrainingRequestSet


def _write_pool(
    root: Path,
    specifications: list[tuple[str, int, int]],
) -> Path:
    pool_dir = root / "train"
    pool_dir.mkdir()
    manifest_entries: list[dict[str, str]] = []
    for pdb_id, occurrence_count, context_count in specifications:
        occurrence = np.arange(occurrence_count, dtype=np.int32)
        center = np.stack(
            [occurrence, occurrence + 100, occurrence + 200],
            axis=1,
        )
        bias = np.repeat(center[:, None, :], 30, axis=1)
        if occurrence_count:
            bias[:, :, 2] += np.arange(30, dtype=np.int32)[None, :]
        context_axis = np.arange(context_count, dtype=np.int32)
        context = np.stack(
            [context_axis + 300, context_axis + 400, context_axis + 500],
            axis=1,
        )
        relative_path = f"train/{pdb_id}.npz"
        np.savez(
            root / relative_path,
            pdb_id=np.asarray(pdb_id),
            occurrence_id=occurrence,
            center_start_zyx=center,
            bias_start_zyx=bias,
            context_start_zyx=context,
        )
        manifest_entries.append({"pdb_id": pdb_id, "path": relative_path})

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {"train": manifest_entries, "validation": []},
            }
        ),
        encoding="utf-8",
    )
    (root / "_COMPLETE").write_text("", encoding="utf-8")
    return pool_dir


def test_cap_ratio_expand_uses_per_pdb_ceiling_and_existing_pool_starts(
    tmp_path: Path,
) -> None:
    pool_dir = _write_pool(
        tmp_path,
        [("small", 3, 8), ("large", 60, 8)],
    )
    source = Stage1TrainingRequestSet(
        pool_dir,
        seed=17,
        occurrence_cap_per_pdb=50,
        occurrence_ratio=0.428,
        entry_ratio={"center": 0, "bias": 1, "context": 1},
    )

    requests = tuple(source.requests)
    assert Counter(request.pdb_id for request in requests) == {
        "small": 4,
        "large": 44,
    }
    assert Counter(request.role for request in requests) == {
        "bias": 24,
        "context": 24,
    }
    assert all(
        count == 2
        for count in Counter(
            (request.pdb_id, request.occurrence_id) for request in requests
        ).values()
    )

    pdb_order = [
        request.pdb_id
        for index, request in enumerate(requests)
        if index == 0 or requests[index - 1].pdb_id != request.pdb_id
    ]
    assert sorted(pdb_order) == ["large", "small"]

    for request in requests:
        with np.load(
            pool_dir / f"{request.pdb_id}.npz",
            allow_pickle=False,
        ) as pool:
            if request.role == "bias":
                occurrence_rows = np.flatnonzero(
                    pool["occurrence_id"] == request.occurrence_id
                )
                assert occurrence_rows.size == 1
                expected = pool["bias_start_zyx"][
                    occurrence_rows[0], request.candidate_index
                ]
            else:
                expected = pool["context_start_zyx"][request.candidate_index]
        assert request.box_start_zyx == tuple(expected.tolist())


def test_request_identity_rebuilds_by_epoch_and_is_reproducible(
    tmp_path: Path,
) -> None:
    pool_dir = _write_pool(tmp_path, [("only", 60, 8)])
    source = Stage1TrainingRequestSet(
        pool_dir,
        seed=29,
        occurrence_ratio=0.428,
        entry_ratio={"center": 0, "bias": 1, "context": 1},
    )
    epoch0 = tuple(source.requests)
    source.set_epoch(1)
    epoch1 = tuple(source.requests)

    repeated = Stage1TrainingRequestSet(
        pool_dir,
        seed=29,
        occurrence_ratio=0.428,
        entry_ratio={"center": 0, "bias": 1, "context": 1},
    )
    repeated.set_epoch(1)

    assert len(epoch0) == len(epoch1) == 44
    assert epoch1 != epoch0
    assert tuple(repeated.requests) == epoch1


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("nan")])
def test_occurrence_ratio_rejects_values_outside_open_closed_unit_interval(
    tmp_path: Path,
    ratio: float,
) -> None:
    pool_dir = _write_pool(tmp_path, [("only", 1, 1)])
    with pytest.raises(ValueError, match="occurrence_ratio"):
        Stage1TrainingRequestSet(
            pool_dir,
            seed=3,
            occurrence_ratio=ratio,
        )


@pytest.mark.parametrize("cap", [0, -1])
def test_occurrence_cap_must_be_positive(
    tmp_path: Path,
    cap: int,
) -> None:
    pool_dir = _write_pool(tmp_path, [("only", 1, 1)])
    with pytest.raises(ValueError, match="occurrence_cap_per_pdb"):
        Stage1TrainingRequestSet(
            pool_dir,
            seed=3,
            occurrence_cap_per_pdb=cap,
        )


@pytest.mark.parametrize(
    ("entry_ratio", "message"),
    [
        ({"center": 2, "bias": 1, "context": 1}, "entry_ratio.center"),
        ({"center": 0, "bias": 31, "context": 1}, "entry_ratio.bias"),
        ({"center": 0, "bias": 1, "context": -1}, "entry_ratio.context"),
        ({"center": 0, "bias": 0, "context": 0}, "至少启用"),
    ],
)
def test_entry_ratio_rejects_invalid_role_counts(
    tmp_path: Path,
    entry_ratio: dict[str, int],
    message: str,
) -> None:
    pool_dir = _write_pool(tmp_path, [("only", 1, 1)])
    with pytest.raises(ValueError, match=message):
        Stage1TrainingRequestSet(
            pool_dir,
            seed=3,
            entry_ratio=entry_ratio,
        )


@pytest.mark.parametrize(
    ("entry_ratio", "expected_roles"),
    [
        ({"center": 0, "bias": 1, "context": 1}, {"bias": 1, "context": 1}),
        (
            {"center": 1, "bias": 2, "context": 3},
            {"center": 1, "bias": 2, "context": 3},
        ),
        ({"center": 0, "bias": 5, "context": 5}, {"bias": 5, "context": 5}),
        ({"center": 1, "bias": 0, "context": 0}, {"center": 1}),
        ({"center": 0, "bias": 0, "context": 2}, {"context": 2}),
    ],
)
def test_entry_ratio_expands_each_selected_occurrence(
    tmp_path: Path,
    entry_ratio: dict[str, int],
    expected_roles: dict[str, int],
) -> None:
    pool_dir = _write_pool(tmp_path, [("only", 1, 3)])
    source = Stage1TrainingRequestSet(
        pool_dir,
        seed=11,
        occurrence_ratio=1.0,
        entry_ratio=entry_ratio,
    )
    assert Counter(request.role for request in source.requests) == expected_roles
    bias_indices = [
        request.candidate_index for request in source.requests if request.role == "bias"
    ]
    assert len(bias_indices) == len(set(bias_indices))
    context_indices = [
        request.candidate_index
        for request in source.requests
        if request.role == "context"
    ]
    if len(context_indices) <= 3:
        assert len(context_indices) == len(set(context_indices))


def test_empty_occurrence_pool_and_empty_pure_context_pool_produce_no_request(
    tmp_path: Path,
) -> None:
    empty_occurrence_root = tmp_path / "empty_occurrence"
    empty_context_root = tmp_path / "empty_context"
    empty_occurrence_root.mkdir()
    empty_context_root.mkdir()
    empty_occurrence_pool = _write_pool(
        empty_occurrence_root,
        [("empty_occurrence", 0, 3)],
    )
    empty_context_pool = _write_pool(
        empty_context_root,
        [("empty_context", 1, 0)],
    )

    assert not Stage1TrainingRequestSet(
        empty_occurrence_pool,
        seed=5,
    ).requests
    assert not Stage1TrainingRequestSet(
        empty_context_pool,
        seed=5,
        entry_ratio={"center": 0, "bias": 0, "context": 2},
    ).requests
