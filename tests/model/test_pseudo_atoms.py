from __future__ import annotations

import pytest
import torch

from src.model import pseudo_atoms
from src.model.pseudo_atoms import (
    build_layout,
    build_pseudo_mask,
    build_real_mask,
    extract_pseudo_tensor_from_mixed,
    extract_real_tensor_from_mixed,
    inject_pseudo_atoms,
    interleave_real_and_pseudo_tensor,
    remove_pseudo_atoms,
)


def _make_real_batch() -> dict[str, torch.Tensor]:
    """
    构造两个 BOX 的 real-only batch。

    输出:
        - batch: dict[str, torch.Tensor], real_counts=[2, 1], atom_feat 形状为 (3, 2)
    """
    return {
        "box_shape_zyx": torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long),
        "atom_counts": torch.tensor([2, 1], dtype=torch.long),
        "atom_offsets": torch.tensor([2, 3], dtype=torch.long),
        "atom_batch_index": torch.tensor([0, 0, 1], dtype=torch.long),
        "atom_feat": torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        "atom_coord_centered_world": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "atom_coord_local_voxel": torch.tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]),
        "atom_coord_world": torch.tensor([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        "atom_label": torch.tensor([1, 0, 1], dtype=torch.long),
        "atom_is_in_core_box": torch.tensor([True, True, False]),
        "atom_global_indices": torch.tensor([100, 101, 102], dtype=torch.long),
        "voxel_size_world": torch.ones(2, 3),
    }


def _make_pseudo_dict() -> dict[str, torch.Tensor]:
    """
    构造两个 BOX 的 P anchor 字典。

    输出:
        - pseudo_dict: dict[str, torch.Tensor], pseudo_counts=[1, 2], pseudo_feat 形状为 (3, 2)
    """
    return {
        "pseudo_counts": torch.tensor([1, 2], dtype=torch.long),
        "pseudo_batch_index": torch.tensor([0, 1, 1], dtype=torch.long),
        "pseudo_feat": torch.tensor([[9.0, 90.0], [8.0, 80.0], [7.0, 70.0]]),
        "pseudo_coord_centered_world": torch.tensor([[9.0, 0.0, 0.0], [8.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        "pseudo_coord_local_voxel": torch.tensor([[3.5, 0.5, 0.5], [2.5, 1.5, 0.5], [1.5, 1.5, 0.5]]),
        "pseudo_coord_world": torch.tensor([[19.0, 0.0, 0.0], [18.0, 0.0, 0.0], [17.0, 0.0, 0.0]]),
    }


def test_build_layout_counts_match() -> None:
    """
    验证 layout 的 real/pseudo/mixed 逐 BOX 计数。
    """
    layout = build_layout(_make_real_batch(), _make_pseudo_dict())
    assert layout.real_counts.tolist() == [2, 1]
    assert layout.pseudo_counts.tolist() == [1, 2]
    assert layout.mixed_counts.tolist() == [3, 3]
    assert layout.batch_size == 2


def test_inject_pseudo_atoms_interleaves_fields() -> None:
    """
    验证 inject_pseudo_atoms 按 `[real_i, pseudo_i]` mixed 顺序交错字段。
    """
    mixed_batch, _layout = inject_pseudo_atoms(_make_real_batch(), _make_pseudo_dict())
    assert mixed_batch["atom_feat"].tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
        [9.0, 90.0],
        [3.0, 30.0],
        [8.0, 80.0],
        [7.0, 70.0],
    ]
    assert mixed_batch["atom_offsets"].tolist() == [3, 6]
    assert mixed_batch["atom_batch_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert mixed_batch["atom_label"].tolist() == [1, 0, 0, 1, 0, 0]
    assert mixed_batch["atom_is_in_core_box"].tolist() == [True, True, True, False, True, True]
    assert mixed_batch["atom_global_indices"].tolist() == [100, 101, -1, 102, -1, -1]


def test_build_real_and_pseudo_mask() -> None:
    """
    验证 real/pseudo mask 与 mixed 顺序一致。
    """
    layout = build_layout(_make_real_batch(), _make_pseudo_dict())
    assert build_real_mask(layout).tolist() == [True, True, False, True, False, False]
    assert build_pseudo_mask(layout).tolist() == [False, False, True, False, True, True]


def test_remove_pseudo_atoms_restores_real_batch() -> None:
    """
    验证 inject 后 remove 能恢复 real-only atom 字段与计数。
    """
    real_batch = _make_real_batch()
    mixed_batch, layout = inject_pseudo_atoms(real_batch, _make_pseudo_dict())
    restored_batch = remove_pseudo_atoms(mixed_batch, layout)
    for key in (
        "atom_feat",
        "atom_coord_centered_world",
        "atom_coord_local_voxel",
        "atom_coord_world",
        "atom_label",
        "atom_is_in_core_box",
        "atom_global_indices",
        "atom_counts",
        "atom_offsets",
        "atom_batch_index",
    ):
        assert torch.equal(restored_batch[key], real_batch[key])
    assert "pseudo_mask" not in restored_batch
    assert "real_mask" not in restored_batch


def test_extract_real_and_pseudo_tensor_from_mixed() -> None:
    """
    验证 mixed 张量可以按 layout 裁成 real-only 与 pseudo-only。
    """
    layout = build_layout(_make_real_batch(), _make_pseudo_dict())
    mixed_tensor = torch.tensor([10, 11, 90, 12, 80, 70])
    assert extract_real_tensor_from_mixed(mixed_tensor, layout).tolist() == [10, 11, 12]
    assert extract_pseudo_tensor_from_mixed(mixed_tensor, layout).tolist() == [90, 80, 70]


def test_interleave_rejects_wrong_lengths() -> None:
    """
    验证 interleave_real_and_pseudo_tensor 对错误 real/pseudo 长度 fail-fast。
    """
    layout = build_layout(_make_real_batch(), _make_pseudo_dict())
    with pytest.raises(RuntimeError):
        interleave_real_and_pseudo_tensor(torch.ones(4, 2), layout)
    with pytest.raises(RuntimeError):
        interleave_real_and_pseudo_tensor(torch.ones(3, 2), layout, torch.ones(4, 2))


def test_no_legacy_generator_symbols() -> None:
    """
    验证新 pseudo_atoms 模块不再暴露旧随机生成器符号。
    """
    assert not hasattr(pseudo_atoms, "PseudoAtomGenerator")
    assert not hasattr(pseudo_atoms, "generate")
    assert not hasattr(pseudo_atoms, "recycle_policy")
