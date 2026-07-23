from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

import src.datasets.stage1_dataset as stage1_dataset_module
from src.datasets.density_channel_builder import ALL_CHANNEL_NAMES
from src.datasets.stage1_collate import Stage1BatchCollator
from src.datasets.stage1_dataset import Stage1Dataset
from src.datasets.stage1_box_pool import build_stage1_box_pools, generate_context_starts
from src.datasets.stage1_requests import (
    Stage1TrainingRequestSet,
    build_request_source,
    centered_start_from_centroid_zyx,
    centered_start_from_sparse_mask,
    load_validation_selection,
    resolve_stage1_start,
)


def _write_upstream(root: Path, pdb_id: str = "1abc", shape: tuple[int, int, int] = (80, 80, 80)) -> None:
    density_dir = root / "density" / pdb_id
    parse_dir = root / "parse" / pdb_id
    label_dir = root / "labels" / pdb_id
    density_dir.mkdir(parents=True)
    parse_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    voxel_size = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    origin = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    z, y, x = np.indices(shape, dtype=np.float32)
    exp = (x + 2.0 * y + 3.0 * z)[None]
    sim = (0.5 * x + y + 0.25 * z + 1.0)[None]
    np.savez(density_dir / "exp.npz", grid=exp, voxel_size=voxel_size, origin=origin)
    np.savez(density_dir / "sim.npz", grid=sim, voxel_size=voxel_size, origin=origin)
    union = np.zeros((1, *shape), dtype=bool)
    union[0, 4, 3, 2] = True
    np.savez_compressed(density_dir / "ligand_area.npz", union_mask=union)
    distance = np.full((1, *shape), 100.0, dtype=np.float16)
    distance[0, 4, 3, 2] = np.float16(3.0)
    np.savez_compressed(
        density_dir / "ligand_dist.npz",
        distance=distance,
        schema_version=np.asarray(1, dtype=np.uint16),
        grid_shape_zyx=np.asarray(shape, dtype=np.int64),
        voxel_size_xyz=voxel_size,
        origin_xyz=origin,
        distance_unit=np.asarray("angstrom"),
    )

    coords = np.asarray(
        [
            [10.5, 20.5, 30.5],       # core home z/y/x = 0/0/0
            [89.5, 99.5, 109.5],      # core home z/y/x = 79/79/79
            [94.0, 50.0, 60.0],       # core 外 4 Å，属于 8 Å point buffer
            [99.0, 50.0, 60.0],       # core 外 9 Å，不应加载
        ],
        dtype=np.float32,
    )
    feat = np.arange(coords.shape[0] * 49, dtype=np.float32).reshape(coords.shape[0], 49)
    np.savez(
        parse_dir / "receptor_tokens.npz",
        coords=coords,
        feat=feat,
        res_type=np.asarray([0, 0, 20, 28], dtype=np.uint8),
        atom_name=np.asarray([b"CA", b"N", b"P", b"CB"], dtype="S4"),
    )
    np.savez(label_dir / "atom_labels.npz", binding_atom=np.asarray([True, False, True, False]))


def _write_request(path: Path, require_targets: bool) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "pdb_id": "1ABC",
                    "box_start_zyx": [0, 0, 0],
                    "require_targets": require_targets,
                    "role": "centered",
                    "occurrence_id": 0,
                }
            ]
        ),
        encoding="utf-8",
    )


def _density_config(channels: list[str]) -> dict[str, object]:
    return {
        "clip_percentile": [0.001, 0.999],
        "fit_mask_percentile": 0.003,
        "enabled_channels": channels,
    }


def _write_pool_upstream(root: Path, pdb_id: str) -> None:
    """写一份足以执行 BOX pool 一键入口的 schema-v3 轻量资产。"""

    density_dir = root / "density" / pdb_id
    parse_dir = root / "parse" / pdb_id
    density_dir.mkdir(parents=True)
    parse_dir.mkdir(parents=True)
    shape = (80, 80, 80)
    origin = np.zeros(3, dtype=np.float32)
    voxel_size = np.ones(3, dtype=np.float32)
    np.savez(
        density_dir / "exp.npz",
        grid=np.zeros((1, *shape), dtype=np.float32),
        origin=origin,
        voxel_size=voxel_size,
    )
    sparse = np.asarray([[39, 39, 39], [40, 40, 40], [41, 41, 41]], dtype=np.int32)
    union = np.zeros((1, *shape), dtype=bool)
    union[(0, sparse[:, 0], sparse[:, 1], sparse[:, 2])] = True
    np.savez_compressed(
        density_dir / "ligand_area.npz",
        schema_version=np.asarray(3, dtype=np.int32),
        grid_shape_zyx=np.asarray(shape, dtype=np.int64),
        union_mask=union,
        mask_7=sparse,
    )
    coords = np.full((1000, 3), 40.0, dtype=np.float32)
    np.savez(parse_dir / "receptor_tokens.npz", coords=coords)


def test_resolve_stage1_start_clamps_without_padding() -> None:
    assert resolve_stage1_start((-5, 99, 12), (90, 100, 80)) == (0, 20, 0)
    with pytest.raises(ValueError, match="不小于"):
        resolve_stage1_start((0, 0, 0), (79, 80, 80))


def test_centroid_start_matches_sparse_mask_and_clamps_boundaries() -> None:
    """验证 forest/centered 质心入口与 occurrence sparse-mask 入口完全同口径。"""

    sparse = np.asarray([[0, 2, 4], [2, 4, 6], [4, 6, 8]], dtype=np.int32)
    centroid = sparse.astype(np.float64).mean(axis=0)
    direct = centered_start_from_centroid_zyx(centroid, (120, 130, 140))
    from_sparse = centered_start_from_sparse_mask(sparse, (120, 130, 140))

    assert direct == from_sparse == (0, 0, 0)
    assert centered_start_from_centroid_zyx((119.0, 129.0, 139.0), (120, 130, 140)) == (40, 50, 60)


def test_centroid_start_rejects_non_finite_values() -> None:
    """非有限 blob 质心不得静默生成错误 BOX。"""

    with pytest.raises(ValueError, match="NaN/Inf"):
        centered_start_from_centroid_zyx((np.nan, 40.0, 40.0), (100, 100, 100))


def test_find_dataset_materializes_direct_core8_and_union_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_upstream(tmp_path)
    monkeypatch.setattr(
        "src.datasets.stage1_dataset.build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros((56, *exp_raw.shape), dtype=np.float32),
    )
    manifest = tmp_path / "requests.json"
    _write_request(manifest, require_targets=True)
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=str(manifest),
        mode="val",
        stage1_model_name="Find_1",
        box_pool_root=None,
        density_channel_config=_density_config(list(ALL_CHANNEL_NAMES)),
        enable_random_rotation=False,
    )

    sample = dataset[0]
    assert sample["pdb_id"] == "1abc"
    assert sample["density_input"].shape == (56, 80, 80, 80)
    assert sample["density_input"].dtype == torch.float32
    assert sample["atom_global_indices"].tolist() == [0, 1, 2]
    assert sample["atom_is_in_core_box"].tolist() == [True, True, False]
    assert sample["hardmask"].sum().item() == 2
    assert sample["voxel_label"].sum().item() == 1
    assert sample["ligand_area_target"][4, 3, 2].item() is True
    assert sample["protein_mainchain_target"][0, 0, 0].item() == 2
    assert sample["protein_mainchain_target"][79, 79, 79].item() == 1
    assert sample["nucleic_mainchain_target"].sum().item() == 0
    assert sample["ligand_inverse_distance_target"][4, 3, 2].item() == pytest.approx(0.25)
    assert sample["ligand_inverse_distance_target"][0, 0, 0].item() == pytest.approx(
        1.0 / 101.0
    )
    assert "ligand_dist_map" not in sample


def test_ligand_distance_loader_rejects_mixed_finite_and_infinite_values(
    tmp_path: Path,
) -> None:
    """有配体距离图必须全部有限；无配体距离图才允许全部为正无穷。"""

    _write_upstream(tmp_path)
    distance_path = tmp_path / "density" / "1abc" / "ligand_dist.npz"
    distance = np.ones((1, 80, 80, 80), dtype=np.float16)
    distance[0, 0, 0, 0] = np.inf
    np.savez_compressed(
        distance_path,
        distance=distance,
        schema_version=np.asarray(1, dtype=np.uint16),
        grid_shape_zyx=np.asarray([80, 80, 80], dtype=np.int64),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
        origin_xyz=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        distance_unit=np.asarray("angstrom"),
    )
    request_path = tmp_path / "requests.json"
    _write_request(request_path, require_targets=True)
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=str(request_path),
        mode="val",
        stage1_model_name="Find_1",
        box_pool_root=None,
        density_channel_config=_density_config(list(ALL_CHANNEL_NAMES)),
        enable_random_rotation=False,
    )

    with pytest.raises(ValueError, match="全部有限且非负，或全部为正无穷"):
        dataset._load_ligand_distance(
            "1abc",
            (80, 80, 80),
            np.ones(3, dtype=np.float32),
            np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        )


def test_unet_dataset_does_not_read_sim_or_return_atoms(tmp_path: Path) -> None:
    _write_upstream(tmp_path)
    (tmp_path / "density" / "1abc" / "sim.npz").unlink()
    manifest = tmp_path / "requests.json"
    _write_request(manifest, require_targets=True)
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=str(manifest),
        mode="val",
        stage1_model_name="unet_c1",
        box_pool_root=None,
        density_channel_config=_density_config(["exp_clipnorm_nopost"]),
        enable_random_rotation=False,
    )
    sample = dataset[0]
    assert sample["density_input"].shape == (1, 80, 80, 80)
    assert "atom_feat" not in sample
    assert sample["hardmask"].sum().item() == 2
    assert sample["voxel_label"].sum().item() == 1


def test_dataset_excludes_pdb_without_rewriting_request_file(tmp_path: Path) -> None:
    """训练配置排除 PDB 时只改变可读样本，不改写原请求文件。"""

    request_path = tmp_path / "requests.json"
    request_rows = [
        {
            "pdb_id": pdb_id,
            "box_start_zyx": [0, 0, 0],
            "require_targets": True,
            "role": "centered",
        }
        for pdb_id in ("1ABC", "2DEF")
    ]
    original_text = json.dumps(request_rows)
    request_path.write_text(original_text, encoding="utf-8")

    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=str(request_path),
        mode="val",
        stage1_model_name="unet_c1",
        box_pool_root=None,
        density_channel_config=_density_config(["exp_clipnorm_nopost"]),
        excluded_pdb_ids=["1AbC"],
        enable_random_rotation=False,
    )

    assert len(dataset) == 1
    assert dataset.describe_index(0).startswith("pdb_id=2def")
    assert request_path.read_text(encoding="utf-8") == original_text


def test_targets_toggle_does_not_change_model_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_upstream(tmp_path)
    monkeypatch.setattr(
        "src.datasets.stage1_dataset.build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros((56, *exp_raw.shape), dtype=np.float32),
    )
    with_targets = tmp_path / "with.json"
    without_targets = tmp_path / "without.json"
    _write_request(with_targets, require_targets=True)
    _write_request(without_targets, require_targets=False)
    kwargs = dict(
        all_data_path=str(tmp_path),
        mode="centered",
        stage1_model_name="Find_0",
        box_pool_root=None,
        density_channel_config=_density_config(list(ALL_CHANNEL_NAMES)),
        enable_random_rotation=False,
    )
    sample_true = Stage1Dataset(split_file=str(with_targets), **kwargs)[0]
    sample_false = Stage1Dataset(split_file=str(without_targets), **kwargs)[0]
    for field_name in (
        "density_input",
        "hardmask",
        "atom_global_indices",
        "atom_feat",
        "atom_coord_local_voxel",
        "atom_coord_centered_world",
        "atom_is_in_core_box",
    ):
        assert torch.equal(sample_true[field_name], sample_false[field_name])
    assert "ligand_area_target" not in sample_false
    assert "voxel_label" not in sample_false
    assert "atom_label" not in sample_false


def test_repeated_windows_reuse_bounded_full_grid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 worker 连续物化同一 PDB 时，exp/sim 各只解压一次。"""

    _write_upstream(tmp_path)
    request_path = tmp_path / "requests.json"
    _write_request(request_path, require_targets=False)
    read_count: Counter[str] = Counter()
    from src.datasets import stage1_dataset as dataset_module

    original_loader = dataset_module._grid_from_npz

    def counted_loader(path: Path):
        read_count[path.name] += 1
        return original_loader(path)

    monkeypatch.setattr(dataset_module, "_grid_from_npz", counted_loader)
    monkeypatch.setattr(
        dataset_module,
        "build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros(
            (56, *exp_raw.shape), dtype=np.float32
        ),
    )
    dataset = Stage1Dataset(
        all_data_path=str(tmp_path),
        split_file=str(request_path),
        mode="centered",
        stage1_model_name="Find_1",
        box_pool_root=None,
        density_channel_config=_density_config(list(ALL_CHANNEL_NAMES)),
        cache_max_bytes=64 * 1024 * 1024,
        enable_random_rotation=False,
    )

    first = dataset[0]
    second = dataset[0]

    assert read_count == Counter({"exp.npz": 1, "sim.npz": 1})
    assert torch.equal(first["density_input"], second["density_input"])


def test_collator_exposes_b_plus_one_offsets_and_handles_empty_atoms() -> None:
    common = {
        "pdb_id": "x",
        "request_role": "centered",
        "occurrence_id": None,
        "candidate_index": None,
        "box_start_zyx": torch.zeros(3, dtype=torch.int32),
        "box_shape_zyx": torch.full((3,), 80, dtype=torch.int64),
        "box_origin_world": torch.zeros(3),
        "voxel_size_world": torch.ones(3),
        "density_input": torch.zeros(1, 2, 2, 2),
        "hardmask": torch.zeros(2, 2, 2, dtype=torch.bool),
    }
    atom_tail = {
        "atom_global_indices": torch.empty(0, dtype=torch.int64),
        "atom_feat": torch.empty(0, 49),
        "atom_coord_world": torch.empty(0, 3),
        "atom_coord_local_voxel": torch.empty(0, 3),
        "atom_coord_centered_world": torch.empty(0, 3),
        "atom_is_in_core_box": torch.empty(0, dtype=torch.bool),
    }
    sample0 = {**common, **atom_tail}
    sample1 = {**common, **atom_tail, "atom_feat": torch.ones(2, 49)}
    sample1.update(
        {
            "atom_global_indices": torch.arange(2),
            "atom_coord_world": torch.zeros(2, 3),
            "atom_coord_local_voxel": torch.zeros(2, 3),
            "atom_coord_centered_world": torch.zeros(2, 3),
            "atom_is_in_core_box": torch.ones(2, dtype=torch.bool),
        }
    )
    batch = Stage1BatchCollator()([sample0, sample1])
    assert batch["atom_counts"].tolist() == [0, 2]
    assert batch["atom_offsets"].tolist() == [0, 0, 2]
    assert batch["atom_batch_index"].tolist() == [1, 1]


def test_training_pool_rebuilds_fixed_1_5_3_ratio(tmp_path: Path) -> None:
    pool_dir = tmp_path / "train"
    pool_dir.mkdir()
    occurrence = np.arange(55, dtype=np.int32)
    center = np.stack([occurrence, occurrence, occurrence], axis=1)
    bias = np.repeat(center[:, None, :], 30, axis=1)
    context = np.stack([np.arange(10), np.arange(10), np.arange(10)], axis=1).astype(np.int32)
    np.savez(
        pool_dir / "1abc.npz",
        occurrence_id=occurrence,
        center_start_zyx=center,
        bias_start_zyx=bias,
        context_start_zyx=context,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "train": [{"pdb_id": "1abc", "path": "train/1abc.npz"}],
                    "validation": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "_COMPLETE").write_text("", encoding="utf-8")
    source = Stage1TrainingRequestSet(pool_dir, seed=7)
    epoch0 = tuple(source.requests)
    assert len(epoch0) == 50 * 9
    assert sum(request.role == "center" for request in epoch0) == 50
    assert sum(request.role == "bias" for request in epoch0) == 250
    assert sum(request.role == "context" for request in epoch0) == 150
    source.set_epoch(1)
    assert tuple(source.requests) != epoch0
    source_again = Stage1TrainingRequestSet(pool_dir, seed=7)
    source_again.set_epoch(1)
    assert tuple(source_again.requests) == tuple(source.requests)

    reduced = Stage1TrainingRequestSet(pool_dir, seed=7, box_sample_fraction=0.1)
    frozen = tuple(reduced.requests)
    assert len(frozen) == 45
    assert sum(request.role == "center" for request in frozen) == 5
    assert sum(request.role == "bias" for request in frozen) == 25
    assert sum(request.role == "context" for request in frozen) == 15
    assert (tmp_path / "train_selection_0.1_seed7.npz").is_file()
    with np.load(tmp_path / "train_selection_0.1_seed7.npz", allow_pickle=False) as saved:
        assert float(saved["box_sample_fraction"]) == pytest.approx(0.1)
        assert int(saved["request_seed"]) == 7
        assert int(saved["selection_epoch"]) == 0
        assert int(saved["schema_version"]) == 1
        assert len(str(saved["source_manifest_sha256"].item())) == 64
    reduced.set_epoch(9)
    assert tuple(reduced.requests) == frozen
    assert tuple(Stage1TrainingRequestSet(pool_dir, seed=7, box_sample_fraction=0.1).requests) == frozen
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="元数据与当前来源不一致"):
        Stage1TrainingRequestSet(pool_dir, seed=7, box_sample_fraction=0.1)


@pytest.mark.parametrize("context_count", (0, 1, 2))
def test_training_pool_context_underflow_does_not_abort(
    tmp_path: Path,
    context_count: int,
) -> None:
    """context 尝试耗尽后保留真实池；1–2 个可复用，0 个则只省略 context。"""

    pool_dir = tmp_path / "train"
    pool_dir.mkdir()
    context = np.zeros((context_count, 3), dtype=np.int32)
    np.savez(
        pool_dir / "1abc.npz",
        occurrence_id=np.asarray([7], dtype=np.int32),
        center_start_zyx=np.asarray([[0, 0, 0]], dtype=np.int32),
        bias_start_zyx=np.zeros((1, 30, 3), dtype=np.int32),
        context_start_zyx=context,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "train": [{"pdb_id": "1abc", "path": "train/1abc.npz"}],
                    "validation": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "_COMPLETE").write_text("", encoding="utf-8")

    requests = tuple(Stage1TrainingRequestSet(pool_dir, seed=7).requests)

    context_requests = [request for request in requests if request.role == "context"]
    assert len(context_requests) == (3 if context_count else 0)
    assert all(0 <= request.candidate_index < context_count for request in context_requests)
    assert sum(request.role == "center" for request in requests) == 1
    assert sum(request.role == "bias" for request in requests) == 5


def test_context_generator_uses_core_atom_count_and_stable_legal_starts() -> None:
    """验证 context 只按合法起点与 core receptor 重原子数筛选。"""

    coords = np.full((4, 3), 40.0, dtype=np.float32)
    first = generate_context_starts(
        receptor_coords_world=coords,
        full_origin_world=(0.0, 0.0, 0.0),
        voxel_size_world=(1.0, 1.0, 1.0),
        full_shape_zyx=(80, 80, 80),
        rng=np.random.default_rng(9),
        target_count=5,
        max_attempts=8,
        min_core_atoms=4,
    )
    second = generate_context_starts(
        receptor_coords_world=coords,
        full_origin_world=(0.0, 0.0, 0.0),
        voxel_size_world=(1.0, 1.0, 1.0),
        full_shape_zyx=(80, 80, 80),
        rng=np.random.default_rng(9),
        target_count=5,
        max_attempts=8,
        min_core_atoms=4,
    )

    assert first.shape == (5, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.zeros((5, 3), dtype=np.int32))


def test_synced_rotation_swaps_anisotropic_voxel_axes_and_keeps_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """90° 旋转必须同步置换轴尺度、体素监督和 Find 原子坐标。"""

    side = 4
    origin = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    voxel_size = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    local_xyz = np.asarray([[0.5, 1.5, 2.5]], dtype=np.float32)
    center = origin + 0.5 * side * voxel_size
    world = origin[None, :] + local_xyz * voxel_size[None, :]
    label = np.zeros((side, side, side), dtype=np.bool_)
    label[2, 1, 0] = True
    sample = {
        "density_input": label[None].astype(np.float32),
        "hardmask": label.copy(),
        "voxel_label": label.copy(),
        "ligand_area_target": label.copy(),
        "box_shape_zyx": np.asarray([side, side, side], dtype=np.int64),
        "box_origin_world": origin,
        "voxel_size_world": voxel_size,
        "atom_coord_local_voxel": local_xyz,
        "atom_coord_centered_world": world - center[None, :],
        "atom_coord_world": world,
    }
    monkeypatch.setattr(
        stage1_dataset_module.np.random,
        "choice",
        lambda *_args, **_kwargs: np.asarray([0, 1]),
    )
    monkeypatch.setattr(stage1_dataset_module.random, "randint", lambda *_args: 1)

    rotated = stage1_dataset_module._apply_synced_rotation(sample)

    np.testing.assert_array_equal(rotated["voxel_size_world"], [1.0, 3.0, 2.0])
    label_position = np.argwhere(rotated["voxel_label"])[0]
    atom_position = np.floor(rotated["atom_coord_local_voxel"][0]).astype(np.int64)[
        [2, 1, 0]
    ]
    np.testing.assert_array_equal(atom_position, label_position)
    np.testing.assert_allclose(
        rotated["atom_coord_world"],
        origin[None, :]
        + rotated["atom_coord_local_voxel"]
        * rotated["voxel_size_world"][None, :],
    )
    rotated_center = origin + 0.5 * side * rotated["voxel_size_world"]
    np.testing.assert_allclose(
        rotated["atom_coord_centered_world"],
        rotated["atom_coord_world"] - rotated_center[None, :],
    )


def test_box_pool_one_click_entry_publishes_train_validation_and_selection(tmp_path: Path) -> None:
    """验证一键入口生成 pool、冻结 validation 1:5:3，并最后发布完成标记。"""

    data_root = tmp_path / "data"
    _write_pool_upstream(data_root, "1abc")
    _write_pool_upstream(data_root, "2def")
    train_split = tmp_path / "train.json"
    validation_split = tmp_path / "validation.json"
    train_split.write_text(json.dumps([{"pdb_id": "1ABC"}]), encoding="utf-8")
    validation_split.write_text(json.dumps([{"pdb_id": "2DEF"}]), encoding="utf-8")
    output_root = tmp_path / "box_pool"
    (output_root / "train").mkdir(parents=True)
    (output_root / "validation").mkdir(parents=True)
    np.savez(output_root / "train" / "stale.npz", broken=np.asarray([1]))
    np.savez(output_root / "validation" / "stale.npz", broken=np.asarray([1]))

    summary = build_stage1_box_pools(
        data_root=data_root,
        train_split=train_split,
        validation_split=validation_split,
        output_root=output_root,
        seed=23,
    )

    assert summary["train"]["published_pdb"] == 1
    assert summary["validation"]["published_pdb"] == 1
    assert (output_root / "_COMPLETE").is_file()
    assert (output_root / "config.json").is_file()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"] == {
        "train": [{"pdb_id": "1abc", "path": "train/1abc.npz"}],
        "validation": [{"pdb_id": "2def", "path": "validation/2def.npz"}],
    }
    assert len(Stage1TrainingRequestSet(output_root / "train", seed=23)) == 9
    with np.load(output_root / "train" / "1abc.npz", allow_pickle=False) as pool:
        assert pool["occurrence_id"].tolist() == [7]
        assert pool["center_start_zyx"].shape == (1, 3)
        assert pool["bias_start_zyx"].shape == (1, 30, 3)
        assert pool["context_start_zyx"].shape == (500, 3)
    requests = load_validation_selection(
        output_root / "validation_selection.npz",
        output_root,
    )
    assert len(requests) == 9
    assert [request.role for request in requests].count("center") == 1
    assert [request.role for request in requests].count("bias") == 5
    assert [request.role for request in requests].count("context") == 3
    reduced_validation = build_request_source(
        split_file=output_root / "validation_selection.npz",
        mode="val",
        box_pool_root=output_root,
        seed=23,
        box_sample_fraction=0.5,
    )
    assert len(reduced_validation) == 4
    assert [request.role for request in reduced_validation].count("center") == 1
    assert [request.role for request in reduced_validation].count("bias") == 2
    assert [request.role for request in reduced_validation].count("context") == 1
    assert (output_root / "validation_selection_0.5_seed23.npz").is_file()
    with np.load(
        output_root / "validation_selection_0.5_seed23.npz",
        allow_pickle=False,
    ) as saved:
        assert len(str(saved["source_manifest_sha256"].item())) == 64
        assert len(str(saved["source_validation_sha256"].item())) == 64
