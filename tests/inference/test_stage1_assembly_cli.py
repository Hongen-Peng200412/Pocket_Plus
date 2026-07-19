"""A—G 装配、固定清单 CLI 与 Selected role 的端到端轻量测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.artifacts.io import load_npz_strict
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import is_role_complete, mark_role_complete
from src.component_lineage import ComponentForest, ComponentNode, ComponentTree
from src.datasets.density_channel_builder import ALL_CHANNEL_NAMES
from src.inference.assembly import (
    AGOccurrenceVoxelLoader,
    Stage1RuntimeAssembly,
    build_production_tasks,
    load_pdb_id_list,
)
from src.inference.centered import CenteredRequest
from src.inference.cli import DEFAULT_MAX_VOXELS, build_parser, main
from src.inference.runner import (
    ProductionTask,
    Stage1ProductionRunner,
    make_selected_refined_role_producer,
)


SHAPE = (80, 80, 80)


class _ProbabilityWrapper:
    """为完整图测试返回零 logit，并记录统一 Dataset batch。"""

    def __init__(self) -> None:
        self.last_batch = None
        self.device = None

    def to(self, device: str):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def forward_voxel_probability(self, batch):
        self.last_batch = batch
        batch_size = int(batch["density_input"].shape[0])
        return torch.zeros((batch_size, *SHAPE), dtype=torch.float32)

    def __call__(self, batch):
        self.last_batch = batch
        return {}


def _write_ag_fixture(root: Path, pdb_id: str = "1abc") -> None:
    """写入可由正式 A—G loaders 消费的最小 schema-v3 PDB。"""
    density_dir = root / "density" / pdb_id
    parse_dir = root / "parse" / pdb_id
    density_dir.mkdir(parents=True)
    parse_dir.mkdir(parents=True)
    voxel_size = np.ones(3, dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    exp = np.zeros((1, *SHAPE), dtype=np.float32)
    sim = np.ones((1, *SHAPE), dtype=np.float32)
    np.savez(density_dir / "exp.npz", grid=exp, voxel_size=voxel_size, origin=origin)
    np.savez(density_dir / "sim.npz", grid=sim, voxel_size=voxel_size, origin=origin)

    sparse = np.asarray([[5, 5, 5]], dtype=np.int32)
    union = np.zeros((1, *SHAPE), dtype=np.bool_)
    union[(0, sparse[:, 0], sparse[:, 1], sparse[:, 2])] = True
    np.savez_compressed(
        density_dir / "ligand_area.npz",
        schema_version=np.asarray(3, dtype=np.int32),
        grid_shape_zyx=np.asarray(SHAPE, dtype=np.int64),
        union_mask=union,
        mask_7=sparse,
    )
    coords = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32)
    feat = np.zeros((1, 49), dtype=np.float32)
    np.savez(parse_dir / "receptor_tokens.npz", coords=coords, feat=feat)


def _write_config(path: Path, producer: str = "Find_0") -> None:
    """写入只含 runtime Dataset 契约的 resolved YAML。"""
    path.write_text(
        "\n".join(
            (
                "dataset:",
                f"  stage1_model_name: {producer}",
                "  atom_buffer_radius: 8.0",
                "  density_channel_config:",
                "    clip_percentile: [0.001, 0.999]",
                "    fit_mask_percentile: 0.003",
                "    enabled_channels: [ALL]",
            )
        ),
        encoding="utf-8",
    )


def _common_cli_arguments(
    command: str,
    data_root: Path,
    output_root: Path,
    checkpoint: Path,
    config: Path,
    pdb_list: Path,
) -> list[str]:
    """返回五个生产命令共用的显式参数。"""
    return [
        command,
        "--producer",
        "Find_0",
        "--pdb-list",
        str(pdb_list),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--device",
        "cpu",
    ]


def test_fixed_list_sharding_is_stable_and_rejects_duplicates(tmp_path: Path) -> None:
    """固定清单只按原始行号取模，不扫描输出目录追求最新样本。"""
    source = tmp_path / "pdb_ids.json"
    source.write_text(json.dumps(["3CCC", "1AAA", "2BBB", "4DDD"]), encoding="utf-8")
    assert load_pdb_id_list(source) == ("3ccc", "1aaa", "2bbb", "4ddd")
    tasks = build_production_tasks("Find_0", "train", source, 1, 2)
    assert [task.pdb_id for task in tasks] == ["1aaa", "4ddd"]

    source.write_text(json.dumps(["1AAA", "1aaa"]), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_pdb_id_list(source)


def test_runtime_assembly_uses_in_memory_dataset_and_find_full_hardmask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A—G→统一 Dataset/collator→voxel-only 的过程中不创建临时 split 文件。"""
    data_root = tmp_path / "ag"
    _write_ag_fixture(data_root)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "BEST.ckpt"
    _write_config(config)
    checkpoint.touch()
    wrapper = _ProbabilityWrapper()
    from src.datasets import stage1_dataset as stage1_dataset_module

    original_grid_loader = stage1_dataset_module._grid_from_npz
    grid_loads: list[str] = []

    def record_grid_load(path):
        grid_loads.append(Path(path).name)
        return original_grid_loader(path)

    monkeypatch.setattr(stage1_dataset_module, "_grid_from_npz", record_grid_load)
    monkeypatch.setattr(
        "src.datasets.stage1_dataset.build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros(
            (len(ALL_CHANNEL_NAMES), *exp_raw.shape), dtype=np.float32
        ),
    )
    def reject_temporary_split(*args, **kwargs):
        del args, kwargs
        raise AssertionError("runtime assembly 不得创建临时 split 文件")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", reject_temporary_split)
    runtime = Stage1RuntimeAssembly(
        data_root=data_root,
        stage1_model_name="Find_0",
        checkpoint_path=checkpoint,
        resolved_config_path=config,
        device="cpu",
        window_batch_size=1,
        wrapper_loader=lambda **kwargs: wrapper,
    )
    task = ProductionTask("Find_0", "calibration", "1abc")
    inputs = runtime.full_map_input(task)
    batch = inputs.window_batch_builder([(0, 0, 0)])
    centered_batch = runtime.centered_batch_builder(task)(
        CenteredRequest(
            stage1_model_name="Find_0",
            split="calibration",
            pdb_id="1abc",
            centered_role="F1_centered",
            centered_box_index=0,
            box_start_zyx=(0, 0, 0),
            source_tree_id=0,
            source_node_id=0,
            source_threshold_grid_index=16384,
        )
    )

    assert batch["density_input"].shape == (1, 56, *SHAPE)
    assert centered_batch["density_input"].shape == (1, 56, *SHAPE)
    assert batch["atom_global_indices"].tolist() == [0]
    assert bool(inputs.receptor_hardmask_full[0, 0, 0])
    assert grid_loads.count("exp.npz") == 1
    assert grid_loads.count("sim.npz") == 1
    occurrences = runtime.occurrence_voxels(task, SHAPE)
    assert occurrences[7].tolist() == [np.ravel_multi_index((5, 5, 5), SHAPE)]


def test_cli_calibration_then_cal_centered_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合成 A—G fixture 贯通 probability→freeze→cal F1/CLG，并验证续跑跳过。"""
    data_root = tmp_path / "ag"
    output_root = tmp_path / "outputs"
    _write_ag_fixture(data_root)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "BEST.ckpt"
    pdb_list = tmp_path / "cal.json"
    _write_config(config)
    checkpoint.touch()
    pdb_list.write_text(json.dumps(["1abc"]), encoding="utf-8")
    wrapper = _ProbabilityWrapper()
    monkeypatch.setattr(
        "src.datasets.stage1_dataset.build_density_channels",
        lambda exp_raw, sim_raw, config, receptor_mask: np.zeros(
            (len(ALL_CHANNEL_NAMES), *exp_raw.shape), dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "src.inference.assembly.load_stage1_wrapper",
        lambda **kwargs: wrapper,
    )

    probability_args = _common_cli_arguments(
        "cal-probability",
        data_root,
        output_root,
        checkpoint,
        config,
        pdb_list,
    )
    assert main(probability_args) == 0
    paths = Stage1ArtifactPaths(output_root, "Find_0", "calibration", "1abc")
    assert is_role_complete(paths, "probability")
    probability = load_npz_strict(paths.probability_npz)["probability_map"]
    assert probability[0, 0, 0] == 0.0
    assert probability[5, 5, 5] == pytest.approx(0.5)
    # 同一命令再次运行只读现有 marker，不重新 forward。
    first_batch = wrapper.last_batch
    assert main(probability_args) == 0
    assert wrapper.last_batch is first_batch

    assert main(
        [
            "freeze-thresholds",
            "--producer",
            "Find_0",
            "--pdb-list",
            str(pdb_list),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ]
    ) == 0
    threshold_payload = json.loads(paths.thresholds_json.read_text(encoding="utf-8"))
    assert threshold_payload["max_voxels"] == 1023

    assert main(
        _common_cli_arguments(
            "cal-produce-f1-clg",
            data_root,
            output_root,
            checkpoint,
            config,
            pdb_list,
        )
    ) == 0
    assert is_role_complete(paths, "components")
    assert is_role_complete(paths, "F1_centered")
    assert is_role_complete(paths, "CLG_centered")

    for command, split in (
        ("val-produce-prob-f1-clg", "validation"),
        ("train-produce-prob-f1-clg", "train"),
    ):
        assert main(
            _common_cli_arguments(
                command,
                data_root,
                output_root,
                checkpoint,
                config,
                pdb_list,
            )
        ) == 0
        split_paths = Stage1ArtifactPaths(output_root, "Find_0", split, "1abc")
        for role in ("probability", "components", "F1_centered", "CLG_centered"):
            assert is_role_complete(split_paths, role)


def test_freeze_cli_default_max_voxels_is_frozen_q95_value() -> None:
    """CLI 缺省值只能是全量 Q95=682×1.5 得到的正式 1023。"""
    arguments = build_parser().parse_args(
        [
            "freeze-thresholds",
            "--producer",
            "unet_c1",
            "--pdb-list",
            "ids.json",
            "--data-root",
            "ag",
            "--output-root",
            "out",
        ]
    )
    assert DEFAULT_MAX_VOXELS == 1023
    assert arguments.max_voxels == 1023


def test_selected_role_loads_selection_reruns_and_publishes(tmp_path: Path) -> None:
    """Selected producer 恢复 source node，以原阈值重跑并发布成功 refined blob。"""
    output_root = tmp_path / "outputs"
    paths = Stage1ArtifactPaths(output_root, "unet_c1", "train", "1abc")
    paths.forest_npz.parent.mkdir(parents=True)
    voxel = np.asarray([np.ravel_multi_index((10, 10, 10), SHAPE)], dtype=np.int64)
    node = ComponentNode(
        tree_id=0,
        node_id=3,
        threshold_grid_index=16384,
        threshold_value=0.5,
        voxel_global_linear_index=voxel,
        bbox_min_zyx=np.asarray([10, 10, 10], dtype=np.int32),
        bbox_max_zyx=np.asarray([10, 10, 10], dtype=np.int32),
        centroid_zyx=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        probability_mean=0.9,
        probability_max=0.9,
        candidate_eligible=True,
        ineligible_reason_code=0,
    )
    np.savez_compressed(
        paths.forest_npz,
        **ComponentForest((ComponentTree(0, (node,)),)).to_arrays(),
    )
    np.savez_compressed(
        paths.clg_npz,
        CLG_id=np.asarray([0], dtype=np.int32),
        tree_id=np.asarray([0], dtype=np.int32),
        candidate_offsets=np.asarray([0, 1], dtype=np.int64),
        candidate_node_id=np.asarray([3], dtype=np.int32),
    )
    selection_path = paths.pdb_root / "selector" / "selection.npz"
    selection_path.parent.mkdir(parents=True)
    np.savez_compressed(
        selection_path,
        CLG_id=np.asarray([0], dtype=np.int32),
        CLG_gate_pass=np.asarray([True], dtype=np.bool_),
        selected_candidate_offsets=np.asarray([0, 1], dtype=np.int64),
        selected_candidate_index=np.asarray([0], dtype=np.int16),
    )
    paths.probability_geometry_json.parent.mkdir(parents=True, exist_ok=True)
    paths.probability_geometry_json.write_text(
        json.dumps(
            {
                "full_shape_zyx": list(SHAPE),
                "origin_xyz": [0.0, 0.0, 0.0],
                "voxel_size_xyz": [1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    mark_role_complete(paths, "components")

    probability = np.zeros(SHAPE, dtype=np.float32)
    probability[10, 10, 10] = 0.9
    voxel_final = np.broadcast_to(
        np.zeros((48, 1, 1, 1), dtype=np.float16),
        (48, *SHAPE),
    )

    def adapter(output, batch):
        del output, batch
        return {
            "ligand_probability": probability,
            "hardmask": np.zeros(SHAPE, dtype=np.bool_),
            "voxel_aux_probability_grid": np.zeros(SHAPE, dtype=np.float32),
            "voxel_final_grid": voxel_final,
            "voxel_ds_2": np.zeros((256, 20, 20, 20), dtype=np.float16),
            "voxel_ds_3": np.zeros((256, 10, 10, 10), dtype=np.float16),
            "voxel_ds_4": np.zeros((256, 5, 5, 5), dtype=np.float16),
            "voxel_c4": np.zeros((256, 5, 5, 5), dtype=np.float16),
        }

    wrapper = _ProbabilityWrapper()
    producer = make_selected_refined_role_producer(
        wrapper_provider=lambda task: wrapper,
        batch_builder_provider=lambda task: lambda request: {},
        output_adapter=adapter,
    )
    runner = Stage1ProductionRunner.for_current_process(
        output_root=str(output_root),
        role_producers={"Selected_Refined_Centered": producer},
    )
    record = runner.run_selected((ProductionTask("unet_c1", "train", "1abc"),))[0]
    assert record.status == "completed"
    assert is_role_complete(paths, "Selected_Refined_Centered")
    selected = load_npz_strict(paths.centered_npz("Selected_Refined_Centered"))
    assert selected["source_node_id"].tolist() == [3]
    assert selected["refine_status"].tolist() == [0]
    assert selected["voxel_index_local_zyx"].tolist() == [[10, 10, 10]]


def test_occurrence_loader_rejects_union_drift(tmp_path: Path) -> None:
    """occurrence 稀疏 mask 与 union 不一致时必须 fail-fast。"""
    _write_ag_fixture(tmp_path)
    path = tmp_path / "density" / "1abc" / "ligand_area.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["union_mask"] = np.zeros((1, *SHAPE), dtype=np.bool_)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="并集"):
        AGOccurrenceVoxelLoader(tmp_path)("1abc", SHAPE)
