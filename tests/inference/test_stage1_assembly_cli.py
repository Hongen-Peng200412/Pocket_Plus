"""A—G 装配、固定清单 CLI 与 Selected role 的端到端轻量测试. """

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
from src.inference.cli import (
    DEFAULT_MAX_VOXELS,
    _standard_role_producers,
    build_parser,
    main,
)
from src.inference.runner import (
    ProductionTask,
    Stage1ProductionRunner,
    make_selected_refined_role_producer,
)


SHAPE = (80, 80, 80)


class _ProbabilityWrapper:
    """为完整图测试返回零 logit, 并记录统一 Dataset batch. """

    def __init__(self, centered_probability: np.ndarray | None = None) -> None:
        self.last_batch = None
        self.device = None
        self.centered_probability = centered_probability
        self.backbone = SimpleNamespace(
            voxel_backbone=SimpleNamespace(
                feature_channels_by_name={"voxel_final": 48}
            )
        )

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
        batch_size = int(batch["density_input"].shape[0])
        if self.centered_probability is None:
            ligand_logits = np.zeros((batch_size, 1, *SHAPE), dtype=np.float32)
        else:
            probability = np.asarray(self.centered_probability, dtype=np.float32)
            logits = np.full(SHAPE, -80.0, dtype=np.float32)
            positive = probability > 0.0
            logits[positive] = np.log(
                probability[positive] / (1.0 - probability[positive])
            )
            ligand_logits = np.broadcast_to(logits, (batch_size, 1, *SHAPE))
        voxel_features = {
            "voxel_final": np.zeros((batch_size, 48, *SHAPE), dtype=np.float16),
        }
        output = {
            "voxel_logits_ligand": ligand_logits,
            "voxel_logits_aux": np.zeros((batch_size, 1, *SHAPE), dtype=np.float32),
            "voxel_features": voxel_features,
        }
        if "atom_counts" not in batch:
            return output
        atom_count = int(batch["atom_global_indices"].shape[0])
        output.update(
            {
                "atom_counts": batch["atom_counts"],
                "atom_global_indices": batch["atom_global_indices"],
                "atom_coord_local_voxel": batch["atom_coord_local_voxel"],
                "atom_logits": np.zeros(atom_count, dtype=np.float32),
                "anchor_batch_index": np.empty(0, dtype=np.int64),
                "anchor_coord_local_voxel": np.empty((0, 3), dtype=np.float32),
                "pseudo_logits": np.empty(0, dtype=np.float32),
                **{
                    f"A_feat_L{level}": np.zeros((atom_count, level), dtype=np.float32)
                    for level in range(1, 5)
                },
                **{
                    f"P_feat_L{level}": np.empty((0, level), dtype=np.float32)
                    for level in range(2, 5)
                },
            }
        )
        return output


def _write_ag_fixture(root: Path, pdb_id: str = "1abc") -> None:
    """写入可由正式 A—G loaders 消费的最小 schema-v3 PDB. """
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
    np.savez(
        parse_dir / "receptor_tokens.npz",
        coords=coords,
        feat=feat,
        is_backbone=np.asarray([True], dtype=np.bool_),
    )


def _write_config(path: Path, producer: str = "Find_0") -> None:
    """写入只含 runtime Dataset 契约的 resolved YAML. """
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
    """返回八个 PDB 级模型推理命令共用的显式参数。"""
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
    """固定清单只按原始行号取模, 不扫描输出目录追求最新样本. """
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
    """A—G→统一 Dataset/collator→voxel-only 的过程中不创建临时 split 文件. """
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
        centered_batch_size=12,
        wrapper_loader=lambda **kwargs: wrapper,
    )
    assert runtime.cache_max_bytes == 500 * 1024**3
    task = ProductionTask("Find_0", "calibration", "1abc")
    inputs = runtime.full_map_input(task)
    batch = inputs.window_batch_builder([(0, 0, 0)])
    centered_batch = runtime.centered_batch_builder(task)(
        [CenteredRequest(
            stage1_model_name="Find_0",
            split="calibration",
            pdb_id="1abc",
            centered_role="F1_centered",
            centered_box_index=0,
            box_start_zyx=(0, 0, 0),
            source_tree_id=0,
            source_node_id=0,
            source_threshold_grid_index=16384,
        )]
    )

    assert batch["density_input"].shape == (1, 56, *SHAPE)
    assert centered_batch["density_input"].shape == (1, 56, *SHAPE)
    assert batch["atom_global_indices"].tolist() == [0]
    assert centered_batch["atom_label"].dtype == torch.bool
    assert centered_batch["atom_label"].tolist() == [False]
    assert bool(inputs.receptor_hardmask_full[0, 0, 0])
    assert grid_loads.count("exp.npz") == 1
    assert grid_loads.count("sim.npz") == 1
    occurrences = runtime.occurrence_voxels(task, SHAPE)
    assert occurrences[7].tolist() == [np.ravel_multi_index((5, 5, 5), SHAPE)]


def test_cli_calibration_then_cal_centered_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """贯通 F1 优先生产、同目录补充 CLG 和再次运行跳过。"""
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
    assert probability[0, 0, 0] == pytest.approx(0.5)
    assert probability[5, 5, 5] == pytest.approx(0.5)
    # 同一命令再次运行只读现有 marker, 不重新 forward. 
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
    assert threshold_payload["max_voxels"] == 2046

    assert main(
        _common_cli_arguments(
            "cal-produce-f1",
            data_root,
            output_root,
            checkpoint,
            config,
            pdb_list,
        )
    ) == 0
    assert is_role_complete(paths, "components")
    assert is_role_complete(paths, "F1_centered")
    assert not is_role_complete(paths, "CLG_centered")
    assert not paths.centered_npz("CLG_centered").exists()

    f1_bytes = paths.centered_npz("F1_centered").read_bytes()
    assert main(
        [
            *_common_cli_arguments(
                "produce-falpha",
                data_root,
                output_root,
                checkpoint,
                config,
                pdb_list,
            ),
            "--split",
            "calibration",
            "--alpha",
            "2/3",
        ]
    ) == 0
    assert is_role_complete(paths, "F_2_3_centered")
    assert paths.centered_npz("F1_centered").read_bytes() == f1_bytes

    li_output_root = tmp_path / "li_outputs"
    assert main(
        [
            *_common_cli_arguments(
                "produce-li-centered",
                data_root,
                li_output_root,
                checkpoint,
                config,
                pdb_list,
            ),
            "--split",
            "calibration",
            "--probability-output-root",
            str(output_root),
            "--min-voxels",
            "10",
        ]
    ) == 0
    li_paths = Stage1ArtifactPaths(
        li_output_root, "Find_0", "calibration", "1abc"
    )
    assert is_role_complete(li_paths, "Li_centered")
    assert not li_paths.forest_npz.exists()

    preserved = (
        paths.probability_npz,
        paths.probability_geometry_json,
        paths.forest_npz,
        paths.clg_npz,
        paths.overlap_npz,
        paths.component_summary_json,
        paths.centered_npz("F1_centered"),
        paths.role_complete_path("probability"),
        paths.role_complete_path("components"),
        paths.role_complete_path("F1_centered"),
    )
    before_clg = {path: path.read_bytes() for path in preserved}
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
    assert is_role_complete(paths, "CLG_centered")
    assert before_clg == {path: path.read_bytes() for path in preserved}

    clg_bytes = paths.centered_npz("CLG_centered").read_bytes()
    clg_marker_bytes = paths.role_complete_path("CLG_centered").read_bytes()
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
    assert paths.centered_npz("CLG_centered").read_bytes() == clg_bytes
    assert paths.role_complete_path("CLG_centered").read_bytes() == clg_marker_bytes

    for f1_command, clg_command, split in (
        ("val-produce-prob-f1", "val-produce-prob-f1-clg", "validation"),
        ("train-produce-prob-f1", "train-produce-prob-f1-clg", "train"),
    ):
        assert main(
            _common_cli_arguments(
                f1_command,
                data_root,
                output_root,
                checkpoint,
                config,
                pdb_list,
            )
        ) == 0
        split_paths = Stage1ArtifactPaths(output_root, "Find_0", split, "1abc")
        for role in ("probability", "components", "F1_centered"):
            assert is_role_complete(split_paths, role)
        assert not is_role_complete(split_paths, "CLG_centered")
        before_clg = {
            path: path.read_bytes()
            for path in (
                split_paths.probability_npz,
                split_paths.forest_npz,
                split_paths.centered_npz("F1_centered"),
            )
        }
        assert main(
            _common_cli_arguments(
                clg_command,
                data_root,
                output_root,
                checkpoint,
                config,
                pdb_list,
            )
        ) == 0
        assert is_role_complete(split_paths, "CLG_centered")
        assert before_clg == {path: path.read_bytes() for path in before_clg}


def test_freeze_cli_default_max_voxels_is_frozen_q95_value() -> None:
    """CLI 缺省值只能是全量 Q95=682×3.0 得到的正式 2046. """
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
    assert DEFAULT_MAX_VOXELS == 2046
    assert arguments.max_voxels == 2046


def test_centered_cli_default_batch_size_is_twelve() -> None:
    arguments = build_parser().parse_args(
        [
            "cal-produce-f1-clg",
            "--producer",
            "unet_c1",
            "--pdb-list",
            "ids.json",
            "--data-root",
            "ag",
            "--checkpoint",
            "BEST.ckpt",
            "--device",
            "cuda:0",
            "--output-root",
            "out",
        ]
    )
    assert arguments.centered_batch_size == 10


def test_inference_cli_default_cache_allows_five_hundred_gibibytes() -> None:
    """正式 CLI 缺省缓存上限为 500 GiB，且该值不会预先分配内存。"""
    arguments = build_parser().parse_args(
        [
            "cal-probability",
            "--producer",
            "Find_0",
            "--pdb-list",
            "ids.json",
            "--data-root",
            "ag",
            "--checkpoint",
            "BEST.ckpt",
            "--device",
            "cuda:0",
            "--output-root",
            "out",
        ]
    )

    assert arguments.cache_max_bytes == 500 * 1024**3


def test_new_cli_switches_are_explicit_and_default_to_compatibility() -> None:
    """生产续跑开关默认关闭，评估超限样本开关默认关闭。"""

    freeze = build_parser().parse_args(
        [
            "freeze-thresholds",
            "--producer",
            "Find_0",
            "--pdb-list",
            "ids.json",
            "--data-root",
            "ag",
            "--output-root",
            "out",
            "--evaluate-on-blob-exceed",
        ]
    )
    falpha = build_parser().parse_args(
        [
            "produce-falpha",
            "--split",
            "calibration",
            "--producer",
            "Find_0",
            "--pdb-list",
            "ids.json",
            "--data-root",
            "ag",
            "--checkpoint",
            "BEST.ckpt",
            "--device",
            "cpu",
            "--output-root",
            "out",
            "--alpha",
            "2/3",
        ]
    )

    assert freeze.evaluate_on_blob_exceed is True
    assert falpha.continue_on_blob_exceed is False
    assert falpha.alpha == "2/3"


def test_component_producer_activates_checkpoint_source_before_dataset_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """组件算法按需导入 Dataset 前，必须先激活 checkpoint 的唯一源码快照。"""
    activated = False

    def wrapper_provider(task):
        nonlocal activated
        del task
        activated = True
        return object()

    def component_factory(**kwargs):
        del kwargs

        def produce(task, paths):
            del task, paths
            assert activated

        return produce

    monkeypatch.setattr(
        "src.inference.cli.make_component_role_producer",
        component_factory,
    )
    runtime = SimpleNamespace(
        occurrence_voxels=lambda *args: {},
        wrapper_provider=wrapper_provider,
        centered_batch_builder=lambda task: None,
        centered_batch_size=12,
    )
    arguments = SimpleNamespace(
        max_split_events=1,
        max_merge_events=1,
        max_nodes_per_clg=32,
        f1_eligible_limit=200,
    )
    producers = _standard_role_producers(
        runtime,
        arguments,
        include_probability=False,
        include_clg=False,
    )

    producers["components"](ProductionTask("Find_0", "calibration", "1abc"), object())
    assert activated


def test_selected_role_loads_selection_reruns_and_publishes(tmp_path: Path) -> None:
    """Selected producer 恢复 source node, 以原阈值重跑并发布成功 refined blob. """
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
    wrapper = _ProbabilityWrapper(centered_probability=probability)
    producer = make_selected_refined_role_producer(
        wrapper_provider=lambda task: wrapper,
        batch_builder_provider=lambda task: lambda requests: {
            "density_input": np.zeros((len(requests), 1, *SHAPE), dtype=np.float32),
            "hardmask": np.zeros((len(requests), *SHAPE), dtype=np.bool_),
            "box_shape_zyx": np.asarray([SHAPE] * len(requests), dtype=np.int64),
            "voxel_size_world": np.ones((len(requests), 3), dtype=np.float32),
        },
        centered_batch_size=12,
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
    """occurrence 稀疏 mask 与 union 不一致时必须 fail-fast. """
    _write_ag_fixture(tmp_path)
    path = tmp_path / "density" / "1abc" / "ligand_area.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["union_mask"] = np.zeros((1, *SHAPE), dtype=np.bool_)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="并集"):
        AGOccurrenceVoxelLoader(tmp_path)("1abc", SHAPE)
