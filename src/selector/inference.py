"""生成 Selector scores.npz 并按冻结 tau_G 解码 selection.npz。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.artifacts import atomic_savez_compressed, load_npz_strict, validate_offsets
from src.component_lineage import ComponentForest, ComponentNode

from .calibration import calibrate_tau_g
from .dataset import SelectorDataset
from .structured.antichain_dp import build_candidate_tree_closure
from .structured.decode import decode_gated_antichain
from .train import move_sample_to_device
from .wrapper import SelectorWrapper, build_selector_wrapper_from_config


def _validate_scores_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """校验 scores.npz 的精确字段、dtype、offsets 和有限值。"""
    expected = {
        "CLG_id": np.int32,
        "CLG_logit": np.float32,
        "CLG_valid_probability": np.float32,
        "candidate_offsets": np.int64,
        "predicted_max_iou": np.float32,
        "selection_logit": np.float32,
    }
    if set(arrays) != set(expected):
        raise KeyError(f"scores.npz 字段集合不一致: {sorted(arrays)}")
    for name, dtype in expected.items():
        if np.asarray(arrays[name]).dtype != np.dtype(dtype):
            raise ValueError(f"scores.{name}.dtype 必须为 {np.dtype(dtype)}。")
    clg_count = int(np.asarray(arrays["CLG_id"]).size)
    candidate_count = int(np.asarray(arrays["predicted_max_iou"]).size)
    clg_fields = ("CLG_id", "CLG_logit", "CLG_valid_probability")
    if any(np.asarray(arrays[name]).shape != (clg_count,) for name in clg_fields):
        raise ValueError("scores 的三个 CLG value 表必须严格逐行对齐。")
    validate_offsets(np.asarray(arrays["candidate_offsets"]), candidate_count, "candidate_offsets")
    if np.asarray(arrays["candidate_offsets"]).shape != (clg_count + 1,):
        raise ValueError("scores.candidate_offsets 长度必须为 N_CLG+1。")
    if np.asarray(arrays["selection_logit"]).shape != (candidate_count,):
        raise ValueError("selection_logit 必须与 predicted_max_iou 逐 candidate 对齐。")
    for name in ("CLG_logit", "CLG_valid_probability", "predicted_max_iou", "selection_logit"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"scores.{name} 含 NaN/Inf。")
    for name in ("CLG_valid_probability", "predicted_max_iou"):
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise ValueError(f"scores.{name} 必须位于 [0,1]。")


def _validate_selection_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """校验 selection.npz 的精确字段、dtype 和 ragged offsets。"""
    expected = {
        "CLG_id": np.int32,
        "CLG_gate_pass": np.bool_,
        "selected_candidate_offsets": np.int64,
        "selected_candidate_index": np.int16,
    }
    if set(arrays) != set(expected):
        raise KeyError(f"selection.npz 字段集合不一致: {sorted(arrays)}")
    for name, dtype in expected.items():
        if np.asarray(arrays[name]).dtype != np.dtype(dtype):
            raise ValueError(f"selection.{name}.dtype 必须为 {np.dtype(dtype)}。")
    clg_count = int(np.asarray(arrays["CLG_id"]).size)
    if np.asarray(arrays["CLG_gate_pass"]).shape != (clg_count,):
        raise ValueError("CLG_gate_pass 必须与 CLG_id 对齐。")
    selected = np.asarray(arrays["selected_candidate_index"])
    offsets = np.asarray(arrays["selected_candidate_offsets"])
    validate_offsets(offsets, selected.size, "selected_candidate_offsets")
    if offsets.shape != (clg_count + 1,):
        raise ValueError("selected_candidate_offsets 长度必须为 N_CLG+1。")


def load_selector_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SelectorWrapper, dict[str, Any]]:
    """
    从 Selector BEST checkpoint 严格恢复完整 Wrapper。

    输入参数:
        - checkpoint_path: str | Path, Selector `checkpoints/BEST.ckpt`
        - device: torch.device, 推理设备

    输出:
        - wrapper: SelectorWrapper，strict 恢复并置 eval
        - checkpoint: dict[str,Any]，含 resolved config/source dimensions 等运行身份
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    wrapper = build_selector_wrapper_from_config(
        checkpoint["resolved_config"],
        checkpoint["source_dimensions"],
    )
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper.to(device)
    wrapper.eval()
    return wrapper, checkpoint


def produce_scores(
    checkpoint_path: str | Path,
    input_clg_list_path: str | Path,
    stage1_outputs_root: str | Path,
    upstream_root: str | Path,
    selector_run_dir: str | Path,
    split: str,
    device_name: str,
) -> tuple[Path, ...]:
    """
    按来源 clg.npz 顺序为一个 split 的每个 PDB 发布 scores.npz。

    输入参数:
        - checkpoint_path: str | Path, validation total loss 选出的 Selector BEST
        - input_clg_list_path: str | Path, 当前 run 冻结输入清单
        - stage1_outputs_root: str | Path, Stage1 producer 输出根
        - upstream_root: str | Path, A–G 上游数据根
        - selector_run_dir: str | Path, 当前 selector_run_dir
        - split: str, train/validation/calibration 等单一 split
        - device_name: str, 显式 cpu 或 cuda

    输出:
        - score_paths: tuple[Path,...]，按 pdb_id 排序的正式 scores.npz 路径
    """
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device_name 只允许 cpu 或 cuda。")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前进程没有可用 GPU。")
    device = torch.device(device_name)
    wrapper, checkpoint = load_selector_checkpoint(checkpoint_path, device)
    config = checkpoint["resolved_config"]
    dataset = SelectorDataset(
        input_clg_list_path=input_clg_list_path,
        stage1_outputs_root=stage1_outputs_root,
        upstream_root=upstream_root,
        split=split,
        lambda_count=float(config["data"]["lambda_count"]),
        require_oracle=False,
        density_clip_percentile=tuple(float(value) for value in config["data"]["density_clip_percentile"]),
        pdb_cache_size=int(config["data"]["pdb_cache_size"]),
    )
    by_pdb: dict[str, dict[int, dict[str, np.ndarray | float]]] = defaultdict(dict)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = move_sample_to_device(dataset[index], device)
            output = wrapper.model(sample)
            by_pdb[str(sample["pdb_id"])][int(sample["CLG_id"])] = {
                "CLG_logit": float(output["CLG_logit"].detach().cpu()),
                "CLG_valid_probability": float(output["CLG_valid_probability"].detach().cpu()),
                "predicted_max_iou": output["predicted_max_iou"].detach().cpu().numpy().astype(np.float32),
                "selection_logit": output["selection_logit"].detach().cpu().numpy().astype(np.float32),
            }

    output_paths: list[Path] = []
    stage1_root = Path(stage1_outputs_root)
    output_root = Path(selector_run_dir)
    stage1_model_name = str(config["stage1_model_name"])
    for pdb_id in sorted(by_pdb):
        source_path = stage1_root / stage1_model_name / split / pdb_id / "components" / "clg.npz"
        with np.load(source_path, allow_pickle=False) as source:
            source_clg_id = np.asarray(source["CLG_id"], dtype=np.int32)
            source_offsets = np.asarray(source["candidate_offsets"], dtype=np.int64)
        result_by_id = by_pdb[pdb_id]
        if set(result_by_id) != set(source_clg_id.tolist()):
            raise ValueError(f"scores 必须覆盖来源 PDB 的全部 CLG: split={split}, pdb_id={pdb_id}")
        clg_logits: list[float] = []
        clg_probabilities: list[float] = []
        predicted_values: list[np.ndarray] = []
        selection_values: list[np.ndarray] = []
        for row, clg_id in enumerate(source_clg_id.tolist()):
            result = result_by_id[int(clg_id)]
            expected_count = int(source_offsets[row + 1] - source_offsets[row])
            if np.asarray(result["predicted_max_iou"]).shape != (expected_count,):
                raise ValueError("predicted_max_iou 行数与来源 candidate_offsets 不一致。")
            clg_logits.append(float(result["CLG_logit"]))
            clg_probabilities.append(float(result["CLG_valid_probability"]))
            predicted_values.append(np.asarray(result["predicted_max_iou"], dtype=np.float32))
            selection_values.append(np.asarray(result["selection_logit"], dtype=np.float32))
        score_path = output_root / split / pdb_id / "scores.npz"
        atomic_savez_compressed(
            score_path,
            {
                "CLG_id": source_clg_id,
                "CLG_logit": np.asarray(clg_logits, dtype=np.float32),
                "CLG_valid_probability": np.asarray(clg_probabilities, dtype=np.float32),
                "candidate_offsets": source_offsets,
                "predicted_max_iou": np.concatenate(predicted_values).astype(np.float32, copy=False),
                "selection_logit": np.concatenate(selection_values).astype(np.float32, copy=False),
            },
            validator=_validate_scores_arrays,
        )
        output_paths.append(score_path)
    return tuple(output_paths)


def produce_selection_for_pdb(
    scores_path: str | Path,
    forest_path: str | Path,
    clg_path: str | Path,
    selection_path: str | Path,
    tau_g: float,
    lambda_count: float,
) -> Path:
    """
    对一个 PDB 的全部 CLG 执行门控和精确非空 MAP，发布 selection.npz。

    输入参数:
        - scores_path: str | Path, 当前 PDB 的 scores.npz
        - forest_path: str | Path, 来源 forest.npz
        - clg_path: str | Path, 来源 clg.npz
        - selection_path: str | Path, 正式 selection.npz 输出路径
        - tau_g: float, calibration 冻结的 CLG gate 阈值
        - lambda_count: float, 预测反链计数惩罚

    输出:
        - output_path: Path, 原子发布后的 selection.npz
    """
    with np.load(scores_path, allow_pickle=False) as source:
        scores = {name: np.asarray(source[name]).copy() for name in source.files}
    with np.load(forest_path, allow_pickle=False) as source:
        forest = {name: np.asarray(source[name]).copy() for name in source.files}
    with np.load(clg_path, allow_pickle=False) as source:
        clg = {name: np.asarray(source[name]).copy() for name in source.files}
    if not np.array_equal(scores["CLG_id"], clg["CLG_id"]):
        raise ValueError("scores.CLG_id 必须与来源 clg.npz 完全同序。")
    if not np.array_equal(scores["candidate_offsets"], clg["candidate_offsets"]):
        raise ValueError("scores.candidate_offsets 必须逐元素复制来源 clg.npz。")

    gate_pass_values: list[bool] = []
    selected_offsets = [0]
    selected_local_values: list[int] = []
    forest_tree_id = np.asarray(forest["tree_id"], dtype=np.int64)
    forest_node_id = np.asarray(forest["node_id"], dtype=np.int64)
    forest_parent_id = np.asarray(forest["parent_node_id"], dtype=np.int64)
    candidate_offsets = np.asarray(clg["candidate_offsets"], dtype=np.int64)
    for row in range(clg["CLG_id"].shape[0]):
        begin, end = int(candidate_offsets[row]), int(candidate_offsets[row + 1])
        candidate_node_id = np.asarray(clg["candidate_node_id"][begin:end], dtype=np.int64)
        tree_id = int(clg["tree_id"][row])
        tree_rows = np.flatnonzero(forest_tree_id == tree_id)
        closure = build_candidate_tree_closure(
            node_id=forest_node_id[tree_rows].tolist(),
            parent_node_id=forest_parent_id[tree_rows].tolist(),
            candidate_node_id=candidate_node_id.tolist(),
        )
        selection_logit = torch.from_numpy(np.asarray(scores["selection_logit"][begin:end], dtype=np.float32))
        gate_pass, selected = decode_gated_antichain(
            clg_valid_probability=torch.tensor(float(scores["CLG_valid_probability"][row])),
            tau_g=float(tau_g),
            selection_logit=selection_logit,
            parent_index=closure.parent_index,
            candidate_index_by_node=closure.candidate_index_by_node,
            lambda_count=float(lambda_count),
        )
        selected_values = sorted(int(value) for value in selected.cpu().tolist())
        gate_pass_values.append(gate_pass)
        selected_local_values.extend(selected_values)
        selected_offsets.append(len(selected_local_values))

    output_path = Path(selection_path)
    atomic_savez_compressed(
        output_path,
        {
            "CLG_id": np.asarray(clg["CLG_id"], dtype=np.int32),
            "CLG_gate_pass": np.asarray(gate_pass_values, dtype=np.bool_),
            "selected_candidate_offsets": np.asarray(selected_offsets, dtype=np.int64),
            "selected_candidate_index": np.asarray(selected_local_values, dtype=np.int16),
        },
        validator=_validate_selection_arrays,
    )
    return output_path


def load_selected_nodes_for_pdb(
    selection_path: str | Path,
    forest_path: str | Path,
    clg_path: str | Path,
) -> tuple[ComponentNode, ...]:
    """
    把 selection 的 CLG 局部 candidate 下标恢复为 centered runner 所需 forest nodes。

    输入参数:
        - selection_path: str | Path, 当前 PDB 冻结 tau_G 后的 selection.npz
        - forest_path: str | Path, 同 PDB 来源 forest.npz
        - clg_path: str | Path, 同 PDB 来源 clg.npz

    输出:
        - selected_nodes: tuple[ComponentNode,...], 按 CLG/来源 candidate 顺序，供
          `produce_selected_refined_entries(selected_nodes=...)` 直接消费
    """
    selection = load_npz_strict(selection_path)
    forest_arrays = load_npz_strict(forest_path)
    clg = load_npz_strict(clg_path)
    if not np.array_equal(selection["CLG_id"], clg["CLG_id"]):
        raise ValueError("selection.CLG_id 必须与来源 clg.npz 完全同序。")
    selected_offsets = np.asarray(selection["selected_candidate_offsets"], dtype=np.int64)
    selected_local = np.asarray(selection["selected_candidate_index"], dtype=np.int64)
    gate_pass = np.asarray(selection["CLG_gate_pass"], dtype=np.bool_)
    candidate_offsets = np.asarray(clg["candidate_offsets"], dtype=np.int64)
    validate_offsets(selected_offsets, selected_local.size, "selected_candidate_offsets")
    if selected_offsets.shape != candidate_offsets.shape or gate_pass.shape != np.asarray(clg["CLG_id"]).shape:
        raise ValueError("selected_candidate_offsets 必须按 CLG 完整切分 selected_candidate_index。")

    forest = ComponentForest.from_arrays(forest_arrays)
    nodes: list[ComponentNode] = []
    identities: set[tuple[int, int]] = set()
    for clg_row in range(np.asarray(clg["CLG_id"]).size):
        selected_begin, selected_end = int(selected_offsets[clg_row]), int(selected_offsets[clg_row + 1])
        candidate_begin, candidate_end = int(candidate_offsets[clg_row]), int(candidate_offsets[clg_row + 1])
        local_values = selected_local[selected_begin:selected_end]
        if not bool(gate_pass[clg_row]) and local_values.size:
            raise ValueError("CLG_gate_pass=False 的 CLG 必须对应空 selection 段。")
        if bool(gate_pass[clg_row]) and not local_values.size:
            raise ValueError("CLG_gate_pass=True 的 CLG 必须对应非空 MAP selection。")
        if local_values.size and (
            np.unique(local_values).size != local_values.size
            or not np.array_equal(local_values, np.sort(local_values))
        ):
            raise ValueError("selected_candidate_index 必须按来源顺序严格递增且无重复。")
        for local_index in local_values.tolist():
            if local_index < 0 or candidate_begin + local_index >= candidate_end:
                raise ValueError("selected_candidate_index 越过所属 CLG candidate 段。")
            absolute_row = candidate_begin + int(local_index)
            identity = (int(clg["tree_id"][clg_row]), int(clg["candidate_node_id"][absolute_row]))
            if identity in identities:
                raise ValueError(f"selection 重复引用同一 forest node: {identity}")
            identities.add(identity)
            nodes.append(forest.node(*identity))
    return tuple(nodes)


def main(argv: Sequence[str] | None = None) -> int:
    """
    提供 scores、calibrate 与 selection 三个显式 Selector 推理子命令。

    输入参数:
        - argv: Sequence[str] | None, CLI 参数；None 表示读取 sys.argv

    输出:
        - exit_code: int, 成功为 0
    """
    parser = argparse.ArgumentParser(description="AdaLigand Stage1 Selector 推理")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scores_parser = subparsers.add_parser("scores")
    scores_parser.add_argument("--checkpoint", required=True)
    scores_parser.add_argument("--input-clg-list", required=True)
    scores_parser.add_argument("--stage1-outputs-root", required=True)
    scores_parser.add_argument("--upstream-root", required=True)
    scores_parser.add_argument("--selector-run-dir", required=True)
    scores_parser.add_argument("--split", required=True)
    scores_parser.add_argument("--device", required=True, choices=("cpu", "cuda"))

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--selector-run-dir", required=True)
    calibrate_parser.add_argument("--stage1-outputs-root", required=True)
    calibrate_parser.add_argument("--input-clg-list", required=True)
    calibrate_parser.add_argument(
        "--stage1-model-name",
        required=True,
        choices=("Find_0", "Find_1", "unet_c1"),
    )
    calibrate_parser.add_argument("--lambda-count", required=True, type=float)
    calibrate_parser.add_argument("--split", default="calibration")

    selection_parser = subparsers.add_parser("selection")
    selection_parser.add_argument("--scores", required=True)
    selection_parser.add_argument("--forest", required=True)
    selection_parser.add_argument("--clg", required=True)
    selection_parser.add_argument("--output", required=True)
    selection_parser.add_argument("--calibration", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "scores":
        paths = produce_scores(
            checkpoint_path=arguments.checkpoint,
            input_clg_list_path=arguments.input_clg_list,
            stage1_outputs_root=arguments.stage1_outputs_root,
            upstream_root=arguments.upstream_root,
            selector_run_dir=arguments.selector_run_dir,
            split=arguments.split,
            device_name=arguments.device,
        )
        print(f"published_scores={len(paths)}")
    elif arguments.command == "calibrate":
        payload = calibrate_tau_g(
            selector_run_dir=arguments.selector_run_dir,
            stage1_outputs_root=arguments.stage1_outputs_root,
            input_clg_list_path=arguments.input_clg_list,
            stage1_model_name=arguments.stage1_model_name,
            lambda_count=arguments.lambda_count,
            split=arguments.split,
        )
        print(f"tau_G={payload['tau_G']}")
    else:
        calibration = json.loads(Path(arguments.calibration).read_text(encoding="utf-8"))
        output = produce_selection_for_pdb(
            scores_path=arguments.scores,
            forest_path=arguments.forest,
            clg_path=arguments.clg,
            selection_path=arguments.output,
            tau_g=float(calibration["tau_G"]),
            lambda_count=float(calibration["lambda_count"]),
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
