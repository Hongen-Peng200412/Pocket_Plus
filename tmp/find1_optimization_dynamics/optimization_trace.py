"""在真实 Stage1 batch 上记录 Find_1 体素参数的优化动力学 JSON."""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import os
import secrets
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


VOXEL_PARAMETER_PREFIXES = (
    "backbone.embed_head.voxel_input_proj.",
    "backbone.embed_head.voxel_out_proj_with_offset.",
    "backbone.voxel_backbone.",
)
CONTROLLED_RECYCLE_SEQUENCE = (1, 2, 3, 2, 1, 3, 1, 2)
POINT_LOSS_NAMES = frozenset(("atom", "pseudo"))
PROCESS_NONCE = secrets.token_hex(16)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--role", choices=("trunk", "full"), required=True)
    parser.add_argument(
        "--track",
        choices=("controlled", "natural", "replay"),
        required=True,
    )
    parser.add_argument("--recycle-sequence-from", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accumulate-steps", type=int, default=8)
    parser.add_argument("--optimizer-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(project_root: Path) -> str:
    """计算目标项目 ``src`` 与 ``configs`` 的确定性内容摘要."""

    digest = hashlib.sha256()
    source_paths = []
    for relative_root in (Path("src"), Path("configs")):
        root = project_root / relative_root
        for pattern in ("*.py", "*.yaml", "*.yml"):
            source_paths.extend(root.rglob(pattern))
    for path in sorted(set(source_paths)):
        relative_path = path.relative_to(project_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _json_value(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sample_positions(numel: int) -> tuple[int, ...]:
    """用整数运算生成最多十六个首尾覆盖的合法位置."""

    sample_count = min(16, numel)
    if sample_count == 0:
        return ()
    if sample_count == 1:
        return (0,)
    last_position = numel - 1
    return tuple(
        index * last_position // (sample_count - 1)
        for index in range(sample_count)
    )


def tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """用原始字节摘要、有限性和统计量记录一个完整张量."""

    contiguous = tensor.detach().contiguous()
    values = contiguous.float().reshape(-1)
    raw_bytes = contiguous.view(torch.uint8).cpu().numpy().tobytes()
    sample_positions = _sample_positions(int(values.numel()))
    if sample_positions:
        indices = torch.tensor(
            sample_positions,
            dtype=torch.long,
            device=values.device,
        )
        samples = values.index_select(0, indices).cpu().tolist()
    else:
        samples = []
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "all_finite": bool(torch.isfinite(values).all().item()),
        "l2": float(torch.linalg.vector_norm(values).item()),
        "mean": float(values.mean().item()) if values.numel() else 0.0,
        "max_abs": float(values.abs().max().item()) if values.numel() else 0.0,
        "samples": samples,
    }


def _parameter_signatures(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    source: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    signatures = {}
    for name, parameter in named_parameters:
        if source == "parameter":
            tensor = parameter
        elif source == "gradient":
            tensor = parameter.grad
        else:
            if optimizer is None:
                raise ValueError("optimizer state 签名需要 optimizer。")
            tensor = optimizer.state[parameter].get(source)
        signatures[name] = None if tensor is None else tensor_signature(tensor)
    return signatures


def _optimizer_state_steps(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, int | None]:
    """逐参数记录 AdamW step；未参与本次更新的参数记为 ``None``."""

    return {
        name: (
            int(optimizer.state[parameter]["step"].item())
            if "step" in optimizer.state[parameter]
            else None
        )
        for name, parameter in named_parameters
    }


def _snapshot_parameters(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    """把 optimizer step 前的体素参数复制到 CPU."""

    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in named_parameters
    }


def _parameter_update_signatures(
    before_step: Mapping[str, torch.Tensor],
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    """记录每个体素参数在一个 optimizer step 中的实际增量."""

    return {
        name: tensor_signature(
            parameter.detach().float().cpu() - before_step[name]
        )
        for name, parameter in named_parameters
    }


def _group_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def _gradient_tuple_norm(
    gradients: Sequence[torch.Tensor | None],
) -> float:
    norms = [
        torch.linalg.vector_norm(gradient.detach().float())
        for gradient in gradients
        if gradient is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _compose_config(project_root: Path, experiment: str) -> Any:
    with initialize_config_dir(
        config_dir=str(project_root / "configs"),
        version_base=None,
    ):
        return compose(config_name="base", overrides=[f"+experiment={experiment}"])


def _build_dataset(config: Any) -> Any:
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.dataset, resolve=True)
    )
    dataset_config.enable_random_rotation = False
    dataset = instantiate(
        dataset_config,
        split_file=dataset_config.split_train,
        all_data_path=dataset_config.all_data_path,
        mode="train",
    )
    dataset.set_epoch(0)
    return dataset


def _build_wrapper(config: Any, role: str) -> torch.nn.Module:
    arguments = {
        "optimizer": config.train.optimizer,
        "scheduler": config.train.scheduler,
        "compile": False,
    }
    if role == "full":
        arguments["gradient_clip_mode"] = config.train.gradient_clip_mode
    return instantiate(config.model, **arguments)


def _materialize_batch(dataset: Any, positions: Sequence[int]) -> dict[str, Any]:
    samples = [dataset[int(position)] for position in positions]
    return dataset.collate_fn(samples)


def _initialize_input_channels(wrapper: torch.nn.Module, batch: Mapping[str, Any]) -> int:
    input_tensor = batch.get("density_input", batch.get("voxel_grid"))
    if not torch.is_tensor(input_tensor) or input_tensor.ndim != 5:
        raise ValueError("Stage1 collate batch 必须包含五维 density_input 或 voxel_grid。")
    input_channels = int(input_tensor.shape[1])
    wrapper.backbone.set_input_channels(input_channels)
    return input_channels


def _load_shared_checkpoint(
    wrapper: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["state_dict"]
    target_state = wrapper.state_dict()
    shared_state = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    wrapper.load_state_dict(shared_state, strict=False)
    missing_voxel_state = sorted(
        name
        for name in target_state
        if name.startswith(VOXEL_PARAMETER_PREFIXES) and name not in shared_state
    )
    if missing_voxel_state:
        raise RuntimeError(
            f"checkpoint 缺少 {len(missing_voxel_state)} 个体素状态："
            f"{missing_voxel_state[:5]}"
        )
    return {
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "shared_state_count": len(shared_state),
    }


def _apply_frozen_patterns(wrapper: torch.nn.Module, config: Any) -> None:
    frozen_config = config.get("frozen_module")
    if frozen_config is None:
        return
    patterns = tuple(str(pattern) for pattern in frozen_config.patterns)
    patterns += tuple(
        str(pattern) for pattern in (frozen_config.get("optional_patterns") or ())
    )
    for name, parameter in wrapper.named_parameters():
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            parameter.requires_grad = False


def _configure_candidate_runtime(
    wrapper: torch.nn.Module,
    *,
    optimizer_step: int,
    warmup_steps: int,
) -> None:
    backbone = wrapper.backbone
    if hasattr(backbone, "set_sparse_candidate_runtime"):
        backbone.set_sparse_candidate_runtime(
            global_step=optimizer_step,
            candidate_warmup_steps=warmup_steps,
            allow_warmup_fixed_topk=True,
        )
    if hasattr(backbone, "set_sparse_candidate_thresholds"):
        backbone.set_sparse_candidate_thresholds(None, None)


def _loss_values(loss_terms: Sequence[Any]) -> dict[str, float]:
    return {
        str(term.name): float(term.value.detach().float().item())
        for term in loss_terms
    }


def _voxel_output_signatures(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """记录两套模型都应产生的最终体素输出."""

    names = (
        "voxel_logits_aux",
        "voxel_logits_ligand",
        "voxel_logits_protein",
        "voxel_logits_nucleic",
        "voxel_logits_distance",
        "voxel_recycle_out",
    )
    return {
        name: tensor_signature(outputs[name])
        for name in names
        if torch.is_tensor(outputs.get(name))
    }


def _probe_point_to_voxel_gradient(
    loss_terms: Sequence[Any],
    voxel_parameters: Sequence[torch.nn.Parameter],
) -> dict[str, Any]:
    """分别测量每个点监督对体素参数产生的梯度范数."""

    selected_terms = [term for term in loss_terms if term.name in POINT_LOSS_NAMES]
    names = [str(term.name) for term in selected_terms]
    if len(names) != len(set(names)):
        raise RuntimeError(f"点损失名称重复：{names}")
    per_loss_gradient_norm = {}
    for term in selected_terms:
        if term.value.requires_grad:
            gradients = torch.autograd.grad(
                term.value * float(term.weight),
                voxel_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            gradient_norm = _gradient_tuple_norm(gradients)
        else:
            gradient_norm = 0.0
        per_loss_gradient_norm[str(term.name)] = gradient_norm
    return {
        "loss_term_names": sorted(names),
        "per_loss_voxel_gradient_norm": per_loss_gradient_norm,
    }


def _optimizer_step_trace(
    *,
    role: str,
    wrapper: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    voxel_named: Sequence[tuple[str, torch.nn.Parameter]],
    other_named: Sequence[tuple[str, torch.nn.Parameter]],
    all_trainable: Sequence[torch.nn.Parameter],
    optimizer_step: int,
    last_microbatch: int,
) -> dict[str, Any]:
    voxel_parameters = tuple(parameter for _, parameter in voxel_named)
    other_parameters = tuple(parameter for _, parameter in other_named)
    before_step = _snapshot_parameters(voxel_named)
    other_before_step = _snapshot_parameters(other_named)
    trace = {
        "optimizer_step": optimizer_step,
        "last_microbatch": last_microbatch,
        "lr_before_step": float(optimizer.param_groups[0]["lr"]),
        "pre_clip_group_norm": _group_gradient_norm(voxel_parameters),
        "other_pre_clip_group_norm": _group_gradient_norm(other_parameters),
        "parameters_before_step": _parameter_signatures(
            voxel_named,
            source="parameter",
        ),
        "pre_clip_gradients": _parameter_signatures(
            voxel_named,
            source="gradient",
        ),
        "other_parameters_before_step": _parameter_signatures(
            other_named,
            source="parameter",
        ),
        "other_pre_clip_gradients": _parameter_signatures(
            other_named,
            source="gradient",
        ),
    }
    if role == "full":
        wrapper.configure_gradient_clipping(
            optimizer,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )
    else:
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=0.5)
    trace["post_clip_group_norm"] = _group_gradient_norm(voxel_parameters)
    trace["other_post_clip_group_norm"] = _group_gradient_norm(other_parameters)
    pre_clip_norm = float(trace["pre_clip_group_norm"])
    post_clip_norm = float(trace["post_clip_group_norm"])
    trace["voxel_clip_coefficient"] = (
        post_clip_norm / pre_clip_norm if pre_clip_norm > 0.0 else 1.0
    )
    trace["post_clip_gradients"] = _parameter_signatures(
        voxel_named,
        source="gradient",
    )
    trace["other_post_clip_gradients"] = _parameter_signatures(
        other_named,
        source="gradient",
    )

    optimizer.step()
    scheduler.step()
    trace["lr_after_step"] = float(optimizer.param_groups[0]["lr"])
    trace["parameters_after_step"] = _parameter_signatures(
        voxel_named,
        source="parameter",
    )
    trace["parameter_updates"] = _parameter_update_signatures(
        before_step,
        voxel_named,
    )
    trace["other_parameters_after_step"] = _parameter_signatures(
        other_named,
        source="parameter",
    )
    trace["other_parameter_updates"] = _parameter_update_signatures(
        other_before_step,
        other_named,
    )
    trace["exp_avg_after_step"] = _parameter_signatures(
        voxel_named,
        source="exp_avg",
        optimizer=optimizer,
    )
    trace["exp_avg_sq_after_step"] = _parameter_signatures(
        voxel_named,
        source="exp_avg_sq",
        optimizer=optimizer,
    )
    trace["other_exp_avg_after_step"] = _parameter_signatures(
        other_named,
        source="exp_avg",
        optimizer=optimizer,
    )
    trace["other_exp_avg_sq_after_step"] = _parameter_signatures(
        other_named,
        source="exp_avg_sq",
        optimizer=optimizer,
    )
    trace["optimizer_state_step"] = _optimizer_state_steps(
        voxel_named,
        optimizer,
    )
    trace["other_optimizer_state_step"] = _optimizer_state_steps(
        other_named,
        optimizer,
    )
    optimizer.zero_grad(set_to_none=True)
    return trace


def run_trace(arguments: argparse.Namespace) -> dict[str, Any]:
    """运行一条 AUTO trunk 或完整 Find_1 优化轨迹."""

    project_root = arguments.project_root.resolve()
    sys.path.insert(0, str(project_root))
    config = _compose_config(project_root, arguments.experiment)
    resolved_config = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    dataset = _build_dataset(config)
    microbatch_count = arguments.accumulate_steps * arguments.optimizer_steps
    replay_sequence = None
    replay_source_sha256 = None
    if arguments.track == "replay":
        if arguments.recycle_sequence_from is None:
            raise ValueError("replay 轨迹要求 --recycle-sequence-from。")
        replay_source = arguments.recycle_sequence_from.resolve()
        replay_payload = json.loads(replay_source.read_text(encoding="utf-8"))
        replay_sequence = [
            int(record["recycle_passes"])
            for record in replay_payload["microbatches"]
        ]
        if len(replay_sequence) < microbatch_count:
            raise ValueError(
                f"replay 序列只有 {len(replay_sequence)} 项，少于 {microbatch_count}。"
            )
        replay_source_sha256 = _sha256_file(replay_source)
    elif arguments.recycle_sequence_from is not None:
        raise ValueError("--recycle-sequence-from 只允许用于 replay 轨迹。")
    rng = np.random.default_rng(arguments.seed)
    request_positions = rng.choice(
        len(dataset),
        size=microbatch_count * arguments.batch_size,
        replace=False,
    ).tolist()
    first_positions = request_positions[: arguments.batch_size]
    first_batch = _materialize_batch(dataset, first_positions)

    torch.manual_seed(arguments.seed)
    wrapper = _build_wrapper(config, arguments.role)
    input_channels = _initialize_input_channels(wrapper, first_batch)
    checkpoint_record = _load_shared_checkpoint(wrapper, arguments.checkpoint)
    _apply_frozen_patterns(wrapper, config)

    named_trainable = [
        (name, parameter)
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    ]
    voxel_named = [
        (name, parameter)
        for name, parameter in named_trainable
        if name.startswith(VOXEL_PARAMETER_PREFIXES)
    ]
    other_named = [
        (name, parameter)
        for name, parameter in named_trainable
        if not name.startswith(VOXEL_PARAMETER_PREFIXES)
    ]
    if arguments.role == "trunk" and other_named:
        raise RuntimeError("AUTO trunk 出现了体素边界之外的可训练参数。")
    if arguments.role == "full" and not other_named:
        raise RuntimeError("完整 Find_1 缺少体素边界之外的可训练参数。")

    device = torch.device("cuda")
    wrapper.to(device)
    wrapper.train()
    all_trainable = tuple(parameter for _, parameter in named_trainable)
    optimizer = torch.optim.AdamW(
        all_trainable,
        lr=5.0e-5,
        weight_decay=0.01,
    )
    warmup_steps = 10
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.33,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    optimizer.zero_grad(set_to_none=True)
    if arguments.track == "natural":
        torch.manual_seed(arguments.seed + 100_000)
        torch.cuda.manual_seed_all(arguments.seed + 100_000)

    microbatch_records = []
    optimizer_step_records = []
    point_gradient_probe = None
    for microbatch_index in range(microbatch_count):
        begin = microbatch_index * arguments.batch_size
        positions = request_positions[begin : begin + arguments.batch_size]
        cpu_batch = first_batch if microbatch_index == 0 else _materialize_batch(
            dataset,
            positions,
        )
        batch = _move_to_device(cpu_batch, device)
        if microbatch_index == 0:
            first_batch = None
        optimizer_step = microbatch_index // arguments.accumulate_steps
        _configure_candidate_runtime(
            wrapper,
            optimizer_step=optimizer_step,
            warmup_steps=warmup_steps,
        )
        if arguments.track == "controlled":
            recycle_steps = CONTROLLED_RECYCLE_SEQUENCE[
                microbatch_index % len(CONTROLLED_RECYCLE_SEQUENCE)
            ]
            wrapper.backbone.max_recycles = recycle_steps
            wrapper.backbone.randomize_recycles = False
            torch.manual_seed(arguments.seed + 100_000 + microbatch_index)
            torch.cuda.manual_seed_all(arguments.seed + 100_000 + microbatch_index)
        elif arguments.track == "replay":
            recycle_steps = replay_sequence[microbatch_index]
            wrapper.backbone.max_recycles = recycle_steps
            wrapper.backbone.randomize_recycles = False
            torch.manual_seed(arguments.seed + 100_000 + microbatch_index)
            torch.cuda.manual_seed_all(arguments.seed + 100_000 + microbatch_index)
        else:
            wrapper.backbone.max_recycles = 3
            wrapper.backbone.randomize_recycles = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = wrapper(batch)
            total_loss, loss_terms, _ = wrapper._compute_total_loss(outputs, batch)
            scaled_loss = total_loss / arguments.accumulate_steps
        if microbatch_index == 0:
            point_gradient_probe = _probe_point_to_voxel_gradient(
                loss_terms,
                tuple(parameter for _, parameter in voxel_named),
            )
        scaled_loss.backward()
        voxel_parameters = tuple(parameter for _, parameter in voxel_named)
        microbatch_records.append(
            {
                "microbatch": microbatch_index,
                "optimizer_step": optimizer_step,
                "request_positions": positions,
                "requests": [
                    _json_value(dataset.request_source[int(position)])
                    for position in positions
                ],
                "recycle_passes": int(outputs["recycle_passes_used"]),
                "total_loss": float(total_loss.detach().float().item()),
                "loss_terms": _loss_values(loss_terms),
                "voxel_outputs": _voxel_output_signatures(outputs),
                "accumulated_voxel_gradient_norm": _group_gradient_norm(
                    voxel_parameters
                ),
                "accumulated_voxel_gradients": _parameter_signatures(
                    voxel_named,
                    source="gradient",
                ),
            }
        )
        del batch, cpu_batch, outputs, total_loss, scaled_loss, loss_terms

        if (microbatch_index + 1) % arguments.accumulate_steps == 0:
            optimizer_step_records.append(
                _optimizer_step_trace(
                    role=arguments.role,
                    wrapper=wrapper,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    voxel_named=voxel_named,
                    other_named=other_named,
                    all_trainable=all_trainable,
                    optimizer_step=optimizer_step,
                    last_microbatch=microbatch_index,
                )
            )

    return {
        "schema_version": 2,
        "role": arguments.role,
        "track": arguments.track,
        "source_identity": arguments.source_identity,
        "process_nonce": PROCESS_NONCE,
        "process_pid": os.getpid(),
        "project_root": str(project_root),
        "project_source_sha256": _source_tree_sha256(project_root),
        "experiment": arguments.experiment,
        "resolved_config_sha256": hashlib.sha256(
            resolved_config.encode("utf-8")
        ).hexdigest(),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": _sha256_file(arguments.checkpoint),
        "batch_size": arguments.batch_size,
        "accumulate_steps": arguments.accumulate_steps,
        "optimizer_steps": arguments.optimizer_steps,
        "seed": arguments.seed,
        "input_channels": input_channels,
        "trainable_parameter_tensors": len(named_trainable),
        "voxel_parameter_tensors": len(voxel_named),
        "other_parameter_tensors": len(other_named),
        "voxel_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(name for name, _ in voxel_named)).encode("utf-8")
        ).hexdigest(),
        "point_to_voxel_gradient_probe": point_gradient_probe,
        "recycle_sequence_source_sha256": replay_source_sha256,
        "recycle_sequence": [
            int(record["recycle_passes"])
            for record in microbatch_records
        ],
        **checkpoint_record,
        "microbatches": microbatch_records,
        "optimizer_step_records": optimizer_step_records,
    }


def main() -> None:
    arguments = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("optimization trace 需要 CUDA GPU。")
    if arguments.batch_size <= 0 or arguments.accumulate_steps <= 0:
        raise ValueError("batch-size 与 accumulate-steps 必须为正数。")
    if arguments.optimizer_steps <= 0:
        raise ValueError("optimizer-steps 必须为正数。")
    result = run_trace(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
