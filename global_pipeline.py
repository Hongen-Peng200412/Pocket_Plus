from __future__ import annotations

"""
单实验训练与评估总控入口。

服务器典型用法:
    python /home/penghongen/My_Project/Pocket_Plus/global_pipeline.py \
        --train-config /home/penghongen/My_Project/Pocket_Plus/configs/experiment/emb_unet_abl0.yaml \
        --val-config-standard /home/penghongen/My_Project/Pocket_Plus/configs/infer_or_eval/Abl/emb_unet0_stardard_40.yaml \
        --test-config-standard /home/penghongen/My_Project/Pocket_Plus/configs/infer_or_eval/Abl/emb_unet0_stardard.yaml \
        --val-config-strict /home/penghongen/My_Project/Pocket_Plus/configs/infer_or_eval/Abl/emb_unet0_strict_40.yaml \
        --test-config-strict /home/penghongen/My_Project/Pocket_Plus/configs/infer_or_eval/Abl/emb_unet0_strict.yaml \
        --vis-enable false

本脚本不申请 Slurm 资源、不管理 try_lock，只适合写入已有 allocation 的 run_cmd_<JOBID>.sh。
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment"
DEFAULT_FEEDBACK_ROOT = PROJECT_PARENT / "feedback_plus"


@dataclass(frozen=True)
class TrainRunSpec:
    """
    训练运行目录推导所需的稳定元信息。

    输入参数:
        - config_path: Path, configs/experiment 下的训练配置路径
        - experiment_name: str, Hydra experiment group 名, 即配置文件 stem
        - tag: str, ExperimentManager 生成 run_dir 时使用的 tag
        - experiment_group: str, ExperimentManager 生成 run_dir 时使用的父目录
        - monitor_mode: str, TOP checkpoint 分数选择方向, 取值 max 或 min
    """

    config_path: Path
    experiment_name: str
    tag: str
    experiment_group: str
    monitor_mode: str


def parse_bool(value: str) -> bool:
    """
    解析 CLI 布尔值。

    输入参数:
        - value: str, 命令行文本, 支持 true/false/1/0/yes/no/on/off

    输出:
        - enabled: bool, 解析后的布尔值
    """
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单实验: train.py 训练 -> TOP ckpt 选择 -> standard/strict two_stage_basic 评估。")
    parser.add_argument("--train-config", required=True, help="configs/experiment 下的训练配置 YAML 路径。")
    parser.add_argument("--val-config-standard", required=True, help="standard 的 protein_40 验证配置 YAML 路径。")
    parser.add_argument("--test-config-standard", required=True, help="standard 的 protein_110 测试配置 YAML 路径。")
    parser.add_argument("--val-config-strict", required=True, help="strict 的 protein_40 验证配置 YAML 路径。")
    parser.add_argument("--test-config-strict", required=True, help="strict 的 protein_110 测试配置 YAML 路径。")
    parser.add_argument("--vis-enable", type=parse_bool, default=True, help="是否在 Stage2/test 导出可视化资产; 默认 true。")
    parser.add_argument(
        "--feedback-root",
        default=os.environ.get("EXPERIMENT_FEEDBACK_ROOT", str(DEFAULT_FEEDBACK_ROOT)),
        help="训练输出根目录; 默认读取 EXPERIMENT_FEEDBACK_ROOT, 否则为项目父目录下 feedback_plus。",
    )
    parser.add_argument("--device", default="cuda:0", help="two_stage_basic 推理评估设备; 默认 cuda:0。")
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="追加给 train.py 的 Hydra override; 可重复传入。",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="追加给 two_stage_basic.py 的 OmegaConf dotlist override; 可重复传入。",
    )
    parser.add_argument(
        "--write-config-ckpt",
        action="store_true",
        help="将选出的 ckpt_path 写回四个 infer/eval YAML; 默认只通过命令行覆盖。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令和推导出的路径, 不实际训练或评估。",
    )
    return parser.parse_args(argv)


def sanitize_path_component(value: str) -> str:
    """
    复刻 ExperimentManager 的路径片段清理规则。

    输入参数:
        - value: str, tag 或 experiment_group 原始文本

    输出:
        - text: str, 可用于 run_dir 的路径片段
    """
    text = str(value).strip()
    if not text:
        return "Unnamed"
    return text.replace("\\", "-").replace("/", "-").replace(" ", "_")


def resolve_existing_path(value: str, *, base_dir: Path = PROJECT_ROOT) -> Path:
    """
    将命令行路径解析为真实存在的绝对路径。

    输入参数:
        - value: str, 绝对路径或相对项目根目录的路径
        - base_dir: Path, 相对路径解析基准目录

    输出:
        - path: Path, 已存在的绝对路径
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    return path


def load_train_spec(train_config_value: str) -> TrainRunSpec:
    """
    读取训练配置并提取 run_dir 与 checkpoint 选择所需信息。

    输入参数:
        - train_config_value: str, configs/experiment 下的 YAML 路径

    输出:
        - spec: TrainRunSpec, 单次训练运行元信息
    """
    config_path = resolve_existing_path(train_config_value)
    try:
        config_path.relative_to(EXPERIMENT_CONFIG_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"--train-config 必须位于 configs/experiment 下: {config_path}") from exc

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    monitor_mode = str(cfg.get("model", {}).get("monitor_mode", "max"))
    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"{config_path} 的 model.monitor_mode 必须为 max 或 min, 实际为 {monitor_mode}")
    return TrainRunSpec(
        config_path=config_path,
        experiment_name=config_path.stem,
        tag=str(cfg.get("tag", config_path.stem)),
        experiment_group=str(cfg.get("experiment_group", "default")),
        monitor_mode=monitor_mode,
    )


def resolve_run_stamp_from_env() -> str | None:
    """
    按 ExperimentManager 规则从 Slurm 环境推导 run stamp。

    输出:
        - run_stamp: str | None, Slurm 环境下可精确推导; 非 Slurm 环境返回 None
    """
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    step_id = os.environ.get("SLURM_STEP_ID", "").strip()
    if job_id and step_id and step_id.lower() != "batch":
        return f"job{job_id}_step{sanitize_path_component(step_id)}"
    if job_id:
        return f"job{job_id}"
    return None


def expected_run_dir(spec: TrainRunSpec, feedback_root: Path, run_stamp: str) -> Path:
    """
    根据 ExperimentManager 命名规则推导本次训练 run_dir。

    输入参数:
        - spec: TrainRunSpec, 训练元信息
        - feedback_root: Path, feedback_plus 根目录
        - run_stamp: str, Slurm job 派生的 run stamp

    输出:
        - run_dir: Path, 本次训练输出目录
    """
    return (
        feedback_root
        / "logs"
        / sanitize_path_component(spec.experiment_group)
        / f"{sanitize_path_component(spec.tag)}____{sanitize_path_component(run_stamp)}"
    )


def command_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def run_subprocess(command: Sequence[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    """
    以子进程执行训练或评估命令。

    输入参数:
        - command: Sequence[str], 完整命令参数列表
        - dry_run: bool, True 时只打印不执行
        - env: dict[str, str] | None, 子进程环境变量; None 表示继承当前环境

    输出:
        - None, 子进程失败时由 subprocess.run 抛出 CalledProcessError
    """
    print("=" * 88, flush=True)
    print(f"[global_pipeline] RUN: {command_text(command)}", flush=True)
    print("=" * 88, flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(PROJECT_PARENT), env=env, check=True)


def snapshot_existing_run_dirs(spec: TrainRunSpec, feedback_root: Path) -> set[Path]:
    """
    非 Slurm 调试时记录训练前已有 run_dir。

    输入参数:
        - spec: TrainRunSpec, 训练元信息
        - feedback_root: Path, feedback_plus 根目录

    输出:
        - run_dirs: set[Path], 训练前已存在的候选 run_dir 集合
    """
    group_dir = feedback_root / "logs" / sanitize_path_component(spec.experiment_group)
    pattern = f"{sanitize_path_component(spec.tag)}____*"
    return {path.resolve() for path in group_dir.glob(pattern) if path.is_dir()}


def find_new_local_run_dir(spec: TrainRunSpec, feedback_root: Path, before: set[Path]) -> Path:
    """
    非 Slurm 调试时寻找本轮新生成的 run_dir。

    输入参数:
        - spec: TrainRunSpec, 训练元信息
        - feedback_root: Path, feedback_plus 根目录
        - before: set[Path], 训练前已有 run_dir 集合

    输出:
        - run_dir: Path, 训练后新增的唯一或最新 run_dir
    """
    group_dir = feedback_root / "logs" / sanitize_path_component(spec.experiment_group)
    pattern = f"{sanitize_path_component(spec.tag)}____*"
    candidates = [path.resolve() for path in group_dir.glob(pattern) if path.is_dir() and path.resolve() not in before]
    if not candidates:
        raise FileNotFoundError(f"未找到本轮新增 run_dir: {group_dir / pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_training(spec: TrainRunSpec, feedback_root: Path, train_overrides: Sequence[str], *, dry_run: bool) -> Path:
    """
    子进程运行 train.py 并返回本轮 run_dir。

    输入参数:
        - spec: TrainRunSpec, 训练元信息
        - feedback_root: Path, feedback_plus 根目录
        - train_overrides: Sequence[str], 传给 train.py 的 Hydra override
        - dry_run: bool, True 时只打印命令并返回推导路径

    输出:
        - run_dir: Path, 本次训练输出目录
    """
    run_stamp = resolve_run_stamp_from_env()
    before = set() if run_stamp is not None else snapshot_existing_run_dirs(spec, feedback_root)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train.py"),
        f"+experiment={spec.experiment_name}",
        *train_overrides,
    ]
    env = os.environ.copy()
    env["EXPERIMENT_FEEDBACK_ROOT"] = str(feedback_root)
    run_subprocess(command, dry_run=dry_run, env=env)
    if run_stamp is not None:
        run_dir = expected_run_dir(spec, feedback_root, run_stamp)
    elif dry_run:
        run_dir = feedback_root / "logs" / sanitize_path_component(spec.experiment_group) / f"{sanitize_path_component(spec.tag)}____<localtime>"
    else:
        run_dir = find_new_local_run_dir(spec, feedback_root, before)
    if not dry_run and not run_dir.exists():
        raise FileNotFoundError(f"训练结束后未找到 run_dir: {run_dir}")
    print(f"[global_pipeline] run_dir: {run_dir}", flush=True)
    return run_dir


def parse_score_from_top_ckpt(path: Path) -> float:
    """
    从 TOP checkpoint 文件名解析 monitor score。

    输入参数:
        - path: Path, 文件名形如 TOP_epoch_05_score_0.7472.ckpt

    输出:
        - score: float, checkpoint 文件名中的分数
    """
    match = re.search(r"score_([-+]?\d+(?:\.\d+)?)", path.stem)
    if match is None:
        raise ValueError(f"TOP checkpoint 文件名缺少 score_<float>: {path}")
    return float(match.group(1))


def select_best_top_checkpoint(run_dir: Path, monitor_mode: str) -> Path:
    """
    从本次 run_dir/checkpoints 中选择最佳 TOP checkpoint。

    输入参数:
        - run_dir: Path, 本次训练输出目录
        - monitor_mode: str, max 取最高分, min 取最低分

    输出:
        - ckpt_path: Path, 最佳 TOP checkpoint 路径
    """
    ckpt_dir = run_dir / "checkpoints"
    top_ckpts = sorted(ckpt_dir.glob("TOP_*.ckpt"))
    if not top_ckpts:
        raise FileNotFoundError(f"未找到 TOP checkpoint: {ckpt_dir / 'TOP_*.ckpt'}")
    scored = [(parse_score_from_top_ckpt(path), path) for path in top_ckpts]
    reverse = monitor_mode == "max"
    best_score, best_path = sorted(scored, key=lambda item: item[0], reverse=reverse)[0]
    print(f"[global_pipeline] best TOP ckpt: {best_path} (score={best_score:.6f}, mode={monitor_mode})", flush=True)
    return best_path


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_ckpt_path_to_config(config_path: Path, ckpt_path: Path) -> None:
    """
    将 ckpt_path 写回 infer/eval YAML。

    输入参数:
        - config_path: Path, infer_or_eval YAML 路径
        - ckpt_path: Path, 选出的 TOP checkpoint 路径

    输出:
        - None, 原地更新 YAML 中的 ckpt_path 行
    """
    text = config_path.read_text(encoding="utf-8")
    quoted = yaml_quote(str(ckpt_path))
    new_text, count = re.subn(
        r"^(\s*ckpt_path:\s*)(.*?)(\s*(?:#.*)?)$",
        lambda match: f"{match.group(1)}{quoted}{match.group(3)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"未能在配置中定位唯一 ckpt_path 行: {config_path}")
    config_path.write_text(new_text, encoding="utf-8")
    print(f"[global_pipeline] wrote ckpt_path: {config_path}", flush=True)


def run_two_stage_pair(
    *,
    label: str,
    val_config: Path,
    test_config: Path,
    ckpt_path: Path,
    device: str,
    vis_enable: bool,
    eval_overrides: Sequence[str],
    write_config_ckpt: bool,
    dry_run: bool,
) -> None:
    """
    子进程运行一组 two_stage_basic 评估。

    输入参数:
        - label: str, 当前评估体系名称, standard 或 strict
        - val_config: Path, protein_40 验证 YAML
        - test_config: Path, protein_110 测试 YAML
        - ckpt_path: Path, 训练选出的 TOP checkpoint
        - device: str, torch device 文本
        - vis_enable: bool, Stage2/test 是否导出可视化
        - eval_overrides: Sequence[str], 额外 two_stage_basic override
        - write_config_ckpt: bool, 是否写回 YAML
        - dry_run: bool, True 时只打印命令不执行

    输出:
        - None
    """
    if write_config_ckpt and not dry_run:
        write_ckpt_path_to_config(val_config, ckpt_path)
        write_ckpt_path_to_config(test_config, ckpt_path)
    print(f"[global_pipeline] start eval: {label}", flush=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "inference" / "main" / "two_stage_basic.py"),
        "--val_config",
        str(val_config),
        "--test_config",
        str(test_config),
        f"ckpt_path={ckpt_path}",
        f"device={device}",
        f"pipeline_vis_enable={str(vis_enable).lower()}",
        *eval_overrides,
    ]
    run_subprocess(command, dry_run=dry_run)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    feedback_root = Path(args.feedback_root).expanduser().resolve()
    spec = load_train_spec(args.train_config)
    val_standard = resolve_existing_path(args.val_config_standard)
    test_standard = resolve_existing_path(args.test_config_standard)
    val_strict = resolve_existing_path(args.val_config_strict)
    test_strict = resolve_existing_path(args.test_config_strict)

    print("[global_pipeline] experiment:", flush=True)
    print(f"  train_config: {spec.config_path}", flush=True)
    print(f"  experiment:   {spec.experiment_name}", flush=True)
    print(f"  tag/group:    {spec.tag} / {spec.experiment_group}", flush=True)
    print(f"  monitor_mode: {spec.monitor_mode}", flush=True)
    print(f"  vis_enable:   {args.vis_enable}", flush=True)

    run_dir = run_training(spec, feedback_root, args.train_override, dry_run=bool(args.dry_run))
    if args.dry_run:
        ckpt_path = run_dir / "checkpoints" / "TOP_epoch_XX_score_X.XXXX.ckpt"
        print(f"[global_pipeline] dry-run ckpt placeholder: {ckpt_path}", flush=True)
    else:
        ckpt_path = select_best_top_checkpoint(run_dir, spec.monitor_mode)

    run_two_stage_pair(
        label="standard",
        val_config=val_standard,
        test_config=test_standard,
        ckpt_path=ckpt_path,
        device=str(args.device),
        vis_enable=bool(args.vis_enable),
        eval_overrides=args.eval_override,
        write_config_ckpt=bool(args.write_config_ckpt),
        dry_run=bool(args.dry_run),
    )
    run_two_stage_pair(
        label="strict",
        val_config=val_strict,
        test_config=test_strict,
        ckpt_path=ckpt_path,
        device=str(args.device),
        vis_enable=bool(args.vis_enable),
        eval_overrides=args.eval_override,
        write_config_ckpt=bool(args.write_config_ckpt),
        dry_run=bool(args.dry_run),
    )
    print("[global_pipeline] done.", flush=True)


if __name__ == "__main__":
    main()
