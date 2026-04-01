# -*- coding: utf-8 -*-
"""
聚合运行 Stage1 伪原子 / recycle 相关的关键测试。

运行内容:
    - 低层伪原子生成 / 注入 / 移除单测
    - Stage1 model 基础单测
    - 三种 pseudo recycle policy 的整网集成 smoke 测试
    - wrapper 层的损失与训练封装单测

建议用法:
    - 快速检查: `python scripts/check_stage1_pseudo_atom_pipeline.py --quick`
    - 完整检查: `python scripts/check_stage1_pseudo_atom_pipeline.py`
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest


def main() -> int:
    """
    解析命令行并调用 pytest 运行 Stage1 伪原子回归测试集合。
    输出:
        - exit_code: int, pytest 的退出码; 0 表示全部通过
    """
    parser = argparse.ArgumentParser(description="Run Stage1 pseudo-atom recycle regression tests.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只运行新增的三种 recycle policy 整网 smoke 测试。",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    full_targets = [
        repo_root / "tests/model/test_pseudo_atoms.py",
        repo_root / "tests/model/test_stage1_model.py",
        repo_root / "tests/model/test_stage1_pseudo_recycle_integration.py",
        repo_root / "tests/wrappers/test_voxel_point_stage1.py",
    ]
    quick_targets = [
        repo_root / "tests/model/test_stage1_pseudo_recycle_integration.py",
    ]
    selected_targets = quick_targets if args.quick else full_targets
    pytest_args = [str(path) for path in selected_targets] + ["-q"]
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
