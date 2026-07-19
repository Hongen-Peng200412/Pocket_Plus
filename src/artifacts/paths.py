"""AdaLigand Stage1 正式产物的固定路径契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE1_MODEL_NAMES: tuple[str, ...] = ("Find_0", "Find_1", "unet_c1")
OUTPUT_ROLES: tuple[str, ...] = (
    "probability",
    "components",
    "F1_centered",
    "CLG_centered",
    "Selected_Refined_Centered",
)


@dataclass(frozen=True)
class Stage1ArtifactPaths:
    """
    解析一个 producer/split/PDB 的全部正式 Stage1 路径。

    输入参数:
        - output_root: Path, `stage1_outputs` 根目录
        - stage1_model_name: str, `Find_0`、`Find_1` 或 `unet_c1`
        - split: str, 当前数据划分名
        - pdb_id: str, 当前 PDB 身份
    """

    output_root: Path
    stage1_model_name: str
    split: str
    pdb_id: str

    def __post_init__(self) -> None:
        if self.stage1_model_name not in STAGE1_MODEL_NAMES:
            raise ValueError(
                f"stage1_model_name={self.stage1_model_name!r} 不在 {STAGE1_MODEL_NAMES} 中"
            )
        if not self.split or not self.pdb_id:
            raise ValueError("split 与 pdb_id 必须是非空字符串")
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def producer_root(self) -> Path:
        """返回当前 producer 的正式根目录。"""
        return self.output_root / self.stage1_model_name

    @property
    def pdb_root(self) -> Path:
        """返回当前 split/PDB 的正式根目录。"""
        return self.producer_root / self.split / self.pdb_id

    @property
    def running_dir(self) -> Path:
        """返回 PDB 级运行互斥目录。"""
        return self.pdb_root / "_RUNNING"

    @property
    def blob_exceed_path(self) -> Path:
        """返回组件数量超限终态文件。"""
        return self.pdb_root / "_BLOB_EXCEED"

    def role_complete_path(self, output_role: str) -> Path:
        """
        返回指定 role 的 `_COMPLETE` 标记路径。

        输入参数:
            - output_role: str, `OUTPUT_ROLES` 中的正式 role

        输出:
            - path: Path, `status/{output_role}/_COMPLETE`
        """
        if output_role not in OUTPUT_ROLES:
            raise ValueError(f"未知 output_role={output_role!r}")
        return self.pdb_root / "status" / output_role / "_COMPLETE"

    @property
    def probability_npz(self) -> Path:
        """返回完整图概率归档路径。"""
        return self.pdb_root / "probability" / "probability_map.npz"

    @property
    def probability_geometry_json(self) -> Path:
        """返回完整图几何 JSON 路径。"""
        return self.pdb_root / "probability" / "geometry.json"

    @property
    def forest_npz(self) -> Path:
        """返回组件森林归档路径。"""
        return self.pdb_root / "components" / "forest.npz"

    @property
    def clg_npz(self) -> Path:
        """返回 CLG 归档路径。"""
        return self.pdb_root / "components" / "clg.npz"

    @property
    def overlap_npz(self) -> Path:
        """返回 candidate-occurrence 重叠归档路径。"""
        return self.pdb_root / "components" / "overlap.npz"

    @property
    def component_summary_json(self) -> Path:
        """返回组件与 CLG 统计 JSON 路径。"""
        return self.pdb_root / "components" / "summary.json"

    def centered_npz(self, centered_role: str) -> Path:
        """
        返回一个居中 role 的聚合 NPZ 路径。

        输入参数:
            - centered_role: str, `F1_centered`、`CLG_centered` 或
              `Selected_Refined_Centered`

        输出:
            - path: Path, 当前 PDB 的单个 role 级聚合文件
        """
        if centered_role not in OUTPUT_ROLES[2:]:
            raise ValueError(f"未知 centered_role={centered_role!r}")
        return self.pdb_root / "centered" / f"{centered_role}.npz"

    @property
    def calibration_root(self) -> Path:
        """返回当前 producer 的 calibration 根目录。"""
        return self.producer_root / "calibration"

    @property
    def thresholds_json(self) -> Path:
        """返回当前 producer 的冻结阈值表路径。"""
        return self.calibration_root / "thresholds.json"

    @property
    def calibration_metrics_json(self) -> Path:
        """返回当前 producer 的 calibration-fitted 指标路径。"""
        return self.calibration_root / "metrics.json"

    @property
    def threshold_scan_npz(self) -> Path:
        """返回 micro-Fα 完整扫描曲线与 TP/FP/FN 计数归档路径。"""
        return self.calibration_root / "threshold_scan.npz"

    @property
    def calibration_complete_path(self) -> Path:
        """返回当前 producer 的 calibration 完成标记。"""
        return self.calibration_root / "_COMPLETE"
