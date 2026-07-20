"""从 `producer/split/pdb_id` 身份解析 AdaLigand Stage1 正式产物路径。

主要入口:
    - `Stage1ArtifactPaths`: 统一生成完整图概率、组件谱系、三类 centered 归档、完成标记和 producer 级 calibration 文件路径。

本模块只计算 `Path`，不创建目录、不读取文件也不发布产物。PDB 级文件位于 `<output_root>/<producer>/<split>/<pdb_id>/`，calibration 文件位于 `<output_root>/<producer>/calibration/`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# 三个正式 Stage1 producer 名称；该顺序只用于稳定展示，不表示模型优先级。
STAGE1_MODEL_NAMES: tuple[str, ...] = ("Find_0", "Find_1", "unet_c1")
# 五个可独立发布完成标记的 PDB 级产物角色；列表顺序表达生产依赖，最后一个角色在 Selector 选择结果冻结后独立补跑。
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
        - output_root: Path, `stage1_outputs` 根目录；构造时只规范化路径，不要求目录已经存在。
        - stage1_model_name: str, `STAGE1_MODEL_NAMES` 中的模型来源身份，决定 producer 级目录。
        - split: str, 非空数据划分名，例如 `calibration`、`validation` 或 `train`。
        - pdb_id: str, 非空 PDB 身份，决定 PDB 级目录名。

    路径关系:
        - `producer_root`: `<output_root>/<stage1_model_name>`，包含该模型来源的全部数据划分和 calibration 产物。
        - `pdb_root`: `<producer_root>/<split>/<pdb_id>`，包含一个 PDB 的 probability、components、centered 与 status 子目录。
    """
    output_root: Path
    stage1_model_name: str
    split: str
    pdb_id: str

    def __post_init__(self) -> None:
        """
        校验产物身份并把输出根规范化为 Path。

        输出:
            - None, 原地规范化 frozen dataclass 的 `output_root`；非法 producer 或空 split/PDB 直接报错。
        """
        if self.stage1_model_name not in STAGE1_MODEL_NAMES:
            raise ValueError(f"stage1_model_name={self.stage1_model_name!r} 不在 {STAGE1_MODEL_NAMES} 中")
        if not self.split or not self.pdb_id:
            raise ValueError("split 与 pdb_id 必须是非空字符串")
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def producer_root(self) -> Path:
        """
        返回当前 producer 的正式根目录。

        输出:
            - path: Path, `output_root/stage1_model_name`
        """
        return self.output_root / self.stage1_model_name

    @property
    def pdb_root(self) -> Path:
        """
        返回当前 split/PDB 的正式根目录。

        输出:
            - path: Path, `producer_root/split/pdb_id`
        """
        return self.producer_root / self.split / self.pdb_id

    @property
    def running_dir(self) -> Path:
        """
        返回 PDB 级运行互斥目录。

        输出:
            - path: Path, `pdb_root/_RUNNING`
        """
        return self.pdb_root / "_RUNNING"

    @property
    def blob_exceed_path(self) -> Path:
        """
        返回组件数量超限终态文件。

        输出:
            - path: Path, `pdb_root/_BLOB_EXCEED`
        """
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
        """
        返回完整图概率归档路径。

        输出:
            - path: Path, `pdb_root/probability/probability_map.npz`
        """
        return self.pdb_root / "probability" / "probability_map.npz"

    @property
    def probability_geometry_json(self) -> Path:
        """
        返回完整图几何 JSON 路径。

        输出:
            - path: Path, `pdb_root/probability/geometry.json`
        """
        return self.pdb_root / "probability" / "geometry.json"

    @property
    def forest_npz(self) -> Path:
        """
        返回组件森林归档路径。

        输出:
            - path: Path, `pdb_root/components/forest.npz`
        """
        return self.pdb_root / "components" / "forest.npz"

    @property
    def clg_npz(self) -> Path:
        """
        返回 CLG 归档路径。

        输出:
            - path: Path, `pdb_root/components/clg.npz`
        """
        return self.pdb_root / "components" / "clg.npz"

    @property
    def overlap_npz(self) -> Path:
        """
        返回 candidate-occurrence 重叠归档路径。

        输出:
            - path: Path, `pdb_root/components/overlap.npz`
        """
        return self.pdb_root / "components" / "overlap.npz"

    @property
    def component_summary_json(self) -> Path:
        """
        返回组件与 CLG 统计 JSON 路径。

        输出:
            - path: Path, `pdb_root/components/summary.json`
        """
        return self.pdb_root / "components" / "summary.json"

    def centered_npz(self, centered_role: str) -> Path:
        """
        返回一个居中 role 的聚合 NPZ 路径。

        输入参数:
            - centered_role: str, `F1_centered`、`CLG_centered` 或 `Selected_Refined_Centered`。

        输出:
            - path: Path, `pdb_root/centered/{centered_role}.npz`，一个 PDB 的单个 role 级聚合文件。
        """
        if centered_role not in OUTPUT_ROLES[2:]:
            raise ValueError(f"未知 centered_role={centered_role!r}")
        return self.pdb_root / "centered" / f"{centered_role}.npz"

    @property
    def calibration_root(self) -> Path:
        """
        返回当前 producer 的 calibration 根目录。

        输出:
            - path: Path, `producer_root/calibration`
        """
        return self.producer_root / "calibration"

    @property
    def thresholds_json(self) -> Path:
        """
        返回当前 producer 的冻结阈值表路径。

        输出:
            - path: Path, `calibration_root/thresholds.json`
        """
        return self.calibration_root / "thresholds.json"

    @property
    def calibration_metrics_json(self) -> Path:
        """
        返回当前 producer 的 calibration-fitted 指标路径。

        输出:
            - path: Path, `calibration_root/metrics.json`
        """
        return self.calibration_root / "metrics.json"

    @property
    def threshold_scan_npz(self) -> Path:
        """
        返回 micro-Falpha 完整扫描曲线与 TP/FP/FN 计数归档路径。

        输出:
            - path: Path, `calibration_root/threshold_scan.npz`
        """
        return self.calibration_root / "threshold_scan.npz"

    @property
    def calibration_complete_path(self) -> Path:
        """
        返回当前 producer 的 calibration 完成标记。

        输出:
            - path: Path, `calibration_root/_COMPLETE`
        """
        return self.calibration_root / "_COMPLETE"
