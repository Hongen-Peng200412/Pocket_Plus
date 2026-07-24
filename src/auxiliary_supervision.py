"""定义 Stage1 蛋白主链、核酸主链和配体距离辅助监督的共享类别契约。

``stage1_dataset.py`` 按这里的原子名称映射生成体素类别编号；
``stage1_voxel_backbone.py`` 按同一类别顺序确定输出通道和初始化先验；
损失、验证指标与推理快照也按该顺序解释每个通道。配体距离没有离散类别，
Dataset 另行生成 ``1 / (1 + distance_Å)``，因此本文件不保存距离分箱。
"""

from __future__ import annotations

# 蛋白体素类别编号：0 表示该体素没有命中指定蛋白主链原子，
# 1、2、3、4 分别表示 N、CA、C、O 原子的 home voxel。
PROTEIN_MAINCHAIN_CLASS_NAMES = ("background", "N", "CA", "C", "O")
# 核酸体素类别编号：0 表示该体素没有命中指定核酸主链原子，
# 1 至 6 依次表示 P、O5'、C5'、C4'、C3'、O3' 原子的 home voxel。
NUCLEIC_MAINCHAIN_CLASS_NAMES = (
    "background",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "C3'",
    "O3'",
)

# dict[str, int]，蛋白主链原子名称到上方类别编号的映射；不收录背景类别。
PROTEIN_MAINCHAIN_CLASS_BY_ATOM_NAME = {
    atom_name: class_id
    for class_id, atom_name in enumerate(PROTEIN_MAINCHAIN_CLASS_NAMES)
    if class_id > 0
}
# dict[str, int]，核酸主链原子名称到上方类别编号的映射；不收录背景类别。
NUCLEIC_MAINCHAIN_CLASS_BY_ATOM_NAME = {
    atom_name: class_id
    for class_id, atom_name in enumerate(NUCLEIC_MAINCHAIN_CLASS_NAMES)
    if class_id > 0
}

# 每项与 PROTEIN_MAINCHAIN_CLASS_NAMES 同序；四个非背景类别的初始概率均为 0.01。
PROTEIN_MAINCHAIN_PRIORS = (0.96, 0.01, 0.01, 0.01, 0.01)
# 每项与 NUCLEIC_MAINCHAIN_CLASS_NAMES 同序；六个非背景类别的初始概率均为 0.01。
NUCLEIC_MAINCHAIN_PRIORS = (0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
