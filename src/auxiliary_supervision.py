"""定义新版 Find_1 三项辅助监督共享的固定类别契约。

Dataset、体素输出头、损失指标和推理快照都从本模块读取同一类别顺序，避免
蛋白或核酸类别在不同文件中分别维护。
"""

from __future__ import annotations


PROTEIN_MAINCHAIN_CLASS_NAMES = ("background", "N", "CA", "C", "O")
NUCLEIC_MAINCHAIN_CLASS_NAMES = (
    "background",
    "P",
    "O5'",
    "C5'",
    "C4'",
    "C3'",
    "O3'",
)

PROTEIN_MAINCHAIN_CLASS_BY_ATOM_NAME = {
    atom_name: class_id
    for class_id, atom_name in enumerate(PROTEIN_MAINCHAIN_CLASS_NAMES)
    if class_id > 0
}
NUCLEIC_MAINCHAIN_CLASS_BY_ATOM_NAME = {
    atom_name: class_id
    for class_id, atom_name in enumerate(NUCLEIC_MAINCHAIN_CLASS_NAMES)
    if class_id > 0
}

PROTEIN_MAINCHAIN_PRIORS = (0.96, 0.01, 0.01, 0.01, 0.01)
NUCLEIC_MAINCHAIN_PRIORS = (0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
