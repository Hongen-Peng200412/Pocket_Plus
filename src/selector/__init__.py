"""AdaLigand Stage1 Selector 的数据、模型、训练与推理入口。"""

# 顶层包不提前导入 Dataset/Wrapper，避免训练入口与轻量结构化算法相互牵连。
__all__ = ["calibration", "dataset", "inference", "model", "structured", "train", "wrapper"]
