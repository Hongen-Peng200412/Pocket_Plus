"""AdaLigand Stage1 Selector 的数据、模型、结构化求解、训练与推理包。"""

# 顶层包只声明子模块，不提前导入 Dataset、模型或训练包装器。这样仅使用精确反链
# 动态规划时不会连带导入密度处理、PyTorch DataLoader 和命令行训练依赖。
__all__ = ["calibration", "dataset", "inference", "model", "structured", "train", "wrapper"]
