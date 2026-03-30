# -*- coding: utf-8 -*-
"""
legacy/__init__.py

旧版体素推断代码归档入口。
保留导入别名，以便需要回退时直接 from src.inference.legacy import ...
"""

from .get_pred_voxel import load_model as load_voxel_model
from .get_pred_voxel import run_inference as run_voxel_inference
from .postprocess_voxel import assign_prob_to_atoms
