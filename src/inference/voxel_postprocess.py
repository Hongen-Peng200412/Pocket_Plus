from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.inference.utils.voxel_types import VoxelCandidate, VoxelPostprocessResult


# --------------------------------- 工具函数 ---------------------------------
# 导出 candidates 的信息
def _build_candidates(
    instance_label: np.ndarray,
    score_map: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
) -> list[VoxelCandidate]:
    """
    从 instance 标签图中导出候选区域元信息。

    输入参数:
        - instance_label: np.ndarray, (D,H,W), int32, instance 标签
        - score_map: np.ndarray, (D,H,W), float32, 评分图
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)

    输出:
        - candidates: list[VoxelCandidate], 候选区域列表, 包含 instance_id、voxel_count、score_mean、center_voxel_zyx 等等
    """
    candidates: list[VoxelCandidate] = []
    for instance_id in [int(v) for v in np.unique(instance_label) if int(v) > 0]:
        # np.ndarray, (N,3), int64, 当前 instance 内体素坐标(z,y,x)
        coords_zyx = np.argwhere(instance_label == instance_id)
        if coords_zyx.shape[0] == 0:
            continue
        # np.ndarray, (N,), float32, 当前 instance 内评分
        instance_scores = score_map[instance_label == instance_id]
        # np.ndarray, (3,), float64, 当前 instance 中心体素坐标(z,y,x)
        center_zyx = coords_zyx.mean(axis=0)
        # tuple[float,float,float], 当前 instance 中心世界坐标(x,y,z)
        center_world = voxel_zyx_to_world_xyz(tuple(float(v) for v in center_zyx), origin, voxel_size)
        # np.ndarray, (3,), int64, 包围盒最小坐标(z,y,x)
        bbox_min = coords_zyx.min(axis=0)
        # np.ndarray, (3,), int64, 包围盒最大坐标(z,y,x)
        bbox_max = coords_zyx.max(axis=0)
        candidates.append(
            VoxelCandidate(
                instance_id=instance_id,
                voxel_count=int(coords_zyx.shape[0]),
                score_mean=float(instance_scores.mean()),
                score_max=float(instance_scores.max()),
                center_voxel_zyx=tuple(float(v) for v in center_zyx.tolist()),
                center_world_xyz=center_world,
                bbox_min_zyx=tuple(int(v) for v in bbox_min.tolist()),
                bbox_max_zyx=tuple(int(v) for v in bbox_max.tolist()),
            )
        )
    return candidates

# 为 instance_map 中的 candidates 做 "最小平均得分"+"最小体素数" 的过滤
def _filter_instances_by_score(
    instance_label: np.ndarray,
    score_map: np.ndarray,
    min_component_voxels: int,
    instance_score_min: float,
) -> np.ndarray:
    """
    按体素数和实例平均分过滤 instance。

    输入参数:
        - instance_label: np.ndarray, (D,H,W), int32, instance 标签
        - score_map: np.ndarray, (D,H,W), float32, 评分图
        - min_component_voxels: int, 最小体素数
        - instance_score_min: float, 最低实例平均分

    输出:
        - filtered_label: np.ndarray, (D,H,W), int32, 过滤并重新编号后的标签
    """
    filtered_label = np.zeros_like(instance_label, dtype=np.int32)
    next_id = 1
    for instance_id in [int(v) for v in np.unique(instance_label) if int(v) > 0]:
        mask = instance_label == instance_id
        voxel_count = int(mask.sum())
        if voxel_count < int(min_component_voxels):
            continue
        score_mean = float(score_map[mask].mean())
        if score_mean < float(instance_score_min):
            continue
        filtered_label[mask] = next_id
        next_id += 1
    return filtered_label

# 高斯滤波计算
def _gaussian_filter_with_kernel(
    data: np.ndarray,
    sigma: float,
    kernel_size: int,
) -> np.ndarray:
    """
    按显式 kernel_size 控制 truncate 的 3D Gaussian 滤波。

    输入参数:
        - data: np.ndarray, (D,H,W), float32, 输入体素图
        - sigma: float, Gaussian sigma
        - kernel_size: int, Gaussian 核窗口大小

    输出:
        - filtered: np.ndarray, (D,H,W), float32, 滤波结果
    """
    if sigma <= 0:
        raise ValueError(f"sigma 必须大于 0, 实际为 {sigma}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size 必须为正奇数, 实际为 {kernel_size}")
    truncate = ((int(kernel_size) - 1) / 2.0) / float(sigma)
    return ndimage.gaussian_filter(data, sigma=float(sigma), mode="reflect", truncate=truncate).astype(np.float32)









# ------------------------------------------------ 主逻辑 -------------------------------------------------
def postprocess_ligand_probability_map(
    ligand_pred: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    threshold: float,
    min_component_voxels: int,
    filter_strength: str,
    connectivity_policy: str,
    sigma_nearby: float,
    kernel_nearby: int,
    receptor_pred: np.ndarray | None,
    sigma_response: float,
    kernel_response: int,
    score_add: float,
    score_minus: float,
    voxel_score_min: float,
    instance_score_min: float,
) -> VoxelPostprocessResult:
    """
    对单张 ligand 概率图执行 voxel-only 后处理。

    输入参数:
        输入基本信息：
        - ligand_pred: np.ndarray, (D,H,W), 已按 hardmask 约束的 ligand 概率图
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
        - threshold: float, 初始概率阈值
        - min_component_voxels: int, 最小实例体素数
        - filter_strength: str, "basic" 或 "advanced"


        第一次过滤(base): 简单的单阈值连通分析 + 去除小连通域
        - connectivity_policy: str, 两阶段连通域策略, 如 "7_none"
        - sigma_nearby: float, ligand 自身近邻高斯 sigma
        - kernel_nearby: int, ligand 自身近邻高斯核大小


        第二次过滤: 算 "概率得分"(用高斯滤波) + "响应得分"(按照正负得分系数) 
        - receptor_pred: np.ndarray | None, (D,H,W), receptor 概率图
        - sigma_response: float, receptor 响应高斯 sigma
        - kernel_response: int, receptor 响应高斯核大小
        - score_add: float, receptor positive 加分系数
        - score_minus: float, receptor negative 扣分系数

        - voxel_score_min: float, advanced 低分体素删除阈值
        - instance_score_min: float, advanced 低均分 instance 删除阈值

    输出:
        - result: VoxelPostprocessResult, 后处理结果对象
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold 必须在 [0,1], 实际为 {threshold}")
    if filter_strength not in {"basic", "advanced"}:
        raise ValueError(f"filter_strength 必须为 basic 或 advanced, 实际为 {filter_strength}")

    # np.ndarray, (D,H,W), float32, ligand 概率图
    prob_map = np.asarray(ligand_pred, dtype=np.float32)
    if prob_map.ndim != 3:
        raise ValueError(f"ligand_pred 必须为 (D,H,W), 实际为 {prob_map.shape}")
    if receptor_pred is not None and np.asarray(receptor_pred).shape != prob_map.shape:
        raise ValueError(f"receptor_pred.shape={np.asarray(receptor_pred).shape} 与 ligand_pred.shape={prob_map.shape} 不一致")

    first_conn, second_conn = parse_connectivity_policy(connectivity_policy)
    # np.ndarray, (D,H,W), bool, 初始阈值掩码
    binary_mask_raw = prob_map >= float(threshold)
    # np.ndarray, (D,H,W), int32, 初始连通域标签
    instance_label_raw = _label_connected_components(binary_mask_raw, first_conn)

    if filter_strength == "basic":
        filtered_label = _filter_small_components(instance_label_raw, min_component_voxels)
        score_map = prob_map.astype(np.float32, copy=True)
        candidates = _build_candidates(filtered_label, score_map, origin, voxel_size)
        return VoxelPostprocessResult(
            prob_map=prob_map,
            score_map=score_map,
            binary_mask_raw=binary_mask_raw,
            instance_label_raw=instance_label_raw,
            binary_mask_filtered=filtered_label > 0,
            instance_label_filtered=filtered_label,
            candidates=candidates,
        )


    if receptor_pred is None:
        raise ValueError("receptor_pred is None, 但是开启了advanced后处理")
    # np.ndarray, (D,H,W), float32, ligand 概率近邻响应
    nearby_score = _gaussian_filter_with_kernel(prob_map, sigma_nearby, kernel_nearby)
    score_map = prob_map + nearby_score
    # np.ndarray, (D,H,W), float32, receptor 正响应图
    receptor_float = np.asarray(receptor_pred, dtype=np.float32)
    # np.ndarray, (D,H,W), float32, receptor 正响应高斯图
    response_positive = _gaussian_filter_with_kernel(receptor_float, sigma_response, kernel_response)
    # np.ndarray, (D,H,W), bool, receptor 有效区域
    receptor_valid = receptor_float > 0
    # np.ndarray, (D,H,W), float32, receptor 负响应仅在 receptor 有效区域参与
    response_negative_input = np.where(receptor_valid, 1.0 - receptor_float, 0.0).astype(np.float32)
    response_negative = _gaussian_filter_with_kernel(response_negative_input, sigma_response, kernel_response)
    score_map = score_map + float(score_add) * response_positive - float(score_minus) * response_negative
    score_map = score_map.astype(np.float32, copy=False)

    # np.ndarray, (D,H,W), bool, advanced 体素级过滤掩码
    score_mask = binary_mask_raw & (score_map >= float(voxel_score_min))
    if second_conn == "none":
        # np.ndarray, (D,H,W), int32, 沿用第一次连通域但去掉低分体素
        score_instance_label = np.where(score_mask, instance_label_raw, 0).astype(np.int32)
    else:
        score_instance_label = _label_connected_components(score_mask, second_conn)
    filtered_label = _filter_instances_by_score(
        instance_label=score_instance_label,
        score_map=score_map,
        min_component_voxels=min_component_voxels,
        instance_score_min=instance_score_min,
    )
    candidates = _build_candidates(filtered_label, score_map, origin, voxel_size)
    return VoxelPostprocessResult(
        prob_map=prob_map,
        score_map=score_map,
        binary_mask_raw=binary_mask_raw,
        instance_label_raw=instance_label_raw,
        binary_mask_filtered=filtered_label > 0,
        instance_label_filtered=filtered_label,
        candidates=candidates,
    )








# ------------------------------------ 工具函数 --------------------------------------
def parse_connectivity_policy(connectivity_policy: str) -> tuple[str, str]:
    """
    解析两阶段连通域策略。

    输入参数:
        - connectivity_policy: str, 形如 7_none、7_7、19_19、27_none

    输出:
        - first_conn: str, 第一次连通域策略
        - second_conn: str, 第二次连通域策略或 none
    """
    parts = str(connectivity_policy).split("_")
    if len(parts) != 2:
        raise ValueError(f"connectivity_policy 必须形如 7_none, 实际为 {connectivity_policy}")
    first_conn, second_conn = parts
    build_connectivity_structure(first_conn)   # 只是顺便检查异常值而已
    build_connectivity_structure(second_conn)
    if first_conn == "none":
        raise ValueError("第一次连通域不能为 none")
    return first_conn, second_conn

def build_connectivity_structure(conn: str) -> np.ndarray | None:
    """
    构造 scipy.ndimage.label 使用的 3D 连通结构。

    输入参数:
        - conn: str, 允许 7/19/27/none

    输出:
        - structure: np.ndarray | None, (3,3,3) 连通结构或 None
    """
    if conn == "none":
        return None
    if conn == "7":
        return ndimage.generate_binary_structure(3, 1)
    if conn == "19":
        return ndimage.generate_binary_structure(3, 2)
    if conn == "27":
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError(f"未知 connectivity: {conn}")

def _label_connected_components(
    binary_mask: np.ndarray,
    conn: str,
) -> np.ndarray:
    """
    对三维二值掩码执行连通域标记。

    输入参数:
        - binary_mask: np.ndarray, (D,H,W), bool, 二值掩码
        - conn: str, 允许 7/19/27

    输出:
        - label: np.ndarray, (D,H,W), int32, 连通域标签
    """
    structure = build_connectivity_structure(conn)
    if structure is None:
        raise ValueError("连通域标记阶段不能使用 none")
    # np.ndarray, (D,H,W), int32, 连通域标签
    label, _ = ndimage.label(binary_mask, structure=structure)
    return label.astype(np.int32, copy=False)



def _filter_small_components(
    instance_label: np.ndarray,
    min_component_voxels: int,
) -> np.ndarray:
    """
    删除体素数过少的连通域。

    输入参数:
        - instance_label: np.ndarray, (D,H,W), int32, instance 标签
        - min_component_voxels: int, 最小体素数

    输出:
        - filtered_label: np.ndarray, (D,H,W), int32, 过滤并重新编号后的标签
    """
    if min_component_voxels < 1:
        raise ValueError(f"min_component_voxels 必须 >= 1, 实际为 {min_component_voxels}")

    filtered_label = np.zeros_like(instance_label, dtype=np.int32)
    next_id = 1
    for instance_id in [int(v) for v in np.unique(instance_label) if int(v) > 0]:
        # np.ndarray, (D,H,W), bool, 当前 instance 掩码
        mask = instance_label == instance_id
        if int(mask.sum()) < int(min_component_voxels):
            continue
        filtered_label[mask] = next_id
        next_id += 1
    return filtered_label



def voxel_zyx_to_world_xyz(
    voxel_zyx: tuple[float, float, float],
    origin: np.ndarray,
    voxel_size: np.ndarray,
) -> tuple[float, float, float]:
    """
    将体素中心坐标转换为世界坐标。

    输入参数:
        - voxel_zyx: tuple[float,float,float], 体素坐标(z,y,x)
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)

    输出:
        - world_xyz: tuple[float,float,float], 世界坐标(x,y,z)
    """
    z, y, x = [float(v) for v in voxel_zyx]
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(3)
    return (
        float(origin[0] + (x + 0.5) * voxel_size[0]),
        float(origin[1] + (y + 0.5) * voxel_size[1]),
        float(origin[2] + (z + 0.5) * voxel_size[2]),
    )

