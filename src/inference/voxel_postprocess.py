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
    从 instance 标签图中导出 candidates。

    输入参数:
        - instance_label: np.ndarray, (D,H,W), int32, instance 标签; 0 为背景, 正 id 从 1 开始连续
        - score_map: np.ndarray, (D,H,W), float32, 评分图
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)

    输出:
        - candidates: list[VoxelCandidate], 可变长度, 按 instance_id 升序排列的候选区域列表
    """
    # np.ndarray, (D,H,W), int32, 输入 instance 标签副本视图; 0 为背景, 正 id 从 1 开始连续
    label = np.asarray(instance_label, dtype=np.int32)
    # np.ndarray, (D,H,W), bool, 预测 instance 正区域掩码
    positive_mask = label > 0
    if not bool(positive_mask.any()):
        return []

    # tuple[np.ndarray, np.ndarray, np.ndarray], 每项形状 (N,), 正区域体素坐标(z,y,x)
    coords_zyx = np.nonzero(positive_mask)
    # np.ndarray, (N,), int64, 每个正区域体素对应的原始 instance id
    labels = label[positive_mask].astype(np.int64, copy=False)
    # np.ndarray, (N,), float64, 每个正区域体素对应的评分
    scores = np.asarray(score_map, dtype=np.float32)[positive_mask].astype(np.float64, copy=False)
    # int, 当前标签图中最大的正 instance id
    max_label = int(labels.max())

    # np.ndarray, (max_label+1,), int64, 每个原始 instance id 的体素数
    counts = np.bincount(labels, minlength=max_label + 1)
    # np.ndarray, (max_label+1,), float64, 每个原始 instance id 的评分总和
    score_sum = np.bincount(labels, weights=scores, minlength=max_label + 1)
    # np.ndarray, (max_label+1,), float64, 每个原始 instance id 的最高体素评分
    score_max = np.full(max_label + 1, -np.inf, dtype=np.float64)
    np.maximum.at(score_max, labels, scores)

    # list[np.ndarray], 长度 3, 每项形状 (max_label+1,), 每个 instance 在 z/y/x 轴的坐标总和
    coord_sum = [np.bincount(labels, weights=coords_zyx[axis], minlength=max_label + 1) for axis in range(3)]
    # np.ndarray, (max_label+1,3), int64, 每个 instance 的包围盒最小坐标(z,y,x)
    # np.iinfo(np.int64).max：获取 64 位有符号整数（int64）所能表示的最大数值
    bbox_min = np.full((max_label + 1, 3), np.iinfo(np.int64).max, dtype=np.int64)   
    # np.ndarray, (max_label+1,3), int64, 每个 instance 的包围盒最大坐标(z,y,x)
    bbox_max = np.full((max_label + 1, 3), np.iinfo(np.int64).min, dtype=np.int64)
    for axis in range(3):
        np.minimum.at(bbox_min[:, axis], labels, coords_zyx[axis])
        np.maximum.at(bbox_max[:, axis], labels, coords_zyx[axis])

    # list[VoxelCandidate], 可变长度, 输出候选区域列表
    candidates: list[VoxelCandidate] = []
    for instance_id in np.flatnonzero(counts > 0):
        if int(instance_id) == 0:
            continue
        # np.ndarray, (3,), float64, 当前 instance 的中心体素坐标(z,y,x)
        center_zyx = np.array([coord_sum[axis][instance_id] / counts[instance_id] for axis in range(3)], dtype=np.float64)
        candidates.append(
            VoxelCandidate(
                instance_id=int(instance_id),
                voxel_count=int(counts[instance_id]),
                score_mean=float(score_sum[instance_id] / counts[instance_id]),
                score_max=float(score_max[instance_id]),
                center_voxel_zyx=tuple(float(v) for v in center_zyx.tolist()),
                center_world_xyz=voxel_zyx_to_world_xyz(tuple(float(v) for v in center_zyx), origin, voxel_size),
                bbox_min_zyx=tuple(int(v) for v in bbox_min[instance_id].tolist()),
                bbox_max_zyx=tuple(int(v) for v in bbox_max[instance_id].tolist()),
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
    按体素数和实例平均分过滤 instance, 并将保留项重新编号为连续正 id。

    输入参数:
        - instance_label: np.ndarray, (D,H,W), int32, instance 标签; 0 为背景, 正 id 允许不连续
        - score_map: np.ndarray, (D,H,W), float32, 评分图
        - min_component_voxels: int, 最小体素数
        - instance_score_min: float, 最低实例平均分

    输出:
        - filtered_label: np.ndarray, (D,H,W), int32, 过滤后，id 仍从 1 开始连续
    """
    # np.ndarray, (D,H,W), int32, 输入 instance 标签副本视图; 0 为背景
    label = np.asarray(instance_label, dtype=np.int32)
    # np.ndarray, (D,H,W), bool, 预测 instance 正区域掩码
    positive_mask = label > 0
    if not bool(positive_mask.any()):
        return np.zeros_like(label, dtype=np.int32)

    # np.ndarray, (N,), int64, 每个正区域体素对应的原始 instance id
    labels = label[positive_mask].astype(np.int64, copy=False)
    # np.ndarray, (N,), float64, 每个正区域体素对应的评分
    scores = np.asarray(score_map, dtype=np.float32)[positive_mask].astype(np.float64, copy=False)
    # np.ndarray, (max_label+1,), int64, 每个原始 instance id 的体素数
    counts = np.bincount(labels)
    # np.ndarray, (max_label+1,), float64, 每个原始 instance id 的评分总和
    score_sum = np.bincount(labels, weights=scores, minlength=counts.shape[0])
    # np.ndarray, (max_label+1,), float64, 每个原始 instance id 的平均评分
    score_mean = np.zeros_like(score_sum, dtype=np.float64)
    score_mean[counts > 0] = score_sum[counts > 0] / counts[counts > 0]
    # np.ndarray, (max_label+1,), bool, 每个原始 instance id 是否通过体素数和平均分过滤
    keep = (counts >= int(min_component_voxels)) & (score_mean >= float(instance_score_min))
    keep[0] = False
    # np.ndarray, (max_label+1,), int32, 原始 instance id 到连续 instance id 的映射表
    new_ids = np.zeros_like(counts, dtype=np.int32)
    new_ids[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new_ids[label]

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
    删除体素数过少的连通域, 并将保留项重新编号为连续正 id。

    输入参数:
        - instance_label: np.ndarray, (D,H,W), int32, instance 标签; 0 为背景, 正 id 允许不连续
        - min_component_voxels: int, 最小体素数

    输出:
        - filtered_label: np.ndarray, (D,H,W), int32, 过滤后正 id 从 1 开始连续的标签
    """
    if min_component_voxels < 1:
        raise ValueError(f"min_component_voxels 必须 >= 1, 实际为 {min_component_voxels}")

    # np.ndarray, (D,H,W), int32, 输入 instance 标签副本视图; 0 为背景
    label = np.asarray(instance_label, dtype=np.int32)
    # np.ndarray, (max_label+1,), int64, 每个原始 instance id 的体素数
    counts = np.bincount(label.ravel())
    # np.ndarray, (max_label+1,), bool, 每个原始 instance id 是否通过体素数过滤
    keep = counts >= int(min_component_voxels)
    keep[0] = False
    # np.ndarray, (max_label+1,), int32, 原始 instance id 到连续 instance id 的映射表
    new_ids = np.zeros_like(counts, dtype=np.int32)
    new_ids[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new_ids[label]



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

