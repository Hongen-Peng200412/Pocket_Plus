from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree



def _build_voxel_gt_from_ligand_coords(
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    ligand_candidate_ids: np.ndarray,
    mapped_ligand_class_ids: np.ndarray,
    ligand_coords_map: dict[int, np.ndarray],
    distance_threshold: float,
) -> dict[str, Any]:
    """
    根据 per-ligand 原子坐标距离阈值构建整图 voxel GT。

    输入参数:
        - origin: np.ndarray, (3,), 密度图世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
        - grid_shape_zyx: tuple[int,int,int], 整图形状(D,H,W)
        - ligand_candidate_ids: np.ndarray, (N_ligand,), 配体 candidate_id
        - mapped_ligand_class_ids: np.ndarray, (N_ligand,), 映射后的配体类别 ID
        - ligand_coords_map: dict[int, np.ndarray], candidate_id → (M_i,3), ligand 原子世界坐标(x,y,z)
        - distance_threshold: float, 体素中心到最近 ligand 原子的前景距离阈值, 建议值 1.7

    输出:
        - result: dict[str, Any], voxel GT 字典, 包含:
            - "gt_ligand_mask": np.ndarray, (D,H,W), bool, 取值0/1: 这个体素处有/无 ligand
            - "gt_instance_label": np.ndarray, (D,H,W), int32, union ligand instance 标签图, 取值为 0~所有配体总数(不论类别)
            - "gt_ligand_mask_by_class_id": dict[int, np.ndarray], class_id -> (D,H,W), bool, 类别内 GT ligand mask, 取值为 0/1(是否是这个类别的配体)
            - "gt_instance_label_by_class_id": dict[int, np.ndarray], class_id -> (D,H,W), int32, 类别内 GT instance 标签, 取值为 0~这个类别的总配体数
            - "gt_instance_meta": list[dict[str, Any]], 每个保留 ligand 的 instance 来源元信息
                - candidate_id: int, 上游 ligand candidate ID
                - class_id: int, 映射后的任务类别 ID
                - union_instance_id: int, union instance 标签中的 ID(1~配体总数)
                - class_instance_id: int, 类别内 instance 标签中的 ID(1~这个类别的配体数)
                - voxel_count: int, 当前 ligand 写入 union GT 的体素数
    """
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(3)
    grid_shape_zyx = tuple(int(v) for v in grid_shape_zyx)

    # np.ndarray, (N_ligand,), int64, 配体 candidate_id
    ligand_candidate_ids = np.asarray(ligand_candidate_ids, dtype=np.int64)
    # np.ndarray, (N_ligand,), int64, 映射后的配体类别 ID
    mapped_ligand_class_ids = np.asarray(mapped_ligand_class_ids, dtype=np.int64)
    # np.ndarray, (D*H*W, 3), float32, 整图体素中心世界坐标(x,y,z)
    voxel_center_coords = _build_voxel_center_coords_xyz(
        grid_shape_zyx=grid_shape_zyx,
        origin_xyz=origin,
        voxel_size_xyz=voxel_size,
    )
    # np.ndarray, (D,H,W), int32, union GT instance 标签; 0表示背景
    gt_instance_label = np.zeros(grid_shape_zyx, dtype=np.int32)
    # dict[int, np.ndarray], class_id -> (D,H,W), 这个类别的 GT instance 标签
    gt_instance_label_by_class_id: dict[int, np.ndarray] = {}
    # dict[int, int], class_id -> 这个类别的下一个 instance ID
    next_class_instance_id: dict[int, int] = {}
    # list[dict[str, Any]], 每个保留 ligand 的 GT instance 来源元信息
    gt_instance_meta: list[dict[str, Any]] = []
    # int, 下一个 union GT instance ID; 0 保留给背景
    next_instance_id = 1

    for candidate_id, mapped_class_id in zip(
        ligand_candidate_ids.tolist(),
        mapped_ligand_class_ids.tolist(),
    ):
        # int, 当前 ligand 映射后的任务类别 ID; 0 表示不参与前景评估
        mapped_class_id = int(mapped_class_id)
        if mapped_class_id <= 0:
            continue
        # np.ndarray, (M,3), float32, 当前 ligand 原子世界坐标(x,y,z)
        ligand_coords = np.asarray(ligand_coords_map[int(candidate_id)], dtype=np.float32)
        if ligand_coords.shape[0] == 0:
            continue
        if mapped_class_id not in gt_instance_label_by_class_id:
            gt_instance_label_by_class_id[mapped_class_id] = np.zeros(grid_shape_zyx, dtype=np.int32)
            next_class_instance_id[mapped_class_id] = 1

        # np.ndarray, (D*H*W,), float32, 每个体素中心到当前 ligand 最近原子的距离
        distances, _ = cKDTree(ligand_coords).query(voxel_center_coords, k=1)
        # np.ndarray, (D,H,W), bool, 当前 ligand 距离阈值前景掩码
        instance_mask = distances.reshape(grid_shape_zyx) < float(distance_threshold)
        # np.ndarray, (D,H,W), bool, 尚未被其他 GT instance 占用且属于当前 ligand 的 union 体素
        writable_mask = np.logical_and(instance_mask, gt_instance_label == 0)
        # np.ndarray, (D,H,W), int32, 当前类别内 GT instance 标签
        class_instance_label = gt_instance_label_by_class_id[mapped_class_id]  # 绑定非拷贝
        # np.ndarray, (D,H,W), bool, 尚未被同类 GT instance 占用且属于当前 ligand 的体素
        class_writable_mask = np.logical_and(instance_mask, class_instance_label == 0)
        # int, 当前 ligand 在 union GT 中的 instance ID
        union_instance_id = int(next_instance_id)
        # int, 当前 ligand 在类别内 GT 中的 instance ID
        class_instance_id = int(next_class_instance_id[mapped_class_id])
        gt_instance_label[writable_mask] = union_instance_id
        class_instance_label[class_writable_mask] = class_instance_id
        # int, 当前 ligand 写入 union GT 的体素数
        voxel_count = int(writable_mask.sum())
        gt_instance_meta.append(
            {
                "candidate_id": int(candidate_id),
                "class_id": mapped_class_id,
                "union_instance_id": union_instance_id,
                "class_instance_id": class_instance_id,
                "voxel_count": voxel_count,
            }
        )
        next_instance_id += 1
        next_class_instance_id[mapped_class_id] = class_instance_id + 1

    # np.ndarray, (D,H,W), bool, 所有保留 GT instance 的并集
    gt_ligand_mask = gt_instance_label > 0
    # dict[int, np.ndarray], class_id -> (D,H,W), 类别内 GT ligand mask
    gt_ligand_mask_by_class_id = {
        class_id: class_label > 0
        for class_id, class_label in gt_instance_label_by_class_id.items()
    }
    return {
        "gt_ligand_mask": gt_ligand_mask,
        "gt_instance_label": gt_instance_label,
        "gt_ligand_mask_by_class_id": gt_ligand_mask_by_class_id,
        "gt_instance_label_by_class_id": gt_instance_label_by_class_id,
        "gt_instance_meta": gt_instance_meta,
    }





# ---------------------------------- 推理时得到 voxel-label 的第一种方式：加载.npz ----------------------------------
def load_ligand_gt_from_labels_npz(
    labels_npz_path: str,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    class_mapping: list[int] | None,
    ligand_gt_distance_threshold: float,
) -> dict[str, Any]:
    """
    从 labels.npz 读取 ligand 坐标并构建整图 voxel GT。

    输入参数:
        - labels_npz_path: str, Make_Data 生成的 labels.npz 路径
        - origin: np.ndarray, (3,), 密度图世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
        - grid_shape_zyx: tuple[int,int,int], 整图形状(D,H,W)
        - class_mapping: list[int] | None, 类别映射表; 映射后 >0 的 ligand 作为前景
        - ligand_gt_distance_threshold: float, 体素中心到最近 ligand 原子的前景距离阈值, 建议值 1.7

    输出:
        - result: dict[str, Any], voxel GT 字典, 包含:
            - "gt_ligand_mask": np.ndarray, (D,H,W), bool, 取值0/1: 这个体素处有/无 ligand
            - "gt_instance_label": np.ndarray, (D,H,W), int32, union ligand instance 标签图, 取值为 0~所有配体总数(不论类别)
            - "gt_ligand_mask_by_class_id": dict[int, np.ndarray], class_id -> (D,H,W), bool, 类别内 GT ligand mask, 取值为 0/1(是否是这个类别的配体)
            - "gt_instance_label_by_class_id": dict[int, np.ndarray], class_id -> (D,H,W), int32, 类别内 GT instance 标签, 取值为 0~这个类别的总配体数
            - "gt_instance_meta": list[dict[str, Any]], 每个保留 ligand 的 instance 来源元信息
                - candidate_id: int, 上游 ligand candidate ID
                - class_id: int, 映射后的任务类别 ID
                - union_instance_id: int, union instance 标签中的 ID(1~配体总数)
                - class_instance_id: int, 类别内 instance 标签中的 ID(1~这个类别的配体数)
                - voxel_count: int, 当前 ligand 写入 union GT 的体素数
    """
    with np.load(labels_npz_path, allow_pickle=False) as data:
        # np.ndarray, (N_ligand,), int64, labels.npz 中的配体 candidate_id
        ligand_candidate_ids = data["ligand_candidate_ids"].astype(np.int64, copy=False)
        # np.ndarray, (N_ligand,), int64, labels.npz 中的原始配体类别 ID
        ligand_class_ids = data["ligand_class_ids"].astype(np.int64, copy=False)

        # np.ndarray, (N_ligand,), int64, 映射后的配体类别 ID
        mapped_ligand_class_ids = np.asarray(
            [_map_class_id(int(class_id), class_mapping) for class_id in ligand_class_ids.tolist()],
            dtype=np.int64,
        )
        # dict[int, np.ndarray], candidate_id → (M_i,3), labels.npz 中的 ligand 原子坐标
        ligand_coords_map: dict[int, np.ndarray] = {}
        for candidate_id in ligand_candidate_ids.tolist():
            coord_key = f"ligand_coords_{int(candidate_id)}"
            if coord_key not in data.files:
                raise KeyError(f"labels.npz 缺少字段: {coord_key}")
            ligand_coords_map[int(candidate_id)] = data[coord_key].astype(np.float32, copy=False)

    return _build_voxel_gt_from_ligand_coords(
        origin=origin,
        voxel_size=voxel_size,
        grid_shape_zyx=grid_shape_zyx,
        ligand_candidate_ids=ligand_candidate_ids,
        mapped_ligand_class_ids=mapped_ligand_class_ids,
        ligand_coords_map=ligand_coords_map,
        distance_threshold=ligand_gt_distance_threshold,
    )





# ---------------------------------- 推理时得到 voxel-label 的第二种方式：从头构建 ----------------------------------
def load_ligand_gt_from_structure(
    ligand_structure_path: str,
    receptor_structure_path: str,
    filter_preset: str,
    class_mapping: list[int] | None,
    select_first_model: bool,
    error_dir: str | None,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    ligand_gt_distance_threshold: float,
) -> dict[str, Any]:
    """
    现场调用 Make_Data 逻辑构建 ligand voxel GT。

    输入参数:
        - ligand_structure_path: str, 用于提取 ligand GT 的结构路径
        - receptor_structure_path: str, 用于计算受体 binding label 的结构路径
        - filter_preset: str, 配体筛选预设名
        - class_mapping: list[int] | None, 类别映射表; 映射后 >0 的 ligand 作为前景
        - select_first_model: bool, 多 model 结构处理策略
        - error_dir: str | None, 结构解析错误输出目录
        - origin: np.ndarray, (3,), 密度图世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)
        - grid_shape_zyx: tuple[int,int,int], 整图形状(D,H,W)
        - ligand_gt_distance_threshold: float, 体素中心到最近 ligand 原子的前景距离阈值, 建议值 1.7

    输出:
        - result: dict[str, Any], voxel GT 字典, 包含:
            - "gt_ligand_mask": np.ndarray, (D,H,W), bool, 取值0/1: 这个体素处有/无 ligand
            - "gt_instance_label": np.ndarray, (D,H,W), int32, union ligand instance 标签图, 取值为 0~所有配体总数(不论类别)
            - "gt_ligand_mask_by_class_id": dict[int, np.ndarray], class_id -> (D,H,W), bool, 类别内 GT ligand mask, 取值为 0/1(是否是这个类别的配体)
            - "gt_instance_label_by_class_id": dict[int, np.ndarray], class_id -> (D,H,W), int32, 类别内 GT instance 标签, 取值为 0~这个类别的总配体数
            - "gt_instance_meta": list[dict[str, Any]], 每个保留 ligand 的 instance 来源元信息
                - candidate_id: int, 上游 ligand candidate ID
                - class_id: int, 映射后的任务类别 ID
                - union_instance_id: int, union instance 标签中的 ID(1~配体总数)
                - class_instance_id: int, 类别内 instance 标签中的 ID(1~这个类别的配体数)
                - voxel_count: int, 当前 ligand 写入 union GT 的体素数
    """
    from src.inference.parse_input import load_gt_from_structure

    # dict[str, Any], 结构现场求解的 atom/pocket/ligand GT 信息
    gt_data = load_gt_from_structure(
        ligand_structure_path=ligand_structure_path,
        receptor_structure_path=receptor_structure_path,
        filter_preset=filter_preset,
        class_mapping=class_mapping,
        select_first_model=select_first_model,
        error_dir=error_dir,
    )
    return _build_voxel_gt_from_ligand_coords(
        origin=origin,
        voxel_size=voxel_size,
        grid_shape_zyx=grid_shape_zyx,
        ligand_candidate_ids=gt_data["ligand_candidate_ids"],
        mapped_ligand_class_ids=gt_data["mapped_ligand_class_ids"],
        ligand_coords_map=gt_data["ligand_coords"],
        distance_threshold=ligand_gt_distance_threshold,
    )





# ========================================== 工具函数 =============================================
def _map_class_id(
    class_id: int,
    class_mapping: list[int] | None,
) -> int:
    """
    将原始 ligand class_id 映射为评估类别。

    输入参数:
        - class_id: int, 原始配体类别 ID
        - class_mapping: list[int] | None, 类别映射表; None 表示保留原始类别

    输出:
        - mapped_class_id: int, 映射后的类别 ID
    """
    if class_mapping is None:
        return int(class_id)
    if class_id < 0 or class_id >= len(class_mapping):
        raise IndexError(f"ligand_class_id={class_id} 超出 class_mapping 长度 {len(class_mapping)}")
    return int(class_mapping[class_id])

def _build_voxel_center_coords_xyz(
    grid_shape_zyx: tuple[int, int, int],
    origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
) -> np.ndarray:
    """
    构造整图每个体素中心的世界坐标。

    输入参数:
        - grid_shape_zyx: tuple[int,int,int], 整图形状(D,H,W)
        - origin_xyz: np.ndarray, (3,), 密度图世界坐标原点(x,y,z)
        - voxel_size_xyz: np.ndarray, (3,), 体素大小(x,y,z)

    输出:
        - coords_xyz: np.ndarray, (D*H*W, 3), 每个体素中心的世界坐标(x,y,z)
    """
    depth, height, width = tuple(int(v) for v in grid_shape_zyx)
    # np.ndarray, (D,H,W), int64, 三个空间轴的体素索引网格
    z_idx, y_idx, x_idx = np.mgrid[0:depth, 0:height, 0:width]
    # np.ndarray, (D*H*W, 3), float32, 体素中心世界坐标(x,y,z)
    coords_xyz = np.stack(
        [
            x_idx.ravel() * voxel_size_xyz[0] + origin_xyz[0] + voxel_size_xyz[0] * 0.5,
            y_idx.ravel() * voxel_size_xyz[1] + origin_xyz[1] + voxel_size_xyz[1] * 0.5,
            z_idx.ravel() * voxel_size_xyz[2] + origin_xyz[2] + voxel_size_xyz[2] * 0.5,
        ],
        axis=1,
    ).astype(np.float32)
    return coords_xyz

