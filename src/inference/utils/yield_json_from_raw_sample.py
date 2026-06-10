"""
这个文件主要负责找到用于推断或评估的所有样本, 生成它们的 .json 文件.
.json 形如 list[dict], 每项含 {"cif_path": ..., "map_path": ...} 即实际的路径对, 这是 Pocket_Plus/src/inference/run.py 里面  def _run_raw_batch_mode 直接接受的.

支持两种使用场景:
- 场景A (仅推断/用真实结构做特征+评估): 每项仅含 {"cif_path": ..., "map_path": ...}
- 场景B (预测结构做特征, 真实结构做评估): 每项含 {"cif_gt_path": ..., "cif_path": ..., "map_path": ...}
  其中 cif_path 指向 AF3/CryoAtom 等建模输出，cif_gt_path 指向实验解析的真实结构

它的主要功能:

1.
输入:
-   一个密度图根文件夹 map_folder, 一个结构根文件夹 pdb_folder. 还有一个必选的.json文件, 它格式为list[dict], 每项形如{emdb_id:pdb_id} 或 {"emd_20235": "6P1H"}, 本文件可以灵活识别(见NOTE). 这个.json文件的作用是筛选有效样本的范围(不在.json里面的就忽略).
-   可选的 pdb_gt_folder: 真实结构根文件夹（如原始PDB结构目录）。若提供，则在输出 dict 中额外添加 cif_gt_path。
-   可选的 sim_map_folder: 真实 receptor 模拟密度图根目录。若提供，则在输出 dict 中额外添加 sim_map_path。
-   可选的 sim_map_folder_cryoatom: cryoatom receptor 模拟密度图根目录。若提供，则在输出 dict 中额外添加 sim_map_path_cryoatom。
-   可选的 labels_npz_root: 预处理标签根目录。若提供，则在输出 dict 中额外添加 labels_npz_path。
-   用户给定一个可选的扫描数目限制 max_scan(只随机取出max_scan个进行后面的筛选条件检查) 和可选的最终接受数目限制 max_accept; 一系列可选的筛选条件(目前只支持 nucleic_ratio=非HETATOM的氨基酸残基数目/ 非HETATOM的核酸残基数目)
-   output_folder: 存放生成的.json文件的文件目录

输出:
-   .json 文件; list[dict], 每项含 {"cif_path": ..., "map_path": ...}, 按可选根目录额外写入 cif_gt_path/sim_map_path/sim_map_path_cryoatom/labels_npz_path

NOTE:
-   从提供的.json识别出的emdb_id只关注数字部分(如emd_4651和EMDB-4651.map效果等同, 匹配时也只看数字); pdb_id也不计大小写.
"""

import os
import json
import random
import re
import sys
from pathlib import Path
from typing import Any
random.seed(42)
from Bio.PDB import MMCIFParser, PDBParser
# Pocket_Plus 根目录
_CURRENT_DIR = Path(__file__).resolve().parent
_Pocket_Plus_ROOT = _CURRENT_DIR.parent.parent.parent
root_path = str(_Pocket_Plus_ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

from Make_Data.PDB_processor.config import (
    AMINO_ACIDS,
    DNA_NUCLEOTIDES,
    MODIFIED_RESIDUE_TO_PARENT,
    NUCLEOTIDES,
)
from utils.validate_density_map_pairs import (
    DEFAULT_SIM_MAP_KEYS,
    validate_sample_density_maps,
)

# --------------------------- 工具函数 --------------------------
def load_raw_pairs(raw_pairs: str | list[dict]) -> list[dict[str, str | None]]:
    """
    将 raw_pairs 配置统一解析为推理样本字典列表。容纳: cif_path/map_path/cif_gt_path/sim_map_path/sim_map_path_cryoatom/labels_npz_path/sample_name

    输入参数:
        - raw_pairs: str 或 list[dict], JSON 路径或已加载的样本列表; 每项必须包含 cif_path/map_path, 可包含 cif_gt_path/sim_map_path/sim_map_path_cryoatom/labels_npz_path/sample_name

    输出:
        - pairs: list[dict[str, str | None]], 可变长度, 规范化后的样本路径字典列表
    """
    if not raw_pairs:
        raise ValueError("raw_pairs/raw_pairs_json 不能为空。")

    if isinstance(raw_pairs, str):
        if not raw_pairs.endswith(".json"):
            raise ValueError(f"raw_pairs 若为字符串，当前仅支持 JSON 文件路径，收到: {raw_pairs}")
        with open(raw_pairs, "r", encoding="utf-8-sig") as file_obj:
            # list[dict], 可变长度, JSON 中的原始样本条目
            raw_pairs = json.load(file_obj)

    if not isinstance(raw_pairs, list):
        raise TypeError(f"raw_pairs 必须是 JSON 路径或 list[dict]，当前类型为: {type(raw_pairs)}")

    # list[dict[str, str | None]], 可变长度, 规范化后的样本路径字典
    pairs: list[dict[str, str | None]] = []
    for item_index, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise TypeError(f"raw_pairs 第 {item_index} 项必须是 dict，当前收到: {type(item)}")
        for required_key in ("cif_path", "map_path"):
            if required_key not in item or item[required_key] is None:
                raise KeyError(f"raw_pairs 第 {item_index} 项缺少必需字段: {required_key}")

        # str, 当前样本名; 未提供时从 cif_path 推断
        sample_name = item["sample_name"] if item.get("sample_name") is not None else Path(item["cif_path"]).stem
        pairs.append(
            {
                "cif_path": str(item["cif_path"]),
                "map_path": str(item["map_path"]),
                "cif_gt_path": None if item.get("cif_gt_path") is None else str(item["cif_gt_path"]),
                "sim_map_path": None if item.get("sim_map_path") is None else str(item["sim_map_path"]),
                "sim_map_path_cryoatom": None if item.get("sim_map_path_cryoatom") is None else str(item["sim_map_path_cryoatom"]),
                "labels_npz_path": None if item.get("labels_npz_path") is None else str(item["labels_npz_path"]),
                "sample_name": str(sample_name),
            }
        )

    return pairs

# str, 支持的结构文件扩展名集合
_STRUCTURE_EXTS = {".cif", ".pdb", ".mmcif"}
# str, 支持的密度图扩展名集合
_DENSITY_MAP_EXTS = {".map", ".mrc"}

def _scan_density_map_dir(folder: str) -> dict:
    """
    扫描密度图目录，按文件名中的数字部分建立索引。

    输入参数:
        - folder: str, 密度图根目录路径

    输出:
        - id_to_path: dict[str, str], EMDB 数字 ID → 密度图路径
    """
    # dict[str, str], EMDB 数字 ID → 密度图路径
    id_to_path = {}
    if not os.path.exists(folder):
        return id_to_path

    for file_name in os.listdir(folder):
        # str, 当前文件扩展名小写形式
        file_ext = os.path.splitext(file_name)[-1].lower()
        if file_ext not in _DENSITY_MAP_EXTS:
            continue
        # Match | None, 文件名中的 EMDB 数字部分
        num_match = re.search(r'\d+', file_name)
        if num_match is None:
            continue
        # str, EMDB 数字 ID
        emdb_id = num_match.group()
        if emdb_id not in id_to_path:
            id_to_path[emdb_id] = os.path.join(folder, file_name)
    return id_to_path


def _scan_labels_npz_root(folder: str) -> dict:
    """
    扫描 labels.npz 根目录，兼容 {folder}/{pdb_id}/labels.npz 与 {folder}/{pdb_id}.npz。

    输入参数:
        - folder: str, labels.npz 根目录路径

    输出:
        - id_to_path: dict[str, str], 小写 PDB ID → labels.npz 路径
    """
    # dict[str, str], 小写 PDB ID → labels.npz 路径
    id_to_path = {}
    if not os.path.exists(folder):
        return id_to_path

    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)
        if os.path.isdir(entry_path):
            # str, 嵌套布局中的 labels.npz 路径
            labels_path = os.path.join(entry_path, "labels.npz")
            if os.path.isfile(labels_path):
                id_to_path[entry.lower()] = labels_path
        elif entry.lower().endswith(".npz"):
            # str, 扁平布局中的文件名 stem
            stem = os.path.splitext(entry)[0].lower()
            if stem != "labels":
                id_to_path[stem] = entry_path
    return id_to_path


def _scan_structure_dir(folder: str) -> dict:
    """
    扫描结构文件目录，同时兼容 扁平({folder}/{id}.cif) 和 嵌套({folder}/{id}/{id}.cif) 两种布局。

    扫描优先级: 先收嵌套, 再收扁平, 扁平覆盖嵌套（即扁平优先）。

    输入参数:
        - folder: str, 标量, 结构文件根目录绝对路径

    输出:
        - id_to_path: dict[str, str], 小写文件名 stem(pdb_id) → 绝对路径
    """
    # dict, str:str, 小写stem → 绝对路径
    id_to_path = {}
    if not os.path.exists(folder):
        return id_to_path

    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)

        if os.path.isdir(entry_path):
            # 嵌套布局: {folder}/{subdir}/{subdir}.cif|pdb|mmcif
            # str, 子目录名的小写形式, 作为候选 stem
            subdir_lower = entry.lower()
            for ext in _STRUCTURE_EXTS:
                # str, 候选嵌套文件路径
                nested_path = os.path.join(entry_path, f"{entry}{ext}")
                if os.path.isfile(nested_path):
                    if subdir_lower not in id_to_path:
                        id_to_path[subdir_lower] = nested_path
                    break
        else:
            # 扁平布局: {folder}/{id}.cif|pdb|mmcif
            # str, 扩展名小写
            p_ext = os.path.splitext(entry)[-1].lower()
            if p_ext in _STRUCTURE_EXTS:
                # str, 文件名stem小写
                p_base = os.path.splitext(entry)[0].lower()
                # 扁平优先: 无条件覆盖(可能覆盖嵌套的条目)
                id_to_path[p_base] = entry_path

    return id_to_path


def get_nucleic_ratio(file_path: str) -> float:
    """
    读取并过滤结构文件，计算核酸与蛋白的核酸/氨基酸比例.
    
    输入参数:
    - file_path: str , 标量, 表示结构文件绝对路径
    
    输出结果:
    - nucleic_ratio: float , 标量, 计算出的比例值（非HETATM核酸数量 / 非HETATM氨基酸数量）。
      只要核酸为空，都是 0.0; 如果没有氨基酸但有核酸，则返回 inf
    """
    # str, 抽取到的文件扩展名后缀
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdb':
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
        
    try:
        structure = parser.get_structure("struct", file_path)
    except Exception:
        return -1.0
    # int,  用来统计蛋白质氨基酸数量的累加器
    num_aa = 0
    # int,  用来统计核酸分子数量的累加器
    num_na = 0
    aa_set = set(AMINO_ACIDS)
    na_set = set(NUCLEOTIDES) | set(DNA_NUCLEOTIDES)
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # tuple, (3,), 残基标识符，格式如 (' ', 15, ' ')
                het_flag, resseq, icode = residue.id
                if het_flag != ' ':
                    # 非空即是配体或水，丢弃
                    continue
                    
                # str,  获取去除两端无用空格的大写残基识别字串
                resname = residue.resname.strip().upper()
                # str,  看看该序列被记录为修饰残留的话替换为对应的标准祖先残基，否则原样抛回
                resname = MODIFIED_RESIDUE_TO_PARENT.get(resname, resname)
                # bool,  判断当前解析字符是否确为蛋白序列
                if resname in aa_set:
                    num_aa += 1
                elif resname in na_set:
                    num_na += 1
        # 只取结构中第一个模型作为代表即可终止
        break
        
    if num_na == 0:
        return 0.0
    if num_aa == 0 and num_na > 0:
        return float('inf')
        
    # float, 标量, 返回 核酸个数 / 氨基酸个数
    return float(num_na) / float(num_aa)


def _scan_candidate_pair(
    pair: tuple[str, str],
    map_dict: dict[str, str],
    pdb_dict: dict[str, str],
    pdb_gt_dict: dict[str, str],
    sim_map_dict: dict[str, str],
    sim_map_cryoatom_dict: dict[str, str],
    labels_npz_dict: dict[str, str],
    pdb_gt_folder: str | None,
    sim_map_folder: str | None,
    sim_map_folder_cryoatom: str | None,
    labels_npz_root: str | None,
    min_nucleic_ratio: float | None,
    max_nucleic_ratio: float | None,
    validate_density_map_pairs: bool,
    target_voxel_size: float,
    density_origin_atol: float,
    density_voxel_rtol: float,
    density_voxel_atol: float,
) -> dict[str, Any]:
    """
    检查单个候选 EMDB/PDB 配对是否可写入样本 JSON。
    输入参数:
        - pair: tuple[str, str], (emdb_id, pdb_id), 二者均已标准化为小写或数字 ID
        - map_dict: dict[str, str], EMDB 数字 ID 到真实密度图路径的映射
        - pdb_dict: dict[str, str], 小写 PDB ID 到输入结构路径的映射
        - pdb_gt_dict: dict[str, str], 小写 PDB ID 到真实结构路径的映射
        - sim_map_dict: dict[str, str], EMDB 数字 ID 到真实 receptor 模拟图路径的映射
        - sim_map_cryoatom_dict: dict[str, str], EMDB 数字 ID 到 cryoatom 模拟图路径的映射
        - labels_npz_dict: dict[str, str], 小写 PDB ID 到 labels.npz 路径的映射
        - pdb_gt_folder: str | None, 是否要求真实结构路径
        - sim_map_folder: str | None, 是否要求真实 receptor 模拟图路径
        - sim_map_folder_cryoatom: str | None, 是否要求 cryoatom 模拟图路径
        - labels_npz_root: str | None, 是否要求 labels.npz 路径
        - min_nucleic_ratio: float | None, 核酸比例下界
        - max_nucleic_ratio: float | None, 核酸比例上界
        - validate_density_map_pairs: bool, 是否校验真实图与模拟图几何一致性
        - target_voxel_size: float, 推理重采样目标体素大小
        - density_origin_atol: float, origin 绝对误差容忍度
        - density_voxel_rtol: float, voxel_size 相对误差容忍度
        - density_voxel_atol: float, voxel_size 绝对误差容忍度

    输出:
        - result: dict[str, Any], 包含:
            - "status": str, accepted/missing_files/filtered_ratio/density_mismatch
            - "sample": dict[str, str] | None, 通过时的样本路径字典
            - "message": str, 失败说明或空字符串
    """
    emdb_id = pair[0]
    pdb_id = pair[1].lower()

    map_path = map_dict.get(emdb_id)
    if map_path is None or not os.path.exists(map_path):
        return {"status": "missing_files", "sample": None, "message": f"missing map_path for emdb_id={emdb_id}"}

    cif_path = pdb_dict.get(pdb_id)
    if cif_path is None or not os.path.exists(cif_path):
        return {"status": "missing_files", "sample": None, "message": f"missing cif_path for pdb_id={pdb_id}"}

    cif_gt_path = None
    if pdb_gt_folder is not None:
        cif_gt_path = pdb_gt_dict.get(pdb_id)
        if cif_gt_path is None or not os.path.exists(cif_gt_path):
            return {"status": "missing_files", "sample": None, "message": f"missing cif_gt_path for pdb_id={pdb_id}"}

    sim_map_path = None
    if sim_map_folder is not None:
        sim_map_path = sim_map_dict.get(emdb_id)
        if sim_map_path is None or not os.path.exists(sim_map_path):
            return {"status": "missing_files", "sample": None, "message": f"missing sim_map_path for emdb_id={emdb_id}"}

    sim_map_path_cryoatom = None
    if sim_map_folder_cryoatom is not None:
        sim_map_path_cryoatom = sim_map_cryoatom_dict.get(emdb_id)
        if sim_map_path_cryoatom is None or not os.path.exists(sim_map_path_cryoatom):
            return {"status": "missing_files", "sample": None, "message": f"missing sim_map_path_cryoatom for emdb_id={emdb_id}"}

    labels_npz_path = None
    if labels_npz_root is not None:
        labels_npz_path = labels_npz_dict.get(pdb_id)
        if labels_npz_path is None or not os.path.exists(labels_npz_path):
            return {"status": "missing_files", "sample": None, "message": f"missing labels_npz_path for pdb_id={pdb_id}"}

    if (min_nucleic_ratio is not None) or (max_nucleic_ratio is not None):
        nucleic_ratio = get_nucleic_ratio(cif_path)
        if nucleic_ratio < 0:
            return {"status": "filtered_ratio", "sample": None, "message": f"parse structure failed for pdb_id={pdb_id}"}
        ratio_valid = True
        if min_nucleic_ratio is not None and nucleic_ratio < min_nucleic_ratio:
            ratio_valid = False
        if max_nucleic_ratio is not None and nucleic_ratio > max_nucleic_ratio:
            ratio_valid = False
        if not ratio_valid:
            return {"status": "filtered_ratio", "sample": None, "message": f"nucleic_ratio={nucleic_ratio} for pdb_id={pdb_id}"}

    sample = {
        "cif_path": cif_path,
        "map_path": map_path,
    }
    if pdb_gt_folder is not None:
        sample["cif_gt_path"] = cif_gt_path
    if sim_map_folder is not None:
        sample["sim_map_path"] = sim_map_path
    if sim_map_folder_cryoatom is not None:
        sample["sim_map_path_cryoatom"] = sim_map_path_cryoatom
    if labels_npz_root is not None:
        sample["labels_npz_path"] = labels_npz_path

    if validate_density_map_pairs:
        validations = validate_sample_density_maps(
            sample=sample,
            target_voxel_size=float(target_voxel_size),
            origin_atol=float(density_origin_atol),
            voxel_rtol=float(density_voxel_rtol),
            voxel_atol=float(density_voxel_atol),
            sim_map_keys=DEFAULT_SIM_MAP_KEYS,
        )
        failed_validations = [item for item in validations if not item.ok]
        if failed_validations:
            failed = failed_validations[0]
            return {
                "status": "density_mismatch",
                "sample": None,
                "message": f"{pdb_id}/{emdb_id} {failed.sim_map_key}: {failed.reason}",
            }

    return {"status": "accepted", "sample": sample, "message": ""}






# ----------- 总函数 -----------
def generate_sample_json(
    map_folder: str=None,
    pdb_folder: str=None,
    pdb_gt_folder: str=None,   # 可选，真实结构根文件夹（如原始PDB目录），若提供则在输出 dict 中添加 cif_gt_path
    sim_map_folder: str=None,  # 可选，模拟密度图根目录，若提供则在输出 dict 中添加 sim_map_path
    sim_map_folder_cryoatom: str=None, # 可选，cryoatom模拟密度图根目录，若提供则在输出 dict 中添加 sim_map_path_cryoatom
    labels_npz_root: str=None, # 可选，预处理标签根目录，若提供则在输出 dict 中添加 labels_npz_path
    output_folder: str=None,
    valid_json_path: str = None,   # /home/penghongen/My_Project/Data/raw.json, 单条目形如: {"emd_63092": "9LHB"}

    max_scan: int = None,
    max_accept: int = None,
    min_nucleic_ratio: float = None,
    max_nucleic_ratio: float = None,
    output_json_name: str = "raw_samples.json",

    validate_density_map_pairs: bool = True,
    target_voxel_size: float = 1.0,
    density_origin_atol: float = 1e-4,
    density_voxel_rtol: float = 1e-5,
    density_voxel_atol: float = 1e-6,
) -> str:
    """
    生成推断和评估读取原始样本的 JSON 路径映射文件。

    输入参数:
        - map_folder:      str, 标量, 表示密度图所在的根文件夹绝对或相对路径
        - pdb_folder:      str, 标量, 表示蛋白核酸等结构文件所在的根文件夹(用于提取模型输入特征)
        - pdb_gt_folder:   str | None, 标量, 可选，真实结构根文件夹（用于提取 GT 标签评估）。
                           若提供，则在输出 dict 中为每个样本额外添加 "cif_gt_path" 键（若在该目录找到同名文件）。
        - sim_map_folder:  str | None, 模拟密度图根目录; 若提供则要求每个样本存在对应 sim_map_path
        - sim_map_folder_cryoatom: str | None, cryoatom 模拟密度图根目录; 若提供则要求每个样本存在对应 sim_map_path_cryoatom
        - labels_npz_root: str | None, labels.npz 根目录; 若提供则要求每个样本存在对应 labels_npz_path
        - output_folder:   str, 标量, 表示最后生成的json文件要存放的目录
        - valid_json_path: str, 标量, 必选的范围界定json文件(也提供emdb-pdb映射), 单条目形如: {"emd_63092": "9LHB"}

        - max_scan:        int, 标量, 代表从初始候选集合里随机取出验证的最多数目。默认为 None
        - max_accept:      int, 标量, 可选参数，代表最终保留进入生成的 json 的组合对数量上限。
                           达到上限后立即停止扫描后续候选。默认为 None
        - min_nucleic_ratio: float, 标量, 核酸数量 / 氨基酸数量（排除HETATM异构物与配体）下界截断。默认为 None
        - max_nucleic_ratio: float, 标量, 核酸数量 / 氨基酸数量（排除HETATM异构物与配体）上界截断。默认为 None
        - output_json_name: str, 标量, 生成 json 目标字典的默认具体文件名。默认为 "raw_samples.json"

        - validate_density_map_pairs: bool, 是否校验真实密度图和已存在模拟密度图的推理几何一致性
        - target_voxel_size: float, 密度图重采样目标体素大小, 应与推理配置保持一致
        - density_origin_atol: float, origin 绝对误差容忍度
        - density_voxel_rtol: float, voxel_size 相对误差容忍度
        - density_voxel_atol: float, voxel_size 绝对误差容忍度

    输出:
        - output_json_path: str, .json 文件的绝对路径, 内容为 list[dict[str,str]], 每项至少包含 cif_path/map_path, 并按输入根目录追加 cif_gt_path / sim_map_path / sim_map_path_cryoatom / labels_npz_path
    """
    # list, (N,), 用来预先存储后续所有可匹配探测配对；其内部的每个元素的形状均为 (2,) 的元组，分别代表(emdb_id, pdb_id)~(数字,小写字母)的元组
    candidate_pairs = []
    print(f"---------- 开始生成样本 JSON 文件 ----------")
    if valid_json_path is None:
        raise ValueError("必须提供 valid_json_path，以界定可用样本的 json 清单范围。")
    with open(valid_json_path, "r", encoding="utf-8-sig") as f_json:
        json_list = json.load(f_json)
    for item in json_list:
        keys = list(item.keys())
        if len(keys) == 0:
            continue
        # str, 标量, 把其第一个提取的 key 作为原始的 emdb 索引
        orig_emdb_id = str(keys[0])
        # Match, 标量, 尝试使用正则仅拾取数字部分
        num_match = re.search(r'\d+', orig_emdb_id)
        if not num_match:
            continue
        emdb_id = num_match.group()
        pdb_id = str(item[keys[0]]).lower()
        candidate_pairs.append((emdb_id, pdb_id))
    num_candidates = len(candidate_pairs)
    print(f"提取到的初始候选配对总数: {num_candidates}")
    if max_scan is not None and max_scan < num_candidates:
        random.shuffle(candidate_pairs)
        candidate_pairs = candidate_pairs[:max_scan]
        print(f"应用 max_scan 限制，将扫描验证的对数截断为: {max_scan}")
    else:
        random.shuffle(candidate_pairs)



    # dict[str, str], EMDB 数字 ID → 真实密度图路径
    map_dict = _scan_density_map_dir(map_folder)
    # dict[str, str], 小写 PDB ID → 推理输入结构路径
    pdb_dict = _scan_structure_dir(pdb_folder)
    # dict[str, str], 小写 PDB ID → 真实结构路径
    pdb_gt_dict = _scan_structure_dir(pdb_gt_folder) if (pdb_gt_folder is not None) else {}
    # dict[str, str], EMDB 数字 ID → 模拟密度图路径
    sim_map_dict = _scan_density_map_dir(sim_map_folder) if (sim_map_folder is not None) else {}
    # dict[str, str], EMDB 数字 ID → cryoatom 模拟密度图路径
    sim_map_cryoatom_dict = _scan_density_map_dir(sim_map_folder_cryoatom) if (sim_map_folder_cryoatom is not None) else {}
    # dict[str, str], 小写 PDB ID → labels.npz 路径
    labels_npz_dict = _scan_labels_npz_root(labels_npz_root) if (labels_npz_root is not None) else {}





    os.makedirs(output_folder, exist_ok=True)
    worker_kwargs = dict(
        map_dict=map_dict,
        pdb_dict=pdb_dict,
        pdb_gt_dict=pdb_gt_dict,
        sim_map_dict=sim_map_dict,
        sim_map_cryoatom_dict=sim_map_cryoatom_dict,
        labels_npz_dict=labels_npz_dict,
        pdb_gt_folder=pdb_gt_folder,
        sim_map_folder=sim_map_folder,
        sim_map_folder_cryoatom=sim_map_folder_cryoatom,
        labels_npz_root=labels_npz_root,
        min_nucleic_ratio=min_nucleic_ratio,
        max_nucleic_ratio=max_nucleic_ratio,
        validate_density_map_pairs=bool(validate_density_map_pairs),
        target_voxel_size=float(target_voxel_size),
        density_origin_atol=float(density_origin_atol),
        density_voxel_rtol=float(density_voxel_rtol),
        density_voxel_atol=float(density_voxel_atol),
    )
    # list[dict[str, str]], (A,), 最终写入 JSON 的合格样本; 达到 max_accept 后立即停止扫描
    accepted_samples = []
    # int, 标量, 为了获得 accepted_samples 实际扫描过的候选数
    actual_scanned_count = 0
    # int, 标量, 已扫描候选中因缺失结构、真实图、模拟图或 labels.npz 被过滤的数量
    missing_files_count = 0
    # int, 标量, 已扫描候选中因核酸比例不合格被过滤的数量
    filtered_out_count = 0
    # int, 标量, 已扫描候选中因真实图/模拟图几何不匹配被过滤的数量
    density_mismatch_count = 0

    for pair in candidate_pairs:
        if max_accept is not None and len(accepted_samples) >= max_accept:
            break

        result = _scan_candidate_pair(pair, **worker_kwargs)
        actual_scanned_count += 1
        result_status = result["status"]
        if result_status == "missing_files":
            missing_files_count += 1
        elif result_status == "filtered_ratio":
            filtered_out_count += 1
        elif result_status == "density_mismatch":
            density_mismatch_count += 1
        elif result_status == "accepted" and result["sample"] is not None:
            accepted_samples.append(result["sample"])

    accepted_count = len(accepted_samples)
        
    print(f"---------- 扫描完成，统计信息 ----------")
    print(f"  - 初始检查候选配对总数: {num_candidates}")
    print(f"  - 实际执行文件和比例探测运算的配对次数: {actual_scanned_count}")
    print(f"  - 缺失结构或密度图文件配对数: {missing_files_count}")
    print(f"  - 因核酸比例不合格而过滤抛弃数: {filtered_out_count}")
    print(f"  - 因真实/模拟密度图几何不匹配而过滤的配对数: {density_mismatch_count}")
    print(f"  - 最终保留且符合要求的有效配对数: {accepted_count}")
    output_json_path = os.path.join(output_folder, output_json_name)
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(accepted_samples, out_f, indent=4, ensure_ascii=False)
        
    return output_json_path






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成匹配验证样本列表")
    # see me: 第一套是不涉及cryoAtom的, 第二套是涉及cryoAtom的

    # -------------------------- 输入输出 --------------------------
    parser.add_argument("--map_folder", type=str, help="密度图根目录", 
    default="/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc")   # 更全的在: /storage/chenzhaoyang/cryo_em/EMDB

    # parser.add_argument("--pdb_folder", type=str, help="结构根目录（用于提取模型输入特征，如 AF3/CryoAtom 输出目录）", 
    # default="/storage/chenzhaoyang/cryo_em/PDB_3.5_cc_qscore")
    parser.add_argument("--pdb_folder", type=str, help="结构根目录", 
    default="/storage/chenzhaoyang/cryo_em/result_split")



    # parser.add_argument("--pdb_gt_folder", type=str, default=None,
    # help="可选，真实结构根文件夹（实验解析PDB结构目录，用于 GT 评估）。"
    #      "若提供，则在输出 JSON 中为每个样本额外添加 cif_gt_path 字段。")
    parser.add_argument("--pdb_gt_folder", type=str, default="/storage/chenzhaoyang/cryo_em/CIF_3.5_cc_qscore",
    help="可选，真实结构根文件夹（实验解析PDB结构目录，用于 GT 评估）。"
         "若提供，则在输出 JSON 中为每个样本额外添加 cif_gt_path 字段。")

    parser.add_argument("--sim_map_folder", type=str, default="/storage/penghongen/simulated_receptor_map/",
    help="可选，真实 receptor 模拟密度图根目录。若提供，则输出 JSON 中为每个样本额外添加 sim_map_path 字段。")

    parser.add_argument("--sim_map_folder_cryoatom", type=str, default="/storage/penghongen/simulated_cryoatom_map/",
    help="可选，cryoatom receptor 模拟密度图根目录。若提供，则输出 JSON 中为每个样本额外添加 sim_map_path_cryoatom 字段。")

    parser.add_argument("--labels_npz_root", type=str, default="/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/",
    help="可选，labels.npz 根目录。若提供，则输出 JSON 中为每个样本额外添加 labels_npz_path 字段。")



    # parser.add_argument("--valid_json_path", type=str, help="约束可选集合列表的 json 参照字典路径 (必须传入)",
    # default="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/test_0.json")   # test/val 在这里切换
    parser.add_argument("--valid_json_path", type=str, help="约束可选集合列表的 json 参照字典路径 (必须传入)",
    default="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/test_3.json")   # test/val 在这里切换
    # parser.add_argument("--valid_json_path", type=str, help="约束可选集合列表的 json 参照字典路径 (必须传入)", 
    # default="/home/penghongen/My_Project/Data/raw.json")



    parser.add_argument("--output_folder", type=str, help="输出.json的存放目录", 
    default="/home/penghongen/My_Project/Pocket_Plus/src/inference/utils")

    parser.add_argument("--output_json_name", type=str, help="生成的 json 文件名称", 
    default="nucleic_40.json")





    # -------------------------- 参数 --------------------------
    parser.add_argument("--max_scan", type=int, default=None, help="从全部样本里随机选取验证扫描的最多个数")
    parser.add_argument("--max_accept", type=int, default=40, help="最多写入输出 JSON 的合规样本个数")
    parser.add_argument("--min_nucleic_ratio", type=float, default=0.0001, help="核酸数目/蛋白数目的比率阈值下限(none则不限制), 允许端点")
    parser.add_argument("--max_nucleic_ratio", type=float, default=None, help="核酸数目/蛋白数目的比率阈值上限(none则不限制), 允许端点")
    parser.add_argument("--skip_density_map_validation", action="store_true", help="跳过真实密度图与模拟密度图几何一致性校验")
    parser.add_argument("--target_voxel_size", type=float, default=1.0, help="密度图重采样目标体素大小")
    parser.add_argument("--density_origin_atol", type=float, default=1e-4, help="密度图 origin 绝对误差容忍度")
    parser.add_argument("--density_voxel_rtol", type=float, default=1e-5, help="密度图 voxel_size 相对误差容忍度")
    parser.add_argument("--density_voxel_atol", type=float, default=1e-6, help="密度图 voxel_size 绝对误差容忍度")
    
    
    # Namespace,  获取解析和捕获成功后的全用户输入环境数据载体
    args = parser.parse_args()
    
    # str,  反馈成功生成 json 且最终落盘生效的主结果存储指向路径
    out_path = generate_sample_json(
        map_folder=args.map_folder,
        pdb_folder=args.pdb_folder,
        pdb_gt_folder=args.pdb_gt_folder,
        sim_map_folder=args.sim_map_folder,
        sim_map_folder_cryoatom=args.sim_map_folder_cryoatom,
        labels_npz_root=args.labels_npz_root,
        output_folder=args.output_folder,
        valid_json_path=args.valid_json_path,
        max_scan=args.max_scan,
        max_accept=args.max_accept,
        min_nucleic_ratio=args.min_nucleic_ratio,
        max_nucleic_ratio=args.max_nucleic_ratio,
        output_json_name=args.output_json_name,
        validate_density_map_pairs=not args.skip_density_map_validation,
        target_voxel_size=args.target_voxel_size,
        density_origin_atol=args.density_origin_atol,
        density_voxel_rtol=args.density_voxel_rtol,
        density_voxel_atol=args.density_voxel_atol,
    )
    print(f"✅ json 生成完毕! 输出在: {out_path}")



# NOTE：用来调参的.json——————老版本：真实受体做的密度图
"""
[Attempt] Executing dynamic command from /home/penghongen/run_cmd_255120.sh...
#!/bin/bash
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py
=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 269
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 269
  - 实际执行文件和比例探测运算的配对次数: 236
  - 缺失结构或密度图文件配对数: 156
  - 因核酸比例不合格而过滤抛弃数: 0
  - 最终保留且符合要求的有效配对数: 80
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/val_cryoatom.json
"""





# NOTE：用来调参的.json——————新版本：cryoatom 做的密度图
"""
=================================================================================
[Attempt] Executing dynamic command from /home/penghongen/run_cmd_255120.sh...
#!/bin/bash
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py
=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 269
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 269
  - 实际执行文件和比例探测运算的配对次数: 236
  - 缺失结构或密度图文件配对数: 156
  - 因核酸比例不合格而过滤抛弃数: 0
  - 最终保留且符合要求的有效配对数: 80
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/val_cryoatom_new.json
=================================================================================
[Success] Execution finished successfully.
=================================================================================
"""


# NOTE: 最终测试集.json——————纯蛋白的100个样本(含cryoAtom套)
"""
[Attempt] Executing dynamic command from /home/penghongen/run_cmd_293101.sh...

#!/bin/bash
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py

=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 269
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 269
  - 实际执行文件和比例探测运算的配对次数: 100
  - 缺失结构或密度图文件配对数: 0
  - 因核酸比例不合格而过滤抛弃数: 0
  - 最终保留且符合要求的有效配对数: 100
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/test_protein.json
"""
