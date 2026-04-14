"""
这个文件主要负责找到用于推断或评估的所有样本, 生成它们的 .json 文件.
.json 形如 list[dict], 每项含 {"cif_path": ..., "map_path": ...} 即实际的路径对, 这是 Pocket_Plus\src\inference\run.py 里面  def _run_raw_batch_mode 直接接受的.

支持两种使用场景:
- 场景A (仅推断/用真实结构做特征+评估): 每项仅含 {"cif_path": ..., "map_path": ...}
- 场景B (预测结构做特征, 真实结构做评估): 每项含 {"cif_gt_path": ..., "cif_path": ..., "map_path": ...}
  其中 cif_path 指向 AF3/CryoAtom 等建模输出，cif_gt_path 指向实验解析的真实结构

它的主要功能:

1.
输入:
-   一个密度图根文件夹 map_folder, 一个结构根文件夹 pdb_folder. 还有一个必选的.json文件, 它格式为list[dict], 每项形如{emdb_id:pdb_id} 或 {"emd_20235": "6P1H"}, 本文件可以灵活识别(见NOTE). 这个.json文件的作用是筛选有效样本的范围(不在.json里面的就忽略).
-   可选的 pdb_gt_folder: 真实结构根文件夹（如原始PDB结构目录）。若提供，则在输出 dict 中额外添加 cif_gt_path。
-   用户给定一个可选的扫描数目限制 max_scan(只随机取出max_scan个进行后面的筛选条件检查) 和可选的最终接受数目限制 max_accept; 一系列可选的筛选条件(目前只支持 nucleic_ratio=非HETATOM的氨基酸残基数目/ 非HETATOM的核酸残基数目)
-   output_folder: 存放生成的.json文件的文件目录

输出:
-   .json 文件; list[dict], 每项含 {"cif_path": ..., "map_path": ...}, 当提供 pdb_gt_folder 且找到对应文件时，额外含 {"cif_gt_path": ...}

NOTE:
-   从提供的.json识别出的emdb_id只关注数字部分(如emd_4651和EMDB-4651.map效果等同, 匹配时也只看数字); pdb_id也不计大小写.
"""

import os
import json
import random
import re
import sys
from pathlib import Path
random.seed(42)

# Pocket_Plus 根目录
_CURRENT_DIR = Path(__file__).resolve().parent
_Pocket_Plus_ROOT = _CURRENT_DIR.parent.parent.parent
if str(_Pocket_Plus_ROOT) not in sys.path:
    sys.path.insert(0, str(_Pocket_Plus_ROOT))
from Bio.PDB import PDBParser, MMCIFParser
# 导入已经定义好的常量配置字典
from Make_Data.PDB_processor.config import AMINO_ACIDS, NUCLEOTIDES, DNA_NUCLEOTIDES, MODIFIED_RESIDUE_TO_PARENT

# --------------------------- 工具函数 --------------------------
def load_raw_pairs(raw_pairs: str | list[dict]) -> list[tuple[str, str, str | None]]:
    """
    将 raw_pairs 配置统一解析为 `(cif_path, map_path, cif_gt_path)` 三元组列表。
    """
    if not raw_pairs:
        raise ValueError("raw_pairs/raw_pairs_json 不能为空。")

    if isinstance(raw_pairs, str):
        if not raw_pairs.endswith(".json"):
            raise ValueError(f"raw_pairs 若为字符串，当前仅支持 JSON 文件路径，收到: {raw_pairs}")
        with open(raw_pairs, "r", encoding="utf-8-sig") as file_obj:
            raw_pairs = json.load(file_obj)

    if not isinstance(raw_pairs, list):
        raise TypeError(f"raw_pairs 必须是 JSON 路径或 list[dict]，当前类型为: {type(raw_pairs)}")

    pairs: list[tuple[str, str, str | None]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise TypeError(f"raw_pairs 的每一项都必须是 dict，当前收到: {type(item)}")
        if "cif_path" not in item or "map_path" not in item:
            raise KeyError("raw_pairs 的每一项都必须包含 `cif_path` 和 `map_path`。")
        pairs.append((item["cif_path"], item["map_path"], item.get("cif_gt_path", None)))

    return pairs

# str, 支持的结构文件扩展名集合
_STRUCTURE_EXTS = {".cif", ".pdb", ".mmcif"}

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






# ----------- 总函数 -----------
def generate_sample_json(
    map_folder: str=None,
    pdb_folder: str=None,
    pdb_gt_folder: str=None,   # 可选，真实结构根文件夹（如原始PDB目录），若提供则在输出 dict 中添加 cif_gt_path
    output_folder: str=None,
    valid_json_path: str = None,   # /home/penghongen/My_Project/Data/raw.json, 单条目形如: {"emd_63092": "9LHB"}

    max_scan: int = None,
    max_accept: int = None,
    min_nucleic_ratio: float = None,
    max_nucleic_ratio: float = None,
    output_json_name: str = "raw_samples.json"
) -> str:
    """
    生成推断和评估读取原始样本的 JSON 路径映射文件。

    输入参数:
        - map_folder:      str, 标量, 表示密度图所在的根文件夹绝对或相对路径
        - pdb_folder:      str, 标量, 表示蛋白核酸等结构文件所在的根文件夹(用于提取模型输入特征)
        - pdb_gt_folder:   str | None, 标量, 可选，真实结构根文件夹（用于提取 GT 标签评估）。
                           若提供，则在输出 dict 中为每个样本额外添加 "cif_gt_path" 键（若在该目录找到同名文件）。
        - output_folder:   str, 标量, 表示最后生成的json文件要存放的目录
        - valid_json_path: str, 标量, 必选的范围界定json文件(也提供emdb-pdb映射), 单条目形如: {"emd_63092": "9LHB"}

        - max_scan:        int, 标量, 代表从初始候选集合里随机取出验证的最多数目。默认为 None
        - max_accept:      int, 标量, 可选参数，代表最终保留进入生成的 json 的组合对数量上限。默认为 None
        - min_nucleic_ratio: float, 标量, 核酸数量 / 氨基酸数量（排除HETATM异构物与配体）下界截断。默认为 None
        - max_nucleic_ratio: float, 标量, 核酸数量 / 氨基酸数量（排除HETATM异构物与配体）上界截断。默认为 None
        - output_json_name: str, 标量, 生成 json 目标字典的默认具体文件名。默认为 "raw_samples.json"

    输出:
        - output_json_path: str, .json 文件的绝对路径, 内容为 List[dict[str,str]], 每项为 {"cif_gt_path": cif_gt_path, "cif_path": cif_path, "map_path": map_path} 或 {"cif_path": cif_path, "map_path": map_path}
    """
    # list, (N,), 用来预先存储后续所有可匹配探测配对；其内部的每个元素的形状均为 (2,) 的元组，分别代表(emdb_id, pdb_id)~(数字,小写字母)的元组
    candidate_pairs = []
    print(f"---------- 开始生成样本 JSON 文件 ----------")
    if valid_json_path is None:
        raise ValueError("必须提供 valid_json_path，以界定可用样本的 json 清单范围。")
    with open(valid_json_path, "r", encoding="utf-8") as f_json:
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



    # dict, str:str, 表示 这个map的数字id:对应全路径
    map_dict = {}
    for m_file in os.listdir(map_folder):
        m_lower = m_file.lower()
        if m_lower.endswith(".map") or m_lower.endswith(".mrc"):
            # Match, 标量, 提取文件名里的数字
            num_match = re.search(r'\d+', m_file)
            if num_match:
                # str, 标量, 获取的纯数字ID
                num_str = num_match.group()
                if num_str not in map_dict:
                        map_dict[num_str] = os.path.join(map_folder, m_file)

    # dict, str:str, 表示 这个结构文件的小写id:对应全路径（用于提取模型输入特征的结构，即 pdb_folder 中的文件, 兼容扁平和嵌套布局）
    pdb_dict = _scan_structure_dir(pdb_folder)
    pdb_gt_dict = _scan_structure_dir(pdb_gt_folder) if (pdb_gt_folder is not None) else {}





    # list, (A,), 最终的路径结构列表
    accepted_samples = []

    # int, 标量, 用于记录实际进入检查过程的候选组合对数
    actual_scanned_count = 0
    # int, 标量, 目前已被验证和保留进 accepted_samples 里的个数计时器
    accepted_count = 0
    # int, 标量, 记录因缺失对应的 map 或 pdb 等文件而跳过的对数统计
    missing_files_count = 0
    # int, 标量, 记录因蛋白与核酸比例不在要求范围内而抛弃的对数统计
    filtered_out_count = 0

    

    os.makedirs(output_folder, exist_ok=True)
    for pair in candidate_pairs:
        if max_accept is not None and accepted_count >= max_accept:
            break
        actual_scanned_count += 1
            
        # str, 已提取为纯数字
        emdb_id = pair[0]
        # str, 已统一转小写
        pdb_id = pair[1].lower()
        # str/None, 标量, 尝试从预设好的按数字索引的字典内直接获取 map 绝对路径
        map_path = map_dict.get(emdb_id)
        if map_path is None or not os.path.exists(map_path):
            missing_files_count += 1
            continue
        # str/None, 标量, 尝试从预设好的通过小写全名索引的结构字典里面获取真正的 pdb/cif 途径
        cif_path = pdb_dict.get(pdb_id)
        if cif_path is None or not os.path.exists(cif_path):
            missing_files_count += 1
            continue
        cif_gt_path = None
        if pdb_gt_folder is not None:
            cif_gt_path = pdb_gt_dict.get(pdb_id)
            if cif_gt_path is None or not os.path.exists(cif_gt_path):
                missing_files_count += 1
                continue



        if (min_nucleic_ratio is not None) or (max_nucleic_ratio is not None):
            nucleic_ratio = get_nucleic_ratio(cif_path)
            if nucleic_ratio < 0:
                filtered_out_count += 1
                continue
            ratio_valid = True
            if min_nucleic_ratio is not None and nucleic_ratio < min_nucleic_ratio:
                ratio_valid = False
            if max_nucleic_ratio is not None and nucleic_ratio > max_nucleic_ratio:
                ratio_valid = False
                
            if not ratio_valid:
                filtered_out_count += 1
                continue
        acc_dict = {
            "cif_path": cif_path,
            "map_path": map_path
        }
        if pdb_gt_folder is not None:
            acc_dict = {
                "cif_gt_path": cif_gt_path,
                "cif_path":    cif_path,
                "map_path":    map_path,
            }
        accepted_samples.append(acc_dict)
        accepted_count += 1
        
    print(f"---------- 扫描完成，统计信息 ----------")
    print(f"  - 初始检查候选配对总数: {num_candidates}")
    print(f"  - 实际执行文件和比例探测运算的配对次数: {actual_scanned_count}")
    print(f"  - 缺失结构或密度图文件配对数: {missing_files_count}")
    print(f"  - 因核酸比例不合格而过滤抛弃数: {filtered_out_count}")
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
    default="/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc")   # 更全的在: /storage/chenzhaoyang/cryo_em/EMDB, 用于生成for_cryoAtom_hard  !! 目前映射关系不全，无法真正完成！！


    parser.add_argument("--pdb_folder", type=str, help="结构根目录（用于提取模型输入特征，如 AF3/CryoAtom 输出目录）", 
    default="/storage/chenzhaoyang/cryo_em/PDB_3.5_cc_qscore")
    # parser.add_argument("--pdb_folder", type=str, help="结构根目录", 
    # default="/storage/chenzhaoyang/cryo_em/result_split")



    parser.add_argument("--pdb_gt_folder", type=str, default=None,
    help="可选，真实结构根文件夹（实验解析PDB结构目录，用于 GT 评估）。"
         "若提供，则在输出 JSON 中为每个样本额外添加 cif_gt_path 字段。")
    # parser.add_argument("--pdb_gt_folder", type=str, default="/storage/chenzhaoyang/cryo_em/PDB_3.5_cc_qscore",
    # help="可选，真实结构根文件夹（实验解析PDB结构目录，用于 GT 评估）。"
    #      "若提供，则在输出 JSON 中为每个样本额外添加 cif_gt_path 字段。")



    parser.add_argument("--valid_json_path", type=str, help="约束可选集合列表的 json 参照字典路径 (必须传入)", 
    default="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/test.json")   # test/val 在这里切换
    # parser.add_argument("--valid_json_path", type=str, help="约束可选集合列表的 json 参照字典路径 (必须传入)", 
    # default="/home/penghongen/My_Project/Data/raw.json")



    parser.add_argument("--output_folder", type=str, help="输出.json的存放目录", 
    default="/home/penghongen/My_Project/Pocket_Plus/src/inference/utils")

    parser.add_argument("--output_json_name", type=str, help="生成的 json 文件名称", 
    default="for_v1_protein_eval.json")





    # -------------------------- 参数 --------------------------
    parser.add_argument("--max_scan", type=int, default=None, help="从全部样本里随机选取验证扫描的最多个数")
    parser.add_argument("--max_accept", type=int, default=100, help="只要扫描出这么多合规的就算成功并且退出运行")
    parser.add_argument("--min_nucleic_ratio", type=float, default=None, help="核酸数目/蛋白数目的比率阈值下限(none则不限制), 允许端点")
    parser.add_argument("--max_nucleic_ratio", type=float, default=0.0, help="核酸数目/蛋白数目的比率阈值上限(none则不限制), 允许端点")
    
    
    # Namespace,  获取解析和捕获成功后的全用户输入环境数据载体
    args = parser.parse_args()
    
    # str,  反馈成功生成 json 且最终落盘生效的主结果存储指向路径
    out_path = generate_sample_json(
        map_folder=args.map_folder,
        pdb_folder=args.pdb_folder,
        pdb_gt_folder=args.pdb_gt_folder,
        output_folder=args.output_folder,
        valid_json_path=args.valid_json_path,
        max_scan=args.max_scan,
        max_accept=args.max_accept,
        min_nucleic_ratio=args.min_nucleic_ratio,
        max_nucleic_ratio=args.max_nucleic_ratio,
        output_json_name=args.output_json_name
    )
    print(f"✅ json 生成完毕! 输出在: {out_path}")



# NOTE：用来调参的.json, 对核酸比例无要求, 先跑这个找最优参数, 然后再跑下面的(基准.json为"/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/val.json")
"""
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py
    
=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 268
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 268
  - 实际执行文件和比例探测运算的配对次数: 100
  - 缺失结构或密度图文件配对数: 0
  - 因核酸比例不合格而过滤抛弃数: 0
  - 最终保留且符合要求的有效配对数: 100
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/for_v1_search.json
=================================================================================
"""


# NOTE: 纯蛋白测试样本(基准.json为"/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/test.json")
"""
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py
    
=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 268
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 268
  - 实际执行文件和比例探测运算的配对次数: 60
  - 缺失结构或密度图文件配对数: 0
  - 因核酸比例不合格而过滤抛弃数: 10
  - 最终保留且符合要求的有效配对数: 50
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/for_v1_protein_eval.json
=================================================================================
"""


# NOTE: 核酸 / 蛋白 > 0.1(基准.json为"/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v0/test.json")
"""
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py
    
=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 268
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 268
  - 实际执行文件和比例探测运算的配对次数: 268
  - 缺失结构或密度图文件配对数: 0
  - 因核酸比例不合格而过滤抛弃数: 255
  - 最终保留且符合要求的有效配对数: 13
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/for_v1_nucleic_eval.json
=================================================================================
"""





# NOTE: cryoAtom的测试(要求cc<3.5A)
"""
python /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/yield_json_from_raw_sample.py
    
=================================================================================
---------- 开始生成样本 JSON 文件 ----------
提取到的初始候选配对总数: 268
---------- 扫描完成，统计信息 ----------
  - 初始检查候选配对总数: 268
  - 实际执行文件和比例探测运算的配对次数: 268
  - 缺失结构或密度图文件配对数: 221
  - 因核酸比例不合格而过滤抛弃数: 0
  - 最终保留且符合要求的有效配对数: 47
✅ json 生成完毕! 输出在: /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/for_v1_cryoatom_eval.json
=================================================================================
"""





