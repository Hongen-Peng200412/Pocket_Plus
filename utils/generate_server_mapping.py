import os
import csv
import json
import re
from typing import List, Optional

def get_emdb_num(emdb_id: str) -> str:
    """
    从 EMDB 标识符中提取数字部分。

    输入参数:
        - emdb_id: str, 标量, 原始 EMDB 编号字符串, 如 'EMD-63092'

    输出:
        - num_str: str, 标量, 提取出的数字字符串, 如 '63092'; 若未找到则返回空字符串
    """
    # list[str], 可变长度, 匹配到的数字列表
    nums = re.findall(r'\d+', emdb_id)
    return nums[0] if nums else ""

def get_files_with_suffix(directory: str, suffix: str) -> List[str]:
    """
    获取指定目录下具有特定后缀的所有文件名。

    输入参数:
        - directory: str, 标量, 需要扫描的目录路径
        - suffix: str, 标量, 文件后缀名, 如 '.map'

    输出:
        - file_list: list[str], 可变长度, 匹配的文件名列表
    """
    if not os.path.exists(directory):
        print(f"Warning: 目录未找到: {directory}")
        return []
    
    # list[str], 可变长度, 过滤后的文件名列表
    return [f for f in os.listdir(directory) if f.endswith(suffix)]

def find_file(files_list: List[str], pdb_id: str, emdb_num: str) -> Optional[str]:
    """
    根据 PDB ID 或 EMDB 编号在一组文件命中寻找匹配项。

    输入参数:
        - files_list: list[str], 可变长度, 可选的文件名列表
        - pdb_id: str, 标量, 小写的 PDB 标识, 长度通常为 4
        - emdb_num: str, 标量, EMDB 纯数字编号

    输出:
        - matched_file: str 或 None, 标量, 匹配到的文件名; 若未找到则返回 None
    """
    for f in files_list:
        if pdb_id and (pdb_id.lower() in f.lower()):
            return f
            
    if emdb_num:
        for f in files_list:
            if emdb_num in f:
                return f
                
    return None

def generate_mapping(csv_path: str,
                     cif_dir: str,
                     emdb_dir: str,
                     simu_atom_dir: str,
                     simu_all_dir: str,
                     out_json_path: str) -> None:
    """
    读取 CSV 文件, 在对应的文件夹中匹配相关的文件, 最终生成 JSON 映射文件。

    输入参数:
        - csv_path: str, 标量, 包含文件映射信息的 CSV 路径
        - cif_dir: str, 标量, 真实结构文件所在目录
        - emdb_dir: str, 标量, EMDB 密度文件所在目录
        - simu_atom_dir: str, 标量, 受体模拟密度图文件所在目录
        - simu_all_dir: str, 标量, 整体模拟密度图所在目录
        - out_json_path: str, 标量, 生成的 JSON 保存位置

    输出:
        - 无返回值, 将结果写入磁盘对应的 JSON 文件
    """
    if not os.path.exists(csv_path):
        print(f"Error: 找不到指定的 CSV 映射文件: {csv_path}")
        return

    # list[dict[str, str]], 可变长度, 从原 CSV 文件抽取的有效样本标识 
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # csv.DictReader, 无固定形状, CSV 阅读器对象
        reader = csv.DictReader(f)
        for row in reader:
            # str, 标量, 原始 EMDB 编号
            emdb_id = row.get('emdb_id', '')
            # str, 标量, PDB ID 转为小写去空格
            pdb_id = row.get('fitted_pdbs', '').split(",")[0].strip().lower()
            if emdb_id and pdb_id:
                samples.append({
                    'emdb_num': get_emdb_num(emdb_id),
                    'pdb_id': pdb_id,
                    'raw_emdb_id': emdb_id
                })

    print(f"解析到 {len(samples)} 个目标对应关系，现在开始列举目录...")
    
    # list[str], 可变长度, 对应目录下的后缀文件列表
    cif_files = get_files_with_suffix(cif_dir, '.cif')
    emdb_files = get_files_with_suffix(emdb_dir, '.map')
    sim_atom_files = get_files_with_suffix(simu_atom_dir, '.mrc')
    sim_all_files = get_files_with_suffix(simu_all_dir, '.mrc')

    # --- 诊断输出: 帮助排查文件名格式与 PDB/EMDB 编号之间的对应关系 ---
    print(f"CIF 目录下共 {len(cif_files)} 个文件, 前5个文件名: {cif_files[:5]}")
    print(f"EMDB 目录下共 {len(emdb_files)} 个文件, 前5个文件名: {emdb_files[:5]}")
    print(f"SimAtom 目录下共 {len(sim_atom_files)} 个文件, 前5个文件名: {sim_atom_files[:5]}")
    print(f"SimAll 目录下共 {len(sim_all_files)} 个文件, 前5个文件名: {sim_all_files[:5]}")
    print(f"CSV 中前3个样本: {samples[:3]}")

    # list[dict[str, str]], 可变长度, 符合全部要求(有三项必备文件)的样本列表
    valid_samples = []
    # int, 标量, 各类匹配统计
    cif_hit = 0
    map_hit = 0
    sim_atom_hit = 0
    print("开始匹配文件...")
    for s in samples:
        # str, 标量, 提取暂存变量
        pdb = s['pdb_id']
        enum = s['emdb_num']

        # str或None, 标量, 按编号进行字符串匹配找出的文件名
        cif_f = find_file(cif_files, pdb, enum)
        map_f = find_file(emdb_files, pdb, enum)
        sim_atom_f = find_file(sim_atom_files, pdb, enum)
        sim_all_f = find_file(sim_all_files, pdb, enum)

        if cif_f: cif_hit += 1
        if map_f: map_hit += 1
        if sim_atom_f: sim_atom_hit += 1

        # bool, 标量, 只有当结构、密度图、受体模拟密度图全都在时，才认为是完整的
        if cif_f and map_f and sim_atom_f:
            valid_samples.append({
                "pdb_id": pdb,
                "emdb_id": s['raw_emdb_id'],
                "cif_path": os.path.join(cif_dir, cif_f),
                "map_path": os.path.join(emdb_dir, map_f),
                "sim_cif_path": os.path.join(simu_atom_dir, sim_atom_f),
                "sim_all_path": os.path.join(simu_all_dir, sim_all_f) if sim_all_f else ""
            })

    print(f"匹配统计: CIF命中={cif_hit}, MAP命中={map_hit}, SimAtom命中={sim_atom_hit} (共{len(samples)}个样本)")

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_samples, f, indent=4, ensure_ascii=False)

    print(f"匹配完成! 共生成 {len(valid_samples)} 个完整的样本。已写入至: {out_json_path}")

if __name__ == "__main__":
    generate_mapping(
        csv_path="/storage/penghongen/EMDB_PDB_resolution_3.5.csv",  # 请务必确保此文件包含 emdb_id 和 fitted_pdbs 列，而不是只含有 real_file 列
        cif_dir="/storage/chenzhaoyang/cryo_em/CIF_3.5_cc_qscore",
        emdb_dir="/storage/chenzhaoyang/cryo_em/EMDB_3.5",
        simu_atom_dir="/storage/chenzhaoyang/cryo_em/EMDB_simu_atom",
        simu_all_dir="/storage/chenzhaoyang/cryo_em/EMDB_simu",
        out_json_path="dataset_mapping.json"
    )

"""
conda activate Pocket_Plus_centos7_cu121_allgpu
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/utils/generate_server_mapping.py
(base) [penghongen@master ~]$ conda activate Pocket_Plus_centos7_cu121_allgpu
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python /home/penghongen/My_Project/Pocket_Plus/utils/generate_server_mapping.py
解析到 11036 个目标对应关系，现在开始列举目录...
CIF 目录下共 6544 个文件, 前5个文件名: ['6TAY.cif', '8V3U.cif', '9OJU.cif', '8BEJ.cif', '7NVV.cif']
EMDB 目录下共 11008 个文件, 前5个文件名: ['emd_60857.map', 'emd_26879.map', 'emd_10632.map', 'emd_29414.map', 'emd_50173.map']
SimAtom 目录下共 10037 个文件, 前5个文件名: ['emd_51149.mrc', 'emd_26941.mrc', 'emd_33671.mrc', 'emd_47360.mrc', 'emd_47554.mrc']
SimAll 目录下共 10989 个文件, 前5个文件名: ['emd_51149.mrc', 'emd_26941.mrc', 'emd_33671.mrc', 'emd_47360.mrc', 'emd_47554.mrc']
CSV 中前3个样本: [{'emdb_num': '63092', 'pdb_id': '9lhb', 'raw_emdb_id': 'EMD-63092'}, {'emdb_num': '48793', 'pdb_id': '9n0t', 'raw_emdb_id': 'EMD-48793'}, {'emdb_num': '61836', 'pdb_id': '9jv1', 'raw_emdb_id': 'EMD-61836'}]
开始匹配文件...
匹配统计: CIF命中=6544, MAP命中=11007, SimAtom命中=10356 (共11036个样本)
匹配完成! 共生成 6193 个完整的样本。已写入至: dataset_mapping.json
(Pocket_Plus_centos7_cu121_allgpu) [penghongen@master ~]$ 

"""
