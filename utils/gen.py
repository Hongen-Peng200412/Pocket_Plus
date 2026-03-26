import json
import os
import random
import pandas as pd
from pathlib import Path

# str, EMDB-PDB 映射 csv 路径
EMDB_TO_PDB_CSV_PATH = "/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv"
EMDB_COLUMN_NAME = 'emdb_id'
PDB_COLUMN_NAME = 'fitted_pdbs'


def make_EMDB_PDB_dict(emdb_to_pdb_csv_path=EMDB_TO_PDB_CSV_PATH, emdb_column_name=EMDB_COLUMN_NAME, pdb_column_name=PDB_COLUMN_NAME, 
                        add_prefix_of_emdb: str='emd_', upper_lower_of_pdb: str='upper', 
                        return_pdb_mode: str='Dict[str, str]', 
                        exclude_pdb_list: list[str] = None):
    """ 返回 EMDB-PDB 字典.返回格式为 return_pdb_mode
    - return_pdb_mode: 'Dict[str, List[str]]' 或 'Dict[str, str]', 对应 {'EMD-611081':['9j37', ...], 'EMD-...':['...']} 和 {'EMD-611081':'9j37', 'EMD-...':'...'}(一个EMDB可能对应多个pdb, 但按照后者的模式, 在返回时只选第一个pdb)
    - prefix_of_emdb: 注意, 总是同时支持关于EMDB_ID的好几种格式(按'-'或'_'划分取最后). 但返回的字典中的key = prefix_of_emdb + emdb_id(数字部分)
    - upper_lower_of_pdb: 'upper' 或 'lower', 控制返回字典中 PDB ID 的大小写
    - exclude_pdb_list: list[str], 需要排除的pdb的文件名, 如9j37. 如果一个EMDB的对应PDB列表与exclude_pdb_list的交集非空, 则不返回该EMDB对应的键值对.注意这是在应用 upper_lower_of_pdb调整了emdb_to_pdb_csv_path之后再检测的.
    """
    df_dict = pd.read_csv(emdb_to_pdb_csv_path)

    # 检查第一行数据的格式来确定emdb_id的分隔符
    if len(df_dict) == 0:
        raise ValueError("CSV文件为空")
    sample_id = df_dict[emdb_column_name].iloc[0]
    if exclude_pdb_list is None:
        exclude_pdb_set = set()
    else:
        exclude_pdb_set = set(exclude_pdb_list)

    if '-' in str(sample_id):   # 如'...-61081'
        df_dict['temp_emdb_id'] = df_dict[emdb_column_name].apply(   # 对每一列(如emd-61081或EMDB-61081)使用
            lambda x:   add_prefix_of_emdb + x.split('-')[-1]
        )
    elif '_' in str(sample_id):   # 如'..._61081'
        df_dict['temp_emdb_id'] = df_dict[emdb_column_name].apply(
            lambda x:   add_prefix_of_emdb + x.split('_')[-1]
        )
    else:
        raise ValueError(f"EMDB_ID 格式无法识别: {sample_id}")


    if return_pdb_mode == 'Dict[str, List[str]]':
        if upper_lower_of_pdb == "upper":
            df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(   # 对每一列(如['9j37', '9j38'])使用
                lambda x: [item.strip().upper() for item in str(x).split(',')]    if ',' in str(x)
                else [str(x).upper()]
            )
        elif upper_lower_of_pdb == "lower":
            df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(   # 对每一列(如['9j37', '9j38'])使用
                lambda x: [item.strip().lower() for item in str(x).split(',')]    if ',' in str(x)
                else [str(x).lower()]
            )
        df_dict['temp_pdbs_list'] = df_dict['temp_pdbs_list'].apply(lambda x: x if set(x).isdisjoint(exclude_pdb_set) else None)  # 过滤

    elif return_pdb_mode == 'Dict[str, str]':
        if upper_lower_of_pdb == "upper":
            df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(
                lambda x: str(x).split(',')[0].strip().upper()    if ',' in str(x)
                else str(x).upper()
            )
        elif upper_lower_of_pdb == 'lower':
            df_dict['temp_pdbs_list'] = df_dict[pdb_column_name].apply(
                lambda x: str(x).split(',')[0].strip().lower()    if ',' in str(x)
                else str(x).lower()
            )
        df_dict['temp_pdbs_list'] = df_dict['temp_pdbs_list'].apply(lambda x: x if x not in exclude_pdb_set else None)
    else:
        raise ValueError("return_pdb_mode 格式无法识别")


    processed_dict = dict(zip(df_dict['temp_emdb_id'], df_dict['temp_pdbs_list']))
    filtered_dict = {}
    for emdb_id, pdb in processed_dict.items():
        if pdb is not None:
            filtered_dict[emdb_id] = pdb

    return filtered_dict


def dict_to_list_of_single_dicts(data_dict: dict[str, str]) -> list[dict[str, str]]:
    """
    将包含多个键值对的字典Dict[str, str]转换为包含多个单键值对字典的列表List[Dict[str, str]]
    - data_dict: dict[str, str], 形状为 {k1:v1, k2:v2, ...}, 原始存储 EMDB-PDB 对应关系的字典[{k1:v1}, {k2:v2}, ...]
    输出:
    - list[dict[str, str]]: list, 形状为 (N,), 长度为 N 的列表，每个元素是形如 {key: value} 的小字典
    """
    return [{k: v} for k, v in data_dict.items()]


def save_json(data, path: Path) -> None:
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {path.name} ({len(data)} 样本)")


if __name__ == '__main__':
    raw_json = make_EMDB_PDB_dict()
    raw_json = dict_to_list_of_single_dicts(raw_json)
    save_json(raw_json, Path("/home/penghongen/My_Project/Data/raw.json"))