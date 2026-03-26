"""
划分逻辑:
1. 我们要构造三种数据集, 对应于三个划分文件夹(Split_output_FOLDER)。 每个文件夹含有 "包含.json文件的子文件夹"————它们目前名称将会是：metal_ion, small_molecule, peptide, nucleic(来自 Pocket\processedPDB_EMDB_binder\bind.py 和 Pocket\processedPDB_EMDB_binder\split_and_select_box.py 的保存逻辑) 每个子文件夹对应 "关于这个数据集、针对这种配体的划分".
2. 目前, 文件夹 split_0 对应的数据集是: 只包含所有以"_C"为结尾的样本, 对应所有结合位点的"中心摄影图"; 
         文件夹 split_3 对应的数据集是: 包含: 所有以"_C"为结尾的样本 + 每个结合位点额外随机选取的3个BOX; 
         文件夹 split_all 对应的数据集是: 所有样本.
3. .json格式形如["9e01_3_0_0_0_C", ...], 它是去后缀的文件名. 某个条目在.json列表内当且仅当这个文件同时存在于EMDB_BOX_FOLDER, PDB_Feaure_BOX_FOLDER, PBD_Label_BOX_FOLDER中.

Note:
(1). 实际操作中，就是依据 Instance_Json = ".../all.json" 来遍历/过滤的，也就是说 Instance_Json = ".../all.json" 相当于"最全的emdb_pdb"映射, 不在里面的样本都忽略.
(2). 本脚本产生的.json与 Pocket\Make_Data\split_data\generate_full_json.py 完全对齐.
具体而言, Json_Root_Folder 这个文件夹里面的.json文件完全由 Pocket\Make_Data\split_data\generate_full_json.py 产生.
本脚本将读取 Json_Root_Folder 里面所有的.json文件, 然后在 Split_output_FOLDER 里面每个子文件夹里生成完全相同的.json. 某个BOX被划分到了 Split_output_FOLDER 里面的X.json 当且仅当 这个BOX所在的样本在 Json_Root_Folder 里面的X.json中.
"""
import json
import os
import random
import pandas as pd
from pathlib import Path

# ============================================================================
# 配置 / Configuration
# ============================================================================

random.seed(42)

# str, Json根目录, 其中包含从Make_Data生成的各式X.json，如train.json, val.json, test.json等, 这些json规定了PDB_ID的归属，本脚本将完全遵照其分配逻辑
Json_Root_Folder = "/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/"
# str, 全部原样本的基础信息json(通常是 Json_Root_Folder 目录下的 all.json)
Instance_Json = os.path.join(Json_Root_Folder, "all.json")

# 各种切好的BOX的存放文件夹, 注意它们都有子文件夹(metal_ion, small_molecule, peptide, nucleic), 将全部遍历处理
EMDB_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v_1/emdb_BOX"
PDB_Feaure_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v_1/pdb_feature_BOX"
PBD_Label_BOX_FOLDER = "/storage/penghongen/Pocket_classic/v_1/pdb_label_BOX"

# list[int], 三种 split_mode 参数集合 （0为主中心，3为中心外加三非中心BOX, -1为全部)
Split_Modes = [0, 3, -1]
# list[str], 指定上方的 split_modes 所对应的生成目标子文件夹名
Split_Mode_Names = ['split_0', 'split_3', 'split_all']
# list[str], 指定上方的 split_modes 所对应的生成目标子文件夹名
Split_output_FOLDER = [
    "/storage/penghongen/Pocket_classic/v_1/split/split_0", 
    "/storage/penghongen/Pocket_classic/v_1/split/split_3", 
    "/storage/penghongen/Pocket_classic/v_1/split/split_all"
]




# 下面是或许会用到的工具函数
def save_json(data, path: Path) -> None:
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {path.name} ({len(data)} 样本)")







# 新增工具函数
def collect_box_stems_per_class(folder: str) -> dict:
    """
    遍历 BOX 文件夹下的所有类别子文件夹, 按类别分别收集 .npz 文件的去后缀名(stem).
    folder的文件结构: folder/{class_name}/{pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C].npz

    输入:
        - folder: str, BOX 根文件夹路径 (内含 metal_ion/peptide 等类别子文件夹)
    输出:
        - class_stems: dict[str, set[str]], key=类别名(如 'metal_ion'), value=该类别下所有 stem 的集合
    """
    # dict[str, set[str]], key=class_name, value=该类别下所有stem集合
    class_stems = {}
    for class_name in os.listdir(folder):
        # str, 类别子文件夹的完整路径
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        # set[str], 当前类别下所有 stem
        stems = set()
        for fname in os.listdir(class_dir):
            # str, 去掉后缀的文件名, 如 '9e01_3_0_0_0_C'
            stem = os.path.splitext(fname)[0]
            stems.add(stem)
        class_stems[class_name] = stems
    return class_stems


def get_valid_stems_per_class(
    emdb_folder: str,
    feature_folder: str,
    label_folder: str,
) -> dict:
    """
    对三个 BOX 文件夹按类别分别取三方交集, 返回每个类别下公共合法的 stem 集合.

    输入:
        - emdb_folder:    str, EMDB BOX 根文件夹
        - feature_folder: str, PDB Feature BOX 根文件夹
        - label_folder:   str, PDB Label BOX 根文件夹
    输出:
        - valid_per_class: dict[str, set[str]], key=类别名(如metal_ion), value=该类别下三方公共 stem 集合
    """
    # dict[str, set[str]], 三个文件夹各自的分类别 stem 字典
    emdb_cls    = collect_box_stems_per_class(emdb_folder)
    feature_cls = collect_box_stems_per_class(feature_folder)
    label_cls   = collect_box_stems_per_class(label_folder)
    # set[str], 三个文件夹中同时出现的类别名
    all_classes = set(emdb_cls.keys()) & set(feature_cls.keys()) & set(label_cls.keys())

    # dict[str, set[str]], 对每个类别取三方交集
    valid_per_class = {}
    for cls in sorted(all_classes):
        # set[str], 当前类别在三个文件夹中均存在的 stem
        valid_per_class[cls] = emdb_cls[cls] & feature_cls[cls] & label_cls[cls]
        print(f"    [{cls}] EMDB:{len(emdb_cls[cls])}  Feature:{len(feature_cls[cls])}  Label:{len(label_cls[cls])}  交集:{len(valid_per_class[cls])}")
    return valid_per_class


def get_pdb_id(stem: str) -> str:
    """
    从 BOX 文件名 stem 中提取 pdb_id.
    stem 格式: {pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C], pdb_id 固定为第一个下划线前的部分 (4位小写字母, 如 '9e01').

    输入:
        - stem: str, 如 '9e01_3_0_12_-6_C'
    输出:
        - str, pdb_id, 如 '9e01'
    """
    return stem.split('_')[0]


def get_binding_site(stem: str) -> str:
    """
    从 BOX 文件名 stem 中提取结合位点标识 ({pdb_id}_{inst_id}).
    stem 格式: {pdb_id}_{inst_id}_{Rxx}_{Ryy}_{Rzz}[_C], inst_id 是第二个下划线分隔的整数字符串 (如 '3').

    输入:
        - stem: str, 如 '9e01_3_0_12_-6_C'
    输出:
        - str, '{pdb_id}_{inst_id}', 如 '9e01_3'
    """
    # list[str], 按下划线切分
    parts = stem.split('_')
    # str, 结合位点标识 = pdb_id + '_' + inst_id
    return parts[0] + '_' + parts[1]


def select_split_stems(all_stems: list, split_mode: int) -> list:
    """
    根据 split_mode 从 all_stems 中筛选出该数据集类型对应的样本子集.

    输入:
        - all_stems:  list[str], 所有合法 BOX stem 的列表 (已过滤交集+Instance_Json)
        - split_mode: int, 0 = 只含 _C; k(k>=1) = _C + 每结合位点至多k个非中心; -1 = 全部
    输出:
        - selected: list[str], 筛选后的 stem 列表 (split_mode=3 时有随机性)
    """
    if split_mode == -1:
        # list[str], 全部样本, 直接返回副本
        return list(all_stems)

    if split_mode == 0:
        # list[str], 只保留以 '_C' 结尾的中心 BOX
        return [s for s in all_stems if s.endswith('_C')]

    if split_mode >= 1:
        # dict[str, list[str]], 按结合位点分组; key='pdb_id_inst_id', value=stem列表
        site_to_center = {}      # dict[str, list[str]], 中心 BOX
        site_to_nonCenter = {}   # dict[str, list[str]], 非中心 BOX
        for stem in all_stems:
            # str, 当前 stem 的结合位点标识
            site = get_binding_site(stem)
            if stem.endswith('_C'):
                # setdefault(key, default) 它是字典（dict）的一个内置方法，逻辑如下：
                # 如果键（key）已经在字典里了： 它会返回该键对应的值（Value）。
                # 如果键（key）不在字典里： 它会把这个键插入字典，并将它的值设为 default，然后返回这个新创建的 default 值。
                site_to_center.setdefault(site, []).append(stem)
            else:
                site_to_nonCenter.setdefault(site, []).append(stem)

        # list[str], 最终筛选结果
        selected = []
        # 先把全部中心 BOX 加入
        for stems_list in site_to_center.values():
            selected.extend(stems_list)
        # 再对【有中心BOX的】结合位点随机抽取至多 split_mode 个非中心 BOX(没有 _C 的结合位点不在 site_to_center 中, 不会被额外附加非中心 BOX)
        for site, nc_list in site_to_nonCenter.items():
            if site not in site_to_center:
                # 该结合位点无中心 BOX, 跳过（不额外附加）
                continue
            # int, 实际抽取数量 (不足split_mode个则全取)
            k = min(split_mode, len(nc_list))
            selected.extend(random.sample(nc_list, k))
        return selected

    raise ValueError(f"未知 split_mode: {split_mode}, 合法值为 0 / k(k>=1) / -1")




# ============================================================================
# 主函数 / Main Function
# ============================================================================

def main():
    print(f"开始划分，正在依凭{Json_Root_Folder} 进行文件划分归属对齐...")

    # ========== Step 0: 按类别分别取三个BOX文件夹的公共 stem ==========
    print("\n[Step 0] 按类别收集三个BOX文件夹的各自合法 stem（分类别取三方交集）...")
    # dict[str, set[str]], 形状：比如 {"metal_ion": {"9e01_3_0...", ...}, ...}。 字典的值由三方公共stem名组成，作为去缀的独立基底文件名。
    valid_per_class = get_valid_stems_per_class(
        EMDB_BOX_FOLDER, PDB_Feaure_BOX_FOLDER, PBD_Label_BOX_FOLDER
    )
    # list[str], 包含的所有类别名，例如 ['metal_ion', 'nucleic', 'peptide', 'small_molecule']
    all_class_names = sorted(valid_per_class.keys())
    # int, 全部类别合法 stem 总数
    total_valid = sum(len(v) for v in valid_per_class.values())
    print(f"  发现类别: {all_class_names}")
    print(f"  各类别三方交集 stem 总数: {total_valid}")



    # ========== Step 1: 读取外部配置 Json_Root_Folder 中的全部 JSON ==========
    print(f"\n[Step 1] 扫描 Json_Root_Folder ({Json_Root_Folder})，构建全局 PDB 归属映射表...")
    # dict[str, set[str]], 存储外源 json文件名(例如: 'train_0.json'): 其内部所含的不重复pdb_id小写集合
    json_to_pdb_ids = {}
    json_root = Path(Json_Root_Folder)
    for json_file in json_root.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            # list[dict[str, str]], 典型形状如 [{"emd_123": "9j37"}, {"emd_xxx": "5ab1"}, ...]
            data = json.load(f)
        # set[str], 从上面 data 中提取出所有 .values() 作为唯一标识pdb_id的值的集合
        pdb_ids = {list(item.values())[0].lower() for item in data}
        # 将上述集合录入对齐字典, key:str 指向这批ID
        json_to_pdb_ids[json_file.name] = pdb_ids
        print(f"  已从目标配置读取到外部约束: {json_file.name} (含 {len(pdb_ids)} 个不重复的 PDB_ID 样本)")



    # ========== Step 2: 获取全体名单并在 all.json 里过滤 ==========
    # set[str], 获取囊括全体的主集 all.json 中的包含要素集合
    all_pdb_ids = json_to_pdb_ids.get("all.json", set())
    if not all_pdb_ids:
        print("  警告：在根目录下没有找到 all.json 或者内容为空。这可能导致后续交集过滤落空。")
    for cls in all_class_names:
        before = len(valid_per_class[cls])
        # 剔除在 all.json 中未被记录的样本
        valid_per_class[cls] = {s for s in valid_per_class[cls]   if get_pdb_id(s).lower() in all_pdb_ids or get_pdb_id(s).upper() in all_pdb_ids}
        print(f"    [{cls}] 初始根据 all.json 过滤前:{before}  过滤后:{len(valid_per_class[cls])}")



    # ========== Step 3: 对各种split_mode的数据集分别生成 json ==========
    for ds_idx, (split_mode, split_name) in enumerate(zip(Split_Modes, Split_Mode_Names)):
        print(f"\n{'='*60}")
        print(f"[Dataset {ds_idx}] {split_name}  (split_mode={split_mode})")
        print(f"{'='*60}")
        ds_output_dir = Path(Split_output_FOLDER[ds_idx])
        ds_output_dir.mkdir(parents=True, exist_ok=True)
        # dict[str, int], 追踪每个生成的划分文件内写入了多少个stem数目，例如 {'train.json':15152}
        json_to_stem_count = {jname: 0 for jname in json_to_pdb_ids.keys()}

        for cls in all_class_names:
            print(f"\n  -- [{cls}] --")
            cls_output_dir = ds_output_dir / cls
            cls_output_dir.mkdir(parents=True, exist_ok=True)
            # list[str], 对于当前数据集(split_mode)、当前类别(cls) 产生的有效样本的stem(去后缀)
            all_cls_stems = select_split_stems(list(valid_per_class[cls]), split_mode)
            
            for json_name, pdb_ids_set in json_to_pdb_ids.items():
                # list[str], 根据 pdb_ids_set 对候选 stem 进行过滤录入
                # 剔除在 all.json 中未被记录的样本
                target_stems = [s for s in all_cls_stems   if get_pdb_id(s).lower() in pdb_ids_set or get_pdb_id(s).upper() in pdb_ids_set]
                save_json(sorted(target_stems), cls_output_dir / json_name)
                json_to_stem_count[json_name] += len(target_stems)

        print(f"\n  [{split_name}] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 {len(all_class_names)} 个类别):")
        for json_name, count in sorted(json_to_stem_count.items()):
            print(f"    {json_name}: 共分配 {count} 个 stem")

    # ========== 全局完成 ==========
    print("\n" + "=" * 60)
    print("全部外部依赖同步划分完成! 请前往以下目录查收生成的对齐 JSON 文件群:")
    for folder in Split_output_FOLDER:
        print(f"  {folder}")
    print("=" * 60)

if __name__ == "__main__":
    main()



# 针对 v0 数据集BOX的划分结果:
"""
(base) [penghongen@master ~]$ conda activate vnegnn
/home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/processedPDB_EMDB_binder/split_data/generate_full_json.py
WARNING: overwriting environment variables set in the machine
overwriting variable {'LD_LIBRARY_PATH'}
(vnegnn) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/processedPDB_EMDB_binder/split_data/generate_full_json.py
开始划分，正在依凭/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v0/ 进行文件划分归属对齐...

[Step 0] 按类别收集三个BOX文件夹的各自合法 stem（分类别取三方交集）...
    [metal_ion] EMDB:33929  Feature:33929  Label:33929  交集:33929
    [nucleic] EMDB:18  Feature:18  Label:18  交集:18
    [peptide] EMDB:1278  Feature:1278  Label:1278  交集:1278
    [small_molecule] EMDB:91618  Feature:91618  Label:91618  交集:91618
  发现类别: ['metal_ion', 'nucleic', 'peptide', 'small_molecule']
  各类别三方交集 stem 总数: 126843

[Step 1] 扫描 Json_Root_Folder (/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v0/)，构建全局 PDB 归属映射表...
  已从目标配置读取到外部约束: val_2.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: all.json (含 2680 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_8.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_6.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_4.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_0.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val.json (含 268 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train.json (含 2144 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_5.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_1.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test.json (含 268 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_3.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_9.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_7.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_9.json (含 26 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_7.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_3.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_1.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_5.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_0.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_4.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_8.json (含 26 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_6.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_2.json (含 215 个不重复的 PDB_ID 样本)
    [metal_ion] 初始根据 all.json 过滤前:33929  过滤后:33929
    [nucleic] 初始根据 all.json 过滤前:18  过滤后:18
    [peptide] 初始根据 all.json 过滤前:1278  过滤后:1278
    [small_molecule] 初始根据 all.json 过滤前:91618  过滤后:91618

============================================================
[Dataset 0] split_0  (split_mode=0)
============================================================

  -- [metal_ion] --
  已保存: val_2.json (81 样本)
  已保存: all.json (7633 样本)
  已保存: train_8.json (505 样本)
  已保存: val_6.json (184 样本)
  已保存: train_4.json (742 样本)
  已保存: train_0.json (789 样本)
  已保存: val.json (813 样本)
  已保存: train.json (6084 样本)
  已保存: train_5.json (733 样本)
  已保存: train_1.json (536 样本)
  已保存: test.json (736 样本)
  已保存: val_3.json (174 样本)
  已保存: train_9.json (512 样本)
  已保存: val_7.json (49 样本)
  已保存: val_9.json (33 样本)
  已保存: train_7.json (569 样本)
  已保存: train_3.json (543 样本)
  已保存: val_1.json (66 样本)
  已保存: val_5.json (35 样本)
  已保存: val_0.json (59 样本)
  已保存: val_4.json (67 样本)
  已保存: val_8.json (65 样本)
  已保存: train_6.json (637 样本)
  已保存: train_2.json (518 样本)

  -- [nucleic] --
  已保存: val_2.json (0 样本)
  已保存: all.json (4 样本)
  已保存: train_8.json (3 样本)
  已保存: val_6.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train_0.json (0 样本)
  已保存: val.json (0 样本)
  已保存: train.json (4 样本)
  已保存: train_5.json (0 样本)
  已保存: train_1.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_3.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_6.json (1 样本)
  已保存: train_2.json (0 样本)

  -- [peptide] --
  已保存: val_2.json (0 样本)
  已保存: all.json (222 样本)
  已保存: train_8.json (17 样本)
  已保存: val_6.json (0 样本)
  已保存: train_4.json (17 样本)
  已保存: train_0.json (28 样本)
  已保存: val.json (7 样本)
  已保存: train.json (186 样本)
  已保存: train_5.json (30 样本)
  已保存: train_1.json (12 样本)
  已保存: test.json (29 样本)
  已保存: val_3.json (0 样本)
  已保存: train_9.json (17 样本)
  已保存: val_7.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: train_7.json (20 样本)
  已保存: train_3.json (18 样本)
  已保存: val_1.json (0 样本)
  已保存: val_5.json (1 样本)
  已保存: val_0.json (2 样本)
  已保存: val_4.json (4 样本)
  已保存: val_8.json (0 样本)
  已保存: train_6.json (5 样本)
  已保存: train_2.json (22 样本)

  -- [small_molecule] --
  已保存: val_2.json (212 样本)
  已保存: all.json (23575 样本)
  已保存: train_8.json (2228 样本)
  已保存: val_6.json (298 样本)
  已保存: train_4.json (1955 样本)
  已保存: train_0.json (1760 样本)
  已保存: val.json (2429 样本)
  已保存: train.json (19009 样本)
  已保存: train_5.json (2207 样本)
  已保存: train_1.json (2054 样本)
  已保存: test.json (2137 样本)
  已保存: val_3.json (287 样本)
  已保存: train_9.json (1913 样本)
  已保存: val_7.json (245 样本)
  已保存: val_9.json (194 样本)
  已保存: train_7.json (1842 样本)
  已保存: train_3.json (1818 样本)
  已保存: val_1.json (308 样本)
  已保存: val_5.json (155 样本)
  已保存: val_0.json (242 样本)
  已保存: val_4.json (216 样本)
  已保存: val_8.json (272 样本)
  已保存: train_6.json (1695 样本)
  已保存: train_2.json (1537 样本)

  [split_0] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 31434 个 stem
    test.json: 共分配 2902 个 stem
    train.json: 共分配 25283 个 stem
    train_0.json: 共分配 2577 个 stem
    train_1.json: 共分配 2602 个 stem
    train_2.json: 共分配 2077 个 stem
    train_3.json: 共分配 2379 个 stem
    train_4.json: 共分配 2714 个 stem
    train_5.json: 共分配 2970 个 stem
    train_6.json: 共分配 2338 个 stem
    train_7.json: 共分配 2431 个 stem
    train_8.json: 共分配 2753 个 stem
    train_9.json: 共分配 2442 个 stem
    val.json: 共分配 3249 个 stem
    val_0.json: 共分配 303 个 stem
    val_1.json: 共分配 374 个 stem
    val_2.json: 共分配 293 个 stem
    val_3.json: 共分配 461 个 stem
    val_4.json: 共分配 287 个 stem
    val_5.json: 共分配 191 个 stem
    val_6.json: 共分配 482 个 stem
    val_7.json: 共分配 294 个 stem
    val_8.json: 共分配 337 个 stem
    val_9.json: 共分配 227 个 stem

============================================================
[Dataset 1] split_3  (split_mode=3)
============================================================

  -- [metal_ion] --
  已保存: val_2.json (286 样本)
  已保存: all.json (24522 样本)
  已保存: train_8.json (1686 样本)
  已保存: val_6.json (556 样本)
  已保存: train_4.json (2307 样本)
  已保存: train_0.json (2513 样本)
  已保存: val.json (2607 样本)
  已保存: train.json (19639 样本)
  已保存: train_5.json (2338 样本)
  已保存: train_1.json (1788 样本)
  已保存: test.json (2276 样本)
  已保存: val_3.json (538 样本)
  已保存: train_9.json (1738 样本)
  已保存: val_7.json (167 样本)
  已保存: val_9.json (106 样本)
  已保存: train_7.json (1828 样本)
  已保存: train_3.json (1766 样本)
  已保存: val_1.json (203 样本)
  已保存: val_5.json (127 样本)
  已保存: val_0.json (181 样本)
  已保存: val_4.json (199 样本)
  已保存: val_8.json (244 样本)
  已保存: train_6.json (2067 样本)
  已保存: train_2.json (1608 样本)

  -- [nucleic] --
  已保存: val_2.json (0 样本)
  已保存: all.json (14 样本)
  已保存: train_8.json (10 样本)
  已保存: val_6.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train_0.json (0 样本)
  已保存: val.json (0 样本)
  已保存: train.json (14 样本)
  已保存: train_5.json (0 样本)
  已保存: train_1.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_3.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_6.json (4 样本)
  已保存: train_2.json (0 样本)

  -- [peptide] --
  已保存: val_2.json (0 样本)
  已保存: all.json (832 样本)
  已保存: train_8.json (68 样本)
  已保存: val_6.json (0 样本)
  已保存: train_4.json (68 样本)
  已保存: train_0.json (97 样本)
  已保存: val.json (26 样本)
  已保存: train.json (704 样本)
  已保存: train_5.json (111 样本)
  已保存: train_1.json (48 样本)
  已保存: test.json (102 样本)
  已保存: val_3.json (0 样本)
  已保存: train_9.json (62 样本)
  已保存: val_7.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: train_7.json (76 样本)
  已保存: train_3.json (72 样本)
  已保存: val_1.json (0 样本)
  已保存: val_5.json (4 样本)
  已保存: val_0.json (6 样本)
  已保存: val_4.json (16 样本)
  已保存: val_8.json (0 样本)
  已保存: train_6.json (18 样本)
  已保存: train_2.json (84 样本)

  -- [small_molecule] --
  已保存: val_2.json (654 样本)
  已保存: all.json (70591 样本)
  已保存: train_8.json (6878 样本)
  已保存: val_6.json (816 样本)
  已保存: train_4.json (5822 样本)
  已保存: train_0.json (5159 样本)
  已保存: val.json (7262 样本)
  已保存: train.json (57003 样本)
  已保存: train_5.json (6572 样本)
  已保存: train_1.json (6286 样本)
  已保存: test.json (6326 样本)
  已保存: val_3.json (866 样本)
  已保存: train_9.json (5877 样本)
  已保存: val_7.json (675 样本)
  已保存: val_9.json (580 样本)
  已保存: train_7.json (5391 样本)
  已保存: train_3.json (5399 样本)
  已保存: val_1.json (1008 样本)
  已保存: val_5.json (515 样本)
  已保存: val_0.json (735 样本)
  已保存: val_4.json (650 样本)
  已保存: val_8.json (763 样本)
  已保存: train_6.json (5159 样本)
  已保存: train_2.json (4460 样本)

  [split_3] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 95959 个 stem
    test.json: 共分配 8704 个 stem
    train.json: 共分配 77360 个 stem
    train_0.json: 共分配 7769 个 stem
    train_1.json: 共分配 8122 个 stem
    train_2.json: 共分配 6152 个 stem
    train_3.json: 共分配 7237 个 stem
    train_4.json: 共分配 8197 个 stem
    train_5.json: 共分配 9021 个 stem
    train_6.json: 共分配 7248 个 stem
    train_7.json: 共分配 7295 个 stem
    train_8.json: 共分配 8642 个 stem
    train_9.json: 共分配 7677 个 stem
    val.json: 共分配 9895 个 stem
    val_0.json: 共分配 922 个 stem
    val_1.json: 共分配 1211 个 stem
    val_2.json: 共分配 940 个 stem
    val_3.json: 共分配 1404 个 stem
    val_4.json: 共分配 865 个 stem
    val_5.json: 共分配 646 个 stem
    val_6.json: 共分配 1372 个 stem
    val_7.json: 共分配 842 个 stem
    val_8.json: 共分配 1007 个 stem
    val_9.json: 共分配 686 个 stem

============================================================
[Dataset 2] split_all  (split_mode=-1)
============================================================

  -- [metal_ion] --
  已保存: val_2.json (413 样本)
  已保存: all.json (33929 样本)
  已保存: train_8.json (2451 样本)
  已保存: val_6.json (744 样本)
  已保存: train_4.json (3073 样本)
  已保存: train_0.json (3352 样本)
  已保存: val.json (3670 样本)
  已保存: train.json (27221 样本)
  已保存: train_5.json (3132 样本)
  已保存: train_1.json (2551 样本)
  已保存: test.json (3038 样本)
  已保存: val_3.json (706 样本)
  已保存: train_9.json (2516 样本)
  已保存: val_7.json (256 样本)
  已保存: val_9.json (158 样本)
  已保存: train_7.json (2557 样本)
  已保存: train_3.json (2461 样本)
  已保存: val_1.json (291 样本)
  已保存: val_5.json (195 样本)
  已保存: val_0.json (267 样本)
  已保存: val_4.json (269 样本)
  已保存: val_8.json (371 样本)
  已保存: train_6.json (2874 样本)
  已保存: train_2.json (2254 样本)

  -- [nucleic] --
  已保存: val_2.json (0 样本)
  已保存: all.json (18 样本)
  已保存: train_8.json (10 样本)
  已保存: val_6.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train_0.json (0 样本)
  已保存: val.json (0 样本)
  已保存: train.json (18 样本)
  已保存: train_5.json (0 样本)
  已保存: train_1.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_3.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_6.json (8 样本)
  已保存: train_2.json (0 样本)

  -- [peptide] --
  已保存: val_2.json (0 样本)
  已保存: all.json (1278 样本)
  已保存: train_8.json (128 样本)
  已保存: val_6.json (0 样本)
  已保存: train_4.json (114 样本)
  已保存: train_0.json (133 样本)
  已保存: val.json (42 样本)
  已保存: train.json (1072 样本)
  已保存: train_5.json (153 样本)
  已保存: train_1.json (60 样本)
  已保存: test.json (164 样本)
  已保存: val_3.json (0 样本)
  已保存: train_9.json (92 样本)
  已保存: val_7.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: train_7.json (124 样本)
  已保存: train_3.json (118 样本)
  已保存: val_1.json (0 样本)
  已保存: val_5.json (8 样本)
  已保存: val_0.json (10 样本)
  已保存: val_4.json (24 样本)
  已保存: val_8.json (0 样本)
  已保存: train_6.json (30 样本)
  已保存: train_2.json (120 样本)

  -- [small_molecule] --
  已保存: val_2.json (836 样本)
  已保存: all.json (91618 样本)
  已保存: train_8.json (9101 样本)
  已保存: val_6.json (1007 样本)
  已保存: train_4.json (7489 样本)
  已保存: train_0.json (6630 样本)
  已保存: val.json (9416 样本)
  已保存: train.json (74013 样本)
  已保存: train_5.json (8423 样本)
  已保存: train_1.json (8427 样本)
  已保存: test.json (8189 样本)
  已保存: val_3.json (1170 样本)
  已保存: train_9.json (7676 样本)
  已保存: val_7.json (866 样本)
  已保存: val_9.json (730 样本)
  已保存: train_7.json (6866 样本)
  已保存: train_3.json (6938 样本)
  已保存: val_1.json (1379 样本)
  已保存: val_5.json (726 样本)
  已保存: val_0.json (918 样本)
  已保存: val_4.json (821 样本)
  已保存: val_8.json (963 样本)
  已保存: train_6.json (6752 样本)
  已保存: train_2.json (5711 样本)

  [split_all] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 126843 个 stem
    test.json: 共分配 11391 个 stem
    train.json: 共分配 102324 个 stem
    train_0.json: 共分配 10115 个 stem
    train_1.json: 共分配 11038 个 stem
    train_2.json: 共分配 8085 个 stem
    train_3.json: 共分配 9517 个 stem
    train_4.json: 共分配 10676 个 stem
    train_5.json: 共分配 11708 个 stem
    train_6.json: 共分配 9664 个 stem
    train_7.json: 共分配 9547 个 stem
    train_8.json: 共分配 11690 个 stem
    train_9.json: 共分配 10284 个 stem
    val.json: 共分配 13128 个 stem
    val_0.json: 共分配 1195 个 stem
    val_1.json: 共分配 1670 个 stem
    val_2.json: 共分配 1249 个 stem
    val_3.json: 共分配 1876 个 stem
    val_4.json: 共分配 1114 个 stem
    val_5.json: 共分配 929 个 stem
    val_6.json: 共分配 1751 个 stem
    val_7.json: 共分配 1122 个 stem
    val_8.json: 共分配 1334 个 stem
    val_9.json: 共分配 888 个 stem

============================================================
全部外部依赖同步划分完成! 请前往以下目录查收生成的对齐 JSON 文件群:
  /storage/penghongen/Pocket_classic/v_0/split/split_0
  /storage/penghongen/Pocket_classic/v_0/split/split_3
  /storage/penghongen/Pocket_classic/v_0/split/split_all
============================================================
(vnegnn) [penghongen@master ~]$  
"""


"""
(vnegnn) [penghongen@master ~]$ /home/penghongen/anaconda3/envs/vnegnn/bin/python /home/penghongen/My_Project/Pocket/processedPDB_EMDB_binder/split_data/generate_full_json.py
开始划分，正在依凭/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/ 进行文件划分归属对齐...

[Step 0] 按类别收集三个BOX文件夹的各自合法 stem（分类别取三方交集）...
    [metal_ion] EMDB:30642  Feature:30642  Label:30642  交集:30642
    [nucleic] EMDB:22  Feature:22  Label:22  交集:22
    [peptide] EMDB:1035  Feature:1035  Label:1035  交集:1035
    [small_molecule] EMDB:90477  Feature:90477  Label:90477  交集:90477
  发现类别: ['metal_ion', 'nucleic', 'peptide', 'small_molecule']
  各类别三方交集 stem 总数: 122176

[Step 1] 扫描 Json_Root_Folder (/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v1/)，构建全局 PDB 归属映射表...
  已从目标配置读取到外部约束: train_6.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_2.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_9.json (含 26 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_5.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: test.json (含 268 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_1.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_4.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_0.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_7.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_8.json (含 26 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_3.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val.json (含 268 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_6.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_2.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_9.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_5.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: all.json (含 2680 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_1.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_4.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train.json (含 2144 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_0.json (含 215 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_7.json (含 27 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: train_8.json (含 214 个不重复的 PDB_ID 样本)
  已从目标配置读取到外部约束: val_3.json (含 27 个不重复的 PDB_ID 样本)
    [metal_ion] 初始根据 all.json 过滤前:30642  过滤后:30642
    [nucleic] 初始根据 all.json 过滤前:22  过滤后:22
    [peptide] 初始根据 all.json 过滤前:1035  过滤后:1035
    [small_molecule] 初始根据 all.json 过滤前:90477  过滤后:90477

============================================================
[Dataset 0] split_0  (split_mode=0)
============================================================

  -- [metal_ion] --
  已保存: train_6.json (637 样本)
  已保存: train_2.json (518 样本)
  已保存: val_9.json (33 样本)
  已保存: val_5.json (35 样本)
  已保存: test.json (736 样本)
  已保存: val_1.json (66 样本)
  已保存: val_4.json (67 样本)
  已保存: val_0.json (59 样本)
  已保存: train_7.json (569 样本)
  已保存: val_8.json (65 样本)
  已保存: train_3.json (543 样本)
  已保存: val.json (813 样本)
  已保存: val_6.json (184 样本)
  已保存: val_2.json (81 样本)
  已保存: train_9.json (512 样本)
  已保存: train_5.json (733 样本)
  已保存: all.json (7633 样本)
  已保存: train_1.json (536 样本)
  已保存: train_4.json (742 样本)
  已保存: train.json (6084 样本)
  已保存: train_0.json (789 样本)
  已保存: val_7.json (49 样本)
  已保存: train_8.json (505 样本)
  已保存: val_3.json (174 样本)

  -- [nucleic] --
  已保存: train_6.json (1 样本)
  已保存: train_2.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val.json (0 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: train_5.json (0 样本)
  已保存: all.json (4 样本)
  已保存: train_1.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train.json (4 样本)
  已保存: train_0.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (3 样本)
  已保存: val_3.json (0 样本)

  -- [peptide] --
  已保存: train_6.json (5 样本)
  已保存: train_2.json (22 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (1 样本)
  已保存: test.json (29 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (4 样本)
  已保存: val_0.json (2 样本)
  已保存: train_7.json (20 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (18 样本)
  已保存: val.json (7 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (17 样本)
  已保存: train_5.json (30 样本)
  已保存: all.json (222 样本)
  已保存: train_1.json (12 样本)
  已保存: train_4.json (17 样本)
  已保存: train.json (186 样本)
  已保存: train_0.json (28 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (17 样本)
  已保存: val_3.json (0 样本)

  -- [small_molecule] --
  已保存: train_6.json (1695 样本)
  已保存: train_2.json (1537 样本)
  已保存: val_9.json (194 样本)
  已保存: val_5.json (155 样本)
  已保存: test.json (2137 样本)
  已保存: val_1.json (308 样本)
  已保存: val_4.json (216 样本)
  已保存: val_0.json (242 样本)
  已保存: train_7.json (1842 样本)
  已保存: val_8.json (272 样本)
  已保存: train_3.json (1818 样本)
  已保存: val.json (2429 样本)
  已保存: val_6.json (298 样本)
  已保存: val_2.json (212 样本)
  已保存: train_9.json (1913 样本)
  已保存: train_5.json (2207 样本)
  已保存: all.json (23575 样本)
  已保存: train_1.json (2054 样本)
  已保存: train_4.json (1955 样本)
  已保存: train.json (19009 样本)
  已保存: train_0.json (1760 样本)
  已保存: val_7.json (245 样本)
  已保存: train_8.json (2228 样本)
  已保存: val_3.json (287 样本)

  [split_0] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 31434 个 stem
    test.json: 共分配 2902 个 stem
    train.json: 共分配 25283 个 stem
    train_0.json: 共分配 2577 个 stem
    train_1.json: 共分配 2602 个 stem
    train_2.json: 共分配 2077 个 stem
    train_3.json: 共分配 2379 个 stem
    train_4.json: 共分配 2714 个 stem
    train_5.json: 共分配 2970 个 stem
    train_6.json: 共分配 2338 个 stem
    train_7.json: 共分配 2431 个 stem
    train_8.json: 共分配 2753 个 stem
    train_9.json: 共分配 2442 个 stem
    val.json: 共分配 3249 个 stem
    val_0.json: 共分配 303 个 stem
    val_1.json: 共分配 374 个 stem
    val_2.json: 共分配 293 个 stem
    val_3.json: 共分配 461 个 stem
    val_4.json: 共分配 287 个 stem
    val_5.json: 共分配 191 个 stem
    val_6.json: 共分配 482 个 stem
    val_7.json: 共分配 294 个 stem
    val_8.json: 共分配 337 个 stem
    val_9.json: 共分配 227 个 stem

============================================================
[Dataset 1] split_3  (split_mode=3)
============================================================

  -- [metal_ion] --
  已保存: train_6.json (1996 样本)
  已保存: train_2.json (1596 样本)
  已保存: val_9.json (102 样本)
  已保存: val_5.json (130 样本)
  已保存: test.json (2221 样本)
  已保存: val_1.json (212 样本)
  已保存: val_4.json (230 样本)
  已保存: val_0.json (197 样本)
  已保存: train_7.json (1734 样本)
  已保存: val_8.json (221 样本)
  已保存: train_3.json (1708 样本)
  已保存: val.json (2494 样本)
  已保存: val_6.json (508 样本)
  已保存: val_2.json (242 样本)
  已保存: train_9.json (1602 样本)
  已保存: train_5.json (2261 样本)
  已保存: all.json (23596 样本)
  已保存: train_1.json (1678 样本)
  已保存: train_4.json (2251 样本)
  已保存: train.json (18881 样本)
  已保存: train_0.json (2460 样本)
  已保存: val_7.json (152 样本)
  已保存: train_8.json (1595 样本)
  已保存: val_3.json (500 样本)

  -- [nucleic] --
  已保存: train_6.json (4 样本)
  已保存: train_2.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val.json (0 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: train_5.json (0 样本)
  已保存: all.json (14 样本)
  已保存: train_1.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train.json (14 样本)
  已保存: train_0.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (10 样本)
  已保存: val_3.json (0 样本)

  -- [peptide] --
  已保存: train_6.json (16 样本)
  已保存: train_2.json (70 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (4 样本)
  已保存: test.json (106 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (14 样本)
  已保存: val_0.json (6 样本)
  已保存: train_7.json (70 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (62 样本)
  已保存: val.json (24 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (62 样本)
  已保存: train_5.json (89 样本)
  已保存: all.json (761 样本)
  已保存: train_1.json (40 样本)
  已保存: train_4.json (62 样本)
  已保存: train.json (631 样本)
  已保存: train_0.json (94 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (66 样本)
  已保存: val_3.json (0 样本)

  -- [small_molecule] --
  已保存: train_6.json (5194 样本)
  已保存: train_2.json (4578 样本)
  已保存: val_9.json (564 样本)
  已保存: val_5.json (523 样本)
  已保存: test.json (6384 样本)
  已保存: val_1.json (990 样本)
  已保存: val_4.json (640 样本)
  已保存: val_0.json (687 样本)
  已保存: train_7.json (5509 样本)
  已保存: val_8.json (824 样本)
  已保存: train_3.json (5499 样本)
  已保存: val.json (7364 样本)
  已保存: val_6.json (831 样本)
  已保存: val_2.json (672 样本)
  已保存: train_9.json (5848 样本)
  已保存: train_5.json (6642 样本)
  已保存: all.json (71260 样本)
  已保存: train_1.json (6162 样本)
  已保存: train_4.json (6029 样本)
  已保存: train.json (57512 样本)
  已保存: train_0.json (5198 样本)
  已保存: val_7.json (752 样本)
  已保存: train_8.json (6853 样本)
  已保存: val_3.json (881 样本)

  [split_3] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 95631 个 stem
    test.json: 共分配 8711 个 stem
    train.json: 共分配 77038 个 stem
    train_0.json: 共分配 7752 个 stem
    train_1.json: 共分配 7880 个 stem
    train_2.json: 共分配 6244 个 stem
    train_3.json: 共分配 7269 个 stem
    train_4.json: 共分配 8342 个 stem
    train_5.json: 共分配 8992 个 stem
    train_6.json: 共分配 7210 个 stem
    train_7.json: 共分配 7313 个 stem
    train_8.json: 共分配 8524 个 stem
    train_9.json: 共分配 7512 个 stem
    val.json: 共分配 9882 个 stem
    val_0.json: 共分配 890 个 stem
    val_1.json: 共分配 1202 个 stem
    val_2.json: 共分配 914 个 stem
    val_3.json: 共分配 1381 个 stem
    val_4.json: 共分配 884 个 stem
    val_5.json: 共分配 657 个 stem
    val_6.json: 共分配 1339 个 stem
    val_7.json: 共分配 904 个 stem
    val_8.json: 共分配 1045 个 stem
    val_9.json: 共分配 666 个 stem

============================================================
[Dataset 2] split_all  (split_mode=-1)
============================================================

  -- [metal_ion] --
  已保存: train_6.json (2523 样本)
  已保存: train_2.json (2150 样本)
  已保存: val_9.json (147 样本)
  已保存: val_5.json (186 样本)
  已保存: test.json (2855 样本)
  已保存: val_1.json (290 样本)
  已保存: val_4.json (305 样本)
  已保存: val_0.json (282 样本)
  已保存: train_7.json (2304 样本)
  已保存: val_8.json (307 样本)
  已保存: train_3.json (2212 样本)
  已保存: val.json (3239 样本)
  已保存: val_6.json (625 样本)
  已保存: val_2.json (296 样本)
  已保存: train_9.json (2125 样本)
  已保存: train_5.json (2888 样本)
  已保存: all.json (30642 样本)
  已保存: train_1.json (2206 样本)
  已保存: train_4.json (2868 样本)
  已保存: train.json (24548 样本)
  已保存: train_0.json (3125 样本)
  已保存: val_7.json (210 样本)
  已保存: train_8.json (2147 样本)
  已保存: val_3.json (591 样本)

  -- [nucleic] --
  已保存: train_6.json (8 样本)
  已保存: train_2.json (0 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (0 样本)
  已保存: test.json (0 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (0 样本)
  已保存: val_0.json (0 样本)
  已保存: train_7.json (0 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (0 样本)
  已保存: val.json (0 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (0 样本)
  已保存: train_5.json (0 样本)
  已保存: all.json (22 样本)
  已保存: train_1.json (0 样本)
  已保存: train_4.json (0 样本)
  已保存: train.json (22 样本)
  已保存: train_0.json (0 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (14 样本)
  已保存: val_3.json (0 样本)

  -- [peptide] --
  已保存: train_6.json (20 样本)
  已保存: train_2.json (88 样本)
  已保存: val_9.json (0 样本)
  已保存: val_5.json (8 样本)
  已保存: test.json (146 样本)
  已保存: val_1.json (0 样本)
  已保存: val_4.json (16 样本)
  已保存: val_0.json (10 样本)
  已保存: train_7.json (88 样本)
  已保存: val_8.json (0 样本)
  已保存: train_3.json (82 样本)
  已保存: val.json (34 样本)
  已保存: val_6.json (0 样本)
  已保存: val_2.json (0 样本)
  已保存: train_9.json (88 样本)
  已保存: train_5.json (110 样本)
  已保存: all.json (1035 样本)
  已保存: train_1.json (54 样本)
  已保存: train_4.json (90 样本)
  已保存: train.json (855 样本)
  已保存: train_0.json (135 样本)
  已保存: val_7.json (0 样本)
  已保存: train_8.json (100 样本)
  已保存: val_3.json (0 样本)

  -- [small_molecule] --
  已保存: train_6.json (6600 样本)
  已保存: train_2.json (5756 样本)
  已保存: val_9.json (695 样本)
  已保存: val_5.json (713 样本)
  已保存: test.json (8052 样本)
  已保存: val_1.json (1312 样本)
  已保存: val_4.json (771 样本)
  已保存: val_0.json (857 样本)
  已保存: train_7.json (7035 样本)
  已保存: val_8.json (1043 样本)
  已保存: train_3.json (6990 样本)
  已保存: val.json (9354 样本)
  已保存: val_6.json (1030 样本)
  已保存: val_2.json (833 样本)
  已保存: train_9.json (7511 样本)
  已保存: train_5.json (8402 样本)
  已保存: all.json (90477 样本)
  已保存: train_1.json (7837 样本)
  已保存: train_4.json (7660 样本)
  已保存: train.json (73071 样本)
  已保存: train_0.json (6493 样本)
  已保存: val_7.json (961 样本)
  已保存: train_8.json (8787 样本)
  已保存: val_3.json (1139 样本)

  [split_all] 当前模式下各 JSON 文件总输出 stem 数汇总 (总计跨 4 个类别):
    all.json: 共分配 122176 个 stem
    test.json: 共分配 11053 个 stem
    train.json: 共分配 98496 个 stem
    train_0.json: 共分配 9753 个 stem
    train_1.json: 共分配 10097 个 stem
    train_2.json: 共分配 7994 个 stem
    train_3.json: 共分配 9284 个 stem
    train_4.json: 共分配 10618 个 stem
    train_5.json: 共分配 11400 个 stem
    train_6.json: 共分配 9151 个 stem
    train_7.json: 共分配 9427 个 stem
    train_8.json: 共分配 11048 个 stem
    train_9.json: 共分配 9724 个 stem
    val.json: 共分配 12627 个 stem
    val_0.json: 共分配 1149 个 stem
    val_1.json: 共分配 1602 个 stem
    val_2.json: 共分配 1129 个 stem
    val_3.json: 共分配 1730 个 stem
    val_4.json: 共分配 1092 个 stem
    val_5.json: 共分配 907 个 stem
    val_6.json: 共分配 1655 个 stem
    val_7.json: 共分配 1171 个 stem
    val_8.json: 共分配 1350 个 stem
    val_9.json: 共分配 842 个 stem

============================================================
全部外部依赖同步划分完成! 请前往以下目录查收生成的对齐 JSON 文件群:
  /storage/penghongen/Pocket_classic/v_1/split/split_0
  /storage/penghongen/Pocket_classic/v_1/split/split_3
  /storage/penghongen/Pocket_classic/v_1/split/split_all
============================================================
(vnegnn) [penghongen@master ~]$ 
"""
