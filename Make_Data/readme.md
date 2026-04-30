Pocket\Make_Data\process_and_label.py 是总的处理接口, 可以把下面的1~3都做了, 再手动运行 4 即可


1. Pocket\Make_Data\PDB_processor 负责解析PDB, 并产生一系列.npz文件(除了 label.npz)。各层次特征(原子级/残基级/图)、以及非常广泛的"候选ligand"(仅去除了水和缓冲液)的信息都在这里.

2. (运行完上一步之后)再运行另一个文件夹 Pocket\Make_Data\split_data , 给定需要排除的测试集, 它会按照自己的意图灵活划分训练集、验证集。

3. 下面是选择"候选ligand" 的逻辑:
(1). 如果要定义一种候选模式, 直接在 Pocket\Make_Data\labels\filter_config.py 里面添加
(2). Pocket\Make_Data\labels\ligand_filter.py 控制着根据"候选模式"筛选ligand的逻辑
(3). 最终根据选好的候选ligand, 打标签 + 生成label.npz (逻辑在 Pocket\Make_Data\labels\instance_labels.py, 这个逻辑相对固定, 一般不用动)
     label.npz 除了包含逐原子的 pocket_class_ids / instance_ids / binding_mask 等字段外, 还持久化了:
       - ligand_class_ids: 逐配体的口袋类别 ID (candidate_id → class_id 直接映射, 由 filter_and_classify 产出)
       - pocket_atom_indices_{id}: 每个配体阈值内的所有结合原子全局索引 (不受独占约束, 同一原子可出现在多个配体中)
如果要细化ligand的选取方式(比如定义一种规则: ligand满足"埋藏深度必须超过5", 那么首先要在 Pocket\Make_Data\PDB_processor\ligand_candidates.py 里面定义出"埋藏深度"这个属性, 然后在 Pocket\Make_Data\labels\ligand_filter.py 里面定义出"埋藏深度的识别逻辑", 最后在 Pocket\Make_Data\labels\filter_config.py 里面添加想要的配置逻辑)

4. 最后, 手动运行 Pocket\Make_Data\split_data\generate_full_json.py 产生训练集、验证集、测试集的.json划分, 即List[dict[str, str]], 每个条目(dict)如  {"emd_48166": "9MD3"}.
它的生成逻辑是: 
(1). 以原始emdb_pdb映射文件 (目前是"/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv") 为基础. 如果它的一个条目同时在PDB根目录文件("/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc")和EMDB
根目录文件("/storage/chenzhaoyang/cryo_em/PDB_3.5_cc_qscore")中出现, 那么就写入 all.json(My_Project/Data/split/3.5_cc_qscore_v0/all.json)
(2). 给定一个用作测试集的PDB根目录文件("/home/penghongen/My_Project/Data/cryatom_output_cif/"), 如果它对应的一个条目在all.json内出现【这是目前的限制;意味着合法的测试样本必须满足cc/qscore等前置条件】, 那才写入 My_Project/Data/split/3.5_cc_qscore_v0/test.json . 剩余的all.json再划分训练集与验证集.



以上是第一大阶段的处理流程, 注意目前为止完全没有关于EMDB的处理.
