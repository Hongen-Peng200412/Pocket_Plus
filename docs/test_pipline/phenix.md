# Phenix 差图 baseline 预生成说明

> 若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；再阅读本目录 `general.md` 与 `运行指导.md` 确认测试 pipeline 的统一评估口径。本文只说明 phenix baseline 的预生成与消费契约。

## 1. 预生成、只消费

`phenix_real_space_diff_map` 是六组 baseline 中唯一依赖外部二进制的一组：

```text
/home/yangjy/software/phenix/build/bin/phenix.real_space_diff_map
```

它涉及外部命令调用、结构输入格式、resolution 参数、输出文件定位，以及把 phenix 输出 map 对齐到 cache 网格。为保证评估阶段稳定，pipeline 采用：

1. `src/inference/main/generate_phenix_diff_maps.py` 在服务器预生成每个样本的对齐差图。
2. `src/inference/main/build_baseline_cache.py` 按约定路径读取对齐差图，构造 DL-compatible baseline cache。
3. `src/inference/main/run_baseline_two_stage.py` 复用三段评估流程。

消费端不在线运行 phenix、不重采样 phenix 输出、不向样本 JSON 写额外字段。

## 2. 路径契约

生成端和消费端共用 A2 派生规则：

```text
derive_phenix_map_path(phenix_output_root, system, sample_name)
```

默认输出路径：

```text
/home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps/<system>/<sample_name>/phenix_diff_aligned.mrc
```

其中：

- `phenix_output_root` 默认 `/home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps`。
- `system` 为 `stardard` 或 `strict`。
- `sample_name` 与 cache 文件名使用同一解析规则。
- `protein_40.json` / `protein_110.json` 保持原始样本字段，不写 phenix 专用路径字段。

## 3. 输入选择

对 `protein_40.json` / `protein_110.json` 中的每个样本、每个系统：

```text
stardard:
  phenix model = cif_gt_path 的 receptor-only mmCIF
  map          = map_path

strict:
  phenix model = cif_path
  map          = map_path
```

`stardard` 的 `cif_gt_path` 可能包含 ligand，因此生成脚本会先写出 receptor-only mmCIF，仅用于 phenix model 输入；不修改原始结构文件、不修改 GT 构造口径。

`strict` 的 `cif_path` 按系统语义直接使用。Phenix 可接受 mmCIF，因此不需要强制转换为 PDB，避免多字符 chain id 被 PDB 格式截断。

## 4. resolution 来源

逐样本 resolution 解析顺序：

```text
JSON 显式字段 resolution
> resolution_csv 按 EMDB id 查询
> fallback_3.5
```

默认 CSV：

```text
/home/penghongen/My_Project/Data/EMDB_PDB_resolution_3.5.csv
```

落到 `fallback_3.5` 时，生成日志需要保留 `resolution_source=fallback_3.5`，便于结果解释时区分。

## 5. 对齐契约

生成脚本负责把 phenix 原始输出 map 对齐到 `load_from_raw_cif(...)` 得到的 cache 网格：

```text
target_voxel_size = 1.0
shape             = reference["full_shape_zyx"]
origin            = reference["origin"]
voxel_size         = reference["voxel_size"]
```

写出的 `phenix_diff_aligned.mrc` 必须与同一样本 cache 中的 `resampled_emdb` 形状、origin 和 voxel_size 一致。

## 6. 与 baseline cache 的衔接

`build_baseline_cache.py` 的 phenix 分支：

1. 调 `load_from_raw_cif` 得到 `hardmask / resampled_emdb / origin / voxel_size / GT`。
2. 按 A2 约定读取 `phenix_diff_aligned.mrc`。
3. 把差图作为 raw score map。
4. 执行 `per_sample_rank_equalize(valid=hardmask==0)`。
5. 写出 DL-compatible `.npz`：`ligand_pred`、`receptor_pred=None`、`hardmask`、`resampled_emdb`、GT 和 meta。
6. 可视化需要时，raw 差图 sidecar MRC 路径写入 `meta["raw_diff_map_path"]`。

## 7. 推荐运行

先预生成差图：

```bash
python src/inference/main/generate_phenix_diff_maps.py \
  --phenix_output_root /home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps \
  --val_json /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json \
  --test_json /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json \
  --systems stardard strict
```

再运行 baseline 三段评估：

```bash
python src/inference/main/run_baseline_two_stage.py \
  --base_dir /home/penghongen/My_Project/EVAL_OUT \
  --phenix_output_root /home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps \
  --val_json /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json \
  --test_json /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json \
  --systems stardard strict \
  --baselines phenix_real_space_diff_map \
  --device cuda:0
```
