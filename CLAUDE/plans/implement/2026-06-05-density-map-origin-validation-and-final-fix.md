# 密度图 origin 不一致的临时过滤与最终修复计划

## 背景

`unet_c2` 推理在 `emd_25763 / 7t9n` 上报错:

```text
真实密度图与模拟密度图重采样后 origin 不一致:
exp=[-0.844 -0.844 -0.844], sim=[-0.71233594 -0.71233594 -0.71233594]
```

`unet_c1` 能继续运行的原因不是该样本几何正确，而是 `unet_c1` 只启用 `exp_clipnorm_nopost`，不会读取 `sim_map_path`。`unet_c2` 启用 `diff_clipnorm_nopost`，必须同时读取真实密度图和模拟密度图，因此触发 `src/inference/parse_input.py` 中的 fail-fast 校验。

## 已确认事实

1. 当前推理读图口径来自 `processedPDB_EMDB_binder.utils.mrc_tools.load_map()` 和 `make_model_grid()`。
2. `emd_25763` 的真实图和模拟图 raw shape、voxel size 一致，但 header 表达 origin 的方式不同。
3. 真实图 header 近似为 `origin=(0,0,0), nstart=(-1,-1,-1)`，当前 loader 读成 `-0.844 Å`。
4. 模拟图 header 近似为 `origin=(-0.844,-0.844,-0.844), nstart=(0,0,0)`，当前 loader 把 `origin` 再乘一次 voxel size，读成 `-0.71233597 Å`。
5. `Bundle_of_Maps/simulated_map/gen_chimera_cmds.py` 使用 Chimera:

```text
open {receptor_cif_path}
open {emdb_map_path}
volume #1 step 1
molmap #0 {resolution} onGrid #1
volume #2 save {simu_map_path}
```

这说明模拟图数据网格大概率来自 `onGrid #1`，但保存后的 MRC header 约定没有与原始 EMDB map 保持一致。

## 本轮临时修复

本轮实现的临时修复目标是: 生成样本 JSON 时提前过滤会在当前推理口径下失败的条目。

改动边界:

- 新增 `utils/validate_density_map_pairs.py`。
- 修改 `src/inference/utils/yield_json_from_raw_sample.py`。
- 默认开启真实图与模拟图几何校验。
- 同时校验两类模拟图字段:
  - `sim_map_path`: 真实 receptor 模拟图, 默认来自 `/storage/penghongen/simulated_receptor_map`
  - `sim_map_path_cryoatom`: cryoatom receptor 模拟图, 默认来自 `/storage/penghongen/simulated_cryoatom_map`
- `yield_json_from_raw_sample.py` 保持顺序扫描; 达到 `max_accept` 后立即停止扫描后续候选。
- 独立批量审计脚本 `utils/validate_density_map_pairs.py` 支持 `joblib` CPU 并行。
- `--skip_density_map_validation` 可关闭临时校验。
- `load_raw_pairs()` 会保留两套模拟图路径字段; 当前推理主链路仍按既有 `sim_map_path` 字段选择实际输入模拟图。

校验口径:

1. 按当前推理路径使用 `load_map()` 读取真实图和模拟图。
2. 按 `make_model_grid()` 的几何计算口径得到重采样后 `shape / voxel_size / origin`。
3. 只要任意已存在模拟图字段不匹配，就过滤该候选条目。

## 最终修复建议

最终修复不建议直接放宽 `parse_input.py` 中的 origin 校验，也不建议本轮立刻修改 `load_map()` 的 origin 解释方式。

原因:

1. `load_map()` 已经被训练数据处理、BOX 切分、GT 构造、推理缓存等多条链路依赖。
2. 直接修 loader 到标准 MRC 物理口径可能改变历史训练数据与现有 checkpoint 的坐标解释。
3. 当前异常更像是模拟图生成产物 header 与真实 EMDB map header 约定不一致，而不是模型推理需要接受错位网格。

推荐最终方案:

1. 在 `Bundle_of_Maps/simulated_map` 下新增模拟图 header 归一化步骤。
2. 对每个 `{emdb_id}.mrc`，读取对应真实 EMDB map 的几何 header。
3. 将模拟图的几何字段归一化到真实 EMDB map:
   - `nxstart / nystart / nzstart`
   - `origin`
   - `mx / my / mz`
   - `cella`
   - `mapc / mapr / maps`
   - `voxel_size`
4. 保留模拟图数据数组本身，只修正 header 几何约定。
5. 对真实 receptor 模拟图目录和 cryoatom 模拟图目录分别执行:
   - `/storage/penghongen/simulated_receptor_map`
   - `/storage/penghongen/simulated_cryoatom_map`
6. 修正后用 `utils/validate_density_map_pairs.py` 批量验收。
7. 生成修复 manifest，记录每个样本修复前后的 header、校验状态和输出路径。

## 推荐执行步骤

1. 在服务器上抽样审计两套模拟图:

```bash
python /home/penghongen/My_Project/Pocket_Plus/utils/validate_density_map_pairs.py \
  --input_json /home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_100.json \
  --output_json /home/penghongen/My_Project/tmp/protein_100_checked.json \
  --report_json /home/penghongen/My_Project/tmp/protein_100_density_report.json \
  --n_jobs 16
```

2. 统计失败模式:
   - 只差 origin
   - shape 不一致
   - voxel size 不一致
   - 文件无法读取

3. 如果绝大多数失败是 header origin 表达差异，则实现 header 归一化脚本。
4. 归一化前备份原模拟图目录或写入新的输出目录，避免覆盖原始产物。
5. 用同一个校验器验收归一化后的目录。
6. 重新生成推理 JSON。
7. 重新跑 `unet_c2` 参数搜索。

## 非目标

- 本轮不修改 `Bundle_of_Maps/simulated_map/gen_chimera_cmds.py`。
- 本轮不修改 `processedPDB_EMDB_binder.utils.mrc_tools.load_map()`。
- 本轮不批量修改服务器 `/storage` 中已有 `.mrc` 产物。
- 本轮不放宽 `src/inference/parse_input.py` 的 fail-fast 校验。

## 风险与注意事项

1. 如果某些模拟图不仅 header 不一致，而且数据数组本身没有对齐到真实 EMDB 网格，单纯复制 header 会掩盖真实问题。必须先按失败类型分层统计。
2. 如果未来决定修 `load_map()` 到更标准的 MRC 物理口径，需要重新审计历史训练数据 `.npz`、推理 cache 和 checkpoint 输入语义。
3. 独立校验脚本的 `joblib` 并行会同时读取多张大 map，`--n_jobs` 不宜超过 Slurm 分配 CPU，也要留意节点内存。
