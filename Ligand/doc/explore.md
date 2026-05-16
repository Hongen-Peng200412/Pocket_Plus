# RCSB 单源路线探索记录

本文记录为什么最终选择 RCSB PDB 单源路线，为 Make_Data 的 small molecule ligand 补充 SMILES、InChI/InChIKey 和 RCSB native mol2。

本文偏“探索与判断”。正式实现计划见 `Ligand/doc/implement_plan.md`，实际执行和调试过程见 `Ligand/doc/exec_plan.md`，最终数据字段见 `Ligand/notes_of_dataset.md`。

文档分工：

```text
explore.md          # 解释为什么走 RCSB 单源路线
implement_plan.md   # 解释正式程序如何设计
exec_plan.md        # 解释实际运行中如何修复和验收
notes_of_dataset.md # 解释最终产物、字段、枚举值和统计口径
```

## 1. 初始问题

Make_Data 目前已经能从 PDB/mmCIF 解析结构、候选 ligand、标签和 pocket 相关数据，但没有小分子 SMILES / mol2 enrichment 逻辑。

后续要接入的工具包括：

```text
DockEM
EMERALD-ID / Automated identification of small molecules in cryo-EM data
PocketXMol
```

这些工具对输入的最低要求不同，但对短期方案而言，最核心的是：

```text
1. 知道 Make_Data 里的某个 ligand candidate 对应 RCSB 的哪个 ligand instance。
2. 能拿到该 instance 的 RCSB native mol2。
3. 能拿到该 CCD 的 SMILES / InChI / InChIKey。
4. 能证明 Make_Data candidate、RCSB CIF instance、native mol2 三者确实对齐。
```

## 2. 为什么不是直接从本地 CIF “一键恢复” SMILES/mol2

本地 full CIF 有全原子坐标，但坐标本身不能唯一恢复完整化学图。

仅凭坐标通常无法高置信恢复：

```text
bond order
aromaticity
formal charge
protonation
tautomer
partial charge
force-field atom type
```

mol2 不只是坐标文件，通常还包含：

```text
原子类型
键
部分电荷
三维构象
```

SMILES 则主要表达化学身份和 2D 化学图。SMILES 确定后，也不能唯一确定 docking 所需的 3D 构象、质子化、电荷和参数化。

因此，“从本地 CIF 一键无损得到 SMILES + docking-ready mol2”并不是高置信路径。更稳妥的做法是直接向权威结构源索取：

```text
RCSB chemical component descriptor -> SMILES/InChI/InChIKey
RCSB ModelServer ligand endpoint   -> native ligand mol2
RCSB full CIF                      -> instance 对齐和坐标校验
```

## 3. 为什么短期选择 RCSB 单源

RCSB 对一个 PDB 条目同时提供：

```text
1. full structure CIF
2. chemical component descriptor
3. ligand instance native mol2
4. ligand instance SDF/mmCIF 等文件
5. 页面可见的 ligand table 和 chain/residue 信息
```

这意味着 Make_Data 已有的字段：

```text
pdb_id
candidate_id
resname / CCD ID
chain_id
res_id
insertion_code
n_heavy_atoms
candidate_coords
```

足以构造一条 RCSB 单源对齐路线。

RCSB 单源的优点：

```text
1. 数据来源统一，避免多数据库化学身份冲突。
2. PDB full CIF、chemcomp descriptor、native mol2 来自同一条 RCSB 记录。
3. 不需要 SDF/mmCIF 转 mol2，符合“只认原生 mol2”的约束。
4. 可用 Make_Data 坐标反向验证 RCSB instance。
5. 可作为长期 Q-BioLiP/生物学相关性注释的底层结构缓存。
```

## 4. Q-BioLiP 的位置

Q-BioLiP 更适合作为后续的 biological relevance / interaction / binding site annotation 来源，而不是短期 `(SMILES, native mol2)` 的唯一来源。

短期目标是：

```text
Make_Data candidate -> RCSB ligand instance -> SMILES/native mol2
```

长期可在这个映射表上追加：

```text
qbiolip_id
qbiolip_site_id
biological_relevance
binding residues
interaction annotation
```

因此当前路线不是否定 Q-BioLiP，而是把它放在更适合的位置：

```text
RCSB: 化学身份、结构文件、native mol2、instance 坐标
Q-BioLiP: 生物相关性、相互作用、binding site 语义
```

## 5. 初始本地端到端实验

在本地下载的 Make_Data parsed_pdb 子集上，优先测试用户指定样本：

```text
3j7a
5bki
5gmk
5i68
```

随后扩展到本地样本目录：

```text
C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb
```

本地实验结果：

```text
class-4 ligand instances = 58
PASS_HIGH = 58
失败 = 0
```

通过的 CCD 包括：

```text
34G, 6ES, 6ET, 6EU, 6O8, 6O9, 6OE, CL, FAD, GNP, GTP, PAR, PGW
```

这证明了 RCSB 单源路线在样本级别可行。

## 6. 大规模运行后的关键发现

### 6.1 糖类不在 nonpoly scheme

第一版只查 `_pdbx_nonpoly_scheme`，导致大量糖类失败。

诊断结果：

```text
RCSB_INSTANCE_MATCH_FAILED 中 98.6% 是 NAG/MAN/BMA/FUC/GAL 等糖类或支链糖类。
这些 monomer 通常记录在 _pdbx_branch_scheme。
```

修复：

```text
增加 _pdbx_branch_scheme 解析和匹配。
```

### 6.2 一些 HETATM 只适合从 atom_site 兜底

剩余非糖类失败中，许多不在 scheme 表里，但 `_atom_site` 中有完整 comp/chain/seq/coords。

例子：

```text
6AP1 ACE chain G res 0
6VMI Y5P chain A5 res 1
6VMI P5P chain A5 res 2
7FGI GTA chain M res 1003
8CEP KBE/DPP/UAL/MYN chain V res 1-4
9IF4 S0R chain Y res 1
```

修复：

```text
增加 _atom_site fallback。
优先使用 Make_Data candidate_coords 做坐标唯一化。
坐标无法唯一化时，再使用 auth/label chain + seq 精确匹配。
```

### 6.3 ModelServer 参数名容易误导

RCSB ModelServer ligand endpoint 参数名是：

```text
auth_seq_id
```

但实际对本任务应传入 RCSB 页面可见的 residue number，即：

```text
pdb_seq_num
```

尤其在 branch glycan 中，`auth_seq_num` 可能与 `pdb_seq_num` 顺序相反。误用 `auth_seq_num` 会下载相邻 monomer 或导致坐标错配。

### 6.4 altLoc 不能只接受 A

少数 ligand instance 只有 `label_alt_id=B`。如果只接受空 alt / A / 1，会把 CIF instance 提取成 0 个原子。

修复策略：

```text
优先保留空 alt + A/1。
如果只有 B/C/...，选择排序后的第一个可用 altLoc。
```

### 6.5 mol2 元素推断需要完整周期表

`BEF` 等无机配体中，mol2 atom type 可能是 `BE`。旧逻辑误推为 `B`。

修复：

```text
mol2_parser.py 使用完整周期表推断元素。
```

## 7. 最终可行性判断

最终服务器验收：

```text
class4_candidate_count = 190256
PASS_HIGH = 190093
RCSB_NATIVE_MOL2_MISSING = 156
VALIDATION_FAILED = 7
RCSB_INSTANCE_MATCH_FAILED = 0
```

成功率：

```text
全量 class-4 口径:
190093 / 190256 = 99.914%

排除 RCSB ModelServer native mol2 外部缺失口径:
190093 / (190256 - 156) = 99.996%
```

最终结论：

```text
RCSB 单源路线足够支撑短期方案。
Make_Data ligand -> RCSB ligand instance 的映射已经打通。
native mol2 可获得条目中，SMILES/native mol2 pair 的高置信获取率超过 99.5%。
```

## 8. 剩余边界

最终 `RCSB_NATIVE_MOL2_MISSING = 156` 来自 RCSB ModelServer 外部源不可用，主要错误：

```text
HTTP 404: Could not find source file for 'pdb-bcif/{pdb_id}'
```

最终 `VALIDATION_FAILED = 7`：

```text
6JLU / CLA / chain 18 / res 311
  Make_Data 重原子数为 5，RCSB CIF/native mol2 为 46。

7V68 / IXO / chain R / res 501
7V68 / 2CU / chain R / res 502
9O7S / 1KP / chain E/F/G/H / res 201
  重原子数一致，但 Make_Data vs RCSB CIF 坐标超过严格阈值。
```

这些残留不构成路线失败，更适合作为数据审计或 Make_Data candidate 质量复查对象。

## 9. 对 docking 工具的意义

当前 enrichment 不运行 docking，只判断输入文件是否满足格式要求。

最终 readiness：

```text
DockEM ready = 190093
EMERALD-ID ready = 190093
PocketXMol ready = 190093
NOT_READY = 163 = 156 native mol2 missing + 7 validation hard cases
```

注意：

```text
1. RCSB native mol2 可能没有氢原子。
2. 单原子 ligand 可能没有 bond。
3. DockEM 真正运行前仍可能需要按工具要求准备 protein.mol2、加氢、map.mrc、binding site 文件和参数。
4. 本项目当前只保证 ligand 侧 SMILES/native mol2 的高置信 enrichment。
```
