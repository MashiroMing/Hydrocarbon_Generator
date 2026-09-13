# PROJECT_GUIDE — 有机化学异构体生成工具

> 本文档面向后续 AI Agent 与开发者，用于快速理解项目结构与功能。
> 文档中的模块名/函数名/签名均以实际代码为准。

---

## 1. 项目概述

本工具用于有机化学异构体的**生成、去重、3D/2D 可视化**及 **Gaussian 输入文件（.gjf）导出**。

- 支持**烃类**（烷、烯、炔、二烯、环烷、环烯、多烯、多环等）、**卤代烃**（F/Cl/Br/I 取代）和**含氧衍生物**（醇/酚/醚/醛/酮/羧酸/酯及含 O 杂环）的自动枚举。
- **GUI**：Tkinter 实现，支持分子式输入、类型筛选、高级结构约束筛选、列表选择、3D 交互（matplotlib）和键线式渲染（RDKit）。
- **核心引擎**：`GeneratorManager` 统一调度；烷烃使用专用生成器，其他类型使用 `RobustPolycyclicPolyeneGenerator`（基于图论枚举）。
- **数据表示**：所有分子以 `networkx.Graph` 存储，节点和边携带化学属性。
- **质量保证**：全管线末端设**分子式一致性门禁**（`filter_by_expected_formula`），保证输出与用户输入分子式严格一致（防泄漏）；含氧生成采用"骨架不饱和度下探 + 骨架 O 桥/环氧/过氧/双醚桥 + 数学口径过滤"策略，与 MAYGEN 的数学完备口径一致（C7H6O2 覆盖率 ≈98.7%）；附独立工具 `tools/maygen_validate.py` 与 MAYGEN 做异构体数量交叉验证。

---

## 2. 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│  GUI 层 (Main.py – MoleculeApp)                             │
│  - 用户交互、事件绑定、进度反馈、列表/图像显示                │
│  - 坐标计算（_rdkit_coords_* 等模块级函数）                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  生成管理 (utils.py – GeneratorManager)                      │
│  - 分子式解析 → 确定类型 → 分发至具体生成器                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  生成器层                                                   │
│  - AlkaneIsomerGenerator (烷烃)                             │
│  - RobustPolycyclicPolyeneGenerator (其余所有烃类)          │
│  - HalogenSubstitutionGenerator (卤代, halogen/Halide.py)   │
│  - UnifiedOxoGenerator / OxoSubstituentGenerator (含氧)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  筛选层 (structure_filter.py)                               │
│  - StructureConstraint 约束定义（主链长度/环大小/必含基团）   │
│  - filter_isomers_by_constraint 统一后置结构过滤             │
│  - filter_by_expected_formula 分子式一致性门禁（最终收口）    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  工具层 (utils.py, structure_filter.py, molecular_constants) │
│  - 图转换、去重（WL哈希 + 子图同构）                         │
│  - 分子式统计/校验（graph_formula_parts / formula_matches_input）│
│  - 结构约束过滤（主链长度、环大小、必须含基团）              │
│  - 键线式渲染（RDKit SVG → PNG / matplotlib 回退）           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  可视化层                                                   │
│  - 3D：matplotlib + mpl_toolkits.mplot3d                    │
│  - 2D：RDKit MolDraw2D / matplotlib                         │
└─────────────────────────────────────────────────────────────┘
```

**位置说明**：坐标计算（RDKit 嵌入 + 键长松弛）函数定义在 `Main.py` **模块级**（非 `utils.py`），包括 `_rdkit_coords_single` / `_rdkit_coords_with_bonds` / `_embed_best_conformer` / `_relax_bond_lengths` / `_fix_cumulated_diene_coords_standalone` / `_fallback_coords_from_graph`。

**独立工具（不参与程序运行/打包）**：
- `tools/boundary_report.py`：**内部边界验证**（无 MAYGEN 依赖，见 4.9）；
- `tests/test_formula_guard.py`：分子式门禁 + 官能团覆盖回归测试（三级档位见第 8 节）；
- `tools/maygen_validate.py`：MAYGEN 外部对照脚本（**可选**，程序本体不包含 MAYGEN 接口）。

---

## 3. 核心数据结构

### 3.1 分子图 (networkx.Graph)

异构体用无向图表示，节点为原子，边为化学键。

**节点属性**（通过 `G.nodes[node]` 访问）：

| 属性名 | 类型 | 含义 | 示例 |
|--------|------|------|------|
| `label` | str | 原子符号 | `'C'`, `'O'` |
| `halogen_counts` | tuple (F, Cl, Br, I) | 该碳上连接的卤素原子数 | `(1,0,0,0)` 表示一个 F |
| `oxo_counts` | tuple (OH_count, CO_count) | 羟基(-OH) 和 羰基/醚氧 (=O/-O-) 数量 | `(1,0)` 表示一个 OH |

**边属性**（`G[u][v]['bond_type']`）：

- `'single'` / `'double'` / `'triple'`

**关键区分**：
- **骨架氧原子**（如醚键中的 O）作为**独立节点**，`label='O'`；
- **取代基氧**（如 `-OH`、`=O`）通过 `oxo_counts` 记录在碳节点上，不单独建节点。

### 3.2 约束对象 (structure_filter.StructureConstraint)

用于高级筛选，各字段可为 `None`：

```python
@dataclass
class StructureConstraint:
    required_fragment: Optional[str]           # 如 'benzene'
    main_chain: Optional[Tuple[int, int]]      # (min, max) 主链碳数
    ring_size: Optional[Tuple[int, int]]       # (min, max) 环元数
    persist: bool = False                      # 是否跨分子式复用
```

### 3.3 分子式解析结果

`utils.parse_compound_formula()` 返回：

```python
(mol_types, n_carbon, n_hydrogen, n_oxygen,
 halogen_spec, n_deuterium, error)
```

- `mol_types`：list，如 `['alkene', 'cycloalkane']`
- `halogen_spec`：(F, Cl, Br, I) 原子数元组
- 支持氘代 D 标记；卤素视为 H 替代参与不饱和度计算

---

## 4. 关键模块与函数（开发常用）

### 4.1 utils.GeneratorManager

- `generate(mol_type, n_carbon, n_hydrogen=None)`：生成特定类型异构体列表。
- `generate_all(n_carbon, n_hydrogen)`：生成所有可能类型，返回 `[(mol_type, Graph), ...]`，自动去重。

### 4.2 去重机制 (utils.dedup_add_to_buckets)

使用 **Weisfeiler-Lehman 图哈希** 作为粗过滤桶，桶内用 **度签名** 预过滤，最后调用 `nx.is_isomorphic`（带 `node_match` 和 `edge_match`）精确比较。

```python
def dedup_add_to_buckets(G, mol_type, buckets, node_match,
                         node_label_fn, node_sig_fn,
                         canon_key_fn, degree_signature_fn):
```

各生成器因节点/边特征不同，通过 `node_match / node_label_fn / node_sig_fn / canon_key_fn / degree_signature_fn` 参数注入节点特征编码（含 `label`、`halogen_counts`、`oxo_counts`）。

> **凯库勒式差异刻意保留（图口径计数）**：苯环等含环分子的凯库勒式（双键位置不同的
> 两种表示，如邻二氯苯的两种凯库勒式）在数学口径下是**不同的图**（WL 哈希不同、
> 非同构），因此按图计数时各算一个结构，**不去重**。注意这与 MAYGEN 的
> "一个分子一条 SMILES"分子口径不同——本项目计数为**图口径**，用户明确要求保留
> 这一特性（例：C8H8O2+苯环 严格口径 40 个图中仅 34 个唯一分子，属预期）。

### 4.3 结构筛选 (structure_filter.filter_isomers_by_constraint)

统一后置过滤入口，覆盖 `required_fragment`（子图同构检测）、`main_chain`、`ring_size` 三种约束。

- 链状与环状分子自动适用不同特征值（**主链长度** vs **环大小**）。
- 调用方需传入 `feature_fn(mol_type, graph) -> int` 计算特征值。
- 幂等设计，可安全重复调用；对所有生成路径（含氧/纯烃/卤代）统一生效。
- 分子式一致性由 **4.7 分子式一致性门禁**（`filter_by_expected_formula`）最终兜底。

### 4.4 坐标计算（Main.py 模块级函数）

> **注意**：下列函数定义在 `Main.py` 顶部（模块级），而非 `utils.py`。

- 使用 **RDKit ETKDG 嵌入 + MMFF 优化**。
- 对大分子（≥13 碳）采用多种子策略选择最佳构象（最小 C-C 键长）。
- 包含键长松弛（`_relax_bond_lengths`）和累积双键修正（`_fix_cumulated_diene_coords_standalone`）。
- 相关函数：`_rdkit_coords_single` / `_rdkit_coords_with_bonds` / `_embed_best_conformer` / `_fallback_coords_from_graph`。

### 4.5 键线式渲染 (utils.render_skeletal_formula)

```python
def render_skeletal_formula(isomer_data, mol_type, gen_mgr=None, img_size=(400, 300)):
```

- 输入 `nx.Graph` 或规范字符串 → 转换 RDKit Mol → 生成 SVG → 经 `cairosvg` 转 PNG。
- 甲烷单独绘制（无 C-C 键时）。
- 回退方案：matplotlib 手动绘制。

### 4.6 片段库 (oxygen/fragment_library)

- 注册 16 种基团片段（苯环/萘/蒽/菲/联苯/呋喃/吡喃/二氧六环/THF/环氧乙烷/环丙烷/环丁烷/环戊烷/环己烷/降冰片烷/金刚烷/乙基）。
- 常用函数：`list_fragments()`、`get_fragment_info(name)`、`graph_contains_fragment(G, name)`（VF2 子图同构）。

### 4.7 分子式一致性门禁（utils + structure_filter）

统一分子式统计/校验，单一事实源（不信任调用方传入的碳数参数）：

```python
utils.graph_formula_parts(G)                 # → (nC, nH, nO, (f,cl,br,i)) 从图统计
utils.format_graph_formula(G)                # → 分子式字符串（与各 compute_graph_formula 口径一致）
utils.expected_formula_parts(n_carbon, n_hydrogen, n_oxygen,
                             halogen_spec, n_deuterium=0)   # 输入期望组成
utils.formula_matches_input(G, n_carbon, n_hydrogen, n_oxygen,
                            halogen_spec, n_deuterium=0)    # 门禁判定

structure_filter.filter_by_expected_formula(
    all_isomers, n_carbon, n_hydrogen, n_oxygen,
    halogen_spec=(0,0,0,0), n_deuterium=0, to_graph_fn=None)  # 统一兜底过滤
```

- `filter_by_expected_formula` 在 `Main.py::_generate_in_background` 的"统一后置过滤层"之后调用，三条路径（纯烃/含氧/卤代）统一收口；
- 字符串条目经 `to_graph_fn(mol_type, data)` 转图校验，转图失败保守剔除；幂等、纯函数；
- 历史缺陷修复：`oxygen/oxo_generator.py` 与 `halogen/Halide.py` 的 `compute_graph_formula` 的 C 前缀已改为**图内统计**（参数 `n_carbon` 仅作兼容保留）。

### 4.8 含氧生成策略 (oxygen/unified_oxo_generator.py)

O 分配采用"**骨架不饱和度下探 + 骨架 O 桥/环氧/过氧/双醚桥 + 数学口径过滤**"策略，
目标是与 MAYGEN 一致（数学可行/八隅律口径）覆盖醛/酮/羧酸/醚/酯/过氧/环氧/含 O 杂环：

- **骨架 k 下探**：主循环枚举 `k_skel ∈ [max(0, k_min−n_oxygen), k_max]`。每个 `=O`/环氧/双醚桥使产物 k 比骨架高 1（OH / 醚桥 / OOH / 过氧桥 / O 在环不改变 k），目标 k 的含羰基结构由低 k 骨架 + 提升组合而成；`add()` 公式门禁放行命中目标 H 的产物、拦截 k+1 溢出；
- **骨架 O 桥**：`need_bridges` 恒为 True；`BridgeGenerator` 支持三种桥：
  - 醚桥（`ether`，1 O）：C-C → C-O-C；
  - 过氧桥（`peroxy`，2 O）：C-C → C-O-O-C，且支持 C=C → C-C + O-O（1,2-二氧杂环丁烷等，双键降单、k 不变）；
  - 双醚桥（`dioxy`，2 O）：C-C → C-O-C-O 平行双 O（4 元 1,3-二氧杂环，k+1）；
- **环氧插入**：`oxygen/epoxide_generator.py` 的 `EpoxideGenerator` 在 C-C 键上"附加"O 成 3 元 C-C-O 环——单键式（保留单键，k+1）与双键式（C=C 降单，k 不变）；
- **氢过氧化物取代基**：`oxygen/peroxy_substituent_generator.py` 的 `PeroxySubstituentGenerator` 为可用 H 位的 C 附加 R-O-O-H 链（数据模型零改动，k 不变）；
- **O 在环（oring）**：跨 C 迭代对所有 k 启用，模块常量 `_ORING_MAX_ATOMS = 10` 上限保护枚举量（可生成苯并二氧杂环戊烯等含 O 杂环）；
- **片段路径醚桥**：`_generate_from_fragment` 新增"片段 − O − 侧链"合并（辅助函数 `_attach_side_chain_via_o`，与 `_attach_side_chain` 并列，均返回 bool 成功标志），约束模式（必含苯环等）下可生成醚/酯；
- **片段严格校验（"必含基团"口径）**：`_generate_from_fragment` 的 `add()` 收口处对每个
  产物执行 `graph_contains_fragment(g, fragment_name)`（VF2 子图同构，label+bond_type 匹配）。
  产物必须**仍包含**所需片段子图：苯环等片段被醚桥/环氧/双醚桥/O 替换破坏后
  （如苯并氧杂环已无交替单双键六元碳环）不再属于"必含苯环"，会被剔除 —— 与 GUI
  类型筛选（`_has_benzene_ring`）口径一致，保证命令行与 GUI 计数统一
  （C8H8O2+苯环：70 图 → 严格口径 40 图，乙酸苯酯/苯甲酸甲酯均保留）。
  想枚举苯环被破坏的衍生结构时用无约束生成（它们本就在全集中）；
- `_pure_hydrocarbon` 为类内方法（n_oxygen=0 时使用），尊重 k_range。

**加速（2026-06 实施，实测 C7H8O2 无约束 327s → 114~151s）**：
- **进程池并行（`use_parallel`，默认自动：原子数≥7 且 k_skel ≥2 档时启用）**：
  Phase 2d 每个 `k_skel` 一个任务 + oring 跨C 每个 `(extra_c, nc_big, k)` 一个任务；
  worker 本地桶+本地去重，父进程按完成序并入全局（跨任务全局去重）。
  实测并行 2.17×；进程池不可用（打包/环境）自动回退串行。
- **快速模式（`enable_oring_cross_c=False`，GUI"快速生成"勾选）**：跳过 oring 跨C
  迭代（省 C+n 骨架枚举），C7H8O2 串行 327→129s（2.5×）；代价少 ~3% 结构
  （苯并二氧杂环戊烯类稠合 O 杂环），数学完备口径下慎用、缺结构可用完整模式补。
- **add() 门禁前移**：公式门禁先于 WL 哈希，拦截 k+1 溢出候选，省哈希。
- **碰撞安全去重（`_seen_add_safe`）**：`global_seen` 从 set 改为 {wl: [graph,...]}，
  同哈希须 `_graphs_isomorphic`（label+oxo+halo+bond_type）确认才视为重复——
  修复统一生成器长期存在的 **WL 哈希碰撞误删**（与矩阵预言机 `_seen_add` 同源修复；
  C7H8O2 找回 170 个被误删的真实结构：99622 → 99792；小公式与原子预言机完全一致）。
- **骨架缓存**：`_hydrocarbon_skeleton_cache(nc, nh)` 模块级 lru_cache（深拷贝返回防污染），
  跨 generate() 调用复用（GUI 重复生成同一分子式时生效）。

**过滤口径（`oxygen/chem_filter.py`）双模式**：
- `'math'`（**默认**，匹配"数学可行/八隅律"目的）：仅保留价态硬规则（C≤4 / O≤2），
  移除化学稳定性启发式（偕二醇 `gem_diol`、累积烯酮 `allene_cumulene`）——与 MAYGEN 数学枚举口径一致；
- `'chem'`：启用全部规则（供未来需要稳定结构筛选的场景）；
- 全局开关 `oxygen.unified_oxo_generator.set_chem_filter('math'|'chem')`（影响之后新建实例）；
  实例级参数 `UnifiedOxoGenerator(chem_mode=...)`。

> 边界覆盖（C7H6O2，数学口径）：89.9% → 93.2%（环氧）→ 98.47%（OOH/过氧/数学模式）→ **98.69%**（双醚桥）。
> 残余缺失（~1,594）为高不饱和度骨架上的多环稠合 O 杂环（如 `C#CC(=C)C12OC1(C)O2`）；
> "我方多余" 1,013 个全部价态/公式合法（完备预言机自身也确认其为合法结构），数学口径下保留。

### 4.9 边界情况验证（内部为主，MAYGEN 仅可选外部复核）

**内部完备性验证（主要手段，零外部依赖）**：
- `oxygen/boundary.py::internal_boundary_report(formula)`：以**原子矩阵目标 H 剪枝枚举**
  （`AtomicMatrixGenerator.generate_for_target_h`，八隅律价态模型下数学上与 MAYGEN 等价）
  作为**完备性预言机**，对比"树机制集合"
  （`UnifiedOxoGenerator(enable_atomic_fallback=False)`）：
  - **边界缺失 = 完备 − 树**（树机制无法表达的结构），按官能团 SMARTS 分类；
  - **内部覆盖率 = |树 ∩ 完备| / |完备|**；
  - "我方多余 = 树 − 完备"（应为 ~0，出现即需核验）；
- CLI：`python tools/boundary_report.py C3H6O2 C6H6O2 [--top-missing N] [--max-atoms 8]`
  （无需 Java / MAYGEN；N≥9 时矩阵枚举显著变慢，默认上限 8）；
- 等价性实证（N≤6 的 18 公式 + N=7 的 10 公式电池，覆盖醇/醚/醛/酮/酸/酯/过氧/环氧/环/多氧/高不饱和）：
  **全部用例 0 缺失**（预言机绝不缺失 MAYGEN 已有的结构）；其中多数**双向完全等价**，
  其余为**严格超集**（如 C5H6O +1、C6H8O +4、C6H6O +20）——多出的均为 MAYGEN 的 CDK
  生成器系统不枚举的**环丁二烯（反芳香 4 元环）衍生物**（如 `CC1=C(O)C=C1`），八隅律完全合法；
- 交叉验证曾发现并修复预言机的 **WL 哈希冲突 bug**（`C1OC1C1CO1` 与 `C1OC2COC12` 同哈希被
  误删，现改为哈希→列表 + `is_isomorphic` 精确判定）；
- **P0 性能优化（有序生成）**：回溯中施加"单新键类型内前缀序"约束（C 原子按 0..nc-1、
  O 原子按 nc..N-1 顺序首次成键；双新键放行并由末端连通检查兜底），消除 ~N! 倍标号冗余。
  效果（实测）：C6H8O 317→28s（~11×）、C6H10O 139→14s、C5H8O2 78→10s；
  N=7 全部用例（含此前超时的 C6H6O/C5H6O2/C5H4O2）现均 ≤35s 完成。

**MAYGEN（可选外部复核，不参与项目流程）**：
- `tools/maygen_validate.py` 保留为开发期可选外部复核（需本机 Java）；
- **程序本体不含 MAYGEN 接口**；项目运行、回归测试、边界判断均不依赖 MAYGEN。

> **开发政策**：边界/完备性判断一律使用内部预言机（`oxygen/boundary.py`）；
> MAYGEN 仅可作一次性外部对照，禁止将其作为项目功能/测试/文档的依赖。

### 4.10 含氧官能团分类 (oxygen/funcgroup.py)

GUI 结果列表精细筛选用（含氧时显示）：`classify(G) -> frozenset[str]` 返回
**标签集**（一个结构可命中多类，多选"或/与"过滤），`o_distribution(G) -> str`
返回**互斥**的 O 分布形态：

- **标签**（11 类）：醇 / 酚 / 醚 / 过氧(含氢过氧) / 醛 / 酮 / 羧酸 / 酯 /
  环氧 / 含氧杂环 / 偕二醇；
- **O 分布**：`substituent_only`（仅 -OH/=O）`backbone_only`（仅醚/过氧/环氧/杂环）
  `mixed`（混合）`none`（无 O）——与"骨架 k 下探"生成策略对应；
- **判定原则**：纯图快判优先（O 节点 / `oxo_counts` / `bond_type`），零 RDKit；
  苯环判定与 fragment_library 凯库勒苯环语义一致（支持被取代/稠合苯环）；
  酯的 C-O-C 不算醚、环氧/含氧杂环/醚按 O 环情况互斥；
- 分类口径与 `tests.coverage_stats` 同源（酯/酸/醚逻辑一致）；
- 测试：`tests/test_funcgroup.py`（手工构造图 14 例 + C3H6O/C7H8O+苯环/
  C8H8O2+苯环/C6H6O 生成集成），已挂入 `test_formula_guard.py --fast`；
- 已知特性：C6H6O 全集合无醇标签（唯一 -OH 在苯环上 → 酚，无非环 sp3 C 挂 OH）。

**GUI 集成（Main.py）**：
- `_display_meta` 元组扩展为 `(formula, cn_name, feature, fg_tags, o_dist)`；
  `_build_display_meta` 含氧分支用 `classify`/`o_distribution` 一次性计算并缓存
  （纯图判定，无 RDKit；苯环检测按需单节点判定，无 OH 的图零开销）；
- 列表行尾追加官能团标签（`_format_isomer_entry`，如 `  [醚,酯]`）；
- **含氧官能团筛选面板**（`_update_fg_panel`，仅 n_oxygen>0 时 pack 显示）：
  官能团复选框（实时计数，如 "酯 (3)"）+ 或/与 模式单选 + O 分布下拉（只列实际
  存在的形态）+ 清空按钮；
- `_repopulate_from_current_filter` 组合过滤：**特征值下拉 ∧ 官能团勾选(或/与) ∧
  O 分布**，结果标签显示命中数与筛选摘要；
- 计数为**图口径**（凯库勒变体分开计数，见 4.2），与列表行数一致。

**GUI 加速（配合 4.8 生成侧加速）**：
- **阶段进度 + 实时计数**：`generate(progress_cb=...)` 每完成一个任务回调
  （阶段/任务进度/去重后已并入结构数），进度条 10→90 分阶段推进；
- **列表无显示上限**：`_populate_isomer_listbox` 批量插入（5000/批），
  实测 10 万行 ~1.6s 渲染完成（旧版逐行插入 + 5000 行上限已移除）；
- **官能团标签惰性化**：≤3000 行同步计算；更大集合由后台线程分块（1000/块）计算
  并逐块刷新面板计数（面板显示"标签计算中 X/Y…"），不阻塞界面；
- **快速生成开关**（公式输入框旁的"快速生成"复选框）→ `enable_oring_cross_c=False`；
- 已移除"按分子式分组"复选框（用户按分子式检索，分组无意义；相关
  `_group_view`/`_row_is_group_header`/分组标题行逻辑一并删除）。

**GUI 响应式布局（窗口缩放自适应）**：
- FG 面板 O分布/模式行与官能团复选框行采用**流式换行布局**（`_flow_widgets`，
  绑定 `<Configure>` + `root.after` 兜底重排）：窄窗口自动换行（800px 时复选框
  2 行），宽窗口自动回到单行，任何尺寸都完整可见；
- `list_frame` 子级改为 **grid**（filter 行/ FG 面板/列表区），FG 面板用
  `grid_remove/grid` 显隐并保留配置（避免 pack 重排顺序导致宽度塌缩）；
- **结果区列权重 3:1**（列表列优先）+ 键线式 Canvas 默认宽 300（图片按实际宽等比
  缩放），右侧信息栏不再抢占列表宽度（800px 时 list_frame 282→379px）。

---

## 5. 开发规范（AI Agent 必读）

### 5.1 命名约定

- **模块**：小写下划线，如 `structure_filter.py`
- **类**：大驼峰，如 `GeneratorManager`
- **函数/变量**：小写下划线，如 `parse_compound_formula`
- **私有函数**：前缀 `_`，如 `_rdkit_coords_single`

### 5.2 导入规范

- 所有导入集中在文件头部，按 **标准库 → 第三方库 → 项目模块** 分组。
- 项目内导入使用绝对路径（如 `from utils import ...`），避免相对导入引起打包问题。
- PyInstaller 打包后通过 `sys._MEIPASS` 动态添加路径（已在 `runtime_hook.py` 处理）。

### 5.3 线程安全

- 耗时的生成/坐标计算在**后台线程**执行（`threading`）。
- GUI 更新必须通过 `root.after(0, callback)` 回到主线程。
- 取消操作通过 `cancel_requested` 标志和窗口关闭检查实现。
- 注意：`tkinter.Image.__del__` 的线程安全问题已修复（见 `Main.py` 开头）。

### 5.4 添加新分子类型（扩展指南）

1. 在 `MOL_NAMES_CN` / `MOL_NAMES_FILTER` / `MOL_NAMES_EN` 字典（`utils.py`）中添加名称映射。
2. 在 `_rdt_to_mol_type` 中添加 `(环数, 双键数, 三键数)` 到新类型的映射。
3. 若生成逻辑不同于现有类型，需在 `GeneratorManager.generate` 或 `RobustPolycyclicPolyeneGenerator` 中增加分支。
4. 在 `_TYPE_ORDER` 列表中纳入排序顺序。
5. 若需新约束，在 `structure_filter.StructureConstraint` 添加字段并更新 `filter_isomers_by_constraint`。

### 5.5 调试技巧

- 启用详细日志：`import logging; logging.basicConfig(level=logging.DEBUG)`
- 打印中间图：`nx.write_adjlist(G, "graph.adjlist")`
- 检查 RDKit 分子：`Chem.MolToSmiles(mol)`

### 5.6 AI 技能集（chemistry-and-biology）

- 项目根 `chemistry-and-biology/` 收录 **27 个化学/生物 AI 技能**（`<name>/SKILL.md` 规范），索引与维护说明见该目录 `skills.md`；
- 已通过**目录联接**注册到 DSH 项目级技能根：`.dsh/skills/<name> → chemistry-and-biology/<name>`，可直接按技能名调用（`skill` 工具）加载完整指令；
- 修改技能正文直接编辑 `SKILL.md`（DSH 每次调用实时重读）；新增/删除技能需同步维护 junction（重建脚本见 `skills.md`）。

---

## 6. 常见问题与解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| 生成卡顿 | 大分子枚举量巨大 | 使用 ProcessPoolExecutor 并行计算坐标（已实现） |
| 3D 坐标扭曲 | RDKit 嵌入陷入局部极小 | 增大种子数（`n_seeds`）或回退至 2D 坐标 |
| 键线式渲染空白 | RDKit 无法处理某些图 | 确保图节点属性完整（`label` 必须设置） |
| 卤代/含氧显示异常 | 节点属性未正确传递 | 检查 `halogen_counts` 和 `oxo_counts` 的编码格式 |
| 保存 .gjf 时坐标缺失 | 图数据序列化失败 | 使用 pickle 序列化图对象时确保所有属性可 pickle |
| 结果出现分子式不符结构 | 生成路径泄漏（=O 使 k+1、原子矩阵全 k 枚举、片段合并改变 k） | 末端分子式门禁 `filter_by_expected_formula` 兜底剔除（见 4.7） |
| 含氧结果缺酯/醚/羰基类 | 旧版骨架 k 硬对齐 + 桥接仅限 k≤2 | 新版 k 下探 + 骨架 O 桥放开 + 片段醚桥（见 4.8） |
| 约束模式（必含苯环等）缺"侧链内部 O 桥"型结构（如 C8H8O2 缺苯甲酸甲酯、只有乙酸苯酯） | `_generate_from_fragment` 只做直连 C-C 合并与"片段−O−侧链"交界醚桥，从不放置**侧链内部 O 桥** | 方案 A：直连合并骨架复用主循环 `_apply_plan` backbone 机制（任意 C-C 单键醚桥 + =O 等组合），add() 门禁自动收口（见 4.8） |
| 结果缺过氧/偕二醇/环氧等数学可行结构 | 化学稳定口径过滤（`ChemFilter` 默认 'chem' 风格规则） | 默认切换为**数学口径** `set_chem_filter('math')`，仅保留价态硬规则（见 4.8） |
| 约束模式计数：命令行 70 vs GUI 40（C8H8O2+苯环） | GUI 在约束生成后**叠加了类型筛选"含苯环"** 的 `_has_benzene_ring` 严格 Kekulé 环检查，剔除了苯环被氧机制破坏的 30 个衍生结构（苯并氧杂环类，确实无苯环） | 已按**严格口径**统一：`_generate_from_fragment.add()` 收口处做片段子图校验（见 4.8），命令行 = GUI = 40；类型筛选"含苯环"与"必含苯环"约束语义一致 |
| 结果计数与"唯一分子数"不一致（如 40 图 = 34 个唯一分子） | 凯库勒式双键位置不同的结构是**不同的图**（图口径计数） | **刻意保留不去重**（见 4.2 说明），这是特性而非缺陷 |
| 如何判断边界/完备性 | 早期依赖 MAYGEN 差集（外部工具，需 Java） | **已完全解除对 MAYGEN 的依赖**：内部预言机 `oxygen/boundary.py` + `tools/boundary_report.py`（原子矩阵目标 H 剪枝，数学上与 MAYGEN 等价，见 4.9） |
| 无约束含氧大公式生成很慢（如 C7H8O2 全集 5 分钟+） | 骨架枚举 + O 机制组合爆炸（单线程） | 已加速：**进程池并行（2.17×）+ 快速生成开关（2.5×）+ 进度实时反馈**（见 4.8）；约束模式走片段路径本身很快 |

> **特定分子结构测试技巧**：针对某一分子/官能团做定向测试时，**先查 `oxygen/fragment_library.py` 是否已有预结构**（苯环 `get_fragment('benzene')`、呋喃/吡喃/THF/环氧乙烷/环烷烃/萘/蒽等 16 种），直接用预结构构造测试图，**避免从头生成分子**（可显著加速调试）。
> 新增预结构：在 `fragment_library.py` 添加 `_build_xxx` 函数并在 `_registry` 注册即可。

> **去 MAYGEN 依赖政策（重要）**：本项目的运行、回归测试与边界判断**均不依赖 MAYGEN**；
> `MAYGEN/` 目录与 `tools/maygen_validate.py` 仅保留为**可选的一次性外部对照**，不参与项目流程。
> **后续开发禁止再引入对 MAYGEN（或其接口/结果）的任何依赖**——边界/完备性一律用内部预言机判断。

---

## 7. 外部依赖

- `networkx` (>=2.5)
- `rdkit-pypi` 或 `rdkit` (2021.09+)
- `matplotlib` (>=3.3)
- `cairosvg`（可选，用于 PNG 导出）
- `Pillow`（用于图像处理）
- `scipy`（用于环优化）

---

## 8. 打包与辅助文件

- `Hydrocarbon_Generator.spec` / `Hydrocarbon_Generator_linux.spec`：PyInstaller 打包配置。
- `runtime_hook.py`：PyInstaller 运行时钩子，确保子进程能找到项目模块。
- `icon.ico`：应用图标。
- `molecular_constants.py`：存储分子常数相关的静态数据（含 base64 图像资源）。
- `tests/test_formula_guard.py`：分子式门禁 + 官能团覆盖回归测试。三级档位：
  `--smoke`（秒级，小分子+单元断言）/ `--fast`（分钟级，中小分子+约束用例）/ 默认完整档
  （重量级用例并行，约 10-15 分钟，瓶颈为 C7H6O2 等大公式生成本身）。
- `tests/test_funcgroup.py`：官能团分类模块测试（手工构造图 14 例 + 生成集成，
  已挂入 `test_formula_guard.py --fast`）。
- `oxygen/funcgroup.py`：含氧官能团分类模块（GUI 精细筛选用，见 4.10）。
- `O_project_guide.md`：**含氧生成专项总结**（化学口径 / 生成机制 / 去重验证 / 性能 /
  踩坑分类收录 / N、P 扩展蓝图），扩展 N/P 元素前必读——本文档含氧细节从简，见该文件。
- `oxygen/boundary.py` + `tools/boundary_report.py`：**内部边界验证**（原子矩阵目标 H 剪枝
  预言机 vs 树机制，无 MAYGEN 依赖，见 4.9）。
- `tools/maygen_validate.py`：MAYGEN 外部对照工具（**可选，仅一次性复核**，需本机 Java，
  不参与项目流程）。
- `MAYGEN/`：MAYGEN 开源异构体生成器（Java JAR 与源码，**仅保留为可选外部对照，程序不调用**）。
- `chemistry-and-biology/`：AI 技能集（27 个 `SKILL.md` + `skills.md` 索引）。
- `.dsh/skills/`：技能注册目录联接（指向 `chemistry-and-biology/`，勿删除，重建脚本见 `skills.md`）。
