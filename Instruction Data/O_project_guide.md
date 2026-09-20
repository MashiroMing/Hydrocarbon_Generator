# O_project_guide.md — 含氧异构体生成：AI Agent 操作手册

> **面向对象**：需要修改/调试/扩展含氧生成的 AI Agent 与开发者。
> **本文只讲含氧生成**，与 `PROJECT_GUIDE.md`（全项目架构/函数级指南）分工：
> 全局架构、数据模型、WL 机制细节见 PROJECT_GUIDE；本文给 agent **可执行的操作手册**——
> 代码地图、不可破坏的不变量、验证命令、避坑自查、N/P 扩展 TODO。
>
> **30 秒启动**：
> - 改任何生成代码前先跑：`python tests/test_formula_guard.py --fast`（当前基线 PASS=72）；
> - 两个"绝对不能破坏"：① 公式门禁全链路口径一致；② 去重必须碰撞安全（`_seen_add_safe`）；
> - 含氧生成一句话：**纯 C 骨架（k 下探）× O 机制组合，公式门禁 + 碰撞安全去重双保险**。

---

## 1. 代码地图（模块 → 关键函数 → 职责 → 常见改动场景）

| 文件 | 关键函数 | 职责 | 常见改动场景 |
|---|---|---|---|
| `oxygen/unified_oxo_generator.py` | `UnifiedOxoGenerator.generate()` | 含氧生成**主入口**（约束解析、任务派发、F3 补漏） | 加参数/加机制/改并行 |
| 同上 | `_phase2d_single` / `_oring_single` / `_parallel_worker` | 单任务执行（本地桶+本地去重） | 任务粒度调整 |
| 同上 | `_apply_plan` | **机制步骤链**（醚→过氧→环氧→OOH→双醚桥→oring→OH/=O） | 加新 O 机制 / N 机制 |
| 同上 | `_plan_backbone_allocations` / `_plan_substituent_allocations` | O 分配方案枚举 | 加新 backbone/取代基类型 |
| 同上 | `_seen_add_safe` / `_graphs_isomorphic` | **碰撞安全去重**（哈希→列表+VF2） | ⚠️ 别改语义（见 §2） |
| 同上 | `_generate_from_fragment` | 约束模式（片段路径 + 方案 A 侧链内部 O 桥） | 加片段约束 / 改严格口径 |
| `oxygen/atomic_matrix_gen.py` | `AtomicMatrixGenerator.generate_for_target_h` | **完备性预言机**（目标 H 剪枝） | N/P 原子类型扩展 |
| `oxygen/funcgroup.py` | `classify` / `o_distribution` | 官能团标签 + O 分布分类（GUI 筛选用） | 加官能团 / N/P 分类 |
| `oxygen/fragment_library.py` | `_build_xxx` + `_registry` / `graph_contains_fragment` | 预结构片段库 + 子图判定 | 加预结构（苯环模板 `_build_benzene`） |
| `oxygen/boundary.py` + `tools/boundary_report.py` | `internal_boundary_report` | 树机制 vs 预言机覆盖率验证 | 验证完备性 |
| `utils.py` | `graph_formula_parts` / `format_graph_formula` / `expected_formula_parts` | **公式计算（单一事实来源）** | ⚠️ H 规则改动=全盘重算 |
| `structure_filter.py` | `filter_by_expected_formula` / `filter_isomers_by_constraint` | 末端公式门禁 + 约束过滤 | N/P 公式支持 |
| `tests/test_formula_guard.py` / `tests/test_funcgroup.py` | 各断言函数 | 回归（smoke/fast/完整三档） | 加断言 |

---

## 2. 不可破坏的不变量（改代码的底线）

> 违反任何一条 = 静默产出错误结果。每条给出"破坏后的症状"供自查。

| # | 不变量 | 破坏后的症状 | 相关代码 |
|---|---|---|---|
| I1 | **公式门禁全链路口径一致**：`graph_formula_parts` → `format_graph_formula` → `expected_formula_parts` → `filter_by_expected_formula` 与生成内 `add()` 门禁必须同口径 | 结果混入分子式不符结构（历史：C7H8O→C6H6O 泄漏） | utils.py + unified_oxo_generator.add() |
| I2 | **去重必须碰撞安全**：同 WL 哈希必须 VF2 同构（含 label/oxo/halogen/bond_type 全属性）才算重复；禁止退回裸 `set` | 计数莫名减少（历史：C4H6O2 262、C7H8O2 99622） | `_seen_add_safe` / `atomic_matrix_gen._seen_add` |
| I3 | **图口径计数**：凯库勒式变体（双键位置不同）是不同图，**不去重** | 计数被"合并"（违背用户明确要求） | 各 add() |
| I4 | **并行 == 串行**：`use_parallel=True` 与 `False` 结果集合必须完全一致 | 计数不一致（必查） | generate() 任务派发/合并 |
| I5 | **严格片段口径**：约束"必含基团"= 产物**仍含**片段子图（VF2）；苯环被氧机制破坏的结构不算 | 约束计数与 GUI 不一致（历史：70 vs 40） | `_generate_from_fragment.add()` |
| I6 | **参考计数**（回归锚点，改动后必须逐一核对）：C3H6O2=34、C4H6O2=263、C5H8O2=1168、C6H10O=747、C6H8O=1623、C6H12O=211；C7H8O2 完整=99,792、快速=96,785；约束：C8H8O2+苯环=40、C7H6O2+苯环=17、C7H8O+苯环=7 | 计数漂移 | 见 §7 验证命令 |
| I7 | **k/H 贡献表**（§4 表）：`=O`/环氧(单键)/dioxy 各 k+1；其余机制 k 不变——改它=所有生成逻辑重算 | 全面错乱 | graph_formula_parts + _apply_plan + 预筛公式 |

**改动影响面矩阵（改 X 必须重跑 Y）**：

| 改动 | 必跑验证 |
|---|---|
| `_apply_plan` / `_plan_*`（机制/分配） | 计数电池（I6）+ 覆盖断言 + fast 套件 |
| 去重逻辑（`_seen_add_safe` 等） | 并行/串行等价 + C7H8O2 计数 |
| `graph_formula_parts` / 公式门禁 | 全量公式门禁测试 + 约束回归 |
| 并行派发/合并 | 等价电池 + 基准 |
| `fragment_library`（加片段） | 约束回归 + funcgroup 测试 |
| `funcgroup`（分类） | `tests/test_funcgroup.py` |

---

## 3. 关键决策点（改参数前先确认）

| 参数 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `chem_mode` | `UnifiedOxoGenerator(chem_mode=...)` / `set_chem_filter()` | `'math'` | `math`=八隅律硬规则（完备口径）；`chem`=加稳定性启发式。**测试/GUI 默认 math** |
| `use_parallel` | `generate(use_parallel=...)` | `None`=自动 | 自动条件：`nC+nO ≥ 7` 且 k_skel ≥ 2 档；worker ≤ 4；失败自动回退串行 |
| `enable_oring_cross_c` | `generate(...)` | `True` | `False`=快速模式（跳过 oring 跨C，少 ~3% 稠合 O 杂环，提速 ~2.5×） |
| F3 原子回退 | generate() 内条件 | 仅 `k_max < 4` | k≥4 时矩阵枚举极慢且收益低，**不要无差别开启** |
| 约束严格口径 | `_generate_from_fragment.add()` | 严格 | 产物必须仍含片段子图（I5） |

---

## 4. 化学模型速查（数学口径 = 八隅律）

- 口径：只认**价态饱和/电子配对**，不考虑稳定性/芳香性/键能。
- 两种 O 身份（**最关键数据模型决策**）：
  - **骨架 O**：`label='O'` 独立节点（醚/过氧/环氧/杂环中的 O）；
  - **取代基 O**：C 节点属性 `oxo_counts=(OH, CO)`（羟基/羰基，**不建独立节点**）。
- 基础：`k = (2·nC + 2 − nH − 卤素) / 2`；单节点 H：`h = max(0, max_v − bond_load − oh − 2·co − halo)`，
  `总 H = Σh + Σoh`，`nO = Σ(oh+co) + 骨架O数`。

**O 机制对 k/H 的贡献（I7，全部生成逻辑的地基）**：

| 机制 | 产物 k | 对 H | 说明 |
|---|---|---|---|
| 取代基 `=O` | k+1 | −2H | 醛/酮/酸/酯的羰基 |
| 环氧（单键式） | k+1 | −2H | C-C + O 成 3 元环 |
| 双醚桥 dioxy | k+1 | −2H | C-C → C-O-C-O 4 元环 |
| 环氧（双键式） | 不变 | 不变 | C=C → C-C + O 桥 |
| `-OH` / OOH / 醚桥 / 过氧桥 / O 在环 | 不变 | 不变 | |

> 推论：目标 k 的含羰基结构只能来自 `k' ∈ [k−nO, k]` 的骨架 + 提升组合（骨架 k 下探）。

---

## 5. 生成流程（每步对应的函数）

```
generate(nC, nO, k_range, constraints, use_parallel, enable_oring_cross_c, progress_cb)
 ├─ 约束模式？ → _generate_from_fragment（片段路径，快）
 ├─ 主循环：k_skel ∈ [max(0, k_min−nO), k_max]          （骨架不饱和度下探）
 │    └─ 每个 k_skel → _phase2d_single（串行）或 _parallel_worker（并行）
 │         ├─ _plan_backbone_allocations + _plan_substituent_allocations（O 分配两阶段）
 │         ├─ 方案级 H 范围预筛：
 │         │    lo = h_skel − 2·(sp[CO]+bb[dioxy]+bb[epoxide])
 │         │    hi = h_skel − 2·(sp[CO]+bb[dioxy])
 │         │    （[lo,hi] 与目标 H 集合无交集 → 整体跳过）
 │         └─ _apply_plan 步骤链：A醚桥 → B过氧 → A'环氧 → A''OOH → A'''双醚桥 → C或ing → D(OH/=O)
 ├─ oring 跨C 迭代（enable_oring_cross_c 时）：(extra_c, nc_big, k) → _oring_single
 ├─ Phase F3 原子矩阵补漏（仅 k_max < 4）：generate_for_target_h → 合并
 ├─ Phase 2e 化学过滤（chem_mode）
 └─ 返回 {分子式: [(mol_type, graph), ...]}
```

合并去重：worker 本地 `_seen_add_safe` 去重 → 父进程按任务完成序并入全局（跨任务全局去重）。

---

## 6. 去重与口径（O 特有的坑，改代码必读）

1. **WL 碰撞**：`nx.weisfeiler_lehman_graph_hash` 对非同构图可能碰撞（实测例
   `C1OC1C1CO1` vs `C1OC2COC12`）。**必须用 `_seen_add_safe`（哈希→列表 + `_graphs_isomorphic`
   VF2 全属性同构）**；矩阵预言机用 `atomic_matrix_gen._seen_add`。**禁止裸 set。**
2. **凯库勒式变体**：双键位置不同的图是**不同结构**，刻意不去重（图口径，见 I3）。
3. **公式门禁前置**：`add()` 先公式（O(n) 便宜）后 WL 哈希，拦截 k+1 溢出候选、省哈希。

---

## 7. 验证工作流（agent 必跑）

### 7.1 快速回归（每次改动后）
```bash
python tests/test_formula_guard.py --smoke    # 秒级：小分子 + 单元断言
python tests/test_formula_guard.py --fast     # 约 20s：中小分子 + 约束用例 + funcgroup（基线 PASS=72）
python tests/test_funcgroup.py                 # 官能团分类专项（41 断言）
```

### 7.2 计数电池（I6 锚点，改机制/去重后必跑）
```bash
python -c "
from utils import parse_compound_formula
from oxygen.unified_oxo_generator import UnifiedOxoGenerator
for f, e in [('C3H6O2',34),('C4H6O2',263),('C5H8O2',1168),('C6H10O',747),('C6H8O',1623),('C6H12O',211)]:
    _,nc,nh,no,*_ = parse_compound_formula(f); k=(2*nc+2-nh)//2
    b = UnifiedOxoGenerator().generate(nc,no,k_range=(k,k))
    n = sum(len(v) for v in b.values())
    print(f, n, 'OK' if n==e else f'!= {e} (回归!)')
"
```

### 7.3 并行/串行等价（改并行或去重后必跑；完整档已含 C6H10O/C6H12O）
```bash
python -c "
from utils import parse_compound_formula
from oxygen.unified_oxo_generator import UnifiedOxoGenerator
for f in ('C6H10O','C6H12O'):
    _,nc,nh,no,*_ = parse_compound_formula(f); k=(2*nc+2-nh)//2
    ns = sum(len(v) for v in UnifiedOxoGenerator().generate(nc,no,k_range=(k,k),use_parallel=False).values())
    np = sum(len(v) for v in UnifiedOxoGenerator().generate(nc,no,k_range=(k,k),use_parallel=True).values())
    print(f, ns, np, 'OK' if ns==np else '不等价!')
"
```

### 7.4 完备性（加机制/改覆盖后）
```bash
python tools/boundary_report.py C7H6O2 --top-missing 10   # 覆盖率应 ≥ 98.69%（数学口径）
```

### 7.5 约束回归（改片段路径/严格口径后）
```bash
# 期望：C8H8O2+苯环=40、C7H6O2+苯环=17、C7H8O+苯环=7；且集合含
# 乙酸苯酯 CC(=O)Oc1ccccc1 与 苯甲酸甲酯 COC(=O)c1ccccc1（funcgroup 测试已断言）
```

### 7.6 性能基准（改并行/快速模式后）
```bash
# C7H8O2 无约束（基线：串行完整 327s / 并行完整 150.6s / 并行快速 114.4s）
# 临时脚本：对 use_parallel × enable_oring_cross_c 四种组合计时，核对计数 99,792 / 96,785
```

---

## 8. 避坑清单（过程中的错：症状 → 根因 → 修复 → 自查）

### 8.1 生成正确性类

| # | 症状 | 根因 | 修复 | 自查 |
|---|---|---|---|---|
| 1 | 结果混入分子式不符结构 | `=O` k+1 / 原子矩阵全 k 枚举 / 片段合并改 k 三处泄漏 | 末端公式门禁 + 生成内 `add()` 门禁统一收口 | 7.1 fast（含门禁断言） |
| 2 | C4H6O2 计数 262（少 1） | WL 碰撞误删（`C1OC1C1CO1` vs `C1OC2COC12`） | `_seen_add`：哈希→列表 + VF2 | 7.2 计数电池 |
| 3 | C7H8O2 计数 99,622（少 170） | 统一生成器 `add()` 裸 WL set（与预言机不同步） | `_seen_add_safe` 全链路替换 | 7.2 + 7.3 |
| 4 | 有序生成丢结构（34→33、263→247） | ①全原子 BFS 前缀丢内部 O（`C=COOC`）②禁双新键剪枝过严 ③双新键都必须"下一原子"过严 | 单新键=同类型内下一原子；双新键放行；末端 `_is_graph_connected` | 7.2（原子预言机口径计数） |
| 5 | 苯环约束 70 vs GUI 40 不一致 | GUI 类型筛选 `_has_benzene_ring`（严格 Kekulé）剔除了苯并氧杂环 | 统一严格口径：`_generate_from_fragment.add()` 收口 VF2 片段校验 | 7.5 约束回归 |
| 6 | C8H8O2+苯环缺苯甲酸甲酯（只有乙酸苯酯） | 片段路径从不放**侧链内部 O 桥** | 方案 A：直连合并骨架复用 `_apply_plan` backbone 机制 | 7.5（双酯断言） |
| 7 | 覆盖率缺口 | 环氧→OOH/过氧→双醚桥逐级缺失 | 补机制后用边界报告验证覆盖爬坡 | 7.4 |

### 8.2 验证 / 测试类

| # | 症状 | 修复 | 自查 |
|---|---|---|---|
| 1 | 芳香 SMILES→图后门禁假阴性 | 测试中先 `Chem.Kekulize` 再取键型（否则全单键环判 False） | 7.1 |
| 2 | 超时机制在 Windows 报错 | 用 `deadline` 参数，不用 SIGALRM | 7.2 |
| 3 | 计数"回归"误报 | 厘清口径：原子预言机计数 vs 统一生成器计数（树⊆预言机时相等，否则树=子集+多余）；设期望前先验证口径 | 7.2（先跑预言机对拍） |
| 4 | 测试脚本被 Windows spawn 重复执行 | 入口加 `if __name__ == '__main__'` 保护 | 观察输出无重复执行 |

### 8.3 工程 / 性能类

| # | 症状 | 修复 | 自查 |
|---|---|---|---|
| 1 | k≥4 公式生成极慢 | 原子矩阵回退仅在 `k_max < 4` 启用（k≥4 收益低） | 7.2（C7H6O2 k=5 不走 F3） |
| 2 | 并行收益仅 ~2× | 任务不平衡（最重单任务主导墙钟） | 7.6 基准 |
| 3 | 无效候选白算哈希 | 公式门禁前移到 WL 前 | 7.3 等价性 |
| 4 | 跨调用骨架重复枚举 | `_hydrocarbon_skeleton_cache(nc,nh)`（深拷贝返回） | 7.1 |
| 5 | GUI 显示截断/布局问题 | 显示层问题，不影响生成正确性（已修，见 PROJECT_GUIDE 4.10） | — |

---

## 9. N/P 扩展任务清单（agent 可执行 TODO）

> 设计目标：O 的框架（k 下探、两阶段分配、H 预筛、`_seen_add_safe`、原子回退、片段路径、
> 并行化）**全部复用**；N/P = 新增原子类型 + 新增机制 + 扩展公式/分类/预言机。

### Step 0 — 决策（**动手前必须与用户确认**，输出口径文档）
- [ ] 价态口径：N/P 是否允许超出八隅律（5 价 N/P、配位键）？放开则偏离 O 的"八隅律硬规则"；
- [ ] 是否计入电荷/配位（质子化胺、磷鎓）；
- [ ] 氮规则 H 公式：DBE = nC − nH/2 + nN/2 + 1 → `graph_formula_parts` 按元素参数化。

### Step 1 — 数据模型与公式（`utils.py`）
- [ ] `parse_compound_formula`：解析支持 N/P 原子；
- [ ] `graph_formula_parts`：H 规则参数化（C=4、O=2、N=3、P=3/5；N 的 nN/2 修正）；新增
      N/P 取代基属性（`amino_counts`/`nitro_counts` 等，参照 `oxo_counts` 模式）；
- [ ] `format_graph_formula` / `expected_formula_parts`：分子式字符串支持 N/P；
- **验证**：手构乙胺 CH3CH2NH2、硝基苯 C6H5NO2 图，断言 `graph_formula_parts` 输出正确公式。

### Step 2 — 完备性预言机（`oxygen/atomic_matrix_gen.py`）
- [ ] 原子类型扩展 'N'/'P'（`self.atom_types` / `self.valences` 参数化）；
- [ ] `_matrix_to_graph` label 映射；
- **验证**：`generate_for_target_h` 对 C2H7N 与 MAYGEN 对拍（仅一次性外部对照，见去 MAYGEN 政策）。

### Step 3 — 树机制（`oxygen/unified_oxo_generator.py`）
- [ ] `_plan_substituent_allocations`：增加 N 取代基分配（NH₂/CN/NO₂，注意 NO₂=2O+1N 组合约束）；
- [ ] `_plan_backbone_allocations`：增加 N 桥（-NH-、-N=N-、N 杂环），与 ether/peroxy/… 并列；
- [ ] `_apply_plan`：新增 Step（参照 A→D 步骤链模式）；
- [ ] `add()` / `_make_phase_add` 公式门禁：H 集合按氮规则扩展；
- **验证**：C2H7N 覆盖乙胺/二甲胺/甲胺+CH4？；计数电池（I6 O 系锚点不能变！）+ 7.3 等价。

### Step 4 — 片段与约束（`oxygen/fragment_library.py`）
- [ ] 新增预结构：吡啶/吡咯/咪唑/苯胺/硝基苯…（`_build_xxx` + `_registry`，模板 `_build_benzene`）；
- [ ] 约束模式复用 `_generate_from_fragment` 与严格口径（I5）；
- **验证**：C5H5N+吡啶 约束计数 + 7.5。

### Step 5 — 分类与 GUI（`oxygen/funcgroup.py` + `Main.py`）
- [ ] `TAG_CN` 增加：胺/腈/硝基/酰胺/含 N 杂环/磷酰基…；`o_distribution` 泛化"杂原子分布"；
- [ ] GUI 类型筛选/官能团面板/分子式显示支持 N/P；
- **验证**：`tests/test_funcgroup.py` 扩展 + 无窗口 GUI 构造冒烟。

### Step 6 — 完备性收口（`oxygen/boundary.py` + `tools/boundary_report.py` + 测试）
- [ ] 预言机扩展到 N/P（价态集合、目标 H 剪枝公式）；
- [ ] 新增 N 系计数/覆盖断言进 `tests/test_formula_guard.py`；
- **验证**：全套 7.1–7.5 + 新增 N 系电池。

**可直接复用的框架**：k 下探 / 两阶段分配 / H 范围预筛 / `_seen_add_safe`（新原子后碰撞概率只增不减，**必须沿用**）/ F3 原子回退 / 片段路径约束 / 进程池并行 / funcgroup 过滤体系 / 边界报告验证流程。

---

## 10. 速查与参考计数

| 项 | 值 |
|---|---|
| k 下探区间 | `[max(0, k_min − nO), k_max]` |
| O 机制 H 变化 | `=O`/环氧(单键)/dioxy 各 −2H（k+1）；OH/OOH/醚/过氧/oring 不变 |
| 方案级预筛 | `lo = h_skel − 2·(sp[CO]+bb[dioxy]+bb[epoxide])`, `hi = h_skel − 2·(sp[CO]+bb[dioxy])` |
| oring 跨C | `extra_c ≤ min(nO, 2)`，`_ORING_MAX_ATOMS = 10` |
| F3 原子回退 | 仅 `k_max < 4` 时启用，重原子上限 10 |
| 并行默认 | `nC+nO ≥ 7` 且 k_skel ≥ 2 档；worker ≤ 4；失败回退串行 |
| 参考计数 | 34 / 263 / 1168 / 747 / 1623 / 211；C7H8O2 完整 99,792 / 快速 96,785 |
| 约束回归 | C8H8O2+苯环=40、C7H6O2+苯环=17、C7H8O+苯环=7 |
| 覆盖率 | C7H6O2 数学口径 98.69%（残余=多环稠合 O 杂环） |
| 测试档位 | smoke（秒级）/ fast（~20s，基线 PASS=72）/ 完整（10-15 分钟） |

---

## 附：与 PROJECT_GUIDE.md 的交叉引用（不重复展开）

- 全局架构 / 数据模型 / WL 去重机制 → `PROJECT_GUIDE.md` 2、3、4.2
- 含氧生成策略细节（机制列表、ChemFilter）→ 4.8
- 边界验证 / MAYGEN 等价实证 / 有序生成 P0 → 4.9
- 官能团分类模块与 GUI 集成（惰性计算、流式布局）→ 4.10
- 常见问题表 / 去 MAYGEN 政策 → 6
