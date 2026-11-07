# Nitrogen — 含氮有机物异构体生成模块

> 面向：后续开发此模块的 AI Agent 与开发者。
> 本文记录 N 系的化学口径、架构决策、验证方法与已知边界。
> 与其他文档分工：全项目见 `PROJECT_GUIDE.md`，含氧专项见 `O_project_guide.md`，
> N 系细节**以本文为准**。
>
> 最近更新：2026-09-19（验证判据内部化 + 两处缺陷修复）

---

## 1. 定位与范围（第一阶段）

**已实现**：纯 C/H/N 有机物（胺/腈/亚胺/偶氮/含 N 杂环等）的异构体**完备枚举**。
- 化学口径：**中性八隅律**（C = 4 价，N = 3 价）；
- 数据表示：`networkx.Graph`（节点 `label='C'/'N'`，边 `bond_type`），
  与 O 系同构（PROJECT_GUIDE 3.1），仅原子类型扩展；
- 验证：**判据 100% 内部化**（内部预言机 + 内部实测基线 + 化学事实锚点），
  对外部工具**零依赖**（合规红线 §8.4-3）。MAYGEN 对照已剥离为可选脚本
  `maygen_compare.py`。

**已实现**：
- **GUI 接入（第一期，2026-09-21）**：生成 / 列表 / 分子式文本
  （Main.py `_generate_in_background` 的含氮分支走 `AtomicMatrixNGenerator`）；
- **渲染与导出（第二期，2026-09-21）**：**键线式 2D / 3D 可视化 / Gaussian `.gjf` 导出**
  ——含氮结构与烃 / 含氧 / 卤代**共用同一渲染与导出链路**，元素由 `element_of` 回调
  按节点 `label` 判定。实施记录见 `Instruction Data/开发过程.md` §十四。
- **路线 A：N+O 组合（第三期，2026-09-21）**：预言机扩到 **C/N/O**（价态 4/3/2），
  支持 `C2H5NO`、`CH4N2O`（尿素）、`C2H5NO2`、`C5H5NO` 等含氮含氧分子式；
  15 个 N+O 公式经 MAYGEN 一次性对拍 **0 缺 0 多**（§十五）。
  同轮完成**预言机并行化**（前缀切分 + 进程池 + 父进程去重，C6H5N 1.9~2.8×、大公式更高）。

**未实现（后续阶段）**：
- **硝基 –NO₂**：中性八隅律 **N=3** 下不在枚举空间内（需电荷分离式 N⁺–O⁻）。
  ⚠ 与 **MAYGEN 默认价态一致**（CH3NO2 两侧同为 15 个结构，均不含硝基），非"缺失"；
  已提供**按需挂载**通道 `special_groups.py`（`attach_group(G, node, 'nitro')`，
  可渲染 / 导出，**不进枚举**）；
- 含 N 杂环约束路径（必含吡啶等）、funcgroup 含氮分类；
- **O 系预言机并行化**（本次先落在 N 系主生成器，O 系可照搬同一套前缀机制）。

> ✅ **互斥已全部解除（2026-09-21）**：含氮可自由组合 **O 与卤素**——
> `C2H5NO`、`CH4ClN`、`C2H4ClNO` 均可用；N+卤素经 MAYGEN 对拍 **0 缺 0 多**
> （见 `开发过程.md` §16.3）。

> ⚠ **含氧 / 卤代的 `.gjf` 导出仍是待办**（`开发过程.md` §10.7 待办 2）：
> `_build_rdkit_mol` 在"骨架 + 取代基"模型下不建 `oxo_counts`（OH/=O）与
> `halogen_counts` 原子；本次只在**节点模型**（N / N+O 预言机产出）下新增了
> 按元素建 O 的能力，刻意不改含氧路径以免行为漂移。

---

## 2. 决策记录

### 2.1 首轮决策（动手前与用户确认）

| # | 决策点 | 结论 | 依据 |
|---|---|---|---|
| 1 | 功能范围 | **纯 C/H/N 核心全集**（胺/腈/亚胺/偶氮/含 N 杂环） | 用户确认；硝基等含 O 组合留待 N+O 阶段 |
| 2 | 架构策略 | **独立模块 + 复用公共设施**（不改 O 系 / Main.py / GUI） | 用户确认；风险隔离，可独立验证 |
| 3 | 价态口径 | **中性八隅律 N=3**（默认），不含铵/配位 4~5 价 | MAYGEN 源码实证：`valences.put("N", 3)`；支持 `N(5)` 显式覆盖但默认关闭 |
| 4 | 验证标准 | **计数锚点 + 预言机对拍** | 用户确认；预言机为主 |

### 2.2 验证判据内部化（2026-09-19，合规红线 §8.4-3）

**背景冲突**：首版 `validate.py` 把 MAYGEN 计数硬编码为 `MAYGEN_ANCHORS`
并用作**回归断言的期望值**。这违反红线 §8.4-3「边界/完备性一律使用内部预言机，
**禁止引入对 MAYGEN（或其接口、结果）的任何依赖（运行、测试、文档均不得依赖）**」——
属**结果依赖**（把外部工具数值变成测试判定基准）。

**处置**：判定基准改为三类内部/独立证据，MAYGEN 对拍降级为可选脚本。

| 证据来源 | 内容 | 独立性 |
|---|---|---|
| ① 内部预言机 | `atomic_matrix_n.py` 原子矩阵目标 H 剪枝枚举 | 内部实现 |
| ② 内部实测基线 `INTERNAL_BASELINE` | 预言机实测值，作**回归锁**防计数漂移 | 内部实测 |
| ③ 化学事实锚点 `CHEM_FACTS` | 教科书可查结果（C2H7N=2 等） | **独立于任何程序** |

- `Nitrogen/maygen_compare.py`（新增）：MAYGEN 仅经 `subprocess` 外部调用，
  需**显式运行**、不参与回归、不参与 pass/fail 判定 —— 与 O 系
  `tools/maygen_validate.py` 同等待遇；
- 首版 12 个公式的外部对拍结论**作为历史记录保留**（见 §5.1），但不再承担判据角色。

### 2.3 MAYGEN 价态处理结论（源码实证，`MAYGEN/src/main/java/maygen/Maygen.java`）

- 默认价态表 C=4 / N=3 / O=2 / S=2 / P=3 / 卤素=1 / H=1；
- 氢分配规则：每原子氢容量 = `valence − 1`（N 最多 2 个 H → `-NH₂`）；
- 可选 `元素(价态)` 语法（如 `N(5)`）显式覆盖价态，默认关闭；
- 生成方式为**度序列驱动**（不查形式电荷，只查价态）→ 硝基在 3 价 N 下
  合法表示为 N−C + N=O + N−O（八隅律满足）。

---

## 3. 化学口径速查

### 3.1 价态与公式（单一事实源：`formula.py`）

| 原子 | 价态 | 说明 |
|---|---|---|
| C | 4 | 与 O 系一致 |
| N | 3 | 中性八隅律 |

- **氮规则 DBE**：`k = nC − nH/2 + nN/2 + 1` ⟺ `nH = 2·nC + nN + 2 − 2·k`；
- 图统计：每节点 `h = max(0, VALENCE[label] − bond_load)`，总 H = Σh；
- **双重门禁**（顺序不可颠倒，见 §3.3）：
  1. **价态硬校验** `is_valence_legal(G)` —— C ≤ 4、N ≤ 3；
  2. **公式门禁** `formula_matches_input(G, nC, nH, nN)` —— H/C/N 严格一致。

### 3.2 树机制对 k 的贡献（`generator.py`）

| 机制 | 操作 | 对 k | 覆盖 |
|---|---|---|---|
| A：C→N 替换 | (nC+nN) 碳骨架中选 nN 个 C 替换为 N（键型不变，H −1/位） | 不变 | 含 N 杂环（吡啶←苯）、叔胺中心、亚胺 C=N、腈（←末端炔）、偶氮（←C=C 两端） |
| B：端基 N | 在 nC 骨架 H 位挂 N 端基 | -NH₂: +0 / =NH: +1 / -C≡N: +2 | 伯胺、亚胺端基、腈端基 |
| C：桥 N | 在 nC 骨架 C-C 键插入 N | -NH-: +0 / -N=C-: +1 | 仲胺桥、亚胺桥 |

> 机制 B 的 k 守恒推导：`a·(-NH₂) + b·(=NH) + c·(-C≡N)` 使 `b + 2c = k_target − k_skel`，
> 且 `a + b + c = nN`（整数解枚举位置分配）。

> **⚠ 机制 B/C 必须逐 H 档位调用**：其 `k_target` 由 `n_hydrogen` 决定
> （`delta = k_target − k_skel`），故 `generate()` 中对 `target_h_set` 的
> **每一个 H 档位**各调用一次。历史缺陷见 §8.1。

### 3.3 为什么"公式门禁"单独不够（红线 §8.4-1 的 N 系教训）

`graph_formula_parts` 用 `h = max(0, max_v − bond_load)` 计算氢数——
当原子**超价**时（如 N 载 4 键），该式把"超价"**静默吸收成 h = 0**，
于是一个非法结构仍可能凑出目标 H 总数而**通过纯公式门禁**。

实证：机制 A 对键载为 4 的碳做 C→N 替换会产生 N 超价结构，
`C2H6N2` 多出 1 个、`C2H4N2` 多出 5 个，全部蒙混过关（详见 §8.2）。

**结论**：价态校验必须作为**独立前置门禁**与公式门禁并存，
且应放在**统一收口点**（`NitrogenGenerator.add()`）。

---

## 4. 模块地图

| 文件 | 关键函数/类 | 职责 | 改动场景 |
|---|---|---|---|
| `formula.py` | `parse_formula` / `graph_formula_parts` / `formula_matches_input` / `is_valence_legal` / `dbe_from_parts` | 公式与价态单一事实源 | N+O 阶段加 O 解析与价态表 |
| `atomic_matrix_n.py` | `AtomicMatrixNGenerator.generate_for_target_h` | **完备性预言机**（邻接矩阵原子级枚举，C=4/N=3） | 加 P/S 原子（价态表参数化） |
| `generator.py` | `NitrogenGenerator.generate` / `_mechanism_replace` / `_mechanism_terminal` / `_mechanism_bridge` | 树机制生成器（价态+公式双门禁，碰撞安全去重） | 加 N+O 机制（硝基/酰胺） |
| `validate.py` | `check_anchors` / `check_oracle_selfconsistent` / `check_coverage` / `check_all_h_levels` / `smoke_gate` | CLI 验证入口（六项内部检查） | 加锚点/断言 |
| `maygen_compare.py` | `compare` / `graph_to_smiles` | **可选** MAYGEN 外部对照（不参与回归） | 一次性外部复核 |

**复用**：WL 哈希 + VF2 同构去重（碰撞安全，禁止裸 set——O 系历史教训）、
骨架自举（预言机生成纯 C 骨架）、公式门禁模式（O 系 `add()` 同构）。

---

## 5. 验证方法（agent 必跑）

```bash
# 全套内部验证（六项检查，约 2.5 分钟，瓶颈 C6H5N/C6H7N）
python -m Nitrogen.validate --anchors

# 交互模式：逐条输入分子式回车即计算
python -m Nitrogen.validate            # quit/exit/空行 退出；anchors 跑锚点；help 用法

# 批量：命令行给出一个或多个分子式（打印样例 + 锚点 + 覆盖率）
python -m Nitrogen.validate C5H5N
python -m Nitrogen.validate C2H7N C3H9N

# 强制模式
python -m Nitrogen.validate --oracle C6H5N    # 强制预言机（带超时兜底）
python -m Nitrogen.validate --tree  C6H5N     # 树机制预览（部分集合，不快于预言机）

# 可选：MAYGEN 一次性外部对照（需本机 Java；不参与回归）
python -m Nitrogen.maygen_compare                 # 默认电池
python -m Nitrogen.maygen_compare C5H5N --set     # 指定 + SMILES 集合级对比（需 RDKit）
```

### 5.1 六项内部检查（`--anchors`）

| # | 检查 | 断言 | 函数 |
|---|---|---|---|
| 1 | 门禁冒烟 | 产物分子式 ≡ 输入分子式 | `smoke_gate` |
| 2 | 内部基线锚点 | 预言机计数 ≡ `INTERNAL_BASELINE`（回归锁） | `check_internal_baseline` |
| 3 | 化学事实锚点 | 预言机计数 ≡ `CHEM_FACTS`（独立化学证据） | `check_chem_facts` |
| 4 | 预言机自洽 | 门禁恒真 + 无同构重复 + 价态合法 | `check_oracle_selfconsistent` |
| 5 | 双实现交叉 | 树机制 ↔ 预言机**双向差集**：missing = 0 **且 extra = 0** | `check_coverage` |
| 6 | 全 H 档位守卫 | `generate(nC,nN)` 不给 `n_hydrogen` 时逐档等同预言机 | `check_all_h_levels` |

> 检查 5 的 **extra（树机制 − 预言机）必须恒为 0**：非 0 即"非法泄漏"，
> 历史上正是该指标缺失，才让 §8.2 的 N 超价结构长期未被发现。

### 5.2 计数基线（图口径）

| 分子式 | 计数 | 分子式 | 计数 |
|---|---|---|---|
| C2H7N | 2 | C4H9N | 35 |
| C3H9N | 4 | C4H11N | 8 |
| C2H5N | 4 | C5H5N | 685 |
| C2H3N | 5 | C6H5N | 4394 |
| C2H8N2 | 6 | C6H7N | 4378 |
| C3H7N | 12 | C6H15N | 39 |

> **来源**：内部预言机实测（`INTERNAL_BASELINE`）。
> 其中 C2H7N=2、C3H9N=4、C4H11N=8 另有**化学教科书事实**独立印证（`CHEM_FACTS`）。
>
> **历史记录**（不再承担判据角色）：2026-08-26 曾用 MAYGEN 一次性对拍，
> 12 个公式计数与 MAYGEN **完全一致**；树机制在上述公式覆盖率为 100%。

### 5.3 不变式（与 O 系同源，改代码底线）

| # | 不变量 | 破坏症状 |
|---|---|---|
| I1 | 价态 + 公式**双门禁**全链路同口径（`formula.py` 单一事实源） | 结果混入超价 / 分子式不符结构 |
| I2 | 去重碰撞安全（同 WL 哈希必须 VF2 全属性同构） | 计数莫名减少 |
| I3 | 图口径计数（凯库勒式变体不去重） | 计数被"合并" |
| I4 | 树机制 − 预言机 = 0（无非法泄漏） | 非法结构混入产出 |
| I5 | 验证判据内部化，对外部工具零依赖 | 违反红线 §8.4-3 |

---

## 6. 关键实测事实（2026-09-19 测定，供后续开发决策）

| 事实 | 实测数据 | 对开发的影响 |
|---|---|---|
| **机制 A 单独即完备** | 8/8 公式上 A 的产物集合 ≡ 预言机（含 C5H5N=685） | A 可作完备生成路径；理论上也可由"碳化映射"证明 |
| **机制 A 比预言机慢** | C5H5N 3.9s→7.0s（0.56×）、C6H7N 79.5s→104.0s（0.76×）、C6H5N 72.6s→103.7s（0.70×） | **`--tree` 不是加速档**："用树机制规避原子矩阵组合爆炸"的假设**不成立**（A 在 (nC+nN) 碳上做全不饱和度枚举，未做目标 H 剪枝） |
| **机制 B/C 是部分路径** | 关闭 A 后 B+C only 仅覆盖 8/13（C2N1 全档位） | B/C 仅覆盖端基/桥类常见官能团，不含含 N 杂环；完备性由 A / 预言机承担 |
| **预言机是完备真值** | 双向差集：`仅预言机有 = 0`（不漏），修门禁后 `仅树机制有 = 0`（不多） | 与 O 系一致：**生成用预言机最快且完备** |

**碳化映射（机制 A 完备性的理论依据）**：任一合法 C/H/N 图，把 N 全部改标为 C
后，各原子键载 ≤ 4，即得一张合法全碳骨架图；反向由价态 + 公式双门禁保证
A 的产出 ⊆ 合法 C/H/N 图。故 A ⊇ 全集且 A ⊆ 全集。

---

## 7. 已知边界与后续 TODO

1. **N+O 组合（第二阶段）**：硝基 -NO₂ / 酰胺 -CONH₂ / 硝基苯 / 含 N 含 O 杂环。
   需扩展：`formula.py` 解析 O、价态表加 O=2、预言机 `_max_bond_type` 加 C-N/N-O/O-N 键型、
   `generator.py` 加 NO₂ 组合约束（NO₂ = 2O + 1N）、funcgroup 分类（胺/腈/硝基/酰胺）。
2. **含 N 杂环约束路径**：`C5H5N + 必含吡啶` 约束（参照 O 系 `_generate_from_fragment` +
   `fragment_library.py` 预结构模式）。
3. **大公式性能**：预言机对 ≥8 重原子变慢（C6H5N ≈ 64~73s）。
   ⚠ 注意：**改用树机制并不能加速**（见 §6），真正的优化方向是：
   ① 预言机的进程池并行（参照 O 系 `use_parallel`）；② 对高不饱和度档位增加剪枝。
4. **GUI 集成（已完成，第一期）**：Main.py 已接入。接入点：
   `utils.parse_compound_formula` 扩展为 **8 元组**（末位 `n_nitrogen`）、
   新增 `utils.graph_formula_parts_full`（支持 `label='N'`，C=4/N=3/O=2）、
   `MOL_NAMES_CN/FILTER/EN` 增加 `'nitrogen'`、`structure_filter.filter_by_expected_formula`
   增加 `n_nitrogen` 关键字、`Main.py` 新增含氮生成分支与展示分支。
   **第二期（2026-09-21）已补齐**键线式 / 3D / `.gjf` 导出，接入点：
   `utils.graph_to_rdkit_mol` 元素价态表支持 `label='N'`（C=4 / N=3 / O=2）、
   `Main._build_rdkit_mol` 把 N 建为 **3 价氮原子**、`_extract_coords` 改为收集
   **全部重原子**、`_parallel_compute_coords` 增加 `'nitrogen'` 分支、
   `gaussian_export.build_gjf` 坐标段按元素渲染（原实现硬编码 `'C'`）。
5. **高价态 N（预留）**：MAYGEN 支持 `N(5)` 显式覆盖；本项目默认 N=3，如需铵盐/配位
   形态，在价态表参数化处扩展（`formula.VALENCE` / 预言机 `valences`）。

---

## 8. 缺陷修复记录

### 8.1 机制 B/C 在"全 H 档位"下覆盖缺失（2026-09-19 修复）

- **现象**：`NitrogenGenerator(enable_replace=False).generate(2, 1)`（不给 `n_hydrogen`）
  仅产出 `{'C2H5N': 1}`；按文档承诺应枚举全部 H 档位。
- **根因**：`generate()` 对机制 B/C 只传 `min(target_h_set)`（= 最高不饱和度那一档）
  作为 `n_hydrogen`，而 B/C 的 `delta = k_target − k_skel` **依赖** `n_hydrogen`，
  故只产出该单档结构（机制 A 不依赖 `n_hydrogen`，被其完备性掩盖，默认配置下不可见）。
- **修法**：对 `target_h_set` 的每一个 H 档位各调用一次 B/C。
- **修复效果**：B+C only 由 1 个 → 8 个结构。
- **回归守卫**：新增检查 6 `check_all_h_levels`（防复发）。

### 8.2 机制 A 产出 N 超价结构并被公式门禁放行（2026-09-19 修复）

- **发现途径**：新增的检查 6 报告 `(nC=2,nN=2)` 三档"树机制 **多于** 预言机"
  （H=6: 19 vs 18；H=4: 32 vs 27；H=2: 28 vs 19）——这是不该出现的方向。
- **根因**：① `_mechanism_replace` 对**任何**碳（含键载为 4 的碳）做 C→N 替换，
  产生 N 载 4 键的非法结构；② `graph_formula_parts` 用 `max(0, max_v − bond_load)`
  把超价**静默吸收为 h = 0**，故纯公式门禁**无法拦截**（见 §3.3）。
- **定性证据**：差集中"仅预言机有 = 0"（预言机不漏），"仅树机制有"的 6 个结构
  **全部** N 超价（`load=4`），预言机正确地将其排除。
- **修法**：在统一收口点 `add()` **前置** `is_valence_legal(g)` 校验
  （红线 §8.4-1 加固，价态门禁与公式门禁并存）；
  并在检查 5 中补上 **extra（树 − 预言机）** 指标。
- **修复效果**：`C2H6N2` 19→18（= 预言机）、`C2H4N2` 32→27（= 预言机），extra 全为 0。
- **全量复验**：六项检查全部通过（2m32s）。

---

## 9. 快速示例

```python
from Nitrogen import NitrogenGenerator, AtomicMatrixNGenerator, parse_formula

# 1) 树机制生成器（胺/腈/亚胺/偶氮/杂环）
gen = NitrogenGenerator()
res = gen.generate_for_formula('C3H9N')          # 4 个异构体
for fml, items in res.items():
    print(fml, len(items))

# 2) 完备性预言机（指定分子式的全部结构）
n_c, n_h, n_n, err = parse_formula('C5H5N')      # 吡啶等
graphs = AtomicMatrixNGenerator(n_c, n_n).generate_for_target_h(n_h)
print(len(graphs))                                # 685
```
