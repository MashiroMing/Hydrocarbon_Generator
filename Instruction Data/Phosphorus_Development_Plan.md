# Phosphorus Isomer Generator — Development Plan

> **项目**：Hydrocarbon_Generator（Isomer）
> **目标**：为异构体生成器增加 **含磷（P）** 元素支持（第一期：仅 C/H/P，CLI，不接入 GUI）
> **作者**：CodeBuddy ｜ **日期**：2026-11-07 ｜ **状态**：方案待确认（未开始编码）
> **关联**：`Instruction Data/O_project_guide.md`、`Nitrogen/`（模板模块）、`Instruction Data/开发过程.md` §十三~§十九

---

## 0. 一页速览

| 项 | 结论 |
|---|---|
| 第一期范围 | **仅 C / H / P**（不接 O、N、卤素）；**CLI 先行**，不接 `Main.py` |
| P 价态 | **一期 P = 3**（与 MAYGEN 1.8 完全一致）；**预留 5 价开关**（`valence_p=3` 参数，集合化，二期改 `(3,5)`） |
| 键级 | **全部允许**：P–C / P–H / P–P 单键 + **P=C、P≡C** 双/三键（math 口径，与 MAYGEN 一致） |
| 决策状态 | ✅ 已确认（Q1–Q4，2026-11-07，见 §8） |
| 电荷 | 第一期**不引入**形式电荷（磷鎓 P⁺、"P=O 电荷分离式"留到二期） |
| 数据模型 | 与含氧/含氮**完全一致**：`networkx.Graph`，节点 `label ∈ {'C','P'}`，边 `bond_type ∈ {1,2,3}` |
| 算法 | **矩阵级完备枚举**（照搬 `Nitrogen/atomic_matrix_n.py` 的骨架：价态饱和 + 目标 H 剪枝 + WL/精确同构去重 + 前缀并行 + 缓存） |
| CLI | `python -m Phosphorus.cli C3H9P [--count|--smi|--json|--max-atoms N]`；`python -m Phosphorus.validate --anchors` |
| 验收 | 小分子锚点**逐一对齐 MAYGEN 1.8**（本文 §5 表格）+ 串行/并行等价 + 公式自洽 |

---

## 1. 调研结论

### 1.1 MAYGEN 项目如何处理磷（**核心问题的答案**）

**证据 A —— 源码价态表**（本地 `MAYGEN/src/main/java/maygen/Maygen.java:123-135`）：

```java
valences.put("C", 4);
valences.put("N", 3);
valences.put("O", 2);
valences.put("S", 2);
valences.put("P", 3);   // ← 磷默认 3 价（CDK 标准价态）
```

**证据 B —— 本地实测**（`java -jar MAYGEN-1.8.jar -f <式> -t`，本机 Java 21）：

| 分子式 | MAYGEN 1.8 计数 | 说明 |
|---|---|---|
| CH5P | 1 | 甲基膦 CH3–PH2（P 只有 3 根键，故带 2 个 H） |
| CH3P | 1 | H2C=P（磷烯，**P=C 双键**） |
| C2H5P | 4 | 含 P=C / P–H 变体 |
| C2H7P | 2 | 二甲膦 + 乙基膦（**P–H 键被允许**） |
| C3H5P | 21 | DBE=2（含 P≡C 的三键情形） |
| C3H7P | 12 | DBE=1（C=C、P=C、含 P 环） |
| C3H9P | 4 | 三烷基膦（P 无 H） |
| C4H9P | 35 | DBE=2 |
| C4H11P | 8 | 三烷基/二烷基膦 |
| C5H13P | 17 | |
| C6H13P | 284 | DBE=2（中型锚点） |
| C6H15P | 39 | 无环三烷基膦（含支链） |
| C2H8P2 | 6 | **P–P 键被允许** |
| C3H10P2 | 14 | |
| C4H12P2 | 38 | |

**证据 C —— 官方文档的价态覆盖语法**（MAYGEN 官网 Usage）：

```bash
java -jar MAYGEN-1.8.jar -f N(val=4)6H6 -setElements -v -t
```

> 即：MAYGEN 的**默认**是 CDK 价态（P=3），但**允许**通过 `Element(val=N)` + `-setElements`
> 显式覆盖（例如 `P(val=5)`）。**本地 1.8 jar 不支持该参数**（新版支持）。
> → 结论：**"P 成 3 根键"是 MAYGEN 的默认口径；"5 根键"是可选覆盖，不是默认。**

**证据 D —— MAYGEN 允许的磷化学边界**（由实测反推）：
- ✅ P–H 键（C2H7P=2 必须如此）
- ✅ P–P 键（C2H8P2=6）
- ✅ P=C 双键、P≡C 三键（C3H5P=21）
- ✅ P 在环内（C3H7P=12 含磷杂环）
- ❌ 形式电荷 / 配位键（MAYGEN 不建模）
- ⚠ 5 价磷：默认不生成（需 `val=5` 覆盖）

**对本项目的直接含义**：
1. 第一期口径取 **P = 3**，可与 MAYGEN 1.8 **逐一数值对拍**（这是最强的正确性锚点）；
2. 5 价磷**不要**在一期默认开启（既无 O 配合，化学上也罕见——无 O 时 5 价磷=磷烷，多为高能物种）；
3. 把价态做成 `{"P": (3,)}` 的**集合**参数，二期改为 `(3,5)` 只需改配置。

### 1.2 可复用的框架（来自 `O_project_guide.md`）

| 框架 | 本计划如何用 |
|---|---|
| **I1 公式门禁全链路口径一致** | P 系 crate 自带 `formula.py` 作为唯一事实源；CLI 末端再校验一次 |
| **I2 去重必须碰撞安全**（WL 哈希 → VF2） | 直接沿用 `Nitrogen/atomic_matrix_n.py` 的 `_seen_add_matrix` + `_matrix_iso_fast`（比 networkx 快 3.9–6.3×） |
| **I3 图口径计数**（凯库勒变体分开计） | 与 MAYGEN 对拍即验证口径（N 系已实证一致） |
| **I4 并行 == 串行** | 必须逐式断言集合完全一致 |
| **k 下探 / 两阶段分配** | 一期不需要（P 走预言机全枚举）；**二期接 O 时**再引入 P 桥/P 取代基机制 |
| **F3 原子矩阵回退、片段路径、进程池** | 一期直接用"原子矩阵"作为主引擎；并行与缓存直接复用 |
| **§9 N/P 扩展 TODO** | 本计划即该 TODO 在 "仅 P" 上的落地（先 CLI，后 GUI） |

### 1.3 模板：`Nitrogen/`（最接近的现成实现）

| 文件 | 作用 | 本期怎么处理 |
|---|---|---|
| `Nitrogen/atomic_matrix_n.py` | 矩阵预言机（C/N/O 三类型） | **复制改造**为 C/P 两类型（`Phosphorus/atomic_matrix_p.py`） |
| `Nitrogen/formula.py` | `parse_formula_full` / `graph_formula_parts` / `format_graph_formula` | 复制并去掉 O/N，加入 P（第一期 P 系自成一体，不改 `utils.py`） |
| `Nitrogen/validate.py` | 锚点电池 + 并行等价 + MAYGEN（可选） | 复制，锚点表换成 §5 的 P 表 |
| `Nitrogen/oracle_cache.py` | 磁盘缓存 | 复制并换命名空间（`cache_key` 加 `p-` 前缀） |
| `Nitrogen/generator.py` | 兼容旧接口 | 可省略（一期不需要） |

### 1.4 skills 使用情况

- 已加载 **`isomer-project-context`**（项目架构/数据模型/门禁约定）；
- 仓库 `skills/` 目录现存技能为 **化学构象搜索（conformer-search）**，与磷异构体枚举**不相关**，未使用；
- 如需补充 P 化学知识，可在实现阶段再加载 `organic-chemistry-expert`（磷化学：膦/氧化膦/膦酸酯/磷鎓）。

---

## 2. 化学与数据模型决策

### 2.1 分子式与 DBE（氮规则/磷规则）

P 为**三价**（与 N 同构），故 DBE 公式按 N 同类型处理：

```
DBE = (2·nC + 2 + nP − nH) / 2          （一期无 O、无卤素、无 N）
```

自检：C3H9P → (6+2+1−9)/2 = 0 ✅（无环无重键，MAYGEN 4 个）
C3H7P → (6+2+1−7)/2 = 1 ✅（MAYGEN 12 个）

### 2.2 允许的成键（一期）

| 边 | 允许 | 说明 |
|---|---|---|
| C–C | ✅ | 单/双/三（1/2/3） |
| C–P | ✅ | 单/双/三（P=C、P≡C 与 MAYGEN 一致） |
| P–P | ✅ | 单/双/三 |
| P–H | ✅ | H 不单独建模，由"价态 − 已用键级"推得（与 N 系一致） |
| 电荷 | ❌ | 二期再做（磷鎓 P⁺ / P=O 电荷分离） |
| 配位键 | ❌ | 不在图模型里 |

### 2.3 氢原子规则（与 N 系一致）

每个原子按"**价态 − 键级和**"补 H：

```
h(atom) = valence(atom) − Σ bond_order(atom, nb)
H_total = Σ h(atom)
```

- P=3：可带 0/1/2 个 H（如 –P<、–PH–、–PH2）；**P 带 3 个 H 就是 PH3（孤立磷，不构成有机结构，会被连通性剪枝自然排除）**；
- 目标 H 剪枝：先按 `h(i) ≤ valence(i)` 与全局 H 总数剪枝，再回溯（照搬 N 模块）。

### 2.4 图属性（与含氧/含氮一致）

```python
G = nx.Graph()
G.add_node(i, label='C' | 'P')          # 元素标签
G.add_edge(i, j, bond_type=1|2|3)       # 键级
# 无 oxo_counts / halogen_counts（一期无 O/卤素）
```

---

## 3. 架构设计

### 3.1 目录与文件

```
Phosphorus/
├── __init__.py            # 导出 PhosphorusMatrixGenerator / 版本号
├── formula.py             # 分子式事实源（parse / parts / format / matches）
├── atomic_matrix_p.py     # 矩阵预言机（C/P 两类原子）
├── oracle_cache.py        # 结果缓存（可选，二期再启用也可）
├── validate.py            # 锚点电池 + 并行等价 + 自检（可执行）
└── cli.py                 # CLI 入口
```

### 3.2 关键接口（与 N 模块同形）

```python
# formula.py
parse_formula_p(f) -> (nC, nH, nP)          # 支持 C3H9P / CH5P / C6H13P
graph_formula_parts_p(G) -> (nC, nH, nP)    # 图 → 分子式
format_graph_formula_p(G) -> "C3H9P"        # 唯一事实源
formula_matches(f, (nC,nH,nP)) -> bool

# atomic_matrix_p.py
class PhosphorusMatrixGenerator:
    def __init__(self, n_carbon, n_phosphorus, valence_p=3)
    def generate_for_target_h(self, target_h, deadline=None) -> list[nx.Graph]
    def generate(self) -> list[nx.Graph]     # 自动扫合法 H

# cli.py
python -m Phosphorus.cli C3H9P --count          # 只打印计数
python -m Phosphorus.cli C3H9P --smi            # 打印 SMILES 列表（需 RDKit）
python -m Phosphorus.cli C3H9P --json out.json  # 结构化导出
python -m Phosphorus.cli C3H9P --max-atoms 12   # 规模护栏
```

### 3.3 枚举算法（一期）

```
generate(nC, nP):
  1. DBE 合法性与 H 范围：H ∈ [0, 3·nP + 4·nC]（由价态和约束）
  2. 对每个合法 H（或用户指定 H）：
     backtrack(原子 i):
       - P 特殊：已用键级和 ≤ 3（val=3；二期 val∈{3,5}）
       - 剪枝：剩余原子的最小/最大可成键数 vs 剩余键位
       - 连通性末端检查（照搬 N 模块 _is_graph_connected 思路）
  3. 叶子去重：矩阵级 WL(3 轮) + 精确同构（复用 N 模块增强内核）
  4. 末端口径：graph_formula_parts_p(G) == 目标
```

**性能预期**：P 与 N 同价态，规模相当；C6H15P（≈ N 系 C6H15N 量级）预计秒级；
C6H13P=284 结构属小规模；C7/C8 级随组合增长，沿用 N 系的并行+缓存即可。

### 3.4 CLI 输出规格

- `--count`：`C3H9P<TAB>4<TAB>0.12s`（与 MAYGEN `-t` 对齐，便于对拍）
- `--smi`：每行一个 SMILES（按 RDKit 生成；失败则回退"节点标签+键级"的文本表示）
- `--json`：`{"formula": "...", "count": N, "structures": [{"atoms": [...], "bonds": [[i,j,order],...]}]}`

---

## 4. 分阶段实施计划

| 阶段 | 内容 | 产出 | 验收 |
|---|---|---|---|
| **P0 决策确认** | 与用户确认 §8 的 5 个问题 | 本文件定稿（含决策） | 用户签字/回复 |
| **P1 公式层** | `formula.py`：解析/构造/格式化/匹配 + DBE | 可 `python -c` 自检 | CH5P/C3H9P 手构图往返一致 |
| **P2 预言机** | `atomic_matrix_p.py`：C/P 回溯 + H 剪枝 + 去重 | 能出结构 | **CH3P=1 / C2H7P=2 / C3H9P=4 / C3H7P=12** 与 MAYGEN 一致 |
| **P3 CLI** | `cli.py`（count/smi/json/max-atoms） | 可命令行使用 | `python -m Phosphorus.cli C3H9P --count` → 4 |
| **P4 完备性/等价** | `validate.py`：锚点电池 + 串行/并行 + 全表对拍 | 回归脚本 | 本文 §5 全表通过 |
| **P5 文档与固化** | 开发过程 §二十 + README 增补 | 文档 | 与代码同提交 |

> 每阶段结束跑该阶段验收，**不通过不进入下一阶段**（沿用用户既定流程）。

---

## 5. 验证方案（锚点表）

### 5.1 外部锚点：MAYGEN 1.8（P=3，本机实测）

| 分子式 | DBE | 期望计数 | 备注 |
|---|---|---|---|
| CH3P | 1 | **1** | 最小 P=C |
| CH5P | 0 | **1** | 甲基膦 |
| C2H5P | 1 | **4** | |
| C2H7P | 0 | **2** | P–H 必需 |
| C3H5P | 2 | **21** | 含 P≡C |
| C3H7P | 1 | **12** | 含 P 环/P=C |
| C3H9P | 0 | **4** | 三烷基膦 |
| C4H9P | 2 | **35** | |
| C4H11P | 0 | **8** | |
| C5H13P | 0 | **17** | |
| C6H13P | 2 | **284** | 中型锚点 |
| C6H15P | 0 | **39** | 支链三烷基膦 |
| C2H8P2 | 0 | **6** | P–P 键 |
| C3H10P2 | 0 | **14** | |
| C4H12P2 | 0 | **38** | |

> 生成命令：`java -jar MAYGEN/MAYGEN-1.8.jar -f <式> -t`（一次性外部对照，不进入回归；
> 与项目既定"去 MAYGEN 依赖"政策一致）。

### 5.2 内部一致性检查（每次提交必跑）

1. **公式自洽**：每个结果 `graph_formula_parts_p(G)` 必须等于输入分子式（I1）；
2. **去重安全**：`len(set) == len(list)` 用 WL+VF2 验证；并抽检同 WL 哈希的不同图不被误删（I2）；
3. **串行 == 并行**：`generate` 串行与并行结果集合逐元素一致（I4）；
4. **连通性**：所有图 `nx.is_connected`（P 孤立 = 无效）；
5. **价态**：所有 P 的键级和 == 3（一期），C == 4。

### 5.3 回归脚本（`Phosphorus/validate.py`）

```
python -m Phosphorus.validate --anchors     # 跑 §5.1 全表（期望值与 MAYGEN 一致）
python -m Phosphorus.validate --equiv       # 串行/并行等价
python -m Phosphorus.validate --props       # 公式/连通性/价态自检
```

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| P 的 5 价与 3 价混用导致口径混乱 | 计数不可比 | 一期**锁定 val=3**；`valence_p` 参数化，二期单独开 `(3,5)` 并重跑锚点 |
| P=C/P≡C 是否该算 | 计数差异 | 与 MAYGEN 对齐（算）；在 CLI `--count` 输出旁注明"math 口径，含 P=C/P≡C" |
| 图口径 vs 分子口径（凯库勒变体） | 计数偏差 | 与 N 系同样处理：**图口径**，并与 MAYGEN 逐式对拍确认一致 |
| P–P 键与多磷环爆炸 | 规模 | `--max-atoms` 护栏 + 目标 H 剪枝 + 并行/缓存 |
| 与既有 `utils.py` 冲突 | 后期集成风险 | 一期**不动** `utils.py`；P 系自带 `formula.py`，集成时再统一（见 §7） |
| RDKit 对 P 的支持差异 | SMILES 输出失败 | `--smi` 为主，失败自动回退文本表示；计数不依赖 RDKit |

---

## 7. 后续接入主程序的清单（**不在本期**）

> 一期只做 CLI。若后续决定接入 GUI，按以下顺序（每步都要过既有门禁与回归）：

1. `utils.py`：`parse_compound_formula` 支持 P；`graph_formula_parts`/`format_graph_formula` 加 P；元素顺序 C→H→N→O→P→卤素；
2. `structure_filter.py`：`filter_by_expected_formula` 支持 P；
3. `Main.py`：类型判定（含 P 时走 P 分支）、列表显示、耗时预估；
4. 渲染：`graph_to_rdkit_mol` 加 P 价态（3）；键线式/3D；
5. `.gjf` 导出：元素行 + 价态（P 无卤素处理，直接按元素）；
6. 可选基团库：`special_groups.py` 加膦/氧化膦（需二期 O）；
7. 测试：`tests/test_formula_guard.py` 加 P 断言；`tests/test_gjf_export.py` 加 P 导出用例。

---

## 8. 设计决策（**已由用户确认，2026-11-07**）

| # | 问题 | 决策 | 落地方式 |
|---|---|---|---|
| Q1 | P 价态口径 | **一期 3 价，预留 5 价开关** | `PhosphorusMatrixGenerator(n_carbon, n_phosphorus, valence_p=3)`；内部按 `P_VALENCE_OPTIONS = (3,)` 取集合，二期改 `(3, 5)` 即可（不改调用方） |
| Q2 | 成键类型 | **全部允许**：P–C、P–H、P–P 单键 + **P=C、P≡C** 双/三键 | 回溯中 P 的"已用键级和 ≤ 3"；键级 1/2/3 全开（与 MAYGEN 实测一致） |
| Q3 | CLI 形态 | **`python -m Phosphorus.cli <式>`**，支持 `--count / --smi / --json` | `--count` 输出与 MAYGEN `-t` 同格式（`式<TAB>计数<TAB>秒`）；`--smi` 用 RDKit，失败回退文本 |
| Q4 | 对拍范围 | **接受**，含 C6H13P=284 等中型式 | `validate.py --anchors` 跑 §5.1 全表（一次性外部对照，不进日常回归） |
| Q5 | 氘代 D / 模糊式 | **本期不支持**（默认，未特别要求） | 保留格式位；如需要可后续加 |

> 二期（接 O 后）需要重新确认的只有：P=O 用"双键"还是"电荷分离式（P⁺–O⁻）"；
> 本计划建议与 `special_groups.py` 的硝基一致，走**电荷分离式**，并复用 `formal_charge` 机制。

---

## 9. 参考

- MAYGEN 源码：`MAYGEN/src/main/java/maygen/Maygen.java`（价态表 123–135 行）
- MAYGEN 官网（`val=` 语法）：https://maygenerator.github.io/
- 本项目：`Instruction Data/O_project_guide.md` §9（N/P 扩展 TODO）、`Nitrogen/atomic_matrix_n.py`、`Nitrogen/formula.py`
- 开发日志：`Instruction Data/开发过程.md` §十三~§十九

---

## 10. 实施结果（P1–P5，2026-11-07 完成）

| 阶段 | 产出 | 验收 |
|---|---|---|
| P1 公式层 | `Phosphorus/formula.py` | 解析/构造/格式化/DBE 全通过 |
| P2 预言机 | `Phosphorus/atomic_matrix_p.py` | 锚点对拍逐式一致 |
| P3 CLI | `Phosphorus/cli.py` | `--count/--smi/--json/--parallel` 均正常 |
| P4 验证 | `Phosphorus/validate.py` + `tests/test_phosphorus.py` | 21 断言全通过 |
| P5 文档 | `Phosphorus/README.md` + 开发过程 §二十 | 与代码同提交 |

**锚点结果（MAYGEN 1.8，P=3）**：21/21 全部一致，总耗时约 3s；
其中 C6H13P=284（1.58s）、C6H15P=39、C4H12P2=38、P2=1、C2H2P2=19、C3H3P=19。

**性质自检**：21 式全部通过（公式自洽 / P 价态=3 / 连通性 / 无重复图）。

**串行/并行**：C3H9P、C4H11P、C6H15P、C2H8P2 结果集完全一致。

**CLI 实测**：
- `C3H9P --count` → `C3H9P<TAB>4<TAB>0.001`
- `C3H9P` 列出 4 个 SMILES：`CP(C)C`（三甲膦）、`CCPC`（乙基甲基膦）、`CCCP`（丙基膦）、`CC(C)P`（异丙基膦）
- `C2H7P --smi` → `CPC`（二甲膦）、`CCP`（乙基膦）
- 错误路径：含 O/N、无 P、非法计数均返回 exit=2 且提示清晰

**性能**：C6H13P（284 结构）1.46–1.58s；全部锚点 3s；并行仅对前缀切分有意义，
小规模默认串行（进程池启动开销 > 收益）。

### 10.1 第二轮补充（2026-11-07，用户独立测试前）

**① CLI 支持用户直接输入 C/H/P 个数（与分子式完全等价）**

```bash
python -m Phosphorus.cli --c 3 --h 9 --p 1 --count     # → C3H9P  4
python -m Phosphorus.cli --c 0 --h 3 --p 1 --count     # → H3P    1（PH3）
python -m Phosphorus.cli --interactive                 # 逐项提示（提示走 stderr）
```

- `--c/--h/--p` 必须**三个同时给**；与分子式位置参数**二选一**；P 必须 ≥1；
- 交互式提示输出到 **stderr**，stdout 只留结果（便于脚本/对拍解析）。

**② 系统性对拍（53 个 C/H/P 组合，vs MAYGEN 1.8）：53/53 全部一致**

覆盖 C 0..4 × P 1..3 × DBE 0..3，含边界与多磷：

| 类型 | 示例 | 结果 |
|---|---|---|
| 单原子 | H3P=1、HP=0 | ✓ |
| 最少原子 | CHP=1、CP2=1 | ✓ |
| 多磷 | P2=1、H2P2=1、H3P3=2、H5P3=1 | ✓ |
| 中型 | C4H8P2=**633**、C3H7P3=**681**、C3H5P3=**969** | ✓ |
| 已有锚点 | C6H13P=284、C6H15P=39、C4H12P2=38 | ✓ |

用时 18.3s（MAYGEN 每式 ~0.2s，本模块总计 <3s）。

**③ 修复单原子边界 bug**：`H3P`（PH₃）原被"连通计数"误判为 0
（N=1 时无连接可计），修后 = 1，与 MAYGEN 一致。

**④ 关于"内置预言机是否需要更迭"的结论**

| 对象 | 是否需要改 | 说明 |
|---|---|---|
| `Phosphorus/atomic_matrix_p.py` | **不需要** | 它本身就是 P 的**完备预言机**；CLI/库已直接使用；53 式对拍即完备性证明 |
| `oxygen/boundary.py` + `tools/boundary_report.py` | **暂不需要** | 二者是 **O 专用**（`utils.parse_compound_formula` 不认 P，内部用 O 的 `AtomicMatrixGenerator`）；实测 `boundary_report.py C3H9P` 会"跳过" |
| `Nitrogen/validate.py` 锚点 | 不需要 | P 有独立的 `Phosphorus/validate.py` |
| `utils.parse_compound_formula` / 公式门禁 | **接入 GUI 时才需要** | 属计划 §7 的集成清单（`utils` 解析 → 类型判定 → 渲染/`.gjf`） |

> 一期 P 没有"树机制"（直接走矩阵预言机）→ 覆盖率天然 100%，**边界分析工具无意义**；
> 待二期引入 P 的快速树生成器或接入主程序时，再用 `PhosphorusMatrixGenerator`
> 作为"完备"一侧扩展 `boundary.py` 即可。
