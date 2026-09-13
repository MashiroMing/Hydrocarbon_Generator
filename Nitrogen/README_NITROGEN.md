# Nitrogen — 含氮有机物异构体生成模块

> 面向：后续开发此模块的 AI Agent 与开发者。
> 本文记录 N 系的化学口径、架构决策、验证方法与已知边界，与 `PROJECT_GUIDE.md`
> （全项目）和 `O_project_guide.md`（含氧专项）分工：N 系细节以本文为准。

---

## 1. 定位与范围（第一阶段）

**已实现**：纯 C/H/N 有机物（胺/腈/亚胺/偶氮/含 N 杂环等）的异构体**完备枚举**。
- 化学口径：**中性八隅律**（C = 4 价，N = 3 价），与 MAYGEN 默认价态表一致；
- 数据表示：`networkx.Graph`（节点 `label='C'/'N'`，边 `bond_type`），
  与 O 系同构（PROJECT_GUIDE 3.1），仅原子类型扩展；
- 验证：内部预言机（完备枚举）为主 + MAYGEN 一次性外部对拍（开发期实证，
  不引入依赖，符合项目"去 MAYGEN 依赖"政策）。

**未实现（后续阶段）**：N+O 组合（硝基 -NO₂、酰胺 -CONH₂、硝基苯等，需 2 价 O 参与）、
含 N 杂环约束路径（必含吡啶等）、funcgroup 分类、GUI 集成、大公式性能优化。
按 `O_project_guide.md` §9 蓝图，NO₂ = 2O + 1N 组合约束属于 N/P 扩展的后续 Step。

---

## 2. 决策记录（动手前与用户确认的四个分支）

| # | 决策点 | 结论 | 依据 |
|---|---|---|---|
| 1 | 功能范围 | **纯 C/H/N 核心全集**（胺/腈/亚胺/偶氮/含 N 杂环） | 用户确认；硝基等含 O 组合留待 N+O 阶段 |
| 2 | 架构策略 | **独立模块 + 复用公共设施**（不改 O 系 / Main.py / GUI） | 用户确认；风险隔离，可独立验证 |
| 3 | 价态口径 | **中性八隅律 N=3**（默认），不含铵/配位 4~5 价 | MAYGEN 源码实证：`valences.put("N", 3)`；支持 `N(5)` 显式覆盖但默认关闭 |
| 4 | 验证标准 | **计数锚点 + 预言机对拍 + MAYGEN 一次性外部对拍** | 用户确认；预言机为主 |

**MAYGEN 价态处理结论（源码实证，`MAYGEN/src/main/java/maygen/Maygen.java`）**：
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
| N | 3 | 中性八隅律，与 MAYGEN 一致 |

- **氮规则 DBE**：`k = nC − nH/2 + nN/2 + 1` ⟺ `nH = 2·nC + nN + 2 − 2·k`；
- 图统计：每节点 `h = max(0, VALENCE[label] − bond_load)`，总 H = Σh；
- 门禁：`formula_matches_input(G, nC, nH, nN)`（H/C/N 严格一致，全链路唯一口径）。

### 3.2 树机制对 k 的贡献（generator.py 三种机制）

| 机制 | 操作 | 对 k | 覆盖 |
|---|---|---|---|
| A：C→N 替换 | (nC+nN) 碳骨架中选 nN 个 C 替换为 N（键型不变，H −1/位） | 不变 | 含 N 杂环（吡啶←苯）、叔胺中心（←异丁烷）、亚胺 C=N、腈（←末端炔）、偶氮（←C=C 两端） |
| B：端基 N | 在 nC 骨架 H 位挂 N 端基 | -NH₂: +0 / =NH: +1 / -C≡N: +2 | 伯胺、亚胺端基、腈端基 |
| C：桥 N | 在 nC 骨架 C-C 键插入 N | -NH-: +0 / -N=C-: +1 | 仲胺桥、亚胺桥 |

> 机制 B 的 k 守恒推导：`a·(-NH₂) + b·(=NH) + c·(-C≡N)` 使 `b + 2c = k_target − k_skel`，
> 且 `a + b + c = nN`（整数解枚举位置分配）。

---

## 4. 模块地图

| 文件 | 关键函数/类 | 职责 | 改动场景 |
|---|---|---|---|
| `formula.py` | `parse_formula` / `graph_formula_parts` / `formula_matches_input` / `dbe_from_parts` | 公式单一事实源（解析/统计/门禁/氮规则） | N+O 阶段加 O 解析与价态表 |
| `atomic_matrix_n.py` | `AtomicMatrixNGenerator.generate_for_target_h` / `generate_all_unsat` | **完备性预言机**（邻接矩阵原子级枚举，C=4/N=3） | 加 P/S 原子（价态表参数化） |
| `generator.py` | `NitrogenGenerator.generate` / `_mechanism_replace` / `_mechanism_terminal` / `_mechanism_bridge` | 树机制生成器（替换/端基/桥，公式门禁+碰撞安全去重） | 加 N+O 机制（硝基/酰胺） |
| `validate.py` | `check_anchors` / `check_coverage` / `smoke_gate` / `sample_structures` | CLI 验证入口（锚点电池 + 完备性 + 门禁冒烟） | 加锚点/断言 |

**复用**：WL 哈希 + VF2 同构去重（`_seen_add`，碰撞安全，禁止裸 set——O 系历史教训）、
骨架自举（预言机生成纯 C 骨架）、公式门禁模式（O 系 `add()` 同构）。

---

## 5. 验证方法（agent 必跑）

```bash
# 交互模式：进入后逐条输入分子式回车即计算（默认行为）
python -m Nitrogen.validate
#   C2H7N   -> 打印样例 + 锚点核对 + 覆盖率
#   C5H5N   -> 同上（重原子 >7 时自动跳过覆盖率枚举）
#   quit/exit/空行 退出；anchors 跑锚点电池；help 显示用法

# 批量：命令行直接给出一个或多个分子式（打印结构样例 + 锚点 + 覆盖率）
python -m Nitrogen.validate C5H5N
python -m Nitrogen.validate C2H7N C3H9N

# 强制进入交互模式（即使给了命令行分子式）
python -m Nitrogen.validate -i

# 默认计数锚点电池 + 完备性 + 门禁冒烟（约 3 分钟，瓶颈 C6H5N/C6H7N）
python -m Nitrogen.validate --anchors

# 详细模式（打印缺失结构）
python -m Nitrogen.validate -v

# MAYGEN 一次性外部对拍（需本机 Java；开发期实证用，不写入依赖/测试流程）
java -jar MAYGEN/MAYGEN-1.8.jar -f C5H5N -t        # 期望 685
```

### 5.1 计数锚点电池（图口径，预言机 = MAYGEN 实证）

| 分子式 | 计数 | 分子式 | 计数 |
|---|---|---|---|
| C2H7N | 2 | C4H9N | 35 |
| C3H9N | 4 | C5H5N | 685 |
| C2H5N | 4 | C6H5N | 4394 |
| C2H3N | 5 | C6H7N | 4378 |
| C2H8N2 | 6 | C6H15N | 39 |
| C3H7N | 12 | C4H11N | 8 |

> 全部 12 个公式预言机计数与 MAYGEN 完全一致（2026-08-26 实证）。
> 树机制覆盖率：上述 ≤6 重原子公式 **100%**（零缺失，含吡啶 C5H5N=685）。

### 5.2 不变式（与 O 系同源，改代码底线）

| # | 不变量 | 破坏症状 |
|---|---|---|
| I1 | 公式门禁全链路同口径（formula.py 单一事实源） | 结果混入分子式不符结构 |
| I2 | 去重碰撞安全（同 WL 哈希必须 VF2 全属性同构） | 计数莫名减少 |
| I3 | 图口径计数（凯库勒式变体不去重） | 计数被"合并" |
| I4 | 预言机与 MAYGEN 计数一致（锚点电池） | 锚点漂移 |

---

## 6. 已知边界与后续 TODO

1. **N+O 组合（第二阶段）**：硝基 -NO₂ / 酰胺 -CONH₂ / 硝基苯 / 含 N 含 O 杂环。
   需扩展：`formula.py` 解析 O、价态表加 O=2、预言机 `_max_bond_type` 加 C-N/N-O/O-N 键型、
   `generator.py` 加 NO₂ 组合约束（NO₂ = 2O + 1N）、funcgroup 分类（胺/腈/硝基/酰胺）。
2. **含 N 杂环约束路径**：`C5H5N + 必含吡啶` 约束（参照 O 系 `_generate_from_fragment` +
   `fragment_library.py` 预结构模式）。
3. **大公式性能**：原子预言机对 ≥8 重原子变慢（C6H5N ≈ 68s）；树机制 + 进程池并行
   （参照 O 系 `use_parallel`）为后续优化点。
4. **GUI 集成**：暂不动 Main.py；后续接入 GeneratorManager 需同步 `parse_compound_formula`、
   `MOL_NAMES_CN`、类型分发、funcgroup 面板。
5. **高价态 N（预留）**：MAYGEN 支持 `N(5)` 显式覆盖；本项目默认 N=3，如需铵盐/配位
   形态，在价态表参数化处扩展（`formula.VALENCE` / 预言机 `valences`）。

---

## 7. 快速示例

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
