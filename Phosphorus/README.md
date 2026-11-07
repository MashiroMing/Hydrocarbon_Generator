# Phosphorus — 含磷异构体生成（第一期：仅 C / H / P）

> 独立模块，**不接入** `Main.py`；命令行 / 库两种用法。
> 规划与决策见 [`Instruction Data/Phosphorus_Development_Plan.md`](../Instruction%20Data/Phosphorus_Development_Plan.md)。

## 口径（已确认）

| 项 | 值 |
|---|---|
| 元素 | **C / H / P**（不含 O、N、卤素） |
| P 价态 | **3**（与 MAYGEN 1.8 默认一致）；`valence_p=5` 为预留开关（实验性，无锚点保证） |
| 键级 | 单键 + **双键/三键全允许**（P–C / P–H / P–P；含 P=C、P≡C、P≡P） |
| 电荷 | 不建模（二期接 O 时再引入） |
| 计数口径 | **图口径**（凯库勒/键级变体分开计），与 MAYGEN 逐式对拍一致 |

## 快速开始

```bash
python -m Phosphorus.cli C3H9P              # 列出结构（SMILES）
python -m Phosphorus.cli C3H9P --count      # 式<TAB>计数<TAB>秒（便于与 MAYGEN 对拍）
python -m Phosphorus.cli C6H13P --smi       # 每行一个 SMILES
python -m Phosphorus.cli C3H7P --json out.json
python -m Phosphorus.cli C6H15P --parallel  # 前缀并行（结果与串行一致）
python -m Phosphorus.cli PH3 --count        # 也支持无碳含磷式
```

```python
from Phosphorus import PhosphorusMatrixGenerator, parse_formula_p

n_c, n_h, n_p = parse_formula_p('C3H9P')
graphs = PhosphorusMatrixGenerator(n_c, n_p).generate(n_h)   # [nx.Graph, ...]
```

## 验证

```bash
python -m Phosphorus.validate --anchors   # 21 个分子式 vs MAYGEN 1.8（P=3）
python -m Phosphorus.validate --props     # 公式自洽 / 价态 / 连通性 / 无重复
python -m Phosphorus.validate --equiv     # 串行 == 并行
python tests/test_phosphorus.py           # 项目测试（21 断言，约 16s）
python tests/test_phosphorus.py --quick   # 小锚点快速档
```

**当前状态**：锚点 21/21 全对（含 C6H13P=284、`P2`=1、`C2H2P2`=19 等），
全套测试 PASS=21 / FAIL=0。

## 架构

| 文件 | 职责 |
|---|---|
| `formula.py` | 分子式唯一事实源：解析 / 图→式 / 格式化 / DBE / 门禁 |
| `atomic_matrix_p.py` | 矩阵级完备预言机：有序生成 + 目标价剪枝 + 矩阵级 WL/精确同构去重 + 前缀并行 |
| `cli.py` | 命令行（--count / --smi / --json / --parallel / --timeout / --max-atoms） |
| `validate.py` | 验证入口（锚点 / 性质 / 等价） |

## 锚点表（MAYGEN 1.8，P=3）

| 式 | 计数 | 式 | 计数 |
|---|---|---|---|
| CH3P | 1 | C4H11P | 8 |
| CH5P | 1 | C5H13P | 17 |
| C2H5P | 4 | C6H13P | 284 |
| C2H7P | 2 | C6H15P | 39 |
| C3H5P | 21 | C2H8P2 | 6 |
| C3H7P | 12 | C3H10P2 | 14 |
| C3H9P | 4 | C4H12P2 | 38 |
| C4H9P | 35 | P2 | 1 |
| C2H2P2 | 19 | C2H4P2 | 27 |
| CH2P2 | 4 | C2P2 | 5 |
| C3H3P | 19 | | |

## 已知限制（一期）

1. 仅 C/H/P；接 O/N/卤素需按计划 §7 的清单逐步集成（`utils.py` 未改动）；
2. P=5 价未验证（无锚点），仅供实验；
3. 无 `.gjf` / 2D / 3D（属 GUI 集成阶段）；
4. 无形式电荷（磷鎓、P=O 电荷分离式留二期）。
