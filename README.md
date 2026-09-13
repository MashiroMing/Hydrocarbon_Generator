# Hydrocarbon_Generator — 有机分子异构体生成与可视化工具

基于 **Python + Tkinter** 的有机分子同分异构体生成与可视化软件：输入分子式，即可自动枚举全部异构体，并进行 **2D 键线式渲染**、**3D 交互可视化** 与 **Gaussian 输入文件导出**。

支持 **烃类**（烷 / 烯 / 炔 / 环 / 多环等）、**含氧衍生物**（醇 / 酚 / 醚 / 醛 / 酮 / 羧酸 / 酯 / 环氧 / 过氧 / 含氧杂环等）与 **卤代烃**（F / Cl / Br / I）。

## 功能特性

- **异构体自动枚举**：输入分子式，按不饱和度自动判定类型并分派生成器，枚举全部同分异构体
- **严格去重**：Weisfeiler-Lehman 图哈希 + VF2 全属性同构校验，确保不重复、不遗漏
- **分子式一致性门禁**：全管线末端统一校验，输出分子式与用户输入严格一致
- **高级结构筛选**：按主链长度、环尺寸、必含基团（预置片段库）筛选
- **官能团筛选**：醇 / 酚 / 醚 / 过氧 / 醛 / 酮 / 羧酸 / 酯 / 环氧 / 含氧杂环等标签，支持「或 / 与」模式与 O 分布形态过滤
- **2D 键线式渲染**：RDKit MolDraw2D + CairoSVG 生成标准键线式，matplotlib 手动绘制作回退
- **3D 交互可视化**：Matplotlib 3D（TkAgg）+ 自研坐标算法（RDKit 嵌入 + 键长松弛），多进程并行加速
- **Gaussian 导出**：批量导出 `.gjf` 文件，支持仅导出选中项
- **内部边界分析（预言机）**：不依赖 MAYGEN / Java 即可量化生成器对任意分子式的完备性边界
- **一键打包**：提供 PyInstaller 目录模式配置（Windows / Linux）

## 支持的分子类型

| 类别 | 覆盖范围 |
|---|---|
| 烃类 | 烷烃、烯烃、炔烃、二烯 / 三烯 / 四烯 / 多烯烃、烯炔烃、环烷烃、环烯烃、单环多烯烃、多环烷烃、多环多烯炔烃 |
| 含氧衍生物 | 醇、酚、醚、醛、酮、羧酸、酯、环氧、过氧 / 氢过氧、含氧杂环（呋喃 / 四氢呋喃 / 二氧六环等） |
| 卤代烃 | F / Cl / Br / I 的单取代与混合取代 |

> 分子类型由分子式的不饱和度自动判定，无需手动选择。

## 分子式输入格式

```
C{n}H{m}[O{o}][卤素...]
```

- **骨架元素**：`C`、`H`、`O`（个数可省略，省略时为 1，如 `C6H6O`）
- **卤素**：`F`、`Cl`、`Br`、`I`，可混合，如 `C2H4ClBr`
- **示例**：`C6H14`、`C4H8`、`C3H6O`、`C7H6O2`、`C6H5Cl`、`C2H4BrCl`

> 单个碳写作 `CH4`；卤素按 1 价计入不饱和度。另接受氘代标记 `D`（用于分子式合法性校验）。

## 高级筛选

| 约束 | 说明 |
|---|---|
| 主链长度 | 限制链状分子最长碳链的碳原子数范围 |
| 环尺寸 | 限制环状分子环的元数范围 |
| 必含基团 | 要求结构必须包含某个预置片段 |

**预置片段库**：

- 芳香环：苯环、萘、蒽、菲、联苯
- 杂环：呋喃、2H-吡喃、1,4-二氧六环、四氢呋喃、环氧乙烷
- 饱和环：环丙烷、环丁烷、环戊烷、环己烷
- 多环饱和：降冰片烷、金刚烷
- 碳链：乙基

**官能团筛选（含氧）**：醇、酚、醚、过氧、醛、酮、羧酸、酯、环氧、含氧杂环、偕二醇；支持「任一命中（或）/ 全部命中（与）」模式，并可按下拉选择 **O 分布**（全部取代基型 / 全部骨架型 / 混合）。

> **快速生成模式**：跳过稠合 O 杂环的跨碳深环枚举，大分子式约提速 30%（代价：缺少该类稠合杂环）。

## 项目结构

```
├── Main.py                              # 主界面（Tkinter GUI）
├── utils.py                             # 分子式解析、生成分发、去重与键线式渲染
├── structure_filter.py                  # 统一结构约束筛选 + 分子式一致性门禁
├── molecular_constants.py               # 内部编码数据
├── icon.ico                             # 应用图标
├── original_programs/                   # 纯烃生成与可视化模块
├── oxygen/                              # 含氧衍生物生成模块
│   ├── fragment_library.py              #   预置片段库
│   ├── unified_oxo_generator.py         #   统一含氧生成器（含片段加速通道）
│   ├── funcgroup.py                     #   含氧官能团分类
│   └── ...                              #   官能团 / 环氧 / 醚桥 / 过氧 / 含氧环等子生成器
├── halogen/                             # 卤代物生成模块
├── Nitrogen/                            # 含氮化合物生成模块（独立，尚未接入 GUI）
├── tools/                               # 独立工具（边界分析 / MAYGEN 对照）
├── tests/                               # 回归测试
└── *.spec                               # PyInstaller 打包配置（Windows / Linux）
```

> **技术栈说明**：GUI 基于 **Tkinter**（标准库），3D 可视化采用 **Matplotlib 3D**（TkAgg 后端），键线式渲染基于 **RDKit + CairoSVG**。项目**不依赖** PyQt5 / PySide 等 Qt 框架。

## 独立模块与工具

> 以下内容不参与主程序运行与打包。

| 路径 | 说明 |
|---|---|
| `Nitrogen/` | **含氮化合物**（胺 / 腈 / 亚胺 / 偶氮 / 含 N 杂环）的完备枚举模块，内置完备性预言机与计数锚点验证。当前为独立 CLI / 库，**尚未接入 GUI**：`python -m Nitrogen.validate C5H5N` |
| `tools/boundary_report.py` | 内部边界分析，量化生成器对任意分子式的覆盖缺口（无 MAYGEN 依赖） |
| `tools/maygen_validate.py` | 与 MAYGEN 的异构体数量交叉验证（**可选**，需自行获取 MAYGEN） |
| `tests/` | 分子式门禁与官能团分类回归测试 |

## 内部边界分析（预言机）

本项目内置**不依赖 MAYGEN / Java / 任何外部工具**的完备性预言机，用于量化生成器的完备性边界：

- **树机制集合（Tree）**：生成器「碳骨架 + O 修饰组合」能够表达的全部结构，覆盖 OH / =O、醚 / 过氧 / 双醚桥、环氧、OOH、O 环等含氧修饰；
- **完备集合（Matrix）**：按目标氢数（八隅律 / 价态模型）做邻接矩阵全图枚举，数学上与 MAYGEN 等价；
- **边界（Boundary）** = 完备 − 树，即树机制无法表达的结构，并按官能团 SMARTS 自动分类（酯键、羧酸、羰基、醚、环氧、过氧、芳环等）；
- **覆盖率** = `|树 ∩ 完备| / |完备|`。

```bash
python tools/boundary_report.py C7H6O2
python tools/boundary_report.py C3H6O2 C6H6O2 --top-missing 15
python tools/boundary_report.py C7H6O2 --max-atoms 10
```

## 环境依赖

- Python 3.10+（开发环境为 3.11）
- Tkinter（Python 标准库，通常随 Python 自带）
- RDKit、NetworkX、NumPy、Matplotlib（含 `mpl_toolkits.mplot3d`）、CairoSVG、Pillow

## 安装与运行

```bash
git clone https://github.com/MashiroMing/Hydrocarbon_Generator.git
cd Hydrocarbon_Generator
pip install rdkit networkx numpy matplotlib cairosvg pillow
python Main.py
```

> 若提示缺少 Tkinter，请安装系统 tk 支持（Linux：`sudo apt install python3-tk`）。

## 使用方法

1. 输入分子式（如 `C6H14`、`C4H8`、`C3H6O`、`C6H5Cl`）
2. 点击生成，程序自动判定类型并枚举全部异构体
3. 左侧列表浏览结果，可启用「高级筛选」与「含氧官能团筛选」精确定位目标结构
4. 右侧查看分子信息、键线式图片与 3D 结构
5. 通过「另存为」导出 Gaussian `.gjf` 文件（支持仅导出选中项）

## 打包发布

```bash
pyinstaller Hydrocarbon_Generator.spec         # Windows
pyinstaller Hydrocarbon_Generator_linux.spec   # Linux
```

产物位于 `dist/Hydrocarbon_Generator/`。如需自定义，可修改 `.spec` 中的 `name`（程序名）、`icon`（图标）、`console`（是否显示控制台）、`datas` / `hiddenimports` / `excludes` 等字段。

> **注意**：请勿将 `tkinter` 加入 `excludes`，否则 GUI 无法运行。Linux 下若中文显示为方框，请安装中文字体：`sudo apt install -y fonts-wqy-zenhei fonts-noto-cjk`。

## License

MIT
