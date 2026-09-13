# Hydrocarbon_Generator - 分子异构体可视化工具

基于 Python + Tkinter 的有机化学分子异构体生成与可视化软件，支持多种烃类（及含氧、含卤代物）的同分异构体自动生成、3D 分子结构可视化及键线式渲染。

## 功能特性

- **异构体自动生成**：输入分子式，自动枚举所有同分异构体
- **3D 分子可视化**：基于 Matplotlib 3D（TkAgg 后端）的自研坐标算法交互式分子结构展示
- **键线式渲染**：基于 RDKit + CairoSVG 自动生成标准键线式（Skeletal Formula），支持锯齿状链结构
- **多类型支持**：烷烃、烯烃、炔烃、二烯烃、环烷烃、环烯烃、多烯烃、多环烷烃、多环多烯炔烃，以及含氧、含卤代有机物
- **结构约束筛选**：支持按环尺寸、主链长度、必需片段（苯环、呋喃等预置片段）筛选异构体
- **内部边界分析（预言机）**：内置完备性预言机，无需外部工具即可量化生成器对任意分子式的边界覆盖情况（详见下方[核心算法](#内部边界分析预言机)章节）
- **化学约束验证**：基于键负载的化学有效性检查，兼容桥头碳等环结构情形
- **WL 哈希去重**：采用 Weisfeiler-Lehman 图哈希 + 同构检查确保异构体无重复
- **并行计算**：多进程加速大规模异构体的 3D 坐标计算
- **Gaussian 输出**：支持导出 `.gjf` 格式文件用于 Gaussian 计算对接
- **一键打包**：支持 PyInstaller 目录模式封装为独立可执行程序

## 支持的分子类型

| 分子式通式 | 类型 |
|---|---|
| CₙH₂ₙ₊₂ | 烷烃 |
| CₙH₂ₙ | 单烯烃 / 单环烷烃 |
| CₙH₂ₙ₋₂ | 单炔烃 / 二烯烃 / 单环烯烃 / 双环烷烃 / 多环多烯炔烃 |
| CₙH₂ₙ₋₄ | 烯炔烃 / 三烯烃 / 单环二烯烃 / 三环烷烃 / 多环多烯炔烃 |
| CₙH₂ₙ₋₆ | 四烯烃 / 单环三烯烃 / 四环烷烃 / 多环多烯炔烃 |
| CₙH₂ₙ₋₂ₖ | k-烯烃 / 单环多烯烃 / k环烷烃 / 多环多烯炔烃 |

## 项目结构

```
├── Main.py                              # 主界面（Tkinter GUI）
├── utils.py                             # 分子式解析与生成分发
├── structure_filter.py                  # 统一的异构体约束筛选层
├── molecular_constants.py               # 内部编码数据
├── icon.ico                             # 应用图标
├── original_programs/                   # 纯烃生成模块
│   ├── alkane_isomer_visualizer.py      # 烷烃异构体生成与可视化
│   ├── alkene.py                        # 烯烃异构体生成
│   ├── alkene_visualizer.py             # 烯烃可视化
│   ├── alkyne.py                        # 炔烃异构体生成
│   ├── alkenyl_generator.py             # 烯炔烃生成器
│   ├── polyene_generator.py             # 多烯烃生成器
│   ├── polyalkenyne.py                  # 无环多烯炔烃生成器
│   ├── cycloalkene_generator.py         # 单环烯烃生成器
│   ├── cyclopolyene_generator.py        # 单环多烯烃生成器
│   ├── multcycloalkane.py               # 多环烷烃生成器（含单环）
│   └── multcyclomultalkane.py           # 多环多烯炔烃生成器（组合式分派）
├── oxygen/                              # 含氧有机物生成模块
│   ├── fragment_library.py              # 预置片段库（苯环、呋喃等）
│   ├── unified_oxo_generator.py         # 统一含氧生成器（含片段加速通道）
│   └── ...                              # 含氧功能团/环/醚桥等子生成器
├── halogen/                             # 卤代物生成模块
├── MAYGEN/                              # 外部图生成器（Java）
├── tools/                               # 辅助工具
└── tests/                               # 测试用例
```

> **技术栈说明**：GUI 基于 **Tkinter**（标准库）构建，3D 可视化采用 **Matplotlib 3D**（TkAgg 后端），键线式渲染基于 **RDKit + CairoSVG**。项目不依赖 PyQt5/PySide 等 Qt 框架。

## 核心算法

### 多环多烯炔烃生成器（multcyclomultalkane）

采用组合式分派策略，根据 `(环数, 双键数, 三键数)` 分派到已验证的子模块：

- **n_rings=0** → 委托 `polyalkenyne`（无环多烯炔）
- **n_rings=1, n_triple=0** → 委托 `cyclopolyene_generator`（单环多烯）
- **n_rings≥2, n_db=0, n_tb=0** → 委托 `multcycloalkane`（多环烷烃）
- **通用情况** → 先生成多环骨架，再逐步插入双键/三键，每步通过化学约束验证和 WL 哈希去重

### 去重策略

使用 Weisfeiler-Lehman 图哈希作为桶键，同桶内做度序列预过滤 + `is_isomorphic` 精确同构检查，兼顾效率与正确性。

### 内部边界分析预言机

> 这是本项目的一大特色：**不依赖 MAYGEN / Java / 任何外部工具**，即可对任意分子式量化生成器的完备性边界。

**核心思想**——"树生成为主体，矩阵补边缘"的可量化版本：

- **树机制集合（Tree）**：由 `UnifiedOxoGenerator`（关闭矩阵补漏）生成的产物，即"碳骨架 + O 修饰组合"机制能够表达的所有分子，覆盖 OH/=O、醚/过氧/双醚桥、环氧、OOH、oring 等含氧修饰；
- **完备集合（Matrix）**：由 `AtomicMatrixGenerator` 按目标氢数（八隅律/价态模型）做邻接矩阵全图枚举得到，数学上与 MAYGEN 等价，因此可作为**完备性预言机**；
- **边界（Boundary）** = 完备集合 − 树集合，即树机制无法表达的分子，并按官能团 SMARTS 自动分类（酯键、羧酸、羰基、醚、环氧、过氧、芳环等）；
- **内部覆盖率（Coverage）** = `|树 ∩ 完备| / |完备|`，直观反映生成器覆盖完备空间的比例。

**用法**：通过 `tools/boundary_report.py` 命令行工具对单个或多个分子式批量分析：

```bash
python tools/boundary_report.py C7H6O2
python tools/boundary_report.py C3H6O2 C6H6O2 --top-missing 15
python tools/boundary_report.py C7H6O2 --max-atoms 10
```

输出包含：树/矩阵/交集/边界数量、覆盖率、缺失结构的官能团分类统计与示例 SMILES。当某分子式的边界（缺失）分子过多时，即可据此定位生成机制盲区并针对性补齐；MAYGEN 仍可作为**可选的外部交叉复核**工具。

## 环境依赖

- Python 3.10+（开发环境为 3.11）
- Tkinter（Python 标准库，通常随 Python 自带）
- RDKit
- NetworkX
- NumPy
- Matplotlib（含 `mpl_toolkits.mplot3d` 3D 模块）
- CairoSVG
- Pillow

> 无需安装 PyQt5/PySide 等 Qt 框架。

## 安装与运行

1. 克隆仓库：

```bash
git clone https://github.com/MashiroMing/Hydrocarbon_Generator.git
cd Hydrocarbon_Generator
```

2. 安装依赖：

```bash
pip install rdkit networkx numpy matplotlib cairosvg pillow
```

> 若运行后提示缺失 Tkinter，请安装系统 tk 支持（Linux: `sudo apt install python3-tk`）。

3. 运行程序：

```bash
python Main.py
```

## 使用方法

1. 在输入框中输入分子式（如 `C6H14`、`C4H8`）
2. 点击生成按钮，程序将自动识别分子类型并枚举所有异构体
3. 左侧列表显示所有异构体，点击选中查看详情
4. 右侧展示分子信息、键线式图片和 3D 结构
5. 可通过筛选下拉框过滤特定类型的异构体
6. 支持导出 Gaussian `.gjf` 格式文件

## 打包发布

项目提供了 PyInstaller 的 `.spec` 配置文件，可直接用于目录模式打包。

**Windows：**

```bash
pyinstaller Hydrocarbon_Generator.spec
```

**Linux：**

```bash
pyinstaller Hydrocarbon_Generator_linux.spec
```

打包产物均位于 `dist/Hydrocarbon_Generator/` 目录下。

### 自定义打包配置

如果你需要自行封装，请修改 `.spec` 文件中的以下内容：

| 配置项 | 所在文件 | 说明 |
|---|---|---|
| `name` | `EXE()` 中的 `name='Hydrocarbon_Generator'` | 可执行文件名称 |
| `icon` | `EXE()` 中的 `icon=os.path.join(root_dir, 'icon.ico')` | 应用图标路径 |
| `console` | `EXE()` 中的 `console=True` | 设为 `False` 可隐藏控制台窗口 |
| `datas` | `Analysis()` 中 | 需要打包的额外数据文件/目录 |
| `hiddenimports` | `Analysis()` 中 | PyInstaller 无法自动检测的隐式依赖 |
| `excludes` | `Analysis()` 中 | 需要排除的模块（可减小打包体积） |

> **注意**：请勿将 `tkinter` 加入 `excludes` 列表，否则 GUI 将无法正常运行。

### 运行打包后的程序

**Windows：**

```bash
dist\Hydrocarbon_Generator\Hydrocarbon_Generator.exe
```

**Linux：**

```bash
./dist/Hydrocarbon_Generator/Hydrocarbon_Generator
```

> **Linux 用户注意**：如果中文显示为方框，请安装中文字体：
> ```bash
> sudo apt install -y fonts-wqy-zenhei fonts-wqy-microhei fonts-noto-cjk
> ```

## License

MIT
