# Hydrocarbon_Generator - 分子异构体可视化工具

https://zenodo.org/records/20844657

基于 Python + PyQt5 的有机化学分子异构体生成与可视化软件，支持多种烃类的同分异构体自动生成、3D 分子结构可视化及键线式渲染。

## 功能特性

- **异构体自动生成**：输入分子式，自动枚举所有同分异构体
- **3D 分子可视化**：基于 RDKit + Matplotlib 的交互式 3D 分子结构展示
- **键线式渲染**：自动生成标准键线式（Skeletal Formula），支持锯齿状链结构
- **多类型支持**：烷烃、烯烃、炔烃、二烯烃、环烷烃、环烯烃、多烯烃、多环烷烃、多环多烯炔烃等
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
├── Main.py                              # 主界面（PyQt5 GUI）
├── utils.py                             # 分子式解析与生成分发
├── icon.ico                             # 应用图标
├── molecular_constants.py               # 内部编码数据
├── original_programs/                   # 生成模块
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
```

## 核心算法

### 多环多烯炔烃生成器（multcyclomultalkane）

采用组合式分派策略，根据 `(环数, 双键数, 三键数)` 分派到已验证的子模块：

- **n_rings=0** → 委托 `polyalkenyne`（无环多烯炔）
- **n_rings=1, n_triple=0** → 委托 `cyclopolyene_generator`（单环多烯）
- **n_rings≥2, n_db=0, n_tb=0** → 委托 `multcycloalkane`（多环烷烃）
- **通用情况** → 先生成多环骨架，再逐步插入双键/三键，每步通过化学约束验证和 WL 哈希去重

### 去重策略

使用 Weisfeiler-Lehman 图哈希作为桶键，同桶内做度序列预过滤 + `is_isomorphic` 精确同构检查，兼顾效率与正确性。

## 环境依赖

- Python 3.10+
- PyQt5
- RDKit
- NetworkX
- NumPy
- Matplotlib
- CairoSVG
- Pillow

## 安装与运行

1. 克隆仓库：

```bash
git clone https://github.com/MashiroMing/Hydrocarbon_Generator.git
cd molvis
```

2. 安装依赖：

```bash
pip install PyQt5 rdkit networkx numpy matplotlib cairosvg pillow
```

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
