# MolVis - 分子异构体可视化工具

基于 Python + Tkinter 的有机化学分子异构体生成与可视化软件，支持多种烃类的同分异构体自动生成、3D 分子结构可视化及键线式渲染。

## 功能特性

- **异构体自动生成**：输入分子式，自动枚举所有同分异构体
- **3D 分子可视化**：基于 RDKit + Matplotlib 的交互式 3D 分子结构展示
- **键线式渲染**：自动生成标准键线式（Skeletal Formula），支持锯齿状链结构
- **多类型支持**：烷烃、烯烃、炔烃、二烯烃、环烷烃、环烯烃、多烯烃等
- **并行计算**：多进程加速大规模异构体的 3D 坐标计算
- **一键打包**：支持 PyInstaller 目录模式封装为独立可执行程序

## 支持的分子类型

| 分子式通式 | 类型 |
|---|---|
| CₙH₂ₙ₊₂ | 烷烃 |
| CₙH₂ₙ | 单烯烃 / 单环烷烃 |
| CₙH₂ₙ₋₂ | 单炔烃 / 二烯烃 / 单环烯烃 |
| CₙH₂ₙ₋₄ | 烯炔烃 / 三烯烃 / 单环二烯烃 |
| CₙH₂ₙ₋₆ | 四烯烃 / 单环三烯烃 |
| CₙH₂ₙ₋₂ₖ | k-烯烃 / 单环多烯烃 |

## 项目结构

```
├── Main.py                              # 主界面（Tkinter GUI）
├── utils.py                             # 分子式解析与生成分发
├── icon.ico                             # 应用图标
├── core_modules/                        # 核心生成模块
│   ├── cycloalkane_app.py               # 环烷烃生成器
│   ├── cycloalkene_generator.py         # 环烯烃生成器
│   └── cyclopolyene_generator.py        # 环多烯烃生成器
├── original_programs/                   # 基础生成模块
│   ├── alkane_isomer_visualizer.py      # 烷烃异构体生成与可视化
│   ├── alkene.py                        # 烯烃异构体生成
│   ├── alkene_visualizer.py             # 烯烃可视化
│   ├── alkyne.py                        # 炔烃异构体生成
│   ├── alkenyl_generator.py             # 烯炔烃生成器
│   └── polyene_generator.py             # 多烯烃生成器
```

## 环境依赖

- Python 3.10+
- RDKit
- NetworkX
- NumPy
- Matplotlib
- CairoSVG
- Pillow

## 安装与运行

1. 克隆仓库：

```bash
git clone https://github.com/MashiroMing/molvis.git
cd molvis
```

2. 安装依赖：

```bash
pip install rdkit networkx numpy matplotlib cairosvg pillow
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

## 打包发布

使用 PyInstaller 进行目录模式打包。

**Windows：**

```bash
pyinstaller molvis.spec
```

**Linux：**

```bash
pyinstaller molvis_linux.spec
```

打包产物均位于 `dist/molvis/` 目录下。

### 运行打包后的程序

**Windows：**

```bash
dist\molvis\molvis.exe
```

**Linux：**

```bash
./dist/molvis/molvis
```

> **Linux 用户注意**：如果中文显示为方框，请安装中文字体：
> ```bash
> sudo apt install -y fonts-wqy-zenhei fonts-wqy-microhei fonts-noto-cjk
> ```

## License

MIT
