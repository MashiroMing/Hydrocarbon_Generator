# -*- mode: python ; coding: utf-8 -*-
# Windows 平台目录模式打包（精简版，干净 venv 专用）
# 使用方法: pyinstaller molvis.spec

import os
import sys

block_cipher = None
root_dir = os.path.dirname(SPEC)

# 排除 original_programs 下的非代码目录
_exclude_dirs = ['.codebuddy', '__pycache__', '.git']

a = Analysis(
    ['Main.py'],
    pathex=[root_dir],
    binaries=[],
    datas=[
        ('original_programs', 'original_programs'),
        ('utils.py', '.'),
    ],
    hiddenimports=[
        # RDKit（pip 版 rdkit 需显式声明子模块）
        'rdkit',
        'rdkit.Chem',
        'rdkit.Chem.AllChem',
        'rdkit.Chem.BondType',
        'rdkit.RDLogger',
        'rdkit.Chem.Draw',
        'rdkit.Chem.Draw.rdMolDraw2D',
        # NumPy（PyInstaller hook 自动处理大部分，仅需核心声明）
        'numpy',
        # NetworkX
        'networkx',
        # Matplotlib（TkAgg 后端）
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'mpl_toolkits.mplot3d',
        # PIL / CairoSVG（键线式渲染）
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cairosvg',
        # 项目模块
        'original_programs.alkane_isomer_visualizer',
        'original_programs.alkene_visualizer',
        'original_programs.alkene',
        'original_programs.alkyne',
        'original_programs.alkenyl_generator',
        'original_programs.polyene_generator',
        'original_programs.multcycloalkane',
        'original_programs.multcyclomultalkane',
        'original_programs.cycloalkene_generator',
        'original_programs.cyclopolyene_generator',
        'original_programs.polyalkenyne',
        'utils',
        # 多进程（并行加速）
        'multiprocessing',
        'concurrent.futures',
    ],
    hookspath=[],
    runtime_hooks=[os.path.join(root_dir, 'runtime_hook.py')],
    excludes=[
        # 不属于项目的 GUI 框架
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'qtpy',
        # 科学计算 / 数据科学生态（干净 venv 中不存在，安全排除）
        'pandas', 'scipy', 'sympy', 'sklearn', 'skimage',
        'statsmodels', 'astropy', 'xarray', 'numba', 'llvmlite',
        'dask', 'distributed', 'pyarrow', 'numexpr', 'bottleneck',
        # 交互式可视化
        'bokeh', 'panel', 'holoviews', 'plotly', 'altair',
        'seaborn', 'plotnine',
        # Web / Jupyter
        'flask', 'aiohttp', 'werkzeug', 'jupyter', 'ipykernel',
        'notebook', 'ipywidgets', 'tornado',
        # 杂项
        'sqlalchemy', 'h5py', 'conda', 'anaconda',
        'sphinx', 'pytest', 'test', 'docutils',
        'MAYGEN', 'MolGen', 'PMG',
    ],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='molvis',
    debug=False,
    strip=False,
    upx=True,
    console=True,
    icon=os.path.join(root_dir, 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='molvis',
)
