"""
生成并可视化烃类的同分异构体 - 主界面
统一管理烷烃、烯烃、二烯烃、环烷烃的异构体生成与可视化
"""

import sys
import os
import io
from pathlib import Path
from molecular_constants import B64_SHOKO_CN, B64_SHOKO_EN
# 添加项目根目录到 Python 路径
_project_root = Path(sys.executable).resolve().parent
if hasattr(sys, '_MEIPASS'):
    # PyInstaller 打包后，项目根目录为 _MEIPASS
    _project_root = Path(sys._MEIPASS)
else:
    _project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 在导入 tkinter 之前设置 matplotlib 后端
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端以支持 GUI 可视化窗口

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import traceback

# 修复 tkinter.Image.__del__ 的线程安全问题
# matplotlib TkAgg 后端在后台线程中关闭 figure 时，会触发 tkinter.Image 的垃圾回收，
# 而 __del__ 尝试在非主线程中调用 Tcl 命令，导致 RuntimeError: main thread is not in main loop
_original_image_del = tk.Image.__del__
def _threadsafe_image_del(self):
    try:
        _original_image_del(self)
    except RuntimeError:
        pass  # 忽略非主线程中的 Tcl 调用错误
tk.Image.__del__ = _threadsafe_image_del

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入分子式解析与生成分发工具
from utils import (
    GeneratorManager,
    parse_molecule_input,
    compute_formula,
    format_formula,
    MOL_NAMES_CN,
    MOL_NAMES_FILTER,
    MOL_NAMES_EN,
    render_skeletal_formula,
)

# 键线式渲染特性检测
try:
    from rdkit.Chem.Draw import rdMolDraw2D
    import cairosvg
    from PIL import Image, ImageTk
    HAS_SKELETAL_DRAWING = True
except ImportError:
    HAS_SKELETAL_DRAWING = False


def _parallel_compute_coords(args):
    """多进程并行计算分子3D坐标（顶层函数，可被pickle）"""
    idx, graph_data, mol_type, n_carbon = args

    try:
        import networkx as nx
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem, BondType
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        # 反序列化图对象
        if isinstance(graph_data, bytes):
            import pickle
            G = pickle.loads(graph_data)
        elif isinstance(graph_data, nx.Graph):
            G = graph_data
        else:
            return idx, {}, []

        # 根据分子类型计算坐标
        if mol_type == 'alkane':
            return idx, *_rdkit_coords_single(G)

        elif mol_type in ('alkene', 'diene', 'cycloalkene', 'cyclopolyene',
                          'triene', 'tetraene', 'polyene', 'multcycloalkane'):
            return idx, *_rdkit_coords_with_bonds(G, double_only=True)

        elif mol_type in ('alkyne', 'alkenyl', 'multcyclomultalkane'):
            return idx, *_rdkit_coords_with_bonds(G, double_only=False)

        elif mol_type == 'cycloalkane':
            return idx, *_rdkit_coords_single(G)

        else:
            return idx, {}, []

    except Exception:
        return idx, {}, []


# ============================================================
# C-C 键长松弛：修复 RDKit 嵌入产生的异常键长（>1.7 Å 或 <1.3 Å）
# ============================================================
def _relax_bond_lengths(G, c_coords, h_coords, n_carbons,
                        target=1.54, tol=0.08, max_iters=200):
    """对异常 C-C 键长进行弹簧力迭代松弛

    Args:
        G: 分子图
        c_coords: {node_id: (x,y,z)}
        h_coords: [(x,y,z), ...]
        n_carbons: 碳原子数
        target: 目标 C-C 键长 (Å)
        tol: 容差，键长超出 target±tol 时触发松弛
        max_iters: 最大迭代次数

    Returns:
        (c_coords, h_coords)
    """
    import numpy as np

    c_arr = np.zeros((n_carbons, 3))
    for i in range(n_carbons):
        if i in c_coords:
            c_arr[i] = c_coords[i]

    h_list = [list(h) for h in h_coords]

    for iteration in range(max_iters):
        # 查找异常边
        bad_edges = []
        for u, v in G.edges():
            if u >= n_carbons or v >= n_carbons:
                continue
            d = np.linalg.norm(c_arr[u] - c_arr[v])
            if d > target + tol or d < target - tol:
                bad_edges.append((u, v, d))

        if not bad_edges:
            break

        # 弹簧力：方向 from u to v，大小正比于 (d - target)
        # 力常数随迭代衰减
        k = max(0.1, 1.0 - iteration / max_iters * 0.9)
        forces = np.zeros((n_carbons, 3))

        for u, v, d in bad_edges:
            direction = c_arr[v] - c_arr[u]
            if d < 1e-8:
                direction = np.random.randn(3) * 0.01
                d = 1e-8
            force = direction / d * (d - target) * k * 0.5
            forces[u] += force
            forces[v] -= force

        # 更新碳坐标
        for i in range(n_carbons):
            if i in c_coords:
                c_arr[i] += forces[i]

    # 更新氢坐标（跟随最近碳移动）
    h_result = []
    for h_pos in h_list:
        h_arr = np.array(h_pos)
        # 找最近碳
        min_d = float('inf')
        nearest = 0
        for ci in range(n_carbons):
            if ci in c_coords:
                d = np.linalg.norm(h_arr - c_arr[ci])
                if d < min_d:
                    min_d = d
                    nearest = ci
        # 保持 C-H 键长 1.09 Å
        vec = h_arr - c_arr[nearest]
        dist = np.linalg.norm(vec)
        if dist > 1e-8:
            h_arr = c_arr[nearest] + vec / dist * 1.09
        h_result.append(tuple(float(x) for x in h_arr))

    # 还原 c_coords 字典
    c_result = {}
    for i in range(n_carbons):
        if i in c_coords:
            c_result[i] = (float(c_arr[i, 0]), float(c_arr[i, 1]), float(c_arr[i, 2]))

    return c_result, h_result


def _rdkit_coords_single(G):
    """RDKit构建全单键分子并生成3D坐标

    对大分子（≥13碳）使用多种子策略，选择最大C-C键长最短的构象，
    避免单种子陷入局部最小值导致的断键（C-C > 1.7 Å 被 GaussView 判定为断键）。
    """
    import networkx as nx
    import math
    from rdkit import Chem
    from rdkit.Chem import AllChem, BondType
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    n_carbons = G.number_of_nodes()
    mol = Chem.RWMol()
    for i in range(n_carbons):
        mol.AddAtom(Chem.Atom('C'))

    added = set()
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key not in added:
            added.add(key)
            mol.AddBond(u, v, BondType.SINGLE)

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)

    # 大分子（≥13碳）或高分支：多种子策略
    if n_carbons >= 13:
        best_mol = None
        best_max_bond = float('inf')
        n_seeds = min(30, max(10, n_carbons))  # 随碳数增加种子数
        for seed in range(n_seeds):
            mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            res = AllChem.EmbedMolecule(mol_trial, params)
            if res == -1:
                res = AllChem.EmbedMolecule(mol_trial, randomSeed=seed,
                                            useRandomCoords=True)
                if res == -1:
                    continue
            try:
                AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
            except Exception:
                pass
            conf = mol_trial.GetConformer()
            max_bond = 0.0
            for u, v in G.edges():
                p1 = conf.GetAtomPosition(u)
                p2 = conf.GetAtomPosition(v)
                d = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                              + (p1.z - p2.z) ** 2)
                max_bond = max(max_bond, d)
            if max_bond < best_max_bond:
                best_max_bond = max_bond
                best_mol = Chem.Mol(mol_trial.ToBinary())
        if best_mol is not None:
            mol = best_mol
        else:
            # 所有种子都失败，回退单种子
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        # 小分子：单种子即可
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    c_coords, h_coords = _extract_coords(mol)
    # 键长松弛：修复 RDKit 产生的异常 C-C 键长
    if c_coords:
        c_coords, h_coords = _relax_bond_lengths(
            G, c_coords, h_coords, n_carbons)
    return c_coords, h_coords


def _rdkit_coords_with_bonds(G, double_only=True):
    """RDKit构建含多重键分子并生成3D坐标（含回退方案）"""
    import networkx as nx
    import numpy as np
    import math
    from rdkit import Chem
    from rdkit.Chem import AllChem, BondType
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    n_carbons = G.number_of_nodes()
    double_bonds = set()
    triple_bonds = set()

    for u, v, data in G.edges(data=True):
        bt = data.get('bond_type', 'single')
        key = (min(u, v), max(u, v))
        if bt == 'double':
            double_bonds.add(key)
        elif bt == 'triple':
            triple_bonds.add(key)

    mol = Chem.RWMol()
    for i in range(n_carbons):
        mol.AddAtom(Chem.Atom('C'))

    added = set()
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key in added:
            continue
        added.add(key)
        if key in triple_bonds:
            mol.AddBond(u, v, BondType.TRIPLE)
        elif key in double_bonds:
            mol.AddBond(u, v, BondType.DOUBLE)
        else:
            mol.AddBond(u, v, BondType.SINGLE)

    # 设置显式氢
    for node in sorted(G.nodes()):
        bond_load = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            if bt == 'double':
                bond_load += 2
            elif bt == 'triple':
                bond_load += 3
            else:
                bond_load += 1
        h_count = 4 - bond_load
        if h_count > 0:
            mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # SanitizeMol 失败：回退到基于图的2D坐标
        return _fallback_coords_from_graph(G)

    mol = Chem.AddHs(mol)

    # 检测是否含环（含环分子使用多种子策略避免高张力构象）
    has_ring = len(nx.cycle_basis(G)) > 0

    if has_ring:
        # 含环分子：多种子选择策略，选择最大C-C键长最短的构象
        best_mol = None
        best_max_bond = float('inf')
        for seed in range(50):
            mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            result = AllChem.EmbedMolecule(mol_trial, params)
            if result == -1:
                result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                if result2 == -1:
                    continue
            try:
                AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
            except Exception:
                pass
            # 计算最大C-C键长
            conf = mol_trial.GetConformer()
            max_bond = 0.0
            for u, v in G.edges():
                if u < n_carbons and v < n_carbons:
                    p1 = conf.GetAtomPosition(u)
                    p2 = conf.GetAtomPosition(v)
                    dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
                    max_bond = max(max_bond, dist)
            if max_bond < best_max_bond:
                best_max_bond = max_bond
                best_mol = Chem.Mol(mol_trial.ToBinary())
        if best_mol is not None:
            mol = best_mol
        else:
            # 所有种子都失败，回退到2D坐标
            AllChem.Compute2DCoords(mol)
    else:
        # 非环分子：大分子使用多种子策略，避免局部最小值导致的断键
        if n_carbons >= 13:
            best_mol = None
            best_max_bond = float('inf')
            n_seeds = min(30, max(10, n_carbons))
            for seed in range(n_seeds):
                mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                res = AllChem.EmbedMolecule(mol_trial, params)
                if res == -1:
                    res = AllChem.EmbedMolecule(mol_trial, randomSeed=seed,
                                                useRandomCoords=True)
                    if res == -1:
                        continue
                try:
                    AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                except Exception:
                    pass
                conf = mol_trial.GetConformer()
                max_bond = 0.0
                for u, v in G.edges():
                    if u < n_carbons and v < n_carbons:
                        p1 = conf.GetAtomPosition(u)
                        p2 = conf.GetAtomPosition(v)
                        d = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                                      + (p1.z - p2.z) ** 2)
                        max_bond = max(max_bond, d)
                if max_bond < best_max_bond:
                    best_max_bond = max_bond
                    best_mol = Chem.Mol(mol_trial.ToBinary())
            if best_mol is not None:
                mol = best_mol
            else:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            # 小分子：单种子3D嵌入
            embed_ok = False
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != -1:
                embed_ok = True
            else:
                result2 = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                if result2 != -1:
                    embed_ok = True
            if not embed_ok:
                AllChem.Compute2DCoords(mol)
            else:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                except Exception:
                    pass

    c_coords, h_coords = _extract_coords(mol)

    # 验证坐标有效性（非全零）
    if c_coords and all(abs(c[0]) < 0.001 and abs(c[1]) < 0.001 and abs(c[2]) < 0.001
                        for c in c_coords.values()):
        return _fallback_coords_from_graph(G)

    # 修正累积双键段(C=C=C)的共线问题
    if double_bonds:
        c_coords = _fix_cumulated_diene_coords_standalone(G, c_coords)

    # 键长松弛：修复 RDKit 产生的异常 C-C 键长
    if c_coords:
        c_coords, h_coords = _relax_bond_lengths(
            G, c_coords, h_coords, n_carbons)

    return c_coords, h_coords


def _fallback_coords_from_graph(G):
    """当RDKit嵌入失败时，基于图结构生成回退2D坐标"""
    import networkx as nx
    import math

    n_carbons = G.number_of_nodes()
    c_coords = {}

    # 检测环结构
    cycles = nx.cycle_basis(G)
    cycle_nodes = set()
    main_cycle = cycles[0] if cycles else []

    if main_cycle:
        cycle_nodes = set(main_cycle)
        # 环上节点：等间距放置在圆上
        ring_len = len(main_cycle)
        for idx, node in enumerate(main_cycle):
            angle = 2 * math.pi * idx / ring_len
            c_coords[node] = (1.5 * math.cos(angle), 1.5 * math.sin(angle), 0.0)

    # 非环节点：BFS 从环上节点扩展
    visited = set(c_coords.keys())
    queue = list(visited)

    while queue:
        current = queue.pop(0)
        if current not in c_coords:
            continue
        cx, cy, cz = c_coords[current]
        unplaced_neighbors = [nb for nb in G.neighbors(current) if nb not in visited]

        for j, nb in enumerate(unplaced_neighbors):
            # 从父节点向外延伸
            parent_angle = math.atan2(cy, cx) if abs(cx) > 0.001 or abs(cy) > 0.001 else 0
            spread = math.pi / 3  # 60度间隔
            branch_angle = parent_angle + math.pi + (j - len(unplaced_neighbors) / 2 + 0.5) * spread
            nx_pos = (cx + 1.54 * math.cos(branch_angle),
                      cy + 1.54 * math.sin(branch_angle),
                      0.0)
            c_coords[nb] = nx_pos
            visited.add(nb)
            queue.append(nb)

    # 处理没有环的分子或遗漏的节点
    for node in sorted(G.nodes()):
        if node not in c_coords:
            angle = 2 * math.pi * node / n_carbons
            c_coords[node] = (1.5 * math.cos(angle), 1.5 * math.sin(angle), 0.0)

    # 生成氢原子坐标
    h_coords = []
    for node in sorted(G.nodes()):
        bond_load = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            bond_load += 2 if bt == 'double' else 1
        h_count = 4 - bond_load
        if h_count > 0 and node in c_coords:
            cx, cy, cz = c_coords[node]
            for j in range(h_count):
                ha = 2 * math.pi * j / max(h_count, 1) + node * 0.5
                h_coords.append((cx + 1.09 * math.cos(ha), cy + 1.09 * math.sin(ha), cz))

    return c_coords, h_coords


def _extract_coords(mol):
    """从RDKit分子对象提取碳/氢坐标"""
    from rdkit import Chem
    conf = mol.GetConformer()
    c_coords = {}
    h_coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coord = (float(pos.x), float(pos.y), float(pos.z))
        if atom.GetSymbol() == 'C':
            c_coords[atom.GetIdx()] = coord
        else:
            h_coords.append(coord)
    return c_coords, h_coords


def _fix_cumulated_diene_coords_standalone(G, c_coords):
    """修正累积双键段(C=C=C)的共线坐标问题（独立函数，不依赖self）"""
    import numpy as np

    if not c_coords or len(c_coords) < 3:
        return c_coords

    cumulated_nodes = set()
    for node in G.nodes():
        double_count = sum(
            1 for nb in G.neighbors(node)
            if G[node][nb].get('bond_type', 'single') == 'double'
        )
        if double_count >= 2:
            cumulated_nodes.add(node)

    if not cumulated_nodes:
        return c_coords

    for iteration in range(5):
        still_colinear = False
        for node in cumulated_nodes:
            if node not in c_coords:
                continue
            db_nbs = [nb for nb in G.neighbors(node) if G[node][nb].get('bond_type', 'single') == 'double']
            if len(db_nbs) < 2:
                continue
            pb = np.array(c_coords[node])
            for i_idx in range(len(db_nbs)):
                for j_idx in range(i_idx + 1, len(db_nbs)):
                    a, c = db_nbs[i_idx], db_nbs[j_idx]
                    if a not in c_coords or c not in c_coords:
                        continue
                    pa = np.array(c_coords[a])
                    pc = np.array(c_coords[c])
                    v1 = pa - pb
                    v2 = pc - pb
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 < 1e-8 or n2 < 1e-8:
                        continue
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_a))
                    if angle > 170.0:
                        still_colinear = True
                        normal = np.cross(v1, v2)
                        nn = np.linalg.norm(normal)
                        if nn < 1e-8:
                            if abs(v1[0]) < abs(v1[1]):
                                normal = np.cross(v1, [1, 0, 0])
                            else:
                                normal = np.cross(v1, [0, 1, 0])
                            nn = np.linalg.norm(normal)
                        if nn > 1e-8:
                            normal = normal / nn
                            avg_bond = (n1 + n2) / 2
                            offset_dist = avg_bond * 0.50
                            new_pb = pb + normal * offset_dist
                            c_coords[node] = (float(new_pb[0]), float(new_pb[1]), float(new_pb[2]))
                            pb = new_pb
        if not still_colinear:
            break

    return c_coords


# ============================================================
# Tooltip：悬停提示小窗口
# ============================================================
class CreateToolTip:
    """为 tkinter 控件添加鼠标悬停提示"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffcc",
                         foreground="#333333", font=('Microsoft YaHei', 10),
                         relief=tk.SOLID, borderwidth=1, padx=6, pady=2)
        label.pack()

    def _hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class MoleculeApp:
    """分子异构体可视化主程序"""

    # 内部编码数据，请勿修改
    _B64_SHOKO_CN = B64_SHOKO_CN
    _B64_SHOKO_EN = B64_SHOKO_EN

    def __init__(self, root):
        self.root = root
        self.root.title("分子异构体生成及可视化")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # 窗口状态标志
        self.window_closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # 初始化各生成器
        self._init_generators()

        # 当前状态
        self.current_isomers = []
        self.current_molecule_type = None
        self.current_carbon_count = None
        self._filter_data = []  # 存储筛选数据 [(主链长度, 索引), ...]

        # 生成控制
        self.is_generating = False
        self.cancel_requested = False

        # 创建界面
        self._create_widgets()

        # 配置样式
        self._configure_styles()

    def _show_usage(self):
        """弹窗显示使用说明"""
        win = tk.Toplevel(self.root)
        win.title("使用说明")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # 内容框架
        frame = ttk.Frame(win, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(frame, text="使用说明",
                  font=('Microsoft YaHei', 14, 'bold'),
                  foreground='#1a3a5c').pack(anchor=tk.W, pady=(0, 5))

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))

        # 正文（固定宽度，自动换行）
        usage_lines = [
            ("bold", "1.  整体功能说明："),
            ("normal", "本工具用于生成烃类异构体并进行 3D 交互可视化。"),
            ("", ""),
            ("bold", "2.  使用方法"),
            ("bold", "2.1  输入分子式"),
            ("normal", '在\u201c分子式\u201d输入框中键入任意碳氢化合物的分子式（如CH4、C5H12、C4H8、C6H6，不区分大小写）'),
            ("bold", "2.2  选择类型"),
            ("normal", '选择\u201c全部\u201d或指定单一类别'),
            ("bold", "2.3  生成异构体"),
            ("normal", "点击【生成异构体】按钮，程序将枚举所有符合化学约束的结构，并显示在列表中"),
            ("bold", "2.4  查看结构"),
            ("normal", "单击列表中的某个异构体，右侧会展示其键线式（2D 结构图）信息"),
            ("bold", "2.5  可视化当前选中异构体"),
            ("normal", "点击【可视化当前分子】按钮，即可在新窗口中查看该分子的 3D 交互模型（支持旋转）"),
            ("bold", "2.6  另存为"),
            ("normal", "可以将选中的分子导出为 Gaussian 输入文件（.gjf）"),
            ("bold", "2.7  快捷键与操作技巧"),
            ("normal", "• Ctrl + 单击 — 多选列表中的异构体"),
            ("normal", "• Shift + 单击 — 范围选择"),
            ("normal", '• 右键单击键线式图片 — 弹出菜单，可选择\u201c复制图像\u201d或\u201c另存为...\u201d'),
            ("", ""),
            ("bold", "3.  开发者说明和引用说明"),
            ("normal", "开发者：张曾继明，甘利华"),
            ("normal", "联系方式：2678605701@qq.com ; ganlh@swu.edu.cn"),
            ("normal", "技术栈：Python 3.11，Tkinter，RDKit，Py3Dmol"),
            ("normal", "许可协议：MIT License"),
            ("normal", "引用说明：若本工具对您的研究或工作有帮助，请引用："),
            ("normal", "张曾继明，甘利华. (2026). Hydrocarbon Generator (v2.0) [计算机软件]. Zenodo. https://doi.org/10.5281/zenodo.20844657"),
            ("normal", "源代码地址：https://github.com/MashiroMing/Hydrocarbon_Generator"),
        ]

        for style_tag, line in usage_lines:
            if not line:
                # 空行作为间距
                spacer = ttk.Frame(frame, height=6)
                spacer.pack(fill=tk.X)
                continue
            kwargs = {
                'text': line,
                'wraplength': 500,
                'anchor': tk.W,
                'justify': tk.LEFT,
                'font': ('Microsoft YaHei', 11),
            }
            if style_tag == "bold":
                kwargs.update({'font': ('Microsoft YaHei', 11, 'bold'),
                               'foreground': '#2a2a2a'})
            ttk.Label(frame, **kwargs).pack(fill=tk.X, pady=1)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(10, 10))

        # 关闭按钮
        ttk.Button(frame, text="关闭", style='Action.TButton',
                   command=win.destroy).pack(pady=(0, 5))

        # 居中于主窗口
        win.update_idletasks()
        w = win.winfo_width()
        h = win.winfo_height()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        x = rx + (rw - w) // 2
        y = ry + (rh - h) // 2
        win.geometry(f"+{x}+{y}")

        # 关闭时释放 grab
        def _on_close():
            win.grab_release()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

    # ================================================================
    # 彩蛋
    # ================================================================
    def _show_easter_egg(self, b64_data: str):
        """显示彩蛋图片（Base64 解码，无外部文件依赖）"""
        import base64
        import io
        try:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
            photo = ImageTk.PhotoImage(img)

            win = tk.Toplevel(self.root)
            win.title("🥚")
            win.resizable(False, False)
            win.transient(self.root)

            label = ttk.Label(win, image=photo)
            label.image = photo
            label.pack(padx=10, pady=10)

            close_btn = ttk.Button(win, text="关闭", command=win.destroy)
            close_btn.pack(pady=(0, 10))
            win.bind("<Escape>", lambda e: win.destroy())

            win.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - win.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - win.winfo_height()) // 2
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass  # 彩蛋静默失败，不影响正常使用

    def _on_window_close(self):
        """窗口关闭处理"""
        self.window_closed = True
        self.cancel_requested = True
        try:
            plt.close('all')  # 关闭所有 matplotlib 窗口
        except:
            pass
        self.root.destroy()

    def _init_generators(self):
        """初始化各分子生成器（委托给 GeneratorManager）"""
        self.gen_mgr = GeneratorManager()
        # 烷烃专用生成器（独立于统一生成器）
        self.alkane_generator = self.gen_mgr.alkane_generator
        self.alkane_visualizer = self.gen_mgr.alkane_visualizer
        self.alkene_visualizer = self.gen_mgr.alkene_visualizer
        # 统一生成器（所有非烷烃类型）
        self.unified_generator = self.gen_mgr.unified_generator
        self.generators = self.gen_mgr.generators

        # 兼容属性：从统一生成器内部获取子模块（用于可视化/坐标计算等）
        self.multcycloalkane_generator = self.unified_generator.ring_gen
        self.cyclopolyene_generator = self.unified_generator.cyclopolyene_gen
        self.diene_generator = self.unified_generator.polyenyne_gen

        # 炔烃可视化器（独立实例，用于坐标计算和可视化）
        from original_programs.alkyne import AlkyneIsomerVisualizer
        self.alkyne_visualizer = AlkyneIsomerVisualizer()

    def _configure_styles(self):
        """配置界面样式"""
        style = ttk.Style()
        style.configure('TLabel', font=('Microsoft YaHei', 12))
        style.configure('TLabelframe.Label', font=('Microsoft YaHei', 12))
        style.configure('Title.TLabel', font=('Microsoft YaHei', 14, 'bold'))
        style.configure('Subtitle.TLabel', font=('Microsoft YaHei', 12))
        style.configure('Info.TLabel', font=('Microsoft YaHei', 12))
        style.configure('Action.TButton', font=('Microsoft YaHei', 12), padding=10)

    def _create_widgets(self):
        """创建界面组件"""
        # 主容器（使用 grid 确保底部按钮始终可见）
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(2, weight=1)  # 只有结果区域可伸缩
        main_frame.columnconfigure(0, weight=1)

        # ===== 标题区域 =====
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        # 三列布局：左弹性 | 标题居中 | 按钮右对齐
        title_frame.columnconfigure(0, weight=1)
        title_frame.columnconfigure(2, weight=1)

        ttk.Label(
            title_frame,
            text="生成并可视化烃类的同分异构体",
            style='Title.TLabel'
        ).grid(row=0, column=1)

        # 使用说明按钮（右侧）
        self.help_btn = ttk.Button(
            title_frame, text="\u2753", width=3,
            style='Action.TButton',
            command=self._show_usage
        )
        self.help_btn.grid(row=0, column=2, sticky='e', padx=(10, 0))
        # 悬停提示
        self._help_tooltip = CreateToolTip(self.help_btn, "使用说明")

        # ===== 输入区域 =====
        input_frame = ttk.LabelFrame(main_frame, text="分子输入", padding="10")
        input_frame.grid(row=1, column=0, sticky="ew", pady=10)

        # 分子式输入行
        formula_frame = ttk.Frame(input_frame)
        formula_frame.pack(fill=tk.X, pady=5)

        ttk.Label(formula_frame, text="分子式:").pack(side=tk.LEFT, padx=5)
        self.molecule_input = tk.StringVar(value='')
        self.formula_entry = ttk.Entry(
            formula_frame,
            textvariable=self.molecule_input,
            width=20,
            font=('Times New Roman', 12)
        )
        self.formula_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(formula_frame, text='(例: CH4, C4H8, C5H6, C6H6, C10H20, C15H32)', style='Info.TLabel').pack(side=tk.LEFT, padx=5)

        # 识别结果显示行
        self.parsed_info_var = tk.StringVar(value="")
        self.parsed_info_label = ttk.Label(input_frame, textvariable=self.parsed_info_var, foreground='gray', style='Info.TLabel')
        self.parsed_info_label.pack(fill=tk.X, pady=(2, 5), padx=5)

        # 分子类型筛选（用于同分异构情况）
        filter_type_frame = ttk.Frame(input_frame)
        filter_type_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_type_frame, text="类型筛选:").pack(side=tk.LEFT, padx=5)
        self.type_filter_var = tk.StringVar(value='all')
        self.type_filter_frame = filter_type_frame  # 保存引用以便动态更新
        self.type_filter_radio_frame = ttk.Frame(filter_type_frame)
        self.type_filter_radio_frame.pack(side=tk.LEFT, padx=5)

        # 初始隐藏筛选
        self._update_type_filter_visibility()

        # 绑定分子式输入变化事件
        self.molecule_input.trace_add('write', self._on_molecule_input_changed)

        # 生成按钮和进度条
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.generate_btn = ttk.Button(
            btn_frame,
            text="生成异构体",
            style='Action.TButton',
            command=self._generate_isomers
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = ttk.Button(
            btn_frame,
            text="取消",
            style='Action.TButton',
            command=self._cancel_generation,
            state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        # 进度条
        progress_frame = ttk.Frame(btn_frame)
        progress_frame.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5)

        self.progress_label = ttk.Label(progress_frame, text="就绪", style='Info.TLabel')
        self.progress_label.pack(fill=tk.X, padx=5)

        # ===== 结果显示区域 =====
        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        result_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(1, weight=1)
        result_frame.rowconfigure(0, weight=1)

        # 左侧：异构体列表
        list_frame = ttk.LabelFrame(result_frame, text="异构体列表", padding="5")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        # 筛选区域
        filter_frame = ttk.Frame(list_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        self.filter_var = tk.StringVar(value="全部")
        self.filter_label = ttk.Label(filter_frame, text="筛选:")
        self.filter_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_combo = ttk.Combobox(
            filter_frame, 
            textvariable=self.filter_var,
            values=["全部"],
            state='readonly',
            width=10
        )
        self.filter_combo.pack(side=tk.LEFT)
        self.filter_combo.bind('<<ComboboxSelected>>', self._on_filter_changed)
        
        # 筛选结果标签
        self.filter_result_var = tk.StringVar(value="")
        ttk.Label(filter_frame, textvariable=self.filter_result_var, foreground='blue').pack(side=tk.LEFT, padx=(10, 0))

        # 列表容器：Listbox + Scrollbar 作为一个整体
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)

        list_scroll = ttk.Scrollbar(list_container)
        list_scroll.grid(row=0, column=1, sticky="ns")

        self.isomer_listbox = tk.Listbox(
            list_container,
            yscrollcommand=list_scroll.set,
            font=('Times New Roman', 12),
            selectmode=tk.EXTENDED
        )
        self.isomer_listbox.grid(row=0, column=0, sticky="nsew")
        self.isomer_listbox.bind('<<ListboxSelect>>', self._on_isomer_selected)
        list_scroll.config(command=self.isomer_listbox.yview)

        # 右侧：信息显示
        info_frame = ttk.LabelFrame(result_frame, text="分子信息", padding="5")
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))

        # 占位行，与左侧筛选行等高，使浅色内容区顶部对齐
        info_spacer = ttk.Frame(info_frame, height=24)
        info_spacer.pack(fill=tk.X, pady=(0, 5))
        info_spacer.pack_propagate(False)

        # 键线式图片区域
        self._formula_canvas = tk.Canvas(
            info_frame,
            height=200,
            bg='white',
            highlightthickness=1,
            highlightbackground='#cccccc',
        )
        self._formula_canvas.pack(fill=tk.X, pady=(0, 5))
        self._formula_photo = None  # 保持对 PhotoImage 的引用，防止被 GC
        self._skeletal_png_data = None  # 原始 PNG 字节

        # 右键菜单：复制图像 / 另存为
        self._canvas_context_menu = tk.Menu(self._formula_canvas, tearoff=0)
        self._canvas_context_menu.add_command(label="复制图像", command=self._on_copy_skeletal_image)
        self._canvas_context_menu.add_command(label="另存为...", command=self._on_save_skeletal_image)
        self._formula_canvas.bind("<Button-3>", self._on_formula_canvas_right_click)
        # macOS 兼容
        self._formula_canvas.bind("<Button-2>", self._on_formula_canvas_right_click)

        # 文本信息区域
        self.info_text = scrolledtext.ScrolledText(
            info_frame,
            width=40,
            font=('Microsoft YaHei', 12),
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # ===== 可视化按钮区域 =====
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=3, column=0, sticky="ew", pady=10)

        self.visualize_btn = ttk.Button(
            viz_frame,
            text="可视化当前选中异构体",
            style='Action.TButton',
            command=self._visualize_current,
            state=tk.DISABLED
        )
        self.visualize_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            viz_frame,
            text="另存为...",
            style='Action.TButton',
            command=self._save_as,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # ===== 状态栏 =====
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_type_changed(self):
        """分子类型改变时的处理（已弃用，保留兼容）"""
        pass

    def _parse_molecule_input(self, formula_str):
        """解析分子式输入（委托给 utils.parse_molecule_input）

        返回: (mol_types, n_carbon, error_msg)
            mol_types: 分子类型列表，如 ['alkene', 'cycloalkane'] 或 ['alkane']
            n_carbon: 碳原子数
            error_msg: 错误信息，无错误时为 None
        """
        mol_types, n, m, error = parse_molecule_input(formula_str)
        return mol_types, n, error

    def _update_type_filter_visibility(self, mol_types=None):
        """更新类型筛选区域的显示"""
        # 清除旧的 radiobutton
        for widget in self.type_filter_radio_frame.winfo_children():
            widget.destroy()
        
        # 检查是否需要显示苯环筛选（不饱和度 k ≥ 4）
        show_benzene = False
        if mol_types:
            import re
            formula_str = self.molecule_input.get().strip()
            match = re.match(r'^C(\d*)H(\d+)$', formula_str, re.IGNORECASE)
            if match:
                n = int(match.group(1)) if match.group(1) else 1
                m = int(match.group(2))
                k = (2 * n + 2 - m) // 2
                if k >= 4:
                    show_benzene = True

        if (mol_types and len(mol_types) > 1) or show_benzene:
            # 有同分异构或可选苯环筛选
            self.type_filter_var.set('all')
            
            mol_names = MOL_NAMES_FILTER
            
            ttk.Radiobutton(
                self.type_filter_radio_frame,
                text="全部",
                variable=self.type_filter_var,
                value='all'
            ).pack(side=tk.LEFT, padx=8)
            
            for mt in mol_types:
                ttk.Radiobutton(
                    self.type_filter_radio_frame,
                    text=mol_names.get(mt, mt),
                    variable=self.type_filter_var,
                    value=mt
                ).pack(side=tk.LEFT, padx=8)
            
            # 苯环筛选（仅在不饱和度足够时显示）
            if show_benzene:
                ttk.Radiobutton(
                    self.type_filter_radio_frame,
                    text="含苯环",
                    variable=self.type_filter_var,
                    value='benzene'
                ).pack(side=tk.LEFT, padx=8)
            
            self.type_filter_frame.pack(fill=tk.X, pady=5)
        else:
            # 无需筛选，隐藏
            self.type_filter_frame.pack_forget()

    def _on_molecule_input_changed(self, *args):
        """分子式输入变化时的实时提示"""
        try:
            formula_str = self.molecule_input.get()
            mol_types, n, error = self._parse_molecule_input(formula_str)
            
            if error:
                self.parsed_info_var.set(f"⚠ {error}")
                self.parsed_info_label.config(foreground='red')
                self._update_type_filter_visibility(None)
            else:
                mol_names = MOL_NAMES_FILTER
                type_str = " / ".join([mol_names.get(t, t) for t in mol_types])
                self.parsed_info_var.set(f"✓ 识别为: {type_str}  (C{n})")
                self.parsed_info_label.config(foreground='green')
                self._update_type_filter_visibility(mol_types)
        except Exception:
            pass

    def _on_carbon_changed(self):
        """碳原子数改变时的处理（已弃用）"""
        pass

    # -----------------------------------------------------------------
    # 苯环检测
    # -----------------------------------------------------------------
    @staticmethod
    def _has_benzene_ring(G):
        """检测图中是否包含凯库勒式苯环（6元环，3条双键与3条单键交替排列）

        Args:
            G: networkx.Graph，边含 bond_type 属性

        Returns:
            bool: 是否至少含一个凯库勒式苯环
        """
        import networkx as nx
        try:
            all_cycles = nx.cycle_basis(G)
        except Exception:
            return False

        for cycle_nodes in all_cycles:
            if len(cycle_nodes) != 6:
                continue

            # 检查这6个节点是否确实构成简单6元环（诱导子图度数为2）
            sub = G.subgraph(cycle_nodes)
            if sub.number_of_edges() != 6:
                continue
            if not all(d == 2 for _, d in sub.degree()):
                continue

            # 遍历子图获取环序（nx.find_cycle 返回环序边列表）
            try:
                edges_in_order = nx.find_cycle(sub, orientation='ignore')
            except nx.NetworkXNoCycle:
                continue

            if len(edges_in_order) != 6:
                continue

            # 收集按环序排列的键类型
            bond_types = []
            for u, v, _ in edges_in_order:
                bt = sub[u][v].get('bond_type', 'single')
                if bt == 'triple':
                    break  # 三键不可能在苯环上
                bond_types.append(bt)
            if len(bond_types) != 6:
                continue

            # 凯库勒式：严格单双交替 [单,双,单,双,单,双] 或 [双,单,双,单,双,单]
            single = 'single'
            double = 'double'
            patterns = [
                [single, double, single, double, single, double],
                [double, single, double, single, double, single],
            ]
            for pat in patterns:
                for shift in range(6):
                    if [bond_types[(shift + i) % 6] for i in range(6)] == pat:
                        return True

        return False

    def _generate_isomers(self):
        """生成异构体"""
        # 彩蛋检测
        raw = self.molecule_input.get().strip()
        if raw == "牧之原翔子":
            self._show_easter_egg(self._B64_SHOKO_CN)
            return
        if raw.lower() == "makinohara shoko":
            self._show_easter_egg(self._B64_SHOKO_EN)
            return

        formula_str = self.molecule_input.get().strip()
        
        # 解析分子式
        mol_types, n_carbon, error = self._parse_molecule_input(formula_str)
        if error:
            messagebox.showwarning("输入错误", error)
            return
        
        # 从分子式提取氢原子数（供 polyene 推导双键数用）
        import re
        match = re.match(r'^C(\d*)H(\d+)$', formula_str, re.IGNORECASE)
        self._current_n_hydrogen = int(match.group(2)) if match else None
        
        # 确定要生成的分子类型
        type_filter = self.type_filter_var.get()
        if type_filter in ('all', 'benzene'):
            # "全部"或"苯环"：先生成全部，苯环由后置筛选
            target_types = mol_types
        else:
            if type_filter in mol_types:
                target_types = [type_filter]
            else:
                messagebox.showwarning("输入错误", f"所选类型与分子式不匹配")
                return

        # 重置状态
        self.is_generating = True
        self.cancel_requested = False

        # 更新界面状态
        self.generate_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="准备生成...")
        self.status_var.set("正在生成异构体...")

        # 清空列表
        self.isomer_listbox.delete(0, tk.END)
        self._formula_canvas.delete("all")
        self._formula_photo = None
        self._show_skeletal_placeholder("生成中...")
        self.info_text.delete(1.0, tk.END)

        # 在后台线程中生成
        thread = threading.Thread(target=self._generate_in_background, args=(target_types, n_carbon, type_filter))
        thread.daemon = True
        thread.start()

    def _cancel_generation(self):
        """取消生成"""
        self.cancel_requested = True
        self.status_var.set("正在取消...")
        self.progress_label.config(text="正在取消...")

    def _validate_input(self, mol_type, n_carbon):
        """验证输入（已弃用，由 _parse_molecule_input 替代）"""
        return True

    def _generate_in_background(self, target_types, n_carbon, type_filter='all'):
        """在后台线程中生成异构体（支持多种分子类型）"""
        try:
            cancelled = False
            all_isomers = []       # [(mol_type, isomer_data), ...]
            type_counts = {}       # {mol_type: count}

            # 检查窗口是否已关闭
            if self.window_closed:
                return

            self._update_progress(5, "准备生成...")

            n_hydrogen = getattr(self, '_current_n_hydrogen', None)

            if type_filter in ('all', 'benzene'):
                # "全部"或"苯环"：使用 generate_all 统一生成
                self._update_progress(10, "统一生成异构体...")
                all_isomers = self.gen_mgr.generate_all(n_carbon, n_hydrogen)
                # 统计各类型数量
                for mol_type, iso in all_isomers:
                    type_counts[mol_type] = type_counts.get(mol_type, 0) + 1
            else:
                # 单类型模式：只生成指定类型
                isomers = self.gen_mgr.generate(type_filter, n_carbon, n_hydrogen)
                type_counts[type_filter] = len(isomers)
                for iso in isomers:
                    all_isomers.append((type_filter, iso))

            # 苯环后置筛选：从全部异构体中筛选含凯库勒苯环的结构
            if type_filter == 'benzene':
                import networkx as nx
                all_isomers = [
                    (mt, iso) for mt, iso in all_isomers
                    if isinstance(iso, nx.Graph) and self._has_benzene_ring(iso)
                ]
                type_counts = {}
                for mol_type, iso in all_isomers:
                    type_counts.setdefault('benzene', 0)
                    type_counts['benzene'] += 1

            self._update_progress(90, "生成完成")

            # 检查是否被取消或窗口已关闭
            if self.cancel_requested or self.window_closed:
                cancelled = True
                all_isomers = []
                print(f"\n生成已取消")
            else:
                total_count = len(all_isomers)
                print(f"\n{'='*50}")
                print(f"生成完成: 共 {total_count} 个异构体")
                for mt, cnt in type_counts.items():
                    print(f"  - {MOL_NAMES_CN.get(mt, mt)}: {cnt} 个")
                print(f"{'='*50}\n")

            # 更新界面
            if not self.window_closed:
                self.root.after(0, self._update_isomer_list, all_isomers, target_types, n_carbon, type_counts, cancelled)

        except Exception as e:
            if not self.window_closed:
                self.root.after(0, self._show_error, str(e))

    def _generate_alkane_with_progress(self, generator, n_carbon):
        """带进度追踪的烷烃生成"""
        import time

        # 分阶段模拟进度
        stages = [
            (10, "生成中...", 0.15),
            (35, "去重处理中...", 0.4),
            (60, "整理结果中...", 0.6),
            (80, "完成准备...", 0.85),
        ]

        isomers = []
        for progress, label, _ in stages:
            if self.cancel_requested or self.window_closed:
                break
            self._update_progress(progress, label)
            time.sleep(0.3)  # 给 UI 更新的时间

        if not self.cancel_requested and not self.window_closed:
            self._update_progress(85, "实际生成异构体...")
            isomers = generator.generate_isomers(n_carbon)
            self._update_progress(100, f"生成完成，共 {len(isomers)} 个")

        return isomers

    def _update_progress(self, value, label):
        """更新进度条（在线程中调用），同时输出到终端"""
        if self.window_closed:
            return
        print(f"[{value:3d}%] {label}")
        sys.stdout.flush()
        self.root.after(0, self._set_progress, value, label)

    def _set_progress(self, value, label):
        """设置进度（在主线程中执行）"""
        if self.window_closed:
            return
        try:
            self.progress_var.set(value)
            self.progress_label.config(text=label)
            self.root.update_idletasks()
        except tk.TclError:
            pass  # 忽略窗口已关闭时的错误

    def _update_isomer_list(self, all_isomers, target_types, n_carbon, type_counts, cancelled=False):
        """更新异构体列表"""
        if self.window_closed:
            return

        try:
            # 重置生成状态
            self.is_generating = False
            self.generate_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)

            # 如果被取消
            if cancelled:
                self.progress_var.set(0)
                self.progress_label.config(text="已取消")
                self.status_var.set("生成已取消")
                return

            # 存储数据: all_isomers = [(mol_type, isomer_data), ...]
            self.current_isomers = all_isomers
            self.current_molecule_type = target_types[0] if len(target_types) == 1 else 'mixed'
            self.current_carbon_count = n_carbon

            # 清空列表
            self.isomer_listbox.delete(0, tk.END)

            # 填充列表
            mol_names = MOL_NAMES_CN
            n_hydrogen = getattr(self, '_current_n_hydrogen', None)
            formulas = {mt: compute_formula(mt, n_carbon, n_hydrogen) for mt in mol_names}

            for i, (mol_type, _) in enumerate(all_isomers):
                self.isomer_listbox.insert(tk.END, f"#{i+1:3d}  {formulas[mol_type]}  {mol_names[mol_type]}")

            # 强制刷新滚动区域（修复非最大化窗口滚动不到底部的问题）
            self.isomer_listbox.update_idletasks()
            self.isomer_listbox.yview_moveto(0)

            # 计算筛选选项（主链长度/环大小）
            self._update_filter_options([iso for _, iso in all_isomers],
                                        target_types[0] if len(target_types) == 1 else 'mixed',
                                        n_carbon)

            # 更新状态
            total_count = len(all_isomers)
            if len(target_types) == 1:
                self.status_var.set(f"已生成 {total_count} 个 {mol_names[target_types[0]]} 异构体")
            else:
                type_detail = ", ".join([f"{mol_names[t]} {type_counts.get(t, 0)}个" for t in target_types])
                self.status_var.set(f"已生成 {total_count} 个异构体 ({type_detail})")

            self.progress_var.set(100)
            self.progress_label.config(text=f"完成，共 {total_count} 个异构体")

            # 更新信息
            self._show_skeletal_placeholder("选择异构体查看键线式")
            self.info_text.delete(1.0, tk.END)
            if len(target_types) == 1:
                info = f"""分子式: {formulas[target_types[0]]}

分子类型: {mol_names[target_types[0]]}

异构体数量: {total_count}

提示:
- 点击列表中的异构体查看详情
- 选择后点击"可视化当前选中异构体"进行3D展示
"""
            else:
                type_lines = "\n".join([f"  - {mol_names[t]}: {type_counts.get(t, 0)} 个" for t in target_types])
                info = f"""分子式: {formulas[target_types[0]]}

包含类型:
{type_lines}

异构体总数: {total_count}

提示:
- 点击列表中的异构体查看详情
- 选择后点击"可视化当前选中异构体"进行3D展示
"""
            self.info_text.insert(1.0, info)

            # 启用按钮
            if len(all_isomers) > 0:
                self.save_btn.config(state=tk.NORMAL)

        except tk.TclError:
            pass  # 窗口已关闭，忽略错误

    def _calculate_main_chain_length(self, isomer_data, mol_type):
        """计算主链长度或环大小"""
        import networkx as nx
        
        G = isomer_data
        
        # 如果是字符串（规范表示），需要转换为图
        if isinstance(isomer_data, str):
            try:
                if mol_type == 'alkene':
                    adj = self.alkene_visualizer.canon_to_adjacency(isomer_data)
                else:
                    adj = self.alkane_generator.canon_to_adjacency(isomer_data)
                G = nx.Graph()
                for node, neighbors in adj.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
            except:
                return 0  # 默认返回0
        
        if mol_type in ('cycloalkane', 'cycloalkene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane'):
            # 环烷烃/单环烯烃：计算环的大小
            # 优化：对于恰好含一个环的图，环长 = 边数 - 节点数 + 1
            # 更快的方式：直接用 cycle_basis，但先尝试用度数<=2的节点剥离法
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            # 对于恰好含一个环的连通图: 环长 = n_edges - n_nodes + 1
            # 但这是 cyclomatic number，不等于环长
            # 使用更高效的方法：逐步剥离去叶节点
            try:
                deg = dict(G.degree())
                remaining = set(G.nodes())
                changed = True
                while changed:
                    changed = False
                    leaves = [n for n in remaining if deg.get(n, 0) <= 1]
                    if leaves:
                        changed = True
                        for leaf in leaves:
                            remaining.discard(leaf)
                            for nb in G.neighbors(leaf):
                                if nb in remaining and deg.get(nb, 0) > 0:
                                    deg[nb] = deg.get(nb, 0) - 1
                return len(remaining) if remaining else n_nodes
            except Exception:
                cycles = nx.cycle_basis(G)
                if cycles:
                    return len(cycles[0])
                return n_nodes
        else:
            # 链状分子：计算最长路径（主链长度）
            if G.number_of_nodes() == 0:
                return 0
            
            # 使用两次BFS找树的直径（最长路径）
            try:
                # 找一个端点（度为1的节点）
                endpoints = [n for n in G.nodes() if G.degree(n) == 1]
                if not endpoints:
                    endpoints = list(G.nodes())
                
                # 从第一个端点开始BFS，找到最远节点
                start = endpoints[0]
                distances = nx.single_source_shortest_path_length(G, start)
                farthest = max(distances, key=distances.get)
                
                # 从最远节点开始BFS，找到最长路径
                distances = nx.single_source_shortest_path_length(G, farthest)
                max_dist = max(distances.values())
                
                # 主链长度 = 路径上的节点数 = 最长距离 + 1
                return max_dist + 1
            except:
                return G.number_of_nodes()

    def _update_filter_options(self, isomers, mol_type, n_carbon):
        """更新筛选选项"""
        import networkx as nx
        
        # 计算每个异构体的主链长度/环大小
        chain_lengths = []
        for isomer in isomers:
            # isomer 可能是元组 (mol_type, data) 或直接是 data
            if isinstance(isomer, tuple) and len(isomer) == 2:
                iso_mol_type, iso_data = isomer
            else:
                iso_mol_type = mol_type
                iso_data = isomer
            length = self._calculate_main_chain_length(iso_data, iso_mol_type if iso_mol_type != 'mixed' else 'alkane')
            chain_lengths.append(length)
        
        # 获取唯一值并排序
        unique_lengths = sorted(set(chain_lengths))
        
        # 更新下拉框选项
        # 根据分子类型确定筛选标签
        if mol_type == 'cycloalkane':
            filter_label = "环大小:"
        elif mol_type == 'cycloalkene':
            filter_label = "环大小:"
        elif mol_type == 'cyclopolyene':
            filter_label = "环大小:"
        elif mol_type == 'multcycloalkane':
            filter_label = "环大小:"
        elif mol_type == 'multcyclomultalkane':
            filter_label = "环大小:"
        elif mol_type == 'mixed':
            # 混合类型：检查是否包含环烷烃
            has_cyclo = any(isinstance(isomer, tuple) and isomer[0] in ('cycloalkane', 'cycloalkene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane') for isomer in isomers)
            has_chain = any(isinstance(isomer, tuple) and isomer[0] in ('alkane', 'alkene', 'alkyne', 'diene', 'alkenyl') for isomer in isomers)
            if has_cyclo and has_chain:
                filter_label = "特征值:"
            elif has_cyclo:
                filter_label = "环大小:"
            else:
                filter_label = "主链长度:"
        else:
            filter_label = "主链长度:"

        self.filter_label.config(text=filter_label)
        self._current_filter_label = filter_label
        
        self.filter_var.set("全部")
        self.filter_combo['values'] = ["全部"] + [str(x) for x in unique_lengths]
        self.filter_result_var.set(f"(共 {len(isomers)} 个)")
        
        # 存储筛选数据
        self._filter_data = list(zip(chain_lengths, list(range(len(isomers)))))

    def _on_filter_changed(self, event):
        """筛选条件改变时的处理"""
        if self.window_closed:
            return
        
        try:
            selected = self.filter_var.get()
            all_isomers = self.current_isomers
            n_carbon = self.current_carbon_count
            
            formulas = {mt: compute_formula(mt, n_carbon, getattr(self, '_current_n_hydrogen', None)) for mt in MOL_NAMES_CN}
            
            mol_names = MOL_NAMES_CN
            
            # 清空并重新填充列表
            self.isomer_listbox.delete(0, tk.END)
            
            if selected == "全部":
                # 显示所有异构体
                for i, item in enumerate(all_isomers):
                    if isinstance(item, tuple) and len(item) == 2:
                        mt, _ = item
                    else:
                        mt = self.current_molecule_type
                    self.isomer_listbox.insert(tk.END, f"#{i+1:3d}  {formulas[mt]}  {mol_names[mt]}")
                self.filter_result_var.set(f"(共 {len(all_isomers)} 个)")
            else:
                # 按筛选条件过滤
                target_length = int(selected)
                filtered_count = 0
                for chain_len, idx in self._filter_data:
                    if chain_len == target_length:
                        item = all_isomers[idx]
                        if isinstance(item, tuple) and len(item) == 2:
                            mt, _ = item
                        else:
                            mt = self.current_molecule_type
                        self.isomer_listbox.insert(tk.END, f"#{idx+1:3d}  {formulas[mt]}  {mol_names[mt]}")
                        filtered_count += 1
                
                # 更新筛选结果标签
                filter_name = getattr(self, '_current_filter_label', '主链长度:').replace(':', '')
                self.filter_result_var.set(f"({filter_name}={target_length}: {filtered_count} 个)")
            
            # 强制刷新滚动区域（修复非最大化窗口滚动不到底部的问题）
            self.isomer_listbox.update_idletasks()
            self.isomer_listbox.yview_moveto(0)
        
        except Exception as e:
            pass  # 忽略错误

    def _on_isomer_selected(self, event):
        """异构体选中事件"""
        if self.window_closed:
            return

        try:
            selection = self.isomer_listbox.curselection()
            if selection:
                self.visualize_btn.config(state=tk.NORMAL)

                list_idx = selection[0]
                
                # 获取列表项的内容来解析原始索引
                item_text = self.isomer_listbox.get(list_idx)
                # 格式: "#001  C5H12  烷烃"
                try:
                    original_idx = int(item_text.split('#')[1].split()[0]) - 1
                except:
                    original_idx = list_idx
                
                mol_type, isomer = self.current_isomers[original_idx]

                # 显示详细信息
                self._show_isomer_details(original_idx, isomer, mol_type)
        except tk.TclError:
            pass  # 窗口已关闭

    def _show_isomer_details(self, idx, isomer_data, mol_type=None):
        """显示异构体详细信息（含键线式图片）"""
        if self.window_closed:
            return

        try:
            if mol_type is None:
                mol_type = self.current_molecule_type
            n_carbon = self.current_carbon_count

            import networkx as nx

            mol_names = MOL_NAMES_FILTER

            formulas = {mt: compute_formula(mt, n_carbon, getattr(self, '_current_n_hydrogen', None)) for mt in mol_names}
            if mol_type == 'alkane':
                # 烷烃返回的是规范字符串
                info = f"""【异构体 #{idx+1} 详细信息】

分子类型: {mol_names[mol_type]}
分子式: {formulas[mol_type]}

结构表示: {isomer_data}

提示: 这是烷烃的规范树表示法
"""
            elif mol_type == 'alkene':
                # 烯烃返回 nx.Graph 对象
                info = self._get_graph_info_text(isomer_data, idx, mol_type, formulas)
            elif mol_type in ['diene', 'cycloalkane', 'cycloalkene', 'alkenyl', 'triene', 'tetraene', 'polyene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane']:
                # 二烯烃、环烷烃和单环烯烃返回 Graph
                if isinstance(isomer_data, str):
                    info = f"""【异构体 #{idx+1} 详细信息】

分子类型: {mol_names[mol_type]}
分子式: {formulas[mol_type]}

结构表示: {isomer_data}
"""
                else:
                    info = self._get_graph_info_text(isomer_data, idx, mol_type, formulas)
            else:
                info = f"异构体 #{idx+1}"

            # 更新文本信息
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)

            # 渲染键线式图片
            self._render_skeletal_image(isomer_data, mol_type)

        except tk.TclError:
            pass  # 窗口已关闭

    def _render_skeletal_image(self, isomer_data, mol_type):
        """渲染并显示分子的键线式图片"""
        # 清除旧图片，释放内存
        self._formula_canvas.delete("all")
        self._formula_photo = None
        self._skeletal_png_data = None  # 原始 PNG 字节（供右键复制/另存用）

        if not HAS_SKELETAL_DRAWING:
            self._show_skeletal_placeholder("键线式渲染不可用\n请安装: pip install cairosvg Pillow")
            return

        # 在后台线程中生成图片（避免阻塞 UI）
        def _render_in_thread():
            try:
                png_data = render_skeletal_formula(
                    isomer_data, mol_type,
                    gen_mgr=self.gen_mgr,
                    img_size=(380, 220)
                )
                if png_data is None:
                    self.root.after(0, self._show_skeletal_placeholder, "键线式渲染不可用")
                    return

                # 保存原始 PNG 字节（供右键复制/另存用）
                self._skeletal_png_data = png_data

                # PNG 字节 → PIL Image（纯内存，无临时文件）
                from PIL import Image
                img = Image.open(io.BytesIO(png_data))
                self.root.after(0, self._display_skeletal_image, img)
            except Exception:
                self.root.after(0, self._show_skeletal_placeholder, "键线式渲染不可用")

        thread = threading.Thread(target=_render_in_thread)
        thread.daemon = True
        thread.start()

    def _display_skeletal_image(self, pil_img):
        """在 Canvas 上显示键线式图片（在主线程中调用）"""
        try:
            if self.window_closed:
                return

            from PIL import Image, ImageTk

            # 获取 Canvas 实际宽度，适配缩放
            canvas_w = self._formula_canvas.winfo_width()
            if canvas_w < 50:
                canvas_w = 380

            # 等比缩放
            w, h = pil_img.size
            scale = min(canvas_w / w, 200 / h, 1.0)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

            self._formula_photo = ImageTk.PhotoImage(pil_img)

            # 居中放置
            self._formula_canvas.delete("all")
            self._formula_canvas.create_image(
                canvas_w // 2, 100,
                image=self._formula_photo,
                anchor=tk.CENTER
            )

        except Exception:
            self._show_skeletal_placeholder("键线式渲染不可用")

    def _on_formula_canvas_right_click(self, event):
        """右键弹出菜单：复制图像 / 另存为"""
        if self._skeletal_png_data is not None:
            try:
                self._canvas_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self._canvas_context_menu.grab_release()

    def _on_copy_skeletal_image(self):
        """将键线式图片复制到剪贴板"""
        if self._skeletal_png_data is None:
            return
        try:
            from PIL import Image
            import io as io_pil
            img = Image.open(io_pil.BytesIO(self._skeletal_png_data))

            # Windows: 通过 Win32 API 复制 PNG 到剪贴板
            import subprocess
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.write(self._skeletal_png_data)
            tmp.close()
            # 使用 PowerShell 复制图像到剪贴板
            subprocess.run([
                'powershell', '-NoProfile', '-Command',
                f'Add-Type -AssemblyName System.Windows.Forms;'
                f'$img = [System.Drawing.Image]::FromFile("{tmp.name}");'
                f'[System.Windows.Forms.Clipboard]::SetImage($img);'
                f'$img.Dispose()'
            ], capture_output=True, timeout=10)
            import os
            os.unlink(tmp.name)
        except Exception:
            pass  # 复制失败时静默忽略

    def _on_save_skeletal_image(self):
        """将键线式图片另存为 PNG 文件"""
        if self._skeletal_png_data is None:
            return
        try:
            import tkinter.filedialog
            file_path = tkinter.filedialog.asksaveasfilename(
                title="保存键线式图片",
                defaultextension=".png",
                filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")]
            )
            if file_path:
                with open(file_path, 'wb') as f:
                    f.write(self._skeletal_png_data)
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存图片: {str(e)}")

    def _show_skeletal_placeholder(self, message="键线式渲染不可用"):
        """显示键线式占位提示"""
        try:
            if self.window_closed:
                return
            self._formula_canvas.delete("all")
            self._formula_canvas.create_text(
                190, 100,
                text=message,
                font=('Microsoft YaHei', 12),
                fill='#999999',
                justify=tk.CENTER
            )
        except tk.TclError:
            pass

    def _get_graph_info_text(self, G, idx, mol_type, formulas):
        """获取图的详细信息文本"""
        import networkx as nx

        mol_names = MOL_NAMES_FILTER

        # 获取图的基本信息
        n_atoms = G.number_of_nodes()
        n_bonds = G.number_of_edges()

        # 统计双键和三键数量
        double_bonds = sum(1 for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double')
        triple_bonds = sum(1 for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple')

        # 获取邻接关系
        adj_list = []
        for node in sorted(G.nodes()):
            neighbors = sorted([n for n in G.neighbors(node)])
            adj_list.append(f"C{node+1}: 连接的碳原子 -> {[f'C{n+1}' for n in neighbors]}")

        # 计算氢原子数
        if mol_type == 'alkane':
            h_count = 2 * n_atoms + 2
        elif mol_type == 'alkyne':
            h_count = 2 * n_atoms - 2
        elif mol_type == 'alkenyl':
            h_count = 2 * n_atoms - 4
        elif mol_type == 'triene':
            h_count = 2 * n_atoms - 4
        elif mol_type == 'tetraene':
            h_count = 2 * n_atoms - 6
        elif mol_type == 'polyene':
            # 从图中直接计算（每个碳4个价键，减去已用键数）
            h_count = sum(4 - sum(2 if G[node][nb].get('bond_type')=='double' else 3 if G[node][nb].get('bond_type')=='triple' else 1 for nb in G.neighbors(node)) for node in G.nodes())
        elif mol_type == 'multcycloalkane':
            # 多环烷烃：每个碳4个价键，减去C-C键数
            h_count = sum(4 - G.degree(node) for node in G.nodes())
        elif mol_type == 'multcyclomultalkane':
            # 多环多烯炔烃：每个碳4个价键，减去键负载
            h_count = sum(4 - sum(2 if G[node][nb].get('bond_type')=='double' else 3 if G[node][nb].get('bond_type')=='triple' else 1 for nb in G.neighbors(node)) for node in G.nodes())
        else:
            h_count = 2 * n_atoms

        info = f"""【异构体 #{idx+1} 详细信息】

分子类型: {mol_names[mol_type]}
分子式: {formulas[mol_type]}

原子统计:
  - 总原子数: {n_atoms + h_count}
  - 碳原子数: {n_atoms}
  - 氢原子数: {h_count}

化学键统计:
  - 总键数: {n_bonds}
  - 双键数: {double_bonds}
  - 三键数: {triple_bonds}

碳原子连接关系:
"""
        for adj in adj_list:
            info += f"  {adj}\n"

        return info

    def _get_mol_display_name(self, mol_type, n_carbon):
        """获取分子类型的英文名和分子式"""
        name = MOL_NAMES_EN.get(mol_type, mol_type)
        formula = compute_formula(mol_type, n_carbon, getattr(self, '_current_n_hydrogen', None))
        return f"{formula} {name}"

    def _generate_hydrogens_for_multi_bond(self, G, c_coords, main_cycle=None):
        """根据图G的bond_type信息生成正确的氢原子坐标

        CycloalkaneBuilder.build_coordinates 不知道双键信息，按全部单键(sp3)
        计算氢原子数，导致含双键的环状分子（如累积双烯）氢数翻倍。
        此方法根据bond_type正确计算每个碳的价键负载，进而得出正确的氢原子数，
        并利用四面体/sp2/sp几何放置氢原子。

        Args:
            G: networkx图对象(含bond_type属性)
            c_coords: 碳原子坐标字典 {node_id: (x, y, z)}
            main_cycle: 主环节点列表(可选，用于确定环法线方向)

        Returns:
            氢原子坐标列表 [(x, y, z), ...]，按碳原子顺序排列
        """
        import numpy as np
        import math

        n_c = G.number_of_nodes()
        ring_set = set(main_cycle) if main_cycle else set()

        # 计算环法线（用于确定氢原子在环的哪一侧）
        ring_normal = np.array([0.0, 0.0, 1.0])
        if len(ring_set) >= 3:
            cycle_list = list(ring_set)
            c0, c1, c2 = cycle_list[0], cycle_list[1], cycle_list[2]
            if c0 in c_coords and c1 in c_coords and c2 in c_coords:
                v1 = np.array(c_coords[c1]) - np.array(c_coords[c0])
                v2 = np.array(c_coords[c2]) - np.array(c_coords[c0])
                normal = np.cross(v1, v2)
                norm_n = np.linalg.norm(normal)
                if norm_n > 1e-6:
                    ring_normal = normal / norm_n

        h_coords = []

        for c_idx in range(n_c):
            if c_idx not in c_coords:
                continue

            coord = np.array(c_coords[c_idx])

            # 计算价键负载（考虑双键贡献2、三键贡献3）
            bond_load = 0
            for nb in G.neighbors(c_idx):
                bt = G[c_idx][nb].get('bond_type', 'single')
                if bt == 'double':
                    bond_load += 2
                elif bt == 'triple':
                    bond_load += 3
                else:
                    bond_load += 1

            h_count = max(0, 4 - bond_load)
            if h_count <= 0:
                continue

            # 收集已有C-C键的方向向量
            cc_vectors = []
            for nb in G.neighbors(c_idx):
                if nb in c_coords:
                    vec = np.array(c_coords[nb]) - coord
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        cc_vectors.append(vec / norm)

            # 判断杂化类型来决定氢原子方向
            num_double_bonds = sum(
                1 for nb in G.neighbors(c_idx)
                if G[c_idx][nb].get('bond_type', 'single') == 'double'
            )

            if num_double_bonds >= 2:
                # sp 杂化碳（累积双烯中心碳，如 C=C=C）：
                # 两个双键方向近似180°，氢原子应与双键平面垂直
                h_dirs = self._sp_hydrogen_dirs(cc_vectors, ring_normal, h_count)
            elif num_double_bonds == 1:
                # sp2 杂化碳：双键+单键约120°平面，氢在平面内第三个方向
                h_dirs = self._sp2_hydrogen_dirs(cc_vectors, ring_normal, h_count)
            else:
                # sp3 杂化碳：四面体几何
                h_dirs = self._sp3_hydrogen_dirs(cc_vectors, ring_normal, h_count, c_idx in ring_set)

            for h_dir in h_dirs:
                h_coord = coord + 1.09 * h_dir
                h_coords.append((float(h_coord[0]), float(h_coord[1]), float(h_coord[2])))

        return h_coords

    def _sp3_hydrogen_dirs(self, cc_vectors, ring_normal, num_h, in_ring=False):
        """sp3碳的氢原子方向（四面体几何，109.47°）"""
        import numpy as np
        import math

        if not cc_vectors:
            sqrt3 = math.sqrt(3)
            dirs = [
                np.array([sqrt3/3, sqrt3/3, sqrt3/3]),
                np.array([sqrt3/3, -sqrt3/3, -sqrt3/3]),
                np.array([-sqrt3/3, sqrt3/3, -sqrt3/3]),
                np.array([-sqrt3/3, -sqrt3/3, sqrt3/3])
            ]
            return dirs[:num_h]

        cos_theta = -1.0 / 3.0  # cos(109.47°)
        sin_theta = math.sqrt(8) / 3.0

        if len(cc_vectors) == 1:
            v1 = cc_vectors[0]
            arb = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp = np.cross(v1, arb)
            perp /= (np.linalg.norm(perp) + 1e-6)
            h_dirs = []
            for i in range(num_h):
                angle = i * 2 * math.pi / num_h
                v_new = cos_theta * v1 + sin_theta * (math.cos(angle) * perp + math.sin(angle) * np.cross(v1, perp))
                h_dirs.append(v_new / (np.linalg.norm(v_new) + 1e-6))
            return h_dirs

        elif len(cc_vectors) == 2:
            v1, v2 = cc_vectors
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-6:
                normal = ring_normal
            else:
                normal /= norm_n

            cos_beta = np.clip(np.dot(v1, v2), -1, 1)
            beta = math.acos(cos_beta)

            # 当两个碳邻居近似反平行（角度>150°，如C3-C5-C6直线排列），
            # 标准四面体计算会失效（z_squared<0），氢方向会落在邻居平面内导致坍缩
            # 这种情况下，氢原子应沿法线方向放置，与每个碳邻居约109.47°
            if beta > math.radians(150):
                # 两个碳邻居近似180°，氢原子沿法线方向放置
                # 两个氢分别在法线的正反方向，与v1/v2约90°（实际取决于法线与v1的夹角）
                h1 = normal
                h2 = -normal
                h_dirs = [h1, h2][:num_h]
                return h_dirs

            u = v1
            w = v2 - np.dot(v2, u) * u
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-6:
                w = np.cross(normal, u)
                w /= (np.linalg.norm(w) + 1e-6)
            else:
                w /= norm_w

            sin_beta = math.sin(beta)
            if abs(sin_beta) < 1e-6:
                sin_beta = 1e-6

            a = cos_theta
            b = (cos_theta - a * cos_beta) / sin_beta

            plane_component = a * u + b * w
            z_squared = 1 - a**2 - b**2

            h_dirs = []
            if z_squared > 0:
                z = math.sqrt(z_squared)
                h1 = plane_component + z * normal
                h2 = plane_component - z * normal
            else:
                # z_squared <= 0: 标准四面体方向超出范围，
                # 使用法线方向偏移来修正
                h1 = plane_component + 0.1 * normal
                h2 = plane_component - 0.1 * normal

            h1_norm = h1 / (np.linalg.norm(h1) + 1e-6)
            h2_norm = h2 / (np.linalg.norm(h2) + 1e-6)

            # 检查氢方向是否与碳邻居方向过于接近（防坍缩）
            for h_candidate in [h1_norm, h2_norm]:
                for cc_v in cc_vectors:
                    cos_a = np.clip(np.dot(h_candidate, cc_v), -1, 1)
                    angle = math.acos(cos_a)
                    if angle < math.radians(60):
                        # 氢方向与碳邻居夹角太小，需要修正
                        # 使用法线方向重新计算
                        h1 = normal
                        h2 = -normal
                        h1_norm = h1
                        h2_norm = h2
                        break
                else:
                    continue
                break

            h_dirs.append(h1_norm)
            if num_h >= 2:
                h_dirs.append(h2_norm)
            return h_dirs[:num_h]

        elif len(cc_vectors) == 3:
            sum_vec = np.sum(cc_vectors, axis=0)
            v4 = -sum_vec
            norm_v4 = np.linalg.norm(v4)
            if norm_v4 > 1e-6:
                h_dir = v4 / norm_v4
            else:
                # 三个方向几乎完美对称（norm≈0），无法确定四面体第四方向
                # 使用邻居平面的法线方向
                h_dir = ring_normal
                return [h_dir]

            # 检查氢方向是否与某个碳邻居方向过于接近
            # 如果所有碳邻居共面（如都在z=0平面），-sum_vec会落在该平面内，
            # 可能指向某个碳邻居方向，导致氢原子与该碳坍缩
            min_angle_to_cc = float('inf')
            nearest_cc = cc_vectors[0]
            for cc_v in cc_vectors:
                cos_a = np.clip(np.dot(h_dir, cc_v), -1, 1)
                angle = math.acos(cos_a)
                if angle < min_angle_to_cc:
                    min_angle_to_cc = angle
                    nearest_cc = cc_v

            # 理想四面体角度是109.47°，如果氢方向与最近碳邻居的夹角 < 80°，
            # 说明方向有问题（氢可能指向碳），需要修正
            if min_angle_to_cc < math.radians(80):
                # 计算碳邻居所在平面的法线
                v1, v2, v3 = cc_vectors[0], cc_vectors[1], cc_vectors[2]
                normal = np.cross(v2 - v1, v3 - v1)
                norm_n = np.linalg.norm(normal)
                if norm_n > 1e-6:
                    normal /= norm_n
                else:
                    normal = ring_normal

                # 精确计算四面体第4个方向：
                # 设 h = a * plane_dir + b * normal，需要 |h|=1 且与每个cc_v夹角≈109.47°
                # plane_dir 是 h 在邻居平面内的投影方向（取-sum_vec的平面分量）
                target_cos = -1.0 / 3.0  # cos(109.47°)
                h_in_plane = h_dir - np.dot(h_dir, normal) * normal
                norm_plane = np.linalg.norm(h_in_plane)
                if norm_plane > 1e-6:
                    plane_dir = h_in_plane / norm_plane
                else:
                    plane_dir = np.cross(normal, cc_vectors[0])
                    norm_pd = np.linalg.norm(plane_dir)
                    if norm_pd > 1e-6:
                        plane_dir /= norm_pd
                    else:
                        plane_dir = np.array([1.0, 0.0, 0.0])

                # cos(angle) = dot(a*plane_dir + b*normal, cc_v)
                # = a * dot(plane_dir, cc_v) + b * dot(normal, cc_v)
                # 对最近的cc_v: cos(109.47°) = a * dot(plane_dir, nearest_cc) + b * dot(normal, nearest_cc)
                # 且 a^2 + b^2 = 1
                d1 = np.dot(plane_dir, nearest_cc)
                d2 = np.dot(normal, nearest_cc)
                # a * d1 + b * d2 = target_cos, a^2 + b^2 = 1
                # 从 a^2 + b^2 = 1: b = sqrt(1 - a^2) (取正，让H偏向法线正方向)
                # a * d1 + sqrt(1 - a^2) * d2 = target_cos
                # 用数值求解
                best_a = 0.0
                best_err = float('inf')
                for a_try in np.linspace(-1, 1, 1000):
                    b_sq = 1.0 - a_try ** 2
                    if b_sq < 0:
                        continue
                    b_try = math.sqrt(b_sq)
                    predicted_cos = a_try * d1 + b_try * d2
                    err = abs(predicted_cos - target_cos)
                    if err < best_err:
                        best_err = err
                        best_a = a_try

                best_b = math.sqrt(max(0, 1.0 - best_a ** 2))
                h_dir = best_a * plane_dir + best_b * normal
                h_dir /= (np.linalg.norm(h_dir) + 1e-10)

                # 验证所有cc角度
                valid = True
                for cc_v in cc_vectors:
                    cos_a = np.clip(np.dot(h_dir, cc_v), -1, 1)
                    angle = math.acos(cos_a)
                    if angle < math.radians(70):
                        valid = False
                        break
                if not valid:
                    # 回退：使用法线方向
                    h_dir = normal / (np.linalg.norm(normal) + 1e-10)

            return [h_dir]

        return []

    def _sp2_hydrogen_dirs(self, cc_vectors, ring_normal, num_h):
        """sp2碳的氢原子方向（平面三角几何，120°）"""
        import numpy as np
        import math

        if not cc_vectors:
            dirs = [np.array([1.0, 0.0, 0.0])]
            if num_h >= 2:
                dirs.append(np.array([-0.5, math.sqrt(3)/2, 0.0]))
            return dirs[:num_h]

        if len(cc_vectors) == 1:
            v1 = cc_vectors[0]
            # 氢原子方向与v1成120°，在v1和ring_normal定义的平面内
            arb = ring_normal if np.linalg.norm(np.cross(v1, ring_normal)) > 1e-6 else np.array([0.0, 0.0, 1.0])
            perp = np.cross(v1, arb)
            norm_p = np.linalg.norm(perp)
            if norm_p < 1e-6:
                arb = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                perp = np.cross(v1, arb)
                norm_p = np.linalg.norm(perp)
            perp /= (norm_p + 1e-6)
            # sp2: 120° from v1 in the plane
            h_dir = -0.5 * v1 + math.sqrt(3)/2 * perp
            h_dir /= (np.linalg.norm(h_dir) + 1e-6)
            if num_h == 1:
                return [h_dir]
            h_dir2 = -0.5 * v1 - math.sqrt(3)/2 * perp
            h_dir2 /= (np.linalg.norm(h_dir2) + 1e-6)
            return [h_dir, h_dir2][:num_h]

        elif len(cc_vectors) >= 2:
            v1, v2 = cc_vectors[0], cc_vectors[1]
            # sp2碳有2个C邻居 + 可能1个H
            # H方向是v1和v2的角平分线反方向
            bisector = v1 + v2
            norm_b = np.linalg.norm(bisector)
            if norm_b > 1e-6:
                h_dir = -bisector / norm_b
            else:
                # v1和v2近似相反方向（不常见但可能），使用法线方向
                h_dir = ring_normal
            return [h_dir]

        return []

    def _sp_hydrogen_dirs(self, cc_vectors, ring_normal, num_h):
        """sp碳的氢原子方向（线性几何，180°）"""
        import numpy as np
        import math

        if not cc_vectors:
            return [np.array([1.0, 0.0, 0.0])][:num_h]

        # sp碳：两个双键方向近似相反，没有氢原子（键负载=4）
        # 如果有氢原子（键负载<4的sp碳），放在与双键平面垂直的方向
        if len(cc_vectors) >= 2:
            v1, v2 = cc_vectors[0], cc_vectors[1]
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n > 1e-6:
                normal /= norm_n
                return [normal, -normal][:num_h]

        # 退化为使用任意垂直方向
        v1 = cc_vectors[0]
        arb = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(v1, arb)
        perp /= (np.linalg.norm(perp) + 1e-6)
        return [perp, -perp][:num_h]

    def _build_ring_coords_with_bond_types(self, G, main_cycle):
        """根据键类型信息生成正确键长的环骨架坐标

        CycloalkaneBuilder.build_coordinates 生成等边多边形（所有键1.54A），
        但含双键的环中双键应为1.34A，单键应为1.54A。
        此方法使用scipy优化内角，精确闭合不等边多边形环。

        Args:
            G: networkx图对象(含bond_type属性)
            main_cycle: 主环节点列表

        Returns:
            碳原子坐标字典 {node_id: (x, y, z)}，包含环内和环外碳原子
        """
        import numpy as np
        import math
        from scipy.optimize import minimize

        TARGET_DOUBLE = 1.34
        TARGET_SINGLE = 1.54

        cycle_set = set(main_cycle)
        n_ring = len(main_cycle)

        # 收集环内每条边的目标键长
        ring_edge_lengths = []
        for i in range(n_ring):
            u = main_cycle[i]
            v = main_cycle[(i + 1) % n_ring]
            bt = G[u][v].get('bond_type', 'single') if G.has_edge(u, v) else 'single'
            target = TARGET_DOUBLE if bt == 'double' else (1.20 if bt == 'triple' else TARGET_SINGLE)
            ring_edge_lengths.append(target)

        # 使用scipy优化内角来闭合不等边多边形
        # 多边形内角和 = (n-2)*pi，优化前n-1个内角，第n个由约束决定
        def ring_closure_error(angles_flat, edge_lengths):
            n = len(edge_lengths)
            interior = list(angles_flat)
            last_angle = (n - 2) * math.pi - sum(interior)
            if last_angle <= 0.1 or last_angle >= math.pi - 0.1:
                return 1e10
            interior.append(last_angle)
            for a in interior:
                if a <= 0.1 or a >= math.pi - 0.1:
                    return 1e10

            direction = 0.0
            x = edge_lengths[0]
            y = 0.0
            for i in range(1, n):
                turn = math.pi - interior[i]
                direction += turn
                x += edge_lengths[i] * math.cos(direction)
                y += edge_lengths[i] * math.sin(direction)
            return x**2 + y**2

        # 正多边形内角作为初始值
        regular_angle = (n_ring - 2) * math.pi / n_ring
        x0 = [regular_angle] * (n_ring - 1)
        result = minimize(ring_closure_error, x0, args=(ring_edge_lengths,),
                          method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 50000})

        optimized_angles = list(result.x)
        last_angle = (n_ring - 2) * math.pi - sum(optimized_angles)
        optimized_angles.append(last_angle)

        # 用优化后的内角构建坐标
        coords = {}
        coords[main_cycle[0]] = np.array([0.0, 0.0, 0.0])
        direction = 0.0
        coords[main_cycle[1]] = np.array([ring_edge_lengths[0], 0.0, 0.0])

        for i in range(1, n_ring - 1):
            # i 从 1 到 n_ring-2，放置 main_cycle[2] 到 main_cycle[n_ring-1]
            # 注意：i=n_ring-1 时对应的顶点是 main_cycle[0]（闭合），不需要放置
            turn = math.pi - optimized_angles[i]
            direction += turn
            prev_pos = coords[main_cycle[i]]
            edge_len = ring_edge_lengths[i]
            new_pos = prev_pos + edge_len * np.array([math.cos(direction), math.sin(direction), 0.0])
            coords[main_cycle[i + 1]] = new_pos

        # 居中
        center = np.mean([coords[n] for n in main_cycle if n in coords], axis=0)
        for node in main_cycle:
            if node in coords:
                coords[node] = coords[node] - center

        # 处理环外碳原子（BFS逐层处理，支持链式环外碳如 -CH2-CH3）
        processed = set(cycle_set)
        queue = list(cycle_set)
        while queue:
            parent = queue.pop(0)
            if parent not in coords:
                continue
            parent_pos = coords[parent]

            # 找到parent的所有未处理邻居
            for node in G.neighbors(parent):
                if node in processed or node in coords:
                    continue

                # 计算放置方向：取parent已有邻居方向的反方向之和
                other_dirs = []
                for nb in G.neighbors(parent):
                    if nb != node and nb in coords:
                        d = coords[nb] - parent_pos
                        norm = np.linalg.norm(d)
                        if norm > 1e-6:
                            other_dirs.append(d / norm)

                if other_dirs:
                    sum_dir = -np.sum(other_dirs, axis=0)
                    norm_s = np.linalg.norm(sum_dir)
                    if norm_s > 1e-6:
                        direction = sum_dir / norm_s
                    else:
                        direction = np.array([0.0, 0.0, 1.0])
                else:
                    direction = np.array([0.0, 0.0, 1.0])

                bt = G[parent][node].get('bond_type', 'single')
                target = TARGET_DOUBLE if bt == 'double' else (1.20 if bt == 'triple' else TARGET_SINGLE)
                coords[node] = parent_pos + direction * target
                processed.add(node)
                queue.append(node)

        return {i: (float(c[0]), float(c[1]), float(c[2])) for i, c in coords.items()}

    def _fix_cumulated_diene_coords(self, G, c_coords):
        """修正累积双键段(C=C=C)的共线坐标问题

        RDKit的MMFF力场会将累积双键(C=C=C)优化为180°共线构型，
        导致GaussView按距离判断时误将非成键的C-C对显示为三键。

        修正策略：检测图中的累积双键段，对共线(角度>170°)的碳原子
        进行角度修正，将中间碳偏移使键角从180°修正为~120°。

        Args:
            G: networkx图对象(含bond_type属性)
            c_coords: 碳原子坐标字典 {node_id: (x, y, z)}

        Returns:
            修正后的坐标字典
        """
        import numpy as np

        if not c_coords or len(c_coords) < 3:
            return c_coords

        # 找出所有累积双键节点(参与2个双键的碳原子)
        cumulated_nodes = set()
        for node in G.nodes():
            double_count = sum(
                1 for nb in G.neighbors(node)
                if G[node][nb].get('bond_type', 'single') == 'double'
            )
            if double_count >= 2:
                cumulated_nodes.add(node)

        if not cumulated_nodes:
            return c_coords

        # 构建累积双键段的路径
        # 从每个累积节点出发，沿双键边找到完整的累积段
        visited = set()
        segments = []  # 每个segment是 [endpoint1, cum_node1, cum_node2, ..., endpoint2]

        for start in cumulated_nodes:
            if start in visited:
                continue
            # 沿双键边向两个方向扩展
            segment = [start]
            visited.add(start)

            # 向左扩展
            current = start
            while True:
                # 找到current的双键邻居中未访问的
                db_neighbors = [
                    nb for nb in G.neighbors(current)
                    if G[current][nb].get('bond_type', 'single') == 'double'
                    and nb not in visited
                ]
                if not db_neighbors:
                    break
                next_node = db_neighbors[0]
                segment.insert(0, next_node)
                visited.add(next_node)
                if next_node not in cumulated_nodes:
                    # 到达端点，停止
                    break
                current = next_node

            # 向右扩展
            current = start
            while True:
                db_neighbors = [
                    nb for nb in G.neighbors(current)
                    if G[current][nb].get('bond_type', 'single') == 'double'
                    and nb not in visited
                ]
                if not db_neighbors:
                    break
                next_node = db_neighbors[0]
                segment.append(next_node)
                visited.add(next_node)
                if next_node not in cumulated_nodes:
                    break
                current = next_node

            if len(segment) >= 3:
                segments.append(segment)

        # 对每个累积段进行角度修正
        for seg in segments:
            # 对于段中每个三元组 (A, B, C)，检查角度
            # B是累积双键中心碳，A和C是B的双键邻居
            for i in range(len(seg) - 2):
                a, b, c = seg[i], seg[i+1], seg[i+2]
                if a not in c_coords or b not in c_coords or c not in c_coords:
                    continue

                pa = np.array(c_coords[a])
                pb = np.array(c_coords[b])
                pc = np.array(c_coords[c])

                v1 = pa - pb
                v2 = pc - pb
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 < 1e-8 or n2 < 1e-8:
                    continue

                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))

                # 只修正接近共线的角度(>170°)
                if angle > 170.0:
                    # 将B沿垂直于v1-v2平面的方向偏移
                    # 使A-B-C角度修正为~120°
                    normal = np.cross(v1, v2)
                    nn = np.linalg.norm(normal)
                    if nn < 1e-8:
                        # v1和v2几乎平行，选择任意垂直方向
                        if abs(v1[0]) < abs(v1[1]):
                            normal = np.cross(v1, [1, 0, 0])
                        else:
                            normal = np.cross(v1, [0, 1, 0])
                        nn = np.linalg.norm(normal)

                    if nn > 1e-8:
                        normal = normal / nn
                        # 偏移量：使角度从180°修正为120°
                        # 对于等边情况，offset = bond_length * tan(30°) ≈ 0.577 * bond_length
                        avg_bond = (n1 + n2) / 2
                        offset_dist = avg_bond * 0.50  # 使角度从180°修正到~120°
                        # 将B向法线方向偏移
                        new_pb = pb + normal * offset_dist
                        c_coords[b] = (float(new_pb[0]), float(new_pb[1]), float(new_pb[2]))

                        # 递归修正：偏移B后，需要检查B的其他键角
                        # 检查B的所有邻居，确保没有其他共线问题
                        for nb in G.neighbors(b):
                            if nb == a or nb == c or nb not in c_coords:
                                continue
                            p_nb = np.array(c_coords[nb])
                            # 检查 B-Nb 与 B-A 或 B-C 的角度
                            for ref in [a, c]:
                                if ref not in c_coords:
                                    continue
                                p_ref = np.array(c_coords[ref])
                                v_ref = p_ref - new_pb
                                v_nb = p_nb - new_pb
                                nr = np.linalg.norm(v_ref)
                                nnb = np.linalg.norm(v_nb)
                                if nr < 1e-8 or nnb < 1e-8:
                                    continue
                                cos_a = np.dot(v_ref, v_nb) / (nr * nnb)
                                cos_a = np.clip(cos_a, -1.0, 1.0)
                                if np.degrees(np.arccos(cos_a)) > 170.0:
                                    # Nb也需要修正
                                    offset2 = normal * offset_dist * 0.5
                                    new_pnb = p_nb + offset2
                                    c_coords[nb] = (float(new_pnb[0]), float(new_pnb[1]), float(new_pnb[2]))

        # 多轮迭代修正：修正后某些角度可能仍接近180°，需要反复检查
        # 最多迭代5轮
        for iteration in range(5):
            still_colinear = False
            for node in cumulated_nodes:
                if node not in c_coords:
                    continue
                db_nbs = [nb for nb in G.neighbors(node) if G[node][nb].get('bond_type', 'single') == 'double']
                if len(db_nbs) < 2:
                    continue
                pb = np.array(c_coords[node])
                # 检查所有双键邻居对的角度
                for i_idx in range(len(db_nbs)):
                    for j_idx in range(i_idx + 1, len(db_nbs)):
                        a, c = db_nbs[i_idx], db_nbs[j_idx]
                        if a not in c_coords or c not in c_coords:
                            continue
                        pa = np.array(c_coords[a])
                        pc = np.array(c_coords[c])
                        v1 = pa - pb
                        v2 = pc - pb
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 < 1e-8 or n2 < 1e-8:
                            continue
                        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_a))
                        if angle > 170.0:
                            still_colinear = True
                            # 将该碳沿垂直方向偏移
                            normal = np.cross(v1, v2)
                            nn = np.linalg.norm(normal)
                            if nn < 1e-8:
                                if abs(v1[0]) < abs(v1[1]):
                                    normal = np.cross(v1, [1, 0, 0])
                                else:
                                    normal = np.cross(v1, [0, 1, 0])
                                nn = np.linalg.norm(normal)
                            if nn > 1e-8:
                                normal = normal / nn
                                avg_bond = (n1 + n2) / 2
                                offset_dist = avg_bond * 0.50
                                new_pb = pb + normal * offset_dist
                                c_coords[node] = (float(new_pb[0]), float(new_pb[1]), float(new_pb[2]))
                                pb = new_pb
            if not still_colinear:
                break

        return c_coords

    def _scale_multi_bond_distances(self, G, c_coords, h_coords):
        """缩放双键/三键的C-C距离至真实键长，使GaussView能根据距离正确识别键级

        GaussView根据原子间距离自动判断键级：
        - C=C 双键典型距离 ~1.34 Å，但MMFF优化后常为 ~1.45 Å，被误判为单键
        - C≡C 三键典型距离 ~1.20 Å，但MMFF优化后常为 ~1.30 Å

        修正策略：
        - 链状分子：将双键C-C距离缩放至1.34 Å，三键缩放至1.20 Å，
          同时等比例移动相邻的氢原子以保持C-H键的几何关系
        - 环状分子：使用居中缩放策略，双键两端各移动一半偏移量，
          避免BFS在环中导致的偏移叠加破坏环几何

        Args:
            G: networkx图对象(含bond_type属性)
            c_coords: 碳原子坐标字典 {node_id: (x, y, z)}
            h_coords: 氢原子坐标列表 [(x, y, z), ...]（按碳原子顺序排列）

        Returns:
            (修正后的c_coords, 修正后的h_coords)
        """
        import numpy as np
        import networkx as nx

        if not c_coords or len(c_coords) < 2:
            return c_coords, h_coords

        # 目标键长
        TARGET_DOUBLE = 1.34  # Å，C=C双键
        TARGET_TRIPLE = 1.20  # Å，C≡C三键
        TARGET_SINGLE = 1.54  # Å，C-C单键

        # 收集多重键边
        multi_bond_edges = []
        for u, v, data in G.edges(data=True):
            bt = data.get('bond_type', 'single')
            if bt == 'double':
                multi_bond_edges.append((u, v, TARGET_DOUBLE))
            elif bt == 'triple':
                multi_bond_edges.append((u, v, TARGET_TRIPLE))

        if not multi_bond_edges:
            # 即使没有多重键，也需要缩放C-H键长（Compute2DCoords默认C-H距离~1.5Å）
            n_c = G.number_of_nodes()
            c_h_counts_early = []
            for c_idx in range(n_c):
                bond_load = 0
                for nb in G.neighbors(c_idx):
                    bt = G[c_idx][nb].get('bond_type', 'single')
                    if bt == 'double':
                        bond_load += 2
                    elif bt == 'triple':
                        bond_load += 3
                    else:
                        bond_load += 1
                c_h_counts_early.append(max(0, 4 - bond_load))
            h_to_c_early = {}
            h_idx = 0
            for c_idx in range(n_c):
                for _ in range(c_h_counts_early[c_idx]):
                    if h_idx < len(h_coords):
                        h_to_c_early[h_idx] = c_idx
                        h_idx += 1
            h_coords = self._scale_ch_bond_lengths(c_coords, h_coords, h_to_c_early)
            return c_coords, h_coords

        # 检测是否为环状分子（含环的分子）
        has_ring = bool(nx.cycle_basis(G))

        # 计算每个碳的氢原子数，建立H到C的映射
        n_c = G.number_of_nodes()
        c_h_counts = []
        for c_idx in range(n_c):
            bond_load = 0
            for nb in G.neighbors(c_idx):
                bt = G[c_idx][nb].get('bond_type', 'single')
                if bt == 'double':
                    bond_load += 2
                elif bt == 'triple':
                    bond_load += 3
                else:
                    bond_load += 1
            c_h_counts.append(max(0, 4 - bond_load))

        # 建立氢原子索引到碳原子的映射
        h_to_c = {}
        h_idx = 0
        for c_idx in range(n_c):
            for _ in range(c_h_counts[c_idx]):
                if h_idx < len(h_coords):
                    h_to_c[h_idx] = c_idx
                    h_idx += 1

        if has_ring:
            # 环状分子：使用居中缩放策略
            # 对每条多重键，两端各移动一半偏移量，保持环几何
            new_c_coords = {i: np.array(coord) for i, coord in c_coords.items()}

            # 检测环内共轭/离域：如果环内所有MMFF键长近似相等，
            # 说明RDKit已将其芳香化（如1,3,5-环己三烯→苯），
            # 需要对环内双键和单键分别缩放到不同目标距离以保持对称性
            cycles = nx.cycle_basis(G)
            if cycles:
                main_cycle = cycles[0]
                cycle_set = set(main_cycle)
                # 收集环内所有边的MMFF距离
                ring_double_dists = []
                ring_single_dists = []
                for u, v, data in G.edges(data=True):
                    if u in cycle_set and v in cycle_set:
                        if u in new_c_coords and v in new_c_coords:
                            d = np.linalg.norm(new_c_coords[u] - new_c_coords[v])
                            if data.get('bond_type', 'single') == 'double':
                                ring_double_dists.append(d)
                            else:
                                ring_single_dists.append(d)

                # 如果环内双键和单键的MMFF距离非常接近（差值<0.05Å），
                # 说明RDKit将其识别为离域/芳香体系
                is_delocalized = (
                    ring_double_dists and ring_single_dists and
                    abs(np.mean(ring_double_dists) - np.mean(ring_single_dists)) < 0.05
                )

                if is_delocalized:
                    # 共轭环（如1,3,5-环己三烯→苯的离域体系）
                    # RDKit将其芳香化后所有键长近似相等（~1.5Å的2D坐标），
                    # 不宜按double/single分别缩放，也不宜逐条居中缩放（会破坏对称性）
                    # 正确策略：以环中心为基准整体等比例缩放，保持环的对称性
                    TARGET_AROMATIC = 1.40  # Å，芳香键长

                    # 计算环中心
                    cycle_coords = np.array([new_c_coords[i] for i in main_cycle if i in new_c_coords])
                    if len(cycle_coords) > 0:
                        ring_center = np.mean(cycle_coords, axis=0)

                        # 计算当前环内平均键长
                        ring_dists = []
                        for u, v, data in G.edges(data=True):
                            if u in cycle_set and v in cycle_set and u in new_c_coords and v in new_c_coords:
                                ring_dists.append(np.linalg.norm(new_c_coords[u] - new_c_coords[v]))
                        avg_ring_dist = np.mean(ring_dists) if ring_dists else 1.5

                        # 整体等比例缩放：以环中心为基准，将环内所有碳原子按统一比例缩放
                        global_scale = TARGET_AROMATIC / avg_ring_dist
                        for i in cycle_set:
                            if i in new_c_coords:
                                new_c_coords[i] = ring_center + (new_c_coords[i] - ring_center) * global_scale

                    # 环外多重键也需处理
                    exo_multi = [(u, v, td) for u, v, td in multi_bond_edges
                                 if not (u in cycle_set and v in cycle_set)]
                    for u, v, target_dist in exo_multi:
                        if u not in new_c_coords or v not in new_c_coords:
                            continue
                        pu = new_c_coords[u]
                        pv = new_c_coords[v]
                        vec = pv - pu
                        current_dist = np.linalg.norm(vec)
                        if current_dist < 1e-8 or abs(current_dist - target_dist) < 0.02:
                            continue
                        # 环外键：环内端不动，环外端移动
                        if u in cycle_set:
                            new_c_coords[v] = new_c_coords[u] + vec / current_dist * target_dist
                        elif v in cycle_set:
                            new_c_coords[u] = new_c_coords[v] - vec / current_dist * target_dist
                        else:
                            scale = target_dist / current_dist
                            mid = (pu + pv) / 2.0
                            new_c_coords[u] = mid - vec / 2.0 * scale
                            new_c_coords[v] = mid + vec / 2.0 * scale

                    # 转换回元组格式
                    result_c = {i: (float(c[0]), float(c[1]), float(c[2]))
                                for i, c in new_c_coords.items()}

                    # 修正非键连原子碰撞
                    result_c = self._resolve_atom_collisions(G, result_c)

                    # 同步偏移氢原子坐标
                    result_h = list(h_coords)
                    for h_i in range(len(result_h)):
                        if h_i in h_to_c:
                            c_i = h_to_c[h_i]
                            if c_i in new_c_coords:
                                h_coord = np.array(result_h[h_i]) + (new_c_coords[c_i] - np.array(c_coords[c_i]))
                                result_h[h_i] = (float(h_coord[0]), float(h_coord[1]), float(h_coord[2]))

                    # 缩放C-H键长至真实键长1.09Å（Compute2DCoords默认C-H距离~1.5Å，GaussView无法识别）
                    result_h = self._scale_ch_bond_lengths(result_c, result_h, h_to_c)

                    # 如果坐标是纯平面(2D回退)，添加Z轴扰动生成合理的3D构象
                    result_c, result_h = self._add_z_perturbation(G, result_c, result_h, h_to_c)

                    # Z扰动可能改变了C-H键长，重新缩放
                    result_h = self._scale_ch_bond_lengths(result_c, result_h, h_to_c)

                    return result_c, result_h

            # 非共轭环：使用迭代力导向缩放，同时处理所有键
            # 逐条居中缩放会导致同一碳参与多条双键时偏移被覆盖（如累积双烯 C=C=C）
            # 改为迭代法：每步对所有键同时施加弹簧力，逐步收敛到目标键长
            # 同时对单键施加弱约束，防止环几何被过度扭曲
            all_bond_targets = []
            for u, v, data in G.edges(data=True):
                if u not in new_c_coords or v not in new_c_coords:
                    continue
                bt = data.get('bond_type', 'single')
                if bt == 'double':
                    all_bond_targets.append((u, v, TARGET_DOUBLE, 1.0))  # 强约束
                elif bt == 'triple':
                    all_bond_targets.append((u, v, TARGET_TRIPLE, 1.0))  # 强约束
                else:
                    all_bond_targets.append((u, v, TARGET_SINGLE, 0.3))  # 弱约束，防止单键被过度拉伸

            for iteration in range(100):
                max_error = 0.0
                offsets = {i: np.zeros(3) for i in new_c_coords}

                for u, v, target_dist, stiffness in all_bond_targets:
                    if u not in new_c_coords or v not in new_c_coords:
                        continue
                    pu = new_c_coords[u]
                    pv = new_c_coords[v]
                    vec = pv - pu
                    current_dist = np.linalg.norm(vec)
                    if current_dist < 1e-8:
                        continue

                    error = current_dist - target_dist
                    if stiffness >= 1.0:
                        max_error = max(max_error, abs(error))

                    # 弹簧力：两端各移一半偏移量，乘以刚度系数
                    correction = vec / current_dist * error * 0.5 * stiffness
                    offsets[u] += correction
                    offsets[v] -= correction

                # 一次性应用所有偏移
                for i in new_c_coords:
                    new_c_coords[i] = new_c_coords[i] + offsets[i] * 0.5  # 阻尼因子0.5防止振荡

                if max_error < 0.01:
                    break

            # 转换回元组格式
            result_c = {i: (float(c[0]), float(c[1]), float(c[2]))
                        for i, c in new_c_coords.items()}

            # 修正非键连原子碰撞
            result_c = self._resolve_atom_collisions(G, result_c)

            # 同步偏移氢原子坐标
            result_h = list(h_coords)
            for h_i in range(len(result_h)):
                if h_i in h_to_c:
                    c_i = h_to_c[h_i]
                    if c_i in new_c_coords:
                        h_coord = np.array(result_h[h_i]) + (new_c_coords[c_i] - np.array(c_coords[c_i]))
                        result_h[h_i] = (float(h_coord[0]), float(h_coord[1]), float(h_coord[2]))

            # 缩放C-H键长至真实键长1.09Å（Compute2DCoords默认C-H距离~1.5Å，GaussView无法识别）
            result_h = self._scale_ch_bond_lengths(result_c, result_h, h_to_c)

            # 如果坐标是纯平面(2D回退)，添加Z轴扰动生成合理的3D构象
            result_c, result_h = self._add_z_perturbation(G, result_c, result_h, h_to_c)

            # Z扰动可能改变了C-H键长，重新缩放
            result_h = self._scale_ch_bond_lengths(result_c, result_h, h_to_c)

            return result_c, result_h

        # 链状分子：使用原有的BFS偏移策略
        c_offsets = {i: np.zeros(3) for i in range(n_c)}

        for u, v, target_dist in multi_bond_edges:
            if u not in c_coords or v not in c_coords:
                continue

            pu = np.array(c_coords[u])
            pv = np.array(c_coords[v])
            vec = pv - pu
            current_dist = np.linalg.norm(vec)

            if current_dist < 1e-8:
                continue

            # 计算缩放比例
            scale = target_dist / current_dist
            # 将v向u方向移动，使距离达到target_dist
            new_pv = pu + vec * scale
            offset = new_pv - pv

            # 计算v的邻居碳原子（除了u之外），它们需要跟随v一起偏移
            # 使用BFS确定v一侧的所有碳原子
            visited = {u}
            queue = [v]
            affected_carbons = []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                affected_carbons.append(node)
                for nb in G.neighbors(node):
                    if nb not in visited:
                        queue.append(nb)

            # 对受影响的碳原子应用偏移
            for c in affected_carbons:
                c_offsets[c] = c_offsets.get(c, np.zeros(3)) + offset

        # 应用偏移到碳原子坐标
        new_c_coords = {}
        for i, coord in c_coords.items():
            new_coord = np.array(coord) + c_offsets.get(i, np.zeros(3))
            new_c_coords[i] = (float(new_coord[0]), float(new_coord[1]), float(new_coord[2]))

        # 修正非键连原子碰撞
        new_c_coords = self._resolve_atom_collisions(G, new_c_coords)

        # 同步偏移氢原子坐标
        new_h_coords = list(h_coords)
        for h_i in range(len(new_h_coords)):
            if h_i in h_to_c:
                c_i = h_to_c[h_i]
                h_coord = np.array(new_h_coords[h_i]) + c_offsets.get(c_i, np.zeros(3))
                new_h_coords[h_i] = (float(h_coord[0]), float(h_coord[1]), float(h_coord[2]))

        # 缩放C-H键长至真实键长1.09Å（Compute2DCoords默认C-H距离~1.5Å，GaussView无法识别）
        new_h_coords = self._scale_ch_bond_lengths(new_c_coords, new_h_coords, h_to_c)

        # 如果坐标是纯平面(2D回退)，添加Z轴扰动生成合理的3D构象
        new_c_coords, new_h_coords = self._add_z_perturbation(G, new_c_coords, new_h_coords, h_to_c)

        # Z扰动可能改变了C-H键长，重新缩放
        new_h_coords = self._scale_ch_bond_lengths(new_c_coords, new_h_coords, h_to_c)

        return new_c_coords, new_h_coords

    def _resolve_atom_collisions(self, G, c_coords, min_dist=0.8):
        """检测并修正非键连碳原子之间的碰撞（距离过近）

        RDKit 的 Compute2DCoords 对某些多环/稠环分子可能生成原子重叠的2D坐标，
        导致 GaussView 显示异常。此函数对非键连且距离 < min_dist 的原子对施加排斥力。

        Args:
            G: networkx 图对象
            c_coords: 碳原子坐标字典 {node_id: (x, y, z)}
            min_dist: 非键连碳原子间最小允许距离（Å）

        Returns:
            修正后的碳原子坐标字典
        """
        import numpy as np

        if not c_coords or len(c_coords) < 2:
            return c_coords

        # 建立键连关系集合
        bonded = set()
        for u, v in G.edges():
            bonded.add((min(u, v), max(u, v)))

        new_coords = {i: np.array(coord, dtype=float) for i, coord in c_coords.items()}
        nodes = sorted(new_coords.keys())

        # 迭代排斥力修正（最多50轮，避免死循环）
        for iteration in range(50):
            max_violation = 0.0
            offsets = {i: np.zeros(3) for i in nodes}

            for i_idx in range(len(nodes)):
                for j_idx in range(i_idx + 1, len(nodes)):
                    i, j = nodes[i_idx], nodes[j_idx]
                    # 跳过键连原子
                    if (min(i, j), max(i, j)) in bonded:
                        continue
                    pi = new_coords[i]
                    pj = new_coords[j]
                    vec = pj - pi
                    dist = np.linalg.norm(vec)
                    if dist < min_dist and dist > 1e-8:
                        violation = min_dist - dist
                        max_violation = max(max_violation, violation)
                        # 排斥力：将两个原子沿连线方向各推一半
                        push = vec / dist * violation * 0.5 * 1.1  # 1.1倍确保超过阈值
                        offsets[i] -= push
                        offsets[j] += push

            if max_violation < 0.01:
                break

            # 应用偏移
            for i in nodes:
                new_coords[i] = new_coords[i] + offsets[i]

        return {i: (float(c[0]), float(c[1]), float(c[2])) for i, c in new_coords.items()}

    def _add_z_perturbation(self, G, c_coords, h_coords, h_to_c):
        """对纯平面(2D)坐标添加Z轴扰动并优化，生成合理的3D构象

        当RDKit 3D嵌入失败回退到Compute2DCoords时，Z坐标全为0，
        非键连原子可能因2D投影距离过近导致GaussView误判键连接。
        此方法通过Z偏移+XY微调+迭代力场优化，确保：
        1) 所有键连原子距离接近真实键长
        2) 所有非键连原子3D距离 >= 阈值（避免GaussView误判）

        对于苯环等单环共轭体系，2D坐标中非键连原子距离已在安全范围内，
        不会施加Z扰动，保持其平面芳香性。

        策略：
        1. 检查是否存在2D距离 < MIN_NONBOND的非键连冲突对
        2. 若无冲突，直接返回（保持平面结构如苯环）
        3. 若有冲突，构建冲突图二部着色分配Z符号
        4. 施加初始Z偏移
        5. 迭代力场优化：键约束 + 非键排斥，直到收敛

        Args:
            G: networkx 图对象
            c_coords: 碳原子坐标字典 {node_id: (x, y, z)}
            h_coords: 氢原子坐标列表 [(x, y, z), ...]
            h_to_c: 氢原子索引到碳原子索引的映射 {h_idx: c_idx}

        Returns:
            (c_coords, h_coords) 修正后的坐标
        """
        import numpy as np
        import networkx as nx
        import math

        # 检查是否真的是纯平面坐标
        max_z = max(abs(c[2]) for c in c_coords.values()) if c_coords else 0
        if max_z > 0.1:
            return c_coords, h_coords  # 已有3D坐标，不需要处理

        # 建立键连关系集合
        bonded = set()
        for u, v in G.edges():
            bonded.add((min(u, v), max(u, v)))

        TARGET_SINGLE = 1.54   # C-C单键目标键长
        TARGET_DOUBLE_BOND = 1.34  # C=C双键目标键长
        MIN_NONBOND = 1.8     # 非键连原子3D最小距离（GaussView C-C成键阈值~1.7A，需留余量）

        # --- Step 1: 找出2D近距离非键连冲突对 ---
        # 只有2D距离 < MIN_NONBOND的非键连对才真正需要Z扰动来解决
        # 2D距离已 >= MIN_NONBOND的对在3D中不会误判，无需处理
        nodes = sorted(c_coords.keys())
        critical_conflicts = []  # 2D距离 < MIN_NONBOND，真正需要解决的冲突
        moderate_conflicts = []  # 2D距离 < CONFLICT_DIST 但 >= MIN_NONBOND
        CONFLICT_DIST = 2.5
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                i, j = nodes[i_idx], nodes[j_idx]
                if (min(i, j), max(i, j)) in bonded:
                    continue
                pi = c_coords[i]
                pj = c_coords[j]
                d2d = math.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                if d2d < MIN_NONBOND:
                    critical_conflicts.append((i, j, d2d))
                elif d2d < CONFLICT_DIST:
                    moderate_conflicts.append((i, j, d2d))

        # 若无关键冲突，说明2D坐标中所有非键连距离已安全，不需要Z扰动
        # 这样可以保持苯环等单环共轭体系的平面芳香性
        if not critical_conflicts:
            return c_coords, h_coords

        # --- Step 2: 冲突图二部着色 -> Z符号 ---
        # 使用关键冲突对构建冲突图（2D距离 < MIN_NONBOND的真正需要Z扰动的对）
        z_sign = {}
        if critical_conflicts:
            CG = nx.Graph()
            for i in c_coords:
                CG.add_node(i)
            for i, j, d in critical_conflicts:
                CG.add_edge(i, j)

            from networkx.algorithms import bipartite as bp_module
            if bp_module.is_bipartite(CG):
                color = bp_module.color(CG)
                z_sign = {node: 1 if color[node] == 0 else -1 for node in color}
            else:
                # 贪心着色：最近的冲突对优先分配相反符号
                sorted_conflicts = sorted(critical_conflicts, key=lambda x: x[2])
                for i, j, d in sorted_conflicts:
                    si = z_sign.get(i)
                    sj = z_sign.get(j)
                    if si is not None and sj is not None:
                        continue
                    elif si is not None and sj is None:
                        z_sign[j] = -si
                    elif sj is not None and si is None:
                        z_sign[i] = -sj
                    else:
                        z_sign[i] = 1
                        z_sign[j] = -1

        # 对没有分配Z符号的原子补全（仅在有关键冲突时需要）
        for node in c_coords:
            if node not in z_sign:
                neighbor_signs = [z_sign[nb] for nb in G.neighbors(node) if nb in z_sign]
                if neighbor_signs:
                    avg = sum(neighbor_signs) / len(neighbor_signs)
                    z_sign[node] = -1 if avg > 0 else 1
                else:
                    z_sign[node] = 1 if node % 2 == 0 else -1

        # --- Step 3: 施加初始Z偏移 ---
        Z_SCALE = 1.0
        coords = {}
        for i, (x, y, z) in c_coords.items():
            coords[i] = np.array([x, y, z + z_sign.get(i, 0) * Z_SCALE], dtype=float)

        # 收集键类型信息
        bond_targets = {}
        for u, v, data in G.edges(data=True):
            bt = data.get('bond_type', 'single')
            target = TARGET_DOUBLE_BOND if bt == 'double' else TARGET_SINGLE
            bond_targets[(u, v)] = target
            bond_targets[(v, u)] = target

        # --- Step 4: 迭代力场优化 ---
        for iteration in range(500):
            offsets = {i: np.zeros(3) for i in coords}
            max_error = 0.0

            # 键约束（适度刚度）
            for u, v in G.edges():
                target = bond_targets.get((u, v), TARGET_SINGLE)
                vec = coords[v] - coords[u]
                dist = np.linalg.norm(vec)
                if dist < 1e-8:
                    continue
                error = dist - target
                max_error = max(max_error, abs(error) * 0.5)
                correction = vec / dist * error * 0.15
                offsets[u] += correction
                offsets[v] -= correction

            # 非键排斥（强排斥，确保不误判成键）
            for i_idx in range(len(nodes)):
                for j_idx in range(i_idx + 1, len(nodes)):
                    i, j = nodes[i_idx], nodes[j_idx]
                    if (min(i, j), max(i, j)) in bonded:
                        continue
                    if i not in coords or j not in coords:
                        continue
                    vec = coords[j] - coords[i]
                    dist = np.linalg.norm(vec)
                    if dist < MIN_NONBOND and dist > 1e-8:
                        violation = MIN_NONBOND - dist
                        max_error = max(max_error, violation)
                        # 距离越近排斥越强，使用1/violation作为权重确保极近距离也能推开
                        weight = max(1.5, MIN_NONBOND / max(violation, 0.1))
                        push = vec / dist * violation * 0.5 * weight
                        offsets[i] -= push
                        offsets[j] += push

            # 应用偏移（阻尼0.4防止振荡）
            for i in coords:
                coords[i] += offsets[i] * 0.4

            if max_error < 0.005:
                break

        # --- Step 5: 输出结果 ---
        new_c = {i: (float(c[0]), float(c[1]), float(c[2])) for i, c in coords.items()}

        # 氢原子跟随碳原子的偏移
        new_h = []
        for h_i, (x, y, z) in enumerate(h_coords):
            c_i = h_to_c.get(h_i)
            if c_i is not None and c_i in new_c and c_i in c_coords:
                dx = new_c[c_i][0] - c_coords[c_i][0]
                dy = new_c[c_i][1] - c_coords[c_i][1]
                dz = new_c[c_i][2] - c_coords[c_i][2]
                new_h.append((x + dx, y + dy, z + dz * 0.5))
            else:
                new_h.append((x, y, z))

        return new_c, new_h

    def _scale_ch_bond_lengths(self, c_coords, h_coords, h_to_c):
        """将C-H键长缩放至真实键长1.09Å

        Compute2DCoords生成的2D坐标中，C-H键长默认为~1.5Å（RDKit 2D默认值），
        而GaussView根据原子间距离判断键级，C-H键的典型值为1.09Å，
        超过~1.2Å就不会显示键连接，因此需要将氢原子位置沿C-H方向缩放到1.09Å。

        若 h_to_c 不存在或某H未映射，自动寻找最近的碳原子作为归属碳。

        Args:
            c_coords: 碳原子坐标字典 {node_id: (x, y, z)}
            h_coords: 氢原子坐标列表 [(x, y, z), ...]
            h_to_c: 氢原子索引到碳原子索引的映射 {h_idx: c_idx}

        Returns:
            修正后的氢原子坐标列表
        """
        import numpy as np

        TARGET_CH = 1.09  # Å，C-H键真实键长
        result_h = list(h_coords)

        for h_i in range(len(result_h)):
            h_pos = np.array(result_h[h_i])

            # 确定归属碳：优先使用 h_to_c 映射，否则找最近的碳
            c_i = None
            if h_to_c is not None and h_i in h_to_c and h_to_c[h_i] in c_coords:
                c_i = h_to_c[h_i]
            else:
                # 回退：找距离最近的碳原子
                min_dist = float('inf')
                for c_idx, c_pos_val in c_coords.items():
                    c_pos_arr = np.array(c_pos_val)
                    d = np.linalg.norm(h_pos - c_pos_arr)
                    if d < min_dist:
                        min_dist = d
                        c_i = c_idx

            if c_i is not None and c_i in c_coords:
                c_pos = np.array(c_coords[c_i])
                vec = h_pos - c_pos
                dist = np.linalg.norm(vec)
                if dist > 1e-8:
                    # 沿C-H方向缩放氢原子位置至1.09Å
                    new_h_pos = c_pos + vec / dist * TARGET_CH
                    result_h[h_i] = (float(new_h_pos[0]), float(new_h_pos[1]), float(new_h_pos[2]))

        return result_h

    def _visualize_current(self):
        """可视化当前选中的异构体"""
        if self.window_closed:
            return

        selection = self.isomer_listbox.curselection()
        if not selection:
            messagebox.showwarning("未选择", "请先从列表中选择一个异构体")
            return

        list_idx = selection[0]
        
        # 获取列表项的内容来解析原始索引
        item_text = self.isomer_listbox.get(list_idx)
        # 格式: "#001  C5H12  烷烃"
        try:
            idx = int(item_text.split('#')[1].split()[0]) - 1
        except:
            idx = list_idx
        
        mol_type = self.current_molecule_type
        n_carbon = self.current_carbon_count

        try:
            # 先关闭所有matplotlib窗口，避免tkinter状态冲突
            plt.close('all')
        except:
            pass

        try:
            # 从 all_isomers 元组中获取 mol_type 和 isomer
            if isinstance(self.current_isomers[idx], tuple) and len(self.current_isomers[idx]) == 2:
                isomer_mol_type, isomer = self.current_isomers[idx]
            else:
                isomer_mol_type = mol_type
                isomer = self.current_isomers[idx]

            if isomer_mol_type == 'alkane':
                # 使用烷烃可视化器
                self.alkane_visualizer.visualize_isomer(n_carbon, idx, show=True)
            elif isomer_mol_type == 'diene':
                # 二烯烃：使用RDKit可视化（含双键标记）
                mol_name = self._get_mol_display_name(isomer_mol_type, n_carbon)
                self._visualize_polyene(isomer, f"{mol_name} #{idx+1}")
            elif isomer_mol_type == 'alkene':
                # 烯烃：使用 AlkeneIsomerVisualizer
                self.alkene_visualizer.visualize_isomer(isomer, show=True)
            elif isomer_mol_type == 'alkyne':
                # 炔烃：使用炔烃可视化器
                self.alkyne_visualizer.visualize_isomer(isomer, show=True)
            elif isomer_mol_type == 'cycloalkane':
                # 环烷烃：手动可视化
                self._visualize_cycloalkane(isomer, f"C{n_carbon} 环烷烃 #{idx+1}")
            elif isomer_mol_type == 'cycloalkene':
                # 单环烯烃：使用环烷烃方式可视化（含双键标记）
                self._visualize_cycloalkene(isomer, f"C{n_carbon} 单环烯烃 #{idx+1}")
            elif isomer_mol_type == 'alkenyl':
                # 烯炔烃：使用烯炔烃生成器的可视化方法
                self._visualize_alkenyl(isomer, f"{format_formula(n_carbon, 2*n_carbon-4)} 烯炔烃 #{idx+1}")
            elif isomer_mol_type in ('triene', 'tetraene', 'polyene'):
                # 多烯烃：使用多烯烃可视化方法（基于RDKit，含双键标记）
                mol_name = self._get_mol_display_name(isomer_mol_type, n_carbon)
                self._visualize_polyene(isomer, f"{mol_name} #{idx+1}")
            elif isomer_mol_type == 'cyclopolyene':
                # 单环多烯烃：使用单环多烯烃可视化方法（含双键标记+环标记）
                desc = self.cyclopolyene_generator.describe_isomer(isomer)
                formula = format_formula(n_carbon, self._current_n_hydrogen) if hasattr(self, '_current_n_hydrogen') and self._current_n_hydrogen is not None else (f"CH?" if n_carbon == 1 else f"C{n_carbon}H?")
                self._visualize_cyclopolyene(isomer, f"{formula} 单环多烯烃 #{idx+1} ({desc})")
            elif isomer_mol_type == 'multcycloalkane':
                # 多环烷烃：使用多环烷烃可视化方法
                desc = self.multcycloalkane_generator.describe_isomer(isomer)
                formula = format_formula(n_carbon, self._current_n_hydrogen) if hasattr(self, '_current_n_hydrogen') and self._current_n_hydrogen is not None else (f"CH?" if n_carbon == 1 else f"C{n_carbon}H?")
                self._visualize_multcycloalkane(isomer, f"{formula} 多环烷烃 #{idx+1} ({desc})")
            elif isomer_mol_type == 'multcyclomultalkane':
                # 多环多烯炔烃：使用统一生成器的可视化方法
                formula = format_formula(n_carbon, self._current_n_hydrogen) if hasattr(self, '_current_n_hydrogen') and self._current_n_hydrogen is not None else (f"CH?" if n_carbon == 1 else f"C{n_carbon}H?")
                desc = self.unified_generator.describe_isomer(isomer)
                self.unified_generator.visualize_isomer(isomer, title=f"{formula} 多环多烯炔烃 #{idx+1} ({desc})")
        except tk.TclError:
            pass  # 窗口已关闭
        except Exception as e:
            self._show_error(str(e))

    def _get_original_index(self, listbox_index):
        """从列表项文本中还原异构体的原始索引"""
        item_text = self.isomer_listbox.get(listbox_index)
        try:
            return int(item_text.split('#')[1].split()[0]) - 1
        except Exception:
            return listbox_index

    def _save_as(self):
        """另存为 - 保存异构体信息（支持多选导出）"""
        if self.window_closed:
            return

        if not self.current_isomers:
            messagebox.showwarning("无异构体", "请先生成异构体")
            return

        # 获取用户选中的列表项索引
        selected_indices = self.isomer_listbox.curselection()
        if selected_indices:
            # 用户有选中 → 仅导出选中的异构体
            indices = [self._get_original_index(i) for i in selected_indices]
            label = f"已选中 {len(indices)} 个异构体"
        else:
            # 无选中 → 导出全部
            indices = list(range(len(self.current_isomers)))
            label = f"共 {len(indices)} 个异构体"

        mol_type = self.current_molecule_type
        n_carbon = self.current_carbon_count

        mol_names = MOL_NAMES_EN

        formulas = {mt: compute_formula(mt, n_carbon, getattr(self, '_current_n_hydrogen', None)) for mt in mol_names}

        # 让用户选择保存位置
        import tkinter.filedialog
        folder_path = tkinter.filedialog.askdirectory(
            title=f"选择保存文件夹 — {label}",
            initialdir=str(Path.home())
        )

        if not folder_path:
            return  # 用户取消

        folder_path = Path(folder_path)

        # 使用输入的分子式作为文件夹名
        formula_name = self.molecule_input.get().strip().upper()
        if selected_indices:
            formula_name = f"{formula_name}_selected"
        save_folder = folder_path / formula_name
        try:
            save_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法创建文件夹: {str(e)}")
            return

        self.status_var.set("正在保存...")
        self.progress_var.set(0)
        self.progress_label.config(text="正在保存...")

        # 在后台线程中保存
        thread = threading.Thread(
            target=self._save_isomers_in_background,
            args=(save_folder, n_carbon, formulas, mol_names, indices)
        )
        thread.daemon = True
        thread.start()

    def _save_isomers_in_background(self, save_folder, n_carbon, formulas, mol_names, selected_indices=None):
        """在后台线程中保存异构体（多进程并行计算坐标）

        Args:
            selected_indices: 指定要导出的异构体索引列表，None 表示导出全部
        """
        import multiprocessing
        import pickle

        try:
            all_isomers = self.current_isomers

            # 过滤：仅处理选中的异构体（或全部）
            if selected_indices is not None:
                targets = [all_isomers[i] for i in selected_indices]
                target_indices = list(selected_indices)
            else:
                targets = all_isomers
                target_indices = list(range(len(all_isomers)))

            total = len(targets)

            # 解析目标异构体的元数据
            tasks = []
            for idx_in_filtered, item in enumerate(targets):
                # 使用原始异构体编号（用户看到的编号）
                i = target_indices[idx_in_filtered]
                if isinstance(item, tuple) and len(item) == 2:
                    mol_type, isomer_data = item
                else:
                    mol_type = self.current_molecule_type
                    isomer_data = item

                # 序列化图对象供子进程使用
                import networkx as nx
                if isinstance(isomer_data, nx.Graph):
                    graph_data = pickle.dumps(isomer_data)
                elif isinstance(isomer_data, str):
                    # 烷烃等返回规范字符串，需先转换为图再序列化
                    try:
                        if mol_type == 'alkene':
                            adj = self.alkene_visualizer.canon_to_adjacency(isomer_data)
                        else:
                            adj = self.alkane_generator.canon_to_adjacency(isomer_data)
                        G = nx.Graph()
                        for node, neighbors in adj.items():
                            for neighbor in neighbors:
                                G.add_edge(node, neighbor)
                        graph_data = pickle.dumps(G)
                    except Exception:
                        graph_data = pickle.dumps(isomer_data)
                else:
                    graph_data = pickle.dumps(isomer_data)

                tasks.append((i, graph_data, mol_type, n_carbon))

            # 阶段1: 多进程并行计算坐标
            n_workers = min(multiprocessing.cpu_count(), 8)
            self._update_progress(5, f"并行计算坐标 ({n_workers}核)...")
            print(f"\n{'='*50}")
            print(f"保存异构体: 共 {total} 个, 使用 {n_workers} 个进程并行计算坐标")
            print(f"{'='*50}")

            coords_results = {}
            completed = 0

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_parallel_compute_coords, task): task[0] for task in tasks}

                for future in as_completed(futures):
                    if self.window_closed:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return

                    idx = futures[future]
                    try:
                        result_idx, c_coords, h_coords = future.result()
                        coords_results[result_idx] = (c_coords, h_coords)
                    except Exception:
                        coords_results[idx] = ({}, [])

                    completed += 1
                    progress = int(completed / total * 70) + 5
                    self._update_progress(progress, f"计算坐标 {completed}/{total}")

            # 阶段2: 串行写文件（IO不是瓶颈）
            self._update_progress(75, "写入文件...")
            print(f"坐标计算完成，开始写入 .gjf 文件...")

            for idx_in_filtered, item in enumerate(targets):
                i = target_indices[idx_in_filtered]  # 原始异构体编号
                if self.window_closed:
                    return

                if isinstance(item, tuple) and len(item) == 2:
                    mol_type, isomer_data = item
                else:
                    mol_type = self.current_molecule_type
                    isomer_data = item

                # 使用预计算的坐标（key 为原始索引）
                c_coords, h_coords = coords_results.get(i, ({}, []))

                # 还原图对象
                import networkx as nx
                if isinstance(isomer_data, nx.Graph):
                    G = isomer_data
                elif isinstance(isomer_data, str):
                    # 规范字符串需转换为图（与序列化时一致的逻辑）
                    try:
                        if mol_type == 'alkene':
                            adj = self.alkene_visualizer.canon_to_adjacency(isomer_data)
                        else:
                            adj = self.alkane_generator.canon_to_adjacency(isomer_data)
                        G = nx.Graph()
                        for node, neighbors in adj.items():
                            for neighbor in neighbors:
                                G.add_edge(node, neighbor)
                    except Exception:
                        G = None
                else:
                    G = None
                content = self._generate_gjf_content(G, c_coords, h_coords, mol_type, n_carbon, i + 1)

                # 写文件（使用原始编号保持与坐标一致）
                filename = f"{formulas[mol_type]}_{mol_names.get(mol_type, mol_type)}_{i + 1:03d}.gjf"
                filepath = save_folder / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                progress = int((idx_in_filtered + 1) / total * 25) + 75
                self._update_progress(progress, f"已保存 {idx_in_filtered + 1}/{total}")

            # 完成
            if not self.window_closed:
                print(f"\n{'='*50}")
                print(f"保存完成: {total} 个 .gjf 文件 -> {save_folder}")
                print(f"{'='*50}\n")
                self.root.after(0, self._save_complete, save_folder, total)

        except Exception as e:
            if not self.window_closed:
                self.root.after(0, messagebox.showerror, "保存错误", str(e))

    def _generate_isomer_content(self, isomer_data, mol_type, n_carbon, index):
        """生成单个异构体的 Gaussian .gjf 格式内容"""
        import networkx as nx

        # 处理规范字符串格式（烷烃返回字符串，其他可能返回图对象）
        G = isomer_data
        adj = None
        if isinstance(isomer_data, str):
            # 规范字符串转换为图对象
            try:
                if mol_type == 'alkene':
                    # 烯烃使用特殊的转换（带双键信息）
                    adj = self.alkene_visualizer.canon_to_adjacency(isomer_data)
                else:
                    # 烷烃使用标准转换
                    adj = self.alkane_generator.canon_to_adjacency(isomer_data)
                G = nx.Graph()
                for node, neighbors in adj.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
            except Exception as e:
                return f"错误：无法解析异构体 #{index} 的结构\n{e}"

        # 确保 G 是图对象
        if not isinstance(G, nx.Graph):
            return f"错误：异构体 #{index} 的数据结构无效 (类型: {type(isomer_data).__name__})"

        # 分子式
        formulas = {mt: compute_formula(mt, n_carbon, getattr(self, '_current_n_hydrogen', None)) for mt in MOL_NAMES_EN}

        mol_names = MOL_NAMES_EN

        # 计算坐标
        c_coords, h_coords = self._calculate_molecule_coords(isomer_data, mol_type, n_carbon)

        # 缩放键长至真实值（包括C-H键长，Compute2DCoords默认C-H距离~1.5Å，GaussView无法识别）
        # GaussView根据原子间距离判断键级，MMFF优化后双键距离偏长(~1.45A)会被误判为单键
        if mol_type in ('cycloalkane', 'cycloalkene', 'alkene', 'diene', 'alkyne', 'alkenyl', 'triene', 'tetraene', 'polyene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane'):
            c_coords, h_coords = self._scale_multi_bond_distances(G, c_coords, h_coords)

        # ========== 生成 Gaussian .gjf 格式 ==========
        lines = []

        # 1. 文件头注释（三行，避免 GaussView 报错）
        lines.append(f"# {formulas[mol_type]} {mol_names[mol_type]} Isomer #{index}")
        lines.append("")
        lines.append(f"# {formulas[mol_type]} {mol_names[mol_type]} - Isomer #{index}")
        lines.append("# Generated by Molecule Isomer Visualizer")
        lines.append("")
        lines.append("0 1")  # 电荷=0, 自旋多重度=1

        # 2. 碳原子坐标
        for i in range(G.number_of_nodes()):
            if i in c_coords:
                coord = c_coords[i]
                lines.append(f"C     {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}")

        # 3. 氢原子坐标
        for coord in h_coords:
            lines.append(f"H     {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}")

        # 4. 对于含双键的类型，添加键连接信息（GaussView 依赖此信息识别键级）
        if mol_type in ('cycloalkene', 'alkene', 'diene', 'alkyne', 'alkenyl', 'triene', 'tetraene', 'polyene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane'):
            lines.append("")
            n_c = G.number_of_nodes()
            # 收集C-C键信息
            bond_lines = []
            for u, v, data in G.edges(data=True):
                bond_type = data.get('bond_type', 'single')
                if bond_type == 'double':
                    bond_order = 2
                elif bond_type == 'triple':
                    bond_order = 3
                else:
                    bond_order = 1
                bond_lines.append(f"{u+1:4d} {v+1:4d} {bond_order:4d} 0.0 0.0 0.0")

            # 添加C-H键信息，使GaussView能正确显示所有氢原子连接
            h_global_idx = n_c + 1  # 氢原子在GJF中的1-based全局索引
            for c_idx in range(n_c):
                # 计算该碳原子的氢原子数
                bond_load = 0
                for nb in G.neighbors(c_idx):
                    bt = G[c_idx][nb].get('bond_type', 'single')
                    if bt == 'double':
                        bond_load += 2
                    elif bt == 'triple':
                        bond_load += 3
                    else:
                        bond_load += 1
                h_count = 4 - bond_load
                for _ in range(max(0, h_count)):
                    bond_lines.append(f"{c_idx+1:4d} {h_global_idx:4d}    1 0.0 0.0 0.0")
                    h_global_idx += 1

            lines.extend(bond_lines)

        lines.append("")

        return "\n".join(lines)

    def _generate_gjf_content(self, G, c_coords, h_coords, mol_type, n_carbon, index):
        """使用预计算坐标生成 .gjf 文件内容（用于并行导出）"""
        formulas = {mt: compute_formula(mt, n_carbon, getattr(self, '_current_n_hydrogen', None)) for mt in MOL_NAMES_EN}
        mol_names = MOL_NAMES_EN

        # 缩放键长至真实值（包括C-H键长）
        if G is not None and mol_type in ('cycloalkane', 'cycloalkene', 'alkene', 'diene', 'alkyne', 'alkenyl', 'triene', 'tetraene', 'polyene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane'):
            c_coords, h_coords = self._scale_multi_bond_distances(G, c_coords, h_coords)

        lines = []
        lines.append(f"# {formulas[mol_type]} {mol_names[mol_type]} Isomer #{index}")
        lines.append("")
        lines.append(f"# {formulas[mol_type]} {mol_names[mol_type]} - Isomer #{index}")
        lines.append("# Generated by Molecule Isomer Visualizer")
        lines.append("")
        lines.append("0 1")

        # 碳原子坐标（按节点索引0,1,2,...顺序）
        if c_coords:
            n_nodes = max(c_coords.keys()) + 1
            for i in range(n_nodes):
                if i in c_coords:
                    coord = c_coords[i]
                    lines.append(f"C     {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}")
        elif G is not None:
            # RDKit嵌入失败：使用2D投影作为回退坐标，确保GJF有有效坐标
            import math
            n_nodes = G.number_of_nodes()
            for i in range(n_nodes):
                angle = 2 * math.pi * i / n_nodes
                x = 1.5 * math.cos(angle)
                y = 1.5 * math.sin(angle)
                lines.append(f"C     {x:>12.6f} {y:>12.6f}    0.000000")
            c_coords = {i: (1.5 * math.cos(2 * math.pi * i / n_nodes),
                            1.5 * math.sin(2 * math.pi * i / n_nodes),
                            0.0) for i in range(n_nodes)}
            # 生成氢原子坐标回退
            for node in sorted(G.nodes()):
                bond_load = 0
                for nb in G.neighbors(node):
                    bt = G[node][nb].get('bond_type', 'single')
                    bond_load += 2 if bt == 'double' else 1
                h_count = 4 - bond_load
                cx, cy, cz = c_coords[node]
                for j in range(max(0, h_count)):
                    ha = 2 * math.pi * j / max(h_count, 1) + node * 0.5
                    hx = cx + 1.09 * math.cos(ha)
                    hy = cy + 1.09 * math.sin(ha)
                    h_coords.append((hx, hy, cz))

        # 氢原子坐标
        for coord in h_coords:
            lines.append(f"H     {coord[0]:>12.6f} {coord[1]:>12.6f} {coord[2]:>12.6f}")

        # 键连接信息（含多重键的分子需要）
        has_valid_coords = bool(c_coords) and any(
            abs(c[0]) > 0.001 or abs(c[1]) > 0.001 or abs(c[2]) > 0.001
            for c in c_coords.values()
        ) if c_coords else False
        if G is not None and has_valid_coords and mol_type in ('cycloalkene', 'alkene', 'diene', 'alkyne', 'alkenyl', 'triene', 'tetraene', 'polyene', 'cyclopolyene', 'multcycloalkane', 'multcyclomultalkane'):
            lines.append("")
            n_c = G.number_of_nodes()
            bond_lines = []
            for u, v, data in G.edges(data=True):
                bt = data.get('bond_type', 'single')
                if bt == 'double':
                    bo = 2
                elif bt == 'triple':
                    bo = 3
                else:
                    bo = 1
                bond_lines.append(f"{u+1:4d} {v+1:4d} {bo:4d} 0.0 0.0 0.0")

            h_global_idx = n_c + 1
            for c_idx in range(n_c):
                bond_load = 0
                for nb in G.neighbors(c_idx):
                    bt = G[c_idx][nb].get('bond_type', 'single')
                    if bt == 'double':
                        bond_load += 2
                    elif bt == 'triple':
                        bond_load += 3
                    else:
                        bond_load += 1
                h_count = 4 - bond_load
                for _ in range(max(0, h_count)):
                    bond_lines.append(f"{c_idx+1:4d} {h_global_idx:4d}    1 0.0 0.0 0.0")
                    h_global_idx += 1

            lines.extend(bond_lines)

        lines.append("")
        return "\n".join(lines)

    def _calculate_molecule_coords(self, isomer_data, mol_type=None, n_carbon=None):
        """计算分子的坐标信息
        
        支持两种调用方式:
        1. isomer_data 为元组 (mol_type, data) - 新格式
        2. mol_type 单独传入 - 旧格式兼容
        """
        if mol_type is None and isinstance(isomer_data, tuple) and len(isomer_data) == 2:
            mol_type, isomer_data = isomer_data
        import networkx as nx
        import numpy as np

        c_coords = {}
        h_coords = []
        adj = None

        # 处理规范字符串格式（烷烃返回字符串，其他可能返回图对象）
        G = isomer_data
        if isinstance(isomer_data, str):
            # 规范字符串转换为邻接表和图对象
            try:
                if mol_type == 'alkene':
                    # 烯烃使用特殊的转换（带双键信息）
                    adj = self.alkene_visualizer.canon_to_adjacency(isomer_data)
                else:
                    # 烷烃使用标准转换
                    adj = self.alkane_generator.canon_to_adjacency(isomer_data)
                G = nx.Graph()
                for node, neighbors in adj.items():
                    for neighbor in neighbors:
                        G.add_edge(node, neighbor)
            except Exception:
                return c_coords, h_coords

        # 确保 G 是图对象
        if not isinstance(G, nx.Graph):
            return c_coords, h_coords

        try:
            if mol_type == 'alkane':
                # 烷烃使用 RDKit 生成高质量3D坐标
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                if adj is None:
                    adj = self.alkane_generator.canon_to_adjacency(isomer_data)

                n_carbons = G.number_of_nodes()

                # 构建RDKit分子
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                for node, neighbors in adj.items():
                    for n in neighbors:
                        if n > node:
                            mol.AddBond(node, n, BondType.SINGLE)

                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                return c_coords, h_coords

            elif mol_type == 'alkyne':
                # 使用炔烃可视化器
                coords = self.alkyne_visualizer._calculate_coordinates(G)
                hydrogens = self.alkyne_visualizer._generate_hydrogen_coords(G, coords)
                return coords, hydrogens

            elif mol_type == 'alkene':
                # 烯烃使用 RDKit 生成高质量3D坐标（与烷烃一致）
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取邻接表和双键边信息
                double_bond_edges = set()
                adj_dict = {}
                for u, v, data in G.edges(data=True):
                    if data.get('bond_type') == 'double':
                        double_bond_edges.add((min(u, v), max(u, v)))
                    if u not in adj_dict:
                        adj_dict[u] = []
                    if v not in adj_dict:
                        adj_dict[v] = []
                    adj_dict[u].append(v)
                    adj_dict[v].append(u)

                # 构建 RDKit 分子（含双键信息）
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                return c_coords, h_coords

            elif mol_type == 'diene':
                # 二烯烃：使用 RDKit 生成高质量3D坐标（含双键信息 + MMFF优化）
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取双键边信息
                double_bond_edges = set()
                for u, v, data in G.edges(data=True):
                    if data.get('bond_type') == 'double':
                        double_bond_edges.add((min(u, v), max(u, v)))

                # 构建 RDKit 分子
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢
                for node in sorted(G.nodes()):
                    bond_load = sum(
                        2 if G[node][nb].get('bond_type') == 'double' else 1
                        for nb in G.neighbors(node)
                    )
                    h_count = 4 - bond_load
                    if h_count > 0:
                        mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)

                # 二烯烃：先尝试3D嵌入+MMFF优化，失败回退2D坐标
                has_ring = len(nx.cycle_basis(G)) > 0
                if has_ring:
                    import math as _math
                    best_mol = None
                    best_max_bond = float('inf')
                    for seed in range(50):
                        mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                        params = AllChem.ETKDGv3()
                        params.randomSeed = seed
                        result = AllChem.EmbedMolecule(mol_trial, params)
                        if result == -1:
                            result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                            if result2 == -1:
                                continue
                        try:
                            AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                        except Exception:
                            pass
                        conf = mol_trial.GetConformer()
                        max_bond = 0.0
                        for u, v in G.edges():
                            if u < n_carbons and v < n_carbons:
                                p1 = conf.GetAtomPosition(u)
                                p2 = conf.GetAtomPosition(v)
                                dist = _math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                                max_bond = max(max_bond, dist)
                        if max_bond < best_max_bond:
                            best_max_bond = max_bond
                            best_mol = Chem.Mol(mol_trial.ToBinary())
                    if best_mol is not None:
                        mol = best_mol
                    else:
                        AllChem.Compute2DCoords(mol)
                else:
                    # 非环二烯烃：使用单种子3D嵌入
                    embed_ok = False
                    result = AllChem.EmbedMolecule(mol, randomSeed=42)
                    if result != -1:
                        embed_ok = True
                    else:
                        result2 = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                        if result2 != -1:
                            embed_ok = True
                    if not embed_ok:
                        AllChem.Compute2DCoords(mol)
                    else:
                        try:
                            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                        except Exception:
                            pass

                # 提取坐标
                c_coords = {}
                h_coords = []
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 修正累积双键段(C=C=C)的共线问题
                c_coords = self._fix_cumulated_diene_coords(G, c_coords)

                return c_coords, h_coords

            elif mol_type == 'cycloalkane':
                # 环烷烃：使用 RDKit 生成高质量3D坐标（含力场优化）
                c_coords_rdkit, h_coords_rdkit, h_to_c = self._build_cycloalkane_rdkit(G)

                if c_coords_rdkit is not None:
                    return c_coords_rdkit, h_coords_rdkit
                else:
                    # RDKit 嵌入失败，尝试 PolycycloalkaneGenerator 的多种子策略
                    mol = self.multcycloalkane_generator.graph_to_rdkit_mol(G, optimize=True)
                    if mol is not None and mol.GetNumConformers() > 0:
                        conf = mol.GetConformer()
                        c_coords = {}
                        h_coords = []
                        for atom in mol.GetAtoms():
                            pos = conf.GetAtomPosition(atom.GetIdx())
                            coord = np.array([float(pos.x), float(pos.y), float(pos.z)])
                            if atom.GetSymbol() == 'C':
                                c_coords[atom.GetIdx()] = coord
                            else:
                                h_coords.append(coord)
                        return c_coords, h_coords
                    # 最终回退：2D坐标
                    c_coords = {}
                    for node in G.nodes():
                        angle = 2 * np.pi * node / G.number_of_nodes()
                        c_coords[node] = np.array([np.cos(angle), np.sin(angle), 0.0])
                    return c_coords, []

            elif mol_type == 'cycloalkene':
                # 单环烯烃：使用 RDKit 生成高质量3D坐标（含双键信息 + 力场优化）
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取双键边信息
                double_bond_edges = set()
                for u, v, data in G.edges(data=True):
                    if data.get('bond_type') == 'double':
                        double_bond_edges.add((min(u, v), max(u, v)))

                # 构建 RDKit 分子
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢数量
                for node in sorted(G.nodes()):
                    total_bonds = 0
                    for nb in G.neighbors(node):
                        bond_type = G[node][nb].get('bond_type', 'single')
                        total_bonds += 2 if bond_type == 'double' else 1
                    h_count = 4 - total_bonds
                    mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)

                # 单环烯烃：先尝试3D嵌入+MMFF优化，失败回退2D坐标
                import math as _math
                best_mol = None
                best_max_bond = float('inf')
                for seed in range(50):
                    mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    result = AllChem.EmbedMolecule(mol_trial, params)
                    if result == -1:
                        result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                        if result2 == -1:
                            continue
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                    except Exception:
                        pass
                    conf = mol_trial.GetConformer()
                    max_bond = 0.0
                    for u, v in G.edges():
                        if u < n_carbons and v < n_carbons:
                            p1 = conf.GetAtomPosition(u)
                            p2 = conf.GetAtomPosition(v)
                            dist = _math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                            max_bond = max(max_bond, dist)
                    if max_bond < best_max_bond:
                        best_max_bond = max_bond
                        best_mol = Chem.Mol(mol_trial.ToBinary())
                if best_mol is not None:
                    mol = best_mol
                else:
                    AllChem.Compute2DCoords(mol)

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 修正累积双键段(C=C=C)的共线问题
                c_coords = self._fix_cumulated_diene_coords(G, c_coords)

                return c_coords, h_coords

            elif mol_type == 'cyclopolyene':
                # 单环多烯烃：使用 RDKit 生成高质量3D坐标（含双键信息 + 力场优化）
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取双键边信息
                double_bond_edges = set()
                for u, v, data in G.edges(data=True):
                    if data.get('bond_type') == 'double':
                        double_bond_edges.add((min(u, v), max(u, v)))

                # 构建 RDKit 分子
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢数量
                for node in sorted(G.nodes()):
                    total_bonds = 0
                    for nb in G.neighbors(node):
                        bond_type = G[node][nb].get('bond_type', 'single')
                        total_bonds += 2 if bond_type == 'double' else 1
                    h_count = 4 - total_bonds
                    if h_count > 0:
                        mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)

                # 单环多烯烃：先尝试3D嵌入+MMFF优化，失败回退2D坐标
                import math as _math
                best_mol = None
                best_max_bond = float('inf')
                for seed in range(50):
                    mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    result = AllChem.EmbedMolecule(mol_trial, params)
                    if result == -1:
                        result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                        if result2 == -1:
                            continue
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                    except Exception:
                        pass
                    conf = mol_trial.GetConformer()
                    max_bond = 0.0
                    for u, v in G.edges():
                        if u < n_carbons and v < n_carbons:
                            p1 = conf.GetAtomPosition(u)
                            p2 = conf.GetAtomPosition(v)
                            dist = _math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                            max_bond = max(max_bond, dist)
                    if max_bond < best_max_bond:
                        best_max_bond = max_bond
                        best_mol = Chem.Mol(mol_trial.ToBinary())
                if best_mol is not None:
                    mol = best_mol
                else:
                    AllChem.Compute2DCoords(mol)

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 修正累积双键段(C=C=C)的共线问题
                c_coords = self._fix_cumulated_diene_coords(G, c_coords)

                return c_coords, h_coords

            elif mol_type in ('triene', 'tetraene', 'polyene'):
                # 多烯烃：使用 RDKit 生成3D坐标（含双键信息 + MMFF优化）
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取双键边信息
                double_bond_edges = set()
                for u, v, data in G.edges(data=True):
                    bt = data.get('bond_type', 'single')
                    key = (min(u, v), max(u, v))
                    if bt == 'double':
                        double_bond_edges.add(key)

                # 构建 RDKit 分子
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢
                for node in sorted(G.nodes()):
                    total_bonds = 0
                    for nb in G.neighbors(node):
                        bt = G[node][nb].get('bond_type', 'single')
                        if bt == 'double':
                            total_bonds += 2
                        else:
                            total_bonds += 1
                    h_count = 4 - total_bonds
                    if h_count > 0:
                        mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return c_coords, h_coords
                mol = Chem.AddHs(mol)

                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result == -1:
                    AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                except Exception:
                    pass

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 修正累积双键段(C=C=C)的共线问题
                # RDKit/MMFF会将累积双键优化为180°共线，导致GaussView误判为三键
                # 将共线碳的角度修正为~120°(sp杂化碳的实际合理键角)
                c_coords = self._fix_cumulated_diene_coords(G, c_coords)

                return c_coords, h_coords

            elif mol_type == 'alkenyl':
                # 烯炔烃：使用 RDKit 生成高质量3D坐标（含双键+三键信息 + 力场优化）
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取双键和三键边信息
                double_bond_edges = set()
                triple_bond_edges = set()
                for u, v, data in G.edges(data=True):
                    bt = data.get('bond_type', 'single')
                    key = (min(u, v), max(u, v))
                    if bt == 'double':
                        double_bond_edges.add(key)
                    elif bt == 'triple':
                        triple_bond_edges.add(key)

                # 构建 RDKit 分子
                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in triple_bond_edges:
                        mol.AddBond(u, v, BondType.TRIPLE)
                    elif bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢数量
                for node in sorted(G.nodes()):
                    total_bonds = 0
                    for nb in G.neighbors(node):
                        bt = G[node][nb].get('bond_type', 'single')
                        if bt == 'double':
                            total_bonds += 2
                        elif bt == 'triple':
                            total_bonds += 3
                        else:
                            total_bonds += 1
                    h_count = 4 - total_bonds
                    if h_count > 0:
                        mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return c_coords, h_coords
                mol = Chem.AddHs(mol)

                # 嵌入3D坐标 + MMFF力场优化
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result == -1:
                    AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                except Exception:
                    pass

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 修正累积双键段(C=C=C)的共线问题
                c_coords = self._fix_cumulated_diene_coords(G, c_coords)

                return c_coords, h_coords

            elif mol_type == 'multcycloalkane':
                # 多环烷烃：使用 RDKit 生成3D坐标（全单键，含力场优化）
                import math
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢
                for node in sorted(G.nodes()):
                    h_count = 4 - G.degree(node)
                    if h_count > 0:
                        mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return self._fallback_coords_from_graph(G)

                mol = Chem.AddHs(mol)

                # 多环烷烃：多种子选择策略，选择最大C-C键长最短的构象
                best_mol = None
                best_max_bond = float('inf')
                for seed in range(50):
                    mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    result = AllChem.EmbedMolecule(mol_trial, params)
                    if result == -1:
                        result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                        if result2 == -1:
                            continue
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                    except Exception:
                        pass
                    # 计算最大C-C键长
                    conf_t = mol_trial.GetConformer()
                    max_bond = 0.0
                    for u, v in G.edges():
                        if u < n_carbons and v < n_carbons:
                            p1 = conf_t.GetAtomPosition(u)
                            p2 = conf_t.GetAtomPosition(v)
                            dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
                            max_bond = max(max_bond, dist)
                    if max_bond < best_max_bond:
                        best_max_bond = max_bond
                        best_mol = Chem.Mol(mol_trial.ToBinary())

                if best_mol is not None:
                    mol = best_mol
                else:
                    # 所有种子都失败，回退到2D坐标
                    AllChem.Compute2DCoords(mol)

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []

                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 坐标为空则回退
                if not c_coords:
                    return self._fallback_coords_from_graph(G)

                return c_coords, h_coords

            elif mol_type == 'multcyclomultalkane':
                # 多环多烯炔烃：使用 RDKit 生成3D坐标（含双键+三键信息 + 力场优化）
                import math
                from rdkit import Chem
                from rdkit.Chem import AllChem, BondType
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')

                n_carbons = G.number_of_nodes()

                # 从图中提取双键和三键边信息
                double_bond_edges = set()
                triple_bond_edges = set()
                for u, v, data in G.edges(data=True):
                    bt = data.get('bond_type', 'single')
                    key = (min(u, v), max(u, v))
                    if bt == 'double':
                        double_bond_edges.add(key)
                    elif bt == 'triple':
                        triple_bond_edges.add(key)

                mol = Chem.RWMol()
                for i in range(n_carbons):
                    mol.AddAtom(Chem.Atom('C'))

                added_bonds = set()
                for u, v in G.edges():
                    bond_key = (min(u, v), max(u, v))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    if bond_key in triple_bond_edges:
                        mol.AddBond(u, v, BondType.TRIPLE)
                    elif bond_key in double_bond_edges:
                        mol.AddBond(u, v, BondType.DOUBLE)
                    else:
                        mol.AddBond(u, v, BondType.SINGLE)

                # 设置显式氢（根据键负载计算）
                for node in sorted(G.nodes()):
                    bond_load = 0
                    for nb in G.neighbors(node):
                        bt = G[node][nb].get('bond_type', 'single')
                        if bt == 'double':
                            bond_load += 2
                        elif bt == 'triple':
                            bond_load += 3
                        else:
                            bond_load += 1
                    h_count = 4 - bond_load
                    if h_count > 0:
                        mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

                mol = mol.GetMol()
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    return self._fallback_coords_from_graph(G)

                mol = Chem.AddHs(mol)

                # 多环多烯炔烃：多种子选择策略，选择最大C-C键长最短的构象
                best_mol = None
                best_max_bond = float('inf')
                for seed in range(50):
                    mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    result = AllChem.EmbedMolecule(mol_trial, params)
                    if result == -1:
                        result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                        if result2 == -1:
                            continue
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                    except Exception:
                        pass
                    # 计算最大C-C键长
                    conf_t = mol_trial.GetConformer()
                    max_bond = 0.0
                    for u, v in G.edges():
                        if u < n_carbons and v < n_carbons:
                            p1 = conf_t.GetAtomPosition(u)
                            p2 = conf_t.GetAtomPosition(v)
                            dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
                            max_bond = max(max_bond, dist)
                    if max_bond < best_max_bond:
                        best_max_bond = max_bond
                        best_mol = Chem.Mol(mol_trial.ToBinary())

                if best_mol is not None:
                    mol = best_mol
                else:
                    # 所有种子都失败，回退到2D坐标
                    AllChem.Compute2DCoords(mol)

                # 提取坐标
                conf = mol.GetConformer()
                c_coords = {}
                h_coords = []
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coord = (float(pos.x), float(pos.y), float(pos.z))
                    if atom.GetSymbol() == 'C':
                        c_coords[atom.GetIdx()] = coord
                    else:
                        h_coords.append(coord)

                # 修正累积双键段(C=C=C)的共线问题
                c_coords = self._fix_cumulated_diene_coords(G, c_coords)

                # 坐标为空则回退
                if not c_coords:
                    return self._fallback_coords_from_graph(G)

                return c_coords, h_coords

        except Exception as e:
            import traceback
            print(f"【坐标计算异常】{mol_type} C{n_carbon}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return c_coords, h_coords

    def _save_complete(self, save_folder, total):
        """保存完成"""
        self.status_var.set(f"保存完成")
        self.progress_var.set(100)
        self.progress_label.config(text=f"已保存 {total} 个文件")
        messagebox.showinfo("保存完成", f"已保存 {total} 个文件到:\n{save_folder}")

    def _visualize_cycloalkene(self, G, title):
        """可视化单个单环烯烃异构体（使用 RDKit 生成高质量3D坐标）"""
        try:
            import networkx as nx
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

            n_carbons = G.number_of_nodes()

            # 从图中提取双键边信息
            double_bond_edges = set()
            for u, v, data in G.edges(data=True):
                if data.get('bond_type') == 'double':
                    double_bond_edges.add((min(u, v), max(u, v)))

            # 构建 RDKit 分子
            mol = Chem.RWMol()
            for i in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = (min(u, v), max(u, v))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)
                if bond_key in double_bond_edges:
                    mol.AddBond(u, v, BondType.DOUBLE)
                else:
                    mol.AddBond(u, v, BondType.SINGLE)

            for node in sorted(G.nodes()):
                total_bonds = 0
                for nb in G.neighbors(node):
                    bond_type = G[node][nb].get('bond_type', 'single')
                    total_bonds += 2 if bond_type == 'double' else 1
                h_count = 4 - total_bonds
                mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)

            # 单环烯烃：先尝试3D嵌入+MMFF优化，失败回退2D坐标
            import math as _math
            best_mol = None
            best_max_bond = float('inf')
            for seed in range(50):
                mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                result = AllChem.EmbedMolecule(mol_trial, params)
                if result == -1:
                    result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                    if result2 == -1:
                        continue
                try:
                    AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                except Exception:
                    pass
                conf = mol_trial.GetConformer()
                max_bond = 0.0
                for u, v in G.edges():
                    if u < n_carbons and v < n_carbons:
                        p1 = conf.GetAtomPosition(u)
                        p2 = conf.GetAtomPosition(v)
                        dist = _math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                        max_bond = max(max_bond, dist)
                if max_bond < best_max_bond:
                    best_max_bond = max_bond
                    best_mol = Chem.Mol(mol_trial.ToBinary())
            if best_mol is not None:
                mol = best_mol
            else:
                AllChem.Compute2DCoords(mol)

            # 提取坐标
            conf = mol.GetConformer()
            c_coords = {}
            h_coords = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coord = (float(pos.x), float(pos.y), float(pos.z))
                if atom.GetSymbol() == 'C':
                    c_coords[atom.GetIdx()] = coord
                else:
                    h_coords.append(coord)

            # 获取环信息
            main_cycle = nx.cycle_basis(G)[0]
            cycle_set = set(main_cycle)

            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 绘制C-C键（双键用红色粗线）
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in c_coords and neighbor in c_coords:
                            is_double = G[node][neighbor].get('bond_type') == 'double'
                            if is_double:
                                # 双键：主线 + 偏移缩短线
                                import numpy as np
                                p1 = np.array(c_coords[node])
                                p2 = np.array(c_coords[neighbor])
                                direction = p2 - p1
                                length = np.linalg.norm(direction)
                                if length > 1e-8:
                                    # 垂直方向（在分子所在平面内偏移）
                                    mid = (p1 + p2) / 2
                                    up = np.array([0, 0, 1])
                                    perp = np.cross(direction, up)
                                    perp_norm = np.linalg.norm(perp)
                                    if perp_norm < 1e-8:
                                        perp = np.cross(direction, np.array([0, 1, 0]))
                                        perp_norm = np.linalg.norm(perp)
                                    perp = perp / perp_norm * length * 0.04
                                    # 主线（不偏移，连接两碳原子）
                                    ax.plot(
                                        [p1[0], p2[0]],
                                        [p1[1], p2[1]],
                                        [p1[2], p2[2]],
                                        color='red', linewidth=3
                                    )
                                    # 第二条线：缩短至键中段 + 偏移
                                    t1, t2 = 0.25, 0.75
                                    sp1 = p1 + direction * t1 + perp
                                    sp2 = p1 + direction * t2 + perp
                                    ax.plot(
                                        [sp1[0], sp2[0]],
                                        [sp1[1], sp2[1]],
                                        [sp1[2], sp2[2]],
                                        color='red', linewidth=3
                                    )
                                else:
                                    ax.plot(
                                        [c_coords[node][0], c_coords[neighbor][0]],
                                        [c_coords[node][1], c_coords[neighbor][1]],
                                        [c_coords[node][2], c_coords[neighbor][2]],
                                        color='red', linewidth=3
                                    )
                            else:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='black', linewidth=2
                                )
                        drawn_bonds.add(bond_key)

            # 绘制碳原子（环碳用蓝色，双键碳用红色）
            double_carbons = set()
            for u, v, d in G.edges(data=True):
                if d.get('bond_type') == 'double':
                    double_carbons.add(u)
                    double_carbons.add(v)

            for n in sorted(c_coords.keys()):
                if n in double_carbons:
                    color = 'red'
                elif n in cycle_set:
                    color = 'blue'
                else:
                    color = 'black'
                ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                          c=color, s=150, zorder=5)

            # 绘制氢原子和碳氢键
            if h_coords:
                hx = [h[0] for h in h_coords]
                hy = [h[1] for h in h_coords]
                hz = [h[2] for h in h_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

                # 找每个氢连接的碳原子
                h_to_c = {}
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H':
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() == 'C':
                                h_idx = atom.GetIdx()
                                c_idx = neighbor.GetIdx()
                                h_to_c[h_idx] = c_idx

                for h_idx, c_idx in h_to_c.items():
                    if h_idx - n_carbons < len(h_coords) and c_idx in c_coords:
                        # 氢原子在RDKit中的索引大于碳原子
                        h_pos_idx = None
                        h_count = 0
                        for atom in mol.GetAtoms():
                            if atom.GetIdx() == h_idx:
                                break
                            if atom.GetSymbol() == 'H':
                                h_count += 1
                        if h_count < len(h_coords):
                            h_coord = h_coords[h_count]
                            c_coord = c_coords[c_idx]
                            ax.plot(
                                [c_coord[0], h_coord[0]],
                                [c_coord[1], h_coord[1]],
                                [c_coord[2], h_coord[2]],
                                'gray', linewidth=1, alpha=0.7, zorder=4
                            )

            # 标记环原子编号
            for i, node in enumerate(main_cycle):
                if node in c_coords:
                    coord = c_coords[node]
                    ax.text(coord[0], coord[1], coord[2], f'{i+1}', fontsize=9, color='blue')

            ax.set_xlabel("X (A)")
            ax.set_ylabel("Y (A)")
            ax.set_zlabel("Z (A)")
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("可视化错误", f"单环烯烃可视化失败: {str(e)}")

    def _visualize_cyclopolyene(self, G, title):
        """可视化单个单环多烯烃异构体（使用 RDKit 生成高质量3D坐标，含双键标记+环标记）"""
        try:
            import networkx as nx
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

            n_carbons = G.number_of_nodes()

            # 从图中提取双键边信息
            double_bond_edges = set()
            for u, v, data in G.edges(data=True):
                if data.get('bond_type') == 'double':
                    double_bond_edges.add((min(u, v), max(u, v)))

            # 构建 RDKit 分子
            mol = Chem.RWMol()
            for i in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = (min(u, v), max(u, v))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)
                if bond_key in double_bond_edges:
                    mol.AddBond(u, v, BondType.DOUBLE)
                else:
                    mol.AddBond(u, v, BondType.SINGLE)

            # 设置显式氢
            for node in sorted(G.nodes()):
                total_bonds = 0
                for nb in G.neighbors(node):
                    bond_type = G[node][nb].get('bond_type', 'single')
                    total_bonds += 2 if bond_type == 'double' else 1
                h_count = 4 - total_bonds
                if h_count > 0:
                    mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)

            # 单环多烯烃：先尝试3D嵌入+MMFF优化，失败回退2D坐标
            import math as _math
            best_mol = None
            best_max_bond = float('inf')
            for seed in range(50):
                mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                result = AllChem.EmbedMolecule(mol_trial, params)
                if result == -1:
                    result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                    if result2 == -1:
                        continue
                try:
                    AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                except Exception:
                    pass
                conf = mol_trial.GetConformer()
                max_bond = 0.0
                for u, v in G.edges():
                    if u < n_carbons and v < n_carbons:
                        p1 = conf.GetAtomPosition(u)
                        p2 = conf.GetAtomPosition(v)
                        dist = _math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                        max_bond = max(max_bond, dist)
                if max_bond < best_max_bond:
                    best_max_bond = max_bond
                    best_mol = Chem.Mol(mol_trial.ToBinary())
            if best_mol is not None:
                mol = best_mol
            else:
                AllChem.Compute2DCoords(mol)

            # 提取坐标
            conf = mol.GetConformer()
            c_coords = {}
            h_coords = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coord = (float(pos.x), float(pos.y), float(pos.z))
                if atom.GetSymbol() == 'C':
                    c_coords[atom.GetIdx()] = coord
                else:
                    h_coords.append(coord)

            # 验证环内键长是否合理
            import math as _math
            _cycles = nx.cycle_basis(G)
            _ring_distorted = False
            for _cycle in _cycles:
                for _ci in range(len(_cycle)):
                    _u = _cycle[_ci]
                    _v = _cycle[(_ci + 1) % len(_cycle)]
                    if _u in c_coords and _v in c_coords:
                        _dist = _math.sqrt(sum((a - b) ** 2
                                               for a, b in zip(c_coords[_u], c_coords[_v])))
                        if _dist > 2.0:
                            _ring_distorted = True
                            break
                if _ring_distorted:
                    break

            if _ring_distorted:
                # 2D坐标仍然畸变，回退到手动环坐标
                try:
                    main_cycle_fb = nx.cycle_basis(G)[0]
                    c_coords = self._build_ring_coords_with_bond_types(G, main_cycle_fb)
                    h_coords = self._generate_hydrogens_for_multi_bond(G, c_coords, main_cycle_fb)
                except Exception:
                    _n = G.number_of_nodes()
                    c_coords = {}
                    h_coords = []
                    for idx, node in enumerate(sorted(G.nodes())):
                        angle = 2 * _math.pi * idx / _n
                        c_coords[node] = (_math.cos(angle), _math.sin(angle), 0.0)
                    for idx, node in enumerate(sorted(G.nodes())):
                        total_bonds = 0
                        for nb in G.neighbors(node):
                            bond_type = G[node][nb].get('bond_type', 'single')
                            total_bonds += 2 if bond_type == 'double' else 1
                        h_count = 4 - total_bonds
                        cx, cy, cz = c_coords[node]
                        for j in range(h_count):
                            ha = 2 * _math.pi * j / max(h_count, 1) + idx * 0.5
                            h_coords.append((cx + 0.6 * _math.cos(ha), cy + 0.6 * _math.sin(ha), cz))

            # 修正累积双键共线问题
            c_coords = self._fix_cumulated_diene_coords(G, c_coords)

            # 获取环信息
            main_cycle = nx.cycle_basis(G)[0]
            cycle_set = set(main_cycle)

            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 绘制C-C键（双键用红色平行线，单键用黑色线）
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in c_coords and neighbor in c_coords:
                            is_double = G[node][neighbor].get('bond_type') == 'double'
                            if is_double:
                                # 双键：主线 + 偏移缩短线
                                p1 = np.array(c_coords[node])
                                p2 = np.array(c_coords[neighbor])
                                direction = p2 - p1
                                length = np.linalg.norm(direction)
                                if length > 1e-8:
                                    up = np.array([0, 0, 1])
                                    perp = np.cross(direction, up)
                                    perp_norm = np.linalg.norm(perp)
                                    if perp_norm < 1e-8:
                                        perp = np.cross(direction, np.array([0, 1, 0]))
                                        perp_norm = np.linalg.norm(perp)
                                    perp = perp / perp_norm * length * 0.04
                                    # 主线
                                    ax.plot(
                                        [p1[0], p2[0]],
                                        [p1[1], p2[1]],
                                        [p1[2], p2[2]],
                                        color='red', linewidth=3
                                    )
                                    # 第二条缩短偏移线
                                    t1, t2 = 0.25, 0.75
                                    sp1 = p1 + direction * t1 + perp
                                    sp2 = p1 + direction * t2 + perp
                                    ax.plot(
                                        [sp1[0], sp2[0]],
                                        [sp1[1], sp2[1]],
                                        [sp1[2], sp2[2]],
                                        color='red', linewidth=3
                                    )
                                else:
                                    ax.plot(
                                        [c_coords[node][0], c_coords[neighbor][0]],
                                        [c_coords[node][1], c_coords[neighbor][1]],
                                        [c_coords[node][2], c_coords[neighbor][2]],
                                        color='red', linewidth=3
                                    )
                            else:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='black', linewidth=2
                                )
                        drawn_bonds.add(bond_key)

            # 绘制碳原子（双键碳用红色，环碳用蓝色，其他用黑色）
            double_carbons = set()
            for u, v, d in G.edges(data=True):
                if d.get('bond_type') == 'double':
                    double_carbons.add(u)
                    double_carbons.add(v)

            for n in sorted(c_coords.keys()):
                if n in double_carbons:
                    color = 'red'
                elif n in cycle_set:
                    color = 'blue'
                else:
                    color = 'black'
                ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                          c=color, s=150, zorder=5)

            # 绘制氢原子和碳氢键
            if h_coords:
                hx = [h[0] for h in h_coords]
                hy = [h[1] for h in h_coords]
                hz = [h[2] for h in h_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

                # 找每个氢连接的碳原子
                h_to_c = {}
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H':
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() == 'C':
                                h_idx = atom.GetIdx()
                                c_idx = neighbor.GetIdx()
                                h_to_c[h_idx] = c_idx

                h_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H':
                        if h_count < len(h_coords):
                            c_idx = h_to_c.get(atom.GetIdx())
                            if c_idx is not None and c_idx in c_coords:
                                h_coord = h_coords[h_count]
                                c_coord = c_coords[c_idx]
                                ax.plot(
                                    [c_coord[0], h_coord[0]],
                                    [c_coord[1], h_coord[1]],
                                    [c_coord[2], h_coord[2]],
                                    'gray', linewidth=1, alpha=0.7, zorder=4
                                )
                        h_count += 1

            # 标记环原子编号
            for i, node in enumerate(main_cycle):
                if node in c_coords:
                    coord = c_coords[node]
                    ax.text(coord[0], coord[1], coord[2], f'{i+1}', fontsize=9, color='blue')

            # 图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2, label='C-C 单键'),
                Line2D([0], [0], color='red', linewidth=3, label='C=C 双键'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='环碳'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='双键碳'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")
            ax.set_zlabel("Z (Å)")
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("可视化错误", f"单环多烯烃可视化失败: {str(e)}")

    def _visualize_alkenyl(self, G, title):
        """可视化单个烯炔烃异构体（含双键+三键，使用RDKit 3D坐标 + matplotlib）"""
        try:
            import networkx as nx
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

            n_carbons = G.number_of_nodes()

            # 从图中提取键类型信息
            double_bond_edges = set()
            triple_bond_edges = set()
            double_nodes = set()
            triple_nodes = set()
            for u, v, data in G.edges(data=True):
                bt = data.get('bond_type', 'single')
                key = (min(u, v), max(u, v))
                if bt == 'double':
                    double_bond_edges.add(key)
                    double_nodes.add(u)
                    double_nodes.add(v)
                elif bt == 'triple':
                    triple_bond_edges.add(key)
                    triple_nodes.add(u)
                    triple_nodes.add(v)

            # 构建 RDKit 分子
            mol = Chem.RWMol()
            for i in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = (min(u, v), max(u, v))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)
                if bond_key in triple_bond_edges:
                    mol.AddBond(u, v, BondType.TRIPLE)
                elif bond_key in double_bond_edges:
                    mol.AddBond(u, v, BondType.DOUBLE)
                else:
                    mol.AddBond(u, v, BondType.SINGLE)

            # 设置显式氢
            for node in sorted(G.nodes()):
                total_bonds = 0
                for nb in G.neighbors(node):
                    bt = G[node][nb].get('bond_type', 'single')
                    if bt == 'double':
                        total_bonds += 2
                    elif bt == 'triple':
                        total_bonds += 3
                    else:
                        total_bonds += 1
                h_count = 4 - total_bonds
                if h_count > 0:
                    mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)

            # 3D坐标嵌入
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass

            # 提取坐标
            conf = mol.GetConformer()
            c_coords = {}
            h_coords = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coord = (float(pos.x), float(pos.y), float(pos.z))
                if atom.GetSymbol() == 'C':
                    c_coords[atom.GetIdx()] = coord
                else:
                    h_coords.append(coord)

            # 修正累积双键段(C=C=C)的共线问题
            c_coords = self._fix_cumulated_diene_coords(G, c_coords)

            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 绘制C-C键（双键橙色双线，三键红色三线，单键黑色）
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in c_coords and neighbor in c_coords:
                            p1 = np.array(c_coords[node])
                            p2 = np.array(c_coords[neighbor])
                            direction = p2 - p1
                            length = np.linalg.norm(direction)

                            if bond_key in triple_bond_edges:
                                # 三键：主线 + 两条偏移缩短线
                                if length > 1e-8:
                                    up = np.array([0, 0, 1])
                                    perp = np.cross(direction, up)
                                    perp_norm = np.linalg.norm(perp)
                                    if perp_norm < 1e-8:
                                        perp = np.cross(direction, np.array([0, 1, 0]))
                                        perp_norm = np.linalg.norm(perp)
                                    perp = perp / perp_norm * length * 0.04
                                    # 主线
                                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                            color='#8B008B', linewidth=3)
                                    # 两条缩短偏移线
                                    t1, t2 = 0.25, 0.75
                                    for sign in [1, -1]:
                                        sp1 = p1 + direction * t1 + perp * sign
                                        sp2 = p1 + direction * t2 + perp * sign
                                        ax.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]], [sp1[2], sp2[2]],
                                                color='#8B008B', linewidth=2)
                                else:
                                    ax.plot(
                                        [c_coords[node][0], c_coords[neighbor][0]],
                                        [c_coords[node][1], c_coords[neighbor][1]],
                                        [c_coords[node][2], c_coords[neighbor][2]],
                                        color='#8B008B', linewidth=3
                                    )
                            elif bond_key in double_bond_edges:
                                # 双键：主线 + 偏移缩短线
                                if length > 1e-8:
                                    up = np.array([0, 0, 1])
                                    perp = np.cross(direction, up)
                                    perp_norm = np.linalg.norm(perp)
                                    if perp_norm < 1e-8:
                                        perp = np.cross(direction, np.array([0, 1, 0]))
                                        perp_norm = np.linalg.norm(perp)
                                    perp = perp / perp_norm * length * 0.04
                                    # 主线
                                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                            color='#FF6600', linewidth=3)
                                    # 第二条缩短偏移线
                                    t1, t2 = 0.25, 0.75
                                    sp1 = p1 + direction * t1 + perp
                                    sp2 = p1 + direction * t2 + perp
                                    ax.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]], [sp1[2], sp2[2]],
                                            color='#FF6600', linewidth=3)
                                else:
                                    ax.plot(
                                        [c_coords[node][0], c_coords[neighbor][0]],
                                        [c_coords[node][1], c_coords[neighbor][1]],
                                        [c_coords[node][2], c_coords[neighbor][2]],
                                        color='#FF6600', linewidth=3
                                    )
                            else:
                                # 单键
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='black', linewidth=2
                                )
                        drawn_bonds.add(bond_key)

            # 绘制碳原子（不同杂化不同颜色）
            for n in sorted(c_coords.keys()):
                if n in triple_nodes:
                    color = '#CC0000'   # sp碳 - 红色
                elif n in double_nodes:
                    color = '#FF6600'   # sp2碳 - 橙色
                else:
                    color = '#333333'   # sp3碳 - 深灰色
                ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                          c=color, s=150, zorder=5)

            # 绘制氢原子和碳氢键
            if h_coords:
                hx = [h[0] for h in h_coords]
                hy = [h[1] for h in h_coords]
                hz = [h[2] for h in h_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

                # 找每个氢连接的碳原子
                h_to_c = {}
                h_list = []
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H':
                        h_list.append(atom.GetIdx())
                        for nb in atom.GetNeighbors():
                            if nb.GetSymbol() == 'C':
                                h_to_c[atom.GetIdx()] = nb.GetIdx()

                for i, h_idx in enumerate(h_list):
                    if i < len(h_coords) and h_idx in h_to_c:
                        c_idx = h_to_c[h_idx]
                        if c_idx in c_coords:
                            h_c = h_coords[i]
                            c_c = c_coords[c_idx]
                            ax.plot(
                                [c_c[0], h_c[0]],
                                [c_c[1], h_c[1]],
                                [c_c[2], h_c[2]],
                                'gray', linewidth=1, alpha=0.7, zorder=4
                            )

            # 标记碳原子编号（带杂化标注）
            for n in sorted(c_coords.keys()):
                coord = c_coords[n]
                label = f'C{n+1}'
                if n in triple_nodes:
                    label += '(sp)'
                elif n in double_nodes:
                    label += '(sp2)'
                ax.text(coord[0], coord[1], coord[2], label, fontsize=7, color='blue')

            # 图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#CC0000', linewidth=3, label='C\u2261C'),
                Line2D([0], [0], color='#FF6600', linewidth=3, label='C=C'),
                Line2D([0], [0], color='black', linewidth=2, label='C-C'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            ax.set_xlabel("X (A)")
            ax.set_ylabel("Y (A)")
            ax.set_zlabel("Z (A)")
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("可视化错误", f"烯炔烃可视化失败: {str(e)}")

    def _visualize_polyene(self, G, title):
        """可视化单个多烯烃异构体（含多个双键，使用RDKit 3D坐标 + matplotlib）"""
        try:
            import networkx as nx
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

            n_carbons = G.number_of_nodes()

            # 收集双键信息
            double_bond_edges = set()
            double_nodes = set()
            for u, v, data in G.edges(data=True):
                bt = data.get('bond_type', 'single')
                key = (min(u, v), max(u, v))
                if bt == 'double':
                    double_bond_edges.add(key)
                    double_nodes.add(u)
                    double_nodes.add(v)

            # 构建 RDKit 分子
            mol = Chem.RWMol()
            for i in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = (min(u, v), max(u, v))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)
                if bond_key in double_bond_edges:
                    mol.AddBond(u, v, BondType.DOUBLE)
                else:
                    mol.AddBond(u, v, BondType.SINGLE)

            # 设置显式氢
            for node in sorted(G.nodes()):
                total_bonds = 0
                for nb in G.neighbors(node):
                    bt = G[node][nb].get('bond_type', 'single')
                    if bt == 'double':
                        total_bonds += 2
                    else:
                        total_bonds += 1
                h_count = 4 - total_bonds
                if h_count > 0:
                    mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)

            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass

            # 提取坐标
            conf = mol.GetConformer()
            c_coords = {}
            h_coords = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coord = (float(pos.x), float(pos.y), float(pos.z))
                if atom.GetSymbol() == 'C':
                    c_coords[atom.GetIdx()] = coord
                else:
                    h_coords.append(coord)

            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 绘制键
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in c_coords and neighbor in c_coords:
                            p1 = np.array(c_coords[node])
                            p2 = np.array(c_coords[neighbor])
                            direction = p2 - p1
                            length = np.linalg.norm(direction)

                            if bond_key in double_bond_edges:
                                if length > 1e-8:
                                    up = np.array([0, 0, 1])
                                    perp = np.cross(direction, up)
                                    perp_norm = np.linalg.norm(perp)
                                    if perp_norm < 1e-8:
                                        perp = np.cross(direction, np.array([0, 1, 0]))
                                        perp_norm = np.linalg.norm(perp)
                                    perp = perp / perp_norm * length * 0.04
                                    # 主线
                                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                            color='#FF6600', linewidth=3)
                                    # 第二条缩短偏移线
                                    t1, t2 = 0.25, 0.75
                                    sp1 = p1 + direction * t1 + perp
                                    sp2 = p1 + direction * t2 + perp
                                    ax.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]], [sp1[2], sp2[2]],
                                            color='#FF6600', linewidth=3)
                            else:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='black', linewidth=2
                                )
                        drawn_bonds.add(bond_key)

            # 绘制碳原子
            for n in sorted(c_coords.keys()):
                color = '#FF6600' if n in double_nodes else '#333333'
                ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                          c=color, s=150, zorder=5)

            # 绘制氢原子
            if h_coords:
                hx = [h[0] for h in h_coords]
                hy = [h[1] for h in h_coords]
                hz = [h[2] for h in h_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

                h_to_c = {}
                h_list = []
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H':
                        h_list.append(atom.GetIdx())
                        for nb in atom.GetNeighbors():
                            if nb.GetSymbol() == 'C':
                                h_to_c[atom.GetIdx()] = nb.GetIdx()

                for i, h_idx in enumerate(h_list):
                    if i < len(h_coords) and h_idx in h_to_c:
                        c_idx = h_to_c[h_idx]
                        if c_idx in c_coords:
                            h_c = h_coords[i]
                            c_c = c_coords[c_idx]
                            ax.plot(
                                [c_c[0], h_c[0]],
                                [c_c[1], h_c[1]],
                                [c_c[2], h_c[2]],
                                'gray', linewidth=1, alpha=0.7, zorder=4
                            )

            # 标记碳原子编号
            for n in sorted(c_coords.keys()):
                coord = c_coords[n]
                label = f'C{n+1}'
                if n in double_nodes:
                    double_count = sum(1 for nb in G.neighbors(n) if G[n][nb].get('bond_type')=='double')
                    if double_count >= 2:
                        label += '(cum)'
                    else:
                        label += '(sp2)'
                ax.text(coord[0], coord[1], coord[2], label, fontsize=7, color='blue')

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#FF6600', linewidth=3, label='C=C'),
                Line2D([0], [0], color='black', linewidth=2, label='C-C'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            ax.set_xlabel("X (A)")
            ax.set_ylabel("Y (A)")
            ax.set_zlabel("Z (A)")
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("可视化错误", f"多烯烃可视化失败: {str(e)}")

    def _build_cycloalkane_rdkit(self, G):
        """使用 RDKit 为环烷烃构建高质量3D坐标
        
        Returns:
            (c_coords, h_coords, h_to_c) 或 (None, None, None) 如果失败
            - c_coords: {碳原子索引: (x,y,z)}
            - h_coords: [(x,y,z), ...] 氢原子坐标列表
            - h_to_c: {氢原子列表索引: 对应的碳原子索引}
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            import networkx as nx
            import numpy as np

            n_carbons = G.number_of_nodes()

            # 构建 RDKit 分子
            mol = Chem.RWMol()
            for i in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = (min(u, v), max(u, v))
                if bond_key not in added_bonds:
                    mol.AddBond(u, v, BondType.SINGLE)
                    added_bonds.add(bond_key)

            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)

            # 环烷烃：先尝试3D嵌入+MMFF优化，失败回退2D坐标
            import math as _math
            best_mol = None
            best_max_bond = float('inf')
            for seed in range(50):
                mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                result = AllChem.EmbedMolecule(mol_trial, params)
                if result == -1:
                    result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                    if result2 == -1:
                        continue
                try:
                    AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                except Exception:
                    pass
                conf = mol_trial.GetConformer()
                max_bond = 0.0
                for u, v in G.edges():
                    if u < n_carbons and v < n_carbons:
                        p1 = conf.GetAtomPosition(u)
                        p2 = conf.GetAtomPosition(v)
                        dist = _math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                        max_bond = max(max_bond, dist)
                if max_bond < best_max_bond:
                    best_max_bond = max_bond
                    best_mol = Chem.Mol(mol_trial.ToBinary())
            if best_mol is not None:
                mol = best_mol
            else:
                AllChem.Compute2DCoords(mol)

            # 提取坐标
            c_coords = {}
            h_coords = []
            h_to_c = {}

            for atom in mol.GetAtoms():
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                coord = (float(pos.x), float(pos.y), float(pos.z))
                if atom.GetSymbol() == 'C':
                    c_coords[atom.GetIdx()] = coord
                elif atom.GetSymbol() == 'H':
                    h_idx = len(h_coords)
                    h_coords.append(coord)
                    # 找到相邻的碳原子
                    for nb in atom.GetNeighbors():
                        if nb.GetSymbol() == 'C':
                            h_to_c[h_idx] = nb.GetIdx()
                            break

            return c_coords, h_coords, h_to_c

        except Exception:
            return None, None, None

    def _visualize_cycloalkane(self, G, title):
        """可视化单个环烷烃异构体（优先使用 RDKit 构建高质量3D坐标）"""
        try:
            import networkx as nx
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

            # 尝试 RDKit 生成坐标
            c_coords, h_coords, h_to_c = self._build_cycloalkane_rdkit(G)

            if c_coords is None:
                # RDKit 失败，尝试 PolycycloalkaneGenerator 的多种子策略
                mol = self.multcycloalkane_generator.graph_to_rdkit_mol(G, optimize=True)
                if mol is not None and mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    c_coords = {}
                    h_coords = []
                    h_to_c = {}
                    for atom in mol.GetAtoms():
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        coord = [float(pos.x), float(pos.y), float(pos.z)]
                        if atom.GetSymbol() == 'C':
                            c_coords[atom.GetIdx()] = coord
                        else:
                            h_to_c[len(h_coords)] = next((a.GetIdx() for a in atom.GetNeighbors() if a.GetSymbol() == 'C'), 0)
                            h_coords.append(coord)
                else:
                    # 最终回退：2D坐标
                    import math
                    c_coords = {}
                    n_nodes = G.number_of_nodes()
                    for node in G.nodes():
                        angle = 2 * math.pi * node / n_nodes
                        c_coords[node] = [math.cos(angle), math.sin(angle), 0.0]
                    h_coords = []
                    h_to_c = {}

            ring_set = set(nx.cycle_basis(G)[0])

            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 绘制C-C键
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in c_coords and neighbor in c_coords:
                            ax.plot(
                                [c_coords[node][0], c_coords[neighbor][0]],
                                [c_coords[node][1], c_coords[neighbor][1]],
                                [c_coords[node][2], c_coords[neighbor][2]],
                                'k-', linewidth=2
                            )
                        drawn_bonds.add(bond_key)

            # 绘制碳原子（环碳用红色）
            cx = [c_coords[n][0] for n in sorted(c_coords.keys())]
            cy = [c_coords[n][1] for n in sorted(c_coords.keys())]
            cz = [c_coords[n][2] for n in sorted(c_coords.keys())]
            ring_colors = ['red' if n in ring_set else 'black' for n in sorted(c_coords.keys())]
            ax.scatter(cx, cy, cz, c=ring_colors, s=150, zorder=5)

            # 绘制氢原子和碳氢键
            if len(h_coords) > 0:
                hx = [h[0] for h in h_coords]
                hy = [h[1] for h in h_coords]
                hz = [h[2] for h in h_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

                # 绘制碳氢键
                for i, h_coord in enumerate(h_coords):
                    carbon_node = h_to_c.get(i)
                    if carbon_node is not None and carbon_node in c_coords:
                        c_c = c_coords[carbon_node]
                        ax.plot(
                            [c_c[0], h_coord[0]],
                            [c_c[1], h_coord[1]],
                            [c_c[2], h_coord[2]],
                            'gray', linewidth=1, alpha=0.7, zorder=4
                        )

            # 标记环原子编号
            main_cycle = nx.cycle_basis(G)[0]
            for i, node in enumerate(main_cycle):
                if node in c_coords:
                    coord = c_coords[node]
                    ax.text(coord[0], coord[1], coord[2], f'{i+1}', fontsize=9, color='blue')

            ax.set_xlabel("X (A)")
            ax.set_ylabel("Y (A)")
            ax.set_zlabel("Z (A)")

            # 调整视角以获得更好的观察效果
            ax.view_init(elev=20, azim=45)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("可视化错误", f"环烷烃可视化失败: {str(e)}")

    def _visualize_multcycloalkane(self, G, title):
        """可视化多环烷烃异构体（优先使用 RDKit 构建高质量3D坐标）"""
        try:
            import math
            import networkx as nx
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

            n_carbons = G.number_of_nodes()

            # 构建 RDKit 分子
            mol = Chem.RWMol()
            for i in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = (min(u, v), max(u, v))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)
                mol.AddBond(u, v, BondType.SINGLE)

            for node in sorted(G.nodes()):
                h_count = 4 - G.degree(node)
                if h_count > 0:
                    mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

            mol = mol.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass

            mol = Chem.AddHs(mol)

            # 多环烷烃：多种子选择策略，选择最大C-C键长最短的构象
            best_mol = None
            best_max_bond = float('inf')
            for seed in range(50):
                mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                result = AllChem.EmbedMolecule(mol_trial, params)
                if result == -1:
                    result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                    if result2 == -1:
                        continue
                try:
                    AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                except Exception:
                    pass
                # 计算最大C-C键长
                conf_t = mol_trial.GetConformer()
                max_bond = 0.0
                for u, v in G.edges():
                    if u < n_carbons and v < n_carbons:
                        p1 = conf_t.GetAtomPosition(u)
                        p2 = conf_t.GetAtomPosition(v)
                        dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
                        max_bond = max(max_bond, dist)
                if max_bond < best_max_bond:
                    best_max_bond = max_bond
                    best_mol = Chem.Mol(mol_trial.ToBinary())

            if best_mol is not None:
                mol = best_mol
            else:
                # 所有种子都失败，回退到2D坐标
                AllChem.Compute2DCoords(mol)

            # 提取坐标
            conf = mol.GetConformer()
            c_coords = {}
            h_coords = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coord = (float(pos.x), float(pos.y), float(pos.z))
                if atom.GetSymbol() == 'C':
                    c_coords[atom.GetIdx()] = coord
                else:
                    h_coords.append(coord)

            # 如果RDKit坐标失败，回退到基于图的方法
            if not c_coords:
                c_coords, h_coords = self._fallback_coords_from_graph(G)

            # 获取环信息
            cycles = nx.cycle_basis(G)

            # 为不同环分配不同颜色
            ring_colors = ['#CC0000', '#0066CC', '#009933', '#CC6600', '#6600CC', '#009999']
            node_ring_color = {}
            ring_edges = set()
            for ci, cycle in enumerate(cycles):
                color = ring_colors[ci % len(ring_colors)]
                cycle_set = set(cycle)
                for node in cycle:
                    if node not in node_ring_color:
                        node_ring_color[node] = color
                    # 标记环内边
                    for j in range(len(cycle)):
                        u = cycle[j]
                        v = cycle[(j + 1) % len(cycle)]
                        ring_edges.add(tuple(sorted([u, v])))

            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontsize=14, fontweight='bold')

            # 绘制 C-C 键
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in c_coords and neighbor in c_coords:
                            if bond_key in ring_edges:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='#CC0000', linewidth=2.5
                                )
                            else:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='black', linewidth=2
                                )
                        drawn_bonds.add(bond_key)

            # 绘制碳原子（不同环用不同颜色）
            for n in sorted(c_coords.keys()):
                if n in node_ring_color:
                    color = node_ring_color[n]
                else:
                    color = '#333333'
                ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                           c=color, s=150, zorder=5)

            # 绘制氢原子和碳氢键
            if h_coords:
                h_to_c = {}
                h_idx = 0
                for bond in mol.GetBonds():
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    if a1.GetSymbol() == 'C' and a2.GetSymbol() == 'H':
                        h_to_c[h_idx] = a1.GetIdx()
                        h_idx += 1
                    elif a1.GetSymbol() == 'H' and a2.GetSymbol() == 'C':
                        h_to_c[h_idx] = a2.GetIdx()
                        h_idx += 1

                hx = [h[0] for h in h_coords]
                hy = [h[1] for h in h_coords]
                hz = [h[2] for h in h_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

                for i, h_coord in enumerate(h_coords):
                    if i in h_to_c and h_to_c[i] in c_coords:
                        c_coord = c_coords[h_to_c[i]]
                        ax.plot(
                            [c_coord[0], h_coord[0]],
                            [c_coord[1], h_coord[1]],
                            [c_coord[2], h_coord[2]],
                            color='gray', linewidth=1, alpha=0.7
                        )

            # 标记环原子编号
            for ci, cycle in enumerate(cycles):
                for i, node in enumerate(cycle):
                    if node in c_coords:
                        coord = c_coords[node]
                        ax.text(coord[0], coord[1], coord[2],
                                f'{i+1}', fontsize=9, color='blue')

            # 图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#CC0000', linewidth=2.5, label='环内 C-C'),
                Line2D([0], [0], color='black', linewidth=2, label='环外 C-C'),
                Line2D([0], [0], color='gray', linewidth=1, label='C-H'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")
            ax.set_zlabel("Z (Å)")
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("可视化错误", f"多环烷烃可视化失败: {str(e)}")

    def _visualize_alkynes_batch(self, graphs, title):
        """批量可视化炔烃异构体"""
        n = len(graphs)
        if n > 20:
            n_display = 20
            messagebox.showinfo("提示", f"共有 {n} 个异构体，将显示前 20 个")
        else:
            n_display = n

        cols = min(4, n_display)
        rows = (n_display + cols - 1) // cols

        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        for i in range(n_display):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            G = graphs[i]

            try:
                # 获取三键信息
                triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']

                # 计算坐标
                coords = self.alkyne_visualizer._calculate_coordinates(G)
                if not coords:
                    continue

                # 生成氢原子
                h_coords = self.alkyne_visualizer._generate_hydrogen_coords(G, coords)

                # 绘制分子
                self.alkyne_visualizer._draw_molecule(ax, G, coords, h_coords, triple_edges)

                # 设置标题
                if triple_edges:
                    u, v = triple_edges[0]
                    ax.set_title(f"C{u+1}≡C{v+1}", fontsize=10, pad=10)
                else:
                    ax.set_title(f"#{i+1}", fontsize=10, pad=10)

                # 设置等比例
                all_coords = list(coords.values()) + h_coords
                self.alkyne_visualizer._set_equal_aspect_ratio(ax, all_coords)

            except Exception as e:
                ax.set_title(f"#{i+1} (error)", fontsize=10)
                ax.text(0.5, 0.5, 0.5, str(e), transform=ax.transAxes)

        plt.tight_layout()
        plt.show()

    def _visualize_cycloalkanes_batch(self, graphs, title):
        """批量可视化环烷烃异构体（优先使用 RDKit 构建高质量3D坐标）"""
        n = len(graphs)
        if n > 20:
            n_display = 20
            messagebox.showinfo("提示", f"共有 {n} 个异构体，将显示前 20 个")
        else:
            n_display = n

        cols = min(4, n_display)
        rows = (n_display + cols - 1) // cols

        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        import networkx as nx

        for i in range(n_display):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            G = graphs[i]

            try:
                c_coords, h_coords, h_to_c = self._build_cycloalkane_rdkit(G)
                if c_coords is None:
                    # RDKit 失败，尝试 PolycycloalkaneGenerator 的多种子策略
                    mol = self.multcycloalkane_generator.graph_to_rdkit_mol(G, optimize=True)
                    if mol is not None and mol.GetNumConformers() > 0:
                        conf = mol.GetConformer()
                        c_coords = {}
                        h_coords = []
                        h_to_c = {}
                        for atom in mol.GetAtoms():
                            pos = conf.GetAtomPosition(atom.GetIdx())
                            coord = [float(pos.x), float(pos.y), float(pos.z)]
                            if atom.GetSymbol() == 'C':
                                c_coords[atom.GetIdx()] = coord
                            else:
                                h_to_c[len(h_coords)] = next((a.GetIdx() for a in atom.GetNeighbors() if a.GetSymbol() == 'C'), 0)
                                h_coords.append(coord)
                    else:
                        import math
                        c_coords = {}
                        n_nodes = G.number_of_nodes()
                        for node in G.nodes():
                            angle = 2 * math.pi * node / n_nodes
                            c_coords[node] = [math.cos(angle), math.sin(angle), 0.0]
                        h_coords = []
                        h_to_c = {}

                ring_set = set(nx.cycle_basis(G)[0])

                # 绘制C-C键
                drawn_bonds = set()
                for node in G.nodes():
                    for neighbor in G.neighbors(node):
                        bond_key = tuple(sorted([node, neighbor]))
                        if bond_key not in drawn_bonds:
                            if node in c_coords and neighbor in c_coords:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    'k-', linewidth=1.5
                                )
                            drawn_bonds.add(bond_key)

                # 绘制碳原子（环碳用红色）
                for n in sorted(c_coords.keys()):
                    color = 'red' if n in ring_set else 'black'
                    ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                               c=color, s=80, zorder=5)

                # 绘制氢原子
                for i2, h_coord in enumerate(h_coords):
                    ax.scatter([h_coord[0]], [h_coord[1]], [h_coord[2]],
                              c='white', edgecolors='gray', s=40, zorder=5)

                ax.set_title(f"#{i+1}", fontsize=10)

            except Exception:
                # 回退方案：使用简单的2D表示
                pos = nx.spring_layout(G, seed=42)
                for node in G.nodes():
                    x, y = pos[node]
                    ax.scatter([x], [y], [0], c='black', s=50)
                for u, v in G.edges():
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], [0, 0], 'k-', linewidth=1)
                ax.set_title(f"#{i+1} (fallback)", fontsize=10)

        plt.tight_layout()
        plt.show()

    def _show_error(self, error_msg):
        """显示错误信息"""
        if self.window_closed:
            return

        try:
            self.is_generating = False
            self.generate_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
            self.progress_label.config(text="错误")
            self.status_var.set("生成失败")
            messagebox.showerror("错误", f"可视化时出错:\n{error_msg}")
            traceback.print_exc()
        except tk.TclError:
            pass  # 窗口已关闭


def main():
    """主函数"""
    root = tk.Tk()

    # 配置中文显示
    try:
        root.tk.call('encoding', 'system', 'utf-8')
    except:
        pass

    app = MoleculeApp(root)
    root.mainloop()


if __name__ == "__main__":
    # PyInstaller 打包后 multiprocessing 必须在 __main__ 入口调用 freeze_support
    # 否则 spawn 模式的子进程会重复执行主模块，导致无限递归或卡死
    import multiprocessing
    multiprocessing.freeze_support()
    main()
