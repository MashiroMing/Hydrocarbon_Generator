"""
分子式解析与异构体生成分发工具

职责：解析分子式 → 识别分子类型 → 分发到对应生成器
        键线式（skeletal formula）渲染
"""

import re
import io
import tempfile
import os

from original_programs.alkane_isomer_visualizer import AlkaneIsomerGenerator, AlkaneIsomerVisualizer
from original_programs.alkene_visualizer import AlkeneIsomerVisualizer as AlkeneViz
from original_programs.alkyne import AlkyneIsomerGenerator, AlkyneIsomerVisualizer
from core_modules.cycloalkane_app import CycloalkaneGenerator, CycloalkaneBuilder
from core_modules.cycloalkene_generator import CycloalkeneGenerator
from core_modules.cyclopolyene_generator import CyclopolyeneGenerator
from original_programs.alkenyl_generator import AlkenylIsomerGenerator
from original_programs.polyene_generator import PolyeneIsomerGenerator

# 中文名 / 筛选用中文名 / 英文名
MOL_NAMES_CN = {
    'alkane': '烷烃',
    'alkene': '烯烃',
    'alkyne': '炔烃',
    'diene': '二烯烃',
    'cycloalkane': '环烷烃',
    'cycloalkene': '单环烯烃',
    'alkenyl': '烯炔烃',
    'triene': '三烯烃',
    'tetraene': '四烯烃',
    'polyene': '多烯烃',
    'cyclopolyene': '单环多烯烃',
}

MOL_NAMES_FILTER = {
    'alkane': '饱和链烃',
    'alkene': '单烯烃',
    'alkyne': '单炔烃',
    'diene': '二烯烃',
    'cycloalkane': '单环烷烃',
    'cycloalkene': '单环烯烃',
    'alkenyl': '烯炔烃',
    'triene': '三烯烃',
    'tetraene': '四烯烃',
    'polyene': '多烯烃',
    'cyclopolyene': '单环多烯烃',
}

MOL_NAMES_EN = {
    'alkane': 'Alkane',
    'alkene': 'Alkene',
    'alkyne': 'Alkyne',
    'diene': 'Diene',
    'cycloalkane': 'Cycloalkane',
    'cycloalkene': 'Cycloalkene',
    'alkenyl': 'Alkenyl',
    'triene': 'Triene',
    'tetraene': 'Tetraene',
    'polyene': 'Polyene',
    'cyclopolyene': 'Cyclopolyene',
}

# 不支持的分子式错误提示
_UNSUPPORTED_MSG = (
    "分子式 C{n}H{m} 不是本软件支持的类型\n"
    "支持的类型:\n"
    "• 烷烃 CnH2n+2\n"
    "• 单烯烃/单环烷烃 CnH2n (n>=2)\n"
    "• 单炔烃/二烯烃/单环烯烃 CnH2n-2 (n>=2)\n"
    "• 烯炔烃/三烯烃/单环二烯烃 CnH2n-4 (n>=4)\n"
    "• 四烯烃/单环三烯烃 CnH2n-6 (n>=5)\n"
    "• k-烯烃/单环多烯烃 CnH2n-2k (k>=5)"
)


def parse_molecule_input(formula_str):
    """解析分子式，返回分子类型列表、碳氢原子数及错误信息"""
    formula_str = formula_str.strip()
    if not formula_str:
        return None, None, None, "请输入分子式"

    pattern = r'^C(\d+)H(\d+)$'
    match = re.match(pattern, formula_str, re.IGNORECASE)
    if not match:
        return None, None, None, "分子式格式不正确，请使用如 C5H12、C4H8 的格式"

    n = int(match.group(1))
    m = int(match.group(2))

    if n < 1:
        return None, None, None, "碳原子数必须 >= 1"
    if m < 0:
        return None, None, None, "氢原子数不能为负"

    mol_types = []

    # CnH2n+2: 烷烃
    if m == 2 * n + 2:
        if n >= 1:
            mol_types.append('alkane')

    # CnH2n: 单烯烃 或 单环烷烃
    elif m == 2 * n:
        if n >= 3:
            mol_types.extend(['alkene', 'cycloalkane'])
        elif n >= 2:
            mol_types.append('alkene')

    # CnH2n-2: 单炔烃 或 二烯烃 或 单环烯烃
    elif m == 2 * n - 2:
        if n >= 3:
            mol_types.extend(['alkyne', 'diene', 'cycloalkene'])
        elif n >= 2:
            mol_types.append('alkyne')

    # CnH2n-4: 烯炔烃 或 三烯烃 或 单环二烯烃
    elif m == 2 * n - 4:
        if n >= 4:
            mol_types.append('alkenyl')
        if n >= 4:
            mol_types.append('triene')
        if n >= 4:
            mol_types.append('cyclopolyene')

    # CnH2n-6: 四烯烃 或 单环三烯烃
    elif m == 2 * n - 6:
        if n >= 5:
            mol_types.append('tetraene')
        if n >= 5:
            mol_types.append('cyclopolyene')

    # CnH(2n+2-2k) (k>=5): 更高阶多烯烃 或 单环多烯烃
    elif m < 2 * n - 6 and m >= 0 and (2 * n - m) % 2 == 0:
        k = (2 * n + 2 - m) // 2
        if n >= k + 1 and k >= 1:
            mol_types.append('polyene')
        cyclo_k = (2 * n - m) // 2
        if n >= 3 and cyclo_k >= 2:
            mol_types.append('cyclopolyene')

    else:
        return None, n, m, _UNSUPPORTED_MSG.format(n=n, m=m)

    if not mol_types:
        return None, n, m, f"分子式 C{n}H{m} 无法生成异构体"

    return mol_types, n, m, None


def compute_formula(mol_type, n_carbon, n_hydrogen=None):
    """根据分子类型和碳原子数计算分子式"""
    formula_map = {
        'alkane':      lambda n, m: f"C{n}H{2*n + 2}",
        'alkene':      lambda n, m: f"C{n}H{2*n}",
        'alkyne':      lambda n, m: f"C{n}H{2*n - 2}",
        'diene':       lambda n, m: f"C{n}H{2*n - 2}",
        'cycloalkane': lambda n, m: f"C{n}H{2*n}",
        'cycloalkene': lambda n, m: f"C{n}H{2*n - 2}",
        'alkenyl':     lambda n, m: f"C{n}H{2*n - 4}",
        'triene':      lambda n, m: f"C{n}H{2*n - 4}",
        'tetraene':    lambda n, m: f"C{n}H{2*n - 6}",
    }
    if mol_type in formula_map:
        return formula_map[mol_type](n_carbon, n_hydrogen)
    # polyene / cyclopolyene 需要实际氢数
    if n_hydrogen is not None:
        return f"C{n_carbon}H{n_hydrogen}"
    return f"C{n_carbon}H?"


class GeneratorManager:
    """统一管理异构体生成器初始化与分发"""

    def __init__(self):
        self._init_generators()

    def _init_generators(self):
        """初始化各分子生成器"""
        self.alkane_generator = AlkaneIsomerGenerator(use_parallel=True)
        self.alkane_visualizer = AlkaneIsomerVisualizer(self.alkane_generator)
        self.alkene_visualizer = AlkeneViz(generator=None)
        self.alkyne_generator = AlkyneIsomerGenerator()
        self.alkyne_visualizer = AlkyneIsomerVisualizer(generator=self.alkyne_generator)
        self.diene_generator = PolyeneIsomerGenerator()
        self.cycloalkane_generator = CycloalkaneGenerator()
        self.cycloalkane_builder = CycloalkaneBuilder()
        self.cycloalkene_generator = CycloalkeneGenerator()
        self.cyclopolyene_generator = CyclopolyeneGenerator()
        self.alkenyl_generator = AlkenylIsomerGenerator()
        self.polyene_generator = PolyeneIsomerGenerator()

        self.generators = {
            'alkane':      self.alkane_generator,
            'alkene':      self.alkene_visualizer,
            'alkyne':      self.alkyne_generator,
            'diene':       self.diene_generator,
            'cycloalkane': self.cycloalkane_generator,
            'cycloalkene': self.cycloalkene_generator,
            'cyclopolyene': self.cyclopolyene_generator,
            'alkenyl':     self.alkenyl_generator,
            'triene':      self.polyene_generator,
            'tetraene':    self.polyene_generator,
            'polyene':     self.polyene_generator,
        }

    def generate(self, mol_type, n_carbon, n_hydrogen=None):
        """根据分子类型分发到对应生成器，返回异构体列表"""
        generator = self.generators[mol_type]

        if mol_type == 'alkane':
            return generator.generate_isomers(n_carbon)

        elif mol_type == 'alkene':
            return generator.generate_isomers_optimized(n_carbon, show_progress=False)

        elif mol_type == 'alkyne':
            return generator.generate_isomers(n_carbon)

        elif mol_type == 'diene':
            return generator.generate_diene(n_carbon)

        elif mol_type == 'cycloalkane':
            return generator.generate_isomers(n_carbon)

        elif mol_type == 'cycloalkene':
            return generator.generate_isomers(n_carbon)

        elif mol_type == 'alkenyl':
            return generator.generate_isomers(n_carbon)

        elif mol_type in ('triene', 'tetraene', 'polyene'):
            k_map = {'triene': 3, 'tetraene': 4}
            k = k_map.get(mol_type)
            if k is None and n_hydrogen is not None:
                k = (2 * n_carbon + 2 - n_hydrogen) // 2
            if k and k >= 1:
                return generator.generate_k_ene(n_carbon, k)
            return []

        elif mol_type == 'cyclopolyene':
            cyclo_k = 2  # 默认双键数
            if n_hydrogen is not None:
                cyclo_k = (2 * n_carbon - n_hydrogen) // 2
            if cyclo_k >= 2:
                return self.cyclopolyene_generator.generate_k_cycloene(n_carbon, cyclo_k)
        return []


# ============================================================
# 键线式（Skeletal Formula / 结构式）渲染
# ============================================================

# 分子类型 → RDKit 可识别的 SMILES 片段标记
_BOND_TYPE_MAP = {
    'single': None,
    'double': '=',
    'triple': '#',
}


def graph_to_rdkit_mol(G):
    """将 networkx.Graph（含 bond_type 属性）转换为 RDKit Mol 对象

    Args:
        G: networkx.Graph，节点为碳原子索引(0-based)，边含 bond_type 属性

    Returns:
        rdkit.Chem.rdchem.Mol 或 None（失败时）
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BondType
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
            return None
        return mol
    except Exception:
        return None


def canon_str_to_graph(canon_str, mol_type, gen_mgr=None):
    """将规范字符串转换为 networkx.Graph

    Args:
        canon_str: 规范树表示法字符串（如 'CC(C)CC'）
        mol_type: 分子类型
        gen_mgr: GeneratorManager 实例（用于获取生成器）

    Returns:
        networkx.Graph 或 None
    """
    import networkx as nx
    try:
        if mol_type == 'alkene':
            if gen_mgr is not None:
                adj = gen_mgr.alkene_visualizer.canon_to_adjacency(canon_str)
            else:
                from original_programs.alkene_visualizer import AlkeneIsomerVisualizer
                viz = AlkeneIsomerVisualizer(generator=None)
                adj = viz.canon_to_adjacency(canon_str)
        else:
            if gen_mgr is not None:
                adj = gen_mgr.alkane_generator.canon_to_adjacency(canon_str)
            else:
                from original_programs.alkane_isomer_visualizer import AlkaneIsomerGenerator
                gen = AlkaneIsomerGenerator()
                adj = gen.canon_to_adjacency(canon_str)

        G = nx.Graph()
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G
    except Exception:
        return None


def render_skeletal_formula(isomer_data, mol_type, gen_mgr=None, img_size=(400, 300)):
    """渲染分子的键线式（2D skeletal formula）为 PNG 图片字节

    使用 RDKit MolDraw2DSVG 生成标准化学键线式（专业渲染质量），
    再通过 cairosvg 转换为 PNG。回退到 matplotlib 手动绘制。

    按键线式惯例：碳骨架以折线表示，双键/三键以平行线标注，
    不显式标注 C 和 H 原子。

    Args:
        isomer_data: nx.Graph 或规范字符串
        mol_type: 分子类型
        gen_mgr: GeneratorManager 实例（可选，用于字符串→图转换）
        img_size: (width, height) 输出图片尺寸（像素）

    Returns:
        bytes (PNG 图片数据) 或 None（渲染失败时）
    """
    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # 1. 获取 nx.Graph
    if isinstance(isomer_data, nx.Graph):
        G = isomer_data
    elif isinstance(isomer_data, str):
        G = canon_str_to_graph(isomer_data, mol_type, gen_mgr)
        if G is None:
            return None
    else:
        return None

    # 2. 转换为 RDKit Mol 并生成 2D 坐标
    mol = graph_to_rdkit_mol(G)
    if mol is None:
        return None

    # 在无氢分子上生成2D坐标，确保碳骨架呈锯齿状（zigzag）
    # 键线式不需要显式添加氢原子，RDKit Draw会根据NumExplicitHs自动处理
    AllChem.Compute2DCoords(mol)

    # 3. 尝试 RDKit 原生渲染（MolDraw2DSVG + cairosvg → PNG）
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        import cairosvg as _cairosvg

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        png_data = _cairosvg.svg2png(
            bytestring=svg_text.encode('utf-8'),
            output_width=img_size[0],
            output_height=img_size[1],
        )
        if png_data and len(png_data) > 100:
            return png_data
    except Exception:
        pass  # 回退到 matplotlib

    # 4. 回退：matplotlib 手动绘制键线式
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        conf = mol.GetConformer()
        n_c = G.number_of_nodes()
        c_positions = []
        for i in range(n_c):
            pos = conf.GetAtomPosition(i)
            c_positions.append((pos.x, pos.y))

        if not c_positions:
            return None

        w_inch = img_size[0] / 120.0
        h_inch = img_size[1] / 120.0
        fig, ax = plt.subplots(figsize=(w_inch, h_inch), facecolor='white')
        ax.set_aspect('equal')
        ax.axis('off')

        for u, v, data in G.edges(data=True):
            bt = data.get('bond_type', 'single')
            x1, y1 = c_positions[u]
            x2, y2 = c_positions[v]

            if bt == 'double':
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 1e-8:
                    nx_vec = -dy / length * 0.045
                    ny_vec = dx / length * 0.045
                    ax.plot([x1 + nx_vec, x2 + nx_vec], [y1 + ny_vec, y2 + ny_vec],
                            'k-', linewidth=2.0, solid_capstyle='round')
                    ax.plot([x1 - nx_vec, x2 - nx_vec], [y1 - ny_vec, y2 - ny_vec],
                            'k-', linewidth=2.0, solid_capstyle='round')
                else:
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2.0)
            elif bt == 'triple':
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 1e-8:
                    nx_vec = -dy / length * 0.06
                    ny_vec = dx / length * 0.06
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2.0, solid_capstyle='round')
                    ax.plot([x1 + nx_vec, x2 + nx_vec], [y1 + ny_vec, y2 + ny_vec],
                            'k-', linewidth=2.0, solid_capstyle='round')
                    ax.plot([x1 - nx_vec, x2 - nx_vec], [y1 - ny_vec, y2 - ny_vec],
                            'k-', linewidth=2.0, solid_capstyle='round')
                else:
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2.0)
            else:
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2.0, solid_capstyle='round')

        xs = [p[0] for p in c_positions]
        ys = [p[1] for p in c_positions]
        x_margin = max(0.5, (max(xs) - min(xs)) * 0.2)
        y_margin = max(0.5, (max(ys) - min(ys)) * 0.2)
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

        fig.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None
