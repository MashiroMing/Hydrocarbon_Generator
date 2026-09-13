"""
分子式解析与异构体生成分发工具

职责：解析分子式 → 按 (环数, 双键数, 三键数) 分类 → 统一由 multcyclomultalkane 生成
      键线式（skeletal formula）渲染
"""

import re
import io
import tempfile
import os

import networkx as nx

from original_programs.multcyclomultalkane import RobustPolycyclicPolyeneGenerator, _enumerate_rdt_combinations
from original_programs.alkane_isomer_visualizer import AlkaneIsomerGenerator, AlkaneIsomerVisualizer
from original_programs.alkene_visualizer import AlkeneIsomerVisualizer as AlkeneViz

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
    'multcycloalkane': '多环烷烃',
    'multcyclomultalkane': '多环多烯炔烃',
    'benzene': '苯环',
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
    'multcycloalkane': '多环烷烃',
    'multcyclomultalkane': '多环多烯炔烃',
    'benzene': '苯环',
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
    'multcycloalkane': 'Polycycloalkane',
    'multcyclomultalkane': 'Polycyclic Polyene',
    'benzene': 'Benzene',
}

# 不支持的分子式错误提示
_UNSUPPORTED_MSG = (
    "分子式 C{n}H{m} 不是本软件支持的类型\n"
    "支持的类型:\n"
    "• 烷烃 CnH2n+2\n"
    "• 单烯烃/单环烷烃 CnH2n (n>=2)\n"
    "• 单炔烃/二烯烃/单环烯烃/双环烷烃 CnH2n-2 (n>=2)\n"
    "• 烯炔烃/三烯烃/单环二烯烃/三环烷烃 CnH2n-4 (n>=4)\n"
    "• 四烯烃/单环三烯烃/四环烷烃 CnH2n-6 (n>=5)\n"
    "• k-烯烃/单环多烯烃/k环烷烃 CnH2n-2k (k>=5)"
)


def _rdt_to_mol_type(n_r: int, n_d: int, n_t: int) -> str:
    """将 (环数, 双键数, 三键数) 映射为传统分子类型标签"""
    if n_r == 0 and n_t == 0 and n_d == 1:
        return 'alkene'
    if n_r == 0 and n_t == 1 and n_d == 0:
        return 'alkyne'
    if n_r == 0 and n_t == 0 and n_d >= 2:
        # 多烯烃：按双键数细分
        _diene_names = {2: 'diene', 3: 'triene', 4: 'tetraene'}
        return _diene_names.get(n_d, 'polyene')
    if n_r == 0 and n_t >= 1 and n_d >= 1:
        return 'alkenyl'
    if n_r == 0 and n_t >= 2:
        return 'alkenyl'
    if n_r == 1 and n_t == 0 and n_d == 1:
        return 'cycloalkene'
    if n_r == 1 and n_t == 0 and n_d >= 2:
        return 'cyclopolyene'
    if n_r == 1 and n_t == 0 and n_d == 0:
        return 'cycloalkane'
    if n_r >= 2 and n_t == 0 and n_d == 0:
        return 'multcycloalkane'
    # 其余：含环 + 含重键的复杂情况
    return 'multcyclomultalkane'


def parse_molecule_input(formula_str):
    """解析分子式，返回分类标签列表、碳氢原子数及错误信息

    返回值: (mol_types, n, m, error)
        mol_types: 按 (r,d,t) 分类后的分子类型列表（去重排序）
    """
    formula_str = formula_str.strip()
    if not formula_str:
        return None, None, None, "请输入分子式"

    pattern = r'^C(\d*)H(\d+)$'
    match = re.match(pattern, formula_str, re.IGNORECASE)
    if not match:
        return None, None, None, "分子式格式不正确，请使用如 C5H12、C4H8 的格式"

    n = int(match.group(1)) if match.group(1) else 1
    m = int(match.group(2))

    if n < 1:
        return None, None, None, "碳原子数必须 >= 1"
    if m < 0:
        return None, None, None, "氢原子数不能为负"

    mol_types = []

    # CnH2n+2: 烷烃（无环、无双键三键）
    if m == 2 * n + 2:
        if n >= 1:
            mol_types.append('alkane')

    # CnH2n+2-2k (k>=1): 不饱和度 k = (2n+2-m)/2
    elif m <= 2 * n and m >= 0 and (2 * n + 2 - m) % 2 == 0:
        k = (2 * n + 2 - m) // 2
        if k < 1:
            return None, n, m, _UNSUPPORTED_MSG.format(n=n, m=m)
        if n < 2:
            return None, n, m, _UNSUPPORTED_MSG.format(n=n, m=m)

        # 枚举所有 (r, d, t) 组合，收集对应的分子类型
        combos = _enumerate_rdt_combinations(k)
        type_set = set()
        for n_r, n_d, n_t in combos:
            # 检查该组合是否可能产生异构体（基本可行性）
            # 无环至少需要 n_d + 2*n_t + 1 <= n（确保有足够碳原子）
            if n_r == 0:
                if n_d == 0 and n_t == 0:
                    continue  # 纯烷烃，已被 CnH2n+2 覆盖
                min_carbons = n_d + 2 * n_t + 1
                if n < min_carbons:
                    continue
            elif n_r == 1 and n_d == 0 and n_t == 0:
                if n < 3:
                    continue
            elif n_r >= 2 and n_d == 0 and n_t == 0:
                if n < n_r + 2:
                    continue
            mol_type = _rdt_to_mol_type(n_r, n_d, n_t)
            type_set.add(mol_type)

        # 按固定顺序排列
        _TYPE_ORDER = [
            'alkene', 'alkyne', 'diene', 'alkenyl',
            'triene', 'tetraene', 'polyene',
            'cycloalkane', 'cycloalkene', 'cyclopolyene',
            'multcycloalkane', 'multcyclomultalkane',
        ]
        for t in _TYPE_ORDER:
            if t in type_set:
                mol_types.append(t)

    else:
        return None, n, m, _UNSUPPORTED_MSG.format(n=n, m=m)

    if not mol_types:
        return None, n, m, f"分子式 C{n}H{m} 无法生成异构体"

    return mol_types, n, m, None


# ─────────────────────────────────────────────
# 整合分子式解析：C{n}H{m}O{o} + 卤素 + 氘
# ─────────────────────────────────────────────

_SUPPORTED_ELEMENTS = "C、H、O 以及卤素 F、Cl、Br、I / 氘代 D"


def parse_compound_formula(formula_str: str):
    """解析完整分子式（含 O、卤素、氘代），返回统一结构

    格式: C{n}H{m}O{o} 后跟任意顺序的卤素 (F/Cl/Br/I) 和氘 (D)
    例: C3H6O3, C2H6OClBr, C6H6, C2H6OD, C3H6Cl2

    Returns:
        (mol_types, n_carbon, n_hydrogen, n_oxygen,
         halogen_spec, n_deuterium, error)
        halogen_spec: (f, cl, br, i) 元组
    """
    formula = formula_str.strip()
    if not formula:
        return None, None, None, None, None, None, "请输入分子式"

    # 逐元素提取（2 字符元素优先避免 Cl/Br 误匹配为 C/B）
    p = r'(Cl|Br|C|H|O|D|F|I)(\d*)'
    matches = re.findall(p, formula, re.IGNORECASE)
    if not matches:
        return None, None, None, None, None, None, "无法识别该分子式"

    counts = {'C': 0, 'H': 0, 'O': 0, 'D': 0,
              'F': 0, 'CL': 0, 'BR': 0, 'I': 0}
    used = set()
    for elem, num in matches:
        elem_upper = elem.upper()
        if elem_upper in used:
            return None, None, None, None, None, None, f"{elem_upper} 重复出现"
        used.add(elem_upper)
        cnt = int(num) if num else 1
        counts[elem_upper] = cnt

    n_carbon = counts['C']
    n_hydrogen = counts['H']
    n_oxygen = counts['O']
    n_deuterium = counts['D']
    halogen_spec = (counts['F'], counts['CL'], counts['BR'], counts['I'])

    # 验证
    if n_carbon < 1:
        return (None, None, None, None, None, None,
                "分子式需含碳原子。纯无机分子不在支持范围。")
    if n_hydrogen == 0 and n_carbon >= 1:
        # C6 不加 H 等价于 C6H?: 不允许
        return (None, None, None, None, None, None,
                "分子式需含氢原子，如 CH4、C2H6O")
    # 卤素视为 H 替代参与价态/k 计算（每 Cl/Br/I 占 1H，F 也占 1H）
    total_halogen = sum(halogen_spec)
    total_h_equiv = n_hydrogen + total_halogen
    if total_h_equiv > 2 * n_carbon + 2:
        return (None, None, None, None, None, None,
                f"H 原子数 ({n_hydrogen}) + 卤素数 ({total_halogen}) "
                f"超出碳原子价态上限")
    if (2 * n_carbon + 2 - total_h_equiv) % 2 != 0:
        return (None, None, None, None, None, None,
                "不饱和度 (k) 非整数，请检查 H 原子数")

    if n_deuterium > n_hydrogen:
        return (None, None, None, None, None, None,
                f"氘代原子数 ({n_deuterium}) 不能超过氢原子总数")

    # 验证仅支持的元素
    for ch in formula:
        if ch.upper() not in 'CHODFCLBRI0123456789':
            return (None, None, None, None, None, None,
                    f"不支持的字符 '{ch}'，仅支持 {_SUPPORTED_ELEMENTS}")

    # 复用 parse_molecule_input 推导 mol_types（基于 C+H 等价骨架）
    base_formula = f"C{n_carbon}H{total_h_equiv}"
    mol_types, _, _, base_error = parse_molecule_input(base_formula)
    if base_error:
        return None, n_carbon, n_hydrogen, n_oxygen, halogen_spec, n_deuterium, base_error

    return mol_types, n_carbon, n_hydrogen, n_oxygen, halogen_spec, n_deuterium, None


def format_compound_formula(n_carbon, n_hydrogen, n_oxygen,
                            halogen_spec, n_deuterium=0) -> str:
    """将解析结果格式化为标准化学式字符串"""
    parts = []
    parts.append(f"CH{n_hydrogen}" if n_carbon == 1 else f"C{n_carbon}H{n_hydrogen}")
    if n_oxygen > 0:
        parts.append(f"O{n_oxygen}" if n_oxygen > 1 else "O")
    if n_deuterium > 0:
        parts.append(f"D{n_deuterium}" if n_deuterium > 1 else "D")
    for elem, cnt in [('Br', halogen_spec[2]), ('Cl', halogen_spec[1]),
                       ('F', halogen_spec[0]), ('I', halogen_spec[3])]:
        if cnt > 0:
            parts.append(f"{elem}{cnt}" if cnt > 1 else elem)
    return ''.join(parts)


def format_formula(n_carbon, n_hydrogen):
    """格式化分子式为标准化学写法：单碳时省略 1，如 CH4 而非 C1H4"""
    if n_carbon == 1:
        return f"CH{n_hydrogen}"
    return f"C{n_carbon}H{n_hydrogen}"


def compute_formula(mol_type, n_carbon, n_hydrogen=None):
    """根据分子类型和碳原子数计算分子式"""
    formula_map = {
        'alkane':      lambda n, m: format_formula(n, 2*n + 2),
        'alkene':      lambda n, m: format_formula(n, 2*n),
        'alkyne':      lambda n, m: format_formula(n, 2*n - 2),
        'diene':       lambda n, m: format_formula(n, 2*n - 2),
        'cycloalkane': lambda n, m: format_formula(n, 2*n),
        'cycloalkene': lambda n, m: format_formula(n, 2*n - 2),
        'alkenyl':     lambda n, m: format_formula(n, 2*n - 4),
        'triene':      lambda n, m: format_formula(n, 2*n - 4),
        'tetraene':    lambda n, m: format_formula(n, 2*n - 6),
    }
    if mol_type in formula_map:
        return formula_map[mol_type](n_carbon, n_hydrogen)
    # polyene / cyclopolyene / multcycloalkane / multcyclomultalkane 需要实际氢数
    if n_hydrogen is not None:
        return format_formula(n_carbon, n_hydrogen)
    return f"{'CH' if n_carbon == 1 else f'C{n_carbon}H'}?"


class GeneratorManager:
    """统一管理异构体生成器：所有非烷烃类型均由 multcyclomultalkane 统一生成"""

    def __init__(self):
        self._init_generators()

    def _init_generators(self):
        """初始化各分子生成器"""
        # 烷烃：独立生成器（不经过 multcyclomultalkane）
        self.alkane_generator = AlkaneIsomerGenerator(use_parallel=True)
        self.alkane_visualizer = AlkaneIsomerVisualizer(self.alkane_generator)
        self.alkene_visualizer = AlkeneViz(generator=None)

        # 统一生成器：所有含不饱和度的烃类
        self.unified_generator = RobustPolycyclicPolyeneGenerator()

        self.generators = {
            'alkane':      self.alkane_generator,
            'unified':     self.unified_generator,
        }

    def generate(self, mol_type, n_carbon, n_hydrogen=None):
        """根据分子类型分发到对应生成器，返回异构体列表

        对于非烷烃类型，统一使用 multcyclomultalkane.generate()，
        按 (r,d,t) 组合枚举，并自动分类打标签。
        """
        if mol_type == 'alkane':
            return self.alkane_generator.generate_isomers(n_carbon)

        # 计算不饱和度
        k = 2
        if n_hydrogen is not None:
            k = (2 * n_carbon + 2 - n_hydrogen) // 2
        if k < 1 or n_carbon < 2:
            return []

        combos = _enumerate_rdt_combinations(k)

        # 按 mol_type 筛选对应的 (r,d,t) 组合
        result = []
        for n_r, n_d, n_t in combos:
            combo_type = _rdt_to_mol_type(n_r, n_d, n_t)
            if combo_type != mol_type:
                continue
            isomers = self.unified_generator.generate(n_carbon, n_r, n_d, n_t)
            result.extend(isomers)
        return result

    def generate_all(self, n_carbon, n_hydrogen):
        """生成所有类型的异构体，返回 [(mol_type, isomer_graph), ...]

        使用统一的 WL 哈希去重，避免跨类型重复。
        烷烃单独处理（不会与其他类型重复）。
        """
        all_isomers = []

        # 1. 烷烃（CnH2n+2）
        if n_hydrogen == 2 * n_carbon + 2 and n_carbon >= 1:
            alkane_isomers = self.alkane_generator.generate_isomers(n_carbon)
            for iso in alkane_isomers:
                all_isomers.append(('alkane', iso))

        # 2. 非烷烃：统一生成并分类
        k = (2 * n_carbon + 2 - n_hydrogen) // 2
        if k < 1 or n_carbon < 2:
            return all_isomers

        combos = _enumerate_rdt_combinations(k)
        for n_r, n_d, n_t in combos:
            # 跳过纯烷烃组合
            if n_r == 0 and n_d == 0 and n_t == 0:
                continue
            isomers = self.unified_generator.generate(n_carbon, n_r, n_d, n_t)
            combo_type = _rdt_to_mol_type(n_r, n_d, n_t)
            for iso in isomers:
                all_isomers.append((combo_type, iso))

        return all_isomers


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
    """将 networkx.Graph（含 bond_type / halogen_counts / oxo_counts）→ RDKit Mol

    节点 label='C'/'O' 区分碳骨架与醚氧骨架节点。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BondType
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        n_atoms = G.number_of_nodes()
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
        # 根据节点 label 创建原子（C 或 O 骨架）—— 按节点集合遍历避免索引间隙
        node_to_atom_idx = {}
        for n, data in G.nodes(data=True):
            label = data.get('label', 'C')
            atom_idx = mol.AddAtom(Chem.Atom(label))
            node_to_atom_idx[n] = atom_idx

            added = set()
        for u, v in G.edges():
            key = (min(u, v), max(u, v))
            if key in added:
                continue
            added.add(key)
            u_idx, v_idx = node_to_atom_idx[u], node_to_atom_idx[v]
            if key in triple_bonds:
                mol.AddBond(u_idx, v_idx, BondType.TRIPLE)
            elif key in double_bonds:
                mol.AddBond(u_idx, v_idx, BondType.DOUBLE)
            else:
                mol.AddBond(u_idx, v_idx, BondType.SINGLE)

        # 添加卤素/氧取代基并设置氢原子数
        _HALOGEN_RDKIT = ['F', 'Cl', 'Br', 'I']
        next_idx = n_atoms
        for node in sorted(G.nodes()):
            atom_idx = node_to_atom_idx[node]
            label = G.nodes[node].get('label', 'C')
            max_valence = 4 if label == 'C' else 2

            # 计算键负载
            bond_load = 0
            for nb in G.neighbors(node):
                bt = G[node][nb].get('bond_type', 'single')
                if bt == 'double':
                    bond_load += 2
                elif bt == 'triple':
                    bond_load += 3
                else:
                    bond_load += 1

            # O 骨架节点：无取代基，直接算 H
            if label == 'O':
                h_count = max(0, max_valence - bond_load)
                if h_count > 0:
                    mol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(h_count)
                continue

            # ── C 节点：处理卤素 + 氧取代基 ──
            hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
            total_halogen = sum(hc)
            for h_idx, count in enumerate(hc):
                for _ in range(count):
                    mol.AddAtom(Chem.Atom(_HALOGEN_RDKIT[h_idx]))
                    mol.AddBond(atom_idx, next_idx, BondType.SINGLE)
                    next_idx += 1

            oxo = G.nodes[node].get('oxo_counts', (0, 0))
            oh_count, co_count = oxo
            for _ in range(oh_count):
                mol.AddAtom(Chem.Atom('O'))
                mol.AddBond(atom_idx, next_idx, BondType.SINGLE)
                mol.GetAtomWithIdx(next_idx).SetNumExplicitHs(1)
                next_idx += 1
            for _ in range(co_count):
                mol.AddAtom(Chem.Atom('O'))
                mol.AddBond(atom_idx, next_idx, BondType.DOUBLE)
                next_idx += 1

            oxo_h_used = oh_count + 2 * co_count
            h_count = max(0, max_valence - bond_load - total_halogen - oxo_h_used)
            if h_count > 0:
                mol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(h_count)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # 重试：跳过 KEKULIZE，让 RDKit 自动感知芳香性
            try:
                from rdkit.Chem import SanitizeFlags
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=(
                        SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE
                    )
                )
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
            G.add_node(node, label='C')  # 显式添加节点（处理孤立碳，如甲烷）
            for neighbor in neighbors:
                G.add_edge(node, neighbor, bond_type='single')
        return G
    except Exception:
        return None


def _render_methane_label(img_size=(400, 300)):
    """为甲烷 (CH4) 绘制分子结构示意图

    甲烷是单碳四面体分子，中心碳连接四个氢原子。
    以2D平面投影绘制：C在中心，四个H通过单键连接呈十字形分布。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        w_inch = img_size[0] / 120.0
        h_inch = img_size[1] / 120.0
        fig, ax = plt.subplots(figsize=(w_inch, h_inch), facecolor='white')
        ax.set_aspect('equal')
        ax.axis('off')

        # 中心碳原子位置
        cx, cy = 0.5, 0.5
        # 四个氢原子位置（上下左右，模拟四面体投影）
        bond_len = 0.25
        h_positions = [
            (cx, cy + bond_len),      # 上
            (cx, cy - bond_len),      # 下
            (cx - bond_len * 0.9, cy), # 左（稍短，透视效果）
            (cx + bond_len * 0.9, cy), # 右
        ]

        # 绘制 C-H 键
        for hx, hy in h_positions:
            ax.plot([cx, hx], [cy, hy], 'k-', linewidth=3.0, solid_capstyle='round')

        # 绘制中心碳原子（黑色圆点）
        ax.plot(cx, cy, 'ko', markersize=22, markerfacecolor='#333333',
                markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        # 碳原子标签
        ax.text(cx, cy, "C", fontsize=14, ha='center', va='center',
                color='white', weight='bold', zorder=4)

        # 绘制四个氢原子（浅灰圆点）
        for hx, hy in h_positions:
            ax.plot(hx, hy, 'o', markersize=16, markerfacecolor='#e0e0e0',
                    markeredgecolor='#666666', markeredgewidth=1.2, zorder=3)
            ax.text(hx, hy, "H", fontsize=10, ha='center', va='center',
                    color='#333333', weight='bold', zorder=4)

        # 设置显示范围
        margin = 0.1
        ax.set_xlim(0.5 - bond_len - margin, 0.5 + bond_len + margin)
        ax.set_ylim(0.5 - bond_len - margin, 0.5 + bond_len + margin)

        fig.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor='white', pad_inches=0.15)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def render_skeletal_formula(isomer_data, mol_type, gen_mgr=None, img_size=(400, 300)):
    """渲染分子的键线式（2D skeletal formula）为 PNG 图片字节

    使用 RDKit MolDraw2DSVG 生成标准化学键式（专业渲染质量），
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

    # 1a. 特殊处理：甲烷 (CH4) — 单碳无键，键线式无骨架可绘
    #     单独绘制 "CH₄" 标签以区别于空白图像
    if G.number_of_nodes() == 1 and G.number_of_edges() == 0:
        return _render_methane_label(img_size)

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


# ============================================================
# 通用去重框架（halogen/ 与 oxygen/ 各生成器共用）
# ============================================================

def _dedup_canon_key(G, node_label_fn):
    """WL 图哈希（节点标签由调用方 node_label_fn 编码）

    Args:
        G: networkx.Graph
        node_label_fn: callable(node) -> str，节点标签编码函数
    """
    H = nx.Graph()
    for n in G.nodes():
        H.add_node(n, label=node_label_fn(G, n))
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))
    return nx.weisfeiler_lehman_graph_hash(
        H, edge_attr='bond_type', node_attr='label'
    )


def _dedup_degree_signature(G, node_sig_fn):
    """度签名（节点特征由调用方 node_sig_fn 编码），用于同构检查的预过滤"""
    return tuple(sorted(
        (G.degree(v), node_sig_fn(G, v))
        for v in G.nodes()
    ))


def dedup_add_to_buckets(G, mol_type, buckets, node_match, node_label_fn,
                         node_sig_fn, canon_key_fn, degree_signature_fn):
    """将图添加到去重桶中（WL哈希→桶→度签名预过滤→is_isomorphic精确比较）

    各生成器因节点/边特征不同，通过参数传入：
        node_match / node_label_fn / node_sig_fn / canon_key_fn / degree_signature_fn

    buckets: Dict[str, List[Tuple[str, nx.Graph]]]
    """
    h = canon_key_fn(G)
    deg_sig = degree_signature_fn(G)

    if h not in buckets:
        buckets[h] = [(mol_type, G)]
        return

    for _, existing_G in buckets[h]:
        if degree_signature_fn(existing_G) != deg_sig:
            continue

        def edge_match(e1, e2):
            return e1.get('bond_type') == e2.get('bond_type')

        if nx.is_isomorphic(G, existing_G,
                            node_match=node_match,
                            edge_match=edge_match):
            return  # 已存在同构图

    buckets[h].append((mol_type, G))


# ============================================================
# 分子式一致性校验（统一口径，单一事实源）
# ============================================================
# 所有生成器 / 展示 / 门禁的分子式统计都走 graph_formula_parts，
# 不信任调用方传入的碳数参数（历史缺陷：compute_graph_formula 的
# C 前缀曾直接使用调用方 n_carbon，导致桶键公式与实际图不符）。

def graph_formula_parts(G):
    """从图中统计真实分子式组成 (nC, nH, nO, (f, cl, br, i))

    口径与氧取代基/卤素图表示一致（PROJECT_GUIDE 3.1 数据结构）：
      - 节点 label='C' 最大价 4，label='O'（骨架氧）最大价 2；
      - 键负载 bond_load = Σ(单键1 / 双键2 / 三键3)；
      - 取代基占位：-OH 占 1 个 H 位（自身带 1 个 H），=O 占 2 个 H 位，
        每个卤素占 1 个 H 位；
      - h = max(0, max_v - bond_load - oh - 2*co - halo)；total_h += h + oh；
      - nO = Σ(oxo_counts) + 骨架 O 节点数。
    """
    n_c = 0
    n_o = 0
    total_h = 0
    total_f = total_cl = total_br = total_i = 0

    for node in G.nodes():
        label = G.nodes[node].get('label', 'C')
        bond_load = sum(
            3 if G[node][nb].get('bond_type') == 'triple' else
            2 if G[node][nb].get('bond_type') == 'double' else 1
            for nb in G.neighbors(node)
        )
        oh, co = G.nodes[node].get('oxo_counts', (0, 0))
        hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
        halo_on_node = sum(hc)

        n_o += oh + co
        total_f += hc[0]
        total_cl += hc[1]
        total_br += hc[2]
        total_i += hc[3]

        if label == 'C':
            n_c += 1
        elif label == 'O':
            n_o += 1

        max_v = 4 if label == 'C' else 2
        h_on_atom = max(0, max_v - bond_load - oh - 2 * co - halo_on_node)
        total_h += h_on_atom
        total_h += oh  # OH 上的 H

    return (n_c, total_h, n_o, (total_f, total_cl, total_br, total_i))


def format_graph_formula(G) -> str:
    """从图生成标准分子式字符串（与各 compute_graph_formula 口径一致）

    格式: C{n}H{h}[O{o}][Br..][Cl..][F..][I..]（卤素按字母序）
    """
    n_c, total_h, total_o, (f, cl, br, i) = graph_formula_parts(G)
    base = f"CH{total_h}" if n_c <= 1 else f"C{n_c}H{total_h}"
    if total_o > 0:
        base += f"O{total_o}" if total_o > 1 else "O"
    for elem, cnt in [('Br', br), ('Cl', cl), ('F', f), ('I', i)]:
        if cnt > 0:
            base += f"{elem}{cnt}" if cnt > 1 else elem
    return base


def expected_formula_parts(n_carbon, n_hydrogen, n_oxygen,
                           halogen_spec=(0, 0, 0, 0),
                           n_deuterium=0):
    """用户输入对应的期望分子式组成 (nC, nH, nO, (f, cl, br, i))

    参数即 utils.parse_compound_formula 的解析结果（卤素视为 H 替代；
    氘代 D 仅参与合法性校验，不进入图比较——生成器产出的是非氘代母体）。
    """
    return (n_carbon, n_hydrogen, n_oxygen, tuple(halogen_spec))


def formula_matches_input(G, n_carbon, n_hydrogen, n_oxygen,
                          halogen_spec=(0, 0, 0, 0), n_deuterium=0):
    """分子式门禁：图 G 的真实分子式是否与用户输入完全一致

    任何生成路径（=O 使 k+1、原子矩阵补漏枚举全部不饱和度、片段合并
    改变 k 等）只要产出分子式不符的结构，该函数即返回 False。
    """
    if n_deuterium < 0 or n_deuterium > n_hydrogen:
        return False
    return graph_formula_parts(G) == expected_formula_parts(
        n_carbon, n_hydrogen, n_oxygen, halogen_spec, n_deuterium)
