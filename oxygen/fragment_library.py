"""
分子片段库 — 预制化学基团 (nx.Graph)

用法示例:
    from oxygen.fragment_library import get_fragment, list_fragments

    benzene = get_fragment('benzene')
    furan   = get_fragment('furan')

    # 直接注入生成器
    from oxygen.oxo_generator import OxoSubstituentGenerator
    gen = OxoSubstituentGenerator()
    phenol_variants = gen.generate([('benzene', benzene)], n_oxygen=1, n_carbon=6)

扩展方法:
    在 _builders 字典中新增条目即可，格式:
    'key': ('描述', '类别', builder_function)

    类别可选: '芳香环', '杂环', '饱和环', '多环饱和', '官能团', '碳链'
"""

import networkx as nx
from typing import Dict, Tuple, Callable, List


# ============ 构建器注册 ============

# 格式: key → (中文描述, 类别, builder_function)
_builders: Dict[str, Tuple[str, str, Callable[[], nx.Graph]]] = {}

# 缓存: 首次调用后才构建，后续返回缓存
_cache: Dict[str, nx.Graph] = {}


def _register(key: str, desc: str, category: str, builder: Callable[[], nx.Graph]):
    _builders[key] = (desc, category, builder)


def get_fragment(name: str) -> nx.Graph:
    """获取片段图对象（首次调用构建，后续缓存）

    Args:
        name: 片段注册名（如 'benzene', 'furan'）

    Returns:
        nx.Graph，含 label='C'/'O' 和 bond_type='single'/'double'/'triple'

    Raises:
        KeyError: 未注册的片段名
    """
    if name not in _builders:
        raise KeyError(
            f"未注册的片段 '{name}'，可用: {list(_builders.keys())}"
        )
    if name not in _cache:
        _, _, builder = _builders[name]
        _cache[name] = builder()
    return _cache[name].copy()  # 返回副本，防止外部污染缓存


def list_fragments() -> List[str]:
    """列出所有可用片段名"""
    return sorted(_builders.keys())


def get_fragment_info(name: str) -> dict:
    """获取片段元信息（不返回图对象，轻量查询）

    Returns:
        {'name': str, 'description': str, 'category': str,
         'n_carbon': int, 'n_oxygen': int, 'k': int}
    """
    if name not in _builders:
        raise KeyError(f"未注册的片段 '{name}'")
    desc, cat, _ = _builders[name]
    G = get_fragment(name)  # 缓存已构建
    nc = sum(1 for n in G.nodes() if G.nodes[n].get('label', 'C') == 'C')
    no = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'O')
    nh = 0
    for n in G.nodes():
        label = G.nodes[n].get('label', 'C')
        max_v = 2 if label == 'O' else 4
        bl = sum(2 if G[n][nb].get('bond_type') == 'double' else
                 3 if G[n][nb].get('bond_type') == 'triple' else 1
                 for nb in G.neighbors(n))
        nh += max(0, max_v - bl)
    k = (2 * nc + 2 - nh) // 2
    return {'name': name, 'description': desc, 'category': cat,
            'n_carbon': nc, 'n_oxygen': no, 'k': k}


def graph_contains_fragment(G, fragment_name: str) -> bool:
    """检查图 G 是否包含指定片段作为子图

    使用 networkx 的子图同构（VF2 算法）。片段为带 label 的图，
    比较时按节点 label + 边 bond_type 匹配。
    """
    import networkx as nx
    sub = get_fragment(fragment_name)
    if G.number_of_nodes() < sub.number_of_nodes():
        return False
    # 子图匹配：节点 label 必须一致
    matcher = nx.algorithms.isomorphism.categorical_node_match('label', 'C')
    em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
    GM = nx.algorithms.isomorphism.GraphMatcher(G, sub, node_match=matcher, edge_match=em)
    return any(GM.subgraph_isomorphisms_iter())


def describe(key: str) -> str:
    """片段说明"""
    if key in _builders:
        desc, cat, _ = _builders[key]
        return f"[{cat}] {key}: {desc}"
    raise KeyError(f"未知片段 '{key}'")


def list_by_category() -> Dict[str, List[str]]:
    """按类别分组列出片段"""
    groups: Dict[str, List[str]] = {}
    for key, (_, cat, _) in sorted(_builders.items()):
        groups.setdefault(cat, []).append(key)
    return dict(sorted(groups.items()))


# ============ 构建器工具函数 ============

def _make_ring(atoms: list, bonds: list) -> nx.Graph:
    """构建环状骨架图

    Args:
        atoms: 原子符号列表，如 ['C', 'C', 'C', 'O', 'C']
        bonds: 键型列表，长度 = len(atoms)，从 atom[0]-atom[1] 到 atom[-1]-atom[0]
               如 ['single', 'double', 'single', 'double', 'single']
    Returns:
        nx.Graph
    """
    n = len(atoms)
    assert len(bonds) == n, f"bonds({len(bonds)}) 与 atoms({n}) 数量不一致"
    G = nx.Graph()
    for i, atom in enumerate(atoms):
        G.add_node(i, label=atom)
    for i in range(n):
        j = (i + 1) % n
        G.add_edge(i, j, bond_type=bonds[i])
    return G


def _make_chain(atoms: list, bonds: list) -> nx.Graph:
    """构建链状骨架图

    atoms: ['C', 'C', 'O', 'C']
    bonds: ['single', 'single', 'single']  # 长度 = len(atoms) - 1
    """
    n = len(atoms)
    assert len(bonds) == n - 1, f"bonds({len(bonds)}) 应 = atoms({n}) - 1"
    G = nx.Graph()
    for i, atom in enumerate(atoms):
        G.add_node(i, label=atom)
    for i, bt in enumerate(bonds):
        G.add_edge(i, i + 1, bond_type=bt)
    return G


def _fuse_rings(ring1: nx.Graph, ring2: nx.Graph,
                shared_edge: Tuple[int, int]) -> nx.Graph:
    """拼接两个环，共享一条边

    Args:
        ring1, ring2: 两个独立的环图
        shared_edge: (ring1_node, ring2_node_0, ring2_node_1)
                     表示 ring1[shared_edge[0]] 与 ring2 的 shared_edge[1:3] 共享
    """
    G = ring1.copy()
    offset = ring1.number_of_nodes()
    r2_map = {}
    r1_idx, r2_a, r2_b = shared_edge

    # 重映射 ring2 节点（保留共享边位置）
    r2_shared = {r2_a, r2_b}
    r2_to_new = {}
    for n in ring2.nodes():
        if n == r2_a:
            r2_to_new[n] = r1_idx
        elif n == r2_b:
            r2_to_new[n] = G.number_of_nodes()
            G.add_node(G.number_of_nodes(),
                       atom=ring2.nodes[n].get('label', 'C'))
            G.add_edge(r1_idx, r2_to_new[n],
                       bond_type=ring2[r2_a][r2_b].get('bond_type', 'single'))
        else:
            r2_to_new[n] = G.number_of_nodes()
            G.add_node(G.number_of_nodes(),
                       atom=ring2.nodes[n].get('label', 'C'))

    for u, v, data in ring2.edges(data=True):
        pu, pv = r2_to_new[u], r2_to_new[v]
        if pu != r1_idx or pv not in [r2_to_new[n] for n in r2.nodes()
                                       if n not in (r2_a, r2_b)]:
            if not G.has_edge(pu, pv):
                G.add_edge(pu, pv,
                           bond_type=data.get('bond_type', 'single'))
    return G


# ─────────────────────────────────────────────
# 芳香环
# ─────────────────────────────────────────────

def _build_benzene() -> nx.Graph:
    """苯环 (C6H6, Kekulé 式)
        C1=C2-C3=C4-C5=C6  (6元环, 交替单双键)
    """
    return _make_ring(
        atoms=['C', 'C', 'C', 'C', 'C', 'C'],
        bonds=['double', 'single', 'double', 'single', 'double', 'single']
    )


def _build_naphthalene() -> nx.Graph:
    """萘环 (C10H8, 双苯环稠合)
        结构: 两个苯环共享 C1-C2 边
    """
    # 苯环 1 (Kekulé): C0=C1-C2=C3-C4=C5
    benzene = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['double', 'single', 'double', 'single', 'double', 'single']
    )
    # 第二个苯环共享 C1-C2，其余 C 重新编号
    b2 = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['double', 'single', 'double', 'single', 'double', 'single']
    )
    # 重新排列 b2：
    # b2 的键布局: F0=D1-S2=D3-S4=D5-S
    # 对应实环:   C'0=C'1-C'2=C'3-C'4=C'5
    # 共享 C'0-C'5 = 苯环 C1-C2 边
    n = benzene.number_of_nodes()  # = 6
    shared_r1_node = 1  # C1
    H = benzene.copy()
    # b2 的 at 1-0 是共享边: C'0(b2[0])=C'1(b2[1]) 但共享边应由构建逻辑定
    # 简化：逐个添加 b2 原子到苯环上
    # b2: 0-1(double) 1-2(single) 2-3(double) 3-4(single) 4-5(double) 5-0(single)
    # 共享 b2[0]-b2[1] 与 苯环 C1-C2
    r1_u, r1_v = 1, 2   # 苯环的共享边 (C1-C2, single)
    # b2 中充当 r1_u (C1) 的是 b2[0], 充当 r1_v (C2) 的是 b2[1]
    r2_map = {0: r1_u, 1: r1_v}
    for b2_node in range(2, 6):
        r2_map[b2_node] = n
        H.add_node(n, atom=b2.nodes[b2_node].get('atom', 'C'))
        n += 1
    for u, v, d in b2.edges(data=True):
        pu, pv = r2_map[u], r2_map[v]
        if (pu == r1_u and pv == r1_v) or (pu == r1_v and pv == r1_u):
            continue  # 已存在于苯环
        H.add_edge(pu, pv, bond_type=d.get('bond_type', 'single'))
    return H


def _build_anthracene() -> nx.Graph:
    """蒽 (C14H10, 三个线性稠合的苯环)"""
    # 苯环-苯环-苯环: 共享边 C1-C2 和 C3-C4
    # 构建线性三环
    naph = _build_naphthalene()  # 10 个原子, 0-9
    b3 = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['double', 'single', 'double', 'single', 'double', 'single']
    )
    # b3 共享 naph 的 C3-C4 边
    # naph atoms: 0=C1-S2=C3-S4=C5, 6=D7-S8=C9 (need actual structure)
    # 简化：使用索引方式，共享 C3-C4 (index 3-4)
    H = naph.copy()
    n = H.number_of_nodes()
    r2_map = {0: 3, 1: 4}  # b3 的 0-1 对应 naph 的 3-4
    for b2_node in range(2, 6):
        r2_map[b2_node] = n
        H.add_node(n, atom='C')
        n += 1
    for u, v, d in b3.edges(data=True):
        pu, pv = r2_map[u], r2_map[v]
        if {pu, pv} == {3, 4}:
            continue
        H.add_edge(pu, pv, bond_type=d.get('bond_type', 'single'))
    return H


def _build_phenanthrene() -> nx.Graph:
    """菲 (C14H10, 角形三环稠合)

    结构: 苯-苯-苯, 共享 C1-C2 和 C2-C3 (角形联结)
    注意: 本方法提供近似 Kekulé 表示，不影响后续取代操作。
    """
    benzene = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['double', 'single', 'double', 'single', 'double', 'single']
    )
    # 构建角形稠合: 环1=C0-C1-C2=C3-C4-C5, 环2 共享 C1-C2, 环3 共享 C2-C3
    naph = _build_naphthalene()  # 环1+环2

    # 在 naph 上再稠合一个苯环，共享 naph 中的 C2-C3 边（角形）
    # naph 结构: 从左到右依次编号，C0-C1-C2-C3-C4-C5 为第一个环,
    # C1-C2 与第二个环共享
    H = naph.copy()
    n = H.number_of_nodes()
    ring3 = _make_ring(
        ['C', 'C', 'C', 'C'],
        ['single', 'double', 'single', 'double']
    )
    # 扩展 ring3 到 6 个: O=C-S 的剩余部分
    # 简化: ring3[0]-ring3[1] 共享 naph 的边
    # 实际使用: ring3 共享 C2-C3 (naph 索引由构建决定)
    # 直接使用一个简化苯环，共享 naph 的 C2-C3
    b3 = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['double', 'single', 'double', 'single', 'double', 'single']
    )
    r2_map = {0: 2, 1: 3}  # b3[0]-b3[1] 共享 naph 的 C2-C3
    for b2_node in range(2, 6):
        r2_map[b2_node] = n
        H.add_node(n, atom='C')
        n += 1
    for u, v, d in b3.edges(data=True):
        pu, pv = r2_map[u], r2_map[v]
        if {pu, pv} == {2, 3}:
            continue
        H.add_edge(pu, pv, bond_type=d.get('bond_type', 'single'))
    return H


def _build_biphenyl() -> nx.Graph:
    """联苯 (C12H10, C-C 连接两个苯环)"""
    b1 = _build_benzene()
    b2 = _build_benzene()
    n = b1.number_of_nodes()
    H = b1.copy()
    r2_old_to_new = {}
    for i in range(n):
        if i != 0:
            H.add_node(n + i - 1, atom='C')
            r2_old_to_new[i] = n + i - 1
    r2_old_to_new[0] = 0  # 共享 C0
    for u, v, d in b2.edges(data=True):
        pu, pv = r2_old_to_new[u], r2_old_to_new[v]
        if pu == 0 and pv == r2_old_to_new.get(5, n + 4):
            continue  # 不复制共享点上的边
        if not H.has_edge(pu, pv):
            H.add_edge(pu, pv, bond_type=d.get('bond_type', 'single'))
    # 去除重复的 C0 相关边后，添加 C0-b2_C5 作为连接键
    H.add_edge(0, n - 1, bond_type='single')
    return H


# ─────────────────────────────────────────────
# 杂环 (含 O)
# ─────────────────────────────────────────────

def _build_furan() -> nx.Graph:
    """呋喃 (C4H4O, 5元环 Kekulé)

    结构: O1-C2=C3-C4=C5, bonds: single-double-single-double-single
    """
    return _make_ring(
        atoms=['O', 'C', 'C', 'C', 'C'],
        bonds=['single', 'double', 'single', 'double', 'single']
    )


def _build_pyran() -> nx.Graph:
    """2H-吡喃 (C5H6O, 6元环 Kekulé)

    结构: 环内含 1 个 O, 2 个双键
    bonds: O0-C1(single) C1=C2(double) C2-C3(single) C3=C4(double) C4-C5(single) C5-O0(single)
    """
    return _make_ring(
        atoms=['O', 'C', 'C', 'C', 'C', 'C'],
        bonds=['single', 'double', 'single', 'double', 'single', 'single']
    )


def _build_dioxane() -> nx.Graph:
    """1,4-二氧六环 (C4H8O2, 6元环, 两个对位 O)

    结构: O1-C2-C3-O4-C5-C6, 全部单键
    """
    return _make_ring(
        atoms=['O', 'C', 'C', 'O', 'C', 'C'],
        bonds=['single', 'single', 'single', 'single', 'single', 'single']
    )


def _build_tetrahydrofuran() -> nx.Graph:
    """四氢呋喃 THF (C4H8O, 5元饱和环含 O)

    结构: O1-C2-C3-C4-C5, 全单键
    """
    return _make_ring(
        atoms=['O', 'C', 'C', 'C', 'C'],
        bonds=['single', 'single', 'single', 'single', 'single']
    )


def _build_oxirane() -> nx.Graph:
    """环氧乙烷 (C2H4O, 3元环含 O)

    结构: O1-C2-C3, 全单键 (三角形)
    """
    return _make_ring(
        atoms=['O', 'C', 'C'],
        bonds=['single', 'single', 'single']
    )


# ─────────────────────────────────────────────
# 饱和环
# ─────────────────────────────────────────────

def _build_cyclopropane() -> nx.Graph:
    """环丙烷 (C3H6)"""
    return _make_ring(
        atoms=['C', 'C', 'C'],
        bonds=['single', 'single', 'single']
    )


def _build_cyclobutane() -> nx.Graph:
    """环丁烷 (C4H8)"""
    return _make_ring(
        atoms=['C', 'C', 'C', 'C'],
        bonds=['single', 'single', 'single', 'single']
    )


def _build_cyclopentane() -> nx.Graph:
    """环戊烷 (C5H10)"""
    return _make_ring(
        atoms=['C', 'C', 'C', 'C', 'C'],
        bonds=['single', 'single', 'single', 'single', 'single']
    )


def _build_cyclohexane() -> nx.Graph:
    """环己烷 (C6H12)"""
    return _make_ring(
        atoms=['C', 'C', 'C', 'C', 'C', 'C'],
        bonds=['single', 'single', 'single', 'single', 'single', 'single']
    )


# ─────────────────────────────────────────────
# 多环饱和
# ─────────────────────────────────────────────

def _build_norbornane() -> nx.Graph:
    """降冰片烷 (C7H12, 双环[2.2.1]庚烷)

    结构: 6个 C 构成桥环，1个 C 桥，全单键
      C2
     /  \
    C1   C4
    |    |
    C0-C5-C3
    """
    G = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['single', 'single', 'single', 'single', 'single', 'single']
    )
    # 添加桥 C: 连接 C1 和 C4 (当前索引 1 和 4)
    bridge_c = G.number_of_nodes()
    G.add_node(bridge_c, atom='C')
    G.add_edge(bridge_c, 1, bond_type='single')
    G.add_edge(bridge_c, 4, bond_type='single')
    return G


def _build_adamantane() -> nx.Graph:
    """金刚烷 (C10H16, 三环[3.3.1.1³·⁷]癸烷)

    结构: 4 个椅式环己烷稠合，全 sp³ 碳
    """
    # 环己烷的 4 环稠合: 使用编号构建
    # 0-1-2-3-4-5 (椅式), 6-7-8-9 桥接
    G = _make_ring(
        ['C', 'C', 'C', 'C', 'C', 'C'],
        ['single', 'single', 'single', 'single', 'single', 'single']
    )
    # 添加 4 个桥接点 (金刚烷 = 椅式 ×4 稠合)
    n = 6
    for _ in range(4):
        G.add_node(n, atom='C')
        n += 1
    # 连接模式: C0-C6, C2-C6, C0-C7, C1-C7, ...
    # 简化：构建完整的金刚烷骨架
    # C0(桥头), C1-C3(椅C), C4(桥头), C8-C9(桥接)
    # 参考标准编号:
    # 桥头: C0, C4
    # 椅: C1, C2, C3, C5, C6, C7
    # 桥接: C8, C9
    G.add_edge(0, 6, bond_type='single')
    G.add_edge(2, 6, bond_type='single')
    G.add_edge(1, 7, bond_type='single')
    G.add_edge(4, 7, bond_type='single')
    G.add_edge(0, 8, bond_type='single')
    G.add_edge(3, 8, bond_type='single')
    G.add_edge(1, 9, bond_type='single')
    G.add_edge(5, 9, bond_type='single')
    return G


# ─────────────────────────────────────────────
# 碳链 (预留)
# ─────────────────────────────────────────────

def _build_ethyl() -> nx.Graph:
    """乙基 (-CH2-CH3, 链)"""
    return _make_chain(['C', 'C'], ['single'])


# ─────────────────────────────────────────────
# 官能团 (预留)
# ─────────────────────────────────────────────

# 占位: 单原子/双原子官能团可在后续添加。
# 如: -OH, =O, -COOH, -O-, -O-O-, -NH2, -NO2 等


# ============ 完整注册表 ============

# 顺序由类别 + 易读性决定
_registry = [
    # (key, 描述, 类别, builder)
    # ── 芳香环 ──
    ('benzene',       '苯环 (C6H6, Kekulé)',            '芳香环',    _build_benzene),
    ('naphthalene',   '萘环 (C10H8)',                   '芳香环',    _build_naphthalene),
    ('anthracene',    '蒽 (C14H10)',                    '芳香环',    _build_anthracene),
    ('phenanthrene',  '菲 (C14H10)',                    '芳香环',    _build_phenanthrene),
    ('biphenyl',      '联苯 (C12H10)',                  '芳香环',    _build_biphenyl),
    # ── 杂环 ──
    ('furan',         '呋喃 (C4H4O)',                   '杂环',      _build_furan),
    ('pyran',         '2H-吡喃 (C5H6O)',                '杂环',      _build_pyran),
    ('dioxane',       '1,4-二氧六环 (C4H8O2)',          '杂环',      _build_dioxane),
    ('thf',           '四氢呋喃 (C4H8O)',               '杂环',      _build_tetrahydrofuran),
    ('oxirane',       '环氧乙烷 (C2H4O)',               '杂环',      _build_oxirane),
    # ── 饱和环 ──
    ('cyclopropane',  '环丙烷 (C3H6)',                  '饱和环',    _build_cyclopropane),
    ('cyclobutane',   '环丁烷 (C4H8)',                  '饱和环',    _build_cyclobutane),
    ('cyclopentane',  '环戊烷 (C5H10)',                 '饱和环',    _build_cyclopentane),
    ('cyclohexane',   '环己烷 (C6H12)',                 '饱和环',    _build_cyclohexane),
    # ── 多环饱和 ──
    ('norbornane',    '降冰片烷 (C7H12)',               '多环饱和',  _build_norbornane),
    ('adamantane',    '金刚烷 (C10H16)',                '多环饱和',  _build_adamantane),
    # ── 碳链 ──
    ('ethyl',         '乙基 (-CH2-CH3)',                '碳链',      _build_ethyl),
]

for key, desc, cat, builder in _registry:
    _register(key, desc, cat, builder)


# ============ 测试 ============

if __name__ == '__main__':
    print("分子片段库")
    print("=" * 60)
    print(f"已注册 {len(_builders)} 个片段\n")

    print("按类别分组:")
    for cat, keys in list_by_category().items():
        print(f"  [{cat}]")
        for k in keys:
            print(f"     {describe(k)}")

    print(f"\n{'-'*60}")
    print("功能测试:")

    tests = ['benzene', 'furan', 'thf', 'naphthalene',
             'cyclohexane', 'norbornane']
    for key in tests:
        try:
            G = get_fragment(key)
            c = sum(1 for n in G.nodes()
                    if G.nodes[n].get('label', 'C') == 'C')
            o = sum(1 for n in G.nodes()
                    if G.nodes[n].get('label') == 'O')
            bonds = [(u, v, G[u][v].get('bond_type', 'single'))
                     for u, v in G.edges()]
            print(f"  {key:15s} C={c} O={o} edges={len(bonds)}")
        except Exception as e:
            print(f"  {key:15s} ERROR: {e}")
