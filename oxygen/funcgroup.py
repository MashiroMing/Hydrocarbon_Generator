# -*- coding: utf-8 -*-
"""含氧官能团分类（GUI 精细筛选用）

一个结构可命中多个标签（返回 frozenset），与 O 分布（互斥）配合使用：

标签:
  'alcohol'       醇        -OH（非苯环 C 上，且该 C 无 =O）
  'phenol'        酚        -OH 直接连凯库勒苯环 C
  'ether'         醚        开链/桥 C-O-C（O 邻接 2 个 C、不在环中、非酯 O）
  'peroxide'      过氧/氢过氧  O-O 键
  'aldehyde'      醛        -CHO（=O C 上无 OH、C 邻居 ≤1、无 O 邻居）
  'ketone'        酮        >C=O（=O C 上无 OH、C 邻居 ≥2）
  'acid'          羧酸      同碳 (OH, =O)
  'ester'         酯        C(=O)-O-C（=O C 邻接骨架 O）
  'epoxide'       环氧      3 元 C-C-O 环
  'o_heterocycle' 含氧杂环   O 在 ≥4 元环中（呋喃/THF/二氧六环/二氧戊环…）
  'gem_diol'      偕二醇    同碳双 -OH（数学口径合法）

O 分布（互斥单选）:
  'substituent_only'  全部取代基 O（仅 -OH/=O，无骨架 O）
  'backbone_only'     全部骨架 O（仅醚/过氧/环氧/杂环，无 -OH/=O）
  'mixed'             混合
  'none'              无 O（不应出现在含氧调用）

设计原则:
- 纯图快判优先（O 节点 / oxo_counts / bond_type），零 RDKit、零 SMARTS；
- 苯环判定与 fragment_library 的凯库勒苯环语义一致（6 元 C 环、单双键交替），
  支持被烷基/羟基取代的苯环（环 C 度可 >2）与稠合苯环（萘类）；
- 酯/酸/醚判定与 tests.test_formula_guard.coverage_stats 口径同源；
- 酯的 C-O-C 不算醚（邻接 =O C 的骨架 O 仅标注酯），环氧/含氧杂环/醚三类
  按 O 所在环情况互斥区分。
"""
import networkx as nx

TAG_CN = {
    'alcohol': '醇',
    'phenol': '酚',
    'ether': '醚',
    'peroxide': '过氧',
    'aldehyde': '醛',
    'ketone': '酮',
    'acid': '羧酸',
    'ester': '酯',
    'epoxide': '环氧',
    'o_heterocycle': '含氧杂环',
    'gem_diol': '偕二醇',
}

O_DIST_CN = {
    'substituent_only': '全部取代基O',
    'backbone_only': '全部骨架O',
    'mixed': '混合',
    'none': '无O',
}

__all__ = ['classify', 'o_distribution', 'TAG_CN', 'O_DIST_CN']


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _bond_type(G, u, v):
    return G[u][v].get('bond_type', 'single')


def _is_alternating(bonds):
    """6 条边是否构成交替单双键（凯库勒式）"""
    if any(b == 'triple' for b in bonds):
        return False
    for pat in (('single', 'double'), ('double', 'single')):
        if all(bonds[i] == pat[i % 2] for i in range(6)):
            return True
    return False


def _on_kekule_benzene_ring(G, n):
    """n（C 原子）是否位于凯库勒苯环（6 个 C、单双键交替）上（纯图判定）。

    对 n 的 C 邻居两两组合，找 4 边路径构成 6 环，再校验键交替：
    - 支持被烷基/羟基/羧基等取代的苯环（环 C 在 C 子图中度可 >2）；
    - 支持稠合苯环（萘类，两两组合只会命中环内 4 边路径）；
    - 按需单节点判定：无 OH 的图完全跳过本函数（大集合性能关键）。
    """
    c_nodes = [x for x in G.nodes() if G.nodes[x].get('label', 'C') == 'C']
    c_sub = G.subgraph(c_nodes)
    cnb = list(c_sub.neighbors(n))
    for i in range(len(cnb)):
        for j in range(i + 1, len(cnb)):
            a, b = cnb[i], cnb[j]
            for p in nx.all_simple_paths(c_sub, a, b, cutoff=4):
                if len(p) != 5 or n in p:
                    continue
                cycle = [n] + p  # n-a-…-b → 6 个 C
                bonds = [_bond_type(G, cycle[k], cycle[k + 1])
                         for k in range(5)]
                bonds.append(_bond_type(G, cycle[5], cycle[0]))
                if _is_alternating(bonds):
                    return True
    return False


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def classify(G) -> frozenset:
    """返回 G 的含氧官能团标签集合（可多类，纯图判定，无 RDKit）"""
    tags = set()

    # ── 取代基 O（oxo_counts）统计 ──
    oh_nodes = []          # (node, oh, co)
    co_nodes = []
    gem_diol = False
    for n, d in G.nodes(data=True):
        if d.get('label', 'C') != 'C':
            continue
        oh, co = d.get('oxo_counts', (0, 0))
        if oh >= 1:
            oh_nodes.append((n, oh, co))
        if co >= 1:
            co_nodes.append(n)
        if oh >= 2:
            gem_diol = True

    # ── 骨架 O（label='O' 节点）分类 ──
    o_nodes = [n for n, d in G.nodes(data=True)
               if d.get('label', 'C') == 'O']

    # 过氧 / 氢过氧：存在 O-O 键
    if any(G.has_edge(o1, o2) for i, o1 in enumerate(o_nodes)
           for o2 in o_nodes[i + 1:]):
        tags.add('peroxide')

    # 醚 / 环氧 / 含氧杂环：按 O 邻居情况互斥区分
    for o in o_nodes:
        nb = list(G.neighbors(o))
        if len(nb) != 2:
            continue
        a, b = nb
        if (G.nodes[a].get('label', 'C') != 'C'
                or G.nodes[b].get('label', 'C') != 'C'):
            continue
        if G.has_edge(a, b):
            # 两 C 直接相连 → 3 元 C-C-O 环氧环
            tags.add('epoxide')
            continue
        # 绕开 o 后 a、b 仍连通 → o 在 ≥4 元环中
        G2 = G.copy()
        G2.remove_node(o)
        if nx.has_path(G2, a, b):
            tags.add('o_heterocycle')
            continue
        # 开链/桥醚：且邻接 C 不带 =O（酯 O 单独标注，不算醚）
        if not (G.nodes[a].get('oxo_counts', (0, 0))[1]
                or G.nodes[b].get('oxo_counts', (0, 0))[1]):
            tags.add('ether')

    # ── 醇 / 酚（无 OH 的图自动跳过苯环检测）──
    has_alcohol = has_phenol = False
    for n, oh, co in oh_nodes:
        if co >= 1:
            continue  # 羧酸 OH（酸单独标注）
        if _on_kekule_benzene_ring(G, n):
            has_phenol = True
        else:
            has_alcohol = True
        if has_phenol and has_alcohol:
            break
    if has_phenol:
        tags.add('phenol')
    if has_alcohol:
        tags.add('alcohol')
    if gem_diol:
        tags.add('gem_diol')

    # ── 酸 / 酯 / 醛 / 酮 ──
    for n in co_nodes:
        oh, _co = G.nodes[n].get('oxo_counts', (0, 0))
        c_neighbors = [x for x in G.neighbors(n)
                       if G.nodes[x].get('label', 'C') == 'C']
        o_neighbors = [x for x in G.neighbors(n)
                       if G.nodes[x].get('label', 'C') == 'O']
        if oh >= 1:
            tags.add('acid')      # 同碳 (OH, =O)
            continue
        if o_neighbors:
            tags.add('ester')     # C(=O)-O-C
            continue
        if len(c_neighbors) <= 1:
            tags.add('aldehyde')  # R-CHO（含 HCHO）
        else:
            tags.add('ketone')    # R-CO-R'

    return frozenset(tags)


def o_distribution(G) -> str:
    """O 原子分布形态（互斥）：substituent_only / backbone_only / mixed / none

    骨架 O = label=='O' 节点（醚/过氧/环氧/杂环 O）；
    取代基 O = Σ(oxo_counts 的 OH + CO)。
    """
    n_backbone = 0
    n_subst = 0
    for n, d in G.nodes(data=True):
        if d.get('label', 'C') == 'O':
            n_backbone += 1
        else:
            oh, co = d.get('oxo_counts', (0, 0))
            n_subst += oh + co
    if n_backbone == 0 and n_subst == 0:
        return 'none'
    if n_backbone == 0:
        return 'substituent_only'
    if n_subst == 0:
        return 'backbone_only'
    return 'mixed'
