# -*- coding: utf-8 -*-
"""
官能团分类模块测试 (oxygen/funcgroup.py)

手工构造图（优先用 fragment_library 苯环预结构）+ 小型生成集成：
  1. 14 个代表性分子的标签集与 O 分布断言（纯图判定）
  2. 生成集成：C3H6O 全集合标签覆盖；苯环约束集合中
     苯甲醚/苯甲醇/乙酸苯酯/苯甲酸甲酯 的标签正确；C6H6O 含酚/环氧/醇

运行：
  python tests/test_funcgroup.py        # 直接运行
  被 tests/test_formula_guard.py --fast 集成调用（run_all）
  退出码 0 = 全部通过
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from oxygen.funcgroup import classify, o_distribution
from oxygen.fragment_library import get_fragment

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# 手工构造图（苯环一律来自 fragment_library 预结构）
# ---------------------------------------------------------------------------

def anisole():
    """苯甲醚 C7H8O：苯环 - O - CH3"""
    G = get_fragment('benzene')
    G.add_node(6, label='O')
    G.add_node(7, label='C')
    G.add_edge(0, 6, bond_type='single')
    G.add_edge(6, 7, bond_type='single')
    return G


def phenol():
    """苯酚 C6H6O：苯环 C 上 -OH"""
    G = get_fragment('benzene')
    G.nodes[0]['oxo_counts'] = (1, 0)
    return G


def benzyl_alcohol():
    """苯甲醇 C7H8O：苯环 - CH2OH"""
    G = get_fragment('benzene')
    G.add_node(6, label='C')
    G.nodes[6]['oxo_counts'] = (1, 0)
    G.add_edge(0, 6, bond_type='single')
    return G


def cyclohexanol():
    """环己醇（全单键 6 元环，非苯环 → 醇而非酚）"""
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, label='C')
    for i in range(6):
        G.add_edge(i, (i + 1) % 6, bond_type='single')
    G.nodes[0]['oxo_counts'] = (1, 0)
    return G


def benzaldehyde():
    """苯甲醛 C7H6O：苯环 - CHO"""
    G = get_fragment('benzene')
    G.add_node(6, label='C')
    G.nodes[6]['oxo_counts'] = (0, 1)
    G.add_edge(0, 6, bond_type='single')
    return G


def acetophenone():
    """苯乙酮 C8H8O：苯环 - C(=O) - CH3"""
    G = get_fragment('benzene')
    G.add_node(6, label='C')
    G.nodes[6]['oxo_counts'] = (0, 1)
    G.add_node(7, label='C')
    G.add_edge(0, 6, bond_type='single')
    G.add_edge(6, 7, bond_type='single')
    return G


def benzoic_acid():
    """苯甲酸 C7H6O2：苯环 - C(=O)OH"""
    G = get_fragment('benzene')
    G.add_node(6, label='C')
    G.nodes[6]['oxo_counts'] = (1, 1)
    G.add_edge(0, 6, bond_type='single')
    return G


def phenyl_acetate():
    """乙酸苯酯 C8H8O2：苯环 - O - C(=O) - CH3"""
    G = get_fragment('benzene')
    G.add_node(6, label='O')
    G.add_node(7, label='C')
    G.nodes[7]['oxo_counts'] = (0, 1)
    G.add_node(8, label='C')
    G.add_edge(0, 6, bond_type='single')
    G.add_edge(6, 7, bond_type='single')
    G.add_edge(7, 8, bond_type='single')
    return G


def methyl_benzoate():
    """苯甲酸甲酯 C8H8O2：苯环 - C(=O) - O - CH3"""
    G = get_fragment('benzene')
    G.add_node(6, label='C')
    G.nodes[6]['oxo_counts'] = (0, 1)
    G.add_node(7, label='O')
    G.add_node(8, label='C')
    G.add_edge(0, 6, bond_type='single')
    G.add_edge(6, 7, bond_type='single')
    G.add_edge(7, 8, bond_type='single')
    return G


def propylene_oxide():
    """环氧丙烷 C3H6O：3 元 C-C-O 环 + CH3"""
    G = nx.Graph()
    for i in range(3):
        G.add_node(i, label='C')
    G.add_node(3, label='O')
    G.add_edge(0, 1, bond_type='single')
    G.add_edge(1, 3, bond_type='single')
    G.add_edge(3, 0, bond_type='single')
    G.add_edge(1, 2, bond_type='single')
    return G


def dimethyl_ether():
    """二甲醚 C2H6O：CH3 - O - CH3"""
    G = nx.Graph()
    G.add_node(0, label='C')
    G.add_node(1, label='O')
    G.add_node(2, label='C')
    G.add_edge(0, 1, bond_type='single')
    G.add_edge(1, 2, bond_type='single')
    return G


def ethyl_hydroperoxide():
    """乙基过氧化氢 C2H6O2：CH3 - CH2 - O - O - H"""
    G = nx.Graph()
    for i in range(3):
        G.add_node(i, label='C')
    G.add_node(3, label='O')
    G.add_node(4, label='O')
    G.add_edge(0, 1, bond_type='single')
    G.add_edge(1, 3, bond_type='single')
    G.add_edge(3, 4, bond_type='single')
    return G


def gem_diol():
    """偕二醇 CH2(OH)2"""
    G = nx.Graph()
    G.add_node(0, label='C')
    G.nodes[0]['oxo_counts'] = (2, 0)
    return G


def dioxane():
    """1,4-二氧六环 C4H8O2：C-O-C-C-O-C 六元环"""
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, label='C' if i in (0, 2, 3, 5) else 'O')
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
        G.add_edge(u, v, bond_type='single')
    return G


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------

def _gen_buckets(formula, constraint=None):
    """复刻 GUI 含氧分支生成调用"""
    from utils import parse_compound_formula
    from oxygen.unified_oxo_generator import UnifiedOxoGenerator
    res = parse_compound_formula(formula)
    _, nc, nh, no, *_ = res
    k = (2 * nc + 2 - nh) // 2
    return UnifiedOxoGenerator().generate(nc, no, k_range=(k, k),
                                          constraints=constraint)


def _smi_tag_map(buckets):
    """图集合 → {规范 SMILES: 标签集}（凯库勒变体合并）"""
    from utils import graph_to_rdkit_mol
    from rdkit import Chem
    out = {}
    for items in buckets.values():
        for mt, G in items:
            m = graph_to_rdkit_mol(G)
            if m is None:
                continue
            out.setdefault(Chem.MolToSmiles(m), set()).update(classify(G))
    return out


def run_all():
    """返回失败断言数（供 test_formula_guard.py 集成调用）"""
    global PASS, FAIL
    PASS = FAIL = 0

    # ── 1) 手工构造图：标签集 + O 分布 ──
    cases = [
        ("苯甲醚", anisole(), {'ether'}, 'backbone_only'),
        ("苯酚", phenol(), {'phenol'}, 'substituent_only'),
        ("苯甲醇", benzyl_alcohol(), {'alcohol'}, 'substituent_only'),
        ("环己醇", cyclohexanol(), {'alcohol'}, 'substituent_only'),
        ("苯甲醛", benzaldehyde(), {'aldehyde'}, 'substituent_only'),
        ("苯乙酮", acetophenone(), {'ketone'}, 'substituent_only'),
        ("苯甲酸", benzoic_acid(), {'acid'}, 'substituent_only'),
        ("乙酸苯酯", phenyl_acetate(), {'ester'}, 'mixed'),
        ("苯甲酸甲酯", methyl_benzoate(), {'ester'}, 'mixed'),
        ("环氧丙烷", propylene_oxide(), {'epoxide'}, 'backbone_only'),
        ("二甲醚", dimethyl_ether(), {'ether'}, 'backbone_only'),
        ("乙基过氧化氢", ethyl_hydroperoxide(), {'peroxide'}, 'backbone_only'),
        ("偕二醇", gem_diol(), {'alcohol', 'gem_diol'}, 'substituent_only'),
        ("1,4-二氧六环", dioxane(), {'o_heterocycle'}, 'backbone_only'),
    ]
    for name, G, exp_tags, exp_dist in cases:
        tags = classify(G)
        check(f"分类: {name} 标签 == {sorted(exp_tags)}",
              tags == frozenset(exp_tags), f"实际 {sorted(tags)}")
        check(f"分类: {name} O分布 == {exp_dist}",
              o_distribution(G) == exp_dist, f"实际 {o_distribution(G)}")

    # ── 2) 生成集成 ──
    from structure_filter import StructureConstraint

    b = _gen_buckets("C3H6O")
    tags_all = set()
    for items in b.values():
        for _mt, G in items:
            tags_all |= classify(G)
    for t, nm in [('aldehyde', '醛'), ('ketone', '酮'), ('epoxide', '环氧'),
                  ('alcohol', '醇'), ('ether', '醚')]:
        check(f"集成: C3H6O 全集合含{nm}标签", t in tags_all,
              f"标签全集 {sorted(tags_all)}")

    b = _gen_buckets("C7H8O", StructureConstraint(required_fragment='benzene'))
    smi_tag = _smi_tag_map(b)
    check("集成: C7H8O+苯环 苯甲醚 标签含醚",
          'COc1ccccc1' in smi_tag and 'ether' in smi_tag['COc1ccccc1'],
          f"苯甲醚标签 {smi_tag.get('COc1ccccc1')}")
    check("集成: C7H8O+苯环 苯甲醇 标签含醇",
          'OCc1ccccc1' in smi_tag and 'alcohol' in smi_tag['OCc1ccccc1'],
          f"苯甲醇标签 {smi_tag.get('OCc1ccccc1')}")

    b = _gen_buckets("C8H8O2", StructureConstraint(required_fragment='benzene'))
    smi_tag = _smi_tag_map(b)
    check("集成: C8H8O2+苯环 乙酸苯酯 标签含酯",
          'CC(=O)Oc1ccccc1' in smi_tag and 'ester' in smi_tag['CC(=O)Oc1ccccc1'],
          f"乙酸苯酯标签 {smi_tag.get('CC(=O)Oc1ccccc1')}")
    check("集成: C8H8O2+苯环 苯甲酸甲酯 标签含酯",
          'COC(=O)c1ccccc1' in smi_tag and 'ester' in smi_tag['COC(=O)c1ccccc1'],
          f"苯甲酸甲酯标签 {smi_tag.get('COC(=O)c1ccccc1')}")

    b = _gen_buckets("C6H6O")  # k=3 全集合（含苯酚/苯并环氧/环己二烯酮/醛）
    tags_all = set()
    for items in b.values():
        for _mt, G in items:
            tags_all |= classify(G)
    # C6H6O 唯一的 -OH 在苯环上（酚），无非环 sp3 C 可挂 OH → 无醇标签
    check("集成: C6H6O 含酚标签", 'phenol' in tags_all,
          f"标签全集 {sorted(tags_all)}")
    check("集成: C6H6O 含环氧标签（苯并环氧）", 'epoxide' in tags_all)
    check("集成: C6H6O 含酮标签（环己二烯酮）", 'ketone' in tags_all)
    check("集成: C6H6O 含醛标签", 'aldehyde' in tags_all)

    return FAIL


if __name__ == "__main__":
    print("=" * 60)
    print("官能团分类测试 (oxygen/funcgroup.py)")
    print("=" * 60)
    n_fail = run_all()
    print(f"\n结果: PASS={PASS}  FAIL={n_fail}")
    sys.exit(1 if n_fail else 0)
