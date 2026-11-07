# -*- coding: utf-8 -*-
"""特殊基团（硝基等）回归测试

覆盖：
  1. 基团库数据完整性（键名/原子数/键级/电荷个数一致）
  2. `attach_group` 挂载正确性：苯 + 硝基 = 硝基苯 C6H5NO2（分子式 / 门禁 / H 数）
  3. 价态规则：形式正电荷允许的额外键数（N⁺ 允许 4 键）
  4. 渲染与导出链路可用：RDKit 构建成功、`.gjf` 坐标段含 N/O、连通性块可用、
     校验零问题、电荷行按图总电荷
  5. 口径隔离：预言机**不会**产出硝基结构（CH3NO2 与 MAYGEN 一致，均不含）

运行：python tests/test_special_groups.py
"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  —— {detail}" if detail else ""))


def benzene():
    import networkx as nx
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, label='C')
    for i in range(6):
        G.add_edge(i, (i + 1) % 6,
                   bond_type='double' if i % 2 == 0 else 'single')
    return G


def test_library():
    print("\n== 1. 基团库数据完整性 ==")
    from special_groups import list_groups, GROUP_LIBRARY
    check("库非空且含 nitro", 'nitro' in GROUP_LIBRARY, str(sorted(GROUP_LIBRARY)))
    for g in list_groups():
        # 单原子基团（–NH₂/–OH/–F…）没有"组内键"，只与宿主成键 → 不要求 ≥1 条
        ok = ((not g.charges or len(g.charges) == g.n_atoms)
              and g.n_atoms == 1 + len(g.extra_atoms)
              and all(0 <= i < g.n_atoms and 0 <= j < g.n_atoms
                      for i, j, _bt in g.group_bonds)
              and all(i != j for i, j, _bt in g.group_bonds))
        check(f"{g.key}: 原子/组内键/电荷数一致（{g.n_atoms} 原子）", ok,
              f"bonds={g.group_bonds} charges={g.charges}")


def test_attach():
    print("\n== 2. 挂载正确性（苯 + 硝基 → 硝基苯） ==")
    from special_groups import attach_group, describe, net_charge
    from utils import (format_graph_formula, formula_matches_input,
                       graph_formula_parts_full)
    B = benzene()
    check("苯：C6H6", format_graph_formula(B) == 'C6H6', format_graph_formula(B))
    NB = attach_group(B, 0, 'nitro')
    check("硝基苯：分子式 C6H5NO2",
          format_graph_formula(NB) == 'C6H5NO2', format_graph_formula(NB))
    check("硝基苯：parts = (6,5,1,2,(0,0,0,0))",
          graph_formula_parts_full(NB) == (6, 5, 1, 2, (0, 0, 0, 0)),
          str(graph_formula_parts_full(NB)))
    check("硝基苯：分子式门禁通过（含 N=1/O=2）",
          formula_matches_input(NB, 6, 5, 2, n_nitrogen=1))
    check("硝基苯：净电荷 = 0（N⁺/O⁻ 相抵）", net_charge(NB) == 0,
          str(net_charge(NB)))
    check("硝基苯：describe 含『硝基×1』", '硝基×1' in describe(NB), describe(NB))
    check("原图未被修改（纯函数）", format_graph_formula(B) == 'C6H6')
    # 挂两次：第二次应因宿主已无 H 而失败（苯环 C0 已被取代）
    try:
        attach_group(NB, 0, 'nitro')
        check("已取代位置再次挂载被拒绝", False, "未抛 ValueError")
    except ValueError:
        check("已取代位置再次挂载被拒绝", True)
    # 其他基团
    NS = attach_group(B, 0, 'nitroso')
    check("苯 + 亚硝基 = C6H5NO", format_graph_formula(NS) == 'C6H5NO',
          format_graph_formula(NS))
    AZ = attach_group(B, 0, 'azide')
    check("苯 + 叠氮 = C6H5N3", format_graph_formula(AZ) == 'C6H5N3',
          format_graph_formula(AZ))


def _raises(fn):
    try:
        fn()
        return None
    except Exception:
        return True


def test_valence_rule():
    print("\n== 3. 价态规则（形式电荷修正） ==")
    from utils import atom_valence
    check("中性 N 允许 3 键", atom_valence('N', {}) == 3)
    check("N⁺ 允许 4 键（阳离子多成键）",
          atom_valence('N', {'formal_charge': 1}) == 4)
    check("O⁻ 只允许 1 键（阴离子少成键 → 不补 H）",
          atom_valence('O', {'formal_charge': -1}) == 1)
    check("C⁻ 允许 3 键", atom_valence('C', {'formal_charge': -1}) == 3)
    check("valence_override 优先",
          atom_valence('N', {'valence_override': 5}) == 5)
    from Nitrogen.formula import atom_valence as n_val
    check("Nitrogen.formula 同口径（N⁺=4 / O⁻=1）",
          n_val('N', {'formal_charge': 1}) == 4
          and n_val('O', {'formal_charge': -1}) == 1)


def test_render_export():
    print("\n== 4. 渲染与导出链路 ==")
    from special_groups import attach_group
    from utils import render_skeletal_formula
    from gaussian_export import build_gjf, validate_gjf, hydrogen_counts
    try:
        from Main import _build_rdkit_mol
    except Exception as e:                      # 无 GUI 依赖时跳过
        check("导入 Main 构建器", False, str(e))
        return
    NB = attach_group(benzene(), 0, 'nitro')
    mol = _build_rdkit_mol(NB)
    check("RDKit 构建成功（电荷分离式可 sanitize）", mol is not None)
    if mol is not None:
        from rdkit import Chem
        check("RDKit 中含 1 个 N 与 2 个 O",
              sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N') == 1
              and sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O') == 2)
        check("RDKit 总 H = 5（无额外 H）",
              mol.GetNumAtoms() - NB.number_of_nodes() == 5,
              str(mol.GetNumAtoms()))
        check("SMILES 可生成", bool(Chem.MolToSmiles(mol)))
    png = render_skeletal_formula(NB, 'nitrogen', img_size=(380, 220))
    check("2D 键线式出图", bool(png) and png[:8] == b'\x89PNG\r\n\x1a\n')

    elem = lambda n, _G=NB: _G.nodes[n].get('label', 'C')       # noqa: E731
    counts = hydrogen_counts(NB, elem)
    heavy = {0: (0.0, 0.0, 0.0), 1: (1.39, 0.0, 0.0), 2: (2.09, 1.20, 0.0),
             3: (1.39, 2.41, 0.0), 4: (0.0, 2.41, 0.0), 5: (-0.70, 1.20, 0.0),
             6: (1.40, -1.35, 0.0), 7: (2.62, -1.35, 0.0), 8: (0.79, -1.50, 0.9)}
    from tests.test_gjf_export import place_hydrogens, _parse_blocks
    res = build_gjf("C6H5NO2 nitrobenzene Isomer #1", heavy,
                    place_hydrogens(heavy, counts), NB, element_of=elem)
    elems = [l.split()[0] for l in _parse_blocks(res.text)["atoms"]]
    check("导出：坐标段元素 = 6C + 1N + 2O", elems.count('C') == 6
          and elems.count('N') == 1 and elems.count('O') == 2, str(elems))
    check("导出：H 数 = 5", elems.count('H') == 5, str(elems))
    check("导出：连通性块可用", res.connectivity, str(res.warnings))
    check("导出：电荷行 = 0 1（净电中性）",
          _parse_blocks(res.text)["charge_line"].strip() == '0 1',
          _parse_blocks(res.text)["charge_line"])
    check("导出：校验零问题", validate_gjf(res.text) == [],
          str(validate_gjf(res.text)))


def test_not_in_enumeration():
    print("\n== 5. 口径隔离：硝基不进枚举 ==")
    from Nitrogen.atomic_matrix_n import AtomicMatrixNGenerator
    from Nitrogen.formula import graph_formula_parts
    # CH3NO2：N=3 口径下的结构数（已与 MAYGEN 对拍 = 15，且其中无硝基）
    gs = AtomicMatrixNGenerator(1, 1, 2).generate_for_target_h(3)
    check("CH3NO2 结构数 = 15（与 MAYGEN 默认价态一致）", len(gs) == 15,
          str(len(gs)))
    check("全部结构都是中性八隅律（无 valence_override/formal_charge）",
          all(all('valence_override' not in d and 'formal_charge' not in d
                  for _, d in g.nodes(data=True)) for g in gs))
    # 硝基需要的键级组合（N 双键连两个 O）在枚举中不存在
    has_n_double2 = any(
        sum(1 for nb in g.neighbors(n)
            if g.nodes[nb].get('label') == 'O'
            and g[n][nb].get('bond_type') == 'double') >= 2
        for g in gs for n in g.nodes() if g.nodes[n].get('label') == 'N')
    check("枚举中不存在 N(=O)(=O)（硝基）", not has_n_double2)


def main():
    print("=" * 70)
    print("特殊基团（硝基 / 亚硝基 / 叠氮）回归测试")
    print("=" * 70)
    test_library()
    test_attach()
    test_valence_rule()
    test_render_export()
    test_not_in_enumeration()
    print("\n" + "=" * 70)
    print(f"结果: PASS={PASS}  FAIL={FAIL}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
