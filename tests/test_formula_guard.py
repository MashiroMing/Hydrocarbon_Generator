# -*- coding: utf-8 -*-
"""
分子式一致性门禁回归测试

覆盖（对应修复方案第 6 步）：
  1. 公式电池：对多种分子式断言 UnifiedOxoGenerator 返回的所有桶公式 == 输入公式
     （修复前：C7H8O 泄漏 C7H6O、C6H8O 泄漏 C6H6O、C5H6O 泄漏 C5H12O...C5H0O 等）
  2. 约束用例：C7H8O+苯环 / C7H8O+呋喃（片段路径）
  3. 负例：formula_matches_input 对已知泄漏图返回 False，对合法图返回 True
  4. GUI 后端流程模拟：展平桶 → 门禁 → 断言全部公式匹配
  5. _pure_hydrocarbon 缩进修复：generate(n_carbon, 0) 不再崩溃且桶公式正确

运行（分级，按需选择）：
  python tests/test_formula_guard.py --smoke   # 秒级：小分子 + 单元断言
  python tests/test_formula_guard.py --fast    # 分钟级：中小分子 + 约束用例
  python tests/test_formula_guard.py           # 完整：全部（重量级用例并行）
  退出码 0 = 全部通过

提速说明：
  - coverage_stats 对 RDKit 子结构统计采用固定种子抽样（每桶 ≤ _COV_SAMPLE），
    覆盖断言只需"存在性"，抽样结论等价；
  - 重量级公式电池用 ProcessPoolExecutor 并行（机器多核时墙钟大幅下降）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from utils import (parse_compound_formula, graph_formula_parts,
                   formula_matches_input, format_graph_formula)
from structure_filter import filter_by_expected_formula, StructureConstraint
from oxygen.unified_oxo_generator import UnifiedOxoGenerator

PASS = 0
FAIL = 0

# RDKit 子结构统计抽样上限（覆盖断言只判存在性，抽样结论等价）
_COV_SAMPLE = 300


def check(name, cond, detail="", results=None):
    """断言：results 非空时收集到列表（并行子进程用，不打印）；否则全局计数并打印"""
    if results is not None:
        results.append((name, bool(cond), detail))
        return
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def gen_buckets(formula, constraint=None, use_parallel=False):
    """复刻 GUI 含氧分支的生成调用：k_range=(k_user, k_user)

    use_parallel 默认 False：测试保持串行确定性；并行路径由专门的
    "并行/串行等价" 用例覆盖（见 main() 完整档）。
    """
    res = parse_compound_formula(formula)
    _, nc, nh, no, halo, _, err = res
    assert not err, f"parse error: {err}"
    h_equiv = nh + sum(halo)
    k = (2 * nc + 2 - h_equiv) // 2 if h_equiv else 3
    k = max(k, 0)
    uog = UnifiedOxoGenerator()
    return uog.generate(nc, no, k_range=(k, k), constraints=constraint,
                        use_parallel=use_parallel)


def expected_formula_str(formula):
    res = parse_compound_formula(formula)
    _, nc, nh, no, halo, _, _ = res
    s = f"CH{nh}" if nc == 1 else f"C{nc}H{nh}"
    if no:
        s += f"O{no}" if no > 1 else "O"
    for elem, cnt in [('Br', halo[2]), ('Cl', halo[1]),
                      ('F', halo[0]), ('I', halo[3])]:
        if cnt:
            s += f"{elem}{cnt}" if cnt > 1 else elem
    return s


def test_battery(formula, constraint=None, tag="", allow_empty=False,
                 results=None):
    print(f"\n== 公式电池: {formula}{tag} ==")
    buckets = gen_buckets(formula, constraint)
    expect = expected_formula_str(formula)
    fmls = sorted(buckets.keys())
    total = sum(len(v) for v in buckets.values())
    if allow_empty and not buckets:
        # 片段与输入 k 不兼容（如呋喃 k=3 无法产出 k=4 的 C7H8O）→ 空结果合法
        check(f"{formula}{tag}: 无产物（允许），且无旧版泄漏", True,
              results=results)
        return buckets
    check(f"{formula}{tag}: 唯一桶公式 == {expect}",
          fmls == [expect], f"实际桶: {fmls}", results=results)
    # 抽查每个桶前 3 个图：图实际公式与桶键一致（校验 compute_graph_formula 口径）
    for fml in fmls:
        for mt, G in buckets[fml][:3]:
            actual = format_graph_formula(G)
            check(f"{formula}{tag}: 图公式 {actual} == 桶键 {fml}",
                  actual == fml, f"桶键 {fml} vs 图 {actual}", results=results)
    # 全量：所有图公式 == 输入（纯图计算，无 RDKit，成本可忽略）
    bad = sum(1 for items in buckets.values()
              for mt, G in items if format_graph_formula(G) != expect)
    check(f"{formula}{tag}: 全部 {total} 个图公式 == {expect}（不符 {bad} 个）",
          bad == 0, results=results)
    return buckets


# ── 手工构造图 ──

def kekule_ring(n=6, first_double=True):
    """Kekulé 环：节点 0..n-1，交替单双键"""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, label='C')
    for i in range(n):
        j = (i + 1) % n
        double = (i % 2 == 0) if first_double else (i % 2 == 1)
        G.add_edge(i, j, bond_type='double' if double else 'single')
    return G


def benzene_with(node0_substituent=None, node0_oxo=None, node0_halo=None):
    """苯环 + C0 上接取代基/氧/卤素（返回 7 节点苯甲醛/苯甲醇骨架）"""
    G = kekule_ring(6)
    if node0_substituent is not None:
        n = G.number_of_nodes()
        G.add_node(n, label='C')
        G.add_edge(0, n, bond_type='single')
        if node0_oxo:
            G.nodes[n]['oxo_counts'] = node0_oxo
        if node0_halo:
            G.nodes[n]['halogen_counts'] = node0_halo
    elif node0_oxo or node0_halo:
        if node0_oxo:
            G.nodes[0]['oxo_counts'] = node0_oxo
        if node0_halo:
            G.nodes[0]['halogen_counts'] = node0_halo
    return G


def pentane_with_oh():
    """戊烷骨架 C1-C2-C3-C4-C5 + C1 上 OH → 戊醇 C5H12O"""
    G = nx.Graph()
    for i in range(5):
        G.add_node(i, label='C')
    for i in range(4):
        G.add_edge(i, i + 1, bond_type='single')
    G.nodes[0]['oxo_counts'] = (1, 0)
    return G


def test_negative_cases():
    print("\n== 负例：formula_matches_input 单元断言 ==")
    # 苯甲醛 C6H5-CHO：=O 使 k+1（输入 C7H8O 时应被拒）
    benzaldehyde = benzene_with(node0_substituent=True, node0_oxo=(0, 1))
    check("苯甲醛 C7H6O: 图公式 == C7H6O",
          format_graph_formula(benzaldehyde) == "C7H6O",
          format_graph_formula(benzaldehyde))
    check("苯甲醛: 对输入 C7H8O 返回 False", not formula_matches_input(
        benzaldehyde, 7, 8, 1))
    check("苯甲醛: 对输入 C7H6O 返回 True", formula_matches_input(
        benzaldehyde, 7, 6, 1))

    # 苯甲醇 C6H5-CH2OH：合法 C7H8O
    benzyl_alcohol = benzene_with(node0_substituent=True, node0_oxo=(1, 0))
    check("苯甲醇: 图公式 == C7H8O", format_graph_formula(benzyl_alcohol) == "C7H8O",
          format_graph_formula(benzyl_alcohol))
    check("苯甲醇: 对输入 C7H8O 返回 True", formula_matches_input(
        benzyl_alcohol, 7, 8, 1))

    # 戊醇 C5H12O：输入 C5H6O 时应被拒（原子矩阵全 k 泄漏的典型）
    pentanol = pentane_with_oh()
    check("戊醇: 图公式 == C5H12O", format_graph_formula(pentanol) == "C5H12O",
          format_graph_formula(pentanol))
    check("戊醇: 对输入 C5H6O 返回 False", not formula_matches_input(
        pentanol, 5, 6, 1))
    check("戊醇: 对输入 C5H12O 返回 True", formula_matches_input(
        pentanol, 5, 12, 1))

    # 苯酚 C6H6O（合法）+ 氯苯 C6H5Cl（卤素口径）
    phenol = benzene_with(node0_oxo=(1, 0))
    check("苯酚: 图公式 == C6H6O", format_graph_formula(phenol) == "C6H6O",
          format_graph_formula(phenol))
    check("苯酚: 对输入 C6H8O 返回 False", not formula_matches_input(
        phenol, 6, 8, 1))
    chlorobenzene = benzene_with(node0_halo=(0, 1, 0, 0))
    check("氯苯: 图公式 == C6H5Cl", format_graph_formula(chlorobenzene) == "C6H5Cl",
          format_graph_formula(chlorobenzene))
    check("氯苯: 对输入 C6H5Cl(H=5) 返回 True",
          formula_matches_input(chlorobenzene, 6, 5, 0, (0, 1, 0, 0)))
    check("氯苯: 对无卤输入 C6H6 返回 False",
          not formula_matches_input(chlorobenzene, 6, 6, 0, (0, 0, 0, 0)))


def test_gui_gate_simulation(formula, results=None):
    print(f"\n== GUI 后端流程模拟: {formula} ==")
    buckets = gen_buckets(formula)
    all_isomers = [(mt, G) for items in buckets.values()
                   for mt, G in items]
    res = parse_compound_formula(formula)
    _, nc, nh, no, halo, nd, _ = res
    before = len(all_isomers)
    filtered = filter_by_expected_formula(all_isomers, nc, nh, no, halo, nd)
    expect = expected_formula_str(formula)
    after = len(filtered)
    ok_all = all(format_graph_formula(G) == expect for mt, G in filtered)
    check(f"{formula}: 门禁后 {before} → {after}，全部公式匹配（{ok_all}）",
          ok_all, f"仍有不符: {sum(1 for mt, G in filtered if format_graph_formula(G) != expect)}",
          results=results)
    return before, after


def test_pure_hydrocarbon_fix():
    print("\n== _pure_hydrocarbon 缩进修复 ==")
    uog = UnifiedOxoGenerator()
    try:
        buckets = uog.generate(7, 0, k_range=(0, 4))
        fmls = sorted(buckets.keys())
        check("generate(7,0,(0,4)) 不再崩溃且桶公式正确",
              fmls == ['C7H10', 'C7H12', 'C7H14', 'C7H16', 'C7H8'],
              f"实际桶: {fmls}")
    except AttributeError as e:
        check("generate(7,0,(0,4)) 不再崩溃", False, f"AttributeError: {e}")


def coverage_stats(buckets):
    """统计桶集合的官能团覆盖（完备性断言用，数学口径）

    n_co: 含 C=O（醛/酮/羧酸/酯）
    n_ether: 含骨架 O（醚/酯/杂环）
    n_ester: 含酯键 -C-O-C(=O)-
    n_acid: 含羧基（同碳 (OH, =O)）
    n_epoxide: 含 3 元 CCO 环氧环
    n_epoxide_co: 环氧 + 羰基组合
    n_oo: 含 O-O 键（过氧/氢过氧/过氧环）
    n_gem_diol: 含偕二醇 C(OH)2（数学口径下应存在）

    提速：oxo_counts 类统计（n_co/n_ether/n_ester/n_acid/total）全量遍历（纯图操作，
    无 RDKit，成本可忽略）；RDKit 子结构统计（环氧/过氧/偕二醇）按固定种子抽样，
    每桶 ≤ _COV_SAMPLE 个 —— 覆盖断言只判存在性，抽样结论等价。
    """
    import random
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    p_epox = Chem.MolFromSmarts('[#6]1-[#6]-[#8]-1')
    p_co = Chem.MolFromSmarts('[CX3]=[OX1]')
    p_oo = Chem.MolFromSmarts('[#8]-[#8]')
    p_gem = Chem.MolFromSmarts('[#6]([#8H1])([#8H1])')
    from utils import graph_to_rdkit_mol

    rng = random.Random(20260814)  # 固定种子，可复现

    (n_co, n_ether, n_ester, n_acid, n_epoxide,
     n_epoxide_co, n_oo, n_gem_diol) = (0, 0, 0, 0, 0, 0, 0, 0)
    total = 0
    for items in buckets.values():
        # ── 全量：oxo_counts 类统计（纯图操作）──
        for mt, G in items:
            total += 1
            has_co = has_ether = has_ester = has_acid = False
            for o_node in G.nodes():
                if G.nodes[o_node].get('label') != 'O':
                    continue
                has_ether = True
                for nb in G.neighbors(o_node):
                    if (G.nodes[nb].get('label') == 'C'
                            and G.nodes[nb].get('oxo_counts', (0, 0))[1] >= 1):
                        has_ester = True
            for n in G.nodes():
                oh, co = G.nodes[n].get('oxo_counts', (0, 0))
                if co >= 1:
                    has_co = True
                if oh >= 1 and co >= 1:
                    has_acid = True
            n_co += has_co
            n_ether += has_ether
            n_ester += has_ester
            n_acid += has_acid
        # ── 抽样：RDKit 子结构统计（每桶 ≤ _COV_SAMPLE）──
        pool = items if len(items) <= _COV_SAMPLE else rng.sample(items, _COV_SAMPLE)
        for mt, G in pool:
            m = graph_to_rdkit_mol(G)
            if m is None:
                continue
            if m.HasSubstructMatch(p_epox):
                n_epoxide += 1
                if m.HasSubstructMatch(p_co):
                    n_epoxide_co += 1
            if m.HasSubstructMatch(p_oo):
                n_oo += 1
            if m.HasSubstructMatch(p_gem):
                n_gem_diol += 1
    return {'total': total, 'n_co': n_co, 'n_ether': n_ether,
            'n_ester': n_ester, 'n_acid': n_acid,
            'n_epoxide': n_epoxide, 'n_epoxide_co': n_epoxide_co,
            'n_oo': n_oo, 'n_gem_diol': n_gem_diol}


def canonical_smiles_set(buckets):
    """桶集合 → RDKit 规范 SMILES 集合（探针用，仅用于小公式）"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from utils import graph_to_rdkit_mol
    out = set()
    for items in buckets.values():
        for mt, G in items:
            m = graph_to_rdkit_mol(G)
            if m is not None:
                out.add(Chem.MolToSmiles(m))
    return out


# ── 重量级公式电池（完整档并行执行）──
# job 结构: (formula, tag, allow_empty, 覆盖断言规格[(label, stat_key, min_val), ...] 或 None,
#            是否附加 GUI 门禁模拟)
# 顺序按生成耗时降序（长任务优先入池，减少末波尾效应）
_HEAVY_JOBS = [
    ("C7H6O2", "", False, [
        ("覆盖: C7H6O2 含酯键", 'n_ester', 1),
        ("覆盖: C7H6O2 含羧基", 'n_acid', 1),
        ("覆盖: C7H6O2 含环氧", 'n_epoxide', 1),
        ("覆盖: C7H6O2 含环氧+羰基组合", 'n_epoxide_co', 1),
        ("覆盖: C7H6O2 含 O-O（过氧/氢过氧/过氧环）", 'n_oo', 1),
        ("覆盖: C7H6O2 含偕二醇（数学口径）", 'n_gem_diol', 1)], False),
    ("C7H8O2", "", False, None, False),
    ("C6H8O", "", False, None, True),   # 附加 GUI 门禁模拟（复用本次生成）
    ("C7H6O", "", False, [
        ("覆盖: C7H6O 含 C=O（苯甲醛类）", 'n_co', 1)], False),
    ("C6H10O", "", False, [
        ("覆盖: C6H10O 含环氧（环己烯氧化物类）", 'n_epoxide', 1)], False),
    ("C7H8O", "", False, [
        ("覆盖: C7H8O 含骨架O（苯甲醚类）", 'n_ether', 1),
        ("覆盖: C7H8O 含 C=O（酮类，k=3 骨架+ =O 提升）", 'n_co', 1)], False),
    ("C5H6O", "", False, None, False),
]


def _job_worker(job):
    """子进程执行单个重量级公式电池 + 覆盖断言（返回结果列表，不打印断言）"""
    formula, tag, allow_empty, spec, do_sim = job
    results = []
    buckets = test_battery(formula, None, tag, allow_empty, results=results)
    if spec:
        cs = coverage_stats(buckets)
        for label, key, minv in spec:
            results.append((label, cs.get(key, 0) >= minv, f"stats={cs}"))
    if do_sim:
        test_gui_gate_simulation(formula, results=results)
    return formula, results


def main():
    smoke = "--smoke" in sys.argv
    fast = smoke or "--fast" in sys.argv
    mode = "完整" if not fast else ("smoke" if smoke else "fast")
    print("=" * 70)
    print(f"分子式一致性门禁回归测试（{mode}）")
    print("=" * 70)

    # 1) 小分子电池（smoke 必跑，秒级）
    b = test_battery("C3H6O")
    cs = coverage_stats(b)
    check("覆盖: C3H6O 含环氧（环氧丙烷类）", cs['n_epoxide'] >= 1,
          f"stats={cs}")
    check("覆盖: C3H6O 含环氧丙烷(CC1CO1)", 'CC1CO1' in canonical_smiles_set(b))
    b = test_battery("C2H6O2")
    cs = coverage_stats(b)
    check("覆盖: C2H6O2 含 O-O（二甲过氧/乙基过氧化氢）", cs['n_oo'] >= 1,
          f"stats={cs}")
    check("覆盖: C2H6O2 含二甲过氧(COOC)", 'COOC' in canonical_smiles_set(b))
    b = test_battery("C2H4O2")
    cs = coverage_stats(b)
    check("覆盖: C2H4O2 含 O-O 或 4 元 C-O-C-O（过氧环/1,3-二氧杂环）",
          cs['n_oo'] >= 1, f"stats={cs}")
    test_battery("C2H6O")

    # 2) 负例单元断言 + 纯烃修复（秒级）
    test_negative_cases()
    test_pure_hydrocarbon_fix()

    if not smoke:
        # 3) 中小分子 + 约束用例（fast 起跑，分钟级）
        test_battery("C4H8O")
        test_battery("C6H6O")
        b = test_battery("C7H8O", StructureConstraint(required_fragment='benzene'),
                         " 约束=苯环")
        cs = coverage_stats(b)
        check("覆盖: C7H8O+苯环 含骨架O（苯甲醚）", cs['n_ether'] >= 1,
              f"stats={cs}")
        b = test_battery("C7H6O2", StructureConstraint(required_fragment='benzene'),
                         " 约束=苯环")
        cs = coverage_stats(b)
        check("覆盖: C7H6O2+苯环 含酯键（甲酸苯酯）", cs['n_ester'] >= 1,
              f"stats={cs}")
        check("覆盖: C7H6O2+苯环 含羧基（苯甲酸）", cs['n_acid'] >= 1,
              f"stats={cs}")
        b = test_battery("C8H8O2", StructureConstraint(required_fragment='benzene'),
                         " 约束=苯环")
        ss8 = canonical_smiles_set(b)
        check("覆盖: C8H8O2+苯环 含乙酸苯酯(ph-O-C(=O)-CH3)",
              'CC(=O)Oc1ccccc1' in ss8)
        check("覆盖: C8H8O2+苯环 含苯甲酸甲酯(ph-C(=O)-O-CH3)",
              'COC(=O)c1ccccc1' in ss8)
        buckets_furan = test_battery("C7H8O",
                                     StructureConstraint(required_fragment='furan'),
                                     " 约束=呋喃", allow_empty=True)
        check("C7H8O+呋喃: 旧泄漏 C7H10O 已消失",
              'C7H10O' not in buckets_furan)
        # 官能团分类模块（oxygen/funcgroup.py，GUI 精细筛选用）
        import test_funcgroup
        _fg_fail = test_funcgroup.run_all()
        check("funcgroup: 官能团分类全部断言通过", _fg_fail == 0,
              f"失败 {_fg_fail} 个")
        test_gui_gate_simulation("C4H8O")

    if not fast:
        # 4) 重量级公式电池（并行，墙钟大幅下降）
        print("\n== 重量级公式电池（并行） ==")
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=4) as ex:
            for formula, results in ex.map(_job_worker, _HEAVY_JOBS):
                for name, ok, detail in results:
                    check(name, ok, detail)

        # 4.5) 并行/串行等价（进程池并行与串行必须完全一致）
        print("\n== 并行/串行等价（进程池） ==")
        for _formula in ("C6H12O", "C6H10O"):
            res = parse_compound_formula(_formula)
            _, nc, nh, no, *_ = res
            _k = (2 * nc + 2 - nh) // 2
            b_s = UnifiedOxoGenerator().generate(nc, no, k_range=(_k, _k),
                                                 use_parallel=False)
            b_p = UnifiedOxoGenerator().generate(nc, no, k_range=(_k, _k),
                                                 use_parallel=True)
            n_s = sum(len(v) for v in b_s.values())
            n_p = sum(len(v) for v in b_p.values())
            check(f"并行等价: {_formula} 计数一致 ({n_s} == {n_p})",
                  n_s == n_p, f"串行 {n_s} vs 并行 {n_p}")
            check(f"并行等价: {_formula} 桶公式一致",
                  sorted(b_s.keys()) == sorted(b_p.keys()),
                  f"{sorted(b_s.keys())} vs {sorted(b_p.keys())}")

    print("\n" + "=" * 70)
    print(f"结果: PASS={PASS}  FAIL={FAIL}")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
