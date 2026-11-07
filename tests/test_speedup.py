# -*- coding: utf-8 -*-
"""提速特性回归测试（A 基团构建 / B 去重内核 / D 缓存 / F 代价预估）

覆盖：
  A. `special_groups` 基团增量 + `group_builder.derive_parent` 反推 +
     硝基苯可在**秒级**产出（且完整枚举给不出）
  B. 去重内核（矩阵级 WL + 精确同构）计数与内部基线逐位一致
  D. 结果缓存：写盘→命中→内容一致；损坏条目自动失效
  F. `estimate_cost` 能给出估计或明确"过慢"

运行：python tests/test_speedup.py
"""
import io
import os
import sys
import time

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


def test_group_deltas():
    print("\n== A1. 基团库与增量（数据驱动） ==")
    from special_groups import GROUP_LIBRARY, group_delta
    expect = {
        'nitro':    (0, -1, 1, 2, 0),
        'amino':    (0, 1, 1, 0, 0),
        'hydroxyl': (0, 0, 0, 1, 0),
        'cyano':    (1, -1, 1, 0, 0),
        'carboxyl': (1, 0, 0, 2, 0),
        'chloro':   (0, -1, 0, 0, 1),
    }
    for key, exp in expect.items():
        d = group_delta(key)
        check(f"{key}: 增量 = {exp}", d[:5] == exp, str(d[:5]))
    check("基团库 ≥ 11 个", len(GROUP_LIBRARY) >= 11, str(len(GROUP_LIBRARY)))


def test_derive_parent():
    print("\n== A2. 母体骨架反推 ==")
    from utils import parse_compound_formula
    from group_builder import derive_parent
    cases = [
        ('C6H5NO2', {'nitro': 1}, (6, 6)),
        ('C6H7N', {'amino': 1}, (6, 6)),
        ('C6H6O', {'hydroxyl': 1}, (6, 6)),
        ('C7H5N', {'cyano': 1}, (6, 6)),
        ('C7H6O2', {'carboxyl': 1}, (6, 6)),
        ('C6H5Cl', {'chloro': 1}, (6, 6)),
        ('C6H4N2O4', {'nitro': 2}, (6, 6)),
        ('C6H4ClN', {'amino': 1, 'chloro': 1}, (6, 4)),
    ]
    for fml, groups, exp in cases:
        r = parse_compound_formula(fml)
        _mt, nc, nh, no, halo, _nd, err, nn = r
        parent, e = derive_parent((nc, nh, nn, no, halo), groups)
        check(f"{fml} + {groups} → 母体 {exp}",
              e is None and parent == exp, f"{parent} / {e}")
    # 不匹配的目标应拒绝
    r = parse_compound_formula('C6H5NO2')
    parent, e = derive_parent((r[1], r[2], r[7], r[3], r[4]), {'amino': 1})
    check("C6H5NO2 + 仅氨基 → 拒绝（O 无法解释）", e is not None, str(e))


def test_nitrobenzene_fast():
    print("\n== A3. 硝基苯：秒级产出（完整枚举给不出） ==")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from utils import parse_compound_formula, graph_to_rdkit_mol, GeneratorManager
    from group_builder import build_from_groups, describe_mode

    gm = GeneratorManager()
    r = parse_compound_formula('C6H5NO2')
    _mt, nc, nh, no, halo, _nd, err, nn = r
    t0 = time.time()
    res = build_from_groups((nc, nh, nn, no, halo), {'nitro': 1},
                            lambda c, h: gm.generate_all(c, h))
    dt = time.time() - t0
    check("构建模式无错误", res.error is None, str(res.error))
    check(f"耗时 < 5s（实测 {dt:.2f}s）", dt < 5.0, f"{dt:.2f}s")
    check("母体骨架 = C6H6", res.parent_formula == 'C6H6', res.parent_formula)
    check("结果非空", len(res.graphs) > 0, str(len(res.graphs)))
    check("门禁 0 剔除", res.n_gate_rejected == 0, str(res.n_gate_rejected))

    want = Chem.MolToSmiles(Chem.MolFromSmiles('O=[N+]([O-])c1ccccc1'))
    found = False
    for g in res.graphs:
        m = graph_to_rdkit_mol(g)
        if m is not None and Chem.MolToSmiles(m) == want:
            found = True
            break
    check("✅ 硝基苯在结果中（O=[N+]([O-])c1ccccc1）", found)
    # 全部结果分子式 = C6H5NO2
    from utils import formula_matches_input
    bad = sum(1 for g in res.graphs
              if not formula_matches_input(g, nc, nh, no, halo, 0, n_nitrogen=nn))
    check("全部结果分子式 = C6H5NO2", bad == 0, str(bad))
    check("模式说明含『非完备』", '非完备' in describe_mode())


def test_dedup_kernel():
    print("\n== B. 去重内核（矩阵级 WL + 精确同构）计数一致性 ==")
    from Nitrogen.atomic_matrix_n import AtomicMatrixNGenerator
    from Nitrogen.validate import INTERNAL_BASELINE
    for fml, expect in [('C2H7N', 2), ('C3H9N', 4), ('C2H5N', 4),
                        ('C2H3N', 5), ('C5H5N', 685), ('C3H7N', 12)]:
        from Nitrogen.formula import parse_formula_full
        nc, nh, nn, no, err = parse_formula_full(fml)
        gs = AtomicMatrixNGenerator(nc, nn, no).generate_for_target_h(nh)
        check(f"{fml} = {expect}", len(gs) == expect, str(len(gs)))
    check("INTERNAL_BASELINE 含 C5H5N=685 锚点",
          INTERNAL_BASELINE.get('C5H5N') == 685)


def test_cache():
    print("\n== D. 结果缓存 ==")
    from Nitrogen import oracle_cache as oc
    from Nitrogen.atomic_matrix_n import AtomicMatrixNGenerator
    key_args = (4, 1, 0, 9)          # C4H9N
    oc.clear_cache()
    p = oc._entry_path(*key_args)
    if p.exists():
        p.unlink()
    check("初始未命中", oc.load(*key_args) is None)

    gen = AtomicMatrixNGenerator(4, 1, 0)
    graphs = gen.generate_for_target_h(9)
    check("写盘成功", oc.save(*key_args, graphs))
    got = oc.load(*key_args)
    check("命中且数量一致", got is not None and len(got) == len(graphs),
          f"{None if got is None else len(got)} vs {len(graphs)}")
    check("命中内容含 label/bond_type",
          got is not None and all('label' in d
                                  for _, d in got[0].nodes(data=True)))
    # 端到端：generate_cached 第二次应命中
    t0 = time.time(); _g1, i1 = oc.generate_cached(3, 1, 0, 9); t1 = time.time() - t0
    t0 = time.time(); _g2, i2 = oc.generate_cached(3, 1, 0, 9); t2 = time.time() - t0
    check("generate_cached 第二次命中", i2['hit'] is True, str(i2))
    # 小公式两次都极快（近似相等），只要求"命中不更慢"且数量一致
    check("命中不慢于首次且数量一致",
          t2 <= t1 + 0.05 and i1['n'] == i2['n'],
          f"{t1:.3f}s → {t2:.3f}s")
    # 版本失效：篡改版本号后应未命中
    from Nitrogen import oracle_cache as oc2
    old_ver = oc2.CACHE_VERSION
    try:
        oc2.CACHE_VERSION = 'unit-test-bump'
        check("引擎版本变更 → 旧缓存不命中", oc2.load(*key_args) is None)
    finally:
        oc2.CACHE_VERSION = old_ver
    # 损坏条目自动失效
    with open(p, 'wb') as f:
        f.write(b'not-a-pickle')
    check("损坏条目 → 未命中并删除",
          oc.load(*key_args) is None and not p.exists())
    oc.clear_cache()


def test_estimate():
    print("\n== F. 代价预估 ==")
    from Nitrogen.atomic_matrix_n import estimate_cost
    e = estimate_cost(4, 1, 0, 11, t_cap=2.0)          # C4H11N（很快）
    check("小公式可估计（estimable）", e['estimable'] is True, str(e))
    check("小公式预计耗时 < 60s", e['est_seconds_parallel'] < 60,
          str(e['est_seconds_parallel']))
    e2 = estimate_cost(6, 1, 2, 5, t_cap=1.5)          # C6H5NO2（极大）
    check("超大公式标记过慢或给出巨大估计",
          e2['too_slow'] or e2['est_seconds_parallel'] > 300,
          str({k: e2[k] for k in ('too_slow', 'est_seconds_parallel')}))


def main():
    print("=" * 70)
    print("提速特性回归测试（A 基团构建 / B 去重内核 / D 缓存 / F 预估）")
    print("=" * 70)
    test_group_deltas()
    test_derive_parent()
    test_nitrobenzene_fast()
    test_dedup_kernel()
    test_cache()
    test_estimate()
    print("\n" + "=" * 70)
    print(f"结果: PASS={PASS}  FAIL={FAIL}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
