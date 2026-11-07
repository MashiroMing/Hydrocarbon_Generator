"""
Nitrogen/validate.py — N 系验证入口（CLI）

═══ 验证判据 100% 内部化（合规红线 §8.4-3）═══
判定基准只来自三类**内部/独立**证据，不含对 MAYGEN（或任何外部工具）
接口与结果的任何依赖：

  ① 内部预言机自身（atomic_matrix_n.py 的原子矩阵目标 H 剪枝枚举）
  ② 内部预言机实测基线 INTERNAL_BASELINE（回归锁，防计数漂移）
  ③ 化学教科书事实锚点 CHEM_FACTS（可由化学推理直接得出，独立于任何程序）

MAYGEN 对照已剥离为**可选**脚本 Nitrogen/maygen_compare.py：
需显式调用、不参与本回归、不参与 pass/fail 判定。

═══ 六项检查 ═══
  1. 门禁冒烟        产物分子式 ≡ 输入分子式
  2. 内部基线锚点    预言机计数 vs INTERNAL_BASELINE（回归锁）
  3. 化学事实锚点    预言机计数 vs CHEM_FACTS（独立化学证据）
  4. 预言机自洽      门禁恒真 + 无同构重复 + 价态合法（内部正确性证据）
  5. 双实现交叉完备  树机制 ⊆ 预言机（覆盖率报告）
  6. 全 H 档位守卫   n_hydrogen=None 时机制 B/C 覆盖全部 H 档位（防缺陷复发）

═══ 关键实测事实（2026-09-19 测定，供后续开发参考）═══
  · **机制 A（C→N 替换）单独即完备**：8/8 公式上 A 的产物集合与预言机
    完全等值（含 C5H5N=685）。理论上也可由"碳化映射"证明——任一 C/H/N 图
    把 N 改标为 C 后即得合法全碳骨架，故 A ⊇ 全部 C/H/N 图；反向由公式门禁
    保证 A ⊆ 合法图。
  · **机制 A 比预言机慢**（实测 0.56~0.76×）：A 在 (nC+nN) 个碳上做**全不饱和度**
    枚举，未施加预言机的目标 H 剪枝。故"用树机制规避原子矩阵组合爆炸"的
    假设**不成立**——生成任务应直接用预言机，`--tree` 仅用于快速查看常见结构。
  · **机制 B/C 是部分路径**：仅覆盖端基/桥类常见官能团（不含含 N 杂环），
    且其 k_target 依赖 n_hydrogen，必须逐 H 档位调用（见检查 6）。

用法：
  python -m Nitrogen.validate                     # 默认：全套内部验证
  python -m Nitrogen.validate C5H5N               # 指定分子式，打印结构样例
  python -m Nitrogen.validate -i                  # 交互模式（逐条输入分子式）
  python -m Nitrogen.validate --anchors           # 只跑锚点电池 + 覆盖率
"""

import os
import sys
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ── 兼容直跑：python Nitrogen/validate.py（无父包上下文）──
# 推荐用法仍是 `python -m Nitrogen.validate`；直跑时把项目根加入 sys.path 并转绝对导入。
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Nitrogen.formula import (parse_formula, parse_formula_full,
                                  graph_formula_parts,
                                  format_graph_formula, dbe_from_parts)
    from Nitrogen.atomic_matrix_n import AtomicMatrixNGenerator
    from Nitrogen.generator import NitrogenGenerator
else:
    from .formula import (parse_formula, parse_formula_full, graph_formula_parts,
                          format_graph_formula, dbe_from_parts)
    from .atomic_matrix_n import AtomicMatrixNGenerator
    from .generator import NitrogenGenerator

import networkx as nx

# ══════════════════════════════════════════════════════════════════
# 锚点表
# ══════════════════════════════════════════════════════════════════

# ── 内部基线（回归锁）──────────────────────────────────────────────
# 来源：内部预言机 atomic_matrix_n.py 实测（2026-08-26 首测、2026-09-19 复核，
#       两次结果一致）。口径：图口径（凯库勒式变体分开计数，与 O 系一致）。
# 作用：任何生成/枚举逻辑改动一旦造成计数漂移，此处立即报警。
# 合规：判定过程只读本表与内部预言机，**不读取任何外部工具输出**。
INTERNAL_BASELINE: Dict[str, int] = {
    'C2H7N': 2, 'C3H9N': 4, 'C2H5N': 4, 'C2H3N': 5, 'C5H5N': 685,
    'C2H8N2': 6, 'C3H7N': 12, 'C4H11N': 8, 'C6H5N': 4394, 'C6H7N': 4378,
    'C6H15N': 39, 'C4H9N': 35,
}

# ── 化学事实锚点（教科书可查，独立于任何软件实现）────────────────
# 这些数目可直接由化学推理得出（有机化学教科书的同分异构体计数题），
# 因此为内部预言机提供了一条**外部独立性**证据——不依赖任何枚举程序的输出。
CHEM_FACTS: Dict[str, Tuple[int, str]] = {
    'C2H7N': (2, '乙胺 / 二甲胺'),
    'C3H9N': (4, '正丙胺 / 异丙胺 / 甲乙胺 / 三甲胺'),
    'C4H11N': (8, '四种丁胺（正/仲/异/叔）+ N-甲基丙胺（正/异）'
                  ' + N-乙基乙胺 + N,N-二甲基乙胺'),
}


# ══════════════════════════════════════════════════════════════════
# 预言机（带缓存；同一 (nC,nN,targetH) 只枚举一次，被后续多项检查复用）
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _oracle_cached(n_c: int, n_n: int, target_h: int, n_o: int = 0) -> tuple:
    """内部预言机枚举（无超时路径的缓存层）"""
    gen = AtomicMatrixNGenerator(n_c, n_n, n_o)
    if not gen._can_enumerate():
        return ()
    return tuple(gen.generate_for_target_h(target_h))


def oracle_count(n_c: int, n_n: int, target_h: int,
                 deadline: Optional[float] = None, n_o: int = 0) -> List:
    """预言机生成指定分子式的全部结构

    Args:
        deadline: 可选超时（time.time() 绝对时刻）。超时则返回 [] 并由
                  调用方走降级路径。带 deadline 的调用不走缓存。
        n_o: 氧原子数（**末位默认 0** → 既有调用逐位不变；路线 A 起支持 C/N/O）。
    """
    if n_o == 0 and deadline is None:
        return list(_oracle_cached(n_c, n_n, target_h))
    gen = AtomicMatrixNGenerator(n_c, n_n, n_o)
    if not gen._can_enumerate():
        return []
    try:
        return gen.generate_for_target_h(target_h, deadline=deadline)
    except TimeoutError:
        return []


def tree_generate(n_c: int, n_n: int, target_h: int,
                  **kwargs) -> List:
    """树机制生成器产出（平铺图列表）

    kwargs 透传 NitrogenGenerator 构造参数（enable_replace / enable_terminal /
    enable_bridge），用于消融实验与部分路径检查。
    """
    gen = NitrogenGenerator(**kwargs)
    res = gen.generate(n_c, n_n, n_hydrogen=target_h)
    out = []
    for fml, items in res.items():
        for mt, g in items:
            out.append(g)
    return out


def _isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    nm = nx.algorithms.isomorphism.categorical_node_match('label', 'C')
    em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
    return nx.is_isomorphic(g1, g2, node_match=nm, edge_match=em)


# ══════════════════════════════════════════════════════════════════
# 检查 1：门禁冒烟
# ══════════════════════════════════════════════════════════════════

def smoke_gate(formulas: Optional[List[str]] = None) -> int:
    """分子式门禁冒烟：预言机全部产物必须通过 graph_formula_parts 核对"""
    if formulas is None:
        formulas = ['C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C2H8N2']
    n_fail = 0
    for fml in formulas:
        n_c, n_h, n_n, err = parse_formula(fml)
        if err:
            continue
        for g in oracle_count(n_c, n_n, n_h):
            parts = graph_formula_parts(g)
            if (parts[0], parts[1], parts[2]) != (n_c, n_h, n_n):
                print(f"门禁失败 {fml}: {parts}")
                n_fail += 1
    return n_fail


# ══════════════════════════════════════════════════════════════════
# 检查 2：内部基线锚点（回归锁）
# ══════════════════════════════════════════════════════════════════

def check_internal_baseline(formulas: Optional[List[str]] = None) -> int:
    """内部基线锚点电池：预言机计数 vs INTERNAL_BASELINE

    Returns:
        失败数（解析失败 / 计数漂移）
    """
    if formulas is None:
        formulas = sorted(INTERNAL_BASELINE)
    n_fail = 0
    print(f"{'分子式':<8} {'预言机':>7} {'内部基线':>8} {'一致':>4}  {'DBE':>4}  耗时")
    print("-" * 62)
    for fml in formulas:
        n_c, n_h, n_n, err = parse_formula(fml)
        if err:
            print(f"{fml:<8} 解析失败: {err}")
            n_fail += 1
            continue
        t0 = time.time()
        graphs = oracle_count(n_c, n_n, n_h)
        dt = time.time() - t0
        n_oracle = len(graphs)
        base = INTERNAL_BASELINE.get(fml)
        ok = (base is None) or (n_oracle == base)
        if not ok:
            n_fail += 1
        mark = "OK " if ok else "FAIL"
        dbe = dbe_from_parts(n_c, n_h, n_n)
        print(f"{fml:<8} {n_oracle:>7} "
              f"{(str(base) if base is not None else '-'):>8} "
              f"{mark:>4} {dbe:>4.0f}  {dt:6.1f}s")
    print("-" * 62)
    return n_fail


# ── 内部基线（回归锁）：路线 A —— 含氮含氧 C/H/N/O ─────────────────
# 来源：内部预言机 atomic_matrix_n.py（C=4/N=3/O=2）实测 2026-09-21；
#       每个值另经 **一次性 MAYGEN 外部对拍** 验证（计数与 SMILES 集合 0 缺 0 多，
#       见开发过程.md §十五）——外部对拍只用于**首次确认**，回归只读本表。
# 口径提醒：中性八隅律 N=3 ⇒ **硝基 -NO₂ 不在空间内**（MAYGEN 默认价态同样不含）。
INTERNAL_BASELINE_NO: Dict[str, int] = {
    # ≤5 重原子（默认电池，秒级）
    'CH3NO': 5, 'CH5NO': 3, 'CH3NO2': 15, 'CH4N2O': 21, 'C2H5NO': 22,
    'C2H3NO': 26, 'C2H5NO2': 84, 'C2H4N2O': 177, 'C3H7NO': 84, 'C3H5NO': 154,
    'C3H3NO': 136, 'C2H2N2O': 114,
    # ≥6 重原子（较慢，仅在 --full 档跑）
    'C4H9NO': 299, 'C4H7NO': 764, 'C5H5NO': 7687,
}

# 默认电池：只跑 ≤5 重原子的那些（快）
_N_O_FAST = ['CH3NO', 'CH5NO', 'CH3NO2', 'CH4N2O', 'C2H5NO', 'C2H3NO',
             'C2H5NO2', 'C2H4N2O', 'C3H7NO', 'C3H5NO', 'C3H3NO', 'C2H2N2O']


def check_n_plus_o_baseline(formulas: Optional[List[str]] = None) -> int:
    """路线 A 基线电池：预言机计数（C/H/N/O）vs INTERNAL_BASELINE_NO + 公式门禁

    Returns:
        失败数（解析失败 / 计数漂移 / 公式不符）
    """
    from .formula import parse_formula_full, formula_matches_input
    if formulas is None:
        formulas = _N_O_FAST
    n_fail = 0
    print(f"{'分子式':<10} {'预言机':>7} {'基线':>7} {'一致':>4}  {'DBE':>4}  耗时")
    print("-" * 62)
    for fml in formulas:
        n_c, n_h, n_n, n_o, err = parse_formula_full(fml)
        if err:
            print(f"{fml:<10} 解析失败: {err}")
            n_fail += 1
            continue
        t0 = time.time()
        graphs = oracle_count(n_c, n_n, n_h, n_o=n_o)
        dt = time.time() - t0
        n_oracle = len(graphs)
        base = INTERNAL_BASELINE_NO.get(fml)
        bad_gate = sum(1 for g in graphs
                       if not formula_matches_input(g, n_c, n_h, n_n, n_o))
        ok = ((base is None or n_oracle == base) and bad_gate == 0)
        if not ok:
            n_fail += 1
        mark = "OK " if ok else "FAIL"
        dbe = dbe_from_parts(n_c, n_h, n_n)
        print(f"{fml:<10} {n_oracle:>7} "
              f"{(str(base) if base is not None else '-'):>7} "
              f"{mark:>4} {dbe:>4.0f}  {dt:6.1f}s"
              + (f"  [门禁不符 {bad_gate}]" if bad_gate else ""))
    print("-" * 62)
    return n_fail


def check_parallel_equiv(formulas: Optional[List[str]] = None) -> int:
    """并行预言机等价性回归：``generate_for_target_h_parallel`` == 串行结果

    锁定不变量：前缀切分 + 父进程 WL/VF2 跨子树去重后，唯一图集合与串行
    **逐元素一致**（只差顺序）。选小公式保证秒级完成。
    """
    import networkx as nx
    if formulas is None:
        formulas = ['C3H7N', 'C4H9N', 'C2H5NO', 'CH4N2O', 'C5H7N']
    n_fail = 0
    print(f"{'分子式':<9} {'串行':>6} {'并行':>6} {'一致':>4}  加速")
    print("-" * 46)
    for fml in formulas:
        n_c, n_h, n_n, n_o, err = parse_formula_full(fml)
        if err:
            print(f"{fml:<9} 解析失败: {err}")
            n_fail += 1
            continue
        gen = AtomicMatrixNGenerator(n_c, n_n, n_o)
        t0 = time.time()
        ser = gen.generate_for_target_h(n_h)
        t_ser = time.time() - t0
        t0 = time.time()
        par = gen.generate_for_target_h_parallel(n_h)
        t_par = time.time() - t0
        key = lambda g: nx.weisfeiler_lehman_graph_hash(       # noqa: E731
            g, node_attr='label', edge_attr='bond_type')
        ok = (len(ser) == len(par) and {key(g) for g in ser} == {key(g) for g in par})
        if not ok:
            n_fail += 1
        print(f"{fml:<9} {len(ser):>6} {len(par):>6} {'OK ' if ok else 'FAIL':>4}  "
              f"{(t_ser / t_par) if t_par else 0:4.2f}x")
    print("-" * 46)
    return n_fail


# ══════════════════════════════════════════════════════════════════
# 检查 3：化学事实锚点（独立化学证据）
# ══════════════════════════════════════════════════════════════════

def check_chem_facts(formulas: Optional[List[str]] = None) -> int:
    """化学事实锚点核对：预言机计数 vs 教科书可查结果

    这是验证体系中唯一**不依赖任何程序输出**的证据来源，用于交叉印证
    内部预言机的正确性（数目可由化学推理独立得出）。

    Returns:
        失败数
    """
    if formulas is None:
        formulas = sorted(CHEM_FACTS)
    n_fail = 0
    print(f"{'分子式':<8} {'预言机':>7} {'化学事实':>8} {'一致':>4}  来源")
    print("-" * 62)
    for fml in formulas:
        fact = CHEM_FACTS.get(fml)
        if fact is None:
            continue
        expected, note = fact
        n_c, n_h, n_n, err = parse_formula(fml)
        if err:
            print(f"{fml:<8} 解析失败: {err}")
            n_fail += 1
            continue
        n_oracle = len(oracle_count(n_c, n_n, n_h))
        ok = (n_oracle == expected)
        if not ok:
            n_fail += 1
        print(f"{fml:<8} {n_oracle:>7} {expected:>8} "
              f"{('OK ' if ok else 'FAIL'):>4}  {note}")
    print("-" * 62)
    return n_fail


def check_anchors(formulas: Optional[List[str]] = None) -> int:
    """锚点电池总入口 = 内部基线（回归锁）+ 化学事实锚点（独立证据）"""
    return (check_internal_baseline(formulas) +
            check_chem_facts(formulas))


# ══════════════════════════════════════════════════════════════════
# 检查 4：预言机自洽（内部正确性证据，无需任何外部工具）
# ══════════════════════════════════════════════════════════════════

_VALENCE = {'C': 4, 'N': 3}
_BOND_LOAD = {'single': 1, 'double': 2, 'triple': 3}


def check_oracle_selfconsistent(formulas: Optional[List[str]] = None) -> int:
    """预言机自洽三项断言

      ① 门禁恒真：每个产物的 (nC,nH,nN) 等于输入
      ② 无同构重复：产物集合内两两不同构（验证去重完备——碰撞安全去重的
         反向保证：既不能漏删重复，也不能误删真实结构）
      ③ 价态合法：每个原子 bond_load ≤ 其最大价（八隅律口径）

    Returns:
        失败数
    """
    if formulas is None:
        formulas = ['C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C5H5N', 'C2H8N2']
    n_fail = 0
    print(f"{'分子式':<8} {'结构数':>7} {'门禁':>5} {'无重复':>7} {'价态':>5}  桶数")
    print("-" * 58)
    for fml in formulas:
        n_c, n_h, n_n, err = parse_formula(fml)
        if err:
            print(f"{fml:<8} 解析失败: {err}")
            n_fail += 1
            continue
        graphs = oracle_count(n_c, n_n, n_h)

        # ① 门禁恒真
        bad_gate = sum(1 for g in graphs
                       if graph_formula_parts(g)[:3] != (n_c, n_h, n_n))

        # ② 无同构重复（WL 分桶 + 桶内成对 VF2，避免 O(n²) 全比）
        buckets: Dict[str, List] = {}
        for g in graphs:
            wl = nx.weisfeiler_lehman_graph_hash(
                g, node_attr='label', edge_attr='bond_type')
            buckets.setdefault(wl, []).append(g)
        dup = 0
        for group in buckets.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if _isomorphic(group[i], group[j]):
                        dup += 1
                        break

        # ③ 价态合法
        bad_val = 0
        for g in graphs:
            for nd in g.nodes():
                load = sum(_BOND_LOAD.get(g[nd][nb].get('bond_type', 'single'), 1)
                           for nb in g.neighbors(nd))
                if load > _VALENCE.get(g.nodes[nd].get('label', 'C'), 4):
                    bad_val += 1
                    break

        ok = (bad_gate == 0 and dup == 0 and bad_val == 0)
        if not ok:
            n_fail += 1
        print(f"{fml:<8} {len(graphs):>7} {bad_gate:>5} {dup:>7} {bad_val:>5}  "
              f"{len(buckets)}{'' if ok else '  <== FAIL'}")
    print("-" * 58)
    return n_fail


# ══════════════════════════════════════════════════════════════════
# 检查 5：双实现交叉完备性（树机制 ⊆ 预言机）
# ══════════════════════════════════════════════════════════════════

def coverage_report(n_c: int, n_n: int, target_h: int,
                    verbose: bool = False, **tree_kwargs) -> Dict:
    """树机制 vs 预言机 双向差集检查（对齐 O 系 boundary report 的"我方多余"口径）

    双向都查，缺一不可：
      · missing = 预言机 − 树机制  → 树机制的**覆盖缺口**（不完备）
      · extra   = 树机制 − 预言机  → 树机制的**非法泄漏**（必须恒为 0；
        非 0 说明树机制产出了预言机不认的结构，通常是价态/公式门禁漏网）

    Returns:
        {'oracle': n, 'tree': n, 'covered': n, 'missing': n, 'missing_fmls': [...],
         'extra': n, 'extra_fmls': [...]}
    """
    oracle = oracle_count(n_c, n_n, target_h)
    tree = tree_generate(n_c, n_n, target_h, **tree_kwargs)
    covered = 0
    missing = []
    for g in oracle:
        if any(_isomorphic(g, t) for t in tree):
            covered += 1
        else:
            missing.append(format_graph_formula(g))
    extra = []
    for t in tree:
        if not any(_isomorphic(t, o) for o in oracle):
            extra.append(format_graph_formula(t))
    return {
        'oracle': len(oracle), 'tree': len(tree), 'covered': covered,
        'missing': len(missing), 'missing_fmls': missing,
        'extra': len(extra), 'extra_fmls': extra,
    }


def check_coverage(formulas: Optional[List[str]] = None,
                   verbose: bool = False) -> int:
    """树机制 vs 预言机 双向差集检查；返回失败数

    判据：missing（覆盖缺口）与 extra（非法泄漏）**都必须为 0**。
    extra > 0 即树机制产出了预言机不认的结构 —— 历史实证：机制 A 对键载
    为 4 的碳做 C→N 替换会产生 N 超价结构，且被纯公式门禁放行。
    """
    if formulas is None:
        formulas = ['C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C2H8N2',
                    'C3H7N', 'C4H9N', 'C5H5N', 'C2H6N2', 'C2H4N2']
    n_fail = 0
    print(f"{'分子式':<8} {'预言机':>7} {'树机制':>7} {'覆盖':>6} "
          f"{'缺失':>5} {'多余':>5}  覆盖率")
    print("-" * 74)
    for fml in formulas:
        n_c, n_h, n_n, err = parse_formula(fml)
        if err:
            print(f"{fml:<8} 解析失败: {err}")
            n_fail += 1
            continue
        if n_c + n_n > 7:
            print(f"{fml:<8} 重原子 {n_c+n_n} > 7，跳过（枚举过慢）")
            continue
        t0 = time.time()
        rep = coverage_report(n_c, n_n, n_h, verbose=verbose)
        dt = time.time() - t0
        ratio = rep['covered'] / rep['oracle'] if rep['oracle'] else 1.0
        ok = (rep['missing'] == 0 and rep['extra'] == 0)
        if not ok:
            n_fail += 1
        mark = "OK " if ok else ("GAP" if rep['missing'] else "LEAK")
        print(f"{fml:<8} {rep['oracle']:>7} {rep['tree']:>7} "
              f"{rep['covered']:>6} {rep['missing']:>5} {rep['extra']:>5}  "
              f"{ratio*100:5.1f}%  {mark}  {dt:5.1f}s")
        if verbose and rep['missing'] > 0 and rep['missing'] <= 10:
            for m in rep['missing_fmls'][:10]:
                print(f"        缺失: {m}")
        if rep['extra'] > 0:
            for m in rep['extra_fmls'][:10]:
                print(f"        多余(非法泄漏): {m}")
    print("-" * 74)
    return n_fail


# ══════════════════════════════════════════════════════════════════
# 检查 6：全 H 档位守卫（防历史缺陷复发）
# ══════════════════════════════════════════════════════════════════

def check_all_h_levels(pairs: Optional[List[Tuple[int, int]]] = None) -> int:
    """全 H 档位守卫：generate(nC, nN)（**不给** n_hydrogen）必须覆盖每一档

    背景：机制 B/C 的 delta = k_target − k_skel 依赖 n_hydrogen，历史实现
    只传 min(target_h_set)（最高不饱和度那一档），导致不带 n_hydrogen 时
    B/C 只产出极少数结构；默认配置下被机制 A 的完备性掩盖，
    仅 enable_replace=False 时暴露。本检查为该缺陷的**回归守卫**。

    断言（机制 A 启用时机制整体应为完备）：
      ① 产出分子式集合 == 预言机在该 (nC,nN) 下的全部合法 H 档位集合
      ② 每一档的计数 == 预言机同档计数

    Returns:
        失败数
    """
    if pairs is None:
        pairs = [(2, 1), (3, 1), (2, 2)]
    n_fail = 0
    print(f"{'(nC,nN)':<10} {'H 档位':>7} {'树机制':>7} {'预言机':>7} {'一致':>4}")
    print("-" * 46)
    for n_c, n_n in pairs:
        # 合法 H 档位：nH = 2nC + nN + 2 − 2k, k ≥ 0 且 nH ≥ 1（含 N 需 H）
        k_max = (2 * n_c + n_n + 2) // 2
        h_levels = sorted({2 * n_c + n_n + 2 - 2 * k for k in range(k_max + 1)
                           if 2 * n_c + n_n + 2 - 2 * k >= 1}, reverse=True)
        gen = NitrogenGenerator()
        res = gen.generate(n_c, n_n)          # 关键：不给 n_hydrogen
        tree_total = sum(len(v) for v in res.values())
        level_fail = 0
        for h in h_levels:
            tree_by_h = sum(len(v) for fml, v in res.items()
                            if parse_formula(fml)[1] == h)
            oracle_by_h = len(oracle_count(n_c, n_n, h))
            ok = (tree_by_h == oracle_by_h)
            if not ok:
                level_fail += 1
            print(f"{f'({n_c},{n_n})':<10} {f'H={h}':>7} {tree_by_h:>7} "
                  f"{oracle_by_h:>7} {('OK ' if ok else 'FAIL'):>4}")
        if level_fail:
            n_fail += level_fail
            print(f"  ^^ ({n_c},{n_n}) 存在 {level_fail} 个档位不一致；"
                  f"树机制总产出 {tree_total}")
    print("-" * 46)
    return n_fail


# ══════════════════════════════════════════════════════════════════
# 结构样例打印
# ══════════════════════════════════════════════════════════════════

def sample_structures(formula: str, limit: int = 12,
                      graphs: Optional[List] = None,
                      source: str = '预言机') -> None:
    """打印指定分子式的结构样例（图边列表 + 分子式）

    Args:
        graphs: 已枚举的图列表；None 时内部调用预言机（小分子式）。
        source: 来源标注，如 '预言机' / '树机制(部分，非完备)'。
    """
    n_c, n_h, n_n, n_o, err = parse_formula_full(formula)
    if err:
        print(err)
        return
    if graphs is None:
        graphs = oracle_count(n_c, n_n, n_h, n_o=n_o)
    print(f"\n{formula}: 共 {len(graphs)} 个结构（{source}）。样例：")
    for i, g in enumerate(graphs[:limit]):
        edges = []
        for u, v, d in g.edges(data=True):
            edges.append(f"{g.nodes[u]['label']}{u}-"
                         f"{d.get('bond_type','single')[:1]}-"
                         f"{g.nodes[v]['label']}{v}")
        h_total = graph_formula_parts(g)[1]
        print(f"  [{i}] H={h_total}  {'  '.join(edges)}")
    if len(graphs) > limit:
        print(f"  ... 其余 {len(graphs) - limit} 个略")


_USAGE = """\
用法：
  python -m Nitrogen.validate [分子式...] [选项]      # 批量计算（命令行分子式）
  python -m Nitrogen.validate                        # 交互模式：逐条输入分子式再计算
  python -m Nitrogen.validate -i                     # 强制进入交互模式
  python -m Nitrogen.validate --anchors              # 只跑锚点电池 + 完备性

选项：
  分子式...      待计算的分子式（**C/H/N/O**），如 C2H7N C5H5N C2H5NO CH4N2O
  -i, --interactive   进入交互模式（逐条输入分子式）
  --anchors           运行锚点电池（内部基线 N + N+O 基线 + 化学事实 + 覆盖率）
  -v, --verbose       打印覆盖率缺失结构明细
  --tree              强制树机制（部分路径，非完备；**不比预言机快**）
  --oracle            强制预言机完备枚举（重原子>7 带 30s 超时兜底）
  -h, --help          显示本帮助

验证判据（100% 内部化，合规红线 §8.4-3）：
  内部预言机 · 内部实测基线（回归锁）· 化学教科书事实锚点
MAYGEN 对照为**可选**，见 Nitrogen/maygen_compare.py（不参与本回归）。

模式说明：
  重原子 ≤ 7 ：预言机完备枚举（推荐，最快且完备）+ 覆盖率校验；
  重原子 > 7 ：默认仍走预言机（带超时兜底）。注意 **--tree 不是加速档**：
              实测机制 A 比预言机慢 0.56~0.76×（见模块 docstring）。
"""

# 重原子上限：超过此值时预言机带超时兜底
# ⚠ 历史误述更正（2026-09-19 实测）：本档位原先标注为"降级到快速树机制以避免
#   组合爆炸"，但实测机制 A 的产物 ≡ 预言机产物且**更慢**（0.56~0.76×），
#   因其在 (nC+nN) 个碳上做全不饱和度枚举、未做目标 H 剪枝。故 tree 模式
#   不构成加速手段，仅作为"只看常见官能团部分集合"的快速预览。
_ORACLE_ATOM_LIMIT = 7
_ORACLE_DEADLINE = 30.0   # 秒；auto/oracle 模式下预言机的超时兜底


def _run_single_formula(fml: str, verbose: bool = False,
                        mode: str = 'auto') -> int:
    """对单个分子式执行：样例 + 锚点 + 覆盖率

    mode:
      'auto'  : 重原子 ≤ _ORACLE_ATOM_LIMIT 用预言机；更大时用树机制预览
      'tree'  : 强制树机制（部分路径，非完备，**不快于预言机**）
      'oracle': 强制预言机（重原子超限时带 _ORACLE_DEADLINE 超时兜底）

    Returns:
        失败项计数（解析失败 / 预言机超时 / 覆盖率缺失）
    """
    n_c, n_h, n_n, n_o, err = parse_formula_full(fml)
    if err:
        print(f"{fml}: 解析失败 -> {err}")
        return 1
    n_atoms = n_c + n_n + n_o
    n_fail = 0

    if n_o == 0 and (mode == 'tree' or (mode == 'auto' and n_atoms > _ORACLE_ATOM_LIMIT)):
        # ── 树机制预览路径（部分集合，非完备） ──
        t0 = time.time()
        tree = tree_generate(n_c, n_n, n_h)
        dt = time.time() - t0
        note = f"树机制(部分，非完备，{dt:.1f}s)"
        print(f"  （{fml} 重原子 {n_atoms} 较大，采用 {note}；"
              f"输出为部分集合，如需完备结果请用 --oracle）")
        sample_structures(fml, graphs=tree, source=note)
        base = INTERNAL_BASELINE.get(fml)
        if base is not None:
            mark = "OK " if len(tree) == base else "GAP"
            print(f"  {fml:8s} 树机制 {len(tree):>6}  内部基线 {base:>6}  {mark}")
            if len(tree) != base:
                n_fail += 1
        return n_fail

    # ── 预言机完备路径（N ≤ _ORACLE_ATOM_LIMIT，或 --oracle 强制） ──
    deadline = time.time() + _ORACLE_DEADLINE if n_atoms > _ORACLE_ATOM_LIMIT else None
    graphs = oracle_count(n_c, n_n, n_h, deadline=deadline, n_o=n_o)
    if not graphs and n_atoms > _ORACLE_ATOM_LIMIT:
        print(f"  （{fml} 重原子 {n_atoms} 预言机枚举超过 {_ORACLE_DEADLINE:.0f}s，"
              f"已中断；建议放宽超时或改用 --tree 预览）")
        return 1
    sample_structures(fml, graphs=graphs, source='预言机')
    if n_o > 0:
        # 含氧含氮（路线 A）：走 N+O 基线电池（树机制不覆盖 O，故跳过覆盖率）
        n_fail += check_n_plus_o_baseline([fml])
    else:
        n_fail += check_anchors([fml])
        if n_atoms <= _ORACLE_ATOM_LIMIT:
            n_fail += check_coverage([fml], verbose)
        else:
            print(f"  （{fml} 重原子 {n_atoms} > {_ORACLE_ATOM_LIMIT}，跳过覆盖率枚举）")
    return n_fail


def _interactive_loop(verbose: bool = False, mode: str = 'auto') -> int:
    """交互模式：循环提示用户输入分子式，逐条计算，直到退出"""
    print()
    print("交互模式：输入 C/H/N 分子式后回车即计算（如 C2H7N / C5H5N）")
    print("  特殊命令：quit/exit/空行=退出；anchors=跑锚点电池；help=用法")
    if mode == 'tree':
        print("  当前模式：树机制（--tree，部分集合，非完备，不快于预言机）")
    elif mode == 'oracle':
        print("  当前模式：预言机完备（--oracle，重原子>7 带超时）")
    else:
        print(f"  当前模式：自动（重原子 ≤{_ORACLE_ATOM_LIMIT} 预言机，"
              f">{_ORACLE_ATOM_LIMIT} 树机制预览）")
    print("-" * 68)
    total_fail = 0
    while True:
        try:
            raw = input("分子式> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not raw:
            print("已退出。")
            break
        low = raw.lower()
        if low in ('quit', 'exit'):
            print("已退出。")
            break
        if low in ('help', '?'):
            print(_USAGE)
            continue
        if low == 'anchors':
            total_fail += check_anchors()
            continue
        total_fail += _run_single_formula(raw, verbose, mode=mode)
        print("-" * 68)
    return total_fail


def main(argv: Optional[List[str]] = None) -> int:
    # stdout UTF-8 兜底：避免 `✓`/`✗` 在 GBK 控制台触发 UnicodeEncodeError。
    # 放在 main() 内而非仅 __main__ 块，保证被 import 后调用同样安全。
    # （reconfigure 仅 Python 3.7+；pyright 认为 TextIO 无此属性，故用 getattr 规避）
    for _stream in (sys.stdout, sys.stderr):
        _reconf = getattr(_stream, 'reconfigure', None)
        if callable(_reconf):
            _reconf(encoding='utf-8', errors='replace')

    argv = argv if argv is not None else sys.argv[1:]
    verbose = '-v' in argv or '--verbose' in argv
    interactive = ('-i' in argv) or ('--interactive' in argv)
    anchors_only = '--anchors' in argv
    help_requested = '-h' in argv or '--help' in argv
    maygen_requested = '--maygen' in argv
    if '--tree' in argv:
        mode = 'tree'
    elif '--oracle' in argv:
        mode = 'oracle'
    else:
        mode = 'auto'
    positional = [a for a in argv if not a.startswith('-')]

    print("=" * 68)
    print("Nitrogen 验证入口：内部自证（锚点 / 自洽 / 完备性 / 全 H 档位）")
    print("=" * 68)
    print("判据 100% 内部化：内部预言机 · 内部实测基线 · 化学事实锚点")
    print("（MAYGEN 对照为可选，见 Nitrogen/maygen_compare.py，不参与本回归）")

    if maygen_requested:
        print()
        print("提示：--maygen 已剥离。对外部对照请运行：")
        print("      python -m Nitrogen.maygen_compare")
        print("      该脚本为可选一次性外部复核，不参与回归、不影响 pass/fail。")

    if help_requested:
        print(_USAGE)
        return 0

    n_fail = 0
    print()
    n_fail += smoke_gate()

    if anchors_only:
        print()
        n_fail += check_anchors()
        print()
        n_fail += check_n_plus_o_baseline()
        print()
        n_fail += check_parallel_equiv()
        print()
        n_fail += check_oracle_selfconsistent()
        print()
        n_fail += check_coverage(verbose=verbose)
        print()
        n_fail += check_all_h_levels()
    elif positional and not interactive:
        for fml in positional:
            n_fail += _run_single_formula(fml, verbose, mode=mode)
    else:
        n_fail += _interactive_loop(verbose, mode=mode)

    print()
    if n_fail == 0:
        print("全部验证通过 ✓")
    else:
        print(f"存在 {n_fail} 项失败 ✗")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
