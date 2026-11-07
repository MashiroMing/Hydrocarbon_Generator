"""Phosphorus 验证入口（第一期：仅 C/H/P）

三类检查：
    --anchors  与 MAYGEN 1.8 一次性对拍（P=3 口径，计数必须逐一相等）
    --props    内部性质：公式自洽 / P 价态=3 / 连通性 / 去重无重复
    --equiv    串行 vs 并行 结果集完全一致

用法：
    python -m Phosphorus.validate --anchors
    python -m Phosphorus.validate --quick
    python -m Phosphorus.validate --props
    python -m Phosphorus.validate --equiv
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .atomic_matrix_p import PhosphorusMatrixGenerator
from .formula import (
    bond_order,
    FormulaError,
    dbe,
    format_graph_formula_p,
    graph_formula_parts_p,
    parse_formula_p,
)

#: MAYGEN 1.8 实测计数（P=3 口径；一次性外部对照，不进入日常回归）
#: 生成命令：java -jar MAYGEN/MAYGEN-1.8.jar -f <式> -t
ANCHORS: Dict[str, int] = {
    'CH3P': 1,
    'CH5P': 1,
    'C2H5P': 4,
    'C2H7P': 2,
    'C3H5P': 21,
    'C3H7P': 12,
    'C3H9P': 4,
    'C4H9P': 35,
    'C4H11P': 8,
    'C5H13P': 17,
    'C6H13P': 284,
    'C6H15P': 39,
    'C2H8P2': 6,
    'C3H10P2': 14,
    'C4H12P2': 38,
    # 下列用于覆盖 P≡P / P≡C 等极端键型
    'P2': 1,
    'C2H2P2': 19,
    'C2H4P2': 27,
    'CH2P2': 4,
    'C2P2': 5,
    'C3H3P': 19,
}

_QUICK = ['CH3P', 'CH5P', 'C2H7P', 'C3H9P', 'C3H7P', 'C2H8P2']


def _count(formula: str, valence_p: int, deadline: Optional[float] = None,
           parallel: bool = False) -> Tuple[int, float]:
    n_c, n_h, n_p = parse_formula_p(formula)
    gen = PhosphorusMatrixGenerator(n_c, n_p, valence_p=valence_p)
    t0 = time.time()
    graphs = gen.generate(n_h, parallel=parallel, deadline=deadline)
    return len(graphs), time.time() - t0


def check_anchors(formulas: Optional[List[str]] = None,
                  valence_p: int = 3,
                  timeout_per: float = 300.0) -> bool:
    """一次性对拍：本模块计数 vs MAYGEN 1.8 实测值"""
    names = formulas or list(ANCHORS)
    print(f"=== 锚点对拍（MAYGEN 1.8，P={valence_p} 价）===")
    print(f"{'分子式':<10}{'期望':>6}{'实得':>8}{'用时':>9}  结果")
    ok_all = True
    for f in names:
        expect = ANCHORS.get(f)
        if expect is None:
            print(f"{f:<10}{'—':>6}{'—':>8}{'—':>9}  跳过（无锚点）")
            continue
        try:
            got, dt = _count(f, valence_p,
                             deadline=time.time() + timeout_per)
        except TimeoutError:
            print(f"{f:<10}{expect:>6}{'超时':>8}{timeout_per:>8.0f}s  FAIL")
            ok_all = False
            continue
        good = (got == expect)
        ok_all = ok_all and good
        print(f"{f:<10}{expect:>6}{got:>8}{dt:>8.2f}s  "
              + ("OK" if good else "FAIL"))
    print(f"\n锚点结果：{'全部通过 ✓' if ok_all else '存在不一致 ✗'}")
    return ok_all


def check_props(formulas: Optional[List[str]] = None,
                valence_p: int = 3) -> bool:
    """内部性质：公式自洽 / 价态 / 连通性 / 去重无重复"""
    names = formulas or list(ANCHORS)
    print("=== 内部性质自检 ===")
    ok_all = True
    for f in names:
        n_c, n_h, n_p = parse_formula_p(f)
        gen = PhosphorusMatrixGenerator(n_c, n_p, valence_p=valence_p)
        if not gen.feasible(n_h):
            print(f"{f:<10} 不可行（跳过）")
            continue
        graphs = gen.generate(n_h)
        problems = []
        for G in graphs:
            if not nx.is_connected(G):
                problems.append('不连通')
                continue
            try:
                parts = graph_formula_parts_p(G, valence_p)
            except FormulaError as exc:
                problems.append(f'公式异常：{exc}')
                continue
            if parts != (n_c, n_h, n_p):
                problems.append(f'公式不符：{parts} != {(n_c, n_h, n_p)}')
            for nd, d in G.nodes(data=True):
                lab = d.get('label', 'C')
                used = sum(bond_order(G[nd][nb].get('bond_type', 1))
                           for nb in G[nd])
                want = valence_p if lab == 'P' else 4
                if used > want:
                    problems.append(f'{lab}{nd} 超价 {used}>{want}')
        # 精确去重检查（WL 哈希 + VF2 全属性）
        seen = {}
        dup = 0
        for G in graphs:
            key = nx.weisfeiler_lehman_graph_hash(
                G, node_attr='label', edge_attr='bond_type')
            bucket = seen.setdefault(key, [])
            if any(nx.is_isomorphic(G, H, node_match=_node_match,
                                    edge_match=_edge_match) for H in bucket):
                dup += 1
            else:
                bucket.append(G)
        if dup:
            problems.append(f'重复图 {dup} 个')
        status = 'OK' if not problems else 'FAIL'
        ok_all = ok_all and not problems
        print(f"{f:<10} {len(graphs):>5} 结构  {status}"
              + (f"  {'; '.join(problems[:3])}" if problems else ""))
    print(f"\n性质自检：{'全部通过 ✓' if ok_all else '存在问题 ✗'}")
    return ok_all


def check_equiv(formulas: Optional[List[str]] = None,
                valence_p: int = 3) -> bool:
    """串行 vs 并行：结果集合必须逐元素一致"""
    names = formulas or ['C3H9P', 'C4H11P', 'C6H15P', 'C2H8P2']
    print("=== 串行 / 并行 等价 ===")
    ok_all = True
    for f in names:
        n_c, n_h, n_p = parse_formula_p(f)
        gen = PhosphorusMatrixGenerator(n_c, n_p, valence_p=valence_p)
        a = gen.generate(n_h, parallel=False)
        b = gen.generate(n_h, parallel=True)
        ka = {nx.weisfeiler_lehman_graph_hash(g, node_attr='label',
                                              edge_attr='bond_type') for g in a}
        kb = {nx.weisfeiler_lehman_graph_hash(g, node_attr='label',
                                              edge_attr='bond_type') for g in b}
        good = (len(a) == len(b) and ka == kb)
        ok_all = ok_all and good
        print(f"  {f:<10} 串行={len(a):<5} 并行={len(b):<5} "
              + ("OK" if good else "FAIL"))
    print(f"\n等价性：{'通过 ✓' if ok_all else '不一致 ✗'}")
    return ok_all


def _node_match(a, b) -> bool:
    return a.get('label', 'C') == b.get('label', 'C')


def _edge_match(a, b) -> bool:
    return bond_order(a.get('bond_type', 1)) == bond_order(b.get('bond_type', 1))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='python -m Phosphorus.validate',
        description='Phosphorus 模块验证（一期 C/H/P）')
    ap.add_argument('--anchors', action='store_true', help='与 MAYGEN 对拍')
    ap.add_argument('--quick', action='store_true', help='小锚点集（快速）')
    ap.add_argument('--props', action='store_true', help='内部性质自检')
    ap.add_argument('--equiv', action='store_true', help='串行/并行等价')
    ap.add_argument('--valence-p', type=int, default=3, choices=(3, 5))
    ap.add_argument('--timeout', type=float, default=300.0,
                    help='单个分子式超时秒数（默认 300）')
    ap.add_argument('--formulas', nargs='*', default=None,
                    help='仅检查指定分子式')
    args = ap.parse_args(argv)

    do_any = args.anchors or args.props or args.equiv or args.quick
    if not do_any:
        args.anchors = True

    ok = True
    if args.quick:
        ok &= check_anchors(_QUICK, args.valence_p, args.timeout)
    if args.anchors:
        ok &= check_anchors(args.formulas, args.valence_p, args.timeout)
    if args.props:
        ok &= check_props(args.formulas, args.valence_p)
    if args.equiv:
        ok &= check_equiv(args.formulas, args.valence_p)
    print("\n" + "=" * 60)
    print("全部验证通过 ✓" if ok else "存在失败项 ✗")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
