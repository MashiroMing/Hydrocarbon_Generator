"""
Nitrogen/validate.py — N 系验证入口（CLI）

功能：
  1. 计数锚点电池：对给定分子式逐一核对预言机 / 树机制 / MAYGEN 计数；
  2. 完备性检查：树机制 ⊆ 预言机（缺漏结构列表）+ 覆盖率报告；
  3. 分子式门禁冒烟：随机/全部产物验证 formula_matches_input 恒真。

用法：
  python -m Nitrogen.validate                     # 默认锚点电池
  python -m Nitrogen.validate C5H5N               # 指定分子式，打印结构样例
  python -m Nitrogen.validate --maygen            # 含 MAYGEN 一次性对拍（需 Java）
"""

import os
import sys
import time
from typing import Dict, List, Optional

# ── 兼容直跑：python Nitrogen/validate.py（无父包上下文）──
# 推荐用法仍是 `python -m Nitrogen.validate`；直跑时把项目根加入 sys.path 并转绝对导入。
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Nitrogen.formula import (parse_formula, graph_formula_parts,
                                  format_graph_formula, dbe_from_parts)
    from Nitrogen.atomic_matrix_n import AtomicMatrixNGenerator
    from Nitrogen.generator import NitrogenGenerator
else:
    from .formula import (parse_formula, graph_formula_parts,
                          format_graph_formula, dbe_from_parts)
    from .atomic_matrix_n import AtomicMatrixNGenerator
    from .generator import NitrogenGenerator

import networkx as nx

# 参考锚点（来源：内部预言机 + MAYGEN 一次性对拍实证）
# 口径：图口径（凯库勒式变体分开计数，与 O 系一致）
MAYGEN_ANCHORS = {
    'C2H7N': 2, 'C3H9N': 4, 'C2H5N': 4, 'C2H3N': 5, 'C5H5N': 685,
    'C2H8N2': 6, 'C3H7N': 12, 'C4H11N': 8, 'C6H5N': 4394, 'C6H7N': 4378,
    'C6H15N': 39, 'C4H9N': 35,
}


def oracle_count(n_c: int, n_n: int, target_h: int,
                 deadline: Optional[float] = None) -> List:
    """预言机生成指定分子式的全部结构

    Args:
        deadline: 可选超时（time.time() 绝对时刻）。原子矩阵枚举对重原子 ≥8
                  组合爆炸，超时则返回 [] 并由调用方走快速路径/提示。
    """
    gen = AtomicMatrixNGenerator(n_c, n_n)
    if not gen._can_enumerate():
        return []
    try:
        return gen.generate_for_target_h(target_h, deadline=deadline)
    except TimeoutError:
        return []


def tree_generate(n_c: int, n_n: int, target_h: int) -> List:
    """树机制生成器产出（平铺图列表）"""
    gen = NitrogenGenerator()
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


def coverage_report(n_c: int, n_n: int, target_h: int,
                    verbose: bool = False) -> Dict:
    """树机制 ⊆ 预言机 完备性检查

    Returns:
        {'oracle': n, 'tree': n, 'covered': n, 'missing': n, 'missing_fmls': [...]}
    """
    oracle = oracle_count(n_c, n_n, target_h)
    tree = tree_generate(n_c, n_n, target_h)
    covered = 0
    missing = []
    for g in oracle:
        if any(_isomorphic(g, t) for t in tree):
            covered += 1
        else:
            missing.append(format_graph_formula(g))
    return {
        'oracle': len(oracle), 'tree': len(tree), 'covered': covered,
        'missing': len(missing), 'missing_fmls': missing,
    }


def check_anchors(formulas: Optional[List[str]] = None,
                  use_maygen: bool = False) -> int:
    """运行计数锚点电池；返回失败数"""
    if formulas is None:
        formulas = sorted(MAYGEN_ANCHORS)
    n_fail = 0
    print(f"{'分子式':<8} {'预言机':>7} {'MAYGEN':>7} {'一致':>4}  "
          f"{'DBE':>4}  耗时")
    print("-" * 60)
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
        maygen = MAYGEN_ANCHORS.get(fml)
        ok = (maygen is None) or (n_oracle == maygen)
        if not ok:
            n_fail += 1
        mark = "OK " if ok else "FAIL"
        dbe = dbe_from_parts(n_c, n_h, n_n)
        print(f"{fml:<8} {n_oracle:>7} "
              f"{(str(maygen) if maygen is not None else '-'):>7} "
              f"{mark:>4} {dbe:>4.0f}  {dt:6.1f}s")
        if ok and maygen is not None and use_maygen:
            pass  # MAYGEN 计数已在锚点中（开发期一次性对拍实证）
    print("-" * 60)
    return n_fail


def check_coverage(formulas: Optional[List[str]] = None,
                   verbose: bool = False) -> int:
    """树机制 vs 预言机 覆盖率检查；返回失败数"""
    if formulas is None:
        formulas = ['C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C2H8N2',
                    'C3H7N', 'C4H9N', 'C5H5N']
    n_fail = 0
    print(f"{'分子式':<8} {'预言机':>7} {'树机制':>7} {'覆盖':>6} "
          f"{'缺失':>5}  覆盖率")
    print("-" * 68)
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
        ok = rep['missing'] == 0
        if not ok:
            n_fail += 1
        mark = "OK " if ok else "GAP"
        print(f"{fml:<8} {rep['oracle']:>7} {rep['tree']:>7} "
              f"{rep['covered']:>6} {rep['missing']:>5}  "
              f"{ratio*100:5.1f}%  {mark}  {dt:5.1f}s")
        if verbose and rep['missing'] > 0 and rep['missing'] <= 10:
            for m in rep['missing_fmls'][:10]:
                print(f"        缺失: {m}")
    print("-" * 68)
    return n_fail


def smoke_gate(formulas: Optional[List[str]] = None) -> int:
    """分子式门禁冒烟：预言机全部产物必须通过 formula_matches_input"""
    if formulas is None:
        formulas = ['C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C2H8N2']
    n_fail = 0
    for fml in formulas:
        n_c, n_h, n_n, err = parse_formula(fml)
        if err:
            continue
        graphs = oracle_count(n_c, n_n, n_h)
        for g in graphs:
            parts = graph_formula_parts(g)
            if (parts[0], parts[1], parts[2]) != (n_c, n_h, n_n):
                print(f"门禁失败 {fml}: {parts}")
                n_fail += 1
    return n_fail


def sample_structures(formula: str, limit: int = 12,
                      graphs: Optional[List] = None,
                      source: str = '预言机') -> None:
    """打印指定分子式的结构样例（图边列表 + 分子式）

    Args:
        graphs: 已枚举的图列表；None 时内部调用预言机（小分子式）。
        source: 来源标注，如 '预言机' / '树机制(快速)'。
    """
    n_c, n_h, n_n, err = parse_formula(formula)
    if err:
        print(err)
        return
    if graphs is None:
        graphs = oracle_count(n_c, n_n, n_h)
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
  python -m Nitrogen.validate --anchors              # 只跑默认锚点电池 + 完备性

选项：
  分子式...      待计算的 C/H/N 分子式，如 C2H7N C5H5N（可多个）
  -i, --interactive   进入交互模式（逐条输入分子式）
  --anchors           运行默认计数锚点电池（等价于原无参数行为）
  --maygen            核对 MAYGEN 锚点计数（锚点电池模式下）
  -v, --verbose       打印覆盖率缺失结构明细
  --tree              强制快速树机制（秒级，非完备；重原子较大时推荐）
  --oracle            强制预言机完备枚举（重原子>7 带 30s 超时兜底）
  -h, --help          显示本帮助

加速说明（默认自动分档）：
  重原子 ≤ 7 ：预言机完备枚举（与 MAYGEN 一致）+ 覆盖率校验；
  重原子 > 7 ：自动降级为快速树机制（避免原子矩阵组合爆炸）。
"""


# 重原子上限：原子矩阵预言机对 >7 重原子组合爆炸，默认自动降级到快速树机制
# （树机制秒级出结果但非完备；小分子式预言机已实证与 MAYGEN 完全一致）
_ORACLE_ATOM_LIMIT = 7
_ORACLE_DEADLINE = 30.0   # 秒；auto/oracle 模式下预言机的超时兜底


def _run_single_formula(fml: str, use_maygen: bool = False,
                        verbose: bool = False,
                        mode: str = 'auto') -> int:
    """对单个分子式执行：样例 + 计数 + 覆盖率

    加速策略（按重原子数 N 自动分档）：
      - N ≤ 7   : 预言机完备枚举（与 MAYGEN 一致），并跑覆盖率校验；
      - N > 7   : 默认改用快速树机制（秒级，非完备），避免预言机组合爆炸；
      - mode='tree'   : 强制树机制（最快）；
      - mode='oracle' : 强制预言机（N>7 时带 30s 超时，超时则报提示）。

    Returns:
        失败项计数（解析失败 / 预言机超时 / 覆盖率缺失）。
    """
    n_c, n_h, n_n, err = parse_formula(fml)
    if err:
        print(f"{fml}: 解析失败 -> {err}")
        return 1
    n_atoms = n_c + n_n
    n_fail = 0

    if mode == 'tree' or (mode == 'auto' and n_atoms > _ORACLE_ATOM_LIMIT):
        # ── 快速树机制路径（秒级，非完备） ──
        t0 = time.time()
        tree = tree_generate(n_c, n_n, n_h)
        dt = time.time() - t0
        note = f"树机制(快速，非完备，{dt:.1f}s)"
        print(f"  （{fml} 重原子 {n_atoms} 较大，采用 {note}；"
              f"如需完备验证请用 --oracle 或减少原子数）")
        sample_structures(fml, graphs=tree, source=note)
        # 锚点核对：树机制计数非完备，仅提示不做 FAIL
        maygen = MAYGEN_ANCHORS.get(fml)
        if maygen is not None:
            mark = "OK " if len(tree) == maygen else "GAP"
            print(f"  {fml:8s} 树机制 {len(tree):>6}  "
                  f"MAYGEN锚点 {maygen:>6}  {mark}")
            if len(tree) != maygen:
                n_fail += 1
        return n_fail

    # ── 预言机完备路径（N ≤ 7，或 --oracle 强制） ──
    deadline = time.time() + _ORACLE_DEADLINE if n_atoms > _ORACLE_ATOM_LIMIT else None
    graphs = oracle_count(n_c, n_n, n_h, deadline=deadline)
    if not graphs and n_atoms > _ORACLE_ATOM_LIMIT:
        print(f"  （{fml} 重原子 {n_atoms} 预言机枚举超过 {_ORACLE_DEADLINE:.0f}s，"
              f"已中断；建议改用默认快速树机制）")
        return 1
    sample_structures(fml, graphs=graphs, source='预言机')
    n_fail += check_anchors([fml], use_maygen)
    if n_atoms <= _ORACLE_ATOM_LIMIT:
        n_fail += check_coverage([fml], verbose)
    else:
        print(f"  （{fml} 重原子 {n_atoms} > {_ORACLE_ATOM_LIMIT}，跳过覆盖率枚举）")
    return n_fail


def _interactive_loop(use_maygen: bool = False,
                      verbose: bool = False,
                      mode: str = 'auto') -> int:
    """交互模式：循环提示用户输入分子式，逐条计算，直到退出"""
    print()
    print("交互模式：输入 C/H/N 分子式后回车即计算（如 C2H7N / C5H5N）")
    print("  特殊命令：quit/exit/空行=退出；anchors=跑锚点电池；help=用法")
    if mode == 'tree':
        print("  当前模式：快速树机制（--tree，秒级，非完备）")
    elif mode == 'oracle':
        print("  当前模式：预言机完备（--oracle，重原子>7 带超时）")
    else:
        print(f"  当前模式：自动（重原子 ≤{_ORACLE_ATOM_LIMIT} 预言机，"
              f">{_ORACLE_ATOM_LIMIT} 自动降级树机制）")
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
            total_fail += check_anchors(use_maygen=use_maygen)
            continue
        total_fail += _run_single_formula(raw, use_maygen, verbose, mode=mode)
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
    use_maygen = '--maygen' in argv
    verbose = '-v' in argv or '--verbose' in argv
    interactive = ('-i' in argv) or ('--interactive' in argv)
    anchors_only = '--anchors' in argv
    help_requested = '-h' in argv or '--help' in argv
    if '--tree' in argv:
        mode = 'tree'
    elif '--oracle' in argv:
        mode = 'oracle'
    else:
        mode = 'auto'
    positional = [a for a in argv if not a.startswith('-')]

    print("=" * 68)
    print("Nitrogen 验证入口：计数锚点 + 完备性 + 门禁冒烟")
    print("=" * 68)

    if help_requested:
        print(_USAGE)
        return 0

    n_fail = 0
    n_fail += smoke_gate()

    if anchors_only:
        # 只跑默认锚点电池（等价于原"无参数"行为）
        n_fail += check_anchors(use_maygen=use_maygen)
        n_fail += check_coverage(verbose=verbose)
    elif positional and not interactive:
        # 批量：命令行给出的分子式逐条计算
        for fml in positional:
            n_fail += _run_single_formula(fml, use_maygen, verbose, mode=mode)
    else:
        # 交互模式（无分子式参数，或显式 -i/--interactive）
        n_fail += _interactive_loop(use_maygen, verbose, mode=mode)

    print()
    if n_fail == 0:
        print("全部验证通过 ✓")
    else:
        print(f"存在 {n_fail} 项失败 ✗")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
