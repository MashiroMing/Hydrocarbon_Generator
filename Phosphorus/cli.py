"""Phosphorus CLI（第一期：仅 C/H/P）

用法示例：
    python -m Phosphorus.cli C3H9P --count
    python -m Phosphorus.cli C3H9P --smi
    python -m Phosphorus.cli C6H13P --json out.json --parallel
    python -m Phosphorus.cli CH5P --count --valence-p 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional, Tuple

import networkx as nx

from .atomic_matrix_p import PhosphorusMatrixGenerator
from .formula import (
    formula_string,
    FormulaError,
    dbe,
    format_graph_formula_p,
    graph_formula_parts_p,
    parse_formula_p,
)


def graph_to_smiles(G: nx.Graph) -> Optional[str]:
    """图 → SMILES（RDKit 可用时）；失败返回 None"""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return None
    try:
        rw = Chem.RWMol()
        order = sorted(G.nodes())
        for node in order:
            label = G.nodes[node].get('label', 'C')
            rw.AddAtom(Chem.Atom(label))
        idx = {node: k for k, node in enumerate(order)}
        for a, b, d in G.edges(data=True):
            bt = int(d.get('bond_type', 1))
            btype = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                     3: Chem.BondType.TRIPLE}[bt]
            rw.AddBond(idx[a], idx[b], btype)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, kekuleSmiles=True)
    except Exception:
        return None


def graph_to_text(G: nx.Graph) -> str:
    """无 RDKit 时的文本表示：``标签(键级)标签`` 的邻接列表"""
    parts = []
    for node in sorted(G.nodes()):
        label = G.nodes[node].get('label', 'C')
        nbs = []
        for nb in sorted(G.neighbors(node)):
            if nb > node:
                nbs.append(f"{nb}:{G[node][nb].get('bond_type', 1)}")
        parts.append(f"{node}{label}" + (f"({','.join(nbs)})" if nbs else ''))
    return ' '.join(parts)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m Phosphorus.cli',
        description='含磷异构体生成器（第一期：仅 C/H/P，P 默认 3 价）',
        epilog=('三种输入方式等价：\n'
                '  1) 分子式：      python -m Phosphorus.cli C3H9P\n'
                '  2) 分开给个数：  python -m Phosphorus.cli --c 3 --h 9 --p 1\n'
                '  3) 交互式：      python -m Phosphorus.cli --interactive'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('formula', nargs='?',
                   help='分子式，如 C3H9P / CH5P / C6H13P（与 --c/--h/--p 二选一）')
    p.add_argument('--c', '--carbon', dest='n_c', type=int, metavar='N',
                   help='碳原子数')
    p.add_argument('--h', '--hydrogen', dest='n_h', type=int, metavar='N',
                   help='氢原子数')
    p.add_argument('--p', '--phosphorus', dest='n_p', type=int, metavar='N',
                   help='磷原子数（需要 ≥1）')
    p.add_argument('--interactive', action='store_true',
                   help='交互式输入 C/H/P 个数（逐项提示）')
    p.add_argument('--count', action='store_true',
                   help='只输出计数（格式：式<TAB>计数<TAB>秒，便于与 MAYGEN 对拍）')
    p.add_argument('--smi', action='store_true', help='每行输出一个 SMILES')
    p.add_argument('--json', metavar='PATH', help='把结构导出为 JSON')
    p.add_argument('--max-atoms', type=int, default=10,
                   help='重原子上限（默认 10）')
    p.add_argument('--valence-p', type=int, default=3, choices=(3, 5),
                   help='磷价态：3=默认（一期），5=预留（实验性，无锚点保证）')
    p.add_argument('--parallel', action='store_true',
                   help='按前缀并行（结果集与串行一致）')
    p.add_argument('--timeout', type=float, default=0,
                   help='超时秒数（0=不限制）')
    return p


def _read_int(prompt: str, default: Optional[int] = None) -> int:
    """交互式读取一个非负整数（回车用默认值）

    提示信息走 **stderr**，stdout 只保留结果（便于脚本/对拍直接解析）。
    """
    while True:
        hint = f"（回车默认 {default}）" if default is not None else ""
        print(f"{prompt}{hint}：", end='', file=sys.stderr, flush=True)
        raw = sys.stdin.readline().strip()
        if raw == '':
            if default is not None:
                return default
            print("请输入整数。", file=sys.stderr)
            continue
        try:
            val = int(raw)
        except ValueError:
            print("请输入整数。", file=sys.stderr)
            continue
        if val < 0:
            print("请输入非负整数。", file=sys.stderr)
            continue
        return val


def _resolve_input(args) -> Tuple[str, int, int, int]:
    """统一入口：分子式 / 分开个数 / 交互式 → (显示用式, nC, nH, nP)"""
    has_counts = any(v is not None for v in (args.n_c, args.n_h, args.n_p))
    given = sum(v is not None for v in (args.n_c, args.n_h, args.n_p))
    if args.formula and has_counts:
        raise FormulaError("分子式与 --c/--h/--p 只能二选一")
    if args.interactive and (args.formula or has_counts):
        raise FormulaError("--interactive 不能与分子式/--c/--h/--p 同时使用")

    if args.interactive:
        print("交互式输入（P 个数需要 ≥1）", file=sys.stderr)
        n_c = _read_int("碳原子数 C", 0)
        n_h = _read_int("氢原子数 H", 0)
        n_p = _read_int("磷原子数 P", 1)
        if n_p < 1:
            raise FormulaError("磷原子数必须 ≥1")
        return (formula_string(n_c, n_h, n_p), n_c, n_h, n_p)

    if has_counts:
        if given != 3:
            raise FormulaError("--c / --h / --p 必须同时给出（三个都要）")
        n_c, n_h, n_p = args.n_c, args.n_h, args.n_p
        if n_c < 0 or n_h < 0 or n_p < 1:
            raise FormulaError("个数必须非负，且 P ≥ 1")
        return (formula_string(n_c, n_h, n_p), n_c, n_h, n_p)

    if not args.formula:
        raise FormulaError("请提供分子式，或使用 --c/--h/--p，或 --interactive")
    n_c, n_h, n_p = parse_formula_p(args.formula)
    return (args.formula, n_c, n_h, n_p)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        formula, n_c, n_h, n_p = _resolve_input(args)
    except FormulaError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.valence_p == 5:
        print("警告：P=5 为预留口径（无 MAYGEN 锚点保证），仅供实验。",
              file=sys.stderr)

    k = dbe(n_c, n_h, n_p)
    if k is None:
        print(f"错误：分子式 {args.formula} 的不饱和度非法"
              "（H 数奇偶/大小不符）", file=sys.stderr)
        return 2

    gen = PhosphorusMatrixGenerator(n_c, n_p, valence_p=args.valence_p,
                                    max_atoms=args.max_atoms)
    if gen.N > args.max_atoms:
        print(f"错误：重原子数 {gen.N} 超过上限 {args.max_atoms}"
              "（用 --max-atoms 调整）", file=sys.stderr)
        return 2
    if not gen.feasible(n_h):
        print(f"错误：{formula} 在该口径下无解"
              f"（N={gen.N}, H={n_h}）", file=sys.stderr)
        return 1

    deadline = (time.time() + args.timeout) if args.timeout > 0 else None
    t0 = time.time()
    try:
        graphs = gen.generate(n_h, parallel=args.parallel, deadline=deadline)
    except TimeoutError as exc:
        print(f"超时：{exc}", file=sys.stderr)
        return 3
    elapsed = time.time() - t0

    # 公式自检（门禁）：每个结果都必须与输入分子式一致
    bad = 0
    for G in graphs:
        try:
            if graph_formula_parts_p(G, args.valence_p) != (n_c, n_h, n_p):
                bad += 1
        except FormulaError:
            bad += 1
    if bad:
        print(f"错误：{bad} 个结果分子式不符（内部一致性失败）", file=sys.stderr)
        return 4

    if args.count:
        print(f"{formula}\t{len(graphs)}\t{elapsed:.3f}")
    elif args.smi:
        for G in graphs:
            smi = graph_to_smiles(G)
            print(smi if smi is not None else graph_to_text(G))
    elif args.json:
        data = {
            'formula': format_graph_formula_p(graphs[0], args.valence_p) if graphs else formula,
            'input_formula': formula,
            'dbe': k,
            'valence_p': args.valence_p,
            'count': len(graphs),
            'seconds': round(elapsed, 6),
            'structures': [
                {
                    'atoms': [G.nodes[i].get('label', 'C') for i in sorted(G.nodes())],
                    'bonds': [[int(a), int(b), int(d.get('bond_type', 1))]
                              for a, b, d in G.edges(data=True)],
                }
                for G in graphs
            ],
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"已写入 {args.json}（{len(graphs)} 个结构，{elapsed:.3f}s）")
    else:
        print(f"分子式 {formula}：{len(graphs)} 个结构（{elapsed:.3f}s）")
        for G in graphs[:20]:
            smi = graph_to_smiles(G)
            print('  ' + (smi if smi is not None else graph_to_text(G)))
        if len(graphs) > 20:
            print(f"  …（共 {len(graphs)} 个，仅显示前 20）")
    return 0


if __name__ == '__main__':
    sys.exit(main())
