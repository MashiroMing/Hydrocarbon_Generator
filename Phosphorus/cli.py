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
from typing import List, Optional

import networkx as nx

from .atomic_matrix_p import PhosphorusMatrixGenerator
from .formula import (
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
        description='含磷异构体生成器（第一期：仅 C/H/P，P 默认 3 价）')
    p.add_argument('formula', help='分子式，如 C3H9P / CH5P / C6H13P')
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


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        n_c, n_h, n_p = parse_formula_p(args.formula)
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
        print(f"错误：{args.formula} 在该口径下无解"
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
        print(f"{args.formula}\t{len(graphs)}\t{elapsed:.3f}")
    elif args.smi:
        for G in graphs:
            smi = graph_to_smiles(G)
            print(smi if smi is not None else graph_to_text(G))
    elif args.json:
        data = {
            'formula': format_graph_formula_p(graphs[0], args.valence_p) if graphs else args.formula,
            'input_formula': args.formula,
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
        print(f"分子式 {args.formula}：{len(graphs)} 个结构（{elapsed:.3f}s）")
        for G in graphs[:20]:
            smi = graph_to_smiles(G)
            print('  ' + (smi if smi is not None else graph_to_text(G)))
        if len(graphs) > 20:
            print(f"  …（共 {len(graphs)} 个，仅显示前 20）")
    return 0


if __name__ == '__main__':
    sys.exit(main())
