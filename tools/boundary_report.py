#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内部边界分析报告工具（无 MAYGEN / Java 依赖）

对给定分子式，用项目自身的"完备性预言机"（原子矩阵目标 H 剪枝枚举）
对比"树机制集合"，量化边界情况（树机制无法表达的结构），
并按官能团分类 —— 全部内部计算，MAYGEN 仅可作可选外部复核。

用法:
    python tools/boundary_report.py C7H6O2
    python tools/boundary_report.py C3H6O2 C6H6O2 --top-missing 15
    python tools/boundary_report.py C7H6O2 --max-atoms 10
"""
import argparse
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def format_report(r: dict, top_missing: int) -> str:
    if r.get('error'):
        return f"{r['formula']}: 跳过 — {r['error']}"

    lines = []
    lines.append("=" * 72)
    lines.append(f"公式: {r['formula']}  (k={r['k']}) — 内部边界分析（无 MAYGEN）")
    lines.append("=" * 72)
    lines.append(f"  树机制集合: {r['tree']} 个")
    lines.append(f"  完备集合(矩阵预言机): {r['matrix']} 个")
    lines.append(f"  交集: {r['intersection']}")
    lines.append(f"  内部覆盖率(交集/完备): {r['coverage'] * 100:.2f}%")
    lines.append(f"  边界缺失(完备−树): {r['missing']} 个")
    lines.append(f"  我方多余(树−完备): {r['extra']} 个"
                 f"  {'⚠ 需核验' if r['extra'] else '（0 ✓）'}")
    if r['missing']:
        lines.append("  边界结构按官能团分类（可重叠）:")
        for name, cnt in sorted(r['categories'].items(), key=lambda x: -x[1]):
            if cnt:
                lines.append(f"    {name}: {cnt}")
        lines.append(f"  缺失示例（前 {min(top_missing, r['missing'])} 条）:")
        for s, tags in r['examples'][:top_missing]:
            lines.append(f"    {s}   [{', '.join(tags) or '其他'}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="内部边界分析（无 MAYGEN 依赖）")
    ap.add_argument("formulas", nargs="+", help="分子式，如 C7H6O2")
    ap.add_argument("--top-missing", type=int, default=10)
    ap.add_argument("--max-atoms", type=int, default=8,
                    help="矩阵预言机重原子数上限（默认 8；9~10 显著变慢，谨慎使用）")
    ap.add_argument("--chem-mode", default='math',
                    choices=['math', 'chem'],
                    help="过滤口径（默认 math 八隅律）")
    args = ap.parse_args()

    from oxygen.boundary import internal_boundary_report
    from utils import parse_compound_formula

    print("内部边界分析（树机制 vs 矩阵完备预言机）— 不依赖 MAYGEN")
    rows = []
    for f in args.formulas:
        res = parse_compound_formula(f)
        if res and res[1] is not None and res[1] + res[3] > args.max_atoms:
            print(f"{f}: 重原子数 {res[1] + res[3]} > {args.max_atoms}"
                  f"（--max-atoms 上限）→ 跳过，提示：N≥9 时矩阵枚举显著变慢")
            rows.append((f, '-', '-', '-', '-', '-'))
            continue
        r = internal_boundary_report(f, chem_mode=args.chem_mode,
                                     max_atoms=args.max_atoms,
                                     top_missing=args.top_missing)
        print(format_report(r, args.top_missing))
        print()
        if r.get('error'):
            rows.append((f, '-', '-', '-', '-', '-'))
        else:
            rows.append((f, r['tree'], r['matrix'], r['intersection'],
                         r['missing'], f"{r['coverage'] * 100:.2f}%"))

    print("汇总:")
    print(f"{'公式':10s} {'树':>8s} {'完备':>8s} {'交集':>8s} {'缺失':>6s} {'内部覆盖率':>10s}")
    for f, t, m, i, miss, cov in rows:
        print(f"{f:10s} {t:>8} {m:>8} {i:>8} {miss:>6} {cov:>10s}")


if __name__ == "__main__":
    main()
