# -*- coding: utf-8 -*-
"""Phosphorus 模块回归测试（第一期：仅 C/H/P）

覆盖：
  A. 锚点计数（与 MAYGEN 1.8 对拍，P=3 口径）
  B. 内部性质（公式自洽 / P 价态=3 / 连通性 / 无重复图）
  C. 串行 == 并行
  D. CLI（--count / --smi / --json）
  E. 分子式解析错误路径

运行：
    python tests/test_phosphorus.py
    python tests/test_phosphorus.py --quick     # 仅小锚点
"""
import sys
import os
import io
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from Phosphorus.atomic_matrix_p import PhosphorusMatrixGenerator
from Phosphorus.formula import (FormulaError, dbe, format_graph_formula_p,
                                graph_formula_parts_p, parse_formula_p)
from Phosphorus.validate import ANCHORS, check_anchors, check_props, check_equiv

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}" + (f"  —— {detail}" if detail else ''))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f"  —— {detail}" if detail else ''))


def test_formula():
    print("\n== A. 分子式解析 ==")
    for f, want in [('C3H9P', (3, 9, 1)), ('CH5P', (1, 5, 1)),
                    ('C6H13P', (6, 13, 1)), ('H3P', (0, 3, 1)),
                    ('C2H8P2', (2, 8, 2))]:
        check(f"parse {f}", parse_formula_p(f) == want, str(parse_formula_p(f)))
    check("DBE C3H9P=0", dbe(3, 9, 1) == 0, str(dbe(3, 9, 1)))
    check("DBE C3H7P=1", dbe(3, 7, 1) == 1, str(dbe(3, 7, 1)))
    check("DBE CH4P 非法(半整数)", dbe(1, 4, 1) is None, str(dbe(1, 4, 1)))
    for bad in ('C2H5NO', 'C3H8', 'XYZ'):
        try:
            parse_formula_p(bad)
            check(f"拒绝非法式 {bad}", False, '未抛异常')
        except FormulaError:
            check(f"拒绝非法式 {bad}", True)


def test_formula_graph():
    print("\n== B. 图 ↔ 分子式 ==")
    G = nx.Graph()
    G.add_node(0, label='P')
    G.add_node(1, label='C')
    G.add_edge(0, 1, bond_type=1)
    # P: 3-1=2H；C: 4-1=3H → CH5P
    check("手构图 CH5P", graph_formula_parts_p(G) == (1, 5, 1),
          str(graph_formula_parts_p(G)))
    check("格式化 CH5P", format_graph_formula_p(G) == 'CH5P',
          format_graph_formula_p(G))


def test_anchors(quick):
    print("\n== C. 锚点计数（MAYGEN 1.8 对拍）==")
    names = (['CH3P', 'CH5P', 'C2H7P', 'C3H9P', 'C3H7P', 'C2H8P2']
             if quick else list(ANCHORS))
    check("锚点全对", check_anchors(names) is True)


def test_props(quick):
    print("\n== D. 内部性质 ==")
    names = (['C3H9P', 'C2H8P2'] if quick
             else ['CH3P', 'C3H9P', 'C6H13P', 'C2H8P2', 'P2', 'C2H2P2'])
    check("性质自检", check_props(names) is True)


def test_equiv():
    print("\n== E. 串行 == 并行 ==")
    check("等价", check_equiv(['C3H9P', 'C4H11P', 'C2H8P2']) is True)


def test_cli():
    print("\n== F. CLI ==")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONIOENCODING='utf-8')

    def run(*args):
        return subprocess.run([sys.executable, '-m', 'Phosphorus.cli', *args],
                              cwd=root, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', env=env)

    r = run('C3H9P', '--count')
    parts = r.stdout.split()
    ok = r.returncode == 0 and len(parts) >= 2 and parts[1] == '4'
    check("--count C3H9P=4", ok, r.stdout.strip()[:40])

    r = run('C2H7P', '--smi')
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    check("--smi C2H7P=2 行且有 S", r.returncode == 0 and len(lines) == 2,
          str(lines))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'p.json')
        r = run('C3H7P', '--json', path)
        ok = r.returncode == 0 and os.path.exists(path)
        data = {}
        if ok:
            with io.open(path, encoding='utf-8') as f:
                data = json.load(f)
        check("--json 结构完整",
              ok and data.get('count') == 12
              and len(data.get('structures', [])) == 12
              and data.get('dbe') == 1,
              f"count={data.get('count')}")

    # 并行 CLI 与串行计数一致（第三列是耗时，不参与比较）
    a = run('C4H11P', '--count').stdout.split()
    b = run('C4H11P', '--count', '--parallel').stdout.split()
    check("CLI 串行/并行计数一致",
          len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2],
          f"{' '.join(a[:2])} vs {' '.join(b[:2])}")

    r = run('C2H5NO', '--count')
    check("CLI 拒绝含 O/N", r.returncode == 2,
          (r.stderr or r.stdout).strip()[:40])


def test_cli_counts():
    """C/H/P 独立输入（--c/--h/--p）、交互式、与分子式等价、单原子边界"""
    print("\n== G. CLI 独立个数输入 ==")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONIOENCODING='utf-8')

    def run(*args, stdin=None):
        return subprocess.run([sys.executable, '-m', 'Phosphorus.cli', *args],
                              cwd=root, capture_output=True, text=True,
                              input=stdin, encoding='utf-8',
                              errors='replace', env=env)

    # 个数输入 == 分子式输入
    a = run('C3H9P', '--count').stdout.split()
    b = run('--c', '3', '--h', '9', '--p', '1', '--count').stdout.split()
    check("个数输入 == 分子式输入", len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2],
          f"{' '.join(a[:2])} vs {' '.join(b[:2])}")

    r = run('--c', '6', '--h', '13', '--p', '1', '--count')
    check("--c 6 --h 13 --p 1 → C6H13P=284",
          r.returncode == 0 and r.stdout.split()[:2] == ['C6H13P', '284'],
          r.stdout.strip()[:30])

    # 单原子边界（PH3）：曾因连通计数误判为 0
    r = run('--c', '0', '--h', '3', '--p', '1', '--count')
    check("单原子 H3P=1", r.returncode == 0
          and r.stdout.split()[:2] == ['H3P', '1'], r.stdout.strip()[:30])
    check("单原子 H3P 图性质",
          len(PhosphorusMatrixGenerator(0, 1).generate(3)) == 1)

    # 交互式（管道喂入 3 / 9 / 1）
    r = run('--interactive', '--count', stdin='3\n9\n1\n')
    check("交互式输入 → C3H9P=4", r.returncode == 0
          and r.stdout.split()[:2] == ['C3H9P', '4'],
          r.stdout.strip().splitlines()[-1][:40])

    # 错误路径
    for args, why in [(('--c', '3', '--h', '9'), '缺 --p'),
                      (('C3H9P', '--c', '3', '--h', '9', '--p', '1'), '二选一'),
                      (('--c', '3', '--h', '9', '--p', '0'), 'P≥1')]:
        r = run(*args, '--count')
        check(f"拒绝：{why}", r.returncode == 2,
              (r.stderr or r.stdout).strip()[:40])


def main():
    quick = '--quick' in sys.argv
    test_formula()
    test_formula_graph()
    test_anchors(quick)
    test_props(quick)
    test_equiv()
    test_cli()
    test_cli_counts()
    print("\n" + "=" * 60)
    print(f"结果: PASS={PASS}  FAIL={FAIL}")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
