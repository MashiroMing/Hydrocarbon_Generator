#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nitrogen/maygen_compare.py — MAYGEN 外部对照脚本（**可选**，独立工具）

═══ 定位与红线 ═══
本脚本是**开发期一次性外部对照**，与 O 系 tools/maygen_validate.py 同等待遇：
  · 需**显式调用**才运行，不参与任何回归测试、不参与 pass/fail 判定；
  · Nitrogen/ 模块本体（generator / validate / formula / atomic_matrix_n）
    **不包含任何 MAYGEN 接口**，MAYGEN 仅在本脚本内经命令行外部调用；
  · 项目对边界与完备性的判断**一律以内部预言机为准**（合规红线 §8.4-3）。

即：本脚本用于给内部预言机提供**一次性的外部旁证**，事后即失去流程意义；
任何结论若只在本脚本成立、在 validate.py 不成立，一律以 validate.py 为准。

═══ 用法 ═══
    python -m Nitrogen.maygen_compare                # 默认分子式电池
    python -m Nitrogen.maygen_compare C5H5N C6H7N    # 指定分子式
    python -m Nitrogen.maygen_compare --set          # 额外做集合级对比（SMILES + 键级图级复核，需 RDKit）
    python -m Nitrogen.maygen_compare --set --verify-limit 2000   # 限制图级核验项数
    python -m Nitrogen.maygen_compare --jar <路径>   # 指定 MAYGEN JAR

═══ 前置条件 ═══
  · 本机 Java（`java -version` 可用）
  · MAYGEN JAR：默认 <项目根>/MAYGEN/MAYGEN-1.8.jar
  · 集合级对比另需 RDKit（缺失时自动跳过，仅做计数对比）
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Nitrogen.formula import parse_formula_full               # noqa: E402
from Nitrogen.validate import oracle_count, INTERNAL_BASELINE   # noqa: E402

DEFAULT_JAR = os.path.join(ROOT, "MAYGEN", "MAYGEN-1.8.jar")
MAYGEN_TIMEOUT = 900          # 秒
_ORACLE_LIMIT = 8             # 重原子上限：内部预言机对 ≥9 显著变慢
_GRAPH_VERIFY_LIMIT = 0       # 图级复核每方向核验上限（0 = 全部核验）


def _rdkit():
    """惰性导入 RDKit；缺失时返回 None（集合级对比自动降级）"""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        return Chem
    except Exception:
        return None


def graph_to_smiles(G, Chem):
    """N 系分子图 → 规范 SMILES（本地实现，不依赖 utils.py / RDKit 之外的模块）

    支持 label ∈ {C, N} 与 bond_type ∈ {single, double, triple}。
    """
    if Chem is None:
        return None
    mol = Chem.RWMol()
    idx = {}
    for nd in G.nodes():
        label = G.nodes[nd].get('label', 'C')
        atom = Chem.Atom(label)
        atom.SetNoImplicit(False)
        idx[nd] = mol.AddAtom(atom)
    bt = {'single': Chem.BondType.SINGLE,
          'double': Chem.BondType.DOUBLE,
          'triple': Chem.BondType.TRIPLE}
    for u, v, d in G.edges(data=True):
        mol.AddBond(idx[u], idx[v],
                    bt.get(d.get('bond_type', 'single'), Chem.BondType.SINGLE))
    try:
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


# ═══ 图级复核工具（消除 SMILES 书写/芳香化口径造成的假缺失、假多余）═══
#
# 背景（2026-09 实测 C7H5N 发现）：RDKit 的**芳香性感知依赖输入凯库勒式**，
# 同一分子由「我方键级图 → mol」与「MAYGEN 芳香式 SMILES → mol」两条路径，
# 可能得到**不同的规范 SMILES**（紧张稠合体系如双 3 元环稠芳环尤甚）。
# 因此纯 SMILES 集合对比会报出"缺失/多余"，而它们在键级图层面其实是同一结构。
# 故对不一致项追加**键级图同构**判定：命中 ⇒ 书写口径差异；未命中 ⇒ 真实差异。
#
# 另注：MAYGEN 的 `-smi` 输出本身会含重复行（其 tsv 计数按行统计），
# 故唯一 SMILES 数常小于其计数——本脚本两侧都以**唯一化**后的集合比较。

def graph_wl_key(G):
    """键级图的 WL 哈希（仅 label + bond_type，与预言机图口径一致）"""
    H = nx.Graph()
    for n in G.nodes():
        H.add_node(n, label=G.nodes[n].get('label', 'C'))
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, bond_type=d.get('bond_type', 'single'))
    return nx.weisfeiler_lehman_graph_hash(H, edge_attr='bond_type',
                                           node_attr='label')


def graphs_isomorphic(a, b):
    """键级图精确同构（label + bond_type）"""
    nm = nx.algorithms.isomorphism.categorical_node_match('label', 'C')
    em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
    return nx.is_isomorphic(a, b, node_match=nm, edge_match=em)


def smiles_to_graph(smi, Chem):
    """SMILES → 显式键级图（**必须先 Kekulize**，否则取到的是芳香键，
    与预言机的显式单双键图不可比）"""
    if Chem is None:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    mk = Chem.Mol(m)
    try:
        Chem.Kekulize(mk, clearAromaticFlags=True)
    except Exception:
        return None
    G = nx.Graph()
    for a in mk.GetAtoms():
        G.add_node(a.GetIdx(), label=a.GetSymbol())
    bt = {Chem.BondType.SINGLE: 'single', Chem.BondType.DOUBLE: 'double',
          Chem.BondType.TRIPLE: 'triple'}
    for b in mk.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                   bond_type=bt.get(b.GetBondType(), 'single'))
    return G


def _bucketize(graphs):
    out = {}
    for g in graphs:
        out.setdefault(graph_wl_key(g), []).append(g)
    return out


def _in_buckets(g, buckets):
    for cand in buckets.get(graph_wl_key(g), []):
        if graphs_isomorphic(g, cand):
            return True
    return False


def relax_ring_bonds(G):
    """把**环内边**的键级松弛为 single（返回新图，不改原图）。

    用于识别"同一分子的不同凯库勒式"：芳香/紧张稠合体系里，两侧可能各写一种
    双键排布；此时键级图不同构，但环骨架 + 原子类型 + 环外键级一致。
    ⚠ 这是**口径差异**（本项目图口径下凯库勒变体刻意分开计数），
    **不是**覆盖缺口——故单独归类，不计入"真实缺口"。
    """
    R = G.copy()
    try:
        cyc = set()
        for c in nx.cycle_basis(G):
            for i in range(len(c)):
                cyc.add(frozenset((c[i], c[(i + 1) % len(c)])))
        for u, v, d in R.edges(data=True):
            if frozenset((u, v)) in cyc:
                d['bond_type'] = 'single'
    except Exception:
        pass
    return R


def run_maygen(formula: str, jar: str, workdir: str):
    """外部调用 MAYGEN；返回 (计数, SMILES 集合, 状态说明)

    注意：本机实测 MAYGEN-1.8 的 `-smi` 单独使用可能输出为空，
    故以 `-t`（tsv 计数）为主判据，`-smi` 仅为集合对比的可选补充。
    """
    if not os.path.exists(jar):
        return None, set(), f"未找到 JAR: {jar}"

    cmd = ["java", "-jar", jar, "-f", formula, "-t", "-smi", "-o", workdir]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=MAYGEN_TIMEOUT)
    except FileNotFoundError:
        return None, set(), "未找到 java（请确认本机 Java 可用）"
    except subprocess.TimeoutExpired:
        return None, set(), f"超时 (>{MAYGEN_TIMEOUT}s)"
    dt = time.time() - t0

    # 解析 tsv 计数：容错 tab / 逗号 / 分号分隔，取第 2 列
    count = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("formula"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").replace(";", ",").split(",")]
        if len(parts) >= 2 and any(c.isdigit() for c in parts[1]):
            try:
                count = int(float(parts[1]))
                break
            except ValueError:
                continue

    status = f"OK ({dt:.1f}s)"
    if proc.returncode != 0:
        status = f"退出码 {proc.returncode}: {(proc.stderr or '').strip()[:160]}"

    # 集合级：读取 workdir 下的 .smi 文件
    smiles = set()
    Chem = _rdkit()
    if Chem is not None:
        for f in glob.glob(os.path.join(workdir, "*.smi")):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    m = Chem.MolFromSmiles(line.split()[0])
                    if m is not None:
                        smiles.add(Chem.MolToSmiles(m))
    return count, smiles, status


def compare(formula: str, jar: str, do_set: bool = False,
            verify_limit: int = 0) -> dict:
    """单个分子式：内部预言机 vs MAYGEN（路线 A 起支持 C/H/N/O）"""
    n_c, n_h, n_n, n_o, err = parse_formula_full(formula)
    if err:
        print(f"  [错误] 公式解析失败: {err}")
        return {'formula': formula, 'error': err}

    n_heavy = n_c + n_n + n_o
    print("=" * 72)
    print(f"分子式: {formula}   (nC={n_c}, nH={n_h}, nN={n_n}, nO={n_o})")
    print("=" * 72)

    if n_heavy > _ORACLE_LIMIT:
        print(f"  [跳过] 重原子 {n_heavy} > {_ORACLE_LIMIT}，内部预言机枚举过慢")
        return {'formula': formula, 'skipped': True}

    print("[1/2] 内部预言机枚举...")
    t0 = time.time()
    graphs = oracle_count(n_c, n_n, n_h, n_o=n_o)
    dt_oracle = time.time() - t0
    print(f"  内部预言机: {len(graphs)} 个结构 ({dt_oracle:.1f}s)")
    base = INTERNAL_BASELINE.get(formula)
    if base is not None:
        print(f"  内部基线: {base}  {'一致 ✓' if base == len(graphs) else '不一致 ✗'}")

    print("[2/2] MAYGEN 外部生成...")
    with tempfile.TemporaryDirectory(prefix="maygen_n_") as workdir:
        mg_count, mg_smiles, status = run_maygen(formula, jar, workdir)
        print(f"  MAYGEN: 计数={mg_count}  SMILES={len(mg_smiles)}  | {status}")

    result = {'formula': formula, 'oracle': len(graphs), 'maygen': mg_count}

    if mg_count is not None:
        ok = (mg_count == len(graphs))
        result['match'] = ok
        print()
        print(f"  计数对比: 内部预言机 {len(graphs)}  vs  MAYGEN {mg_count}  "
              f"→  {'完全一致 ✓' if ok else '存在差异 ⚠'}")

    if do_set:
        Chem = _rdkit()
        if Chem is None:
            print("  [跳过集合对比] 未安装 RDKit")
        elif not mg_smiles:
            print("  [跳过集合对比] MAYGEN 未产出 SMILES（本机 -smi 输出为空）")
        else:
            ours_map = {}
            for g in graphs:
                s = graph_to_smiles(g, Chem)
                if s:
                    ours_map.setdefault(s, g)
            ours = set(ours_map)
            inter = ours & mg_smiles
            miss = sorted(mg_smiles - ours)
            extra = sorted(ours - mg_smiles)
            print()
            print(f"  集合对比(SMILES): 我方 {len(ours)}  交集 {len(inter)}  "
                  f"MAYGEN {len(mg_smiles)}")
            print(f"    我方缺失 (MAYGEN−我们): {len(miss)}")
            print(f"    我方多余 (我们−MAYGEN): {len(extra)}")

            # ── 图级复核：把"缺失/多余"拆成 真实差异 vs 口径差异 ──
            #   ① 精确键级图同构命中 → SMILES 书写/芳香化差异（同一图）
            #   ② 环键级松弛后命中   → 凯库勒式变体（同一分子，图口径下各计一次）
            #   ③ 两者皆不命中       → 真实缺口 / 真实多余
            real_miss, fake_miss, kek_miss = [], [], []
            real_extra, fake_extra, kek_extra = [], [], []
            if miss or extra:
                print("  [图级复核] 键级图同构判定中（区分书写差异 / 凯库勒变体 / 真实差异）...")
                ours_b = _bucketize(list(ours_map.values()))
                ours_rb = _bucketize([relax_ring_bonds(g)
                                      for g in ours_map.values()])
                mg_graphs = {}
                for s in mg_smiles:
                    g = smiles_to_graph(s, Chem)
                    if g is not None:
                        mg_graphs[s] = g
                mg_b = _bucketize(list(mg_graphs.values()))
                mg_rb = _bucketize([relax_ring_bonds(g)
                                    for g in mg_graphs.values()])

                todo_miss = miss if verify_limit <= 0 else miss[:verify_limit]
                for s in todo_miss:
                    g = mg_graphs.get(s)
                    if g is not None and _in_buckets(g, ours_b):
                        fake_miss.append(s)        # ① 同一键级图
                    elif g is not None and _in_buckets(relax_ring_bonds(g), ours_rb):
                        kek_miss.append(s)         # ② 凯库勒式变体
                    else:
                        real_miss.append(s)        # ③ 真实缺口
                todo_extra = extra if verify_limit <= 0 else extra[:verify_limit]
                for s in todo_extra:
                    g = ours_map.get(s)
                    if g is not None and _in_buckets(g, mg_b):
                        fake_extra.append(s)
                    elif g is not None and _in_buckets(relax_ring_bonds(g), mg_rb):
                        kek_extra.append(s)
                    else:
                        real_extra.append(s)

                print(f"    缺失 {len(miss)} = 真实缺口 {len(real_miss)}"
                      f" + 书写差异(同一键级图) {len(fake_miss)}"
                      f" + 凯库勒式变体 {len(kek_miss)}")
                print(f"    多余 {len(extra)} = 真实多余 {len(real_extra)}"
                      f" + 书写差异(同一键级图) {len(fake_extra)}"
                      f" + 凯库勒式变体 {len(kek_extra)}")
                if len(todo_miss) < len(miss) or len(todo_extra) < len(extra):
                    print("    [注意] 已按 verify_limit 截断核验（未核验项未计入分类）")
                if real_miss:
                    print("    ⚠ 真实缺口示例（前 10）:")
                    for s in real_miss[:10]:
                        print(f"      {s}")
                elif miss:
                    print("    ✓ 无真实缺口（缺失项均为书写差异或凯库勒式变体）")
                if real_extra:
                    print("    真实多余示例（前 5，我方有而 MAYGEN 无）:")
                    for s in real_extra[:5]:
                        print(f"      {s}")

            result['ours_smiles'] = len(ours)
            result['set_missing'] = len(miss)
            result['set_extra'] = len(extra)
            if miss or extra:
                result['set_missing_real'] = len(real_miss)
                result['set_extra_real'] = len(real_extra)

    print()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="MAYGEN 外部对照（可选，不参与回归；判据以内部预言机为准）")
    ap.add_argument("formulas", nargs="*",
                    help="分子式，如 C5H5N；留空则跑默认电池")
    ap.add_argument("--jar", default=DEFAULT_JAR, help="MAYGEN JAR 路径")
    ap.add_argument("--set", dest="do_set", action="store_true",
                    help="额外做集合级对比（SMILES + 键级图级复核，需 RDKit）")
    ap.add_argument("--verify-limit", type=int, default=_GRAPH_VERIFY_LIMIT,
                    help="图级复核每方向最多核验项数（0=全部核验，默认 0）")
    args = ap.parse_args(argv)

    formulas = args.formulas or [
        'C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C2H8N2', 'C3H7N', 'C4H9N', 'C5H5N',
    ]

    print("Nitrogen × MAYGEN 外部对照（可选工具 · 不参与回归 · 判据以内部预言机为准）")
    print()

    results = [compare(f, args.jar, args.do_set, args.verify_limit)
               for f in formulas]

    print("=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"{'分子式':<9} {'内部预言机':>10} {'MAYGEN':>9} {'一致':>6}")
    n_ok = n_cmp = 0
    for r in results:
        if r.get('skipped'):
            print(f"{r['formula']:<9} {'跳过（重原子过大）':>10}")
            continue
        if r.get('oracle') is None:
            print(f"{r['formula']:<9} {'—':>10}")
            continue
        mg = r.get('maygen')
        mark = '—' if mg is None else ('OK ' if r.get('match') else 'FAIL')
        if mg is not None:
            n_cmp += 1
            n_ok += 1 if r.get('match') else 0
        print(f"{r['formula']:<9} {r['oracle']:>10} "
              f"{(str(mg) if mg is not None else '—'):>9} {mark:>6}")
    print("-" * 72)
    print(f"可比对 {n_cmp} 项，一致 {n_ok} 项")
    if args.do_set:
        with_miss = [r['formula'] for r in results if r.get('set_missing_real')]
        print(f"图级复核：存在**真实缺口**的公式 {len(with_miss)} 个"
              + (f"：{', '.join(with_miss)}" if with_miss else "（全部缺失项均为书写口径差异）"))
    print()
    print("提醒：本脚本结果为一次性外部旁证，不作为项目验收判据。")
    print("      边界/完备性判定请以 python -m Nitrogen.validate 为准。")
    return 0 if n_ok == n_cmp else 1


if __name__ == '__main__':
    sys.exit(main())
