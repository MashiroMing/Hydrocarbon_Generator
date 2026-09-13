#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAYGEN 外部交叉验证工具（独立开发工具，不属于程序本体）

用途：对本程序生成的异构体与 MAYGEN（根目录 MAYGEN/MAYGEN-1.8.jar）
生成的全量同分异构体做覆盖率对比，量化生成缺口（重点：酯/醚/羰基等）。

红线：本脚本是独立工具；程序本体（Main.py / utils.py / oxygen/ / halogen/）
不包含任何 MAYGEN 接口，MAYGEN 仅在本脚本内通过命令行外部调用。

用法:
    python tools/maygen_validate.py C7H6O2
    python tools/maygen_validate.py C7H6O2 C6H6O2 C3H6O2
    python tools/maygen_validate.py C7H6O2 --top-missing 20

输出:
    每个公式: 我方数量 / MAYGEN 数量 / 覆盖率 / 缺失结构按官能团分类统计 /
    我方多余（应恒为 0，出现即泄漏）。
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAYGEN_JAR = os.path.join(ROOT, "MAYGEN", "MAYGEN-1.8.jar")
MAYGEN_TIMEOUT = 900  # 秒；大公式可调大


# ── 官能团 SMARTS（缺失结构分类用，类别允许重叠）──
SMARTS = {
    "酯键 -C-O-C(=O)-": "[CX3](=O)-[OX2H0]",
    "羧酸 -C(=O)OH": "[CX3](=O)-[OX2H1]",
    "羰基 C=O": "[CX3]=[OX1]",
    "醚键 C-O-C": "[OX2H0]([#6])[#6]",
    "含芳环": "a",
}


def canonical_smiles_set_from_graphs(graphs):
    """本程序图集合 → RDKit 规范 SMILES 集合"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from utils import graph_to_rdkit_mol

    smiles_set = set()
    invalid = 0
    for G in graphs:
        mol = graph_to_rdkit_mol(G)
        if mol is None:
            invalid += 1
            continue
        try:
            smiles_set.add(Chem.MolToSmiles(mol))
        except Exception:
            invalid += 1
    return smiles_set, invalid


def generate_ours(formula):
    """GUI 同款入口：解析 → 生成 → 展平 → 图列表"""
    from utils import parse_compound_formula
    from oxygen.unified_oxo_generator import UnifiedOxoGenerator
    from utils import GeneratorManager

    res = parse_compound_formula(formula)
    _, nc, nh, no, halo, _, err = res
    if err:
        print(f"  [错误] 公式解析失败: {err}")
        return []
    h_equiv = nh + sum(halo)
    k = (2 * nc + 2 - h_equiv) // 2 if h_equiv else 3
    k = max(k, 0)

    graphs = []
    if no > 0:
        uog = UnifiedOxoGenerator()
        buckets = uog.generate(nc, no, k_range=(k, k), constraints=None)
        for items in buckets.values():
            for _, G in items:
                graphs.append(G)
    else:
        mgr = GeneratorManager()
        raw = mgr.generate_all(nc, nh)
        from utils import canon_str_to_graph
        for mt, iso in raw:
            if hasattr(iso, 'nodes'):
                graphs.append(iso)
            else:
                g = canon_str_to_graph(iso, mt, mgr)
                if g is not None:
                    graphs.append(g)
    return graphs


def run_maygen(formula, workdir):
    """外部调用 MAYGEN JAR，返回规范 SMILES 集合与原始数量"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    if not os.path.exists(MAYGEN_JAR):
        print(f"  [错误] 未找到 {MAYGEN_JAR}")
        return set(), 0, "缺少 JAR"

    cmd = ["java", "-jar", MAYGEN_JAR, "-f", formula,
           "-smi", "-t", "-o", workdir]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=MAYGEN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return set(), 0, f"超时(>{MAYGEN_TIMEOUT}s)"

    # 从 stdout 的 tsv 行解析数量： "formula,count,time"（容错多种分隔）
    count = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("formula"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) >= 2 and any(c.isdigit() for c in parts[1]):
            try:
                count = int(float(parts[1]))
            except ValueError:
                pass
            break

    smi_files = glob.glob(os.path.join(workdir, "*.smi"))
    smiles_set = set()
    invalid = 0
    for f in smi_files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                mol = Chem.MolFromSmiles(line)
                if mol is None:
                    invalid += 1
                    continue
                try:
                    smiles_set.add(Chem.MolToSmiles(mol))
                except Exception:
                    invalid += 1

    status = "OK"
    if proc.returncode != 0:
        status = f"MAYGEN 退出码 {proc.returncode}: {proc.stderr.strip()[:200]}"
    return smiles_set, (count if count is not None else len(smiles_set)), status


def categorize(smiles_list):
    """按官能团 SMARTS 统计缺失结构分类"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    patts = {name: Chem.MolFromSmarts(smarts)
             for name, smarts in SMARTS.items()}
    stats = {name: 0 for name in SMARTS}
    tagged = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        tags = [name for name, p in patts.items()
                if p is not None and mol.HasSubstructMatch(p)]
        for t in tags:
            stats[t] += 1
        tagged.append((s, tags))
    return stats, tagged


def validate(formula, top_missing=10):
    print("=" * 72)
    print(f"公式: {formula}")
    print("=" * 72)

    print("[1/3] 本程序生成...")
    ours_graphs = generate_ours(formula)
    ours_set, ours_invalid = canonical_smiles_set_from_graphs(ours_graphs)
    print(f"  本程序: {len(ours_graphs)} 个图, 规范SMILES {len(ours_set)} 个"
          f"（RDKit 转换失败 {ours_invalid}）")

    print("[2/3] MAYGEN 外部生成...")
    with tempfile.TemporaryDirectory(prefix="maygen_val_") as workdir:
        maygen_set, maygen_count, status = run_maygen(formula, workdir)
        print(f"  MAYGEN: 原始计数={maygen_count}, 规范SMILES {len(maygen_set)} 个"
              f" | {status}")

    print("[3/3] 对比...")
    missing = sorted(maygen_set - ours_set)
    extra = sorted(ours_set - maygen_set)
    union = maygen_set | ours_set
    coverage = (len(ours_set & maygen_set) / len(maygen_set)
                if maygen_set else 0.0)
    print(f"  交集: {len(ours_set & maygen_set)}")
    print(f"  覆盖率(交集/MAYGEN): {coverage*100:.2f}%")
    print(f"  我方缺失(MAYGEN−我们): {len(missing)}")
    print(f"  我方多余(我们−MAYGEN): {len(extra)}  {'⚠ 出现多余=泄漏!' if extra else '（0 ✓）'}")

    if missing:
        stats, tagged = categorize(missing)
        print("  缺失结构按官能团分类（可重叠）:")
        for name, cnt in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"    {name}: {cnt}")
        print(f"  缺失示例（前 {min(top_missing, len(missing))} 条）:")
        for s, tags in tagged[:top_missing]:
            print(f"    {s}   [{', '.join(tags) or '其他'}]")

    if extra:
        print("  ⚠ 我方存在但 MAYGEN 未生成的 SMILES（前 10 条）:")
        for s in extra[:10]:
            print(f"    {s}")

    print()
    return {'formula': formula, 'ours': len(ours_set), 'maygen': maygen_count,
            'intersection': len(ours_set & maygen_set),
            'missing': len(missing), 'extra': len(extra)}


def main():
    ap = argparse.ArgumentParser(description="MAYGEN 外部交叉验证工具")
    ap.add_argument("formulas", nargs="+", help="分子式，如 C7H6O2")
    ap.add_argument("--top-missing", type=int, default=10,
                    help="缺失示例展示条数")
    args = ap.parse_args()

    summary = []
    for f in args.formulas:
        summary.append(validate(f, top_missing=args.top_missing))

    print("=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"{'公式':10s} {'我方':>8s} {'MAYGEN':>8s} {'交集':>8s} "
          f"{'缺失':>6s} {'多余':>6s} {'覆盖率':>8s}")
    for r in summary:
        cov = (r['intersection'] / r['maygen'] * 100) if r['maygen'] else 0
        print(f"{r['formula']:10s} {r['ours']:8d} {r['maygen']:8d} "
              f"{r['intersection']:8d} {r['missing']:6d} {r['extra']:6d} "
              f"{cov:7.2f}%")


if __name__ == "__main__":
    main()
