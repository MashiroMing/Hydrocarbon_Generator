"""
内部边界分析（独立于 MAYGEN）

原理（"树生成为主体，矩阵补边缘"的可量化版本）：
  - 树机制集合：UnifiedOxoGenerator(enable_atomic_fallback=False) 的产物
    （碳骨架 + O 修饰组合：OH/=O、醚/过氧/双醚桥、环氧、OOH、oring 等）；
  - 完备集合：AtomicMatrixGenerator.generate_for_target_h 的邻接矩阵全图枚举
    （八隅律/价态模型下数学上与 MAYGEN 等价，即完备性预言机）；
  - 边界 = 完备集合 − 树集合（树机制无法表达的分子），按官能团 SMARTS 分类；
  - 内部覆盖率 = |树 ∩ 完备| / |完备|。

因此**无需 MAYGEN / Java / 任何外部工具**，即可对任意公式量化边界情况；
MAYGEN 仅可作可选的外部交叉复核（tools/maygen_validate.py）。
"""
from typing import Dict, List, Tuple


# ── 官能团 SMARTS（边界结构分类，类别允许重叠）──
_SMARTS = {
    "酯键 -C-O-C(=O)-": "[CX3](=O)-[OX2H0]",
    "羧酸 -C(=O)OH": "[CX3](=O)-[OX2H1]",
    "羰基 C=O": "[CX3]=[OX1]",
    "醚键 C-O-C": "[OX2H0]([#6])[#6]",
    "环氧(3元CCO)": "[#6]1-[#6]-[#8]-1",
    "过氧键 O-O": "[#8]-[#8]",
    "含芳环": "a",
}


def _canonical_set(graphs) -> set:
    """图列表 → RDKit 规范 SMILES 集合"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from utils import graph_to_rdkit_mol

    out = set()
    for G in graphs:
        m = graph_to_rdkit_mol(G)
        if m is not None:
            out.add(Chem.MolToSmiles(m))
    return out


def _categorize(smiles_list: List[str]) -> Tuple[Dict[str, int], List[Tuple[str, list]]]:
    """按官能团 SMARTS 统计并标注"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    patts = {name: Chem.MolFromSmarts(smarts)
             for name, smarts in _SMARTS.items()}
    stats = {name: 0 for name in _SMARTS}
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


def internal_boundary_report(formula: str, chem_mode: str = 'math',
                             max_atoms: int = 10,
                             top_missing: int = 10) -> dict:
    """对单个分子式做内部边界分析（无 MAYGEN 依赖）

    Args:
        formula: 如 "C7H6O2"
        chem_mode: 'math'（默认，八隅律）| 'chem'（化学稳定性）
        max_atoms: 矩阵预言机的重原子数上限（超限返回 error 提示）
        top_missing: 缺失示例条数

    Returns:
        dict: formula/k/tree/matrix/intersection/missing/extra/coverage/
              categories/examples/error
    """
    from utils import parse_compound_formula
    from oxygen.unified_oxo_generator import UnifiedOxoGenerator
    from oxygen.atomic_matrix_gen import AtomicMatrixGenerator

    res = parse_compound_formula(formula)
    _, nc, nh, no, halo, _, err = res
    if err:
        return {'formula': formula, 'error': err}
    if no <= 0:
        return {'formula': formula,
                'error': '内部边界分析仅支持含氧分子式（纯烃另走 GeneratorManager）'}
    h_equiv = nh + sum(halo)
    k = (2 * nc + 2 - h_equiv) // 2 if h_equiv else 3
    k = max(k, 0)

    # 1) 树机制集合（关闭矩阵补漏，纯机制产物）
    uog = UnifiedOxoGenerator(enable_atomic_fallback=False, chem_mode=chem_mode)
    buckets = uog.generate(nc, no, k_range=(k, k), constraints=None)
    tree_graphs = [G for items in buckets.values() for _, G in items]
    tree = _canonical_set(tree_graphs)

    # 2) 完备集合（矩阵预言机：目标 H 剪枝全枚举）
    amg = AtomicMatrixGenerator(nc, no, max_atoms=max_atoms)
    matrix = _canonical_set(amg.generate_for_target_h(nh))

    missing = sorted(matrix - tree)
    extra = sorted(tree - matrix)
    inter = tree & matrix
    coverage = (len(inter) / len(matrix)) if matrix else 0.0
    stats, tagged = _categorize(missing)

    return {
        'formula': formula, 'k': k,
        'tree': len(tree), 'matrix': len(matrix),
        'intersection': len(inter),
        'missing': len(missing), 'extra': len(extra),
        'coverage': coverage,
        'categories': stats,
        'examples': tagged[:top_missing],
    }
