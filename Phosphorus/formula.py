"""Phosphorus 分子式事实源（第一期：仅 C / H / P）

单一事实源原则（对齐 O_project_guide.md 的 I1 不变量）：
所有"分子式 ↔ 图"的换算都必须经过本模块，禁止在其它文件中另写一套。

价态口径：
- C = 4（八隅律）
- P = 3（与 MAYGEN 1.8 默认一致；5 价见 ``P_VALENCE_OPTIONS``，仅预留）

DBE（不饱和度）——P 为三价，与 N 同理：
    DBE = (2*nC + 2 + nP - nH) / 2
    （一期无 O / 无卤素 / 无 N；nH 与 nP 共同参与）
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import networkx as nx

#: 元素价态（第一期）
VALENCE: Dict[str, int] = {'C': 4, 'P': 3}

#: P 价态选项：一期只开放 3；5 为**预留**（默认不启用、无锚点保证）
P_VALENCE_OPTIONS: Tuple[int, ...] = (3, 5)

#: 本模块支持的原子标签（H 只在分子式中出现，不会成为图节点）
SUPPORTED_LABELS = frozenset({'C', 'H', 'P'})

_TOKEN_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


class FormulaError(ValueError):
    """分子式解析/构造错误"""


def parse_formula_p(text: str) -> Tuple[int, int, int]:
    """解析含磷分子式 → ``(nC, nH, nP)``

    Args:
        text: 如 ``'C3H9P'`` / ``'CH5P'`` / ``'H3P'``（元素顺序任意）

    Returns:
        ``(nC, nH, nP)``

    Raises:
        FormulaError: 无法解析 / 含不支持的其它元素（O、N、卤素等）
    """
    if not isinstance(text, str):
        raise FormulaError(f"分子式必须是字符串，收到 {type(text).__name__}")
    s = text.strip()
    if not s:
        raise FormulaError("分子式为空")

    counts: Dict[str, int] = {}
    pos = 0
    for m in _TOKEN_RE.finditer(s):
        if m.start() != pos:
            raise FormulaError(f"无法解析的字符：{s[pos:m.start()]!r}")
        sym, num = m.group(1), m.group(2)
        if num == '':
            cnt = 1
        else:
            cnt = int(num)
            if cnt <= 0:
                raise FormulaError(f"元素计数必须 ≥1：{sym}{num}")
        counts[sym] = counts.get(sym, 0) + cnt
        pos = m.end()
    if pos != len(s):
        raise FormulaError(f"无法解析的字符：{s[pos:]!r}")

    unknown = set(counts) - SUPPORTED_LABELS
    if unknown:
        raise FormulaError(
            f"第一期仅支持 C / H / P，收到：{', '.join(sorted(unknown))}"
            "（含 O/N/卤素请用相应模块）")
    if counts.get('P', 0) < 1:
        raise FormulaError("含磷模块要求分子式中至少有一个 P")

    return counts.get('C', 0), counts.get('H', 0), counts.get('P', 0)


def dbe(n_carbon: int, n_hydrogen: int, n_phosphorus: int) -> Optional[int]:
    """不饱和度；不合法（负数/半整数）返回 ``None``"""
    num = 2 * n_carbon + 2 + n_phosphorus - n_hydrogen
    if num < 0 or num % 2 != 0:
        return None
    return num // 2


def _atom_hydrogen(degree: int, label: str, valence_p: int = 3) -> int:
    """单原子氢数 = 价态 − 键级和"""
    base = VALENCE.get('P', 3) if label == 'P' else VALENCE.get(label, 0)
    if label == 'P':
        base = valence_p
    return base - degree


def graph_formula_parts_p(G: nx.Graph,
                          valence_p: int = 3) -> Tuple[int, int, int]:
    """图 → ``(nC, nH, nP)``（含氢数由价态 − 键级和推出）

    Raises:
        FormulaError: 节点标签不支持 / 某原子超价（键级和 > 价态）
    """
    n_c = n_p = n_h = 0
    for node, data in G.nodes(data=True):
        label = data.get('label', 'C')
        if label not in SUPPORTED_LABELS:
            raise FormulaError(f"不支持的原子标签：{label!r}")
        used = 0
        for nb in G.neighbors(node):
            bt = int(G[node][nb].get('bond_type', 1))
            if bt not in (1, 2, 3):
                raise FormulaError(f"非法键级 {bt}（{node}-{nb}）")
            used += bt
        h = _atom_hydrogen(used, label, valence_p)
        if h < 0:
            raise FormulaError(
                f"原子 {node}({label}) 超价：键级和 {used} > 价态 "
                f"{valence_p if label == 'P' else VALENCE[label]}")
        if label == 'C':
            n_c += 1
        else:
            n_p += 1
        n_h += h
    return n_c, n_h, n_p


def formula_string(n_carbon: int, n_hydrogen: int,
                   n_phosphorus: int) -> str:
    """``(nC, nH, nP)`` → 分子式字符串（1 省略计数，与项目其它模块一致）"""
    def part(sym: str, n: int) -> str:
        return '' if n == 0 else (sym if n == 1 else f'{sym}{n}')
    return part('C', n_carbon) + part('H', n_hydrogen) + part('P', n_phosphorus)


def format_graph_formula_p(G: nx.Graph, valence_p: int = 3) -> str:
    """图 → 分子式字符串（唯一事实源）"""
    return formula_string(*graph_formula_parts_p(G, valence_p))


def formula_matches_p(formula: str, G: nx.Graph,
                      valence_p: int = 3) -> bool:
    """图是否与给定分子式严格一致（公式门禁）"""
    want = parse_formula_p(formula)
    try:
        return graph_formula_parts_p(G, valence_p) == want
    except FormulaError:
        return False
