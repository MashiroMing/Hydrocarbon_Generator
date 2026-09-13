"""
Nitrogen/formula.py — N 系分子式与图公式工具（单一事实源）

口径（与 MAYGEN 默认价态表、O 系 math 口径三方对齐）：
  - 中性八隅律：C = 4 价，N = 3 价；
  - 分子式只含 C/H/N（第一阶段纯 C/H/N；N+O 组合留待后续阶段）；
  - 氮规则 DBE：k = nC - nH/2 + nN/2 + 1   ⟺   nH = 2*nC + nN + 2 - 2*k。

图数据模型（与 PROJECT_GUIDE 3.1 一致，仅原子类型扩展）：
  - 节点 label: 'C' / 'N'（N 即骨架原子，亦承载取代基/桥/环内角色）；
  - 边 bond_type: 'single' / 'double' / 'triple'；
  - 每个节点 H 数 = max(0, max_v - bond_load)，总 H = Σ。
"""

import re
import networkx as nx

# 原子价态（中性八隅律）
VALENCE = {'C': 4, 'N': 3}

_FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def parse_formula(formula_str: str):
    """解析 C/H/N 分子式 → (n_carbon, n_hydrogen, n_nitrogen, error)

    支持 'C3H9N'、'C2H7N'、'C6H6'、'N2H4'、'CH5N' 等：
      - C/H/N 三元素任意顺序、可省略任一种（缺省计数为 1，如 'CN' = C1N1）；
      - 只接受 C/H/N，其他元素（O/卤素/D 等）报错（第一阶段纯 C/H/N 边界）。

    Returns:
        (n_carbon, n_hydrogen, n_nitrogen, error)
        error 为 None 表示解析成功。
    """
    if not formula_str:
        return 0, 0, 0, "空分子式"
    counts = {}
    pos = 0
    for m in _FORMULA_RE.finditer(formula_str):
        if m.start() != pos:
            return 0, 0, 0, f"无法解析的字符: {formula_str[pos:m.start()]!r}"
        sym, num = m.group(1), m.group(2)
        n = int(num) if num else 1
        if n == 0:
            return 0, 0, 0, f"元素 {sym} 计数为 0"
        counts[sym] = counts.get(sym, 0) + n
        pos = m.end()
    if pos != len(formula_str):
        return 0, 0, 0, f"无法解析的字符: {formula_str[pos:]!r}"

    unknown = set(counts) - {'C', 'H', 'N'}
    if unknown:
        return 0, 0, 0, f"第一阶段仅支持 C/H/N，发现: {sorted(unknown)}"

    n_c = counts.get('C', 0)
    n_h = counts.get('H', 0)
    n_n = counts.get('N', 0)

    # 合法性：价态和必须为偶数（所有键成对），且满足连通图下界
    total_valence = 4 * n_c + 3 * n_n
    if (total_valence - n_h) % 2 != 0:
        return 0, 0, 0, f"{formula_str}: 价态和与 H 数不匹配（Σv - nH 必须为偶数）"
    n_heavy = n_c + n_n
    if n_heavy > 0 and total_valence < 2 * (n_heavy - 1) + n_h:
        return 0, 0, 0, f"{formula_str}: 无法构成连通重原子骨架（价态不足）"
    return n_c, n_h, n_n, None


def dbe_from_parts(n_c: int, n_h: int, n_n: int) -> float:
    """氮规则不饱和度（DBE）：k = nC - nH/2 + nN/2 + 1"""
    return n_c - n_h / 2 + n_n / 2 + 1


def expected_h_for_dbe(n_c: int, n_n: int, k: int) -> int:
    """给定碳数/氮数与目标 DBE，反推期望 H 数：nH = 2*nC + nN + 2 - 2*k"""
    return 2 * n_c + n_n + 2 - 2 * k


def _bond_load(G: nx.Graph, node) -> int:
    """节点键负载：单键 1 / 双键 2 / 三键 3 之和"""
    used = 0
    for nb in G.neighbors(node):
        bt = G[node][nb].get('bond_type', 'single')
        used += 3 if bt == 'triple' else (2 if bt == 'double' else 1)
    return used


def graph_formula_parts(G: nx.Graph):
    """从图统计真实分子式组成 (nC, nH, nN)

    口径：h = max(0, VALENCE[label] - bond_load)，总 H = Σh。
    与预言机 / 树机制 / 门禁共用此函数，保证全链路一致（不变量 I1 的 N 系对应）。
    """
    n_c = 0
    n_n = 0
    total_h = 0
    for _, d in G.nodes(data=True):
        label = d.get('label', 'C')
        if label == 'C':
            n_c += 1
        elif label == 'N':
            n_n += 1
        max_v = VALENCE.get(label)
        if max_v is None:
            raise ValueError(f"未知原子类型: {label!r}")
        total_h += max(0, max_v - _bond_load(G, _))
    return n_c, total_h, n_n


def format_graph_formula(G: nx.Graph) -> str:
    """从图生成标准分子式字符串（C→H→N 顺序，nC=1 时省略计数）"""
    n_c, n_h, n_n = graph_formula_parts(G)
    base = f"C{n_c}" if n_c > 1 else "C"
    base += f"H{n_h}" if n_h != 1 else "H"
    if n_n > 0:
        base += f"N{n_n}" if n_n > 1 else "N"
    return base


def expected_formula_parts(n_carbon: int, n_hydrogen: int, n_nitrogen: int):
    """用户输入对应的期望分子式组成 (nC, nH, nN)"""
    return n_carbon, n_hydrogen, n_nitrogen


def formula_matches_input(G: nx.Graph, n_carbon: int, n_hydrogen: int,
                          n_nitrogen: int) -> bool:
    """分子式门禁：图 G 的真实分子式是否与输入完全一致"""
    n_c, n_h, n_n = graph_formula_parts(G)
    return (n_c == n_carbon and n_h == n_hydrogen and n_n == n_nitrogen)


def dbe_from_graph(G: nx.Graph) -> float:
    """从图直接计算 DBE（氮规则）"""
    n_c, n_h, n_n = graph_formula_parts(G)
    return dbe_from_parts(n_c, n_h, n_n)


def graph_valences(G: nx.Graph) -> int:
    """图总价态（重原子）"""
    total = 0
    for _, d in G.nodes(data=True):
        total += VALENCE.get(d.get('label', 'C'), 0)
    return total


def is_valence_legal(G: nx.Graph) -> bool:
    """价态合法性检查（C ≤ 4，N ≤ 3；不含 H 节点）"""
    for _, d in G.nodes(data=True):
        label = d.get('label', 'C')
        max_v = VALENCE.get(label)
        if max_v is None:
            return False
        if _bond_load(G, _) > max_v:
            return False
    return True
