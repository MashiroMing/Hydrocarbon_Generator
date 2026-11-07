"""
Nitrogen/formula.py — N 系分子式与图公式工具（单一事实源）

口径（与 MAYGEN 默认价态表、O 系 math 口径三方对齐）：
  - 中性八隅律：C = 4 价，N = 3 价，O = 2 价；
  - 分子式支持 C/H/N/O（**路线 A，2026-09-21 起**；此前第一阶段只允许纯 C/H/N）；
    ⚠ 中性八隅律下 **N=3**，因此 **硝基 -NO₂ 不在枚举空间内**（它需要 5 价 N 或
    电荷分离式 N⁺-O⁻）。可枚举的是酰胺 / 肟 / 亚硝基 / 羟胺 / 氨基醇酸酯等。
  - 氮规则 DBE：k = nC - nH/2 + nN/2 + 1   ⟺   nH = 2*nC + nN + 2 - 2*k
    （**O 不进入 DBE**）。

图数据模型（与 PROJECT_GUIDE 3.1 一致，仅原子类型扩展）：
  - 节点 label: 'C' / 'N' / 'O'（均为重原子节点，O 不是 oxo_counts 那种取代基计数）；
  - 边 bond_type: 'single' / 'double' / 'triple'；
  - 每个节点 H 数 = max(0, max_v - bond_load)，总 H = Σ。
"""

import re
import networkx as nx

# 原子价态（中性八隅律）
VALENCE = {'C': 4, 'N': 3, 'O': 2}

_FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def atom_valence(label: str, data=None) -> int:
    """原子的**允许键数**：基准价 + 形式电荷（**带符号**；``valence_override`` 优先）

    与 ``utils.atom_valence`` 同一口径（单一事实源口径；改一处必须同步）。
    硝基电荷分离式 ``N⁺`` 允许 4 键、``O⁻`` 只允许 1 键（不补 H）即由此规则给出。
    """
    data = data or {}
    ov = data.get('valence_override')
    if ov:
        return int(ov)
    base = VALENCE.get(label, 4)
    q = int(data.get('formal_charge', 0) or 0)
    return max(0, base + q)


def _parse_parts(formula_str: str):
    """解析 C/H/N/O 分子式 → (n_carbon, n_hydrogen, n_nitrogen, n_oxygen, error)"""
    if not formula_str:
        return 0, 0, 0, 0, "空分子式"
    counts = {}
    pos = 0
    for m in _FORMULA_RE.finditer(formula_str):
        if m.start() != pos:
            return 0, 0, 0, 0, f"无法解析的字符: {formula_str[pos:m.start()]!r}"
        sym, num = m.group(1), m.group(2)
        n = int(num) if num else 1
        if n == 0:
            return 0, 0, 0, 0, f"元素 {sym} 计数为 0"
        counts[sym] = counts.get(sym, 0) + n
        pos = m.end()
    if pos != len(formula_str):
        return 0, 0, 0, 0, f"无法解析的字符: {formula_str[pos:]!r}"

    unknown = set(counts) - set(VALENCE) - {'H'}
    if unknown:
        return 0, 0, 0, 0, f"仅支持 C/H/N/O，发现: {sorted(unknown)}"

    n_c = counts.get('C', 0)
    n_h = counts.get('H', 0)
    n_n = counts.get('N', 0)
    n_o = counts.get('O', 0)

    # 合法性：价态和与 H 数同奇偶（所有键成对），且满足连通图价态下界
    total_valence = sum(VALENCE[e] * c for e, c in
                        (('C', n_c), ('N', n_n), ('O', n_o)))
    if (total_valence - n_h) % 2 != 0:
        return 0, 0, 0, 0, f"{formula_str}: 价态和与 H 数不匹配（Σv - nH 必须为偶数）"
    n_heavy = n_c + n_n + n_o
    if n_heavy > 0 and total_valence < 2 * (n_heavy - 1) + n_h:
        return 0, 0, 0, 0, f"{formula_str}: 无法构成连通重原子骨架（价态不足）"
    return n_c, n_h, n_n, n_o, None


def parse_formula_full(formula_str: str):
    """解析 C/H/N/O 分子式 → (n_carbon, n_hydrogen, n_nitrogen, n_oxygen, error)

    支持 'C2H5NO'、'C6H5NO2'、'C2H7N'、'C6H6' 等：C/H/N/O 任意顺序、可省略任一种。
    """
    return _parse_parts(formula_str)


def parse_formula(formula_str: str):
    """【兼容入口】解析纯 C/H/N 分子式 → (n_carbon, n_hydrogen, n_nitrogen, error)

    支持 'C3H9N'、'C2H7N'、'C6H6'、'N2H4'、'CH5N' 等。
    含 O 的分子式请改用 :func:`parse_formula_full`（本函数保持旧 4 元组契约，
    遇到 O 会返回明确错误而不是静默忽略）。
    """
    n_c, n_h, n_n, n_o, err = _parse_parts(formula_str)
    if err:
        return 0, 0, 0, err
    if n_o > 0:
        return 0, 0, 0, (f"{formula_str}: 含氧含氮分子式请使用 parse_formula_full "
                         f"（parse_formula 只服务纯 C/H/N 工具链）")
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
    """从图统计真实分子式组成 **(nC, nH, nN, nO)**

    口径：h = max(0, VALENCE[label] - bond_load)，总 H = Σh。
    与预言机 / 树机制 / 门禁共用此函数，保证全链路一致（不变量 I1 的 N 系对应）。
    节点可选 ``valence_override``（特殊基团，如硝基 N=5；见 ``special_groups.py``）。

    ⚠ 返回 **4 元组**（2026-09-21 路线 A 起加入 nO）。既有调用点若按
      `parts[0..2]` 切片或索引取值不受影响；直接 `n_c, n_h, n_n = ...` 解包的
      调用点需同步改为 4 元组（本仓库仅 generator.py 一处）。
    """
    n_c = 0
    n_n = 0
    n_o = 0
    total_h = 0
    for _, d in G.nodes(data=True):
        label = d.get('label', 'C')
        if label == 'C':
            n_c += 1
        elif label == 'N':
            n_n += 1
        elif label == 'O':
            n_o += 1
        max_v = atom_valence(label, d)
        if max_v is None:
            raise ValueError(f"未知原子类型: {label!r}")
        total_h += max(0, max_v - _bond_load(G, _))
    return n_c, total_h, n_n, n_o


def format_graph_formula(G: nx.Graph) -> str:
    """从图生成标准分子式字符串（C→H→N→O 顺序，nC=1 时省略计数）"""
    n_c, n_h, n_n, n_o = graph_formula_parts(G)
    base = f"C{n_c}" if n_c > 1 else "C"
    base += f"H{n_h}" if n_h != 1 else "H"
    if n_n > 0:
        base += f"N{n_n}" if n_n > 1 else "N"
    if n_o > 0:
        base += f"O{n_o}" if n_o > 1 else "O"
    return base


def expected_formula_parts(n_carbon: int, n_hydrogen: int, n_nitrogen: int,
                           n_oxygen: int = 0):
    """用户输入对应的期望分子式组成 (nC, nH, nN, nO)

    `n_oxygen` 为**末位默认参数**（默认 0）→ 既有三参调用逐位不变。
    """
    return n_carbon, n_hydrogen, n_nitrogen, n_oxygen


def formula_matches_input(G: nx.Graph, n_carbon: int, n_hydrogen: int,
                          n_nitrogen: int, n_oxygen: int = 0) -> bool:
    """分子式门禁：图 G 的真实分子式是否与输入完全一致"""
    n_c, n_h, n_n, n_o = graph_formula_parts(G)
    return (n_c == n_carbon and n_h == n_hydrogen
            and n_n == n_nitrogen and n_o == n_oxygen)


def dbe_from_graph(G: nx.Graph) -> float:
    """从图直接计算 DBE（氮规则；O 不进入 DBE）"""
    n_c, n_h, n_n, _n_o = graph_formula_parts(G)
    return dbe_from_parts(n_c, n_h, n_n)




def is_valence_legal(G: nx.Graph) -> bool:
    """价态合法性检查（C ≤ 4，N ≤ 3，O ≤ 2；不含 H 节点）

    节点带 ``valence_override`` / ``formal_charge`` 时按 :func:`atom_valence`
    修正允许键数（特殊基团用，如硝基电荷分离式的 N⁺ 允许 4 键）。
    """
    for _, d in G.nodes(data=True):
        label = d.get('label', 'C')
        max_v = atom_valence(label, d)
        if max_v is None:
            return False
        if _bond_load(G, _) > max_v:
            return False
    return True
