"""
化学过滤器 (Phase 2e)

两种口径（构造参数 chem_mode）：
  - 'math'（默认）：数学完备口径——仅保留八隅律硬规则（C 价≤4 / O 价≤2），
    移除化学稳定性启发式（偕二醇、累积烯酮等），与 MAYGEN 的数学枚举口径一致；
  - 'chem'：化学稳定性口径——启用全部规则（供未来需要稳定结构筛选的场景）。

用法:
    from oxygen.chem_filter import ChemFilter
    cf = ChemFilter(chem_mode='math')        # 数学完备（默认）
    passed = cf.filter(structure_list)       # 返回通过的结构
    valid, reason = cf.check(G)              # 返回 (bool, str)
"""

import networkx as nx
from typing import List, Tuple, Dict, Optional


# ============ 规则定义 ============

# 化学稳定性启发式规则（数学口径下禁用，化学口径下启用）
_STABILITY_RULES = ('allene_cumulene', 'gem_diol_unstable')


class ChemFilter:
    """化学过滤器：'math' 仅八隅律硬规则；'chem' 全规则"""

    def __init__(self, chem_mode: str = 'math'):
        self.rules = [
            ('allene_cumulene', self._check_allene_cumulene),  # C=C=O 积累体系
            ('gem_diol_unstable', self._check_gem_diol),        # 偕二醇（非羧基）
            ('carbon_valence', self._check_valence),            # C 价态 > 4（八隅律硬规则）
            ('oxygen_valence', self._check_o_valence),          # O 价态 > 2（八隅律硬规则）
            ('o_o_next_to_co', self._check_peroxy_carbonyl),    # O-O 邻位 C=O
        ]
        self.disabled: set = set()   # 禁用的规则名
        self.chem_mode = chem_mode
        if chem_mode == 'math':
            # 数学完备口径：仅保留价态硬规则
            self.disabled.update(_STABILITY_RULES)
        elif chem_mode != 'chem':
            raise ValueError("chem_mode 必须为 'math' 或 'chem'")

    def disable(self, rule_name: str):
        self.disabled.add(rule_name)

    def enable(self, rule_name: str):
        self.disabled.discard(rule_name)

    # ── 过滤入口 ──

    def filter(self, structures: List[Tuple[str, nx.Graph]]
               ) -> List[Tuple[str, nx.Graph]]:
        """批量过滤：仅返回通过的结构"""
        return [(mt, G) for mt, G in structures if self.check(G)[0]]

    def check(self, G: nx.Graph) -> Tuple[bool, str]:
        """检查单个图：返回 (通过, 违规规则名)"""
        for name, func in self.rules:
            if name in self.disabled:
                continue
            ok, msg = func(G)
            if not ok:
                return False, msg
        return True, ""

    # ── 规则实现 ──

    @staticmethod
    def _check_valence(G: nx.Graph) -> Tuple[bool, str]:
        """C 原子价态 ≤ 4"""
        for node in G.nodes():
            if G.nodes[node].get('label', 'C') != 'C':
                continue
            bond_load = sum(
                2 if G[node][nb].get('bond_type', 'single') == 'double' else
                3 if G[node][nb].get('bond_type', 'single') == 'triple' else 1
                for nb in G.neighbors(node)
            )
            oh, co = G.nodes[node].get('oxo_counts', (0, 0))
            hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
            total = bond_load + oh + 2 * co + sum(hc)
            if total > 4:
                return False, f"carbon_valence: C{node} has {total} bonds"
        return True, ""

    @staticmethod
    def _check_o_valence(G: nx.Graph) -> Tuple[bool, str]:
        """O 原子价态 ≤ 2"""
        for node in G.nodes():
            if G.nodes[node].get('label', 'C') != 'O':
                continue
            bond_load = sum(
                2 if G[node][nb].get('bond_type', 'single') == 'double' else
                3 if G[node][nb].get('bond_type', 'single') == 'triple' else 1
                for nb in G.neighbors(node)
            )
            oh, co = G.nodes[node].get('oxo_counts', (0, 0))
            if bond_load + oh + 2 * co > 2:
                return False, f"oxygen_valence: O{node} > 2"
        return True, ""

    @staticmethod
    def _check_allene_cumulene(G: nx.Graph) -> Tuple[bool, str]:
        """C=C=O 累积烯酮 → 排除"""
        for node in G.nodes():
            if G.nodes[node].get('label', 'C') != 'C':
                continue
            n_double = sum(1 for nb in G.neighbors(node)
                           if G[node][nb].get('bond_type') == 'double')
            if n_double < 2:
                continue
            # 检查是否有一条是 C=O
            has_co = False
            for nb in G.neighbors(node):
                if (G[node][nb].get('bond_type') == 'double'
                        and G.nodes[nb].get('label', 'C') == 'O'):
                    has_co = True
                    break
            if has_co:
                return False, f"allene_cumulene: C{node} in C=C=O"
        # Also check C=C=C=O etc through oxo_counts (=O on allene C)
        for node in G.nodes():
            if G.nodes[node].get('label', 'C') != 'C':
                continue
            n_db = sum(1 for nb in G.neighbors(node)
                       if G[node][nb].get('bond_type') == 'double')
            co = G.nodes[node].get('oxo_counts', (0, 0))[1]
            # Allene C with =O substituent → cumulene
            if n_db >= 2 and co > 0:
                return False, f"allene_cumulene: =O on allene C{node}"
        return True, ""

    @staticmethod
    def _check_gem_diol(G: nx.Graph) -> Tuple[bool, str]:
        """偕二醇：同一 C 上 ≥2 个 OH 且无 =O（非羧基）→ 排除"""
        for node in G.nodes():
            if G.nodes[node].get('label', 'C') != 'C':
                continue
            oh, co = G.nodes[node].get('oxo_counts', (0, 0))
            if oh < 2:
                continue
            # 有 =O → 这是羧基，保留
            if co > 0:
                continue
            # 非端基 C 但有 2 个 OH 且邻接 C=O → 可能是缩醛前驱体，保留
            # 但简单偕二醇不稳定 → 排除
            has_adjacent_co = False
            for nb in G.neighbors(node):
                if G.nodes[nb].get('label', 'C') != 'C':
                    continue
                nb_co = G.nodes[nb].get('oxo_counts', (0, 0))[1]
                if nb_co > 0:
                    has_adjacent_co = True
                    break
            if not has_adjacent_co:
                return False, f"gem_diol: C{node} has {oh} OH without adjacent C=O"
        return True, ""

    @staticmethod
    def _check_peroxy_carbonyl(G: nx.Graph) -> Tuple[bool, str]:
        """过氧桥邻位是 C=O 且无 H 时 → 过氧酸可能不稳定 → 警告但不排除"""
        # 宽松策略：仅排除明确过度氧化的结构
        return True, ""


# ============ 多 O 组合重排检测 ============

def detect_redundant_combinations(G: nx.Graph) -> List[str]:
    """检测图中可能因多 O 组合产生的冗余特征。

    返回警告列表（不影响过滤结果），用于调试诊断。
    """
    warnings = []
    o_count = sum(1 for n in G.nodes()
                  if G.nodes[n].get('label', 'C') == 'O')
    oh_total = sum(G.nodes[n].get('oxo_counts', (0, 0))[0]
                   for n in G.nodes())
    co_total = sum(G.nodes[n].get('oxo_counts', (0, 0))[1]
                   for n in G.nodes())

    # 邻位 OH 过多
    for u, v in G.edges():
        oh_u = G.nodes[u].get('oxo_counts', (0, 0))[0]
        oh_v = G.nodes[v].get('oxo_counts', (0, 0))[0]
        if oh_u >= 1 and oh_v >= 1:
            warnings.append(f"vicinal_diol: C{u}-C{v} both have OH")

    # O 骨架过密
    if o_count > sum(1 for n in G.nodes() if G.nodes[n].get('label', 'C') == 'C'):
        warnings.append("o_dense: more O than C in backbone")

    return warnings
