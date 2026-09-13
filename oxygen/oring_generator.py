"""
O在环生成器 (O-in-Ring Generator)

将碳骨架上的 C 原子替换为 O 原子（degree ≤ 2），生成含 O 环结构：
  - 环丙烷 C₃H₆ → C₂H₄O (环氧乙烷)
  - 环丁烷 C₄H₈ → C₃H₆O (氧杂环丁烷)
  - 环戊烷 C₅H₁₀ → C₄H₈O (THF)
  - 环己烷 C₆H₁₂ → C₅H₁₀O (四氢吡喃)

约束：O 的 degree ≤ 2（桥头碳>2 自动过滤）
"""
import itertools
import networkx as nx
from typing import List, Tuple, Dict

from utils import _dedup_canon_key, _dedup_degree_signature, dedup_add_to_buckets


class ORingGenerator:
    """C→O 替换生成器"""

    def __init__(self):
        pass

    def generate(self, carbon_skeletons: List[Tuple[str, nx.Graph]],
                 n_o: int) -> List[Tuple[str, nx.Graph]]:
        """在碳骨架上替换 n_o 个 C 为 O

        Args:
            carbon_skeletons: [(mol_type, nx.Graph), ...]
            n_o: 替换的 O 原子数

        Returns:
            [(mol_type, oring_graph), ...]
        """
        if n_o <= 0 or not carbon_skeletons:
            return carbon_skeletons

        dedup_buckets: Dict[str, List[Tuple[str, nx.Graph]]] = {}

        for mol_type, G in carbon_skeletons:
            # 仅替换环上的 C 原子（cycle_basis 中的节点）
            ring_atoms = self._get_ring_atoms(G)
            candidates = [n for n in G.nodes()
                          if G.degree(n) <= 2
                          and G.nodes[n].get('label', 'C') == 'C'
                          and n in ring_atoms]
            if len(candidates) <= n_o:
                # 防护：确保替换后至少保留 1 个 C 原子
                continue

            for combo in itertools.combinations(candidates, n_o):
                H = self._replace_c_with_o(G, combo)
                if H is not None:
                    self._add_to_buckets(H, mol_type, dedup_buckets)

        results = []
        for bucket in dedup_buckets.values():
            results.extend(bucket)
        return results

    def _replace_c_with_o(self, G: nx.Graph,
                           nodes: Tuple[int, ...]) -> nx.Graph:
        """将指定节点替换为 O"""
        H = nx.Graph()
        replace_set = set(nodes)

        for node in G.nodes():
            label = 'O' if node in replace_set else G.nodes[node].get('label', 'C')
            H.add_node(node, label=label)

        for u, v, data in G.edges(data=True):
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

        # 价态验证
        if not self._validate_valence(H):
            return None
        return H

    @staticmethod
    def _get_ring_atoms(G: nx.Graph) -> set:
        """返回图中属于至少一个环的节点集合"""
        try:
            cycles = nx.cycle_basis(G)
        except Exception:
            return set()
        return {node for cycle in cycles for node in cycle}

    @staticmethod
    def _validate_valence(G: nx.Graph) -> bool:
        """检查全图价态是否符合 C≤4, O≤2"""
        for node in G.nodes():
            label = G.nodes[node].get('label', 'C')
            max_v = 4 if label == 'C' else 2

            bond_load = 0
            for nb in G.neighbors(node):
                bt = G[node][nb].get('bond_type', 'single')
                if bt == 'double':
                    bond_load += 2
                elif bt == 'triple':
                    bond_load += 3
                else:
                    bond_load += 1

            if bond_load > max_v:
                return False
        return True

    # ========== 去重 ==========

    @staticmethod
    def _ring_label(G: nx.Graph, node: int) -> str:
        """节点标签编码：直接使用节点 label"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _ring_sig(G: nx.Graph, node: int) -> str:
        """节点特征：用于度签名预过滤"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        return _dedup_canon_key(G, ORingGenerator._ring_label)

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        return _dedup_degree_signature(G, ORingGenerator._ring_sig)

    @staticmethod
    def _add_to_buckets(G: nx.Graph, mol_type: str,
                         buckets: Dict[str, List[Tuple[str, nx.Graph]]]):
        def node_match(n1, n2):
            return n1.get('label') == n2.get('label')

        dedup_add_to_buckets(
            G, mol_type, buckets, node_match,
            ORingGenerator._ring_label,
            ORingGenerator._ring_sig,
            ORingGenerator._canon_key,
            ORingGenerator._degree_signature,
        )


# ==================== 测试 ====================

if __name__ == "__main__":
    from utils import GeneratorManager, canon_str_to_graph
    import re

    mgr = GeneratorManager()
    gen = ORingGenerator()

    tests = [
        ("C3H6", 1, "环丙烷 → 环氧乙烷"),
        ("C4H8", 1, "环丁烷 → 氧杂环丁烷"),
        ("C5H10", 1, "环戊烷 → THF"),
        ("C6H12", 1, "环己烷 → 四氢吡喃"),
        ("C6H12", 2, "环己烷 → 二氧六环"),
    ]

    for formula, n_o, desc in tests:
        match = re.match(r'^C(\d+)H(\d+)$', formula)
        nc = int(match.group(1)); nh = int(match.group(2))
        raw = mgr.generate_all(nc, nh)
        skels = [(mt, canon_str_to_graph(iso, mt, mgr)) if isinstance(iso, str)
                 else (mt, iso) for mt, iso in raw]

        r = gen.generate(skels, n_o)
        print(f"\n{desc}: {len(r)} structures")
        for i, (mt, G) in enumerate(r[:4]):
            nc2 = sum(1 for nd in G.nodes() if G.nodes[nd].get('label','C')=='C')
            no2 = sum(1 for nd in G.nodes() if G.nodes[nd].get('label')=='O')
            print(f"  #{i+1}: C={nc2}, O={no2} [{mt}]")
