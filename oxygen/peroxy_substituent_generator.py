"""
氢过氧化物取代基生成器 (R-O-O-H) — 数学完备口径 (Phase A)

在碳骨架上为可用 H 位的 C 附加过氧羟基链：C-H → C-O-O-H

  - 数据模型零改动：O-O 链用 2 个骨架 O 节点表达（label='O'），
    终端 O 的 H 由价态自动计算（graph_formula_parts 口径）；
  - 每 1 个 OOH 消耗 2 个 O 原子；H 数净不变（C -1H，终端 O +1H）→ k 不变；
  - 与过氧桥（R-O-O-R'，替换 C-C 键）互补：本生成器是"取代基式"终端 OOH；
  - 同一 C 可承接多个 OOH（每 1 个消耗 1 个可用 H 位）。
"""
import itertools
import networkx as nx
from typing import List, Tuple, Dict

from utils import _dedup_canon_key, _dedup_degree_signature, dedup_add_to_buckets


class PeroxySubstituentGenerator:
    """氢过氧化物取代基生成器：为可用 H 位的 C 附加 R-O-O-H 链"""

    def generate(self, carbon_skeletons: List[Tuple[str, nx.Graph]],
                 n_ooh: int) -> List[Tuple[str, nx.Graph]]:
        """在碳骨架上添加 n_ooh 个 -O-O-H 取代基（每个消耗 2 个 O 原子）

        Args:
            carbon_skeletons: [(mol_type, nx.Graph), ...]
            n_ooh: OOH 取代基数 (>= 1)

        Returns:
            [(mol_type, graph_with_ooh), ...]
        """
        if n_ooh <= 0 or not carbon_skeletons:
            return carbon_skeletons

        dedup_buckets: Dict[str, List[Tuple[str, nx.Graph]]] = {}

        for mol_type, G in carbon_skeletons:
            # 可用 H ≥ 1 的 C（可承接 OOH）：bond_load ≤ 3
            candidates = [n for n, d in G.nodes(data=True)
                          if d.get('label', 'C') == 'C'
                          and self._bond_load(G, n) <= 3]
            if not candidates:
                continue

            # 同一 C 可接多个 OOH（组合允许重复，每 1 个消耗 1 个 H 位）
            for combo in itertools.combinations_with_replacement(
                    candidates, n_ooh):
                counts: Dict[int, int] = {}
                for c in combo:
                    counts[c] = counts.get(c, 0) + 1
                if any(cnt > (4 - self._bond_load(G, c))
                       for c, cnt in counts.items()):
                    continue  # H 位不足
                H = self._insert_ooh(G, counts)
                if H is not None:
                    self._add_to_buckets(H, mol_type, dedup_buckets)

        results = []
        for bucket in dedup_buckets.values():
            results.extend(bucket)
        return results

    @staticmethod
    def _bond_load(G: nx.Graph, node: int) -> int:
        """节点的键负载（单键=1, 双键=2, 三键=3）"""
        total = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            if bt == 'triple':
                total += 3
            elif bt == 'double':
                total += 2
            else:
                total += 1
        return total

    @staticmethod
    def _insert_ooh(G: nx.Graph,
                    counts: Dict[int, int]) -> nx.Graph:
        """为 counts: {C节点: OOH个数} 的每个 C 附加 O-O-H 链"""
        H = nx.Graph()
        for node in G.nodes():
            H.add_node(node, label=G.nodes[node].get('label', 'C'))
        for u, v, data in G.edges(data=True):
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

        next_idx = G.number_of_nodes()
        for c in sorted(counts):
            for _ in range(counts[c]):
                o1 = next_idx
                o2 = next_idx + 1
                next_idx += 2
                H.add_node(o1, label='O')
                H.add_node(o2, label='O')
                H.add_edge(c, o1, bond_type='single')
                H.add_edge(o1, o2, bond_type='single')
        return H

    # ==================== 去重 ====================

    @staticmethod
    def _ooh_label(G: nx.Graph, node: int) -> str:
        """节点标签编码：直接使用节点 label"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _ooh_sig(G: nx.Graph, node: int) -> str:
        """节点特征：用于度签名预过滤"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        return _dedup_canon_key(G, PeroxySubstituentGenerator._ooh_label)

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        return _dedup_degree_signature(G, PeroxySubstituentGenerator._ooh_sig)

    @staticmethod
    def _add_to_buckets(G: nx.Graph, mol_type: str,
                        buckets: Dict[str, List[Tuple[str, nx.Graph]]]):
        def node_match(n1, n2):
            return n1.get('label') == n2.get('label')

        dedup_add_to_buckets(
            G, mol_type, buckets, node_match,
            PeroxySubstituentGenerator._ooh_label,
            PeroxySubstituentGenerator._ooh_sig,
            PeroxySubstituentGenerator._canon_key,
            PeroxySubstituentGenerator._degree_signature,
        )
