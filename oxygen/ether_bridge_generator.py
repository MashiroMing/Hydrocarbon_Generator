"""
O-桥生成器 (Ether & Peroxy & Dioxy Bridge Generator)

在碳骨架上枚举 C-C 键 → O 桥:
  - 醚桥 (ether):     C-C → C-O-C          (1 O，仅单键)
  - 过氧桥 (peroxy):  C-C → C-O-O-C        (2 O，单键)
                     C=C → C-C + O-O       (2 O，双键降单 → 1,2-二氧杂环丁烷等)
  - 双醚桥 (dioxy):   C-C → C-O-C-O 平行双 O (2 O，4 元 1,3-二氧杂环，k+1)

O 原子作为骨架节点（label='O'），与两侧 C 单键连接。
可结合 OxoSubstituentGenerator 在同一骨架上添加 OH/=O。
"""
import itertools
import networkx as nx
from typing import List, Tuple, Dict, Set

from utils import _dedup_canon_key, _dedup_degree_signature, dedup_add_to_buckets


class BridgeGenerator:
    """O-桥插入生成器（支持醚桥 / 过氧桥 / 双醚桥）"""

    def __init__(self):
        pass

    def generate(self, carbon_skeletons: List[Tuple[str, nx.Graph]],
                 n_o: int, bridge_type: str = 'ether') -> List[Tuple[str, nx.Graph]]:
        """在碳骨架上插入 O 桥

        Args:
            carbon_skeletons: [(mol_type, nx.Graph), ...]
            n_o: 用于桥接的总 O 原子数
            bridge_type: 'ether' (1 O/桥) | 'peroxy' (2 O/桥) | 'dioxy' (2 O/桥)

        Returns:
            [(mol_type, bridged_graph), ...]
        """
        o_per_bridge = 2 if bridge_type in ('peroxy', 'dioxy') else 1
        max_bridges = n_o // o_per_bridge
        if max_bridges <= 0 or not carbon_skeletons:
            return carbon_skeletons

        dedup_buckets: Dict[str, List[Tuple[str, nx.Graph]]] = {}

        for mol_type, G in carbon_skeletons:
            cc_single = [
                (u, v, False) for u, v, d in G.edges(data=True)
                if d.get('bond_type', 'single') == 'single'
                and G.nodes[u].get('label', 'C') == 'C'
                and G.nodes[v].get('label', 'C') == 'C'
            ]
            bonds = list(cc_single)
            if bridge_type == 'peroxy':
                # 过氧桥额外支持 C=C 双键：C=C → C-C + O-O（1,2-二氧杂环丁烷等）
                cc_double = [
                    (u, v, True) for u, v, d in G.edges(data=True)
                    if d.get('bond_type', 'single') == 'double'
                    and G.nodes[u].get('label', 'C') == 'C'
                    and G.nodes[v].get('label', 'C') == 'C'
                ]
                bonds = cc_single + cc_double
            elif bridge_type == 'dioxy':
                # 双醚桥：两端 C 需各有 ≥1 个可用 H（净 +1 键，k+1）
                def _ok(u, v):
                    return (self._bond_load(G, u) <= 3
                            and self._bond_load(G, v) <= 3)
                bonds = [(u, v, False) for u, v, _ in cc_single if _ok(u, v)]
            if not bonds:
                continue

            for r in range(1, min(max_bridges, len(bonds)) + 1):
                for combo in itertools.combinations(bonds, r):
                    H = self._insert_bridges(G, combo, o_per_bridge,
                                             dioxy=(bridge_type == 'dioxy'))
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

    def _insert_bridges(self, G: nx.Graph,
                        bonds: Tuple[Tuple[int, int, bool], ...],
                        o_count: int, dioxy: bool = False) -> nx.Graph:
        """在 G 上插入 O 桥。o_count=1 为醚桥，o_count=2 为过氧桥

        bonds 元素: (u, v, was_double) —— was_double=True 表示原为 C=C 双键，
        桥接时双键降为 C-C 单键（仅过氧桥支持，如 1,2-二氧杂环丁烷）。

        dioxy=True：双醚桥 —— 用 2 个平行 O 桥替换 C-C 键（4 元 C-O-C-O 环），
        两端 C 各 -1 H（k+1），如 1,3-二氧杂环丁烷。
        """
        H = nx.Graph()
        for node in G.nodes():
            H.add_node(node, label=G.nodes[node].get('label', 'C'))

        bridge_set = set()
        for u, v, _ in bonds:
            bridge_set.add((min(u, v), max(u, v)))

        for u, v, data in G.edges(data=True):
            key = (min(u, v), max(u, v))
            if key in bridge_set:
                continue
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

        next_idx = G.number_of_nodes()
        for u, v, was_double in bonds:
            if was_double:
                # C=C → C-C（双键降单；π 键消耗 + 环贡献 = k 不变）
                H.add_edge(u, v, bond_type='single')
            if dioxy:
                # 双醚桥：2 个平行 O 桥（C-O-C-O 4 元环），不保留 C-C 键
                o1 = next_idx
                o2 = next_idx + 1
                next_idx += 2
                H.add_node(o1, label='O')
                H.add_node(o2, label='O')
                H.add_edge(u, o1, bond_type='single')
                H.add_edge(o1, v, bond_type='single')
                H.add_edge(u, o2, bond_type='single')
                H.add_edge(o2, v, bond_type='single')
                continue
            prev = u
            for oi in range(o_count):
                H.add_node(next_idx, label='O')
                H.add_edge(prev, next_idx, bond_type='single')
                prev = next_idx
                next_idx += 1
            H.add_edge(prev, v, bond_type='single')

        return H

    # ==================== 去重 ====================

    @staticmethod
    def _bridge_label(G: nx.Graph, node: int) -> str:
        """节点标签编码：直接使用节点 label"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _bridge_sig(G: nx.Graph, node: int) -> str:
        """节点特征：用于度签名预过滤"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        return _dedup_canon_key(G, BridgeGenerator._bridge_label)

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        return _dedup_degree_signature(G, BridgeGenerator._bridge_sig)

    @staticmethod
    def _add_to_buckets(G: nx.Graph, mol_type: str,
                         buckets: Dict[str, List[Tuple[str, nx.Graph]]]):
        def node_match(n1, n2):
            return n1.get('label') == n2.get('label')

        dedup_add_to_buckets(
            G, mol_type, buckets, node_match,
            BridgeGenerator._bridge_label,
            BridgeGenerator._bridge_sig,
            BridgeGenerator._canon_key,
            BridgeGenerator._degree_signature,
        )


# ==================== 测试 ====================

if __name__ == "__main__":
    from utils import GeneratorManager, canon_str_to_graph
    import re

    mgr = GeneratorManager()
    gen = BridgeGenerator()

    tests = [
        ("C2H6", 2, 'ether', "C2H6 + 2O 双醚桥"),
        ("C2H6", 2, 'peroxy', "C2H6 + 2O 过氧桥 → CH3-O-O-CH3"),
        ("C3H8", 2, 'peroxy', "C3H8 + 2O 过氧桥"),
        ("C4H10", 2, 'peroxy', "C4H10 + 2O 过氧桥"),
    ]

    for formula, n_o, btype, desc in tests:
        match = re.match(r'^C(\d+)H(\d+)$', formula)
        nc = int(match.group(1)); nh = int(match.group(2))
        raw = mgr.generate_all(nc, nh)
        skels = [(mt, canon_str_to_graph(iso, mt, mgr)) if isinstance(iso, str)
                 else (mt, iso) for mt, iso in raw]

        r = gen.generate(skels, n_o, bridge_type=btype)
        n_cs = sum(1 for nd in r[0][1].nodes() if r[0][1].nodes[nd].get('label','C')=='C') if r else 0
        n_os = sum(1 for nd in r[0][1].nodes() if r[0][1].nodes[nd].get('label')=='O') if r else 0
        print(f"\n{desc}: {len(r)} structures (典型: C={n_cs}, O={n_os})")
