"""
环氧乙烷（3 元 C-C-O 环）生成器 — O-杂环补齐 (P0)

在碳骨架的 C-C 键上"附加"O 原子形成 3 元氧杂环（oxirane / 环氧乙烷类）：

  - 单键环氧（稠合式）：保留 C-C 单键，O 同时连接两端 C
      C-C + O → C-C-O 三角环，两端 C 各 -1 H（k+1）
      例：丙烷 → 环氧丙烷（甲基环氧乙烷）
  - 双键环氧（经典烯烃环氧化）：C=C 双键降为 C-C 单键，O 桥接
      C=C + O → C-C-O 三角环，H/k 不变
      例：乙烯 → 环氧乙烷；苯（Kekulé C=C）→ 苯并环氧（benzene oxide）

与醚桥（"替换" C-C 键为 C-O-C，产生 4 元环/开链）互补：
醚桥在数学上无法产生 3 元 CCO 环，本生成器补足该类结构。
产物与 oring（C→O 原子替换）也可能重复，由上层 WL 去重统一收敛。
"""
import itertools
import networkx as nx
from typing import List, Tuple, Dict

from utils import _dedup_canon_key, _dedup_degree_signature, dedup_add_to_buckets


class EpoxideGenerator:
    """环氧乙烷生成器：在 C-C 键上附加 O 形成 3 元 C-C-O 环"""

    def generate(self, carbon_skeletons: List[Tuple[str, nx.Graph]],
                 n_o: int) -> List[Tuple[str, nx.Graph]]:
        """在碳骨架上添加 n_o 个环氧 O（每个消耗 1 个 O 原子）

        Args:
            carbon_skeletons: [(mol_type, nx.Graph), ...]
            n_o: 环氧 O 原子数 (>= 1)

        Returns:
            [(mol_type, epoxidized_graph), ...]
        """
        if n_o <= 0 or not carbon_skeletons:
            return carbon_skeletons

        dedup_buckets: Dict[str, List[Tuple[str, nx.Graph]]] = {}

        for mol_type, G in carbon_skeletons:
            # 可环氧化的 C-C 边：
            #   - single：两端 bond_load ≤ 3（附加 O 后各 C 仍 ≤ 4 价）
            #   - double：直接环氧化（双键降单 + O 桥，两端 C 价态不变）
            candidates: List[Tuple[int, int]] = []
            for u, v, d in G.edges(data=True):
                if (G.nodes[u].get('label', 'C') != 'C'
                        or G.nodes[v].get('label', 'C') != 'C'):
                    continue  # 只环氧化 C-C 键（O 节点/杂原子不参与）
                bt = d.get('bond_type', 'single')
                if bt == 'double':
                    candidates.append((u, v))
                elif bt == 'single':
                    if (self._bond_load(G, u) <= 3
                            and self._bond_load(G, v) <= 3):
                        candidates.append((u, v))
                # 三键不可环氧化

            if len(candidates) < n_o:
                continue

            for combo in itertools.combinations(candidates, n_o):
                H = self._insert_epoxides(G, combo)
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
    def _insert_epoxides(G: nx.Graph,
                         bonds: Tuple[Tuple[int, int], ...]) -> nx.Graph:
        """在 G 上对每条选中的 C-C 边附加 1 个环氧 O

        每条边 (u, v)：保留/降为 C-C 单键，新增 O 节点同时单键连接 u、v，
        形成 3 元环 u-v-O。
        """
        H = nx.Graph()
        for node in G.nodes():
            H.add_node(node, label=G.nodes[node].get('label', 'C'))

        ep_set = set()
        for u, v in bonds:
            ep_set.add((min(u, v), max(u, v)))

        for u, v, data in G.edges(data=True):
            key = (min(u, v), max(u, v))
            if key in ep_set:
                continue  # 被环氧化的边，由下方重建
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

        next_idx = G.number_of_nodes()
        for u, v in bonds:
            H.add_node(next_idx, label='O')
            H.add_edge(u, v, bond_type='single')   # 双键→单键 / 单键保留
            H.add_edge(u, next_idx, bond_type='single')
            H.add_edge(v, next_idx, bond_type='single')
            next_idx += 1

        return H

    # ==================== 去重 ====================

    @staticmethod
    def _epoxide_label(G: nx.Graph, node: int) -> str:
        """节点标签编码：直接使用节点 label"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _epoxide_sig(G: nx.Graph, node: int) -> str:
        """节点特征：用于度签名预过滤"""
        return G.nodes[node].get('label', 'C')

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        return _dedup_canon_key(G, EpoxideGenerator._epoxide_label)

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        return _dedup_degree_signature(G, EpoxideGenerator._epoxide_sig)

    @staticmethod
    def _add_to_buckets(G: nx.Graph, mol_type: str,
                        buckets: Dict[str, List[Tuple[str, nx.Graph]]]):
        def node_match(n1, n2):
            return n1.get('label') == n2.get('label')

        dedup_add_to_buckets(
            G, mol_type, buckets, node_match,
            EpoxideGenerator._epoxide_label,
            EpoxideGenerator._epoxide_sig,
            EpoxideGenerator._canon_key,
            EpoxideGenerator._degree_signature,
        )
