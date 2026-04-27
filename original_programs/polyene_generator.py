"""
================================================================================
多烯烃同分异构体统一生成器 (CnH2n-2k, k=0,1,2,...,floor(n/2))
================================================================================

【功能说明】
  - 统一生成任意碳数 n 的烷烃、单烯烃、二烯烃、三烯烃、...、N烯烃
  - 基于"烷烃骨架 → 逐层插入双键"的递推算法
  - 分子通式: CnH(2n-2k)，其中 k 为双键数

【核心算法 — 逐层递推法】
  第0层: 生成 n 个碳的烷烃骨架树（全部单键，CnH2n+2 → 实际是 CnH2n+2）
  第1层: 在烷烃骨架的每条合法边上放置1个双键 → 单烯烃 (CnH2n)
  第2层: 在单烯烃的每条合法剩余单键边上放置第2个双键 → 二烯烃 (CnH2n-2)
  第3层: 在二烯烃的每条合法剩余单键边上放置第3个双键 → 三烯烃 (CnH2n-4)
  ...
  第k层: 在(k-1)烯烃的每条合法剩余单键边上放置第k个双键 → k烯烃 (CnH2n-2k)

  【化学约束】
  - 每个双键端点碳在"当前图"中的度数 ≤ 3（sp²碳最多连3个原子）
  - 累积二烯 (C=C=C) 允许存在（同一碳可参与两个双键，但该碳度数≤2）
  - 每个碳的总价键数不超过4
  - 双键不能与已有的双键是同一条边

【递推公式】
  设 L_k 为第 k 层（k-烯烃）的唯一异构体集合，则：
    L_0 = {烷烃骨架}
    L_k = { 在 G ∈ L_{k-1} 的合法单键边上放置双键 } 去重后

【验证数据】
  单烯烃: C4=3, C5=5, C6=13, C7=27
  二烯烃: C4=2, C5=6, C6=16, C7=44
  三烯烃: C4=1, C5=2, C6=10
  四烯烃: C5=1, C6=3

================================================================================
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import networkx as nx

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 延迟导入烷烃生成器
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_alkane_generator():
    """延迟导入烷烃生成器"""
    from original_programs.alkene import AlkaneIsomerGenerator
    return AlkaneIsomerGenerator


# ==================== 多烯烃统一生成器类 ====================

class PolyeneIsomerGenerator:
    """
    多烯烃同分异构体统一生成器

    【算法原理 — 逐层递推法】
    从烷烃骨架出发，逐层插入双键：
      L_0 = 烷烃骨架（全部单键）
      L_k = 在 L_{k-1} 的每个异构体的合法单键边上插入第k个双键，去重

    【化学约束】
    - sp²碳（双键端点）: 最多连3个原子
    - 累积二烯允许存在
    - 所有碳的价键数 ≤ 4

    【分子通式】
    CnH(2n-2k)，k=0,1,2,...,floor(n/2)
    """

    # 各类型已验证的数据（独立枚举法验证）
    VERIFIED_MONOENE = {3: 1, 4: 3, 5: 5, 6: 13, 7: 27}
    VERIFIED_DIENE = {4: 2, 5: 6, 6: 16, 7: 44}
    VERIFIED_TRIENE = {4: 1, 5: 2}
    VERIFIED_TETRAENE = {4: 0, 5: 1}

    def __init__(self, alkane_generator=None):
        """
        初始化多烯烃生成器

        Args:
            alkane_generator: 烷烃生成器实例（用于生成骨架树）
        """
        self.alkane_gen = alkane_generator if alkane_generator else _get_alkane_generator()()

    # ==================== 主入口方法 ====================

    def generate_all_layers(self, n_carbons: int, max_double_bonds: int = None,
                            verbose: bool = True) -> Dict[int, List[nx.Graph]]:
        """
        生成所有层级的烯烃异构体

        Args:
            n_carbons: 碳原子数
            max_double_bonds: 最大双键数（None表示自动计算上限）
            verbose: 是否打印进度信息

        Returns:
            字典 {双键数k: 异构体列表}
        """
        if n_carbons < 2:
            return {}

        if max_double_bonds is None:
            # 理论上限：n个碳最多有n-1条边，但化学约束限制双键数
            # 每个双键至少消耗2个碳的各1个价键位，最多 floor(n/2) 个双键
            # 但考虑到度约束，实际上限可能更小
            max_double_bonds = (n_carbons - 1)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  逐层递推生成 C{n_carbons} 系列烯烃异构体")
            print(f"{'='*60}")

        # 第0层：烷烃骨架
        results = {}
        alkane_graphs = self._get_alkane_graphs(n_carbons)
        if verbose:
            print(f"  L0 烷烃骨架 (C{n_carbons}H{2*n_carbons+2}): {len(alkane_graphs)} 个")

        # 如果用户需要烷烃层，也可以提供（但通常我们只关心 k>=1）
        current_layer = alkane_graphs  # L_0

        for k in range(1, max_double_bonds + 1):
            next_layer = self._insert_double_bonds(current_layer, k)

            if not next_layer:
                # 无法再放置更多双键，提前终止
                if verbose:
                    print(f"  L{k} {k}烯烃 (C{n_carbons}H{2*n_carbons-2*k}): 0 个 (终止)")
                break

            results[k] = next_layer
            if verbose:
                formula = f"C{n_carbons}H{2*n_carbons-2*k}"
                name = self._chinese_name(k)
                # 验证
                verified = self._get_verified_count(k)
                if n_carbons in verified:
                    expected = verified[n_carbons]
                    status = "[OK]" if len(next_layer) == expected else f"[FAIL](expected {expected})"
                else:
                    status = "(未验证)"
                print(f"  L{k} {name} ({formula}): {len(next_layer)} 个 {status}")

            current_layer = next_layer

        return results

    def generate_monoene(self, n_carbons: int) -> List[nx.Graph]:
        """生成单烯烃异构体 (CnH2n)"""
        return self.generate_k_ene(n_carbons, 1)

    def generate_diene(self, n_carbons: int) -> List[nx.Graph]:
        """生成二烯烃异构体 (CnH2n-2)"""
        return self.generate_k_ene(n_carbons, 2)

    def generate_triene(self, n_carbons: int) -> List[nx.Graph]:
        """生成三烯烃异构体 (CnH2n-4)"""
        return self.generate_k_ene(n_carbons, 3)

    def generate_tetraene(self, n_carbons: int) -> List[nx.Graph]:
        """生成四烯烃异构体 (CnH2n-6)"""
        return self.generate_k_ene(n_carbons, 4)

    def generate_k_ene(self, n_carbons: int, k: int) -> List[nx.Graph]:
        """
        生成 k-烯烃异构体 (CnH2n-2k)

        Args:
            n_carbons: 碳原子数
            k: 双键数量

        Returns:
            唯一异构体图列表
        """
        if n_carbons < k + 1:
            return []
        if k < 1:
            return []

        # 从烷烃骨架开始，逐层递推到第k层
        current_layer = self._get_alkane_graphs(n_carbons)

        for layer_idx in range(1, k + 1):
            current_layer = self._insert_double_bonds(current_layer, layer_idx)
            if not current_layer:
                return []

        return current_layer

    # ==================== 核心递推逻辑 ====================

    def _insert_double_bonds(self, prev_layer: List[nx.Graph],
                             layer_k: int) -> List[nx.Graph]:
        """
        核心递推：在第(k-1)层异构体的合法单键边上插入第k个双键

        算法流程：
        1. 遍历 prev_layer 中的每个异构体 G
        2. 找出 G 中所有可以放置新双键的合法单键边
        3. 对每条合法边，构建新图 H（该边升级为双键）
        4. 化学约束验证
        5. WL哈希分组 + 图同构精确去重

        Args:
            prev_layer: 第(k-1)层的异构体列表
            layer_k: 当前层级（放置第k个双键）

        Returns:
            第k层的唯一异构体列表
        """
        hash_groups = {}  # WL hash -> 图列表
        total_candidates = 0

        if HAS_TQDM:
            iterator = tqdm(prev_layer, desc=f"L{layer_k}递推", unit="异构体")
        else:
            iterator = prev_layer

        for G in iterator:
            edges = list(G.edges())

            for u, v in edges:
                # 该边必须是单键（不能在已有的双键上再放双键）
                bond_type = G[u][v].get('bond_type', 'single')
                if bond_type != 'single':
                    continue

                # 化学约束：放置双键后，两个端点的度数是否合法
                if not self._can_insert_double_bond(G, u, v):
                    continue

                # 构建新图：将边 (u, v) 升级为双键
                H = self._upgrade_edge_to_double(G, u, v)

                # 化学约束全面验证
                if not self._validate_molecule(H):
                    continue

                total_candidates += 1

                # WL 哈希分组加速去重
                h = nx.weisfeiler_lehman_graph_hash(
                    H, edge_attr='bond_type', node_attr='label'
                )

                if h not in hash_groups:
                    hash_groups[h] = [H]
                else:
                    dup = False
                    for existing in hash_groups[h]:
                        if self._is_isomorphic(H, existing):
                            dup = True
                            break
                    if not dup:
                        hash_groups[h].append(H)

        # 合并所有哈希组
        unique = []
        for group in hash_groups.values():
            unique.extend(group)

        return unique

    def _can_insert_double_bond(self, G: nx.Graph, u: int, v: int) -> bool:
        """
        检查是否可以在边 (u, v) 上插入双键

        化学约束：
        - 如果 u 已经是双键端点（sp²），则 u 在图中度数 ≤ 2（放入双键后度数≤3）
          但如果是累积二烯（u同时是两个双键的端点），则 u 度数必须 ≤ 2（放入后仍是sp²）
        - 如果 u 不是双键端点（sp³），则 u 度数 ≤ 3（放入双键后度数不变，但sp²约束度≤3）

        简化规则：放入双键后，每个端点碳的有效键数不超过4
        更精确的规则：放入双键后，该端点的 sp 杂化类型约束

        Args:
            G: 当前图
            u, v: 待升级边的端点

        Returns:
            是否可以放置
        """
        # 计算 u 和 v 当前的"键数负载"
        # 每条双键算2个键，三键算3个键，单键算1个键
        for node in [u, v]:
            bond_load = self._calculate_bond_load(G, node)
            # 升级为双键后，这条边从1个键变成2个键，负载+1
            new_load = bond_load + 1
            # 碳的最大化合价为4
            if new_load > 4:
                return False

            # sp² 碦约束：如果该节点放入双键后变成双键端点
            # 检查放入后该节点的所有双键数量
            # 累积二烯：同一碳参与2个双键，该碳只有2个σ键（度=2）
            # 非累积：sp²碳度数≤3

        # 进一步约束：检查放入双键后端点的度约束
        for node in [u, v]:
            deg = G.degree(node)
            # 计算该节点已有的双键数
            existing_double_bonds = sum(
                1 for nb in G.neighbors(node)
                if G[node][nb].get('bond_type', 'single') == 'double'
            )
            new_double_bonds = existing_double_bonds + 1  # 加上新放的这一个

            if new_double_bonds >= 2:
                # 累积二烯/丙二烯型：同一碳参与2个双键
                # sp杂化（实际是sp-like），该碳最多2个σ键 → 度 ≤ 2
                if deg > 2:
                    return False
            else:
                # 普通sp²碳：最多3个σ键 → 度 ≤ 3
                if deg > 3:
                    return False

        return True

    def _calculate_bond_load(self, G: nx.Graph, node: int) -> int:
        """计算节点当前的总键数负载（单键=1，双键=2，三键=3）"""
        total = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            if bt == 'double':
                total += 2
            elif bt == 'triple':
                total += 3
            else:
                total += 1
        return total

    def _upgrade_edge_to_double(self, G: nx.Graph, u: int, v: int) -> nx.Graph:
        """
        将图 G 中的边 (u, v) 从单键升级为双键，返回新图

        Args:
            G: 原始图
            u, v: 待升级边的端点

        Returns:
            新图（边(u,v)标记为double，其余边保持不变）
        """
        H = nx.Graph()
        for node in G.nodes():
            H.add_node(node, label=G.nodes[node].get('label', 'C'))
        for a, b in G.edges():
            if (a == u and b == v) or (a == v and b == u):
                H.add_edge(a, b, bond_type='double')
            else:
                H.add_edge(a, b, bond_type=G[a][b].get('bond_type', 'single'))
        return H

    # ==================== 化学验证 ====================

    def _validate_molecule(self, G: nx.Graph) -> bool:
        """
        全面验证分子结构的化学有效性

        约束：
        1. 每个碳的总键数 ≤ 4（价键约束）
        2. sp²碳（恰好参与1个双键）度数 ≤ 3
        3. sp-like碳（参与2个双键，累积二烯中心碳）度数 ≤ 2
        4. sp³碳度数 ≤ 4
        5. 氢原子数非负

        Returns:
            是否化学有效
        """
        for node in G.nodes():
            bond_load = self._calculate_bond_load(G, node)
            deg = G.degree(node)

            # 价键约束
            if bond_load > 4:
                return False

            # 计算该节点参与的双键数
            double_bond_count = sum(
                1 for nb in G.neighbors(node)
                if G[node][nb].get('bond_type', 'single') == 'double'
            )

            if double_bond_count >= 2:
                # 累积二烯中心碳：sp杂化，度数 ≤ 2
                if deg > 2:
                    return False
            elif double_bond_count == 1:
                # sp²碳：度数 ≤ 3
                if deg > 3:
                    return False
            else:
                # sp³碳：度数 ≤ 4
                if deg > 4:
                    return False

            # 氢原子数非负
            h_count = 4 - bond_load
            if h_count < 0:
                return False

        return True

    # ==================== 去重工具 ====================

    def _is_isomorphic(self, G1: nx.Graph, G2: nx.Graph) -> bool:
        """检查两个图是否同构（考虑键类型）"""
        def custom_edge_match(e1, e2):
            return e1.get('bond_type') == e2.get('bond_type')

        try:
            return nx.is_isomorphic(
                G1, G2,
                node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
                edge_match=custom_edge_match
            )
        except Exception:
            return False

    # ==================== 辅助方法 ====================

    def _get_alkane_graphs(self, n_carbons: int) -> List[nx.Graph]:
        """获取 n_carbons 个碳的烷烃骨架图列表"""
        alkane_canons = self.alkane_gen.generate_isomers(n_carbons)

        if alkane_canons and hasattr(alkane_canons[0], 'edges'):
            # 已经是图对象 → 添加标签和单键属性
            graphs = []
            for G in alkane_canons:
                H = nx.Graph()
                for node in G.nodes():
                    H.add_node(node, label='C')
                for a, b in G.edges():
                    H.add_edge(a, b, bond_type='single')
                graphs.append(H)
            return graphs

        # 规范字符串列表 → 转换为图
        base_trees_adj = [self.alkane_gen.canon_to_adjacency(canon) for canon in alkane_canons]
        graphs = []
        for adj in base_trees_adj:
            G = nx.Graph()
            for node, neighbors in adj.items():
                G.add_node(node, label='C')
                for neighbor in neighbors:
                    if neighbor > node:
                        G.add_edge(node, neighbor, bond_type='single')
            graphs.append(G)
        return graphs

    @staticmethod
    def _chinese_name(k: int) -> str:
        """k-烯烃的中文名称"""
        names = {1: '单烯烃', 2: '二烯烃', 3: '三烯烃', 4: '四烯烃',
                 5: '五烯烃', 6: '六烯烃', 7: '七烯烃', 8: '八烯烃'}
        return names.get(k, f'{k}烯烃')

    def _get_verified_count(self, k: int) -> Dict[int, int]:
        """获取 k-烯烃的已验证数据"""
        verified_map = {
            1: self.VERIFIED_MONOENE,
            2: self.VERIFIED_DIENE,
            3: self.VERIFIED_TRIENE,
            4: self.VERIFIED_TETRAENE,
        }
        return verified_map.get(k, {})

    def describe_isomer(self, G: nx.Graph) -> str:
        """生成异构体的描述字符串"""
        double_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('bond_type') == 'double']

        desc_parts = []
        for u, v in double_edges:
            desc_parts.append(f"C{u+1}=C{v+1}")

        # 计算各碳的氢数
        h_info = []
        for node in sorted(G.nodes()):
            total_bonds = self._calculate_bond_load(G, node)
            h_count = 4 - total_bonds
            h_info.append(f"C{node+1}({h_count}H)")

        double_count = len(double_edges)
        name = self._chinese_name(double_count)
        desc = " ".join(desc_parts) + " [" + ", ".join(h_info) + "]"
        return f"{name}: {desc}"

    def classify_polyene(self, G: nx.Graph) -> str:
        """
        分类多烯烃类型

        Returns:
            'cumulated' (累积), 'conjugated' (共轭), 'isolated' (隔离), 或混合类型
        """
        double_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('bond_type') == 'double']

        if len(double_edges) <= 1:
            return 'monoene'

        # 检查是否有累积双键（共享端点）
        double_nodes = []
        for u, v in double_edges:
            double_nodes.append(set([u, v]))

        has_cumulated = False
        has_conjugated = False
        has_isolated = False

        for i in range(len(double_edges)):
            for j in range(i + 1, len(double_edges)):
                shared = double_nodes[i] & double_nodes[j]
                if shared:
                    has_cumulated = True
                else:
                    # 检查是否共轭（通过一条单键相连）
                    ni = double_nodes[i]
                    nj = double_nodes[j]
                    # 共轭：一个双键的端点通过单键连接到另一个双键的端点
                    connected = False
                    for a in ni:
                        for b in nj:
                            if G.has_edge(a, b):
                                connected = True
                                break
                    if connected:
                        has_conjugated = True
                    else:
                        has_isolated = True

        parts = []
        if has_cumulated:
            parts.append('cumulated')
        if has_conjugated:
            parts.append('conjugated')
        if has_isolated:
            parts.append('isolated')

        return '+'.join(parts) if parts else 'unknown'

    def graph_to_rdkit_mol(self, G: nx.Graph, optimize: bool = True):
        """将图对象转换为 RDKit Mol 对象"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            return None

        n_carbons = G.number_of_nodes()

        # 收集键类型
        double_bond_edges = set()
        for u, v, data in G.edges(data=True):
            bt = data.get('bond_type', 'single')
            if bt == 'double':
                double_bond_edges.add((min(u, v), max(u, v)))

        # 构建 RDKit 分子
        mol = Chem.RWMol()
        for i in range(n_carbons):
            mol.AddAtom(Chem.Atom('C'))

        added_bonds = set()
        for u, v in G.edges():
            bond_key = (min(u, v), max(u, v))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)
            if bond_key in double_bond_edges:
                mol.AddBond(u, v, BondType.DOUBLE)
            else:
                mol.AddBond(u, v, BondType.SINGLE)

        # 设置显式氢
        for node in sorted(G.nodes()):
            bond_load = self._calculate_bond_load(G, node)
            h_count = 4 - bond_load
            if h_count > 0:
                mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        mol = Chem.AddHs(mol)

        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)

        if optimize:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass

        return mol

    def visualize_isomer(self, G: nx.Graph, title: str = "") -> bool:
        """使用 RDKit + matplotlib 弹出3D可视化窗口"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            print(f"可视化依赖缺失: {e}")
            return False

        mol = self.graph_to_rdkit_mol(G, optimize=True)
        if mol is None:
            print("RDKit 分子构建失败")
            return False

        from rdkit import Chem
        conf = mol.GetConformer()

        c_coords = {}
        h_coords = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coord = (float(pos.x), float(pos.y), float(pos.z))
            if atom.GetSymbol() == 'C':
                c_coords[atom.GetIdx()] = coord
            else:
                h_coords.append(coord)

        # 收集双键信息
        double_bond_edges = set()
        double_nodes = set()
        for u, v, d in G.edges(data=True):
            if d.get('bond_type') == 'single':
                continue
            if d.get('bond_type') == 'double':
                key = (min(u, v), max(u, v))
                double_bond_edges.add(key)
                double_nodes.add(u)
                double_nodes.add(v)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # 绘制键
        drawn_bonds = set()
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                bond_key = tuple(sorted([node, neighbor]))
                if bond_key not in drawn_bonds:
                    if node in c_coords and neighbor in c_coords:
                        p1 = np.array(c_coords[node])
                        p2 = np.array(c_coords[neighbor])
                        direction = p2 - p1
                        length = np.linalg.norm(direction)

                        if bond_key in double_bond_edges:
                            if length > 0:
                                if abs(direction[0]) < abs(direction[1]):
                                    perp = np.cross(direction, [1, 0, 0])
                                else:
                                    perp = np.cross(direction, [0, 1, 0])
                                perp = perp / np.linalg.norm(perp) * 0.12
                                ax.plot(
                                    [p1[0]+perp[0], p2[0]+perp[0]],
                                    [p1[1]+perp[1], p2[1]+perp[1]],
                                    [p1[2]+perp[2], p2[2]+perp[2]],
                                    color='#FF6600', linewidth=3
                                )
                                ax.plot(
                                    [p1[0]-perp[0], p2[0]-perp[0]],
                                    [p1[1]-perp[1], p2[1]-perp[1]],
                                    [p1[2]-perp[2], p2[2]-perp[2]],
                                    color='#FF6600', linewidth=3
                                )
                            else:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='#FF6600', linewidth=3
                                )
                        else:
                            ax.plot(
                                [c_coords[node][0], c_coords[neighbor][0]],
                                [c_coords[node][1], c_coords[neighbor][1]],
                                [c_coords[node][2], c_coords[neighbor][2]],
                                color='black', linewidth=2
                            )
                    drawn_bonds.add(bond_key)

        # 绘制碳原子
        for n in sorted(c_coords.keys()):
            color = '#FF6600' if n in double_nodes else '#333333'
            ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                       c=color, s=150, zorder=5)

        # 绘制氢原子及C-H键
        if h_coords:
            # 建立H到C的映射：RDKit AddHs后，每个C原子后面紧跟其H原子
            # 需要根据分子中C-H键的连接关系确定映射
            h_to_c = {}
            h_idx = 0
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetSymbol() == 'C' and a2.GetSymbol() == 'H':
                    h_to_c[h_idx] = a1.GetIdx()
                    h_idx += 1
                elif a1.GetSymbol() == 'H' and a2.GetSymbol() == 'C':
                    h_to_c[h_idx] = a2.GetIdx()
                    h_idx += 1

            hx = [h[0] for h in h_coords]
            hy = [h[1] for h in h_coords]
            hz = [h[2] for h in h_coords]
            ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

            # 绘制C-H键
            for i, h_coord in enumerate(h_coords):
                if i in h_to_c and h_to_c[i] in c_coords:
                    c_coord = c_coords[h_to_c[i]]
                    ax.plot(
                        [c_coord[0], h_coord[0]],
                        [c_coord[1], h_coord[1]],
                        [c_coord[2], h_coord[2]],
                        color='gray', linewidth=1, alpha=0.7
                    )

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FF6600', linewidth=3, label='C=C'),
            Line2D([0], [0], color='black', linewidth=2, label='C-C'),
            Line2D([0], [0], color='gray', linewidth=1, label='C-H'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_xlabel("X (A)")
        ax.set_ylabel("Y (A)")
        ax.set_zlabel("Z (A)")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()
        return True


# ==================== 命令行入口 ====================

def main():
    """命令行交互入口"""
    print("=" * 60)
    print("  多烯烃同分异构体统一生成器")
    print("  算法: 烷烃骨架 → 逐层插入双键（递推法）")
    print("=" * 60)

    try:
        n = int(input("\n请输入碳原子数: "))
    except (ValueError, EOFError):
        print("输入无效")
        return

    if n < 2:
        print("碳原子数必须 >= 2")
        return

    gen = PolyeneIsomerGenerator()
    results = gen.generate_all_layers(n, verbose=True)

    # 交互式查看
    print(f"\n{'='*60}")
    print("  生成完成！可用命令:")
    print("  list [k]         - 列出k-烯烃的所有异构体")
    print("  show k i         - 可视化第k层第i个异构体")
    print("  classify k i     - 分类第k层第i个异构体")
    print("  quit             - 退出")
    print(f"{'='*60}")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd or cmd == 'quit':
            break

        parts = cmd.split()
        if parts[0] == 'list' and len(parts) >= 2:
            k = int(parts[1])
            if k in results:
                for i, G in enumerate(results[k]):
                    desc = gen.describe_isomer(G)
                    print(f"  #{i+1}: {desc}")
            else:
                print(f"  无 {k}烯烃数据")
        elif parts[0] == 'show' and len(parts) >= 3:
            k, i = int(parts[1]), int(parts[2])
            if k in results and 0 < i <= len(results[k]):
                G = results[k][i-1]
                desc = gen.describe_isomer(G)
                gen.visualize_isomer(G, desc)
            else:
                print("  索引超出范围")
        elif parts[0] == 'classify' and len(parts) >= 3:
            k, i = int(parts[1]), int(parts[2])
            if k in results and 0 < i <= len(results[k]):
                G = results[k][i-1]
                cls = gen.classify_polyene(G)
                print(f"  分类: {cls}")
            else:
                print("  索引超出范围")
        else:
            print("  未知命令")


if __name__ == '__main__':
    main()
