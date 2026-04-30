"""
组合式多环多烯炔烃生成器
策略：根据 (环数, 双键数, 三键数) 分派到已验证的专业子模块
  - n_rings=0: 分派到 polyalkenyne（无环多烯炔）
  - n_rings=1, n_tb=0: 分派到 cyclopolyene_generator（单环多烯）
  - n_rings>=2, n_db=0, n_tb=0: 分派到 multcycloalkane（多环烷烃）
  - 其他: 多环烷烃骨架 + 改进的键插入逻辑
"""

import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple
import networkx as nx

# 确保同目录模块可被导入
_sys_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)

# 导入已通过验证的模块
try:
    from multcycloalkane import PolycycloalkaneGenerator
    from polyalkenyne import PolyalkenyneGenerator
    from cyclopolyene_generator import CyclopolyeneGenerator
except ImportError as e:
    print(f"错误：缺少核心模块 - {e}")
    sys.exit(1)


class RobustPolycyclicPolyeneGenerator:
    """
    多环多烯炔烃完备生成器
    使用方法：
        gen = RobustPolycyclicPolyeneGenerator()
        isomers = gen.generate(n_c, n_rings, n_db, n_tb)
    """

    def __init__(self):
        self.ring_gen = PolycycloalkaneGenerator()
        self.polyenyne_gen = PolyalkenyneGenerator()
        self.cyclopolyene_gen = CyclopolyeneGenerator()

    def generate(self, n_carbons: int, n_rings: int = 0,
                 n_db: int = 0, n_tb: int = 0) -> List[nx.Graph]:
        """
        生成指定环数、双键数、三键数的所有异构体
        """
        if n_carbons < 2:
            return []

        # ---- 分派到已验证的专业子模块 ----

        # 无环结构：完全由 polyalkenyne 处理
        if n_rings == 0:
            return self._generate_acyclic(n_carbons, n_db, n_tb)

        # 单环 + 仅双键（无三键）：由 cyclopolyene_generator 处理
        if n_rings == 1 and n_tb == 0:
            return self._generate_monocyclic_polyene(n_carbons, n_db)

        # 多环 + 仅单键（无双键三键）：由 multcycloalkane 处理
        if n_db == 0 and n_tb == 0:
            return self._generate_polycyclic_alkane(n_carbons, n_rings)

        # 通用情况：多环骨架 + 插入双键/三键
        return self._generate_general(n_carbons, n_rings, n_db, n_tb)

    # ==================== 分派方法 ====================

    def _generate_acyclic(self, n_carbons: int, n_db: int, n_tb: int) -> List[nx.Graph]:
        """无环多烯炔烃：委托给 polyalkenyne"""
        if n_db == 0 and n_tb == 0:
            # 纯烷烃 CnH2n+2
            return []
        print(f"  [分派] 无环: polyalkenyne C{n_carbons} d={n_db} t={n_tb}")
        return self.polyenyne_gen.generate(n_carbons, n_db, n_tb)

    def _generate_monocyclic_polyene(self, n_carbons: int, n_db: int) -> List[nx.Graph]:
        """单环多烯烃（无双键时退化为单环烷烃）：委托给 cyclopolyene_generator"""
        print(f"  [分派] 单环多烯: cyclopolyene C{n_carbons} d={n_db}")
        if n_db == 0:
            return self._generate_polycyclic_alkane(n_carbons, 1)
        return self.cyclopolyene_gen.generate_k_cycloene(n_carbons, n_db)

    def _generate_polycyclic_alkane(self, n_carbons: int, n_rings: int) -> List[nx.Graph]:
        """多环烷烃：委托给 multcycloalkane"""
        print(f"  [分派] 多环烷烃: multcycloalkane C{n_carbons} r={n_rings}")
        ring_graphs = self.ring_gen.generate_isomers(n_carbons, n_rings, verbose=True)
        # 转换为带 bond_type 属性的图（统一格式）
        return self._convert_to_bond_type_graphs(ring_graphs)

    def _generate_general(self, n_carbons: int, n_rings: int,
                          n_db: int, n_tb: int) -> List[nx.Graph]:
        """
        通用情况：先由 multcycloalkane 生成多环骨架，再插入双键/三键
        使用改进的化学约束和去重逻辑
        """
        print(f"  [通用] C{n_carbons} r={n_rings} d={n_db} t={n_tb}")

        # 1. 生成目标环数的多环烷烃骨架
        print(f"  生成 {n_rings} 环骨架 ...")
        ring_graphs = self.ring_gen.generate_isomers(n_carbons, n_rings, verbose=True)
        print(f"  获得 {len(ring_graphs)} 个饱和多环骨架")

        if not ring_graphs:
            return []

        # 2. 转换为带 bond_type 属性的图
        base_graphs = self._convert_to_bond_type_graphs(ring_graphs)

        # 3. 使用改进的键插入逻辑
        return self._insert_multiple_bonds(base_graphs, n_db, n_tb)

    # ==================== 图格式转换 ====================

    @staticmethod
    def _convert_to_bond_type_graphs(graphs: List[nx.Graph]) -> List[nx.Graph]:
        """将无 bond_type 属性的图列表转换为带 bond_type='single' 的图"""
        result = []
        for G in graphs:
            H = nx.Graph()
            for node in G.nodes():
                H.add_node(node, label='C')
            for u, v in G.edges():
                H.add_edge(u, v, bond_type='single')
            result.append(H)
        return result

    # ==================== 键插入（通用） ====================

    def _insert_multiple_bonds(self, base_graphs: List[nx.Graph],
                               n_db: int, n_tb: int) -> List[nx.Graph]:
        """
        在给定图集上放置 n_db 个双键和 n_tb 个三键。
        先插入所有三键，再插入所有双键。
        使用基于最终状态验证的约束：只做轻量预判（键负载），
        最终有效性由 _validate_molecule 在升级后全面检查。
        """
        # 使用 _add_to_dict 而非字典推导式，避免 WL 哈希碰撞导致不同骨架被错误覆盖
        current = {}
        for g in base_graphs:
            self._add_to_dict(g, current)

        # 插入三键
        for _ in range(n_tb):
            next_dict = {}
            for group in current.values():
                for G in group:
                    for u, v in list(G.edges()):
                        if G[u][v].get('bond_type', 'single') != 'single':
                            continue
                        # 轻量预判：键负载+2 ≤ 4（硬上限，不可违反）
                        if self._bond_load(G, u) + 2 > 4 or self._bond_load(G, v) + 2 > 4:
                            continue
                        H = self._upgrade_edge_to_triple(G, u, v)
                        if self._validate_molecule(H):
                            self._add_to_dict(H, next_dict)
            current = next_dict
            if not current:
                break

        # 插入双键
        for _ in range(n_db):
            next_dict = {}
            for group in current.values():
                for G in group:
                    for u, v in list(G.edges()):
                        if G[u][v].get('bond_type', 'single') != 'single':
                            continue
                        # 轻量预判：键负载+1 ≤ 4（硬上限，不可违反）
                        if self._bond_load(G, u) + 1 > 4 or self._bond_load(G, v) + 1 > 4:
                            continue
                        H = self._upgrade_edge_to_double(G, u, v)
                        if self._validate_molecule(H):
                            self._add_to_dict(H, next_dict)
            current = next_dict
            if not current:
                break

        # 展平结果
        result = []
        for group in current.values():
            result.extend(group)
        return result

    # ==================== 化学约束（改进版，兼容环结构） ====================

    def _can_insert_double(self, G: nx.Graph, u: int, v: int) -> bool:
        """
        检查边(u,v)是否可以插入双键（兼容环结构）
        约束（基于键负载，兼容桥头碳）：
          - 键负载 +1 ≤ 4
          - 累积二烯碳度数 ≤ 2（同一碳参与两个双键时度数不超过2）
        """
        for node in (u, v):
            if self._bond_load(G, node) + 1 > 4:
                return False
            new_double = self._count_double_bonds(G, node) + 1
            if new_double >= 2 and G.degree(node) > 2:
                return False
        return True

    def _can_insert_triple(self, G: nx.Graph, u: int, v: int) -> bool:
        """
        检查边(u,v)是否可以插入三键（兼容环结构）
        约束：
          - 键负载 +2 ≤ 4
          - sp碳度数 ≤ 2
          - sp碳不参与其他重键
        """
        for node in (u, v):
            if self._bond_load(G, node) + 2 > 4:
                return False
            if G.degree(node) > 2:
                return False
            if self._count_double_bonds(G, node) > 0 or self._count_triple_bonds(G, node) > 0:
                return False
        return True

    def _validate_molecule(self, G: nx.Graph) -> bool:
        """
        验证分子结构的化学有效性（改进版，兼容环结构）
        约束（基于键负载而非度数，兼容桥头碳等环结构情形）：
          - 每个碳键负载 ≤ 4
          - sp³碳: 无额外约束（键负载已涵盖）
          - sp²碳: 累积二烯碳度数 ≤ 2（同一碳参与两个双键时度数不能超过2）
          - sp碳: 度数 ≤ 2（sp碳最多2个σ键）
          - 氢原子数 ≥ 0（键负载 ≤ 4 已保证）
        """
        for node in G.nodes():
            load = self._bond_load(G, node)
            deg = G.degree(node)

            if load > 4:
                return False

            n_db = self._count_double_bonds(G, node)
            n_tb = self._count_triple_bonds(G, node)

            # sp碳: 度数 ≤ 2（最多2个σ键，如 -C≡）
            if n_tb > 0 and deg > 2:
                return False
            # 累积二烯: 同一碳参与≥2个双键时度数 ≤ 2
            if n_db >= 2 and deg > 2:
                return False

        return True

    # ==================== 键升级 ====================

    @staticmethod
    def _upgrade_edge_to_double(G: nx.Graph, u: int, v: int) -> nx.Graph:
        """将边(u,v)从单键升级为双键"""
        H = nx.Graph()
        for n in G.nodes():
            H.add_node(n, label=G.nodes[n].get('label', 'C'))
        for a, b, data in G.edges(data=True):
            if (a == u and b == v) or (a == v and b == u):
                H.add_edge(a, b, bond_type='double')
            else:
                H.add_edge(a, b, bond_type=data.get('bond_type', 'single'))
        return H

    @staticmethod
    def _upgrade_edge_to_triple(G: nx.Graph, u: int, v: int) -> nx.Graph:
        """将边(u,v)从单键升级为三键"""
        H = nx.Graph()
        for n in G.nodes():
            H.add_node(n, label=G.nodes[n].get('label', 'C'))
        for a, b, data in G.edges(data=True):
            if (a == u and b == v) or (a == v and b == u):
                H.add_edge(a, b, bond_type='triple')
            else:
                H.add_edge(a, b, bond_type=data.get('bond_type', 'single'))
        return H

    # ==================== 辅助函数 ====================

    @staticmethod
    def _bond_load(G: nx.Graph, node: int) -> int:
        """计算节点当前的总键数负载"""
        total = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            if bt == 'triple':   total += 3
            elif bt == 'double': total += 2
            else:                total += 1
        return total

    @staticmethod
    def _count_double_bonds(G: nx.Graph, node: int) -> int:
        """计算节点参与的双键数"""
        return sum(1 for nb in G.neighbors(node)
                   if G[node][nb].get('bond_type') == 'double')

    @staticmethod
    def _count_triple_bonds(G: nx.Graph, node: int) -> int:
        """计算节点参与的三键数"""
        return sum(1 for nb in G.neighbors(node)
                   if G[node][nb].get('bond_type') == 'triple')

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        """计算图的 WL 哈希作为规范键"""
        return nx.weisfeiler_lehman_graph_hash(
            G, edge_attr='bond_type', node_attr='label'
        )

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        """快速度序列签名，用于 is_isomorphic 的廉价预过滤"""
        return tuple(sorted(G.degree(v) for v in G.nodes()))

    @staticmethod
    def _add_to_dict(G: nx.Graph, bucket: Dict[str, List[nx.Graph]]):
        """
        将图添加到去重字典中（改进版）
        同一哈希桶内保存所有图列表，逐一做同构检查
        """
        h = RobustPolycyclicPolyeneGenerator._canon_key(G)
        if h not in bucket:
            bucket[h] = [G]
            return
        # 与桶内所有图做同构比较
        deg_sig = RobustPolycyclicPolyeneGenerator._degree_signature(G)
        for existing in bucket[h]:
            if RobustPolycyclicPolyeneGenerator._degree_signature(existing) != deg_sig:
                continue
            if nx.is_isomorphic(
                G, existing,
                node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
                edge_match=lambda e1, e2: e1.get('bond_type') == e2.get('bond_type')
            ):
                return  # 已存在同构图
        bucket[h].append(G)

    # ==================== 描述与统计 ====================

    def describe_isomer(self, G: nx.Graph) -> str:
        """
        生成异构体的描述字符串。

        格式: 环数=nodes, 双键=[...], 三键=[...], 度分布
        如: "2环(5,5) 双键=[(1,2)] 三键=[] deg=(2:2,3:4,4:2)"
        """
        cycles = nx.cycle_basis(G)
        n_rings = len(cycles)
        ring_sizes = sorted([len(c) for c in cycles], reverse=True)

        from collections import Counter
        deg_dist = Counter(G.degree(v) for v in G.nodes())
        deg_str = ",".join(f"{d}:{c}" for d, c in sorted(deg_dist.items()))

        ring_str = ",".join(str(s) for s in ring_sizes)

        double_bonds = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
        triple_bonds = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']

        return f"{n_rings}环({ring_str}) 双键={double_bonds} 三键={triple_bonds} deg=({deg_str})"

    def ring_size_distribution(self, isomers: List[nx.Graph]) -> Dict[Tuple, int]:
        """统计各环大小组合的异构体数量分布"""
        dist: Dict[Tuple, int] = defaultdict(int)
        for G in isomers:
            cycles = nx.cycle_basis(G)
            ring_sizes = tuple(sorted([len(c) for c in cycles], reverse=True))
            dist[ring_sizes] += 1
        return dict(dist)

    # ==================== 可视化 ====================

    def graph_to_rdkit_mol(self, G: nx.Graph, optimize: bool = True):
        """将图对象转换为 RDKit Mol 对象（含3D坐标 + MMFF力场优化）"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            return None

        n_carbons = G.number_of_nodes()

        # 提取双键和三键边信息
        double_bonds = set()
        triple_bonds = set()
        for u, v, data in G.edges(data=True):
            bt = data.get('bond_type', 'single')
            key = (min(u, v), max(u, v))
            if bt == 'double':
                double_bonds.add(key)
            elif bt == 'triple':
                triple_bonds.add(key)

        mol = Chem.RWMol()
        for i in range(n_carbons):
            mol.AddAtom(Chem.Atom('C'))

        added_bonds = set()
        for u, v in G.edges():
            bond_key = (min(u, v), max(u, v))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)
            if bond_key in triple_bonds:
                mol.AddBond(u, v, BondType.TRIPLE)
            elif bond_key in double_bonds:
                mol.AddBond(u, v, BondType.DOUBLE)
            else:
                mol.AddBond(u, v, BondType.SINGLE)

        # 设置显式氢（根据键负载计算）
        for node in sorted(G.nodes()):
            bond_load = 0
            for nb in G.neighbors(node):
                bt = G[node][nb].get('bond_type', 'single')
                if bt == 'double':
                    bond_load += 2
                elif bt == 'triple':
                    bond_load += 3
                else:
                    bond_load += 1
            h_count = 4 - bond_load
            if h_count > 0:
                mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        mol = Chem.AddHs(mol)

        # 多环多烯炔烃：多种子选择策略，选择最大C-C键长最短的构象
        best_mol = None
        best_max_bond = float('inf')
        for seed in range(50):
            mol_trial = Chem.RWMol(Chem.Mol(mol.ToBinary()))
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            result = AllChem.EmbedMolecule(mol_trial, params)
            if result == -1:
                result2 = AllChem.EmbedMolecule(mol_trial, randomSeed=seed, useRandomCoords=True)
                if result2 == -1:
                    continue
            if optimize:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_trial, maxIters=500)
                except Exception:
                    pass
            # 计算最大C-C键长
            conf = mol_trial.GetConformer()
            max_bond = 0.0
            for u, v in G.edges():
                if u < n_carbons and v < n_carbons:
                    p1 = conf.GetAtomPosition(u)
                    p2 = conf.GetAtomPosition(v)
                    dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
                    max_bond = max(max_bond, dist)
            if max_bond < best_max_bond:
                best_max_bond = max_bond
                best_mol = Chem.Mol(mol_trial.ToBinary())
        if best_mol is not None:
            mol = best_mol
        else:
            # 所有种子都失败，回退到2D坐标
            AllChem.Compute2DCoords(mol)

        return mol

    def visualize_isomer(self, G: nx.Graph, title: str = "") -> bool:
        """使用 RDKit + matplotlib 弹出3D可视化窗口"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from rdkit import Chem
        except ImportError as e:
            print(f"可视化依赖缺失: {e}")
            print("请安装: pip install matplotlib rdkit")
            return False

        mol = self.graph_to_rdkit_mol(G, optimize=True)
        if mol is None:
            print("RDKit 分子构建失败，无法可视化")
            return False

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

        # 获取环信息
        cycles = nx.cycle_basis(G)
        ring_nodes = set()
        for cycle in cycles:
            ring_nodes.update(cycle)

        # 为不同环分配不同颜色
        ring_colors = ['#CC0000', '#0066CC', '#009933', '#CC6600', '#6600CC', '#009999']
        node_ring_color = {}
        for ci, cycle in enumerate(cycles):
            color = ring_colors[ci % len(ring_colors)]
            for node in cycle:
                if node not in node_ring_color:
                    node_ring_color[node] = color

        # 标记环内边
        ring_edges = set()
        for ci, cycle in enumerate(cycles):
            cycle_set = set(cycle)
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ring_edges.add(tuple(sorted([u, v])))

        # 收集双键和三键边
        double_bond_edges = set()
        triple_bond_edges = set()
        for u, v, data in G.edges(data=True):
            bt = data.get('bond_type', 'single')
            key = tuple(sorted([u, v]))
            if bt == 'double':
                double_bond_edges.add(key)
            elif bt == 'triple':
                triple_bond_edges.add(key)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # 绘制 C-C 键
        drawn_bonds = set()
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                bond_key = tuple(sorted([node, neighbor]))
                if bond_key not in drawn_bonds:
                    if node in c_coords and neighbor in c_coords:
                        import numpy as np
                        p1 = np.array(c_coords[node])
                        p2 = np.array(c_coords[neighbor])
                        direction = p2 - p1
                        length = np.linalg.norm(direction)

                        if bond_key in triple_bond_edges:
                            # 三键：三条平行线
                            if length > 1e-8:
                                if abs(direction[0]) < abs(direction[1]):
                                    perp = np.cross(direction, [1, 0, 0])
                                else:
                                    perp = np.cross(direction, [0, 1, 0])
                                perp = perp / np.linalg.norm(perp) * 0.12
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                        color='#8B008B', linewidth=3)
                                ax.plot([p1[0]+perp[0], p2[0]+perp[0]],
                                        [p1[1]+perp[1], p2[1]+perp[1]],
                                        [p1[2]+perp[2], p2[2]+perp[2]],
                                        color='#8B008B', linewidth=2)
                                ax.plot([p1[0]-perp[0], p2[0]-perp[0]],
                                        [p1[1]-perp[1], p2[1]-perp[1]],
                                        [p1[2]-perp[2], p2[2]-perp[2]],
                                        color='#8B008B', linewidth=2)
                            else:
                                ax.plot([c_coords[node][0], c_coords[neighbor][0]],
                                        [c_coords[node][1], c_coords[neighbor][1]],
                                        [c_coords[node][2], c_coords[neighbor][2]],
                                        color='#8B008B', linewidth=3)
                        elif bond_key in double_bond_edges:
                            # 双键：两条平行线
                            if length > 1e-8:
                                if abs(direction[0]) < abs(direction[1]):
                                    perp = np.cross(direction, [1, 0, 0])
                                else:
                                    perp = np.cross(direction, [0, 1, 0])
                                perp = perp / np.linalg.norm(perp) * 0.12
                                ax.plot([p1[0]+perp[0], p2[0]+perp[0]],
                                        [p1[1]+perp[1], p2[1]+perp[1]],
                                        [p1[2]+perp[2], p2[2]+perp[2]],
                                        color='red', linewidth=2.5)
                                ax.plot([p1[0]-perp[0], p2[0]-perp[0]],
                                        [p1[1]-perp[1], p2[1]-perp[1]],
                                        [p1[2]-perp[2], p2[2]-perp[2]],
                                        color='red', linewidth=2.5)
                            else:
                                ax.plot([c_coords[node][0], c_coords[neighbor][0]],
                                        [c_coords[node][1], c_coords[neighbor][1]],
                                        [c_coords[node][2], c_coords[neighbor][2]],
                                        color='red', linewidth=2.5)
                        elif bond_key in ring_edges:
                            # 环内单键
                            ax.plot(
                                [c_coords[node][0], c_coords[neighbor][0]],
                                [c_coords[node][1], c_coords[neighbor][1]],
                                [c_coords[node][2], c_coords[neighbor][2]],
                                color='#CC0000', linewidth=2.5
                            )
                        else:
                            # 环外单键
                            ax.plot(
                                [c_coords[node][0], c_coords[neighbor][0]],
                                [c_coords[node][1], c_coords[neighbor][1]],
                                [c_coords[node][2], c_coords[neighbor][2]],
                                color='black', linewidth=2
                            )
                    drawn_bonds.add(bond_key)

        # 绘制碳原子（不同环用不同颜色，含双键/三键碳用特殊色）
        double_carbons = set()
        triple_carbons = set()
        for u, v, d in G.edges(data=True):
            if d.get('bond_type') == 'double':
                double_carbons.add(u)
                double_carbons.add(v)
            elif d.get('bond_type') == 'triple':
                triple_carbons.add(u)
                triple_carbons.add(v)

        for n in sorted(c_coords.keys()):
            if n in triple_carbons:
                color = '#8B008B'  # 紫色：sp碳
            elif n in double_carbons:
                color = 'red'
            elif n in node_ring_color:
                color = node_ring_color[n]
            else:
                color = '#333333'
            ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                       c=color, s=150, zorder=5)

        # 绘制氢原子和碳氢键
        if h_coords:
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

            for i, h_coord in enumerate(h_coords):
                if i in h_to_c and h_to_c[i] in c_coords:
                    c_coord = c_coords[h_to_c[i]]
                    ax.plot(
                        [c_coord[0], h_coord[0]],
                        [c_coord[1], h_coord[1]],
                        [c_coord[2], h_coord[2]],
                        color='gray', linewidth=1, alpha=0.7
                    )

        # 标记环原子编号
        for ci, cycle in enumerate(cycles):
            for i, node in enumerate(cycle):
                if node in c_coords:
                    coord = c_coords[node]
                    ax.text(coord[0], coord[1], coord[2],
                            f'{i+1}', fontsize=9, color='blue')

        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2.5, label='双键 C=C'),
            Line2D([0], [0], color='#8B008B', linewidth=3, label='三键 C≡C'),
            Line2D([0], [0], color='#CC0000', linewidth=2.5, label='环内 C-C'),
            Line2D([0], [0], color='black', linewidth=2, label='环外 C-C'),
            Line2D([0], [0], color='gray', linewidth=1, label='C-H'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()
        return True


def _enumerate_rdt_combinations(k: int):
    """枚举所有满足 r + d + 2t = k 的非负整数组合 (r, d, t)"""
    combos = []
    for t in range(k // 2 + 1):
        for r in range(k - 2 * t + 1):
            d = k - 2 * t - r
            if d >= 0:
                combos.append((r, d, t))
    return combos


def _print_isomers(isomers, n_r, n_d, n_t, n_c, k):
    """打印一组异构体的详情"""
    h_count = 2 * n_c + 2 - 2 * k
    formula = f"C{n_c}H{h_count}"
    tag = ""
    if n_r:
        tag += f" {n_r}环"
    if n_d:
        tag += f" {n_d}双键"
    if n_t:
        tag += f" {n_t}三键"
    print(f"\n  {formula}{tag} → {len(isomers)} 个异构体")
    if isomers:
        for i, G in enumerate(isomers[:20], 1):
            dbs = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
            tbs = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
            print(f"    {i:3d}: 双键 {dbs}, 三键 {tbs}, 边数={G.number_of_edges()}")
        if len(isomers) > 20:
            print(f"    ... 还有 {len(isomers) - 20} 个")


# ==================== 交互命令行 ====================
if __name__ == '__main__':
    gen = RobustPolycyclicPolyeneGenerator()

    print()
    print("=" * 60)
    print("  多环多烯炔烃同分异构体生成器")
    print("  不饱和度 k = 环数 + 双键数 + 2×三键数")
    print("=" * 60)
    print()
    print("  输入模式:")
    print("    n r d t  → 碳数 环数 双键数 三键数 (精确指定)")
    print("    n k      → 碳数 不饱和度 (枚举所有 r,d,t 组合)")
    print("    q        → 退出")
    print()

    while True:
        try:
            cmd = input("> ").strip()
            if cmd.lower() in ('q', 'quit', 'exit'):
                break
            if not cmd:
                continue

            parts = cmd.split()

            # ---- 模式1: 碳数 + 不饱和度 ----
            if len(parts) == 2:
                n_c, k = map(int, parts)
                if n_c < 2:
                    print("  ⚠ 碳数 ≥ 2")
                    continue
                if k < 0:
                    print("  ⚠ 不饱和度 ≥ 0")
                    continue
                combos = _enumerate_rdt_combinations(k)
                if not combos:
                    print("  无合法 (环,双键,三键) 组合")
                    continue

                h_count = 2 * n_c + 2 - 2 * k
                formula = f"C{n_c}H{h_count}"
                print(f"\n  {formula}  不饱和度 k={k}")
                print(f"  共 {len(combos)} 种 (环,双键,三键) 组合:")
                print("-" * 60)

                grand_total = 0
                for n_r, n_d, n_t in combos:
                    isomers = gen.generate(n_c, n_r, n_d, n_t)
                    grand_total += len(isomers)
                    _print_isomers(isomers, n_r, n_d, n_t, n_c, k)

                print("-" * 60)
                print(f"  合计 {formula} (k={k}) → {grand_total} 个异构体\n")

            # ---- 模式2: 碳数 + 环数 + 双键数 + 三键数 ----
            elif len(parts) == 4:
                n_c, n_r, n_d, n_t = map(int, parts)
                k = n_r + n_d + 2 * n_t
                isomers = gen.generate(n_c, n_r, n_d, n_t)
                _print_isomers(isomers, n_r, n_d, n_t, n_c, k)
                print()

            else:
                print("  格式: n k (不饱和度) 或 n r d t (精确指定)")

        except ValueError:
            print("  ⚠ 请输入有效数字")
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break
