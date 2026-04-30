"""
单环多烯烃 (Cyclopolyene) 同分异构体生成工具

功能: 统一生成任意碳数 n 的单环k烯烃 (k=0,1,2,...)
算法: 逐层递推 - 从环烷烃骨架出发，逐层在合法单键边上插入双键
通式: CnH(2n-2k)，不饱和度=k+1（环贡献1 + 每个双键贡献1）

使用示例:
  gen = CyclopolyeneGenerator()
  gen.generate_cycloalkene(6)    # C6H10 单环单烯烃
  gen.generate_cyclo_diene(6)   # C6H8 单环二烯烃
  gen.generate_k_cycloene(6, 3) # C6H6 单环三烯烃
  gen.generate_all_layers(6)     # 生成所有层级
"""

from typing import List, Dict, Tuple
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

# 确保 original_programs 可以被正确导入
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 尝试导入 tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class CyclopolyeneGenerator:
    """
    单环多烯烃同分异构体统一生成器

    算法: 环烷烃骨架 → 逐层插入双键去重 → 分子通式CnH(2n-2k)
    约束: 每个碳价键数≤4, sp²碳度≤3, 累积二烯碳度≤2
    去重: WL哈希分组 + 图同构精确验证
    """

    # 已验证参考数据
    VERIFIED_CYCLOALKANE = {3: 1, 4: 2, 5: 5, 6: 12, 7: 29, 8: 73, 9: 185, 10: 475}
    VERIFIED_CYCLOALKENE = {3: 1, 4: 4, 5: 12, 6: 37, 7: 108, 8: 320, 9: 929, 10: 2704}
    VERIFIED_CYCLO_DIENE = {}

    def __init__(self, cycloalkane_generator=None):
        """
        Args:
            cycloalkane_generator: 环烷烃生成器实例，None则使用默认实例
        """
        if cycloalkane_generator is not None:
            self.cycloalkane_gen = cycloalkane_generator
        else:
            from original_programs.multcycloalkane import PolycycloalkaneGenerator
            self.cycloalkane_gen = PolycycloalkaneGenerator()

    def generate_all_layers(self, n_carbons: int, max_double_bonds: int = None,
                            verbose: bool = True) -> Dict[int, List[nx.Graph]]:
        """
        生成所有层级的单环多烯烃异构体

        Args:
            n_carbons: 碳原子数
            max_double_bonds: 最大双键数，None表示自动计算
            verbose: 是否打印进度信息

        Returns:
            字典 {双键数k: 异构体列表}，k=0为环烷烃
        """
        if n_carbons < 3:
            return {}

        if max_double_bonds is None:
            max_double_bonds = n_carbons - 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"逐层递推生成 C{n_carbons} 系列单环多烯烃异构体")
            print(f"{'='*60}")

        if verbose:
            print(f"\n--- L0 环烷烃骨架 (C{n_carbons}H{2*n_carbons}) ---")

        cycloalkane_graphs = self._get_cycloalkane_graphs(n_carbons)
        results = {0: cycloalkane_graphs}

        if verbose:
            verified_count = len(cycloalkane_graphs)
            expected = self.VERIFIED_CYCLOALKANE.get(n_carbons)
            status = ""
            if expected is not None:
                status = " [OK]" if verified_count == expected else f" [FAIL](expected {expected})"
            print(f"  - 环烷烃异构体数: {verified_count}{status}")

        current_layer = cycloalkane_graphs

        for k in range(1, max_double_bonds + 1):
            formula = f"C{n_carbons}H{2*n_carbons - 2*k}"
            name = self._chinese_name(k)

            if verbose:
                print(f"\n--- L{k} {name} ({formula}) ---")
                print(f"  - 前驱异构体数: {len(current_layer)}")

            next_layer = self._insert_double_bonds(current_layer, k)

            if not next_layer:
                if verbose:
                    print(f"  - 候选结构: 0")
                    print(f"  - 唯一结构: 0 (终止递推)")
                break

            results[k] = next_layer

            if verbose:
                verified = self._get_verified_count(k)
                status = ""
                if n_carbons in verified:
                    expected = verified[n_carbons]
                    status = " [OK]" if len(next_layer) == expected else f" [FAIL](expected {expected})"
                else:
                    status = " (待验证)"
                print(f"  - 唯一结构: {len(next_layer)}{status}")

            current_layer = next_layer

        return results

    def generate_cycloalkene(self, n_carbons: int) -> List[nx.Graph]:
        """生成单环单烯烃异构体 (CnH2n-2)"""
        return self.generate_k_cycloene(n_carbons, 1)

    def generate_cyclo_diene(self, n_carbons: int) -> List[nx.Graph]:
        """生成单环二烯烃异构体 (CnH2n-4)"""
        return self.generate_k_cycloene(n_carbons, 2)

    def generate_cyclo_triene(self, n_carbons: int) -> List[nx.Graph]:
        """生成单环三烯烃异构体 (CnH2n-6)"""
        return self.generate_k_cycloene(n_carbons, 3)

    def generate_k_cycloene(self, n_carbons: int, k: int) -> List[nx.Graph]:
        """
        生成单环k-烯烃异构体

        Args:
            n_carbons: 碳原子数
            k: 双键数

        Returns:
            唯一异构体图列表
        """
        if n_carbons < 3:
            return []
        if k < 1:
            return self._get_cycloalkane_graphs(n_carbons)

        current_layer = self._get_cycloalkane_graphs(n_carbons)

        for layer_idx in range(1, k + 1):
            current_layer = self._insert_double_bonds(current_layer, layer_idx)
            if not current_layer:
                return []

        return current_layer

    def _insert_double_bonds(self, prev_layer: List[nx.Graph],
                             layer_k: int) -> List[nx.Graph]:
        """
        在第(k-1)层异构体的合法单键边上插入第k个双键

        Args:
            prev_layer: 第(k-1)层的异构体列表
            layer_k: 当前层级

        Returns:
            第k层的唯一异构体列表
        """
        hash_groups = {}
        total_candidates = 0

        if HAS_TQDM:
            iterator = tqdm(prev_layer, desc=f"L{layer_k}递推", unit="异构体")
        else:
            iterator = prev_layer

        for G in iterator:
            edges = list(G.edges())

            for u, v in edges:
                bond_type = G[u][v].get('bond_type', 'single')
                if bond_type != 'single':
                    continue

                if not self._can_insert_double_bond(G, u, v):
                    continue

                H = self._upgrade_edge_to_double(G, u, v)

                if not self._validate_molecule(H):
                    continue

                total_candidates += 1

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

        unique = []
        for group in hash_groups.values():
            unique.extend(group)

        return unique

    def _can_insert_double_bond(self, G: nx.Graph, u: int, v: int) -> bool:
        """
        检查边(u,v)是否可以插入双键（兼容环结构中桥头碳等情形）

        Args:
            G: 当前图
            u, v: 待升级边的端点

        Returns:
            是否可以放置
        """
        for node in [u, v]:
            bond_load = self._calculate_bond_load(G, node)
            new_load = bond_load + 1
            if new_load > 4:
                return False

        for node in [u, v]:
            deg = G.degree(node)
            existing_double_bonds = sum(
                1 for nb in G.neighbors(node)
                if G[node][nb].get('bond_type', 'single') == 'double'
            )
            new_double_bonds = existing_double_bonds + 1

            # 累积二烯: 同一碳参与≥2个双键时度数 ≤ 2
            if new_double_bonds >= 2:
                if deg > 2:
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
        将边(u,v)从单键升级为双键

        Args:
            G: 原始图
            u, v: 待升级边的端点

        Returns:
            新图
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

    def _validate_molecule(self, G: nx.Graph) -> bool:
        """
        验证分子结构的化学有效性（兼容环结构中桥头碳等情形）

        Returns:
            是否化学有效
        """
        for node in G.nodes():
            bond_load = self._calculate_bond_load(G, node)
            deg = G.degree(node)

            if bond_load > 4:
                return False

            double_bond_count = sum(
                1 for nb in G.neighbors(node)
                if G[node][nb].get('bond_type', 'single') == 'double'
            )

            # 累积二烯: 同一碳参与≥2个双键时度数 ≤ 2
            if double_bond_count >= 2:
                if deg > 2:
                    return False

        return True

    def _is_isomorphic(self, G1: nx.Graph, G2: nx.Graph) -> bool:
        """检查两图是否同构（考虑键类型和节点标签）"""
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

    def _get_cycloalkane_graphs(self, n_carbons: int) -> List[nx.Graph]:
        """
        获取环烷烃骨架图列表

        Args:
            n_carbons: 碳原子数

        Returns:
            nx.Graph列表（含bond_type属性）
        """
        cycloalkane_graphs = self.cycloalkane_gen.generate_isomers(n_carbons, n_rings=1, verbose=False)

        graphs = []
        for G in cycloalkane_graphs:
            H = nx.Graph()
            for node in G.nodes():
                H.add_node(node, label='C')
            for a, b in G.edges():
                H.add_edge(a, b, bond_type='single')
            graphs.append(H)

        return graphs

    @staticmethod
    def _chinese_name(k: int) -> str:
        """k-烯烃的中文名称"""
        names = {1: '单环烯烃', 2: '单环二烯烃', 3: '单环三烯烃', 4: '单环四烯烃',
                 5: '单环五烯烃', 6: '单环六烯烃', 7: '单环七烯烃', 8: '单环八烯烃'}
        return names.get(k, f'单环{k}烯烃')

    def _get_verified_count(self, k: int) -> Dict[int, int]:
        """获取k-单环烯烃的已验证数据"""
        verified_map = {
            0: self.VERIFIED_CYCLOALKANE,
            1: self.VERIFIED_CYCLOALKENE,
            2: self.VERIFIED_CYCLO_DIENE,
        }
        return verified_map.get(k, {})

    def describe_isomer(self, G: nx.Graph) -> str:
        """
        生成异构体描述字符串

        Args:
            G: 单环多烯烃图对象

        Returns:
            描述字符串
        """
        double_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('bond_type') == 'double']

        cycles = nx.cycle_basis(G)
        if not cycles:
            return "unknown"

        main_cycle = cycles[0]
        cycle_len = len(main_cycle)
        cycle_set = set(main_cycle)

        ring_double = []
        exo_double = []
        for u, v in double_edges:
            if u in cycle_set and v in cycle_set:
                ring_double.append((u, v))
            else:
                exo_double.append((u, v))

        ring_desc = f"c{cycle_len}"
        if ring_double:
            ring_desc += f"={len(ring_double)}"

        sub_parts = []
        for v in main_cycle:
            for nb in G.neighbors(v):
                if nb not in cycle_set:
                    is_double = any(
                        (u, w) in exo_double or (w, u) in exo_double
                        for u, w in [(v, nb), (nb, v)]
                    )
                    nc = self._count_subtree(G, nb, cycle_set)
                    if is_double:
                        sub_parts.append(f"=C{nc}")
                    else:
                        sub_parts.append(self._short_alkyl_name(nc))

        if sub_parts:
            ring_desc += "(" + ",".join(sub_parts) + ")"

        h_info = []
        for node in sorted(G.nodes()):
            total_bonds = self._calculate_bond_load(G, node)
            h_count = 4 - total_bonds
            h_info.append(f"C{node+1}({h_count}H)")

        double_count = len(double_edges)
        name = self._chinese_name(double_count)
        return f"{name}: {ring_desc} [" + ", ".join(h_info) + "]"

    def classify_cyclopolyene(self, G: nx.Graph) -> str:
        """
        分类单环多烯烃类型

        Returns:
            cumulated/conjugated/isolated的组合
        """
        double_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('bond_type') == 'double']

        if len(double_edges) <= 1:
            return 'monoene'

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
                    ni = double_nodes[i]
                    nj = double_nodes[j]
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

    def _count_subtree(self, G: nx.Graph, root: int, forbidden: set) -> int:
        """计算子树节点数（不进入forbidden集合）"""
        visited = set(forbidden)
        queue = [root]
        count = 0
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            count += 1
            for nb in G.neighbors(node):
                if nb not in visited:
                    queue.append(nb)
        return count

    @staticmethod
    def _short_alkyl_name(n_carbons: int) -> str:
        """获取烷基缩写名"""
        names = {1: "Me", 2: "Et", 3: "Pr", 4: "Bu"}
        return names.get(n_carbons, f"C{n_carbons}")

    def graph_to_rdkit_mol(self, G: nx.Graph, optimize: bool = True):
        """
        转换为RDKit Mol对象（含3D坐标和MMFF优化）

        Args:
            G: 图对象（含bond_type边属性）
            optimize: 是否进行MMFF力场优化

        Returns:
            rdkit.Chem.rdchem.Mol对象，失败返回None
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            return None

        n_carbons = G.number_of_nodes()

        double_bond_edges = set()
        for u, v, data in G.edges(data=True):
            if data.get('bond_type') == 'double':
                double_bond_edges.add((min(u, v), max(u, v)))

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
        embed_ok = result != -1
        if not embed_ok:
            result2 = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            embed_ok = result2 != -1

        if not embed_ok:
            return None

        if optimize:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass

        return mol

    def visualize_isomer(self, G: nx.Graph, title: str = "") -> bool:
        """
        使用RDKit+matplotlib弹出3D可视化窗口

        Args:
            G: 图对象
            title: 窗口标题

        Returns:
            True成功，False失败
        """
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            import numpy as np
            from rdkit import Chem
        except ImportError as e:
            print(f"可视化依赖缺失: {e}")
            return False

        mol = self.graph_to_rdkit_mol(G, optimize=True)
        if mol is None:
            print("RDKit 分子构建失败")
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

        double_bond_edges = set()
        double_nodes = set()
        for u, v, d in G.edges(data=True):
            if d.get('bond_type') == 'double':
                key = (min(u, v), max(u, v))
                double_bond_edges.add(key)
                double_nodes.add(u)
                double_nodes.add(v)

        cycles = nx.cycle_basis(G)
        cycle_set = set(cycles[0]) if cycles else set()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=14, fontweight='bold')

        drawn_bonds = set()
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                bond_key = tuple(sorted([node, neighbor]))
                if bond_key not in drawn_bonds:
                    if node in c_coords and neighbor in c_coords:
                        is_double = bond_key in double_bond_edges
                        if is_double:
                            p1 = np.array(c_coords[node])
                            p2 = np.array(c_coords[neighbor])
                            direction = p2 - p1
                            length = np.linalg.norm(direction)
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

        for n in sorted(c_coords.keys()):
            if n in double_nodes:
                color = '#FF6600'
            elif n in cycle_set:
                color = 'blue'
            else:
                color = '#333333'
            ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                       c=color, s=150, zorder=5)

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

        if cycles:
            for i, node in enumerate(cycles[0]):
                if node in c_coords:
                    coord = c_coords[node]
                    ax.text(coord[0], coord[1], coord[2], f'{i+1}', fontsize=9, color='blue')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FF6600', linewidth=3, label='C=C'),
            Line2D([0], [0], color='black', linewidth=2, label='C-C'),
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


def main():
    """交互式命令行入口"""
    print("=" * 70)
    print("单环多烯烃同分异构体生成器")
    print("分子通式: CnH(2n-2k)，k=0,1,2,... (不饱和度=k+1)")
    print("验证数据:")
    print("  环烷烃 (k=0): C3=1, C4=2, C5=5, C6=12, C7=29, C8=73")
    print("  单环烯烃 (k=1): C3=1, C4=4, C5=12, C6=37, C7=108, C8=320")
    print("=" * 70)
    print()

    generator = CyclopolyeneGenerator()

    while True:
        try:
            user_input = input("请输入碳原子数 (n>=3, 输入 q 退出): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if user_input.lower() == 'q':
            print("再见！")
            break

        try:
            n = int(user_input)
        except ValueError:
            print("错误: 请输入有效的数字")
            continue

        if n < 3:
            print("碳原子数必须 >= 3")
            return

        print()

        try:
            results = generator.generate_all_layers(n, verbose=True)
        except KeyboardInterrupt:
            print("\n\n[中断] 计算已停止")
            continue
        except Exception as e:
            print(f"\n\n[错误] 计算过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not results:
            print(f"未找到 C{n} 的异构体")
            continue

        print()
        for k in sorted(results.keys()):
            isomers = results[k]
            formula = f"C{n}H{2*n - 2*k}"
            name = generator._chinese_name(k)
            verified = generator._get_verified_count(k)
            expected = verified.get(n)
            if expected is not None:
                status = "[OK]" if len(isomers) == expected else "[FAIL]"
                print(f"  L{k} {name} ({formula}): {len(isomers)} 个 (预期 {expected}) {status}")
            else:
                print(f"  L{k} {name} ({formula}): {len(isomers)} 个")

        print()
        print("双键类型分布:")
        for k in sorted(results.keys()):
            isomers = results[k]
            if k == 0:
                continue
            ring_only = 0
            exo_only = 0
            mixed = 0
            for G in isomers:
                cycles = nx.cycle_basis(G)
                cycle_set = set(cycles[0]) if cycles else set()
                ring_double = 0
                exo_double = 0
                for u, v, d in G.edges(data=True):
                    if d.get('bond_type') == 'double':
                        if u in cycle_set and v in cycle_set:
                            ring_double += 1
                        else:
                            exo_double += 1
                if ring_double > 0 and exo_double == 0:
                    ring_only += 1
                elif exo_double > 0 and ring_double == 0:
                    exo_only += 1
                else:
                    mixed += 1
            name = generator._chinese_name(k)
            parts = []
            if ring_only > 0:
                parts.append(f"环内双键={ring_only}")
            if exo_only > 0:
                parts.append(f"环外双键={exo_only}")
            if mixed > 0:
                parts.append(f"混合={mixed}")
            print(f"  L{k} {name}: {', '.join(parts)}")

        print()
        for k in sorted(results.keys()):
            isomers = results[k]
            name = generator._chinese_name(k)
            formula = f"C{n}H{2*n - 2*k}"
            print(f"L{k} {name} ({formula}) 异构体列表 (共 {len(isomers)} 个):")
            for i in range(min(len(isomers), 10)):
                desc = generator.describe_isomer(isomers[i])
                if k >= 2:
                    cls = generator.classify_cyclopolyene(isomers[i])
                    print(f"  {i+1}. {desc}  [{cls}]")
                else:
                    print(f"  {i+1}. {desc}")
            if len(isomers) > 10:
                print(f"  ... 还有 {len(isomers) - 10} 个")
            print()

        total_all = sum(len(v) for v in results.values())
        print(f"共生成 {total_all} 个异构体，分布在 {len(results)} 个层级")

        print("\n选项:")
        print("1. 列出指定层级的全部异构体")
        print("2. 可视化指定异构体")
        print("3. 退出")

        try:
            choice = input("\n请选择 (1/2/3): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if choice == "1":
            try:
                k_input = input(f"请输入层级k (0-{max(results.keys())}): ").strip()
                if k_input:
                    k = int(k_input)
                    if k in results:
                        isomers = results[k]
                        name = generator._chinese_name(k)
                        formula = f"C{n}H{2*n - 2*k}"
                        print(f"\n{name} ({formula}) 异构体列表 (共 {len(isomers)} 个):")
                        for i, G in enumerate(isomers):
                            desc = generator.describe_isomer(G)
                            if k >= 2:
                                cls = generator.classify_cyclopolyene(G)
                                print(f"  {i+1}. {desc}  [{cls}]")
                            else:
                                print(f"  {i+1}. {desc}")
                    else:
                        print("错误: 层级超出范围")
            except ValueError:
                print("错误: 请输入有效的数字")

        elif choice == "2":
            try:
                k_input = input(f"请输入层级k (0-{max(results.keys())}): ").strip()
                i_input = input(f"请输入异构体编号: ").strip()
                if k_input and i_input:
                    k = int(k_input)
                    i = int(i_input)
                    if k in results and 1 <= i <= len(results[k]):
                        G = results[k][i-1]
                        desc = generator.describe_isomer(G)
                        formula = f"C{n}H{2*n - 2*k}"
                        print(f"\n正在显示: {desc}")
                        vis_title = f"{formula} #{i} ({desc})"
                        generator.visualize_isomer(G, title=vis_title)
                    else:
                        print("错误: 编号超出范围")
            except ValueError:
                print("错误: 请输入有效的数字")

        else:
            print("再见！")
            break


if __name__ == "__main__":
    main()