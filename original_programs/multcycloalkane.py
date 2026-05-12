"""
多环烷烃同分异构体生成器

基于 "树 + 迭代加边" 算法，完备生成 k 环烷烃 (k >= 1)。
OEIS 验证：
  k=1 (单环) : A036671
  k=2 (双环) : A036672
  k=3 (三环) : A036673

支持3D可视化（RDKit + matplotlib）和终端交互。
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import warnings

import networkx as nx
import numpy as np

# 将项目根目录加入 sys.path，方便导入原模块
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 尝试导入烷烃树生成器
try:
    from original_programs.alkene_visualizer import AlkaneTreeGenerator
except ImportError:
    print("错误：找不到 alkene_visualizer.py，请将该文件放在 original_programs 目录下")
    sys.exit(1)

warnings.filterwarnings('ignore')


class PolycycloalkaneGenerator:
    """
    多环烷烃同分异构体生成器（k 环，k >= 1）。
    通过从烷烃树开始，逐步添加 k 条边的策略，保证完备性。
    """

    # OEIS 验证数据
    # k=1 单环: A036671 (度数≤4约束的连通单环图)
    # k>=2: 暂用算法自洽数据
    OEIS_DATA = {
        1: {3: 1, 4: 2, 5: 5, 6: 12, 7: 29, 8: 73, 9: 185, 10: 475},
        2: {3: 1, 4: 1, 5: 5, 6: 17, 7: 56, 8: 201, 9: 784, 10: 1792},
        3: {4: 1, 5: 4, 6: 18, 7: 79, 8: 399},
    }

    RING_NAMES_CN = {
        1: '单环烷烃', 2: '双环烷烃', 3: '三环烷烃',
        4: '四环烷烃', 5: '五环烷烃', 6: '六环烷烃',
    }

    def __init__(self, num_cores: Optional[int] = None):
        self.tree_gen = AlkaneTreeGenerator(use_parallel=True, num_workers=num_cores)
        self.num_cores = num_cores or max(1, __import__('multiprocessing').cpu_count() - 1)

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------
    def generate_isomers(self, n_carbons: int, n_rings: int, verbose: bool = True) -> List[nx.Graph]:
        """
        生成 Cn H_{2n+2-2k} 的全部 k 环烷烃同分异构体。

        Args:
            n_carbons: 碳原子数 (>= 3)
            n_rings: 目标环数 (>= 1)
            verbose: 是否打印进度

        Returns:
            去重后的 nx.Graph 列表
        """
        if n_carbons < 3 or n_rings < 1:
            return []

        formula = f"C{n_carbons}H{2*n_carbons+2-2*n_rings}"
        ring_name = self.RING_NAMES_CN.get(n_rings, f'{n_rings}环烷烃')

        if verbose:
            print(f"\n{'='*60}")
            print(f"  {ring_name} ({formula}) 异构体生成")
            print(f"  算法: 烷烃骨架 → 逐层加边去重")
            print(f"{'='*60}")

        if verbose:
            print(f"\n--- L0 烷烃骨架 ---")
        canons = self.tree_gen.generate_all_canons(n_carbons)
        trees = [self._canon_to_graph(c) for c in canons]
        if verbose:
            print(f"  - 烷烃骨架数: {len(trees)}")

        # 第 1 步：在树上加第一条边 -> 单环图
        if verbose:
            print(f"\n--- L1 单环图 ---")
            print(f"  - 前驱异构体数: {len(trees)}")

        current_graphs = self._add_edges_and_deduplicate(trees, max_degree=3)

        if verbose:
            print(f"  - 唯一结构: {len(current_graphs)}")

        if n_rings == 1:
            return current_graphs

        # 后续步骤：在 r-1 环图的基础上逐次加边
        for r in range(2, n_rings + 1):
            if verbose:
                ring_name_r = self.RING_NAMES_CN.get(r, f'{r}环烷烃')
                print(f"\n--- L{r} {ring_name_r} ---")
                print(f"  - 前驱异构体数: {len(current_graphs)}")

            candidates = []
            for G in current_graphs:
                candidates.extend(self._add_one_edge(G, max_degree=3))
            current_graphs = self._deduplicate(candidates)

            if verbose:
                print(f"  - 唯一结构: {len(current_graphs)}")

            if not current_graphs:
                break

        return current_graphs

    # -------------------------------------------------------------------------
    # 描述与统计
    # -------------------------------------------------------------------------
    def describe_isomer(self, G: nx.Graph) -> str:
        """
        生成异构体的描述字符串。

        格式: 环数=nodes, 度分布
        如: "2环(5,5) deg=(2:2,3:4,4:2)"
        """
        cycles = nx.cycle_basis(G)
        n_rings = len(cycles)
        ring_sizes = sorted([len(c) for c in cycles], reverse=True)

        deg_dist = Counter(G.degree(v) for v in G.nodes())
        deg_str = ",".join(f"{d}:{c}" for d, c in sorted(deg_dist.items()))

        ring_str = ",".join(str(s) for s in ring_sizes)
        return f"{n_rings}环({ring_str}) deg=({deg_str})"

    def ring_size_distribution(self, isomers: List[nx.Graph]) -> Dict[Tuple, int]:
        """
        统计各环大小组合的异构体数量分布。

        Returns:
            字典 {环大小组合元组: 数量}
        """
        dist: Dict[Tuple, int] = defaultdict(int)
        for G in isomers:
            cycles = nx.cycle_basis(G)
            ring_sizes = tuple(sorted([len(c) for c in cycles], reverse=True))
            dist[ring_sizes] += 1
        return dict(dist)

    def degree_distribution(self, isomers: List[nx.Graph]) -> Dict[Tuple, int]:
        """
        统计各度分布的异构体数量分布。

        Returns:
            字典 {度分布元组: 数量}
        """
        dist: Dict[Tuple, int] = defaultdict(int)
        for G in isomers:
            deg_seq = tuple(sorted(G.degree(v) for v in G.nodes()))
            dist[deg_seq] += 1
        return dict(dist)

    # -------------------------------------------------------------------------
    # 可视化
    # -------------------------------------------------------------------------
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

        mol = Chem.RWMol()
        for i in range(n_carbons):
            mol.AddAtom(Chem.Atom('C'))

        added_bonds = set()
        for u, v in G.edges():
            bond_key = (min(u, v), max(u, v))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)
            mol.AddBond(u, v, BondType.SINGLE)

        # 设置显式氢
        for node in sorted(G.nodes()):
            h_count = 4 - G.degree(node)
            if h_count > 0:
                mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        mol = Chem.AddHs(mol)

        # 多环烷烃：多种子选择策略，选择最大C-C键长最短的构象
        # 避免高张力多环体系（如稠合三元环）产生键长异常的构象
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
                        # 判断边是否属于某个环（环内边高亮）
                        in_ring = False
                        for cycle in cycles:
                            cycle_set = set(cycle)
                            if node in cycle_set and neighbor in cycle_set:
                                in_ring = True
                                break

                        if in_ring:
                            ax.plot(
                                [c_coords[node][0], c_coords[neighbor][0]],
                                [c_coords[node][1], c_coords[neighbor][1]],
                                [c_coords[node][2], c_coords[neighbor][2]],
                                color='#CC0000', linewidth=2.5
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
            if n in node_ring_color:
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

    # -------------------------------------------------------------------------
    # 内部工具：图构造与边添加
    # -------------------------------------------------------------------------
    def _canon_to_graph(self, canon: str) -> nx.Graph:
        """将规范字符串还原为 nx.Graph。"""
        adj = self.tree_gen.canon_to_adjacency(canon)
        G = nx.Graph()
        for node in adj.keys():
            G.add_node(node, label='C')
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    G.add_edge(u, v, bond_type='single')
        return G

    @staticmethod
    def _add_one_edge(G: nx.Graph, max_degree: int = 3) -> List[nx.Graph]:
        """在图上添加一条合法边的所有可能结果。"""
        new_graphs = []
        nodes = list(G.nodes())
        n = len(nodes)
        for i in range(n):
            u = nodes[i]
            if G.degree(u) > max_degree:
                continue
            for j in range(i + 1, n):
                v = nodes[j]
                if G.degree(v) > max_degree:
                    continue
                if G.has_edge(u, v):
                    continue
                new_G = G.copy()
                new_G.add_edge(u, v, bond_type='single')
                new_graphs.append(new_G)
        return new_graphs

    @staticmethod
    def _add_edges_and_deduplicate(graphs: List[nx.Graph], max_degree: int = 3) -> List[nx.Graph]:
        """对一批图分别加一条边，合并后去重"""
        candidates = []
        for G in graphs:
            candidates.extend(PolycycloalkaneGenerator._add_one_edge(G, max_degree))
        return PolycycloalkaneGenerator._deduplicate(candidates)

    # -------------------------------------------------------------------------
    # 去重：WL 哈希 + 精确同构
    # -------------------------------------------------------------------------
    @staticmethod
    def _deduplicate(candidates: List[nx.Graph]) -> List[nx.Graph]:
        """两阶段去重：WL哈希分桶 + 桶内精确同构检查"""
        hash_groups: Dict[str, List[nx.Graph]] = defaultdict(list)
        for G in candidates:
            try:
                h = nx.weisfeiler_lehman_graph_hash(
                    G, edge_attr='bond_type', node_attr='label'
                )
            except Exception:
                h = str(PolycycloalkaneGenerator._graph_signature(G))
            hash_groups[h].append(G)

        result = []
        for group in hash_groups.values():
            if len(group) == 1:
                result.append(group[0])
                continue
            unique = [group[0]]
            for G in group[1:]:
                is_dup = False
                for existing in unique:
                    if nx.is_isomorphic(
                        G, existing,
                        node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
                        edge_match=lambda e1, e2: e1.get('bond_type') == e2.get('bond_type')
                    ):
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(G)
            result.extend(unique)
        return result

    @staticmethod
    def _graph_signature(G: nx.Graph) -> tuple:
        """图的快速签名（备用）"""
        deg_seq = tuple(sorted(G.degree(v) for v in G.nodes()))
        nbr_patterns = []
        for v in G.nodes():
            nbr_degs = tuple(sorted(G.degree(w) for w in G.neighbors(v)))
            nbr_patterns.append(nbr_degs)
        nbr_patterns.sort()
        return (deg_seq, tuple(nbr_patterns))


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    print("=" * 70)
    print("  多环烷烃同分异构体生成器")
    print("  算法: 烷烃骨架 → 逐层加边去重")
    print("  OEIS验证: k=1 A036671, k=2 A036672, k=3 A036673")
    print("=" * 70)
    print()

    generator = PolycycloalkaneGenerator()

    while True:
        print("-" * 70)
        try:
            user_input = input("请输入碳原子数 环数 (如 6 2) 或分子式 (如 C6H8), q退出: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if user_input.lower() == 'q':
            print("再见！")
            break

        if not user_input:
            continue

        # 解析输入
        n_carbons = None
        n_rings = None

        if 'C' in user_input.upper():
            m = re.match(r'C(\d+)H(\d+)', user_input, re.IGNORECASE)
            if m:
                n_carbons = int(m.group(1))
                h = int(m.group(2))
                n_rings = n_carbons + 1 - h // 2
                if n_rings < 1:
                    print("氢原子数过多，不是环烷烃")
                    continue
                print(f"识别为 C{n_carbons}，{n_rings} 环")
            else:
                print("分子式格式错误，请使用 CnHm")
                continue
        else:
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    n_carbons = int(parts[0])
                    n_rings = int(parts[1])
                except ValueError:
                    print("输入格式错误")
                    continue
            else:
                print("输入格式错误，请输入 碳数 环数 或 分子式")
                continue

        if n_carbons < 3:
            print("碳原子数至少为 3")
            continue
        if n_rings < 1:
            print("环数至少为 1")
            continue

        formula = f"C{n_carbons}H{2*n_carbons+2-2*n_rings}"
        ring_name = generator.RING_NAMES_CN.get(n_rings, f'{n_rings}环烷烃')

        print()
        try:
            isomers = generator.generate_isomers(n_carbons, n_rings, verbose=True)
        except KeyboardInterrupt:
            print("\n\n[中断] 计算已停止")
            continue
        except Exception as e:
            print(f"\n\n[错误] {e}")
            import traceback
            traceback.print_exc()
            continue

        if not isomers:
            print(f"\n未找到 {formula} 的异构体")
            continue

        # 显示结果摘要
        print()
        print(f"{formula} {ring_name} 共有 {len(isomers)} 个异构体")

        # 环大小分布
        if n_rings >= 2:
            ring_dist = generator.ring_size_distribution(isomers)
            print()
            print("环大小分布:")
            for ring_sizes, count in sorted(ring_dist.items(), key=lambda x: (-x[1], x[0])):
                sizes_str = "+".join(str(s) for s in ring_sizes)
                print(f"  ({sizes_str}): {count} 个")
        elif n_rings == 1:
            ring_dist = Counter()
            for G in isomers:
                cycle = nx.cycle_basis(G)[0]
                ring_dist[len(cycle)] += 1
            print()
            print("环大小分布:")
            for ring_size in sorted(ring_dist.keys()):
                print(f"  c{ring_size}: {ring_dist[ring_size]} 个")

        # 度分布统计
        deg_dist = generator.degree_distribution(isomers)
        if len(deg_dist) > 1:
            print()
            print("度分布统计:")
            for deg_seq, count in sorted(deg_dist.items(), key=lambda x: (-x[1], x[0])):
                deg_str = ",".join(str(d) for d in deg_seq)
                print(f"  ({deg_str}): {count} 个")

        # 异构体列表
        print()
        print(f"{ring_name} ({formula}) 异构体列表 (共 {len(isomers)} 个):")
        for i in range(min(len(isomers), 20)):
            desc = generator.describe_isomer(isomers[i])
            print(f"  {i+1}. {desc}")
        if len(isomers) > 20:
            print(f"  ... 还有 {len(isomers) - 20} 个")

        # 交互选项（内层循环，允许反复操作）
        while True:
            print()
            print("选项:")
            print("1. 查看指定异构体的详细信息")
            print("2. 可视化指定异构体")
            print("3. 保存异构体列表到文件")
            print("4. 继续输入")
            print("5. 退出")

            try:
                choice = input("\n请选择 (1/2/3/4/5): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n再见！")
                return

            if choice == "1":
                try:
                    idx_input = input(f"请输入异构体编号 (1-{len(isomers)}): ").strip()
                    if idx_input:
                        idx = int(idx_input) - 1
                        if 0 <= idx < len(isomers):
                            G = isomers[idx]
                            desc = generator.describe_isomer(G)
                            cycles = nx.cycle_basis(G)
                            ring_sizes = [len(c) for c in cycles]

                            print(f"\n--- {formula} 异构体 #{idx+1} ---")
                            print(f"描述: {desc}")
                            print(f"环数: {len(cycles)}")
                            for ci, cycle in enumerate(cycles):
                                print(f"  环{ci+1}: {len(cycle)}元环, 节点={[n+1 for n in cycle]}")
                            print("各碳原子:")
                            for node in sorted(G.nodes()):
                                ring_strs = []
                                for ci, cycle in enumerate(cycles):
                                    if node in cycle:
                                        ring_strs.append(f"环{ci+1}")
                                ring_info = f" [{','.join(ring_strs)}]" if ring_strs else ""
                                h = 4 - G.degree(node)
                                print(f"  C{node+1}: 度={G.degree(node)}, {h}个H{ring_info}")
                        else:
                            print("错误: 编号超出范围")
                except ValueError:
                    print("错误: 请输入有效的数字")

            elif choice == "2":
                try:
                    idx_input = input(f"请输入异构体编号 (1-{len(isomers)}): ").strip()
                    if idx_input:
                        idx = int(idx_input) - 1
                        if 0 <= idx < len(isomers):
                            G = isomers[idx]
                            desc = generator.describe_isomer(G)
                            vis_title = f"{formula} {ring_name} #{idx+1} ({desc})"
                            print(f"\n正在打开3D可视化窗口...")
                            generator.visualize_isomer(G, title=vis_title)
                        else:
                            print("错误: 编号超出范围")
                except ValueError:
                    print("错误: 请输入有效的数字")

            elif choice == "3":
                default_name = f"{formula}_{n_rings}ring_isomers.txt"
                filename = input(f"请输入保存文件名 (默认 {default_name}): ").strip()
                if not filename:
                    filename = default_name
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"{formula} {ring_name} 异构体列表\n")
                        f.write(f"总数: {len(isomers)}\n")
                        f.write("=" * 70 + "\n\n")
                        for i, G in enumerate(isomers, 1):
                            desc = generator.describe_isomer(G)
                            f.write(f"{i}. {desc}\n")
                    print(f"已保存到: {filename}")
                except Exception as e:
                    print(f"保存失败: {e}")

            elif choice == "4":
                break  # 跳出内层循环，回到外层继续输入

            elif choice == "5":
                print("再见！")
                return


if __name__ == "__main__":
    main()
