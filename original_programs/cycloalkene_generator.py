"""
单环烯烃 (Cycloalkene, CnH2n-2) 同分异构体生成工具。

核心算法：环烷烃骨架 + 遍历加双键 → WL哈希分组 + 图同构去重。
验证数据：C3=1, C4=4, C5=12, C6=37, C7=108, C8=320, C9=929, C10=2704。
示例: CycloalkeneGenerator().generate_isomers(6) → C6H10 单环烯烃列表
"""

from typing import List, Dict, Optional, Tuple
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


class CycloalkeneGenerator:
    """
    单环烯烃 (Cycloalkene, CnH2n-2) 同分异构体生成器。
    核心：环烷烃骨架 + 一条双键，WL哈希 + 图同构去重。
    """
    # 验证数据（算法自洽）：C3=1, C4=4, C5=12, C6=37, C7=108, C8=320, C9=929, C10=2704
    OEIS_DATA = {
        3: 1, 4: 4, 5: 12, 6: 37, 7: 108, 8: 320, 9: 929, 10: 2704,
    }

    def __init__(self, cycloalkane_generator=None):
        """
        初始化生成器。

        Args:
            cycloalkane_generator: 可选的环烷烃生成器实例，为None时创建默认实例。
        """
        if cycloalkane_generator is not None:
            self.cycloalkane_gen = cycloalkane_generator
        else:
            from original_programs.multcycloalkane import PolycycloalkaneGenerator
            self.cycloalkane_gen = PolycycloalkaneGenerator()

    def generate_isomers(self, n_carbons: int) -> List[nx.Graph]:
        """
        生成 CnH2n-2 单环烯烃的所有结构异构体。

        步骤：获取环烷烃骨架 → 遍历边放双键 → WL哈希+同构去重。

        Args:
            n_carbons: 碳原子数 (n >= 3)

        Returns:
            去重后的 nx.Graph 列表，每个 Graph 含 bond_type 属性。
        """
        if n_carbons < 3:
            return []

        print(f"正在生成 C{n_carbons}H{2*n_carbons - 2} 单环烯烃...")

        # Step 1: 获取所有 n 碳环烷烃骨架
        cycloalkane_graphs = self._get_cycloalkane_graphs(n_carbons)
        print(f"  - 环烷烃骨架数: {len(cycloalkane_graphs)}")

        # Step 2: 对每个骨架图遍历边放置双键
        hash_groups = {}
        total_candidates = 0

        if HAS_TQDM:
            iterator = tqdm(cycloalkane_graphs, desc="生成候选", unit="骨架")
        else:
            iterator = cycloalkane_graphs

        for G in iterator:
            edges = list(G.edges())
            for u, v in edges:
                # 价键约束：加双键后该碳的总键数不超过4
                if G.degree[u] > 3 or G.degree[v] > 3:
                    continue

                total_candidates += 1

                # 构建候选图并标记双键
                H = nx.create_empty_copy(G)
                for node in H.nodes():
                    H.nodes[node]['label'] = 'C'
                for a, b in edges:
                    if (a, b) == (u, v) or (b, a) == (u, v):
                        H.add_edge(a, b, bond_type='double')
                    else:
                        H.add_edge(a, b, bond_type='single')

                # WL 哈希（对节点标签和边属性敏感）
                h = nx.weisfeiler_lehman_graph_hash(
                    H, edge_attr='bond_type', node_attr='label'
                )

                # 同哈希组内做精确同构比较
                if h not in hash_groups:
                    hash_groups[h] = [H]
                else:
                    dup = False
                    for existing in hash_groups[h]:
                        if nx.is_isomorphic(
                            H, existing,
                            node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
                            edge_match=lambda e1, e2: e1.get('bond_type') == e2.get('bond_type')
                        ):
                            dup = True
                            break
                    if not dup:
                        hash_groups[h].append(H)

        # 合并所有哈希组
        unique = []
        for group in hash_groups.values():
            unique.extend(group)

        print(f"  - 候选结构: {total_candidates}")
        print(f"  - 唯一结构: {len(unique)}")

        # 与参考数据对比
        expected = self.OEIS_DATA.get(n_carbons)
        if expected is not None:
            if len(unique) == expected:
                print(f"  [OK] OEIS verify pass: {len(unique)} == {expected}")
            else:
                print(f"  [FAIL] OEIS verify fail: {len(unique)} != {expected}")

        return unique

    def _get_cycloalkane_graphs(self, n_carbons: int) -> List[nx.Graph]:
        """
        获取 n 碳的环烷烃骨架图列表。

        Args:
            n_carbons: 碳原子数

        Returns:
            nx.Graph 列表，每个图表示一个环烷烃骨架。
        """
        return self.cycloalkane_gen.generate_isomers(n_carbons, n_rings=1, verbose=False)

    def graph_to_rdkit_mol(self, G: nx.Graph, optimize: bool = True) -> object:
        """
        将环烯烃图转换为 RDKit Mol 对象（含3D坐标 + MMFF力场优化）。

        Args:
            G: 环烯烃图对象（含 bond_type 边属性）
            optimize: 是否进行 MMFF 力场优化 (默认 True)

        Returns:
            rdkit.Chem.rdchem.Mol 对象（含3D conformer），失败返回 None。
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, BondType
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            return None

        n_carbons = G.number_of_nodes()

        # 提取双键边信息
        double_bond_edges = set()
        for u, v, data in G.edges(data=True):
            if data.get('bond_type') == 'double':
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

        # 设置显式氢数量
        for node in sorted(G.nodes()):
            total_bonds = 0
            for nb in G.neighbors(node):
                bond_type = G[node][nb].get('bond_type', 'single')
                total_bonds += 2 if bond_type == 'double' else 1
            h_count = 4 - total_bonds
            mol.GetAtomWithIdx(node).SetNumExplicitHs(h_count)

        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        mol = Chem.AddHs(mol)

        # 嵌入3D坐标
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
        """
        使用 RDKit + matplotlib 弹出3D可视化窗口。

        Args:
            G: 环烯烃图对象（含 bond_type 边属性）
            title: 窗口标题

        Returns:
            True 表示成功弹出窗口，False 表示失败。
        """
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            import numpy as np
            from rdkit import Chem
        except ImportError as e:
            print(f"可视化依赖缺失: {e}")
            print("请安装: pip install matplotlib rdkit")
            return False

        mol = self.graph_to_rdkit_mol(G, optimize=True)
        if mol is None:
            print("RDKit 分子构建失败，无法可视化")
            return False

        # 提取坐标
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
        main_cycle = nx.cycle_basis(G)[0]
        cycle_set = set(main_cycle)

        # 双键碳集合
        double_carbons = set()
        double_bond_edges = set()
        for u, v, d in G.edges(data=True):
            if d.get('bond_type') == 'double':
                double_carbons.add(u)
                double_carbons.add(v)
                double_bond_edges.add((min(u, v), max(u, v)))

        # 创建图形
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
                                perp = perp / np.linalg.norm(perp) * 0.15
                                ax.plot(
                                    [p1[0] + perp[0], p2[0] + perp[0]],
                                    [p1[1] + perp[1], p2[1] + perp[1]],
                                    [p1[2] + perp[2], p2[2] + perp[2]],
                                    color='red', linewidth=3
                                )
                                ax.plot(
                                    [p1[0] - perp[0], p2[0] - perp[0]],
                                    [p1[1] - perp[1], p2[1] - perp[1]],
                                    [p1[2] - perp[2], p2[2] - perp[2]],
                                    color='red', linewidth=3
                                )
                            else:
                                ax.plot(
                                    [c_coords[node][0], c_coords[neighbor][0]],
                                    [c_coords[node][1], c_coords[neighbor][1]],
                                    [c_coords[node][2], c_coords[neighbor][2]],
                                    color='red', linewidth=3
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
            if n in double_carbons:
                color = 'red'
            elif n in cycle_set:
                color = 'blue'
            else:
                color = 'black'
            ax.scatter([c_coords[n][0]], [c_coords[n][1]], [c_coords[n][2]],
                       c=color, s=150, zorder=5)

        # 绘制氢原子和碳氢键
        if h_coords:
            hx = [h[0] for h in h_coords]
            hy = [h[1] for h in h_coords]
            hz = [h[2] for h in h_coords]
            ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, zorder=5)

            # 找每个氢连接的碳原子
            h_to_c = {}
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'H':
                    for nb in atom.GetNeighbors():
                        if nb.GetSymbol() == 'C':
                            h_to_c[atom.GetIdx()] = nb.GetIdx()

            # 氢原子按索引顺序排列
            h_list = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'H':
                    h_list.append(atom.GetIdx())

            for i, h_idx in enumerate(h_list):
                if i < len(h_coords) and h_idx in h_to_c:
                    c_idx = h_to_c[h_idx]
                    if c_idx in c_coords:
                        h_c = h_coords[i]
                        c_c = c_coords[c_idx]
                        ax.plot(
                            [c_c[0], h_c[0]],
                            [c_c[1], h_c[1]],
                            [c_c[2], h_c[2]],
                            'gray', linewidth=1, alpha=0.7, zorder=4
                        )

        # 标记环原子编号
        for i, node in enumerate(main_cycle):
            if node in c_coords:
                coord = c_coords[node]
                ax.text(coord[0], coord[1], coord[2], f'{i+1}', fontsize=9, color='blue')

        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()
        return True

    def describe_isomer(self, G: nx.Graph) -> str:
        """
        生成单环烯烃异构体的描述字符串。

        格式: c5=(1-Me)= (五元环1号位甲基+环内双键), c5(1=Me) (五元环1号位亚甲基)

        Args:
            G: 环烯烃图对象（含 bond_type 属性）

        Returns:
            描述字符串。
        """
        # 找到双键位置
        double_bonds = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('bond_type') == 'double']

        # 找到环
        cycles = nx.cycle_basis(G)

        if not cycles:
            return "unknown"

        main_cycle = cycles[0]
        cycle_len = len(main_cycle)
        cycle_set = set(main_cycle)

        # 区分环内/环外双键
        ring_double_bonds = []
        exo_double_bonds = []
        for u, v in double_bonds:
            if u in cycle_set and v in cycle_set:
                ring_double_bonds.append((u, v))
            else:
                exo_double_bonds.append((u, v))

        # 构建描述
        parts = []
        ring_desc = f"c{cycle_len}"

        if ring_double_bonds:
            ring_desc += "="

        sub_info = self._get_substituent_info(G, main_cycle, cycle_set, exo_double_bonds)
        if sub_info:
            ring_desc += f"({sub_info})"

        parts.append(ring_desc)

        return "".join(parts)

    def _get_substituent_info(self, G: nx.Graph, cycle: list,
                              cycle_set: set, exo_double_bonds: list) -> str:
        """获取环上取代基信息（简化版）"""
        sub_parts = []
        for v in cycle:
            neighbors = [n for n in G.neighbors(v) if n not in cycle_set]
            if neighbors:
                for nb in neighbors:
                    is_double = any(
                        (u, v_) in exo_double_bonds or (v_, u) in exo_double_bonds
                        for u, v_ in [(v, nb), (nb, v)]
                    )
                    if is_double:
                        sub_parts.append(f"={self._count_subtree(G, nb, cycle_set)}C")
                    else:
                        nc = self._count_subtree(G, nb, cycle_set)
                        sub_parts.append(self._short_alkyl_name(nc))

        return ",".join(sub_parts) if sub_parts else ""

    def _count_subtree(self, G: nx.Graph, root: int, forbidden: set) -> int:
        """计算从 root 开始的子树节点数（不进入 forbidden 集合）"""
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


def main():
    """
    交互式命令行入口：获取碳原子数 → 生成异构体 → 显示摘要 → 提供查看/保存选项。
    """
    print("=" * 70)
    print("单环烯烃 (Cycloalkene) 同分异构体生成器")
    print("分子式: CnH2n-2 (含一个环 + 一个双键)")
    print("=" * 70)
    print()
    print("算法: 环烷烃骨架 + 遍历加双键")
    print("  Step 1: 生成所有 n 碳环烷烃骨架 (PolycycloalkaneGenerator)")
    print("  Step 2: 对每个骨架图，遍历所有边放置双键")
    print("  Step 3: WL 哈希分组 + 图同构去重")
    print()
    print("提示: 按 Ctrl+C 可以随时中断")
    print()

    # 创建生成器
    generator = CycloalkeneGenerator()

    # 循环交互
    while True:
        print("-" * 70)
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
            print("错误: 碳原子数必须 >= 3 (最小环丙烯需要3个碳)")
            continue

        print()
        try:
            isomers = generator.generate_isomers(n)
        except KeyboardInterrupt:
            print("\n[中断] 计算已停止")
            continue
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
            continue

        # 显示结果摘要
        formula = f"C{n}H{2*n - 2}"
        print()
        print(f"{formula} 共有 {len(isomers)} 个单环烯烃异构体")

        # 分类统计
        endo_count = 0
        exo_count = 0
        by_ring = {}
        for G in isomers:
            double_edges = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get('bond_type') == 'double']
            cycle = nx.cycle_basis(G)
            cycle_set = set(cycle[0]) if cycle else set()
            ring_size = len(cycle[0]) if cycle else 0

            has_endo = any(u in cycle_set and v in cycle_set for u, v in double_edges)
            has_exo = any(not (u in cycle_set and v in cycle_set) for u, v in double_edges)
            if has_endo:
                endo_count += 1
            if has_exo:
                exo_count += 1

            by_ring[ring_size] = by_ring.get(ring_size, 0) + 1

        print(f"  - 环内双键 (endocyclic): {endo_count}")
        print(f"  - 环外双键 (exocyclic): {exo_count}")
        for ring_size in sorted(by_ring.keys()):
            print(f"  - {ring_size}元环: {by_ring[ring_size]} 个")

        # 列出前 20 个异构体的双键位置
        if isomers:
            print()
            for i, G in enumerate(isomers[:20]):
                desc = generator.describe_isomer(G)
                double_edges = [(u, v) for u, v, d in G.edges(data=True)
                                if d.get('bond_type') == 'double']
                cycle = nx.cycle_basis(G)
                cycle_set = set(cycle[0]) if cycle else set()

                for u, v in double_edges:
                    in_ring = u in cycle_set and v in cycle_set
                    ring_str = "环内" if in_ring else "环外"
                    print(f"  #{i+1:3d}: C{u+1}=C{v+1} ({ring_str})  [{desc}]")

            if len(isomers) > 20:
                print(f"  ... 还有 {len(isomers) - 20} 个")

        # 交互选项
        if isomers:
            print()
            print("选项:")
            print("1. 查看指定异构体的详细信息")
            print("2. 保存异构体列表到文件")
            print("3. 继续输入新的碳原子数")
            print("4. 退出")

            try:
                choice = input("\n请选择 (1/2/3/4): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n再见！")
                break

            if choice == "1":
                # 查看指定异构体详情
                try:
                    idx_input = input(f"请输入异构体编号 (1-{len(isomers)}): ").strip()
                    if idx_input:
                        idx = int(idx_input) - 1
                        if 0 <= idx < len(isomers):
                            G = isomers[idx]
                            desc = generator.describe_isomer(G)
                            double_edges = [(u, v) for u, v, d in G.edges(data=True)
                                            if d.get('bond_type') == 'double']
                            cycle = nx.cycle_basis(G)
                            cycle_set = set(cycle[0]) if cycle else set()
                            ring_size = len(cycle[0]) if cycle else 0

                            # 计算每个碳的价键和氢原子数
                            valence_info = []
                            for node in sorted(G.nodes()):
                                total_bonds = 0
                                for nb in G.neighbors(node):
                                    bond_type = G[node][nb].get('bond_type', 'single')
                                    total_bonds += 2 if bond_type == 'double' else 1
                                h = 4 - total_bonds
                                in_ring_str = " [环]" if node in cycle_set else ""
                                valence_info.append(f"  C{node+1}: {G.degree(node)}个C-C键, {h}个H, 价键={total_bonds}{in_ring_str}")

                            print(f"\n--- {formula} 异构体 #{idx+1} ---")
                            print(f"描述: {desc}")
                            print(f"环大小: {ring_size}")
                            for u, v in double_edges:
                                in_ring = u in cycle_set and v in cycle_set
                                ring_str = "环内" if in_ring else "环外"
                                print(f"双键: C{u+1}=C{v+1} ({ring_str})")
                            print("各碳原子:")
                            for v in valence_info:
                                print(v)

                            # 弹出3D可视化窗口
                            vis_title = f"{formula} 异构体 #{idx+1} ({desc})"
                            print(f"\n正在打开3D可视化窗口...")
                            generator.visualize_isomer(G, title=vis_title)
                        else:
                            print("错误: 编号超出范围")
                except ValueError:
                    print("错误: 请输入有效的数字")

            elif choice == "2":
                # 保存到文件
                print("保存格式:")
                print("  a) 文本列表 (默认)")
                print("  b) Gaussian .gjf 文件 (RDKit 3D坐标 + 键连接信息)")
                fmt = input("请选择格式 (a/b, 默认a): ").strip().lower()

                if fmt == 'b':
                    # 使用 RDKit 生成 .gjf 文件
                    import os
                    dirname = input(f"请输入输出目录 (默认 {formula}_gjf): ").strip()
                    if not dirname:
                        dirname = f"{formula}_gjf"
                    try:
                        os.makedirs(dirname, exist_ok=True)
                        saved = 0
                        for i, G in enumerate(isomers, 1):
                            mol = generator.graph_to_rdkit_mol(G, optimize=True)
                            if mol is None:
                                print(f"  #{i}: RDKit 转换失败，跳过")
                                continue

                            # 生成 .gjf 内容
                            desc = generator.describe_isomer(G)
                            n_c = G.number_of_nodes()
                            conf = mol.GetConformer()

                            lines = []
                            lines.append(f"# {formula} Cycloalkene Isomer #{i} ({desc})")
                            lines.append("")
                            lines.append(f"# {formula} Cycloalkene - Isomer #{i}")
                            lines.append("# Generated by CycloalkeneGenerator (RDKit 3D coords + MMFF)")
                            lines.append("")
                            lines.append("0 1")

                            # 碳原子坐标
                            c_coords_map = {}
                            for atom in mol.GetAtoms():
                                if atom.GetSymbol() == 'C':
                                    pos = conf.GetAtomPosition(atom.GetIdx())
                                    c_coords_map[atom.GetIdx()] = (float(pos.x), float(pos.y), float(pos.z))

                            for idx in sorted(c_coords_map.keys()):
                                c = c_coords_map[idx]
                                lines.append(f"C     {c[0]:>12.6f} {c[1]:>12.6f} {c[2]:>12.6f}")

                            # 氢原子坐标
                            for atom in mol.GetAtoms():
                                if atom.GetSymbol() == 'H':
                                    pos = conf.GetAtomPosition(atom.GetIdx())
                                    lines.append(f"H     {float(pos.x):>12.6f} {float(pos.y):>12.6f} {float(pos.z):>12.6f}")

                            # 键连接信息
                            lines.append("")
                            for u, v, data in G.edges(data=True):
                                bond_type = data.get('bond_type', 'single')
                                bond_order = 2 if bond_type == 'double' else 1
                                lines.append(f"{u+1:4d} {v+1:4d} {bond_order:4d} 0.0 0.0 0.0")

                            filepath = os.path.join(dirname, f"{formula}_Cycloalkene_{i:03d}.gjf")
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write("\n".join(lines))
                            saved += 1

                        print(f"已保存 {saved} 个 .gjf 文件到: {dirname}")
                    except Exception as e:
                        print(f"保存失败: {e}")
                else:
                    # 文本列表
                    filename = input(f"请输入输出文件名 (默认 {formula}_cycloalkene.txt): ").strip()
                    if not filename:
                        filename = f"{formula}_cycloalkene.txt"
                    try:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"{formula} 单环烯烃异构体列表\n")
                            f.write(f"总数: {len(isomers)}\n")
                            f.write(f"环内双键: {endo_count}, 环外双键: {exo_count}\n")
                            f.write("=" * 70 + "\n\n")
                            for i, G in enumerate(isomers, 1):
                                desc = generator.describe_isomer(G)
                                double_edges = [(u, v) for u, v, d in G.edges(data=True)
                                                if d.get('bond_type') == 'double']
                                cycle = nx.cycle_basis(G)
                                cycle_set = set(cycle[0]) if cycle else set()
                                ring_size = len(cycle[0]) if cycle else 0
                                for u, v in double_edges:
                                    in_ring = u in cycle_set and v in cycle_set
                                    ring_str = "环内" if in_ring else "环外"
                                    f.write(f"{i}. C{u+1}=C{v+1} ({ring_str}) ring={ring_size} [{desc}]\n")
                        print(f"已保存到: {filename}")
                    except Exception as e:
                        print(f"保存失败: {e}")

            elif choice == "4":
                print("再见！")
                break


if __name__ == "__main__":
    main()
