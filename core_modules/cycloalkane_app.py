"""
环烷烃同分异构体生成工具

支持任意碳原子数(n>=3)的单环烷烃同分异构体生成。
基于"烷烃骨架+加边成环"算法，支持3D可视化。
OEIS A036671验证: C3=1, C4=2, C5=5, C6=12, C7=29, C8=73, C9=185, C10=475
"""

from typing import List, Dict, Set, Tuple, Optional
import sys
from pathlib import Path
import warnings
import signal

_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

RDKIT_AVAILABLE = False
Chem = None
AllChem = None
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    pass

warnings.filterwarnings('ignore')

_original_excepthook = sys.excepthook
_keyboard_interrupt_occurred = False

def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """处理Ctrl+C中断"""
    global _keyboard_interrupt_occurred
    if exc_type is KeyboardInterrupt:
        _keyboard_interrupt_occurred = True
        return
    _original_excepthook(exc_type, exc_value, exc_traceback)

sys.excepthook = _custom_excepthook


class CycloalkaneGenerator:
    """
    单环烷烃同分异构体生成器（OEIS验证）。
    核心定理：单环图 = 生成树 + 一条额外边。
    通过枚举(树, 额外边)对 + 图同构去重生成所有异构体。
    """
    
    def __init__(self, num_cores=None):
        """
        创建生成器实例，导入烷烃树生成器。

        Args:
            num_cores: 并行计算核心数（默认CPU核心数-1）
        """
        try:
            from original_programs.alkene_visualizer import AlkaneTreeGenerator
            self.tree_gen = AlkaneTreeGenerator(use_parallel=True, num_workers=num_cores)
        except ImportError:
            print("错误: 找不到 alkene_visualizer.py")
            raise
        self.num_cores = num_cores or max(1, __import__('multiprocessing').cpu_count() - 1)
    
    def generate_isomers(self, n_carbons: int) -> List[nx.Graph]:
        """
        生成C_n H_{2n}的全部单环烷烃同分异构体。

        Args:
            n_carbons: 碳原子数量（必须≥3）

        Returns:
            去重后的nx.Graph列表
        """
        if n_carbons < 3:
            return []
        
        canons = self.tree_gen.generate_all_canons(n_carbons)
        print(f"  - 烷烃骨架数: {len(canons)}")
        
        candidates = self._enumerate_candidates(canons)
        print(f"  - 候选结构: {len(candidates)}")
        
        print(f"  - 正在去重...")
        unique = self._deduplicate(candidates)
        print(f"  - 唯一结构: {len(unique)}")
        
        return unique
    
    def _enumerate_candidates(self, canons: List[str]) -> List[nx.Graph]:
        """
        对每棵骨架树枚举所有合法加边（不相邻+加边后度数≤4）。

        Args:
            canons: 规范字符串列表

        Returns:
            所有合法的候选图列表
        """
        candidates = []
        
        for canon in canons:
            adj = self.tree_gen.canon_to_adjacency(canon)
            
            T = nx.Graph()
            T.add_nodes_from(adj.keys())
            for u, nbrs in adj.items():
                for v in nbrs:
                    if u < v:
                        T.add_edge(u, v)
            
            nodes = sorted(T.nodes())
            n = len(nodes)
            deg = dict(T.degree())
            edge_set = set(T.edges())
            
            for i in range(n):
                u = nodes[i]
                if deg[u] > 3:
                    continue
                for j in range(i + 1, n):
                    v = nodes[j]
                    if deg[v] > 3:
                        continue
                    if (u, v) in edge_set:
                        continue
                    
                    G = T.copy()
                    G.add_edge(u, v)
                    candidates.append(G)
        
        return candidates
    
    @staticmethod
    def _graph_signature(G: nx.Graph) -> tuple:
        """
        计算图的签名用于快速分组和去重。

        Args:
            G: NetworkX图对象

        Returns:
            元组(度数序列, 邻居度数模式列表)
        """
        deg_seq = tuple(sorted(G.degree(v) for v in G.nodes()))
        
        nbr_patterns = []
        for v in G.nodes():
            nbr_degs = tuple(sorted(G.degree(w) for w in G.neighbors(v)))
            nbr_patterns.append(nbr_degs)
        nbr_patterns.sort()
        
        return (deg_seq, tuple(nbr_patterns))
    
    def _deduplicate(self, candidates: List[nx.Graph]) -> List[nx.Graph]:
        """
        WL哈希分组 + 精确同构两阶段去重。

        Args:
            candidates: 候选图列表

        Returns:
            去重后的唯一图列表
        """
        hash_groups: Dict[str, List[nx.Graph]] = defaultdict(list)
        for G in candidates:
            try:
                h = nx.weisfeiler_lehman_graph_hash(G)
            except Exception:
                h = str(self._graph_signature(G))
            hash_groups[h].append(G)
        
        result = []
        
        for h, group in hash_groups.items():
            if len(group) == 1:
                result.append(group[0])
                continue
            
            unique_in_group = [group[0]]
            for G in group[1:]:
                is_dup = False
                for existing in unique_in_group:
                    if nx.is_isomorphic(G, existing):
                        is_dup = True
                        break
                if not is_dup:
                    unique_in_group.append(G)
            result.extend(unique_in_group)
        
        return result
    
    @staticmethod
    def ring_size_distribution(isomers: List[nx.Graph]) -> Dict[int, int]:
        """
        统计各环大小的异构体数量分布。

        Args:
            isomers: 环烷烃图列表

        Returns:
            字典 {环大小: 数量}
        """
        dist: Dict[int, int] = defaultdict(int)
        for G in isomers:
            cycle = nx.cycle_basis(G)[0]
            cycle_len = len(cycle)
            dist[cycle_len] += 1
        return dict(dist)
    
    def describe_isomer(self, G: nx.Graph) -> str:
        """
        生成异构体的规范描述字符串。

        Args:
            G: 环烷烃图对象

        Returns:
            描述字符串，如 "c6(1-Me)" 或 "c5"
        """
        cycle = nx.cycle_basis(G)[0]
        cycle_len = len(cycle)
        
        canonical_info = self._canonical_ring_info(G)
        
        has_sub = any(subs for subs in canonical_info)
        if not has_sub:
            return f"c{cycle_len}"
        
        subs_parts = []
        for pos_idx, pos_subs in enumerate(canonical_info):
            if pos_subs:
                for n_c, canon in pos_subs:
                    short_name = self._short_alkyl_name(n_c, canon)
                    subs_parts.append(f"{pos_idx+1}-{short_name}")
        
        return f"c{cycle_len}({','.join(subs_parts)})"
    
    @staticmethod
    def _short_alkyl_name(n_carbons: int, canon: str) -> str:
        """
        获取烷基的简短英文命名。

        Args:
            n_carbons: 烷基碳原子数
            canon: 烷基规范字符串

        Returns:
            缩写名称，如 "Me", "Et", "iPr" 等
        """
        short_names = {
            (1, "C"): "Me",
            (2, "C(C)"): "Et",
            (3, "C(C,C)"): "iPr",
            (3, "C(C(C))"): "Pr",
            (4, "C(C,C,C)"): "tBu",
            (4, "C(C(C),C)"): "iBu",
            (4, "C(C(C,C))"): "sBu",
            (4, "C(C(C(C)))"): "Bu",
        }
        name = short_names.get((n_carbons, canon))
        if name:
            return name
        return f"C{n_carbons}"
    
    def _canonical_ring_info(self, G: nx.Graph) -> Tuple:
        """
        计算规范化的环取代基信息（枚举所有对称变换选择字典序最小）。

        Args:
            G: 环烷烃图对象

        Returns:
            规范化信息元组
        """
        cycle = nx.cycle_basis(G)[0]
        n_cyc = len(cycle)
        cycle_set = set(cycle)
        
        raw_info = []
        for v in cycle:
            subs = []
            for w in G.neighbors(v):
                if w not in cycle_set:
                    sig = self._substituent_signature(G, v, w)
                    subs.append(sig)
            subs.sort()
            raw_info.append(tuple(subs))
        
        best = None
        for start in range(n_cyc):
            for direction in (1, -1):
                rotated = []
                for k in range(n_cyc):
                    idx = (start + direction * k) % n_cyc
                    rotated.append(raw_info[idx])
                candidate = tuple(rotated)
                if best is None or candidate < best:
                    best = candidate
        
        return best
    
    def _substituent_signature(self, G: nx.Graph, ring_node: int, sub_root: int) -> Tuple[int, str]:
        """
        提取取代基子树的签名 (碳数, 有根规范字符串)。

        Args:
            G: 完整分子图
            ring_node: 环连接节点
            sub_root: 取代基根节点

        Returns:
            (子树碳数, 有根规范字符串)
        """
        visited = {ring_node}
        queue = [sub_root]
        nodes = []
        edges = []
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            nodes.append(node)
            for nbr in G.neighbors(node):
                if nbr not in visited:
                    edges.append((node, nbr))
                    queue.append(nbr)
        
        n = len(nodes)
        if n == 0:
            return (0, "C")
        
        node_map = {old: new for new, old in enumerate(nodes)}
        
        adj: Dict[int, List[int]] = defaultdict(list)
        for u, v in edges:
            adj[node_map[u]].append(node_map[v])
            adj[node_map[v]].append(node_map[u])
        
        canon = self._rooted_canon(adj, 0)
        return (n, canon)
    
    @staticmethod
    def _rooted_canon(adj: Dict[int, List[int]], root: int) -> str:
        """
        将有根树转换为规范字符串。

        Args:
            adj: 邻接表
            root: 根节点

        Returns:
            规范字符串，如 "C(C,C(C))"
        """
        def dfs(node: int, parent: int) -> str:
            children = []
            for nbr in adj.get(node, []):
                if nbr != parent:
                    children.append(dfs(nbr, node))
            children.sort()
            return "C" if not children else "C(" + ",".join(children) + ")"
        
        return dfs(root, -1)


class CycloalkaneBuilder:
    """
    分子3D坐标构建器。
    根据环烷烃拓扑结构计算原子坐标：环碳放XY平面正多边形，
    取代基和氢原子遵循四面体几何。
    """
    
    def __init__(self):
        self._temp_coords = {}
        self._current_rdkit_mol = None
        self._current_n_carbons = 0
    
    def build_coordinates(self, G: nx.Graph, main_cycle: List[int]) -> Tuple[Dict[int, np.ndarray], List[Tuple[np.ndarray, int]]]:
        """
        根据拓扑结构构建3D坐标。

        Args:
            G: 环烷烃分子图
            main_cycle: 主环节点列表

        Returns:
            (node_to_coord, hydrogen_coords)
        """
        ring_size = len(main_cycle)
        ring_set = set(main_cycle)
        
        ring_coords = self._generate_ring_coordinates(ring_size)
        
        node_to_coord = {}
        for i, node in enumerate(main_cycle):
            node_to_coord[node] = ring_coords[i]
        
        processed_side_nodes = set()
        existing_side_directions = {}
        
        for i, ring_node in enumerate(main_cycle):
            side_neighbors = [nb for nb in G.neighbors(ring_node) if nb not in ring_set]
            
            for side_idx, side_root in enumerate(side_neighbors):
                if side_root in processed_side_nodes:
                    continue
                
                chain_nodes = self._get_chain_nodes(G, side_root, ring_set)
                
                side_factor = 1.0 if side_idx == 0 else -1.0
                
                existing_dirs = existing_side_directions.get(ring_node, [])
                
                branch_dir = self._build_chain_coordinates(
                    G, chain_nodes, ring_node, ring_coords[i],
                    main_cycle, ring_size, node_to_coord, side_factor, ring_set,
                    existing_dirs
                )
                
                if branch_dir is not None:
                    existing_side_directions[ring_node] = existing_dirs + [branch_dir]
                
                processed_side_nodes.update(chain_nodes)
        
        all_nodes = set(G.nodes())
        missing_nodes = all_nodes - set(node_to_coord.keys())
        self._temp_coords = dict(node_to_coord)
        
        for node in missing_nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor = neighbors[0]
                if neighbor in node_to_coord:
                    direction = self._get_chain_direction(G, node, ring_set, {neighbor})
                    node_to_coord[node] = node_to_coord[neighbor] + 1.54 * direction
                    self._temp_coords[node] = node_to_coord[node]
        
        node_to_coord = self._fix_carbon_pseudo_bonds(
            node_to_coord, G, ring_set
        )
        self._temp_coords = dict(node_to_coord)

        n_carbons = G.number_of_nodes()

        rdkit_hydrogen_coords = self._generate_rdkit_hydrogen_coords(
            G, node_to_coord, n_carbons
        )

        if rdkit_hydrogen_coords:
            is_valid, score = self._check_conformer_quality(
                node_to_coord, rdkit_hydrogen_coords, G
            )
            if is_valid:
                hydrogen_coords = rdkit_hydrogen_coords
            else:
                fixed_h = self._fix_conformer(
                    node_to_coord, rdkit_hydrogen_coords, G
                )
                hydrogen_coords = fixed_h
        else:
            hydrogen_coords = self._generate_hydrogens(G, node_to_coord, ring_set)

        return node_to_coord, hydrogen_coords
    
    def _generate_ring_coordinates(self, ring_size: int) -> List[np.ndarray]:
        """
        生成环碳原子坐标（XY平面正多边形）。

        Args:
            ring_size: 环大小

        Returns:
            坐标列表 [x, y, z]
        """
        CC_BOND_LENGTH = 1.54
        coords = []
        
        if ring_size == 3:
            radius = CC_BOND_LENGTH / (2.0 * np.sin(np.pi / 3))
            for i in range(3):
                angle = i * 2 * np.pi / 3 - np.pi / 2
                coords.append(np.array([np.cos(angle), np.sin(angle), 0.0]) * radius)
        
        elif ring_size == 4:
            radius = CC_BOND_LENGTH / (2.0 * np.sin(np.pi / 4))
            for i in range(4):
                angle = i * np.pi / 2 + np.pi / 4
                coords.append(np.array([np.cos(angle), np.sin(angle), 0.0]) * radius)
        
        elif ring_size == 5:
            radius = CC_BOND_LENGTH / (2.0 * np.sin(np.pi / 5))
            for i in range(5):
                angle = i * 2 * np.pi / 5 - np.pi / 2
                coords.append(np.array([np.cos(angle), np.sin(angle), 0.0]) * radius)
        
        elif ring_size == 6:
            radius = CC_BOND_LENGTH
            for i in range(6):
                angle = i * 2 * np.pi / 6 - np.pi / 2
                coords.append(np.array([np.cos(angle), np.sin(angle), 0.0]) * radius)
        
        else:
            radius = CC_BOND_LENGTH / (2.0 * np.sin(np.pi / ring_size))
            for i in range(ring_size):
                angle = i * 2 * np.pi / ring_size - np.pi / 2
                coords.append(np.array([np.cos(angle), np.sin(angle), 0.0]) * radius)
        
        return coords
    
    def _get_chain_nodes(self, G: nx.Graph, root: int, ring_set: Set[int]) -> List[int]:
        """
        BFS遍历获取取代基链节点（不进入环）。

        Args:
            G: 分子图
            root: 根节点
            ring_set: 环节点集合

        Returns:
            链上所有节点列表
        """
        visited = {root}
        queue = [root]
        chain = []
        
        while queue:
            node = queue.pop(0)
            chain.append(node)
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited and neighbor not in ring_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return chain
    
    def _build_chain_coordinates(self, G, chain_nodes, ring_node, ring_coord,
                                main_cycle, ring_size, node_to_coord, side_factor, ring_set,
                                existing_dirs: List[np.ndarray] = None) -> np.ndarray:
        """
        构建链的坐标，返回选择的分支方向。
        """
        if existing_dirs is None:
            existing_dirs = []
        
        self._temp_coords = dict(node_to_coord)
        
        if ring_size >= 3:
            c0 = main_cycle[0]
            c1 = main_cycle[1]
            c2 = main_cycle[2]
            v1 = node_to_coord[c1] - node_to_coord[c0]
            v2 = node_to_coord[c2] - node_to_coord[c0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
        else:
            normal = np.array([0.0, 0.0, 1.0])
        
        if len(chain_nodes) >= 1:
            first_node = chain_nodes[0]
            
            ring_neighbor_coords = []
            ring_neighbors = list(G.neighbors(ring_node))
            for nb in ring_neighbors:
                if nb in node_to_coord and nb != first_node:
                    ring_neighbor_coords.append(node_to_coord[nb])
            
            if len(ring_neighbor_coords) >= 2:
                cc_vecs = []
                for nb_coord in ring_neighbor_coords:
                    vec = nb_coord - ring_coord
                    vec = vec / (np.linalg.norm(vec) + 1e-6)
                    cc_vecs.append(vec)
                
                tetra_dirs = self._get_tetrahedral_vectors(cc_vecs)
                
                if tetra_dirs:
                    best_dir = None
                    best_score = -1.0
                    
                    for d in tetra_dirs:
                        test_pos = ring_coord + 1.54 * d
                        
                        min_dist_to_ring = float('inf')
                        for ring_atom in main_cycle:
                            if ring_atom != ring_node and ring_atom in node_to_coord:
                                dist = np.linalg.norm(test_pos - node_to_coord[ring_atom])
                                min_dist_to_ring = min(min_dist_to_ring, dist)
                        
                        penalty = 0.0
                        for existing_dir in existing_dirs:
                            dot = abs(np.dot(d, existing_dir))
                            if dot > 0.7:
                                penalty += (dot - 0.7) * 10.0
                        
                        score = min_dist_to_ring - penalty
                        
                        if score > best_score:
                            best_score = score
                            best_dir = d
                    
                    if best_dir is not None:
                        branch_direction = best_dir
                    else:
                        branch_direction = tetra_dirs[0]
                    
                    outward = ring_coord - np.mean(ring_neighbor_coords, axis=0)
                    outward = outward / (np.linalg.norm(outward) + 1e-6)
                    if np.dot(branch_direction, outward) < 0:
                        branch_direction = -branch_direction
                else:
                    branch_direction = normal * side_factor
            else:
                branch_direction = normal * side_factor
            
            node_to_coord[first_node] = ring_coord + 1.54 * branch_direction
            self._temp_coords[first_node] = node_to_coord[first_node]
            
            prev = first_node
            prev_direction = branch_direction
            
            for node in chain_nodes[1:]:
                parent_coord = node_to_coord[prev]
                
                parent_key_dir = -prev_direction
                
                direction = self._get_chain_direction_v2(parent_key_dir, parent_coord)
                
                coord = parent_coord + 1.54 * direction
                node_to_coord[node] = coord
                self._temp_coords[node] = coord
                
                prev_direction = direction
                prev = node
        
        return branch_direction if 'branch_direction' in dir() else None
    
    def _get_tetrahedral_vectors(self, cc_vecs: List[np.ndarray]) -> List[np.ndarray]:
        """
        计算四面体方向向量（与已知两键成109.5°）。

        Args:
            cc_vecs: 两个已存在的C-C键方向（归一化）

        Returns:
            两个新的四面体方向向量
        """
        if len(cc_vecs) < 2:
            return []
        
        v1 = cc_vecs[0] / (np.linalg.norm(cc_vecs[0]) + 1e-6)
        v2 = cc_vecs[1] / (np.linalg.norm(cc_vecs[1]) + 1e-6)
        
        dot_v1_v2 = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_v1_v2 = np.arccos(dot_v1_v2)
        
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            if abs(v1[0]) < 0.9:
                normal = np.cross(v1, np.array([1, 0, 0]))
            else:
                normal = np.cross(v1, np.array([0, 1, 0]))
            normal_norm = np.linalg.norm(normal)
        normal /= normal_norm
        
        cos_tetra = -1.0 / 3.0
        
        bisector = (v1 + v2)
        bis_norm = np.linalg.norm(bisector)
        if bis_norm < 1e-6:
            bisector = v1.copy()
            bis_norm = 1.0
        bisector /= bis_norm
        
        perp_to_normal = bisector - np.dot(bisector, normal) * normal
        perp_norm = np.linalg.norm(perp_to_normal)
        if perp_norm < 1e-6:
            perp_to_normal = normal.copy()
        else:
            perp_to_normal /= perp_norm
        
        bis_dot_v1 = np.dot(bisector, v1)
        norm_dot_v1 = np.dot(normal, v1)
        
        diff_norm = np.linalg.norm(v1 - v2)
        
        if diff_norm < 1.2:
            perp = v1.copy()
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, np.array([1, 0, 0]))
            else:
                perp = np.cross(v1, np.array([0, 1, 0]))
            perp /= (np.linalg.norm(perp) + 1e-6)
            
            v2_perp = v2 - np.dot(v2, v1) * v1
            v2_perp_norm = np.linalg.norm(v2_perp)
            if v2_perp_norm > 1e-6:
                v2_perp /= v2_perp_norm
            
            e1 = v1.copy()
            e2 = v2 - np.dot(v2, e1) * e1
            e2_norm = np.linalg.norm(e2)
            if e2_norm > 1e-6:
                e2 /= e2_norm
            else:
                if abs(v1[2]) < 0.9:
                    e2 = np.cross(v1, np.array([0, 0, 1]))
                else:
                    e2 = np.cross(v1, np.array([1, 0, 0]))
                e2 /= (np.linalg.norm(e2) + 1e-6)
            
            e3 = np.cross(e1, e2)
            e3 /= (np.linalg.norm(e3) + 1e-6)
            
            x = cos_tetra
            
            e1_dot_v2 = np.dot(e1, v2)
            e2_dot_v2 = np.dot(e2, v2)
            
            y = (-1/3 - x * e1_dot_v2) / e2_dot_v2
            
            z_squared = 1 - x**2 - y**2
            if z_squared < 0:
                z_squared = 0
            z = np.sqrt(z_squared)
            
            v_new1 = x * e1 + y * e2 + z * e3
            v_new2 = x * e1 + y * e2 - z * e3
            
            n1 = np.linalg.norm(v_new1)
            n2 = np.linalg.norm(v_new2)
            if n1 > 1e-6:
                v_new1 /= n1
            if n2 > 1e-6:
                v_new2 /= n2
            
            return [v_new1, v_new2]
        
        else:
            if abs(norm_dot_v1) < 1e-6:
                perp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(perp, normal)) > 0.9:
                    perp = np.array([0.0, 1.0, 0.0])
                
                tangent = np.cross(normal, perp)
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-6:
                    tangent /= tangent_norm
                
                v1_tangent = np.dot(v1, tangent)
                
                a_tangent = -1.0 / 3.0 / max(abs(v1_tangent), 1e-6) if abs(v1_tangent) > 1e-6 else 0.0
                
                perp2 = np.cross(tangent, normal)
                perp2_norm = np.linalg.norm(perp2)
                if perp2_norm > 1e-6:
                    perp2 /= perp2_norm
                
                v1_perp2 = np.dot(v1, perp2)
                b_perp2 = (-1.0/3.0 - a_tangent * v1_tangent) / max(abs(v1_perp2), 1e-6) if abs(v1_perp2) > 1e-6 else 0.0
                
                v_new1 = a_tangent * tangent + b_perp2 * perp2 + normal
                v_new2 = a_tangent * tangent + b_perp2 * perp2 - normal
            else:
                a = (-1/3 - bis_dot_v1 * cos_tetra) / norm_dot_v1
                
                perp_component = np.sqrt(1 - cos_tetra**2 - a**2 - 2*a*bis_dot_v1*norm_dot_v1)
                
                if np.isnan(perp_component) or perp_component < 0:
                    perp_component = np.sqrt(max(1 - cos_tetra**2, 0))
                
                v_new1 = cos_tetra * bisector + a * normal + perp_component * perp_to_normal
                v_new2 = cos_tetra * bisector + a * normal - perp_component * perp_to_normal
            
            n1 = np.linalg.norm(v_new1)
            n2 = np.linalg.norm(v_new2)
            if n1 > 1e-6:
                v_new1 /= n1
            if n2 > 1e-6:
                v_new2 /= n2
            
            if np.any(np.isnan(v_new1)) or np.any(np.isinf(v_new1)):
                v_new1 = normal.copy()
            if np.any(np.isnan(v_new2)) or np.any(np.isinf(v_new2)):
                v_new2 = -normal.copy()
            
            return [v_new1, v_new2]
    
    def _get_chain_direction_v2(self, parent_key_dir: np.ndarray, parent_pos: np.ndarray) -> np.ndarray:
        """
        简化版链方向计算（新键与父键成四面体角）。

        Args:
            parent_key_dir: 父键方向
            parent_pos: 父节点位置

        Returns:
            新键方向单位向量
        """
        two_dirs = self._calculate_tetrahedral_from_one_bond_both(parent_key_dir)
        
        if two_dirs and len(two_dirs) == 2:
            best_dir = two_dirs[0]
            best_min_dist = -1.0
            
            for d in two_dirs:
                test_pos = parent_pos + 1.54 * d
                
                min_dist = float('inf')
                for other_node in self._temp_coords:
                    dist = np.linalg.norm(test_pos - self._temp_coords[other_node])
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_dir = d
            
            return best_dir
        
        return self._calculate_tetrahedral_from_one_bond(parent_key_dir)
    
    def _get_chain_direction(self, G: nx.Graph, node: int,
                            ring_set: Set[int], forbidden: Set[int],
                            grand_parent_dir: np.ndarray = None) -> np.ndarray:
        """
        获取链方向的单位向量。
        """
        parent_dir = None
        parent_neighbors = [nb for nb in G.neighbors(node) if nb in forbidden]
        if parent_neighbors:
            parent = parent_neighbors[0]
            if parent in self._temp_coords:
                parent_dir = self._temp_coords[parent] - self._temp_coords.get(node, self._temp_coords[parent])

        if parent_dir is None and grand_parent_dir is not None:
            parent_dir = grand_parent_dir

        neighbors = [nb for nb in G.neighbors(node) if nb not in forbidden and nb not in ring_set]

        if len(neighbors) == 0:
            if parent_dir is not None:
                return self._calculate_tetrahedral_from_one_bond(parent_dir)
            return np.array([0.0, 1.0, 0.0])

        if len(neighbors) == 1:
            if parent_dir is not None:
                two_dirs = self._calculate_tetrahedral_from_one_bond_both(parent_dir)
                
                if two_dirs and len(two_dirs) == 2:
                    best_dir = None
                    best_min_dist = -1.0
                    
                    for d in two_dirs:
                        parent_pos = self._temp_coords[parent_neighbors[0]] if parent_neighbors else None
                        if parent_pos is not None:
                            test_pos = parent_pos + 1.54 * (-d)
                            
                            min_dist = float('inf')
                            for other_node in self._temp_coords:
                                if other_node not in forbidden and other_node != parent_neighbors[0]:
                                    dist = np.linalg.norm(test_pos - self._temp_coords[other_node])
                                    min_dist = min(min_dist, dist)
                            
                            if min_dist > best_min_dist:
                                best_min_dist = min_dist
                                best_dir = d
                    
                    if best_dir is not None:
                        return best_dir
                
                return self._calculate_tetrahedral_from_one_bond(parent_dir)
            return np.array([0.0, 1.0, 0.0])

        if len(neighbors) >= 2:
            vectors = []
            for nb in neighbors:
                if nb in self._temp_coords and node in self._temp_coords:
                    vec = self._temp_coords[nb] - self._temp_coords[node]
                    vectors.append(vec)

            if len(vectors) >= 2:
                return self._calculate_tetrahedral_from_two_bonds(vectors[0], vectors[1])
            elif len(vectors) == 1:
                if parent_dir is not None:
                    return self._calculate_tetrahedral_from_one_bond(parent_dir)
                return np.array([0.0, 1.0, 0.0])

        return np.array([0.0, 1.0, 0.0])
    
    def _calculate_tetrahedral_from_one_bond_both(self, vec: np.ndarray) -> List[np.ndarray]:
        """
        返回与vec成109.5°的两个方向向量。
        """
        v1 = vec / (np.linalg.norm(vec) + 1e-6)
        
        if abs(v1[0]) < 0.9:
            arb = np.array([1, 0, 0])
        else:
            arb = np.array([0, 1, 0])
        
        perp = np.cross(v1, arb)
        perp /= (np.linalg.norm(perp) + 1e-6)
        
        cos_theta = -1/3
        sin_theta = np.sqrt(8)/3
        
        v_new1 = cos_theta * v1 + sin_theta * perp
        v_new2 = cos_theta * v1 - sin_theta * perp
        
        n1 = np.linalg.norm(v_new1)
        n2 = np.linalg.norm(v_new2)
        if n1 > 1e-6:
            v_new1 /= n1
        if n2 > 1e-6:
            v_new2 /= n2
        
        return [v_new1, v_new2]
    
    def _calculate_tetrahedral_from_one_bond(self, vec: np.ndarray) -> np.ndarray:
        """
        根据一个已存在的键向量计算四面体方向（优先伸直方向）。
        """
        v1 = vec / (np.linalg.norm(vec) + 1e-6)
        
        if abs(v1[0]) < 0.9:
            arb = np.array([1, 0, 0])
        else:
            arb = np.array([0, 1, 0])
        
        perp = np.cross(v1, arb)
        perp /= (np.linalg.norm(perp) + 1e-6)
        
        cos_theta = -1/3
        sin_theta = np.sqrt(8)/3
        
        v_new1 = cos_theta * v1 + sin_theta * perp
        v_new2 = cos_theta * v1 - sin_theta * perp
        
        n1 = np.linalg.norm(v_new1)
        n2 = np.linalg.norm(v_new2)
        if n1 > 1e-6:
            v_new1 /= n1
        if n2 > 1e-6:
            v_new2 /= n2
        
        neg_v1 = -v1
        dot1 = np.dot(v_new1, neg_v1)
        dot2 = np.dot(v_new2, neg_v1)
        
        if dot1 >= dot2:
            return v_new1
        else:
            return v_new2
    
    def _calculate_tetrahedral_from_two_bonds(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """根据两个已存在的键向量计算四面体方向。"""
        v1 = vec1 / (np.linalg.norm(vec1) + 1e-6)
        v2 = vec2 / (np.linalg.norm(vec2) + 1e-6)
        
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        
        normal /= normal_norm
        cos_theta = -1/3
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        bisector = (v1 + v2) / (np.linalg.norm(v1 + v2) + 1e-6)
        v_new = cos_theta * bisector + sin_theta * normal
        
        return v_new / (np.linalg.norm(v_new) + 1e-6)
    
    def _generate_hydrogens(self, G: nx.Graph,
                           node_to_coord: Dict[int, np.ndarray],
                           ring_set: Set[int]) -> List[Tuple[np.ndarray, int]]:
        """
        生成氢原子坐标（C-H键长1.09Å，遵循四面体几何）。

        Args:
            G: 分子图
            node_to_coord: 碳原子坐标
            ring_set: 环节点集合

        Returns:
            [(氢坐标, 对应碳节点), ...]
        """
        hydrogens = []
        
        if len(ring_set) >= 3:
            cycle_list = list(ring_set)
            c0, c1, c2 = cycle_list[0], cycle_list[1], cycle_list[2]
            v1 = node_to_coord[c1] - node_to_coord[c0]
            v2 = node_to_coord[c2] - node_to_coord[c0]
            ring_normal = np.cross(v1, v2)
            ring_normal = ring_normal / (np.linalg.norm(ring_normal) + 1e-6)
        else:
            ring_normal = np.array([0.0, 0.0, 1.0])
        
        for node in sorted(G.nodes()):
            coord = node_to_coord[node]
            degree = G.degree(node)
            num_h = max(0, 4 - degree)
            
            if num_h <= 0:
                continue
            
            cc_vectors = []
            for neighbor in G.neighbors(node):
                if neighbor in node_to_coord:
                    vec = node_to_coord[neighbor] - coord
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        cc_vectors.append(vec / norm)
            
            if node in ring_set:
                local_normal = ring_normal
            else:
                local_normal = self._compute_local_normal(node, G, node_to_coord)
            
            h_dirs = self._calculate_h_directions(cc_vectors, local_normal, num_h)
            
            for h_dir in h_dirs:
                h_coord = coord + 1.09 * h_dir
                hydrogens.append((h_coord, node))
        
        return hydrogens
    
    def _calculate_h_directions(self, cc_vectors: List[np.ndarray],
                              ring_normal: np.ndarray,
                              num_hydrogens: int) -> List[np.ndarray]:
        """计算氢原子方向。"""
        import math
        
        normalized_cc = [v / (np.linalg.norm(v) + 1e-6) for v in cc_vectors]
        
        if not normalized_cc:
            sqrt3 = math.sqrt(3)
            dirs = [
                np.array([sqrt3/3, sqrt3/3, sqrt3/3]),
                np.array([sqrt3/3, -sqrt3/3, -sqrt3/3]),
                np.array([-sqrt3/3, sqrt3/3, -sqrt3/3]),
                np.array([-sqrt3/3, -sqrt3/3, sqrt3/3])
            ]
            return dirs[:num_hydrogens]
        
        cos_theta = -1/3
        sin_theta = math.sqrt(8)/3
        
        if len(normalized_cc) == 1:
            v1 = normalized_cc[0]
            
            if abs(v1[0]) < 0.9:
                arb = np.array([1.0, 0.0, 0.0])
            else:
                arb = np.array([0.0, 1.0, 0.0])
            
            perp = np.cross(v1, arb)
            perp /= (np.linalg.norm(perp) + 1e-6)
            
            h_dirs = []
            for i in range(num_hydrogens):
                angle = i * 2 * math.pi / num_hydrogens
                v_new = cos_theta * v1 + sin_theta * (math.cos(angle) * perp + math.sin(angle) * np.cross(v1, perp))
                h_dirs.append(v_new / (np.linalg.norm(v_new) + 1e-6))
            
            return h_dirs
        
        elif len(normalized_cc) == 2:
            v1, v2 = normalized_cc
            
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-6:
                normal = ring_normal
            else:
                normal /= norm_n
            
            u = v1
            w = v2 - np.dot(v2, u) * u
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-6:
                w = np.cross(normal, u)
                w /= (np.linalg.norm(w) + 1e-6)
            
            cos_beta = np.clip(np.dot(v1, v2), -1, 1)
            beta = math.acos(cos_beta)
            sin_beta = math.sin(beta)
            if abs(sin_beta) < 1e-6:
                sin_beta = 1e-6
            
            a = cos_theta
            if abs(sin_beta) > 1e-6:
                b = (cos_theta - a * cos_beta) / sin_beta
            else:
                b = 0.0
            
            plane_component = a * u + b * w
            z_squared = 1 - a**2 - b**2
            
            h_dirs = []
            if z_squared > 0:
                z = math.sqrt(z_squared)
                h1 = plane_component + z * normal
                h2 = plane_component - z * normal
            else:
                h1 = plane_component + 0.1 * normal
                h2 = plane_component - 0.1 * normal
            
            h_dirs.append(h1 / (np.linalg.norm(h1) + 1e-6))
            if num_hydrogens >= 2:
                h_dirs.append(h2 / (np.linalg.norm(h2) + 1e-6))
            
            return h_dirs[:num_hydrogens]
        
        elif len(normalized_cc) == 3:
            sum_vec = np.sum(normalized_cc, axis=0)
            v4 = -sum_vec
            norm_v4 = np.linalg.norm(v4)
            if norm_v4 > 1e-6:
                return [v4 / norm_v4]
            return []
        
        return []
    
    def _compute_local_normal(self, node: int, G: nx.Graph, 
                             node_to_coord: Dict[int, np.ndarray]) -> np.ndarray:
        """为非环碳原子计算局部法线。"""
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) >= 2:
            v1 = node_to_coord[neighbors[0]] - node_to_coord[node]
            v2 = node_to_coord[neighbors[1]] - node_to_coord[node]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                return normal / norm
        
        return np.array([0.0, 0.0, 1.0])

    def _generate_rdkit_hydrogen_coords(
        self,
        G: nx.Graph,
        node_to_coord: Dict[int, np.ndarray],
        n_carbons: int
    ) -> List[Tuple[np.ndarray, int]]:
        """
        使用RDKit生成氢原子坐标（力场优化）。

        Args:
            G: 分子图
            node_to_coord: 碳原子坐标
            n_carbons: 碳原子数

        Returns:
            [(氢坐标, 对应碳节点), ...]
        """
        if not RDKIT_AVAILABLE:
            return []

        try:
            mol = Chem.RWMol()

            for _ in range(n_carbons):
                mol.AddAtom(Chem.Atom('C'))

            added_bonds = set()
            for u, v in G.edges():
                bond_key = tuple(sorted([u, v]))
                if bond_key not in added_bonds:
                    mol.AddBond(u, v, Chem.BondType.SINGLE)
                    added_bonds.add(bond_key)

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass

            mol_h = Chem.AddHs(mol)
            conf = Chem.Conformer(n_carbons)

            for node in range(n_carbons):
                if node in node_to_coord:
                    coord = node_to_coord[node]
                    conf.SetAtomPosition(node, Chem.Point3D(float(coord[0]), float(coord[1]), float(coord[2])))

            mol_h.AddConformer(conf)

            try:
                AllChem.MMFFOptimizeMolecule(mol_h, ignoreInterfragInteractions=True)
            except Exception:
                pass

            hydrogen_coords = []
            conf = mol_h.GetConformer()
            for i in range(n_carbons, mol_h.GetNumAtoms()):
                atom = mol_h.GetAtomWithIdx(i)
                if atom.GetSymbol() == 'H':
                    pos = conf.GetAtomPosition(i)
                    h_coord = np.array([float(pos.x), float(pos.y), float(pos.z)])

                    min_dist = float('inf')
                    parent_carbon = 0
                    for c_idx in range(n_carbons):
                        if c_idx in node_to_coord:
                            c_coord = node_to_coord[c_idx]
                            dist = np.linalg.norm(h_coord - c_coord)
                            if dist < min_dist:
                                min_dist = dist
                                parent_carbon = c_idx

                    hydrogen_coords.append((h_coord, parent_carbon))

            return hydrogen_coords

        except Exception as e:
            return []

    def get_rdkit_hydrogen_coords(
        self,
        G: nx.Graph,
        node_to_coord: Dict[int, np.ndarray],
        n_carbons: int
    ) -> List[Tuple[np.ndarray, int]]:
        """
        获取RDKit生成的氢原子坐标。

        Args:
            G: 分子图
            node_to_coord: 碳原子坐标
            n_carbons: 碳原子数

        Returns:
            [(氢坐标, 对应碳节点), ...]
        """
        return self._generate_rdkit_hydrogen_coords(G, node_to_coord, n_carbons)

    def _check_conformer_quality(
        self,
        node_to_coord: Dict[int, np.ndarray],
        hydrogen_coords: List[Tuple[np.ndarray, int]],
        G: nx.Graph
    ) -> Tuple[bool, float]:
        """
        检查构象质量（C-C 1.50~1.60Å, C-H 1.05~1.15Å）。

        Returns:
            (是否通过, 质量分数)
        """
        import numpy as np

        n_carbons = len(node_to_coord)
        score = 0.0

        for u, v in G.edges():
            if u in node_to_coord and v in node_to_coord:
                d = np.linalg.norm(node_to_coord[u] - node_to_coord[v])
                if d < 1.45 or d > 1.65:
                    score += 1.0

        for h_coord, carbon in hydrogen_coords:
            if carbon in node_to_coord:
                d = np.linalg.norm(h_coord - node_to_coord[carbon])
                if d < 1.00 or d > 1.20:
                    score += 0.5

        for i in range(len(hydrogen_coords)):
            for j in range(i + 1, len(hydrogen_coords_coords := hydrogen_coords)):
                h1, _ = hydrogen_coords[i]
                h2, _ = hydrogen_coords[j]
                d = np.linalg.norm(h1 - h2)
                if d < 1.2:
                    score += 0.3

        return score < 1.0, score

    def _fix_conformer(
        self,
        node_to_coord: Dict[int, np.ndarray],
        hydrogen_coords: List[Tuple[np.ndarray, int]],
        G: nx.Graph
    ) -> List[Tuple[np.ndarray, int]]:
        """
        修复氢原子过近问题。

        Returns:
            修复后的氢原子坐标
        """
        import numpy as np

        fixed_h = [(h.copy(), c) for h, c in hydrogen_coords]

        for _ in range(20):
            changed = False

            for i in range(len(fixed_h)):
                for j in range(i + 1, len(fixed_h)):
                    h1, c1 = fixed_h[i]
                    h2, c2 = fixed_h[j]

                    d = np.linalg.norm(h1 - h2)
                    if d < 1.2 and d > 1e-6:
                        direction = h1 - h2
                        direction = direction / np.linalg.norm(direction)

                        displacement = 0.3 * (1.2 - d) * direction
                        h1 += displacement * 0.5
                        h2 -= displacement * 0.5

                        if c1 in node_to_coord:
                            c1_coord = node_to_coord[c1]
                            vec = h1 - c1_coord
                            dist = np.linalg.norm(vec)
                            if dist > 1e-6:
                                h1 = c1_coord + vec / dist * 1.09
                        if c2 in node_to_coord:
                            c2_coord = node_to_coord[c2]
                            vec = h2 - c2_coord
                            dist = np.linalg.norm(vec)
                            if dist > 1e-6:
                                h2 = c2_coord + vec / dist * 1.09

                        fixed_h[i] = (h1, c1)
                        fixed_h[j] = (h2, c2)
                        changed = True

            if not changed:
                break

        return fixed_h

    def _fix_carbon_pseudo_bonds(
        self,
        node_to_coord: Dict[int, np.ndarray],
        G: nx.Graph,
        ring_set: Set[int]
    ) -> Dict[int, np.ndarray]:
        """
        修复碳原子伪键+收紧过长化学键。

        Returns:
            修复后的坐标字典
        """
        fixed_coords = dict(node_to_coord)
        n = len(fixed_coords)
        if n <= 1:
            return fixed_coords

        node_list = list(fixed_coords.keys())
        pos = np.array([fixed_coords[node] for node in node_list], dtype=np.float64)
        idx_map = {node: i for i, node in enumerate(node_list)}

        bonded_pairs = set()
        for u, v in G.edges():
            if u in idx_map and v in idx_map:
                pair = tuple(sorted([idx_map[u], idx_map[v]]))
                bonded_pairs.add(pair)

        ring_idx = set()
        for node in ring_set:
            if node in idx_map:
                ring_idx.add(idx_map[node])

        node_degrees = {}
        for i, node in enumerate(node_list):
            node_degrees[i] = G.degree(node) if node in G else 0

        MIN_NONBOND_DIST = 2.8
        CC_BOND_LEN = 1.54
        BOND_TOLERANCE = 0.15

        for outer_iter in range(20):
            for phase in [(0.35, 400, 0.001), (0.15, 200, 0.0005)]:
                step_size, max_iters, converge = phase
                for _ in range(max_iters):
                    max_move = 0.0
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (i, j) in bonded_pairs:
                                continue

                            diff = pos[j] - pos[i]
                            dist = np.linalg.norm(diff)

                            if dist < 1e-8:
                                rand_dir = np.random.randn(3)
                                rand_dir /= np.linalg.norm(rand_dir) + 1e-10
                                push = rand_dir * 0.5
                                if i in ring_idx and j not in ring_idx:
                                    pos[j] += push * 2
                                elif j in ring_idx and i not in ring_idx:
                                    pos[i] -= push * 2
                                else:
                                    pos[i] -= push
                                    pos[j] += push
                                max_move = max(max_move, 0.5)
                            elif dist < MIN_NONBOND_DIST:
                                overlap = MIN_NONBOND_DIST - dist
                                force = (overlap / dist) * diff
                                step = force * step_size * 0.5

                                i_is_ring = i in ring_idx
                                j_is_ring = j in ring_idx
                                i_deg = node_degrees[i]
                                j_deg = node_degrees[j]

                                if i_is_ring and not j_is_ring:
                                    pos[j] += step * 2
                                elif j_is_ring and not i_is_ring:
                                    pos[i] -= step * 2
                                elif i_deg <= j_deg:
                                    pos[i] -= step
                                    pos[j] += step
                                else:
                                    pos[i] -= step
                                    pos[j] += step
                                max_move = max(max_move, np.linalg.norm(step))

                    if max_move < converge:
                        break

            for _ in range(100):
                max_correction = 0.0
                for (i, j) in bonded_pairs:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-8:
                        continue

                    error = dist - CC_BOND_LEN
                    if abs(error) > 0.01:
                        correction = (error / dist) * diff * 0.3
                        i_is_ring = i in ring_idx
                        j_is_ring = j in ring_idx

                        if i_is_ring and not j_is_ring:
                            pos[j] -= correction
                        elif j_is_ring and not i_is_ring:
                            pos[i] += correction
                        else:
                            pos[i] += correction * 0.5
                            pos[j] -= correction * 0.5
                        max_correction = max(max_correction, abs(error))

                if max_correction < 0.005:
                    break

            still_close = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in bonded_pairs:
                        continue
                    dist = np.linalg.norm(pos[j] - pos[i])
                    if dist < 2.2:
                        still_close += 1
            if still_close == 0:
                break

        for _ in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in bonded_pairs:
                        continue
                    dist = np.linalg.norm(pos[j] - pos[i])
                    if dist < 2.4 and dist > 1e-8:
                        diff = pos[j] - pos[i]
                        direction = diff / np.linalg.norm(diff)
                        push = (2.4 - dist) * 0.5
                        i_is_ring = i in ring_idx
                        j_is_ring = j in ring_idx
                        if i_is_ring and not j_is_ring:
                            pos[j] += direction * push * 2
                        elif j_is_ring and not i_is_ring:
                            pos[i] -= direction * push * 2
                        else:
                            pos[i] -= direction * push
                            pos[j] += direction * push

        for (i, j) in bonded_pairs:
            dist = np.linalg.norm(pos[j] - pos[i])
            if dist < 1.35 and dist > 1e-8:
                diff = pos[j] - pos[i]
                direction = diff / np.linalg.norm(diff)
                stretch = (CC_BOND_LEN - dist) * 0.5
                i_is_ring = i in ring_idx
                j_is_ring = j in ring_idx
                if i_is_ring and not j_is_ring:
                    pos[j] += direction * stretch * 2
                elif j_is_ring and not i_is_ring:
                    pos[i] -= direction * stretch * 2
                else:
                    pos[i] -= direction * stretch
                    pos[j] += direction * stretch

        for i, node in enumerate(node_list):
            fixed_coords[node] = pos[i]

        return fixed_coords


class CycloalkaneApp:
    """
    环烷烃交互式应用程序主类。
    提供异构体生成、显示和保存功能。
    """
    
    # OEIS A036671验证数据
    OEIS_A036671 = {
        3: 1,
        4: 2,
        5: 5,
        6: 12,
        7: 29,
        8: 73,
        9: 185,
        10: 475
    }
    
    def __init__(self):
        """初始化应用程序。"""
        self.generator = CycloalkaneGenerator()
        self.builder = CycloalkaneBuilder()
    
    def run(self, formula: str = None, auto_select: bool = False):
        """
        运行应用程序主循环。

        Args:
            formula: 分子式字符串（如 "C10H20"）
            auto_select: 是否自动选择
        """
        global _interrupted
        _interrupted = False

        def _signal_handler(signum, frame):
            global _interrupted
            _interrupted = True
            print("\n\n[中断] 正在停止计算，请稍候...")
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, _signal_handler)

        print("=" * 70)
        print("环烷烃同分异构体生成器")
        print("OEIS A036671 验证: C3=1, C4=2, C5=5, C6=12, C7=29, C8=73")
        print("=" * 70)
        print()

        if formula is None:
            formula = input("请输入分子式 (例如 C10H20): ").strip()

        try:
            import re
            match = re.match(r'C(\d+)H(\d+)', formula)
            if not match:
                print("分子式格式无效，请使用 C{n}H{m} 格式")
                return

            c = int(match.group(1))
            h = int(match.group(2))

            if h != 2 * c:
                print("本程序只处理单环烷烃 (CnH2n)")
                return

            if c < 3:
                print("碳原子数必须 >= 3")
                return

            print()

            try:
                graphs = self.generator.generate_isomers(c)
            except (KeyboardInterrupt, Exception):
                if isinstance(sys.exc_info()[0], KeyboardInterrupt) or _interrupted:
                    print("\n\n[中断] 计算已停止")
                else:
                    print(f"\n\n[错误] 计算过程中出现错误: {sys.exc_info()[1]}")
                print("感谢使用！")
                return

            if not graphs:
                print(f"未找到 C{c}H{h} 的异构体")
                return

            expected = self.OEIS_A036671.get(c)
            if expected:
                status = "[OK]" if len(graphs) == expected else "[FAIL]"
                print(f"生成了 {len(graphs)} 个异构体 (预期 {expected}) {status}")
            else:
                print(f"生成了 {len(graphs)} 个异构体")

            by_ring = defaultdict(list)
            for G in graphs:
                cycle = nx.cycle_basis(G)[0]
                by_ring[len(cycle)].append(G)

            print()
            print("环大小分布:")
            for ring_size in sorted(by_ring.keys()):
                print(f"  c{ring_size}: {len(by_ring[ring_size])} 个异构体")

            print()
            print(f"异构体列表 (共 {len(graphs)} 个):")
            for i in range(min(len(graphs), 20)):
                desc = self.generator.describe_isomer(graphs[i])
                print(f"  {i+1}. {desc}")

            if len(graphs) > 20:
                print(f"  ... 还有 {len(graphs) - 20} 个")

            if len(graphs) > 0:
                print("\n选项:")
                print("1. 可视化单个异构体")
                print("2. 保存异构体列表到文件")
                print("3. 退出")

                choice = input("\n请选择 (1/2/3): ").strip()

                if choice == "1":
                    idx_input = input(f"请输入异构体编号 (1-{len(graphs)}): ").strip()
                    if idx_input:
                        try:
                            idx = int(idx_input) - 1
                            if 0 <= idx < len(graphs):
                                print(f"\n正在显示: {self.generator.describe_isomer(graphs[idx])}")
                                self._visualize_isomer(graphs[idx])
                            else:
                                print("错误: 编号超出范围")
                        except ValueError:
                            print("错误: 请输入有效的数字")
                    else:
                        print("已取消")

                elif choice == "2":
                    default_name = f"C{c}_cycloalkane_isomers.txt"
                    filename = input(f"请输入保存文件名 (默认 {default_name}): ").strip()
                    if not filename:
                        filename = default_name

                    try:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"C{c}H{2*c} 环烷烃同分异构体列表\n")
                            f.write(f"总计: {len(graphs)} 个\n")
                            f.write("=" * 70 + "\n\n")
                            for i, G in enumerate(graphs, 1):
                                desc = self.generator.describe_isomer(G)
                                f.write(f"{i}. {desc}\n")
                        print(f"已保存到: {filename}")
                    except Exception as e:
                        print(f"保存失败: {e}")

                else:
                    print("已退出")
            else:
                print("没有可显示的异构体")

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_all_isomers(self, graphs: List[nx.Graph]):
        """显示所有异构体的坐标。"""
        for i, G in enumerate(graphs):
            print(f"\n{'='*60}")
            print(f"Isomer {i+1}: {self.generator.describe_isomer(G)}")
            print(f"{'='*60}")
            
            try:
                main_cycle = nx.cycle_basis(G)[0]
                node_to_coord, hydrogen_coords = self.builder.build_coordinates(G, main_cycle)
                
                print(f"Formula: C{G.number_of_nodes()}H{2*G.number_of_nodes()}")
                print(f"Carbon atoms: {len(node_to_coord)}")
                print(f"Hydrogen atoms: {len(hydrogen_coords)}")
                
            except Exception as e:
                print(f"Coordinate generation failed: {e}")
    
    def _visualize_isomer(self, G: nx.Graph):
        """
        可视化单个异构体（3D图形）。
        """
        try:
            main_cycle = nx.cycle_basis(G)[0]
            node_to_coord, hydrogen_coords = self.builder.build_coordinates(G, main_cycle)
            
            name = self.generator.describe_isomer(G)
            n_carbon = G.number_of_nodes()
            
            print(f"\n{'='*60}")
            print(f"Name: {name}")
            print(f"Formula: C{n_carbon}H{2*n_carbon}")
            print(f"{'='*60}")
            
            print("\nCarbon coordinates:")
            for node in sorted(node_to_coord.keys()):
                coord = node_to_coord[node]
                is_ring = "(ring)" if node in set(main_cycle) else ""
                print(f"  C{node+1}{is_ring}: [{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}]")
            
            print(f"\nHydrogen coordinates (total {len(hydrogen_coords)}):")
            for i, (coord, carbon) in enumerate(hydrogen_coords[:12]):
                print(f"  H{i+1}(C{carbon}): [{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}]")
            if len(hydrogen_coords) > 12:
                print(f"  ... and {len(hydrogen_coords) - 12} more hydrogen atoms")

            print(f"\nStatistics:")
            print(f"  Carbon atoms: {len(node_to_coord)}")
            print(f"  Hydrogen atoms: {len(hydrogen_coords)}")
            print(f"  Total atoms: {len(node_to_coord) + len(hydrogen_coords)}")
            
            c_z = [node_to_coord[n][2] for n in sorted(node_to_coord.keys())]
            print(f"\nSpatial distribution:")
            print(f"  Carbon Z range: {min(c_z):.4f} ~ {max(c_z):.4f}")
            
            z_range = max(c_z) - min(c_z)
            if z_range > 0.1:
                print(f"  Status: Molecule has 3D structure")
            else:
                print(f"  Status: Molecule is nearly planar")
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            drawn_bonds = set()
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    bond_key = tuple(sorted([node, neighbor]))
                    if bond_key not in drawn_bonds:
                        if node in node_to_coord and neighbor in node_to_coord:
                            ax.plot(
                                [node_to_coord[node][0], node_to_coord[neighbor][0]],
                                [node_to_coord[node][1], node_to_coord[neighbor][1]],
                                [node_to_coord[node][2], node_to_coord[neighbor][2]],
                                'k-', linewidth=2
                            )
                        drawn_bonds.add(bond_key)
            
            cx = [node_to_coord[n][0] for n in sorted(node_to_coord.keys())]
            cy = [node_to_coord[n][1] for n in sorted(node_to_coord.keys())]
            cz = [node_to_coord[n][2] for n in sorted(node_to_coord.keys())]
            
            ring_colors = ['red' if n in set(main_cycle) else 'black' for n in sorted(node_to_coord.keys())]
            ax.scatter(cx, cy, cz, c=ring_colors, s=150, label='C (Carbon)', zorder=5)
            
            if len(hydrogen_coords) > 0:
                hx = [h[0][0] for h in hydrogen_coords]
                hy = [h[0][1] for h in hydrogen_coords]
                hz = [h[0][2] for h in hydrogen_coords]
                ax.scatter(hx, hy, hz, c='white', edgecolors='gray', s=80, 
                          label=f'H (n={len(hydrogen_coords)})', zorder=5)
                
                for h_coord, carbon_node in hydrogen_coords:
                    if carbon_node in node_to_coord:
                        c_coord = node_to_coord[carbon_node]
                        ax.plot(
                            [c_coord[0], h_coord[0]],
                            [c_coord[1], h_coord[1]],
                            [c_coord[2], h_coord[2]],
                            'gray', linewidth=1, alpha=0.7, zorder=4
                        )
            
            for i, node in enumerate(main_cycle):
                if node in node_to_coord:
                    coord = node_to_coord[node]
                    ax.text(coord[0], coord[1], coord[2], 
                           f'{i+1}', fontsize=9, color='blue')
            
            ax.set_title(f"{name}\nC{n_carbon}H{2*n_carbon}", fontsize=14)
            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")
            ax.set_zlabel("Z (Å)")
            ax.legend(loc='upper left')
            
            h_coords_only = [h[0] for h in hydrogen_coords]
            all_coords = list(node_to_coord.values()) + h_coords_only
            all_x = [c[0] for c in all_coords]
            all_y = [c[1] for c in all_coords]
            all_z = [c[2] for c in all_coords]
            
            max_range = max(
                max(all_x) - min(all_x) if all_x else 1,
                max(all_y) - min(all_y) if all_y else 1,
                max(all_z) - min(all_z) if all_z else 1
            )
            mid_x = (max(all_x) + min(all_x)) / 2 if all_x else 0
            mid_y = (max(all_y) + min(all_y)) / 2 if all_y else 0
            mid_z = (max(all_z) + min(all_z)) / 2 if all_z else 0
            
            ax.set_xlim(mid_x - max_range/1.5, mid_x + max_range/1.5)
            ax.set_ylim(mid_y - max_range/1.5, mid_y + max_range/1.5)
            ax.set_zlim(mid_z - max_range/1.5, mid_z + max_range/1.5)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    程序入口：创建应用实例并运行。
    用法: python cycloalkane_app.py [分子式]
    """
    app = CycloalkaneApp()
    
    import sys
    if len(sys.argv) > 1:
        app.run(sys.argv[1], auto_select=True)
    else:
        app.run()


if __name__ == "__main__":
    main()
