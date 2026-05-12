"""
单炔烃同分异构体生成器
支持任意碳原子数(n>=2)的单炔烃同分异构体生成，基于"烷基骨架+三键插入"算法
支持3D可视化，展示碳骨架和氢原子的空间分布
OEIS A000642 验证: C2=1, C3=1, C4=2, C5=3, C6=7, C7=14, C8=32
"""

import os
import sys
import math
import warnings
import signal
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx

import matplotlib
_plt = None

def _get_plt():
    """延迟导入matplotlib.pyplot"""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt

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

def configure_chinese_font():
    """配置matplotlib支持中文显示"""
    plt = _get_plt()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')

configure_chinese_font()

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _get_alkene_generator():
    from original_programs.alkene import AlkaneIsomerGenerator
    return AlkaneIsomerGenerator

def _get_tree_generator():
    from original_programs.alkene_visualizer import AlkaneTreeGenerator
    return AlkaneTreeGenerator


class AlkyneIsomerGenerator:
    """
    单炔烃同分异构体生成器
    炔烃 CnH(2n-2) 含有碳碳三键 (C≡C)，sp杂化，键角180°
    策略：生成烷烃骨架树 → 选择边升级为三键 → 度数约束验证 → 图同构去重
    化学约束：sp碳总键数为2，原树度数>2的碳不能作为三键端点
    """
    
    def __init__(self, alkane_generator=None, num_cores=1):
        """
        Args:
            alkane_generator: 烷烃生成器实例
            num_cores: CPU核心数
        """
        self.alkane_gen = alkane_generator if alkane_generator else _get_alkene_generator()()
        self.tree_gen = _get_tree_generator()(use_parallel=False)
        self.num_cores = num_cores

    def generate_isomers(self, n_carbons: int) -> List[nx.Graph]:
        """
        生成n个碳原子的单炔烃同分异构体
        Args:
            n_carbons: 碳原子数量 (n >= 2)
        Returns:
            NetworkX图列表
        """
        if n_carbons < 2:
            return []
        
        print(f"\n{'='*60}")
        print(f"生成 C{n_carbons}H{2*n_carbons-2} 单炔烃同分异构体")
        print(f"{'='*60}")
        
        # 步骤1: 获取烷烃骨架树
        alkane_canons = self.alkane_gen.generate_isomers(n_carbons)
        print(f"[1/4] 生成了 {len(alkane_canons)} 个烷烃骨架 (C{n_carbons})")
        
        # 步骤2: 枚举三键位置并构建候选图
        print(f"[2/4] 枚举三键位置...")
        candidate_graphs = []
        total_candidates = 0
        
        for canon in alkane_canons:
            tree_adj = self.alkane_gen.canon_to_adjacency(canon)
            edges = self._get_edges(tree_adj)
            
            for u, v in edges:
                if self._can_form_triple_bond(tree_adj, u, v):
                    G = self._build_alkyne_graph(tree_adj, u, v)
                    if G:
                        candidate_graphs.append(G)
                        total_candidates += 1
        
        print(f"[3/4] 生成了 {total_candidates} 个候选炔烃结构")
        
        # 步骤3: 图同构去重
        print(f"[4/4] 图同构去重...")
        unique_graphs = self._deduplicate_isomers(candidate_graphs)
        
        print(f"\n结果: C{n_carbons} 共有 {len(unique_graphs)} 个单炔烃异构体")
        
        # OEIS验证
        expected = {2: 1, 3: 1, 4: 2, 5: 3, 6: 7, 7: 14, 8: 32}
        if n_carbons in expected:
            match = "[OK]" if len(unique_graphs) == expected[n_carbons] else "[FAIL]"
            print(f"OEIS A000642 验证: 预期 {expected[n_carbons]}, 实际 {len(unique_graphs)} {match}")
        
        return unique_graphs

    def _get_edges(self, adj: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        """从邻接表获取所有边"""
        edges = []
        for u, neighbors in adj.items():
            for v in neighbors:
                if u < v:
                    edges.append((u, v))
        return edges

    def _can_form_triple_bond(self, tree_adj: Dict[int, List[int]], u: int, v: int) -> bool:
        """
        检查是否可以在节点u和v之间形成三键
        Args:
            tree_adj: 树邻接表
            u, v: 边的两个端点
        Returns:
            是否可以形成三键
        """
        degree_u = len(tree_adj.get(u, []))
        degree_v = len(tree_adj.get(v, []))
        # sp碳总键数为2，只有原树度数<=2的节点才能作为三键端点
        return degree_u <= 2 and degree_v <= 2

    def _build_alkyne_graph(
        self, 
        tree_adj: Dict[int, List[int]], 
        u: int, 
        v: int
    ) -> Optional[nx.Graph]:
        """构建带三键的炔烃图"""
        G = nx.Graph()
        G.add_nodes_from(tree_adj.keys())
        
        triple_edge = (u, v)
        for node, neighbors in tree_adj.items():
            for neighbor in neighbors:
                if node < neighbor:
                    if (node, neighbor) == triple_edge or (neighbor, node) == triple_edge:
                        G.add_edge(node, neighbor, bond_type='triple')
                    else:
                        G.add_edge(node, neighbor, bond_type='single')
        
        # 碳原子配位数验证：sp碳max=2, sp3碳max=4
        triple_nodes = {u, v}
        for node in G.nodes():
            deg = G.degree(node)
            max_deg = 2 if node in triple_nodes else 4
            if deg > max_deg:
                return None
        
        for node in G.nodes():
            G.nodes[node]['label'] = 'C'
        
        return G

    def _deduplicate_isomers(self, candidates: List[nx.Graph]) -> List[nx.Graph]:
        """两阶段去重：WL哈希分桶 + 桶内精确同构检查"""
        hash_groups: Dict[str, List[nx.Graph]] = {}
        for G in candidates:
            h = nx.weisfeiler_lehman_graph_hash(
                G, edge_attr='bond_type', node_attr='label'
            )
            if h not in hash_groups:
                hash_groups[h] = [G]
            else:
                is_dup = False
                for existing in hash_groups[h]:
                    if self._is_isomorphic(G, existing):
                        is_dup = True
                        break
                if not is_dup:
                    hash_groups[h].append(G)

        result = []
        for group in hash_groups.values():
            result.extend(group)
        return result

    @staticmethod
    def _is_isomorphic(G1: nx.Graph, G2: nx.Graph) -> bool:
        """检查两个图是否同构（考虑键类型和原子标签）"""
        return nx.is_isomorphic(
            G1, G2,
            node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
            edge_match=lambda e1, e2: e1.get('bond_type') == e2.get('bond_type')
        )

    def get_isomer_name(self, n: int, index: int, triple_edge: Tuple[int, int] = None) -> str:
        """
        获取炔烃异构体的名称
        Args:
            n: 碳原子数
            index: 异构体索引
            triple_edge: 三键位置
        Returns:
            异构体名称
        """
        base_names = {
            2: "乙炔", 3: "丙炔", 4: "丁炔", 5: "戊炔",
            6: "己炔", 7: "庚炔", 8: "辛炔", 9: "壬炔", 10: "癸炔"
        }
        
        if n == 2:
            names = ["乙炔"]
        elif n == 3:
            names = ["丙炔 (1-丙炔)"]
        elif n == 4:
            names = ["1-丁炔", "2-丁炔"]
        elif n == 5:
            names = ["1-戊炔", "2-戊炔", "3-甲基-1-丁炔"]
        else:
            base_name = base_names.get(n, f"C{n}炔")
            return f"{base_name}-异构体{index+1} (C{n}H{2*n-2})"
        
        return f"{names[index] if index < len(names) else f'异构体{index+1}'} (C{n}H{2*n-2})"


class AlkyneIsomerVisualizer:
    """炔烃同分异构体3D可视化器，正确处理sp杂化(180°键角)和三键绘制"""
    
    # 化学键参数
    BOND_LENGTH_CC = 1.54
    BOND_LENGTH_CC_TRIPLE = 1.20
    BOND_LENGTH_CH = 1.09
    TETRAHEDRAL_ANGLE = math.acos(-1/3)  # 109.47°
    
    def __init__(self, generator=None):
        self.generator = generator
        self._setup_styles()

    def _setup_styles(self):
        """设置绘图样式"""
        self.atom_colors = {
            'C': '#333333',
            'H': '#FFFFFF',
            'sp': '#006600',
            'sp3': '#333333'
        }
        self.atom_radii = {'C': 0.6, 'H': 0.25}
        self.bond_color = '#666666'
        self.bond_color_triple = '#CC0000'
        self.bond_width = 2.5
        self.bond_width_triple = 4.5

    def visualize_isomer(self, G: nx.Graph, show: bool = True, save_path: str = None):
        """可视化单个炔烃异构体"""
        n_carbons = G.number_of_nodes()
        
        triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
        
        # 结构有效性验证
        if not self._validate_molecule_structure(G):
            print("⚠️ 结构验证失败: 存在配位数超标的碳原子，跳过此异构体")
            return
        
        coords = self._calculate_coordinates(G)
        if not coords:
            print("坐标计算失败")
            return
        
        h_coords = self._generate_hydrogen_coords(G, coords)
        all_coords = list(coords.values()) + h_coords
        
        self._print_molecule_info(G, coords, h_coords)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        self._draw_molecule(ax, G, coords, h_coords, triple_edges)
        
        if triple_edges:
            u, v = triple_edges[0]
            title = f'C{n_carbons}H{2*n_carbons-2} - C{u+1}≡C{v+1}'
        else:
            title = f'C{n_carbons}H{2*n_carbons-2}'
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.grid(True, alpha=0.3)
        self._set_equal_aspect_ratio(ax, all_coords)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至：{save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def _validate_molecule_structure(self, G: nx.Graph) -> bool:
        """验证分子结构的化学有效性"""
        print(f"\n🔍 [DEBUG] 开始结构验证... 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
        
        triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
        triple_nodes = set()
        for u, v in triple_edges:
            triple_nodes.add(u)
            triple_nodes.add(v)
        
        print(f"🔍 [DEBUG] 三键边: {triple_edges}")
        print(f"🔍 [DEBUG] 三键节点: {triple_nodes}")
        
        total_h = 0
        for node in G.nodes():
            deg = G.degree(node)
            neighbors = list(G.neighbors(node))
            
            if node in triple_nodes:
                max_deg = 2  # sp碳: 三键σ键 + 1个单键
                expected_h = max(0, 2 - len(neighbors))
            else:
                max_deg = 4  # sp3碳: 4个单键
                expected_h = max(0, 4 - len(neighbors))
            
            total_h += expected_h
            
            status = "✓"
            if deg > max_deg:
                status = f"❌ 超标! (>{max_deg})"
            
            print(f"  🔍 C{node+1}: degree={deg}, max={max_deg}, 邻居数={len(neighbors)}, "
                  f"预期H={expected_h}, 邻居列表={neighbors} {status}")
            
            if deg > max_deg:
                print(f"\n❌❌❌ 验证失败: C{node+1} 配位数超标 ❌❌❌\n")
                return False
        
        print(f"🔍 [DEBUG] 验证通过. 总预计H数={total_h}")
        return True

    def _calculate_coordinates(self, G: nx.Graph) -> Dict[int, Tuple[float, float, float]]:
        """
        碳原子3D坐标生成 — 基于BFS+方向分配算法
        三键两端sp杂化(180°)，其他sp3杂化(109.5°四面体)
        """
        import numpy as np
        from math import sqrt, acos, cos, sin, pi
        
        n = G.number_of_nodes()
        if n == 0:
            return {}
        
        def get_sp3_dirs(reference_vec):
            """生成sp3四面体4个方向"""
            ref = np.array(reference_vec, dtype=float)
            ref = ref / np.linalg.norm(ref) if np.linalg.norm(ref) > 1e-10 else np.array([1., 0., 0.])
            
            tetra_angle = acos(-1./3.)
            cos_t, sin_t = cos(tetra_angle), sin(tetra_angle)
            
            if abs(ref[2]) < 0.9:
                up = np.array([0., 0., 1.])
            else:
                up = np.array([1., 0., 0.])
            
            perp1 = np.cross(ref, up)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(ref, perp1)
            
            d1 = ref.copy()
            
            offset_angle = pi / 3  # 60°偏移使分布均匀
            cos_o, sin_o = cos(offset_angle), sin(offset_angle)
            
            d2 = cos_t * ref + sin_t * (cos_o * perp1 + sin_o * perp2)
            d3 = cos_t * ref + sin_t * (cos_o * perp1 - sin_o * perp2)
            d4 = cos_t * ref - sin_t * perp1
            
            return [d1, d2, d3, d4]
        
        def get_sp_dirs(reference_vec):
            """生成sp直线2个方向（180°）"""
            ref = np.array(reference_vec, dtype=float)
            ref = ref / np.linalg.norm(ref) if np.linalg.norm(ref) > 1e-10 else np.array([1., 0., 0.])
            return [ref.copy(), -ref.copy()]
        
        coords = [None] * n
        
        # 放置起始点（三键优先）
        triple_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
        
        if triple_edges:
            t_u, t_v = triple_edges[0][0], triple_edges[0][1]
            coords[t_u] = (0.0, 0.0, 0.0)
            coords[t_v] = (self.BOND_LENGTH_CC_TRIPLE, 0.0, 0.0)
            root = t_u
            triple_nodes = {t_u, t_v}
        else:
            coords[0] = (0.0, 0.0, 0.0)
            root = 0
            triple_nodes = set()
        
        # 构建邻接表
        adjacency = {}
        for node in G.nodes():
            adjacency[node] = [nb for nb in G.neighbors(node)]
        
        # BFS建立父子关系
        visited = {root}
        queue = [root]
        parent_of = {}
        
        while queue:
            curr = queue.pop(0)
            for nb in adjacency.get(curr, []):
                if nb not in visited and nb < n:
                    visited.add(nb)
                    parent_of[nb] = curr
                    queue.append(nb)
        
        # 处理孤立节点
        for i in range(n):
            if i not in visited:
                visited.add(i)
                parent_of[i] = i
                q = [i]
                while q:
                    nd = q.pop(0)
                    for nb in adjacency.get(nd, []):
                        if nb not in visited and nb < n:
                            visited.add(nb)
                            parent_of[nb] = nd
                            q.append(nb)
        
        # 按BFS顺序计算坐标
        child_count = {}
        
        queue = [root]
        visited.clear()
        visited.add(root)
        
        while queue:
            node = queue.pop(0)
            
            for nb in adjacency.get(node, []):
                if nb >= n:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
            
            unplaced_neighbors = [nb for nb in adjacency.get(node, []) 
                                 if nb < n and coords[nb] is None]
            
            if not unplaced_neighbors:
                continue
            
            px, py, pz = coords[node]
            parent_pos = np.array([px, py, pz])
            
            if node not in child_count:
                child_count[node] = 0
            
            is_sp = node in triple_nodes
            
            triple_bond_vec = None
            if is_sp:
                for other_nb in adjacency.get(node, []):
                    if other_nb in triple_nodes and other_nb != node:
                        if coords[other_nb] is not None:
                            v = np.array(coords[other_nb]) - parent_pos
                            vn = np.linalg.norm(v)
                            if vn > 1e-6:
                                triple_bond_vec = v / vn
                        break
            
            existing_dirs = []
            for other_nb in adjacency.get(node, []):
                if coords[other_nb] is not None and other_nb != node:
                    v = np.array(coords[other_nb]) - parent_pos
                    vn = np.linalg.norm(v)
                    if vn > 1e-6:
                        existing_dirs.append(v / vn)
            
            local_child_idx = 0
            
            for neighbor in unplaced_neighbors:
                child_count[node] += 1
                
                is_triple_bond = (node in triple_nodes and neighbor in triple_nodes)
                bond_len = self.BOND_LENGTH_CC_TRIPLE if is_triple_bond else self.BOND_LENGTH_CC
                
                direction = None
                
                if is_triple_bond and triple_bond_vec is not None:
                    direction = triple_bond_vec.copy()
                    
                elif is_sp and triple_bond_vec is not None:
                    direction = -triple_bond_vec
                    
                elif is_sp:
                    dirs = get_sp_dirs(np.array([1., 0., 0.]))
                    direction = dirs[local_child_idx % len(dirs)]
                    
                else:
                    # sp3处理
                    if existing_dirs:
                        base = existing_dirs[0]
                        for ex_dir in existing_dirs[1:]:
                            if abs(np.dot(base, ex_dir)) < 0.9:
                                base = ex_dir
                                break
                    else:
                        base = np.array([1., 0., 0.])
                    
                    dirs = get_sp3_dirs(base)
                    
                    # 排除与已有方向太接近的（阈值0.9 ≈ 25°）
                    available = []
                    for d in dirs:
                        too_close = False
                        for ex in existing_dirs:
                            if abs(np.dot(d, ex)) > 0.9:
                                too_close = True
                                break
                        if not too_close:
                            available.append(d)
                    
                    if available:
                        direction = available[local_child_idx % len(available)]
                    else:
                        direction = dirs[local_child_idx % len(dirs)]
                
                # 归一化并设置坐标
                dn = np.linalg.norm(direction)
                if dn > 1e-8:
                    direction /= dn
                
                new_pos = parent_pos + direction * bond_len
                coords[neighbor] = (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                
                existing_dirs.append(direction.copy())
                local_child_idx += 1
        
        # 处理仍未分配的节点
        for i in range(n):
            if coords[i] is None:
                for nb in adjacency.get(i, []):
                    if nb < n and coords[nb] is not None:
                        cx, cy, cz = coords[nb]
                        coords[i] = (cx + self.BOND_LENGTH_CC, cy, cz)
                        break
                if coords[i] is None:
                    coords[i] = (float((i+1) * 1.54), 0.0, 0.0)
        
        # 非键连原子排斥力优化
        coords_dict = {}
        for i in range(n):
            if coords[i] is not None:
                coords_dict[i] = coords[i]
        
        coords_dict = self._apply_nonbonded_repulsion(coords_dict, G)
        
        return coords_dict

    def _calculate_coordinates_fallback(self, G: nx.Graph) -> Dict[int, Tuple[float, float, float]]:
        """没有三键时的后备坐标计算"""
        coords = {}
        nodes = list(G.nodes())
        
        if not nodes:
            return {}
        
        coords[nodes[0]] = (0.0, 0.0, 0.0)
        
        for node in nodes[1:]:
            for neighbor in G.neighbors(node):
                if neighbor in coords:
                    coord = self._calculate_sp3_child_coord(G, coords, node, neighbor)
                    if coord is not None:
                        coords[node] = coord
                        break
            if node not in coords:
                coords[node] = (len(coords) * 1.5, 0.0, 0.0)
        
        return coords

    def _apply_nonbonded_repulsion(self, coords: Dict, G: nx.Graph) -> Dict:
        """对非键连原子施加强力排斥力，消除视觉上的"假环"
        找出距离过近的非相邻原子对，施加强力推开
        """
        n = len(coords)
        if n <= 1:
            return coords
        
        node_list = list(coords.keys())
        pos = np.array([coords[node] for node in node_list], dtype=np.float64)
        idx_map = {node: i for i, node in enumerate(node_list)}
        
        bonded_pairs = set()
        for u, v in G.edges():
            if u in idx_map and v in idx_map:
                pair = tuple(sorted([idx_map[u], idx_map[v]]))
                bonded_pairs.add(pair)
        
        MIN_DIST = 2.4
        
        # 阶段1：粗推
        phase1_iters = 200
        phase1_step = 0.35
        
        for _ in range(phase1_iters):
            max_move = 0.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in bonded_pairs:
                        continue
                    
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    
                    if dist < 1e-8:
                        rand_dir = np.random.randn(3)
                        rand_dir /= np.linalg.norm(rand_dir)
                        push = rand_dir * 0.5
                        pos[i] -= push
                        pos[j] += push
                        max_move = max(max_move, 0.5)
                        
                    elif dist < MIN_DIST:
                        overlap = MIN_DIST - dist
                        force = (overlap / dist) * diff
                        step = force * phase1_step * 0.5
                        
                        pos[i] -= step
                        pos[j] += step
                        max_move = max(max_move, np.linalg.norm(step))
            
            if max_move < 0.001:
                break
        
        # 阶段2：精调
        phase2_iters = 100
        phase2_step = 0.15
        
        for _ in range(phase2_iters):
            max_move = 0.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in bonded_pairs:
                        continue
                    
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0 and dist < MIN_DIST:
                        overlap = MIN_DIST - dist
                        force = (overlap / dist) * diff
                        step = force * phase2_step * 0.5
                        
                        pos[i] -= step
                        pos[j] += step
                        max_move = max(max_move, np.linalg.norm(step))
            
            if max_move < 0.0005:
                break
        
        # 键长校正
        TRIPLE_BOND_LEN = 1.20
        SINGLE_BOND_LEN = 1.54
        MIN_SINGLE_BOND = 1.45
        
        triple_bonded_pairs = set()
        for u, v, d in G.edges(data=True):
            if d.get('bond_type') == 'triple':
                pair = tuple(sorted([idx_map[u], idx_map[v]]))
                triple_bonded_pairs.add(pair)
        
        for _ in range(50):
            max_correction = 0.0
            for (i, j) in bonded_pairs:
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff)
                
                if dist < 1e-8:
                    continue
                
                if (i, j) in triple_bonded_pairs or (j, i) in triple_bonded_pairs:
                    target_len = TRIPLE_BOND_LEN
                else:
                    target_len = SINGLE_BOND_LEN
                
                error = dist - target_len
                if abs(error) > 0.01:
                    correction = (error / dist) * diff * 0.3
                    pos[i] += correction
                    pos[j] -= correction
                    max_correction = max(max_correction, abs(error))
            
            if max_correction < 0.005:
                break
        
        for (i, j) in bonded_pairs:
            is_triple = (i, j) in triple_bonded_pairs or (j, i) in triple_bonded_pairs
            dist = np.linalg.norm(pos[j] - pos[i])
            
            if not is_triple and dist < MIN_SINGLE_BOND:
                diff = pos[j] - pos[i]
                direction = diff / np.linalg.norm(diff)
                stretch = (MIN_SINGLE_BOND - dist) * 0.5
                pos[i] -= direction * stretch
                pos[j] += direction * stretch
        
        result = {}
        for i, node in enumerate(node_list):
            result[node] = (float(pos[i][0]), float(pos[i][1]), float(pos[i][2]))
        
        return result

    def _calculate_sp3_child_coord(
        self, 
        G: nx.Graph, 
        coords: Dict[int, np.ndarray],
        child: int, 
        parent: int
    ) -> np.ndarray:
        """计算sp3碳子节点的坐标（四面体结构，109.5°键角）"""
        p_coords = coords[parent]
        
        siblings = [nb for nb in G.neighbors(parent) if nb in coords and nb != child]
        bond_len = self.BOND_LENGTH_CC
        depth = self._calculate_node_depth(coords, child, parent)
        
        if not siblings:
            if depth % 3 == 0:
                direction = np.array([1.0, 0.0, 0.0])
            elif depth % 3 == 1:
                direction = np.array([0.0, 1.0, 0.0])
            else:
                direction = np.array([0.0, 0.0, 1.0])
        elif len(siblings) == 1:
            direction = self._tetrahedral_one_bond_v3(p_coords - coords[siblings[0]], depth)
        elif len(siblings) == 2:
            direction = self._tetrahedral_two_bonds_v3(
                p_coords - coords[siblings[0]],
                p_coords - coords[siblings[1]],
                depth
            )
        elif len(siblings) == 3:
            direction = self._tetrahedral_three_bonds(
                p_coords - coords[siblings[0]],
                p_coords - coords[siblings[1]],
                p_coords - coords[siblings[2]]
            )
        else:
            direction = np.array([1.0, 0.0, 0.0])
        
        direction = direction / np.linalg.norm(direction)
        return p_coords + direction * bond_len
    
    def _calculate_node_depth(self, coords: Dict[int, np.ndarray], 
                               child: int, parent: int) -> int:
        """计算节点相对于根的大致深度（用于方向交替）"""
        # 简单启发式：基于父节点坐标到原点的距离估算深度
        dist = np.linalg.norm(coords[parent])
        return int(dist / 1.5)  # 每1.5Å算一代

    def _tetrahedral_one_bond_v3(self, vec: np.ndarray, depth: int) -> np.ndarray:
        """根据depth参数交替选择参考轴，计算四面体方向(109.5°)"""
        v1 = vec / np.linalg.norm(vec)
        
        ref_options = [
            (np.array([0, 0, 1]), "Z"),
            (np.array([0, 1, 0]), "Y"),
            (np.array([1, 0, 0]), "X"),
        ]
        
        arb, _ = ref_options[depth % 3]
        
        if abs(np.dot(v1, arb)) > 0.99:
            arb = np.array([1, 0, 0]) if abs(np.dot(v1, np.array([1,0,0]))) < 0.99 else np.array([0, 1, 0])
        
        perp = np.cross(v1, arb)
        perp_norm = np.linalg.norm(perp)
        
        if perp_norm < 1e-6:
            for alt_arb in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
                perp = np.cross(v1, alt_arb)
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-6:
                    break
        
        perp /= perp_norm
        
        cos_theta = -1/3
        sin_theta = np.sqrt(8)/3
        
        v_new = cos_theta * v1 + sin_theta * np.cross(perp, v1) + (1 - cos_theta) * np.dot(perp, v1) * perp
        
        return v_new / np.linalg.norm(v_new)

    def _tetrahedral_two_bonds_v3(self, vec1: np.ndarray, vec2: np.ndarray, depth: int) -> np.ndarray:
        """根据depth参数计算四面体方向"""
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)
        
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        # 处理退化情况
        if normal_norm < 1e-6:
            return self._tetrahedral_one_bond_v3(v1, depth)
        
        normal /= normal_norm
        
        dot_v1_v2 = np.dot(v1, v2)
        
        # 处理反向情况
        if abs(1 + dot_v1_v2) < 1e-6:
            return self._tetrahedral_one_bond_v3(v1, depth)
        
        # *** 根据深度选择正负法线 ***
        sign = 1 if depth % 2 == 0 else -1
        
        # 计算系数 c 和 d（标准四面体几何）
        dot_product = -1/3  # 目标点积（109.47°）
        s = 1 + dot_product  # = 2/3
        t = 1 + dot_v1_v2
        
        d_sq = (s * t - 2 * (dot_product + dot_v1_v2 * dot_product)) / (
            2 * (1 - dot_v1_v2**2))
        d_sq = max(d_sq, 0)
        
        c = dot_product - dot_v1_v2 * np.sqrt(d_sq) if depth % 2 == 0 else \
            dot_product + dot_v1_v2 * np.sqrt(d_sq)
        d = sign * np.sqrt(max(d_sq, 0))
        
        v4 = c * (v1 + v2) + d * normal
        v4_norm = np.linalg.norm(v4)
        
        if v4_norm < 1e-6:
            return self._tetrahedral_one_bond_v3(v1, depth)
        
        return v4 / v4_norm

    def _tetrahedral_one_bond(self, vec: np.ndarray) -> np.ndarray:
        """根据1个已有键计算四面体方向（109.5°）"""
        v1 = vec / np.linalg.norm(vec)
        
        if abs(v1[2]) < 0.9:
            arb = np.array([0, 0, 1])
        else:
            # v1 有较大 Z 分量 → 用 X 或 Y 轴
            if abs(v1[0]) < 0.9:
                arb = np.array([1, 0, 0])
            else:
                arb = np.array([0, 1, 0])
        
        perp = np.cross(v1, arb)
        perp /= np.linalg.norm(perp)
        
        # 四面体角相关参数
        cos_theta = -1/3          # cos(109.47°)
        sin_theta = np.sqrt(8)/3  # sin(109.47°)
        
        # Rodrigues旋转公式
        v_new = cos_theta * v1 + sin_theta * np.cross(perp, v1) + (1 - cos_theta) * np.dot(perp, v1) * perp
        
        return v_new / np.linalg.norm(v_new)

    def _tetrahedral_two_bonds(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """根据2个已有键计算四面体方向"""
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)
        
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < 1e-6:
            if abs(v1[2]) < 0.9:
                perp = np.cross(v1, [0, 0, 1])
            elif abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp /= np.linalg.norm(perp)
            return self._tetrahedral_one_bond(perp)
        
        normal /= normal_norm
        
        dot_v1_v2 = np.dot(v1, v2)
        
        if abs(1 + dot_v1_v2) < 1e-6:
            if abs(v1[2]) < 0.9:
                perp = np.cross(v1, [0, 0, 1])
            else:
                perp = np.cross(v1, [1, 0, 0] if abs(v1[0]) >= 0.9 else [0, 1, 0])
            perp /= np.linalg.norm(perp)
            return self._tetrahedral_one_bond(perp)
        
        c = -1 / (3 * (1 + dot_v1_v2))
        v1_plus_v2_mag_sq = 2 * (1 + dot_v1_v2)
        normal_mag_sq = 1 - dot_v1_v2**2
        
        term1 = c**2 * v1_plus_v2_mag_sq
        if 1 - term1 < 0:
            v4_unnorm = c * (v1 + v2)
            return v4_unnorm / np.linalg.norm(v4_unnorm)
        
        d_sq = (1 - term1) / normal_mag_sq
        if d_sq < 0:
            v4_unnorm = c * (v1 + v2)
            return v4_unnorm / np.linalg.norm(v4_unnorm)
        
        d = np.sqrt(d_sq)
        v4_unnorm = c * (v1 + v2) + d * normal
        v4_unnorm_check_neg = c * (v1 + v2) - d * normal
        
        # 选择与 v1 成正确角度的方向
        test_dot_pos = np.dot(v4_unnorm / np.linalg.norm(v4_unnorm), v1)
        test_dot_neg = np.dot(v4_unnorm_check_neg / np.linalg.norm(v4_unnorm_check_neg), v1)
        
        if abs(test_dot_pos - (-1/3)) < abs(test_dot_neg - (-1/3)):
            final_v4 = v4_unnorm
        else:
            final_v4 = v4_unnorm_check_neg
        
        return final_v4 / np.linalg.norm(final_v4)

    def _tetrahedral_three_bonds(self, vec1: np.ndarray, vec2: np.ndarray, vec3: np.ndarray) -> np.ndarray:
        """根据3个已有键计算四面体方向"""
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)
        v3 = vec3 / np.linalg.norm(vec3)
        
        v4 = -(v1 + v2 + v3)
        norm = np.linalg.norm(v4)
        
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        
        return v4 / norm

    def _generate_hydrogen_coords(self, G: nx.Graph, c_coords: Dict[int, Tuple]) -> List[Tuple[float, float, float]]:
        """生成氢原子坐标"""
        h_coords = []
        
        triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
        triple_nodes = set()
        if triple_edges:
            triple_nodes.add(triple_edges[0][0])
            triple_nodes.add(triple_edges[0][1])
        
        for node in G.nodes():
            c_pos = np.array(c_coords[node])
            neighbors = list(G.neighbors(node))
            num_attached = len(neighbors)
            
            is_sp = node in triple_nodes
            
            if is_sp:
                num_single_bonds = num_attached - 1
                
                if num_single_bonds >= 1:
                    num_h = 0
                else:
                    num_h = 1
            else:
                num_h = 4 - num_attached
                if num_h < 0:
                    print(f"  ⚠️ C{node+1} 配位数异常: degree={num_attached} > 4, 跳过氢原子生成")
                    continue
            
            if num_h <= 0:
                continue
            
            bond_directions = []
            for neighbor in neighbors:
                if neighbor in c_coords:
                    vec = np.array(c_coords[neighbor]) - c_pos
                    bond_directions.append(vec / np.linalg.norm(vec))
            
            if is_sp:
                triple_dir = None
                for nb in neighbors:
                    if G[node][nb].get('bond_type') == 'triple':
                        triple_dir = np.array(c_coords[nb]) - c_pos
                        triple_dir /= np.linalg.norm(triple_dir)
                        break
                
                if triple_dir is not None:
                    h_dir = -triple_dir
                else:
                    h_dir = np.array([1.0, 0.0, 0.0])
                
                h_pos = c_pos + h_dir * self.BOND_LENGTH_CH
                h_coords.append(tuple(h_pos))
                
            else:
                h_dirs = self._generate_tetrahedral_h_directions(bond_directions, num_h)
                for h_dir in h_dirs:
                    h_pos = c_pos + h_dir * self.BOND_LENGTH_CH
                    h_coords.append(tuple(h_pos))
        
        return h_coords

    def _generate_tetrahedral_h_directions(self, bond_directions: List[np.ndarray], num_h: int) -> List[np.ndarray]:
        """生成四面体分布的氢原子方向"""
        if num_h == 1:
            if not bond_directions:
                return [np.array([1.0, 0.0, 0.0])]
            elif len(bond_directions) == 1:
                return [self._tetrahedral_one_bond(bond_directions[0])]
            elif len(bond_directions) == 2:
                return [self._tetrahedral_two_bonds(bond_directions[0], bond_directions[1])]
            else:
                return [self._tetrahedral_three_bonds(
                    bond_directions[0], bond_directions[1], bond_directions[2]
                )]
        
        elif num_h == 2:
            if not bond_directions:
                return [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
            
            if len(bond_directions) == 1:
                h1 = self._tetrahedral_one_bond(bond_directions[0])
                _cands = [np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.])]
                _ref = min(_cands, key=lambda c: abs(np.dot(h1, c)))
                perp = np.cross(h1, _ref)
                perp /= np.linalg.norm(perp)
                h2 = self._tetrahedral_one_bond(-perp)
                return [h1, h2]
            elif len(bond_directions) == 2:
                v1, v2 = bond_directions[0], bond_directions[1]
                
                perp = np.cross(v1, v2)
                perp_norm = np.linalg.norm(perp)
                
                if perp_norm < 1e-6:
                    _cands = [np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.])]
                    _ref = min(_cands, key=lambda c: abs(np.dot(v1, c)))
                    perp = np.cross(v1, _ref)
                    perp /= np.linalg.norm(perp)
                
                bisector = (v1 + v2) / np.linalg.norm(v1 + v2)
                
                cos_theta = -1/3
                sin_theta = np.sqrt(1 - cos_theta**2)
                
                h1 = cos_theta * bisector + sin_theta * perp
                h2 = cos_theta * bisector - sin_theta * perp
                
                return [h1 / np.linalg.norm(h1), h2 / np.linalg.norm(h2)]
            else:
                return [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
        
        elif num_h == 3:
            if not bond_directions:
                return [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([-0.5, 0.866, 0.0]),
                    np.array([-0.5, -0.866, 0.0])
                ]
            
            bond_dir = bond_directions[0]
            
            candidates = [np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.])]
            ref = min(candidates, key=lambda c: abs(np.dot(bond_dir, c)))
            perp1 = np.cross(bond_dir, ref)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(bond_dir, perp1)
            perp2 /= np.linalg.norm(perp2)
            
            cos_theta = -1/3
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            h_dirs = []
            for angle in [0, 2*np.pi/3, 4*np.pi/3]:
                h_dir = cos_theta * bond_dir + sin_theta * (perp1 * np.cos(angle) + perp2 * np.sin(angle))
                h_dirs.append(h_dir / np.linalg.norm(h_dir))
            
            return h_dirs
        
        return []

    def _print_molecule_info(
        self, 
        G: nx.Graph, 
        c_coords: Dict[int, Tuple], 
        h_coords: List[Tuple]
    ):
        """打印分子信息（含配位数诊断）"""
        n = G.number_of_nodes()
        triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
        triple_nodes = set()
        for u, v in triple_edges:
            triple_nodes.add(u)
            triple_nodes.add(v)
        
        print("\n" + "=" * 60)
        print(f"炔烃分子信息")
        print("=" * 60)
        print(f"分子式: C{n}H{2*n-2}")
        
        if triple_edges:
            u, v = triple_edges[0]
            print(f"三键位置: C{u+1}≡C{v+1}")
        
        # 【关键诊断】碳原子配位数检查
        print(f"\n【⚠️ 配位数诊断】")
        print("-" * 55)
        has_error = False
        for node in sorted(G.nodes()):
            deg = G.degree(node)
            max_deg = 2 if node in triple_nodes else 4
            neighbors = list(G.neighbors(node))
            
            status = "✓"
            if deg > max_deg:
                status = f"❌ 超标! (>{max_deg})"
                has_error = True
            
            h_count = self._count_hydrogens(G, node)
            print(f"  C{node+1}: degree={deg}/{max_deg} 邻居={neighbors} H={h_count} {status}")
        
        if has_error:
            print("  ⚠️ 检测到配位数超标! 结构可能无效 ⚠️")
        print("-" * 55)
        
        # 碳原子坐标
        print("\n【碳原子坐标】")
        print("-" * 50)
        print(f"{'原子':<6} {'X':>10} {'Y':>10} {'Z':>10}")
        print("-" * 50)
        for i in range(n):
            if i in c_coords:
                coord = c_coords[i]
                print(f"C{i+1:<4} {coord[0]:>10.4f} {coord[1]:>10.4f} {coord[2]:>10.4f}")
        
        # 氢原子坐标（详细输出）
        print("\n【氢原子坐标】")
        print("-" * 55)
        print(f"{'原子':<6} {'X':>10} {'Y':>10} {'Z':>10} {'所属碳'}")
        print("-" * 55)
        
        # 构建每个碳的氢原子索引映射
        h_idx = 0
        for node in G.nodes():
            if node not in c_coords:
                continue
            num_h = self._count_hydrogens(G, node)
            
            for j in range(num_h):
                if h_idx < len(h_coords):
                    h_pos = h_coords[h_idx]
                    print(f"H{h_idx+1:<4} {h_pos[0]:>10.4f} {h_pos[1]:>10.4f} {h_pos[2]:>10.4f} C{node+1}")
                    h_idx += 1
        
        if len(h_coords) > h_idx:
            for extra_idx in range(h_idx, len(h_coords)):
                h_pos = h_coords[extra_idx]
                print(f"H{extra_idx+1:<4} {h_pos[0]:>10.4f} {h_pos[1]:>10.4f} {h_pos[2]:>10.4f} (extra)")
        
        print(f"\n总氢原子数: {len(h_coords)}")
        print("=" * 60)

    def _draw_molecule(
        self, 
        ax, 
        G: nx.Graph, 
        c_coords: Dict[int, Tuple],
        h_coords: List[Tuple],
        triple_edges: List[Tuple]
    ):
        """绘制分子结构"""
        n = G.number_of_nodes()
        triple_edges_set = set(triple_edges)
        
        # 绘制 C-C 键
        for u, v, data in G.edges(data=True):
            if u in c_coords and v in c_coords:
                c1 = np.array(c_coords[u])
                c2 = np.array(c_coords[v])
                
                if data.get('bond_type') == 'triple':
                    self._draw_triple_bond(ax, c1, c2)
                else:
                    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                           c=self.bond_color, linewidth=self.bond_width)
        
        # 绘制碳原子
        for node in G.nodes():
            if node in c_coords:
                c_pos = np.array(c_coords[node])
                color = self.atom_colors.get('sp', self.atom_colors['C'])
                ax.scatter(*c_pos, c=color, s=150, edgecolors='black', linewidths=1)
                ax.text(c_pos[0], c_pos[1], c_pos[2], f'C{node+1}',
                       fontsize=7, ha='center', va='center')
        
        # 绘制氢原子和 C-H 键
        h_counter = 0
        for node in G.nodes():
            if node not in c_coords:
                continue
            c_pos = np.array(c_coords[node])
            num_h = self._count_hydrogens(G, node)
            
            for _ in range(num_h):
                if h_counter < len(h_coords):
                    h_pos = np.array(h_coords[h_counter])
                    ax.plot([c_pos[0], h_pos[0]], [c_pos[1], h_pos[1]], [c_pos[2], h_pos[2]],
                           c=self.bond_color, linewidth=self.bond_width * 0.6)
                    ax.scatter(*h_pos, c=self.atom_colors['H'], s=80,
                              edgecolors='gray', linewidths=0.5)
                    h_counter += 1

    def _draw_triple_bond(self, ax, c1: np.ndarray, c2: np.ndarray):
        """绘制三键（三条平行线）"""
        bond_vec = c2 - c1
        bond_len = np.linalg.norm(bond_vec)
        if bond_len < 1e-6:
            return
        
        bond_vec_norm = bond_vec / bond_len
        
        # 寻找垂直方向
        perp = None
        for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
            perp = np.cross(bond_vec_norm, axis)
            if np.linalg.norm(perp) > 1e-6:
                break
        
        perp = perp / np.linalg.norm(perp)
        
        # 三键偏移 - 三条线
        offsets = [-0.12, 0.0, 0.12]
        
        # 绘制三条平行线（中间一条 + 两侧各一条）
        for offset_val in offsets:
            offset = offset_val * perp
            if abs(offset_val) < 1e-6:
                # 中间那条用实线
                ax.plot(
                    [c1[0], c2[0]],
                    [c1[1], c2[1]],
                    [c1[2], c2[2]],
                    c=self.bond_color_triple, linewidth=self.bond_width_triple
                )
            else:
                # 两侧的线稍微细一点
                ax.plot(
                    [c1[0] + offset[0], c2[0] + offset[0]],
                    [c1[1] + offset[1], c2[1] + offset[1]],
                    [c1[2] + offset[2], c2[2] + offset[2]],
                    c=self.bond_color_triple, linewidth=self.bond_width_triple * 0.8
                )

    def _count_hydrogens(self, G: nx.Graph, node: int) -> int:
        """计算某碳原子上的氢原子数"""
        neighbors = list(G.neighbors(node))
        
        is_triple_end = any(
            G[node][nb].get('bond_type') == 'triple' for nb in neighbors
        )
        
        if is_triple_end:
            num_single_bonds = len(neighbors) - 1
            
            if num_single_bonds >= 1:
                return 0
            else:
                return 1
        else:
            h_count = 4 - len(neighbors)
            return max(0, h_count)

    def _set_equal_aspect_ratio(self, ax, coords: List):
        """设置 3D 图形等比例"""
        if not coords:
            return
        
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        z = [c[2] for c in coords]
        
        max_range = max(max(x)-min(x), max(y)-min(y), max(z)-min(z), 1)
        center = ((max(x)+min(x))/2, (max(y)+min(y))/2, (max(z)+min(z))/2)
        limit = max_range / 2 + max_range * 0.1
        
        ax.set_xlim(center[0]-limit, center[0]+limit)
        ax.set_ylim(center[1]-limit, center[1]+limit)
        ax.set_zlim(center[2]-limit, center[2]+limit)





def main():
    """交互式命令行入口"""
    global _interrupted
    _interrupted = False

    def _signal_handler(signum, frame):
        global _interrupted
        _interrupted = True
        print("\n\n[中断] 正在停止计算，请稍候...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _signal_handler)

    print("=" * 70)
    print("单炔烃同分异构体生成工具")
    print("分子式: CnH(2n-2)")
    print("OEIS A000642 验证: C4=2, C5=3, C6=7, C7=14, C8=32")
    print("=" * 70)
    print()

    alkyne_gen = AlkyneIsomerGenerator()
    visualizer = AlkyneIsomerVisualizer(generator=alkyne_gen)

    try:
        n = int(input("请输入碳原子数量 (n >= 2): "))
        if n < 2:
            print("错误：碳原子数量必须 >= 2")
            return
    except ValueError:
        print("错误：请输入有效的数字")
        return

    print(f"\n开始生成 C{n} 的单炔烃异构体...")
    print()

    try:
        isomers = alkyne_gen.generate_isomers(n)
    except (KeyboardInterrupt, Exception):
        if isinstance(sys.exc_info()[0], KeyboardInterrupt) or _interrupted:
            print("\n\n[中断] 计算已停止")
        else:
            print(f"\n\n[错误] 计算过程中出现错误: {sys.exc_info()[1]}")
        print("感谢使用！")
        return

    if isomers:
        print(f"\n异构体列表 (共 {len(isomers)} 个):")
        for i in range(min(len(isomers), 20)):
            G = isomers[i]
            triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
            if triple_edges:
                u, v = triple_edges[0]
                name = alkyne_gen.get_isomer_name(n, i, triple_edges[0])
                print(f"  {i+1}. C{u+1}≡C{v+1} ({name})")
        if len(isomers) > 20:
            print(f"  ... 还有 {len(isomers) - 20} 个")

        print("\n选项:")
        print("1. 可视化单个异构体")
        print("2. 保存异构体列表到文件")
        print("3. 退出")

        choice = input("\n请选择 (1/2/3): ").strip()

        if choice == "1":
            try:
                idx_input = input(f"请输入异构体编号 (1-{len(isomers)}): ")
                if idx_input.strip():
                    idx = int(idx_input) - 1
                    if 0 <= idx < len(isomers):
                        print(f"\n正在显示第 {idx+1} 个异构体...")
                        visualizer.visualize_isomer(isomers[idx], show=True)
                    else:
                        print("错误：异构体编号超出范围")
                else:
                    print("已取消可视化")
            except ValueError:
                print("错误：请输入有效的数字")

        elif choice == "2":
            filename = input("请输入输出文件名 (默认 C{n}_alkyne_isomers.txt): ").strip()
            if not filename:
                filename = f"C{n}_alkyne_isomers.txt"

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"C{n} 单炔烃同分异构体列表\n")
                    f.write(f"分子式: C{n}H{2*n-2}\n")
                    f.write(f"总数: {len(isomers)}\n")
                    f.write("=" * 70 + "\n\n")
                    for i, G in enumerate(isomers, 1):
                        triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'triple']
                        if triple_edges:
                            u, v = triple_edges[0]
                            f.write(f"{i}. C{u+1}≡C{v+1}\n")
                print(f"已保存到: {filename}")
            except Exception as e:
                print(f"保存失败: {e}")

        else:
            print("已退出")

    else:
        print("没有找到异构体")


def test():
    """测试函数：验证 OEIS 数据"""
    print("\n" + "=" * 60)
    print("OEIS A000642 验证测试")
    print("=" * 60)
    
    gen = AlkyneIsomerGenerator()
    expected = {2: 1, 3: 1, 4: 2, 5: 3, 6: 7, 7: 14, 8: 32}
    
    for n, exp_count in expected.items():
        isomers = gen.generate_isomers(n)
        actual = len(isomers)
        status = "[OK]" if actual == exp_count else f"[FAIL] (预期 {exp_count})"
        print(f"C{n}: {actual} {status}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
