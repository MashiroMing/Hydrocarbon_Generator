"""
================================================================================
烷烃/单烯烃同分异构体生成工具
================================================================================

【功能说明】
  - 支持任意碳原子数的饱和烷烃 (CnH2n+2) 同分异构体生成
  - 支持单烯烃 (CnH2n) 同分异构体生成
  - 使用无标号树算法枚举所有可能的同分异构体结构
  - 支持3D可视化，展示碳骨架和氢原子的空间分布
  - 支持并行计算加速大规模异构体生成

【核心算法】
  - 烷烃分子可建模为无环连通图（树结构）
  - 每个碳原子最多形成4个化学键（度≤4）
  - 烯烃在烷烃基础上增加一个双键

【验证数据】
  - 烷烃: C1=1, C2=1, C3=1, C4=2, C5=3, C6=5, C7=9, C8=18, C9=35, C10=75
  - 烯烃 (OEIS A000631): C2=1, C3=1, C4=3, C5=5, C6=13, C7=27, C8=66, C9=153, C10=377

【文件结构】
  - AlkaneIsomerGenerator 类：烷烃异构体生成器
  - AlkeneIsomerGenerator 类：烯烃异构体生成器
  - AlkaneIsomerVisualizer 类：3D可视化器
  - main() 函数：命令行交互入口

【使用方法】
  python alkene.py
  # 然后按提示输入碳原子数量

================================================================================
"""

# ============================================================
# 第一部分：导入依赖库
# ============================================================

# 标准库导入
import os              # 文件路径操作
import sys             # 系统参数和异常处理
import math            # 数学运算
import warnings        # 警告控制
import signal          # 信号处理
import traceback       # 堆栈跟踪
from datetime import datetime

# 类型提示
from typing import List, Tuple, Optional, Dict

# 第三方库
import numpy as np                     # 数值计算
import itertools                        # 迭代器工具
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp            # 多进程并行计算
import matplotlib.pyplot as plt         # 绘图
from matplotlib.gridspec import GridSpec  # 子图网格布局
import matplotlib.cm as cm              # 颜色映射

# 导入 networkx 用于图的规范化
import networkx as nx

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可以显示进度条: pip install tqdm")

# ============================================================
# 第二部分：全局异常钩子
# ============================================================

_original_excepthook = sys.excepthook
_keyboard_interrupt_occurred = False

def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """自定义全局异常钩子，处理Ctrl+C中断"""
    global _keyboard_interrupt_occurred
    if exc_type is KeyboardInterrupt:
        _keyboard_interrupt_occurred = True
        return
    _original_excepthook(exc_type, exc_value, exc_traceback)

sys.excepthook = _custom_excepthook

# ============================================================
# 第三部分：matplotlib延迟导入
# ============================================================

import matplotlib
_plt = None
_GridSpec = None
_cm = None

def _get_plt():
    """延迟导入matplotlib.pyplot"""
    global _plt, _GridSpec, _cm
    if _plt is None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.cm as cm
        _plt = plt
        _GridSpec = GridSpec
        _cm = cm
    return _plt

# ============================================================
# 第四部分：中文字体配置
# ============================================================

def configure_chinese_font():
    """配置matplotlib支持中文显示"""
    plt = _get_plt()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')

configure_chinese_font()

# ============================================================
# 第五部分：快速图签名缓存
# ============================================================

# 使用简单的图字符串表示作为快速哈希
_graph_signature_cache = {}

def get_fast_graph_signature(G):
    """
    生成图的快速签名字符串
    使用邻接矩阵和双键位置的组合作为签名
    """
    # 获取节点数
    n = G.number_of_nodes()
    if n == 0:
        return "empty"

    # 获取排序后的节点列表
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # 生成邻接矩阵字符串
    adj_str_parts = []
    for node in nodes:
        neighbors = sorted([node_to_idx[nbr] for nbr in G.neighbors(node)])
        adj_str_parts.append("_".join(map(str, neighbors)))

    # 获取双键信息（使用节点索引而非节点ID）
    double_bond_edges = []
    for u, v, data in G.edges(data=True):
        if data.get('bond_type') == 'double':
            double_bond_edges.append((node_to_idx[u], node_to_idx[v]))
    double_bond_edges.sort()

    # 组合所有信息
    return f"{n}|{'|'.join(adj_str_parts)}|{double_bond_edges}"
configure_chinese_font()

# ==================== 模块级并行工作函数 ====================
# 这些函数必须在模块级别定义，以便Windows的multiprocessing可以正常工作

def check_isomorphism_worker(G, canonical_forms_bytes_list):
    """
    并行工作函数：检查一个图是否与已有图同构
    这个函数在子进程中运行，必须接收可序列化的参数
    """
    def custom_edge_match(e1, e2):
        """自定义边匹配：严格比较 bond_type 是否完全相同"""
        return e1.get('bond_type') == e2.get('bond_type')

    try:
        # 将字节列表转换回图对象
        canonical_forms = []
        for graph_bytes in canonical_forms_bytes_list:
            try:
                G_restored = nx.node_link_graph(graph_bytes)
                canonical_forms.append(G_restored)
            except:
                continue

        # 为节点添加标签属性
        for node in G.nodes():
            G.nodes[node]['label'] = 'C'

        # 检查是否与已有的图同构
        for existing_graph in canonical_forms:
            try:
                gm = nx.isomorphism.GraphMatcher(
                    G,
                    existing_graph,
                    node_match=nx.isomorphism.categorical_node_match('label', 'C'),
                    edge_match=custom_edge_match
                )
                if gm.is_isomorphic():
                    return (False, None)  # 同构，不添加
            except Exception:
                continue

        # 不同构，返回图的序列化形式
        graph_bytes = nx.node_link_data(G)
        return (True, graph_bytes)

    except Exception as e:
        return (False, None)

# ==================== 导入原有类 ====================
class AlkaneIsomerGenerator:
    """
    饱和烷烃同分异构体生成器
    使用无标号树算法生成所有可能的异构体
    支持并行计算加速大规模异构体生成
    """
    BOND_LENGTH_CC = 1.54
    BOND_LENGTH_CH = 1.09
    TETRAHEDRAL_ANGLE = math.acos(-1/3) # ~109.47 degrees
    ANGLE_120 = 2 * math.pi / 3 # 120 degrees in radians
    SQRT3 = math.sqrt(3)

    def __init__(self, use_parallel=True, num_workers=None):
        """
        初始化生成器
        Args:
            use_parallel: 是否使用并行计算
            num_workers: 并行进程数，None表示自动检测
        """
        self.cache = {}
        self.isomer_cache = {}
        self.use_parallel = use_parallel
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = max(1, num_workers)

    def parse_substrings(self, canon_str: str) -> List[str]:
        """解析规范字符串中的子树"""
        if canon_str == "C":
            return []
        inner = canon_str[2:-1]
        if not inner:
            return []
        subs = []
        depth = 0
        cur = ""
        for char in inner:
            if char == '(':
                depth += 1
                cur += char
            elif char == ')':
                depth -= 1
                cur += char
            elif char == ',' and depth == 0:
                subs.append(cur)
                cur = ""
            else:
                cur += char
        if cur:
            subs.append(cur)
        return subs

    def generate_rooted_with_info(self, n: int, max_branches: int = 4):
        """
        生成 n 个碳原子的有根树异构体（无标号树算法）
        max_branches: 根节点允许的最大子分支数量
         - 对于整个分子，max_branches = 4
         - 对于子树（连接到父节点），max_branches = 3 (因为已占用 1 个键连父)
        """
        cache_key = (n, max_branches)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if n == 1:
            res = [("C", [])]
            self.cache[cache_key] = res
            return res

        results = []
        target = n - 1
        partitions = []

        def find_parts(rem, min_val, path):
            if len(path) >= max_branches:
                if rem == 0:
                    partitions.append(tuple(path))
                return
            if rem == 0:
                if path:
                    partitions.append(tuple(path))
                return
            for i in range(min_val, rem + 1):
                if len(path) + 1 == max_branches:
                    if rem - i == 0:
                        find_parts(0, i, path + [i])
                        continue
                find_parts(rem - i, i, path + [i])

        find_parts(target, 1, [])

        seen_local = set()
        for p in partitions:
            # 对于每个部分，递归生成子树
            # 注意：子树的根节点已经连接到了当前根节点，所以子树的 max_branches 必须是 3
            options_per_size = [self.generate_rooted_with_info(s, max_branches=3) for s in p]
            for combo in itertools.product(*options_per_size):
                items = []
                for i, item in enumerate(combo):
                    items.append((item[0], p[i]))
                # 排序以保证规范形式
                items.sort(key=lambda x: x[0])
                sorted_strs = [x[0] for x in items]
                sorted_sizes = [x[1] for x in items]
                
                canon = "C(" + ",".join(sorted_strs) + ")"
                if canon not in seen_local:
                    seen_local.add(canon)
                    results.append((canon, sorted_sizes))

        self.cache[cache_key] = results
        return results

    def generate_isomers(self, n: int):
        """
        生成 n 个碳原子的所有无标号异构体
        支持并行计算以加速大规模异构体生成
        """
        if n in self.isomer_cache:
            return self.isomer_cache[n]

        if n == 0:
            return []
        if n == 1:
            return ["C"]

        # 生成整个分子时，根节点允许 4 个分支
        all_rooted = self.generate_rooted_with_info(n, max_branches=4)
        limit = n // 2
        
        raw_fingerprints = set()

        # 根据数据量决定是否使用并行计算
        use_parallel = self.use_parallel and len(all_rooted) > 1000
        if use_parallel:
            # 并行处理大规模数据
            raw_fingerprints = self._process_rooted_parallel(all_rooted, limit)
        else:
            # 串行处理小规模数据
            for canon, sub_sizes in all_rooted:
                processed = self._process_single_rooted(canon, sub_sizes, limit)
                if processed:
                    raw_fingerprints.add(processed)

        # 基于邻接表的二次去重
        unique_adj_sets = set()
        final_fingerprints = []

        # 并行或串行处理邻接表去重
        if use_parallel and len(raw_fingerprints) > 1000:
            final_fingerprints = self._deduplicate_parallel(raw_fingerprints, unique_adj_sets)
        else:
            for raw_canon in raw_fingerprints:
                result = self._deduplicate_single(raw_canon, unique_adj_sets)
                if result:
                    final_fingerprints.append(result)

        result = sorted(final_fingerprints)
        self.isomer_cache[n] = result
        return result

    def _process_single_rooted(self, canon: str, sub_sizes: tuple, limit: int) -> Optional[str]:
        """处理单个根树异构体"""
        if not sub_sizes:
            max_s = 0
        else:
            max_s = max(sub_sizes)
        
        if max_s <= limit:
            if max_s < limit:
                return canon
            else:
                # 处理 max_size 等于 limit 的情况
                idx = sub_sizes.index(limit)
                sub_strs = self.parse_substrings(canon)
                big_sub_str = sub_strs[idx]
                rest_subs = sub_strs[:idx] + sub_strs[idx+1:]
                rest_subs.sort()
                rest_canon = "C(" + ",".join(rest_subs) + ")"

                big_children = self.parse_substrings(big_sub_str)
                new_children = big_children + [rest_canon]
                new_children.sort()
                flipped_canon = "C(" + ",".join(new_children) + ")"
                return min(canon, flipped_canon)
        return None

    def _process_rooted_parallel(self, all_rooted: list, limit: int) -> set:
        """并行处理根树异构体"""
        results = set()
        # 准备数据，避免闭包
        items_list = [(canon, sub_sizes, limit) for canon, sub_sizes in all_rooted]
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(process_rooted_worker_parallel, item): item
                for item in items_list
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.add(result)
        return results

    def _deduplicate_single(self, raw_canon: str, unique_adj_sets: set) -> Optional[str]:
        """单个异构体的去重处理"""
        try:
            adj = self.canon_to_adjacency(raw_canon)
            # 验证每个节点的度数是否 <= 4
            valid_structure = True
            for node, neighbors in adj.items():
                if len(neighbors) > 4:
                    valid_structure = False
                    break
            if not valid_structure:
                return None
            
            standardized_adj = tuple(
                sorted((node, tuple(sorted(neighbors))) for node, neighbors in adj.items())
            )
            
            if standardized_adj not in unique_adj_sets:
                unique_adj_sets.add(standardized_adj)
                return raw_canon
        except:
            pass
        return None

    def _deduplicate_parallel(self, raw_fingerprints: set, unique_adj_sets: set) -> List[str]:
        """并行去重处理"""
        results = []
        # 将原始指纹分批处理
        fingerprints_list = list(raw_fingerprints)
        batch_size = max(100, len(fingerprints_list) // self.num_workers)
        batches = [
            fingerprints_list[i:i + batch_size]
            for i in range(0, len(fingerprints_list), batch_size)
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(deduplicate_batch_worker_parallel, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_results = future.result()
                for result in batch_results:
                    if result:
                        standardized_adj = tuple(
                            sorted((node, tuple(sorted(neighbors))) for node, neighbors in result[1].items())
                        )
                        if standardized_adj not in unique_adj_sets:
                            unique_adj_sets.add(standardized_adj)
                            results.append(result[0])
        return sorted(results)

    def canon_to_adjacency(self, canon_str: str) -> Dict[int, List[int]]:
        """将规范字符串转换为邻接表"""
        if canon_str == "C":
            return {0: []}
        node_counter = [0]
        self._temp_adj = {}

        def parse_subtree(s: str, parent: int = None) -> int:
            node_id = node_counter[0]
            node_counter[0] += 1
            if parent is not None:
                if parent not in self._temp_adj:
                    self._temp_adj[parent] = []
                if node_id not in self._temp_adj:
                    self._temp_adj[node_id] = []
                self._temp_adj[parent].append(node_id)
                self._temp_adj[node_id].append(parent)

            if s != "C":
                inner = s[2:-1]
                if inner:
                    subs = self.parse_substrings(s)
                    for sub in subs:
                        parse_subtree(sub, node_id)
            return node_id

        parse_subtree(canon_str)
        return self._temp_adj

    def adjacency_to_coords(self, adjacency: Dict[int, List[int]], n: int) -> List[Tuple[float, float, float]]:
        """将邻接表转换为3D坐标（使用严格四面体几何）"""
        coords = [None] * n
        coords[0] = (0.0, 0.0, 0.0)
        visited = set([0])
        queue = [0]
        parent = {}
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        processing_order = []
        queue = [0]
        visited.clear()
        visited.add(0)
        while queue:
            current = queue.pop(0)
            processing_order.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        for node in processing_order:
            if node == 0:
                continue
            parent_node = parent[node]
            px, py, pz = coords[parent_node]
            siblings_and_child = [child for child in adjacency[parent_node] if coords[child] is not None]
            siblings = [s for s in siblings_and_child if s != node]

            if coords[node] is not None:
                continue

            if not siblings:
                direction = (1.0, 0.0, 0.0)
            elif len(siblings) == 1:
                s1_coords = np.array(coords[siblings[0]])
                parent_coords = np.array([px, py, pz])
                vec_to_s1 = s1_coords - parent_coords
                direction = self._calculate_tetrahedral_direction_from_one_bond(vec_to_s1)
            elif len(siblings) == 2:
                s1_coords = np.array(coords[siblings[0]])
                s2_coords = np.array(coords[siblings[1]])
                parent_coords = np.array([px, py, pz])
                vec_to_s1 = s1_coords - parent_coords
                vec_to_s2 = s2_coords - parent_coords
                direction = self._calculate_tetrahedral_direction_from_two_bonds(vec_to_s1, vec_to_s2)
            elif len(siblings) == 3:
                s1_coords = np.array(coords[siblings[0]])
                s2_coords = np.array(coords[siblings[1]])
                s3_coords = np.array(coords[siblings[2]])
                parent_coords = np.array([px, py, pz])
                vec_to_s1 = s1_coords - parent_coords
                vec_to_s2 = s2_coords - parent_coords
                vec_to_s3 = s3_coords - parent_coords
                direction = self._calculate_tetrahedral_direction_from_three_bonds(vec_to_s1, vec_to_s2, vec_to_s3)
            else:
                direction = (1.0, 0.0, 0.0)

            dir_vec = np.array(direction)
            dir_vec /= np.linalg.norm(dir_vec)
            nx = px + dir_vec[0] * self.BOND_LENGTH_CC
            ny = py + dir_vec[1] * self.BOND_LENGTH_CC
            nz = pz + dir_vec[2] * self.BOND_LENGTH_CC
            coords[node] = (nx, ny, nz)
        return coords

    def _calculate_tetrahedral_direction_from_one_bond(self, vec_to_existing_bond):
        """根据一个已存在的键向量，计算新的四面体方向"""
        v1 = vec_to_existing_bond / np.linalg.norm(vec_to_existing_bond)
        if abs(v1[0]) < 0.9:
            arb = np.array([1, 0, 0])
        else:
            arb = np.array([0, 1, 0])
        perp = np.cross(v1, arb)
        perp /= np.linalg.norm(perp)
        
        cos_theta = -1/3
        sin_theta = np.sqrt(8)/3
        
        v_new = cos_theta * v1 + sin_theta * np.cross(perp, v1) + (1 - cos_theta) * np.dot(perp, v1) * perp
        return tuple(v_new)

    def _calculate_tetrahedral_direction_from_two_bonds(self, vec_to_existing_bond1, vec_to_existing_bond2):
        """根据两个已存在的键向量，计算新的四面体方向"""
        v1 = vec_to_existing_bond1 / np.linalg.norm(vec_to_existing_bond1)
        v2 = vec_to_existing_bond2 / np.linalg.norm(vec_to_existing_bond2)
        
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            # 如果两个向量平行或接近平行
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp /= np.linalg.norm(perp)
        else:
            normal /= normal_norm
            perp = normal

        dot_v1_v2 = np.dot(v1, v2)
        if abs(1 + dot_v1_v2) < 1e-6:
             # 如果两个向量反向平行
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp /= np.linalg.norm(perp)
            return self._calculate_tetrahedral_direction_from_one_bond(perp)

        c = -1 / (3 * (1 + dot_v1_v2))
        v1_plus_v2_mag_sq = 2 * (1 + dot_v1_v2)
        normal_mag_sq = 1 - dot_v1_v2**2
        
        term1 = c**2 * v1_plus_v2_mag_sq
        if 1 - term1 < 0:
            v4_unnorm = c * (v1 + v2)
            return tuple(v4_unnorm / np.linalg.norm(v4_unnorm))

        d_sq = (1 - term1) / normal_mag_sq
        if d_sq < 0:
            v4_unnorm = c * (v1 + v2)
            return tuple(v4_unnorm / np.linalg.norm(v4_unnorm))

        d = np.sqrt(d_sq)
        v4_unnorm = c * (v1 + v2) + d * normal
        v4_unnorm_check_neg = c * (v1 + v2) - d * normal
        
        test_dot_pos = np.dot(v4_unnorm / np.linalg.norm(v4_unnorm), v1)
        test_dot_neg = np.dot(v4_unnorm_check_neg / np.linalg.norm(v4_unnorm_check_neg), v1)
        
        if abs(test_dot_pos - (-1/3)) < abs(test_dot_neg - (-1/3)):
            final_v4 = v4_unnorm
        else:
            final_v4 = v4_unnorm_check_neg
            
        return tuple(final_v4 / np.linalg.norm(final_v4))

    def _calculate_tetrahedral_direction_from_three_bonds(self, vec_to_existing_bond1, vec_to_existing_bond2, vec_to_existing_bond3):
        """根据三个已存在的键向量，计算新的四面体方向（第四个方向）"""
        v1 = vec_to_existing_bond1 / np.linalg.norm(vec_to_existing_bond1)
        v2 = vec_to_existing_bond2 / np.linalg.norm(vec_to_existing_bond2)
        v3 = vec_to_existing_bond3 / np.linalg.norm(vec_to_existing_bond3)
        
        v4 = -(v1 + v2 + v3)
        v4_norm = np.linalg.norm(v4)
        if v4_norm < 1e-6:
            return tuple(-v1) # Fallback if sum is zero vector
        v4_normalized = v4 / v4_norm
        return tuple(v4_normalized)

    def get_isomer_name(self, n: int, index: int) -> str:
        """获取异构体的名称"""
        base_names = {
            1: "甲烷", 2: "乙烷", 3: "丙烷", 4: "丁烷", 5: "戊烷",
            6: "己烷", 7: "庚烷", 8: "辛烷", 9: "壬烷", 10: "癸烷",
            11: "十一烷", 12: "十二烷", 13: "十三烷", 14: "十四烷", 15: "十五烷",
            16: "十六烷", 17: "十七烷", 18: "十八烷", 19: "十九烷", 20: "二十烷",
            21: "二十一烷", 22: "二十二烷", 23: "二十三烷", 24: "二十四烷", 25: "二十五烷",
            26: "二十六烷", 27: "二十七烷", 28: "二十八烷", 29: "二十九烷", 30: "三十烷"
        }
        if n == 4:
            names = ["正丁烷", "异丁烷"]
            return f"{names[index] if index < len(names) else f'异构体{index+1}'} (C{n}H{2*n+2})"
        elif n == 5:
            names = ["正戊烷", "异戊烷", "新戊烷"]
            return f"{names[index] if index < len(names) else f'异构体{index+1}'} (C{n}H{2*n+2})"
        else:
            base_name = base_names.get(n, f"C{n}烷")
            return f"{base_name}-异构体{index+1} (C{n}H{2*n+2})"


# 新增：单烯烃生成器类
class AlkeneIsomerGenerator:
    """
    单烯烃同分异构体生成器
    通过在N-1个碳的烷基骨架中插入双键来生成CnH2n的所有异构体
    使用networkx进行图规范化以实现鲁棒去重
    """
    def __init__(self, alkane_generator=None, num_cores=1):
        """
        初始化生成器
        Args:
            alkane_generator: 可选的烷烃生成器实例。如果为None，则创建默认实例
            num_cores: 使用的CPU核心数，用于并行处理
        """
        self.alkane_gen = alkane_generator if alkane_generator else AlkaneIsomerGenerator()
        self.num_cores = num_cores

    def generate_isomers(self, n_carbons: int):
        """
        生成 n_carbons 个碳原子的单烯烃同分异构体（基于烷烃骨架放置双键的新算法）
        使用 WL 哈希分组加速去重，避免逐个图同构比较
        
        Args:
            n_carbons: 碳原子数 (n >= 2)
        Returns:
            List of networkx.Graph objects with bond_type attributes (single/double)
        """
        if n_carbons < 2:
            return []

        print(f"正在生成 C{n_carbons}H{2*n_carbons} 单烯烃...")

        # 步骤1: 获取所有 n 个碳的烷烃骨架
        alkane_graphs = self._get_alkane_graphs(n_carbons)
        print(f"  - 烷烃骨架数: {len(alkane_graphs)}")

        # 步骤2: 对每条合法边放置双键，使用 WL 哈希分组加速去重
        hash_groups = {}  # hash -> 图列表
        total_candidates = 0

        # 进度条
        if HAS_TQDM:
            iterator = tqdm(alkane_graphs, desc="生成候选", unit="骨架")
        else:
            iterator = alkane_graphs

        for G in iterator:
            edges = list(G.edges())
            for u, v in edges:
                # 双键两端碳的原始度数必须 ≤3（否则变成双键后度数超4）
                if G.degree[u] > 3 or G.degree[v] > 3:
                    continue
                total_candidates += 1

                # 构建候选图，复制所有边并标记 bond_type
                H = nx.create_empty_copy(G)
                for a, b in edges:
                    if (a, b) == (u, v) or (b, a) == (u, v):
                        H.add_edge(a, b, bond_type='double')
                    else:
                        H.add_edge(a, b, bond_type='single')
                for node in H.nodes():
                    H.nodes[node]['label'] = 'C'

                # 计算 WL 哈希（对节点标签和边属性敏感）
                h = nx.weisfeiler_lehman_graph_hash(
                    H, edge_attr='bond_type', node_attr='label'
                )

                # 快速去重：只在同哈希组内进行精确同构比较
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
        return unique

    def _get_alkane_graphs(self, n_carbons: int):
        """
        获取 n_carbons 个碳的烷烃骨架图列表
        兼容两种 AlkaneIsomerGenerator：返回图对象列表或规范字符串列表
        """
        alkane_canons = self.alkane_gen.generate_isomers(n_carbons)

        # 如果已经是图对象列表，直接返回
        if alkane_canons and hasattr(alkane_canons[0], 'edges'):
            return alkane_canons

        # 如果是规范字符串列表，转换为图对象
        base_trees_adj = [self.alkane_gen.canon_to_adjacency(canon) for canon in alkane_canons]
        graphs = []
        for adj in base_trees_adj:
            G = nx.Graph()
            for node, neighbors in adj.items():
                G.add_node(node, label='C')
                for neighbor in neighbors:
                    if neighbor > node:
                        G.add_edge(node, neighbor)
            graphs.append(G)
        return graphs


# 原有的可视化器类保持不变，但可以扩展以支持烯烃
class AlkaneIsomerVisualizer:
    """烷烃同分异构体可视化器"""
    def __init__(self, generator=None):
        """
        初始化可视化器
        Args:
            generator: 可选的异构体生成器实例。如果为None，则创建默认实例
        """
        self.generator = generator if generator else AlkaneIsomerGenerator()
        self.atom_colors = {'C': '#333333', 'H': '#FFFFFF'}
        self.atom_radii = {'C': 0.6, 'H': 0.3}
        self.bond_color = '#808080'
        self.bond_width = 3

    def _generate_methane_hydrogens(self, c_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """生成甲烷分子的氢原子位置（四面体构型）"""
        directions = [
            (self.generator.SQRT3/3, self.generator.SQRT3/3, self.generator.SQRT3/3),
            (self.generator.SQRT3/3, -self.generator.SQRT3/3, -self.generator.SQRT3/3),
            (-self.generator.SQRT3/3, self.generator.SQRT3/3, -self.generator.SQRT3/3),
            (-self.generator.SQRT3/3, -self.generator.SQRT3/3, self.generator.SQRT3/3)
        ]
        h_positions = []
        for direction in directions:
            h_x = c_pos[0] + self.generator.BOND_LENGTH_CH * direction[0]
            h_y = c_pos[1] + self.generator.BOND_LENGTH_CH * direction[1]
            h_z = c_pos[2] + self.generator.BOND_LENGTH_CH * direction[2]
            h_positions.append((h_x, h_y, h_z))
        return h_positions

    def _generate_terminal_hydrogens(self, c_pos: Tuple[float, float, float], adj_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """生成末端碳原子的氢原子位置"""
        vec_to_adj = [adj_pos[0] - c_pos[0], adj_pos[1] - c_pos[1], adj_pos[2] - c_pos[2]]
        len_vec = math.sqrt(sum(v**2 for v in vec_to_adj))
        if len_vec == 0:
            return []
        u_to_adj = [v/len_vec for v in vec_to_adj]

        if abs(u_to_adj[0]) < 0.9:
            ref = (1, 0, 0)
        else:
            ref = (0, 1, 0)

        v1 = [u_to_adj[1]*ref[2] - u_to_adj[2]*ref[1], u_to_adj[2]*ref[0] - u_to_adj[0]*ref[2], u_to_adj[0]*ref[1] - u_to_adj[1]*ref[0]]
        len_v1 = math.sqrt(sum(v**2 for v in v1))
        if len_v1 == 0:
            return []
        v1 = [v/len_v1 for v in v1]

        v2 = [v1[1]*u_to_adj[2] - v1[2]*u_to_adj[1], v1[2]*u_to_adj[0] - v1[0]*u_to_adj[2], v1[0]*u_to_adj[1] - v1[1]*u_to_adj[0]]
        len_v2 = math.sqrt(sum(v**2 for v in v2))
        if len_v2 == 0:
            return []
        v2 = [v/len_v2 for v in v2]

        h_positions = []
        for j in range(3):
            angle = j * self.generator.ANGLE_120
            h_x = c_pos[0] + self.generator.BOND_LENGTH_CH * (
                u_to_adj[0]*math.cos(self.generator.TETRAHEDRAL_ANGLE) +
                v1[0]*math.sin(self.generator.TETRAHEDRAL_ANGLE)*math.cos(angle) +
                v2[0]*math.sin(self.generator.TETRAHEDRAL_ANGLE)*math.sin(angle)
            )
            h_y = c_pos[1] + self.generator.BOND_LENGTH_CH * (
                u_to_adj[1]*math.cos(self.generator.TETRAHEDRAL_ANGLE) +
                v1[1]*math.sin(self.generator.TETRAHEDRAL_ANGLE)*math.cos(angle) +
                v2[1]*math.sin(self.generator.TETRAHEDRAL_ANGLE)*math.sin(angle)
            )
            h_z = c_pos[2] + self.generator.BOND_LENGTH_CH * (
                u_to_adj[2]*math.cos(self.generator.TETRAHEDRAL_ANGLE) +
                v1[2]*math.sin(self.generator.TETRAHEDRAL_ANGLE)*math.cos(angle) +
                v2[2]*math.sin(self.generator.TETRAHEDRAL_ANGLE)*math.sin(angle)
            )
            h_positions.append((h_x, h_y, h_z))
        return h_positions

    def _generate_middle_hydrogens(self, c_pos: Tuple[float, float, float], prev_pos: Tuple[float, float, float], next_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """生成中间碳原子的氢原子位置"""
        vec_to_prev = [prev_pos[0] - c_pos[0], prev_pos[1] - c_pos[1], prev_pos[2] - c_pos[2]]
        vec_to_next = [next_pos[0] - c_pos[0], next_pos[1] - c_pos[1], next_pos[2] - c_pos[2]]

        len_prev = math.sqrt(sum(v**2 for v in vec_to_prev))
        len_next = math.sqrt(sum(v**2 for v in vec_to_next))
        if len_prev == 0 or len_next == 0:
            return []

        u_prev = [v/len_prev for v in vec_to_prev]
        u_next = [v/len_next for v in vec_to_next]

        bisector = [u_prev[0] + u_next[0], u_prev[1] + u_next[1], u_prev[2] + u_next[2]]
        len_bis = math.sqrt(sum(v**2 for v in bisector))
        if len_bis == 0:
            if abs(u_prev[0]) < 0.9:
                perp = np.cross(u_prev, [1, 0, 0])
            else:
                perp = np.cross(u_prev, [0, 1, 0])
            perp /= np.linalg.norm(perp)
            bisector = perp
            len_bis = 1.0
        else:
            bisector = [v/len_bis for v in bisector]

        cross = [u_prev[1]*u_next[2] - u_prev[2]*u_next[1], u_prev[2]*u_next[0] - u_prev[0]*u_next[2], u_prev[0]*u_next[1] - u_prev[1]*u_next[0]]
        len_cross = math.sqrt(sum(v**2 for v in cross))
        if len_cross == 0:
            if abs(bisector[0]) < 0.9:
                perp_fallback = np.cross(bisector, [1, 0, 0])
            else:
                perp_fallback = np.cross(bisector, [0, 1, 0])
            perp_fallback /= np.linalg.norm(perp_fallback)
            cross = perp_fallback
            len_cross = 1.0
        else:
            cross = [v/len_cross for v in cross]

        h_positions = []
        for sign in [+1, -1]:
            h_x = c_pos[0] + self.generator.BOND_LENGTH_CH * (
                bisector[0]*math.cos(self.generator.TETRAHEDRAL_ANGLE) +
                sign*cross[0]*math.sin(self.generator.TETRAHEDRAL_ANGLE)
            )
            h_y = c_pos[1] + self.generator.BOND_LENGTH_CH * (
                bisector[1]*math.cos(self.generator.TETRAHEDRAL_ANGLE) +
                sign*cross[1]*math.sin(self.generator.TETRAHEDRAL_ANGLE)
            )
            h_z = c_pos[2] + self.generator.BOND_LENGTH_CH * (
                bisector[2]*math.cos(self.generator.TETRAHEDRAL_ANGLE) +
                sign*cross[2]*math.sin(self.generator.TETRAHEDRAL_ANGLE)
            )
            h_positions.append((h_x, h_y, h_z))
        return h_positions

    def _generate_tertiary_hydrogens(self, c_pos: Tuple[float, float, float], adjacent_carbon_coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """生成叔碳原子的氢原子位置"""
        total_vec = np.array([0.0, 0.0, 0.0])
        for adj_coord in adjacent_carbon_coords:
            vec_to_adj = np.array(adj_coord) - np.array(c_pos)
            total_vec += vec_to_adj / np.linalg.norm(vec_to_adj)
        
        avg_dir = total_vec / len(adjacent_carbon_coords)
        avg_dir /= np.linalg.norm(avg_dir)
        
        h_dir = -avg_dir
        h_pos = np.array(c_pos) + self.generator.BOND_LENGTH_CH * h_dir
        return [tuple(h_pos)]

    def _generate_hydrogen_coords(self, c_coords: List[Tuple[float, float, float]], adjacency: Dict[int, List[int]]) -> List[Tuple[float, float, float]]:
        """生成所有氢原子的坐标"""
        h_coords = []
        n_carbons = len(c_coords)

        for i in range(n_carbons):
            c_pos = c_coords[i]
            adjacent_carbon_indices = adjacency.get(i, [])
            adjacent_carbon_coords = [c_coords[idx] for idx in adjacent_carbon_indices if idx < n_carbons]
            num_attached_carbons = len(adjacent_carbon_coords)

            if num_attached_carbons == 0: # Methane
                h_positions = self._generate_methane_hydrogens(c_pos)
            elif num_attached_carbons == 1: # Terminal carbon
                adj_pos = adjacent_carbon_coords[0]
                h_positions = self._generate_terminal_hydrogens(c_pos, adj_pos)
            elif num_attached_carbons == 2: # Middle carbon
                adj_indices = adjacency.get(i, [])
                adj_indices_sorted = sorted(adj_indices)
                adj_coords_sorted_by_idx = [c_coords[idx] for idx in adj_indices_sorted if idx < n_carbons]
                if len(adj_coords_sorted_by_idx) >= 2:
                    prev_pos = adj_coords_sorted_by_idx[0]
                    next_pos = adj_coords_sorted_by_idx[1]
                else:
                    prev_pos = adj_coords_sorted_by_idx[0] if len(adj_coords_sorted_by_idx) > 0 else (0,0,0)
                    next_pos = (0,0,0)
                h_positions = self._generate_middle_hydrogens(c_pos, prev_pos, next_pos)
            elif num_attached_carbons == 3: # Tertiary carbon
                h_positions = self._generate_tertiary_hydrogens(c_pos, adjacent_carbon_coords)
            elif num_attached_carbons >= 4: # Quaternary carbon (or error)
                h_positions = []
            else:
                h_positions = []
            
            h_coords.extend(h_positions)

        return h_coords

    def visualize_isomer(self, n_carbons: int, isomer_index: int, show=True, save_path=None):
        """可视化单个异构体"""
        isomers = self.generator.generate_isomers(n_carbons)
        if isomer_index >= len(isomers):
            raise ValueError(f"异构体索引超出范围：{isomer_index} >= {len(isomers)}")

        canon = isomers[isomer_index]
        adjacency = self.generator.canon_to_adjacency(canon)
        c_coords = self.generator.adjacency_to_coords(adjacency, n_carbons)
        all_coords = c_coords + self._generate_hydrogen_coords(c_coords, adjacency)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        self._draw_molecule(ax, all_coords, n_carbons, adjacency)

        name = self.generator.get_isomer_name(n_carbons, isomer_index)
        ax.set_title(name, fontsize=14, pad=20)
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')
        ax.set_zlabel('Z (A)')
        ax.grid(True, alpha=0.3)

        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        z_coords = [coord[2] for coord in all_coords]
        self._set_equal_aspect_ratio(ax, x_coords, y_coords, z_coords)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至：{save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_all_isomers(self, n_carbons: int, show=True, save_dir=None):
        """可视化所有异构体"""
        isomers = self.generator.generate_isomers(n_carbons)
        if not isomers:
            print(f"C{n_carbons}没有异构体")
            return

        n_isomers = len(isomers)
        if n_isomers <= 2:
            cols, rows = 2, 1
        elif n_isomers <= 4:
            cols, rows = 2, 2
        elif n_isomers <= 9:
            cols, rows = 3, 3
        else:
            cols = 5
            rows = (n_isomers + cols - 1) // cols

        fig = plt.figure(figsize=(6*cols, 5*rows))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)

        for i, canon in enumerate(isomers):
            if i >= rows * cols:
                break
            row, col = i // cols, i % cols

            adjacency = self.generator.canon_to_adjacency(canon)
            c_coords = self.generator.adjacency_to_coords(adjacency, n_carbons)
            all_coords = c_coords + self._generate_hydrogen_coords(c_coords, adjacency)

            ax = fig.add_subplot(gs[row, col], projection='3d')
            self._draw_molecule(ax, all_coords, n_carbons, adjacency, show_labels=False)

            name = self.generator.get_isomer_name(n_carbons, i)
            ax.set_title(name, fontsize=10, pad=10)
            ax.set_xlabel('X (A)', fontsize=8)
            ax.set_ylabel('Y (A)', fontsize=8)
            ax.set_zlabel('Z (A)', fontsize=8)
            ax.tick_params(labelsize=7)

            x_coords = [coord[0] for coord in all_coords]
            y_coords = [coord[1] for coord in all_coords]
            z_coords = [coord[2] for coord in all_coords]
            self._set_equal_aspect_ratio(ax, x_coords, y_coords, z_coords)

        fig.suptitle(f'C{n_carbons}H{2*n_carbons+2} 的所有同分异构体', fontsize=16, fontweight='bold')

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'C{n_carbons}_all_isomers.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"图片已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()


    def _draw_molecule(self, ax, coords, n_carbons, adjacency, show_labels=True):
        """绘制分子结构"""
        c_coords = coords[:n_carbons]
        h_coords = coords[n_carbons:]

        # 绘制C-C键
        for parent in range(n_carbons):
            if parent in adjacency:
                for child in adjacency[parent]:
                    if child > parent: # 避免重复绘制
                        c1 = np.array(c_coords[parent])
                        c2 = np.array(c_coords[child])
                        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 
                                c=self.bond_color, linewidth=self.bond_width, alpha=0.8)

        # 绘制C-H键
        h_counter = 0
        for i in range(n_carbons):
            num_h_attached = 4 - len(adjacency.get(i, []))
            c_pos = np.array(c_coords[i])
            for j in range(num_h_attached):
                if h_counter < len(h_coords):
                    h_pos = np.array(h_coords[h_counter])
                    ax.plot([c_pos[0], h_pos[0]], [c_pos[1], h_pos[1]], [c_pos[2], h_pos[2]], 
                            c=self.bond_color, linewidth=self.bond_width*0.7, alpha=0.6)
                    h_counter += 1

        # 绘制碳原子
        for i, coord in enumerate(c_coords):
            ax.scatter(coord[0], coord[1], coord[2], 
                       c=self.atom_colors['C'], s=self.atom_radii['C'] * 100, 
                       alpha=0.9, edgecolors='black', linewidths=1)
            if show_labels:
                ax.text(coord[0], coord[1], coord[2], f'C{i+1}', 
                        fontsize=8, ha='center', va='center')

        # 绘制氢原子
        for coord in h_coords:
            ax.scatter(coord[0], coord[1], coord[2], 
                       c=self.atom_colors['H'], s=self.atom_radii['H'] * 100, 
                       alpha=0.9, edgecolors='gray', linewidths=1)


    def _set_equal_aspect_ratio(self, ax, x_coords, y_coords, z_coords):
        """设置3D图形等比例"""
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        z_range = max(z_coords) - min(z_coords)

        max_range = max(x_range, y_range, z_range)

        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2
        z_center = (max(z_coords) + min(z_coords)) / 2

        padding = max_range * 0.1
        limit = max_range / 2 + padding

        ax.set_xlim(x_center - limit, x_center + limit)
        ax.set_ylim(y_center - limit, y_center + limit)
        ax.set_zlim(z_center - limit, z_center + limit)


def main():
    """
    主函数：交互式命令行入口
    """
    # 全局中断标志
    global _interrupted
    _interrupted = False

    # 设置信号处理器（支持Ctrl+C中断）
    def _signal_handler(signum, frame):
        global _interrupted
        _interrupted = True
        print("\n\n[中断] 正在停止计算，请稍候...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _signal_handler)

    # ---------- 显示欢迎信息 ----------
    print("=" * 70)
    print("烷烃/单烯烃同分异构体生成工具")
    print("支持烷烃 CnH(2n+2) 和单烯烃 CnH2n 的生成")
    print("=" * 70)
    print()

    # ---------- 选择分子类型 ----------
    mol_type = input("请选择分子类型 (1: 烷烃, 2: 单烯烃): ").strip()
    if mol_type not in ['1', '2']:
        print("无效选择")
        return

    # ---------- 获取碳原子数 ----------
    try:
        n = int(input("请输入碳原子数量 (>=1 for 烷烃, >=2 for 烯烃): "))
        if (mol_type == '1' and n < 1) or (mol_type == '2' and n < 2):
            print("错误：碳原子数量不足")
            return
    except ValueError:
        print("错误：请输入有效的数字")
        return

    # ---------- 选择核心数 ----------
    if mol_type == '2':
        max_cores = min(mp.cpu_count(), 11)
        print(f"\n系统可用核心数: {mp.cpu_count()}，最大可用核心数: {max_cores}")
        try:
            cores_input = input(f"请选择使用多少个核心 (1-{max_cores}, 默认使用全部可用核心): ").strip()
            if cores_input == "":
                num_cores = max_cores
            else:
                num_cores = int(cores_input)
                if num_cores < 1:
                    num_cores = 1
                elif num_cores > max_cores:
                    print(f"警告：输入值超过最大核心数，将使用 {max_cores} 个核心")
                    num_cores = max_cores
            print(f"将使用 {num_cores} 个核心进行计算")
        except ValueError:
            print("错误：请输入有效的数字，将使用默认核心数")
            num_cores = max_cores
    else:
        num_cores = 1

    # ---------- 创建生成器 ----------
    if mol_type == '1':
        generator = AlkaneIsomerGenerator()
        formula = f"C{n}H{2*n+2}"
        visualizer = AlkaneIsomerVisualizer(generator)
    else:
        generator = AlkeneIsomerGenerator(AlkaneIsomerGenerator(), num_cores=num_cores)
        formula = f"C{n}H{2*n}"
        visualizer = AlkaneIsomerVisualizer(generator)

    # ---------- 生成异构体 ----------
    print(f"\n开始生成 {formula} 的同分异构体...")
    print()

    try:
        if mol_type == '1':
            isomers = generator.generate_isomers(n)
        else:
            isomers = generator.generate_isomers(n)
    except (KeyboardInterrupt, Exception):
        if isinstance(sys.exc_info()[0], KeyboardInterrupt) or _interrupted:
            print("\n\n[中断] 计算已停止")
        else:
            print(f"\n\n[错误] 计算过程中出现错误: {sys.exc_info()[1]}")
        print("感谢使用！")
        return

    # ---------- 显示结果 ----------
    print(f"\n{formula} 共有 {len(isomers)} 个同分异构体")

    # 打印异构体列表（最多显示前20个）
    if isomers:
        print("\n异构体列表:")
        for i in range(min(len(isomers), 20)):
            if hasattr(isomers[i], 'edges'):
                double_edges = [e for e in isomers[i].edges(data=True) if e[2].get('bond_type') == 'double']
                if double_edges:
                    u, v = double_edges[0][0], double_edges[0][1]
                    print(f"  {i+1}. 双键位置: C{u+1}=C{v+1}")
                else:
                    print(f"  {i+1}. {isomers[i]}")
            else:
                print(f"  {i+1}. {isomers[i]}")
        if len(isomers) > 20:
            print(f"  ... 还有 {len(isomers) - 20} 个")

    # ---------- 提供查看选项 ----------
    if isomers:
        print("\n选项:")
        print("1. 可视化单个异构体")
        print("2. 保存异构体列表到文件")
        print("3. 退出")

        choice = input("\n请选择 (1/2/3): ").strip()

        if choice == "1":
            # 可视化单个
            try:
                idx_input = input(f"请输入异构体编号 (1-{len(isomers)}): ")
                if idx_input.strip():
                    idx = int(idx_input) - 1
                    if 0 <= idx < len(isomers):
                        print(f"\n正在显示第 {idx+1} 个异构体...")
                        if mol_type == '1':
                            visualizer.visualize_isomer(n, idx, show=True)
                        else:
                            visualizer.visualize_isomer(n, idx, show=True)
                    else:
                        print("错误：异构体编号超出范围")
                else:
                    print("已取消可视化")
            except ValueError:
                print("错误：请输入有效的数字")

        elif choice == "2":
            # 保存到文件
            filename = input("请输入输出文件名 (默认 alkane_isomers.txt): ").strip()
            if not filename:
                filename = "alkane_isomers.txt"

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"{formula} 同分异构体列表\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总数: {len(isomers)}\n")
                    f.write("=" * 70 + "\n\n")
                    for i, iso in enumerate(isomers, 1):
                        if hasattr(iso, 'edges'):
                            double_edges = [e for e in iso.edges(data=True) if e[2].get('bond_type') == 'double']
                            if double_edges:
                                u, v = double_edges[0][0], double_edges[0][1]
                                f.write(f"{i}. 双键位置: C{u+1}=C{v+1}\n")
                            else:
                                f.write(f"{i}. {iso}\n")
                        else:
                            f.write(f"{i}. {iso}\n")
                print(f"已保存到: {filename}")
            except Exception as e:
                print(f"保存失败: {e}")

        else:
            print("已退出")

    else:
        print("没有找到异构体")
    
    print(f"\n{formula} 共有 {len(isomers)} 个同分异构体")
    
    if isomers:
        cont = input("是否打印所有结构? (y/N): ").strip().lower()
        if cont == 'y':
            for i, iso in enumerate(isomers):
                if hasattr(iso, 'edges'): # If it's an nx graph object
                    double_edges = [e for e in iso.edges(data=True) if e[2].get('bond_type') == 'double']
                    if double_edges:
                        u, v = double_edges[0][0], double_edges[0][1]
                        min_db, max_db = min(u, v), max(u, v)
                        print(f"{i+1}. Double bond between C{u+1} and C{v+1}")

                        degrees = {}
                        for node in iso.nodes():
                            degrees[node] = 0
                        for u_edge, v_edge, d in iso.edges(data=True):
                            bond_count = 2 if d.get('bond_type') == 'double' else 1
                            degrees[u_edge] += bond_count
                            degrees[v_edge] += bond_count

                        structure_info = []
                        for node in sorted(iso.nodes()):
                            neighbors = list(iso.neighbors(node))
                            structure_info.append(f"C{node+1}({degrees[node]})")
                        print(f"   Structure: {'-'.join(structure_info)}")
                    else:
                        print(f"{i+1}. Error: No double bond found")
                else:
                    print(f"{i+1}. {iso}")

    # ========== 导出为 .txt 文件 ==========
    export_choice = input("\n是否将结果导出为 .txt 文件? (Y/n): ").strip().lower()
    if export_choice != 'n':
        # 默认保存到桌面
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.exists(desktop_path):
            desktop_path = os.path.expanduser("~")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{formula}_isomers_{timestamp}.txt"
        filepath = os.path.join(desktop_path, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"{formula} 同分异构体生成结果\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"异构体总数: {len(isomers)}\n")
                f.write("=" * 70 + "\n\n")

                if not isomers:
                    f.write("未找到同分异构体。\n")
                else:
                    for i, iso in enumerate(isomers):
                        if hasattr(iso, 'edges'):
                            double_edges = [e for e in iso.edges(data=True) if e[2].get('bond_type') == 'double']
                            if double_edges:
                                u, v = double_edges[0][0], double_edges[0][1]

                                degrees = {}
                                for node in iso.nodes():
                                    degrees[node] = 0
                                for u_edge, v_edge, d in iso.edges(data=True):
                                    bond_count = 2 if d.get('bond_type') == 'double' else 1
                                    degrees[u_edge] += bond_count
                                    degrees[v_edge] += bond_count

                                structure_info = []
                                for node in sorted(iso.nodes()):
                                    structure_info.append(f"C{node+1}({degrees[node]})")

                                # 详细边信息
                                edge_info = []
                                for eu, ev, ed in iso.edges(data=True):
                                    bt = ed.get('bond_type', 'single')
                                    edge_info.append(f"C{eu+1}-{bt}-C{ev+1}")
                                edge_info.sort()

                                f.write(f"{i+1}. 双键位置: C{u+1}=C{v+1}\n")
                                f.write(f"   拓扑结构: {'-'.join(structure_info)}\n")
                                f.write(f"   键连接: {', '.join(edge_info)}\n")
                            else:
                                f.write(f"{i+1}. (无双键信息)\n")
                        else:
                            f.write(f"{i+1}. {iso}\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("End of report\n")

            print(f"结果已导出至: {filepath}")
        except Exception as e:
            print(f"导出失败: {e}")







# ==================== 并行计算工作函数 ====================
# 注意：这些函数必须在模块级别定义（不在 if __name__ == "__main__" 内部）
# 以支持 Windows 平台的 multiprocessing

def parse_substrings_parallel(canon_str: str):
    """解析规范字符串中的子树（并行版本）"""
    if canon_str == "C":
        return []
    inner = canon_str[2:-1]
    if not inner:
        return []
    subs = []
    depth = 0
    cur = ""
    for char in inner:
        if char == '(':
            depth += 1
            cur += char
        elif char == ')':
            depth -= 1
            cur += char
        elif char == ',' and depth == 0:
            subs.append(cur)
            cur = ""
        else:
            cur += char
    if cur:
        subs.append(cur)
    return subs

def process_rooted_worker_parallel(item: tuple) -> str | None:
    """
    并行处理单个根树异构体的工作函数
    item: (canon, sub_sizes, limit)
    """
    canon, sub_sizes, limit = item
    if not sub_sizes:
        max_s = 0
    else:
        max_s = max(sub_sizes)
    
    if max_s <= limit:
        if max_s < limit:
            return canon
        else:
            # 处理 max_size 等于 limit 的情况
            idx = sub_sizes.index(limit)
            sub_strs = parse_substrings_parallel(canon)
            big_sub_str = sub_strs[idx]
            rest_subs = sub_strs[:idx] + sub_strs[idx+1:]
            rest_subs.sort()
            rest_canon = "C(" + ",".join(rest_subs) + ")"

            # 解析大子树的子树
            if big_sub_str == "C":
                big_children = []
            else:
                inner_big = big_sub_str[2:-1]
                if not inner_big:
                    big_children = []
                else:
                    big_subs = []
                    depth_big = 0
                    cur_big = ""
                    for char in inner_big:
                        if char == '(':
                            depth_big += 1
                            cur_big += char
                        elif char == ')':
                            depth_big -= 1
                            cur_big += char
                        elif char == ',' and depth_big == 0:
                            big_subs.append(cur_big)
                            cur_big = ""
                        else:
                            cur_big += char
                    if cur_big:
                        big_subs.append(cur_big)
                    big_children = big_subs

            new_children = big_children + [rest_canon]
            new_children.sort()
            flipped_canon = "C(" + ",".join(new_children) + ")"
            return min(canon, flipped_canon)
    return None

def _canon_to_adjacency_worker(canon_str: str) -> Tuple[str, Dict[int, List[int]]]:
    """
    并行处理规范字符串转邻接表的工作函数
    """
    if canon_str == "C":
        return canon_str, {0: []}
    
    node_counter = [0]
    temp_adj = {}

    def parse_subtree(s: str, parent: int = None) -> int:
        node_id = node_counter[0]
        node_counter[0] += 1
        if parent is not None:
            if parent not in temp_adj:
                temp_adj[parent] = []
            if node_id not in temp_adj:
                temp_adj[node_id] = []
            temp_adj[parent].append(node_id)
            temp_adj[node_id].append(parent)

        if s != "C":
            inner = s[2:-1]
            if inner:
                # 解析子树
                subs = []
                depth = 0
                cur = ""
                for char in inner:
                    if char == '(':
                        depth += 1
                        cur += char
                    elif char == ')':
                        depth -= 1
                        cur += char
                    elif char == ',' and depth == 0:
                        subs.append(cur)
                        cur = ""
                    else:
                        cur += char
                if cur:
                    subs.append(cur)
                for sub in subs:
                    parse_subtree(sub, node_id)
        return node_id

    parse_subtree(canon_str)
    return canon_str, temp_adj

def deduplicate_batch_worker_parallel(batch: list) -> List[Tuple[str, Dict[int, List[int]]]]:
    """
    并行去重批次处理的工作函数
    """
    results = []
    for canon_str in batch:
        try:
            canon_str, adj = _canon_to_adjacency_worker(canon_str)
            # 验证每个节点的度数是否 <= 4
            valid_structure = True
            for node, neighbors in adj.items():
                if len(neighbors) > 4:
                    valid_structure = False
                    break
            if not valid_structure:
                continue
            results.append((canon_str, adj))
        except:
            continue
    return results

if __name__ == "__main__":
    main()
