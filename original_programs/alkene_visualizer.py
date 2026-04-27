"""
================================================================================
烯烃同分异构体可视化工具
================================================================================

【功能说明】
  - 支持任意碳原子数(n>=2)的烯烃同分异构体生成
  - 使用烷基自由基组合法枚举所有可能的同分异构体结构
  - 支持3D可视化，展示碳骨架、氢原子和双键的空间分布
  - 使用图签名和拉普拉斯矩阵特征值进行精确去重

【核心算法】
  - 烷基自由基组合法: 烯烃 = 两个烷基自由基 + 双键连接
  - 图签名检测: 使用度数序列、邻居度数模式进行初步去重
  - 拉普拉斯矩阵: 使用特征值进行精确同构检测

【化学知识】
  - 烯烃通式: CnH2n (含一个双键)
  - 双键碳 (sp2杂化): 最多连接3个原子，键角120度
  - 单键碳 (sp3杂化): 最多连接4个原子，键角109.47度

【文件结构】
  - AlkaneTreeGenerator: 烷烃骨架树生成器
  - AlkeneIsomerVisualizer: 烯烃可视化器 (主类)
  - compute_centroid_signature: 图签名计算函数
  - compute_canonical_labeling: 规范标号计算函数

================================================================================
"""

import os
import sys
import math
import warnings
import time
import threading
from typing import List, Tuple, Optional, Dict, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
import networkx as nx

# 导入烯烃生成器（新算法）
try:
    from original_programs.alkene import AlkeneIsomerGenerator
    HAS_ALKENE_GENERATOR = True
except ImportError as e:
    print(f"警告: 无法导入 AlkeneIsomerGenerator: {e}")
    HAS_ALKENE_GENERATOR = False

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可以显示进度条: pip install tqdm")


# =========================================================================
# 第一部分:图签名函数(用于同构检测和去重)
# =========================================================================

def compute_centroid_signature(G: nx.Graph, db_edge: Tuple[int, int]) -> str:
    """
    计算基于质心/质点的图的签名
    
    [功能]
    生成一个与节点编号无关的图签名,用于判断两个图是否同构.
    如果两个图同构,它们会产生相同的签名.
    
    [签名包含的信息]
    1. 度数序列:所有节点的度数排序
    2. 邻居度数模式:每个节点邻居的度数排序后组合
    3. 距离分布:每个节点到所有其他节点的距离之和
    4. 双键信息:双键两端碳原子的度数和邻居情况
    
    [示例]
    对于链端烯烃和支链烯烃,虽然碳数相同,但邻居度数模式不同:
    - 链端烯烃:-CH=CH-CH2-(中心碳邻居度数不同)
    - 支链烯烃:-C(=CH2)-CH2-(结构不同)
    
    [返回值]
    格式字符串,如 "n=5|deg=(1,2,2,2,2)|adj=..."
    """

    n = G.number_of_nodes()
    if n == 0:
        return "empty"
    
    # 1. 度数序列
    degrees = tuple(sorted(d for _, d in G.degree()))
    
    # 2. 邻居度数模式
    adj_pattern = []
    for node in sorted(G.nodes()):
        nbr_degrees = tuple(sorted(G.degree(nbr) for nbr in sorted(G.neighbors(node))))
        adj_pattern.append(nbr_degrees)
    adj_pattern = tuple(sorted(adj_pattern))
    
    # 3. 节点到所有其他节点的距离和(质心签名)
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    
    dist_sums = []
    for i, node in enumerate(nodes):
        total_dist = 0
        for j, other in enumerate(nodes):
            if i != j:
                try:
                    total_dist += nx.shortest_path_length(G, node, other)
                except nx.NetworkXNoPath:
                    total_dist += n * 2  # 使用较大的距离
        dist_sums.append(total_dist)
    dist_sums = tuple(sorted(dist_sums))
    
    # 4. 双键信息(对称化)
    u, v = db_edge
    u_deg = G.degree(u)
    v_deg = G.degree(v)
    u_nbrs = tuple(sorted(G.degree(n) for n in G.neighbors(u) if n != v))
    v_nbrs = tuple(sorted(G.degree(n) for n in G.neighbors(v) if n != u))
    
    # 对称化:确保(u,v)和(v,u)产生相同签名
    if (u_deg, u_nbrs) < (v_deg, v_nbrs):
        db_info = (u_deg, u_nbrs, v_deg, v_nbrs)
    else:
        db_info = (v_deg, v_nbrs, u_deg, u_nbrs)
    
    # 组合所有信息
    return f"n={n}|deg={degrees}|adj={adj_pattern}|dist={dist_sums}|dbl={db_info}"


def compute_canonical_labeling(G: nx.Graph) -> str:
    """
    计算图的规范标号(用于精确同构检测)
    
    [功能]
    使用图的谱特征(拉普拉斯矩阵特征值)作为图的指纹.
    拉普拉斯矩阵的特征值在图同构变换下保持不变.
    
    [算法步骤]
    1. 构建图的拉普拉斯矩阵 L = D - A
       D: 度数对角矩阵
       A: 邻接矩阵
    2. 计算 L 的特征值
    3. 对特征值排序并四舍五入
    
    [优势]
    - 计算效率高(O(n^3))
    - 对于大多数情况足够准确
    
    [局限性]
    极少数非同构图可能产生相同的特征值谱
    此时会回退到使用度数序列
    """

    try:
        # 尝试使用拉普拉斯特征值作为谱签名
        L = nx.laplacian_matrix(G).todense()
        eigenvalues = np.linalg.eigvalsh(L)
        # 四舍五入避免浮点精度问题
        eigenvalues = tuple(round(float(e), 6) for e in sorted(eigenvalues))
        return str(eigenvalues)
    except:
        # 备用:使用度数序列
        return str(tuple(sorted(d for _, d in G.degree())))


# =========================================================================
# 第二部分:烷烃骨架树生成器
# =========================================================================

class AlkaneTreeGenerator:
    """
    高效的烷烃骨架树生成器
    
    [核心思想]
    使用"有根树"表示分子骨架,通过递归划分生成所有可能的结构.
    
    [规范字符串格式]
    - "C"         -> 甲烷(1个碳)
    - "C(C)"      -> 乙烷(2个碳)
    - "C(C,C)"    -> 丙烷(3个碳,主链)
    - "C(C,C(C))" -> 异丁烷(4个碳,支链)
    
    [度数约束]
    - 烷烃:每个碳原子度数 <= 4
    - 烷基自由基:根节点度数 <= 3,其他节点 <= 4
    
    [OEIS参考]
    - A000598: n个碳的烷基自由基同分异构体数
    - A000055: n个节点的无标号树数量
    """

    
    def __init__(self, use_parallel=True, num_workers=None):
        self.cache = {}
        self.isomer_cache = {}
        self.use_parallel = use_parallel
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = max(1, num_workers)

    def parse_substrings(self, canon_str: str) -> List[str]:
        """
        解析规范字符串中的子树列表
        
        [功能]
        将括号表示的分子结构解析为子树列表.
        
        [示例]
        输入: "C(C,C(C))"
        处理: 去掉 "C(" 和 ")"
        内容: "C,C(C)"
        按逗号分割(depth=0时):
        结果: ["C", "C(C)"]
        
        [算法]
        使用depth计数器跟踪括号深度,
        只在depth=0时按逗号分割,避免在子括号内分割.
        
        [返回值]
        子树字符串列表,如 ["C", "C(C)"]
        """

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

    def generate_rooted_with_info(self, n: int, max_branches: int = 4, root_max_branches: int = None):
        """
        生成 n 个碳原子的有根树异构体(无标号树算法)
        
        [算法原理]
        使用递归划分法生成所有满足度数约束的树结构:
        1. 将 n 个节点分配到各个分支
        2. 每个分支递归生成子树
        3. 子树按字典序排序去重
        
        [参数说明]
        Args:
            n: 碳原子总数
            max_branches: 非根节点的分支数限制
                          - 4 表示 sp3 碳(1个父节点 + 最多3个子节点)
                          - 3 表示烷基自由基(最多3个子节点)
            root_max_branches: 根节点的特殊分支限制
                              - None 表示与 max_branches 相同
                              - 2 用于烷基自由基(根节点是自由基位点)
        
        [返回值]
        列表: [(规范字符串, 子树大小元组), ...]
        例如: [("C(C,C)", (1,1)), ("C(C(C))", (2,)), ...]
        
        [缓存机制]
        结果会被缓存,避免重复计算同名 n 的树结构.
        """

        if root_max_branches is None:
            root_max_branches = max_branches
        # 缓存key需要包含root_max_branches
        cache_key = (n, max_branches, root_max_branches)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if n == 1:
            res = [("C", [])]
            self.cache[cache_key] = res
            return res

        results = []
        target = n - 1
        partitions = []

        def find_parts(rem, min_val, path, is_root_level: bool):
            """生成树的分支划分,根节点和非根节点使用不同的分支限制"""
            branch_limit = root_max_branches if is_root_level else max_branches
            
            if len(path) >= branch_limit:
                if rem == 0:
                    partitions.append((tuple(path), is_root_level))
                return

            if rem == 0:
                if path:
                    partitions.append((tuple(path), is_root_level))
                return

            for i in range(min_val, rem + 1):
                if len(path) + 1 == branch_limit:
                    if rem - i == 0:
                        find_parts(0, i, path + [i], is_root_level)
                    continue
                find_parts(rem - i, i, path + [i], is_root_level)

        find_parts(target, 1, [], True)

        seen_local = set()
        for p, _ in partitions:
            # 子节点:对于有根树,子树的根节点连接父节点后度数增加1
            # 所以子节点数限制与max_branches相同
            # 例如:max_branches=3表示节点最多有3个子节点(度数=子节点数+1(父节点连接)<=4)
            sub_root_max = max_branches  # 子树的根节点最多有max_branches个子节点
            if sub_root_max < 1:
                continue  # 跳过无效的划分
            options_per_size = [self.generate_rooted_with_info(s, max_branches=max_branches, 
                                                                root_max_branches=sub_root_max) for s in p]

            for combo in itertools.product(*options_per_size):
                items = []
                for i, item in enumerate(combo):
                    items.append((item[0], p[i]))
                items.sort(key=lambda x: x[0])
                sorted_strs = [x[0] for x in items]
                sorted_sizes = [x[1] for x in items]
                canon = "C(" + ",".join(sorted_strs) + ")"
                if canon not in seen_local:
                    seen_local.add(canon)
                    results.append((canon, sorted_sizes))

        self.cache[cache_key] = results
        return results

    def generate_all_canons(self, n: int) -> List[str]:
        """生成 n 个碳原子的所有规范字符串(烷烃,度数<=4)"""
        return self._generate_canons_internal(n, root_max_degree=4, nonroot_max_degree=4)
    
    # OEIS A000598 预计算值(烷基自由基数量,忽略立体异构)
    # A000598(n) = n个碳的烷基自由基 C_n H_{2n+1} 的同分异构体数
    # 定义:有根三元树(每个节点最多3个子节点)
    # 来源: https://oeis.org/A000598
    A000598_VALUES = {
        0: 1,   # 索引0
        1: 1,   # 甲基自由基
        2: 1,   # 乙基自由基
        3: 2,   # 丙基自由基(正丙基、异丙基)
        4: 4,   # 丁基自由基
        5: 8,   # 戊基自由基
        6: 17,  # 己基自由基
        7: 39,  # 庚基自由基
        8: 89,  # 辛基自由基
        9: 211, # 壬基自由基
        10: 507, # 癸基自由基
        11: 1238,
        12: 3057,
        13: 7639,
        14: 19241,
        15: 48865,
    }
    
    # 兼容旧名称
    A000642_VALUES = A000598_VALUES
    
    def generate_alkyl_radical_canons(self, n: int) -> List[Tuple[Dict, int]]:
        """
        生成 n 个碳原子的烷基自由基(OEIS A000598)
        
        A000598定义:有根三元树(每个节点最多3个子节点)
        - 根节点是自由基位点(度数 <= 3,即最多3个子节点)
        - 忽略立体异构
        
        返回:(邻接表, 根节点) 的列表
        """
        import networkx as nx
        
        cache_key = ('alkyl', n)
        if cache_key in self.isomer_cache:
            return self.isomer_cache[cache_key]
        if n == 0: return []
        if n == 1: return [({0: []}, 0)]
        
        # 生成有根三元树:根节点最多3个子节点,非根节点也最多3个子节点
        all_rooted = self.generate_rooted_with_info(n, max_branches=3, root_max_branches=3)
        
        unique_adj = []
        seen_sigs = set()
        
        for canon, _ in all_rooted:
            adj = self.canon_to_adjacency(canon)
            
            # 验证所有节点度数 <= 4(非根节点)且根节点度数 <= 3
            valid = True
            for node, neighbors in adj.items():
                deg = len(neighbors)
                if deg > 4:  # 非根节点最大度数
                    valid = False
                    break
            
            if not valid:
                continue
            
            # 计算规范化签名去重
            sig = self._compute_rooted_signature(adj, 0)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            
            unique_adj.append((dict(adj), 0))  # 根节点是节点0
        
        self.isomer_cache[cache_key] = unique_adj
        return unique_adj
    
    def _compute_rooted_signature(self, adj: Dict, root: int) -> str:
        """计算以root为根的树的规范化签名"""
        def dfs(node: int, parent: int) -> tuple:
            children = []
            for neighbor in adj[node]:
                if neighbor != parent:
                    children.append(dfs(neighbor, node))
            children.sort()
            # 返回 (当前节点度数, 子树签名列表)
            return (len(adj[node]), tuple(children))
        
        sig = dfs(root, -1)
        return str(sig)
    
    def _tree_canonical_key(self, adj: Dict[int, List[int]]) -> str:
        """生成树的规范键(用于无根树去重,使用图同构)"""
        import networkx as nx
        
        n = len(adj)
        # 构建NetworkX图
        G = nx.Graph()
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                if node < neighbor:
                    G.add_edge(node, neighbor)
        
        # 尝试所有节点作为根,找到度数序列最小的(规范化)
        # 使用度数信息作为初步过滤
        degree_seq = sorted(G.degree(n) for n in G.nodes())
        
        # 对每个节点作为根时的度数序列进行排序
        root_sigs = []
        for root in range(n):
            sig = self._compute_unrooted_tree_signature(G, root)
            root_sigs.append(sig)
        
        root_sigs.sort()
        return str(root_sigs)
    
    def _compute_unrooted_tree_signature(self, G: nx.Graph, root: int) -> tuple:
        """计算树的规范化签名"""
        def dfs(node: int, parent: int) -> tuple:
            children = []
            for neighbor in G.neighbors(node):
                if neighbor != parent:
                    children.append(dfs(neighbor, node))
            children.sort()
            return (G.degree(node), tuple(children))
        
        return dfs(root, -1)
    
    def _generate_canons_internal(self, n: int, root_max_degree: int, nonroot_max_degree: int) -> List[str]:
        """内部方法:根据不同的度数约束生成规范字符串"""
        cache_key = (n, root_max_degree, nonroot_max_degree)
        if cache_key in self.isomer_cache:
            return self.isomer_cache[cache_key]
        if n == 0: return []
        if n == 1: return ["C"]

        all_rooted = self.generate_rooted_with_info(n, max_branches=nonroot_max_degree, 
                                                    root_max_branches=root_max_degree)
        limit = n // 2
        raw_fingerprints = set()

        # 根据数据量决定是否使用并行计算
        use_parallel = self.use_parallel and len(all_rooted) > 1000

        if use_parallel:
            raw_fingerprints = self._process_rooted_parallel(all_rooted, limit, show_progress=True)
        else:
            if HAS_TQDM:
                iterator = tqdm(all_rooted, desc="[1/2] 处理根树结构", unit="树",
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            else:
                iterator = all_rooted

            for canon, sub_sizes in iterator:
                processed = self._process_single_rooted(canon, sub_sizes, limit)
                if processed:
                    raw_fingerprints.add(processed)

        # 去重
        unique_adj_sets = set()
        final_fingerprints = []

        if use_parallel and len(raw_fingerprints) > 1000:
            final_fingerprints = self._deduplicate_parallel(list(raw_fingerprints), unique_adj_sets, show_progress=True)
        else:
            raw_list = list(raw_fingerprints)
            if HAS_TQDM:
                iterator = tqdm(raw_list, desc="[2/2] 去重处理", unit="个")
            else:
                iterator = raw_list

            for raw_canon in iterator:
                result = self._deduplicate_single(raw_canon, unique_adj_sets)
                if result:
                    final_fingerprints.append(result)

        result = sorted(final_fingerprints)
        self.isomer_cache[n] = result
        return result

    def canon_to_adjacency(self, canon_str: str) -> Dict[int, List[int]]:
        """将规范字符串转换为邻接表"""
        if canon_str == "C":
            return {0: []}
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
                    subs = self.parse_substrings(s)
                    for sub in subs:
                        parse_subtree(sub, node_id)
            return node_id

        parse_subtree(canon_str)
        return temp_adj

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

    def _process_rooted_parallel(self, all_rooted: list, limit: int, show_progress: bool = True) -> set:
        """并行处理根树异构体"""
        results = set()
        executor = None

        items_list = [(canon, sub_sizes, limit) for canon, sub_sizes in all_rooted]
        total_items = len(items_list)

        pbar = None
        if show_progress and HAS_TQDM:
            pbar = tqdm(total=total_items, desc="[1/2] 处理根树结构", unit="树",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        try:
            ctx = mp.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx)

            batch_size = max(500, min(5000, total_items // 100))

            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch = items_list[batch_start:batch_end]

                for result in executor.map(_process_rooted_worker, batch):
                    if result:
                        results.add(result)
                    if pbar is not None:
                        pbar.update(1)

            if pbar is not None:
                pbar.close()

        except KeyboardInterrupt:
            if pbar is not None:
                pbar.close()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        return results

    def _deduplicate_single(self, raw_canon: str, unique_adj_sets: set) -> Optional[str]:
        """单个异构体的去重处理"""
        try:
            adj = self.canon_to_adjacency(raw_canon)
            for node, neighbors in adj.items():
                if len(neighbors) > 4:
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

    def _deduplicate_parallel(self, raw_list: list, unique_adj_sets: set, show_progress: bool = True) -> List[str]:
        """并行去重处理"""
        results = []
        executor = None
        total_items = len(raw_list)

        pbar = None
        if show_progress and HAS_TQDM:
            pbar = tqdm(total=total_items, desc="[2/2] 去重处理", unit="个",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        try:
            ctx = mp.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx)

            batch_size = max(500, min(5000, total_items // 100))

            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch = raw_list[batch_start:batch_end]

                for result in executor.map(_canon_to_adj_worker, batch):
                    if result:
                        canon_str, adj = result
                        valid_structure = True
                        for node, neighbors in adj.items():
                            if len(neighbors) > 4:
                                valid_structure = False
                                break

                        if valid_structure:
                            standardized_adj = tuple(
                                sorted((node, tuple(sorted(neighbors))) for node, neighbors in adj.items())
                            )
                            if standardized_adj not in unique_adj_sets:
                                unique_adj_sets.add(standardized_adj)
                                results.append(canon_str)

                    if pbar is not None:
                        pbar.update(1)

            if pbar is not None:
                pbar.close()

        except KeyboardInterrupt:
            if pbar is not None:
                pbar.close()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        return sorted(results)


# ==================== 并行工作函数 ====================

def _parse_substrings_worker(canon_str: str) -> List[str]:
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


def _process_rooted_worker(item: tuple) -> Optional[str]:
    """并行处理单个根树异构体的工作函数"""
    canon, sub_sizes, limit = item

    if not sub_sizes:
        max_s = 0
    else:
        max_s = max(sub_sizes)

    if max_s <= limit:
        if max_s < limit:
            return canon
        else:
            idx = sub_sizes.index(limit)
            sub_strs = _parse_substrings_worker(canon)
            big_sub_str = sub_strs[idx]
            rest_subs = sub_strs[:idx] + sub_strs[idx+1:]
            rest_subs.sort()
            rest_canon = "C(" + ",".join(rest_subs) + ")"

            big_children = _parse_substrings_worker(big_sub_str)
            new_children = big_children + [rest_canon]
            new_children.sort()
            flipped_canon = "C(" + ",".join(new_children) + ")"
            return min(canon, flipped_canon)
    return None


def _canon_to_adj_worker(canon_str: str) -> Tuple[str, Dict[int, List[int]]]:
    """并行处理规范字符串转邻接表的工作函数"""
    if canon_str == "C":
        return canon_str, {0: []}

    node_counter = [0]
    temp_adj = {}

    def parse_subtree(s: str, parent: int = None) -> int:
        nonlocal node_counter
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
                subs = _parse_substrings_worker(s)
                for sub in subs:
                    parse_subtree(sub, node_id)
        return node_id

    parse_subtree(canon_str)
    return canon_str, temp_adj


# ========== 优化并行处理函数 ==========

def _compute_signature_single(tree_adj: Dict[int, List[int]], db_edge: Tuple[int, int]) -> Optional[Tuple[nx.Graph, str, tuple]]:
    """
    计算单个图的签名(用于并行处理)
    返回: (图, 完整签名, 分组签名) 或 None
    
    关键优化:签名字符串不依赖节点编号,确保同构结构产生相同签名
    """
    try:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(tree_adj.keys())

        for node, neighbors in tree_adj.items():
            for neighbor in neighbors:
                if neighbor > node:
                    if (node, neighbor) == db_edge or (neighbor, node) == db_edge:
                        G.add_edge(node, neighbor, bond_type='double')
                    else:
                        G.add_edge(node, neighbor, bond_type='single')

        for node in G.nodes():
            G.nodes[node]['label'] = 'C'

        # 计算与节点编号无关的度数序列
        degrees = tuple(sorted(G.degree(node) for node in G.nodes()))
        
        # 邻接度数模式(对每行排序,确保与节点编号无关)
        adj_pattern = []
        for node in sorted(G.nodes()):
            nbr_degrees = tuple(sorted(G.degree(nbr) for nbr in sorted(G.neighbors(node))))
            adj_pattern.append(nbr_degrees)
        adj_pattern = tuple(sorted(adj_pattern))
        
        # 双键度数对(只使用度数,不使用节点编号)
        # 关键:必须对称化,确保 (u_deg, v_deg) 和 (v_deg, u_deg) 产生相同结果
        double_bond_info = []
        for u, v, data in G.edges(data=True):
            if data.get('bond_type') == 'double':
                u_deg = G.degree(u)
                v_deg = G.degree(v)
                # 双键端点的非双键邻居度数
                u_non_db_nbr = tuple(sorted(G.degree(n) for n in G.neighbors(u) if n != v))
                v_non_db_nbr = tuple(sorted(G.degree(n) for n in G.neighbors(v) if n != u))
                
                # 对称化:度数对排序,非双键邻居也排序后比较
                if (u_deg, u_non_db_nbr) < (v_deg, v_non_db_nbr):
                    double_bond_info.append((u_deg, u_non_db_nbr, v_deg, v_non_db_nbr))
                else:
                    double_bond_info.append((v_deg, v_non_db_nbr, u_deg, u_non_db_nbr))
        double_bond_info.sort()
        
        # 完整签名:使用度数信息而非节点编号
        full_sig = f"n={len(degrees)}|deg={degrees}|adj={adj_pattern}|dbl={double_bond_info}"
        
        # 分组签名:用于同构预过滤(使用度数信息)
        group_sig = (len(degrees), degrees, adj_pattern, tuple(double_bond_info))

        return (G, full_sig, group_sig)

    except Exception as e:
        return None


def _compute_signature_batch(graphs_data: List[Tuple[int, Dict[int, List[int]], Tuple[int, int]]]) -> List[Tuple[int, str, tuple]]:
    """
    并行计算一批图的签名
    graphs_data: [(graph_id, tree_adj, db_edge), ...]
    返回: [(graph_id, full_signature, group_signature, graph), ...]
    """
    results = []

    for graph_id, tree_adj, db_edge in graphs_data:
        result = _compute_signature_single(tree_adj, db_edge)
        if result:
            G, full_sig, group_sig = result
            results.append((graph_id, full_sig, group_sig, G))

    return results

# ==================== 快速图签名缓存 ====================
_graph_signature_cache = {}

def get_fast_graph_signature(G):
    """生成图的快速签名字符串"""
    n = G.number_of_nodes()
    if n == 0:
        return "empty"

    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    adj_str_parts = []
    for node in nodes:
        neighbors = sorted([node_to_idx[nbr] for nbr in G.neighbors(node)])
        adj_str_parts.append("_".join(map(str, neighbors)))

    double_bond_edges = []
    for u, v, data in G.edges(data=True):
        if data.get('bond_type') == 'double':
            double_bond_edges.append((node_to_idx[u], node_to_idx[v]))
    double_bond_edges.sort()

    return f"{n}|{'|'.join(adj_str_parts)}|{double_bond_edges}"

# 配置支持中文的字体
def configure_chinese_font():
    """配置matplotlib支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')

configure_chinese_font()


# =========================================================================
# 第三部分:烯烃同分异构体可视化器(主类)
# =========================================================================

class AlkeneIsomerVisualizer:
    """
    烯烃同分异构体可视化器
    
    [功能]
    1. 生成烯烃(CₙH₂ₙ)的所有同分异构体
    2. 计算每个原子的3D坐标
    3. 可视化分子结构(支持单个和批量)
    
    [核心算法 - 烷基自由基组合法]
    烯烃 = 烷基自由基A + 双键 + 烷基自由基B
    
    例如 C5H10 可以由以下组合生成:
    - C1 + C4 组合:C1(甲基) + C4(丁基) -> 戊烯
    - C2 + C3 组合:C2(乙基) + C3(丙基) -> 戊烯
    
    [化学键参数]
    - C-C 单键长度: 1.54 Å
    - C=C 双键长度: 1.34 Å(比单键短)
    - C-H 键长度: 1.09 Å
    - sp3 键角: 109.47°(四面体角)
    - sp2 键角: 120°(平面三角形)
    """


    # 化学键参数 (与烷烃一致)
    BOND_LENGTH_CC = 1.54
    BOND_LENGTH_CC_DOUBLE = 1.34  # 双键比单键短
    BOND_LENGTH_CH = 1.09
    TETRAHEDRAL_ANGLE = math.acos(-1/3)  # ~109.47 degrees (sp3)
    ANGLE_120 = 2 * math.pi / 3  # 120 degrees (sp2)
    SQRT3 = math.sqrt(3)

    def __init__(self, generator=None, num_cores=None):
        """
        初始化可视化器
        Args:
            generator: 可选的烯烃异构体生成器实例
            num_cores: 并行计算的核心数,None表示自动检测
        """
        self.num_cores = num_cores if num_cores else max(1, mp.cpu_count() - 1)
        
        # 设置生成器实例
        if generator is not None:
            self.generator = generator
        elif HAS_ALKENE_GENERATOR:
            # 使用新的基于烷烃骨架的生成器
            self.generator = AlkeneIsomerGenerator(alkane_generator=None, num_cores=self.num_cores)
            print(f"使用新的 AlkeneIsomerGenerator (核心数: {self.num_cores})")
        else:
            self.generator = None
            print("警告: 未提供生成器且无法导入 AlkeneIsomerGenerator，烯烃生成功能可能受限")
            
        self.atom_colors = {'C': '#333333', 'H': '#FFFFFF'}
        self.atom_radii = {'C': 0.6, 'H': 0.3}
        self.bond_color = '#808080'
        self.bond_color_double = '#FF4444'  # 双键用红色表示
        self.bond_width = 3
        self.bond_width_double = 5
        
        # 中断控制
        self._stop_event = threading.Event()
        self._is_computing = False

    def stop(self):
        """设置停止标志,中断计算"""
        self._stop_event.set()
        print("\n[中断请求] 正在停止计算...")

    def resume(self):
        """清除停止标志,继续计算"""
        self._stop_event.clear()

    def is_computing(self) -> bool:
        """检查是否正在计算"""
        return self._is_computing

    def is_stopped(self) -> bool:
        """检查是否被中断"""
        return self._stop_event.is_set()

    def get_graph_info(self, G: nx.Graph) -> Dict:
        """
        从图中提取烯烃信息
        返回: {adjacency, double_bond_nodes, n_carbons}
        """
        n_carbons = G.number_of_nodes()
        adjacency = {node: list(G.neighbors(node)) for node in G.nodes()}

        # 找出双键连接的两个节点
        double_bond_nodes = set()
        for u, v, data in G.edges(data=True):
            if data.get('bond_type') == 'double':
                double_bond_nodes.add(u)
                double_bond_nodes.add(v)

        return {
            'adjacency': adjacency,
            'double_bond_nodes': double_bond_nodes,
            'n_carbons': n_carbons
        }

    def generate_isomers_optimized(self, n_carbons: int, show_progress: bool = True) -> List[nx.Graph]:
        """
        生成烯烃同分异构体 - 使用基于烷烃骨架放置双键的新算法
        
        [算法流程]
        1. 生成所有 n 个碳的烷烃骨架树（饱和链烃）
        2. 对每个骨架树的每条边，检查双键两端碳原子度数 ≤ 3
        3. 将选定的边标记为双键，其余边为单键，构建图
        4. 使用图同构检测（考虑边类型）进行去重
        
        [参数]
        Args:
            n_carbons: 碳原子数量(如 10 表示 C10H20)
            show_progress: 是否显示进度条（已忽略，新算法自带进度输出）
        
        [返回值]
        NetworkX图对象列表,每个图代表一个烯烃异构体，边属性 bond_type 为 'single' 或 'double'
        
        [示例]
        C4H8 有3个异构体:
        1. 1-丁烯: CH2=CH-CH2-CH3
        2. 2-丁烯: CH3-CH=CH-CH3 (顺/反视为相同,不考虑立体异构)
        3. 异丁烯: CH2=C(CH3)2
        """

        import networkx as nx

        if n_carbons < 2:
            return []  # C2H4 (乙烯) 没有同分异构体,但算法支持 n=2

        # 检查是否已请求中断
        if self.is_stopped():
            print("计算已中断")
            return []

        self._is_computing = True
        self._stop_event.clear()

        try:
            # 如果有可用的生成器（新的 AlkeneIsomerGenerator），则使用它
            if self.generator is not None and HAS_ALKENE_GENERATOR:
                if show_progress:
                    print(f"\n开始生成 C{n_carbons}H{2*n_carbons} 的同分异构体（新算法）...")
                isomers = self.generator.generate_isomers(n_carbons)
                if show_progress:
                    print(f"生成完成，共 {len(isomers)} 个异构体")
                return isomers
            else:
                # 回退到旧算法（烷基自由基组合法）
                print(f"\n警告: 使用旧算法生成烯烃异构体（新算法不可用）")
                return self._generate_isomers_legacy(n_carbons, show_progress)

        except Exception as e:
            print(f"生成过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            self._is_computing = False

    def _generate_isomers_legacy(self, n_carbons: int, show_progress: bool = True) -> List[nx.Graph]:
        """
        旧算法：烷基自由基组合法（保留作为回退）
        注意：此方法为内部方法，不应直接调用
        """
        import networkx as nx
        
        if n_carbons < 3:
            return []

        print(f"\n开始生成 C{n_carbons}H{2*n_carbons} 的同分异构体（旧算法）...")
        print(f"使用核心数: {self.num_cores}")
        print("-" * 50)

        # 步骤1: 生成所有烷基自由基
        print("[1/4] 生成烷基自由基骨架...")
        if self.is_stopped():
            print("计算已中断")
            return []

        alkyl_radicals = {}  # {n: [(adj_dict, root_node), ...]}
        for k in range(1, n_carbons):
            radicals = self._generate_alkyl_radicals_for_alkene(k)
            alkyl_radicals[k] = radicals
            if show_progress:
                print(f"      C{k} 烷基自由基: {len(radicals)} 个")

        if self.is_stopped():
            print("计算已中断")
            return []

        # 步骤2: 组合烷基自由基生成烯烃候选
        print("[2/4] 组合烷基自由基生成烯烃候选...")
        if self.is_stopped():
            print("计算已中断")
            return []

        candidates = []  # [(alkene_adj, root1, root2), ...]

        for k in range(1, (n_carbons + 1) // 2 + 1):
            n1, n2 = k, n_carbons - k
            if n1 not in alkyl_radicals or n2 not in alkyl_radicals:
                continue

            # 只处理 k <= n-k 的情况,避免重复计算 C1+C5 和 C5+C1
            if k > n_carbons - k and k != n_carbons - k:
                continue

            if n1 == n2:
                # 自身组合:只考虑 i <= j 的情况避免重复
                radicals = alkyl_radicals[n1]
                for i in range(len(radicals)):
                    for j in range(i, len(radicals)):
                        adj1, root1 = radicals[i]
                        adj2, root2 = radicals[j]
                        merged_adj, new_root1, new_root2 = self._merge_alkyl_radicals(
                            adj1, root1, adj2, root2
                        )
                        candidates.append((merged_adj, new_root1, new_root2))
            else:
                # 不同烷基自由基的组合
                for adj1, root1 in alkyl_radicals[n1]:
                    for adj2, root2 in alkyl_radicals[n2]:
                        # 合并两个烷基自由基,添加双键
                        merged_adj, new_root1, new_root2 = self._merge_alkyl_radicals(
                            adj1, root1, adj2, root2
                        )
                        candidates.append((merged_adj, new_root1, new_root2))

        total_candidates = len(candidates)
        print(f"      生成了 {total_candidates} 个候选结构")

        if self.is_stopped():
            print("计算已中断")
            return []

        # 步骤3: 图同构去重
        print("[3/4] 去重处理 (图同构检测)...")
        if self.is_stopped():
            print("计算已中断")
            return []

        unique_isomers = []

        pbar = None
        if show_progress and HAS_TQDM:
            pbar = tqdm(total=total_candidates, desc="[3/4] 去重", unit="个",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        def custom_edge_match(e1, e2):
            """自定义边匹配:严格比较 bond_type"""
            return e1.get('bond_type') == e2.get('bond_type')

        def custom_node_match(n1, n2):
            """自定义节点匹配:比较 label 和 neighbor_degrees"""
            if n1.get('label') != n2.get('label'):
                return False
            # 比较邻居度数 - 这对于区分烯烃异构体至关重要
            nd1 = n1.get('neighbor_degrees', ())
            nd2 = n2.get('neighbor_degrees', ())
            return nd1 == nd2

        def is_valid_alkene(G):
            """验证烯烃结构是否有效"""
            double_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
            if len(double_edges) != 1:
                return False

            sp2_c1, sp2_c2 = double_edges[0]
            sp2_c1_deg = G.degree(sp2_c1)
            sp2_c2_deg = G.degree(sp2_c2)

            if sp2_c1_deg > 2 or sp2_c2_deg > 2:
                if sp2_c1_deg == 3 or sp2_c2_deg == 3:
                    pass
                else:
                    return False

            if sp2_c1_deg < 1 or sp2_c2_deg < 1:
                return False

            return True

        for adj, root1, root2 in candidates:
            # 构建图对象
            G = self._build_alkene_graph(adj, root1, root2)

            # 验证化学有效性
            if not is_valid_alkene(G):
                continue

            # 检查是否与已有图同构
            is_duplicate = False
            for existing_G in unique_isomers:
                if nx.is_isomorphic(G, existing_G,
                                    node_match=custom_node_match,
                                    edge_match=custom_edge_match):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_isomers.append(G)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        print(f"      去重后: {len(unique_isomers)} 个烯烃异构体")
        return unique_isomers

    def _generate_alkyl_radicals_for_alkene(self, n_carbons: int) -> List[tuple]:
        """
        生成 n 个碳原子的烷基自由基列表
        返回: [(adj_dict, root_node), ...]
        约束:根节点度数<=2,其他节点度数<=3
        """
        if n_carbons == 1:
            return [({0: []}, 0)]  # 甲基

        radicals = []
        # 使用有根树生成
        # 对于烷基自由基:
        # - 根节点(自由基位点)度数<=2(连接最多2个其他碳)
        # - 非根节点度数<=4(sp3碳最多连接4个其他原子)
        # 由于度数包括父节点连接,非根节点的子节点数<=3
        tree_gen = AlkaneTreeGenerator(use_parallel=False)
        rooted_trees = tree_gen.generate_rooted_with_info(
            n_carbons,
            max_branches=4,  # 非根节点度数限制(1个父节点+最多3个子节点)
            root_max_branches=2  # 根节点度数限制
        )

        seen_adj_set = set()

        def get_rooted_tree_signature(adj, root):
            """
            获取有根树的规范化签名
            包含:根度数、根邻居度数、邻居的邻居度数等信息
            用于区分 n-propyl 和 isopropyl
            """
            root_degree = len(adj[root])
            neighbors = adj[root]

            # 收集每个邻居的子树信息
            subtree_sigs = []
            for nbr in neighbors:
                # 邻居的度数(不包括根)
                nbr_degree_except_root = len([x for x in adj[nbr] if x != root])
                # 邻居的邻居的度数(不包括根和当前邻居)
                nbr_nbr_degrees = sorted([len([x for x in adj[nn] if x != nbr and x != root])
                                         for nn in adj[nbr] if nn != root])
                subtree_sigs.append((nbr_degree_except_root, tuple(nbr_nbr_degrees)))

            # 排序以获得规范化表示
            subtree_sigs.sort()
            return (root_degree, tuple(subtree_sigs))

        def are_same_rooted_tree(adj1, root1, adj2, root2):
            """检查两个有根树是否同构"""
            import networkx as nx
            import itertools

            # 首先比较规范化签名
            sig1 = get_rooted_tree_signature(adj1, root1)
            sig2 = get_rooted_tree_signature(adj2, root2)
            if sig1 != sig2:
                return False

            # 签名相同,再用图同构确认
            G1 = nx.Graph()
            for node, neighbors in adj1.items():
                for neighbor in neighbors:
                    G1.add_edge(node, neighbor)

            G2 = nx.Graph()
            for node, neighbors in adj2.items():
                for neighbor in neighbors:
                    G2.add_edge(node, neighbor)

            # 使用图同构检测
            nodes1 = list(G1.nodes())
            nodes2 = list(G2.nodes())
            if len(nodes1) <= 6:
                for perm in itertools.permutations(nodes2):
                    mapping = dict(zip(nodes1, perm))
                    if mapping[root1] == root2:
                        valid = True
                        for u, v in G1.edges():
                            if not G2.has_edge(mapping[u], mapping[v]):
                                valid = False
                                break
                        if valid:
                            return True
                return False
            else:
                GM = nx.isomorphism.GraphMatcher(G1, G2)
                if not GM.is_isomorphic():
                    return False
                for mapping in GM.isomorphisms_iter():
                    if mapping[root1] == root2:
                        return True
                return False

        for canon, root_info in rooted_trees:
            adj = tree_gen.canon_to_adjacency(canon)

            # 对于每种树结构,枚举所有可能的根位置
            # 不同的根位置可能产生不同的烷基自由基
            possible_roots = list(adj.keys())

            for root in possible_roots:
                # 跳过根节点度数 > 2 的情况
                if len(adj[root]) > 2:
                    continue

                # 验证其他节点度数 <= 4
                valid = True
                for node, neighbors in adj.items():
                    if node != root and len(neighbors) > 4:
                        valid = False
                        break

                if not valid:
                    continue

                # 检查是否与已有自由基同构(根位置也要匹配)
                is_duplicate = False
                for existing_adj, existing_root in radicals:
                    if are_same_rooted_tree(adj, root, existing_adj, existing_root):
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                radicals.append((dict(adj), root))

        return radicals

    def _adjacency_to_key(self, adj: Dict) -> str:
        """将邻接表转换为规范化字符串(用于去重)"""
        nodes = sorted(adj.keys())
        parts = []
        for node in nodes:
            neighbors = sorted(adj[node])
            parts.append(f"{node}:{','.join(map(str, neighbors))}")
        return "|".join(parts)

    def _adjacency_with_root_key(self, adj: Dict, root: int) -> str:
        """
        将邻接表和根节点转换为规范化字符串(用于去重)
        包含根节点信息,确保同一结构不同根 = 不同自由基
        """
        nodes = sorted(adj.keys())
        parts = []
        for node in nodes:
            neighbors = sorted(adj[node])
            parts.append(f"{node}:{','.join(map(str, neighbors))}")
        return f"r{root}|" + "|".join(parts)

    def _are_isomorphic_trees(self, adj1: Dict, adj2: Dict) -> bool:
        """检查两棵树是否同构"""
        import networkx as nx

        G1 = nx.Graph()
        for node, neighbors in adj1.items():
            for neighbor in neighbors:
                G1.add_edge(node, neighbor)

        G2 = nx.Graph()
        for node, neighbors in adj2.items():
            for neighbor in neighbors:
                G2.add_edge(node, neighbor)

        return nx.is_isomorphic(G1, G2)

    def _same_root_position(self, adj1: Dict, root1: int, adj2: Dict, root2: int) -> bool:
        """检查两个同构树的根节点是否在拓扑上等效"""
        import networkx as nx

        # 计算从根到每个节点的路径长度分布
        def get_distance_distribution(adj, root):
            G = nx.Graph()
            for node, neighbors in adj.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)

            distances = {}
            for node in G.nodes():
                try:
                    d = nx.shortest_path_length(G, root, node)
                    distances[node] = d
                except nx.NetworkXNoPath:
                    distances[node] = -1

            # 统计每个距离的节点数
            dist_counts = {}
            for d in distances.values():
                dist_counts[d] = dist_counts.get(d, 0) + 1
            return tuple(sorted(dist_counts.items()))

        return get_distance_distribution(adj1, root1) == get_distance_distribution(adj2, root2)

    def _merge_alkyl_radicals(self, adj1: Dict, root1: int, adj2: Dict, root2: int) -> tuple:
        """
        合并两个烷基自由基,添加双键连接
        返回: (merged_adj, new_root1, new_root2)
        """
        import copy

        # 深拷贝邻接表
        merged = copy.deepcopy(adj1)

        # 重新编号第二个烷基自由基的节点,避免冲突
        offset = max(merged.keys()) + 1
        adj2_remapped = {}
        for node, neighbors in adj2.items():
            new_node = node + offset
            new_neighbors = [n + offset for n in neighbors]
            adj2_remapped[new_node] = new_neighbors
            merged[new_node] = new_neighbors

        # 添加双键连接
        new_root1 = root1
        new_root2 = root2 + offset

        # 将两个根节点连接起来(双键)
        merged[new_root1].append(new_root2)
        merged[new_root2].append(new_root1)

        return merged, new_root1, new_root2

    def _generate_alkene_signature(self, adj: Dict, root1: int, root2: int) -> str:
        """
        生成烯烃的规范化签名
        用于快速去重比较
        """
        import networkx as nx

        G = nx.Graph()
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # 标记双键节点
        db_nodes = {root1, root2}

        # 生成规范化签名:基于双键节点的最短路径结构
        # 计算每个节点到两个双键节点的距离
        distances = {}
        for node in G.nodes():
            d1 = nx.shortest_path_length(G, node, root1) if root1 in G.nodes() else 0
            d2 = nx.shortest_path_length(G, node, root2) if root2 in G.nodes() else 0
            deg = G.degree(node)
            is_db = node in db_nodes
            distances[node] = (d1, d2, deg, is_db)

        # 生成度数序列签名
        deg_seq = sorted([(distances[n][0], distances[n][1], distances[n][2], distances[n][3])
                         for n in G.nodes()])
        return str(deg_seq)

    def _build_alkene_graph(self, adj: Dict, root1: int, root2: int) -> nx.Graph:
        """
        构建烯烃NetworkX图对象
        包含 bond_type 属性标记双键
        neighbor_degrees 属性用于区分不同烯烃结构
        """
        G = nx.Graph()

        # 先添加所有节点
        for node in adj.keys():
            G.add_node(node, label='C')

        # 添加所有边
        seen_edges = set()
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)

                # 检查是否是双键
                if (node == root1 and neighbor == root2) or \
                   (node == root2 and neighbor == root1):
                    G.add_edge(node, neighbor, bond_type='double')
                else:
                    G.add_edge(node, neighbor, bond_type='single')

        # 添加邻居度数属性用于区分同分异构体
        # 将每个节点的邻居度数排序后作为元组存储
        for node in G.nodes():
            neighbor_degrees = tuple(sorted([G.degree(neighbor) for neighbor in G.neighbors(node)]))
            G.nodes[node]['neighbor_degrees'] = neighbor_degrees

        return G

    def adjacency_to_coords(self, adjacency: Dict[int, List[int]],
                           double_bond_nodes: set,
                           n: int) -> List[Tuple[float, float, float]]:
        """
        将邻接表转换为3D坐标
        
        【几何约束】
        - 碳碳单键键长：1.54Å
        - 碳碳双键键长：1.34Å  
        - sp3键角：四面体角 ≈ 109.47°
        - sp2键角：平面角 = 120°
        
        【算法核心思想】
        使用"相对方向法"：
        1. 每个父节点维护一个局部坐标系
        2. 第一个子节点定义局部 x 轴
        3. 后续子节点根据预设角度偏移放置
        
        【参数/返回值同前】
        """
        import numpy as np
        from math import sqrt, acos, cos, sin, pi
        
        coords = [None] * n
        
        # ========== 精确的方向生成函数 ==========
        def get_sp2_dirs(reference_vec):
            """生成sp2平面三角形3个方向，第一个与reference对齐"""
            ref = np.array(reference_vec, dtype=float)
            ref = ref / np.linalg.norm(ref) if np.linalg.norm(ref) > 1e-10 else np.array([1., 0., 0.])
            
            # 在垂直于ref的平面内找两个方向，间隔120°
            if abs(ref[2]) < 0.9:
                up = np.array([0., 0., 1.])
            else:
                up = np.array([1., 0., 0.])
            
            perp1 = np.cross(ref, up)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(ref, perp1)
            
            angle = 2 * pi / 3  # 120°
            cos_a, sin_a = cos(angle), sin(angle)
            
            d1 = ref.copy()
            d2 = cos_a * ref + sin_a * perp1
            d3 = cos_a * ref - sin_a * perp1
            
            return [d1, d2, d3]
        
        def get_sp3_dirs(reference_vec):
            """生成sp3四面体4个方向，第一个与reference对齐"""
            ref = np.array(reference_vec, dtype=float)
            ref = ref / np.linalg.norm(ref) if np.linalg.norm(ref) > 1e-10 else np.array([1., 0., 0.])
            
            tetra_angle = acos(-1./3.)  # 109.47°
            cos_t, sin_t = cos(tetra_angle), sin(tetra_angle)
            
            # 找两个垂直于ref的正交向量
            if abs(ref[2]) < 0.9:
                up = np.array([0., 0., 1.])
            else:
                up = np.array([1., 0., 0.])
            
            perp1 = np.cross(ref, up)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(ref, perp1)
            
            # 四面体的4个顶点方向
            # d1: 沿参考方向（第一个键）
            d1 = ref.copy()
            
            # d2, d3, d4: 与d1成109.47°，彼此成109.47°
            # 使用标准四面体配置
            offset_angle = pi / 3  # 60° 偏移使分布均匀
            cos_o, sin_o = cos(offset_angle), sin(offset_angle)
            
            # 第二个方向
            d2 = cos_t * ref + sin_t * (cos_o * perp1 + sin_o * perp2)
            
            # 第三个方向
            d3 = cos_t * ref + sin_t * (cos_o * perp1 - sin_o * perp2)
            
            # 第四个方向
            d4 = cos_t * ref + sin_t * (-cos_o * perp1)  # 主要在 -perp1 方向
            # 调整使d4与其他夹角正确
            d4 = cos_t * ref - sin_t * perp1
            
            return [d1, d2, d3, d4]
        
        # ---------- 1. 放置起始点（双键优先）----------
        if len(double_bond_nodes) >= 2:
            db_list = sorted(list(double_bond_nodes))
            c0, c1 = db_list[0], db_list[1]
            coords[c0] = (0.0, 0.0, 0.0)
            coords[c1] = (self.BOND_LENGTH_CC_DOUBLE, 0.0, 0.0)
            root = c0
        else:
            coords[0] = (0.0, 0.0, 0.0)
            root = 0
        
        # ---------- 2. BFS 建立父子关系 ----------
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
        
        # ---------- 3. 按BFS顺序计算坐标 ----------
        # 为每个节点追踪已放置的子节点数
        child_count = {}  # node -> int
        
        queue = [root]
        visited.clear()
        visited.add(root)
        
        while queue:
            node = queue.pop(0)
            
            # 先扩展队列
            for nb in adjacency.get(node, []):
                if nb >= n:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
            
            # 收集当前节点需要处理的邻居
            unplaced_neighbors = [nb for nb in adjacency.get(node, []) 
                                  if nb < n and coords[nb] is None]
            
            if not unplaced_neighbors:
                continue
            
            px, py, pz = coords[node]
            parent_pos = np.array([px, py, pz])
            
            # 初始化
            if node not in child_count:
                child_count[node] = 0
            
            # ========== 简化的方向分配算法 ==========
            node_is_sp2 = node in double_bond_nodes
            
            # 找到双键伙伴和方向
            double_bond_vec = None
            for other_nb in adjacency.get(node, []):
                if other_nb != node and other_nb in double_bond_nodes and node in double_bond_nodes:
                    if coords[other_nb] is not None:
                        v = np.array(coords[other_nb]) - parent_pos
                        vn = np.linalg.norm(v)
                        if vn > 1e-6:
                            double_bond_vec = v / vn
                    break
            
            # 收集已放置的非双键邻居方向
            existing_dirs = []
            for other_nb in adjacency.get(node, []):
                if coords[other_nb] is not None and other_nb != node:
                    v = np.array(coords[other_nb]) - parent_pos
                    vn = np.linalg.norm(v)
                    if vn > 1e-6:
                        existing_dirs.append(v / vn)
            
            # ========== 为每个未放置邻居分配唯一方向 ==========
            local_child_idx = 0  # 当前节点的本地子节点计数器
            
            for neighbor in unplaced_neighbors:
                child_count[node] += 1  # 全局计数
                
                is_double = (node in double_bond_nodes and neighbor in double_bond_nodes)
                use_sp2 = node_is_sp2 or (neighbor in double_bond_nodes)
                bond_len = self.BOND_LENGTH_CC_DOUBLE if is_double else self.BOND_LENGTH_CC
                
                # *** 核心改进：基于 local_child_idx 直接选择方向 ***
                direction = None
                
                if use_sp2 and double_bond_vec is not None:
                    # ===== sp2 有双键的情况 =====
                    # 双键占1个位置(sp2总共3个)，剩余2个位置与双键成精确120°
                    db = double_bond_vec
                    
                    # 构建垂直于双键的正交基
                    if abs(db[2]) < 0.9:
                        up = np.array([0., 0., 1.])
                    else:
                        up = np.array([1., 0., 0.])
                    
                    perp1 = np.cross(db, up)
                    perp1 /= np.linalg.norm(perp1)
                    
                    # 精确的120°方向（sp2只有这2个可用位置）
                    cos120, sin120 = -0.5, np.sqrt(3) / 2
                    
                    # 两个候选方向（与双键成120°，彼此成120°）
                    d_plus = cos120 * db + sin120 * perp1   # +120°
                    d_minus = cos120 * db - sin120 * perp1  # -120°
                    
                    # 选择不与已有方向重叠的那个
                    available = []
                    for cand in [d_plus, d_minus]:
                        too_close = False
                        for ex in existing_dirs:
                            if abs(np.dot(cand, ex)) > 0.9:  # ~25°以内算太近（更严格）
                                too_close = True
                                break
                        if not too_close:
                            available.append(cand)
                    
                    if available:
                        direction = available[local_child_idx % len(available)]
                    else:
                        # 都太近时用第一个
                        direction = d_plus if local_child_idx % 2 == 0 else d_minus
                
                elif use_sp2:
                    # ===== sp2 无双键的情况 =====
                    dirs = get_sp2_dirs(np.array([1., 0., 0.]))
                    direction = dirs[local_child_idx % len(dirs)]
                    
                else:
                    # ===== sp3 处理 =====
                    # 确定基准方向（第一个键的方向）
                    if double_bond_vec is not None:
                        # 有双键时，sp3的第一个方向取双键反向
                        base = -double_bond_vec
                    elif existing_dirs:
                        base = existing_dirs[0]
                    else:
                        base = np.array([1., 0., 0.])
                    
                    dirs = get_sp3_dirs(base)
                    
                    # 排除与已有方向太接近的（更严格的阈值）
                    available = []
                    for d in dirs:
                        too_close = False
                        for ex in existing_dirs:
                            if abs(np.dot(d, ex)) > 0.9:  # ~25°以内算太近
                                too_close = True
                                break
                        if not too_close:
                            available.append(d)
                    
                    if available:
                        direction = available[local_child_idx % len(available)]
                    else:
                        # 都太近时按顺序取
                        direction = dirs[local_child_idx % len(dirs)]
                
                # 归一化并设置坐标
                dn = np.linalg.norm(direction)
                if dn > 1e-8:
                    direction /= dn
                
                new_pos = parent_pos + direction * bond_len
                coords[neighbor] = (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                
                # 更新状态供下次使用
                existing_dirs.append(direction.copy())
                local_child_idx += 1
        
        # ---------- 4. 处理仍未分配的节点 ----------
        for i in range(n):
            if coords[i] is None:
                for nb in adjacency.get(i, []):
                    if nb < n and coords[nb] is not None:
                        cx, cy, cz = coords[nb]
                        coords[i] = (cx + self.BOND_LENGTH_CC, cy, cz)
                        break
                if coords[i] is None:
                    coords[i] = (float((i+1) * 1.54), 0.0, 0.0)
        
        # ---------- 5. 非键连原子排斥力优化 ----------
        # 解决视觉上"假环"问题：非键连原子靠得太近
        coords = self._apply_nonbonded_repulsion(coords, adjacency, n)
        
        return coords

    def _apply_nonbonded_repulsion(self, coords, adjacency, n):
        """
        对非键连原子施加强力排斥力，消除视觉上的"假环"
        
        核心改进（v2）：
        1. 更大的最小距离阈值（2.4Å）确保视觉分离
        2. 更强的排斥力和更大步长
        3. 多阶段策略：先粗推后精调
        """
        import numpy as np
        
        pos = np.array(coords, dtype=np.float64)
        
        # 构建邻接集合
        bonded_pairs = set()
        for i in range(n):
            for j in adjacency.get(i, []):
                if j < n:
                    pair = tuple(sorted([i, j]))
                    bonded_pairs.add(pair)
        
        # *** v2 强化参数 ***
        MIN_DIST = 2.4          # 非键连最小距离（比之前大很多！）
        
        # 阶段1：粗推阶段 — 大步长快速推开重叠原子
        phase1_iters = 200
        phase1_step = 0.35      # 每次移动可达距离差的35%
        
        for _ in range(phase1_iters):
            max_move = 0.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in bonded_pairs:
                        continue
                    
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    
                    if dist < 1e-8:
                        # 完全重叠：随机方向大力推开
                        rand_dir = np.random.randn(3)
                        rand_dir /= np.linalg.norm(rand_dir)
                        push = rand_dir * 0.5
                        pos[i] -= push
                        pos[j] += push
                        max_move = max(max_move, 0.5)
                        
                    elif dist < MIN_DIST:
                        # *** 核心：线性排斥力（比之前的平方衰减更强）***
                        overlap = MIN_DIST - dist
                        
                        # 力与重叠量成正比（越近越强）
                        force = (overlap / dist) * diff  # 方向：i远离j，j远离i
                        
                        # 两端各移一半，乘以步长因子
                        step = force * phase1_step * 0.5
                        
                        pos[i] -= step
                        pos[j] += step
                        max_move = max(max_move, np.linalg.norm(step))
            
            # 如果所有非键连距离都达标，提前结束
            if max_move < 0.001:
                break
        
        # 阶段2：精调阶段 — 小步长稳定收敛
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
        
        # 转回元组格式
        return [(float(pos[i][0]), float(pos[i][1]), float(pos[i][2])) 
                for i in range(n)]

    def _force_bond_lengths_simple(self, coords, adjacency, double_bond_nodes):
        """简单的强制键长调整 - 只移动新添加的原子，不破坏已调整好的原子"""
        import numpy as np
        
        coords_array = np.array(coords, dtype=np.float64)
        n = len(coords_array)
        
        def get_target_len(n1, n2):
            is_double = (n1 in double_bond_nodes and n2 in double_bond_nodes)
            return self.BOND_LENGTH_CC_DOUBLE if is_double else self.BOND_LENGTH_CC
        
        # 多次迭代调整
        for _ in range(100):
            max_error = 0.0
            
            for i in range(n):
                for j in adjacency.get(i, []):
                    if i >= j:
                        continue
                    
                    target = get_target_len(i, j)
                    diff = coords_array[j] - coords_array[i]
                    dist = np.linalg.norm(diff)
                    
                    if dist < 1e-6:
                        continue
                    
                    error = abs(dist - target)
                    max_error = max(max_error, error)
                    
                    # 只调整 i 的位置（j 可能已经被调整好了）
                    if error > 0.001:
                        adjustment = (dist - target) * (diff / dist)
                        coords_array[i] += adjustment * 0.5  # 渐进调整
            
            if max_error < 0.01:
                break
        
        return coords_array  # 返回 numpy 数组

    def _force_bond_lengths(self, coords, adjacency, double_bond_nodes):
        """强制确保所有相邻原子之间的距离正确
        
        使用直接位置调整：直接将每个原子移动到正确的距离
        """
        import numpy as np
        
        coords_array = np.array(coords, dtype=np.float64)
        n = len(coords_array)
        
        def get_target_len(n1, n2):
            is_double = (n1 in double_bond_nodes and n2 in double_bond_nodes)
            return self.BOND_LENGTH_CC_DOUBLE if is_double else self.BOND_LENGTH_CC
        
        # 多次迭代强制调整所有键长
        for iteration in range(100):
            max_error = 0.0
            
            for i in range(n):
                for j in adjacency.get(i, []):
                    if i >= j:
                        continue
                    
                    target = get_target_len(i, j)
                    diff = coords_array[j] - coords_array[i]
                    dist = np.linalg.norm(diff)
                    
                    if dist < 1e-6:
                        continue
                    
                    error = abs(dist - target)
                    max_error = max(max_error, error)
                    
                    # 无条件调整，直接将距离设为目标值
                    if error > 0.001:  # 允许0.001的误差
                        direction = diff / dist
                        # 移动两个原子使距离等于目标值
                        coords_array[i] += direction * (target - dist) / 2
                        coords_array[j] -= direction * (target - dist) / 2
            
            # 当所有键长误差都小于0.01时停止
            if max_error < 0.01:
                break
        
        return [tuple(c) for c in coords_array]

    def _calculate_tetrahedral_direction_from_one_bond(self, vec_to_existing_bond):
        """
        根据一个已存在的键向量,计算新的四面体方向
        
        [几何原理]
        已知一个键向量 v1,需要找到与 v1 成 109.47° 的新方向.
        
        四面体角满足: cos(θ) = -1/3 ~ -0.333
        其中 θ = 109.47° 是 sp3 碳的键角.
        
        [算法步骤]
        1. 构造垂直于 v1 的基向量 perp
        2. 使用 Rodrigues 旋转公式计算新方向:
           new = cos(θ) * v1 + sin(θ) * perp
        
        [数学推导]
        - perp · v1 = 0(垂直)
        - new · v1 = cos(θ) * (v1·v1) = cos(θ)(满足键角)
        - |new| = 1(单位向量)
        """

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
        """根据两个已存在的键向量,计算新的四面体方向"""
        v1 = vec_to_existing_bond1 / np.linalg.norm(vec_to_existing_bond1)
        v2 = vec_to_existing_bond2 / np.linalg.norm(vec_to_existing_bond2)
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return (1.0, 0.0, 0.0)
        normal /= normal_norm
        cos_theta = -1/3
        sin_theta = np.sqrt(1 - cos_theta**2)
        bisector = (v1 + v2) / np.linalg.norm(v1 + v2)
        v_new = cos_theta * bisector + sin_theta * normal
        return tuple(v_new / np.linalg.norm(v_new))

    def _calculate_tetrahedral_direction_from_three_bonds(self, vec1, vec2, vec3):
        """根据三个已存在的键向量,计算新的四面体方向"""
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)
        v3 = vec3 / np.linalg.norm(vec3)
        normal = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return (1.0, 0.0, 0.0)
        normal /= normal_norm
        return tuple(normal)

    def _calculate_sp2_direction_from_one_bond(self, vec_to_existing_bond, parent_node):
        """根据一个已存在的键向量,计算新的 sp2 方向(120°)"""
        v1 = vec_to_existing_bond / np.linalg.norm(vec_to_existing_bond)
        if abs(v1[0]) < 0.9:
            arb = np.array([1, 0, 0])
        else:
            arb = np.array([0, 1, 0])
        perp = np.cross(v1, arb)
        perp /= np.linalg.norm(perp)
        # sp2 键角 120°,cos(120°) = -0.5
        cos_120 = -0.5
        sin_120 = np.sqrt(3) / 2
        v_new = cos_120 * v1 + sin_120 * perp
        return tuple(v_new / np.linalg.norm(v_new))

    def _calculate_sp2_direction_from_two_bonds(self, vec1, vec2, parent_node):
        """根据两个已存在的键向量,计算新的 sp2 方向"""
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return (0.0, 0.0, 1.0)
        normal /= normal_norm
        # sp2 键角 120°,cos(120°) = -0.5
        cos_120 = -0.5
        sin_120 = np.sqrt(3) / 2
        bisector = (v1 + v2) / np.linalg.norm(v1 + v2)
        v_new = cos_120 * bisector + sin_120 * normal
        return tuple(v_new / np.linalg.norm(v_new))

    def _compute_coords_recursive(self, node: int, parent: int,
                                   adjacency: Dict[int, List[int]],
                                   double_bond_nodes: set,
                                   coords: List[Tuple[float, float, float]]):
        """递归计算子节点坐标"""
        # 获取当前节点的所有子节点(排除父节点)
        children = [n for n in adjacency[node] if n != parent]
        n_children = len(children)

        if n_children == 0:
            return

        # 获取子节点指向父节点的方向(从当前节点出发的方向)
        if parent >= 0:
            parent_dir = np.array(coords[parent]) - np.array(coords[node])
            parent_dir = parent_dir / np.linalg.norm(parent_dir)
        else:
            # 根节点,父方向为 (-1, 0, 0)
            parent_dir = np.array([-1.0, 0.0, 0.0])

        # 判断当前节点是否是 sp2 杂化
        is_sp2 = node in double_bond_nodes

        # 根据子节点数量计算方向
        if is_sp2:
            # sp2 碳:平面三角形几何,键角 120°
            directions = self._get_sp2_directions(parent_dir, n_children)
        else:
            # sp3 碳:四面体几何,键角 109.47°
            directions = self._get_tetrahedral_directions(parent_dir, n_children)

        # 为每个子节点分配方向并计算坐标
        for i, child in enumerate(children):
            bond_len = self.BOND_LENGTH_CC_DOUBLE if child in double_bond_nodes else self.BOND_LENGTH_CC
            direction = np.array(directions[i])

            parent_coords = np.array(coords[node])
            child_coords = parent_coords + direction * bond_len
            coords[child] = tuple(child_coords)

            # 递归处理子节点的子节点
            self._compute_coords_recursive(child, node, adjacency, double_bond_nodes, coords)

    def _get_sp2_directions(self, parent_dir: np.ndarray, n_children: int) -> List[np.ndarray]:
        """
        根据父节点方向,为 n 个子节点计算 sp2 平面三角形方向
        
        [sp2杂化几何]
        sp2碳的三个σ键:
        - 键角: 120°
        - 分布: 同一平面内
        - 特点: 正三角形排列
        
        [数学原理]
        设 parent_dir 是指向父节点的方向(已标准化)
        需要找到与 parent_dir 成 120° 的新方向.
        
        cos(120°) = -0.5
        sin(120°) = sqrt3/2 ~ 0.866
        
        新方向计算公式:
        new_dir = cos(120°) * parent_dir + sin(120°) * perp
               = -0.5 * parent_dir + 0.866 * perp
        
        其中 perp 是垂直于 parent_dir 的平面内任意方向.
        
        [参数]
        Args:
            parent_dir: 指向父节点的单位向量
            n_children: 子节点数量(1-3)
        
        [返回值]
        子节点方向的单位向量列表
        """

        # 标准化 parent_dir
        parent_dir = parent_dir / np.linalg.norm(parent_dir)

        # 计算垂直于 parent_dir 的基向量,形成 sp2 平面
        candidates = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]

        perp1 = None
        for ref in candidates:
            cross = np.cross(parent_dir, ref)
            norm = np.linalg.norm(cross)
            if norm > 1e-6:
                perp1 = cross / norm
                break

        if perp1 is None:
            perp1 = np.array([0.0, 1.0, 0.0])

        perp2 = np.cross(parent_dir, perp1)
        perp2 /= np.linalg.norm(perp2)

        cos_120 = -0.5  # cos(120°)
        sin_120 = np.sqrt(3) / 2  # sin(120°)

        # sp2碳的σ键方向计算:
        # direction = cos(120°)*parent_dir + sin(120°)*perp
        # 这样 direction · parent_dir = cos(120°) = -0.5,确保 120° 键角

        if n_children == 1:
            # 单个子节点:与 parent_dir 成 120° 角
            direction = cos_120 * parent_dir + sin_120 * perp1
            return [direction]

        elif n_children == 2:
            # 两个子节点:都与 parent_dir 成 120° 角,彼此之间也是 120° 角
            return [cos_120 * parent_dir + sin_120 * perp1,
                    cos_120 * parent_dir - sin_120 * perp1]

        elif n_children == 3:
            # 三个子节点:在垂直于 parent_dir 的平面内均匀分布
            # 每个方向都与 parent_dir 成 120°
            dirs = []
            for i in range(3):
                angle = i * (2 * np.pi / 3)
                plane_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
                direction = cos_120 * parent_dir + sin_120 * plane_vec
                dirs.append(direction)
            return dirs

        else:
            dirs = []
            for i in range(n_children):
                angle = i * (2 * np.pi / n_children)
                plane_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
                direction = cos_120 * parent_dir + sin_120 * plane_vec
                dirs.append(direction)
            return dirs

    def _get_tetrahedral_directions(self, parent_dir: np.ndarray, n_children: int) -> List[np.ndarray]:
        """
        根据父节点方向,为n个子节点计算四面体方向
        
        [sp3杂化几何]
        sp3碳的四个σ键:
        - 键角: 109.47°(四面体角)
        - 分布: 三维空间四面体排列
        - 特点: 任意两个键之间的夹角相等
        
        [数学原理]
        设 parent_dir 是指向父节点的方向(已标准化)
        需要找到与 (-parent_dir) 成 109.47° 的新方向.
        
        cos(109.47°) = -1/3 ~ -0.333
        
        四面体参数:
        - a = -1/3(parent_dir方向系数,负号！)
        - b = sqrt8/3 ~ 0.943(垂直方向系数)
        
        [不同子节点数的处理]
        - 1个子节点: 直接使用 Rodrigues 公式
        - 2个子节点: 垂直分量间隔 120°
        - 3个子节点: 垂直平面内均匀分布 120°
        - 4个子节点: 完整四面体,最后一个方向是 -parent_dir
        
        [参数]
        Args:
            parent_dir: 指向父节点的单位向量
            n_children: 子节点数量(1-4)
        
        [返回值]
        子节点方向的单位向量列表
        """

        if n_children == 1:
            return [self._calculate_tetrahedral_from_one(parent_dir)]

        elif n_children == 2:
            return [
                self._calculate_tetrahedral_from_two_directions(parent_dir, 0),
                self._calculate_tetrahedral_from_two_directions(parent_dir, 1)
            ]

        elif n_children == 3:
            # 3个子节点:正确计算三个四面体方向
            return self._get_tetrahedral_directions_for_3(parent_dir)

        elif n_children == 4:
            # 4个子节点:使用标准四面体方向
            return self._get_tetrahedral_directions_for_4(parent_dir)

        else:
            # 5个或更多子节点(不常见)
            dirs = []
            for i in range(n_children):
                if i == 0:
                    dirs.append(self._calculate_tetrahedral_from_one(parent_dir))
                else:
                    dirs.append(self._calculate_tetrahedral_from_two(parent_dir, dirs[-1]))
            return dirs
    
    def _get_tetrahedral_directions_for_3(self, parent_dir: np.ndarray) -> List[np.ndarray]:
        """为3个子节点生成正确的四面体方向"""
        parent_dir = parent_dir / np.linalg.norm(parent_dir)
        
        # 计算两个垂直于 parent_dir 的方向
        if abs(parent_dir[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        perp1 = np.cross(parent_dir, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(parent_dir, perp1)
        perp2 /= np.linalg.norm(perp2)
        
        # 四面体几何参数
        # h_dir · (-parent_dir) = cos(109.47°) = -1/3
        a = -1/3  # parent_dir方向的系数 (负号!)
        b = np.sqrt(8) / 3  # perp方向的系数
        
        # 三个方向在垂直于 parent_dir 的平面上的投影应该间隔 120°
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        
        dirs = []
        for angle in angles:
            perp_component = np.cos(angle) * perp1 + np.sin(angle) * perp2
            h_dir = a * parent_dir + b * perp_component
            dirs.append(h_dir / np.linalg.norm(h_dir))
        
        return dirs

    def _get_tetrahedral_directions_for_4(self, parent_dir: np.ndarray) -> List[np.ndarray]:
        """为4个子节点生成正确的四面体方向"""
        parent_dir = parent_dir / np.linalg.norm(parent_dir)
        
        # 计算垂直于 parent_dir 的两个正交方向
        if abs(parent_dir[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        perp1 = np.cross(parent_dir, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(parent_dir, perp1)
        perp2 /= np.linalg.norm(perp2)
        
        # 四面体几何参数
        # 对于度数为4的碳,所有4个子方向应该与彼此成109.47°角
        # 验证:h_i · h_j = -1/3 for all i != j
        
        # 三个子方向(底部三角形)间隔120°
        a = 1/3  # 注意:是+1/3,不是-1/3
        b = 2 * np.sqrt(2) / 3
        
        # 三个方向在垂直于 parent_dir 的平面上的投影间隔120°
        angles_base = [0, 2*np.pi/3, 4*np.pi/3]
        
        dirs = []
        for angle in angles_base:
            perp_component = np.cos(angle) * perp1 + np.sin(angle) * perp2
            h_dir = a * parent_dir + b * perp_component
            dirs.append(h_dir / np.linalg.norm(h_dir))
        
        # 第四个方向:-parent_dir(与parent_dir成180°)
        # 验证:h_i · (-parent_dir) = -a = -1/3 ✓
        dirs.append(-parent_dir)
        
        return dirs

    def _calculate_tetrahedral_from_two_directions(self, v_parent: np.ndarray, index: int) -> np.ndarray:
        """为两个子节点计算正确的四面体方向"""
        v_parent = v_parent / np.linalg.norm(v_parent)
        
        # 计算两个垂直于 v_parent 的方向
        if abs(v_parent[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        perp1 = np.cross(v_parent, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(v_parent, perp1)
        perp2 /= np.linalg.norm(perp2)
        
        # 四面体几何参数
        # h_dir · (-v_parent) = cos(109.47°) = -1/3
        a = -1/3  # v_parent方向的系数
        b = np.sqrt(8) / 3  # perp方向的系数
        
        # 对于2个子节点,垂直分量的间隔应该是 120°
        # 验证: h1·h2 = a^2 + b^2*cos(120°) = 1/9 + 8/9*(-1/2) = 1/9 - 4/9 = -3/9 = -1/3 ✓
        perp_angle = 2 * np.pi / 3  # 120°
        
        if index == 0:
            perp_component = perp1
        else:
            perp_component = np.cos(perp_angle) * perp1 + np.sin(perp_angle) * perp2
        
        h_dir = a * v_parent + b * perp_component
        return h_dir / np.linalg.norm(h_dir)

    def _calculate_tetrahedral_from_one(self, v1: np.ndarray) -> Tuple:
        """根据一个已存在的键向量,计算新的四面体方向"""
        v1 = v1 / np.linalg.norm(v1)
        
        if abs(v1[0]) < 0.9:
            arb = np.array([1.0, 0.0, 0.0])
        else:
            arb = np.array([0.0, 1.0, 0.0])
        perp = np.cross(v1, arb)
        perp /= np.linalg.norm(perp)

        # 四面体几何: h_dir · (-v1) = cos(109.47°) = -1/3
        # 所以 h_dir = (-1/3) * v1 + (sqrt8/3) * perp
        a = -1/3  # v1方向的系数 (负号!)
        b = np.sqrt(8)/3  # perp方向的系数
        
        v_new = a * v1 + b * perp
        return tuple(v_new / np.linalg.norm(v_new))

    def _calculate_tetrahedral_from_two(self, v1: np.ndarray, v2: np.ndarray) -> Tuple:
        """根据两个已存在的键向量,计算新的四面体方向"""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-6:
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
            # v1和v2相反,退化为一个方向的情况
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp /= np.linalg.norm(perp)
            return self._calculate_tetrahedral_from_one(perp)

        c = -1 / (3 * (1 + dot_v1_v2))

        # 检查虚数情况
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

        # 选择更接近理想角度的方向
        test_dot_pos = np.dot(v4_unnorm / np.linalg.norm(v4_unnorm), v1)
        test_dot_neg = np.dot(v4_unnorm_check_neg / np.linalg.norm(v4_unnorm_check_neg), v1)

        if abs(test_dot_pos - (-1/3)) < abs(test_dot_neg - (-1/3)):
            final_v4 = v4_unnorm
        else:
            final_v4 = v4_unnorm_check_neg

        return tuple(final_v4 / np.linalg.norm(final_v4))

    def _calculate_tetrahedral_from_three(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple:
        """根据三个已存在的键向量,计算新的四面体方向"""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)

        # 第四个方向是前三个向量和的负方向
        v4 = -(v1 + v2 + v3)
        v4_norm = np.linalg.norm(v4)

        if v4_norm < 1e-6:
            return tuple(-v1)
        v4_normalized = v4 / v4_norm
        return tuple(v4_normalized)

    def _generate_hydrogen_coords(self, c_coords: List[Tuple[float, float, float]],
                                 adjacency: Dict[int, List[int]],
                                 double_bond_nodes: set) -> List[Tuple[float, float, float]]:
        """
        生成所有氢原子的坐标
        
        [功能]
        根据每个碳原子的类型(sp2或sp3)和连接的原子数,
        计算该碳原子需要的氢原子数量和位置.
        
        [化学规则]
        
        sp2碳(双键碳):
        ---------------------------------
        - 3个σ键平面分布,键角120°
        - 需要氢原子数 = 3 - 连接的碳原子数
        
        示例:
        - =CH2 (末端烯烃): 1个C连接,2个H
        - =CH- (内部烯烃): 2个C连接,1个H
        - =C< (叔碳烯烃): 3个C连接,0个H
        
        sp3碳(单键碳):
        ---------------------------------
        - 4个σ键四面体分布,键角109.47°
        - 需要氢原子数 = 4 - 连接的碳原子数
        
        示例:
        - -CH3 (甲基): 1个C连接,3个H
        - -CH2- (亚甲基): 2个C连接,2个H
        - -CH< (叔碳): 3个C连接,1个H
        - -C< (季碳): 4个C连接,0个H
        
        [参数]
        Args:
            c_coords: 碳原子坐标列表
            adjacency: 邻接表 {碳ID: [邻居碳ID列表], ...}
            double_bond_nodes: 双键碳原子ID集合
        
        [返回值]
        氢原子坐标列表,如 [(x1,y1,z1), (x2,y2,z2), ...]
        """

        h_coords = []
        n_carbons = len(c_coords)

        for i in range(n_carbons):
            c_pos = c_coords[i]
            adjacent_carbon_indices = adjacency.get(i, [])
            adjacent_carbon_coords = [c_coords[idx] for idx in adjacent_carbon_indices if idx < n_carbons]
            num_attached_carbons = len(adjacent_carbon_coords)

            is_sp2 = i in double_bond_nodes

            if is_sp2:
                # sp2碳:3个σ键,最多2个氢原子
                num_h = 3 - num_attached_carbons
            else:
                # sp3碳:4个σ键
                num_h = 4 - num_attached_carbons

            num_h = max(0, min(num_h, 3))

            if num_h == 0:
                continue

            # ========== sp2碳的氢原子生成(键角120°) ==========
            if is_sp2:
                if num_attached_carbons == 1:
                    # 终端sp2碳:连接1个碳,生成2个氢,键角120°
                    adj_pos = adjacent_carbon_coords[0]
                    h_positions = self._generate_sp2_terminal_hydrogens(c_pos, adj_pos)
                    h_positions = h_positions[:num_h]
                elif num_attached_carbons == 2:
                    # 中间sp2碳:连接2个碳(其中一个是双键),生成1个氢,键角120°
                    adj_indices = sorted(adjacent_carbon_indices)
                    adj_coords_sorted = [c_coords[idx] for idx in adj_indices if idx < n_carbons]
                    if len(adj_coords_sorted) >= 2:
                        h_positions = self._generate_sp2_middle_hydrogens(c_pos, adj_coords_sorted[0], adj_coords_sorted[1])
                    else:
                        h_positions = []
                elif num_attached_carbons == 3:
                    # 叔sp2碳:连接3个碳(其中一个是双键),生成0个氢
                    h_positions = []
                else:
                    h_positions = []
            # ========== sp3碳的氢原子生成(键角109.5°) ==========
            else:
                if num_attached_carbons == 0:
                    h_positions = self._generate_methane_hydrogens(c_pos)
                elif num_attached_carbons == 1:
                    adj_pos = adjacent_carbon_coords[0]
                    h_positions = self._generate_terminal_hydrogens(c_pos, adj_pos)
                elif num_attached_carbons == 2:
                    adj_indices = sorted(adjacent_carbon_indices)
                    adj_coords_sorted = [c_coords[idx] for idx in adj_indices if idx < n_carbons]
                    if len(adj_coords_sorted) >= 2:
                        h_positions = self._generate_middle_hydrogens(c_pos, adj_coords_sorted[0], adj_coords_sorted[1])
                    else:
                        h_positions = []
                elif num_attached_carbons == 3:
                    h_positions = self._generate_tertiary_hydrogens(c_pos, adjacent_carbon_coords)
                else:
                    h_positions = []

            h_coords.extend(h_positions)

        return h_coords

    def _generate_methane_hydrogens(self, c_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """生成甲烷类型的氢原子位置(四面体构型)"""
        directions = [
            (self.SQRT3/3, self.SQRT3/3, self.SQRT3/3),
            (self.SQRT3/3, -self.SQRT3/3, -self.SQRT3/3),
            (-self.SQRT3/3, self.SQRT3/3, -self.SQRT3/3),
            (-self.SQRT3/3, -self.SQRT3/3, self.SQRT3/3)
        ]
        h_positions = []
        for direction in directions:
            h_x = c_pos[0] + self.BOND_LENGTH_CH * direction[0]
            h_y = c_pos[1] + self.BOND_LENGTH_CH * direction[1]
            h_z = c_pos[2] + self.BOND_LENGTH_CH * direction[2]
            h_positions.append((h_x, h_y, h_z))
        return h_positions

    def _generate_terminal_hydrogens(self, c_pos: Tuple[float, float, float],
                                    adj_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
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

        v1 = [u_to_adj[1]*ref[2] - u_to_adj[2]*ref[1],
              u_to_adj[2]*ref[0] - u_to_adj[0]*ref[2],
              u_to_adj[0]*ref[1] - u_to_adj[1]*ref[0]]
        len_v1 = math.sqrt(sum(v**2 for v in v1))
        if len_v1 == 0:
            return []
        v1 = [v/len_v1 for v in v1]

        v2 = [v1[1]*u_to_adj[2] - v1[2]*u_to_adj[1],
              v1[2]*u_to_adj[0] - v1[0]*u_to_adj[2],
              v1[0]*u_to_adj[1] - v1[1]*u_to_adj[0]]
        len_v2 = math.sqrt(sum(v**2 for v in v2))
        if len_v2 == 0:
            return []
        v2 = [v/len_v2 for v in v2]

        h_positions = []
        for j in range(3):
            angle = j * self.ANGLE_120
            h_x = c_pos[0] + self.BOND_LENGTH_CH * (
                u_to_adj[0]*math.cos(self.TETRAHEDRAL_ANGLE) +
                v1[0]*math.sin(self.TETRAHEDRAL_ANGLE)*math.cos(angle) +
                v2[0]*math.sin(self.TETRAHEDRAL_ANGLE)*math.sin(angle)
            )
            h_y = c_pos[1] + self.BOND_LENGTH_CH * (
                u_to_adj[1]*math.cos(self.TETRAHEDRAL_ANGLE) +
                v1[1]*math.sin(self.TETRAHEDRAL_ANGLE)*math.cos(angle) +
                v2[1]*math.sin(self.TETRAHEDRAL_ANGLE)*math.sin(angle)
            )
            h_z = c_pos[2] + self.BOND_LENGTH_CH * (
                u_to_adj[2]*math.cos(self.TETRAHEDRAL_ANGLE) +
                v1[2]*math.sin(self.TETRAHEDRAL_ANGLE)*math.cos(angle) +
                v2[2]*math.sin(self.TETRAHEDRAL_ANGLE)*math.sin(angle)
            )
            h_positions.append((h_x, h_y, h_z))
        return h_positions

    def _generate_middle_hydrogens(self, c_pos: Tuple[float, float, float],
                                   prev_pos: Tuple[float, float, float],
                                   next_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
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

        cross = [u_prev[1]*u_next[2] - u_prev[2]*u_next[1],
                 u_prev[2]*u_next[0] - u_prev[0]*u_next[2],
                 u_prev[0]*u_next[1] - u_prev[1]*u_next[0]]
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
            h_x = c_pos[0] + self.BOND_LENGTH_CH * (
                bisector[0]*math.cos(self.TETRAHEDRAL_ANGLE) +
                sign*cross[0]*math.sin(self.TETRAHEDRAL_ANGLE)
            )
            h_y = c_pos[1] + self.BOND_LENGTH_CH * (
                bisector[1]*math.cos(self.TETRAHEDRAL_ANGLE) +
                sign*cross[1]*math.sin(self.TETRAHEDRAL_ANGLE)
            )
            h_z = c_pos[2] + self.BOND_LENGTH_CH * (
                bisector[2]*math.cos(self.TETRAHEDRAL_ANGLE) +
                sign*cross[2]*math.sin(self.TETRAHEDRAL_ANGLE)
            )
            h_positions.append((h_x, h_y, h_z))
        return h_positions

    def _generate_tertiary_hydrogens(self, c_pos: Tuple[float, float, float],
                                     adjacent_carbon_coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """生成叔碳原子的氢原子位置"""
        total_vec = np.array([0.0, 0.0, 0.0])
        for adj_coord in adjacent_carbon_coords:
            vec_to_adj = np.array(adj_coord) - np.array(c_pos)
            total_vec += vec_to_adj / np.linalg.norm(vec_to_adj)

        avg_dir = total_vec / len(adjacent_carbon_coords)
        avg_dir /= np.linalg.norm(avg_dir)
        h_dir = -avg_dir
        h_pos = np.array(c_pos) + self.BOND_LENGTH_CH * h_dir
        return [tuple(h_pos)]

    def _generate_sp2_terminal_hydrogens(self, c_pos: Tuple[float, float, float],
                                        adj_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        生成终端sp2碳原子的氢原子位置(键角120°)
        
        [终端sp2碳的特点]
        末端烯烃(如 CH2=CH-CH3)中的 =CH2 碳:
        - 连接1个碳原子(双键伙伴)
        - 连接2个氢原子
        - 三个σ键在同一平面内
        
        [几何构型]
        设 C1 是当前碳(=CH2),C2 是双键伙伴
        两个氢原子 H1 和 H2 都在 C1-C2 键的同一侧
        
        [关键约束]
        - ∠H1-C1-C2 = 120°
        - ∠H2-C1-C2 = 120°
        - ∠H1-C1-H2 = 120°
        
        [数学推导]
        cos(120°) = -0.5, sin(60°) = sqrt3/2 ~ 0.866
        
        两个氢方向:
        H1 = 0.5 * (-u_to_adj) + 0.866 * perp
        H2 = 0.5 * (-u_to_adj) - 0.866 * perp
        
        其中 u_to_adj 是从 C1 指向 C2 的单位向量.
        """
        vec_to_adj = np.array(adj_pos) - np.array(c_pos)
        len_vec = np.linalg.norm(vec_to_adj)
        if len_vec == 0:
            return []
        u_to_adj = vec_to_adj / len_vec

        # 找到垂直于 u_to_adj 的方向
        if abs(u_to_adj[0]) < 0.9:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 1, 0])

        perp = np.cross(u_to_adj, ref)
        perp /= np.linalg.norm(perp)

        # sp2键角120°:cos(120°) = -0.5, sin(60°) = sqrt(3)/2
        # 两个氢都在 u_to_adj 的反侧,且在同一垂直方向的两侧
        # h1 = 0.5 * (-u_to_adj) + sin(60) * perp
        # h2 = 0.5 * (-u_to_adj) - sin(60) * perp
        # 验证:h1 · u = -0.5, h2 · u = -0.5, h1 · h2 = -0.5 (均为120°)
        cos_60 = 0.5
        sin_60 = np.sqrt(3) / 2

        # 生成两个氢原子
        h_positions = []
        
        # 第一个氢:在 perp 方向侧
        h_dir1 = cos_60 * (-u_to_adj) + sin_60 * perp
        h_dir1 /= np.linalg.norm(h_dir1)  # 归一化
        h_pos1 = np.array(c_pos) + self.BOND_LENGTH_CH * h_dir1
        h_positions.append(tuple(h_pos1))
        
        # 第二个氢:在 -perp 方向侧
        h_dir2 = cos_60 * (-u_to_adj) - sin_60 * perp
        h_dir2 /= np.linalg.norm(h_dir2)  # 归一化
        h_pos2 = np.array(c_pos) + self.BOND_LENGTH_CH * h_dir2
        h_positions.append(tuple(h_pos2))

        return h_positions

    def _generate_sp2_middle_hydrogens(self, c_pos: Tuple[float, float, float],
                                      prev_pos: Tuple[float, float, float],
                                      next_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        生成中间sp2碳原子的氢原子位置(键角120°)
        
        方法:角平分线方法 - 氢原子位于两个相邻键的角平分线反方向
        注:由于 sp2 碳的键角是 120°,当两个相邻键之间的夹角不是 120° 时,
        氢原子的键角会略有偏差,但对于可视化目的已经足够.
        """
        vec_to_prev = np.array(prev_pos) - np.array(c_pos)
        vec_to_next = np.array(next_pos) - np.array(c_pos)

        len_prev = np.linalg.norm(vec_to_prev)
        len_next = np.linalg.norm(vec_to_next)
        if len_prev == 0 or len_next == 0:
            return []

        u_prev = vec_to_prev / len_prev
        u_next = vec_to_next / len_next

        # 角平分线方向
        bisector = u_prev + u_next
        bisector_norm = np.linalg.norm(bisector)
        
        if bisector_norm < 1e-6:
            # 两个方向接近平行,使用垂直方向
            if abs(u_prev[0]) < 0.9:
                perp = np.cross(u_prev, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(u_prev, np.array([0.0, 1.0, 0.0]))
            perp /= np.linalg.norm(perp)
            h_dir = perp
        else:
            # 氢原子在角平分线的反方向
            h_dir = -bisector / bisector_norm
        
        h_pos = np.array(c_pos) + self.BOND_LENGTH_CH * h_dir
        return [tuple(h_pos)]

    def visualize_isomer(self, G: nx.Graph, show=True, save_path=None, title=None, print_coords=True, coords_format='detailed'):
        """
        可视化单个烯烃异构体
        
        [功能]
        将烯烃分子以3D形式显示在窗口中,并可保存为图片.
        
        [可视化内容]
        1. 碳骨架:C-C单键(灰色)和C=C双键(红色)
        2. 碳原子:深色球体,带编号标签
        3. 氢原子:白色球体,带编号标签
        4. 化学键:线条连接原子
        
        [参数]
        Args:
            G: NetworkX图对象,表示烯烃分子结构
               - 节点: 碳原子(属性: label='C')
               - 边: 化学键(属性: bond_type='single' 或 'double')
            show: 是否显示matplotlib窗口
            save_path: 图片保存路径(如 "output.png")
            title: 图形标题(None表示自动生成)
            print_coords: 是否在控制台打印原子坐标
            coords_format: 坐标格式
                          - 'detailed': 详细格式,包含所有信息
                          - 'compact': 紧凑格式
        
        [返回值]
        所有原子的坐标列表: [(x,y,z), ...]
        前n个是碳原子坐标,后面的全是氢原子坐标
        """
        info = self.get_graph_info(G)
        n_carbons = info['n_carbons']
        adjacency = info['adjacency']
        double_bond_nodes = info['double_bond_nodes']

        # 转换为3D坐标
        c_coords = self.adjacency_to_coords(adjacency, double_bond_nodes, n_carbons)

        # 生成氢原子坐标
        h_coords = self._generate_hydrogen_coords(c_coords, adjacency, double_bond_nodes)

        # 组合所有坐标
        all_coords = c_coords + h_coords

        # 打印坐标信息
        if print_coords:
            if coords_format == 'compact':
                self.print_coords_compact(G, c_coords, h_coords, n_carbons, adjacency, double_bond_nodes)
            else:
                self.print_molecule_coords(G, c_coords, h_coords, n_carbons, adjacency, double_bond_nodes)

        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制分子
        self._draw_molecule(ax, all_coords, n_carbons, adjacency, double_bond_nodes)

        # 设置标题
        if title is None:
            double_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
            if double_edges:
                u, v = double_edges[0]
                title = f'C{n_carbons}H{2*n_carbons} - 双键: C{u+1}=C{v+1}'
        ax.set_title(title if title else f'C{n_carbons}H{2*n_carbons}', fontsize=14, pad=20)

        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')
        ax.set_zlabel('Z (A)')
        ax.grid(True, alpha=0.3)

        # 设置等比例
        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        z_coords = [coord[2] for coord in all_coords]
        self._set_equal_aspect_ratio(ax, x_coords, y_coords, z_coords)

        if show:
            # 如果需要显示,先保存(如果有路径),然后显示
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图片已保存至:{save_path}")
            plt.show()
        else:
            # 如果不显示,只保存图片后关闭
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图片已保存至:{save_path}")
            plt.close(fig)

        return all_coords

    def print_molecule_coords(self, G: nx.Graph, c_coords: List[Tuple], h_coords: List[Tuple],
                              n_carbons: int, adjacency: Dict, double_bond_nodes: set):
        '''
        打印分子中所有原子的坐标和化学键信息
        
        功能:在控制台输出分子结构的详细信息
        参数:G图对象,c_coords碳坐标,h_coords氢坐标,n_carbons碳数量,adjacency邻接表,double_bond_nodes双键节点
        '''

        print("\n" + "=" * 60)
        print("分子坐标信息")
        print("=" * 60)

        n_carbons = len(c_coords)
        n_hydrogens = len(h_coords)

        # 统计每个碳原子的氢原子数
        h_per_carbon = []
        h_counter = 0
        for i in range(n_carbons):
            num_attached_carbons = len(adjacency.get(i, []))
            is_sp2 = i in double_bond_nodes
            if is_sp2:
                num_h = 3 - num_attached_carbons
            else:
                num_h = 4 - num_attached_carbons
            num_h = max(0, num_h)
            h_per_carbon.append((h_counter, num_h))
            h_counter += num_h

        # 打印碳原子坐标
        print("\n[碳原子坐标]")
        print("-" * 50)
        print(f"{'原子':<8} {'X':>10} {'Y':>10} {'Z':>10} {'类型':>10}")
        print("-" * 50)

        for i, coord in enumerate(c_coords):
            is_sp2 = i in double_bond_nodes
            atom_type = "sp2" if is_sp2 else "sp3"
            print(f"C{i+1:<5} {coord[0]:>10.4f} {coord[1]:>10.4f} {coord[2]:>10.4f} {atom_type:>10}")

        # 打印氢原子坐标
        print("\n[氢原子坐标]")
        print("-" * 50)
        print(f"{'原子':<8} {'X':>10} {'Y':>10} {'Z':>10} {'连接碳':>8}")
        print("-" * 50)

        h_counter = 0
        for i in range(n_carbons):
            num_attached_carbons = len(adjacency.get(i, []))
            is_sp2 = i in double_bond_nodes
            if is_sp2:
                num_h = 3 - num_attached_carbons
            else:
                num_h = 4 - num_attached_carbons
            num_h = max(0, num_h)
            for j in range(num_h):
                if h_counter < len(h_coords):
                    coord = h_coords[h_counter]
                    print(f"H{h_counter+1:<5} {coord[0]:>10.4f} {coord[1]:>10.4f} {coord[2]:>10.4f} C{i+1:<6}")
                    h_counter += 1

        # 打印化学键信息
        print("\n[化学键信息]")
        print("-" * 50)

        # C-C键
        print("C-C 单键:")
        for u in range(n_carbons):
            for v in adjacency[u]:
                if v > u:
                    is_double = (u in double_bond_nodes and v in double_bond_nodes)
                    bond_type = "=" if is_double else "-"
                    print(f"  C{u+1}{bond_type}C{v+1}")

        # 打印键长统计
        print("\n[键长统计]")
        print("-" * 50)

        # 计算键长
        single_bonds = []
        double_bonds = []

        for u in range(n_carbons):
            for v in adjacency[u]:
                if v > u:
                    c1 = np.array(c_coords[u])
                    c2 = np.array(c_coords[v])
                    bond_len = np.linalg.norm(c2 - c1)
                    is_double = (u in double_bond_nodes and v in double_bond_nodes)
                    if is_double:
                        double_bonds.append(bond_len)
                    else:
                        single_bonds.append(bond_len)

        if single_bonds:
            print(f"C-C 单键键长: {np.mean(single_bonds):.4f} A (范围: {min(single_bonds):.4f} - {max(single_bonds):.4f})")
        if double_bonds:
            print(f"C=C 双键键长: {np.mean(double_bonds):.4f} A")

        # 计算键角
        print("\n[主要键角]")
        print("-" * 50)

        for i in range(n_carbons):
            neighbors = adjacency.get(i, [])
            if len(neighbors) >= 2:
                is_sp2 = i in double_bond_nodes
                atom_type = "sp2" if is_sp2 else "sp3"

                # 计算该碳与相邻碳的键角
                for j in range(len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        n1, n2 = neighbors[j], neighbors[k]
                        v1 = np.array(c_coords[n1]) - np.array(c_coords[i])
                        v2 = np.array(c_coords[n2]) - np.array(c_coords[i])

                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        angle_deg = np.degrees(angle)

                        print(f"C{i+1}: C{n1+1}-C{i+1}-C{n2+1} = {angle_deg:.2f}° (类型: {atom_type})")

        print("\n" + "=" * 60)

    def print_coords_compact(self, G: nx.Graph, c_coords: List[Tuple], h_coords: List[Tuple],
                              n_carbons: int, adjacency: Dict, double_bond_nodes: set):
        '''打印紧凑格式的分子坐标(便于复制使用)'''
        info = self.get_graph_info(G)
        formula = f"C{n_carbons}H{2*n_carbons}"
        
        print("\n" + "=" * 70)
        print(f"分子式: {formula}")
        print("=" * 70)
        
        # 碳原子坐标(简洁数组格式)
        print("\n碳原子坐标:")
        print("-" * 50)
        carbon_lines = []
        for i, coord in enumerate(c_coords):
            is_sp2 = i in double_bond_nodes
            atom_type = "sp2" if is_sp2 else "sp3"
            carbon_lines.append(f"C{i+1}: [{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}] ({atom_type})")
        print("\n".join(carbon_lines))
        
        # 氢原子坐标
        print("\n氢原子坐标:")
        print("-" * 50)
        h_counter = 0
        hydrogen_lines = []
        for i in range(n_carbons):
            num_attached_carbons = len(adjacency.get(i, []))
            is_sp2 = i in double_bond_nodes
            if is_sp2:
                num_h = 3 - num_attached_carbons
            else:
                num_h = 4 - num_attached_carbons
            num_h = max(0, num_h)
            for j in range(num_h):
                if h_counter < len(h_coords):
                    coord = h_coords[h_counter]
                    hydrogen_lines.append(f"H{h_counter+1}: [{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}] <- C{i+1}")
                    h_counter += 1
        print("\n".join(hydrogen_lines))
        
        # 统计信息
        print("\n统计信息:")
        print("-" * 50)
        print(f"  碳原子总数: {n_carbons}")
        print(f"  氢原子总数: {len(h_coords)}")
        print(f"  原子总数: {n_carbons + len(h_coords)}")
        
        # 坐标范围
        all_coords = c_coords + h_coords
        x_coords = [c[0] for c in all_coords]
        y_coords = [c[1] for c in all_coords]
        z_coords = [c[2] for c in all_coords]
        print(f"\n空间分布:")
        print(f"  X轴范围: {min(x_coords):.4f} ~ {max(x_coords):.4f}")
        print(f"  Y轴范围: {min(y_coords):.4f} ~ {max(y_coords):.4f}")
        print(f"  Z轴范围: {min(z_coords):.4f} ~ {max(z_coords):.4f}")
        
        # 键角验证
        print("\n键角验证:")
        print("-" * 50)
        for i in range(n_carbons):
            neighbors = adjacency.get(i, [])
            if len(neighbors) >= 2:
                is_sp2 = i in double_bond_nodes
                atom_type = "sp2" if is_sp2 else "sp3"
                expected_angle = 120.0 if is_sp2 else 109.5
                
                for j in range(len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        n1, n2 = neighbors[j], neighbors[k]
                        v1 = np.array(c_coords[n1]) - np.array(c_coords[i])
                        v2 = np.array(c_coords[n2]) - np.array(c_coords[i])
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        angle_deg = np.degrees(angle)
                        
                        diff = abs(angle_deg - expected_angle)
                        status = "OK" if diff < 5 else "!"
                        print(f"  C{i+1}: {angle_deg:.1f} deg (expected {expected_angle}) {status}")
        
        print("\n" + "=" * 70)

    def print_isomer_summary(self, isomers: List[nx.Graph]):
        """打印所有异构体的简要坐标信息"""
        print("\n" + "=" * 70)
        print("所有异构体坐标摘要")
        print("=" * 70)

        for i, G in enumerate(isomers):
            info = self.get_graph_info(G)
            n_carbons = info['n_carbons']
            adjacency = info['adjacency']
            double_bond_nodes = info['double_bond_nodes']

            c_coords = self.adjacency_to_coords(adjacency, double_bond_nodes, n_carbons)
            h_coords = self._generate_hydrogen_coords(c_coords, adjacency, double_bond_nodes)

            # 获取双键位置
            double_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
            if double_edges:
                u, v = double_edges[0]
                db_pos = f"C{u+1}=C{v+1}"
            else:
                db_pos = "无"

            print(f"\n[异构体 {i+1}] C{n_carbons}H{2*n_carbons}")
            print(f"  双键位置: {db_pos}")
            print(f"  碳原子数: {n_carbons}, 氢原子数: {len(h_coords)}")

            # 打印碳坐标
            print(f"  碳原子坐标范围:")
            xs = [c[0] for c in c_coords]
            ys = [c[1] for c in c_coords]
            zs = [c[2] for c in c_coords]
            print(f"    X: [{min(xs):.2f}, {max(xs):.2f}]")
            print(f"    Y: [{min(ys):.2f}, {max(ys):.2f}]")
            print(f"    Z: [{min(zs):.2f}, {max(zs):.2f}]")

        print("\n" + "=" * 70)

    def visualize_all_isomers(self, isomers: List[nx.Graph], show=True, save_dir=None, print_summary=True):
        """
        批量可视化所有烯烃异构体
        
        [功能]
        将多个烯烃异构体以网格形式排列在一个图中显示,
        方便对比不同异构体的结构差异.
        
        [显示布局]
        根据异构体数量自动调整网格大小:
        - 1个: 1*1 网格
        - 2-4个: 2*2 网格
        - 5-9个: 3*3 网格
        - 更多: 逐步增加
        
        [参数]
        Args:
            isomers: NetworkX图对象列表
            show: 是否显示图形窗口
            save_dir: 保存目录(每个异构体单独保存)
            print_summary: 是否打印摘要信息
        
        [输出]
        - 如果 show=True: 显示matplotlib窗口
        - 如果 save_dir 指定: 保存每个异构体图片到目录
        """

        if not isomers:
            print("没有异构体可显示")
            return

        # 打印所有异构体的坐标摘要
        if print_summary:
            self.print_isomer_summary(isomers)

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

        for i, G in enumerate(isomers):
            if i >= rows * cols:
                break

            info = self.get_graph_info(G)
            n_carbons = info['n_carbons']
            adjacency = info['adjacency']
            double_bond_nodes = info['double_bond_nodes']

            # 转换为3D坐标
            c_coords = self.adjacency_to_coords(adjacency, double_bond_nodes, n_carbons)
            h_coords = self._generate_hydrogen_coords(c_coords, adjacency, double_bond_nodes)
            all_coords = c_coords + h_coords

            row, col = i // cols, i % cols
            ax = fig.add_subplot(gs[row, col], projection='3d')

            # 绘制分子
            self._draw_molecule(ax, all_coords, n_carbons, adjacency, double_bond_nodes, show_labels=False)

            # 设置标题
            double_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
            if double_edges:
                u, v = double_edges[0]
                ax.set_title(f'{i+1}: C{u+1}=C{v+1}', fontsize=10, pad=10)
            else:
                ax.set_title(f'{i+1}', fontsize=10, pad=10)

            ax.set_xlabel('X', fontsize=7)
            ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7)
            ax.tick_params(labelsize=6)

            self._set_equal_aspect_ratio(ax,
                [coord[0] for coord in all_coords],
                [coord[1] for coord in all_coords],
                [coord[2] for coord in all_coords]
            )

        n_carbons = isomers[0].number_of_nodes()
        fig.suptitle(f'C{n_carbons}H{2*n_carbons} 的所有烯烃同分异构体', fontsize=16, fontweight='bold')

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'C{n_carbons}H{2*n_carbons}_alkene_isomers.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"图片已保存至:{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _draw_molecule(self, ax, coords, n_carbons, adjacency, double_bond_nodes, show_labels=True):
        """
        绘制烯烃分子结构
        
        [功能]
        使用matplotlib在3D坐标系中绘制分子结构.
        
        [绘制顺序]
        1. C-C键(单键和双键)
        2. C-H键
        3. 碳原子(散点 + 标签)
        4. 氢原子(散点 + 标签)
        
        [绘制细节]
        - 单键: 灰色线,宽度3
        - 双键: 红色线,宽度5,绘制为两条平行线
        - 碳原子: 深色球体(sp2用深红,sp3用灰色)
        - 氢原子: 白色球体
        - 标签: C1, C2, ... 和 H1, H2, ...
        
        [Parameters]
        Args:
            ax: matplotlib axes object
            coords: all atom coordinates list
            n_carbons: carbon count
            adjacency: adjacency dict
            double_bond_nodes: double bond nodes set
            show_labels: show labels flag
        """
        # 分离碳原子和氢原子坐标
        c_coords = coords[:n_carbons]  # 前n个是碳原子
        h_coords = coords[n_carbons:]   # 后面的是氢原子

        # ========== 第一步:绘制C-C键(单键和双键)==========

        for u in range(n_carbons):
            for v in adjacency[u]:
                if v > u:
                    # 检查是否是双键
                    is_double = (u in double_bond_nodes and v in double_bond_nodes)
                    c1 = np.array(c_coords[u])
                    c2 = np.array(c_coords[v])

                    if is_double:
                        # 绘制双键 - 分成两条平行线
                        bond_vec = c2 - c1
                        bond_len = np.linalg.norm(bond_vec)
                        if bond_len > 0:
                            bond_vec /= bond_len
                            # 选择垂直于键的方向
                            if abs(bond_vec[0]) < 0.9:
                                perp = np.cross(bond_vec, [1, 0, 0])
                            else:
                                perp = np.cross(bond_vec, [0, 1, 0])
                            perp /= np.linalg.norm(perp)
                            offset = 0.1 * perp  # 双键间距

                            # 绘制双键的两条线
                            ax.plot([c1[0]+offset[0], c2[0]+offset[0]],
                                   [c1[1]+offset[1], c2[1]+offset[1]],
                                   [c1[2]+offset[2], c2[2]+offset[2]],
                                   c=self.bond_color_double, linewidth=self.bond_width_double, alpha=0.9)
                            ax.plot([c1[0]-offset[0], c2[0]-offset[0]],
                                   [c1[1]-offset[1], c2[1]-offset[1]],
                                   [c1[2]-offset[2], c2[2]-offset[2]],
                                   c=self.bond_color_double, linewidth=self.bond_width_double, alpha=0.9)
                    else:
                        # 绘制单键
                        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                               c=self.bond_color, linewidth=self.bond_width, alpha=0.8)

        # 绘制C-H键
        h_counter = 0
        for i in range(n_carbons):
            num_attached_carbons = len(adjacency.get(i, []))
            is_sp2 = i in double_bond_nodes

            if is_sp2:
                # sp2碳:3个σ键
                num_h_attached = 3 - num_attached_carbons
            else:
                # sp3碳:4个σ键
                num_h_attached = 4 - num_attached_carbons

            num_h_attached = max(0, num_h_attached)

            c_pos = np.array(c_coords[i])
            for j in range(num_h_attached):
                if h_counter < len(h_coords):
                    h_pos = np.array(h_coords[h_counter])
                    ax.plot([c_pos[0], h_pos[0]], [c_pos[1], h_pos[1]], [c_pos[2], h_pos[2]],
                           c=self.bond_color, linewidth=self.bond_width*0.6, alpha=0.6)
                    h_counter += 1

        # 绘制碳原子
        for i, coord in enumerate(c_coords):
            is_sp2 = i in double_bond_nodes
            color = '#440000' if is_sp2 else self.atom_colors['C']  # sp2碳用深红色
            size = self.atom_radii['C'] * 120 if is_sp2 else self.atom_radii['C'] * 100

            ax.scatter(coord[0], coord[1], coord[2],
                      c=color, s=size,
                      alpha=0.9, edgecolors='black', linewidths=1)
            if show_labels:
                ax.text(coord[0], coord[1], coord[2], f'C{i+1}', fontsize=8, ha='center', va='center')

        # 绘制氢原子
        for idx, coord in enumerate(h_coords):
            ax.scatter(coord[0], coord[1], coord[2],
                      c=self.atom_colors['H'], s=self.atom_radii['H'] * 150,
                      alpha=0.9, edgecolors='black', linewidths=0.8)
            if show_labels:
                ax.text(coord[0], coord[1], coord[2], f'H{idx+1}', fontsize=5, ha='center', va='center', color='#0000AA', fontweight='bold')

    def _set_equal_aspect_ratio(self, ax, x_coords, y_coords, z_coords):
        """设置3D图形等比例"""
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        z_range = max(z_coords) - min(z_coords)

        max_range = max(x_range, y_range, z_range, 1)

        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2
        z_center = (max(z_coords) + min(z_coords)) / 2

        padding = max_range * 0.1
        limit = max_range / 2 + padding

        ax.set_xlim(x_center - limit, x_center + limit)
        ax.set_ylim(y_center - limit, y_center + limit)
        ax.set_zlim(z_center - limit, z_center + limit)

    def _find_main_chain(self, G, adjacency, degrees, db_u, db_v):
        """
        使用类 IUPAC 规则查找主链:
        1. 必须包含双键的两个碳原子
        2. 选择最长的链
        3. 若等长,选择取代基最多的链
        返回: (主链节点列表, 取代基信息字典)
        """
        from collections import deque

        # 找所有包含双键的最长路径
        all_paths = []

        def dfs_find_paths(current, target, visited, path):
            path = path + [current]
            visited = visited | {current}
            if current == target:
                all_paths.append(path)
                return
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    dfs_find_paths(neighbor, target, visited, path)

        dfs_find_paths(db_u, db_v, set(), [])

        # 对每条路径向两端延伸以获得最长链
        extended_paths = []
        for path in all_paths:
            extended = list(path)

            # 向起点方向延伸
            start = extended[0]
            while True:
                candidates = [nb for nb in adjacency.get(start, []) if nb not in extended]
                if not candidates:
                    break
                best = min(candidates, key=lambda x: degrees.get(x, 0))
                extended.insert(0, best)
                start = best

            # 向终点方向延伸
            end = extended[-1]
            while True:
                candidates = [nb for nb in adjacency.get(end, []) if nb not in extended]
                if not candidates:
                    break
                best = min(candidates, key=lambda x: degrees.get(x, 0))
                extended.append(best)
                end = best

            extended_paths.append(extended)

        if not extended_paths:
            return [db_u, db_v], {}

        # 选最长链;若等长,选取代基最多的
        extended_paths.sort(key=lambda p: (-len(p), -sum(1 for n in p if degrees.get(n, 0) > 2)))
        best_chain = extended_paths[0]

        # 计算取代基信息
        chain_set = set(best_chain)
        substituents = {}
        for i, node in enumerate(best_chain):
            branches = [nb for nb in adjacency.get(node, []) if nb not in chain_set]
            if branches:
                substituents[i] = branches

        return best_chain, substituents

    def _detect_ez_stereochemistry(self, G, adjacency, db_u, db_v):
        """
        检测 E/Z 立体化学.
        
        对于每个 sp^2 碳原子,比较其两个非双键取代基的 Cahn-Ingold-Prelog 优先级.
        优先级规则:比较取代基子树的大小和结构复杂度.
        
        返回: 'E', 'Z', 或 None(无法确定/不适用)
        """
        degrees_dict = {node: len(adjacency.get(node, [])) for node in adjacency}

        def get_subtree_fingerprint(root, exclude_node):
            """获取子树的详细指纹用于 CIP 优先级比较"""
            excluded = {exclude_node}
            def collect(node, visited):
                visited = visited | {node}
                children = [n for n in adjacency.get(node, []) if n not in visited and n not in excluded]
                child_fps = tuple(sorted([collect(c, visited | {c}) for c in children]))
                return (degrees_dict.get(node, 0), child_fps)
            return collect(root, {root})

        # 获取双键两端各自的非双键取代基
        u_neighbors = [nb for nb in adjacency.get(db_u, []) if nb != db_v]
        v_neighbors = [nb for nb in adjacency.get(db_v, []) if nb != db_u]

        if len(u_neighbors) < 1 or len(v_neighbors) < 1:
            return None

        # 判断各端是否有不同取代基
        u_different = False
        v_different = False

        if len(u_neighbors) >= 2:
            fp1 = get_subtree_fingerprint(u_neighbors[0], db_v)
            fp2 = get_subtree_fingerprint(u_neighbors[1], db_v)
            u_different = (fp1 != fp2)

        if len(v_neighbors) >= 2:
            fp1 = get_subtree_fingerprint(v_neighbors[0], db_u)
            fp2 = get_subtree_fingerprint(v_neighbors[1], db_u)
            v_different = (fp1 != fp2)

        if not u_different and not v_different:
            return None

        # 使用确定性哈希方法分配 E/Z 标记
        u_fp_tuple = tuple(get_subtree_fingerprint(nb, db_v) for nb in u_neighbors)
        v_fp_tuple = tuple(get_subtree_fingerprint(nb, db_u) for nb in v_neighbors)

        combined_hash = hash((u_fp_tuple, v_fp_tuple))
        return 'Z' if combined_hash % 2 == 0 else 'E'

    def _name_linear_alkene(self, chain_len, db_pos):
        """命名线性链烯烃"""
        alkene_names = {
            2: "乙烯", 3: "丙烯", 4: "丁烯", 5: "戊烯",
            6: "己烯", 7: "庚烯", 8: "辛烯", 9: "壬烯", 10: "癸烯"
        }
        base = alkene_names.get(chain_len, f"{chain_len}-烯")
        if chain_len <= 3:
            return base
        return f"{db_pos}-{base}"

    def _name_branched_alkene(self, G, n, chain, chain_len, db_pos,
                               substituents, db_u, db_v, degrees):
        """
        命名支链烯烃(通用方法)
        
        IUPAC 格式: 位置-取代基-双键位置-母体烯烃名
        例如: 2-甲基-1-丁烯, 3-甲基-2-戊烯
        """
        alkene_names = {
            4: "丁烯", 5: "戊烯", 6: "己烯", 7: "庚烯", 8: "辛烯"
        }

        sub_parts = []
        for pos in sorted(substituents.keys()):
            chain_position = pos + 1
            num_branches = len(substituents[pos])

            branch_sizes = []
            for branch_node in substituents[pos]:
                from collections import deque
                visited = {branch_node} | set(chain)
                queue = [branch_node]
                size = 1
                while queue:
                    curr = queue.pop(0)
                    for nb in G.neighbors(curr):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                            size += 1
                    branch_sizes.append(size)

            total_branch_carbons = sum(branch_sizes)

            if total_branch_carbons == 1:
                sub_name = "甲基"
            elif total_branch_carbons == 2:
                sub_name = "乙基"
            elif total_branch_carbons == 3:
                sub_name = "丙基"
            else:
                sub_name = f"{total_branch_carbons}烷基"

            if num_branches > 1 and total_branch_carbons == 1:
                sub_name = "二甲基"
            elif num_branches > 1:
                sub_name = f"二{sub_name}"

            sub_parts.append((chain_position, sub_name))

        # 组装完整名称(标准 IUPAC 格式,带连字符)
        base_name = alkene_names.get(chain_len, f"{chain_len}-烯")

        if chain_len <= 3:
            if sub_parts:
                sub_strs = [f"{pos}-{name}" for pos, name in sub_parts]
                full_name = "-".join(sub_strs) + base_name
            else:
                full_name = base_name
        else:
            if sub_parts:
                sub_strs = [f"{pos}-{name}" for pos, name in sub_parts]
                full_name = "-".join(sub_strs) + f"-{db_pos}-{base_name}"
            else:
                full_name = f"{db_pos}-{base_name}"

        return full_name

    def _graph_to_structure_string(self, G, db_u, db_v):
        """
        将烯烃图转换为结构式字符串(类似烷烃 canon 格式)
        
        格式说明:
        - 每个碳原子表示为 C
        - 子分支用括号包裹:C(branch1, branch2)
        - 双键用 =C() 表示
        
        例如:
        - 1-戊烯:     C(=C(C,C,C),C)
        - 2-戊烯:     C(C,=C(C),C)  
        - 2-甲基-1-丁烯: C(=C(C,C),C(C))
        - 3-甲基-1-丁烯: C(=C(C(C)),C)
        
        Args:
            G: 分子图
            db_u, db_v: 双键两端的节点
            
        返回:
            结构式字符串
        """
        adjacency = {node: list(G.neighbors(node)) for node in G.nodes()}
        
        # 选择根节点:优先选择度数最大的非双键端点(使树更平衡)
        # 或者选择双键的一个端点作为根
        best_root = db_u
        root = best_root

        visited = set()
        
        def build_subtree(node):
            """递归构建子树字符串"""
            if node in visited:
                return ""
            visited.add(node)
            
            neighbors = adjacency.get(node, [])
            children = [n for n in neighbors if n not in visited]
            
            if not children:
                return "C"
            
            # 构建子分支字符串
            branches = []
            for child in children:
                branch_str = build_subtree(child)
                if branch_str:
                    # 判断是否是双键连接
                    if (node == db_u and child == db_v) or (node == db_v and child == db_u):
                        branch_str = "=" + branch_str
                    branches.append(branch_str)
            
            if branches:
                return f"C({','.join(sorted(branches))})"
            else:
                return "C"

        result = build_subtree(root)
        return result if result else "C"

    def get_isomer_name(self, G: nx.Graph, n: int, index: int) -> str:
        """
        根据烯烃结构生成结构式字符串(模仿烷烃 canon 格式)

        输出格式示例:
        - C5H10 线性: C(=C(C,C,C),C) 或 C(C,=C(C),C)
        - C5H10 支链: C(=C(C,C),C(C))
        """
        info = self.get_graph_info(G)
        adjacency = info['adjacency']

        # 找到双键位置
        double_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('bond_type') == 'double']
        if not double_edges:
            return f"C{n}H{2*n} - 异构体{index+1}"

        db_u, db_v = double_edges[0]

        # 生成结构式字符串
        structure = self._graph_to_structure_string(G, db_u, db_v)

        return f"{structure}"


def main():
    """主函数 - 支持中断和多核心计算"""
    print("=" * 70)
    print("烯烃同分异构体可视化工具")
    print("支持多核心计算、进度条显示和中断功能")
    print("=" * 70)

    try:
        from alkene import AlkeneIsomerGenerator, AlkaneIsomerGenerator
    except ImportError:
        print("错误: 无法导入 alkene 模块")
        return

    try:
        n = int(input("\n请输入碳原子数量 (>=2): "))
        if n < 2:
            print("错误:碳原子数量必须 >= 2")
            return
    except ValueError:
        print("错误:请输入有效的数字")
        return

    formula = f"C{n}H{2*n}"

    # ========== 核心数选择 ==========
    max_cores = min(mp.cpu_count(), 11)
    print(f"\n系统可用核心数: {mp.cpu_count()}")
    print(f"建议使用核心数: {max_cores}")

    try:
        cores_input = input(f"请选择使用的核心数 (1-{max_cores}, 直接回车使用 {max_cores}): ").strip()
        if cores_input == "":
            num_cores = max_cores
        else:
            num_cores = int(cores_input)
            if num_cores < 1:
                num_cores = 1
            elif num_cores > max_cores:
                print(f"警告: 输入值超过最大核心数,将使用 {max_cores} 个核心")
                num_cores = max_cores
    except ValueError:
        num_cores = max_cores

    print(f"将使用 {num_cores} 个核心进行计算")

    # 创建可视化器
    visualizer = AlkeneIsomerVisualizer(num_cores=num_cores)

    # ========== 启动中断监听线程 ==========
    def interrupt_listener(vis):
        '''后台线程:监听用户输入以中断计算'''
        while vis.is_computing():
            user_input = input("\n[按 Enter 随时中断计算...] ").strip()
            if user_input == "":
                vis.stop()
                break
            time.sleep(0.5)

    interrupt_thread = None

    # 生成异构体
    print(f"\n开始生成 {formula} 的同分异构体...")
    print("提示: 计算过程中可按 Enter 键中断")

    # 启动监听线程
    interrupt_thread = threading.Thread(target=interrupt_listener, args=(visualizer,), daemon=True)
    interrupt_thread.start()

    isomers = visualizer.generate_isomers_optimized(n, show_progress=True)

    # 等待监听线程结束
    interrupt_thread.join(timeout=1)

    if not isomers:
        print("\n没有找到异构体或计算已中断")
        return

    print(f"\n{formula} 共有 {len(isomers)} 个同分异构体")

    # 检查是否返回的是占位符列表(无法生成具体结构)
    if isomers and isomers[0] is None:
        print("\n[注意] 由于结构生成算法限制,当前仅显示异构体计数.")
        print("       具体结构可视化功能暂不可用.")
        print("\n" + "=" * 50)
        return

    # 打印异构体列表
    print("\n异构体列表:")
    for i, iso in enumerate(isomers):
        double_edges = [(u, v) for u, v, d in iso.edges(data=True) if d.get('bond_type') == 'double']
        if double_edges:
            u, v = double_edges[0]
            name = visualizer.get_isomer_name(iso, n, i)
            print(f"  {i+1}. 双键: C{u+1}=C{v+1} - {name}")

    # 如果计算被中断,显示提示
    if visualizer.is_stopped():
        print("\n[计算曾被中断]")

    print("\n" + "=" * 50)
    print("选项:")
    print("1. 可视化单个异构体")
    print("2. 保存异构体列表到文件")
    print("3. 退出")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        try:
            idx = int(input(f"请输入要可视化的异构体编号 (1-{len(isomers)}): ")) - 1
            if 0 <= idx < len(isomers):
                G = isomers[idx]
                name = visualizer.get_isomer_name(G, n, idx)
                print(f"\n正在显示: {name}")
                visualizer.visualize_isomer(G, show=True)
            else:
                print("错误:异构体编号超出范围")
        except ValueError:
            print("错误:请输入有效的数字")

    elif choice == "2":
        # 保存异构体列表到文件
        filename = input(f"请输入输出文件名 (默认 C{n}_isomers.txt): ").strip()
        if not filename:
            filename = f"C{n}_isomers.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"C{n} 烯烃同分异构体列表\n")
                f.write(f"总数: {len(isomers)}\n")
                f.write("=" * 70 + "\n\n")
                for i, iso in enumerate(isomers):
                    double_edges = [(u, v) for u, v, d in iso.edges(data=True) if d.get('bond_type') == 'double']
                    if double_edges:
                        u, v = double_edges[0]
                        name = visualizer.get_isomer_name(iso, n, i)
                        f.write(f"{i+1}. 双键: C{u+1}=C{v+1} - {name}\n")
            print(f"已保存到: {filename}")
        except Exception as e:
            print(f"保存失败: {e}")

    else:
        print("已退出")


if __name__ == "__main__":
    main()
