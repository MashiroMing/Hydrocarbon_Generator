"""
烷烃同分异构体生成与3D可视化工具。
支持任意碳原子数的饱和烷烃生成(n≥1)，使用无标号树算法枚举异构体结构。
支持3D可视化展示碳骨架和氢原子的空间分布。
n≥15时自动启用并行计算加速。OEIS验证: C1=1,C2=1,C3=1,C4=2,C5=3,C6=5,C7=9,C8=18,C9=35,C10=75。
"""

import os, sys, math, warnings, signal, io, traceback
from typing import List, Tuple, Optional, Dict
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: pip install tqdm 可显示进度条")

import matplotlib
_plt = None
_GridSpec = None
_cm = None

def _get_plt():
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
# 第四部分：全局异常钩子（处理Ctrl+C）
# ============================================================

# 保存原始的异常处理函数
_original_excepthook = sys.excepthook
_keyboard_interrupt_occurred = False  # 标记：是否发生了键盘中断

def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    自定义全局异常钩子
    
    【功能】
    - 捕获KeyboardInterrupt（Ctrl+C中断）
    - 避免输出冗长的堆栈信息
    - 提供友好的中断提示
    
    【参数】
    - exc_type: 异常类型
    - exc_value: 异常值
    - exc_traceback: 异常堆栈
    """
    global _keyboard_interrupt_occurred
    if exc_type is KeyboardInterrupt:
        _keyboard_interrupt_occurred = True
        return  # 直接返回，不输出堆栈
    _original_excepthook(exc_type, exc_value, exc_traceback)

# 注册自定义异常钩子
sys.excepthook = _custom_excepthook

# ============================================================
# 第五部分：中文字体配置
# ============================================================

def configure_chinese_font():
    """
    配置matplotlib支持中文显示
    
    【问题背景】
    - matplotlib默认字体不支持中文
    - 需要显式设置支持中文的字体
    
    【字体优先级】
    1. Microsoft YaHei（微软雅黑）- Windows首选
    2. SimHei（黑体）- Windows备选
    3. Arial Unicode MS - 跨平台
    4. DejaVu Sans - Linux备选
    """
    plt = _get_plt()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.family'] = 'sans-serif'
    warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')


# =========================================================================
# 第六部分：烷烃异构体生成器 - AlkaneIsomerGenerator
# =========================================================================
"""
【类功能】
  AlkaneIsomerGenerator是烷烃异构体生成的核心类，负责：
  1. 生成所有可能的烷烃树结构
  2. 转换为规范字符串表示
  3. 去重得到唯一异构体
  4. 转换为3D坐标用于可视化

【核心概念】
  - 有根树：以某个碳为根的树结构
  - 规范字符串：用括号表示的树结构（如C(C,C(C))）
  - 整数划分：将n个碳分配到各子树
"""

class AlkaneIsomerGenerator:
    """
    饱和烷烃同分异构体生成器
    
    【化学背景】
    - 烷烃是只有单键的饱和烃，分子式为CₙH₂ₙ₊₂
    - 每个碳原子形成4个化学键（度≤4）
    - 烷烃分子图是连通无环图（树）
    
    【算法基础】
    - 使用无标号树算法枚举所有可能的分子结构
    - 支持并行计算加速大规模生成
    
    【使用示例】
      generator = AlkaneIsomerGenerator()
      isomers = generator.generate_isomers(6)  # 生成C6H14的所有异构体
    """
    
    # ============================================================
    # 6.1 类常量定义 - 化学键参数
    # ============================================================
    
    BOND_LENGTH_CC = 1.54      # 碳碳单键键长（埃），典型值1.54Å
    BOND_LENGTH_CH = 1.09      # 碳氢键键长（埃），典型值1.09Å
    TETRAHEDRAL_ANGLE = math.acos(-1/3)  # 四面体键角 ≈ 109.47°
    ANGLE_120 = 2 * math.pi / 3  # 120度（弧度），用于sp2杂化
    SQRT3 = math.sqrt(3)      # √3 ≈ 1.732，用于四面体坐标计算

    # ============================================================
    # 6.2 初始化方法
    # ============================================================
    
    def __init__(self, use_parallel=True, num_workers=None):
        """
        初始化烷烃异构体生成器
        
        【参数】
          use_parallel: 是否使用并行计算（默认True）
          num_workers: 并行进程数
                       - None: 自动检测（CPU核心数-1）
                       - 正整数: 指定进程数
        
        【成员变量】
          cache: 有根树生成缓存 { (n, max_branches): [结果] }
          isomer_cache: 异构体生成缓存 { n: [异构体列表] }
          use_parallel: 是否启用并行
          num_workers: 并行进程数
        """
        self.cache = {}              # 有根树缓存：避免重复计算
        self.isomer_cache = {}       # 异构体缓存：已生成的直接返回
        self.use_parallel = use_parallel  # 并行计算开关
        
        # 设置并行进程数
        if num_workers is None:
            # 默认使用CPU核心数-1，保留一个核心给系统
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = max(1, num_workers)

    # ============================================================
    # 6.3 规范字符串解析 - parse_substrings
    # ============================================================
    
    def parse_substrings(self, canon_str: str) -> List[str]:
        """
        解析规范字符串中的子树列表
        
        【输入示例】
          canon_str = "C(C,C(C))"
        
        【输出示例】
          ["C", "C", "C(C)"]  # 三个子树
        
        【解析逻辑】
        - 跳过首尾的 "C(" 和 ")"
        - 使用depth计数器跟踪括号深度
        - 用逗号在depth=0时分割子树
        
        【参数】
          canon_str: 规范字符串，如 "C(C,C(C))"
        
        【返回值】
          子树字符串列表，如 ["C", "C", "C(C)"]
        """
        # 特殊情况：单原子 "C" 没有子树
        if canon_str == "C":
            return []
        
        # 提取括号内容：去掉首尾 "C(" 和 ")"
        inner = canon_str[2:-1]
        if not inner:  # 空内容，无子树
            return []
        
        subs = []        # 子树列表
        depth = 0        # 当前括号深度
        cur = ""         # 当前累积的字符串
        
        # 逐字符解析
        for char in inner:
            if char == '(':
                depth += 1           # 进入子括号
                cur += char
            elif char == ')':
                depth -= 1           # 退出子括号
                cur += char
            elif char == ',' and depth == 0:
                # 顶级逗号：分割子树
                subs.append(cur)
                cur = ""
            else:
                cur += char
        
        # 处理最后一个子树
        if cur:
            subs.append(cur)
        
        return subs

    # ============================================================
    # 6.4 有根树生成 - generate_rooted_with_info
    # ============================================================
    
    def generate_rooted_with_info(self, n: int, max_branches: int = 4):
        """
        生成n个碳原子的所有有根树异构体
        
        【核心算法：整数划分 + 递归生成】
        
        【数学原理】
          对于有根树，设根节点有k个子分支，子树大小分别为s₁,s₂,...,sₖ
          则满足：s₁ + s₂ + ... + sₖ = n - 1
        
        【约束条件】
          - 根节点：最多4个子分支（对应甲烷CH₄）
          - 子树根节点：最多3个子分支（因为已用1个连接父节点）
        
        【去重策略】
          - 子树按规范字符串排序
          - 保证每种结构只有一种规范表示
        
        【参数】
          n: 树的总节点数（碳原子数）
          max_branches: 根节点允许的最大分支数
        
        【返回值】
          列表，每个元素是 (规范字符串, 子树大小列表)
          例如：[("C(C,C)", [1, 1]), ("C(C(C))", [2])]
        """
        # ---------- 6.4.1 缓存检查 ----------
        cache_key = (n, max_branches)
        if cache_key in self.cache:
            # 命中缓存，直接返回
            return self.cache[cache_key]

        # ---------- 6.4.2 基础情况：单节点 ----------
        if n == 1:
            # 单碳树：C（没有子节点）
            res = [("C", [])]
            self.cache[cache_key] = res
            return res

        results = []           # 存储所有生成的有根树
        target = n - 1          # 剩余需要分配给子树的碳数
        partitions = []         # 所有可能的整数划分

        # ---------- 6.4.3 整数划分：find_parts ----------
        def find_parts(rem, min_val, path):
            """
            递归寻找所有满足约束的整数划分
            
            【参数】
              rem: 剩余需要分配的碳数
              min_val: 当前允许的最小值（保证不重复）
              path: 当前已分配的划分方案
            
            【终止条件】
              1. 已达到最大分支数，且剩余为0
              2. 剩余为0，且已有分支
            """
            # 达到最大分支数限制
            if len(path) >= max_branches:
                if rem == 0:
                    partitions.append(tuple(path))  # 记录有效划分
                return
            
            # 剩余为0：完成一个有效划分
            if rem == 0:
                if path:
                    partitions.append(tuple(path))
                return
            
            # 递归尝试所有可能的值
            for i in range(min_val, rem + 1):
                # 特殊处理：最后一分支必须恰好用完
                if len(path) + 1 == max_branches:
                    if rem - i == 0:
                        find_parts(0, i, path + [i])
                    continue
                
                # 正常递归：分配i个碳，继续分配剩余的
                find_parts(rem - i, i, path + [i])

        # 执行整数划分
        find_parts(target, 1, [])

        # ---------- 6.4.4 递归生成子树 ----------
        seen_local = set()  # 局部去重集合
        
        for p in partitions:
            # 对划分的每个部分，递归生成子树
            # 注意：子树的根节点已连接父节点，所以max_branches=3
            options_per_size = [
                self.generate_rooted_with_info(s, max_branches=3) 
                for s in p
            ]

            # 枚举所有子树组合的笛卡尔积
            for combo in itertools.product(*options_per_size):
                items = []
                for i, item in enumerate(combo):
                    # item[0]是规范字符串，p[i]是对应大小
                    items.append((item[0], p[i]))
                
                # 按规范字符串排序，保证规范形式
                items.sort(key=lambda x: x[0])
                sorted_strs = [x[0] for x in items]
                sorted_sizes = [x[1] for x in items]
                
                # 构造规范字符串：C(子树1, 子树2, ...)
                canon = "C(" + ",".join(sorted_strs) + ")"
                
                # 去重
                if canon not in seen_local:
                    seen_local.add(canon)
                    results.append((canon, sorted_sizes))

        # ---------- 6.4.5 缓存结果 ----------
        self.cache[cache_key] = results
        return results

    # ============================================================
    # 6.5 主生成方法 - generate_isomers
    # ============================================================
    
    def generate_isomers(self, n: int, show_progress: bool = True):
        """
        生成n个碳原子的所有烷烃同分异构体
        
        【算法流程】
        1. 生成所有有根树
        2. 处理每个根树（对称性去重）
        3. 邻接表二次去重
        4. 返回规范字符串列表
        
        【参数】
          n: 碳原子数量
          show_progress: 是否显示进度条
        
        【返回值】
          规范字符串列表，每个字符串代表一个唯一的异构体
        
        【示例】
          generate_isomers(4) → ["C(C,C(C))", "C(C(C))"]
          # 对应：正丁烷、异丁烷
        """
        # ---------- 6.5.1 缓存检查 ----------
        if n in self.isomer_cache:
            return self.isomer_cache[n]
        
        # 边界情况处理
        if n == 0: 
            return []
        if n == 1: 
            return ["C"]  # 甲烷

        # ---------- 6.5.2 生成有根树 ----------
        # 整个分子作为根树，根节点允许4个分支
        all_rooted = self.generate_rooted_with_info(n, max_branches=4)

        # ---------- 6.5.3 对称性处理参数 ----------
        limit = n // 2  # 用于处理对称性：避免重复
        raw_fingerprints = set()  # 初步生成的异构体

        # ---------- 6.5.4 选择处理方式（串行/并行） ----------
        # 启发式：数据量>1000时启用并行
        use_parallel = self.use_parallel and len(all_rooted) > 1000

        if use_parallel:
            # 并行处理大规模数据
            raw_fingerprints = self._process_rooted_parallel(all_rooted, limit, show_progress)
        else:
            # 串行处理（带进度条）
            if show_progress and HAS_TQDM:
                from tqdm import tqdm
                iterator = tqdm(
                    all_rooted, 
                    desc="[1/2] 处理根树结构", 
                    unit="树",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )
            else:
                iterator = all_rooted

            # 逐个处理根树
            for canon, sub_sizes in iterator:
                processed = self._process_single_rooted(canon, sub_sizes, limit)
                if processed:
                    raw_fingerprints.add(processed)

        # ---------- 6.5.5 邻接表二次去重 ----------
        unique_adj_sets = set()     # 已去重的邻接表集合
        final_fingerprints = []     # 最终结果

        # 根据数据量选择串行/并行
        if use_parallel and len(raw_fingerprints) > 1000:
            final_fingerprints = self._deduplicate_parallel(
                raw_fingerprints, unique_adj_sets, show_progress
            )
        else:
            raw_list = list(raw_fingerprints)
            
            if show_progress and HAS_TQDM:
                from tqdm import tqdm
                iterator = tqdm(
                    raw_list, 
                    desc="[2/2] 去重处理", 
                    unit="个",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )
            else:
                iterator = raw_list

            # 逐个去重
            for raw_canon in iterator:
                result = self._deduplicate_single(raw_canon, unique_adj_sets)
                if result:
                    final_fingerprints.append(result)

        # ---------- 6.5.6 排序并缓存 ----------
        result = sorted(final_fingerprints)
        self.isomer_cache[n] = result
        return result

    # ============================================================
    # 6.6 单根树处理 - _process_single_rooted
    # ============================================================
    
    def _process_single_rooted(self, canon: str, sub_sizes: tuple, limit: int) -> Optional[str]:
        """
        处理单个有根树，处理对称性
        
        【对称性处理】
        对于某些结构，如 C(C(C)) 和 C(C)C
        从不同根节点看是同一棵树，需要去重
        
        【策略】
        - 如果最大子树≤limit：直接返回
        - 如果最大子树=limit：比较翻转后的规范字符串
        - 其他情况：返回None（会被丢弃）
        
        【参数】
          canon: 规范字符串
          sub_sizes: 子树大小列表
          limit: 对称性阈值 = n//2
        
        【返回值】
          处理后的规范字符串，或None
        """
        # 计算最大子树大小
        max_s = max(sub_sizes) if sub_sizes else 0
        
        if max_s <= limit:
            if max_s < limit:
                # 最大子树小于阈值：唯一，无需翻转
                return canon
            else:
                # 最大子树等于阈值：需要检查翻转
                idx = sub_sizes.index(limit)  # 最大子树的位置
                sub_strs = self.parse_substrings(canon)
                
                # 提取大子树
                big_sub_str = sub_strs[idx]
                # 其他子树
                rest_subs = sub_strs[:idx] + sub_strs[idx+1:]
                rest_subs.sort()
                rest_canon = "C(" + ",".join(rest_subs) + ")"
                
                # 解析大子树的子树
                big_children = self.parse_substrings(big_sub_str)
                
                # 重新组合：将大子树的子树与rest交换
                new_children = big_children + [rest_canon]
                new_children.sort()
                flipped_canon = "C(" + ",".join(new_children) + ")"
                
                # 返回字典序较小的（保证唯一性）
                return min(canon, flipped_canon)
        return None

    # ============================================================
    # 6.7 并行处理 - _process_rooted_parallel
    # ============================================================
    
    def _process_rooted_parallel(self, all_rooted: list, limit: int, show_progress: bool = True) -> set:
        """
        并行处理有根树
        
        【并行策略】
        - 使用ProcessPoolExecutor创建进程池
        - 手动分批处理，避免Windows spawn模式兼容问题
        - 批大小：500-5000个（根据数据量自适应）
        
        【Windows兼容性】
        - Windows使用spawn进程启动方式
        - 需要显式传递参数，不能捕获外部变量
        
        【参数】
          all_rooted: 所有有根树列表
          limit: 对称性阈值
          show_progress: 是否显示进度条
        
        【返回值】
          处理后的规范字符串集合
        """
        results = set()
        executor = None

        # 准备数据：转换为可序列化的元组列表
        items_list = [(canon, sub_sizes, limit) for canon, sub_sizes in all_rooted]
        total_items = len(items_list)

        # 创建进度条
        pbar = None
        if show_progress and HAS_TQDM:
            pbar = tqdm(
                total=total_items, 
                desc="[1/2] 处理根树结构", 
                unit="树", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate}]"
            )

        try:
            # 创建进程池（Windows使用spawn方式）
            ctx = mp.get_context('spawn')
            executor = ProcessPoolExecutor(
                max_workers=self.num_workers, 
                mp_context=ctx
            )

            # 计算批大小：根据数据量自适应
            batch_size = max(500, min(5000, total_items // 100))
            processed = 0
            
            # 分批提交任务
            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch = items_list[batch_start:batch_end]
                
                # 处理每批
                for result in executor.map(process_rooted_worker_parallel, batch):
                    if result:
                        results.add(result)
                    if pbar is not None:
                        pbar.update(1)
                        processed += 1

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

    # ============================================================
    # 6.8 单个去重 - _deduplicate_single
    # ============================================================
    
    def _deduplicate_single(self, raw_canon: str, unique_adj_sets: set) -> Optional[str]:
        """
        对单个异构体进行去重
        
        【去重方法】
        1. 转换为邻接表
        2. 验证结构有效性（度≤4）
        3. 标准化邻接表（排序）
        4. 检查是否已存在
        
        【参数】
          raw_canon: 规范字符串
          unique_adj_sets: 已去重的邻接表集合
        
        【返回值】
          如果唯一则返回规范字符串，否则返回None
        """
        try:
            # 转换为邻接表
            adj = self.canon_to_adjacency(raw_canon)
            
            # 验证结构有效性
            valid_structure = True
            for node, neighbors in adj.items():
                if len(neighbors) > 4:  # 碳原子最多4个键
                    valid_structure = False
                    break

            if not valid_structure:
                return None

            # 标准化邻接表（用于去重比较）
            standardized_adj = tuple(
                sorted((node, tuple(sorted(neighbors))) for node, neighbors in adj.items())
            )

            # 检查是否唯一
            if standardized_adj not in unique_adj_sets:
                unique_adj_sets.add(standardized_adj)
                return raw_canon
        except:
            pass
        return None

    # ============================================================
    # 6.9 并行去重 - _deduplicate_parallel
    # ============================================================
    
    def _deduplicate_parallel(self, raw_fingerprints: set, unique_adj_sets: set, 
                              show_progress: bool = True) -> List[str]:
        """
        并行执行去重
        
        【流程】
        1. 转换为列表并分批
        2. 多进程转换邻接表
        3. 主进程汇总去重
        """
        results = []
        executor = None

        # 准备数据
        fingerprints_list = list(raw_fingerprints)
        total_items = len(fingerprints_list)

        # 进度条
        pbar = None
        if show_progress and HAS_TQDM:
            pbar = tqdm(
                total=total_items, 
                desc="[2/2] 去重处理", 
                unit="个", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate}]"
            )

        try:
            # 创建进程池
            ctx = mp.get_context('spawn')
            executor = ProcessPoolExecutor(
                max_workers=self.num_workers, 
                mp_context=ctx
            )

            # 分批处理
            batch_size = max(500, min(5000, total_items // 100))
            
            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch = fingerprints_list[batch_start:batch_end]
                
                # 并行转换邻接表
                for result in executor.map(_canon_to_adjacency_worker, batch):
                    if result:
                        canon_str, adj = result
                        
                        # 验证结构
                        valid_structure = True
                        for node, neighbors in adj.items():
                            if len(neighbors) > 4:
                                valid_structure = False
                                break
                        
                        if valid_structure:
                            # 标准化并去重
                            standardized_adj = tuple(
                                sorted((node, tuple(sorted(neighbors))) 
                                      for node, neighbors in adj.items())
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

    # ============================================================
    # 6.10 规范字符串转邻接表 - canon_to_adjacency
    # ============================================================
    
    def canon_to_adjacency(self, canon_str: str) -> Dict[int, List[int]]:
        """
        将规范字符串转换为邻接表
        
        【邻接表格式】
          {节点ID: [邻居ID列表], ...}
          例如：{0: [1, 2], 1: [0], 2: [0]} 表示 C0-C1 和 C0-C2
        
        【递归解析】
        - 深度优先遍历规范字符串
        - 每次遇到新节点分配新ID
        - 更新父子的邻居关系
        
        【参数】
          canon_str: 规范字符串，如 "C(C,C(C))"
        
        【返回值】
          邻接表字典
        """
        # 甲烷：单节点，无邻居
        if canon_str == "C":
            return {0: []}
        
        node_counter = [0]  # 全局节点计数器（用列表以便在嵌套函数中修改）
        self._temp_adj = {}  # 临时邻接表

        def parse_subtree(s: str, parent: int = None) -> int:
            """
            递归解析子树
            
            【参数】
              s: 子树字符串
              parent: 父节点ID（None表示根节点）
            
            【返回值】
              当前节点的ID
            """
            nonlocal node_counter
            
            # 分配新节点ID
            node_id = node_counter[0]
            node_counter[0] += 1
            
            # 如果有父节点，建立父子关系
            if parent is not None:
                if parent not in self._temp_adj:
                    self._temp_adj[parent] = []
                if node_id not in self._temp_adj:
                    self._temp_adj[node_id] = []
                # 双向添加邻居
                self._temp_adj[parent].append(node_id)
                self._temp_adj[node_id].append(parent)
            
            # 递归处理子节点
            if s != "C":
                inner = s[2:-1]  # 去掉 "C(" 和 ")"
                if inner:
                    subs = self.parse_substrings(s)  # 解析子树列表
                    for sub in subs:
                        parse_subtree(sub, node_id)  # 递归
            
            return node_id

        # 从根节点开始解析
        parse_subtree(canon_str)
        return self._temp_adj

    # ============================================================
    # 6.11 邻接表转3D坐标 - adjacency_to_coords
    # ============================================================
    
    def adjacency_to_coords(self, adjacency: Dict[int, List[int]], 
                          n: int) -> List[Tuple[float, float, float]]:
        """
        将邻接表转换为3D坐标
        
        【几何约束】
        - 碳碳键长：1.54Å
        - 键角：四面体角 ≈ 109.47°
        - 键角：平面sp2角 = 120°（未使用）
        
        【算法】
        1. BFS确定节点处理顺序
        2. 跟踪每个节点的父节点
        3. 计算四面体方向：
           - 无邻居：用默认方向(1,0,0)
           - 1个邻居：绕该方向旋转109.47°
           - 2个邻居：在两个向量张成的平面内计算
           - 3个邻居：唯一确定第四个方向
        
        【参数】
          adjacency: 邻接表
          n: 节点数
        
        【返回值】
          坐标列表，每个元素是(x,y,z)元组
        """
        coords = [None] * n  # 坐标数组
        coords[0] = (0.0, 0.0, 0.0)  # 根节点放在原点
        
        visited = set([0])  # 已访问节点
        queue = [0]         # BFS队列
        parent = {}         # 父节点映射

        # ---------- BFS：建立父子关系 ----------
        node_depth = {0: 0}  # 节点深度
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    node_depth[neighbor] = node_depth[current] + 1
                    queue.append(neighbor)

        # ---------- BFS：确定处理顺序 ----------
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

        # ---------- 计算每个节点的坐标 ----------
        for node in processing_order:
            if node == 0:
                continue  # 根节点已在原点
            
            parent_node = parent[node]
            px, py, pz = coords[parent_node]

            # 找出已处理过的兄弟节点（用于计算方向）
            siblings_and_child = [
                child for child in adjacency[parent_node] 
                if coords[child] is not None
            ]
            siblings = [s for s in siblings_and_child if s != node]

            if coords[node] is not None:
                continue  # 已计算过

            # ---------- 根据邻居数计算四面体方向 ----------
            if not siblings:
                # 情况1：无邻居，用默认方向
                direction = (1.0, 0.0, 0.0)
            elif len(siblings) == 1:
                # 情况2：1个邻居 → 绕该边旋转109.47°
                s1_coords = np.array(coords[siblings[0]])
                parent_coords = np.array([px, py, pz])
                vec_to_s1 = s1_coords - parent_coords
                direction = self._calculate_tetrahedral_direction_from_one_bond(vec_to_s1)
            elif len(siblings) == 2:
                # 情况3：2个邻居 → 在两向量张成的平面内计算
                s1_coords = np.array(coords[siblings[0]])
                s2_coords = np.array(coords[siblings[1]])
                parent_coords = np.array([px, py, pz])
                vec_to_s1 = s1_coords - parent_coords
                vec_to_s2 = s2_coords - parent_coords
                direction = self._calculate_tetrahedral_direction_from_two_bonds(
                    vec_to_s1, vec_to_s2
                )
            elif len(siblings) == 3:
                # 情况4：3个邻居 → 唯一确定第四个方向
                s1_coords = np.array(coords[siblings[0]])
                s2_coords = np.array(coords[siblings[1]])
                s3_coords = np.array(coords[siblings[2]])
                parent_coords = np.array([px, py, pz])
                vec_to_s1 = s1_coords - parent_coords
                vec_to_s2 = s2_coords - parent_coords
                vec_to_s3 = s3_coords - parent_coords
                direction = self._calculate_tetrahedral_direction_from_three_bonds(
                    vec_to_s1, vec_to_s2, vec_to_s3
                )
            else:
                # 不应该到达这里（超过3个邻居）
                direction = (1.0, 0.0, 0.0)

            # 归一化方向向量
            dir_vec = np.array(direction, dtype=np.float64)
            dir_vec /= np.linalg.norm(dir_vec)

            # ========== 添加随机扰动避免共面 ==========
            # 根据节点深度分配不同的 z 方向
            depth = node_depth.get(node, 0)
            if abs(dir_vec[2]) < 0.5:
                # 使用随机扰动，基于深度和节点 ID 作为种子
                import random
                random.seed(node * 31 + depth * 17)
                z_offset = 0.8 if random.random() > 0.5 else -0.8
                dir_vec[2] = z_offset
                # 重新归一化
                dir_vec /= np.linalg.norm(dir_vec)

            # 计算新节点坐标
            nx = px + dir_vec[0] * self.BOND_LENGTH_CC
            ny = py + dir_vec[1] * self.BOND_LENGTH_CC
            nz = pz + dir_vec[2] * self.BOND_LENGTH_CC
            coords[node] = (nx, ny, nz)

        # ========== 修复伪键问题 ==========
        # 将坐标列表转换为字典格式
        coords_dict = {i: np.array(coord) for i, coord in enumerate(coords) if coord is not None}
        coords_dict = self._fix_alkane_pseudo_bonds(coords_dict, adjacency)
        # 转换回列表格式
        coords = [tuple(coords_dict[i]) if i in coords_dict else None for i in range(n)]

        return coords

    def _fix_alkane_pseudo_bonds(
        self,
        coords_dict: Dict[int, np.ndarray],
        adjacency: Dict[int, List[int]]
    ) -> Dict[int, np.ndarray]:
        """
        【烷烃坐标微调 - 仅修复化学键问题】

        【策略】
        只修复化学键的断键和压缩问题
        不主动修复伪键，因为伪键修复会破坏化学键

        【参数】
            coords_dict: 碳原子坐标字典
            adjacency: 邻接表

        【返回】
            修复后的碳原子坐标字典
        """
        import numpy as np

        fixed_coords = {i: np.array(c, dtype=np.float64) for i, c in coords_dict.items()}
        n = len(adjacency)

        # 参数设置
        bond_target = 1.54         # A - 化学键目标长度
        bond_upper = 1.65          # A - 化学键上限
        bond_lower = 1.44          # A - 化学键下限
        step_fraction = 0.15       # 调整步长

        def get_degree(node):
            return len(adjacency.get(node, []))

        def validate_move(node, new_pos):
            """验证移动后所有化学键是否在允许范围内"""
            for neighbor in adjacency.get(node, []):
                if neighbor in fixed_coords:
                    d = np.linalg.norm(new_pos - fixed_coords[neighbor])
                    if d < 1.30 or d > 1.75:
                        return False
            return True

        changed = True
        max_iterations = 100

        while changed and max_iterations > 0:
            changed = False
            max_iterations -= 1

            for i in range(n):
                if i not in fixed_coords:
                    continue
                for j in adjacency.get(i, []):
                    if j <= i or j not in fixed_coords:
                        continue

                    d = np.linalg.norm(fixed_coords[i] - fixed_coords[j])

                    if d > bond_upper:  # 断键
                        direction = fixed_coords[j] - fixed_coords[i]
                        dn = np.linalg.norm(direction)
                        if dn < 1e-6:
                            continue
                        direction = direction / dn
                        adjust = (d - bond_target) * step_fraction
                        adjust = max(adjust, 0.02)

                        if get_degree(i) <= get_degree(j):
                            new_pos = fixed_coords[i] + direction * adjust
                            if validate_move(i, new_pos):
                                fixed_coords[i] = new_pos
                                changed = True
                        else:
                            new_pos = fixed_coords[j] - direction * adjust
                            if validate_move(j, new_pos):
                                fixed_coords[j] = new_pos
                                changed = True

                    elif d < bond_lower:  # 压缩键
                        direction = fixed_coords[j] - fixed_coords[i]
                        dn = np.linalg.norm(direction)
                        if dn < 1e-6:
                            continue
                        direction = direction / dn
                        adjust = (bond_target - d) * step_fraction
                        adjust = max(adjust, 0.02)

                        if get_degree(i) <= get_degree(j):
                            new_pos = fixed_coords[i] + direction * adjust
                            if validate_move(i, new_pos):
                                fixed_coords[i] = new_pos
                                changed = True
                        else:
                            new_pos = fixed_coords[j] - direction * adjust
                            if validate_move(j, new_pos):
                                fixed_coords[j] = new_pos
                                changed = True

        return fixed_coords

    # ============================================================
    # 6.12 四面体方向计算（1-3个已有键）
    # ============================================================
    
    def _calculate_tetrahedral_direction_from_one_bond(self, vec_to_existing_bond):
        """
        根据1个已有键计算四面体方向
        
        【几何】
        已知一个键向量v1，需要找一个与v1成109.47°的方向v4
        
        【算法】
        使用Rodrigues旋转公式：
        1. 归一化v1
        2. 找一个垂直于v1的参考向量作为旋转轴
        3. 将v1绕旋转轴旋转109.47°
        """
        import numpy as np
        
        v1 = np.array(vec_to_existing_bond, dtype=np.float64)
        v1_norm = np.linalg.norm(v1)
        
        if v1_norm < 1e-10:
            return (1.0, 0.0, 0.0)
        
        v1 = v1 / v1_norm  # 归一化
        
        # 找一个垂直于v1的向量作为旋转轴
        if abs(v1[0]) < 0.9:
            axis = np.cross(v1, np.array([1.0, 0.0, 0.0]))
        else:
            axis = np.cross(v1, np.array([0.0, 1.0, 0.0]))
        
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return (1.0, 0.0, 0.0)
        axis = axis / axis_norm  # 归一化旋转轴
        
        # 四面体角: 109.47°
        theta = np.arccos(-1.0 / 3.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rodrigues旋转公式
        # v_rot = v * cos(θ) + (k × v) * sin(θ) + k * (k · v) * (1 - cos(θ))
        v_rot = cos_t * v1 + sin_t * np.cross(axis, v1) + (1 - cos_t) * np.dot(axis, v1) * axis
        
        v_rot_norm = np.linalg.norm(v_rot)
        if v_rot_norm < 1e-10:
            return (1.0, 0.0, 0.0)
        
        return tuple(v_rot / v_rot_norm)

    def _calculate_tetrahedral_direction_from_two_bonds(self, vec_to_existing_bond1, 
                                                        vec_to_existing_bond2):
        """
        根据2个已有键计算四面体方向
        
        【几何】
        已知两个键向量v1, v2，它们张成一个平面
        求在该平面内、与v1、v2都成109.47°的方向
        
        【算法】
        稳定算法：使用矩阵求解
        - 构建约束方程: v4·v1 = cos(109.47°), v4·v2 = cos(109.47°), |v4| = 1
        - 在法向量方向上选择正确的一侧
        """
        import numpy as np
        
        v1 = np.array(vec_to_existing_bond1, dtype=np.float64)
        v2 = np.array(vec_to_existing_bond2, dtype=np.float64)
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return (1.0, 0.0, 0.0)
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot = np.dot(v1, v2)
        
        # 处理平行或接近平行的情况
        if abs(dot - 1.0) < 1e-6:  # 平行同向
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(v1, np.array([0.0, 1.0, 0.0]))
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-10:
                perp = perp / perp_norm
                return tuple(perp)
            return (1.0, 0.0, 0.0)
        
        if abs(dot + 1.0) < 1e-6:  # 平行反向
            if abs(v1[0]) < 0.9:
                ref = np.cross(v1, np.array([1.0, 0.0, 0.0]))
            else:
                ref = np.cross(v1, np.array([0.0, 1.0, 0.0]))
            ref_norm = np.linalg.norm(ref)
            if ref_norm > 1e-10:
                ref = ref / ref_norm
                return self._calculate_tetrahedral_direction_from_one_bond(ref)
            return (1.0, 0.0, 0.0)
        
        # 计算两向量张成的法向量
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < 1e-6:  # 非常接近平行
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(v1, np.array([0.0, 1.0, 0.0]))
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-10:
                perp = perp / perp_norm
                return tuple(perp)
            return (1.0, 0.0, 0.0)
        
        normal = normal / normal_norm
        
        # 四面体角度的cos值
        cos_tetra = -1.0 / 3.0  # cos(109.47°)
        
        # 构建方程组求解 v4 = a*v1 + b*v2 + c*normal
        # 约束: v4·v1 = cos_tetra, v4·v2 = cos_tetra, |v4|^2 = 1
        A = np.array([
            [1.0, dot, 0.0],
            [dot, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        b = np.array([cos_tetra, cos_tetra, 0.0])
        
        try:
            coeffs = np.linalg.solve(A, b)
            a, b_coef, c = coeffs
            
            v4 = a * v1 + b_coef * v2 + c * normal
            v4_norm = np.linalg.norm(v4)
            
            if v4_norm > 1e-10:
                v4 = v4 / v4_norm
                # 验证角度
                dot_v4_v1 = np.dot(v4, v1)
                if abs(dot_v4_v1 - cos_tetra) < 0.05:
                    return tuple(v4)
        except np.linalg.LinAlgError:
            pass
        
        # 备用方法：使用几何构造
        # 在 v1+v2 方向和 normal 方向之间插值
        sum_v = v1 + v2
        sum_v_norm = np.linalg.norm(sum_v)
        if sum_v_norm > 1e-10:
            sum_v = sum_v / sum_v_norm
            # 验证
            dot_test = np.dot(sum_v, v1)
            if abs(dot_test - cos_tetra) < 0.1:
                return tuple(sum_v)
        
        # 最后尝试：直接返回法向量
        return tuple(normal)

    def _calculate_tetrahedral_direction_from_three_bonds(self, 
                                                           vec_to_existing_bond1, 
                                                           vec_to_existing_bond2, 
                                                           vec_to_existing_bond3):
        """
        根据3个已有键计算四面体方向
        
        【几何】
        已知三个键向量v1, v2, v3
        在正四面体中，第四个方向满足：
        v1 + v2 + v3 + v4 = 0
        
        【推导】
        v4 = -(v1 + v2 + v3)
        归一化后即为所求方向
        """
        import numpy as np
        
        v1 = np.array(vec_to_existing_bond1, dtype=np.float64)
        v2 = np.array(vec_to_existing_bond2, dtype=np.float64)
        v3 = np.array(vec_to_existing_bond3, dtype=np.float64)
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        v3_norm = np.linalg.norm(v3)
        
        if v1_norm < 1e-10 or v2_norm < 1e-10 or v3_norm < 1e-10:
            return (1.0, 0.0, 0.0)
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        v3 = v3 / v3_norm
        
        v4 = -(v1 + v2 + v3)
        v4_norm = np.linalg.norm(v4)
        
        if v4_norm < 1e-6:
            # 三个向量接近共面，需要找一个新的方向
            # 使用Gram-Schmidt正交化
            u1 = v1
            u2 = v2 - np.dot(v2, u1) * u1 / np.dot(u1, u1)
            u2_norm = np.linalg.norm(u2)
            if u2_norm > 1e-10:
                u2 = u2 / u2_norm
                # v4 应该与 v1, v2, v3 都成109.47°
                # 选择垂直于 v1 和 v2 构成的平面的方向
                normal = np.cross(u1, u2)
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 1e-10:
                    return tuple(normal / normal_norm)
            return (1.0, 0.0, 0.0)
        
        v4_normalized = v4 / v4_norm
        
        # 验证角度
        cos_tetra = -1.0 / 3.0
        for v in [v1, v2, v3]:
            dot = np.dot(v4_normalized, v)
            if abs(dot - cos_tetra) > 0.1:
                # 角度不对，尝试取反
                v4_normalized = -v4_normalized
                break
        
        return tuple(v4_normalized)

    # ============================================================
    # 6.13 异构体命名 - get_isomer_name
    # ============================================================
    
    def get_isomer_name(self, n: int, index: int) -> str:
        """
        获取异构体的中文名称
        
        【命名规则】
        - C1-C5: IUPAC系统名（甲烷、乙烷...戊烷）
        - C4: 正丁烷、异丁烷
        - C5: 正戊烷、异戊烷、新戊烷
        - C6+: 数字编号
        
        【参数】
          n: 碳原子数
          index: 异构体索引（从0开始）
        
        【返回值】
          格式化的名称，如 "正丁烷 (C4H10)"
        """
        # 基础名称映射
        base_names = {
            1: "甲烷", 2: "乙烷", 3: "丙烷", 4: "丁烷", 5: "戊烷",
            6: "己烷", 7: "庚烷", 8: "辛烷", 9: "壬烷", 10: "癸烷",
            11: "十一烷", 12: "十二烷", 13: "十三烷", 14: "十四烷",
            15: "十五烷", 16: "十六烷", 17: "十七烷", 18: "十八烷",
            19: "十九烷", 20: "二十烷", 21: "二十一烷", 22: "二十二烷",
            23: "二十三烷", 24: "二十四烷", 25: "二十五烷", 26: "二十六烷",
            27: "二十七烷", 28: "二十八烷", 29: "二十九烷", 30: "三十烷"
        }

        # 丁烷的两种异构体
        if n == 4:
            names = ["正丁烷", "异丁烷"]
            return f"{names[index] if index < len(names) else f'异构体{index+1}'} (C{n}H{2*n+2})"
        
        # 戊烷的三种异构体
        elif n == 5:
            names = ["正戊烷", "异戊烷", "新戊烷"]
            return f"{names[index] if index < len(names) else f'异构体{index+1}'} (C{n}H{2*n+2})"
        
        # 其他碳数
        else:
            base_name = base_names.get(n, f"C{n}烷")
            return f"{base_name}-异构体{index+1} (C{n}H{2*n+2})"


# =========================================================================
# 第七部分：烷烃可视化器 - AlkaneIsomerVisualizer
# =========================================================================
"""
【类功能】
  AlkaneIsomerVisualizer负责：
  1. 生成氢原子的3D坐标
  2. 使用matplotlib绘制3D分子结构
  3. 支持单个/批量可视化
  4. 支持图片保存

【可视化内容】
  - 碳骨架（灰色球体 + 灰色键）
  - 氢原子（白色球体 + 细键）
  - 原子编号标签（可选）
"""

class AlkaneIsomerVisualizer:
    """
    烷烃异构体3D可视化器
    
    【使用流程】
      1. 创建可视化器实例
      2. 调用visualize_isomer()可视化单个
      3. 或调用visualize_all_isomers()批量可视化
    """
    
    def __init__(self, generator=None):
        """
        初始化可视化器
        
        【参数】
          generator: AlkaneIsomerGenerator实例
                     如果为None，则创建默认实例
        """
        self.generator = generator if generator else AlkaneIsomerGenerator()
        
        # 碳原子样式：深灰色
        self.atom_colors = {'C': '#333333', 'H': '#FFFFFF'}
        
        # 原子大小：碳较大，氢较小
        self.atom_radii = {'C': 0.6, 'H': 0.3}
        
        # 键样式：灰色
        self.bond_color = '#808080'
        
        # 键宽度：碳碳键较粗
        self.bond_width = 3

    # ============================================================
    # 7.1 甲烷氢原子生成 - _generate_methane_hydrogens
    # ============================================================
    
    def _generate_methane_hydrogens(self, c_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        生成甲烷（CH₄）分子的4个氢原子位置
        
        【几何】
        甲烷是四面体结构，4个氢原子位于正四面体的顶点
        
        【坐标计算】
        使用正四面体的标准顶点坐标（归一化到键长）
        """
        # 正四面体顶点方向（已归一化）
        directions = [
            (self.generator.SQRT3/3, self.generator.SQRT3/3, self.generator.SQRT3/3),
            (self.generator.SQRT3/3, -self.generator.SQRT3/3, -self.generator.SQRT3/3),
            (-self.generator.SQRT3/3, self.generator.SQRT3/3, -self.generator.SQRT3/3),
            (-self.generator.SQRT3/3, -self.generator.SQRT3/3, self.generator.SQRT3/3)
        ]
        
        h_positions = []
        for direction in directions:
            # 氢坐标 = 碳坐标 + 方向 × 键长
            h_x = c_pos[0] + self.generator.BOND_LENGTH_CH * direction[0]
            h_y = c_pos[1] + self.generator.BOND_LENGTH_CH * direction[1]
            h_z = c_pos[2] + self.generator.BOND_LENGTH_CH * direction[2]
            h_positions.append((h_x, h_y, h_z))
        
        return h_positions

    # ============================================================
    # 7.2 末端碳氢原子生成 - _generate_terminal_hydrogens
    # ============================================================
    
    def _generate_terminal_hydrogens(self, c_pos: Tuple[float, float, float],
                                    adj_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        生成末端碳原子（如-CH₃）的3个氢原子位置
        
        【化学】
        末端碳连接1个碳原子和3个氢原子
        4个键呈四面体分布
        
        【几何】
        - 已知C-C键方向
        - 需要找3个与C-C键成109.47°的方向
        - 这3个方向在同一平面内，互成120°
        
        【参数】
          c_pos: 碳原子坐标
          adj_pos: 相邻碳原子坐标
        """
        # C-C键方向向量
        vec_to_adj = [
            adj_pos[0] - c_pos[0], 
            adj_pos[1] - c_pos[1], 
            adj_pos[2] - c_pos[2]
        ]
        len_vec = math.sqrt(sum(v**2 for v in vec_to_adj))
        if len_vec == 0:
            return []
        u_to_adj = [v/len_vec for v in vec_to_adj]  # 单位向量

        # 构建正交基：找一个不平行于u_to_adj的向量
        if abs(u_to_adj[0]) < 0.9:
            ref = (1, 0, 0)
        else:
            ref = (0, 1, 0)

        # Gram-Schmidt正交化
        v1 = [
            u_to_adj[1]*ref[2] - u_to_adj[2]*ref[1],
            u_to_adj[2]*ref[0] - u_to_adj[0]*ref[2],
            u_to_adj[0]*ref[1] - u_to_adj[1]*ref[0]
        ]
        len_v1 = math.sqrt(sum(v**2 for v in v1))
        if len_v1 == 0:
            return []
        v1 = [v/len_v1 for v in v1]  # 第一个正交方向

        v2 = [
            v1[1]*u_to_adj[2] - v1[2]*u_to_adj[1],
            v1[2]*u_to_adj[0] - v1[0]*u_to_adj[2],
            v1[0]*u_to_adj[1] - v1[1]*u_to_adj[0]
        ]
        len_v2 = math.sqrt(sum(v**2 for v in v2))
        if len_v2 == 0:
            return []
        v2 = [v/len_v2 for v in v2]  # 第二个正交方向

        # 计算3个氢原子位置
        h_positions = []
        for j in range(3):
            # 在v1、v2张成的平面内，绕u_to_adj旋转120°×j
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

    # ============================================================
    # 7.3 中间碳氢原子生成 - _generate_middle_hydrogens
    # ============================================================
    
    def _generate_middle_hydrogens(self, c_pos: Tuple[float, float, float],
                                   prev_pos: Tuple[float, float, float],
                                   next_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        生成中间碳原子（-CH₂-）的2个氢原子位置
        
        【化学】
        中间碳连接2个碳原子和2个氢原子
        
        【几何】
        - 两个C-C键的方向向量分别为 v_prev 和 v_next
        - 氢原子位于 v_prev 和 v_next 夹角的平分线方向
        - 两个氢原子分别位于平分线两侧
        
        【修复】使用实际的键向量而不是归一化的单位向量来计算平分线
        """
        # 两个C-C键方向向量（使用实际键长）
        vec_to_prev = np.array([
            prev_pos[0] - c_pos[0], 
            prev_pos[1] - c_pos[1], 
            prev_pos[2] - c_pos[2]
        ])
        vec_to_next = np.array([
            next_pos[0] - c_pos[0], 
            next_pos[1] - c_pos[1], 
            next_pos[2] - c_pos[2]
        ])

        # 归一化用于计算
        len_prev = np.linalg.norm(vec_to_prev)
        len_next = np.linalg.norm(vec_to_next)
        if len_prev == 0 or len_next == 0:
            return []
        u_prev = vec_to_prev / len_prev
        u_next = vec_to_next / len_next

        # 【修复】使用实际的键向量来计算平分线，而不是单位向量
        # 这样可以更好地处理不同键长的情况
        bisector = vec_to_prev + vec_to_next
        len_bis = np.linalg.norm(bisector)
        if len_bis == 0:
            # 退化情况：两向量反向，使用垂直方向
            cross = np.cross(u_prev, u_next)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-10:
                # 退化情况：两向量平行
                if abs(u_prev[0]) < 0.9:
                    perp = np.cross(u_prev, np.array([1, 0, 0]))
                else:
                    perp = np.cross(u_prev, np.array([0, 1, 0]))
                perp /= np.linalg.norm(perp)
                bisector = perp * self.generator.BOND_LENGTH_CC
            else:
                bisector = cross / cross_norm
            len_bis = np.linalg.norm(bisector)
        bisector = bisector / len_bis

        # 计算垂直于两个C-C键所在平面的方向
        cross = np.cross(vec_to_prev, vec_to_next)
        len_cross = np.linalg.norm(cross)
        if len_cross == 0:
            # 退化情况：两向量平行
            if abs(bisector[0]) < 0.9:
                perp_fallback = np.cross(bisector, np.array([1, 0, 0]))
            else:
                perp_fallback = np.cross(bisector, np.array([0, 1, 0]))
            perp_fallback /= np.linalg.norm(perp_fallback)
            cross = perp_fallback
        else:
            cross = cross / len_cross

        # 计算两个氢原子位置
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

    # ============================================================
    # 7.4 叔碳氢原子生成 - _generate_tertiary_hydrogens
    # ============================================================
    
    def _generate_tertiary_hydrogens(self, c_pos: Tuple[float, float, float],
                                    adjacent_carbon_coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        生成叔碳原子（-CH-）的1个氢原子位置
        
        【化学】
        叔碳连接3个碳原子和1个氢原子
        
        【几何】
        - 氢原子位于3个C-C键的合成向量的反方向
        - 即 氢方向 = -(v1 + v2 + v3) 的归一化方向
        
        【修复】使用实际键向量而非单位向量，确保四面体几何正确
        """
        total_vec = np.array([0.0, 0.0, 0.0])
        bond_lengths = []
        
        for adj_coord in adjacent_carbon_coords:
            vec_to_adj = np.array(adj_coord) - np.array(c_pos)
            bond_len = np.linalg.norm(vec_to_adj)
            if bond_len > 0:
                total_vec += vec_to_adj  # 使用实际向量，不是单位向量
                bond_lengths.append(bond_len)
        
        if not bond_lengths:
            return []
        
        avg_len = sum(bond_lengths) / len(bond_lengths)
        
        # 计算氢原子方向（合成向量的反方向）
        len_sum = np.linalg.norm(total_vec)
        if len_sum < 1e-10:
            # 退化情况：三个向量几乎相互抵消
            # 使用第一个向量的反方向
            vec = np.array(adjacent_carbon_coords[0]) - np.array(c_pos)
            vec /= np.linalg.norm(vec)
            h_dir = -vec
        else:
            h_dir = -total_vec / len_sum
        
        # 计算氢原子位置
        h_pos = np.array(c_pos) + self.generator.BOND_LENGTH_CH * h_dir
        return [tuple(h_pos)]

    # ============================================================
    # 7.5 统一氢原子生成 - _generate_hydrogen_coords
    # ============================================================
    
    def _generate_hydrogen_coords(self, c_coords: List[Tuple[float, float, float]],
                                  adjacency: Dict[int, List[int]]) -> List[Tuple[float, float, float]]:
        """
        生成所有碳原子的所有氢原子坐标
        
        【核心逻辑】
        根据每个碳原子连接的碳原子数，决定氢原子数：
        - 0个碳（甲烷CH₄）：4个氢
        - 1个碳（CH₃-）：3个氢
        - 2个碳（-CH₂-）：2个氢
        - 3个碳（-CH-）：1个氢
        - 4个碳（-C-）：0个氢
        
        【参数】
          c_coords: 碳原子坐标列表
          adjacency: 邻接表
        
        【返回值】
          所有氢原子的坐标列表
        """
        h_coords = []
        n_carbons = len(c_coords)
        
        # 建立H->C的映射关系
        h_to_c_map = []  # [(h_idx, c_idx), ...]

        # 遍历每个碳原子
        for i in range(n_carbons):
            c_pos = c_coords[i]
            adjacent_carbon_indices = adjacency.get(i, [])
            adjacent_carbon_coords = [
                c_coords[idx] for idx in adjacent_carbon_indices if idx < n_carbons
            ]
            num_attached_carbons = len(adjacent_carbon_coords)

            # 根据连接碳数选择生成方法
            if num_attached_carbons == 0:
                # 甲烷
                h_positions = self._generate_methane_hydrogens(c_pos)
            elif num_attached_carbons == 1:
                # 甲基
                adj_pos = adjacent_carbon_coords[0]
                h_positions = self._generate_terminal_hydrogens(c_pos, adj_pos)
            elif num_attached_carbons == 2:
                # 亚甲基
                adj_indices = adjacency.get(i, [])
                adj_indices_sorted = sorted(adj_indices)
                adj_coords_sorted_by_idx = [
                    c_coords[idx] for idx in adj_indices_sorted if idx < n_carbons
                ]
                if len(adj_coords_sorted_by_idx) >= 2:
                    prev_pos = adj_coords_sorted_by_idx[0]
                    next_pos = adj_coords_sorted_by_idx[1]
                else:
                    prev_pos = adj_coords_sorted_by_idx[0] if adj_coords_sorted_by_idx else (0,0,0)
                    next_pos = (0,0,0)
                h_positions = self._generate_middle_hydrogens(c_pos, prev_pos, next_pos)
            elif num_attached_carbons == 3:
                # 次甲基（叔碳）
                h_positions = self._generate_tertiary_hydrogens(c_pos, adjacent_carbon_coords)
            elif num_attached_carbons >= 4:
                # 季碳（无氢）
                h_positions = []
            else:
                h_positions = []

            # 记录映射关系
            for _ in h_positions:
                h_to_c_map.append((len(h_coords), i))
            h_coords.extend(h_positions)

        # 【修复】氢原子碰撞检测和调整
        h_coords = self._fix_hydrogen_collisions(h_coords, h_to_c_map, c_coords)

        return h_coords
    
    def _fix_hydrogen_collisions(self, h_coords: List[Tuple[float, float, float]],
                                  h_to_c_map: List[Tuple[int, int]],
                                  c_coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        检测并修复氢原子之间的碰撞
        
        【问题】某些异构体中，氢原子可能离非键合的碳原子太近
        【解决】对氢原子位置进行微调，避免碰撞
        
        【参数】
          h_coords: 氢原子坐标列表
          h_to_c_map: 氢->碳映射列表 [(h_idx, c_idx), ...]
          c_coords: 碳原子坐标列表
        
        【返回值】
          调整后的氢原子坐标列表
        """
        import random
        
        fixed_h = [np.array(h, dtype=np.float64) for h in h_coords]
        n_h = len(fixed_h)
        n_c = len(c_coords)
        
        # 最小允许的H-C距离（标准C-H键长约1.09A）
        MIN_H_C_DIST = 1.00  # A - 增加到1.0A以确保安全
        
        # 最小允许的H-H距离（氢原子范德华半径约1.2A）
        MIN_H_H_DIST = 1.00  # A
        
        max_iterations = 100
        changed = True
        
        while changed and max_iterations > 0:
            changed = False
            max_iterations -= 1
            
            # 第一步：处理H-C碰撞（更优先）
            for i in range(n_h):
                h_pos = fixed_h[i]
                h_carbon_idx = h_to_c_map[i][1]
                c_pos = np.array(c_coords[h_carbon_idx])
                
                # 检查到所属碳的距离
                dist_to_own_c = np.linalg.norm(h_pos - c_pos)
                if dist_to_own_c < 0.9 or dist_to_own_c > 1.3:
                    direction = h_pos - c_pos
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-10:
                        direction = direction / dir_norm
                        h_pos = c_pos + direction * self.generator.BOND_LENGTH_CH
                        fixed_h[i] = h_pos
                        changed = True
                        continue
                
                # 检查到其他碳的距离
                closest_c_idx = -1
                closest_dist = float('inf')
                for j in range(n_c):
                    if j == h_carbon_idx:
                        continue
                    dist = np.linalg.norm(h_pos - np.array(c_coords[j]))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_c_idx = j
                
                if closest_c_idx >= 0 and closest_dist < MIN_H_C_DIST:
                    # 计算远离最近碳的方向
                    push_dir = h_pos - np.array(c_coords[closest_c_idx])
                    push_norm = np.linalg.norm(push_dir)
                    if push_norm > 1e-10:
                        push_dir = push_dir / push_norm
                        
                        # 所属碳方向
                        to_own_dir = h_pos - c_pos
                        to_own_norm = np.linalg.norm(to_own_dir)
                        if to_own_norm > 1e-10:
                            to_own_dir = to_own_dir / to_own_norm
                        
                        # 调整权重：距离越近，推力越大
                        weight = 0.5 - (closest_dist / MIN_H_C_DIST - 1) * 0.3
                        weight = max(0.2, min(0.6, weight))
                        
                        new_dir = to_own_dir * (1 - weight) + push_dir * weight
                        new_dir_norm = np.linalg.norm(new_dir)
                        if new_dir_norm > 1e-10:
                            new_dir = new_dir / new_dir_norm
                            h_pos = c_pos + new_dir * self.generator.BOND_LENGTH_CH
                            fixed_h[i] = h_pos
                            changed = True
            
            # 第二步：处理H-H碰撞
            for i in range(n_h):
                for j in range(i + 1, n_h):
                    dist = np.linalg.norm(fixed_h[i] - fixed_h[j])
                    if dist < MIN_H_H_DIST and dist > 1e-10:
                        # 两个氢原子互相推开
                        push_dir = fixed_h[i] - fixed_h[j]
                        push_norm = np.linalg.norm(push_dir)
                        push_dir = push_dir / push_norm
                        
                        # 调整量
                        adjust = (MIN_H_H_DIST - dist) * 0.55
                        
                        # 分别向各自所属碳的方向移动
                        i_c = h_to_c_map[i][1]
                        j_c = h_to_c_map[j][1]
                        
                        to_i_c = fixed_h[i] - np.array(c_coords[i_c])
                        to_i_c_norm = np.linalg.norm(to_i_c)
                        if to_i_c_norm > 1e-10:
                            to_i_c = to_i_c / to_i_c_norm
                            fixed_h[i] = fixed_h[i] + (to_i_c + push_dir) * adjust * 0.5
                        
                        to_j_c = fixed_h[j] - np.array(c_coords[j_c])
                        to_j_c_norm = np.linalg.norm(to_j_c)
                        if to_j_c_norm > 1e-10:
                            to_j_c = to_j_c / to_j_c_norm
                            fixed_h[j] = fixed_h[j] + (to_j_c - push_dir) * adjust * 0.5
                        
                        # 归一化到正确键长
                        for k in [i, j]:
                            k_c = h_to_c_map[k][1]
                            new_dir = fixed_h[k] - np.array(c_coords[k_c])
                            new_norm = np.linalg.norm(new_dir)
                            if new_norm > 1e-10:
                                fixed_h[k] = np.array(c_coords[k_c]) + (new_dir / new_norm) * self.generator.BOND_LENGTH_CH
                        
                        changed = True

        return [tuple(h) for h in fixed_h]

    # ============================================================
    # 7.6 可视化单个异构体 - visualize_isomer
    # ============================================================
    
    def visualize_isomer(self, n_carbons: int, isomer_index: int, show=True, save_path=None):
        """
        可视化单个烷烃异构体
        
        【参数】
          n_carbons: 碳原子数
          isomer_index: 异构体索引（从0开始）
          show: 是否显示图形
          save_path: 保存路径（可选）
        """
        # 生成异构体
        isomers = self.generator.generate_isomers(n_carbons)
        if isomer_index >= len(isomers):
            raise ValueError(f"异构体索引超出范围：{isomer_index} >= {len(isomers)}")
        
        canon = isomers[isomer_index]
        
        # 转换为邻接表
        adjacency = self.generator.canon_to_adjacency(canon)
        
        # 计算3D坐标
        c_coords = self.generator.adjacency_to_coords(adjacency, n_carbons)
        
        # 生成氢原子坐标
        all_coords = c_coords + self._generate_hydrogen_coords(c_coords, adjacency)

        # 创建图形
        plt = _get_plt()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制分子
        self._draw_molecule(ax, all_coords, n_carbons, adjacency)
        
        # 设置标题和标签
        name = self.generator.get_isomer_name(n_carbons, isomer_index)
        ax.set_title(name, fontsize=14, pad=20)
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')
        ax.set_zlabel('Z (A)')
        ax.grid(True, alpha=0.3)
        
        # 设置等比例
        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]
        z_coords = [coord[2] for coord in all_coords]
        self._set_equal_aspect_ratio(ax, x_coords, y_coords, z_coords)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至：{save_path}")
        
        if matplotlib.get_backend().lower() != 'agg':
            if show: 
                plt.show()
            else: 
                plt.close()
        else:
            plt.close()
            if show:
                print("[提示] 当前使用非交互式后端(Agg)，图片已自动保存。请使用 save_path 参数保存图片。")

    # ============================================================
    # 7.7 批量可视化 - visualize_all_isomers
    # ============================================================
    
    def visualize_all_isomers(self, n_carbons: int, show=True, save_dir=None):
        """
        在一个图形中可视化所有烷烃异构体
        
        【布局策略】
        - 2个以下：2×1
        - 4个以下：2×2
        - 9个以下：3×3
        - 更多：5列自适应行数
        
        【参数】
          n_carbons: 碳原子数
          show: 是否显示图形
          save_dir: 保存目录（可选）
        """
        # 生成所有异构体
        isomers = self.generator.generate_isomers(n_carbons)
        if not isomers:
            print(f"C{n_carbons}没有异构体")
            return
        
        n_isomers = len(isomers)
        
        # 计算布局
        if n_isomers <= 2: 
            cols, rows = 2, 1
        elif n_isomers <= 4: 
            cols, rows = 2, 2
        elif n_isomers <= 9: 
            cols, rows = 3, 3
        else: 
            cols = 5
            rows = (n_isomers + cols - 1) // cols

        plt = _get_plt()
        GridSpec = _GridSpec
        fig = plt.figure(figsize=(6*cols, 5*rows))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # 绘制每个异构体
        for i, canon in enumerate(isomers):
            if i >= rows * cols: 
                break
            row, col = i // cols, i % cols
            
            # 计算坐标
            adjacency = self.generator.canon_to_adjacency(canon)
            c_coords = self.generator.adjacency_to_coords(adjacency, n_carbons)
            all_coords = c_coords + self._generate_hydrogen_coords(c_coords, adjacency)
            
            # 创建子图
            ax = fig.add_subplot(gs[row, col], projection='3d')
            self._draw_molecule(ax, all_coords, n_carbons, adjacency, show_labels=False)
            
            # 设置标题
            name = self.generator.get_isomer_name(n_carbons, i)
            ax.set_title(name, fontsize=10, pad=10)
            ax.set_xlabel('X (A)', fontsize=8)
            ax.set_ylabel('Y (A)', fontsize=8)
            ax.set_zlabel('Z (A)', fontsize=8)
            ax.tick_params(labelsize=7)
            
            # 设置等比例
            x_coords = [coord[0] for coord in all_coords]
            y_coords = [coord[1] for coord in all_coords]
            z_coords = [coord[2] for coord in all_coords]
            self._set_equal_aspect_ratio(ax, x_coords, y_coords, z_coords)
        
        # 总标题
        fig.suptitle(f'C{n_carbons}H{2*n_carbons+2} 的所有同分异构体', 
                     fontsize=16, fontweight='bold')
        
        # 保存或显示
        if save_dir:
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'C{n_carbons}_all_isomers.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"图片已保存至：{save_path}")
        
        if matplotlib.get_backend().lower() != 'agg':
            if show: 
                plt.show()
            else: 
                plt.close()
        else:
            plt.close()
            if show:
                print("[提示] 当前使用非交互式后端(Agg)，图片已自动保存。请使用 save_dir 参数保存图片。")

    # ============================================================
    # 7.8 分子绘制 - _draw_molecule
    # ============================================================
    
    def _draw_molecule(self, ax, coords, n_carbons, adjacency, show_labels=True):
        """
        绘制分子结构
        
        【绘制顺序】
        1. C-C键（灰色粗线）
        2. C-H键（灰色细线）
        3. 碳原子（深灰色球体 + 编号标签）
        4. 氢原子（白色球体）
        
        【参数】
          ax: matplotlib 3D坐标轴
          coords: 所有原子坐标（碳在前，氢在后）
          n_carbons: 碳原子数
          adjacency: 邻接表
          show_labels: 是否显示原子编号
        """
        c_coords = coords[:n_carbons]  # 碳原子坐标
        h_coords = coords[n_carbons:]   # 氢原子坐标

        # ---------- 绘制C-C键 ----------
        for parent in range(n_carbons):
            if parent in adjacency:
                for child in adjacency[parent]:
                    if child > parent:  # 每条边只画一次
                        c1 = np.array(c_coords[parent])
                        c2 = np.array(c_coords[child])
                        ax.plot(
                            [c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                            c=self.bond_color, linewidth=self.bond_width, alpha=0.8
                        )

        # ---------- 绘制C-H键 ----------
        h_counter = 0
        for i in range(n_carbons):
            # 氢原子数 = 4 - 连接的碳原子数
            num_h_attached = 4 - len(adjacency.get(i, []))
            c_pos = np.array(c_coords[i])
            
            for j in range(num_h_attached):
                if h_counter < len(h_coords):
                    h_pos = np.array(h_coords[h_counter])
                    ax.plot(
                        [c_pos[0], h_pos[0]], 
                        [c_pos[1], h_pos[1]], 
                        [c_pos[2], h_pos[2]],
                        c=self.bond_color, 
                        linewidth=self.bond_width*0.7, 
                        alpha=0.6
                    )
                    h_counter += 1

        # ---------- 绘制碳原子 ----------
        for i, coord in enumerate(c_coords):
            ax.scatter(
                coord[0], coord[1], coord[2],
                c=self.atom_colors['C'], 
                s=self.atom_radii['C'] * 100,
                alpha=0.9, 
                edgecolors='black', 
                linewidths=1
            )
            if show_labels:
                ax.text(coord[0], coord[1], coord[2], 
                       f'C{i+1}', fontsize=8, ha='center', va='center')

        # ---------- 绘制氢原子 ----------
        for coord in h_coords:
            ax.scatter(
                coord[0], coord[1], coord[2],
                c=self.atom_colors['H'], 
                s=self.atom_radii['H'] * 100,
                alpha=0.9, 
                edgecolors='gray', 
                linewidths=1
            )

    # ============================================================
    # 7.9 设置等比例 - _set_equal_aspect_ratio
    # ============================================================
    
    def _set_equal_aspect_ratio(self, ax, x_coords, y_coords, z_coords):
        """
        设置3D图形为等比例显示
        
        【问题】
        matplotlib默认的3D轴可能会拉伸图形
        导致分子看起来变形
        
        【解决】
        1. 计算X、Y、Z的范围
        2. 取最大范围作为统一尺度
        3. 设置相同的显示范围
        """
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        z_range = max(z_coords) - min(z_coords)
        max_range = max(x_range, y_range, z_range)
        
        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2
        z_center = (max(z_coords) + min(z_coords)) / 2
        
        # 添加10%边距
        padding = max_range * 0.1
        limit = max_range / 2 + padding
        
        ax.set_xlim(x_center - limit, x_center + limit)
        ax.set_ylim(y_center - limit, y_center + limit)
        ax.set_zlim(z_center - limit, z_center + limit)


# =========================================================================
# 第八部分：命令行入口 - main()
# =========================================================================
"""
【功能】
  提供交互式命令行界面，支持：
  1. 选择是否使用并行计算
  2. 指定并行进程数
  3. 输入碳原子数
  4. 查看/可视化异构体
  5. 保存结果到文件
"""

def main():
    """
    主函数：交互式命令行入口
    
    【流程】
    1. 显示欢迎信息和系统信息
    2. 配置并行计算参数
    3. 获取碳原子数
    4. 生成异构体
    5. 提供查看选项
    """
    # 全局中断标志
    global _interrupted
    _interrupted = False

    # 设置信号处理器（支持Ctrl+C中断）
    def _signal_handler(signum, frame):
        global _interrupted
        _interrupted = True
        print("\n\n[中断] 正在停止计算，请稍候...")
        # 重新设置信号处理器，允许第二次Ctrl+C强制退出
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _signal_handler)

    # ---------- 显示欢迎信息 ----------
    print("=" * 70)
    print("烷烃同分异构体可视化工具")
    print("使用无标号树算法生成所有异构体")
    print("支持任意碳原子数，自动启用并行计算加速大规模计算")
    print("=" * 70)
    print()

    # ---------- 配置并行计算 ----------
    print("并行计算设置:")
    print(f"  - 可用CPU核心数: {mp.cpu_count()}")
    print(f"  - 默认使用核心数: {max(1, mp.cpu_count() - 1)}")

    # 是否使用并行
    use_parallel_input = input("\n是否使用并行计算? (Y/n, 默认Y): ").strip().lower()
    use_parallel = use_parallel_input != 'n'

    # 指定进程数
    num_workers_input = input(
        f"请指定并行进程数 (默认{max(1, mp.cpu_count() - 1)}): "
    ).strip()
    if num_workers_input:
        try:
            num_workers = int(num_workers_input)
            num_workers = max(1, min(num_workers, mp.cpu_count()))
        except ValueError:
            num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"使用并行计算: {'是' if use_parallel else '否'}")
    if use_parallel:
        print(f"并行进程数: {num_workers}")
    print()

    # ---------- 创建生成器和可视化器 ----------
    visualizer = AlkaneIsomerVisualizer(
        generator=AlkaneIsomerGenerator(use_parallel=use_parallel, num_workers=num_workers)
    )

    # ---------- 获取碳原子数 ----------
    try:
        n = int(input("请输入碳原子数量 (>=1): "))
        if n < 1:
            print("错误：碳原子数量必须 >= 1")
            return
    except ValueError:
        print("错误：请输入有效的数字")
        return

    # ---------- 生成异构体 ----------
    print(f"\n开始生成 C{n} 的同分异构体...")
    if use_parallel and n >= 15:
        print(f"提示: C{n} 可能产生大量异构体，并行计算将显著加速...")
    print("提示: 按 Ctrl+C 可以中断计算")
    print()

    try:
        isomers = visualizer.generator.generate_isomers(n, show_progress=True)
    except (KeyboardInterrupt, Exception):
        # 静默处理中断
        if isinstance(sys.exc_info()[0], KeyboardInterrupt) or _interrupted:
            print("\n\n[中断] 计算已停止")
        else:
            print(f"\n\n[错误] 计算过程中出现错误: {sys.exc_info()[1]}")
        print("感谢使用！")
        return

    # ---------- 显示结果 ----------
    print(f"\nC{n} 共有 {len(isomers)} 个同分异构体")

    # 打印异构体列表（最多显示前20个）
    for i in range(min(len(isomers), 20)):
        print(f"{i+1}. {isomers[i]}")
    if len(isomers) > 20:
        print(f"... 还有 {len(isomers) - 20} 个")

    # ---------- 提供查看选项 ----------
    if len(isomers) > 0:
        print("\n选项:")
        print("1. 可视化单个异构体")
        print("2. 可视化所有异构体")
        print("3. 保存异构体列表到文件")
        print("4. 退出")

        choice = input("\n请选择 (1/2/3/4): ").strip()

        if choice == "1":
            # 可视化单个
            try:
                idx_input = input(f"请输入要可视化的异构体编号 (1-{len(isomers)}): ")
                if idx_input.strip():
                    idx = int(idx_input) - 1
                    if 0 <= idx < len(isomers):
                        print(f"\n正在显示: {visualizer.generator.get_isomer_name(n, idx)}")
                        visualizer.visualize_isomer(n, idx, show=True)
                    else:
                        print("错误：异构体编号超出范围")
                else:
                    print("已取消可视化")
            except ValueError:
                print("错误：请输入有效的数字")

        elif choice == "2":
            # 批量可视化
            print(f"\n正在显示所有 {len(isomers)} 个异构体...")
            if len(isomers) > 100:
                confirm = input(
                    f"警告: 将显示 {len(isomers)} 个异构体，可能需要较长时间。继续? (y/N): "
                ).strip().lower()
                if confirm != 'y':
                    print("已取消")
                    return
            visualizer.visualize_all_isomers(n, show=True)

        elif choice == "3":
            # 保存到文件
            filename = input(f"请输入输出文件名 (默认 C{n}_isomers.txt): ").strip()
            if not filename:
                filename = f"C{n}_isomers.txt"

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"C{n} 同分异构体列表\n")
                    f.write(f"总数: {len(isomers)}\n")
                    f.write("=" * 70 + "\n\n")
                    for i, iso in enumerate(isomers, 1):
                        f.write(f"{i}. {iso}\n")
                print(f"已保存到: {filename}")
            except Exception as e:
                print(f"保存失败: {e}")

        else:
            print("已退出")

    else:
        print("没有可显示的异构体")


# =========================================================================
# 第九部分：并行计算工作函数
# =========================================================================
"""
【背景】
  由于Windows使用spawn进程启动方式，子进程无法访问父进程的变量
  因此需要将这些函数定义为模块级函数（而非类方法）

【函数列表】
  - parse_substrings_parallel: 解析规范字符串（并行版）
  - process_rooted_worker_parallel: 处理有根树（并行版）
  - _canon_to_adjacency_worker: 转换邻接表（并行版）
  - deduplicate_batch_worker_parallel: 批量去重（并行版）
"""

def parse_substrings_parallel(canon_str: str):
    """
    解析规范字符串中的子树（并行版本）
    
    【功能】
    与AlkaneIsomerGenerator.parse_substrings相同
    但定义为模块级函数，可在子进程中调用
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


def process_rooted_worker_parallel(item: tuple) -> str | None:
    """
    并行处理单个根树异构体的工作函数
    
    【功能】
    对应有根树进行对称性处理
    与AlkaneIsomerGenerator._process_single_rooted相同逻辑
    
    【参数】
      item: (canon, sub_sizes, limit) 元组
    
    【返回值】
      处理后的规范字符串，或None
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
    
    【功能】
    与AlkaneIsomerGenerator.canon_to_adjacency相同逻辑
    但定义为模块级函数，可在子进程中调用
    
    【返回值】
      (规范字符串, 邻接表) 元组
    """
    if canon_str == "C":
        return canon_str, {0: []}

    node_counter = [0]
    temp_adj = {}

    def parse_subtree(s: str, parent: int | None = None) -> int:
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
    
    【功能】
    批量处理规范字符串到邻接表的转换
    用于并行去重流程
    
    【参数】
      batch: 规范字符串列表
    
    【返回值】
      [(规范字符串, 邻接表), ...] 列表
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


# =========================================================================
# 程序入口
# =========================================================================

if __name__ == "__main__":
    """
    程序入口点
    
    【执行流程】
    1. 当直接运行本脚本时执行
    2. 调用main()函数启动交互式界面
    3. 等待用户输入并处理
    
    【__name__ == "__main__"的作用】
    - 当本模块被import时不执行
    - 只有直接运行本脚本时才执行
    - 避免在导入时自动运行
    """
    main()
