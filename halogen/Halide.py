"""
卤代烃同分异构体生成器

支持氟(F)、氯(Cl)、溴(Br)、碘(I) 四种卤素的单取代及混合取代，
在碳骨架异构体基础上枚举卤素取代位置。
使用 WL 图哈希 + 同构检查确保去重。

算法：
  1. 接收碳骨架图列表（nx.Graph，边含 bond_type 属性）
  2. 计算每个碳节点的可用H位置数 = 4 - bond_load
  3. 多维回溯：枚举将 (F, Cl, Br, I) 分配到N个碳上的所有方案
  4. 对每个方案创建带 halogen_counts 属性的图副本
  5. WL哈希 + 度签名预过滤 + is_isomorphic 精确去重
"""

import re
import networkx as nx
from typing import List, Dict, Tuple

from utils import _dedup_canon_key, _dedup_degree_signature, dedup_add_to_buckets

# 卤素元组索引: (F, Cl, Br, I)
HALOGEN_INDEX = {'F': 0, 'Cl': 1, 'Br': 2, 'I': 3}
HALOGEN_NAMES = ['F', 'Cl', 'Br', 'I']
# 分子式输出顺序（字母序）: Br → Cl → F → I
HALOGEN_FORMULA_ORDER = ['Br', 'Cl', 'F', 'I']
HALOGEN_COLORS = {'F': '#90E050', 'Cl': '#1FF01F', 'Br': '#A62929', 'I': '#940094'}


def parse_halogen_spec(user_input: str) -> Tuple[int, int, int, int]:
    """解析用户输入的卤素规格字符串，返回 (f, cl, br, i) 元组

    支持两种格式：
      - 化学式格式: "F2Cl1Br3" 或 "Br2ClF"
      - 冒号格式:   "F:2, Cl:1, Br:3" 或 "F:3"

    Args:
        user_input: 用户输入的卤素规格字符串

    Returns:
        (f_count, cl_count, br_count, i_count)
    """
    user_input = user_input.strip()
    if not user_input or user_input.lower() in ('0', 'none', '无'):
        return (0, 0, 0, 0)

    counts = {'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}

    # 尝试冒号格式: F:2, Cl:1
    if ':' in user_input:
        parts = re.split(r'[,;，；\s]+', user_input)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            match = re.match(r'(F|Cl|Br|I)\s*:\s*(\d+)', part, re.IGNORECASE)
            if match:
                elem = match.group(1).capitalize()
                count = int(match.group(2))
                if count > 0:
                    counts[elem] += count
        return (counts['F'], counts['Cl'], counts['Br'], counts['I'])

    # 尝试化学式格式: F2Cl1Br3
    pattern = re.finditer(r'(F|Cl|Br|I)(\d*)', user_input, re.IGNORECASE)
    for match in pattern:
        elem = match.group(1).capitalize()
        digits = match.group(2)
        count = int(digits) if digits else 1
        if count > 0:
            counts[elem] += count

    return (counts['F'], counts['Cl'], counts['Br'], counts['I'])


def format_halogen_spec(counts: Tuple[int, int, int, int]) -> str:
    """将卤素计数元组格式化为易读字符串

    Args:
        counts: (f, cl, br, i)

    Returns:
        如 "F2Cl1"、"无"、"Br3I1" 等
    """
    parts = []
    for elem in HALOGEN_NAMES:
        idx = HALOGEN_INDEX[elem]
        if counts[idx] > 0:
            parts.append(f"{elem}{counts[idx] if counts[idx] > 1 else ''}")
    return ''.join(parts) if parts else '无'


class HalogenSubstitutionGenerator:
    """卤素取代生成器

    用法:
        gen = HalogenSubstitutionGenerator()
        halogenated = gen.generate([(mol_type, G), ...], halogen_spec, n_carbon)
        返回: [(mol_type, G_with_halogen), ...]

        其中 halogen_spec = (f, cl, br, i)
    """

    def __init__(self):
        pass

    # ==================== 主入口 ====================

    def generate(self, carbon_skeletons: List[Tuple[str, nx.Graph]],
                 halogen_spec: Tuple[int, int, int, int],
                 n_carbon: int) -> List[Tuple[str, nx.Graph]]:
        """
        在碳骨架上枚举卤素取代位置

        Args:
            carbon_skeletons: [(mol_type, nx.Graph), ...]
            halogen_spec: (f_total, cl_total, br_total, i_total)
            n_carbon: 碳原子数（用于验证）

        Returns:
            [(mol_type, halogenated_graph), ...]
            如果所有卤素计数为 0，原样返回 carbon_skeletons
        """
        total_halogen = sum(halogen_spec)
        if total_halogen <= 0:
            return carbon_skeletons

        if not carbon_skeletons:
            return []

        # 使用以 WL 哈希为键的去重桶
        dedup_buckets: Dict[str, List[Tuple[str, nx.Graph]]] = {}

        for mol_type, G in carbon_skeletons:
            # 使用 label 统计真实碳数（兼容含 O 骨架节点的图）
            actual_c = sum(1 for n in G.nodes()
                           if G.nodes[n].get('label', 'C') == 'C')
            if actual_c != n_carbon:
                continue

            # 计算每个碳的可用H位置
            available_h = self._compute_available_h(G)
            total_available = sum(available_h)

            if total_available < total_halogen:
                continue  # 该骨架容纳不下这么多卤素

            # 枚举所有卤素分配方案
            assignments = self._enumerate_halogen_assignments(
                available_h, halogen_spec
            )

            for assignment in assignments:
                # 创建带 halogen_counts 属性的图
                H = self._create_halogenated_graph(G, assignment)
                self._add_to_buckets(H, mol_type, dedup_buckets)

        # 展平桶为列表
        results = []
        for bucket in dedup_buckets.values():
            results.extend(bucket)

        return results

    # ==================== 辅助计算 ====================

    def _compute_available_h(self, G: nx.Graph) -> List[int]:
        """计算每个节点可用的H位置数（C=4, O=2，含 oxo_counts 修正）"""
        available = []
        for node in sorted(G.nodes()):
            label = G.nodes[node].get('label', 'C')
            max_valence = 2 if label == 'O' else 4

            bond_load = self._bond_load(G, node)
            n_db = self._count_double_bonds(G, node)

            max_h = max(0, max_valence - bond_load)

            # 累积二烯中间碳：参与 ≥2 个双键，不能再接 H/卤素
            if n_db >= 2 and label == 'C':
                max_h = 0

            # 扣除已有的 oxo 取代基：-OH 占1位，=O 占2位
            oh, co = G.nodes[node].get('oxo_counts', (0, 0))
            max_h -= (oh + 2 * co)
            if max_h < 0:
                max_h = 0

            available.append(max_h)
        return available

    @staticmethod
    def _bond_load(G: nx.Graph, node: int) -> int:
        """计算节点的键数负载（单键=1, 双键=2, 三键=3）"""
        total = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            if bt == 'triple':
                total += 3
            elif bt == 'double':
                total += 2
            else:
                total += 1
        return total

    @staticmethod
    def _count_double_bonds(G: nx.Graph, node: int) -> int:
        """计算节点参与的双键数"""
        return sum(1 for nb in G.neighbors(node)
                   if G[node][nb].get('bond_type') == 'double')

    # ==================== 多维卤素分配枚举 ====================

    def _enumerate_halogen_assignments(
            self,
            available_h: List[int],
            target_counts: Tuple[int, int, int, int]
    ) -> List[Tuple[Tuple[int, int, int, int], ...]]:
        """
        将 (F, Cl, Br, I) 四种卤素分配到 N 个碳节点上

        使用多维回溯 + 剪枝，每个碳节点的卤素总数 ≤ available_h[i]

        Args:
            available_h: 每个碳节点的可用位置数
            target_counts: (target_f, target_cl, target_br, target_i)

        Returns:
            分配方案列表，每个方案为 ((f0,cl0,br0,i0), (f1,cl1,br1,i1), ...)
        """
        results = []
        n = len(available_h)

        def backtrack(idx: int, remaining: Tuple[int, int, int, int],
                      current: List[Tuple[int, int, int, int]]):
            if idx == n:
                if all(c == 0 for c in remaining):
                    results.append(tuple(current))
                return

            avail = available_h[idx]
            rem_f, rem_cl, rem_br, rem_i = remaining
            total_remaining = sum(remaining)

            # 剪枝1：剩余位置容量不足
            remaining_capacity = sum(available_h[idx:])
            if total_remaining > remaining_capacity:
                return

            # 剪枝2：即使后续碳全部填满也无法满足约束
            future_capacity = sum(available_h[idx + 1:])
            min_f = max(0, rem_f - future_capacity)
            min_cl = max(0, rem_cl - future_capacity)
            min_br = max(0, rem_br - future_capacity)
            min_i = max(0, rem_i - future_capacity)

            if min_f + min_cl + min_br + min_i > avail:
                return  # 当前碳容量不足以满足最小需求

            max_f = min(avail, rem_f)
            for f in range(min_f, max_f + 1):
                max_cl = min(avail - f, rem_cl)
                for cl in range(min_cl, max_cl + 1):
                    max_br = min(avail - f - cl, rem_br)
                    for br in range(min_br, max_br + 1):
                        max_i = min(avail - f - cl - br, rem_i)
                        for i_val in range(min_i, max_i + 1):
                            if f + cl + br + i_val > avail:
                                continue
                            new_remaining = (
                                rem_f - f,
                                rem_cl - cl,
                                rem_br - br,
                                rem_i - i_val
                            )
                            current.append((f, cl, br, i_val))
                            backtrack(idx + 1, new_remaining, current)
                            current.pop()

        backtrack(0, target_counts, [])
        return results

    # ==================== 图创建 ====================

    @staticmethod
    def _create_halogenated_graph(G: nx.Graph,
                                   halogen_assignment: Tuple[Tuple[int, int, int, int], ...]
                                   ) -> nx.Graph:
        """创建带 halogen_counts 属性的碳骨架图副本

        Args:
            G: 原始碳骨架图
            halogen_assignment: 每个碳节点的卤素分配，
                如 ((f0,cl0,br0,i0), (f1,cl1,br1,i1), ...)

        Returns:
            新的 nx.Graph，节点含 halogen_counts 属性
        """
        H = nx.Graph()
        sorted_nodes = sorted(G.nodes())

        for node in G.nodes():
            H.add_node(node, label=G.nodes[node].get('label', 'C'))
            # 保留 oxo_counts（含氧取代基信息）
            oxo = G.nodes[node].get('oxo_counts', None)
            if oxo is not None:
                H.nodes[node]['oxo_counts'] = oxo

        for u, v, data in G.edges(data=True):
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

        for idx, halogen_tuple in enumerate(halogen_assignment):
            if sum(halogen_tuple) > 0:
                node = sorted_nodes[idx]
                H.nodes[node]['halogen_counts'] = halogen_tuple

        return H

    # ==================== 去重逻辑 ====================

    @staticmethod
    def _halogen_label(G: nx.Graph, node: int) -> str:
        """节点标签编码：含 halogen_counts 信息"""
        hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
        return f"C:{hc[0]},{hc[1]},{hc[2]},{hc[3]}"

    @staticmethod
    def _halogen_sig(G: nx.Graph, node: int) -> tuple:
        """节点特征：用于度签名预过滤"""
        return G.nodes[node].get('halogen_counts', (0, 0, 0, 0))

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        """计算 WL 图哈希（含 halogen_counts 信息）"""
        return _dedup_canon_key(G, HalogenSubstitutionGenerator._halogen_label)

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        """计算含 halogen_counts 信息的度签名，用于同构检查的预过滤"""
        return _dedup_degree_signature(G, HalogenSubstitutionGenerator._halogen_sig)

    @staticmethod
    def _add_to_buckets(G: nx.Graph, mol_type: str,
                         buckets: Dict[str, List[Tuple[str, nx.Graph]]]):
        """将图添加到去重桶中（同桶内逐一做同构检查）"""
        def node_match(n1, n2):
            return (n1.get('label') == n2.get('label') and
                    n1.get('halogen_counts', (0, 0, 0, 0)) ==
                    n2.get('halogen_counts', (0, 0, 0, 0)) and
                    n1.get('oxo_counts', (0, 0)) ==
                    n2.get('oxo_counts', (0, 0)))

        dedup_add_to_buckets(
            G, mol_type, buckets, node_match,
            HalogenSubstitutionGenerator._halogen_label,
            HalogenSubstitutionGenerator._halogen_sig,
            HalogenSubstitutionGenerator._canon_key,
            HalogenSubstitutionGenerator._degree_signature,
        )

    # ==================== 工具函数 ====================

    @staticmethod
    def compute_graph_formula(G: nx.Graph, n_carbon: int = None) -> str:
        """从含 halogen_counts 属性的图计算实际分子式

        卤素按字母序排列: Br → Cl → F → I。
        C 前缀从图内 label 统计（不信任调用方 n_carbon，参数仅作兼容保留）；
        口径统一委托 utils.graph_formula_parts / format_graph_formula
        （兼容含 oxo_counts / 骨架 O 节点的图）。

        Args:
            G: 碳骨架图（可能含 halogen_counts 节点属性）
            n_carbon: 兼容参数（已弃用）

        Returns:
            分子式字符串，如 "C2H5BrCl", "C3H5Br2ClF", "CCl2F2"
        """
        from utils import format_graph_formula
        return format_graph_formula(G)

    @staticmethod
    def get_halogen_counts(G: nx.Graph) -> Tuple[int, int, int, int]:
        """获取图中各卤素原子总数

        Returns:
            (total_f, total_cl, total_br, total_i)
        """
        f_total = cl_total = br_total = i_total = 0
        for node in G.nodes():
            hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
            f_total += hc[0]
            cl_total += hc[1]
            br_total += hc[2]
            i_total += hc[3]
        return (f_total, cl_total, br_total, i_total)

    @staticmethod
    def has_halogen(G: nx.Graph) -> bool:
        """判断图中是否含卤素"""
        for node in G.nodes():
            hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
            if sum(hc) > 0:
                return True
        return False

    @staticmethod
    def describe_halogen_distribution(G: nx.Graph) -> str:
        """获取图中卤素分布的易读描述

        Args:
            G: 含 halogen_counts 的图

        Returns:
            如 "C0(F2,Cl1), C2(Br1)" 或 "无卤素取代"
        """
        parts = []
        for node in sorted(G.nodes()):
            hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
            if sum(hc) == 0:
                continue
            atom_parts = []
            for idx, elem in enumerate(HALOGEN_NAMES):
                if hc[idx] > 0:
                    atom_parts.append(f"{elem}{hc[idx] if hc[idx] > 1 else ''}")
            parts.append(f"C{node}({','.join(atom_parts)})")
        return ', '.join(parts) if parts else '无卤素取代'


# ==================== 模块级便捷函数 ====================

def enumerate_halogen_substitutions(
        carbon_skeletons: List[Tuple[str, nx.Graph]],
        halogen_spec: Tuple[int, int, int, int],
        n_carbon: int) -> List[Tuple[str, nx.Graph]]:
    """便捷函数：在碳骨架上枚举卤素取代

    Args:
        carbon_skeletons: [(mol_type, nx.Graph), ...]
        halogen_spec: (f_total, cl_total, br_total, i_total)
        n_carbon: 碳原子数

    Returns:
        [(mol_type, halogenated_graph), ...]
    """
    gen = HalogenSubstitutionGenerator()
    return gen.generate(carbon_skeletons, halogen_spec, n_carbon)


# ==================== 交互式命令行界面 ====================

# 分子类型中文名（用于显示）
_MOL_CN_MAP = {
    'alkane': '烷烃',
    'alkene': '烯烃',
    'alkyne': '炔烃',
    'diene': '二烯烃',
    'cycloalkane': '环烷烃',
    'cycloalkene': '单环烯烃',
    'alkenyl': '烯炔烃',
    'triene': '三烯烃',
    'tetraene': '四烯烃',
    'polyene': '多烯烃',
    'cyclopolyene': '单环多烯烃',
    'multcycloalkane': '多环烷烃',
    'multcyclomultalkane': '多环多烯炔烃',
    'benzene': '苯环',
}


def _build_carbon_skeletons_from_formula(formula_str: str):
    """根据分子式构建所有碳骨架同分异构体

    使用 utils.py 的 parse_molecule_input 解析分子式，
    GeneratorManager 生成所有不饱和度的碳骨架，
    统一转为 networkx.Graph 格式。

    Args:
        formula_str: 分子式如 "C4H6"、"C3H8"

    Returns:
        (skeletons, n_carbon, n_hydrogen, mol_types)
        skeletons: [(mol_type, nx.Graph), ...]
        n_carbon: 碳原子数
        n_hydrogen: 氢原子数（用于显示参考）
        mol_types: 分子类型列表（用于显示分类信息）

    如果解析或生成失败，返回 (None, None, None, None)
    """
    from utils import parse_molecule_input, GeneratorManager, canon_str_to_graph

    mol_types, n_carbon, n_hydrogen, error = parse_molecule_input(formula_str)
    if error:
        return None, n_carbon, n_hydrogen, error

    mgr = GeneratorManager()
    all_skeletons = []

    for mol_type in mol_types:
        try:
            isomers = mgr.generate(mol_type, n_carbon, n_hydrogen)
        except Exception:
            continue

        if not isomers:
            continue

        for iso in isomers:
            if isinstance(iso, str):
                # 烷烃规范字符串 → nx.Graph
                G = canon_str_to_graph(iso, mol_type, mgr)
                if G is not None:
                    all_skeletons.append((mol_type, G))
            else:
                # multcyclomultalkane 已直接返回 nx.Graph
                all_skeletons.append((mol_type, iso))

    if not all_skeletons:
        return None, n_carbon, n_hydrogen, "无法生成碳骨架，请检查分子式是否正确。"

    return all_skeletons, n_carbon, n_hydrogen, None


def interactive_cli():
    """交互式命令行界面：让用户输入参数并查看卤代烃生成结果"""
    print("=" * 65)
    print("    卤代烃同分异构体生成器 - 交互式命令行")
    print("    支持: F(氟) / Cl(氯) / Br(溴) / I(碘)")
    print("=" * 65)
    print()
    print("  支持分子类型:")
    print("    • 烷烃 CnH2n+2     • 烯烃/环烷烃 CnH2n")
    print("    • 炔烃/二烯烃 CnH2n-2   • 烯炔/三烯烃 CnH2n-4")
    print("    • 多环烷烃 • 多环多烯炔烃 等")
    print()

    # 步骤1：输入分子式
    while True:
        formula_str = input("请输入碳骨架分子式（如 C4H6、C3H8）: ").strip()
        if not formula_str:
            print("  [!] 输入不能为空。")
            continue

        skeletons, n_carbon, n_hydrogen, error = _build_carbon_skeletons_from_formula(
            formula_str
        )
        if error or skeletons is None:
            print(f"  [!] {error}")
            continue
        break

    # 按分子类型统计
    type_counts = {}
    for mol_type, _ in skeletons:
        type_counts[mol_type] = type_counts.get(mol_type, 0) + 1

    cn_formula = f"CH{n_hydrogen}" if n_carbon == 1 else f"C{n_carbon}H{n_hydrogen}"
    print(f"\n[1/4] 分子式: {cn_formula}")
    print(f"  -> 碳原子数: {n_carbon}，氢原子数: {n_hydrogen}")
    print(f"  -> 获得 {len(skeletons)} 个碳骨架同分异构体。")
    print("     分类：")
    for mt in sorted(type_counts.keys()):
        cn_name = _MOL_CN_MAP.get(mt, mt)
        print(f"       {cn_name}({mt}): {type_counts[mt]} 个")

    # 步骤2：输入卤素规格
    # 先计算最大可取代H数
    gen = HalogenSubstitutionGenerator()
    max_h_all = 0
    min_h_all = float('inf')
    for _, G in skeletons:
        available = gen._compute_available_h(G)
        total_avail = sum(available)
        max_h_all = max(max_h_all, total_avail)
        min_h_all = min(min_h_all, total_avail)

    print(f"\n  碳骨架可取代 H 范围: {min_h_all} ~ {max_h_all} 个（因不饱和度而异）。")
    print("  支持格式:")
    print("    - 化学式: F2Cl1Br3  (大写首字母, 无数字=1个)")
    print("    - 冒号式: F:2, Cl:1 (逗号或空格分隔)")
    print("    - 留空 / '0' 表示不取代（仅展示碳骨架）")

    while True:
        user_input = input("\n请输入卤素规格: ").strip()
        if not user_input or user_input.lower() in ('0', 'none', '无'):
            halogen_spec = (0, 0, 0, 0)
            break
        try:
            halogen_spec = parse_halogen_spec(user_input)
            total_halo = sum(halogen_spec)
            if total_halo == 0:
                print("  [!] 未识别到有效卤素，请重新输入。")
                continue
            if total_halo > max_h_all:
                print(f"  [!] 卤素总数 ({total_halo}) 超过所有碳骨架可容纳的上限 ({max_h_all})。\n"
                      f"      部分碳骨架可能仍可容纳较少卤素，请输入更小的值。")
                continue
            break
        except Exception:
            print("  [!] 格式错误，请使用如 F2Cl1 或 F:2, Cl:1 的格式。")

    spec_str = format_halogen_spec(halogen_spec)
    print(f"\n  -> 卤素规格: {spec_str}")
    print(f"     F={halogen_spec[0]}, Cl={halogen_spec[1]}, "
          f"Br={halogen_spec[2]}, I={halogen_spec[3]}")

    if sum(halogen_spec) == 0:
        print("\n  [!] 未指定卤素取代，仅展示碳骨架：")
        print("-" * 65)
        for i, (mol_type, G) in enumerate(skeletons, 1):
            formula = gen.compute_graph_formula(G, n_carbon)
            cn_name = _MOL_CN_MAP.get(mol_type, mol_type)
            print(f"  #{i:3d}  {formula:12s}  [{cn_name}]")
        print("-" * 65)
        print("\n感谢使用，再见！")
        return

    # 步骤3：生成卤代烃
    print(f"\n[2/4] 正在生成 {cn_formula} 的卤代烃同分异构体...")
    results = gen.generate(skeletons, halogen_spec, n_carbon)

    if not results:
        print("  [!] 未生成任何结果。可能所有碳骨架的H位置不足以容纳指定的卤素。")
        return

    print(f"  -> 共生成 {len(results)} 个卤代烃同分异构体。")

    # 步骤4：展示结果
    print(f"\n[3/4] 结果展示：")
    print("-" * 65)
    for i, (mol_type, G) in enumerate(results, 1):
        formula = gen.compute_graph_formula(G, n_carbon)
        dist_desc = gen.describe_halogen_distribution(G)
        cn_name = _MOL_CN_MAP.get(mol_type, mol_type)
        print(f"  #{i:3d}  {formula:18s}  [{cn_name:8s}]  取代: {dist_desc}")

    print("-" * 65)
    print(f"\n  总计: {len(results)} 个卤代烃同分异构体。")
    print()

    # 步骤5：询问是否查看图结构详情
    while True:
        choice = input("是否查看某个异构体的图结构详情？(输入编号 / q 退出): ").strip()
        if choice.lower() == 'q':
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                mol_type, G = results[idx]
                formula = gen.compute_graph_formula(G, n_carbon)
                cn_name = _MOL_CN_MAP.get(mol_type, mol_type)
                print(f"\n  --- #{idx + 1}: {formula} [{cn_name}] 图结构 ---")
                print(f"  节点数: {G.number_of_nodes()}")
                print(f"  边数: {G.number_of_edges()}")
                print(f"  节点详情:")
                for node in sorted(G.nodes()):
                    hc = G.nodes[node].get('halogen_counts', (0, 0, 0, 0))
                    halogen_str = []
                    for hidx, elem in enumerate(HALOGEN_NAMES):
                        if hc[hidx] > 0:
                            halogen_str.append(f"{elem}={hc[hidx]}")
                    halogen_info = f" ({', '.join(halogen_str)})" if halogen_str else ""
                    neighbors = list(G.neighbors(node))
                    edge_info = []
                    for nb in neighbors:
                        bt = G[node][nb].get('bond_type', 'single')
                        edge_info.append(f"C{nb}({bt})")
                    print(f"    C{node}:{halogen_info} 连接: {edge_info}")
                print(f"  边列表:")
                for u, v, data in G.edges(data=True):
                    print(f"    C{u} - C{v}  [{data.get('bond_type', 'single')}]")
                print()
            else:
                print(f"  [!] 编号需在 1~{len(results)} 之间。")
        except ValueError:
            print("  [!] 请输入有效编号或 'q' 退出。")

    print("\n感谢使用，再见！")


# ==================== 程序入口 ====================

if __name__ == "__main__":
    interactive_cli()
