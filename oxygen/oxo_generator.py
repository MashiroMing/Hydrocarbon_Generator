"""
含氧官能团取代生成器 (Oxo-Substituent Generator)

在碳骨架异构体基础上枚举氧官能团取代位置：
  - -OH  (羟基) : 1 O, 占 1 个 H 位
  - =O   (羰基) : 1 O, 占 2 个 H 位

支持任意碳骨架（烷/烯/炔/环/多环），使用 WL 图哈希 + 同构检查去重。

算法:
  1. 接收碳骨架图列表（nx.Graph，边含 bond_type，来自 GeneratorManager）
  2. 计算每个碳节点的可用 H 位置数 = 4 - bond_load
  3. 多维回溯枚举 (OH, CO) 分配到各 C 的方案
  4. 创建带 oxo_counts 属性的图副本
  5. WL 哈希 + 度签名预过滤 + is_isomorphic 精确去重
"""

import networkx as nx
from typing import List, Tuple, Dict, Set

from utils import _dedup_canon_key, _dedup_degree_signature, dedup_add_to_buckets


class OxoSubstituentGenerator:
    """氧取代基生成器

    用法:
        gen = OxoSubstituentGenerator()
        oxo_results = gen.generate([(mol_type, G), ...], n_o, n_carbon)
        返回: [(mol_type, G_oxo), ...]
        图中节点含 oxo_counts: (n_OH, n_CO) 属性
    """

    def __init__(self):
        pass

    # ==================== 主入口 ====================

    def generate(self, carbon_skeletons: List[Tuple[str, nx.Graph]],
                 n_o: int, n_carbon: int) -> List[Tuple[str, nx.Graph]]:
        """在碳骨架上枚举氧官能团取代

        Args:
            carbon_skeletons: [(mol_type, nx.Graph), ...]
            n_o: O 原子总数 (>= 1)
            n_carbon: 碳原子数

        Returns:
            [(mol_type, oxo_graph), ...]
            若 n_o == 0，原样返回 carbon_skeletons
        """
        if n_o <= 0:
            return carbon_skeletons

        if not carbon_skeletons:
            return []

        dedup_buckets: Dict[str, List[Tuple[str, nx.Graph]]] = {}

        for mol_type, G in carbon_skeletons:
            # 使用 label 统计真实碳数（兼容含 O 骨架节点的图）
            actual_c = sum(1 for n in G.nodes()
                           if G.nodes[n].get('label', 'C') == 'C')
            if actual_c != n_carbon:
                continue

            available_h = self._compute_available_h(G)
            total_available = sum(available_h)

            # 最多能放的 O 原子数：每个 O 至少占 1 个 H 位（OH），最多 2 个（=O）
            # 保守下界：全部为 =O → max_O = total_available // 2
            # 但实际约束更细，此处仅快速过滤
            if total_available < n_o:
                continue

            assignments = self._enumerate_oxo_assignments(available_h, n_o)

            for assignment in assignments:
                H = self._create_oxo_graph(G, assignment)
                self._add_to_buckets(H, mol_type, dedup_buckets)

        results = []
        for bucket in dedup_buckets.values():
            results.extend(bucket)

        return results

    # ==================== 价态计算 (复用 Halide.py 逻辑) ====================

    def _compute_available_h(self, G: nx.Graph) -> List[int]:
        """计算每个节点的可用 H 位置数（C=4, O=2）"""
        available = []
        for node in sorted(G.nodes()):
            label = G.nodes[node].get('label', 'C')
            max_v = 4 if label == 'C' else 2
            bond_load = self._bond_load(G, node)
            n_db = self._count_double_bonds(G, node)
            max_h = max(0, max_v - bond_load)
            if n_db >= 2 and label == 'C':
                max_h = 0  # 累积二烯中间碳
            available.append(max_h)
        return available

    @staticmethod
    def _bond_load(G: nx.Graph, node: int) -> int:
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
        return sum(1 for nb in G.neighbors(node)
                   if G[node][nb].get('bond_type') == 'double')

    # ==================== O 分配枚举 ====================

    @staticmethod
    def _enumerate_oxo_assignments(
            available_h: List[int],
            n_o: int
    ) -> List[Tuple[Tuple[int, int], ...]]:
        """枚举将 n_o 个 O 分配到 N 个 C 节点上的所有方案

        每个节点输出 (n_OH, n_CO):
          - OH: 1 O, 占 1 个 H 位
          - CO: 1 O, 占 2 个 H 位

        约束: n_OH + 2*n_CO ≤ available_h[i]
              Σ (n_OH + n_CO) = n_o
        """
        n = len(available_h)
        results = []

        def backtrack(idx: int, remaining_o: int,
                      current: List[Tuple[int, int]]):
            if idx == n:
                if remaining_o == 0:
                    results.append(tuple(current))
                return

            avail = available_h[idx]
            future_h_capacity = sum(available_h[idx + 1:])

            # 剪枝1：后续 H 容量不足以放剩余 O
            if remaining_o > avail + future_h_capacity:
                return

            # 剪枝2：即使全部放 OH，也放不下
            if remaining_o > future_h_capacity + avail:
                return

            max_co = min(avail // 2, remaining_o)
            for co in range(max_co + 1):
                max_oh = min(avail - 2 * co, remaining_o - co)
                # 当前节点最少 O 数
                min_o_here = max(0, remaining_o - future_h_capacity)
                min_oh = max(0, min_o_here - co)
                if min_oh > max_oh:
                    continue
                for oh in range(min_oh, max_oh + 1):
                    used = oh + co
                    new_remaining = remaining_o - used
                    if new_remaining > future_h_capacity + sum(
                            available_h[idx + 1:]):
                        continue
                    current.append((oh, co))
                    backtrack(idx + 1, new_remaining, current)
                    current.pop()

        backtrack(0, n_o, [])
        return results

    # ==================== 图创建 ====================

    @staticmethod
    def _create_oxo_graph(
            G: nx.Graph,
            oxo_assignment: Tuple[Tuple[int, int], ...]
    ) -> nx.Graph:
        """创建带 oxo_counts 属性的碳骨架图副本"""
        H = nx.Graph()
        sorted_nodes = sorted(G.nodes())

        for node in G.nodes():
            H.add_node(node, label=G.nodes[node].get('label', 'C'))

        for u, v, data in G.edges(data=True):
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

        for idx, (oh, co) in enumerate(oxo_assignment):
            if oh > 0 or co > 0:
                node = sorted_nodes[idx]
                H.nodes[node]['oxo_counts'] = (oh, co)

        return H

    # ==================== 去重逻辑 ====================

    @staticmethod
    def _oxo_label(G: nx.Graph, node: int) -> str:
        """节点标签编码：含 oxo_counts 信息"""
        oc = G.nodes[node].get('oxo_counts', (0, 0))
        return f"C:{oc[0]},{oc[1]}"

    @staticmethod
    def _oxo_sig(G: nx.Graph, node: int) -> tuple:
        """节点特征：用于度签名预过滤"""
        return G.nodes[node].get('oxo_counts', (0, 0))

    @staticmethod
    def _canon_key(G: nx.Graph) -> str:
        """WL 图哈希（含 oxo_counts 信息）"""
        return _dedup_canon_key(G, OxoSubstituentGenerator._oxo_label)

    @staticmethod
    def _degree_signature(G: nx.Graph) -> tuple:
        """含 oxo_counts 的度签名"""
        return _dedup_degree_signature(G, OxoSubstituentGenerator._oxo_sig)

    @staticmethod
    def _add_to_buckets(
            G: nx.Graph, mol_type: str,
            buckets: Dict[str, List[Tuple[str, nx.Graph]]]
    ):
        def node_match(n1, n2):
            return (n1.get('label') == n2.get('label') and
                    n1.get('oxo_counts', (0, 0)) ==
                    n2.get('oxo_counts', (0, 0)))

        dedup_add_to_buckets(
            G, mol_type, buckets, node_match,
            OxoSubstituentGenerator._oxo_label,
            OxoSubstituentGenerator._oxo_sig,
            OxoSubstituentGenerator._canon_key,
            OxoSubstituentGenerator._degree_signature,
        )

    # ==================== 工具函数 ====================

    @staticmethod
    def compute_graph_formula(G: nx.Graph, n_carbon: int = None) -> str:
        """从图计算分子式（含 oxo_counts + halogen_counts + O 骨架）

        C 前缀从图内 label 统计（不信任调用方 n_carbon，参数仅作兼容保留）；
        口径统一委托 utils.graph_formula_parts / format_graph_formula。
        """
        from utils import format_graph_formula
        return format_graph_formula(G)

    @staticmethod
    def get_total_o(G: nx.Graph) -> int:
        """获取图中 O 原子总数"""
        return sum(
            sum(G.nodes[node].get('oxo_counts', (0, 0)))
            for node in G.nodes()
        )

    @staticmethod
    def has_oxygen(G: nx.Graph) -> bool:
        """判断图中是否含 O"""
        for node in G.nodes():
            oh, co = G.nodes[node].get('oxo_counts', (0, 0))
            if oh + co > 0:
                return True
        return False

    @staticmethod
    def describe_oxo_distribution(G: nx.Graph) -> str:
        """获取氧官能团分布描述"""
        parts = []
        for node in sorted(G.nodes()):
            oh, co = G.nodes[node].get('oxo_counts', (0, 0))
            if oh == 0 and co == 0:
                continue
            desc = []
            if oh > 0:
                desc.append(f"OH{oh if oh > 1 else ''}")
            if co > 0:
                desc.append(f"=O{co if co > 1 else ''}")
            parts.append(f"C{node}({','.join(desc)})")
        return ', '.join(parts) if parts else '无氧取代'


# ==================== 交互式命令行界面 ====================

# 分子类型中文名（从 Halide.py 复用或独立定义）
_MOL_CN_MAP = {
    'alkane': '烷烃', 'alkene': '烯烃', 'alkyne': '炔烃',
    'diene': '二烯烃', 'cycloalkane': '环烷烃', 'cycloalkene': '单环烯烃',
    'alkenyl': '烯炔烃', 'triene': '三烯烃', 'tetraene': '四烯烃',
    'polyene': '多烯烃', 'cyclopolyene': '单环多烯烃',
    'multcycloalkane': '多环烷烃', 'multcyclomultalkane': '多环多烯炔烃',
    'benzene': '苯环',
}


def interactive_cli():
    """交互式命令行界面：碳数 → 氧数 → 不饱和度 → 生成含氧有机物"""
    from utils import GeneratorManager, canon_str_to_graph

    print("=" * 65)
    print("    含氧有机物同分异构体生成器 - 交互式命令行")
    print("    支持: -OH (羟基) / =O (羰基)")
    print("=" * 65)
    print()

    # ── 步骤1：输入碳原子数 ──
    while True:
        try:
            n_carbon = int(input("请输入碳原子数 (≥1): ").strip())
            if n_carbon >= 1:
                break
            print("  [!] 碳原子数需 ≥ 1，请重新输入。")
        except ValueError:
            print("  [!] 请输入有效的整数。")

    # ── 步骤2：输入氧原子数 ──
    while True:
        try:
            n_oxygen = int(input(f"请输入氧原子数 (≥0): ").strip())
            if n_oxygen >= 0:
                break
            print("  [!] 氧原子数需 ≥ 0，请重新输入。")
        except ValueError:
            print("  [!] 请输入有效的整数。")

    # ── 步骤3：选择不饱和度 ──
    # 计算可选不饱和度范围（k ≥ 0, H = 2n+2 - 2k ≥ n_oxygen）
    max_k_by_oxygen = (2 * n_carbon + 2 - n_oxygen) // 2  # H ≥ n_oxygen
    max_k_by_valid = 2 * n_carbon + 1  # H ≥ 0，理论最大值
    max_k = max(0, min(max_k_by_oxygen, max_k_by_valid))
    # 显示上限最多到 max_k 或 2n+2，取较小
    display_max = min(max_k, 2 * n_carbon + 2)

    print(f"\n  C{n_carbon} + {n_oxygen}O — 可选不饱和度:")
    print(f"  {'─' * 45}")

    # 不饱和度类型映射
    def _k_description(k, nc):
        h_count = 2 * nc + 2 - 2 * k
        fml = f"CH{h_count}" if nc == 1 else f"C{nc}H{h_count}"
        if k == 0:
            types = "烷烃"
        elif k == 1:
            types = "烯烃 / 环烷烃"
        elif k == 2:
            types = "炔烃 / 二烯烃 / 环烯烃"
        elif k == 3:
            types = "烯炔烃 / 三烯烃 / 环二烯烃"
        elif k == 4:
            types = "多烯多环 (高不饱和)"
        else:
            types = f"{k} 度不饱和"
        return k, fml, h_count, types

    options = []
    for k in range(display_max + 1):
        h_count = 2 * n_carbon + 2 - 2 * k
        if h_count < 0:
            break
        _, fml, _, types = _k_description(k, n_carbon)
        options.append(k)
        h_note = ""
        if h_count < n_oxygen:
            h_note = f" [!] H={h_count} < O={n_oxygen}，不适用"
        print(f"    [{k}]  k={k}: {fml:10s}  ({types}){h_note}")

    print(f"  {'─' * 45}")

    # 过滤掉 H < n_oxygen 的选项
    valid_options = [k for k in options
                     if (2 * n_carbon + 2 - 2 * k) >= n_oxygen]
    if not valid_options:
        print(f"  [!] 碳数 C{n_carbon} 无法容纳 {n_oxygen} 个 O 原子。")
        return

    while True:
        try:
            choice = input(f"\n请选择不饱和度 k ({valid_options[0]}~{valid_options[-1]}): ").strip()
            k_selected = int(choice)
            if k_selected in valid_options:
                break
            print(f"  [!] 请选择有效的不饱和度: {valid_options}")
        except ValueError:
            print("  [!] 请输入有效的整数。")

    n_hydrogen = 2 * n_carbon + 2 - 2 * k_selected
    _, formula_str, _, type_desc = _k_description(k_selected, n_carbon)

    # ── 步骤4：生成碳骨架 ──
    print(f"\n[1/2] 正在生成 {formula_str} ({type_desc}) 的碳骨架...")
    mgr = GeneratorManager()
    raw = mgr.generate_all(n_carbon, n_hydrogen)
    skeletons = []
    for mol_type, iso in raw:
        if isinstance(iso, str):
            G = canon_str_to_graph(iso, mol_type, mgr)
            if G is not None:
                skeletons.append((mol_type, G))
        else:
            skeletons.append((mol_type, iso))

    if not skeletons:
        print(f"  [!] 无法生成 {formula_str} 的碳骨架。")
        return

    # 统计类型
    type_counts = {}
    for mol_type, _ in skeletons:
        type_counts[mol_type] = type_counts.get(mol_type, 0) + 1

    print(f"  -> 获得 {len(skeletons)} 个碳骨架同分异构体。")
    print("     分类：")
    for mt in sorted(type_counts.keys()):
        cn_name = _MOL_CN_MAP.get(mt, mt)
        print(f"       {cn_name}({mt}): {type_counts[mt]} 个")

    if n_oxygen == 0:
        print(f"\n  [!] 未指定氧取代，仅展示碳骨架：")
        gen = OxoSubstituentGenerator()
        print("-" * 65)
        for i, (mol_type, G) in enumerate(skeletons, 1):
            f = gen.compute_graph_formula(G, n_carbon)
            cn_name = _MOL_CN_MAP.get(mol_type, mol_type)
            print(f"  #{i:3d}  {f:12s}  [{cn_name}]")
        print("-" * 65)
        print("\n感谢使用，再见！")
        return

    # ── 步骤5：生成含氧有机物 ──
    gen = OxoSubstituentGenerator()
    print(f"\n[2/2] 正在生成 {formula_str} 的含 {n_oxygen}O 同分异构体...")
    results = gen.generate(skeletons, n_oxygen, n_carbon)

    if not results:
        print("  [!] 未生成任何结果。")
        return

    print(f"  -> 共生成 {len(results)} 个含氧同分异构体。")

    # ── 结果展示 ──
    print(f"\n  结果展示：")
    print("-" * 75)
    for i, (mol_type, G) in enumerate(results, 1):
        formula = gen.compute_graph_formula(G, n_carbon)
        dist_desc = gen.describe_oxo_distribution(G)
        cn_name = _MOL_CN_MAP.get(mol_type, mol_type)
        print(f"  #{i:3d}  {formula:14s}  [{cn_name:10s}]  {dist_desc}")

    print("-" * 75)
    print(f"\n  总计: {len(results)} 个含氧同分异构体。")

    # 统计官能团类型
    oh_only = sum(1 for _, G in results
                  if all((co == 0) for _, (_, co) in
                         [(n, G.nodes[n].get('oxo_counts', (0, 0)))
                          for n in G.nodes()]))
    co_only = sum(1 for _, G in results
                  if all((oh == 0) for _, (oh, _) in
                         [(n, G.nodes[n].get('oxo_counts', (0, 0)))
                          for n in G.nodes()])
                  and gen.has_oxygen(G))
    mixed = sum(1 for _, G in results
                if any((oh > 0 and co > 0) for _, (oh, co) in
                       [(n, G.nodes[n].get('oxo_counts', (0, 0)))
                        for n in G.nodes()]))

    if n_oxygen > 0:
        print(f"    纯醇/醚类 (-OH):  {oh_only} 个")
        print(f"    纯羰基类 (=O):    {co_only} 个")
        print(f"    混合 (OH+=O):     {mixed} 个")
    print()

    # ── 步骤6：查看详情 ──
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
                    oh, co = G.nodes[node].get('oxo_counts', (0, 0))
                    oxo_parts = []
                    if oh > 0:
                        oxo_parts.append(f"OH={oh}")
                    if co > 0:
                        oxo_parts.append(f"=O={co}")
                    oxo_info = f" ({', '.join(oxo_parts)})" if oxo_parts else ""
                    neighbors = list(G.neighbors(node))
                    edge_info = []
                    for nb in neighbors:
                        bt = G[node][nb].get('bond_type', 'single')
                        edge_info.append(f"C{nb}({bt})")
                    bond_load = gen._bond_load(G, node)
                    h_count = max(0, 4 - bond_load - oh - 2 * co)
                    h_oh = oh
                    print(f"    C{node}:{oxo_info} C-H={h_count}, O-H={h_oh}, 连接: {edge_info}")
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
