"""
Nitrogen/generator.py — N 系树机制生成器（纯 C/H/N，中性八隅律 C=4 / N=3）

设计哲学：与 O 系"骨架 k 下探 + 机制组合 + 公式门禁 + 碰撞安全去重"同构，
但 N 系的第一阶段采用更直接的三种机制（骨架全部来自内部预言机自举）：

  机制 A — C→N 替换（骨架角色 N）：
      在 (nC+nN) 碳骨架中选 nN 个 C 替换为 N（键型不变，C 4 价 → N 3 价，
      H 自动 -1/位）。
      覆盖：吡啶/吡咯/咪唑等含 N 杂环、叔胺中心（N 3 键）、亚胺 C=N、
            腈 -C≡N（末端炔烃替换）、偶氮 -N=N-（烯烃双键两端替换）等。

  机制 B — 端基 N 挂载（取代基角色 N）：
      在 nC 碳骨架的可用 H 位上挂 N 端基：
        -NH₂（占 1 个 H 位，k+0） /  =NH（占 2 个 H 位，k+1） /
        -C≡N（占 3 个 H 位，k+2）
      覆盖：伯胺、亚胺端基、腈端基。

  机制 C — 桥 N 插入（桥角色 N）：
      在 nC 碳骨架的 C-C 键上插入 N：
        -NH-（C-C → C-NH-C，k+0） /  -N=C-（C-C → C-N=C，k+1）
      覆盖：仲胺桥、亚胺桥。

所有机制产物经 **公式门禁**（formula_matches_input，H/C/N 严格一致）
+ **碰撞安全去重**（WL 哈希 → 列表 → VF2 全属性同构）收口。

完备性说明：树机制覆盖常见官能团；含 N 杂环等全部结构由
atomic_matrix_n.py 预言机（完备枚举）兜底，validate.py 负责
"树 ⊆ 预言机"与覆盖率报告。
"""

import networkx as nx
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

from .formula import (graph_formula_parts, format_graph_formula,
                      formula_matches_input, dbe_from_graph)
from .atomic_matrix_n import AtomicMatrixNGenerator, _seen_add, _graphs_isomorphic


# ============ 节点/边工具 ============

def _h_available(G: nx.Graph, node) -> int:
    """节点的可用 H 位（= max_v - bond_load）"""
    label = G.nodes[node].get('label', 'C')
    max_v = 4 if label == 'C' else 3
    used = 0
    for nb in G.neighbors(node):
        bt = G[node][nb].get('bond_type', 'single')
        used += 3 if bt == 'triple' else (2 if bt == 'double' else 1)
    return max(0, max_v - used)


def _carbon_skeletons(n_c: int, max_atoms: int = 10) -> List[nx.Graph]:
    """用内部预言机自举枚举 n_c 碳的纯碳骨架（全不饱和度，连通、价态合法）"""
    gen = AtomicMatrixNGenerator(n_c, 0, max_atoms=max_atoms)
    if not gen._can_enumerate():
        return []
    return gen._generate_all_graphs()


# ============ 机制 A：C→N 替换 ============

def _mechanism_replace(skeletons: List[nx.Graph], n_n: int,
                       add: Callable[[nx.Graph], None]) -> None:
    """在 (nC+nN) 碳骨架中选 nN 个 C 替换为 N"""
    for G in skeletons:
        c_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'C']
        if len(c_nodes) < n_n:
            continue
        for combo in combinations(c_nodes, n_n):
            H = G.copy()
            for node in combo:
                H.nodes[node]['label'] = 'N'
            add(H)


# ============ 机制 B：端基 N 挂载 ============

# 端基类型（所需 H 位 / 对 k 的贡献）
_TERMINAL_REQ = [1, 2, 3]   # -NH₂ / =NH / -C≡N
_TERMINAL_K   = [0, 1, 2]


def _mechanism_terminal(skeletons: List[nx.Graph], n_n: int,
                        n_carbon: int, n_hydrogen: int,
                        add: Callable[[nx.Graph], None]) -> None:
    """在 nC 碳骨架的可用 H 位上挂 N 端基

    -NH₂: 需 1 个 H 位，k+0；  =NH: 需 2 个 H 位，k+1；  -C≡N: 需 3 个 H 位，k+2。
    每个骨架按 k_skel 与目标 k 的差 delta 解 (a, b, c)（NH₂/=NH/C≡N 个数）：
      a + b + c = n_n 且 b + 2c = delta。
    位置分配：逐类型放固定数量，同类型组合、不同类型依次排列。
    """
    k_target = n_carbon - n_hydrogen / 2 + n_n / 2 + 1
    for G in skeletons:
        k_skel = dbe_from_graph(G)
        delta = int(round(k_target - k_skel))  # 合法 N 系分子式 DBE 必为整数
        if delta < 0 or delta > 2 * n_n:
            continue
        # 解整数方程 a+b+c=n_n, b+2c=delta
        solutions = []
        for c in range(0, n_n + 1):
            b = delta - 2 * c
            a = n_n - b - c
            if 0 <= b and a >= 0:
                solutions.append((a, b, c))
        if not solutions:
            continue

        def place(assign, counts, type_idx):
            """按固定计数逐类型放置端基"""
            if type_idx == 3:
                H = G.copy()
                for node, t_idx in assign:
                    if t_idx == 0:      # -NH₂
                        H.add_node(f"tN{node}", label='N')
                        H.add_edge(node, f"tN{node}", bond_type='single')
                    elif t_idx == 1:    # =NH
                        H.add_node(f"tN{node}", label='N')
                        H.add_edge(node, f"tN{node}", bond_type='double')
                    else:               # -C≡N
                        H.add_node(f"tN{node}", label='N')
                        H.add_edge(node, f"tN{node}", bond_type='triple')
                add(H)
                return
            cnt = counts[type_idx]
            if cnt == 0:
                place(assign, counts, type_idx + 1)
                return
            req = _TERMINAL_REQ[type_idx]
            used_nodes = {n for n, _ in assign}
            free = [n for n in G.nodes()
                    if n not in used_nodes and _h_available(G, n) >= req]
            for combo in combinations(free, cnt):
                new_assign = assign + [(n, type_idx) for n in combo]
                place(new_assign, counts, type_idx + 1)

        for sol in solutions:
            place([], sol, 0)


# ============ 机制 C：桥 N 插入 ============

def _mechanism_bridge(skeletons: List[nx.Graph], n_n: int,
                      n_carbon: int, n_hydrogen: int,
                      add: Callable[[nx.Graph], None]) -> None:
    """在 nC 碳骨架的 C-C 键上插入桥 N

    -NH-（k+0，C-C → C-NH-C）与 -N=C-（k+1，C-C → C-N=C）。
    桥数 = n_n（每个桥消耗 1 个 N），其中双键桥数 = delta = k_target - k_skel。
    边组合 + 双键标记组合全枚举，重复由门禁去重兜底。
    """
    k_target = n_carbon - n_hydrogen / 2 + n_n / 2 + 1
    for G in skeletons:
        k_skel = dbe_from_graph(G)
        delta = int(round(k_target - k_skel))  # 合法 N 系分子式 DBE 必为整数
        if delta < 0 or delta > n_n:
            continue
        b_d = delta        # 双键桥（-N=C-）数量
        n_s = n_n - b_d    # 单键桥（-NH-）数量
        edges = list(G.edges())
        if not edges and n_n > 0:
            continue
        # 选 n_n 条边（combinations 无重复），标记其中 b_d 条为双键桥
        for combo in combinations(edges, n_n):
            for dbl_mask in combinations(range(n_n), b_d):
                H = G.copy()
                for k, (u, v) in enumerate(combo):
                    if not H.has_edge(u, v):
                        continue  # 该边可能已被前一条桥拆掉（同一骨架上多桥）
                    H.remove_edge(u, v)
                    n_name = f"bN{k}"
                    H.add_node(n_name, label='N')
                    if k in dbl_mask:
                        H.add_edge(u, n_name, bond_type='single')
                        H.add_edge(n_name, v, bond_type='double')
                    else:
                        H.add_edge(u, n_name, bond_type='single')
                        H.add_edge(n_name, v, bond_type='single')
                add(H)


# ============ 主生成器 ============

class NitrogenGenerator:
    """N 系树机制生成器（纯 C/H/N）

    生成流程：
      1. 骨架枚举：机制 A 用 (nC+nN) 碳骨架；机制 B/C 用 nC 碳骨架
         （均由内部预言机自举，全不饱和度）；
      2. 三机制独立产出候选 → 统一 add() 收口：
         公式门禁前置（H/C/N 严格一致）→ WL 哈希 → 碰撞安全去重；
      3. 返回 {分子式: [(mol_type, graph), ...]}（与 O 系 generate 同签名）。

    用法：
      gen = NitrogenGenerator()
      result = gen.generate(n_carbon=2, n_nitrogen=1)   # C2H7N
      for fml, items in result.items(): ...
    """

    def __init__(self, enable_replace: bool = True,
                 enable_terminal: bool = True,
                 enable_bridge: bool = True,
                 max_atoms: int = 10):
        self.enable_replace = enable_replace
        self.enable_terminal = enable_terminal
        self.enable_bridge = enable_bridge
        self.max_atoms = max_atoms
        self._skeleton_cache: Dict[int, List[nx.Graph]] = {}

    def _skeletons(self, n_c: int) -> List[nx.Graph]:
        """碳骨架缓存（深拷贝返回防污染）"""
        if n_c not in self._skeleton_cache:
            self._skeleton_cache[n_c] = _carbon_skeletons(n_c, self.max_atoms)
        return [g.copy() for g in self._skeleton_cache[n_c]]

    def generate(self, n_carbon: int, n_nitrogen: int,
                 n_hydrogen: Optional[int] = None,
                 progress_cb=None) -> Dict[str, list]:
        """生成所有 C(nC)H(nH)N(nN) 异构体，按分子式分组

        Args:
            n_carbon: 碳原子数
            n_nitrogen: 氮原子数
            n_hydrogen: 期望氢原子数（None 时自动按 DBE=0..max 枚举全部；
                       通常由分子式给定）
            progress_cb: 可选进度回调

        Returns:
            {分子式: [(mol_type, graph), ...]}
        """
        if n_carbon + n_nitrogen > self.max_atoms:
            return {}

        # 目标 H 集合（未给定 → 全部合法 H：nH = 2nC+nN+2-2k, k≥0）
        if n_hydrogen is not None:
            target_h_set = {n_hydrogen}
        else:
            k_min = 0
            k_max = (2 * n_carbon + n_nitrogen + 2) // 2  # H≥0 的 k 上界
            target_h_set = {2 * n_carbon + n_nitrogen + 2 - 2 * k
                            for k in range(k_min, k_max + 1)
                            if 2 * n_carbon + n_nitrogen + 2 - 2 * k >= 0}

        buckets: Dict[str, list] = defaultdict(list)
        seen: Dict[str, List[nx.Graph]] = {}

        def add(g: nx.Graph) -> bool:
            """公式门禁前置 + 碰撞安全去重；返回 True 表示新结构入桶"""
            n_c, n_h, n_n = graph_formula_parts(g)
            if n_c != n_carbon or n_n != n_nitrogen or n_h not in target_h_set:
                return False
            fml = format_graph_formula(g)
            wl = nx.weisfeiler_lehman_graph_hash(
                g, node_attr='label', edge_attr='bond_type')
            # 碰撞安全：同哈希须 VF2 同构（label+bond_type）才视为重复
            existing = seen.get(wl)
            if existing is not None:
                for g_old in existing:
                    if _graphs_isomorphic(g, g_old):
                        return False
                existing.append(g)
            else:
                seen[wl] = [g]
            buckets[fml].append(('nitrogen', g))
            return True

        # 机制 A：C→N 替换（骨架 = (nC+nN) 全碳）
        if self.enable_replace:
            skel_full = self._skeletons(n_carbon + n_nitrogen)
            if progress_cb:
                progress_cb('replace', len(skel_full))
            _mechanism_replace(skel_full, n_nitrogen, add)

        # 机制 B：端基 N（骨架 = nC 全碳）
        if self.enable_terminal:
            skel_nc = self._skeletons(n_carbon)
            if progress_cb:
                progress_cb('terminal', len(skel_nc))
            _mechanism_terminal(skel_nc, n_nitrogen, n_carbon,
                                min(target_h_set) if target_h_set else 0, add)

        # 机制 C：桥 N（骨架 = nC 全碳）
        if self.enable_bridge:
            skel_nc = self._skeletons(n_carbon)
            if progress_cb:
                progress_cb('bridge', len(skel_nc))
            _mechanism_bridge(skel_nc, n_nitrogen, n_carbon,
                              min(target_h_set) if target_h_set else 0, add)

        return dict(buckets)

    def generate_for_formula(self, formula: str) -> Dict[str, list]:
        """按分子式字符串生成（如 'C2H7N'）"""
        from .formula import parse_formula
        n_c, n_h, n_n, err = parse_formula(formula)
        if err:
            raise ValueError(err)
        return self.generate(n_c, n_n, n_hydrogen=n_h)


# ============ 自测 ============

if __name__ == "__main__":
    print("N 系树机制生成器")
    print("=" * 60)
    gen = NitrogenGenerator()
    for fml in ['C2H7N', 'C3H9N', 'C2H5N', 'C2H3N', 'C5H5N', 'C2H8N2']:
        try:
            res = gen.generate_for_formula(fml)
            total = sum(len(v) for v in res.values())
            print(f"\n{fml}: {total} 结构（{len(res)} 公式）")
            for k in sorted(res):
                print(f"  {k:10s} {len(res[k]):4d}")
        except ValueError as e:
            print(f"\n{fml}: {e}")
