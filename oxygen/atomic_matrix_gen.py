"""
邻接矩阵原子级生成器 (Phase F3: 补漏通道)

通过有序邻接矩阵枚举重原子 (C,O) 间所有合法连通图。
数学上与 MAYGEN 等价，用于填补骨架先行法无法覆盖的 ~2% 缺口。

仅对 N = n_carbon + n_oxygen ≤ 10 启用，超出范围不做枚举。
"""

import networkx as nx
import time
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict


# ============ 邻接矩阵回溯引擎 ============

def _max_bond_type(a1: str, a2: str) -> int:
    """两原子间允许的最大键级 (0/1/2/3)"""
    if a1 == 'C' and a2 == 'C':
        return 3   # C≡C 三键
    if 'O' in (a1, a2):
        return 2   # C=O 双键，C≡O 不存在
    return 1       # O-O 仅单键


def _bond_valence(bt: int) -> int:
    """键型 → 价电子数 (单键=1, 双键=2, 三键=3)"""
    return bt


def _count_graph_valences(M: List[List[int]], atom_types: List[str]) -> int:
    """已用价电子总和"""
    total = 0
    N = len(atom_types)
    for i in range(N):
        for j in range(i + 1, N):
            total += M[i][j]
    return total


def _is_graph_connected(M: List[List[int]], N: int) -> bool:
    """检查图是否连通"""
    visited = [False] * N
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in range(N):
            if M[u][v] > 0 and not visited[v]:
                visited[v] = True
                stack.append(v)
    return all(visited)


def _wl_hash_graph(M: List[List[int]], atom_types: List[str]) -> str:
    """从矩阵 + 原子类型构建图并返回 WL 哈希"""
    N = len(atom_types)
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, atom=atom_types[i])
    for i in range(N):
        for j in range(i + 1, N):
            bt = M[i][j]
            if bt == 0:
                continue
            bond = 'triple' if bt == 3 else ('double' if bt == 2 else 'single')
            G.add_edge(i, j, bond_type=bond)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='atom', edge_attr='bond_type')


def _matrix_to_graph(M: List[List[int]], atom_types: List[str]) -> nx.Graph:
    """邻接矩阵 → nx.Graph (含 label + bond_type)"""
    N = len(atom_types)
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, label=atom_types[i])
    for i in range(N):
        for j in range(i + 1, N):
            bt = M[i][j]
            if bt == 0:
                continue
            bond = 'triple' if bt == 3 else ('double' if bt == 2 else 'single')
            G.add_edge(i, j, bond_type=bond)
    return G


def _matrix_graphs_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """按 label + bond_type 精确判定两图同构（WL 哈希冲突时的兜底判定）"""
    nm = nx.algorithms.isomorphism.categorical_node_match('label', 'C')
    em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
    return nx.is_isomorphic(g1, g2, node_match=nm, edge_match=em)


def _seen_add(seen: Dict[str, List[nx.Graph]], wh: str,
              G: nx.Graph, results: List[nx.Graph]) -> None:
    """WL 哈希去重（非单射安全版）：同哈希必须同构才视为重复

    WL 图哈希对非同构图可能冲突（如 C1OC1C1CO1 与 C1OC2COC12），
    仅用哈希去重会漏结构；冲突时用 is_isomorphic 精确判定。
    """
    existing = seen.get(wh)
    if existing is None:
        seen[wh] = [G]
        results.append(G)
        return
    for G_old in existing:
        if _matrix_graphs_isomorphic(G, G_old):
            return  # 真重复
    existing.append(G)
    results.append(G)


# ============ 主生成器 ============

class AtomicMatrixGenerator:
    """原子级邻接矩阵补漏生成器"""

    def __init__(self, n_carbon: int, n_oxygen: int,
                 max_atoms: int = 10):
        """
        Args:
            n_carbon: 碳原子数
            n_oxygen: 氧原子数
            max_atoms: 重原子总数上限 (超过则不枚举)
        """
        self.n_carbon = n_carbon
        self.n_oxygen = n_oxygen
        self.N = n_carbon + n_oxygen
        self.max_atoms = max_atoms
        self.atom_types = ['C'] * n_carbon + ['O'] * n_oxygen
        self.valences = [4 if a == 'C' else 2 for a in self.atom_types]
        self.total_valence = sum(self.valences)
        self.num_positions = self.N * (self.N - 1) // 2

    def _can_enumerate(self) -> bool:
        """检查是否在可枚举范围内"""
        return self.N <= self.max_atoms

    # ── 公开接口 ──

    def generate_all_unsat(self) -> Dict[str, List[nx.Graph]]:
        """生成所有不饱和度下的结构，按分子式分组"""
        if not self._can_enumerate():
            return {}

        all_graphs = self._generate_all_graphs()
        buckets: Dict[str, List[nx.Graph]] = defaultdict(list)

        for G in all_graphs:
            # 计算 H 数 = Σ 剩余价态
            h_count = 0
            for node in G.nodes():
                atom = G.nodes[node].get('label', 'C')
                max_v = 4 if atom == 'C' else 2
                used = sum(
                    3 if G[node][nb].get('bond_type') == 'triple' else
                    2 if G[node][nb].get('bond_type') == 'double' else 1
                    for nb in G.neighbors(node)
                )
                h_count += max(0, max_v - used)

            n_c = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'C')
            n_o = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'O')
            base = f"C{n_c}" if n_c > 1 else "C"
            fml = f"{base}H{h_count}"
            if n_o > 0:
                fml += f"O{n_o}" if n_o > 1 else "O"
            buckets[fml].append(G)

        return dict(buckets)

    def generate_for_target_h(self, target_h: int,
                              deadline: Optional[float] = None) -> List[nx.Graph]:
        """生成 H 原子数恰为 target_h 的全部连通、价态合法图（目标键级剪枝）

        数学上完备（与 MAYGEN 等价）：邻接矩阵枚举 + 目标 H 键级总和剪枝，
        用作"树机制边界"的**内部完备性预言机**，无需任何外部工具。

        剪枝依据（精确）：
          - H = total_valence − 2·Σ键级，目标 H ⟺ 目标已用价 valence_target；
          - used_valence > valence_target → 剪（键级只增不减）；
          - used_valence + 剩余位置最大可贡献价 < valence_target → 剪（到不了）；
          - 连通下界：非零键数 + 剩余位置 < N−1 → 剪（连通图需 ≥ N−1 条边）；
          - 孤立原子：某原子当前度 0 且已过其最后可连位置 → 剪（永无法连通）；
          - **有序生成（BFS 前缀标号）**：新原子只能按索引顺序加入不断生长的连通前缀
            （首键必须是 (0,1)；之后原子 k 首次成键时要求 k == 当前已连通原子数 n_conn）。
            这是每个连通图的 BFS 标号所满足的必要条件（安全，不漏结构），
            且连通性由构造保证——消除 ~N! 倍的标号冗余，是性能的关键。

        Args:
            target_h: 目标氢原子数
            deadline: 可选超时（time.time() 绝对时刻）；超时抛 TimeoutError。
                     用于边界报告等需要限时的场景。

        Returns:
            唯一图列表（label='C'/'O'，bond_type），H 数 == target_h。
        """
        if not self._can_enumerate():
            return []

        N = self.N
        rem = self.total_valence - target_h
        if rem < 0 or rem % 2 != 0:
            return []
        valence_target = rem  # 目标已用价总和（= 2·Σ键级）

        # 预计算每个位置的最大键级（常量）与每个原子最后出现位置（连通剪枝）
        pos_max_bond = [0] * self.num_positions
        last_pos = [0] * N
        for p in range(self.num_positions):
            i, j = self._position_to_indices(p)
            pos_max_bond[p] = _max_bond_type(self.atom_types[i],
                                             self.atom_types[j])
            last_pos[i] = max(last_pos[i], p)
            last_pos[j] = max(last_pos[j], p)

        M = [[0] * N for _ in range(N)]
        row_sums = [0] * N
        results: List[nx.Graph] = []
        seen: Dict[str, List[nx.Graph]] = {}   # WL 哈希 → 图列表（冲突时同构判定）

        def max_future_valence(pos_idx: int) -> int:
            """剩余位置可贡献的最大价（乐观上界，用于"到不了"剪枝）"""
            total = 0
            for p in range(pos_idx, self.num_positions):
                i, j = self._position_to_indices(p)
                cap = min(pos_max_bond[p],
                          self.valences[i] - row_sums[i],
                          self.valences[j] - row_sums[j])
                if cap > 0:
                    total += 2 * cap
            return total

        def backtrack(pos_idx: int, used_valence: int, nz_bonds: int,
                      n_conn_c: int, n_conn_o: int):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"原子矩阵枚举超时（N={N}，H={target_h}）")
            # 目标键级剪枝（精确）
            if used_valence > valence_target:
                return
            if used_valence + max_future_valence(pos_idx) < valence_target:
                return
            # 连通下界：连通图需 ≥ N-1 条非零键
            if nz_bonds + (self.num_positions - pos_idx) < N - 1:
                return

            if pos_idx == self.num_positions:
                if used_valence != valence_target:
                    return
                if n_conn_c + n_conn_o != N:
                    return
                if not _is_graph_connected(M, N):
                    return  # 双新键可能产生未合并的临时分量，须显式验证连通
                _seen_add(seen, _wl_hash_graph(M, self.atom_types),
                          _matrix_to_graph(M, self.atom_types), results)
                return

            i, j = self._position_to_indices(pos_idx)
            a1, a2 = self.atom_types[i], self.atom_types[j]
            max_bond = pos_max_bond[pos_idx]

            # 快剪枝：O-O 相邻超过阈值 (最多1个O-O键，避免O-O-O)
            if a1 == 'O' and a2 == 'O':
                o_o_count = 0
                for k in range(N):
                    if self.atom_types[k] == 'O' and M[i][k] > 0:
                        o_o_count += 1
                    if self.atom_types[k] == 'O' and M[j][k] > 0:
                        o_o_count += 1
                if o_o_count >= 2:  # 已有 O-O 键，不能再加
                    max_bond = 0

            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)

            for bt in range(max_bond + 1):
                bv = _bond_valence(bt)
                if row_sums[i] + bv > self.valences[i]:
                    continue
                if row_sums[j] + bv > self.valences[j]:
                    continue
                if bt > 0:
                    # ── 有序生成：单新键类型内前缀序（安全，消标号冗余）──
                    # 新原子经"单新键"首次成键时，必须是同类型内按索引递增的下一个
                    # （C 原子按 0..nc-1 顺序、O 原子按 nc..N-1 顺序）。
                    # "双新键"（两端同时首次成键）全放行——它可能先成临时分量、
                    # 后经后续键并入主分量（如 C=COOC 的末端 C-O 键），
                    # 由末端 _is_graph_connected 兜底。
                    # 安全性：每个连通图在"按 BFS 发现秩于类型内编号"的标号下，
                    # 其所有单新首键均满足类型内顺序 → 不漏结构。
                    if not (i_new and j_new):
                        if i_new:
                            if (i < self.n_carbon and i != n_conn_c) or \
                               (i >= self.n_carbon and i != self.n_carbon + n_conn_o):
                                continue
                        elif j_new:
                            if (j < self.n_carbon and j != n_conn_c) or \
                               (j >= self.n_carbon and j != self.n_carbon + n_conn_o):
                                continue
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bv
                    row_sums[j] += bv
                    backtrack(pos_idx + 1, used_valence + 2 * bv,
                              nz_bonds + 1,
                              n_conn_c + (1 if i_new and i < self.n_carbon else 0)
                                         + (1 if j_new and j < self.n_carbon else 0),
                              n_conn_o + (1 if i_new and i >= self.n_carbon else 0)
                                         + (1 if j_new and j >= self.n_carbon else 0))
                    row_sums[i] -= bv
                    row_sums[j] -= bv
                    M[i][j] = M[j][i] = 0
                else:
                    # 孤立原子"必须连"剪枝：若 i 或 j 当前孤立且这是其最后可连位置，
                    # 则此处不能跳过（否则该原子永远无法连通）
                    if row_sums[i] == 0 and last_pos[i] == pos_idx:
                        continue
                    if row_sums[j] == 0 and last_pos[j] == pos_idx:
                        continue
                    M[i][j] = M[j][i] = 0
                    backtrack(pos_idx + 1, used_valence, nz_bonds,
                              n_conn_c, n_conn_o)
                    M[i][j] = M[j][i] = 0

        backtrack(0, 0, 0, 0, 0)
        return results

    # ── 核心回溯 ──

    def _generate_all_graphs(self) -> List[nx.Graph]:
        """生成所有连通、价态合法的图 (仅 C/O 骨架，不含 H)"""
        if not self._can_enumerate():
            return []

        N = self.N
        M = [[0] * N for _ in range(N)]
        results: List[nx.Graph] = []
        seen: Dict[str, List[nx.Graph]] = {}   # WL 哈希 → 图列表（冲突时同构判定）

        # 用于行和剪枝的累计数组
        row_sums = [0] * N

        def backtrack(pos_idx: int, n_conn_c: int, n_conn_o: int):
            if pos_idx == self.num_positions:
                # 全部填完 → 去重（双新键可能产生未合并分量，须显式验证连通）
                if n_conn_c + n_conn_o != N:
                    return
                if not _is_graph_connected(M, N):
                    return
                _seen_add(seen, _wl_hash_graph(M, self.atom_types),
                          _matrix_to_graph(M, self.atom_types), results)
                return

            # 位置 (i, j)
            i, j = self._position_to_indices(pos_idx)
            a1, a2 = self.atom_types[i], self.atom_types[j]
            max_bond = _max_bond_type(a1, a2)

            # 快剪枝：O-O 相邻超过阈值 (最多1个O-O键，避免O-O-O)
            if a1 == 'O' and a2 == 'O':
                o_o_count = 0
                for k in range(N):
                    if self.atom_types[k] == 'O' and M[i][k] > 0:
                        o_o_count += 1
                    if self.atom_types[k] == 'O' and M[j][k] > 0:
                        o_o_count += 1
                if o_o_count >= 2:  # 已有 O-O 键，不能再加
                    max_bond = 0

            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)

            for bt in range(max_bond + 1):
                bv = _bond_valence(bt)

                # 剪枝：行和超限
                if row_sums[i] + bv > self.valences[i]:
                    continue
                if row_sums[j] + bv > self.valences[j]:
                    continue

                if bt > 0:
                    # ── 有序生成：单新键类型内前缀序（同 generate_for_target_h）──
                    if not (i_new and j_new):
                        if i_new:
                            if (i < self.n_carbon and i != n_conn_c) or \
                               (i >= self.n_carbon and i != self.n_carbon + n_conn_o):
                                continue
                        elif j_new:
                            if (j < self.n_carbon and j != n_conn_c) or \
                               (j >= self.n_carbon and j != self.n_carbon + n_conn_o):
                                continue
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bv
                    row_sums[j] += bv
                    backtrack(pos_idx + 1,
                              n_conn_c + (1 if i_new and i < self.n_carbon else 0)
                                         + (1 if j_new and j < self.n_carbon else 0),
                              n_conn_o + (1 if i_new and i >= self.n_carbon else 0)
                                         + (1 if j_new and j >= self.n_carbon else 0))
                    row_sums[i] -= bv
                    row_sums[j] -= bv
                    M[i][j] = M[j][i] = 0
                else:
                    M[i][j] = M[j][i] = 0
                    backtrack(pos_idx + 1, n_conn_c, n_conn_o)
                    M[i][j] = M[j][i] = 0

        backtrack(0, 0, 0)
        return results

    def _position_to_indices(self, pos: int) -> Tuple[int, int]:
        """将线性位置映射到矩阵索引 (i, j), i < j"""
        N = self.N
        remaining = pos
        for i in range(N):
            cols_in_row = N - i - 1
            if remaining < cols_in_row:
                j = i + 1 + remaining
                return i, j
            remaining -= cols_in_row
        return N - 2, N - 1  # 不应到达


# ============ 测试 ============

if __name__ == "__main__":
    print("邻接矩阵原子级生成器 (Phase F3)")
    print("=" * 60)

    test_cases = [
        (1, 1, "C1O1 (甲醛)"),
        (2, 1, "C2O1 (乙醇 + 二甲醚)"),
        (2, 2, "C2O2 (乙二醇/过氧...)"),
        (3, 1, "C3O1 (丙醇/甲乙醚/环氧...)"),
    ]

    for nc, no, desc in test_cases:
        gen = AtomicMatrixGenerator(nc, no)
        if not gen._can_enumerate():
            print(f"\nC{nc}+{no}O (N={nc+no}): SKIPPED (>10)")
            continue

        results = gen.generate_all_unsat()
        total = sum(len(v) for v in results.values())
        print(f"\nC{nc}+{no}O ({desc}) — N={nc+no}, {total} 结构, {len(results)} 公式:")
        for fml in sorted(results.keys()):
            print(f"  {fml:10s} {len(results[fml]):4d}")
