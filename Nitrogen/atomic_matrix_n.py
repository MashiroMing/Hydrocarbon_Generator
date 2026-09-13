"""
Nitrogen/atomic_matrix_n.py — N 系完备性预言机（邻接矩阵原子级枚举）

从 oxygen/atomic_matrix_gen.py 移植，价态表参数化为 C=4 / N=3（中性八隅律，
与 MAYGEN 默认价态表一致）。数学上与 MAYGEN 等价：枚举重原子 (C,N) 间所有
合法连通图，目标 H 剪枝，作为 N 系生成器的**内部完备性真值**。

用途：
  1. 小公式（重原子 ≤ 10）直接作为权威生成器输出全部异构体；
  2. 作为树机制生成器（generator.py）的完备性预言机：检查"树 ⊆ 预言机"、
     报告覆盖率与差异（含 N 杂环等树机制未覆盖的结构）。

N 系与 O 系的关键差异（移植点）：
  - 原子类型 ['C']*nC + ['N']*nN，价态 4 / 3；
  - _max_bond_type: C-C=3, C-N=3（腈 C≡N）, N-N=2（偶氮 N=N；N≡N 无法再连
    其他原子，靠连通性自动排除，无需特判）；
  - 无 O-O 快剪枝（N-N-N 链如肼类合法，不能剪）；
  - 有序生成双计数器 (n_conn_c, n_conn_n)：N 原子按索引 n_carbon + n_conn_n。
"""

import networkx as nx
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ============ 邻接矩阵回溯引擎 ============

def _max_bond_type(a1: str, a2: str) -> int:
    """两原子间允许的最大键级 (0/1/2/3)

    - C-C: 3（C≡C 三键）
    - C-N: 3（腈 C≡N；异腈 R-N≡C 中 N 超 3 价，由价态检查自动排除）
    - N-N: 2（偶氮 N=N；N≡N 三键下两端均无剩余价态连 C，连通性检查排除）
    """
    if a1 == 'C' and a2 == 'C':
        return 3
    if a1 == 'C' and a2 == 'N':
        return 3
    if a1 == 'N' and a2 == 'C':
        return 3
    return 2  # N-N


def _bond_valence(bt: int) -> int:
    """键型 → 价电子数 (单键=1, 双键=2, 三键=3)"""
    return bt


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
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='atom',
                                           edge_attr='bond_type')


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


def _graphs_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """按 label + bond_type 精确判定两图同构（WL 哈希冲突时的兜底判定）"""
    nm = nx.algorithms.isomorphism.categorical_node_match('label', 'C')
    em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
    return nx.is_isomorphic(g1, g2, node_match=nm, edge_match=em)


def _seen_add(seen: Dict[str, List[nx.Graph]], wh: str,
              G: nx.Graph, results: List[nx.Graph]) -> None:
    """WL 哈希去重（非单射安全版）：同哈希必须同构才视为重复

    历史教训（O 系踩坑）：WL 图哈希对非同构图可能冲突，仅用哈希去重会漏结构。
    """
    existing = seen.get(wh)
    if existing is None:
        seen[wh] = [G]
        results.append(G)
        return
    for G_old in existing:
        if _graphs_isomorphic(G, G_old):
            return  # 真重复
    existing.append(G)
    results.append(G)


# ============ 主预言机 ============

class AtomicMatrixNGenerator:
    """N 系原子级完备预言机 / 生成器（C/H/N，C=4 价，N=3 价）"""

    def __init__(self, n_carbon: int, n_nitrogen: int,
                 max_atoms: int = 10):
        """
        Args:
            n_carbon: 碳原子数
            n_nitrogen: 氮原子数
            max_atoms: 重原子总数上限（超过则不枚举）
        """
        self.n_carbon = n_carbon
        self.n_nitrogen = n_nitrogen
        self.N = n_carbon + n_nitrogen
        self.max_atoms = max_atoms
        self.atom_types = ['C'] * n_carbon + ['N'] * n_nitrogen
        self.valences = [4 if a == 'C' else 3 for a in self.atom_types]
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
            h_count = 0
            for node in G.nodes():
                atom = G.nodes[node].get('label', 'C')
                max_v = 4 if atom == 'C' else 3
                used = sum(
                    3 if G[node][nb].get('bond_type') == 'triple' else
                    2 if G[node][nb].get('bond_type') == 'double' else 1
                    for nb in G.neighbors(node)
                )
                h_count += max(0, max_v - used)

            n_c = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'C')
            n_n = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'N')
            base = f"C{n_c}" if n_c > 1 else "C"
            fml = f"{base}H{h_count}"
            if n_n > 0:
                fml += f"N{n_n}" if n_n > 1 else "N"
            buckets[fml].append(G)

        return dict(buckets)

    def generate_for_target_h(self, target_h: int,
                              deadline: Optional[float] = None) -> List[nx.Graph]:
        """生成 H 原子数恰为 target_h 的全部连通、价态合法图（目标键级剪枝）

        数学上完备（与 MAYGEN 等价）：邻接矩阵枚举 + 目标 H 键级总和剪枝，
        用作"树机制边界"的**内部完备性预言机**，无需任何外部工具。

        剪枝依据（精确，与 O 系 atomic_matrix_gen 同源）：
          - H = total_valence − 2·Σ键级，目标 H ⟺ 目标已用价 valence_target；
          - used_valence > valence_target → 剪；剩余位置最大可贡献价不足 → 剪；
          - 连通下界：非零键数 + 剩余位置 < N−1 → 剪；
          - 孤立原子：度 0 且已过其最后可连位置 → 剪；
          - 有序生成（BFS 前缀标号）：新原子按索引顺序加入连通前缀
            （C 按 0..nc-1，N 按 nc..N-1），消除 ~N! 倍标号冗余。

        Args:
            target_h: 目标氢原子数
            deadline: 可选超时（time.time() 绝对时刻）；超时抛 TimeoutError。

        Returns:
            唯一图列表（label='C'/'N'，bond_type），H 数 == target_h。
        """
        if not self._can_enumerate():
            return []

        N = self.N
        rem = self.total_valence - target_h
        if rem < 0 or rem % 2 != 0:
            return []
        valence_target = rem  # 目标已用价总和（= 2·Σ键级）

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
        seen: Dict[str, List[nx.Graph]] = {}

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
                      n_conn_c: int, n_conn_n: int):
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
                if n_conn_c + n_conn_n != N:
                    return
                if not _is_graph_connected(M, N):
                    return  # 双新键可能产生未合并的临时分量，须显式验证连通
                _seen_add(seen, _wl_hash_graph(M, self.atom_types),
                          _matrix_to_graph(M, self.atom_types), results)
                return

            i, j = self._position_to_indices(pos_idx)
            max_bond = pos_max_bond[pos_idx]

            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)

            for bt in range(max_bond + 1):
                bv = _bond_valence(bt)
                if row_sums[i] + bv > self.valences[i]:
                    continue
                if row_sums[j] + bv > self.valences[j]:
                    continue
                if bt > 0:
                    # 有序生成：单新键类型内前缀序（安全，消标号冗余）
                    if not (i_new and j_new):
                        if i_new:
                            if (i < self.n_carbon and i != n_conn_c) or \
                               (i >= self.n_carbon and i != self.n_carbon + n_conn_n):
                                continue
                        elif j_new:
                            if (j < self.n_carbon and j != n_conn_c) or \
                               (j >= self.n_carbon and j != self.n_carbon + n_conn_n):
                                continue
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bv
                    row_sums[j] += bv
                    backtrack(pos_idx + 1, used_valence + 2 * bv,
                              nz_bonds + 1,
                              n_conn_c + (1 if i_new and i < self.n_carbon else 0)
                                         + (1 if j_new and j < self.n_carbon else 0),
                              n_conn_n + (1 if i_new and i >= self.n_carbon else 0)
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
                              n_conn_c, n_conn_n)
                    M[i][j] = M[j][i] = 0

        backtrack(0, 0, 0, 0, 0)
        return results

    # ── 核心回溯 ──

    def _generate_all_graphs(self) -> List[nx.Graph]:
        """生成所有连通、价态合法的图 (仅 C/N 骨架，不含 H)"""
        if not self._can_enumerate():
            return []

        N = self.N
        M = [[0] * N for _ in range(N)]
        results: List[nx.Graph] = []
        seen: Dict[str, List[nx.Graph]] = {}
        row_sums = [0] * N

        def backtrack(pos_idx: int, n_conn_c: int, n_conn_n: int):
            if pos_idx == self.num_positions:
                if n_conn_c + n_conn_n != N:
                    return
                if not _is_graph_connected(M, N):
                    return
                _seen_add(seen, _wl_hash_graph(M, self.atom_types),
                          _matrix_to_graph(M, self.atom_types), results)
                return

            i, j = self._position_to_indices(pos_idx)
            max_bond = _max_bond_type(self.atom_types[i], self.atom_types[j])

            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)

            for bt in range(max_bond + 1):
                bv = _bond_valence(bt)
                if row_sums[i] + bv > self.valences[i]:
                    continue
                if row_sums[j] + bv > self.valences[j]:
                    continue

                if bt > 0:
                    # 有序生成：单新键类型内前缀序（同 generate_for_target_h）
                    if not (i_new and j_new):
                        if i_new:
                            if (i < self.n_carbon and i != n_conn_c) or \
                               (i >= self.n_carbon and i != self.n_carbon + n_conn_n):
                                continue
                        elif j_new:
                            if (j < self.n_carbon and j != n_conn_c) or \
                               (j >= self.n_carbon and j != self.n_carbon + n_conn_n):
                                continue
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bv
                    row_sums[j] += bv
                    backtrack(pos_idx + 1,
                              n_conn_c + (1 if i_new and i < self.n_carbon else 0)
                                         + (1 if j_new and j < self.n_carbon else 0),
                              n_conn_n + (1 if i_new and i >= self.n_carbon else 0)
                                         + (1 if j_new and j >= self.n_carbon else 0))
                    row_sums[i] -= bv
                    row_sums[j] -= bv
                    M[i][j] = M[j][i] = 0
                else:
                    M[i][j] = M[j][i] = 0
                    backtrack(pos_idx + 1, n_conn_c, n_conn_n)
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


# ============ 自测 ============

if __name__ == "__main__":
    print("N 系原子级完备预言机")
    print("=" * 60)
    for nc, nn, desc in [
        (2, 1, "C2H7N (乙胺+二甲胺)"),
        (3, 1, "C3H9N (丙胺类)"),
        (2, 3, "C2H5N (乙腈等)"),
        (5, 1, "C5H5N (吡啶等)"),
    ]:
        gen = AtomicMatrixNGenerator(nc, nn)
        t0 = time.time()
        results = gen.generate_all_unsat()
        dt = time.time() - t0
        total = sum(len(v) for v in results.values())
        print(f"\nC{nc}+{nn}N ({desc}) — N={nc+nn}, {total} 结构, "
              f"{len(results)} 公式, {dt:.1f}s:")
        for fml in sorted(results.keys(), key=lambda s: (len(s), s)):
            print(f"  {fml:10s} {len(results[fml]):4d}")
