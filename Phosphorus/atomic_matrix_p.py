"""矩阵级完备预言机（第一期：C / H / P）

算法与 ``Nitrogen/atomic_matrix_n.py`` 同源（有序生成 + 目标键级剪枝），
但**完全自包含**（不 import 其它模块的生成器），便于本包独立演进。

数学口径：
- 原子 i 的键级和 + 氢数 = 价态（C=4，P=3）；
- 目标 H  ⟺  目标已用价 ``valence_target = total_valence - target_h``；
- 邻接矩阵枚举（键级 0/1/2/3），剪枝：
    · used_valence > 目标 → 剪
    · used_valence + 剩余位置最大可贡献价 < 目标 → 剪
    · 连通下界：非零键数 + 剩余位置 < N-1 → 剪
    · 孤立原子：度 0 且已过其最后可连位置 → 剪
    · 有序生成（类型内前缀标号）消除 ~N! 倍标号冗余
- 去重：**矩阵级 WL(3 轮) + 着色邻域签名分桶 + 精确同构**（与 N 模块同款，
  实测比 networkx 版快 3.9–6.3×）

键级上限：C–C / C–P / P–P 一律允许三键
（MAYGEN 1.8 实测 ``P2`` 生成 1 个结构 = P≡P，故 P–P 三键必须允许）。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .formula import VALENCE

# WL 颜色细化轮数（固定 3 轮：更多轮收益递减、成本线性增长）
_WL_ROUNDS = 3


# ============================ 矩阵级工具 ============================

def _max_bond_type(a1: str, a2: str) -> int:
    """两原子间允许的最大键级（C-C / C-P / P-P 均允许三键）"""
    return 3


def _is_graph_connected(M: List[List[int]], n: int) -> bool:
    """邻接矩阵连通性检查"""
    visited = [False] * n
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in range(n):
            if M[u][v] > 0 and not visited[v]:
                visited[v] = True
                stack.append(v)
    return all(visited)


def _matrix_to_graph(M: List[List[int]], atom_types: List[str]) -> nx.Graph:
    """邻接矩阵 → nx.Graph（节点 label / 边 bond_type 用整型 1/2/3）"""
    n = len(atom_types)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, label=atom_types[i])
    for i in range(n):
        for j in range(i + 1, n):
            bt = M[i][j]
            if bt:
                G.add_edge(i, j, bond_type=int(bt))
    return G


def _wl_colors(M: List[List[int]], atom_types: List[str],
               rounds: int = _WL_ROUNDS) -> Tuple[int, ...]:
    """矩阵级 WL 颜色细化（固定轮数，按签名内容排序编号保证规范化）"""
    n = len(atom_types)
    if n == 0:
        return ()
    elems = sorted(set(atom_types))
    rank0 = {t: i for i, t in enumerate(elems)}
    colors = [rank0[t] for t in atom_types]
    for _ in range(rounds):
        sig = []
        for i in range(n):
            mi = M[i]
            nb = [(colors[j], mi[j]) for j in range(n) if mi[j] > 0]
            nb.sort()
            sig.append((colors[i], tuple(nb)))
        rank = {s: i for i, s in enumerate(sorted(set(sig)))}
        new = [rank[s] for s in sig]
        if new == colors:
            break
        colors = new
    return tuple(colors)


def _leaf_invariants(M: List[List[int]], atom_types: List[str]):
    """每叶一次：``(inv, colors, vkeys)``，全部为同构不变量"""
    n = len(atom_types)
    colors = _wl_colors(M, atom_types)
    vkeys = []
    for i in range(n):
        mi = M[i]
        agg: Dict[int, int] = {}
        for j in range(n):
            if mi[j] > 0:
                agg[colors[j]] = agg.get(colors[j], 0) + mi[j]
        vkeys.append((colors[i], tuple(sorted(agg.items()))))
    return tuple(sorted(vkeys)), colors, tuple(vkeys)


def _matrix_iso_fast(M1, vk1, M2, vk2) -> bool:
    """精确同构（候选由着色邻域签名给出，回溯校验全矩阵）"""
    n = len(vk1)
    if n != len(vk2):
        return False
    cand = [[j for j in range(n) if vk2[j] == vk1[i]] for i in range(n)]
    if any(not cc for cc in cand):
        return False
    order = sorted(range(n), key=lambda i: len(cand[i]))
    mapping = [-1] * n
    used = [False] * n

    def dfs(k: int) -> bool:
        if k == n:
            return True
        i = order[k]
        for j in cand[i]:
            if used[j]:
                continue
            ok = True
            for k2 in range(k):
                i2 = order[k2]
                if M1[i][i2] != M2[j][mapping[i2]]:
                    ok = False
                    break
            if ok:
                mapping[i] = j
                used[j] = True
                if dfs(k + 1):
                    return True
                used[j] = False
                mapping[i] = -1
        return False

    return dfs(0)


def _seen_add_matrix(seen: Dict, inv, colors, vkeys, M, atom_types,
                     results: List[nx.Graph]) -> None:
    """矩阵级去重：桶键 = 叶子不变量；桶内精确同构；命中才建图"""
    bucket = seen.get(inv)
    if bucket is None:
        seen[inv] = [([row[:] for row in M], vkeys)]
        results.append(_matrix_to_graph(M, atom_types))
        return
    for M0, vk0 in bucket:
        if _matrix_iso_fast(M, vkeys, M0, vk0):
            return
    bucket.append(([row[:] for row in M], vkeys))
    results.append(_matrix_to_graph(M, atom_types))


# ============================ 并发辅助（模块级，可 pickle） ============================

def _prefix_worker(args):
    (n_c, n_p, valence_p, target_h, prefix, max_atoms, deadline) = args
    gen = PhosphorusMatrixGenerator(n_c, n_p, valence_p=valence_p,
                                    max_atoms=max_atoms)
    return gen.generate_for_target_h(target_h, deadline=deadline,
                                     prefix=tuple(prefix))


# ============================ 主预言机 ============================

class PhosphorusMatrixGenerator:
    """C/H/P 原子级完备预言机（P=3 价；预留 5 价）

    Args:
        n_carbon: 碳原子数（≥0）
        n_phosphorus: 磷原子数（≥1）
        valence_p: 磷价态（3 = 默认/一期；5 = 预留、实验性）
        max_atoms: 重原子上限（护栏，默认 10，与 N 系一致）
    """

    def __init__(self, n_carbon: int, n_phosphorus: int, valence_p: int = 3,
                 max_atoms: int = 10):
        if n_carbon < 0 or n_phosphorus < 1:
            raise ValueError("n_carbon ≥ 0 且 n_phosphorus ≥ 1")
        if valence_p not in (3, 5):
            raise ValueError("valence_p 只支持 3（默认）或 5（预留）")
        self.n_carbon = int(n_carbon)
        self.n_phosphorus = int(n_phosphorus)
        self.valence_p = int(valence_p)
        self.max_atoms = int(max_atoms)

        self.atom_types: List[str] = (['C'] * self.n_carbon
                                      + ['P'] * self.n_phosphorus)
        self.valences: List[int] = ([VALENCE['C']] * self.n_carbon
                                    + [self.valence_p] * self.n_phosphorus)
        self.N = len(self.atom_types)
        self.total_valence = sum(self.valences)
        self._off_p = self.n_carbon  # P 的索引区间起点

        # 位置表：所有 i<j 原子对（按行优先）
        self._positions: List[Tuple[int, int]] = [
            (i, j) for i in range(self.N) for j in range(i + 1, self.N)]
        self.num_positions = len(self._positions)

    # ── 基础查询 ──

    def _can_enumerate(self) -> bool:
        return self.N <= self.max_atoms

    def _position_to_indices(self, pos: int) -> Tuple[int, int]:
        return self._positions[pos]

    def _is_next_of_type(self, idx: int, n_conn_c: int, n_conn_p: int) -> bool:
        """有序生成：原子 idx 是其类型内"应当接入的下一个"（消除标号冗余）"""
        if idx < self._off_p:
            return idx == n_conn_c
        return idx == self._off_p + n_conn_p

    def _conn_increments(self, i_new: bool, i: int,
                         j_new: bool, j: int) -> Tuple[int, int]:
        dc = dp = 0
        for _new, _idx in ((i_new, i), (j_new, j)):
            if not _new:
                continue
            if _idx < self._off_p:
                dc += 1
            else:
                dp += 1
        return dc, dp

    def h_values(self) -> List[int]:
        """该组成（nC, nP）下所有可能的 H 数（用于小规模全量自检）"""
        rem_min = 2 * (self.N - 1) if self.N > 1 else 0
        out = []
        for rem in range(rem_min, self.total_valence + 1, 2):
            out.append(self.total_valence - rem)
        return out

    def feasible(self, target_h: int) -> bool:
        """目标 H 是否有解（奇偶/上下界/连通下界）"""
        if not self._can_enumerate():
            return False
        rem = self.total_valence - target_h
        if rem < 0 or rem % 2 != 0:
            return False
        if self.N == 1:
            return rem == 0
        if rem < 2 * (self.N - 1):          # 连通图至少 N−1 条单键
            return False
        if rem > 2 * 3 * self.num_positions:  # 全部满键的粗上界
            return False
        return True

    # ── 预计算 / 前缀 ──

    def _precompute(self):
        pos_max_bond = [0] * self.num_positions
        last_pos = [0] * self.N
        for p in range(self.num_positions):
            i, j = self._position_to_indices(p)
            pos_max_bond[p] = _max_bond_type(self.atom_types[i],
                                             self.atom_types[j])
            last_pos[i] = max(last_pos[i], p)
            last_pos[j] = max(last_pos[j], p)
        return pos_max_bond, last_pos

    def _replay_prefix(self, prefix: Sequence[int], M, row_sums,
                       pos_max_bond, last_pos):
        """回放并行前缀；非法返回 None（与主回溯同一套校验）"""
        if len(prefix) > self.num_positions:
            return None
        used = nz = cc = cp = 0
        for p, bt in enumerate(prefix):
            i, j = self._position_to_indices(p)
            mb = pos_max_bond[p]
            if bt < 0 or bt > mb:
                return None
            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)
            if bt > 0:
                if (row_sums[i] + bt > self.valences[i]
                        or row_sums[j] + bt > self.valences[j]):
                    return None
                if not (i_new and j_new):
                    if i_new and not self._is_next_of_type(i, cc, cp):
                        return None
                    if j_new and not self._is_next_of_type(j, cc, cp):
                        return None
                dc, dp = self._conn_increments(i_new, i, j_new, j)
                M[i][j] = M[j][i] = bt
                row_sums[i] += bt
                row_sums[j] += bt
                used += 2 * bt
                nz += 1
                cc += dc
                cp += dp
            else:
                if row_sums[i] == 0 and last_pos[i] == p:
                    return None
                if row_sums[j] == 0 and last_pos[j] == p:
                    return None
        return len(prefix), used, nz, cc, cp

    def enumerate_prefixes(self, depth: int) -> List[Tuple[int, ...]]:
        """枚举全部通过校验的深度 ``depth`` 前缀（并行切分的互不重叠任务）"""
        depth = max(0, min(int(depth), self.num_positions))
        if depth == 0:
            return [()]
        pos_max_bond, last_pos = self._precompute()
        out: List[Tuple[int, ...]] = []

        def rec(p: int, prefix: Tuple[int, ...]) -> None:
            if p == depth:
                out.append(prefix)
                return
            i, j = self._position_to_indices(p)
            for bt in range(pos_max_bond[p] + 1):
                M = [[0] * self.N for _ in range(self.N)]
                row_sums = [0] * self.N
                st = self._replay_prefix(prefix + (bt,), M, row_sums,
                                         pos_max_bond, last_pos)
                if st is not None:
                    rec(p + 1, prefix + (bt,))

        rec(0, ())
        return out

    # ── 主枚举 ──

    def generate_for_target_h(self, target_h: int,
                              deadline: Optional[float] = None,
                              prefix: Sequence[int] = ()
                              ) -> List[nx.Graph]:
        """枚举 H 数恰为 ``target_h`` 的全部连通、价态合法图

        Args:
            target_h: 目标氢原子数
            deadline: ``time.time()`` 绝对时刻；超时抛 ``TimeoutError``
            prefix: 可选前缀（并行切分用）

        Returns:
            去重后的图列表（label ∈ {'C','P'}，bond_type ∈ {1,2,3}）
        """
        if not self._can_enumerate() or not self.feasible(target_h):
            return []

        N = self.N
        rem = self.total_valence - target_h
        valence_target = rem  # = 2·Σ键级

        pos_max_bond, last_pos = self._precompute()
        M = [[0] * N for _ in range(N)]
        row_sums = [0] * N
        results: List[nx.Graph] = []
        seen: Dict = {}

        st = self._replay_prefix(prefix, M, row_sums, pos_max_bond, last_pos)
        if st is None:
            return []
        p_start, used0, nz0, cc0, cp0 = st

        def max_future_valence(pos_idx: int) -> int:
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
                      n_conn_c: int, n_conn_p: int):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"原子矩阵枚举超时（N={N}，H={target_h}）")
            if used_valence > valence_target:
                return
            if used_valence + max_future_valence(pos_idx) < valence_target:
                return
            if nz_bonds + (self.num_positions - pos_idx) < N - 1:
                return

            if pos_idx == self.num_positions:
                if used_valence != valence_target:
                    return
                if n_conn_c + n_conn_p != N:
                    return
                if not _is_graph_connected(M, N):
                    return
                inv, cols, vks = _leaf_invariants(M, self.atom_types)
                _seen_add_matrix(seen, inv, cols, vks, M, self.atom_types,
                                 results)
                return

            i, j = self._position_to_indices(pos_idx)
            max_bond = pos_max_bond[pos_idx]
            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)

            for bt in range(max_bond + 1):
                if row_sums[i] + bt > self.valences[i]:
                    continue
                if row_sums[j] + bt > self.valences[j]:
                    continue
                if bt > 0:
                    if not (i_new and j_new):
                        if i_new and not self._is_next_of_type(
                                i, n_conn_c, n_conn_p):
                            continue
                        if j_new and not self._is_next_of_type(
                                j, n_conn_c, n_conn_p):
                            continue
                    dc, dp = self._conn_increments(i_new, i, j_new, j)
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bt
                    row_sums[j] += bt
                    backtrack(pos_idx + 1, used_valence + 2 * bt,
                              nz_bonds + 1, n_conn_c + dc, n_conn_p + dp)
                    row_sums[i] -= bt
                    row_sums[j] -= bt
                    M[i][j] = M[j][i] = 0
                else:
                    if row_sums[i] == 0 and last_pos[i] == pos_idx:
                        continue
                    if row_sums[j] == 0 and last_pos[j] == pos_idx:
                        continue
                    backtrack(pos_idx + 1, used_valence, nz_bonds,
                              n_conn_c, n_conn_p)

        backtrack(p_start, used0, nz0, cc0, cp0)
        return results

    def generate(self, target_h: int, parallel: bool = False,
                 workers: Optional[int] = None,
                 deadline: Optional[float] = None) -> List[nx.Graph]:
        """生成入口：串行 / 并行（并行按前缀切分，结果集完全一致）"""
        if not parallel or self.num_positions == 0:
            return self.generate_for_target_h(target_h, deadline=deadline)
        return self.generate_parallel(target_h, workers=workers,
                                      deadline=deadline)

    def generate_parallel(self, target_h: int, workers: Optional[int] = None,
                          deadline: Optional[float] = None
                          ) -> List[nx.Graph]:
        """按前缀并行枚举，父进程统一去重（串行/并行结果集一致）"""
        import multiprocessing as mp
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if not self.feasible(target_h):
            return []
        n_workers = min(workers or (os.cpu_count() or 4), 12)
        depth = 1
        prefixes = self.enumerate_prefixes(depth)
        if len(prefixes) <= 1 or n_workers <= 1:
            return self.generate_for_target_h(target_h, deadline=deadline)

        # 合并去重（父进程）：直接复用矩阵级不变量
        merged: List[nx.Graph] = []
        seen: Dict = {}
        tasks = [(self.n_carbon, self.n_phosphorus, self.valence_p,
                  target_h, pf, self.max_atoms, deadline) for pf in prefixes]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_prefix_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    for G in fut.result():
                        M = _graph_to_matrix(G, self.atom_types)
                        inv, cols, vks = _leaf_invariants(M, self.atom_types)
                        _seen_add_matrix(seen, inv, cols, vks, M,
                                         self.atom_types, merged)
        except Exception:
            # 进程池不可用（打包/受限环境）→ 回退串行
            return self.generate_for_target_h(target_h, deadline=deadline)
        return merged


def _graph_to_matrix(G: nx.Graph, atom_types: List[str]) -> List[List[int]]:
    """图 → 邻接矩阵（按图节点顺序；用于并行合并去重）"""
    n = len(atom_types)
    idx = {node: k for k, node in enumerate(sorted(G.nodes()))}
    M = [[0] * n for _ in range(n)]
    for a, b, d in G.edges(data=True):
        ia, ib = idx[a], idx[b]
        M[ia][ib] = M[ib][ia] = int(d.get('bond_type', 1))
    return M
