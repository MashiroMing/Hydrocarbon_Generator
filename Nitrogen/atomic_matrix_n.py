"""
Nitrogen/atomic_matrix_n.py — N 系完备性预言机（邻接矩阵原子级枚举）

从 oxygen/atomic_matrix_gen.py 移植，价态表参数化 C=4 / N=3 / O=2（中性八隅律，
与 MAYGEN 默认价态表一致）。数学上与 MAYGEN 等价：枚举重原子 (C,N,O) 间所有
合法连通图，目标 H 剪枝，作为 N 系生成器的**内部完备性真值**。

用途：
  1. 小公式（重原子 ≤ 10）直接作为权威生成器输出全部异构体；
  2. 作为树机制生成器（generator.py）的完备性预言机：检查"树 ⊆ 预言机"、
     报告覆盖率与差异（含 N 杂环等树机制未覆盖的结构）。

N 系与 O 系的关键差异（移植点）：
  - 原子类型 ['C']*nC + ['N']*nN + ['O']*nO，价态 4 / 3 / 2；
  - _max_bond_type: C-C=3, C-N=3（腈 C≡N）, N-N=2（偶氮 N=N；N≡N 无法再连
    其他原子，靠连通性自动排除）, C-O=2 / N-O=2（羰基/亚硝基）, O-O=1（过氧）；
  - O-O 快剪枝：最多 1 个 O-O 键（与 O 系一致，避免 O-O-O 过氧链）；
  - 有序生成三计数器 (n_conn_c, n_conn_n, n_conn_o)：各类型原子按
    [0,nC) / [nC,nC+nN) / [nC+nN,N) 索引区间内的前缀序接入。

路线 A（2026-09-21）口径注意：中性八隅律 **N=3** ⇒ **硝基 -NO₂ 不在空间内**
（需 5 价 N 或 N⁺-O⁻ 电荷分离式）；本枚举覆盖酰胺 / 肟 / 亚硝基 / 羟胺 /
氨基醇·酸·酯 / 含 O 与 N 的杂环等。
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
    - C-O / N-O: 2（羰基 C=O、亚硝基 N=O；C≡O、N≡O 三键不存在）
    - O-O: 1（过氧键仅单键）
    """
    if a1 == 'O' and a2 == 'O':
        return 1
    if a1 == 'O' or a2 == 'O':
        return 2
    if a1 == 'C' and a2 == 'C':
        return 3
    if a1 == 'C' or a2 == 'C':      # C-N
        return 3
    return 2                        # N-N


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


def _wl_hash_from_graph(G: nx.Graph) -> str:
    """图的 WL 哈希（并行合并去重用；属性名 'label' / 'bond_type'）

    与 :func:`_wl_hash_graph`（按矩阵 + atom_types、属性名 'atom'）用途不同：
    后者在单进程枚举内部用，前者用于**父进程跨子树合并**。
    两者都只是"分桶近似不变量"，桶内一律再用 VF2 精确同构确证 → 安全。
    """
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='label',
                                           edge_attr='bond_type')


#: 进程池启动成本估计（秒）。Windows/spawn 下每个 worker 都要重新 import
#: networkx 等依赖，本机实测 8 workers ≈ 8~9 s（12 workers 更高）。
_POOL_STARTUP_SEC = 8.0
#: 开池的最低"预计串行总成本"（秒）。低于此值直接串行——原因是**单子树规模不均**，
#: 用"首个子树 × 前缀数"外推会高估（实测 C5H5N 真值 4.3s，外推却 ~32s），
#: 若照此外推开池会白付 ~9s 启动费还更慢（实测 35s）。
#: 60s 意味着：最坏情况下多花 ≤60s 串行，而误开池的代价更高且有不确定性。
_POOL_MIN_TOTAL_SEC = 60.0


def estimate_cost(n_carbon: int, n_nitrogen: int, n_oxygen: int, target_h: int,
                  t_cap: float = 3.0, depth: int = 2,
                  max_atoms: int = 10) -> Dict:
    """**生成前代价预估**（供 GUI 做 ETA 提示，避免用户白等数小时）

    做法（不依赖任何外部工具）：枚举深度 ``depth`` 的合法前缀，取**第一个子树**
    串行枚举并计时（硬上限 ``t_cap`` 秒）：
      · 完成 → 按前缀数线性外推（对上界友好：各子树规模不均时偏保守）；
      · 超时 → 判定"过慢"，不给出精确数字（避免误导）。

    Returns:
        dict: n_prefixes / sample_seconds / sample_structures /
              est_seconds_serial / est_seconds_parallel / est_structures /
              too_slow / timeout / n_positions
    """
    gen = AtomicMatrixNGenerator(n_carbon, n_nitrogen, n_oxygen,
                                 max_atoms=max_atoms)
    out: Dict = {'n_positions': gen.num_positions, 'n_prefixes': 0,
                 'sample_seconds': 0.0, 'sample_structures': 0,
                 'est_seconds_serial': 0.0, 'est_seconds_parallel': 0.0,
                 'est_structures': 0, 'too_slow': False, 'timeout': None,
                 'estimable': False}
    if not gen._can_enumerate():
        out['timeout'] = f"重原子数 {gen.N} 超过上限 {max_atoms}"
        out['too_slow'] = True
        return out
    try:
        prefixes = gen.enumerate_prefixes(depth)
    except Exception as exc:                     # 前缀枚举本身失败 → 不阻断主流程
        out['timeout'] = f"{type(exc).__name__}: {exc}"
        return out
    out['n_prefixes'] = len(prefixes)
    if not prefixes:
        out['estimable'] = True                  # 空间为空（无合法前缀）
        return out

    t0 = time.time()
    try:
        sub = gen.generate_for_target_h(target_h, deadline=t0 + t_cap,
                                        prefix=prefixes[0])
    except TimeoutError:
        out['sample_seconds'] = time.time() - t0
        out['too_slow'] = True
        out['timeout'] = f"单个子树超过 {t_cap:.0f}s 仍未完成"
        return out
    dt = time.time() - t0
    out['sample_seconds'] = dt
    out['sample_structures'] = len(sub)
    out['estimable'] = True
    out['est_seconds_serial'] = dt * len(prefixes)
    # 并行收益按实测经验取 ~2.5×（子任务数 ≈ 前缀数，负载不均时更低）
    out['est_seconds_parallel'] = out['est_seconds_serial'] / 2.5
    out['est_structures'] = len(sub) * len(prefixes)
    return out


def _oracle_prefix_worker(args):
    """进程池 worker（顶层函数，可 pickle）：枚举**一个前缀子树**的唯一图

    args = (n_carbon, n_nitrogen, n_oxygen, target_h, prefix, max_atoms, deadline)
    """
    n_c, n_n, n_o, target_h, prefix, max_atoms, deadline = args
    gen = AtomicMatrixNGenerator(n_c, n_n, n_o, max_atoms=max_atoms)
    try:
        return gen.generate_for_target_h(target_h, deadline=deadline,
                                         prefix=tuple(prefix))
    except TimeoutError:
        return []


#: WL 颜色细化轮数（固定 3 轮：与 networkx 默认一致——更多轮收益递减而成本线性增长。
#: ⚠ 历史教训：曾用"n 轮 + 每轮 sorted(set(sig))"导致 C6H5N 从 70s 退化到 >10min）
_WL_ROUNDS = 3


def _wl_colors(M: List[List[int]], atom_types: List[str],
               rounds: int = _WL_ROUNDS) -> Tuple[int, ...]:
    """**矩阵级** WL 颜色细化（固定轮数）→ 颜色向量（元组）

    只做极少轮、且每轮用**排序编号**保证规范化（与顶点顺序无关）。
    仅作为"每叶一次性不变量"的输入，不在比较热路径中重复调用。
    """
    n = len(atom_types)
    if n == 0:
        return ()
    elems = sorted(set(atom_types))
    _ct = {t: i for i, t in enumerate(elems)}
    colors = [_ct[t] for t in atom_types]
    for _ in range(rounds):
        sig = []
        for i in range(n):
            mi = M[i]
            nb = []
            for j in range(n):
                if mi[j] > 0:
                    nb.append((colors[j], mi[j]))
            nb.sort()
            sig.append((colors[i], tuple(nb)))
        rank = {s: i for i, s in enumerate(sorted(set(sig)))}
        new = [rank[s] for s in sig]
        if new == colors:
            break
        colors = new
    return tuple(colors)


def _leaf_invariants(M: List[List[int]], atom_types: List[str]):
    """每个**叶子**计算一次：``(inv, colors, vkeys)``（全部为同构不变量）

    · ``colors``：WL 颜色（3 轮，规范化编号）；
    · ``vkeys[i]``：原子 i 的**着色邻域签名** = (自身颜色, 各邻居颜色上的键级之和)；
    · ``inv``：``sorted(vkeys)`` —— 比 WL 多重集**更细**的不变量，直接用作去重桶键
      （桶越细 → 桶内需做精确同构的次数越少）。

    这三者在同构下都不变，且**一次算好、多次复用**（旧实现每次比较都重算，是
    性能崩掉的主因）。
    """
    n = len(atom_types)
    colors = _wl_colors(M, atom_types)
    vkeys = []
    for i in range(n):
        mi = M[i]
        agg: Dict[int, int] = {}
        for j in range(n):
            if mi[j] > 0:
                c = colors[j]
                agg[c] = agg.get(c, 0) + mi[j]
        vkeys.append((colors[i], tuple(sorted(agg.items()))))
    return tuple(sorted(vkeys)), colors, tuple(vkeys)


def _matrix_iso_fast(M1, vk1, M2, vk2) -> bool:
    """精确同构（只做回溯；候选类与签名由 :func:`_leaf_invariants` 预先算好）

    正确性：候选仅限"着色邻域签名相同"的顶点（同构的必要条件），
    回溯逐步校验邻接矩阵完全一致 → 得到双射即同构。
    """
    n = len(vk1)
    if n != len(vk2):
        return False
    cand = [[j for j in range(n) if vk2[j] == vk1[i]] for i in range(n)]
    if any(not cc for cc in cand):
        return False
    # 候选少的先映射（剪枝更强）
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
    """矩阵级去重：桶键 = 叶子不变量 ``inv``；桶内做精确同构；命中才建图"""
    bucket = seen.get(inv)
    if bucket is None:
        seen[inv] = [([row[:] for row in M], vkeys)]
        results.append(_matrix_to_graph(M, atom_types))
        return
    for M0, vk0 in bucket:
        if _matrix_iso_fast(M, vkeys, M0, vk0):
            return                                     # 真重复
    bucket.append(([row[:] for row in M], vkeys))
    results.append(_matrix_to_graph(M, atom_types))


# ============ 主预言机 ============

class AtomicMatrixNGenerator:
    """N 系原子级完备预言机 / 生成器（C/H/N(/O)，C=4 N=3 O=2 价）"""

    def __init__(self, n_carbon: int, n_nitrogen: int, n_oxygen: int = 0,
                 max_atoms: int = 10):
        """
        Args:
            n_carbon: 碳原子数
            n_nitrogen: 氮原子数
            n_oxygen: 氧原子数（**末位默认 0** → 既有两参调用逐位不变）
            max_atoms: 重原子总数上限（超过则不枚举）
        """
        self.n_carbon = n_carbon
        self.n_nitrogen = n_nitrogen
        self.n_oxygen = n_oxygen
        self.N = n_carbon + n_nitrogen + n_oxygen
        self.max_atoms = max_atoms
        self.atom_types = ['C'] * n_carbon + ['N'] * n_nitrogen + ['O'] * n_oxygen
        _V = {'C': 4, 'N': 3, 'O': 2}
        self.valences = [_V[a] for a in self.atom_types]
        self.total_valence = sum(self.valences)
        self.num_positions = self.N * (self.N - 1) // 2
        # 各类型索引区间的起点（C: 0 / N: nC / O: nC+nN）
        self._off_n = n_carbon
        self._off_o = n_carbon + n_nitrogen

    def _can_enumerate(self) -> bool:
        """检查是否在可枚举范围内"""
        return self.N <= self.max_atoms

    # ── 有序生成：类型内前缀索引判定 ──

    def _is_next_of_type(self, idx: int, n_conn_c: int, n_conn_n: int,
                         n_conn_o: int) -> bool:
        """原子 idx 是否是「其所属类型内应当接入的下一个」（有序生成前缀序）

        类型索引区间：C = [0, nC)、N = [nC, nC+nN)、O = [nC+nN, N)。
        安全性：任一连通图按「BFS 发现序在类型内编号」重标号后，
        其所有"单新键"首键均满足该条件 → 不漏结构。
        """
        if idx < self._off_n:
            return idx == n_conn_c
        if idx < self._off_o:
            return idx == self._off_n + n_conn_n
        return idx == self._off_o + n_conn_o

    def _conn_increments(self, i_new: bool, i: int, j_new: bool, j: int):
        """(单新键) 统计本步新增连通的 C / N / O 原子数"""
        dc = dn = do = 0
        for _new, _idx in ((i_new, i), (j_new, j)):
            if not _new:
                continue
            if _idx < self._off_n:
                dc += 1
            elif _idx < self._off_o:
                dn += 1
            else:
                do += 1
        return dc, dn, do

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
                max_v = {'C': 4, 'N': 3, 'O': 2}.get(atom, 4)
                used = sum(
                    3 if G[node][nb].get('bond_type') == 'triple' else
                    2 if G[node][nb].get('bond_type') == 'double' else 1
                    for nb in G.neighbors(node)
                )
                h_count += max(0, max_v - used)

            n_c = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'C')
            n_n = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'N')
            n_o = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'O')
            base = f"C{n_c}" if n_c > 1 else "C"
            fml = f"{base}H{h_count}"
            if n_n > 0:
                fml += f"N{n_n}" if n_n > 1 else "N"
            if n_o > 0:
                fml += f"O{n_o}" if n_o > 1 else "O"
            buckets[fml].append(G)

        return dict(buckets)

    def generate_for_target_h(self, target_h: int,
                              deadline: Optional[float] = None,
                              prefix: Tuple[int, ...] = ()) -> List[nx.Graph]:
        """生成 H 原子数恰为 target_h 的全部连通、价态合法图（目标键级剪枝）

        数学上完备（与 MAYGEN 等价）：邻接矩阵枚举 + 目标 H 键级总和剪枝，
        用作"树机制边界"的**内部完备性预言机**，无需任何外部工具。

        剪枝依据（精确，与 O 系 atomic_matrix_gen 同源）：
          - H = total_valence − 2·Σ键级，目标 H ⟺ 目标已用价 valence_target；
          - used_valence > valence_target → 剪；剩余位置最大可贡献价不足 → 剪；
          - 连通下界：非零键数 + 剩余位置 < N−1 → 剪；
          - 孤立原子：度 0 且已过其最后可连位置 → 剪；
          - 有序生成（BFS 前缀标号）：新原子按索引顺序加入连通前缀
            （C 按 0..nC-1，N 按 nC..，O 按 nC+nN..），消除 ~N! 倍标号冗余。

        Args:
            target_h: 目标氢原子数
            deadline: 可选超时（time.time() 绝对时刻）；超时抛 TimeoutError。
            prefix: **可选前缀**（位置 0..len(prefix)-1 的取值元组）。
                    用于并行切分：每个前缀对应一棵互不重叠的子树，
                    结果并集 = 全空间（配合父进程去重即得完整唯一图集）。

        Returns:
            唯一图列表（label='C'/'N'/'O'，bond_type），H 数 == target_h。
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

        # ── 前缀回放：把 prefix 指定的前若干位置固定下来（并行切分用）──
        _st = self._replay_prefix(prefix, M, row_sums, pos_max_bond, last_pos)
        if _st is None:
            return []          # 前缀本身非法 → 该子树为空
        p_start, used_valence0, nz_bonds0, cc0, cn0, co0 = _st

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
                      n_conn_c: int, n_conn_n: int, n_conn_o: int):
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
                if n_conn_c + n_conn_n + n_conn_o != N:
                    return
                if not _is_graph_connected(M, N):
                    return  # 双新键可能产生未合并的临时分量，须显式验证连通
                _inv, _cols, _vks = _leaf_invariants(M, self.atom_types)
                _seen_add_matrix(seen, _inv, _cols, _vks, M, self.atom_types,
                                 results)
                return

            i, j = self._position_to_indices(pos_idx)
            max_bond = pos_max_bond[pos_idx]

            # O-O 快剪枝（与 O 系一致）：最多 1 个 O-O 键，避免 O-O-O 过氧链
            if self.atom_types[i] == 'O' and self.atom_types[j] == 'O':
                _oo = 0
                for _k in range(N):
                    if self.atom_types[_k] == 'O' and M[i][_k] > 0:
                        _oo += 1
                    if self.atom_types[_k] == 'O' and M[j][_k] > 0:
                        _oo += 1
                if _oo >= 2:          # 已有 O-O 键，不能再加
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
                    # 有序生成：单新键类型内前缀序（安全，消标号冗余）
                    # 双新键全放行（可能先成临时分量后并入主分量，末端连通性兜底）
                    if not (i_new and j_new):
                        if i_new and not self._is_next_of_type(
                                i, n_conn_c, n_conn_n, n_conn_o):
                            continue
                        if j_new and not self._is_next_of_type(
                                j, n_conn_c, n_conn_n, n_conn_o):
                            continue
                    dc, dn, do = self._conn_increments(i_new, i, j_new, j)
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bv
                    row_sums[j] += bv
                    backtrack(pos_idx + 1, used_valence + 2 * bv,
                              nz_bonds + 1,
                              n_conn_c + dc, n_conn_n + dn, n_conn_o + do)
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
                              n_conn_c, n_conn_n, n_conn_o)
                    M[i][j] = M[j][i] = 0

        backtrack(p_start, used_valence0, nz_bonds0, cc0, cn0, co0)
        return results

    # ── 前缀机制（并行切分的基础）──

    def _precompute(self):
        """预计算：每位置最大键级 + 每原子最后出现位置（连通剪枝用）"""
        pos_max_bond = [0] * self.num_positions
        last_pos = [0] * self.N
        for p in range(self.num_positions):
            i, j = self._position_to_indices(p)
            pos_max_bond[p] = _max_bond_type(self.atom_types[i],
                                             self.atom_types[j])
            last_pos[i] = max(last_pos[i], p)
            last_pos[j] = max(last_pos[j], p)
        return pos_max_bond, last_pos

    def _replay_prefix(self, prefix, M, row_sums, pos_max_bond, last_pos):
        """把 prefix（位置 0..k-1 的键级取值）回放到矩阵状态上

        与主回溯使用**同一套**校验（max_bond / O-O / 价态 / 有序生成 / 孤立原子），
        故前缀合法性与主搜索完全一致（单一事实源，避免两处规则漂移）。

        Returns:
            ``(p_start, used_valence, nz_bonds, n_conn_c, n_conn_n, n_conn_o)``
            或 ``None``（前缀非法 → 该子树为空）
        """
        if len(prefix) > self.num_positions:
            return None
        N = self.N
        used = nz = cc = cn = co = 0
        for p, bt in enumerate(prefix):
            i, j = self._position_to_indices(p)
            mb = pos_max_bond[p]
            if self.atom_types[i] == 'O' and self.atom_types[j] == 'O':
                _oo = 0
                for _k in range(N):
                    if self.atom_types[_k] == 'O' and M[i][_k] > 0:
                        _oo += 1
                    if self.atom_types[_k] == 'O' and M[j][_k] > 0:
                        _oo += 1
                if _oo >= 2:
                    mb = 0
            if bt < 0 or bt > mb:
                return None
            bv = _bond_valence(bt)
            i_new = (row_sums[i] == 0)
            j_new = (row_sums[j] == 0)
            if bt > 0:
                if (row_sums[i] + bv > self.valences[i]
                        or row_sums[j] + bv > self.valences[j]):
                    return None
                if not (i_new and j_new):
                    if i_new and not self._is_next_of_type(i, cc, cn, co):
                        return None
                    if j_new and not self._is_next_of_type(j, cc, cn, co):
                        return None
                dc, dn, do = self._conn_increments(i_new, i, j_new, j)
                M[i][j] = M[j][i] = bt
                row_sums[i] += bv
                row_sums[j] += bv
                used += 2 * bv
                nz += 1
                cc += dc
                cn += dn
                co += do
            else:
                if row_sums[i] == 0 and last_pos[i] == p:
                    return None
                if row_sums[j] == 0 and last_pos[j] == p:
                    return None
        return len(prefix), used, nz, cc, cn, co

    def enumerate_prefixes(self, depth: int) -> List[Tuple[int, ...]]:
        """枚举全部**通过校验**的深度 ``depth`` 前缀（并行切分用）

        合法性统一由 :meth:`_replay_prefix` 判定（单一事实源）。
        这些前缀互不重叠，且并集覆盖整个搜索空间。
        """
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
            for bt in range(_max_bond_type(self.atom_types[i],
                                           self.atom_types[j]) + 1):
                cand = prefix + (bt,)
                if self._replay_prefix(cand,
                                       [[0] * self.N for _ in range(self.N)],
                                       [0] * self.N, pos_max_bond,
                                       last_pos) is not None:
                    rec(p + 1, cand)

        rec(0, ())
        return out

    def generate_for_target_h_parallel(self, target_h: int,
                                       max_workers: Optional[int] = None,
                                       split_depth: Optional[int] = None,
                                       deadline: Optional[float] = None,
                                       stream_cb=None) -> List[nx.Graph]:
        """**并行版** generate_for_target_h（前缀切分 + 进程池 + 父进程去重）

        等价性保证：
          · 各前缀子树互不重叠、并集 = 全搜索空间（前缀由同一套规则校验）；
          · 父进程用 WL 哈希分桶 + VF2 精确同构**跨子树去重** →
            返回的唯一图集合与串行版**逐元素一致**（只差顺序）。

        Args:
            stream_cb: 可选回调 ``f(subtree_graphs)``，**每个子树完成即调用**
                （父进程侧、乱序），用于"边算边上屏"。回调异常被吞掉，不影响结果。

        适用：重原子数 ≥ 6 的目标 H 档位（子任务足够大，进程开销可摊薄）。
        进程池不可用时自动退回串行，行为不变。
        """
        import os
        if max_workers is None:
            max_workers = min(12, os.cpu_count() or 4)
        if max_workers <= 1 or self.num_positions <= 2:
            res = self.generate_for_target_h(target_h, deadline=deadline)
            if stream_cb:
                try:
                    stream_cb(res)
                except Exception:
                    pass
            return res

        if split_depth is None:
            # 选**最小**深度使子任务数达到 ~3×工作进程数；深度硬上限 4
            # （前缀枚举本身要廉价：深度 4 最多 4^4=256 条候选）。
            # ⚠ 不可按"前缀数 ≥ k×workers"无限加深：合法前缀数增长饱和
            #   （如 C6H5N 深度 1..5 仅 4/13/20/24/27 条），加深会把
            #   前缀枚举本身变成主成本，反而丧失并行收益（实测教训）。
            best_depth, best_n = 1, len(self.enumerate_prefixes(1))
            for d in (2, 3, 4):
                if d >= self.num_positions or best_n >= 3 * max_workers:
                    break
                n_d = len(self.enumerate_prefixes(d))
                if n_d > best_n:
                    best_depth, best_n = d, n_d
            split_depth = best_depth
        prefixes = self.enumerate_prefixes(split_depth)
        if len(prefixes) <= 1:
            res = self.generate_for_target_h(target_h, deadline=deadline)
            if stream_cb:
                try:
                    stream_cb(res)
                except Exception:
                    pass
            return res

        def _emit(sub):
            if stream_cb:
                try:
                    stream_cb(sub)
                except Exception:
                    pass

        # ── 是否值得开进程池？先串行探一个子树，估算串行总成本 ──
        #   小公式（预计 < _POOL_STARTUP_SEC × _POOL_MIN_GAIN）直接走串行，
        #   避免被 spawn 启动开销吃掉（实测 C4H9NO 串行 2.8s vs 开池 9.5s）。
        _t0 = time.time()
        first = self.generate_for_target_h(target_h, deadline=deadline,
                                           prefix=prefixes[0])
        probe = time.time() - _t0
        est_total = probe * len(prefixes)
        merged: List[nx.Graph] = []
        seen: Dict[str, List[nx.Graph]] = {}
        if est_total < _POOL_MIN_TOTAL_SEC:
            # 不值得开池：把**已探完的子树复用**，其余前缀串行补完（不浪费探测）
            _emit(first)
            for g in first:
                _seen_add(seen, _wl_hash_from_graph(g), g, merged)
            for p in prefixes[1:]:
                sub = self.generate_for_target_h(target_h, deadline=deadline,
                                                 prefix=p)
                _emit(sub)
                for g in sub:
                    _seen_add(seen, _wl_hash_from_graph(g), g, merged)
            return merged

        # 探测结果直接复用（不再重复算 prefixes[0] 的子树）
        _emit(first)
        for g in first:
            _seen_add(seen, _wl_hash_from_graph(g), g, merged)

        args = [(self.n_carbon, self.n_nitrogen, self.n_oxygen, target_h,
                 p, self.max_atoms, deadline) for p in prefixes[1:]]
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                # imap_unordered：子树**完成即回传**（流式上屏 + 更好的负载利用）
                for sub in ex.imap_unordered(_oracle_prefix_worker, args,
                                             chunksize=1):
                    _emit(sub)
                    for g in sub:
                        _seen_add(seen, _wl_hash_from_graph(g), g, merged)
        except Exception as exc:
            # 进程池不可用（受限沙箱 / 冻结环境 / 主模块缺 __main__ 守卫等）
            # → 串行兜底：结果不变，但**必须可见**，避免"以为并行了"的错觉。
            print(f"  [预言机] 进程池不可用（{type(exc).__name__}: "
                  f"{str(exc)[:80]}），已退回串行枚举（结果不变，仅慢）")
            return self.generate_for_target_h(target_h, deadline=deadline)
        return merged

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

        def backtrack(pos_idx: int, n_conn_c: int, n_conn_n: int, n_conn_o: int):
            if pos_idx == self.num_positions:
                if n_conn_c + n_conn_n + n_conn_o != N:
                    return
                if not _is_graph_connected(M, N):
                    return
                _inv, _cols, _vks = _leaf_invariants(M, self.atom_types)
                _seen_add_matrix(seen, _inv, _cols, _vks, M, self.atom_types,
                                 results)
                return

            i, j = self._position_to_indices(pos_idx)
            max_bond = _max_bond_type(self.atom_types[i], self.atom_types[j])

            # O-O 快剪枝（与 generate_for_target_h 一致）
            if self.atom_types[i] == 'O' and self.atom_types[j] == 'O':
                _oo = 0
                for _k in range(N):
                    if self.atom_types[_k] == 'O' and M[i][_k] > 0:
                        _oo += 1
                    if self.atom_types[_k] == 'O' and M[j][_k] > 0:
                        _oo += 1
                if _oo >= 2:
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
                    # 有序生成：单新键类型内前缀序（同 generate_for_target_h）
                    if not (i_new and j_new):
                        if i_new and not self._is_next_of_type(
                                i, n_conn_c, n_conn_n, n_conn_o):
                            continue
                        if j_new and not self._is_next_of_type(
                                j, n_conn_c, n_conn_n, n_conn_o):
                            continue
                    dc, dn, do = self._conn_increments(i_new, i, j_new, j)
                    M[i][j] = M[j][i] = bt
                    row_sums[i] += bv
                    row_sums[j] += bv
                    backtrack(pos_idx + 1, n_conn_c + dc, n_conn_n + dn,
                              n_conn_o + do)
                    row_sums[i] -= bv
                    row_sums[j] -= bv
                    M[i][j] = M[j][i] = 0
                else:
                    M[i][j] = M[j][i] = 0
                    backtrack(pos_idx + 1, n_conn_c, n_conn_n, n_conn_o)
                    M[i][j] = M[j][i] = 0

        backtrack(0, 0, 0, 0)
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
