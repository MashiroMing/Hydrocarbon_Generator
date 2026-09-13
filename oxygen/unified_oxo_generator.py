"""
统一含氧有机物生成器 (Phase 2d: 两阶段O分配 + Phase 2e: 化学过滤)

O 分配策略:
  第一阶段：种类间划分 — backbone_o (醚/过氧/O环) + substituent_o (OH/=O)
  第二阶段：种类内分配 — backbone→ether|peroxy|oring, substituent→OH|=O

优先级 (高→低): 醚桥 > 过氧桥 > O在环 > OH/=O
全局 WL 哈希去重（优先级更高先生成，后续冲突直接跳过）。
"""
import networkx as nx
from typing import Dict, Tuple, List
from collections import defaultdict
import itertools

from oxygen.oxo_generator import OxoSubstituentGenerator
from oxygen.ether_bridge_generator import BridgeGenerator
from oxygen.oring_generator import ORingGenerator
from oxygen.epoxide_generator import EpoxideGenerator
from oxygen.peroxy_substituent_generator import PeroxySubstituentGenerator
from oxygen.chem_filter import ChemFilter
from utils import GeneratorManager, canon_str_to_graph, graph_formula_parts

# O在环 跨 C 迭代的重原子数上限（防止 k≥4 高不饱和度下大骨架枚举爆炸）
_ORING_MAX_ATOMS = 10


# ============ WL 哈希工具 ============

def _graph_wl_key(G: nx.Graph) -> str:
    """计算图的全局 WL 哈希（含 label + oxo_counts + halogen_counts）"""
    H = nx.Graph()
    for n in G.nodes():
        label = G.nodes[n].get('label', 'C')
        oxo = G.nodes[n].get('oxo_counts', (0, 0))
        halo = G.nodes[n].get('halogen_counts', (0, 0, 0, 0))
        key = f"{label}:{oxo[0]},{oxo[1]}:{halo[0]},{halo[1]},{halo[2]},{halo[3]}"
        H.add_node(n, label=key)
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))
    return nx.weisfeiler_lehman_graph_hash(
        H, edge_attr='bond_type', node_attr='label'
    )


def _graphs_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """按 label + oxo_counts + halogen_counts + bond_type 精确判定同构。

    WL 图哈希对非同构图可能冲突（如 C1OC1C1CO1 与 C1OC2COC12），
    仅用哈希去重会漏结构；与 _graph_wl_key 的节点特征口径一致。
    """
    _eq = lambda x, y: x == y
    nm = nx.algorithms.isomorphism.generic_node_match(
        ['label', 'oxo_counts', 'halogen_counts'],
        ['C', (0, 0), (0, 0, 0, 0)],
        [_eq, _eq, _eq])
    em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
    return nx.is_isomorphic(g1, g2, node_match=nm, edge_match=em)


def _seen_add_safe(seen: Dict[str, list], wh: str,
                   g: nx.Graph) -> bool:
    """WL 哈希去重（非单射安全版）：同哈希必须同构才视为重复。

    seen: {wl: [graph, ...]}；返回 True 表示 g 是全新结构（已加入 seen），
    False 表示与既有结构重复（或未加入）。
    """
    existing = seen.get(wh)
    if existing is None:
        seen[wh] = [g]
        return True
    for other in existing:
        if _graphs_isomorphic(g, other):
            return False
    existing.append(g)
    return True


def _count_graph_carbon(G: nx.Graph) -> int:
    """从图节点 label 中统计真实 C 原子数"""
    return sum(1 for n in G.nodes()
               if G.nodes[n].get('label', 'C') == 'C')


def _normalize_atomic_graph(G: nx.Graph) -> nx.Graph:
    """将原子矩阵图 (O 为独立 backbone 节点) 转为骨架先行法兼容格式

    - 终端 O (degree=1) → 转为邻接原子上的 oxo_counts
      · single to C → oxo_counts=(1,0) (OH)
      · double to C → oxo_counts=(0,1) (=O)
      · to O (O-O terminal) → 保留 O 为 backbone 节点
    - 桥接 O (degree=2) → 保留为 label='O' backbone 节点
    """
    H = nx.Graph()
    oxo_map = {}  # node -> (oh_c, co_c)
    skip_nodes = set()
    keep_o_nodes = set()  # O 节点需保留

    for node in G.nodes():
        label = G.nodes[node].get('label', 'C')
        if label == 'C':
            H.add_node(node, label='C')
        elif label == 'O':
            deg = G.degree(node)
            if deg == 1:
                nb = list(G.neighbors(node))[0]
                nb_label = G.nodes[nb].get('label', 'C')
                bt = G[node][nb].get('bond_type', 'single')
                if nb_label == 'C':
                    # 终端 O on C → oxo_counts
                    oh, co = oxo_map.get(nb, (0, 0))
                    if bt == 'double':
                        oxo_map[nb] = (oh, co + 1)
                    else:
                        oxo_map[nb] = (oh + 1, co)
                    skip_nodes.add(node)
                elif nb_label == 'O':
                    # O-O 终端（如 HO-OH）→ 保留为 backbone O
                    keep_o_nodes.add(node)
                    keep_o_nodes.add(nb)
                    H.add_node(node, label='O')
                else:
                    keep_o_nodes.add(node)
                    H.add_node(node, label='O')
            elif deg == 2:
                keep_o_nodes.add(node)
                H.add_node(node, label='O')
            else:
                return None
        else:
            return None

    # 复制边（跳过已转换为 oxo_counts 的终端 O）
    for u, v, data in G.edges(data=True):
        if u in skip_nodes or v in skip_nodes:
            continue
        if u in keep_o_nodes or v in keep_o_nodes:
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))
        else:
            H.add_edge(u, v, bond_type=data.get('bond_type', 'single'))

    # 写入 oxo_counts
    for node, oxo in oxo_map.items():
        if node in H.nodes:
            H.nodes[node]['oxo_counts'] = oxo

    return H if H.number_of_nodes() > 0 else None


# ============ Phase 2d: O 分配规划器 ============

def _plan_backbone_allocations(backbone_o: int, has_ring: bool,
                               allow_oring: bool = True
                               ) -> List[Dict[str, int]]:
    """将 backbone_o 个 O 原子分配到 (ether, peroxy, epoxide, ooh, dioxy, oring)

    - ether / epoxide / oring：各消耗 1 O
    - peroxy 桥（R-O-O-R'）/ ooh 取代基（R-O-O-H）/ dioxy 双醚桥：各消耗 2 O

    allow_oring=False 强制 oring_n=0（当前骨架场景，oring 由跨C迭代处理）

    返回: [{'ether': e, 'peroxy': p, 'epoxide': x, 'ooh': h,
            'dioxy': d, 'oring': r}, ...]
    """
    if backbone_o == 0:
        return [{'ether': 0, 'peroxy': 0, 'epoxide': 0,
                 'ooh': 0, 'dioxy': 0, 'oring': 0}]

    plans = []
    for ether_n in range(backbone_o + 1):
        for peroxy_n in range(0, backbone_o - ether_n + 1, 2):  # 过氧桥成对
            rem = backbone_o - ether_n - peroxy_n
            if rem < 0:
                continue
            for ooh_n in range(0, rem // 2 + 1):                # OOH 成对（2 O）
                rem2 = rem - 2 * ooh_n
                for dioxy_n in range(0, rem2 // 2 + 1):         # 双醚桥成对（2 O）
                    rem3 = rem2 - 2 * dioxy_n
                    for epoxide_n in range(rem3 + 1):
                        oring_n = rem3 - epoxide_n
                        if oring_n < 0:
                            continue
                        if not allow_oring and oring_n > 0:
                            continue  # 当前骨架模式禁用 oring
                        if oring_n > 0 and not has_ring:
                            continue  # 无环骨架无需 O环 分配
                        plans.append({'ether': ether_n, 'peroxy': peroxy_n,
                                      'epoxide': epoxide_n, 'ooh': ooh_n,
                                      'dioxy': dioxy_n, 'oring': oring_n})
    return plans


def _plan_substituent_allocations(subst_o: int) -> List[Dict[str, int]]:
    """将 subst_o 个 O 原子分配到 (OH, CO) 两种取代基

    返回: [{'OH': oh, 'CO': co}, ...]
    """
    if subst_o == 0:
        return [{'OH': 0, 'CO': 0}]

    plans = []
    for co in range(subst_o + 1):
        oh = subst_o - co
        plans.append({'OH': oh, 'CO': co})
    return plans


# ============ Phase 2e: 化学过滤器实例 ============

# 全局默认过滤口径：'math'（八隅律数学完备，默认）| 'chem'（化学稳定性）
_chem_filter_mode = 'math'


# ============ 骨架缓存 / 阶段执行 / 并行任务（加速用） ============

from functools import lru_cache
import copy as _copy_mod


@lru_cache(maxsize=96)
def _hydrocarbon_skeleton_cache(nc: int, nh: int):
    """(nc, nh) → [(mol_type, nx.Graph)] 烃骨架，跨 generate() 调用复用。

    输出仅由 (nc, nh) 决定，缓存避免重复枚举（GUI 重复生成同一分子式时生效）。
    调用方拿到的是深拷贝（防污染），拷贝成本远低于重新枚举。
    """
    from utils import GeneratorManager, canon_str_to_graph
    mgr = GeneratorManager()
    raw = mgr.generate_all(nc, nh)
    result = []
    for mt, iso in raw:
        if isinstance(iso, str):
            g = canon_str_to_graph(iso, mt, mgr)
            if g is not None:
                result.append((mt, g))
        else:
            result.append((mt, iso))
    return result


def _build_skeletons(gen, nc: int, k: int, frag_map: dict,
                     use_fragments: bool = True):
    """骨架获取：片段库命中优先，否则用 (nc, nh) 缓存（深拷贝返回）"""
    nh = 2 * nc + 2 - 2 * k
    if nh < 0:
        return []
    if use_fragments and (nc, k) in frag_map:
        return [(fname, fg) for fname, fg in frag_map[(nc, k)]]
    cached = _hydrocarbon_skeleton_cache(nc, nh)
    return [(mt, _copy_mod.deepcopy(g)) for mt, g in cached]


def _make_phase_add(gen, n_carbon, n_oxygen, expected_h_set, seen, buckets):
    """阶段级 add()：分子式门禁前置（先于 WL 哈希拦截 k+1 溢出等无效候选）。

    seen 为 {wl: [graph, ...]}（碰撞安全去重，见 _seen_add_safe）。
    与 generate() 主 add() 同口径；同构图必有同公式，门禁前移不影响去重语义。
    """
    def add(g, mt, actual_c):
        parts = graph_formula_parts(g)
        if (parts[0] != n_carbon or parts[1] not in expected_h_set
                or parts[2] != n_oxygen):
            return
        wl = _graph_wl_key(g)
        if not _seen_add_safe(seen, wl, g):
            return
        fml = gen.oxo_gen.compute_graph_formula(g, actual_c)
        buckets[fml].append((mt, g))
    return add


def _phase2d_single(gen, k_skel, n_carbon, n_oxygen, k_min, k_max,
                    expected_h_set, frag_map, use_fragments=True):
    """在 gen 上执行单个 k_skel 的 Phase 2d 主体 → (buckets, seen)"""
    buckets = defaultdict(list)
    seen: Dict[str, list] = {}  # wl → [graph, ...]（碰撞安全去重）
    add = _make_phase_add(gen, n_carbon, n_oxygen, expected_h_set, seen,
                          buckets)
    skeletons = _build_skeletons(gen, n_carbon, k_skel, frag_map,
                                 use_fragments)
    if not skeletons:
        return {}, set()
    h_skel = 2 * n_carbon + 2 - 2 * k_skel
    has_ring = (k_skel >= 1)
    for backbone_o in range(n_oxygen + 1):
        subst_o = n_oxygen - backbone_o
        bb_plans = _plan_backbone_allocations(backbone_o, has_ring,
                                              allow_oring=False)
        s_plans = _plan_substituent_allocations(subst_o)
        for bb in bb_plans:
            for sp in s_plans:
                # 方案级 H 范围预筛（与主循环同口径）
                lo = h_skel - 2 * (sp['CO'] + bb['dioxy'] + bb['epoxide'])
                hi = h_skel - 2 * (sp['CO'] + bb['dioxy'])
                if lo < 0 or not any(lo <= h <= hi for h in expected_h_set):
                    continue
                gen._apply_plan(skeletons, bb, sp, n_oxygen, n_carbon, True,
                                add)
    return dict(buckets), seen


def _oring_single(gen, extra_c, nc_big, k, n_carbon, n_oxygen,
                  expected_h_set, frag_map, use_fragments=True):
    """单个 oring 跨C 任务（(extra_c, nc_big, k)）→ (buckets, seen)"""
    buckets = defaultdict(list)
    seen: Dict[str, list] = {}  # wl → [graph, ...]（碰撞安全去重）
    add = _make_phase_add(gen, n_carbon, n_oxygen, expected_h_set, seen,
                          buckets)
    skels = _build_skeletons(gen, nc_big, k, frag_map, use_fragments)
    if not skels:
        return {}, set()
    for mt, G in gen.oring_gen.generate(skels, extra_c):
        actual_c = _count_graph_carbon(G)
        if actual_c != n_carbon:
            continue
        remaining_o = n_oxygen - extra_c
        if remaining_o == 0:
            add(G, mt, actual_c)
        elif remaining_o > 0:
            for _, G2 in gen.oxo_gen.generate([(mt, G)], remaining_o,
                                              actual_c):
                add(G2, mt, actual_c)
    return dict(buckets), seen


# 进程池 worker：每个进程复用同一个 UnifiedOxoGenerator（含骨架缓存）
_worker_gen_ref = [None]


def _parallel_worker(task: dict):
    """任务: {'kind': 'phase2d', 'k': ...} 或 {'kind': 'oring', 'extra_c', 'nc_big', 'k'}"""
    gen = _worker_gen_ref[0]
    if gen is None:
        gen = UnifiedOxoGenerator(enable_chem_filter=False,
                                  enable_atomic_fallback=False)
        _worker_gen_ref[0] = gen
    n_carbon = task['n_carbon']
    n_oxygen = task['n_oxygen']
    exp_h = task['expected_h_set']
    fmap = task['frag_map']
    if task['kind'] == 'phase2d':
        return _phase2d_single(gen, task['k'], n_carbon, n_oxygen,
                               task['k_min'], task['k_max'], exp_h, fmap, True)
    return _oring_single(gen, task['extra_c'], task['nc_big'], task['k'],
                         n_carbon, n_oxygen, exp_h, fmap, True)


def set_chem_filter(mode: str):
    """全局设置化学过滤器口径：'math' 数学完备（默认）| 'chem' 化学稳定性

    仅影响之后新建的 UnifiedOxoGenerator 实例（未显式指定 chem_mode 时）。
    """
    global _chem_filter_mode
    if mode not in ('math', 'chem'):
        raise ValueError("mode 必须为 'math' 或 'chem'")
    _chem_filter_mode = mode


# ============ 主生成器 ============

class UnifiedOxoGenerator:
    """统一含氧有机物生成器 (Phase 2d+2e)"""

    def __init__(self, enable_chem_filter: bool = True,
                 enable_atomic_fallback: bool = True,
                 chem_mode: str = None):
        self.mgr = GeneratorManager()
        self.oxo_gen = OxoSubstituentGenerator()
        self.bridge_gen = BridgeGenerator()
        self.oring_gen = ORingGenerator()
        self.epoxide_gen = EpoxideGenerator()
        self.peroxy_sub_gen = PeroxySubstituentGenerator()
        self.enable_chem_filter = enable_chem_filter
        self.enable_atomic_fallback = enable_atomic_fallback
        # 过滤口径：'math'（八隅律数学完备，默认）| 'chem'（化学稳定性）
        self.chem_mode = chem_mode if chem_mode is not None else _chem_filter_mode

    def generate(self, n_carbon: int, n_oxygen: int,
                 k_range: Tuple[int, int] = None,
                 constraints=None,
                 use_parallel: bool = None,
                 enable_oring_cross_c: bool = True,
                 progress_cb=None) -> Dict[str, list]:
        """生成所有含氧有机物，按分子式分组

        constraints: StructureConstraint | None — 高级结构筛选
        use_parallel: None=自动（原子数≥7 且 k_skel 至少 2 档时进程池并行）|
                      True/False 强制；进程池不可用时自动回退串行
        enable_oring_cross_c: 快速模式开关 —— False 时跳过 oring 跨C 迭代
                      （省去 C+n 骨架枚举与 O 在环生成，约 30% 提速；代价是
                      缺"苯并二氧杂环戊烯"类稠合 O 杂环，数学完备性小尾巴）
        progress_cb(stage, done, total, merged_total) — 阶段进度回调（每完成一个
                      任务调用一次；merged_total 为去重后已并入的结构总数）
        """
        # ── 解析约束 ──
        req_frag = None
        if constraints is not None:
            try:
                req_frag = constraints.required_fragment
            except AttributeError:
                pass

        if n_oxygen == 0 and req_frag is None:
            return self._pure_hydrocarbon(n_carbon, k_range)

        # ── 约束模式：使用预置片段作为骨架 ──
        if req_frag is not None:
            return self._generate_from_fragment(
                n_carbon, n_oxygen, k_range, req_frag)

        if k_range is None:
            if n_carbon <= 4:
                k_limit = n_carbon + 1
            elif n_carbon <= 6:
                k_limit = 4
            else:
                k_limit = 3
            k_max = min((2 * n_carbon + 2) // 2, k_limit)
            k_range = (0, k_max)

        k_min, k_max = k_range
        # 目标公式约束：产物 H 数必须落在输入不饱和度 k 区间对应的 H 集合内
        # （H = 2n+2-2k；=O 取代使 H-2 → k+1，将被此集合排除）
        expected_h_set = {2 * n_carbon + 2 - 2 * k
                          for k in range(k_min, k_max + 1)}
        global_seen: Dict[str, list] = {}  # wl → [graph, ...]（碰撞安全去重）
        formula_buckets: Dict[str, list] = defaultdict(list)

        def _passes_formula_gate(g: nx.Graph) -> bool:
            """分子式门禁：C 数 / H 数 / O 数必须与用户输入一致"""
            parts = graph_formula_parts(g)
            return (parts[0] == n_carbon
                    and parts[1] in expected_h_set
                    and parts[2] == n_oxygen)

        def add(g: nx.Graph, mt: str, actual_c: int) -> bool:
            """返回 True 表示 g 为新结构已入桶（供进度统计）"""
            # 分子式门禁前置（便宜，先于 WL 哈希拦截 k+1 溢出等无效候选；
            # 同构图必有同公式，不影响去重语义）
            if not _passes_formula_gate(g):
                return False
            wl = _graph_wl_key(g)
            if not _seen_add_safe(global_seen, wl, g):
                return False
            fml = self.oxo_gen.compute_graph_formula(g, actual_c)
            formula_buckets[fml].append((mt, g))
            return True

        # ── 片段库索引：为 (nc, k) → [fragment_graphs] 预建查找表 ──
        _frag_map = {}  # {(nc, k): [('fragment_name', nx.Graph), ...]}
        try:
            from oxygen.fragment_library import list_fragments, get_fragment
            for fname in list_fragments():
                fg = get_fragment(fname)
                # 仅纯 C 骨架（无 O 原子）可作碳骨架注入
                if any(fg.nodes[n].get('label', 'C') == 'O' for n in fg.nodes()):
                    continue
                fc = sum(1 for n in fg.nodes()
                         if fg.nodes[n].get('label', 'C') == 'C')
                fh = 0
                for n in fg.nodes():
                    bl = sum(2 if fg[n][nb].get('bond_type') == 'double' else
                             3 if fg[n][nb].get('bond_type') == 'triple' else 1
                             for nb in fg.neighbors(n))
                    fh += max(0, 4 - bl)
                fk = (2 * fc + 2 - fh) // 2
                _frag_map.setdefault((fc, fk), []).append((fname, fg))
        except Exception:
            pass

        # =============================================
        # Phase 2d: 两阶段 O 分配 → 按优先级生成（串行或进程池并行）
        # =============================================
        # ── 骨架不饱和度下探（Phase 2d 主循环）──
        # 每个 =O 取代使产物不饱和度比骨架高 1（OH / 醚桥 / O 在环不改变 k）。
        # 因此目标 k 的含羰基结构（醛/酮/羧酸/酯）由 k' ∈ [k-n_oxygen, k] 的
        # 骨架 + O 修饰组合而成；add() 公式门禁负责放行/拦截（低 k 骨架 + =O
        # 提升到目标 k 的放行，k+1 溢出的拦截）。
        #
        # 并行策略：每个 k_skel 一个 phase2d 任务；oring 跨C 的每个
        # (extra_c, nc_big, k) 一个 oring 任务。worker 本地桶+本地去重，
        # 父进程按任务完成顺序合并进全局 add()（跨任务全局去重）。
        k_skel_min = max(0, k_min - n_oxygen)
        tasks = []
        for k_skel in range(k_skel_min, k_max + 1):
            tasks.append({'kind': 'phase2d', 'k': k_skel,
                          'n_carbon': n_carbon, 'n_oxygen': n_oxygen,
                          'k_min': k_min, 'k_max': k_max,
                          'expected_h_set': expected_h_set,
                          'frag_map': _frag_map})
        if enable_oring_cross_c:
            # ── Priority 3b: O在环 从 C+n 骨架 (跨 C 迭代) ──
            # C→O 替换是"碳骨架不含直接 C-C 边、仅靠 O 桥连"杂环（如苯并二氧
            # 杂环戊烯）的唯一骨架路径；nc_big 上限保护枚举量。
            max_extra_c = min(n_oxygen, 2)
            for extra_c in range(1, max_extra_c + 1):
                nc_big = n_carbon + extra_c
                if nc_big > _ORING_MAX_ATOMS:
                    continue
                big_k_start = max(k_skel_min, 1)
                for k in range(big_k_start,
                               min(k_max, (2 * nc_big + 2) // 2) + 1):
                    tasks.append({'kind': 'oring', 'extra_c': extra_c,
                                  'nc_big': nc_big, 'k': k,
                                  'n_carbon': n_carbon, 'n_oxygen': n_oxygen,
                                  'expected_h_set': expected_h_set,
                                  'frag_map': _frag_map})

        def _merge_task_result(result):
            """把单个任务桶并入全局（跨任务全局去重），返回新增结构数"""
            _b, _seen = result
            added = 0
            for fml, items in _b.items():
                for mt, g in items:
                    if add(g, mt, n_carbon):
                        added += 1
            return added

        if use_parallel is None:
            n_phase2d = sum(1 for t in tasks if t['kind'] == 'phase2d')
            use_parallel = (n_carbon + n_oxygen >= 7 and n_phase2d >= 2)
        _parallel_ok = False
        _merged_total = 0  # 去重后已并入结构数（进度回调用）
        if use_parallel and len(tasks) >= 2:
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                n_workers = min(4, len(tasks))
                with ProcessPoolExecutor(max_workers=n_workers) as _ex:
                    _futs = [_ex.submit(_parallel_worker, t) for t in tasks]
                    for _i, _fut in enumerate(as_completed(_futs), 1):
                        _merged_total += _merge_task_result(_fut.result())
                        if progress_cb is not None:
                            progress_cb('生成O机制', _i, len(tasks),
                                        _merged_total)
                _parallel_ok = True
            except Exception:
                _parallel_ok = False  # 进程池不可用（打包/环境）→ 回退串行
        if not _parallel_ok:
            for _i, t in enumerate(tasks, 1):
                if t['kind'] == 'phase2d':
                    result = _phase2d_single(
                        self, t['k'], n_carbon, n_oxygen, k_min, k_max,
                        expected_h_set, _frag_map, True)
                else:
                    result = _oring_single(
                        self, t['extra_c'], t['nc_big'], t['k'],
                        n_carbon, n_oxygen, expected_h_set, _frag_map, True)
                _merged_total += _merge_task_result(result)
                if progress_cb is not None:
                    progress_cb('生成O机制', _i, len(tasks), _merged_total)

        # =============================================
        # Phase 2e: 化学稳定性/数学完备 过滤
        # =============================================
        if self.enable_chem_filter:
            cf = ChemFilter(chem_mode=self.chem_mode)
            for fml in list(formula_buckets.keys()):
                passed = cf.filter(formula_buckets[fml])
                if passed:
                    formula_buckets[fml] = passed
                else:
                    del formula_buckets[fml]

        # =============================================
        # Phase F3: 邻接矩阵法补漏 (N ≤ 8 或 ≤10 但低不饱和度)
        # 高不饱和度 (k ≥ 4) 时原子矩阵枚举极慢且收益低，跳过
        # =============================================
        max_f3_atoms = 8 if k_max >= 4 else 10
        if (self.enable_atomic_fallback
                and n_carbon + n_oxygen <= max_f3_atoms
                and k_max < 4):
            from oxygen.atomic_matrix_gen import AtomicMatrixGenerator
            amg = AtomicMatrixGenerator(n_carbon, n_oxygen)
            if k_min == k_max:
                # 单点 k（GUI 常态）：目标 H 唯一 → 用目标键级剪枝版，
                # 大幅快于全量枚举且结果等价（内部预言机已验证 == MAYGEN）
                h_target = 2 * n_carbon + 2 - 2 * k_min
                atomic_graphs = amg.generate_for_target_h(h_target)
            else:
                atomic_graphs = [G for items in
                                 amg.generate_all_unsat().values()
                                 for G in items]

            for G in atomic_graphs:
                G_norm = _normalize_atomic_graph(G)
                if G_norm is None:
                    continue
                # 分子式门禁：原子矩阵枚举全部不饱和度，
                # 仅保留与用户输入 k/H 一致的产物（归一化后校验）
                if not _passes_formula_gate(G_norm):
                    continue
                wl = _graph_wl_key(G_norm)
                # 碰撞安全去重：与树机制产物同哈希时须同构才算重复
                if not _seen_add_safe(global_seen, wl, G_norm):
                    continue
                actual_c = _count_graph_carbon(G_norm)
                fml_norm = self.oxo_gen.compute_graph_formula(
                    G_norm, actual_c)
                formula_buckets[fml_norm].append(('atomic', G_norm))

        return dict(formula_buckets)

    # ── 单方案应用 ──

    def _apply_plan(self, skeletons, bb: dict, sp: dict,
                    n_oxygen: int, n_carbon: int,
                    need_bridges: bool, add):
        """应用一个具体的 O 分配方案到所有骨架"""
        ether_n, peroxy_n, epoxide_n, ooh_n, dioxy_n, oring_n = (
            bb['ether'], bb['peroxy'], bb['epoxide'], bb['ooh'],
            bb['dioxy'], bb['oring'])
        oh_n, co_n = sp['OH'], sp['CO']
        backbone_o = (ether_n + peroxy_n + epoxide_n
                      + 2 * ooh_n + 2 * dioxy_n + oring_n)
        subst_o = oh_n + co_n

        # 全纯 OH/=O — 最简单路径，统一处理
        if backbone_o == 0:
            for mt, G in self.oxo_gen.generate(skeletons, n_oxygen, n_carbon):
                add(G, mt, n_carbon)
            return

        # 有 backbone — 先改造骨架，再添加取代基
        if not need_bridges:
            return  # 高不饱和度跳桥接

        # Step A: 应用所有 backbone 策略
        base_graphs: List[Tuple[str, nx.Graph]] = []

        if ether_n > 0:
            for mt, G in self.bridge_gen.generate(skeletons, ether_n,
                                                   bridge_type='ether'):
                o_cnt = sum(1 for n in G.nodes()
                            if G.nodes[n].get('label') == 'O')
                if o_cnt == ether_n:
                    base_graphs.append((mt, G))
        else:
            base_graphs = list(skeletons)

        # Step B: 叠加过氧桥（R-O-O-R'，peroxy_n 为 O 原子数，每桥 2 O）
        if peroxy_n > 0:
            tmp = []
            for mt, G in base_graphs:
                for mt2, G2 in self.bridge_gen.generate(
                        [(mt, G)], peroxy_n, bridge_type='peroxy'):
                    o_cnt = sum(1 for n in G2.nodes()
                                if G2.nodes[n].get('label') == 'O')
                    if o_cnt == ether_n + peroxy_n:
                        tmp.append((mt2, G2))
            base_graphs = tmp if tmp else base_graphs

        # Step A': 叠加环氧（3 元 C-C-O 环；单键式 k+1 / 双键式 k 不变）
        if epoxide_n > 0:
            tmp = []
            for mt, G in base_graphs:
                for mt2, G2 in self.epoxide_gen.generate([(mt, G)], epoxide_n):
                    o_cnt = sum(1 for n in G2.nodes()
                                if G2.nodes[n].get('label') == 'O')
                    if o_cnt == ether_n + peroxy_n + epoxide_n:
                        tmp.append((mt2, G2))
            base_graphs = tmp if tmp else base_graphs

        # Step A'': 叠加氢过氧化物取代基 R-O-O-H（ooh_n 为基团数，每个 2 O）
        if ooh_n > 0:
            tmp = []
            for mt, G in base_graphs:
                for mt2, G2 in self.peroxy_sub_gen.generate([(mt, G)], ooh_n):
                    o_cnt = sum(1 for n in G2.nodes()
                                if G2.nodes[n].get('label') == 'O')
                    if o_cnt == ether_n + peroxy_n + epoxide_n + 2 * ooh_n:
                        tmp.append((mt2, G2))
            base_graphs = tmp if tmp else base_graphs

        # Step A''': 叠加双醚桥（C-O-C-O 4 元环，dioxy_n 为桥数，每个 2 O）
        if dioxy_n > 0:
            tmp = []
            for mt, G in base_graphs:
                for mt2, G2 in self.bridge_gen.generate(
                        [(mt, G)], 2 * dioxy_n, bridge_type='dioxy'):
                    o_cnt = sum(1 for n in G2.nodes()
                                if G2.nodes[n].get('label') == 'O')
                    if o_cnt == (ether_n + peroxy_n + epoxide_n
                                 + 2 * ooh_n + 2 * dioxy_n):
                        tmp.append((mt2, G2))
            base_graphs = tmp if tmp else base_graphs

        # Step C: 叠加 O 在环
        if oring_n > 0:
            tmp = []
            for mt, G in base_graphs:
                for mt2, G2 in self.oring_gen.generate([(mt, G)], oring_n):
                    if G2 is not None:
                        tmp.append((mt2, G2))
            base_graphs = tmp if tmp else base_graphs

        # Step D: 叠加 OH/=O 取代基
        if subst_o == 0:
            for mt, G in base_graphs:
                actual_c = _count_graph_carbon(G)
                add(G, mt, actual_c)
        else:
            for mt, G in base_graphs:
                actual_c = _count_graph_carbon(G)
                for _, G2 in self.oxo_gen.generate(
                        [(mt, G)], subst_o, actual_c):
                    add(G2, mt, _count_graph_carbon(G2))

    # ==================== 片段加速通道 ====================

    def _generate_from_fragment(self, n_carbon: int, n_oxygen: int,
                                 k_range, fragment_name: str) -> Dict[str, list]:
        """使用预置片段作为唯一骨架生成含氧有机物

        跳过全枚举，直接以片段图为起点应用 O/取代基。
        剩余 C 原子作为侧链添加到片段上。
        """
        from oxygen.fragment_library import (
            get_fragment, get_fragment_info, graph_contains_fragment)

        info = get_fragment_info(fragment_name)
        base_G = get_fragment(fragment_name)
        fc = info['n_carbon']
        fo = info['n_oxygen']
        remaining_o = n_oxygen - fo
        remaining_c = n_carbon - fc

        global_seen: Dict[str, list] = {}  # wl → [graph, ...]（碰撞安全去重）
        formula_buckets: Dict[str, list] = defaultdict(list)

        if k_range is None:
            k_range = (0, (2 * n_carbon + 2) // 2)
        k_min, k_max = k_range
        # 目标公式约束：产物 H 数必须落在输入不饱和度区间
        expected_h_set = {2 * n_carbon + 2 - 2 * k
                          for k in range(k_min, k_max + 1)}

        def add(g: nx.Graph, mt: str, actual_c: int):
            # 分子式门禁前置 + 碰撞安全去重（与 generate() 主路径同口径）
            parts = graph_formula_parts(g)
            if (parts[0] != n_carbon or parts[1] not in expected_h_set
                    or parts[2] != n_oxygen):
                return
            wl = _graph_wl_key(g)
            if not _seen_add_safe(global_seen, wl, g):
                return
            # 片段严格校验（"必含基团"口径）：产物必须仍包含所需片段子图。
            # 苯环等片段在醚桥/环氧/双氧/O 替换后可能被破坏（如苯并氧杂环
            # 已无完整交替单双键六元碳环），此类衍生结构不属于"必含苯环"，
            # 与 GUI 类型筛选（_has_benzene_ring）口径保持一致。
            if not graph_contains_fragment(g, fragment_name):
                return
            fml = self.oxo_gen.compute_graph_formula(g, actual_c)
            formula_buckets[fml].append((mt, g))

        # 片段本身
        if remaining_o == 0 and remaining_c == 0:
            add(base_G, fragment_name, fc)
            return dict(formula_buckets)

        # 片段 + 剩余 O（无剩余 C）
        if remaining_c == 0 and remaining_o >= 0:
            base_list = [(fragment_name, base_G)]
            if remaining_o > 0:
                for _, G2 in self.oxo_gen.generate(base_list, remaining_o, fc):
                    add(G2, fragment_name, fc)
            else:
                add(base_G, fragment_name, fc)
            return dict(formula_buckets)

        # 片段 + 剩余 C（含或不含剩余 O）
        # 侧链以烷基形式接入：生成 C{n}H{2n+2} 全饱和链，连接时自动消耗 1 H
        from utils import GeneratorManager, canon_str_to_graph
        mgr = GeneratorManager()

        # 侧链仅枚举 k=0（全饱和烷基链）
        for k_side in range(0, 1):
            nh_side = 2 * remaining_c + 2 - 2 * k_side
            if nh_side < 0:
                continue
            raw = mgr.generate_all(remaining_c, nh_side)
            for mt_side, iso_side in raw:
                if isinstance(iso_side, str):
                    side_G = canon_str_to_graph(iso_side, mt_side, mgr)
                    if side_G is None:
                        continue
                else:
                    side_G = iso_side

                # 合并：片段 + 侧链骨架
                merged = base_G.copy()
                offset = merged.number_of_nodes()
                side_map = {}
                for n in side_G.nodes():
                    new_n = offset + n
                    merged.add_node(new_n, label=side_G.nodes[n].get('label', 'C'))
                    side_map[n] = new_n
                for u, v, d in side_G.edges(data=True):
                    merged.add_edge(side_map[u], side_map[v],
                                    bond_type=d.get('bond_type', 'single'))
                # 连接片段和侧链（使用第一个可用的 H 位）
                if not _attach_side_chain(merged, base_G, side_map, offset):
                    continue  # 无法连接，跳过该合并
                # 合并后校验：碳数必须等于用户输入碳数
                if (sum(1 for n in merged.nodes()
                        if merged.nodes[n].get('label', 'C') == 'C')
                        != n_carbon):
                    continue

                # ── 方案A：复用主循环 backbone 机制（醚/过氧/双醚/环氧/OOH/=O/OH 组合）──
                # 直连合并骨架（如乙苯）可经"任意 C-C 单键醚桥 + =O"构造【侧链内部 O 桥】
                # 拓扑，从而在约束模式下生成 苯甲酸甲酯（苄基−CH3 内部醚桥 + =O）等，
                # 修复"约束模式只有交界处醚桥、缺内部醚桥"的覆盖缺口。
                extra_o = n_oxygen - fo
                if extra_o <= 0:
                    add(merged, fragment_name, n_carbon)
                else:
                    h_merged = graph_formula_parts(merged)[1]
                    for backbone_o in range(extra_o + 1):
                        bb_plans = _plan_backbone_allocations(
                            backbone_o, True, allow_oring=False)
                        s_plans = _plan_substituent_allocations(
                            extra_o - backbone_o)
                        for bb in bb_plans:
                            for sp in s_plans:
                                # 方案级 H 范围预筛（与主循环同口径）
                                lo = h_merged - 2 * (
                                    sp['CO'] + bb['dioxy'] + bb['epoxide'])
                                hi = h_merged - 2 * (
                                    sp['CO'] + bb['dioxy'])
                                if lo < 0 or not any(
                                        lo <= h <= hi for h in expected_h_set):
                                    continue
                                self._apply_plan(
                                    [(fragment_name, merged)], bb, sp,
                                    n_oxygen, n_carbon, True, add)

                # ── 醚桥式合并：片段 − O − 侧链（消耗 1 个骨架 O）──
                # 目标：约束模式下也能生成醚 / 酯（如 C7H8O+苯环 → 苯甲醚，
                # C7H6O2+苯环 → 甲酸苯酯 = 醚桥 + 侧碳上 =O）。
                if remaining_o >= 1:
                    merged_o = base_G.copy()
                    offset_o = merged_o.number_of_nodes()
                    side_map_o = {}
                    for n in side_G.nodes():
                        new_n = offset_o + n
                        merged_o.add_node(new_n,
                                          label=side_G.nodes[n].get('label', 'C'))
                        side_map_o[n] = new_n
                    for u, v, d in side_G.edges(data=True):
                        merged_o.add_edge(side_map_o[u], side_map_o[v],
                                          bond_type=d.get('bond_type', 'single'))
                    if _attach_side_chain_via_o(merged_o, base_G,
                                                side_map_o, offset_o):
                        if (sum(1 for n in merged_o.nodes()
                                if merged_o.nodes[n].get('label', 'C') == 'C')
                                == n_carbon):
                            rem_o2 = remaining_o - 1
                            if rem_o2 > 0:
                                for _, G2 in self.oxo_gen.generate(
                                        [(fragment_name, merged_o)],
                                        rem_o2, n_carbon):
                                    add(G2, fragment_name, n_carbon)
                            else:
                                add(merged_o, fragment_name, n_carbon)

        return dict(formula_buckets)

    # ==================== 纯烃模式 ====================

    def _pure_hydrocarbon(self, n_carbon: int, k_range) -> Dict[str, list]:
        if k_range is None:
            k_max = (2 * n_carbon + 2) // 2
            k_range = (0, k_max)
        k_min, k_max = k_range

        formula_buckets: Dict[str, list] = defaultdict(list)
        global_seen = set()

        for k in range(k_min, k_max + 1):
            nh = 2 * n_carbon + 2 - 2 * k
            if nh < 0:
                break
            raw = self.mgr.generate_all(n_carbon, nh)
            for mt, iso in raw:
                if isinstance(iso, str):
                    G = canon_str_to_graph(iso, mt, self.mgr)
                    if G is None:
                        continue
                else:
                    G = iso
                wl = _graph_wl_key(G)
                if wl in global_seen:
                    continue
                global_seen.add(wl)
                fml = f"C{n_carbon}H{nh}" if n_carbon > 1 else f"CH{nh}"
                formula_buckets[fml].append((mt, G))

        return dict(formula_buckets)

    # ==================== 片段辅助函数 ====================

def _attach_side_chain(merged, base_G, side_map, offset):
    """将侧链连接到片段上（使用第一个可用 H 位的 C 原子）

    Returns:
        bool: 是否成功连接（连接失败时返回 False，调用方应跳过该合并）
    """
    for frag_node in sorted(base_G.nodes()):
        if base_G.nodes[frag_node].get('label', 'C') != 'C':
            continue
        bl = 0
        for nb in merged.neighbors(frag_node):
            bt = merged[frag_node][nb].get('bond_type', 'single')
            bl += 2 if bt == 'double' else 3 if bt == 'triple' else 1
        if bl >= 4:
            continue
        for side_node in sorted(side_map.keys()):
            mapped_node = side_map[side_node]
            if merged.nodes[mapped_node].get('label', 'C') != 'C':
                continue
            sbl = 0
            for snb in merged.neighbors(mapped_node):
                sbt = merged[mapped_node][snb].get('bond_type', 'single')
                sbl += 2 if sbt == 'double' else 3 if sbt == 'triple' else 1
            if sbl >= 4:
                continue
            merged.add_edge(frag_node, mapped_node, bond_type='single')
            return True
    return False


def _attach_side_chain_via_o(merged, base_G, side_map, offset):
    """通过骨架 O 桥连接片段与侧链：片段 C - O - 侧链 C

    与 _attach_side_chain 的区别：连接处插入一个 label='O' 的骨架节点
    （醚键），用于生成醚 / 酯类结构（约束模式下的苯甲醚、甲酸苯酯等）。

    Returns:
        bool: 是否成功连接
    """
    for frag_node in sorted(base_G.nodes()):
        if base_G.nodes[frag_node].get('label', 'C') != 'C':
            continue
        bl = 0
        for nb in merged.neighbors(frag_node):
            bt = merged[frag_node][nb].get('bond_type', 'single')
            bl += 2 if bt == 'double' else 3 if bt == 'triple' else 1
        if bl >= 4:
            continue  # 无可用 H 位
        for side_node in sorted(side_map.keys()):
            mapped_node = side_map[side_node]
            if merged.nodes[mapped_node].get('label', 'C') != 'C':
                continue
            sbl = 0
            for snb in merged.neighbors(mapped_node):
                sbt = merged[mapped_node][snb].get('bond_type', 'single')
                sbl += 2 if sbt == 'double' else 3 if sbt == 'triple' else 1
            if sbl >= 4:
                continue
            o_node = merged.number_of_nodes()
            merged.add_node(o_node, label='O')
            merged.add_edge(frag_node, o_node, bond_type='single')
            merged.add_edge(o_node, mapped_node, bond_type='single')
            return True
    return False


# ==================== 测试 ====================

if __name__ == "__main__":
    gen = UnifiedOxoGenerator()

    test_cases = [(2, 1), (3, 1), (2, 2)]

    for nc, no in test_cases:
        print(f"\n{'='*60}")
        print(f"  C{nc} + {no}O — 统一生成 (优先级+跨C)")
        print(f"{'='*60}")
        results = gen.generate(nc, no)

        total = sum(len(v) for v in results.values())
        print(f"  分子式种类: {len(results)}, 总结构: {total}")

        for fml in sorted(results.keys()):
            cnt = len(results[fml])
            types = set(mt for mt, _ in results[fml])
            print(f"    {fml:10s}  {cnt:4d}  [{', '.join(sorted(types))}]")
