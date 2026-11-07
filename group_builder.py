# -*- coding: utf-8 -*-
"""group_builder.py — **基团构建模式**：由目标分子式反推母体骨架 + 基团挂载

## 定位（重要）

这是**独立模式**，与"完整枚举"（原子矩阵预言机）并列，二者语义不同：

| | 完整枚举（预言机） | **基团构建模式（本模块）** |
|---|---|---|
| 空间 | 中性八隅律下**全部** C/H/N/O(/卤) 连通图 | **碳骨架 + 所选基团**连接的产物 |
| 完备性 | 完备（数学真值） | **非完备**（不含杂原子在环内/链内者，如吡啶、环氧） |
| 速度 | 随重原子数指数增长（C6H5NO2 ≈ 44 万结构 / 数小时） | 秒级（骨架秒级 + 挂载组合枚举） |
| 能否出硝基苯 | **不能**（硝基需 N⁺–O⁻，超出 N=3 口径） | **能**（毫秒级） |

典型用途：`C6H5NO2` + 勾选[硝基] → 苯环/椅式等 C6H6 骨架 + 硝基 → **硝基苯**。

## 流程

1. **反推母体骨架分子式**：`parent = target − Σ count·group_delta(g)`
   （增量由 `special_groups.group_delta` 从结构自动推导，数据驱动）；
2. **枚举母体骨架**：调用项目既有烃类生成器（`GeneratorManager`），成本低；
3. **基团挂载位置枚举**：按原子顺序分配各基团（每基团消耗宿主 1 个 H，
   受元素价态约束；同一原子可挂多个，如 –CCl₃ / 偕二醇）；
4. **去重**：WL 哈希分桶 + 精确同构（含形式电荷比较）；
5. **分子式门禁**：`utils.formula_matches_input` 收口（含卤素与形式电荷口径）。

所有产物都是**显式原子图**（每个重原子是节点），可直接走既有
渲染 / 3D / `.gjf` 导出链路。
"""

import itertools
from typing import Dict, List, Optional, Tuple

__all__ = ['derive_parent', 'build_from_groups', 'placement_count',
           'GroupBuildResult', 'describe_mode']


class GroupBuildResult:
    """构建模式结果（graph 列表 + 诊断信息）"""

    def __init__(self):
        self.graphs: List = []
        self.parent_formula: str = ''
        self.n_skeletons: int = 0
        self.n_placements: int = 0
        self.n_gate_rejected: int = 0
        self.error: Optional[str] = None
        self.warnings: List[str] = []

    def __repr__(self):
        return (f"<GroupBuildResult {len(self.graphs)} 结构 "
                f"parent={self.parent_formula} skel={self.n_skeletons} "
                f"place={self.n_placements} gate_rej={self.n_gate_rejected}>")


def _fmt_fml(n_c, n_h, n_n, n_o, halo) -> str:
    base = f"C{n_c}" if n_c > 1 else "C"
    base += f"H{n_h}"
    if n_n:
        base += f"N{n_n}" if n_n > 1 else "N"
    if n_o:
        base += f"O{n_o}" if n_o > 1 else "O"
    for idx, elem in ((2, 'Br'), (1, 'Cl'), (0, 'F'), (3, 'I')):
        if halo[idx]:
            base += f"{elem}{halo[idx]}" if halo[idx] > 1 else elem
    return base


def derive_parent(target: Tuple[int, int, int, int, Tuple[int, int, int, int]],
                  counts: Dict[str, int]):
    """由目标组成与所选基团反推**母体骨架**组成

    Args:
        target: ``(nC, nH, nN, nO, (f, cl, br, i))``（同
            ``utils.graph_formula_parts_full`` 口径）
        counts: ``{group_key: 个数}``

    Returns:
        ``((parent_C, parent_H), error)``；parent 仅含 C/H（卤素交给基团），
        error 为 None 表示成功。

    校验：① 基团键名存在；② 各元素差值非负；③ 母体 H 数合法（≥0）。
    """
    from special_groups import GROUP_LIBRARY, group_delta

    n_c, n_h, n_n, n_o, halo = target
    d_c = d_h = d_n = d_o = d_x = 0
    for key, m in counts.items():
        if m <= 0:
            continue
        if key not in GROUP_LIBRARY:
            return None, f"未知基团 {key!r}"
        dc, dh, dn, do, dx, _dq = group_delta(key)
        d_c += m * dc
        d_h += m * dh
        d_n += m * dn
        d_o += m * do
        d_x += m * dx

    p_c = n_c - d_c
    p_h = n_h - d_h
    p_n = n_n - d_n
    p_o = n_o - d_o
    p_x = sum(halo) - d_x

    if p_n != 0 or p_o != 0 or p_x != 0:
        return None, (f"所选基团无法解释目标分子式中的杂原子："
                      f"母体剩余 N={p_n} O={p_o} 卤={p_x}"
                      f"（构建模式只覆盖「碳骨架 + 所选基团」）")
    if p_c < 0:
        return None, f"碳数不足（母体 C={p_c}）"
    if p_h < 0:
        return None, f"氢数不足（母体 H={p_h}）"
    if p_c == 0 and p_h == 0:
        return None, "母体骨架为空（请减少基团数量或增大目标分子）"
    return (p_c, p_h), None


def placement_count(G, total_groups: int) -> int:
    """粗略估计某骨架上的挂载方案数（仅用于规模预警，非精确可行性判断）"""
    from utils import atom_valence
    cap = 0
    for node in G.nodes():
        d = G.nodes[node]
        load = sum(3 if G[node][nb].get('bond_type') == 'triple'
                   else 2 if G[node][nb].get('bond_type') == 'double' else 1
                   for nb in G.neighbors(node))
        cap += max(0, atom_valence(d.get('label', 'C'), d) - load)
    if cap < total_groups:
        return 0
    # 组合上界：把 total_groups 个基团放到 cap 个"位置"上（可重复）
    from math import comb
    return comb(cap + total_groups - 1, total_groups) if total_groups else 1


def _avail_h(G, node: int) -> int:
    from utils import atom_valence
    d = G.nodes[node]
    load = sum(3 if G[node][nb].get('bond_type') == 'triple'
               else 2 if G[node][nb].get('bond_type') == 'double' else 1
               for nb in G.neighbors(node))
    return max(0, atom_valence(d.get('label', 'C'), d) - load)


def _distributions(remaining: List[Tuple[str, int]], max_total: int):
    """把 remaining 中的基团分配到**当前原子**上（总量 ≤ max_total）

    产出 ``[(key, 个数), ...]``（仅含 >0 项）。按类型顺序枚举，
    保证每个分配方案恰好出现一次（配合"按原子顺序"外层循环 → 无重复）。
    """
    keys = [k for k, _ in remaining]

    def rec(i: int, left: int, cur: List[Tuple[str, int]]):
        if i == len(keys):
            yield list(cur)
            return
        k, m = remaining[i]
        for take in range(0, min(m, left) + 1):
            if take:
                cur.append((k, take))
            yield from rec(i + 1, left - take, cur)
            if take:
                cur.pop()

    yield from rec(0, max_total, [])


def build_from_groups(target: Tuple[int, int, int, int,
                                    Tuple[int, int, int, int]],
                      counts: Dict[str, int],
                      skeleton_provider,
                      max_graphs: int = 200000,
                      max_skel_atoms: int = 16) -> GroupBuildResult:
    """基团构建模式主入口

    Args:
        target: 目标组成 ``(nC, nH, nN, nO, (f,cl,br,i))``
        counts: ``{group_key: 个数}``（如 ``{'nitro': 1}``）
        skeleton_provider: 回调 ``f(n_carbon, n_hydrogen) -> [(mol_type, graph|str), ...]``
            （通常传 ``Main.gen_mgr.generate_all``；便于测试时注入假骨架）
        max_graphs: 结果上限（防爆）
        max_skel_atoms: 母体骨架重原子上限

    Returns:
        :class:`GroupBuildResult`
    """
    import networkx as nx
    from utils import (canon_str_to_graph, formula_matches_input,
                       graph_formula_parts_full)

    res = GroupBuildResult()
    counts = {k: int(v) for k, v in (counts or {}).items() if int(v) > 0}
    if not counts:
        res.error = "未选择任何基团"
        return res

    parent, err = derive_parent(target, counts)
    if err:
        res.error = err
        return res
    p_c, p_h = parent
    res.parent_formula = _fmt_fml(p_c, p_h, 0, 0, (0, 0, 0, 0))
    if p_c + p_h == 0 or p_c == 0 and p_h <= 2:
        res.error = f"母体骨架过小（{res.parent_formula}）"
        return res
    if p_c > max_skel_atoms:
        res.error = (f"母体骨架 C={p_c} 超过构建模式上限 {max_skel_atoms}，"
                     f"请减少基团数量")

    # ① 母体骨架枚举
    try:
        skel_items = skeleton_provider(p_c, p_h)
    except Exception as exc:
        res.error = f"母体骨架枚举失败：{type(exc).__name__}: {exc}"
        return res
    res.n_skeletons = len(skel_items)

    # 展平基团序列（按 key 排序 → 枚举顺序稳定）
    flat = sorted(k for k, m in counts.items() for _ in range(m))
    total_groups = len(flat)

    # ② 逐骨架做挂载位置枚举
    seen: Dict[str, list] = {}
    out: List = []
    from special_groups import GROUP_LIBRARY

    def _canon_key(G):
        """WL 哈希 + 形式电荷编码（同哈希才做精确同构）"""
        H = nx.Graph()
        for n, d in G.nodes(data=True):
            H.add_node(n, k=f"{d.get('label', 'C')}{int(d.get('formal_charge', 0) or 0)}")
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, bond_type=d.get('bond_type', 'single'))
        return nx.weisfeiler_lehman_graph_hash(H, node_attr='k',
                                               edge_attr='bond_type')

    def _iso(a, b) -> bool:
        nm = nx.algorithms.isomorphism.categorical_node_match(
            ('label', 'formal_charge'), ('C', 0))
        em = nx.algorithms.isomorphism.categorical_edge_match('bond_type', 'single')
        return nx.is_isomorphic(a, b, node_match=nm, edge_match=em)

    for mol_type, item in skel_items:
        if isinstance(item, str):
            G0 = canon_str_to_graph(item, mol_type, None)
            if G0 is None:
                continue
        else:
            G0 = item
        if G0.number_of_nodes() != p_c:
            continue
        nodes = sorted(G0.nodes())

        def rec(ai: int, remaining: List[Tuple[str, int]], Gcur):
            """按原子顺序分配基团（每个分配方案唯一）"""
            if ai == len(nodes):
                if any(m for _, m in remaining):
                    return
                res.n_placements += 1
                if not formula_matches_input(Gcur, target[0], target[1],
                                             target[3], target[4], 0,
                                             n_nitrogen=target[2]):
                    res.n_gate_rejected += 1
                    return
                k = _canon_key(Gcur)
                bucket = seen.get(k)
                if bucket is not None:
                    for g_old in bucket:
                        if _iso(Gcur, g_old):
                            return          # 真重复
                seen.setdefault(k, []).append(Gcur)
                out.append(Gcur)
                return

            node = nodes[ai]
            cap = _avail_h(Gcur, node)
            if cap <= 0:
                rec(ai + 1, remaining, Gcur)
                return
            for alloc in _distributions(remaining, cap):
                G2 = Gcur
                for key, m in alloc:
                    for _ in range(m):
                        from special_groups import attach_group
                        G2 = attach_group(G2, node, key)
                left = [(k, m - dict(alloc).get(k, 0)) for k, m in remaining]
                rec(ai + 1, left, G2)
                if len(out) >= max_graphs:
                    return

        try:
            remain0 = [(k, flat.count(k)) for k in sorted(set(flat))]
            rec(0, remain0, G0)
        except RecursionError:
            res.warnings.append("挂载枚举递归过深，已跳过该骨架")
        if len(out) >= max_graphs:
            res.warnings.append(f"结果数达到上限 {max_graphs}，已截断")
            break

    res.graphs = out
    if not out and not res.error:
        res.error = ("构建模式未产出结构：请检查所选基团是否与目标分子式匹配"
                     "（目标：%s，母体：%s）"
                     % (_fmt_fml(target[0], target[1], target[2], target[3],
                                 target[4]), res.parent_formula))
    return res


def describe_mode() -> str:
    """模式说明（GUI 展示用）"""
    return ("基团构建模式（非完备）：只生成「碳骨架 + 所选基团」的结构，"
            "不含杂原子位于环内/链内的结构（如吡啶、环氧、胺基在链中段）。")
