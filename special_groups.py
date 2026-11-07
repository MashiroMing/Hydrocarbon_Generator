# -*- coding: utf-8 -*-
"""special_groups.py — 特殊小分子基团库（超出中性八隅律口径，**按需显式调用**）

## 为什么需要这个模块

本项目枚举口径是 **中性八隅律：C = 4 / N = 3 / O = 2**（与 MAYGEN 默认价态表对齐）。
少数常见基团需要**电荷分离式**才能表达，最典型的是 **硝基 –NO₂**：

- 中性 Lewis 式要 N 与两个 O 都成双键 → N 价态 5（超八隅律）；
- 标准写法是**电荷分离式** `R–N⁺(–O⁻)=O`（一个 =O、一个 –O⁻，N 带 +1）；
- 若把 N=5 放进通用枚举，会**改变全部 N 系基线计数**，且与 MAYGEN 默认价态不再对齐
  （实测 CH3NO2 两侧同为 15 个结构，均不含硝基）——因此**不进枚举**。

但"在已知骨架上挂一个硝基，然后可视化 / 导出 `.gjf`"是合理需求。本模块提供
**数据化的基团库 + 显式挂载接口**：

```python
from special_groups import attach_group, describe, list_groups
G2 = attach_group(G, node=0, group='nitro')     # 苯环 → 硝基苯
print(describe(G2))                             # C6H5NO2，含 硝基×1
```

挂载结果是一个**新的显式原子图**，其特征：
- 基团原子都是显式节点（`label` 为 N/O…），键级与**形式电荷**（`formal_charge`）
  按电荷分离式写入；
- 基团原子带 `special_group=<key>` 标记（可追溯、可裁剪）；
- 分子式统计、2D/3D 渲染、`.gjf` 导出全部可直接复用现有链路
  （`formal_charge` 会在渲染时写入 RDKit，并在导出时计入电荷行）。

## 口径声明（务必遵守）

1. **不进枚举**：预言机 / 树机制不会产出此类结构；挂载图只走"渲染 / 导出"通道，
   不要回灌通用生成或完备性判定，否则会破坏"枚举空间 = 中性八隅律"的定义域；
2. **形式电荷参与合法性判定**：阳离子允许的键数 = 元素基准价 + 形式正电荷
   （硝基 N⁺ 允许 4 键），这是标准价键规则，已在 `utils` / `Nitrogen.formula` 落实；
3. **净电荷**：本库的基团均设计为**净电中性**（如硝基 N⁺/O⁻ 相抵）。若将来加入
   带电基团，`.gjf` 的电荷行会自动跟随图的总电荷（见 `gaussian_export.build_gjf`），
   但多重度需人工确认。

## 扩展方式

往 :data:`_GROUPS` 追加一条 :class:`SpecialGroup` 即可，无需改动其它代码
（:func:`attach_group` 完全由数据驱动）。
"""

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    'SpecialGroup', 'GROUP_LIBRARY', 'attach_group', 'describe',
    'groups_for_host', 'list_groups', 'net_charge',
]


@dataclass(frozen=True)
class SpecialGroup:
    """特殊基团定义（纯数据；`attach_group` 由它驱动）

    原子索引约定：**0 = 中心原子**（与宿主成单键），1..n = ``extra_atoms``。
    组内连接由 ``group_bonds`` 给出（**任意拓扑**，不限于星形）：
    ``(i, j, 键级)``，i/j 为上述原子索引。
    """

    key: str                                  # API 键名，如 'nitro'
    name: str                                 # 中文名，如 '硝基'
    formula: str                              # 基团分子式（不含宿主被取代的 H），如 'NO2'
    central: str                              # 中心原子元素
    extra_atoms: Tuple[Tuple[str, int], ...]  # 额外原子 (label, 显式 H 数)
    group_bonds: Tuple[Tuple[int, int, str], ...]   # 组内键 (i, j, 键级)
    charges: Tuple[int, ...] = ()             # 各原子形式电荷（下标同原子序；空=全 0）
    note: str = ''                            # 口径/化学说明

    @property
    def n_atoms(self) -> int:
        return 1 + len(self.extra_atoms)

    def charge_of(self, i: int) -> int:
        return self.charges[i] if i < len(self.charges) else 0


# ── 基团库（加一条即可扩展）──────────────────────────────────────────
# 全部为**一价取代基**（经 1 个单键挂在宿主上，消耗宿主 1 个 H）。
_GROUPS: Tuple[SpecialGroup, ...] = (
    SpecialGroup(
        key='nitro',
        name='硝基',
        formula='NO2',
        central='N',
        extra_atoms=(('O', 0), ('O', 0)),
        group_bonds=((0, 1, 'double'), (0, 2, 'single')),
        charges=(+1, 0, -1),
        note='–NO₂ 电荷分离式 R–N⁺(–O⁻)=O（净电中性）。中性 Lewis 式需 N=5，'
             '超中性八隅律口径 → 不参与通用枚举，仅按需挂载。',
    ),
    SpecialGroup(
        key='nitroso',
        name='亚硝基',
        formula='NO',
        central='N',
        extra_atoms=(('O', 0),),
        group_bonds=((0, 1, 'double'),),
        charges=(0, 0),
        note='–N=O：N 价态 3，**本就落在中性八隅律口径内**（预言机可枚举）；'
             '此处提供仅为"按需挂载"的对称入口。',
    ),
    SpecialGroup(
        key='azide',
        name='叠氮基',
        formula='N3',
        central='N',
        extra_atoms=(('N', 0), ('N', 0)),
        group_bonds=((0, 1, 'single'), (1, 2, 'triple')),
        charges=(-1, +1, 0),
        note='–N⁻–N⁺≡N 电荷分离式（净电中性）；链状拓扑，端基 N≡N 与中心 N⁻。',
    ),
    SpecialGroup(
        key='amino',
        name='氨基',
        formula='NH2',
        central='N',
        extra_atoms=(),
        group_bonds=(),
        charges=(0,),
        note='–NH₂：N 价态 3（1 键 + 2 H），落在中性八隅律内。',
    ),
    SpecialGroup(
        key='hydroxyl',
        name='羟基',
        formula='OH',
        central='O',
        extra_atoms=(),
        group_bonds=(),
        charges=(0,),
        note='–OH：O 价态 2（1 键 + 1 H）。',
    ),
    SpecialGroup(
        key='cyano',
        name='氰基',
        formula='CN',
        central='C',
        extra_atoms=(('N', 0),),
        group_bonds=((0, 1, 'triple'),),
        charges=(0, 0),
        note='–C≡N：中心 C 为 1 键(宿主)+3 键(N)=4 价，N 为腈氮。',
    ),
    SpecialGroup(
        key='carboxyl',
        name='羧基',
        formula='COOH',
        central='C',
        extra_atoms=(('O', 0), ('O', 0)),
        group_bonds=((0, 1, 'double'), (0, 2, 'single')),
        charges=(0, 0, 0),
        note='–COOH：中心 C 为 1+2+1=4 价；单键 O 按价态自动带 1 个 H。',
    ),
    SpecialGroup(key='fluoro', name='氟', formula='F', central='F',
                 extra_atoms=(), group_bonds=(), charges=(0,),
                 note='–F：卤素取代（1 价）。'),
    SpecialGroup(key='chloro', name='氯', formula='Cl', central='Cl',
                 extra_atoms=(), group_bonds=(), charges=(0,),
                 note='–Cl：卤素取代（1 价）。'),
    SpecialGroup(key='bromo', name='溴', formula='Br', central='Br',
                 extra_atoms=(), group_bonds=(), charges=(0,),
                 note='–Br：卤素取代（1 价）。'),
    SpecialGroup(key='iodo', name='碘', formula='I', central='I',
                 extra_atoms=(), group_bonds=(), charges=(0,),
                 note='–I：卤素取代（1 价）。'),
)

GROUP_LIBRARY: Dict[str, SpecialGroup] = {g.key: g for g in _GROUPS}

#: 基团挂载时的"H 净变化"缓存（由结构自动推导，见 :func:`group_delta`）
_DELTA_CACHE: Dict[str, Tuple[int, int, int, int, int, int]] = {}


def group_delta(key: str) -> Tuple[int, int, int, int, int, int]:
    """基团挂载到宿主后，**相对宿主**的组成净变化

    返回 ``(dC, dH, dN, dO, dHalogen, dCharge)``（dHalogen 为卤素总数）。

    实现方式**数据驱动**（不手工维护增量表）：把基团挂到一个甲烷宿主上，
    用统一事实源 :func:`utils.graph_formula_parts_full` 比较"宿主 CH4"与
    "宿主+基团"的分子式差 → 得到增量。这样新增基团无需改本函数。
    """
    if key in _DELTA_CACHE:
        return _DELTA_CACHE[key]
    import networkx as nx
    from utils import graph_formula_parts_full

    host = nx.Graph()
    host.add_node(0, label='C')
    base = graph_formula_parts_full(host)                    # CH4
    attached = attach_group(host, 0, key)
    got = graph_formula_parts_full(attached)
    d = (got[0] - base[0],          # C
         got[1] - base[1],          # H
         got[2] - base[2],          # N
         got[3] - base[3],          # O
         sum(got[5]) - sum(base[5]),  # 卤素总数
         net_charge(attached) - net_charge(host))
    _DELTA_CACHE[key] = d
    return d


def list_groups() -> Tuple[SpecialGroup, ...]:
    """列出全部已定义的特殊基团"""
    return _GROUPS


def net_charge(G) -> int:
    """图的总形式电荷（无 `formal_charge` 属性时为 0）"""
    return sum(int(d.get('formal_charge', 0) or 0) for _, d in G.nodes(data=True))


def _host_free_valence(G, node) -> int:
    """宿主原子的可取代氢数（按元素价态 + 形式电荷 + valence_override）"""
    from utils import atom_valence
    d = G.nodes[node]
    max_v = atom_valence(d.get('label', 'C'), d)
    load = sum(3 if G[node][nb].get('bond_type') == 'triple'
               else 2 if G[node][nb].get('bond_type') == 'double' else 1
               for nb in G.neighbors(node))
    return max(0, max_v - load)


def groups_for_host(G, node) -> Tuple[SpecialGroup, ...]:
    """列出"可挂到该宿主原子"的基团（宿主需有 ≥1 个可取代 H）"""
    try:
        avail = _host_free_valence(G, node)
    except Exception:
        return ()
    return tuple(g for g in _GROUPS if avail >= 1)


def attach_group(G, node, group: str = 'nitro'):
    """在宿主原子 ``node`` 上挂一个特殊基团 → **新的显式原子图**（不改原图）

    Args:
        G: 宿主图（骨架+取代基模型或节点模型均可）
        node: 挂载点（宿主原子，需有 ≥1 个可取代 H）
        group: 基团键名（见 :data:`GROUP_LIBRARY`）

    Returns:
        新的 ``nx.Graph``（显式原子；基团原子带 ``special_group`` 与 ``formal_charge``）

    Raises:
        KeyError: 未知基团键名
        ValueError: 宿主原子不存在 / 无可取代 H
    """
    if group not in GROUP_LIBRARY:
        raise KeyError(f"未知基团 {group!r}；可用：{sorted(GROUP_LIBRARY)}")
    gd = GROUP_LIBRARY[group]
    if node not in G.nodes:
        raise ValueError(f"宿主原子 {node!r} 不在图中")
    if _host_free_valence(G, node) < 1:
        raise ValueError(f"宿主原子 {node!r} 没有可取代的 H，无法挂载 {gd.name}")

    # 统一成显式原子图（含氧/卤代骨架 → 展开；节点模型 → 恒等），再复制以免改原图
    from utils import expand_substituent_graph
    H = expand_substituent_graph(G).copy()

    next_id = (max(H.nodes()) + 1) if H.number_of_nodes() else 0
    ids = []
    # 中心原子（索引 0）：与宿主成单键
    cid = next_id
    next_id += 1
    H.add_node(cid, label=gd.central, special_group=gd.key)
    if gd.charge_of(0):
        H.nodes[cid]['formal_charge'] = gd.charge_of(0)
    H.add_edge(node, cid, bond_type='single')
    ids.append(cid)

    # 其余原子（索引 1..n）
    for k, (label, n_h) in enumerate(gd.extra_atoms, start=1):
        H.add_node(next_id, label=label, special_group=gd.key)
        if n_h:
            H.nodes[next_id]['fixed_h'] = n_h
        if gd.charge_of(k):
            H.nodes[next_id]['formal_charge'] = gd.charge_of(k)
        ids.append(next_id)
        next_id += 1

    # 组内键（任意拓扑：如叠氮 N⁻–N⁺≡N 是链状而非星形）
    for i, j, bt in gd.group_bonds:
        H.add_edge(ids[i], ids[j], bond_type=bt)
    return H


def describe(G) -> str:
    """返回"分子式 + 特殊基团统计"的可读字符串（自检/日志用）"""
    from utils import graph_formula_parts_full
    n_c, n_h, n_n, n_o, _n_p, halo = graph_formula_parts_full(G)
    base = f"C{n_c}" if n_c > 1 else "C"
    base += f"H{n_h}"
    if n_n:
        base += f"N{n_n}" if n_n > 1 else "N"
    if n_o:
        base += f"O{n_o}" if n_o > 1 else "O"
    for idx, elem in ((2, 'Br'), (1, 'Cl'), (0, 'F'), (3, 'I')):
        if halo[idx]:
            base += f"{elem}{halo[idx]}" if halo[idx] > 1 else elem
    tags: Dict[str, int] = {}
    for _, d in G.nodes(data=True):
        k = d.get('special_group')
        if k:
            tags[k] = tags.get(k, 0) + 1
    parts = []
    for k, cnt in tags.items():
        gd = GROUP_LIBRARY.get(k)
        if gd:
            parts.append(f"{gd.name}×{cnt // gd.n_atoms}")
    q = net_charge(G)
    if parts:
        base += '，含 ' + '、'.join(parts)
    if q:
        base += f'（净电荷 {q:+d}）'
    return base


# ── 自检 ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import networkx as nx
    from utils import format_graph_formula, formula_matches_input

    print("特殊基团库：")
    for g in list_groups():
        print(f"  {g.key:<8} {g.name:<5} –{g.formula:<4} 中心={g.central} "
              f"组内键={g.group_bonds} 电荷={g.charges or (0,) * g.n_atoms}")

    B = nx.Graph()
    for i in range(6):
        B.add_node(i, label='C')
    for i in range(6):
        B.add_edge(i, (i + 1) % 6,
                   bond_type='double' if i % 2 == 0 else 'single')
    print(f"\n苯: {describe(B)}")
    for key in ('nitro', 'nitroso', 'azide'):
        NB = attach_group(B, 0, key)
        print(f"苯 + {key}: {describe(NB)}  格式式={format_graph_formula(NB)}  "
              f"净电荷={net_charge(NB)}")
    NB = attach_group(B, 0, 'nitro')
    print(f"硝基苯门禁 C6H5NO2 → "
          f"{formula_matches_input(NB, 6, 5, 2, n_nitrogen=1)}")
