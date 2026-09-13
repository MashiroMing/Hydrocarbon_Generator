"""
高级结构筛选 — 约束定义与片段兼容性校验

用法:
    from structure_filter import StructureConstraint, check_fragment_compatibility

    c = StructureConstraint()
    c.required_fragment = 'benzene'
    c.main_chain = (5, 7)

    compat = check_fragment_compatibility(c, n_carbon=6, n_hydrogen=6, n_oxygen=0, k=4)
    # → [(True, None, frag_info), ...]
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 约束数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StructureConstraint:
    """高级结构筛选约束"""
    required_fragment: Optional[str] = None   # 必须含有的基团名 ('benzene', ...)
    main_chain: Optional[Tuple[int, int]] = None  # 主链长度范围 (min, max)
    ring_size: Optional[Tuple[int, int]] = None   # 环大小范围 (min, max)
    persist: bool = False  # 跨分子式复用

    def is_active(self) -> bool:
        """是否有任何约束生效"""
        return (self.required_fragment is not None
                or self.main_chain is not None
                or self.ring_size is not None)

    def summary(self) -> str:
        """单行摘要供状态栏显示"""
        parts = []
        if self.required_fragment:
            parts.append(self.required_fragment)
        if self.main_chain:
            parts.append(f"主链 C{self.main_chain[0]}~C{self.main_chain[1]}")
        if self.ring_size:
            parts.append(f"环 {self.ring_size[0]}~{self.ring_size[1]} 元")
        return " | ".join(parts) if parts else "无约束"

    def to_dict(self) -> dict:
        return {
            'required_fragment': self.required_fragment,
            'main_chain': self.main_chain,
            'ring_size': self.ring_size,
            'persist': self.persist,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'StructureConstraint':
        return cls(
            required_fragment=d.get('required_fragment'),
            main_chain=tuple(d['main_chain']) if d.get('main_chain') else None,
            ring_size=tuple(d['ring_size']) if d.get('ring_size') else None,
            persist=d.get('persist', False),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 片段兼容性校验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_fragment_compatibility(
    fragment_name: str,
    n_carbon: int,
    n_hydrogen: int,
    n_oxygen: int,
    k: int
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """检查给定片段是否兼容当前分子式

    Args:
        fragment_name: 片段名
        n_carbon: 用户输入的碳原子总数
        n_hydrogen: 用户输入的氢原子总数
        n_oxygen: 用户输入的氧原子总数
        k: 用户输入的不饱和度

    Returns:
        (compatible, reason_if_not, fragment_info_dict)
        fragment_info_dict 来自 fragment_library.get_fragment_info()
    """
    try:
        from oxygen.fragment_library import get_fragment_info
    except ImportError:
        return False, "无法加载片段库", None

    try:
        info = get_fragment_info(fragment_name)
    except KeyError:
        return False, f"未知片段 '{fragment_name}'", None

    fc = info['n_carbon']
    fo = info['n_oxygen']
    fk = info['k']

    # C 数检查
    remaining_c = n_carbon - fc
    if remaining_c < 0:
        return False, f"碳数不足（需 ≥{fc}，当前 {n_carbon})", info

    # O 数检查
    remaining_o = n_oxygen - fo
    if remaining_o < 0:
        return False, f"氧数不足（需 ≥{fo}，当前 {n_oxygen})", info

    # 不饱和度检查：片段 k 不得超过总 k（片段是子图）
    if fk > k:
        return False, f"不饱和度不足（需 ≥{fk}，当前 {k})", info

    # 注：不再单独检查 H 数。
    # 当分子式要求"含某片段"时，只要 C/O/k 兼容即可。
    # H 数差异由取代基自然调节（如苯环 C6H5-Cl → 苯环贡献 5H + 1H 给 Cl）。
    return True, None, info


def validate_constraints(
    constraint: StructureConstraint,
    n_carbon: int,
    n_hydrogen: int,
    n_oxygen: int,
    k: int
) -> Tuple[bool, Optional[str]]:
    """前置校验：约束是否与分子式自洽

    Returns:
        (valid, error_message_if_not)
    """
    if not constraint.is_active():
        return True, None

    # 片段校验
    if constraint.required_fragment:
        ok, reason, info = check_fragment_compatibility(
            constraint.required_fragment, n_carbon, n_hydrogen, n_oxygen, k
        )
        if not ok:
            return False, f"所需基团 '{constraint.required_fragment}' 不兼容: {reason}"

    # 主链长度校验
    if constraint.main_chain:
        cmin, cmax = constraint.main_chain
        if cmin > n_carbon:
            return False, f"主链最小长度 ({cmin}) 不能超过总碳数 ({n_carbon})"
        if cmin < 1:
            return False, "主链最小长度至少为 1"

    # 环大小校验
    if constraint.ring_size:
        rmin, rmax = constraint.ring_size
        if rmin > n_carbon:
            return False, f"最小环尺寸 ({rmin}) 不能超过总碳数 ({n_carbon})"
        if rmin < 3:
            return False, "最小环尺寸至少为 3"

    return True, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 统一后置过滤层
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 环状分子类型集合：其特征值按"环大小"解释（否则按"主链长度"）
_RING_MOL_TYPES = {
    'cycloalkane', 'cycloalkene', 'cyclopolyene',
    'multcycloalkane', 'multcyclomultalkane',
}

# 链状分子类型集合：主链长度约束只作用于这些类型
_CHAIN_MOL_TYPES = {
    'alkane', 'alkene', 'alkyne', 'diene', 'alkenyl',
    'triene', 'tetraene', 'polyene',
}


def _is_ring_type(mol_type: str) -> bool:
    """判断某分子类型是否为环状骨架"""
    return mol_type in _RING_MOL_TYPES


def filter_isomers_by_constraint(
    all_isomers,
    constraint: Optional[StructureConstraint],
    feature_fn=None,
    to_graph_fn=None,
) -> list:
    """统一后置过滤层 —— 对已生成异构体应用全部约束。

    该函数是纯烃 / 含氧 / 卤代三条生成路径"生成完成后"的统一过滤入口，
    确保 required_fragment / main_chain / ring_size 三种约束在所有路径下一致生效。

    Args:
        all_isomers: 已生成的异构体列表，元素为 (mol_type, data)；
                     data 可为 networkx.Graph 或规范字符串表示。
        constraint:  当前约束；为 None 或未启用时原样返回。
        feature_fn:  可选回调 feature_fn(mol_type, data) -> int，
                     计算主链长度/环大小的特征值。
                     仅当约束含 main_chain 或 ring_size 时需要。
        to_graph_fn: 可选回调 to_graph_fn(data) -> networkx.Graph | None，
                     将规范字符串转换为图。仅当约束含 required_fragment 且
                     data 为字符串时需要（图直接跳过转换）。

    Returns:
        过滤后的 [(mol_type, data), ...]，顺序与原列表一致。
    """
    if not constraint or not constraint.is_active():
        return list(all_isomers)

    need_fragment = constraint.required_fragment is not None
    need_feature = (constraint.main_chain is not None
                    or constraint.ring_size is not None)

    # 仅当确实需要时才延迟加载片段库
    if need_fragment:
        from oxygen.fragment_library import graph_contains_fragment

    result = []
    for mol_type, data in all_isomers:
        # ── 必含基团约束 ──
        if need_fragment:
            G = data
            if not hasattr(G, 'nodes'):
                # 规范字符串 → 图
                if to_graph_fn is None:
                    continue  # 无法转换，保守排除
                G = to_graph_fn(data)
                if G is None:
                    continue
            if not graph_contains_fragment(G, constraint.required_fragment):
                continue

        # ── 主链长度 / 环大小约束 ──
        if need_feature:
            if feature_fn is None:
                continue  # 无特征值计算能力，保守排除
            feat = feature_fn(mol_type, data)
            # main_chain 仅作用于链状骨架，ring_size 仅作用于环状骨架
            if constraint.main_chain:
                lo, hi = constraint.main_chain
                if _is_ring_type(mol_type):
                    # 环状分子不适用主链长度约束，视为通过（交由 ring_size 把关）
                    pass
                elif not (lo <= feat <= hi):
                    continue
            if constraint.ring_size:
                lo, hi = constraint.ring_size
                if _is_ring_type(mol_type):
                    if not (lo <= feat <= hi):
                        continue
                else:
                    # 链状分子不适用环大小约束，视为通过
                    pass

        result.append((mol_type, data))

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分子式一致性门禁（统一兜底过滤）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def filter_by_expected_formula(
    all_isomers,
    n_carbon: int,
    n_hydrogen: int,
    n_oxygen: int = 0,
    halogen_spec=(0, 0, 0, 0),
    n_deuterium: int = 0,
    to_graph_fn=None,
) -> list:
    """统一分子式门禁：仅保留真实分子式 == 用户输入分子式的异构体。

    该函数是纯烃 / 含氧 / 卤代三条生成路径的统一兜底过滤，
    防止任何生成路径产出分子式不符的结构（如 =O 使 k+1、原子矩阵
    补漏枚举全部不饱和度、片段合并改变 k 等）。

    Args:
        all_isomers: [(mol_type, data), ...]；data 为 nx.Graph 或规范字符串。
        n_carbon/n_hydrogen/n_oxygen/halogen_spec/n_deuterium:
            用户输入解析结果（utils.parse_compound_formula 口径）。
        to_graph_fn: 可选回调 to_graph_fn(mol_type, data) -> nx.Graph | None，
                     用于将规范字符串转图（图条目跳过转换）。

    Returns:
        过滤后的 [(mol_type, data), ...]，顺序与原列表一致。
    """
    from utils import formula_matches_input

    result = []
    for mol_type, data in all_isomers:
        G = data
        if not hasattr(G, 'nodes'):
            if to_graph_fn is None:
                continue  # 无法校验，保守剔除
            G = to_graph_fn(mol_type, data)
            if G is None:
                continue
        if formula_matches_input(G, n_carbon, n_hydrogen, n_oxygen,
                                 halogen_spec, n_deuterium):
            result.append((mol_type, data))
    return result
