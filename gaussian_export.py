# -*- coding: utf-8 -*-
"""Gaussian 输入卡（.gjf）生成 —— 单一事实源

本模块是项目内**唯一**生成 .gjf 文本的位置，`Main.py` 与
`original_programs/alkenyl_generator.py` 的导出路径全部收口到这里。

## 为什么需要它

旧实现直接拼接文本，存在四个会导致 Gaussian 拒绝或无提示错算的缺陷：

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | 第一行写作 `# <分子式> <类型> Isomer #N` | 该行占据 **route section** 位置，Gaussian 把 `C6H6`/`alkene` 当关键词解析 → `GAUSSIAN_PARSE_INPUT_FAILURE`。GaussView 容错（只读坐标），所以此前"验证过"是假象 |
| 2 | 无 Link 0（`%Chk`/`%Mem`/`%NProcShared`） | 无法 restart、无法 `formchk`/`cubegen`；内存与核数只能用默认值 |
| 3 | 电荷/自旋多重度硬编码 `0 1` | 开壳层/离子体系必错；且不可配置 |
| 4 | 键连接块写作 `M N Order 0.0 0.0 0.0`，且 route 未声明 `Geom=Connectivity` | 语法既非标准，又因未声明而被 Gaussian 直接忽略 —— 写了等于没写 |

## 规范的块序（依据 Gaussian 官方 Geom 文档与本 skill 的输入解剖）

```
%Chk=<name>.chk            ← Link 0（可多行）
%Mem=4GB
%NProcShared=4
#p B3LYP/6-31G(d) Opt Geom=Connectivity   ← route section，必须以 # 开头
                            ← 空行
C6H6 alkene Isomer #1       ← title，纯文本，**不得以 # 开头**
                            ← 空行
0 1                         ← 电荷 自旋多重度
C   0.000000 ...            ← 坐标（重原子在前，氢在后）
H   ...
                            ← 空行（坐标段终止）
1 2 1.0 3 1.0               ← 连通性块：每个原子一行，M N1 Order1 N2 Order2 ...
2 1 1.0
                            ← 空行（连通性段终止）
```

连通性块的行格式为 `M N1 Order1 [N2 Order2 ...]`（官方原文：
*"requires one line of input per atom, ordered the same as in the molecule
specification ... syntax: M N1 Order1 [N2 Order2 …]"*），**每个原子恰好一行**，
顺序必须与坐标段一致。

## 可靠性守卫

连通性的正确性依赖"H 坐标 → 母重原子"的对应关系。旧代码用"按碳序、每碳
`4 - 键负载` 个氢"的隐式假设（RDKit 的 H 追加顺序未经契约保证）。本模块改为
**按空间最近重原子判定**，并施加两道守卫；任一不通过就**整体降级**为
"不输出连通性块 + route 不带 `Geom=Connectivity`"，让 Gaussian 自行按距离
感知连接性 —— 宁可不给，也不给错的。

1. 每个 H 与其最近重原子的距离必须 ≤ `MAX_CH_DISTANCE`（默认 1.4 Å，C-H 实测 1.09 Å）；
2. 按最近重原子统计的每重原子氢数，必须与图中价态推算的氢数完全一致，
   且图中每个重原子都必须在坐标字典中存在。

## 与其他模块的关系

- 本模块**不依赖** RDKit / networkx（只按鸭子类型使用传入的图对象），
  因此可在无化学库的打包环境中工作；
- 3D 坐标的来源（ETKDG + MMFF、多种子择优、键长缩放等）仍由 `Main.py`
  负责，本模块只负责**把坐标写成合法输入卡**。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ELEMENT_VALENCE",
    "MAX_CH_DISTANCE",
    "GjfSettings",
    "GjfResult",
    "DEFAULT_ROUTE_KEYWORDS",
    "build_gjf",
    "hydrogen_counts",
    "sanitize_name",
    "normalize_route_keywords",
    "compose_route_line",
    "validate_gjf",
    "settings_path",
    "load_settings",
    "save_settings",
]

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 元素默认价态（中性八隅律口径，与 Nitrogen/ 模块保持一致：C=4 / N=3 / O=2）
ELEMENT_VALENCE: Dict[str, int] = {
    "C": 4, "N": 3, "O": 2, "H": 1, "D": 1,
    "F": 1, "Cl": 1, "Br": 1, "I": 1,
}

#: 键型 → 键级
BOND_ORDER: Dict[str, float] = {"single": 1.0, "double": 2.0, "triple": 3.0}

#: C-H 键长上限（Å）。超过此值认为坐标与图不一致，放弃输出连通性
MAX_CH_DISTANCE = 1.4

#: 默认 route 关键词（不含前导 #）
DEFAULT_ROUTE_KEYWORDS = "B3LYP/6-31G(d) Opt"
DEFAULT_MEM = "4GB"
DEFAULT_NPROC = 4
DEFAULT_CHARGE = 0
DEFAULT_MULTIPLICITY = 1

#: 合法元素符号（用于 title 与原子行校验）
_ELEMENT_RE = r"(?:H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Br|I|D)"
_COORD_RE = re.compile(rf"^\s*{_ELEMENT_RE}\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*$")
_LINK0_RE = re.compile(r"^\s*%[A-Za-z]+(?:=[^\s]+)?\s*$")
_CHARGE_MULT_RE = re.compile(r"^\s*(-?\d+)\s+(\d+)\s*$")
_CONN_LINE_RE = re.compile(r"^\s*(\d+)((?:\s+\d+\s+\d+(?:\.\d+)?)+)\s*$")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

@dataclass
class GjfSettings:
    """导出配置。可由 GUI / JSON 覆盖，默认值适用于有机小分子的第一轮优化。

    注意：``route_keywords`` **不含前导 ``#`` 与 ``p``**，二者由
    :func:`compose_route_line` 统一补齐；传入 `"#p B3LYP/..."` 也能被
    :func:`normalize_route_keywords` 正确归一化。
    """

    route_keywords: str = DEFAULT_ROUTE_KEYWORDS
    mem: str = DEFAULT_MEM
    nproc: int = DEFAULT_NPROC
    charge: int = DEFAULT_CHARGE
    multiplicity: int = DEFAULT_MULTIPLICITY
    #: 是否写 Link 0（关闭则得到最小输入卡，全部参数走 Gaussian 默认）
    write_link0: bool = True
    #: 是否写 %Chk（写入后可用 formchk / cubegen / restart）
    write_chk: bool = True
    #: 是否尝试输出连通性块（几何不可信时仍会自动降级）
    use_connectivity: bool = True
    #: route 是否使用详细输出（#p 而非 #）
    verbose_route: bool = True

    # -- 校验与归一化 ------------------------------------------------------
    def normalized(self) -> "GjfSettings":
        """返回经合法性校验的副本；越界值回落到默认值而不是抛异常。"""
        kw = normalize_route_keywords(self.route_keywords)
        mem = (self.mem or DEFAULT_MEM).strip() or DEFAULT_MEM
        try:
            nproc = int(self.nproc)
        except (TypeError, ValueError):
            nproc = DEFAULT_NPROC
        nproc = max(1, min(nproc, 512))
        try:
            charge = int(self.charge)
        except (TypeError, ValueError):
            charge = DEFAULT_CHARGE
        try:
            mult = int(self.multiplicity)
        except (TypeError, ValueError):
            mult = DEFAULT_MULTIPLICITY
        mult = max(1, mult)
        return GjfSettings(
            route_keywords=kw or DEFAULT_ROUTE_KEYWORDS,
            mem=mem,
            nproc=nproc,
            charge=charge,
            multiplicity=mult,
            write_link0=bool(self.write_link0),
            write_chk=bool(self.write_chk),
            use_connectivity=bool(self.use_connectivity),
            verbose_route=bool(self.verbose_route),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GjfSettings":
        if not isinstance(data, dict):
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def normalize_route_keywords(text: str) -> str:
    """把用户输入归一化为「不含前导 # 与 p」的关键词串。"""
    s = (text or "").strip()
    s = re.sub(r"^#+\s*[pP]?\s*", "", s)      # 去掉 # / #p / ##p
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compose_route_line(settings: GjfSettings, connectivity: bool) -> str:
    """拼出完整 route section（以 ``#`` 开头）。"""
    s = settings.normalized()
    kw = s.route_keywords
    if connectivity:
        # 幂等：避免用户手写 Geom=Connectivity 时重复
        kw = re.sub(r"Geom\s*=\s*Connectivity", "", kw, flags=re.IGNORECASE)
        kw = re.sub(r"\s+", " ", kw).strip()
        kw = (kw + " Geom=Connectivity").strip()
    return "#" + ("p" if s.verbose_route else "") + (" " + kw if kw else "")


# --------------------------------------------------------------------------
# 配置持久化（失败绝不抛异常，退回默认值）
# --------------------------------------------------------------------------

def settings_path() -> Path:
    """配置文件路径。可用环境变量 ``HYDROCARBON_GJF_CONFIG`` 覆盖。"""
    env = os.environ.get("HYDROCARBON_GJF_CONFIG")
    if env:
        return Path(env)
    return Path.home() / ".hydrocarbon_generator" / "gaussian_export.json"


def load_settings(path: Optional[Path] = None) -> GjfSettings:
    p = Path(path) if path else settings_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return GjfSettings.from_dict(json.load(f)).normalized()
    except Exception:
        return GjfSettings()


def save_settings(settings: GjfSettings, path: Optional[Path] = None) -> bool:
    """写回配置，成功返回 True。任何异常都吞掉（配置不可写不应阻断导出）。"""
    p = Path(path) if path else settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(settings.normalized().to_dict(), f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 文本工具
# --------------------------------------------------------------------------

def sanitize_name(text: str, fallback: str = "mol") -> str:
    """把任意标题变成可用作文件名的 ASCII 安全串。"""
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", (text or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or fallback


def _as_xyz(coord: Any) -> Tuple[float, float, float]:
    """接受 tuple / list / numpy 数组，统一成三元浮点组。"""
    return (float(coord[0]), float(coord[1]), float(coord[2]))


def _element_of(node: Any, G: Any, element_of: Optional[Callable[[Any], str]]) -> str:
    """推断节点元素符号；默认按碳骨架处理。"""
    if element_of is not None:
        try:
            e = element_of(node)
            if e:
                return str(e).capitalize()
        except Exception:
            pass
    if G is not None:
        try:
            data = G.nodes[node]
        except Exception:
            data = {}
        if isinstance(data, dict):
            for key in ("element", "symbol", "atom"):
                e = data.get(key)
                if isinstance(e, str) and e.strip():
                    m = re.match(r"[A-Za-z]{1,2}", e.strip())
                    if m:
                        return m.group(0).capitalize()
    return "C"


def _bond_order(data: Any) -> float:
    if isinstance(data, dict):
        bt = data.get("bond_type", "single")
    else:
        bt = "single"
    return BOND_ORDER.get(str(bt).lower(), 1.0)


def _fmt_order(order: float) -> str:
    """键级输出格式：整数键级写成 ``1.0`` / ``2.0`` / ``3.0``（与官方示例一致）。"""
    if abs(order - round(order)) < 1e-9:
        return f"{int(round(order))}.0"
    return f"{order:g}"


# --------------------------------------------------------------------------
# 氢数推算与连通性
# --------------------------------------------------------------------------

def hydrogen_counts(G: Any, element_of: Optional[Callable[[Any], str]] = None) -> Dict[Any, int]:
    """按价态推算每个**重原子**应带的氢数：``价态 − 成键负载``。

    价态口径：``ELEMENT_VALENCE`` 基准价 + 该节点的 ``formal_charge``
    （**带符号**：阳离子多成键、阴离子少成键；特殊基团如硝基电荷分离式
    ``N⁺`` 允许 4 键、``O⁻`` 只允许 1 键，均由此规则给出）。
    氢节点本身返回 0（显式氢图不重复计数）。
    """
    counts: Dict[Any, int] = {}
    for node in G.nodes():
        elem = _element_of(node, G, element_of)
        if elem == "H":
            counts[node] = 0
            continue
        valence = ELEMENT_VALENCE.get(elem, 4)
        try:
            q = int(G.nodes[node].get("formal_charge", 0) or 0)
        except Exception:
            q = 0
        valence = max(0, valence + q)
        load = 0.0
        for nb in G.neighbors(node):
            load += _bond_order(G[node][nb])
        counts[node] = max(0, int(round(valence - load)))
    return counts


def _assign_hydrogen_parents(
    heavy_keys: Sequence[Any],
    heavy_pts: Sequence[Tuple[float, float, float]],
    h_coords: Sequence[Any],
) -> Tuple[List[Any], List[float]]:
    """为每个 H 判定最近重原子。返回 (母原子列表, 对应距离列表)。"""
    parents: List[Any] = []
    dists: List[float] = []
    for h in h_coords:
        hx, hy, hz = _as_xyz(h)
        best_key, best_d2 = None, float("inf")
        for key, (px, py, pz) in zip(heavy_keys, heavy_pts):
            d2 = (px - hx) ** 2 + (py - hy) ** 2 + (pz - hz) ** 2
            if d2 < best_d2:
                best_d2, best_key = d2, key
        parents.append(best_key)
        dists.append(math.sqrt(best_d2))
    return parents, dists


def _can_build_connectivity(
    G: Any,
    heavy_keys: Sequence[Any],
    h_coords: Sequence[Any],
    h_parents: Sequence[Any],
    h_dists: Sequence[float],
    element_of: Optional[Callable[[Any], str]],
) -> Tuple[bool, List[str]]:
    """连通性可靠性守卫。返回 (是否可用, 原因列表)。"""
    reasons: List[str] = []

    # 守卫 -1：键级必须可信。连通性块会把键级写死给 Gaussian，若图中存在
    # 「未声明 bond_type」的边，我们只能猜成单键 —— 一旦该边实为 C=C/C≡C，
    # 就等于向 Gaussian 提交了错误的键级，比不写连通性更糟，故直接降级。
    if hasattr(G, "edges"):
        try:
            unlabeled = [(u, v) for u, v, d in G.edges(data=True)
                         if not isinstance(d, dict) or "bond_type" not in d]
        except Exception:
            unlabeled = []
        if unlabeled:
            reasons.append(
                f"{len(unlabeled)} 条边缺少 bond_type 属性（键级不可信），"
                f"例如 {unlabeled[:3]}；请由调用方显式标注单键"
            )

    # 守卫 0：图中所有重原子都必须有坐标（否则索引无法对齐）
    graph_heavy = [n for n in G.nodes() if _element_of(n, G, element_of) != "H"]
    missing = [n for n in graph_heavy if n not in set(heavy_keys)]
    if missing:
        reasons.append(f"图中 {len(missing)} 个重原子缺少坐标：{missing[:5]}")

    # 守卫 1：H 与最近重原子的距离必须在合理键长内
    oversize = [i for i, d in enumerate(h_dists) if d > MAX_CH_DISTANCE]
    if oversize:
        worst = max(h_dists) if h_dists else 0.0
        reasons.append(
            f"{len(oversize)} 个 H 距最近重原子 > {MAX_CH_DISTANCE} Å（最远 {worst:.2f} Å），"
            "坐标与图不一致"
        )

    # 守卫 2：按最近重原子归属的氢数，必须与价态推算完全一致
    expected = hydrogen_counts(G, element_of)
    expected = {k: v for k, v in expected.items() if v > 0}
    actual: Dict[Any, int] = {}
    for p in h_parents:
        actual[p] = actual.get(p, 0) + 1
    if expected != actual:
        diff = []
        for key in sorted(set(expected) | set(actual), key=str):
            e, a = expected.get(key, 0), actual.get(key, 0)
            if e != a:
                diff.append(f"{key}: 期望 {e} 实际 {a}")
        reasons.append("氢分配与价态不符 → " + "; ".join(diff[:6]))
        reasons.append(f"氢总数：坐标 {len(h_coords)} vs 价态推算 {sum(expected.values())}")

    return (not reasons), reasons


# --------------------------------------------------------------------------
# 主构建函数
# --------------------------------------------------------------------------

@dataclass
class GjfResult:
    """构建结果。``text`` 为最终写入磁盘的内容。"""

    text: str
    connectivity: bool
    warnings: List[str] = field(default_factory=list)
    route_line: str = ""
    n_heavy: int = 0
    n_hydrogen: int = 0

    def __str__(self) -> str:  # 方便直接当字符串用
        return self.text


def build_gjf(
    title: str,
    heavy_coords: Dict[Any, Any],
    h_coords: Sequence[Any],
    G: Any = None,
    settings: Optional[GjfSettings] = None,
    chk_name: Optional[str] = None,
    element_of: Optional[Callable[[Any], str]] = None,
) -> GjfResult:
    """构建一份可被 Gaussian 直接提交的输入卡。

    Args:
        title: 人类可读标题（写入 title section，**必须不含换行**）。
        heavy_coords: 重原子坐标 ``{节点: (x, y, z)}``，节点序号需与图中编号一致。
        h_coords: 氢坐标序列 ``[(x, y, z), ...]``。
        G: 分子图（含 ``bond_type`` 边属性）；为 None 时不输出连通性。
        settings: 导出配置；None 时使用默认值。
        chk_name: checkpoint 文件名（不含扩展名）；None 时由 title 推导。
        element_of: 可选回调 ``f(node) -> 元素符号``；默认按碳骨架处理。

    Returns:
        :class:`GjfResult`。即使几何不可信也会产出**合法**的输入卡
        （只是不含连通性块），并在 ``warnings`` 中说明。
    """
    cfg = (settings or GjfSettings()).normalized()
    warnings: List[str] = []

    # -- 标题归一化：单行、纯文本（绝不能以 # 开头，否则被当 route 解析） ----
    clean_title = re.sub(r"[\r\n]+", " ", str(title or "")).strip()
    if clean_title.startswith("#"):
        clean_title = clean_title.lstrip("#").strip()
        warnings.append("标题以 # 开头，已剥离（否则 Gaussian 会当成 route 关键词）")
    if not clean_title:
        clean_title = "molecule"

    # -- 重原子排序：与图节点编号一致（连通性索引依赖它） ------------------
    heavy_items = sorted(heavy_coords.items(), key=lambda kv: str(kv[0])) \
        if _non_int_keys(heavy_coords) else sorted(heavy_coords.items(), key=lambda kv: kv[0])
    heavy_keys = [k for k, _ in heavy_items]
    heavy_pts = [_as_xyz(v) for _, v in heavy_items]
    c_index = {k: i + 1 for i, k in enumerate(heavy_keys)}

    # -- 连通性决策 --------------------------------------------------------
    conn_lines: Optional[List[str]] = None
    if cfg.use_connectivity and G is not None and heavy_keys:
        h_parents, h_dists = _assign_hydrogen_parents(heavy_keys, heavy_pts, h_coords)
        ok, reasons = _can_build_connectivity(
            G, heavy_keys, h_coords, h_parents, h_dists, element_of
        )
        if ok:
            conn_lines = _render_connectivity(G, heavy_keys, c_index, h_parents, element_of)
        else:
            warnings.append("连通性块已跳过，route 不带 Geom=Connectivity；原因：" + "；".join(reasons))
    elif cfg.use_connectivity and G is None:
        warnings.append("未提供分子图，跳过连通性块（Gaussian 将按距离感知连接性）")

    # -- route ------------------------------------------------------------
    route_line = compose_route_line(cfg, connectivity=conn_lines is not None)

    # -- 组装 --------------------------------------------------------------
    lines: List[str] = []
    if cfg.write_link0:
        if cfg.write_chk:
            stem = sanitize_name(chk_name or clean_title, fallback="mol")
            lines.append(f"%Chk={stem}.chk")
        lines.append(f"%Mem={cfg.mem}")
        lines.append(f"%NProcShared={cfg.nproc}")
    lines.append(route_line)
    lines.append("")                       # route 段终止
    lines.append(clean_title)              # title 段
    lines.append("")                       # title 段终止
    # -- 电荷：仅当**图中确有形式电荷信息**（特殊基团如硝基 N⁺/O⁻）时才以图为准；
    #    否则尊重用户的配置（历史行为不变：自由基等按配置电荷导出） ----------
    total_q = 0
    has_charge_info = False
    if G is not None:
        try:
            for _n, _d in G.nodes(data=True):
                if 'formal_charge' in _d:
                    has_charge_info = True
                    total_q += int(_d.get('formal_charge', 0) or 0)
        except Exception:
            has_charge_info = False
            total_q = 0
    if has_charge_info and total_q != cfg.charge:
        warnings.append(f"图中形式电荷合计 {total_q:+d}，已覆盖配置电荷 {cfg.charge:+d}")
    charge = total_q if has_charge_info else cfg.charge
    lines.append(f"{charge} {cfg.multiplicity}")

    # 坐标段：每个重原子按其**元素符号**输出（此前硬编码 'C'，含杂原子
    # 导出会把 N/O 写成碳；元素由 element_of 回调 / 图节点属性判定，
    # 碳骨架情形与旧输出逐字节一致）。
    for _k, (x, y, z) in zip(heavy_keys, heavy_pts):
        _elem = _element_of(_k, G, element_of)
        lines.append(f"{_elem:<2s}{x:>14.6f}{y:>14.6f}{z:>14.6f}")
    for h in h_coords:
        hx, hy, hz = _as_xyz(h)
        lines.append(f"{'H':<2s}{hx:>14.6f}{hy:>14.6f}{hz:>14.6f}")

    lines.append("")                       # 坐标段终止
    if conn_lines:
        lines.extend(conn_lines)
        lines.append("")                   # 连通性段终止

    text = "\n".join(lines) + "\n"
    return GjfResult(
        text=text,
        connectivity=conn_lines is not None,
        warnings=warnings,
        route_line=route_line,
        n_heavy=len(heavy_keys),
        n_hydrogen=len(h_coords),
    )


def _non_int_keys(mapping: Dict[Any, Any]) -> bool:
    return any(not isinstance(k, int) for k in mapping)


def _render_connectivity(
    G: Any,
    heavy_keys: Sequence[Any],
    c_index: Dict[Any, int],
    h_parents: Sequence[Any],
    element_of: Optional[Callable[[Any], str]],
) -> List[str]:
    """输出连通性块：每个原子一行，``M N1 Order1 [N2 Order2 ...]``。

    连通性写成**双向对称**形式（碳那行列出处在该碳上的氢，氢那行也列回母碳），
    与 GaussView 导出的 .gjf 一致；单侧声明虽可能被接受，但对称写法没有歧义。
    """
    h_start = len(heavy_keys) + 1
    # 母重原子 → 该原子上的氢的全局编号（保持 h_coords 顺序）
    h_of_heavy: Dict[Any, List[int]] = {}
    for j, parent in enumerate(h_parents):
        h_of_heavy.setdefault(parent, []).append(h_start + j)

    lines: List[str] = []
    for node in heavy_keys:
        parts: List[str] = [str(c_index[node])]
        nbrs = [nb for nb in G.neighbors(node) if nb in c_index]
        try:
            nbrs.sort(key=lambda n: c_index[n])
        except Exception:
            pass
        for nb in nbrs:
            parts.append(str(c_index[nb]))
            parts.append(_fmt_order(_bond_order(G[node][nb])))
        for hidx in h_of_heavy.get(node, ()):
            parts.append(str(hidx))
            parts.append("1.0")
        lines.append(" ".join(parts))

    for j, parent in enumerate(h_parents):
        lines.append(f"{h_start + j} {c_index[parent]} 1.0")
    return lines


# --------------------------------------------------------------------------
# 结构自检（内部预言机：不依赖 Gaussian 也能判定输入卡是否自洽）
# --------------------------------------------------------------------------

def validate_gjf(text: str) -> List[str]:
    """按 block 规范逐段校验输入卡，返回问题列表（空列表 = 通过）。

    对应 hpc-gaussian 的 input 解剖：Link 0 → route(#) → 空行 → title →
    空行 → 电荷/多重度 → 坐标 → 空行 → [连通性] → 空行。
    """
    problems: List[str] = []
    if not isinstance(text, str) or not text.strip():
        return ["输入卡为空"]
    if "\r" in text:
        problems.append("存在 CR 字符（应为纯 LF 换行）")
    if not text.endswith("\n"):
        problems.append("文件未以换行结尾（Gaussian 需要坐标段后的空行终止）")

    lines = text.split("\n")
    i = 0
    n = len(lines)

    # 1. Link 0
    link0_count = 0
    while i < n and lines[i].strip().startswith("%"):
        if not _LINK0_RE.match(lines[i]):
            problems.append(f"第 {i + 1} 行 Link 0 语法可疑：{lines[i]!r}")
        link0_count += 1
        i += 1
    if i < n and lines[i].strip() == "" and link0_count:
        problems.append(f"第 {i + 1} 行：Link 0 与 route 之间不应有空行")

    # 2. route
    if i >= n:
        return problems + ["缺少 route section"]
    route = lines[i]
    if not route.lstrip().startswith("#"):
        problems.append(f"第 {i + 1} 行不是 route section（须以 # 开头）：{route[:60]!r}")
    route_has_conn = bool(re.search(r"Geom\s*=\s*Connectivity", route, re.IGNORECASE))
    # 启发式：真实 route 一定带「方法/基组」（如 B3LYP/6-31G(d)）。旧实现把
    # "# C6H6 alkene Isomer #1" 当 route，正好缺这个斜杠 —— 据此可自动识别该缺陷。
    if route.lstrip().startswith("#") and "/" not in route:
        problems.append(
            f"第 {i + 1} 行 route 段未见「方法/基组」（缺少 '/'），疑似标题误占 route 位：{route[:60]!r}"
        )
    route_idx = i
    i += 1

    # 3. route 后的空行
    if i < n and lines[i].strip() == "":
        i += 1
    else:
        problems.append(f"route 段之后缺少空行（第 {route_idx + 1} 行之后）")

    # 4. title
    if i >= n:
        return problems + ["缺少 title section"]
    title = lines[i]
    if title.strip().startswith("#"):
        problems.append(f"第 {i + 1} 行 title 以 # 开头 → Gaussian 会把它当 route 解析")
    if title.strip() == "":
        problems.append(f"第 {i + 1} 行 title 为空（Gaussian 需要一行标题占位）")
    i += 1

    # 5. title 后的空行
    if i < n and lines[i].strip() == "":
        i += 1
    else:
        problems.append("title 段之后缺少空行")

    # 6. 电荷 / 多重度
    if i >= n:
        return problems + ["缺少电荷/多重度行"]
    m = _CHARGE_MULT_RE.match(lines[i])
    if not m:
        problems.append(f"第 {i + 1} 行不是合法的「电荷 多重度」：{lines[i]!r}")
    else:
        if int(m.group(2)) < 1:
            problems.append(f"第 {i + 1} 行自旋多重度必须 ≥ 1")
    i += 1

    # 7. 坐标段
    atom_idx: List[int] = []
    while i < n and lines[i].strip() != "":
        if not _COORD_RE.match(lines[i]):
            problems.append(f"第 {i + 1} 行坐标格式非法：{lines[i]!r}")
        else:
            atom_idx.append(len(atom_idx) + 1)
        i += 1
    if not atom_idx:
        problems.append("坐标段为空")
    if i >= n:
        problems.append("坐标段未以空行终止")
        return problems

    # 8. 连通性块
    i += 1  # 跳过坐标段终止空行
    conn_seen: List[int] = []
    has_conn_lines = False
    while i < n and lines[i].strip() != "":
        mm = _CONN_LINE_RE.match(lines[i])
        if not mm:
            problems.append(f"第 {i + 1} 行连通性格式非法（应为 M N1 Order1 ...）：{lines[i]!r}")
        else:
            has_conn_lines = True
            conn_seen.append(int(mm.group(1)))
        i += 1

    if has_conn_lines:
        if not route_has_conn:
            problems.append(
                "存在连通性块但 route 未声明 Geom=Connectivity → Gaussian 会忽略该块"
            )
        if conn_seen != list(range(1, len(atom_idx) + 1)):
            problems.append(
                f"连通性块须每原子恰好一行且顺序一致：得到 {len(conn_seen)} 行 / "
                f"{len(atom_idx)} 个原子，编号 {conn_seen[:8]}…"
            )
    elif route_has_conn:
        problems.append("route 声明了 Geom=Connectivity 但缺少连通性块 → 报 End of file reading connectivity")
    if i < n and lines[i].strip() == "" and has_conn_lines:
        pass  # 正常：连通性段以空行终止

    return problems
