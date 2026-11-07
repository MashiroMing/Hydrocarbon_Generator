"""Phosphorus — 含磷（P）异构体生成模块（第一期：仅 C/H/P，CLI 优先）

设计原则（与项目既有模块保持一致）：
- **低耦合**：本包不 import `Nitrogen` / `oxygen` / `utils`，自成一体；
  分子式、价态、去重内核全部自带（便于后续独立演进与测试）。
- **高内聚**：公式事实源（formula.py）→ 完备枚举（atomic_matrix_p.py）→
  CLI（cli.py）/ 验证（validate.py）四层清晰分离。
- **数据结构**：与含氧/含氮一致 —— `networkx.Graph`，节点 `label ∈ {'C','P'}`，
  边 `bond_type ∈ {1,2,3}`。

口径（已与用户确认，见 Instruction Data/Phosphorus_Development_Plan.md §8）：
- 一期 P 价态 = **3**（与 MAYGEN 1.8 默认完全一致，可直接对拍）；
  `valence_p=5` 作为**预留开关**，默认关闭、不做锚点保证；
- 允许 P–C / P–H / P–P 单键，**允许 P=C、P≡C、P≡P** 双/三键（math 口径）；
- 不建模形式电荷 / 配位键（二期接 O 时再引入）。
"""

__version__ = '1.0.0-phase1'

from .formula import (  # noqa: F401
    P_VALENCE_OPTIONS,
    VALENCE,
    dbe,
    format_graph_formula_p,
    formula_matches_p,
    formula_string,
    graph_formula_parts_p,
    parse_formula_p,
)
from .atomic_matrix_p import PhosphorusMatrixGenerator  # noqa: F401

__all__ = [
    'PhosphorusMatrixGenerator',
    'parse_formula_p',
    'formula_matches_p',
    'graph_formula_parts_p',
    'format_graph_formula_p',
    'formula_string',
    'dbe',
    'VALENCE',
    'P_VALENCE_OPTIONS',
    '__version__',
]
