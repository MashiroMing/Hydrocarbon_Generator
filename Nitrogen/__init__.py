"""
Nitrogen — 含氮有机物异构体生成模块（第一阶段：纯 C/H/N）

模块地图：
  formula.py           N 系分子式/图公式工具（单一事实源，氮规则 DBE）
  atomic_matrix_n.py   完备性预言机（邻接矩阵原子级枚举，C=4/N=3 价）
  generator.py         树机制生成器（C→N 替换 / 端基 N / 桥 N）
  validate.py          验证入口（计数锚点 + 预言机对拍 + MAYGEN 一次性对照）
"""

from .formula import (parse_formula, graph_formula_parts,
                      format_graph_formula, formula_matches_input,
                      dbe_from_parts, dbe_from_graph, expected_h_for_dbe)
from .atomic_matrix_n import AtomicMatrixNGenerator
from .generator import NitrogenGenerator

__all__ = [
    'parse_formula', 'graph_formula_parts', 'format_graph_formula',
    'formula_matches_input', 'dbe_from_parts', 'dbe_from_graph',
    'expected_h_for_dbe', 'AtomicMatrixNGenerator', 'NitrogenGenerator',
]
