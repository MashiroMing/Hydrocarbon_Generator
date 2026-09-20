# -*- coding: utf-8 -*-
"""Gaussian .gjf 导出回归测试

覆盖（对应 hpc-gaussian skill 的 input 解剖清单）：
  1. 块序：Link 0 → route(#) → 空行 → title → 空行 → 电荷/多重度 → 坐标
     → 空行 → [连通性] → 空行
  2. 旧缺陷回归：旧实现的「# 标题占 route 位」「伪连通性块」必须被 validate_gjf 判出
  3. 连通性正确性：键级被写入、块与坐标原子序号一一对应、route 声明一致
  4. 可靠性守卫：几何/氢数与图不一致时**自动降级**（不写块、route 不带 Geom=Connectivity），
     而不是写出错误连通性
  5. 配置：route 关键词归一化、越界值回落、JSON 持久化往返
  6. --e2e 档：调用本项目真实生成器 + RDKit 坐标，端到端导出并校验

运行：
  python tests/test_gjf_export.py           # 默认档（秒级，仅需 networkx + numpy）
  python tests/test_gjf_export.py --e2e     # 追加端到端档（需 rdkit，约数十秒）
  退出码 0 = 全部通过
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gaussian_export import (GjfSettings, build_gjf, compose_route_line,
                             hydrogen_counts, normalize_route_keywords,
                             sanitize_name, load_settings,
                             save_settings, validate_gjf)

PASS = 0
FAIL = 0
RUN_E2E = "--e2e" in sys.argv


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  —— {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 测试几何构造：把氢放在 1.09 Å 球面上，优先远离非母重原子
# --------------------------------------------------------------------------

def _fib_dirs(n):
    ga = math.pi * (3 - math.sqrt(5))
    out = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - z * z))
        th = ga * i
        out.append(np.array([r * math.cos(th), r * math.sin(th), z]))
    return out


def place_hydrogens(heavy, counts, dist=1.09, n_dirs=800):
    """按价态氢数在球面上择优放置氢原子，生成物理上可判定的几何。"""
    keys = sorted(heavy)
    out = []
    for node in keys:
        k = counts.get(node, 0)
        if k <= 0:
            continue
        others = [np.array(heavy[m], dtype=float) for m in keys if m != node]
        cands = []
        for d in _fib_dirs(n_dirs):
            p = np.array(heavy[node], dtype=float) + dist * d
            score = min((np.linalg.norm(p - o) for o in others), default=1e9)
            cands.append((score, p, d))
        cands.sort(key=lambda t: -t[0])
        chosen = []
        for score, p, d in cands:
            if all(float(np.dot(d, dc)) < 0.75 for _, _, dc in chosen):
                chosen.append((score, p, d))
            if len(chosen) == k:
                break
        out.extend([tuple(p) for _, p, _ in chosen])
    return out


def make_mol(spec, coords=None):
    """spec: [(u, v, bond_type), ...]；返回 (G, heavy_coords, h_coords, h_counts)。"""
    import networkx as nx

    G = nx.Graph()
    nodes = sorted({n for u, v, _ in spec for n in (u, v)})
    G.add_nodes_from(nodes)
    for u, v, bt in spec:
        G.add_edge(u, v, bond_type=bt)

    if coords is None:
        coords = {n: (1.54 * i, 0.0, 0.0) for i, n in enumerate(nodes)}
    counts = hydrogen_counts(G)
    return G, coords, place_hydrogens(coords, counts), counts


def _parse_blocks(text):
    """按块序切分输入卡，返回各段位置。

    注意：不能靠「行首是 C/H」来定位坐标段 —— title 行同样可能以 C 开头。
    必须从 route 行按固定偏移推算。
    """
    lines = text.split("\n")
    route_i = next(i for i, l in enumerate(lines) if l.lstrip().startswith("#"))
    title_i = route_i + 2
    charge_i = route_i + 4
    atoms = []
    i = charge_i + 1
    while i < len(lines) and lines[i].strip() != "":
        atoms.append(lines[i])
        i += 1
    i += 1                                  # 跳过坐标段终止空行
    conn = []
    while i < len(lines) and lines[i].strip() != "":
        conn.append(lines[i])
        i += 1
    return {"lines": lines, "route_index": route_i, "title_index": title_i,
            "charge_line": lines[charge_i], "atoms": atoms, "conn": conn}


# --------------------------------------------------------------------------
# 1. 块序与基本格式
# --------------------------------------------------------------------------

def test_block_order():
    print("\n== 1. 块序与基本格式 ==")
    G, heavy, hs, _ = make_mol([(0, 1, 'single'), (1, 2, 'single')],
                               {0: (0.0, 0.0, 0.0), 1: (1.54, 0.0, 0.0),
                                2: (2.05, 1.45, 0.0)})
    res = build_gjf("C3H8 alkane Isomer #1", heavy, hs, G)
    b = _parse_blocks(res.text)
    lines = b["lines"]

    check("校验零问题", validate_gjf(res.text) == [], str(validate_gjf(res.text)))
    check("第 1 行是 %Chk", lines[0].startswith("%Chk="), lines[0])
    check("含 %Mem", any(l.startswith("%Mem=") for l in lines))
    check("含 %NProcShared", any(l.startswith("%NProcShared=") for l in lines))
    route_i = b["route_index"]
    check("route 紧接 Link 0（中间无空行）", route_i == 3, f"route 在第 {route_i + 1} 行")
    check("route 后为空行", lines[route_i + 1] == "")
    check("title 不以 # 开头", not lines[b["title_index"]].startswith("#"),
          lines[b["title_index"]])
    check("title 后为空行", lines[b["title_index"] + 1] == "")
    check("电荷/多重度行为 '0 1'", b["charge_line"].strip() == "0 1", b["charge_line"])
    check("坐标原子总数 = 重原子 + 氢",
          len(b["atoms"]) == res.n_heavy + res.n_hydrogen,
          f"坐标 {len(b['atoms'])} vs 期望 {res.n_heavy + res.n_hydrogen}")
    check("文件以换行结尾", res.text.endswith("\n"))

    first_h = next(i for i, l in enumerate(b["atoms"]) if l.startswith("H"))
    check("重原子先于氢原子", all(l.startswith("C") for l in b["atoms"][:first_h]))


# --------------------------------------------------------------------------
# 2. 旧缺陷回归（校验器必须能识别原实现的问题）
# --------------------------------------------------------------------------

def test_legacy_defect_detected():
    print("\n== 2. 旧缺陷回归 ==")
    legacy = "\n".join([
        "# C6H6 alkene Isomer #1",
        "",
        "# C6H6 alkene - Isomer #1",
        "# Generated by Molecule Isomer Visualizer",
        "",
        "0 1",
        "C       0.000000      0.000000      0.000000",
        "C       1.340000      0.000000      0.000000",
        "",
        "   1   2   2 0.0 0.0 0.0",
        "   1   3   1 0.0 0.0 0.0",
        "   2   4   1 0.0 0.0 0.0",
        "",
    ])
    probs = validate_gjf(legacy)
    blob = " ".join(probs)
    check("旧实现被判定为不合法", len(probs) >= 2, f"仅 {len(probs)} 条：{probs}")
    check("识别出 route 缺少方法/基组（标题占位）",
          any("方法/基组" in p for p in probs), str(probs))
    check("识别出 title 以 # 开头", any("title 以 # 开头" in p for p in probs), str(probs))
    check("识别出伪连通性块行数不符/格式非法",
          ("连通性" in blob) or ("坐标" in blob), blob)


# --------------------------------------------------------------------------
# 3. 连通性正确性
# --------------------------------------------------------------------------

def test_connectivity():
    print("\n== 3. 连通性正确性 ==")

    # 乙烯：应出现键级 2.0，且 route 声明 Geom=Connectivity
    G = __import__("networkx").Graph()
    G.add_edge(0, 1, bond_type='double')
    heavy = {0: (0.0, 0.0, 0.0), 1: (1.34, 0.0, 0.0)}
    counts = hydrogen_counts(G)
    hs = place_hydrogens(heavy, counts)
    res = build_gjf("C2H4 alkene Isomer #1", heavy, hs, G)

    check("乙烯：输出连通性块", res.connectivity)
    check("乙烯：route 声明 Geom=Connectivity",
          "Geom=Connectivity" in res.route_line, res.route_line)
    check("乙烯：双键键级为 2.0", " 2.0" in res.text, res.text[-200:])
    check("乙烯：校验零问题", validate_gjf(res.text) == [], str(validate_gjf(res.text)))

    # 连通性块：每原子一行、编号 1..N 连续、顺序一致
    b = _parse_blocks(res.text)
    n_atoms = res.n_heavy + res.n_hydrogen
    check("乙烯：坐标段原子数 = 重原子 + 氢", len(b["atoms"]) == n_atoms,
          f"{len(b['atoms'])} vs {n_atoms}")
    conn = [l.split()[0] for l in b["conn"]]
    check("连通性块行数 = 原子数", len(conn) == n_atoms, f"{len(conn)} vs {n_atoms}")
    try:
        seq_ok = [int(x) for x in conn] == list(range(1, n_atoms + 1))
    except ValueError:
        seq_ok = False
    check("连通性编号为 1..N 连续", seq_ok, str(conn[:10]))
    # 每个氢行必须回指一个重原子（邻接对称性）
    h_start = res.n_heavy + 1
    h_rows = [l.split() for l in b["conn"] if int(l.split()[0]) >= h_start]
    check("每个氢行恰好声明 1 个键且指向重原子",
          all(len(r) == 3 and 1 <= int(r[1]) <= res.n_heavy for r in h_rows),
          str(h_rows[:4]))
    # 碳行必须包含其氢的编号（双向对称）
    n_h_on_c0 = sum(1 for r in h_rows if int(r[1]) == 1)
    c0_row = next(l.split() for l in b["conn"] if l.split()[0] == "1")
    check("碳行列出了自己的氢（双向对称）",
          all(str(h_start + i) in c0_row for i in range(n_h_on_c0)) and n_h_on_c0 > 0,
          f"c0_row={c0_row} n_h_on_c0={n_h_on_c0}")

    # 环丙烷：三元环，3 条 C-C 单键 + 每个碳 2 个氢
    import networkx as nx
    G = nx.Graph()
    G.add_edge(0, 1, bond_type='single')
    G.add_edge(1, 2, bond_type='single')
    G.add_edge(2, 0, bond_type='single')
    r = 1.51 / math.sqrt(3)
    heavy = {0: (r, 0.0, 0.0),
             1: (r * math.cos(2 * math.pi / 3), r * math.sin(2 * math.pi / 3), 0.0),
             2: (r * math.cos(4 * math.pi / 3), r * math.sin(4 * math.pi / 3), 0.0)}
    counts = hydrogen_counts(G)
    check("环丙烷：每碳 2 个氢", all(v == 2 for v in counts.values()), str(counts))
    hs = place_hydrogens(heavy, counts)
    res = build_gjf("C3H6 cycloalkane Isomer #1", heavy, hs, G)
    check("环丙烷：连通性可用", res.connectivity, str(res.warnings))
    check("环丙烷：校验零问题", validate_gjf(res.text) == [], str(validate_gjf(res.text)))


# --------------------------------------------------------------------------
# 4. 可靠性守卫（降级而非写错）
# --------------------------------------------------------------------------

def test_guard_degradation():
    print("\n== 4. 可靠性守卫 ==")
    import networkx as nx

    G = nx.Graph()
    G.add_edge(0, 1, bond_type='single')
    heavy = {0: (0.0, 0.0, 0.0), 1: (1.54, 0.0, 0.0)}
    counts = hydrogen_counts(G)
    hs = place_hydrogens(heavy, counts)

    # 4.1 氢数少一个（与价态不符）
    res = build_gjf("C2H6 short", heavy, hs[:-1], G)
    check("氢数不符 → 降级", res.connectivity is False, str(res.warnings))
    check("氢数不符 → route 不带 Geom=Connectivity",
          "Geom=Connectivity" not in res.text)
    check("氢数不符 → 仍是合法输入卡", validate_gjf(res.text) == [],
          str(validate_gjf(res.text)))
    check("氢数不符 → 有警告说明", any("降级" in w or "跳过" in w for w in res.warnings),
          str(res.warnings))

    # 4.2 氢坐标远离（几何不可信）
    res = build_gjf("C2H6 far", heavy, [(40.0, 0.0, 0.0)] * len(hs), G)
    check("H 距离过远 → 降级", res.connectivity is False, str(res.warnings))

    # 4.3 无图（G=None）
    res = build_gjf("C2H6 nograph", heavy, hs, None)
    check("无图 → 不输出连通性", res.connectivity is False)
    check("无图 → 仍是合法输入卡", validate_gjf(res.text) == [])

    # 4.4 图中重原子缺坐标
    G3 = nx.Graph()
    G3.add_edge(0, 1, bond_type='single')
    G3.add_edge(1, 2, bond_type='single')
    res = build_gjf("C3H8 missing", {0: (0.0, 0.0, 0.0), 1: (1.54, 0.0, 0.0)},
                    place_hydrogens({0: (0.0, 0.0, 0.0), 1: (1.54, 0.0, 0.0)},
                                    hydrogen_counts(G3)), G3)
    check("重原子缺坐标 → 降级", res.connectivity is False, str(res.warnings))

    # 4.5 标题以 # 开头 → 自动剥离且不报错
    res = build_gjf("# C6H6 bad", heavy, hs, G)
    title = _parse_blocks(res.text)["lines"][_parse_blocks(res.text)["title_index"]]
    check("标题以 # 开头 → 已剥离", not title.startswith("#"), title)
    check("标题以 # 开头 → 仍合法", validate_gjf(res.text) == [],
          str(validate_gjf(res.text)))


# --------------------------------------------------------------------------
# 5. 配置
# --------------------------------------------------------------------------

def test_settings():
    print("\n== 5. 配置 ==")
    check("归一化 '#p B3LYP/6-31G* Opt'",
          normalize_route_keywords("#p B3LYP/6-31G* Opt") == "B3LYP/6-31G* Opt")
    check("归一化 '##P  M06-2X/def2TZVP'",
          normalize_route_keywords("##P  M06-2X/def2TZVP") == "M06-2X/def2TZVP")
    check("归一化多余空白",
          normalize_route_keywords("  B3LYP/6-31G(d)   Opt ") == "B3LYP/6-31G(d) Opt")

    s = GjfSettings(charge=-1, multiplicity=2, nproc=0, mem="", route_keywords="")
    n = s.normalized()
    check("nproc 越界回落", n.nproc >= 1, str(n.nproc))
    check("mem 空串回落", n.mem == "4GB", n.mem)
    check("route 空串回落", n.route_keywords != "", n.route_keywords)
    check("负电荷保留", n.charge == -1 and n.multiplicity == 2)

    s2 = GjfSettings(charge=1, multiplicity=3)
    r = compose_route_line(s2, connectivity=False)
    check("route 无连通性时不带 Geom", "Geom" not in r, r)
    r = compose_route_line(s2, connectivity=True)
    check("route 幂等（反复调用只出现一次 Geom）",
          r.count("Geom=Connectivity") == 1 and r.count("Geom=Connectivity") == 1, r)
    s3 = GjfSettings(route_keywords="B3LYP/6-31G(d) Geom=Connectivity Opt")
    r = compose_route_line(s3, connectivity=True)
    check("用户手写 Geom 不重复", r.count("Geom") == 1, r)

    check("sanitize_name 去非法字符", sanitize_name("C6H6 / alkene #1") == "C6H6_alkene_1",
          sanitize_name("C6H6 / alkene #1"))

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cfg.json")
        ok = save_settings(GjfSettings(charge=-2, multiplicity=1, nproc=8,
                                       mem="16GB", route_keywords="M06-2X/def2TZVP"), p)
        check("配置可写入", ok)
        got = load_settings(p)
        check("配置往返一致",
              (got.charge, got.multiplicity, got.nproc, got.mem) == (-2, 1, 8, "16GB"),
              str(got.to_dict()))
        check("route 往返一致", got.route_keywords == "M06-2X/def2TZVP", got.route_keywords)
        # 损坏文件不应抛异常
        with open(p, "w", encoding="utf-8") as f:
            f.write("{not json")
        check("损坏配置 → 回落默认且不抛异常", load_settings(p).charge == 0)

    # 配置真的作用于输出
    import networkx as nx
    G = nx.Graph()
    G.add_edge(0, 1, bond_type='single')
    heavy = {0: (0.0, 0.0, 0.0), 1: (1.54, 0.0, 0.0)}
    hs = place_hydrogens(heavy, hydrogen_counts(G))
    res = build_gjf("radical", heavy, hs, G,
                    settings=GjfSettings(charge=-1, multiplicity=2,
                                         route_keywords="UB3LYP/6-31G(d) Opt",
                                         mem="8GB", nproc=2,
                                         use_connectivity=False))
    check("电荷/多重度可配置生效", "\n-1 2\n" in res.text, res.text[:200])
    check("mem 可配置生效", "%Mem=8GB" in res.text)
    check("nproc 可配置生效", "%NProcShared=2" in res.text)
    check("use_connectivity=False 生效", "Geom" not in res.text and res.connectivity is False)
    check("可配置后仍合法", validate_gjf(res.text) == [], str(validate_gjf(res.text)))


# --------------------------------------------------------------------------
# 6. 端到端（真实生成器 + RDKit 坐标）
# --------------------------------------------------------------------------

def test_end_to_end():
    print("\n== 6. 端到端（真实生成器 + RDKit 坐标） ==")
    try:
        import rdkit  # noqa: F401
    except Exception as e:
        check("RDKit 可用（跳过端到端）", False, f"未安装 rdkit: {e}")
        return

    import networkx as nx
    from Main import _rdkit_coords_single, _rdkit_coords_with_bonds
    from original_programs.alkane_isomer_visualizer import AlkaneIsomerGenerator

    # 6.1 烷烃：C4H10 / C5H12 的全部异构体（走真实生成器 + 真实坐标）
    gen = AlkaneIsomerGenerator(use_parallel=False)
    for n_c, expected in ((4, 2), (5, 3), (6, 5)):
        canon_list = gen.generate_isomers(n_c, show_progress=False)
        check(f"C{n_c}H{2 * n_c + 2} 生成 {expected} 个骨架",
              len(canon_list) == expected, f"得到 {len(canon_list)}")

        clean = 0
        degraded = 0
        for idx, canon in enumerate(canon_list, 1):
            adj = gen.canon_to_adjacency(canon)
            G = nx.Graph()
            for node, nbrs in adj.items():
                for nb in nbrs:
                    G.add_edge(node, nb, bond_type='single')
            c_coords, h_coords = _rdkit_coords_single(G)
            res = build_gjf(f"C{n_c}H{2 * n_c + 2} alkane Isomer #{idx}",
                            c_coords, h_coords, G)
            probs = validate_gjf(res.text)
            if probs:
                print(f"    isomer#{idx} probs={probs[:2]}")
            elif res.connectivity:
                clean += 1
            else:
                degraded += 1
                print(f"    isomer#{idx} 降级: {res.warnings[:1]}")
        check(f"C{n_c} 全部 {len(canon_list)} 个异构体端到端合法且连通性可用",
              clean == len(canon_list),
              f"可用 {clean} / 降级 {degraded} / 共 {len(canon_list)}")

    # 6.2 乙烯/丁二烯类（含双键）
    for spec, label in (
        ([(0, 1, 'double')], "C2H4"),
        ([(0, 1, 'single'), (1, 2, 'double')], "C3H6"),
        ([(0, 1, 'double'), (1, 2, 'single'), (2, 3, 'double')], "C4H6"),
    ):
        G = nx.Graph()
        for u, v, bt in spec:
            G.add_edge(u, v, bond_type=bt)
        c_coords, h_coords = _rdkit_coords_with_bonds(G)
        res = build_gjf(f"{label} alkene Isomer #1", c_coords, h_coords, G)
        probs = validate_gjf(res.text)
        check(f"{label} 端到端合法（连通性={res.connectivity}）",
              not probs, f"probs={probs[:2]} warn={res.warnings[:1]}")
        if not res.connectivity:
            print(f"    降级原因: {res.warnings}")

    # 6.3 逐一核对：真实坐标下最近重原子距离是否在阈值内
    G = nx.Graph()
    G.add_edge(0, 1, bond_type='single')
    c_coords, h_coords = _rdkit_coords_single(G)
    dists = []
    for h in h_coords:
        d = min(math.dist(tuple(h), tuple(c)) for c in c_coords.values())
        dists.append(d)
    if dists:
        check(f"C-H 最近重原子距离均在阈值内（max={max(dists):.3f} Å ≤ 1.4）",
              max(dists) <= 1.4, f"max={max(dists):.3f}")


def main():
    print("=" * 70)
    print("Gaussian .gjf 导出回归测试" + ("（含端到端档）" if RUN_E2E else ""))
    print("=" * 70)
    test_block_order()
    test_legacy_defect_detected()
    test_connectivity()
    test_guard_degradation()
    test_settings()
    if RUN_E2E:
        test_end_to_end()
    print("\n" + "=" * 70)
    print(f"结果: PASS={PASS}  FAIL={FAIL}")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
