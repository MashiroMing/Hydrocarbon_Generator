"""
多烯炔烃同分异构体统一生成器（不饱和度正确版）
不饱和度计算：双键 +1，三键 +2
生成策略：分层递推（双键来自 k-1 层，三键来自 k-2 层）
"""

import sys
from collections import defaultdict
from typing import List, Dict, Tuple
import networkx as nx

# ---------- 烷烃骨架生成器导入 ----------
import os
_sys_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)

def _get_alkane_generator():
    from alkene import AlkaneIsomerGenerator
    return AlkaneIsomerGenerator


class PolyalkenyneGenerator:
    """
    多烯炔烃生成器
    usage:
        gen = PolyalkenyneGenerator()
        isomers = gen.generate(n_c, n_db, n_tb)           # 指定双/三键数量
        all_at_k = gen.generate_all_at_k(n_c, k)          # 总不饱和度 k 的所有组合
    """

    def __init__(self, alkane_generator=None):
        self.alkane_gen = alkane_generator or _get_alkane_generator()()

    # ================== 主要接口 ==================
    def generate(self, n_carbons: int, n_db: int, n_tb: int) -> List[nx.Graph]:
        """生成指定双键数、三键数的异构体"""
        if n_carbons < 2:
            return []
        k_target = n_db + 2 * n_tb          # 不饱和度
        if k_target == 0:
            return []

        layers = self._build_layers(n_carbons, k_target)

        if k_target in layers and (n_db, n_tb) in layers[k_target]:
            result = []
            for group in layers[k_target][(n_db, n_tb)].values():
                result.extend(group)
            return result
        return []

    def generate_all_at_k(self, n_carbons: int, k: int) -> Dict[Tuple[int,int], List[nx.Graph]]:
        """生成不饱和度 = k 的所有异构体，按 (双键数,三键数) 分组"""
        layers = self._build_layers(n_carbons, k)
        result = {}
        if k in layers:
            for (db, tb), bucket in layers[k].items():
                result[(db, tb)] = [g for group in bucket.values() for g in group]
        return result

    # ================== 核心递推（修正） ==================
    def _build_layers(self, n_c: int, max_k: int):
        """
        构建分层
        layers[t] = {(db,tb): {canon: [Graph, ...]}}
        t = db + 2*tb
        """
        # Layer 0: 烷烃
        layer0 = self._get_alkane_dict(n_c)
        layers = {0: {(0, 0): layer0}}
        print(f"L0: 烷烃 {n_c}C → {sum(len(g) for g in layer0.values())} 棵骨架")

        # 递推
        for t in range(1, max_k + 1):
            new_layer = defaultdict(dict)   # (db, tb) -> canon -> Graph

            # 从 t-1 层加双键
            if t - 1 in layers:
                for (db, tb), bucket in layers[t - 1].items():
                    self._extend_by_double(bucket, db + 1, tb, new_layer)

            # 从 t-2 层加三键 (当 t >= 2)
            if t - 2 in layers:
                for (db, tb), bucket in layers[t - 2].items():
                    self._extend_by_triple(bucket, db, tb + 1, new_layer)

            # 统计输出
            total = sum(len(g) for b in new_layer.values() for g in b.values())
            if total == 0:
                break
            layers[t] = new_layer
            parts = []
            for (db, tb) in sorted(new_layer.keys()):
                cnt = sum(len(g) for g in new_layer[(db, tb)].values())
                parts.append(f"{db}db+{tb}tb:{cnt}")
            print(f"L{t} (α={t}): {total} 个 | {' '.join(parts)}")

        return layers

    # ================== 加键操作 ==================
    def _extend_by_double(self, bucket, new_db, new_tb, target_layer):
        target_bucket = target_layer[(new_db, new_tb)]
        for group in bucket.values():
            for G in group:
                for u, v in list(G.edges()):
                    if G[u][v].get('bond_type', 'single') != 'single':
                        continue
                    if not self._can_insert_double(G, u, v):
                        continue
                    H = self._upgrade_edge_to_double(G, u, v)
                    if self._validate_molecule(H):
                        self._add_unique(H, target_bucket)

    def _extend_by_triple(self, bucket, new_db, new_tb, target_layer):
        target_bucket = target_layer[(new_db, new_tb)]
        for group in bucket.values():
            for G in group:
                for u, v in list(G.edges()):
                    if G[u][v].get('bond_type', 'single') != 'single':
                        continue
                    if not self._can_insert_triple(G, u, v):
                        continue
                    H = self._upgrade_edge_to_triple(G, u, v)
                    if self._validate_molecule(H):
                        self._add_unique(H, target_bucket)

    # ================== 化学约束 ==================
    def _can_insert_double(self, G, u, v):
        for node in (u, v):
            if self._bond_load(G, node) + 1 > 4:
                return False
            new_double = self._count_double_bonds(G, node) + 1
            if new_double >= 2 and G.degree(node) > 2:
                return False
        return True

    def _can_insert_triple(self, G, u, v):
        for node in (u, v):
            if self._bond_load(G, node) + 2 > 4:
                return False
            if G.degree(node) > 2:
                return False
            if self._count_double_bonds(G, node) > 0 or self._count_triple_bonds(G, node) > 0:
                return False
        return True

    def _upgrade_edge_to_double(self, G, u, v):
        H = nx.Graph()
        for n in G.nodes():
            H.add_node(n, label=G.nodes[n].get('label', 'C'))
        for a, b, data in G.edges(data=True):
            if (a == u and b == v) or (a == v and b == u):
                H.add_edge(a, b, bond_type='double')
            else:
                H.add_edge(a, b, bond_type=data.get('bond_type', 'single'))
        return H

    def _upgrade_edge_to_triple(self, G, u, v):
        H = nx.Graph()
        for n in G.nodes():
            H.add_node(n, label=G.nodes[n].get('label', 'C'))
        for a, b, data in G.edges(data=True):
            if (a == u and b == v) or (a == v and b == u):
                H.add_edge(a, b, bond_type='triple')
            else:
                H.add_edge(a, b, bond_type=data.get('bond_type', 'single'))
        return H

    def _validate_molecule(self, G):
        for node in G.nodes():
            load = self._bond_load(G, node)
            deg = G.degree(node)
            if load > 4:
                return False
            n_db = self._count_double_bonds(G, node)
            n_tb = self._count_triple_bonds(G, node)
            if n_tb > 0 and deg > 2:
                return False
            if n_db >= 2 and deg > 2:
                return False
        return True

    # ================== 辅助函数 ==================
    def _bond_load(self, G, node):
        total = 0
        for nb in G.neighbors(node):
            bt = G[node][nb].get('bond_type', 'single')
            if bt == 'triple':   total += 3
            elif bt == 'double': total += 2
            else:                total += 1
        return total

    def _count_double_bonds(self, G, node):
        return sum(1 for nb in G.neighbors(node) if G[node][nb].get('bond_type') == 'double')

    def _count_triple_bonds(self, G, node):
        return sum(1 for nb in G.neighbors(node) if G[node][nb].get('bond_type') == 'triple')

    def _get_alkane_dict(self, n_c):
        canons = self.alkane_gen.generate_isomers(n_c)
        graphs = {}  # h -> [Graph, ...]，同一桶内可存多个不同构图
        for canon in canons:
            adj = self.alkane_gen.canon_to_adjacency(canon)
            G = nx.Graph()
            for node, nbrs in adj.items():
                G.add_node(node, label='C')
                for v in nbrs:
                    if v > node:
                        G.add_edge(node, v, bond_type='single')
            self._add_unique(G, graphs)
        return graphs

    def _add_unique(self, G, bucket):
        """将图添加到去重字典，同一WL哈希桶内保留所有不同构的图"""
        h = nx.weisfeiler_lehman_graph_hash(G, edge_attr='bond_type', node_attr='label')
        if h not in bucket:
            bucket[h] = [G]
            return
        # 与桶内所有图做同构比较
        for existing in bucket[h]:
            if self._is_isomorphic(G, existing):
                return  # 已存在同构图
        bucket[h].append(G)

    def _is_isomorphic(self, G1, G2):
        return nx.is_isomorphic(
            G1, G2,
            node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
            edge_match=lambda e1, e2: e1.get('bond_type') == e2.get('bond_type')
        )


# ==================== 交互式命令行 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("  多烯炔烃同分异构体生成器（不饱和度修正版）")
    print("  双键+1α，三键+2α")
    print("=" * 60)

    gen = PolyalkenyneGenerator()

    while True:
        try:
            cmd = input("\n输入示例:\n"
                        "  10 1       → 碳数+总不饱和度k\n"
                        "  6 2 1      → 碳数+双键数+三键数\n"
                        "  6 ene      → 只烯烃 (三键=0)\n"
                        "  6 yne      → 只炔烃 (双键=0)\n"
                        "  q          → 退出\n"
                        "> ").strip()
            if cmd.lower() == 'q':
                break

            parts = cmd.split()
            if not parts:
                continue

            if len(parts) == 2 and parts[1].lower() in ('ene', 'yne'):
                n_c = int(parts[0])
                only_type = parts[1].lower()
                max_k = n_c - 1
                print(f"生成 C{n_c} {'烯烃' if only_type=='ene' else '炔烃'} 系列 ...")
                if only_type == 'ene':
                    for k in range(1, max_k + 1):
                        res = gen.generate(n_c, k, 0)   # n_db=k, n_tb=0
                        if res:
                            print(f"  k={k}, {k}烯烃 (C{n_c}H{2*n_c+2-2*k}): {len(res)} 个")
                        else:
                            break
                else:  # yne
                    # 炔烃的最小k=2（一个三键），最大k由碳数决定
                    for k in range(2, max_k + 1, 2):  # 只能偶数不饱和度
                        # 不饱和度k，全部为三键：k/2个三键
                        n_tb = k // 2
                        res = gen.generate(n_c, 0, n_tb)
                        if res:
                            print(f"  k={k}, {n_tb}炔烃 (C{n_c}H{2*n_c+2-2*k}): {len(res)} 个")
                        else:
                            break

            elif len(parts) == 2:
                n_c, k = int(parts[0]), int(parts[1])
                print(f"\n不饱和度 k={k} 的所有组合 ...")
                all_combos = gen.generate_all_at_k(n_c, k)
                total = 0
                for (db, tb) in sorted(all_combos.keys()):
                    cnt = len(all_combos[(db, tb)])
                    print(f"  双键={db} 三键={tb} : {cnt} 个")
                    total += cnt
                print(f"  总计: {total} 个异构体")
                detail = input("列出结构？(y/n): ").strip().lower()
                if detail == 'y':
                    for (db, tb), isomers in sorted(all_combos.items()):
                        print(f"\n--- {db}db + {tb}tb ---")
                        for i, G in enumerate(isomers[:20], 1):
                            dbs = [(u,v) for u,v,d in G.edges(data=True) if d.get('bond_type')=='double']
                            tbs = [(u,v) for u,v,d in G.edges(data=True) if d.get('bond_type')=='triple']
                            print(f"  {i:3d}: 双键 {dbs}, 三键 {tbs}")
                        if len(isomers) > 20:
                            print(f"  ... 还有 {len(isomers)-20} 个")

            elif len(parts) == 3:
                n_c, n_db, n_tb = int(parts[0]), int(parts[1]), int(parts[2])
                isomers = gen.generate(n_c, n_db, n_tb)
                k = n_db + 2 * n_tb
                formula = f"C{n_c}H{2*n_c+2-2*k}"
                print(f"\n{formula} (双键={n_db}, 三键={n_tb}): {len(isomers)} 个")
                if isomers:
                    show = input("列出结构？(y/n): ").strip().lower()
                    if show != 'n':
                        for i, G in enumerate(isomers[:30], 1):
                            dbs = [(u,v) for u,v,d in G.edges(data=True) if d.get('bond_type')=='double']
                            tbs = [(u,v) for u,v,d in G.edges(data=True) if d.get('bond_type')=='triple']
                            print(f"  {i:3d}: 双键 {dbs}, 三键 {tbs}")
                        if len(isomers) > 30:
                            print(f"  ... 还有 {len(isomers)-30} 个")
            else:
                print("输入格式错误，请重新输入")
        except (ValueError, KeyboardInterrupt, EOFError):
            if isinstance(sys.exc_info()[0], (KeyboardInterrupt, EOFError)):
                break
            print("输入无效，请重试")

    print("感谢使用！")