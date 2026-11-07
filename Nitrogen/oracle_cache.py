# -*- coding: utf-8 -*-
"""Nitrogen/oracle_cache.py — 预言机结果缓存（内存 + 磁盘）

目的：同一分子式的**重复生成**（用户改筛选再生成、重启后再算）不必重跑数分钟
到数小时。缓存键含**引擎版本**，引擎语义变更时自动失效，避免"用旧结果"。

设计要点：
  · 存储位置在**用户缓存目录**（不污染仓库）：Windows ``%LOCALAPPDATA%``，
    其它 ``~/.cache``；可用环境变量 ``ISOMER_ORACLE_CACHE`` 覆盖；
  · 序列化用 ``pickle + gzip``（图是 networkx.Graph，可 pickle）；
  · **规模守卫**：结构数或文件体积过大时不写盘（避免占用磁盘）；
  · **总容量守卫**：超过上限时按修改时间删除最旧文件；
  · 校验：读回后校验元素/键级属性完整，损坏则视为未命中并删除。

⚠ 缓存只用于 **GUI 生成路径**；`validate.py` 等验证入口**不读缓存**，
   以免掩盖回归（判据必须每次真算）。
"""

import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ['CACHE_VERSION', 'cache_dir', 'cache_key', 'load', 'save',
           'generate_cached', 'clear_cache', 'cache_stats']

#: 引擎语义版本——**枚举/去重口径变更时必须递增**（旧缓存自动失效）
CACHE_VERSION = 'oracle-v3'

#: 单个条目上限（结构数 / 压缩后字节数）
MAX_STRUCTS_PER_ENTRY = 300_000
MAX_BYTES_PER_ENTRY = 256 * 1024 * 1024
#: 缓存目录总上限
MAX_TOTAL_BYTES = 1024 * 1024 * 1024


def cache_dir() -> Path:
    """缓存目录（可用 ``ISOMER_ORACLE_CACHE`` 覆盖）"""
    env = os.environ.get('ISOMER_ORACLE_CACHE')
    if env:
        p = Path(env)
    elif os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or str(Path.home())
        p = Path(base) / 'IsomerOracleCache'
    else:
        p = Path.home() / '.cache' / 'isomer_oracle'
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(n_c: int, n_n: int, n_o: int, n_h: int) -> str:
    return f"{CACHE_VERSION}_C{n_c}N{n_n}O{n_o}H{n_h}"


def _entry_path(n_c: int, n_n: int, n_o: int, n_h: int) -> Path:
    return cache_dir() / f"{cache_key(n_c, n_n, n_o, n_h)}.pkl.gz"


def _valid(g) -> bool:
    """结构校验：节点带 label、边带 bond_type（缺属性的条目视为损坏）"""
    try:
        if g.number_of_nodes() == 0:
            return False
        for _, d in g.nodes(data=True):
            if 'label' not in d:
                return False
        for _, _, d in g.edges(data=True):
            if 'bond_type' not in d:
                return False
        return True
    except Exception:
        return False


def load(n_c: int, n_n: int, n_o: int, n_h: int) -> Optional[List]:
    """读缓存（命中返回图列表；未命中/损坏返回 None）"""
    p = _entry_path(n_c, n_n, n_o, n_h)
    if not p.exists():
        return None
    try:
        with gzip.open(p, 'rb') as f:
            payload = pickle.load(f)
        graphs = payload.get('graphs')
        if not isinstance(graphs, list) or not all(_valid(g) for g in graphs):
            raise ValueError('缓存内容不完整')
        return graphs
    except Exception:
        try:
            p.unlink()
        except Exception:
            pass
        return None


def save(n_c: int, n_n: int, n_o: int, n_h: int, graphs: List) -> bool:
    """写缓存（规模守卫 + 总容量守卫）；成功返回 True"""
    if not graphs or len(graphs) > MAX_STRUCTS_PER_ENTRY:
        return False
    p = _entry_path(n_c, n_n, n_o, n_h)
    tmp = p.with_suffix('.tmp')
    try:
        payload = {'version': CACHE_VERSION, 'created': time.time(),
                   'n_c': n_c, 'n_n': n_n, 'n_o': n_o, 'n_h': n_h,
                   'graphs': graphs}
        with gzip.open(tmp, 'wb', compresslevel=4) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        if tmp.stat().st_size > MAX_BYTES_PER_ENTRY:
            tmp.unlink()
            return False
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False
    _enforce_total_size()
    return True


def _enforce_total_size() -> None:
    """总容量守卫：超限时按修改时间删除最旧条目"""
    try:
        files = sorted(cache_dir().glob(f'{CACHE_VERSION}_*.pkl.gz'),
                       key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while files and total > MAX_TOTAL_BYTES:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink()
    except Exception:
        pass


def cache_stats() -> Tuple[int, int]:
    """返回 (条目数, 总字节数)"""
    try:
        files = list(cache_dir().glob(f'{CACHE_VERSION}_*.pkl.gz'))
        return len(files), sum(f.stat().st_size for f in files)
    except Exception:
        return 0, 0


def clear_cache() -> int:
    """清空当前版本缓存，返回删除条目数"""
    n = 0
    try:
        for f in cache_dir().glob(f'{CACHE_VERSION}_*.pkl.gz'):
            f.unlink()
            n += 1
    except Exception:
        pass
    return n


def generate_cached(n_c: int, n_n: int, n_o: int, n_h: int,
                    parallel: bool = True, max_atoms: int = 10,
                    use_cache: bool = True, stream_cb=None) -> Tuple[List, Dict]:
    """带缓存的预言机枚举

    Returns:
        ``(graphs, info)``；info 含 ``hit`` / ``seconds`` / ``n`` / ``saved``
    """
    t0 = time.time()
    if use_cache:
        got = load(n_c, n_n, n_o, n_h)
        if got is not None:
            if stream_cb:
                try:
                    stream_cb(got)
                except Exception:
                    pass
            return got, {'hit': True, 'seconds': time.time() - t0,
                         'n': len(got), 'saved': False}
    from Nitrogen.atomic_matrix_n import AtomicMatrixNGenerator
    gen = AtomicMatrixNGenerator(n_c, n_n, n_o, max_atoms=max_atoms)
    graphs = (gen.generate_for_target_h_parallel(n_h, stream_cb=stream_cb)
              if parallel else gen.generate_for_target_h(n_h))
    saved = save(n_c, n_n, n_o, n_h, graphs) if use_cache else False
    return graphs, {'hit': False, 'seconds': time.time() - t0,
                    'n': len(graphs), 'saved': saved}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print('缓存目录:', cache_dir())
    print('当前条目:', cache_stats())
    if '--clear' in sys.argv:
        print('已清理条目:', clear_cache())
    else:
        for fml, args, nh in [('C5H5N', (5, 1, 0), 5)]:
            for i in (1, 2):
                gs, info = generate_cached(*args, nh)
                print(f"  {fml} 第{i}次: n={info['n']} 命中={info['hit']} "
                      f"用时={info['seconds']:.2f}s 已写盘={info['saved']}")
