# 生成算法

## 阶段一：按结构类型分派

```
 1: Generate(n, r, d, t)
 2:   if r=0 ∧ d=0 ∧ t=0    → Generate_Alkane(n)
 3:   if r=0                  → Generate_Polyalkenyne(n, d, t)
 4:   if r=1 ∧ t=0           → Generate_Cyclopolyene(n, d)
 5:   if d=0 ∧ t=0           → Generate_Polycycloalkane(n, r)
 6:   otherwise               → Generate_General(n, r, d, t)
 7: End
```

## 阶段二：饱和链烃骨架生成

```
 1: Generate_Alkane(n)
 2:   R ← RootedTrees(n, max_branch=4)         // 算法2
 3:   F ← {ProcessSymmetry(t) | t ∈ R}         // 质心去重 → 自由树
 4:   return Deduplicate(F)                     // 算法4
 5: End

 1: RootedTrees(n, b)                           // 算法2
 2:   if n=1 return {("C", ∅)}
 3:   for each partition (s₁,…,sₖ) of n−1 into ≤b parts
 4:     for each combination in SubTrees(s₁)×…×SubTrees(sₖ)
 5:       emit canonical string "C(sort(substrings))"
 6:   SubTrees(s) = RootedTrees(s, b−1)
 7: End

 1: ProcessSymmetry(t)                          // 算法2a：质心去重
 2:   limit ← ⌊n/2⌋
 3:   s_max ← max subtree size of t
 4:   if s_max < limit   → return t            // 唯一质心，保留
 5:   if s_max > limit   → return ∅            // 非质心，丢弃
 6:   if s_max = limit                          // 双质心情形
 7:     r₂ ← root of the size-limit subtree
 8:     t' ← re-root at r₂ (翻转操作)
 9:     return min(canonical(t), canonical(t')) // 取字典序较小者
10: End
```

## 阶段三：不饱和度处理 —— 环骨架 → 三键 → 双键

```
 1: Generate_Polycycloalkane(n, r)              // 算法3
 2:   G ← Generate_Alkane(n)
 3:   for i ← 1 to r
 4:     G ← {AddEdge(g) | g∈G, non-adjacent (u,v), deg(u),deg(v)≤3}
 5:     G ← Deduplicate(G)
 6:   return G
 7: End

 1: Generate_Polyalkenyne(n, d, t)              // 算法3a：无环情形
 2:   L₀ ← AlkaneSkeletons(n)
 3:   for α ← 1 to d+2t
 4:     Lα ← ∅
 5:     if Lα₋₁ exists: Lα ← {Single→Double(g,e) | g∈Lα₋₁, e∈g, Valid(g')}
 6:     if Lα₋₂ exists: Lα ← Lα ∪ {Single→Triple(g,e) | g∈Lα₋₂, e∈g, Valid(g')}
 7:   return Ld+2t
 8: End

 1: Generate_General(n, r, d, t)                // 算法3b：多环情形
 2:   G ← Generate_Polycycloalkane(n, r)
 3:   for i ← 1 to t                            // 优先插入三键
 4:     G ← {Single→Triple(g,e) | g∈G, e∈g, Valid(g')}; G ← Deduplicate(G)
 5:   for i ← 1 to d                            // 再插入双键
 6:     G ← {Single→Double(g,e) | g∈G, e∈g, Valid(g')}; G ← Deduplicate(G)
 7:   return G
 8: End
```

## 阶段四：两阶段去重（WL 哈希 + 精确同构）

```
 1: Deduplicate(G-set)                          // 算法4
 2:   for each G in G-set: bucket[WL_hash(G)] ← G
 3:   for each bucket: remove isomorphic duplicates
 4:   return union of all buckets
 5: End
```

---

## 数据结构与算法描述

> **术语说明**（与参考论文之差异）：本项目的去重管线完全基于**图同构**——判断两个完整分子图在忽略原子编号差异后是否等价。项目在生成过程中不涉及子图同构或图同态，不做子结构匹配或功能团嵌入。因此第 2 节标题为"同构检查与规范表示"而非"同构与同态"。

---

### 1. 结构与子结构的表达

生成管线采用四种逐级丰富的图表示形式，每种主导不同的生成阶段。这种递进式的丰富化反映了化学语义：烷烃骨架是纯树（规范字符串即可表达），而含不饱和键或环的分子需要键类型感知的图结构。

#### 1.1 规范字符串表示

**作用域**：阶段二 —— 有根树枚举。

**格式**：`"C(...)"` —— 嵌套括号编码有根树的分支结构，子分支按字典序排序。

```
"C"                     → 甲烷（单碳原子）
"C(C)"                  → 乙烷
"C(C,C)"                → 丙烷
"C(C,C(C))"             → 异丁烷（中心碳连接两个—CH₃和一个—CH₂CH₃）
```

**构造规则**（算法 2，第 5 行）：
```
canon = "C(" + join(",", sort(canon_of_child₁, ..., canon_of_childₖ)) + ")"
```
子字符串的字典序排序消除了排列冗余：`C(C, C(C))` 与 `C(C(C), C)` 均解析为 `C(C, C(C))`。

**关键性质**：
- **唯一性**：与有根树同构类之间构成双射。
- **可比较性**：字典序支持质心去重中的 `min(canon, flipped_canon)` 操作。
- **可解析性**：`parse_substrings(canon)` 通过深度计数器匹配恢复子分支，每次解析代价 O(|s|)。

#### 1.2 邻接表

**作用域**：阶段二到阶段三的桥梁（字符串到图的转换）。

**格式**：`Dict[int, List[int]]` —— 以 0 为起始索引的节点 ID 映射到邻接列表。

```
canon = "C(C, C(C))"  →  adj = {0: [1, 2], 1: [0], 2: [0, 3], 3: [2]}
```

**构造方式**：`canon_to_adjacency(canon)` 执行递归 DFS 解析——每个 `"C(...)"` 分配一个连续节点 ID，随后递归处理子字符串并维护双向邻接关系。

**规范化**：为去重目的，邻接表被标准化为排序不变的元组形式：
```python
tuple(sorted((node, tuple(sorted(nbrs))) for node, nbrs in adj.items()))
```
此举消除了原子编号的差异性——同一分子以不同编号方案表示时，产生完全相同的元组。

#### 1.3 属性分子图

**作用域**：阶段三 —— 所有键插入操作。

**格式**：`networkx.Graph`，带有边级和节点级属性：

| 属性 | 类型 | 取值 | 含义 |
|---|---|---|---|
| `bond_type`（边） | `str` | `"single"`, `"double"`, `"triple"` | 两碳原子间的键级 |
| `label`（节点） | `str` | `"C"` | 原子类型（本项目统一为碳） |

```
nx.Graph: (0,1) bond_type='single', (1,2) bond_type='double', (2,3) bond_type='triple'
```

**键升级**（`_upgrade_edge_to_double` / `_upgrade_edge_to_triple`）：创建图的深拷贝，将目标边的 `bond_type` 从 `'single'` 替换为 `'double'`（或 `'triple'`）。其余所有边保持不变。

**化学验证**（`_validate_molecule`）逐原子施加约束：

| 约束 | 规则 | 化学含义 |
|---|---|---|
| 四价性 | bond_load ≤ 4 | 碳原子最多四次成键（单键=1，双键=2，三键=3） |
| sp 碳直线性 | n_triple > 0 ⇒ degree ≤ 2 | sp 杂化碳最多 2 个 σ 键 |
| 禁止累积双烯 | n_double ≥ 2 ⇒ degree ≤ 2 | 防止不稳定的 =C= 丙二烯几何构型 |
| sp/sp² 互斥 | n_triple > 0 ⇒ n_double = 0 | 同一碳原子上 sp 与 sp² 不可共存 |

#### 1.4 表示形式流水线

```
阶段二（DFS树枚举）         阶段二→三桥梁          阶段三（BFS键插入）            阶段四（去重）
┌──────────────────┐      ┌──────────────┐      ┌─────────────────────────┐    ┌──────────────────┐
│ 规范字符串         │─解析→│   邻接表      │─添加→│  属性 nx.Graph           │──→ │  去重桶           │
│ "C(C,C(C))"      │      │ {0:[1,2],...} │      │ 边带有 bond_type 属性     │    │ {WL_hash: [G,...]}│
└──────────────────┘      └──────────────┘      └─────────────────────────┘    └──────────────────┘
```

---

### 2. 同构检查与规范表示

本项目采用**三种不同的同构避免机制**，每种针对其目标域的结构复杂度进行了优化：

| 机制 | 适用域 | 复杂度 | 应用时机 |
|---|---|---|---|
| 基于质心的树去重（算法 2a） | 无环树（烷烃） | O(n) | 阶段二有根→自由树转换时 |
| 邻接表规范化 | 度数 ≤ 4 的树 | O(n log n) | 阶段二最终遍历 |
| WL 哈希 + VF2 同构 | 带键类型属性的通用图 | O(d·m) + VF2 | 阶段三与阶段四 |

#### 2.1 基于质心的树去重

**问题**：一棵自由树（无根树）可因根节点选择的不同而对应多棵有根树。算法 2a 消除此冗余。

**质心定理**：任意 n 节点树具有 1 个唯一质心或 2 个相邻质心。节点 v 为质心，当且仅当以 v 的所有邻居为根的子树的规模均 ≤ ⌊n/2⌋。

**算法**（`ProcessSymmetry`，算法 2a）：

| max_s < ⌊n/2⌋ | max_s = ⌊n/2⌋ | max_s > ⌊n/2⌋ |
|---|---|---|
| 根为**唯一质心** → 直接保留 | **双质心**（n 为偶数）：以大子树为根重新定根，保留 `min(t, flipped_t)` | 根**非**质心 → 丢弃 |

翻转操作在子树规模恰好等于 ⌊n/2⌋ 的子节点处重新定根，产生 t' —— 从另一质心视角观察的同一棵自由树。`min(t, t')` 操作确定性地打破平局。

**质心去重之后**：通过标准化邻接表（§2.2）进行二次遍历，捕获任何残余的重复结构。

#### 2.2 邻接表规范化

对烷烃树，精确同构可归结为比较规范邻接元组：

```python
standardized = tuple(sorted((node, tuple(sorted(nbrs))) for node, nbrs in adj.items()))
```

该做法对树有效，因为：(a) 对有界度小树的同构类，度序列唯一确定；(b) 排序元组表示消除了标号差异。其运行时间为 O(n log n)，避免了通用图同构的复杂机制。

#### 2.3 WL 哈希分桶

对带有键类型属性的通用分子图，Weisfeiler-Lehman（WL）图哈希提供**同构不变**的预过滤器：

```python
h = nx.weisfeiler_lehman_graph_hash(G, edge_attr='bond_type', node_attr='label')
```

哈希通过迭代细化节点颜色计算：
1. **初始化**：每个碳节点 → 颜色 `'C'`；边按 `bond_type` 标记
2. **迭代**（默认 3 轮）：每个节点的新颜色 = `hash(旧颜色, 多重集{(邻居颜色, 边颜色)})`
3. **最终哈希**：`hash(多重集{最终节点颜色})`

**保证**：WL 哈希不同的图必定不同构。哈希相同的图可能同构也可能不同构（对一般图，WL 为必要而非充分检验）。

#### 2.4 精确同构（VF2）

在每个 WL 哈希桶内，通过 `networkx` 的 VF2 算法进行成对同构判定：

```python
nx.is_isomorphic(G1, G2,
    node_match=lambda n1,n2: n1.get('label') == n2.get('label'),
    edge_match=lambda e1,e2: e1.get('bond_type') == e2.get('bond_type'))
```

`node_match` 与 `edge_match` 回调函数实施语义比较——两个原子必须具有相同的元素标签，两条键必须具有相同的键级。VF2 的最坏情况复杂度为指数级，但在 WL 预分桶和下述快速过滤器的配合下，在化学图上极少呈现最坏行为。

**度签名快速过滤**：在调用 VF2 之前，以 O(n log n) 的代价比较排序度序列 `_degree_signature(G)`。度分布不匹配即可保证非同理，无需 VF2。

#### 2.5 两阶段去重管线（算法 4）

```
候选图集
      │
      ▼
  WL_hash(G, edge_attr, node_attr)        ← 每图 O(d·m)
      │
      ├─ 唯一哈希 → 桶大小为1 → 直接保留
      │
      └─ 冲突桶 → 桶内成对 VF2             ← 仅在桶内执行
              │
              ▼
         去重后的集合
```

两阶段设计大幅减少了 VF2 调用：不同 WL 桶中的图永不被比较；在同一桶内，度签名快速过滤器在 VF2 之前排除了多数不匹配项。

---

### 3. 结构生成

结构生成分四个阶段推进，采用**DFS + BFS 混合遍历**策略以适应各子问题的计算特性。

#### 3.1 按结构类型分派（阶段一）

生成问题按结构参数 (n, r, d, t) 进行分解：

```
 1: Generate(n, r, d, t)
 2:   if r=0 ∧ d=0 ∧ t=0    → Generate_Alkane(n)
 3:   if r=0                  → Generate_Polyalkenyne(n, d, t)
 4:   if r=1 ∧ t=0           → Generate_Cyclopolyene(n, d)
 5:   if d=0 ∧ t=0           → Generate_Polycycloalkane(n, r)
 6:   otherwise               → Generate_General(n, r, d, t)
 7: End
```

每条分派路径采用针对其特定化学约束优化的结构生成算法。

#### 3.2 基于整数划分的有根树枚举（算法 2，DFS）

函数 `generate_rooted_with_info(n, max_branches)` 是管线中唯一的 DFS 组件，通过**两阶段递归**枚举所有有根树：

**第一阶段——整数划分**：枚举所有非递增序列 (s₁ ≥ s₂ ≥ ... ≥ sₖ)，满足 Σsᵢ = n−1，k ≤ max_branches，且每项 sᵢ ≥ 1。非递增约束消除了划分顺序的重复。

**第二阶段——笛卡尔积组装**：对每个划分，通过 `generate_rooted_with_info(sᵢ, max_branches−1)` 递归计算所有子树枚举，取笛卡尔积，排序子规范字符串并拼接。

```
generate_rooted_with_info(n=4, b=4)
  ├─ 划分 (3)     → "C(C(C(C)))", "C(C(C,C))"
  ├─ 划分 (2,1)   → "C(C,C(C))"
  └─ 划分 (1,1,1) → "C(C,C,C)"
```

**关键优化**：
- **记忆化缓存** `(n, max_branches)`：避免重复计算相同子结构。
- **子树分支限制**：子树调用使用 `max_branches=3`（而非 4）——子树根节点已使用一个化合价连接父节点。

#### 3.3 迭代加边构建环（算法 3，BFS）

从烷烃树出发，r 轮加边产生 r 环图：

```
L₀: {n碳烷烃树}                                ← 来自算法2 + 2a
     ↓  每图加一条边，WL去重
L₁: {唯一单环图}
     ↓  每图加一条边，WL去重
L₂: {唯一双环图}
     ↓  ...
Lᵣ: {唯一r环图}
```

**每轮操作**：对每个图 G，枚举所有不相邻的节点对 (u, v)，满足 deg(u), deg(v) ≤ 3。以 `bond_type='single'` 添加边 (u, v)。

**正确性证明**：任意 n 节点的连通 r 环图，若移除 r 条边（保持连通性），必得到一棵 n 节点树。反向过程——对全部树分别添加 r 条边——因此是完备的。

**此处采用 BFS 的原因**：同一 r 环图可从多个 (r−1) 环前驱通过不同的加边得到。每轮去重避免了因携带重复图进入下一轮而导致的组合爆炸。

#### 3.4 无环不饱和键：分层递推（算法 3a，BFS）

以不饱和度索引的递推关系：双键 ≡ Δα = 1，三键 ≡ Δα = 2。

```
L₀: {烷烃骨架}                                  // α = 0
L₁: {L₀ + 双键}                                  // L₀ → 所有单键→双键升级
L₂: {L₁ + 双键} ∪ {L₀ + 三键}                     // 两个来源，因三键 Δα = 2
L₃: {L₂ + 双键} ∪ {L₁ + 三键}
...
Lₖ: 选取匹配目标 (d, t) 的子集                    // α_target = d + 2t
```

**完备性保证**：任意总不饱和度 d+2t=k 的 (d, t) 分布，必可从 (d−1, t) [通过 Lₖ₋₁ 加双键] 或 (d, t−1) [通过 Lₖ₋₂ 加三键] 达到。

#### 3.5 多环不饱和键：顺序键插入（算法 3b，BFS）

对多环骨架，键按严格顺序插入：**先全部三键，再全部双键**。每次单个键的插入扫描当前图集中所有单键边，随后 WL 去重。

```
G₀: {r环烷烃骨架}
     ↓  插入三键 #1 → 去重
G₁: {r环，1个三键}
     ↓  插入三键 #2 → 去重
...
G_t: {r环，t个三键}
     ↓  插入双键 #1 → 去重
...
G_{t+d}: {r环，t个三键，d个双键}
```

**三键优先排序的理由**：三键施加更严格的化学约束（sp 碳：度数 ≤ 2，不可共存其他重键），有效插入位点较少。优先插入三键可最大化可达搜索空间；若顺序颠倒，双键可能占据三键所需的那少数位点。

#### 3.6 遍历策略总结

| 算法 | 策略 | 理由 |
|---|---|---|
| 树枚举（算法2） | **DFS** | 递归空间无界；DFS 避免存储巨量中间状态 |
| 环构造（算法3） | **BFS** | 同一 r 环图来自多个 (r−1) 环前驱；逐层去重至关重要 |
| 无环不饱和键（算法3a） | **BFS** | 层索引 α = d+2t 是自然的生成前沿 |
| 多环不饱和键（算法3b） | **BFS** | 不同插入顺序可产生同种分子；逐键去重防止冗余 |

**设计原则**：DFS 用于无界递归枚举（树生成）；当变换可从多个输入到达相同输出（加边、键升级）时采用 BFS，使中间去重得以控制搜索空间增长。

#### 3.7 化学验证规则

所有键插入操作均通过三级验证管线：

| 阶段 | 操作 | 代价 |
|---|---|---|
| **预检** | 轻量检测：bond_load + delta ≤ 4；sp/sp² 度数限制 | 每候选 O(1) |
| **深拷贝 + 升级** | 全图拷贝，设置目标 `bond_type` | O(n + m) |
| **后验证** | 全节点约束检查 | O(n) |

执行的约束：四价性（bond_load ≤ 4），sp 碳度数 ≤ 2，禁止累积双键 =C=，sp/sp² 互斥。

---

### 4. WL 哈希去重一致性

尽管 §2 已覆盖完整的去重管线，WL 哈希作为**首要可扩展性机制**仍值得单独论述——正是该技术使得大规模生成（例如 C11 时 90,111 个中间 5 环骨架）在计算上可行。

#### 4.1 WL 迭代颜色精炼

Weisfeiler-Lehman 算法作为一种确定性、同构不变的图哈希过程运作：

```
第0轮：  ∀v ∈ V: color₀(v) = label(v)                // 节点标签 ('C')
第k轮：  ∀v ∈ V: colorₖ(v) = hash(colorₖ₋₁(v),       // 自身先前颜色
                                    {{(colorₖ₋₁(u), edge_attr(v,u)) | u ∈ N(v)}})  // 邻居多重集
最终：    WL_hash(G) = hash({{color_d(v) | v ∈ V}})    // 最终颜色多重集
```

**边属性变体**：哈希将 `bond_type`（单键/双键/三键）作为边颜色纳入计算，确保仅键级不同的图产生相异哈希。这对于区分例如 C—C=C—C—C 与 C—C—C≡C—C（键分布不同，碳骨架拓扑相同）至关重要。

**迭代深度**：本项目使用默认的 3 轮迭代。对项目中的化学图（有界度数、小直径），3 轮足以将结构信息传播至整个分子。

#### 4.2 跨模块统一集成

`original_programs/` 目录下的全部 9 个分子图生成器均采用相同的两阶段 WL + VF2 去重模式：

```python
# 阶段一：WL 分桶
h = nx.weisfeiler_lehman_graph_hash(G, edge_attr='bond_type', node_attr='label')
if h not in buckets:
    buckets[h] = [G]
else:
    # 阶段二：桶内 VF2
    for existing in buckets[h]:
        if nx.is_isomorphic(G, existing,
                node_match=lambda n1,n2: n1.get('label') == n2.get('label'),
                edge_match=lambda e1,e2: e1.get('bond_type') == e2.get('bond_type')):
            break   # 重复，丢弃
    else:
        buckets[h].append(G)
```

#### 4.3 性能特征

| 组件 | 时间复杂度 | 作用 |
|---|---|---|
| WL 哈希（每图） | O(d · m)，d 次迭代，m 条边 | 划分：将候选集分割为桶 |
| 度签名过滤器 | O(n log n) | 桶内预过滤：消除约 80% 的非匹配项 |
| VF2 同构 | 最坏 O(n! · n)，化学图上典型 O(n²) | 精确验证：仅在冲突桶内调用 |

**经验观察**：在 C11 生成运行中（每层最多 90,111 个中间图），两阶段管线较朴素成对比较减少了约 3–4 个数量级的 VF2 调用，使大规模构型异构体生成进入可行运行时间。
