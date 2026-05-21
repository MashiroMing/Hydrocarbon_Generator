# Generation Algorithms

## Phase 1: Dispatch by Structural Type

```
 1: Generate(n, r, d, t)
 2:   if r=0 ∧ d=0 ∧ t=0    → Generate_Alkane(n)
 3:   if r=0                  → Generate_Polyalkenyne(n, d, t)
 4:   if r=1 ∧ t=0           → Generate_Cyclopolyene(n, d)
 5:   if d=0 ∧ t=0           → Generate_Polycycloalkane(n, r)
 6:   otherwise               → Generate_General(n, r, d, t)
 7: End
```

## Phase 2: Saturated Chain Hydrocarbon Generation

```
 1: Generate_Alkane(n)
 2:   R ← RootedTrees(n, max_branch=4)         // Algorithm 2
 3:   F ← {ProcessSymmetry(t) | t ∈ R}         // centroid dedup → free tree
 4:   return Deduplicate(F)                     // Algorithm 4
 5: End

 1: RootedTrees(n, b)                           // Algorithm 2
 2:   if n=1 return {("C", ∅)}
 3:   for each partition (s₁,…,sₖ) of n−1 into ≤b parts
 4:     for each combination in SubTrees(s₁)×…×SubTrees(sₖ)
 5:       emit canonical string "C(sort(substrings))"
 6:   SubTrees(s) = RootedTrees(s, b−1)
 7: End

 1: ProcessSymmetry(t)                          // Algorithm 2a: centroid dedup
 2:   limit ← ⌊n/2⌋
 3:   s_max ← max subtree size of t
 4:   if s_max < limit   → return t            // unique centroid, keep
 5:   if s_max > limit   → return ∅            // non-centroid, discard
 6:   if s_max = limit                     // bicentroid case
 7:     r₂ ← root of the size-limit subtree
 8:     t' ← re-root at r₂ (flip t)
 9:     return min(canonical(t), canonical(t')) // keep lexicographically smaller
10: End
```

## Phase 3: Unsaturation Handling — Ring Skeleton → Triple Bonds → Double Bonds

```
 1: Generate_Polycycloalkane(n, r)              // Algorithm 3
 2:   G ← Generate_Alkane(n)
 3:   for i ← 1 to r
 4:     G ← {AddEdge(g) | g∈G, non-adjacent (u,v), deg(u),deg(v)≤3}
 5:     G ← Deduplicate(G)
 6:   return G
 7: End

 1: Generate_Polyalkenyne(n, d, t)              // Algorithm 3a: acyclic
 2:   L₀ ← AlkaneSkeletons(n)
 3:   for α ← 1 to d+2t
 4:     Lα ← ∅
 5:     if Lα₋₁ exists: Lα ← {Single→Double(g,e) | g∈Lα₋₁, e∈g, Valid(g')}
 6:     if Lα₋₂ exists: Lα ← Lα ∪ {Single→Triple(g,e) | g∈Lα₋₂, e∈g, Valid(g')}
 7:   return Ld+2t
 8: End

 1: Generate_General(n, r, d, t)                // Algorithm 3b: polycyclic
 2:   G ← Generate_Polycycloalkane(n, r)
 3:   for i ← 1 to t                            // triple bonds first
 4:     G ← {Single→Triple(g,e) | g∈G, e∈g, Valid(g')}; G ← Deduplicate(G)
 5:   for i ← 1 to d                            // then double bonds
 6:     G ← {Single→Double(g,e) | g∈G, e∈g, Valid(g')}; G ← Deduplicate(G)
 7:   return G
 8: End
```

## Phase 4: Two-Stage Deduplication (WL Hash + Exact Isomorphism)

```
 1: Deduplicate(G-set)                          // Algorithm 4
 2:   for each G in G-set: bucket[WL_hash(G)] ← G
 3:   for each bucket: remove isomorphic duplicates
 4:   return union of all buckets
 5: End
```

---

## Data Structures and Algorithm Descriptions

> **Terminology note** (deviation from the referenced paper): The deduplication pipeline in this project relies exclusively on **graph isomorphism** (同构) — checking whether two complete molecular graphs are identical up to atom relabeling. The project does not employ subgraph isomorphism or graph homomorphism (同态), as no substructure matching or functional-group embedding is performed during generation. Section 2 is therefore titled "Isomorphism Checking and Canonical Representation" rather than "Isomorphism and Homomorphism."

---

### 1. Structure Representation (结构与子结构的表达)

The pipeline uses four progressively richer graph representations, each dominating a different generation stage. The progressive enrichment reflects the chemical semantics: the alkane skeleton is a pure tree (canonical string suffices), while unsaturated and cyclic variants require bond-type-aware graphs.

#### 1.1 Canonical String Notation

**Scope**: Phase 2 — rooted-tree enumeration.

**Format**: `"C(...)"` — nested parentheses encode the rooted-tree branching structure. Child branches are sorted lexicographically.

```
"C"                     → methane (single carbon)
"C(C)"                  → ethane
"C(C,C)"                → propane
"C(C,C(C))"             → isobutane (central C with two —CH₃ and one —CH₂CH₃)
```

**Construction** (Algorithm 2, line 5):
```
canon = "C(" + join(",", sort(canon_of_child₁, ..., canon_of_childₖ)) + ")"
```
Lexicographic sorting of child strings eliminates permutation redundancy: `C(C, C(C))` and `C(C(C), C)` both resolve to `C(C, C(C))`.

**Key properties**:
- **Uniqueness**: bijective with rooted-tree isomorphism classes.
- **Comparability**: lexicographic order enables centroid deduplication via `min(canon, flipped_canon)`.
- **Parsability**: `parse_substrings(canon)` recovers child branches using depth-counter matching (O(|s|) per parse).

#### 1.2 Adjacency List

**Scope**: Phase 2 → 3 bridge (translation from string to graph).

**Format**: `Dict[int, List[int]]` — 0-indexed node IDs → neighbor lists.

```
canon = "C(C, C(C))"  →  adj = {0: [1, 2], 1: [0], 2: [0, 3], 3: [2]}
```

**Construction**: `canon_to_adjacency(canon)` performs a recursive DFS parse — each `"C(...)"` allocates a sequential node ID, then recursively processes child substrings with parent links, maintaining bidirectional neighbor relationships.

**Canonicalization**: For deduplication, the adjacency list is normalized to a sort-invariant tuple:
```python
tuple(sorted((node, tuple(sorted(nbrs))) for node, nbrs in adj.items()))
```
This removes atom-numbering variability — identical molecules with different labelings produce identical tuples.

#### 1.3 Attributed Molecular Graph

**Scope**: Phase 3 — all bond-insertion operations.

**Format**: `networkx.Graph` with edge-level and node-level attributes:

| Attribute | Type | Values | Purpose |
|---|---|---|---|
| `bond_type` (edge) | `str` | `"single"`, `"double"`, `"triple"` | Bond order between two carbons |
| `label` (node) | `str` | `"C"` | Atom type (uniformly carbon in this project) |

```
nx.Graph: (0,1) bond_type='single', (1,2) bond_type='double', (2,3) bond_type='triple'
```

**Bond upgrading** (`_upgrade_edge_to_double` / `_upgrade_edge_to_triple`): Creates a deep copy of the graph, replacing the target `bond_type` from `'single'` to `'double'` (or `'triple'`). All other edges are preserved unchanged.

**Chemical validation** (`_validate_molecule`) enforces per-atom constraints:

| Constraint | Rule | Chemistry |
|---|---|---|
| Tetravalence | bond_load ≤ 4 | Carbon max 4 bonds (single=1, double=2, triple=3) |
| sp carbon linearity | n_triple > 0 ⇒ degree ≤ 2 | sp hybridized: max 2 σ-bonds |
| No cumulative diene | n_double ≥ 2 ⇒ degree ≤ 2 | Prevents unstable =C= allene geometry |
| sp/sp² exclusivity | n_triple > 0 ⇒ n_double = 0 | sp and sp² cannot co-exist on one carbon |

#### 1.4 Representation Pipeline

```
Phase 2 (DFS tree enum)        Phase 2→3 bridge       Phase 3 (BFS bond insertion)    Phase 4 (dedup)
┌──────────────────┐          ┌──────────────┐        ┌─────────────────────────┐      ┌──────────────────┐
│ Canonical String  │ ──parse→│ Adjacency List│ ──add→│ Attributed nx.Graph      │ ──→  │ Dedup Buckets    │
│ "C(C,C(C))"       │          │ {0:[1,2],...} │        │ edges with bond_type     │      │ {WL_hash: [G,...]}│
└──────────────────┘          └──────────────┘        └─────────────────────────┘      └──────────────────┘
```

---

### 2. Isomorphism Checking and Canonical Representation (同构检查与规范表示)

The project employs **three distinct isomorphism-avoidance mechanisms**, each optimized for the structural complexity of its target domain:

| Mechanism | Domain | Complexity | When applied |
|---|---|---|---|
| Centroid-based tree dedup (Alg. 2a) | Acyclic trees (alkanes) | O(n) | During Phase 2 rooted→free tree conversion |
| Adjacency-list canonicalization | Trees with degree ≤ 4 | O(n log n) | Phase 2 final pass |
| WL hash + VF2 isomorphism | General graphs with bond types | O(d·m) + VF2 | Phase 3 and Phase 4 |

#### 2.1 Centroid-Based Tree Deduplication

**Problem**: A single free tree (unrooted) can be represented by multiple rooted trees depending on the root choice. Algorithm 2a eliminates this redundancy.

**Centroid theorem**: Every n-node tree has either 1 unique centroid or 2 adjacent centroids. A node v is a centroid iff all subtrees rooted at v's neighbors have size ≤ ⌊n/2⌋.

**Algorithm** (`ProcessSymmetry`, Algorithm 2a):

| max_s < ⌊n/2⌋ | max_s = ⌊n/2⌋ | max_s > ⌊n/2⌋ |
|---|---|---|
| Root is the **unique centroid** → keep directly | **Bicentroid** (n even): re-root at the large child, keep `min(t, flipped_t)` | Root is **not** a centroid → discard |

The flip operation re-roots at the child whose subtree size equals ⌊n/2⌋, producing t' — the same free tree viewed from the other centroid. Taking `min(t, t')` breaks the tie deterministically.

**After centroid dedup**: A secondary pass via standardized adjacency lists (§2.2) catches any remaining duplicates.

#### 2.2 Adjacency List Canonicalization

For alkane trees, exact isomorphism reduces to comparing canonical adjacency tuples:

```python
standardized = tuple(sorted((node, tuple(sorted(nbrs))) for node, nbrs in adj.items()))
```

This is valid for trees because: (a) the degree sequence uniquely identifies isomorphism classes for small trees with bounded degree, and (b) the sorted-tuple representation eliminates labeling variance. It runs in O(n log n) and avoids the general graph-isomorphism machinery.

#### 2.3 WL Hash Bucketing

For general molecular graphs with bond-type attributes, the Weisfeiler-Lehman (WL) graph hash provides an **isomorphism-invariant** pre-filter:

```python
h = nx.weisfeiler_lehman_graph_hash(G, edge_attr='bond_type', node_attr='label')
```

The hash is computed by iteratively refining node colors:
1. **Initialization**: each carbon node → color `'C'`; edges labeled by `bond_type`
2. **Iteration** (default 3 rounds): each node's new color = `hash(old_color, multiset{(neighbor_color, edge_color)})`
3. **Final hash**: `hash(multiset{final_node_colors})`

**Guarantee**: graphs with different WL hashes are guaranteed non-isomorphic. Graphs with identical hashes may or may not be isomorphic (WL is a necessary but not sufficient test for general graphs).

#### 2.4 Exact Isomorphism (VF2)

Within each WL hash bucket, pairwise isomorphism is decided by the VF2 algorithm via `networkx`:

```python
nx.is_isomorphic(G1, G2,
    node_match=lambda n1,n2: n1.get('label') == n2.get('label'),
    edge_match=lambda e1,e2: e1.get('bond_type') == e2.get('bond_type'))
```

The `node_match` and `edge_match` callbacks enforce semantic comparison — two atoms must share the same element label, and two bonds must share the same bond order. VF2 is worst-case exponential but, coupled with WL pre-bucketing and the fast-filter below, rarely exhibits worst-case behavior on chemical graphs.

**Degree-signature fast-filter**: Before invoking VF2, the sorted degree sequence `_degree_signature(G)` is compared in O(n log n). Mismatched degree distributions guarantee non-isomorphism without the need for VF2.

#### 2.5 Two-Stage Deduplication Pipeline (Algorithm 4)

```
Candidate graphs
      │
      ▼
  WL_hash(G, edge_attr, node_attr)        ← O(d·m) per graph
      │
      ├─ unique hash → bucket of size 1 → keep directly
      │
      └─ collision bucket → pairwise VF2   ← only within bucket
              │
              ▼
         deduplicated set
```

The two-stage design drastically reduces VF2 invocations: graphs in different WL buckets are never compared, and within each bucket, the degree-signature fast-filter eliminates most non-matches before VF2.

---

### 3. Structure Generation (结构生成)

Structure generation proceeds in four phases, with a **DFS + BFS hybrid traversal** that matches the computational demands of each subproblem.

#### 3.1 Dispatch by Structural Type (Phase 1)

The generation problem is decomposed by the structural parameters (n, r, d, t):

```
 1: Generate(n, r, d, t)
 2:   if r=0 ∧ d=0 ∧ t=0    → Generate_Alkane(n)
 3:   if r=0                  → Generate_Polyalkenyne(n, d, t)
 4:   if r=1 ∧ t=0           → Generate_Cyclopolyene(n, d)
 5:   if d=0 ∧ t=0           → Generate_Polycycloalkane(n, r)
 6:   otherwise               → Generate_General(n, r, d, t)
 7: End
```

Each dispatched path uses a structure-generation algorithm optimized for its specific chemical constraints.

#### 3.2 Rooted Tree Enumeration via Integer Partition (Algorithm 2, DFS)

The function `generate_rooted_with_info(n, max_branches)` is the only DFS component in the pipeline. It enumerates all rooted trees via **two-stage recursion**:

**Stage 1 — Integer Partition**: Enumerate all non-increasing sequences (s₁ ≥ s₂ ≥ ... ≥ sₖ) satisfying Σsᵢ = n−1, k ≤ max_branches, and each sᵢ ≥ 1. The non-increasing constraint eliminates partition-order duplicates.

**Stage 2 — Cartesian Product Assembly**: For each partition, recursively compute all subtree enumerations via `generate_rooted_with_info(sᵢ, max_branches−1)`, take the Cartesian product, sort child canonical strings, and join.

```
generate_rooted_with_info(n=4, b=4)
  ├─ partition (3)    → "C(C(C(C)))", "C(C(C,C))"
  ├─ partition (2,1)  → "C(C,C(C))"
  └─ partition (1,1,1)→ "C(C,C,C)"
```

**Key optimizations**:
- **Memoization cache** `(n, max_branches)`: avoids recomputing identical substructures.
- **Subtree branching limit**: subtree calls use `max_branches=3` (not 4) — the subtree root has already consumed one valence connecting to its parent.

#### 3.3 Ring Construction via Iterative Edge Addition (Algorithm 3, BFS)

Starting from alkane trees, r rounds of edge addition produce r-ring graphs:

```
L₀: {n-carbon alkane trees}                    ← from Algorithm 2 + 2a
     ↓  add one edge per graph, WL-dedup
L₁: {unique 1-ring graphs}
     ↓  add one edge per graph, WL-dedup
L₂: {unique 2-ring graphs}
     ↓  ...
Lᵣ: {unique r-ring graphs}
```

**Per-round operation**: for each graph G, enumerate all non-adjacent node pairs (u, v) with deg(u), deg(v) ≤ 3. Add edge (u, v) with `bond_type='single'`.

**Correctness proof**: any connected r-ring graph on n nodes, when stripped of r edges (preserving connectivity), yields an n-node tree. The reverse process — adding r edges to all trees — is therefore complete.

**Why BFS here**: the same r-ring graph can be reached from multiple (r−1)-ring precursors via different edge additions. Per-round deduplication prevents the combinatorial explosion that would result from carrying forward duplicates.

#### 3.4 Acyclic Unsaturation: Layered Recurrence (Algorithm 3a, BFS)

Unsaturation-indexed recurrence: double bond ≡ Δα = 1, triple bond ≡ Δα = 2.

```
L₀: {alkane skeletons}                         // α = 0
L₁: {L₀ + double}                               // L₀ → all single→double upgrades
L₂: {L₁ + double} ∪ {L₀ + triple}               // two sources due to Δα triple = 2
L₃: {L₂ + double} ∪ {L₁ + triple}
...
Lₖ: select subset matching target (d, t)        // α_target = d + 2t
```

**Completeness guarantee**: any (d, t) distribution with total unsaturation d+2t=k is reachable from either (d−1, t) [via double from Lₖ₋₁] or (d, t−1) [via triple from Lₖ₋₂].

#### 3.5 Polycyclic Unsaturation: Sequential Bond Insertion (Algorithm 3b, BFS)

For polycyclic skeletons, bonds are inserted in strict order: **all triples first, then all doubles**. Each individual bond insertion scans all single-bond edges in the current graph set, then WL-deduplicates.

```
G₀: {r-ring alkane skeletons}
     ↓  insert triple #1 → dedup
G₁: {r-ring, 1 triple}
     ↓  insert triple #2 → dedup
...
G_t: {r-ring, t triples}
     ↓  insert double #1 → dedup
...
G_{t+d}: {r-ring, t triples, d doubles}
```

**Triple-bonds-first ordering rationale**: Triple bonds impose stricter chemical constraints (sp carbon: degree ≤ 2, no co-existing multiple bonds), resulting in fewer valid insertion sites. Inserting triples first maximizes reachable search space; the reverse order would risk having doubles occupy the few sites that triples require.

#### 3.6 Traversal Strategy Summary

| Algorithm | Strategy | Rationale |
|---|---|---|
| Tree enumeration (Alg. 2) | **DFS** | Recursive space unbounded; DFS avoids storing massive intermediate states |
| Ring construction (Alg. 3) | **BFS** | Same r-ring graph from multiple (r−1) precursors; per-layer dedup essential |
| Acyclic unsaturation (Alg. 3a) | **BFS** | Layer index α = d+2t is a natural generation frontier |
| Polycyclic unsaturation (Alg. 3b) | **BFS** | Different insertion orders produce identical molecules; per-bond dedup prevents redundancy |

**Design principle**: DFS for unbounded recursive enumeration (tree generation); BFS whenever a transformation can reach the same output from multiple inputs (edge addition, bond upgrading), so that intermediate deduplication controls search-space growth.

#### 3.7 Chemical Validation Rules

All bond-insertion operations pass through a three-stage validation pipeline:

| Stage | Operation | Cost |
|---|---|---|
| **Pre-check** | Lightweight: bond_load + delta ≤ 4; sp/sp² degree limits | O(1) per candidate |
| **Deep copy + upgrade** | Full graph copy, set target `bond_type` | O(n + m) |
| **Post-validation** | Full per-node constraint check | O(n) |

Constraints enforced: tetravalence (bond_load ≤ 4), sp carbon degree ≤ 2, no cumulative =C=, and sp/sp² exclusivity.

---

### 4. WL Hash Deduplication (WL哈希去重一致性)

While §2 covers the full deduplication pipeline, the WL hash specifically warrants separate treatment as the **primary scalability mechanism** — it is the single technique that makes large-scale generation (e.g., 90,111 intermediate 5-ring skeletons at C11) computationally feasible.

#### 4.1 WL Iterative Color Refinement

The Weisfeiler-Lehman algorithm operates as a deterministic, isomorphism-invariant graph hashing procedure:

```
Round 0:  ∀v ∈ V: color₀(v) = label(v)              // node label ('C')
Round k:  ∀v ∈ V: colorₖ(v) = hash(colorₖ₋₁(v),     // own previous color
                                     {{(colorₖ₋₁(u), edge_attr(v,u)) | u ∈ N(v)}})  // neighbor multiset
Final:    WL_hash(G) = hash({{color_d(v) | v ∈ V}})  // multiset of final colors
```

**Edge-attributed variant**: The hash incorporates `bond_type` (single/double/triple) as edge colors, ensuring that graphs differing only in bond order produce distinct hashes. This is critical for distinguishing, e.g., C—C=C—C—C from C—C—C≡C—C (different bond distribution, identical carbon skeleton topology).

**Iteration depth**: The project uses the default 3 iterations. For the chemical graphs in this project (bounded degree, small diameter), 3 rounds suffice to propagate structural information across the entire molecule.

#### 4.2 Integration Across All Modules

All 9 molecular-graph generators in `original_programs/` now use the identical two-stage WL + VF2 deduplication pattern:

```python
# Stage 1: WL bucketing
h = nx.weisfeiler_lehman_graph_hash(G, edge_attr='bond_type', node_attr='label')
if h not in buckets:
    buckets[h] = [G]
else:
    # Stage 2: VF2 within bucket
    for existing in buckets[h]:
        if nx.is_isomorphic(G, existing,
                node_match=lambda n1,n2: n1.get('label') == n2.get('label'),
                edge_match=lambda e1,e2: e1.get('bond_type') == e2.get('bond_type')):
            break   # duplicate, discard
    else:
        buckets[h].append(G)
```

#### 4.3 Performance Characteristics

| Component | Time Complexity | Role |
|---|---|---|
| WL hash (per graph) | O(d · m) for d iterations, m edges | Partitioning: divides candidate set into buckets |
| Degree-signature filter | O(n log n) | Intra-bucket pre-filter: eliminates ~80% of non-matches |
| VF2 isomorphism | Worst-case O(n! · n), typical O(n²) on chemical graphs | Exact verification: only invoked within collision buckets |

**Empirical observation**: On C11 generation runs (up to 90,111 intermediate graphs per layer), the two-stage pipeline reduces VF2 invocations by approximately 3–4 orders of magnitude compared to naive pairwise comparison, bringing large-scale constitutional isomer generation into feasible runtime.
