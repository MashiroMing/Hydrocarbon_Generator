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

## Traversal Strategy: Mixed DFS + BFS Architecture

The project employs a deliberate hybrid traversal design — **DFS for combinatorial tree enumeration, BFS for all structural transformations**. The rationale is: DFS suits unbounded combinatorial spaces (tree enumeration), while BFS suits bounded transformations where intermediate deduplication controls the search space.

### Overview

| Phase | Algorithm | Strategy | Rationale |
|---|---|---|---|
| Phase 2 | Tree Enumeration (Algorithm 2) | **DFS** | Integer partition + subtree recursion, deep-first to avoid storing massive intermediate states |
| Phase 2 | Ring Skeleton (Algorithm 3) | **BFS** | Layer-by-layer edge addition: Trees → L₁ → L₂ → ... → Lᵣ, deduplicate at each layer |
| Phase 3a | Acyclic Unsaturation (Algorithm 3a) | **BFS** | Recurrence on unsaturation layers: L₀ → L₁ → ... → Lₖ, Lα draws from Lα₋₁ (double) and Lα₋₂ (triple) |
| Phase 3b | Polycyclic Unsaturation (Algorithm 3b) | **BFS** | Sequential full-pass bond insertion: all triples in current set → dedup → all doubles → dedup |

### DFS: Phase 2 Tree Enumeration

The core function `generate_rooted_with_info(n, max_branches)` uses recursive backtracking to enumerate all rooted trees. The search tree descends from the root carbon through its branching structure to single-atom leaves (base case `n=1`), then backtracks to explore alternative partition paths.

```
generate_rooted_with_info(n=4, b=4)
  ├─ partition (3): subtree s=3 → recursive call (n=3, b=3)
  │    ├─ partition (2): subtree s=2 → (n=2, b=3) → leaf "C(C)"
  │    └─ partition (1,1): "C(C,C)"
  │    → yields "C(C(C(C)))", "C(C(C,C))"
  ├─ partition (2,1): "C(C,C(C))"
  └─ partition (1,1,1): "C(C,C,C)"
```

Key DFS characteristics:
- **Integer partition recursion** (`find_parts`) explores all ways to split `n−1` carbons among ≤4 branches, enforcing non-decreasing order to avoid permutation duplicates.
- **Subtree recursion** calls `generate_rooted_with_info(s, max_branches=3)` — the subtree root has already used one bond connecting to its parent, leaving only 3 free valence slots.
- **Memoization cache** (`cache_key = (n, max_branches)`) avoids recomputing identical substructures across different parent trees.

### BFS: Phase 2 Ring Construction (Algorithm 3)

Starting from all alkane trees, each round adds one edge between two non-adjacent degree-≤3 nodes, incrementing ring count by 1. After each round, the entire layer is deduplicated before proceeding.

```
L₀: {Tree₁, Tree₂, ..., Treeₘ}          ← from DFS (Algorithm 2)
     ↓  add one edge to each, then dedup
L₁: {unique 1-ring graphs}               ← first round
     ↓  add one edge to each, then dedup
L₂: {unique 2-ring graphs}               ← second round
     ↓  ...
Lᵣ: {unique r-ring graphs}              ← final result
```

Why BFS here: the same r-ring graph can be reached from multiple (r−1)-ring precursors via different edge additions. Deduplicating at each layer prevents the next layer from exploding with redundant copies.

### BFS: Phase 3a Acyclic Unsaturation (Algorithm 3a)

Uses **unsaturation-indexed recurrence**: double bond contributes Δα=1, triple bond contributes Δα=2. Each layer Lα draws candidates from Lα₋₁ (add double) and Lα₋₂ (add triple).

```
L₀: {alkane skeletons}                              // α = 0
L₁: {L₀ + double}                                    // α = 1
L₂: {L₁ + double} ∪ {L₀ + triple}                    // α = 2
L₃: {L₂ + double} ∪ {L₁ + triple}                    // α = 3
...
Lₖ: select subset with exact (d, t) match            // α = d + 2t
```

Why BFS here: α is a natural layer index — all structures with the same total unsaturation are generated together. The recurrence guarantees completeness: any (d,t) distribution with d+2t=k must be reachable from either (d−1,t) [α−1] or (d,t−1) [α−2].

### BFS: Phase 3b Polycyclic Unsaturation (Algorithm 3b)

For polycyclic skeletons with unsaturation, bonds are inserted sequentially: all triple bonds first (strictest chemical constraints, fewest viable positions), then all double bonds. Each individual bond insertion does a full pass over the current graph set.

```
G₀: {r-ring alkane skeletons}
     ↓  insert triple bond #1 to all G ∈ G₀ → dedup
G₁: {r-ring, 1 triple}
     ↓  insert triple bond #2 to all G ∈ G₁ → dedup
...
G_t: {r-ring, t triples}
     ↓  insert double bond #1 to all G ∈ G_t → dedup
G_{t+1}: {r-ring, t triples, 1 double}
     ↓  ...
G_{t+d}: {r-ring, t triples, d doubles}          ← final result
```

Why BFS here: the same (t triple, d double) structure can be reached through different insertion orders. Deduplicating after each bond prevents the next round from operating on duplicated graphs. The triple-bonds-first ordering is a pruning strategy — triples have stricter constraints (sp carbon, degree≤2, no other multiple bonds), so inserting them first maximizes the reachable search space.

### Design Principle

The DFS/BFS boundary follows a clean rule — **DFS enumerates combination spaces where the structure graph itself is growing recursively (tree generation); BFS processes all candidates uniformly whenever a transformation may produce duplicates from different sources (edge addition, bond upgrading).** This hybrid design combines DFS's memory efficiency for deep tree enumeration with BFS's natural deduplication rhythm for layered construction.

---

## Data Structures

The project uses four progressively richer graph representations, each active at a different stage of the pipeline.

### 1. Canonical String (Phase 2 — Tree Enumeration)

**Format**: `"C(...)"` where nested parentheses encode the rooted-tree branching structure. Sub-branches are sorted lexicographically.

```
"C"                     → methane (single carbon)
"C(C)"                  → ethane (C—C)
"C(C,C)"                → propane (C—C—C, branched notation)
"C(C,C(C))"             → isobutane (central C with two CH₃ and one CH₂CH₃)
"C(C(C)C(C)C(C(C)))"    → complex heptane isomer
```

**Construction rule** (Algorithm 2, line 5):
```
canon = "C(" + join(",", sort(canon_of_child₁, ..., canon_of_childₖ)) + ")"
```
Sorting child strings eliminates permutation redundancy — `C(C, C(C))` and `C(C(C), C)` both sort to `C(C, C(C))`.

**Properties**:
- Unique for each rooted-tree topology (up to isomorphism)
- Comparable lexicographically (enables centroid dedup via `min(canon, flipped_canon)`)
- Parsable: `parse_substrings(canon)` extracts child substrings using depth-counter matching

### 2. Adjacency List (Phase 2 → 3 Bridge)

**Format**: `Dict[int, List[int]]` — node IDs (0-indexed) map to neighbor lists.

```
canon = "C(C, C(C))"     →    adj = {0: [1, 2], 1: [0], 2: [0, 3], 3: [2]}
                                  ──┬──  ──┬──  ──┬────────  ──┬──
                                   C₀     C₁     C₂          C₃
```

**Construction**: `canon_to_adjacency(canon)` does a recursive DFS parse — each `"C(...)"` allocates a new node ID, then recursively parses child substrings with parent pointers. The function tracks a global `node_counter` to assign sequential IDs.

**Standardization for dedup**: the adjacency list is normalized to a canonical tuple form:
```python
standardized = tuple(sorted((node, tuple(sorted(nbrs))) for node, nbrs in adj.items()))
```
This removes node-labeling variance — the same molecule with different atom numbering yields the same tuple.

### 3. Attributed Molecular Graph (Phase 3 — Bond Insertion)

**Format**: `networkx.Graph` with two edge attributes:

| Attribute | Values | Meaning |
|---|---|---|
| `bond_type` | `"single"`, `"double"`, `"triple"` | Bond order between the two carbons |
| `label` (node) | `"C"` | Atom type (always carbon in this project) |

```
nx.Graph with edges:
  (0,1) bond_type='single'
  (1,2) bond_type='double'       ← C₁=C₂  (upgraded from single)
  (2,3) bond_type='triple'       ← C₂≡C₃  (upgraded from single)
```

**Bond upgrading**: `_upgrade_edge_to_double(G, u, v)` creates a deep copy of the graph, replacing the target edge's `bond_type` from `'single'` to `'double'` (analogous for `'triple'`). All other edges retain their original types.

**Chemical validation** (`_validate_molecule(G)`) checks per-node constraints:

| Constraint | Condition | Rationale |
|---|---|---|
| Bond load ≤ 4 | sum(bond_weights) ≤ 4 | Carbon tetravalence |
| sp carbon degree ≤ 2 | n_triple>0 ⇒ degree≤2 | sp ≡ linear, max 2 σ-bonds |
| No cumulative diene | n_double≥2 ⇒ degree≤2 | Avoids unstable =C= geometry |
| sp carbon exclusive | n_triple>0 ⇒ n_double=0 | sp cannot co-exist with sp² |

where `_bond_load(node)` sums weights: single=1, double=2, triple=3.

### 4. Deduplication Buckets (Phase 4)

**Format**: `Dict[str, List[nx.Graph]]` — WL hash string → list of graphs with the same hash.

```
buckets = {
  "a3f2b1...": [G₁, G₃],
  "7d9e0c...": [G₂],
  "f1a4b8...": [G₄, G₅, G₇],
}
```

**Two-stage pipeline**:

| Stage | Operation | Cost |
|---|---|---|
| 1. WL hashing | `nx.weisfeiler_lehman_graph_hash(G, edge_attr='bond_type', node_attr='label')` | O(d·m) per graph |
| 2. Exact isomorphism | `nx.is_isomorphic(G, existing, node_match=..., edge_match=...)` | VF2 algorithm, worst-case exponential |

Stage 1 partitions graphs into buckets — graphs in different buckets are guaranteed non-isomorphic (WL hash is an isomorphism invariant). Stage 2 only runs inside each bucket, drastically reducing the number of expensive VF2 calls.

**Auxiliary fast-filter**: `_degree_signature(G)` — sorted degree sequence — is checked before VF2. Non-matching degree sequences guarantee non-isomorphism with O(n log n) cost.

---

## Algorithm Descriptions

### Algorithm 2: Rooted Tree Enumeration via Integer Partition

The function `generate_rooted_with_info(n, max_branches)` enumerates all rooted trees of n nodes with bounded branching. It operates in two stages:

**Stage 1 — Integer Partition (`find_parts`)**:
Enumerate all non-increasing sequences (s₁ ≥ s₂ ≥ ... ≥ sₖ) such that:
- Σ sᵢ = n−1 (remaining carbons after root)
- k ≤ max_branches (root's degree limit)
- Each sᵢ ≥ 1

The non-increasing constraint prevents duplicate partitions: (3,1) is enumerated, (1,3) is skipped.

**Stage 2 — Cartesian Product + Canonical Assembly**:
For each partition (s₁, ..., sₖ):
1. Recursively compute all trees of size sᵢ with max_branches−1 for each i
2. Take the Cartesian product across all sᵢ
3. For each combination, sort child canonical strings and join

**Memoization**: Results for `(n, max_branches)` are cached globally. The subtree call uses `max_branches=3` (not 4) because the subtree root already uses one bond connecting to its parent node.

**Time complexity**: proportional to the number of rooted trees generated, with O(n²) overhead per tree for string assembly and sorting.

### Algorithm 2a: Centroid-Based Rooted→Free Tree Deduplication

Given a rooted tree with subtree sizes, determine whether it should be kept as a unique representative of the corresponding free (unrooted) tree.

**Centroid theorem**: Any n-node tree has either one centroid (unique) or two centroids (adjacent). A node v is a centroid if and only if all subtrees rooted at v's neighbors have size ≤ ⌊n/2⌋.

The algorithm works by examining the maximum subtree size max_s of the rooted tree's root:

| max_s vs ⌊n/2⌋ | Centroid status | Action |
|---|---|---|
| max_s < ⌊n/2⌋ | Root is the unique centroid | Keep directly |
| max_s > ⌊n/2⌋ | Root is not a centroid | Discard (centroid perspective already captured) |
| max_s = ⌊n/2⌋ | Bicentroid case (n even) | Keep smaller of canonical(t) vs canonical(flipped t) |

**Flip operation**: When max_s = ⌊n/2⌋, the root r₁ has a child r₂ whose subtree is exactly size ⌊n/2⌋. Re-rooting at r₂ produces t' — the same free tree viewed from the other centroid. Taking `min(t, t')` breaks the tie arbitrarily but deterministically.

**After this step**: `Deduplicate(F)` performs a secondary check via standardized adjacency lists to catch any remaining isomorphic duplicates.

### Algorithm 3: Multi-Ring Construction via Iterative Edge Addition

**Starting point**: All alkane free trees (from Algorithm 2 + 2a).

**Per round** (r rounds total):
1. For each graph G in the current layer, enumerate all pairs (u, v) where:
   - u and v are not adjacent (no existing edge)
   - deg(u) ≤ 3 and deg(v) ≤ 3 (adding edge keeps degree ≤ 4)
2. Create G' = G ∪ {(u,v)} for each valid pair
3. WL-hash-bucket deduplicate the entire layer

**Correctness**: Any connected r-ring graph on n nodes, when any r edges are removed such that the result is still connected, yields an alkane tree on n nodes. The reverse process — adding r edges to all alkane trees — therefore generates all r-ring graphs.

**Pruning**: The degree≤3 constraint prevents carbon atoms from exceeding tetravalence after edge addition. The per-round dedup prevents the same r-ring graph from being generated multiple times from different (r−1)-ring precursors.

### Algorithm 3a/3b: Bond Insertion and Chemical Validation

**Bond insertion** is the core operation that converts C—C (single) to C=C (double) or C≡C (triple). It consists of three stages:

**Pre-check (`_can_insert_double` / `_can_insert_triple`)**:
Lightweight check before full graph copy:
```python
_bond_load(node) + bond_delta ≤ 4        # hard tetravalence limit
# For double: no cumulative =C= (degree > 2 with ≥2 double bonds)
# For triple: degree ≤ 2 and no other multiple bonds
```

**Deep copy + upgrade**: `_upgrade_edge_to_double(G, u, v)` creates a full graph copy (all nodes, all edges), then sets `bond_type='double'` on the target edge.

**Post-validation (`_validate_molecule`)**:
Full per-node constraint check on the upgraded graph, enforcing bond-load ≤ 4, sp carbon degree ≤ 2, cumulative diene restriction, and sp/sp² exclusivity.

**Triple-bonds-first ordering** (Algorithm 3b): Triple bonds have stricter constraints (degree ≤ 2 per end, no co-existing multiple bonds), resulting in fewer valid insertion sites. Inserting triples first maximizes the search space — if doubles were inserted first, they could occupy positions that triple bonds need, causing some valid structures to be missed.

### Algorithm 4: Two-Stage WL + Isomorphism Deduplication

**Stage 1 — WL Hashing**:
The Weisfeiler-Lehman graph hash computes a string invariant under isomorphism by iteratively refining node colors based on neighborhood signatures:
1. Initialize: each carbon node gets color = 'C', edges colored by bond_type
2. Iterate (default 3 rounds): each node's new color = hash(old_color, multiset of neighbor colors with edge colors)
3. Final hash = hash of the multiset of final node colors

**Stage 2 — Exact Isomorphism (VF2)**:
Within each hash bucket, perform pairwise isomorphism checks using networkx's VF2 algorithm:
```python
nx.is_isomorphic(
    G1, G2,
    node_match=lambda n1,n2: n1.get('label') == n2.get('label'),
    edge_match=lambda e1,e2: e1.get('bond_type') == e2.get('bond_type')
)
```
The `node_match` and `edge_match` comparators ensure both atom types and bond orders are respected during the match.

**Fast-filter optimization**: Before VF2, compare `_degree_signature(G)` — the sorted degree sequence. If two graphs differ in degree distribution, they cannot be isomorphic; this O(n log n) check eliminates most non-matches before the potentially exponential VF2.
