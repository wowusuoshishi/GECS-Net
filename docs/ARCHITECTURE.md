# GECS-Net Architecture Details

This document describes the detailed architecture of GECS-Net, matching the IEEE SPL 2026 paper.

## Model Overview

GECS-Net is a 3-stage iterative network for two-view correspondence pruning and essential matrix estimation.

### Input
- N = 2000 initial SIFT correspondences
- Each correspondence: `(x1, y1, x2, y2)` — normalized to [-1, 1]

### Output
- Per-correspondence inlier logits (at each stage)
- Essential matrix E (3×3) via weighted 8-point algorithm

## Stage Architecture

Each of the 3 stages (1 init + 2 refinement) contains:

```
Input (B, C_in, N, 1)
  |
  v
[Conv1]  C_in → 128
  |
  v
[CSGFE cross-stage fusion]  (refinement stages only)
  |-- Cosine similarity k-NN graphs (k=3)
  |-- Annular convolution aggregation
  |-- SE historical calibration (stage 2 only)
  |
  v
[PointCN] → [GEA #1] → [PointCN] ×2
  |           |-- RoPE: Q,K rotation by learned position encoding
  |           |-- MB-FFN: local + GAP + GMP additive fusion
  |
  v
[GNN]  k=3, cosine similarity
  |-- get_graph_feature: [x, x-x_neighbor]
  |-- Annular convolution
  |-- AFF: attention feature fusion
  |
  v
[PointCN] → [GEA #2] → [PointCN]
  |
  v
[Output] → logits, E_hat, Z3 features
```

## GEA: Geometric Embedding Attention

```
Feature (B, 128, N, 1)
  |
  ├──> Q = q_proj(feature) → (B, 128, N)
  ├──> K = k_proj(feature) → (B, 128, N)
  └──> V = v_proj(feature) → (B, 128, N)
  |
  v
[RoPE]
  |-- coord_encoder_1: (x1,y1) → (B, 64, N)
  |-- coord_encoder_2: (x2,y2) → (B, 64, N)
  |-- θ = encoder_1 + encoder_2 → (B, N, 64)
  |-- Q_rot, K_rot = rotate(Q,K) by cos(θ), sin(θ)
  |
  v
[Attention]  Channel-wise (128×128)
  |-- attn = softmax(Q_rot @ K_rot^T / √d)
  |-- out = attn @ V
  |
  v
[+ Residual] → [LayerNorm] → [MB-FFN] → [+ Residual] → [LayerNorm]
  |
  v
Output (B, 128, N, 1)
```

## RoPE: Rotary Position Embedding

Unlike standard positional encodings that add position bias to attention scores, RoPE applies multiplicative rotation to Q and K vectors:

```
Q̃_i = Q_i ⊙ R_Θ(Ĉ_i)
K̃_j = K_j ⊙ R_Θ(Ĉ_j)

(Q̃_i · K̃_j) = Q_i^T R(Ĉ_i - Ĉ_j) K_j
```

This ensures the attention dot product depends on relative geometric displacement `(u_i-u_j, v_i-v_j)`, matching the paper's design.

Implementation:
1. Split feature channels into D/2 = 64 consecutive pairs
2. Learn rotation angles θ_i for each pair via MLP(coords)
3. Apply 2D rotation to each (x_{2i}, x_{2i+1}) pair

## CSGFE: Cross-Stage Graph Feature Extraction

### Within-stage CSGFE (refinement stages)

```
Prev stage Z1, Z2, Z3  →  project to 64-dim each
  |
  v
[Cosine k-NN graph] (k=3) for each feature
  |
  v
[Concatenate graphs]  curr + prev + prev2 (3 sources × k=3)
  |
  v
[Annular Convolution]  → aggregate contextual info
  |-- Stage 1: 3 sources → 2 annular convs
  |-- Stage 2: 3 sources + SE(init) → 3 annular convs
  |
  v
[MLP] → [Fuse with current features]  concat → PointCN(256→128)
```

### Final CSGFE (network level)

```
Stage 0 Z3  ─┐
Stage 1 Z3  ─┼──> Cosine k-NN (k=3) for each
Stage 2 Z3  ─┘
  |
  v
[Concatenate] → [Annular Conv] → [MLP] → [+ Residual from Stage 3]
```

## MB-FFN: Multi-Branch FFN

```
Z_out = Φ(LN(F_out)) + Σ_{Ψ∈{GAP,GMP}} Φ(Ψ(LN(F_out)))

where Φ = Conv(GELU(BN(Conv(·))))
```

- **Local branch**: Conv processing of normalized features
- **GAP branch**: Global average pooling → Conv processing
- **GMP branch**: Global max pooling → Conv processing
- **Fusion**: Simple additive sum (no sigmoid gating)

## Training Details

- **Optimizer**: Adam, lr=3e-4
- **LR Schedule**: WarmupMultiStepLR (100k warmup, milestones at 200k and 400k, γ=0.4)
- **Batch size**: 32
- **Iterations**: 500k
- **Hardware**: Single RTX 3090
- **Loss**: L = L_cls + 0.5 · L_geo

## Differences from Earlier MGCA Implementation

| Aspect | Earlier MGCA | GECS-Net (this code) |
|--------|-------------|---------------------|
| Position encoding | CPT additive bias | RoPE multiplicative rotation |
| Graph distance | Euclidean | Cosine similarity |
| k-NN neighbors | k=6 | k=3 |
| MB-FFN fusion | Sigmoid gating | Additive summation |
| CS fusion | Simple concat (CSMGC) | SE calibration + two-step (CSGFE) |
| OANet block | DP_OA_DOP present | Removed (not in paper) |
| Module names | CGA, CSMGC, MGCANet | GEA, CSGFE, GECSNet |
