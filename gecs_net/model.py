"""
GECS-Net: Geometric Embedding and Cross-Stage Network
for Two-View Correspondence Learning.

Architecture (matching IEEE SPL 2026 paper):
  - GEA (Geometric Embedding Attention): RoPE + MB-FFN
  - CSGFE (Cross-Stage Graph Feature Extraction): SE + cosine k-NN + annular conv
  - 3-stage iterative refinement with weighted 8-point algorithm

Reference:
  Lin et al., "GECS-Net: Two-View Correspondence Learning via Geometric
  Embedding and Cross-Stage Consensus", IEEE SPL, 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d

from .loss import batch_episym


# ==============================================================================
# Basic Building Blocks
# ==============================================================================

class SE(nn.Module):
    """Squeeze-and-Excitation module for channel-wise feature recalibration."""
    def __init__(self, num_channels, reduction_ratio=2):
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        x = self.in0(input_tensor)
        x = self.bn0(x)
        x = self.relu(x)
        input_tensor = self.conv0(x)
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class MBSE(nn.Module):
    """Multi-Branch Squeeze-and-Excitation block for feature enhancement."""
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, bottleneck_width=64):
        super(MBSE, self).__init__()
        SE_channel = int(planes * (bottleneck_width / 64.))
        self.shot_cut = None
        if planes * 2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes * 2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, SE_channel, kernel_size=1, bias=True)
        self.in1 = nn.InstanceNorm2d(inplanes, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = SE(SE_channel)
        self.conv3 = nn.Conv2d(SE_channel, planes * 2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(SE_channel, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(SE_channel)
        self.conv_branch1_1 = nn.Sequential(
            nn.InstanceNorm2d(inplanes, eps=1e-3),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, planes * 2, kernel_size=1),
        )
        self.conv_merge = nn.Sequential(
            nn.InstanceNorm2d(planes * 2, eps=1e-3),
            nn.BatchNorm2d(planes * 2),
            nn.GELU(),
            nn.Conv2d(planes * 2, planes * 2, kernel_size=1),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.in1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.in3(out)
        out = self.bn3(out)
        out = self.conv3(out)
        branch_out = self.conv_branch1_1(x)
        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out = out + branch_out + residual
        out = self.conv_merge(out)
        return out


class PointCN(nn.Module):
    """Point Cloud Network residual block with dual normalization."""
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class MLPs(nn.Module):
    """Simple MLP with dual normalization."""
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class AFF(nn.Module):
    """Attention Feature Fusion: local + global attention with sigmoid gating."""
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# ==============================================================================
# Graph Construction Utilities
# ==============================================================================

def knn(x, k):
    """k-NN using negative squared Euclidean distance."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]


def knn_cosine(x, k):
    """k-NN using cosine similarity distance.

    Following the GECS-Net paper: graph construction uses cosine similarity
    in the feature space to dynamically capture local geometric correlations.
    """
    # Normalize features for cosine similarity
    x_norm = F.normalize(x, p=2, dim=1)
    # Cosine similarity matrix
    cosine_sim = torch.matmul(x_norm.transpose(2, 1), x_norm)
    # Convert to distance (higher similarity = smaller distance)
    cosine_dist = 1.0 - cosine_sim
    idx = cosine_dist.topk(k=k, dim=-1, largest=False)[1]
    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None, use_cosine=False):
    """Build edge features [x, x - x_neighbor] via k-NN graph.

    Args:
        x:       (B, C, N, 1) feature tensor
        k:       number of neighbors
        idx:     optional precomputed neighbor indices
        use_cosine: if True, use cosine similarity for k-NN (GECS-Net paper)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if use_cosine:
            idx_out = knn_cosine(x, k=k)
        else:
            idx_out = knn(x, k=k)
    else:
        idx_out = idx

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def down_sampling(x, indices, features=None):
    """Downsample by selecting top 50% points based on sorted indices."""
    B, _, N, _ = x.size()
    indices = indices[:, :int(N * 0.5)]
    indices = indices.view(B, 1, -1, 1)
    with torch.no_grad():
        x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
    feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
    return x_out, feature_out


# ==============================================================================
# RoPE: Rotary Position Embedding (Paper Section III-B)
# ==============================================================================

class RoPE(nn.Module):
    """Rotary Position Embedding for Geometry-Aware Attention.

    Encodes relative spatial constraints directly into the attention mechanism
    via position-dependent feature rotations.

    As described in the paper:
    - Splits coordinate feature into two orthogonal D/2-dimensional subspaces
      to independently encode horizontal and vertical coordinates.
    - The relative geometric displacement (u_i - u_j, v_i - v_j) is inherently
      encoded through a rotation matrix with learned frequencies.
    - Q_tilde = Q * R_Theta(C_hat),  K_tilde = K * R_Theta(C_hat)
    - Attention = Softmax(Q_tilde * K_tilde^T / sqrt(d)) * V

    This REPLACES the CPT (Context Position Attention) additive bias approach.
    """

    def __init__(self, channels):
        super(RoPE, self).__init__()
        self.channels = channels

        # MLP to encode coordinates (x1,y1,x2,y2) into rotation features
        # Channels 0..D/2-1 encode horizontal (u) direction
        # Channels D/2..D-1 encode vertical (v) direction
        self.coord_encoder_1 = nn.Sequential(
            nn.Conv2d(2, channels // 2, kernel_size=1),
            nn.InstanceNorm2d(channels // 2),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
        )
        self.coord_encoder_2 = nn.Sequential(
            nn.Conv2d(2, channels // 2, kernel_size=1),
            nn.InstanceNorm2d(channels // 2),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
        )

    def _apply_rope(self, x, cos_theta, sin_theta):
        """Apply rotary position embedding to tensor x.

        Groups consecutive pairs of feature channels and rotates each pair
        by the learned position-dependent angle.

        x'_{2i}   = x_{2i} * cos(θ_i) - x_{2i+1} * sin(θ_i)
        x'_{2i+1} = x_{2i+1} * cos(θ_i) + x_{2i} * sin(θ_i)

        Args:
            x:         (B, C, N) feature tensor (C channels, N points)
            cos_theta: (B, N, C/2) cosine of learned rotation angles
            sin_theta: (B, N, C/2) sine of learned rotation angles

        Returns:
            x_rot: (B, C, N) rotated feature tensor
        """
        B, C, N = x.shape
        # Pair consecutive channels: (B, C/2, 2, N)
        x_pair = x.reshape(B, C // 2, 2, N)

        # Transpose to (B, C/2, N, 2) for proper broadcasting
        x_pair = x_pair.permute(0, 1, 3, 2)  # (B, C/2, N, 2)

        # cos/sin: (B, N, C/2) → (B, C/2, N, 1)
        cos = cos_theta.transpose(1, 2).unsqueeze(-1)  # (B, C/2, N, 1)
        sin = sin_theta.transpose(1, 2).unsqueeze(-1)  # (B, C/2, N, 1)

        # Split: (B, C/2, N, 1) each
        x0 = x_pair[..., 0:1]  # first of each channel pair
        x1 = x_pair[..., 1:2]  # second of each channel pair

        # Apply 2D rotation: (B, C/2, N, 1)
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x1 * cos + x0 * sin

        # Reassemble: (B, C/2, N, 2) → (B, C/2, 2, N) → (B, C, N)
        x_rot = torch.cat([x0_rot, x1_rot], dim=-1)  # (B, C/2, N, 2)
        x_rot = x_rot.permute(0, 1, 3, 2)  # (B, C/2, 2, N)
        return x_rot.reshape(B, C, N)

    def forward(self, q, k, coords):
        """Apply RoPE rotation to Q and K features.

        Encodes spatial position as multiplicative rotation of Q and K,
        so that attention (Q_rot @ K_rot^T) naturally depends on relative
        geometric displacement between correspondences.

        Args:
            q:      (B, C, N) query features (C=128 channels, N points)
            k:      (B, C, N) key features
            coords: (B, 4, N, 1) original coordinates (x1, y1, x2, y2)

        Returns:
            q_rot:  (B, C, N) rotated query
            k_rot:  (B, C, N) rotated key
        """
        # Extract coordinates for each view
        coord_1 = coords[:, :2, :, :]   # (B, 2, N, 1) -- image 1 (x1, y1)
        coord_2 = coords[:, 2:4, :, :]  # (B, 2, N, 1) -- image 2 (x2, y2)

        # Encode each view's coordinates to C/2 dimensional rotation features
        feat_1 = self.coord_encoder_1(coord_1)  # (B, C/2, N, 1)
        feat_2 = self.coord_encoder_2(coord_2)  # (B, C/2, N, 1)

        # Combined rotation feature: learned angle per channel-pair per point
        theta = (feat_1 + feat_2).squeeze(-1).transpose(1, 2)  # (B, N, C/2)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Apply channel-pair rotations to Q and K
        q_rot = self._apply_rope(q, cos_theta, sin_theta)
        k_rot = self._apply_rope(k, cos_theta, sin_theta)

        return q_rot, k_rot


# ==============================================================================
# GEA: Geometric Embedding Attention (Paper Section III-B)
# ==============================================================================

class MBFFN(nn.Module):
    """Multi-Branch Feed-Forward Network.

    Paper formula (Equation 4):
      Z_out = Phi(LN(F_out)) + sum_{Psi in {GAP, GMP}} Phi(Psi(LN(F_out)))

    where Phi = Conv(GELU(BN(Conv(.))))

    NOTE: The paper uses ADDITIVE fusion of branches (simple summation),
    unlike the earlier sigmoid-gated variant. This matches the paper's
    description of synthesizing local details with global statistics.
    """

    def __init__(self, in_channels, out_channels, reduction=4):
        super(MBFFN, self).__init__()
        inter_channels = int(in_channels // reduction)

        # Phi for local branch: Conv(GELU(BN(Conv(.))))
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

        # Phi for global average pooling branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

        # Phi for global max pooling branch
        self.gmp_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # F_norm = LN(F_out) — normalization is applied before branches via LayerNorm in GEA
        local_feat = self.local_branch(x)
        gap_feat = self.gap_branch(x)
        gmp_feat = self.gmp_branch(x)

        # Additive fusion: sum all branches (matching paper Equation 4)
        out = local_feat + gap_feat + gmp_feat
        out = self.conv_out(out)
        out = out + x  # Residual connection
        return out


class GEA(nn.Module):
    """Geometric Embedding Attention module.

    Combines Rotary Positional Embeddings (RoPE) with a Multi-Branch
    Feed-Forward Network (MB-FFN) to explicitly encode relative spatial
    constraints into the attention mechanism.

    Structure:
      Input → RoPE Attention → + Residual → LayerNorm → MBFFN → + Residual → LayerNorm → Output
    """

    def __init__(self, channels):
        super(GEA, self).__init__()

        # Q, K, V projections
        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # RoPE: replaces CPT additive coordinate bias with multiplicative rotation
        self.rope = RoPE(channels)

        self.temperature = torch.sqrt(torch.tensor(channels, dtype=torch.float32))

        # MB-FFN with additive fusion (matching paper)
        self.mbffn = MBFFN(channels, channels)

        # Layer normalization
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, feature, position_feature):
        """
        Args:
            feature:          (B, C, N, 1) input features
            position_feature: (B, 4, N, 1) original coordinate data

        Returns:
            out: (B, C, N, 1) geometry-aware features
        """
        # Project to Q, K, V
        q = self.q_proj(feature).squeeze(3)  # (B, C, N)
        k = self.k_proj(feature).squeeze(3)  # (B, C, N)
        v = self.v_proj(feature).squeeze(3)  # (B, C, N)

        # Apply RoPE: multiplicative rotation of Q and K by position
        q_rot, k_rot = self.rope(q, k, position_feature)

        # Geometry-aware attention: Q_rot @ K_rot^T / sqrt(d)
        attn = torch.matmul(q_rot / self.temperature, k_rot.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v).unsqueeze(3)  # (B, C, N, 1)

        # Residual connection + LayerNorm 1
        attn_out = attn_out + feature
        attn_out_ln = attn_out.squeeze(3).transpose(-1, -2)
        attn_out_ln = self.norm1(attn_out_ln)
        attn_out_ln = attn_out_ln.transpose(-1, -2).unsqueeze(3)

        # MB-FFN
        ff_out = self.mbffn(attn_out_ln)

        # LayerNorm 2 + residual
        ff_out_ln = ff_out.squeeze(3).transpose(-1, -2)
        ff_out_ln = self.norm2(ff_out_ln)
        ff_out_ln = ff_out_ln.transpose(-1, -2).unsqueeze(3)

        out = attn_out_ln + ff_out_ln

        return out


# ==============================================================================
# GNN: Graph Neural Network (with cosine similarity, k=3 per paper)
# ==============================================================================

class GNN(nn.Module):
    """Graph Neural Network for local feature aggregation.

    Paper specification:
    - k = 3 nearest neighbors (paper default)
    - Cosine similarity for graph construction
    - Annular convolution for contextual aggregation
    """

    def __init__(self, knn_num=3, in_channel=128):
        super(GNN, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        self.mlp1 = MLPs(2 * in_channel, 2 * in_channel)
        self.change1 = MLPs(2 * in_channel, in_channel)
        self.change2 = MLPs(2 * in_channel, in_channel)
        self.aff = AFF(in_channel, 4)

        # Annular convolution: supports k=3 (paper default) and k=6
        assert self.knn_num in [3, 6, 9], f"knn_num must be 3, 6, or 9, got {self.knn_num}"
        if self.knn_num == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 1)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 1)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )
        elif self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 2)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )
        else:  # k=9
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape
        # Cosine similarity graph construction (paper specification)
        out = get_graph_feature(features, k=self.knn_num, use_cosine=True)
        out_an = self.conv(out)
        out_an = self.change1(out_an)
        out_max = self.mlp1(out)
        out_max = out_max.max(dim=-1, keepdim=False)[0]
        out_max = out_max.unsqueeze(3)
        out_max = self.change2(out_max)
        out = self.aff(out_max, out_an)
        return out


# ==============================================================================
# CSGFE: Cross-Stage Graph Feature Extraction (Paper Section III-C)
# ==============================================================================

class CSGFE(nn.Module):
    """Cross-Stage Graph Feature Extraction module.

    Paper description:
    1. Cross-Stage Feature Extraction: aggregates Z1, Z2, Z3 from previous
       stage via residual fusion.
    2. SE module: dynamically recalibrates historical features from prior
       stages to yield fused consensus Z'.
    3. Graph Consensus Construction: k-NN using cosine similarity.
    4. Annular Convolution Aggregation: contextual refinement via learned
       annular kernels.

    This REPLACES CSMGC (Cross-Stage Multi-Graph Consensus) which used
    simple concatenation without SE calibration or cosine similarity.
    """

    def __init__(self, in_channels, out_channels, num_stages=3):
        nn.Module.__init__(self)
        self.num_stages = num_stages

        # Projection layers for cross-stage feature fusion
        self.proj_z1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.proj_z2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.proj_z3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        # Residual fusion: conv(concat(Z1, Z2, Z3)) → in_channels
        self.res_fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2 * 3, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # SE module for historical feature calibration (paper Equation 6)
        self.se_history = SE(in_channels)

        # Projection for historical features
        self.proj_history = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.InstanceNorm2d(in_channels // 2),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
        )

        # MLP for projecting multi-level graph into aligned features
        self.graph_mlp = MLPs(in_channels * 2, in_channels)

        # Annular convolution for graph consensus aggregation
        self.annular_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 3), stride=(1, 3)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 2)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, (1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Output projection
        self.out_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, stage_features, historical_features=None):
        """Cross-stage graph feature extraction.

        Args:
            stage_features: list of [Z1, Z2, Z3] from the previous stage
                           each of shape (B, C, N, 1)
            historical_features: optional list of Z3 features from all prior
                                stages for SE-based calibration

        Returns:
            out: (B, C, N, 1) cross-stage consensus features
        """
        Z1, Z2, Z3 = stage_features

        # Step 1: Cross-Stage Feature Extraction (paper Equation 5)
        # Z = Res(concat(Z1^(S-1), Z2^(S-1), Z3^(S-1)))
        z1_p = self.proj_z1(Z1)
        z2_p = self.proj_z2(Z2)
        z3_p = self.proj_z3(Z3)
        Z = self.res_fusion(torch.cat([z1_p, z2_p, z3_p], dim=1))

        # Step 2: SE-based historical calibration (paper Equation 6)
        # Z' = SE(concat(Z3^(1), Z3^(2), ..., Z3^(S-2)))
        if historical_features is not None and len(historical_features) > 0:
            if len(historical_features) == 1:
                hist_cat = historical_features[0]
            else:
                # Concatenate features from all prior stages
                h_proj = [self.proj_history(h) for h in historical_features]
                # Average pooling across historical stages
                hist_cat = torch.stack(h_proj, dim=0).mean(dim=0)
            Z_prime = self.se_history(hist_cat)

            # Fuse Z (current) with Z' (historical)
            Z = Z + self.proj_history(Z_prime).repeat(1, 2, 1, 1)
            # Adjust channels back
            Z = self.out_proj(torch.cat([Z, Z_prime], dim=1))

            # Graph construction with cosine similarity on Z
            G = get_graph_feature(Z, k=3, use_cosine=True)
            G_prime = get_graph_feature(Z_prime, k=3, use_cosine=True)

            # Unified graph (paper): G_f = concat(G, G')
            G_f = torch.cat([G, G_prime], dim=-1)

            # Annular convolution aggregation
            out = self.graph_mlp(self.annular_conv(G_f))
            out = out + Z3  # Residual from comprehensive features
        else:
            # No historical features (first stage): direct graph consensus on Z3
            G = get_graph_feature(Z3, k=3, use_cosine=True)
            out = self.graph_mlp(G)
            out = out + Z3

        return out


# ==============================================================================
# Sub-Network: Single Iterative Refinement Stage
# ==============================================================================

class GECSNetStage(nn.Module):
    """Single iterative refinement stage of GECS-Net.

    Each stage contains:
    - Cross-stage feature fusion (CSGFE)
    - Two GEA modules (RoPE + MBFFN)
    - Graph Neural Network (cosine k-NN, k=3)
    - Output: logits, E_hat, features for next stage

    NOTE: No DP_OA_DOP_block (OANet-like) — this module is not described
    in the GECS-Net paper and has been removed for paper fidelity.
    """

    def __init__(self, stage_idx, net_channels, input_channel, depth, is_init=False, knn_num=3):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        self.is_init = is_init
        self.stage_idx = stage_idx

        print(f'  Stage {stage_idx}: channels={channels}, layers={depth}, k-NN={knn_num}'
              + (' (init)' if is_init else ''))

        # Initial projection
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        # CSGFE components (only for refinement stages, not init)
        if not is_init:
            # Projections for extracting Z1/Z2/Z3 from prev stage
            self.proj_z1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
            self.proj_z2 = nn.Conv2d(channels, channels // 2, kernel_size=1)
            self.proj_z3 = nn.Conv2d(channels, channels // 2, kernel_size=1)

            # Cross-stage graph feature extraction
            self.cs_graph_mlp = MLPs(channels, channels)

            # Channel reducer for fusion: x1_1(128) + dgcnn(128) = 256 → 128
            self.fusion_reduce = PointCN(256, channels)

            # SE for init_out historical calibration (used in stage 2, like original i==1)
            if stage_idx == 2:
                self.se_hist = SE(channels // 2)
                self.proj_init_out = nn.Conv2d(channels, channels // 2, kernel_size=1)

            # Annular convolution for cross-stage graph consensus
            # Stage 1: 3 graph sources (curr + prev_out + prev2_out) × k=3 = 9
            # Stage 2: 4 graph sources (+ init_out SE) × k=3 = 12
            if stage_idx == 1:
                # 3 sources → 9 last dim: (1,3) stride(1,3) → 3, (1,3) → 1
                self.cs_annular = nn.Sequential(
                    nn.Conv2d(channels, channels, (1, 3), stride=(1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, (1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            else:
                # 4 sources → 12 last dim: (1,3) stride(1,3) → 4, (1,3) → 2, (1,2) → 1
                self.cs_annular = nn.Sequential(
                    nn.Conv2d(channels, channels, (1, 3), stride=(1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, (1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, (1, 2)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )

        # PointCN blocks
        self.pcn1 = PointCN(channels)
        self.pcn2 = PointCN(channels)
        self.pcn3 = PointCN(channels)

        # GEA modules (RoPE-based, replacing CGA)
        self.gea1 = GEA(channels)
        self.gea2 = GEA(channels)

        # GNN with cosine similarity, k=3 (matching paper)
        self.gnn = GNN(knn_num, channels)

        # Second half PointCN blocks
        self.pcn4 = PointCN(channels)
        self.pcn5 = PointCN(channels)
        self.pcn6 = PointCN(channels)

        # Output heads
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.linear1 = nn.Conv2d(channels, 2, kernel_size=1)

    def _cross_stage_fusion(self, x1_1, x_prev, x_prev2=None, init_out=None):
        """Cross-Stage Graph Feature Extraction (internal CSGFE).

        Fuses features from previous stages using cosine-similarity k-NN
        graphs and annular convolution (paper Section III-C).
        """
        # Project features to lower dimension for graph construction
        curr_64 = self.proj_z3(x1_1)
        prev_64 = self.proj_z1(x_prev)
        prev2_64 = self.proj_z2(x_prev2) if x_prev2 is not None else None

        # Build k-NN graphs using cosine similarity
        curr_graph = get_graph_feature(curr_64, k=3, use_cosine=True)
        prev_graph = get_graph_feature(prev_64, k=3, use_cosine=True)

        if prev2_64 is not None:
            prev2_graph = get_graph_feature(prev2_64, k=3, use_cosine=True)

        # Include init_out features for stage 2 (SE-calibrated, matching original i==1)
        if init_out is not None and self.stage_idx == 2:
            init_64 = self.proj_init_out(init_out)
            init_64 = self.se_hist(init_64)
            init_graph = get_graph_feature(init_64, k=3, use_cosine=True)
            combine_graph = torch.cat([curr_graph, prev_graph, prev2_graph, init_graph], dim=-1)
        elif prev2_64 is not None:
            combine_graph = torch.cat([curr_graph, prev_graph, prev2_graph], dim=-1)
        else:
            combine_graph = torch.cat([curr_graph, prev_graph], dim=-1)

        # Annular convolution + MLP for consensus
        dgcnn = self.cs_annular(combine_graph)
        dgcnn = self.cs_graph_mlp(dgcnn)

        # Fuse with current features
        x1_1 = torch.cat([x1_1, dgcnn], dim=1)
        x1_1 = self.fusion_reduce(x1_1)
        return x1_1

    def forward(self, data, xs, x_prev=None, x_prev2=None, init_out=None):
        """
        Args:
            data:     (B, C_in, N, 1) input features / coordinates
            xs:       (B, 1, N, 4) original correspondences
            x_prev:   previous stage output features (or None for init)
            x_prev2:  two stages ago output (or None)
            init_out: initial stage output (for SE calibration in stage 1)

        Returns:
            logits:   (B, N) classification logits
            e_hat:    (B, 9) estimated essential matrix
            residual: (B, 1, N, 1) epipolar residual
            out:      (B, C, N, 1) output features (Z3 for next stage)
            z1_feat:  (B, C, N, 1) Z1 features (after GEA1, for CSGFE)
        """
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)

        # Cross-stage fusion (CSGFE) for refinement stages
        if x_prev is not None:
            x1_1 = self._cross_stage_fusion(x1_1, x_prev, x_prev2, init_out)

        # GEA1 + PointCN blocks (geometric embedding)
        x1_1 = self.pcn1(x1_1)
        x1_1 = self.gea1(x1_1, data)       # RoPE-based attention
        z1_feat = x1_1                       # Z1: global features
        x1_1 = self.pcn2(x1_1)
        x1_1 = self.pcn3(x1_1)
        z2_feat = x1_1                       # Z2: enriched local-global

        # GNN: cosine similarity k-NN graph (k=3)
        x1_1 = self.gnn(x1_1)

        # GEA2 + PointCN blocks (second geometric refinement)
        x1_1 = self.pcn4(x1_1)
        x1_1 = self.gea2(x1_1, data)
        x1_1 = self.pcn5(x1_1)
        out = self.pcn6(x1_1)               # Z3: comprehensive features

        # Output: logits and essential matrix
        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        logits_sorted, indices = torch.sort(logits, dim=-1, descending=True)

        x_out, feature_out = down_sampling(xs, indices, out)
        w = self.linear1(feature_out)
        e_hat = weighted_8points(x_out, w)

        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual, out, z1_feat


# ==============================================================================
# CSGFE - Final: Cross-Stage Graph Consensus (Network-level)
# ==============================================================================

class CSGFEFinal(nn.Module):
    """Final CSGFE module for fusing all stage outputs at the network level.

    Paper description (Section III-C):
    - Takes comprehensive features Z3 from all three stages
    - Constructs cosine-similarity k-NN graphs
    - Aggregates via annular convolution
    """

    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        # 3 graphs × k=3 neighbors = 9 last dim → reduce to 1
        # (1,3) stride(1,3): 9→3, then (1,3): 3→1
        self.annular_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, (1, 3), stride=(1, 3)),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, (1, 3), stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = MLPs(out_channels, out_channels)

    def forward(self, stage_1, stage_2, stage_3):
        """Fuse three stage outputs via graph consensus.

        Args:
            stage_1/2/3: (B, C, N, 1) features from each stage

        Returns:
            out: (B, C, N, 1) fused consensus features
        """
        # Cosine similarity graph construction (paper specification)
        s1_graph = get_graph_feature(stage_1, k=3, use_cosine=True)
        s2_graph = get_graph_feature(stage_2, k=3, use_cosine=True)
        s3_graph = get_graph_feature(stage_3, k=3, use_cosine=True)

        combine_graph = torch.cat([s1_graph, s2_graph, s3_graph], dim=-1)
        ann_out = self.mlp(self.annular_conv(combine_graph))
        out = ann_out + stage_3  # Residual from final stage
        return out


# ==============================================================================
# GECSNet: Main Network (Paper Section III)
# ==============================================================================

class GECSNet(nn.Module):
    """GECS-Net: Geometric Embedding and Cross-Stage Network.

    Architecture (matching IEEE SPL 2026 paper):
    1. Three cascaded iterative stages (1 init + 2 refinement)
    2. Each stage: GEA (RoPE+MBFFN) ×2 + GNN (cosine k-NN, k=3)
    3. CSGFE: cross-stage graph feature extraction with SE calibration
    4. Final: MBSE enhancement → weighted 8-point algorithm

    Key differences from earlier MGCA code:
    - RoPE replaces CPT additive coordinate bias
    - Cosine similarity replaces Euclidean distance for graph k-NN
    - k=3 (paper default) instead of k=6
    - MB-FFN additive fusion (no sigmoid gating)
    - CSGFE with SE calibration (two-step fusion)
    - No DP_OA_DOP_block (OANet module not in paper)

    Args:
        config: configuration object with network parameters
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = 2  # 2 refinement stages (3 total including init)
        depth_each_stage = config.net_depth // (self.iter_num + 1)

        # ---- Stage 0: Initial ----
        self.stage_init = GECSNetStage(
            stage_idx=0,
            net_channels=config.net_channels,
            input_channel=4,      # (x1, y1, x2, y2)
            depth=depth_each_stage,
            is_init=True,
            knn_num=3,            # paper: k=3
        )

        # ---- Stages 1 & 2: Refinement ----
        self.stages = nn.ModuleList([
            GECSNetStage(
                stage_idx=i + 1,
                net_channels=config.net_channels,
                input_channel=6,  # orig + residual + logits
                depth=depth_each_stage,
                is_init=False,
                knn_num=3,        # paper: k=3
            )
            for i in range(self.iter_num)
        ])

        # ---- CSGFE: Cross-Stage Graph Consensus (final fusion) ----
        self.csgfe = CSGFEFinal(config.net_channels, config.net_channels)

        # ---- MBSE: Feature Enhancement ----
        self.mbse1 = MBSE(config.net_channels, config.net_channels // 2, 1)
        self.mbse2 = MBSE(config.net_channels, config.net_channels // 2, 1)

        # ---- Output heads ----
        self.final_conv = nn.Conv2d(config.net_channels, 1, kernel_size=1)
        self.final_linear = nn.Conv2d(config.net_channels, 2, kernel_size=1)

    def forward(self, data):
        """
        Args:
            data: dict with key 'xs': (B, 1, N, 4) correspondences

        Returns:
            res_weights: list of (B, N) logits at each stage + final
            res_e_hat:   list of (B, 9) essential matrices at each stage + final
        """
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        xs = data['xs']
        input_tensor = xs.transpose(1, 3)  # (B, 4, N, 1)

        res_weights, res_e_hat = [], []
        stage_outputs = []

        # ---- Stage 0: Initial ----
        logits, e_hat, residual, out, z1_feat = self.stage_init(
            input_tensor, xs)
        stage_outputs.append(out)
        init_out = out
        res_weights.append(logits)
        res_e_hat.append(e_hat)

        # ---- Stages 1 & 2: Refinement ----
        # x_prev_feat: intermediate feature from prev stage (like x1_1 in original)
        # x_prev_out: final output feature from prev stage
        x_prev_feat = z1_feat
        x_prev_out = out
        for i in range(self.iter_num):
            stage_input = torch.cat([
                input_tensor,
                residual.detach(),
                torch.relu(torch.tanh(logits)).reshape(residual.shape).detach(),
            ], dim=1)

            logits, e_hat, residual, out, z1_feat = self.stages[i](
                stage_input, xs, x_prev_feat, x_prev_out, init_out)

            x_prev_feat = z1_feat
            x_prev_out = out
            res_weights.append(logits)
            res_e_hat.append(e_hat)
            stage_outputs.append(out)

        # ---- CSGFE: Cross-Stage Graph Consensus ----
        csgfe_out = self.csgfe(stage_outputs[0], stage_outputs[1], stage_outputs[2])

        # ---- MBSE: Feature Enhancement ----
        enhanced = self.mbse2(self.mbse1(csgfe_out))

        # ---- Final Output ----
        logits = torch.squeeze(torch.squeeze(self.final_conv(enhanced), 3), 1)
        logits_sorted, indices = torch.sort(logits, dim=-1, descending=True)

        x_out, feature_out = down_sampling(xs, indices, enhanced)
        w = self.final_linear(feature_out)
        e_hat = weighted_8points(x_out, w)

        res_weights.append(logits)
        res_e_hat.append(e_hat)

        return res_weights, res_e_hat


# ==============================================================================
# Weighted 8-Point Algorithm
# ==============================================================================

def batch_symeig(X):
    """Batch symmetric eigenvalue decomposition (preserves input device)."""
    device = X.device
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='L')
        bv[batch_idx, :, :] = v
    bv = bv.to(device)
    return bv


def weighted_8points(x_in, logits):
    """Weighted 8-point algorithm for Essential matrix estimation.

    Uses learned weights (sigmoid mask + exponential weights) for robust
    estimation, replacing RANSAC's random sampling with deterministic weighting.

    Args:
        x_in:   (B, 1, N, 4) correspondences
        logits: (B, 2, N, 1) [mask_logits, weight_logits]

    Returns:
        e_hat: (B, 9) normalized essential matrix
    """
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    # Create data matrix
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)

    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from smallest eigenvector
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)

    return e_hat
