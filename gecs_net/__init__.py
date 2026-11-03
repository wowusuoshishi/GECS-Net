"""GECS-Net: Geometric Embedding and Cross-Stage Network for Two-View Correspondence Learning.

GECS-Net synergizes Geometry-Aware Embedding (GEA) with Cross-Stage Graph
Feature Extraction (CSGFE) for robust outlier rejection and camera pose estimation.

Reference:
    Lin et al., "GECS-Net: Two-View Correspondence Learning via Geometric
    Embedding and Cross-Stage Consensus", IEEE SPL, 2026.
"""

from .model import GECSNet
from .config import get_config

__all__ = ['GECSNet', 'get_config']
