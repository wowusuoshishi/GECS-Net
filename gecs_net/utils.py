"""Utility functions for GECS-Net."""

import torch
import numpy as np


def tocuda(data):
    """Convert tensor data in dictionary to cuda when it is a tensor."""
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data


def get_pool_result(num_processor, fun, args):
    """Serial execution to avoid multiprocessing spawn issues on Windows."""
    return [fun(arg) for arg in args]


def np_skew_symmetric(v):
    """Numpy skew-symmetric matrix from 3D vector."""
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M


def torch_skew_symmetric(v):
    """PyTorch skew-symmetric matrix from 3D vector."""
    zero = torch.zeros_like(v[:, 0])
    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)
    return M
