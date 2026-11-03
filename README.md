# GECS-Net: Geometric Embedding and Cross-Stage Network for Two-View Correspondence Learning

[![IEEE SPL 2026](https://img.shields.io/badge/Paper-IEEE%20SPL%202026-blue)](https://ieeexplore.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-orange)]()
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-red)]()

This repository provides the official PyTorch implementation of **GECS-Net** (Geometric Embedding and Cross-Stage Network), a deep learning framework for robust two-view correspondence learning and outlier rejection.

GECS-Net synergizes **Geometry-Aware Embedding (GEA)** with **Cross-Stage Graph Feature Extraction (CSGFE)** to achieve state-of-the-art performance on camera pose estimation, even under challenging scenarios with high outlier ratios and complex geometric deformations.

## Overview

Two-view correspondence learning is fundamental to 3D computer vision tasks such as structure-from-motion (SfM) and simultaneous localization and mapping (SLAM). GECS-Net addresses the core challenge of outlier rejection through two key innovations:

1. **Geometric Embedding Attention (GEA)**: Incorporates Rotary Position Embeddings (RoPE) to explicitly encode relative spatial constraints into the attention mechanism, enabling geometry-aware feature learning.

2. **Cross-Stage Graph Feature Extraction (CSGFE)**: Leverages a multi-stage sparse graph neural network with cosine-similarity k-NN graphs and squeeze-and-excitation (SE) calibration to progressively establish geometric consistency across iterative refinement stages.

## Architecture

```
Input: N×4 correspondences (x1, y1, x2, y2)
         │
    ┌────▼────┐
    │ Stage 0 │  Init: Conv → GEA(RoPE+MBFFN)×2 → GNN(k=3,cosine)
    └────┬────┘
         │ Z1, Z2, Z3
    ┌────▼────┐
    │ Stage 1 │  CSGFE cross-stage fusion → GEA×2 → GNN(k=3)
    └────┬────┘
         │
    ┌────▼────┐
    │ Stage 2 │  CSGFE + SE historical calibration → GEA×2 → GNN(k=3)
    └────┬────┘
         │
    ┌────▼────┐
    │  CSGFE  │  Cross-Stage Graph Consensus (all 3 stages)
    └────┬────┘
         │
    ┌────▼────┐
    │  MBSE   │  Multi-Branch SE Enhancement
    └────┬────┘
         │
    ┌────▼────┐
    │ Weighted │  Weighted 8-Point Algorithm
    │ 8-Point  │  → Essential Matrix E
    └─────────┘
```

### Key Modules

| Module | Description |
|--------|-------------|
| **GEA** | Geometric Embedding Attention: RoPE-based positional encoding + Multi-Branch FFN |
| **RoPE** | Rotary Position Embedding: encodes relative spatial displacement via learned rotation |
| **MB-FFN** | Multi-Branch Feed-Forward Network: fuses local + global average + global max pooling |
| **CSGFE** | Cross-Stage Graph Feature Extraction: cosine k-NN + SE calibration + annular convolution |
| **GNN** | Graph Neural Network: cosine similarity k-NN (k=3) + annular convolution + AFF fusion |
| **MBSE** | Multi-Branch Squeeze-and-Excitation for feature enhancement |

### Loss Function

```
L = L_cls + γ · L_geo    (γ = 0.5)
```

- **L_cls**: Balanced binary cross-entropy for inlier/outlier classification
- **L_geo**: Symmetric epipolar distance penalty on virtual correspondences

## Requirements

- Python >= 3.10
- PyTorch >= 2.6.0
- OpenCV (opencv-contrib-python) >= 4.11.0
- h5py >= 3.13.0
- NumPy, Matplotlib, tqdm, six
- tensorboardX >= 2.6.0

### Installation

```bash
# Clone the repository
git clone https://github.com/wowusuoshishi/GECS-Net.git
cd GECS-Net

# Create conda environment
conda create -n gecsnet python=3.12
conda activate gecsnet

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

Download the YFCC100M and SUN3D datasets from [this Google Drive link](https://drive.google.com/drive/folders/1RbBWKy-6QdbKofZGHGhTTRGNlv0rgtVi).

*Note: This download link is originally provided by the OANet project.*

Required files:
- **YFCC100M**: `raw_data_yfcc.tar.gz` (parts 0–8)
- **SUN3D Test**: `raw_sun3d_test.tar.gz` (parts 0–2)
- **SUN3D Train**: `raw_sun3d_train.tar.gz` (parts 0–63)

### Merge and Extract

```bash
# YFCC100M
bash merge_chunks.sh raw_data_yfcc.tar merged_yfcc.tar.gz 0 8
tar -xzvf merged_yfcc.tar.gz

# SUN3D Train
bash merge_chunks.sh raw_sun3d_train.tar.gz merged_sun3d_train.tar.gz 0 63
tar -xzvf merged_sun3d_train.tar.gz

# SUN3D Test
bash merge_chunks.sh raw_sun3d_test.tar.gz merged_sun3d_test.tar.gz 0 2
tar -xzvf merged_sun3d_test.tar.gz
```

### Generate SIFT Correspondences

```bash
cd ../dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```

## Training

Train GECS-Net on YFCC100M or SUN3D:

```bash
# Train on YFCC100M
python main.py --run_mode=train \
    --data_tr=../data_dump/yfcc-sift-2000-train.hdf5 \
    --data_va=../data_dump/yfcc-sift-2000-val.hdf5 \
    --train_batch_size=32 \
    --train_lr=3e-4 \
    --train_iter=500000

# Train on SUN3D
python main.py --run_mode=train \
    --data_tr=../data_dump/sun3d-sift-2000-train.hdf5 \
    --data_va=../data_dump/sun3d-sift-2000-val.hdf5 \
    --train_batch_size=32
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--net_depth` | 12 | Number of layers per stage |
| `--net_channels` | 128 | Feature dimension |
| `--train_batch_size` | 32 | Batch size |
| `--train_lr` | 3e-4 | Learning rate |
| `--train_iter` | 500000 | Training iterations |
| `--loss_essential` | 0.5 | Geometry loss weight (γ in paper) |
| `--loss_classif` | 1.0 | Classification loss weight |
| `--gpu_id` | 0 | GPU device ID |

## Testing

Evaluate a trained model:

```bash
# Test with GECS-Net Essential matrix (no RANSAC)
python main.py --run_mode=test \
    --model_path=../weights/yfcc100m/ \
    --res_path=./results/yfcc/test \
    --data_te=../data_dump/yfcc-sift-2000-test.hdf5 \
    --use_ransac=False

# Test with RANSAC post-processing
python main.py --run_mode=test \
    --model_path=../weights/yfcc100m/ \
    --res_path=./results/yfcc/test_ransac \
    --data_te=../data_dump/yfcc-sift-2000-test.hdf5 \
    --use_ransac=True
```

### Metrics

- **Outlier Rejection**: Precision (P), Recall (R), F-score (F)
- **Pose Estimation**: mAP@5°, mAP@10°, mAP@15°, mAP@20°
- **AUC**: Area Under Curve at 5°, 10°, 20° thresholds

## Pretrained Models

We provide pretrained GECS-Net models trained on YFCC100M and SUN3D datasets.

| Dataset | mAP@5° | F-score | Download |
|---------|--------|---------|----------|
| YFCC100M | TBD | TBD | [weights/yfcc100m/](weights/yfcc100m/) |
| SUN3D | TBD | TBD | [weights/sun3d/](weights/sun3d/) |

*Pretrained weights will be released upon paper publication.*

## Quick Inference Demo

```python
import torch
from gecs_net.model import GECSNet
from gecs_net.config import get_config

# Load model
config, _ = get_config()
model = GECSNet(config)
model.load_state_dict(torch.load('weights/model_best.pth')['state_dict'])
model.cuda().eval()

# Input: N correspondences (x1, y1, x2, y2)
data = {'xs': correspondences_tensor.cuda()}  # (1, 1, N, 4)

with torch.no_grad():
    res_weights, res_e_hat = model(data)

# Get final inlier logits and Essential matrix
inlier_scores = torch.sigmoid(res_weights[-1])
E_matrix = res_e_hat[-1].reshape(3, 3)
```

## Project Structure

```
GECS-Net/
├── README.md                    # This file
├── main.py                      # Entry point
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── gecs_net/                    # Core package
│   ├── __init__.py
│   ├── model.py                 # GECSNet architecture (GEA + CSGFE)
│   ├── config.py                # Configuration
│   ├── train.py                 # Training loop
│   ├── test.py                  # Testing/evaluation
│   ├── loss.py                  # Loss functions
│   ├── data.py                  # HDF5 data loader
│   ├── evaluation.py            # Pose error metrics
│   ├── pose.py                  # mAP/AUC computation
│   ├── utils.py                 # Utilities
│   ├── logger.py                # Logging
│   └── warmup_scheduler.py      # LR scheduler
├── scripts/                     # Utility scripts
├── weights/                     # Pretrained models
└── docs/                        # Documentation
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{lin2026gecsnet,
  title = {GECS-Net: Two-View Correspondence Learning via Geometric
           Embedding and Cross-Stage Consensus},
  author = {Shuyuan Lin and Mengtin Lo and Haosheng Chen and
            Yanjie Liang and Qiangqiang Wu},
  journal = {IEEE Signal Processing Letters},
  year = {2026},
}
```

## Acknowledgement

This code builds upon the excellent work of:

- [OANet](https://github.com/zjhthu/OANet) — Order-Aware Network for two-view correspondence learning
- [BCLNet](https://github.com/guobaoxiao/BCLNet) — Bilateral Consensus Learning
- [MSGSA](https://github.com/shuyuanlin/MSGSA) — Multi-Stage Network with Geometric Semantic Attention

If using parts of the code related to data generation, testing, or evaluation, please also cite these works.

```bibtex
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei
          and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={5845--5854},
  year={2019}
}

@inproceedings{miao2024bclnet,
  title={BCLNet: Bilateral consensus learning for two-view correspondence pruning},
  author={Miao, Xiangyang and Xiao, Guobao and Wang, Shiping and Yu, Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4225--4232},
  year={2024}
}

@article{lin2024multi,
  author={Lin, Shuyuan and Chen, Xiao and Xiao, Guobao and Wang, Hanzi
          and Huang, Feiran and Weng, Jian},
  journal={IEEE Transactions on Image Processing},
  title={Multi-Stage Network with Geometric Semantic Attention for
         Two-View Correspondence Learning},
  year={2024},
  volume={33},
  pages={3031--3046},
}
```

## License

This project is released under the [MIT License](LICENSE).
