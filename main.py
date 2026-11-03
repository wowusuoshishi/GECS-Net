"""GECS-Net: Geometric Embedding and Cross-Stage Network for Two-View Correspondence Learning.

Main entry point for training and testing.

Reference:
    Lin et al., "GECS-Net: Two-View Correspondence Learning via Geometric
    Embedding and Cross-Stage Consensus", IEEE SPL, 2026.

Usage:
    # Training on YFCC100M
    python main.py --run_mode=train --data_tr=../data_dump/yfcc-sift-2000-train.hdf5 --data_va=../data_dump/yfcc-sift-2000-val.hdf5

    # Testing with pretrained model
    python main.py --run_mode=test --model_path=../weights/yfcc100m/ --res_path=./results/yfcc/test --data_te=../data_dump/yfcc-sift-2000-test.hdf5
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gecs_net.config import get_config, print_usage
from gecs_net.data import collate_fn, CorrespondencesDataset
from gecs_net.train import train
from gecs_net.test import test
from gecs_net.model import GECSNet as Model

import torch


def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    result_path = config.log_base
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path + '/train'):
        os.makedirs(result_path + '/train')
    if not os.path.isdir(result_path + '/valid'):
        os.makedirs(result_path + '/valid')
    if not os.path.isdir(result_path + '/test'):
        os.makedirs(result_path + '/test')
    if os.path.exists(result_path + '/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path + '/config.th')
    config.log_path = result_path + '/train'


def main(config):
    """The main function."""

    # Initialize network
    model = Model(config)
    print('GECS-Net parameters:', sum(param.numel() for param in model.parameters()))

    # Run proper mode
    if config.run_mode == "train":
        create_log_dir(config)

        train_dataset = CorrespondencesDataset(config.data_tr, config)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True,
            num_workers=16, pin_memory=False, collate_fn=collate_fn)

        valid_dataset = CorrespondencesDataset(config.data_va, config)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn)

        print('start training GECS-Net .....')
        train(model, train_loader, valid_loader, config)

    elif config.run_mode == "test":
        test_dataset = CorrespondencesDataset(config.data_te, config)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn)

        test(test_loader, model, config)


if __name__ == "__main__":
    print("=" * 60)
    print("GECS-Net: Geometric Embedding and Cross-Stage Network")
    print("Two-View Correspondence Learning via Geometric Embedding")
    print("and Cross-Stage Consensus")
    print("IEEE SPL 2026")
    print("=" * 60)

    # Parse configuration
    config, unparsed = get_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    if config.run_mode == "train":
        create_log_dir(config)

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
