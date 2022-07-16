import argparse
import os

import numpy as np
import torch

from external.pytorch_fid.fid_score import compute_statistics, get_model_instance


def save_feature_stats_given_path(path, batch_size, device, dims, num_workers=8, model_type='inception'):
    """Calculates the inception statistics of the path"""
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)

    filename = '{}/{}_stats_dims{}.npz'.format(path, model_type, dims)
    if os.path.exists(filename):
        print('The statistics file already exists: {}'.format(filename))
        return

    model = get_model_instance(model_type, dims, device)

    mu, sigma = compute_statistics(path, model, batch_size,
                                   model_type, dims, device, num_workers)

    np.savez(filename, mu=mu, sigma=sigma)


def main(config):
    # cuda
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    save_feature_stats_given_path(
        config.path,
        config.batch_size,
        device,
        config.dims,
        config.num_workers,
        config.model_type,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute model feature stats for Frechet distance')

    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA use')
    parser.add_argument('--model-type', type=str, choices=['inception', 'mnist', 'svhn', 'mnist_ae', 'svhn_ae'])
    parser.add_argument('--dims', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
    parser.add_argument('--path', type=str, default="..", help='path to the target image folder')

    config = parser.parse_args()
    main(config)
