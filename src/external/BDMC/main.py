import argparse

import numpy as np
import torch

from external.BDMC.bdmc import bdmc
from external.BDMC.simulate import simulate_data
from external.BDMC.vae import VAE


def main(args):
    # cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = VAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    # bdmc uses simulated data from the model
    loader = simulate_data(
        model,
        batch_size=args.batch_size,
        n_batch=args.n_batch,
        device=device)

    # \b_t for mixing f_1 and f_T to define f_t = (f_1)^{1-\b_t} f_T^{\b_t}
    forward_schedule = np.linspace(0., 1., args.chain_length)
    # run bdmc
    _, _ = bdmc(
        model,
        loader,
        forward_schedule=forward_schedule,
        n_sample=args.iwae_samples,
        likelihood=args.likelihood,
        likelihood_var=args.likelihood_var,
        device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDMC')
    parser.add_argument('--latent-dim', type=int, default=50, metavar='D',
                        help='number of latent variables (default: 50)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='number of examples to eval at once (default: 10)')
    parser.add_argument('--n-batch', type=int, default=10, metavar='B',
                        help='number of batches to eval in total (default: 10)')
    parser.add_argument('--chain-length', type=int, default=500, metavar='L',
                        help='length of ais chain (default: 500)')
    parser.add_argument('--iwae-samples', type=int, default=100, metavar='I',
                        help='number of iwae samples (default: 100)')
    parser.add_argument('--likelihood', type=str, default='bernoulli',
                        choices=['normal', 'bernoulli'],
                        help='likelihood model')
    parser.add_argument('--likelihood-var', type=float, default=0.075,
                        help='likelihood variance for Gaussian observation model')
    parser.add_argument('--ckpt-path', type=str, default='checkpoints/model.pth',
                        metavar='C', help='path to checkpoint')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA use')
    args = parser.parse_args()

    main(args)
