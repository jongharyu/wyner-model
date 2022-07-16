import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from datasets.dataloaders import get_dataloaders_mnist, get_dataloaders_svhn
from datasets.utils import unpack_data
from helpers.generic import Logger, Timer, str2bool, save_model, save_vars
from helpers.plot import plot_grid_samples, save_grid_two_image_samples
from models.adversarial.losses import loss_js_var, loss_symmetric_kl_plugin
from models.adversarial.unimodal.model import Discriminator, MarginalEncoder, Generator

model_names = {'mnist': 'symvae-mnist',
               'svhn': 'symvae-svhn'}


def main(config):
    # cuda
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # random seed
    # Ref: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # set up running path
    assert config.experiment, 'experiment identifier must be specified'
    run_id = datetime.datetime.now().isoformat()
    experiment_dir = Path('{}/experiments/{}/{}'.format(
        config.main_path, model_names[config.dataset], config.experiment
    ))
    experiment_dir.mkdir(parents=True, exist_ok=True if config.temp or config.overwrite else False)
    run_path = str(experiment_dir)
    if config.temp:
        run_path = mkdtemp(prefix=run_id, dir=run_path)
    sys.stdout = Logger('{}/run.log'.format(run_path))
    print('Expt: {}'.format(run_path))
    print('RunID: {}'.format(run_id))

    # save config to running path
    with open('{}/config.json'.format(run_path), 'w') as fp:
        json.dump(config.__dict__, fp)
    # -- also save object because we want to recover these for other things
    torch.save(config, '{}/config.rar'.format(run_path))
    print(config)

    # load models and dataloaders
    batch_size = config.batch_size
    shuffle = True

    if config.dataset == 'mnist':
        cc = 1
        from models import get_networks
        discriminator_conv_net, discriminator_feature_dim, \
            encoder_conv_net, encoder_feature_dim, \
            Decoder = get_networks(cc=cc)
        train_loader, valid_loader, test_loader = get_dataloaders_mnist(
            batch_size, shuffle, device, binarize=False, root_path=config.main_path, size=32,
            drop_last=True,
        )

    elif config.dataset == 'svhn':
        cc = 3
        from models import get_networks
        discriminator_conv_net, discriminator_feature_dim, \
            encoder_conv_net, encoder_feature_dim, \
            Decoder = get_networks(cc=cc)
        train_loader, valid_loader, test_loader = get_dataloaders_svhn(
            batch_size, shuffle, device, binarize=False, root_path=config.main_path,
            drop_last=True,
        )

    else:
        raise Exception('check config.dataset')

    # define models
    dim_z = config.dim_z
    dim_discr_noise = config.dim_discr_noise

    discriminator = Discriminator(dim_z=dim_z,
                                  to_feature=discriminator_conv_net,
                                  noise_dim=dim_discr_noise,
                                  use_batchnorm=config.use_batchnorm,
                                  device=device)
    encoder = MarginalEncoder(dim_z=dim_z,
                              to_feature=encoder_conv_net,
                              use_batchnorm=config.use_batchnorm,
                              device=device)
    decoder = Decoder(dim_z=dim_z, dim_u=0, cc=cc)
    autoencoder = Generator(encoder, decoder, discriminator).to(device)

    # training parameters
    num_epochs = config.num_epochs
    lr = config.lr

    optimizer_model = optim.Adam([*encoder.parameters(), *decoder.parameters()],
                                 lr=lr, betas=(config.beta1, config.beta2), amsgrad=True)
    optimizer_discr = optim.Adam(discriminator.parameters(),
                                 lr=lr, betas=(config.beta1, config.beta2), amsgrad=True)

    # draw reference (reconstruction) samples
    nref = 20
    batch = next(iter(test_loader))
    xref = batch[0][:nref]
    xref = 2 * xref - 1

    def plot_samples(suffix):
        # generation
        z_random = draw_noise(batch_size, dim_z, device=device)
        u_random = draw_noise(batch_size, decoder.dim_u, device=device)  # phantom, zero shape
        x_gen = decoder(z_random, u_random)
        plot_grid_samples((x_gen + 1.) / 2.,
                          '{}/gen_{}.{}'.format(run_path, suffix, config.output_filetype))

        # reconstruction
        u_random = draw_noise(xref.shape[0], decoder.dim_u, device=device)  # phantom, zero shape
        xref_hat = decoder(encoder(xref), u_random)  # (ncond, dim_z)
        save_grid_two_image_samples(
            (xref + 1.) / 2., (xref_hat + 1.) / 2.,
            '{}/rec_{}.{}'.format(run_path, suffix, config.output_filetype))

    def train(epoch, agg):
        autoencoder.train()
        discriminator.train()
        b_loss_discr = 0.
        b_loss_model = 0.

        train_loader_iterator = iter(train_loader)
        num_iters = int(np.ceil(len(train_loader.dataset) / train_loader.batch_size))

        for i in tqdm(range(num_iters), disable=config.silent):
            try:
                dataT = next(train_loader_iterator)
            except StopIteration:
                train_loader_iterator = iter(train_loader)
                dataT = next(train_loader_iterator)
            data = unpack_data(dataT, device=device)

            x_data = 2 * data - 1

            for k in range(config.discr_steps):
                optimizer_discr.zero_grad()
                optimizer_model.zero_grad()

                r_q, r_p, *_ = autoencoder(x_data)
                # discriminator loss
                loss_discr = - loss_js_var(r_p, r_q)

                # update
                loss_discr.backward()
                optimizer_discr.step()

            for j in range(config.model_steps):
                optimizer_discr.zero_grad()
                optimizer_model.zero_grad()

                r_q, r_p, _, z_gen, x_gen, z_random = autoencoder(x_data)

                # model loss
                loss_model = loss_symmetric_kl_plugin(r_p, r_q)

                # reconstruction loss
                u_gen = draw_noise(z_gen.shape[0], decoder.dim_u, device=device)
                x_hat = decoder(z_gen, u_gen)
                loss_rec_x = torch.abs(x_data - x_hat).mean()
                loss_rec_z = torch.abs(z_random - encoder(x_gen)).mean()
                loss_rec = loss_rec_x + loss_rec_z
                loss_model += config.lambda_rec * loss_rec

                # update
                loss_model.backward()
                optimizer_model.step()

            print('epoch: {}; iter: {}; '
                  'loss_discr: {:.4}; loss_model: {:.4}'.format(
                   epoch, i, loss_discr, loss_model))

            if i % 50 == 0:
                suffix = 'e{}_i{}'.format(str(epoch), str(i).zfill(3))
                plot_samples(suffix)

            b_loss_discr += loss_discr.item()  # accumulate losses
            b_loss_model += loss_model.item()  # accumulate losses

        agg['train_loss_discr'].append(b_loss_discr / (num_iters * train_loader.batch_size))
        agg['train_loss_model'].append(b_loss_model / (num_iters * train_loader.batch_size))
        print('\n====> Epoch: {:03d} Train loss (D, G): {:.4f}, {:.4f}'.format(
            epoch,
            agg['train_loss_discr'][-1],
            agg['train_loss_model'][-1])
        )

    print('-' * 89)
    with Timer('Symmetric VAE') as t:
        agg = defaultdict(list)
        for epoch in range(1, num_epochs + 1):
            with Timer('epoch {:03d}'.format(epoch)) as te:
                train(epoch, agg)

    save_model(autoencoder, run_path + '/model.rar', safe=False)
    save_vars(agg, run_path + '/train_history.rar', safe=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarially trained symmetric VAEs')

    # general experiment setting
    parser.add_argument('--experiment', type=str, default='', metavar='E',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # dataset
    parser.add_argument('--dataset', type=str, default="mnist",
                        choices=['mnist', 'svhn'])

    # logging related
    parser.add_argument('--main-path', type=str, default="..",
                        help='main path where datasets live and loggings are saved')
    parser.add_argument('--silent', type=str2bool, default=True, help='mute tqdm verbosity')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--temp', action='store_true', help='work on a temporary path')
    parser.add_argument('--plot-freq', type=int, default=250, help='frequency for plotting results')
    parser.add_argument('--nz', type=int, default=15, help='number of z for generation')
    parser.add_argument('--nuv', type=int, default=15, help='number of different (u,v) with each z for generation')
    parser.add_argument("--output-filetype", default="jpg", choices=["png", "jpg"])

    # model hyperparameters
    parser.add_argument('--dim-z', type=int, default=16, metavar='DZ', help='latent dimensionality (default: 16)')
    parser.add_argument('--dim-discr-noise', type=int, default=8, metavar='DD',
                        help='discriminator noise dimensionality (default: 8)')

    # training hyperparameters (general)
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA use')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='E',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--beta1', type=int, default=.5)
    parser.add_argument('--beta2', type=int, default=.999)

    # adversarial training
    parser.add_argument('--discr-steps', type=int, default=1)
    parser.add_argument('--model-steps', type=int, default=1)

    # objective hyperparameters (adversarial symmetric VAE specifics)
    parser.add_argument('--lambda-rec', type=float, default=1.0, help='loss weight for conditional loss')

    config = parser.parse_args()
    main(config)
