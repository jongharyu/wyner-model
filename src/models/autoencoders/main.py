import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm

from datasets.dataloaders import get_dataloaders_mnist, get_dataloaders_svhn, get_dataloaders_cub_sent
from datasets.utils import unpack_data
from helpers.cub import CUBSentHelper
from helpers.generic import EarlyStopping, Timer, get_running_path, str2bool
from helpers.images import save_image
from models.adversarial.losses import loss_log_prob
from models.autoencoders.model import Encoder, Decoder, Autoencoder
from models.networks.cub_sent import CUBSentFeatureMap, CUBSentDecoderMap
from models.networks.image import ConvNet, SymmetricDeconvNet


def build_image_ae(cc, base_cc, negative_slope, use_batchnorm, n_resblocks):
    encoder_base_cc = base_cc
    encoder_negative_slope = negative_slope
    decoder_base_cc = base_cc

    to_feature = ConvNet(cin=cc,
                         base_cc=encoder_base_cc,
                         negative_slope=encoder_negative_slope,
                         use_batchnorm=use_batchnorm,
                         n_resblocks=n_resblocks)

    to_data = SymmetricDeconvNet(base_cc=decoder_base_cc,
                                 cout=cc,
                                 negative_slope=encoder_negative_slope,
                                 use_batchnorm=use_batchnorm,
                                 n_resblocks=n_resblocks)
    assert to_feature.output_dim == to_data.input_dim, (to_feature.output_dim, to_data.input_dim)
    encoder = Encoder(to_feature, to_latent=None)
    decoder = Decoder(to_data, to_feature=None)
    model = Autoencoder(encoder, decoder)

    return model


def build_cub_sent_ae(base_cc=32, conv2d=True):
    # vocab related parameters
    vocab_size = 1590
    embedding_dim = 128
    # max_sentence_length = 32
    # dim_y = max_sentence_length
    feature_map = CUBSentFeatureMap(vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    base_cc=base_cc,
                                    negative_slope=0,
                                    embedding_noise_scale=0.,
                                    use_batchnorm=True,
                                    batchnorm_kwargs={},
                                    last_activation=True,
                                    use_spectralnorm=False,
                                    conv2d=conv2d)

    decoder_map = CUBSentDecoderMap(vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    base_cc=base_cc,
                                    negative_slope=0,
                                    use_batchnorm=True,
                                    batchnorm_kwargs={},
                                    conv2d=conv2d)

    encoder = Encoder(feature_map, to_latent=None)
    decoder = Decoder(decoder_map, to_feature=None)
    model = Autoencoder(encoder, decoder)

    return model


def train(model, train_loader, optimizer, loss_fn,
          epoch, agg, device):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        x = unpack_data(dataT, device=device)
        xhat = model(x)
        loss = loss_fn(xhat, x)
        b_loss += loss.item()
        loss.backward()
        optimizer.step()

    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('\n====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def test(model, data_loader, early_stopping, loss_fn,
         epoch, agg, run_path, device, mode='test'):
    if not config.do_not_call_eval_during_inference:
        model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(tqdm(data_loader)):
            x = unpack_data(dataT, device=device)
            xhat = model(x)
            loss = loss_fn(xhat, x)
            b_loss += loss.item()
            if i == 0 and mode == 'valid':
                B = 8  # number of data samples to be reconstructed
                xhat = model.reconstruct(x[:B]).squeeze(0)
                filepath = '{}/recon_{:03d}.png'.format(run_path, epoch)
                if config.dataset in ['mnist', 'svhn']:
                    comp = torch.cat([x[:B], xhat]).data.cpu()
                    save_image((comp + 1.) / 2., filepath)
                elif config.dataset == 'cub-sent':
                    helper = CUBSentHelper(config.main_path, 32, 3, 128, 3, reset=False, device=None)
                    helper.save_sent_recon(x, xhat, filepath)

    agg['{}_loss'.format(mode)].append(b_loss / len(data_loader.dataset))
    print('\n====> Epoch: {:03d} {} loss: {:.4f}'.format(epoch, mode, agg['{}_loss'.format(mode)][-1]))

    if mode == 'valid':
        # Early stopping
        # needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(agg['{}_loss'.format(mode)][-1], model, agg)

        if early_stopping.early_stop:
            print("Early stopping")
            return -1


def main(config):
    # parameters
    dataset = config.dataset

    main_path = config.main_path

    # running path
    run_id, run_path = get_running_path(config, 'ae/{}'.format(dataset))

    # check device
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # random seed
    # Ref: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # objectives
    loss_fns = dict(l1=torch.nn.L1Loss, l2squared=torch.nn.MSELoss, )
    loss_fn = loss_fns[config.loss_fn](reduction='mean')

    # define the data loaders
    if dataset in ['mnist', 'svhn']:
        if dataset == 'mnist':
            train_loader, valid_loader, test_loader = get_dataloaders_mnist(config.batch_size, device=device, root_path=main_path,
                                                                            valid_split=0.1, size=32)
            cc = 1
        else:  #  dataset == 'svhn'
            train_loader, valid_loader, test_loader = get_dataloaders_svhn(config.batch_size, device=device, root_path=main_path,
                                                                           valid_split=0.1)
            cc = 3
        model = build_image_ae(cc, config.base_cc, config.negative_slope,
                               config.use_batchnorm, config.n_resblocks).to(device)
    elif dataset == 'cub-sent':
        train_loader, valid_loader, test_loader = get_dataloaders_cub_sent(config.batch_size, shuffle=True,
                                                                           device=device, root_path=main_path)
        model = build_cub_sent_ae(base_cc=config.base_cc, conv2d=config.cub_use_conv2d)

        if config.use_pretrained_embedding:
            # the following parameters are for training word2vec embedding;
            # please note that these are independent from the language model we actually train.
            max_sentence_length = 32
            window_length = 3
            min_occur = 3

            pretrained_embedding = CUBSentHelper(config.main_path, max_sentence_length, min_occur, 128,
                                                 window_length, reset=False, device=None).word2vec_embedding
            model.encoder.to_feature.embedding.weight = nn.Parameter(
                pretrained_embedding.detach().clone()).requires_grad_(True)

        if config.tie_embedding_inverse:
            model.decoder.embedding_inverse.weight = model.encoder.to_feature.embedding.weight

        model = model.to(device)
        loss_fn = loss_log_prob
    else:
        raise ValueError

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.lr, amsgrad=True)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.patience,
                                   verbose=True,
                                   path=run_path)

    print(model)
    summary(model)

    agg = defaultdict(list)
    for epoch in range(1, config.n_epochs + 1):
        with Timer('epoch {:03d}'.format(epoch)):
            train(model, train_loader, optimizer, loss_fn, epoch, agg, device)
            valid_flag = test(model, valid_loader, early_stopping, loss_fn, epoch, agg, run_path, device, mode='valid')
            if valid_flag == -1:
                break
    print("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning autoencoders')
    parser.add_argument('--experiment', type=str, default='')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'svhn', 'cub-sent'])
    parser.add_argument('--temp', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--main-path', type=str, default="..", help='main path where datasets live and loggings are saved')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA use')

    parser.add_argument('--loss-fn', type=str, default='l1', choices=['l1', 'l2squared'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
    parser.add_argument('--n-epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 15)')
    parser.add_argument('--patience', type=int, default=5)

    # CUB specific
    parser.add_argument('--tie-embedding-inverse', action='store_true')
    parser.add_argument('--cub-use-conv2d', type=str2bool, default=True)
    parser.add_argument('--use-pretrained-embedding', type=str2bool, default=False)

    # encoder network hyperparameters
    parser.add_argument('--negative-slope', type=float, default=0.2)
    parser.add_argument('--base-cc', type=int, default=32)
    parser.add_argument('--n-resblocks', type=int, default=0)
    parser.add_argument('--use-batchnorm', type=str2bool, default=True)
    parser.add_argument("--do-not-call-eval-during-inference", action='store_true')

    config = parser.parse_args()
    main(config)
