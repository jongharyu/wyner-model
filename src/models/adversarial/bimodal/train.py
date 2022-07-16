import sys
from collections import defaultdict

import numpy as np
import torch
from reprint import output
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from datasets.utils import unpack_data
from helpers.generic import save_model
from helpers.training import toggle_grad

DISCR_LOSS_TEMPLATE = ('\tloss(Discr): {:+7.3f}\n'
                       '\t\t[dist]             j{:+7.3f}, cx2y{:+7.3f}, cy2x{:+7.3f}, mx{:+7.3f}, my{:+7.3f}, c{:+7.3f}\n'
                       '\t\t[ci  ] var{:+7.3f}, j{:+7.3f}, cx2y{:+7.3f}, cy2x{:+7.3f}\n'
                       '\t\t[agg ] var{:+7.3f},           cx2y{:+7.3f}, cy2x{:+7.3f},                       c{:+7.3f}')

GENER_LOSS_TEMPLATE = ('\tloss(Gener): {:+7.3f}\n'
                       '\t\t[dist]             j{:+7.3f}, cx2y{:+7.3f}, cy2x{:+7.3f}, mx{:+7.3f}, my{:+7.3f}, c{:+7.3f}\n'
                       '\t\t[ci  ] var{:+7.3f}, j{:+7.3f}, cx2y{:+7.3f}, cy2x{:+7.3f}\n'
                       '\t\t[agg ] var{:+7.3f},           cx2y{:+7.3f}, cy2x{:+7.3f},                       c{:+7.3f}\n'
                       '\t\t[recx]             j{:+7.3f},              cy2x{:+7.3f}, mx{:+7.3f}\n'
                       '\t\t[recy]             j{:+7.3f}, cx2y{:+7.3f},                         my{:+7.3f}\n'
                       '\t\t[recz]             j{:+7.3f}')

DISCR_LOSS_KEYS = ['total',
                   'j', 'cx2y', 'cy2x', 'mx', 'my', 'c',
                   'var_ci', 'j_ci', 'cx2y_ci', 'cy2x_ci',
                   'var_agg', 'cx2y_agg', 'cy2x_agg', 'c_agg']

GENER_LOSS_KEYS = DISCR_LOSS_KEYS + ['j_rec_x', 'cy2x_rec_x', 'mx_rec_x',
                                     'j_rec_y', 'cx2y_rec_y', 'my_rec_y',
                                     'j_rec_zuv']


class Trainer:
    def __init__(self, config,
                 model, ema: ExponentialMovingAverage,
                 optimizer_gener, optimizer_discr,
                 train_loader, train_loaders_marginal,
                 run_path, device):
        self.model = model
        self.ema = ema

        self.generator = model.generator
        self.discriminator = model.discriminator

        self.optimizer_gener = optimizer_gener
        self.optimizer_discr = optimizer_discr

        self.gener_steps = config.gener_steps
        self.discr_steps = config.discr_steps
        self.continue_training_after_nan = config.continue_training_after_nan
        self.dataset = config.dataset
        self.ssl = config.ssl
        self.clip_grad_norm = config.clip_grad_norm

        self.train_loader = train_loader
        self.train_loaders_marginal = train_loaders_marginal
        self.run_path = run_path
        self.device = device
        self.config = config

    def train_generator(self, xq, yq):
        if self.config.call_eval_during_training:
            self.generator.train()
            self.discriminator.eval()

        self.optimizer_discr.zero_grad()
        self.optimizer_gener.zero_grad()

        gener_losses = self.model.compute_generator_losses(xq, yq)
        if np.isnan(gener_losses['total'].item()):
            return gener_losses, 1

        # update
        gener_losses['total'].backward()
        #   (optional) clip gradient norm
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_grad_norm)
        self.optimizer_gener.step()

        # (optional) exponential moving average
        self.ema.update()

        return gener_losses, 0

    def train_discriminator(self, xq, yq):
        if self.config.call_eval_during_training:
            self.generator.train()
            self.discriminator.train()

        self.optimizer_discr.zero_grad()
        self.optimizer_gener.zero_grad()

        discr_losses = self.model.compute_discriminator_losses(xq, yq)
        if np.isnan(discr_losses['total'].item()):
            return discr_losses, 1

        # update
        discr_losses['total'].backward()
        #   (optional) clip gradient norm
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
        self.optimizer_discr.step()

        return discr_losses, 0

    def load_checkpoint(self, load_optim=False):
        self.model.load_state_dict(torch.load('{}/model.rar'.format(self.run_path),
                                              map_location=self.device))
        self.ema.load_state_dict(torch.load('{}/ema.rar'.format(self.run_path),
                                            map_location=self.device))

        if load_optim:
            self.optimizer_gener.load_state_dict(torch.load('{}/optimizer_gener.rar'.format(self.run_path),
                                                            map_location=self.device))
            self.optimizer_discr.load_state_dict(torch.load('{}/optimizer_discr.rar'.format(self.run_path),
                                                            map_location=self.device))

        return self

    def save_checkpoint(self, save_optim=False):
        save_model(self.model, '{}/model.rar'.format(self.run_path), safe=False)
        with self.ema.average_parameters():
            save_model(self.ema, '{}/ema.rar'.format(self.run_path), safe=False)

        if save_optim:
            save_model(self.optimizer_gener, '{}/optimizer_gener.rar'.format(self.run_path), safe=False)
            save_model(self.optimizer_discr, '{}/optimizer_discr.rar'.format(self.run_path), safe=False)

        return self

    def initialize_data_iterator(self):
        train_loader_iterator = iter(self.train_loader)
        n_iters = int(np.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size))
        train_loaders_marginal_iterator = []
        if self.ssl:
            train_loaders_marginal_iterator = [iter(loader) for loader in self.train_loaders_marginal]
            n_iters = max([n_iters] + [int(np.ceil(len(loader.dataset) / loader.batch_size))
                                       for loader in self.train_loaders_marginal])

        self.n_iters = n_iters
        self.train_loader_iterator = train_loader_iterator
        self.train_loaders_marginal_iterator = train_loaders_marginal_iterator

    def get_batch(self):
        try:
            dataT = next(self.train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(self.train_loader)
            dataT = next(train_loader_iterator)
        data = unpack_data(dataT, device=self.device, dataset=self.dataset)
        xq, yq = data[0], data[1]

        return xq, yq

    def __call__(self, epoch, agg):
        # Note: we assume len(loader) <= len(loader_marginal) in what follows
        mode = 'train'
        self.model.train()
        losses = defaultdict(list)  # list of losses in an epoch

        logger = sys.stdout  # recover custom logger from sys.stdout
        sys.stdout = logger.terminal  # rollback sys.stdout

        self.initialize_data_iterator()
        with output(output_type='list',
                    initial_len=1 + len(DISCR_LOSS_TEMPLATE.split('\n')) + len(GENER_LOSS_TEMPLATE.split('\n')),
                    interval=0) as output_lines:
            for i in tqdm(range(1, self.n_iters + 1), disable=True):
                xq, yq = self.get_batch()

                # discriminator training;
                #   note that the following order of toggling should not be flipped,
                #   to avoid unexpected behavior when config.tie_embedding is True
                toggle_grad(self.discriminator, True, toggle_embedding=self.config.cub_sent_trainable_embedding)
                toggle_grad(self.generator, False, toggle_embedding=self.config.cub_sent_trainable_embedding)
                if self.config.discriminator_turn_off_additional_gaussian_noise_during_training:
                    self.discriminator.set_additional_gaussian_noise(True)
                for k in range(self.discr_steps):
                    discr_losses, nan_flag = self.train_discriminator(xq, yq)

                    if nan_flag == 1:
                        print('epoch {} ({}/{})\n'.format(epoch, i, self.n_iters))
                        print(DISCR_LOSS_TEMPLATE.format(*[discr_losses[key] for key in DISCR_LOSS_KEYS]))
                        sys.stdout = logger  # redirect stdout to the custom logger
                        break

                    # save
                    losses['discr'].append(discr_losses['total'])
                    for key in DISCR_LOSS_KEYS:
                        losses['discr_{}'.format(key)].append(discr_losses[key])

                if nan_flag == 1:
                    break

                toggle_grad(self.discriminator, False, toggle_embedding=self.config.cub_sent_trainable_embedding)
                toggle_grad(self.generator, True, toggle_embedding=self.config.cub_sent_trainable_embedding)
                if self.config.discriminator_turn_off_additional_gaussian_noise_during_training:
                    self.discriminator.set_additional_gaussian_noise(False)
                for j in range(self.gener_steps):
                    gener_losses, nan_flag = self.train_generator(xq, yq)

                    if nan_flag == 1:
                        print('epoch {} ({}/{})\n'.format(epoch, i, self.n_iters))
                        print(GENER_LOSS_TEMPLATE.format(*[gener_losses[key] for key in GENER_LOSS_KEYS]))
                        sys.stdout = logger  # redirect stdout to the custom logger
                        break

                    # save losses
                    losses['model'].append(gener_losses['total'])
                    for key in GENER_LOSS_KEYS:
                        losses['gener_{}'.format(key)].append(gener_losses[key])

                if nan_flag == 1:
                    break

                # print running average of losses
                discr_loss_strs = DISCR_LOSS_TEMPLATE.format(
                    *[torch.mean(torch.FloatTensor(losses['discr_{}'.format(key)]))
                      for key in DISCR_LOSS_KEYS]).split('\n')
                gener_loss_strs = GENER_LOSS_TEMPLATE.format(
                    *[torch.mean(torch.FloatTensor(losses['gener_{}'.format(key)]))
                      for key in GENER_LOSS_KEYS]).split('\n')
                loss_lines = discr_loss_strs + gener_loss_strs
                output_lines[0] = 'epoch {} ({}/{})'.format(epoch, i, self.n_iters)
                for lineno in range(1, 1 + len(loss_lines)):
                    output_lines[lineno] = loss_lines[lineno - 1]

            if nan_flag == 1:
                print("Warning: nan values encountered!")
                if self.continue_training_after_nan:
                    print("Warning: load last checkpoint and continue...")
                    self.load_checkpoint()

        loss_total = 0.  # logged to detect any nan values
        for key in losses:
            tmp = torch.mean(torch.FloatTensor(losses[key]))
            agg['{}_loss_{}'.format(mode, key)].append(tmp)
            loss_total += tmp
        agg['{}_loss_total'.format(mode)].append(loss_total)

        sys.stdout = logger  # redirect stdout to the custom logger
        print('\n'.join(output_lines))

        return nan_flag


def test(model,
         loader,  # test_loader or valid_loader
         dataset,
         early_stopping,
         epoch, agg,
         device, mode='test'):
    assert mode in ['test', 'valid']
    losses = defaultdict(list)

    with torch.no_grad():
        for dataT in tqdm(loader, disable=False, desc="Testing..."):
            data = unpack_data(dataT, device=device, dataset=dataset)
            xq, yq = data[0], data[1]

            discr_losses = model.compute_discriminator_losses(xq, yq)
            gener_losses = model.compute_generator_losses(xq, yq)

            # save
            losses['discr'].append(discr_losses['total'])
            for key in DISCR_LOSS_KEYS:
                losses['discr_{}'.format(key)].append(discr_losses[key])

            losses['gener'].append(gener_losses['total'])
            for key in GENER_LOSS_KEYS:
                losses['gener_{}'.format(key)].append(gener_losses[key])

    for key in losses:
        agg['{}_loss_{}'.format(mode, key)].append(torch.mean(torch.FloatTensor(losses[key])))

    # print per epoch validation/test losses
    print_loss_summary(epoch, mode, agg)

    if mode == 'valid':
        # Early stopping
        # needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(agg['valid_loss_gener'][-1], model, agg)

        if early_stopping.early_stop:
            print("Early stopping")
            return -1


def print_loss_summary(epoch, mode, agg):
    print('\n====> Epoch {:03d} {} losses'.format(epoch, mode))
    print(DISCR_LOSS_TEMPLATE.format(*[agg['{}_loss_discr_{}'.format(mode, key)][-1] for key in DISCR_LOSS_KEYS]))
    print(GENER_LOSS_TEMPLATE.format(*[agg['{}_loss_gener_{}'.format(mode, key)][-1] for key in GENER_LOSS_KEYS]))
