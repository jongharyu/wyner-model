import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import optim
from torchinfo import summary
from torch_ema import ExponentialMovingAverage

from datasets.dataloaders import get_bimodal_dataloaders
from helpers.generic import Logger, Timer, get_running_path, save_config, save_vars, get_git_revision_hash
from helpers.training import toggle_grad
from models.adversarial.bimodal.evaluate import Evaluator
from models.adversarial.bimodal.model import build_models
from models.adversarial.bimodal.options import Options, SUPPORTED_DATASETS
from models.adversarial.bimodal.train import Trainer

MODEL_NAMES = {dataset: 'adv-bimodal/{}'.format(dataset) for dataset in SUPPORTED_DATASETS}


def main(config):
    if config.evaluate_only and config.path_to_saved_model:
        main_path = config.main_path
        path_to_saved_model = config.path_to_saved_model
        no_cuda = config.no_cuda
        temp = config.temp

        config = torch.load(path_to_saved_model + '/config.rar')

        config.main_path = main_path
        config.evaluate_only = True
        config.path_to_saved_model = path_to_saved_model
        config.no_cuda = no_cuda
        config.temp = temp

    # cuda
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # random seed
    # Ref: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(config.seed)
    random.seed(config.seed)

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(config.seed)

    # set up running path
    run_id, run_path = get_running_path(config, MODEL_NAMES[config.dataset])

    # set up custom logger
    sys.stdout = Logger('{}/run.log'.format(run_path))  # redirect stdout to a logger

    # log experiment configs
    print('Expt: {}'.format(run_path))
    print('RunID: {}'.format(run_id))
    print('GitHash: {}'.format(get_git_revision_hash()))

    # save config as json file to running path
    save_config(config, run_path)

    # get dataloaders
    train_loader, valid_loader, test_loader, train_loaders_marginal = get_bimodal_dataloaders(config, gen, worker_init_fn, device)

    # get model instances
    generator, discriminator, model = build_models(config, device)
    print(model)
    summary(generator)
    summary(discriminator)

    # Reference: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel
    # data parallelism
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ = torch.nn.DataParallel(model)
        model_.forward = model.forward
        model_.compute_generator_losses = model.compute_generator_losses
        model_.compute_discriminator_losses = model.compute_discriminator_losses
        model_.generator = generator
        model_.discriminator = discriminator
        model = model_.to(device)

    # setting up optimizers
    toggle_grad(discriminator, False, toggle_embedding=config.cub_sent_trainable_embedding)
    toggle_grad(generator, True, toggle_embedding=config.cub_sent_trainable_embedding)
    g_params = list(filter(lambda p: p.requires_grad, generator.parameters()))

    toggle_grad(discriminator, True, toggle_embedding=config.cub_sent_trainable_embedding)
    toggle_grad(generator, False, toggle_embedding=config.cub_sent_trainable_embedding)
    d_params = list(filter(lambda p: p.requires_grad, discriminator.parameters()))

    if config.optimizer == 'rmsprop':
        optimizer_gener = optim.RMSprop(g_params, lr=config.gener_lr, alpha=0.99, eps=1e-8)
        optimizer_discr = optim.RMSprop(d_params, lr=config.discr_lr, alpha=0.99, eps=1e-8)

    else:
        optimizer_gener = optim.Adam(g_params,
                                     lr=config.gener_lr,
                                     betas=(config.beta1, config.beta2),
                                     amsgrad=True)
        optimizer_discr = optim.Adam(d_params,
                                     lr=config.discr_lr,
                                     betas=(config.beta1, config.beta2),
                                     amsgrad=True)

    # exponential moving average for generator
    # ema = ExponentialMovingAverage(model.generator, config.ema_decay, skeleton=not config.use_ema).register()  # custom
    toggle_grad(generator, True, toggle_embedding=config.cub_sent_trainable_embedding)
    ema = ExponentialMovingAverage(generator.parameters(), decay=config.ema_decay)  # pytorch_ema library

    trainer = Trainer(config,
                      model, ema,
                      optimizer_gener, optimizer_discr,
                      train_loader, train_loaders_marginal,
                      run_path, device)
    evaluator = Evaluator(config,
                          model, ema,
                          train_loader, valid_loader, test_loader,
                          run_path, device)

    print('-' * 89)
    with Timer('Adversarially trained Wyner model'):
        agg = defaultdict(list)
        if config.evaluate_only:
            config.n_epochs = 1
            if config.path_to_saved_model:
                model.load_state_dict(torch.load('{}/model.rar'.format(config.path_to_saved_model), map_location=device))
                ema.load_state_dict(torch.load('{}/ema.rar'.format(config.path_to_saved_model), map_location=device))

        best_ref_value = -np.inf
        nan_flag = 0
        for epoch in range(1, config.n_epochs + 1):
            if not config.evaluate_only:
                if 'cub' in config.dataset and config.cub_sent_output_embedding_during_training:
                    generator.decoders['y'].to_data.output_embedding = True
                with Timer('epoch {:03d} training'.format(epoch)):
                    nan_flag = trainer(epoch, agg)

            if nan_flag != 1:
                if 'cub' in config.dataset:
                    generator.decoders['y'].to_data.output_embedding = False

                if not config.do_not_evaluate:
                    # at the end of each epoch
                    # if nan_flag encountered, skip evaluation
                    evaluator(epoch, agg)

                if config.ref_value_key in ['test_mAP@all', 'test_P@K', 'cca_j', 'cca_cx2y', 'cca_cy2x']:
                    verb = 'increased'
                    sign = +1
                else:
                    verb = 'decreased'
                    sign = -1

                if not agg[config.ref_value_key]:
                    print('epoch {:03d} saving checkpoint...\n'.format(epoch))
                    agg['checkpoint_epochs'].append(epoch)
                    trainer.save_checkpoint(config.save_optim)
                else:
                    if sign * agg[config.ref_value_key][-1] > best_ref_value:
                        print('epoch {:03d} {} {} to {} (best so far={})\n'.format(
                            epoch, config.ref_value_key, verb, agg[config.ref_value_key][-1], sign * best_ref_value))
                        best_ref_value = sign * agg[config.ref_value_key][-1]
                        print('\tsaving checkpoint...\n')
                        agg['checkpoint_epochs'].append(epoch)
                        trainer.save_checkpoint(config.save_optim)

            save_vars(agg, '{}/history.rar'.format(run_path), safe=False)
            if nan_flag == 1 and (not config.continue_training_after_nan):
                break


if __name__ == '__main__':
    config = Options().parse()
    main(config)
