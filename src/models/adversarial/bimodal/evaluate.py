import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from datasets.dataloaders import get_dataloaders_mnist, get_dataloaders_svhn, get_dataloaders_mnistsvhn, \
    get_dataloaders_mnist_by_label, get_dataloaders_svhn_by_label
from datasets.sketchy import SketchyRetrieval
from datasets.utils import unpack_data, unpack_data_for_classifier
from external.BDMC.ais import ais
from external.pytorch_fid.fid_score import compute_frechet_distance
from helpers.cub import CUBImgHelper, CUBHelper, CUBCanonicalCorrelationAnalysis, CUBSentHelper
from helpers.generic import EarlyStopping, Timer, repeat_interleave
from helpers.get_reference_samples import get_reference_samples_for_plot, get_reference_samples_by_digit
from helpers.plot import save_summary_joint_image_samples, \
    save_summary_conditional_image_samples, save_summary_stylish_image_samples, save_grid_two_image_samples
from helpers.training import toggle_grad
from models.adversarial.bimodal.model import IMAGE_DATASETS
from models.adversarial.bimodal.train import test
from models.classifiers.model import LatentClassifier


class Evaluator:
    def __init__(self, config,
                 model, ema: ExponentialMovingAverage,
                 train_loader, valid_loader, test_loader,
                 run_path, device):
        self.config = config
        self.model = model
        self.ema = ema
        self.train_loader = train_loader  # used in CUB-CCA
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.turn_off_additional_gaussian_noise = config.turn_off_additional_gaussian_noise_during_evaluation
        self.do_not_test = config.do_not_test
        self.do_not_save_samples = config.do_not_save_samples
        self.do_not_classify_samples = config.do_not_classify_samples
        self.do_evaluate_likelihoods = config.do_evaluate_likelihoods
        self.do_not_classify_from_representations = config.do_not_classify_from_representations
        self.do_not_evaluate_fd_scores = config.do_not_evaluate_fd_scores
        self.call_eval_during_inference = config.call_eval_during_inference

        self.run_path = run_path
        self.device = device

        self.dataset = self.config.dataset
        self.evaluation_mode = self.config.evaluation_mode
        self.generator = model.generator
        # initialize early_stopping object
        self.early_stopping = EarlyStopping(patience=config.patience,
                                            verbose=True,
                                            path=run_path,
                                            save=False)
        if self.dataset in IMAGE_DATASETS + ['cub', 'cub-imgft2sent']:
            self.ref_samples_for_plot, self.ref_samples_for_cond_fd = self.get_reference_samples()

        self.sketchy_retrieval = None
        if self.dataset == 'sketchy-vgg':
            self.sketchy_retrieval = SketchyRetrieval(test_loader,
                                                      n_images_to_save=config.n_retrievals_to_save,
                                                      n_retrievals=config.n_retrievals,
                                                      metric=config.sketchy_retrieval_metric,
                                                      run_path=run_path,
                                                      device=device)

        self.cub_cca = None
        if self.dataset in ['cub', 'cub-imgft2sent']:
            self.cub_imgft_helper = CUBImgHelper('imgft' in config.dataset, config.main_path, device)

            # vocab related parameters
            min_occur = 3
            max_sentence_length = 32  # max length of any description for birds dataset

            # the following parameters are for training word2vec embedding;
            # please note that these are independent from the language model we actually train.
            embedding_dim = 300
            window_length = 3

            self.cub_sent_helper = CUBSentHelper(config.main_path, max_sentence_length, min_occur, embedding_dim,
                                                 window_length, reset=False, device=device)
            self.cub_cca = CUBCanonicalCorrelationAnalysis(train_loader, test_loader, self.cub_sent_helper,
                                                           config.batch_size, run_path, device=device)
            self.cub_cca_gt = 0.274576847006335  # self.cub_cca.compute_correlation(self.generator, mode='gt')

            self.cub_helper = CUBHelper(self.cub_imgft_helper, self.cub_sent_helper, config.main_path)

    def __call__(self, epoch, agg):
        self.plot_history(agg, save=True, path=self.run_path)

        toggle_grad(self.model, True, toggle_embedding=self.config.cub_sent_trainable_embedding)  # required to be consistent with ema.average_parameters() below
        with self.ema.average_parameters():
            self.eval()
            if not self.do_not_save_samples and self.dataset in IMAGE_DATASETS + ['cub', 'cub-imgft2sent']:
                with Timer('epoch {:03d} generating and saving samples'.format(epoch)):
                    with torch.no_grad():
                        self.generate_and_save_samples(filename_suffix=('e{:03d}'.format(epoch)
                                                                        if not self.config.evaluate_only else 'eval'))

            if (epoch % self.config.evaluation_freq == 0) or epoch == self.config.n_epochs:
                with Timer('epoch {:03d} evaluating'.format(epoch)):
                    if not self.do_not_test:
                        with Timer('epoch {:03d} testing'.format(epoch)):
                            with torch.no_grad():
                                test(self.model,
                                     self.test_loader,
                                     self.dataset,
                                     self.early_stopping,
                                     epoch, agg,
                                     self.device, mode='test')

                    if self.do_evaluate_likelihoods:
                        with Timer('epoch {:03d} evaluating likelihoods'.format(epoch)):
                            self.evaluate_likelihoods(agg)

                    if not self.do_not_classify_from_representations and self.dataset in ['mnist-mnist', 'mnist-svhn']:
                        if (epoch % self.config.latent_classification_freq == 0) or epoch == self.config.n_epochs:
                            with Timer('epoch {:03d} classifying latents'.format(epoch)):
                                self.classify_from_representation(agg)

                    with torch.no_grad():
                        if not self.do_not_classify_samples and self.dataset in ['mnist-mnist', 'mnist-svhn']:
                            with Timer('epoch {:03d} classifying samples'.format(epoch)):
                                self.classify_samples(agg)

                        if not self.do_not_evaluate_fd_scores and self.dataset in ['mnist-mnist', 'mnist-svhn']:
                            with Timer('epoch {:03d} computing FD scores'.format(epoch)):
                                self.compute_fd(agg)

                        if self.dataset == 'sketchy-vgg':
                            with Timer('epoch {:03d} evaluating Sketchy retrieval'.format(epoch)):
                                self.evaluate_sketchy_retrieval(epoch, agg)

                        if self.dataset == 'cub-imgft2sent':
                            self.analyze_cub_cca(agg)

            self.train()

    def eval(self):
        if self.call_eval_during_inference:
            self.model.eval()
        if self.turn_off_additional_gaussian_noise:
            self.generator.encoder.set_additional_gaussian_noise(False)

    def train(self):
        if self.call_eval_during_inference:
            self.model.train()
        if self.turn_off_additional_gaussian_noise:
            self.generator.encoder.set_additional_gaussian_noise(True)

    def get_reference_samples(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # draw reference (conditioning) samples for visualization
        print("# Drawing reference samples for visualization...", end=" ")
        xref, xref_label, yref, yref_label = get_reference_samples_for_plot(self.dataset,
                                                                            self.config.n_per_class_for_plot,
                                                                            self.test_loader,
                                                                            self.device)

        # draw fixed latent samples for joint generation
        zp_fixed_joint = self.generator.draw_from_prior(n=self.config.n_for_plot, device=self.device)[0]
        zp_fixed_joint = repeat_interleave(zp_fixed_joint, dim=0, n_tile=self.config.n_per_ref_for_plot)
        up_fixed_joint, vp_fixed_joint = self.generator.draw_from_prior(
            n=self.config.n_per_ref_for_plot, device=self.device)[1:]
        up_fixed_joint = up_fixed_joint.repeat(self.config.n_for_plot, 1)
        vp_fixed_joint = vp_fixed_joint.repeat(self.config.n_for_plot, 1)

        # draw fixed latent samples for conditional generation
        up_fixed_cond, vp_fixed_cond = self.generator.draw_from_prior(
            n=self.config.n_per_ref_for_plot, device=self.device)[1:]
        up_fixed_cond = up_fixed_cond.repeat(len(xref), 1)
        vp_fixed_cond = vp_fixed_cond.repeat(len(xref), 1)

        ref_samples_for_plot = (xref, yref, zp_fixed_joint, up_fixed_joint, vp_fixed_joint, up_fixed_cond, vp_fixed_cond)
        print("done.")

        # draw reference (conditioning) samples for FD
        print("# Drawing reference samples for FD score evaluation...", end=" ")
        ref_samples_for_cond_fd = get_reference_samples_by_digit(self.test_loader,
                                                                 self.config.n_per_class_for_cond_fd,
                                                                 self.device) \
            if self.dataset in ['mnist-mnist', 'mnist-svhn'] else None
        print("done.")

        return ref_samples_for_plot, ref_samples_for_cond_fd

    @staticmethod
    def plot_history(agg, show=False, save=False, path='..', dpi=50):
        dists = ['Variational',
                 'Joint',
                 r'Conditional ($x\to y$)',
                 r'Conditional ($y\to x$)',
                 r'Conditional ($x\leftrightarrow y$)']

        dist_loss_keys = [[None, 'j', 'cx2y', 'cy2x', 'c'],
                          ['enc_ci', 'j_ci', 'cx2y_ci', 'cy2x_ci', None],
                          # ['enc_li_u', 'j_li_u', None, 'cy2x_li_u', None],
                          # ['enc_li_v', 'j_li_v', 'cx2y_li_v', None, None],
                          # [None, 'j_agg', 'cx2y_agg', 'cy2x_agg', None],
                          ]
        rec_loss_keys = [[None, 'j_rec_x', 'cy2x_rec_x', 'c_rec_x', None],
                         [None, 'j_rec_y', 'cx2y_rec_y', 'c_rec_y', None],
                         # [None, 'j_rec_z', 'cx2y_rec_z', 'cy2x_rec_z', None],
                         ]
        loss_keys = dist_loss_keys + rec_loss_keys

        configs = {('gener', 'train'): 'bo-', ('gener', 'test'): 'bs--',
                   ('discr', 'train'): 'rx-', ('discr', 'test'): 'r*--'}

        fig, axes = plt.subplots(nrows=len(loss_keys),
                                 ncols=5,  # no. titles
                                 sharex='all',
                                 figsize=(20, 3 * len(loss_keys)),
                                 constrained_layout=False)
        for i, loss_keys in enumerate(loss_keys):
            for j, dist in enumerate(dists):
                ax = axes[i][j]
                if loss_keys[j] is None:
                    ax.axis('off')
                    continue
                choices = ['gener', 'discr'] if i < len(dist_loss_keys) else ['gener']
                for gener_or_discr in choices:
                    ax_ = axes[i][j].twinx() if gener_or_discr == 'discr' else ax
                    for l, train_or_test in enumerate(['train', 'test']):
                        loss_key = '{}_loss_{}_{}'.format(train_or_test, gener_or_discr, loss_keys[j])
                        label = '{}_{}'.format('g' if gener_or_discr=='gener' else 'd', train_or_test) \
                            if i < len(dist_loss_keys) else train_or_test
                        ax_.plot(agg[loss_key], configs[gener_or_discr, train_or_test], label=label)
                    if gener_or_discr == 'gener':
                        ax_.set_ylabel('generator' if i < len(dist_loss_keys) else 'reconstruction', color='blue')
                        ax_.legend(loc='upper left')
                    else:
                        ax_.set_ylabel('discriminator', color='red')
                        ax_.legend(loc='upper right')

                    ax_.grid(True)
                    ax_.set_xlabel('epoch')
                    ax_.set_title(loss_keys[j])

        fig.tight_layout()

        if save:
            plt.savefig(path + '/history.png', bbox_inches='tight', pad_inches=0., dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close()

    # sample quality evaluation by visualization
    def generate_and_save_samples(self, filename_suffix):
        """
        Generate samples from Wyner type models and save some summaries of those

        Parameters
        ----------
        filename_suffix

        Returns
        -------
        None
        """
        config = self.config
        run_path = self.run_path

        generator = self.generator
        encoder, decoders = generator.encoder, generator.decoders

        output_filetype = config.output_filetype
        dataset = config.dataset
        evaluation_mode = config.evaluation_mode
        implicit = config.implicit
        plot_reconstruction = config.plot_reconstruction
        plot_stochastic_reconstruction = config.plot_stochastic_reconstruction

        dim_z = config.dim_z
        dim_u = config.dim_u
        dim_v = config.dim_v

        nz = config.n_for_plot
        n_per_ref = config.n_per_ref_for_plot

        ref_samples_for_plot = self.ref_samples_for_plot
        xref, yref, zp_fixed_joint, up_fixed_joint, vp_fixed_joint, up_fixed_cond, vp_fixed_cond = ref_samples_for_plot
        ncond = len(xref)
        nref = len(xref)

        if evaluation_mode in ['all', 'j'] and \
                'x' in decoders and 'y' in decoders:
            # Joint generation
            filepath = '{}/j_{}.{}'.format(run_path, filename_suffix, output_filetype)
            if dataset in IMAGE_DATASETS + ['cub', 'cub-imgft2sent']:
                xp, yp = generator.decode_zuv_xy(zp_fixed_joint, up_fixed_joint, vp_fixed_joint)
                xp = xp.reshape((nz, n_per_ref, *xp.shape[1:]))
                yp = yp.reshape((nz, n_per_ref, *yp.shape[1:]))
                if dataset in IMAGE_DATASETS:
                    save_summary_joint_image_samples(samples_x=xp, samples_y=yp, filepath=filepath)
                else:  # cub-imgft2sent
                    self.cub_helper.save_joint_samples(xp, yp, filepath)

        if evaluation_mode in ['all', 'c', 'cx2y']:
            if dataset in IMAGE_DATASETS + ['cub', 'cub-imgft2sent'] and \
                    'y' in decoders:
                # Conditional generation (x->y)
                filepath = '{}/cx2y_{}.{}'.format(run_path, filename_suffix, output_filetype)
                zqx = encoder.marginal_encoders['x'].encode_x_z(xref)  # (ncond, dim_z), (ncond, dim_u)
                zqx_repeat = repeat_interleave(zqx, n_tile=n_per_ref, dim=0)  # (ncond * n_per_ref, dim_x)
                ypx = decoders['y'](zqx_repeat, vp_fixed_cond)
                ypx = ypx.reshape((ncond, n_per_ref, *ypx.shape[1:]))

                if dataset in IMAGE_DATASETS:
                    save_summary_conditional_image_samples(conditions=xref,
                                                           samples=ypx,
                                                           filepath=filepath)
                if dataset in ['cub', 'cub-imgft2sent']:
                    self.cub_helper.save_conditional_samples(conditions=xref, samples=ypx,
                                                             from_to='x2y', filepath=filepath,
                                                             ground_truths=yref,
                                                             n_refs=len(xref), n_samples=ypx.shape[1])

                if dim_v > 0 and (not implicit):
                    # Stylistic conditional generation (x->y)
                    filepath = '{}/sx2y_{}.{}'.format(run_path, filename_suffix, output_filetype)

                    # infer z from x: we can recycle zqx from conditional generation
                    zqx_repeat = repeat_interleave(zqx, n_tile=nref, dim=0)  # (ncond * nref, dim_x)
                    # extract style v from y
                    yref_repeat = yref.repeat((ncond, *([1] * len(yref.shape[1:]))))
                    vqyref = encoder.marginal_encoders['y'].encode_zx_u(zqx_repeat, yref_repeat)
                    # mix and match
                    ypx_style = decoders['y'](zqx_repeat, vqyref)
                    ypx_style = ypx_style.reshape((ncond, nref, *ypx_style.shape[1:]))

                    if dataset in IMAGE_DATASETS:
                        save_summary_stylish_image_samples(conditions=xref,
                                                           references=yref,
                                                           samples=ypx_style,
                                                           filepath=filepath)

                    if dataset in ['cub', 'cub-imgft2sent']:
                        pass

        if evaluation_mode in ['all', 'c', 'cy2x']:
            if dataset in IMAGE_DATASETS + ['cub', 'cub-imgft2sent'] and \
                    'x' in decoders and 'y' in encoder.marginal_encoders:
                # Conditional generation (y->x)
                filepath = '{}/cy2x_{}.{}'.format(run_path, filename_suffix, output_filetype)
                zqy = encoder.marginal_encoders['y'].encode_x_z(yref)  # (ncond, dim_z), (ncond, dim_u)
                zqy_repeat = repeat_interleave(zqy, n_tile=n_per_ref, dim=0)  # (ncond * n_per_ref, dim_x)
                xpy = decoders['x'](zqy_repeat, up_fixed_cond)
                xpy = xpy.reshape((ncond, n_per_ref, *xpy.shape[1:]))

                if dataset in IMAGE_DATASETS:
                    save_summary_conditional_image_samples(conditions=yref,
                                                           samples=xpy,
                                                           filepath=filepath)

                if dataset in ['cub', 'cub-imgft2sent']:
                    self.cub_helper.save_conditional_samples(conditions=yref, samples=xpy,
                                                             from_to='y2x', filepath=filepath,
                                                             ground_truths=xref,
                                                             n_refs=len(yref), n_samples=xpy.shape[1])

                if dim_u > 0 and (not implicit):
                    # Stylistic conditional generation (y->x)
                    filepath = '{}/sy2x_{}.{}'.format(run_path, filename_suffix, output_filetype)

                    # infer z from y: we can recycle zqy from conditional generation
                    zqy_repeat = repeat_interleave(zqy, n_tile=nref, dim=0)  # (ncond * nref, dim_x)
                    # extract style u from x
                    xref_repeat = xref.repeat((ncond, *([1] * len(xref.shape[1:]))))
                    uqxref = encoder.marginal_encoders['x'].encode_zx_u(zqy_repeat, xref_repeat)
                    # mix and match
                    xpy_style = decoders['x'](zqy_repeat, uqxref)
                    xpy_style = xpy_style.reshape((ncond, nref, *xpy_style.shape[1:]))

                    if dataset in IMAGE_DATASETS:
                        save_summary_stylish_image_samples(conditions=yref,
                                                           references=xref,
                                                           samples=xpy_style,
                                                           filepath=filepath)

                    if dataset in ['cub', 'cub-imgft2sent']:
                        pass

        if plot_reconstruction and dataset in IMAGE_DATASETS + ['cub', 'cub-imgft2sent']:
            if evaluation_mode in ['all', 'j'] and \
                    'x' in decoders and 'y' in decoders:
                # Joint reconstruction
                zqxy = encoder.encode_xy_z(xref, yref)
                uqxy = encoder.marginal_encoders['x'].encode_zx_u(zqxy, xref)
                vqxy = encoder.marginal_encoders['y'].encode_zx_u(zqxy, yref)
                xref_hat, yref_hat = generator.decode_zuv_xy(zqxy, uqxy, vqxy)

                if dataset in IMAGE_DATASETS:
                    save_grid_two_image_samples(samples_x=xref,
                                                samples_y=xref_hat,
                                                filepath='{}/jrec_x_{}.{}'.format(run_path,
                                                                                  filename_suffix,
                                                                                  output_filetype))
                    save_grid_two_image_samples(samples_x=yref,
                                                samples_y=yref_hat,
                                                filepath='{}/jrec_y_{}.{}'.format(run_path,
                                                                                  filename_suffix,
                                                                                  output_filetype))

            if evaluation_mode in ['all', 'c', 'cx2y'] and \
                    'x' in decoders:
                # Marginal reconstructions
                filepath = '{}/mrec_x_{}.{}'.format(run_path, filename_suffix, output_filetype)
                xref_hatm = decoders['x'](*encoder.marginal_encoders['x'].encode_x_zu(xref))
                if dataset in IMAGE_DATASETS:
                    save_grid_two_image_samples(samples_x=xref,
                                                samples_y=xref_hatm,
                                                filepath=filepath)
                if dataset in ['cub', 'cub-imgft2sent']:
                    self.cub_helper.img_helper.save_imgft_recon(xref, xref_hatm, filepath)

            if evaluation_mode in ['all', 'c', 'cy2x'] and \
                    'y' in decoders:
                filepath = '{}/mrec_y_{}.{}'.format(run_path, filename_suffix, output_filetype)
                yref_hatm = decoders['y'](*encoder.marginal_encoders['y'].encode_x_zu(yref))
                if dataset in IMAGE_DATASETS:
                    save_grid_two_image_samples(samples_x=yref,
                                                samples_y=yref_hatm,
                                                filepath=filepath)
                if dataset in ['cub', 'cub-imgft2sent']:
                    self.cub_helper.sent_helper.save_sent_recon(yref, yref_hatm, filepath)

        if plot_stochastic_reconstruction and dataset in IMAGE_DATASETS:
            # Joint stylistic reconstruction
            if evaluation_mode in ['all', 'j'] and \
                    'x' in decoders and 'y' in decoders:
                if dim_u > 0 and dim_v > 0 and (not implicit):
                    zqxy = encoder.encode_xy_z(xref, yref)
                    zqxy_ = zqxy.unsqueeze(dim=1).repeat((1, n_per_ref, 1)).reshape((-1, dim_z))
                    xrec, yrec = generator.decode_zuv_xy(zqxy_, up_fixed_cond, vp_fixed_cond)
                    xrec = xrec.reshape((ncond, n_per_ref, *xrec.shape[1:]))
                    yrec = yrec.reshape((ncond, n_per_ref, *yrec.shape[1:]))

                    if dataset in IMAGE_DATASETS:
                        save_summary_conditional_image_samples(conditions=xref,
                                                               samples=xrec,
                                                               filepath='{}/srec_jx_{}.{}'.format(run_path,
                                                                                                  filename_suffix,
                                                                                                  output_filetype))
                        save_summary_conditional_image_samples(conditions=yref,
                                                               samples=yrec,
                                                               filepath='{}/srec_jy_{}.{}'.format(run_path,
                                                                                                  filename_suffix,
                                                                                                  output_filetype))

    # likelihood evaluation
    def evaluate_likelihoods(self, agg):
        # annealing schedule
        # \b_t for mixing f_1 and f_T to define f_t = (f_1)^{1-\b_t} f_T^{\b_t}
        forward_schedule = np.linspace(0., 1., self.config.chain_length)

        if self.evaluation_mode in ['all', 'j']:
            # run joint AIS
            joint_loader_for_ais = self.prepare_ais_data_joint(use_encoder=True)
            _, lower_bounds_joint = ais(self.generator,
                                        joint_loader_for_ais,  # (xy, zuv)
                                        forward_schedule=forward_schedule,
                                        n_importance_sample=self.config.iwae_samples,
                                        likelihood=self.config.likelihood,
                                        likelihood_var=self.config.likelihood_var,
                                        use_encoder=True,
                                        posterior_var=self.config.posterior_var,
                                        forward=True,
                                        n_batches=self.config.n_batches_for_ais,
                                        device=self.device)
            print('Likelihood (j): {:.4f}'.format(lower_bounds_joint))
            agg['likelihood_j'].append(lower_bounds_joint)

        if self.evaluation_mode in ['all', 'c', 'cx2y']:
            # run conditional AIS
            cond_loader_for_ais_y = self.prepare_ais_data_conditional(from_to='x2y')
            marginal_decoder_y = self.generator.decoders['y']
            _, lower_bounds_cond_y = ais(marginal_decoder_y,
                                         cond_loader_for_ais_y,  # (y, zv)
                                         forward_schedule=forward_schedule,
                                         n_importance_sample=self.config.iwae_samples,
                                         likelihood=self.config.likelihood,
                                         likelihood_var=self.config.likelihood_var,
                                         use_encoder=True,
                                         posterior_var=self.config.posterior_var,
                                         forward=True,
                                         n_batches=self.config.n_batches_for_ais,
                                         device=self.device)
            print('Likelihood (cx2y): {:.4f}'.format(lower_bounds_cond_y))
            agg['likelihood_cx2y'].append(lower_bounds_cond_y)

        if self.evaluation_mode in ['all', 'c', 'cy2x']:
            cond_loader_for_ais_x = self.prepare_ais_data_conditional(from_to='y2x')
            marginal_decoder_x = self.generator.decoders['x']
            _, lower_bounds_cond_x = ais(marginal_decoder_x,
                                         cond_loader_for_ais_x,  # (x, zu)
                                         forward_schedule=forward_schedule,
                                         n_importance_sample=self.config.iwae_samples,
                                         likelihood=self.config.likelihood,
                                         likelihood_var=self.config.likelihood_var,
                                         use_encoder=True,
                                         posterior_var=self.config.posterior_var,
                                         forward=True,
                                         n_batches=self.config.n_batches_for_ais,
                                         device=self.device)
            print('Likelihood (cy2x): {:.4f}'.format(lower_bounds_cond_x))
            agg['likelihood_cy2x'].append(lower_bounds_cond_x)

    def prepare_ais_data_joint(self, use_encoder=True):
        """Simulate data from a given joint encoder model. Sample from the
        joint distribution q(x,y)q(z|x,y)q(u|z,x)q(v|z,y) if use_encoder is True, else q(x,y)p(z)p(u)p(v)

        Args:
            use_encoder

        Returns:
            iterator that loops over batches of torch Tensor pair x, z
        """
        loader_iterator = iter(self.test_loader)
        num_iters = int(np.ceil(len(self.test_loader.dataset) / self.test_loader.batch_size))

        batches = []
        for i in range(1, num_iters + 1):
            try:
                dataT = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(self.test_loader)
                dataT = next(loader_iterator)
            data = unpack_data(dataT, device=self.device, dataset=self.dataset)

            xq, yq = data[0], data[1]
            xyq = torch.cat([xq.view(xq.shape[0], -1), yq.view(yq.shape[0], -1)], dim=-1)
            if use_encoder:
                z = self.generator.encoder.encode_xy_z(xq, yq)
                u = self.generator.encoder.marginal_encoders['x'].encode_zx_u(z, xq)
                v = self.generator.encoder.marginal_encoders['y'].encode_zx_u(z, yq)
            else:
                z, u, v = self.generator.draw_from_prior(n=xq.shape[0], device=xq.device)

            zuv = torch.cat([z, u, v], dim=-1)
            paired_batch = (xyq, zuv)
            batches.append(paired_batch)

        return iter(batches)

    def prepare_ais_data_conditional(self, from_to='x2y'):
        """Simulate data from a given joint encoder model. Sample from the
        joint distribution q(x,y)q(z|x,y)q(u|z,x)q(v|z,y).

        Args:
            from_to

        Returns:
            iterator that loops over batches of torch Tensor pair x, z
        """
        assert from_to in ['x2y', 'y2x']

        loader_iterator = iter(self.test_loader)
        num_iters = int(np.ceil(len(self.test_loader.dataset) / self.test_loader.batch_size))

        batches = []
        for i in range(1, num_iters + 1):
            try:
                dataT = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(self.test_loader)
                dataT = next(loader_iterator)
            data = unpack_data(dataT, device=self.device, dataset=self.dataset)
            if from_to == 'x2y':
                xq, yq = data[0], data[1]
                vp = self.generator.priors['v'].draw(xq.shape[0], device=self.device)
            elif from_to == 'y2x':
                xq, yq = data[1], data[0]
                vp = self.generator.priors['u'].draw(xq.shape[0], device=self.device)
            else:
                raise ValueError
            ypx, zqx = self.generator.encode_xv_yz(xq, vp, from_to)
            zv = torch.cat([zqx, vp], dim=-1)
            paired_batch = (yq, zv)
            batches.append(paired_batch)

        return iter(batches)

    # sample quality evaluation by classification
    def classify_samples(self, agg):
        assert self.dataset in ['mnist-mnist', 'mnist-svhn']
        from models.classifiers.model import MNISTClassifier
        from models.classifiers.model import SVHNClassifier
        dirname = os.path.dirname(__file__)

        mnist_classifier = MNISTClassifier().to(self.device)
        mnist_classifier.load_state_dict(
            torch.load(os.path.join(dirname, '../../../../pretrained/classifiers/mnist.rar'), map_location=self.device))
        svhn_classifier = SVHNClassifier().to(self.device)
        svhn_classifier.load_state_dict(
            torch.load(os.path.join(dirname, '../../../../pretrained/classifiers/svhn.rar'), map_location=self.device))

        classifier_x = mnist_classifier
        classifier_y = mnist_classifier if self.dataset == 'mnist-mnist' else svhn_classifier
        classifier_x.eval()
        classifier_y.eval()

        if self.evaluation_mode in ['all', 'j'] and \
                'x' in self.generator.decoders and 'y' in self.generator.decoders:
            err_j = _classify_j(self.dataset,
                                self.generator,
                                classifier_x, classifier_y,
                                self.config.n_samples_for_joint_eval,
                                self.config.batch_size,
                                # self.config.n_batch_for_joint_eval,  # FIXME
                                device=self.device)
            agg['err_j'].append(err_j)

        if self.evaluation_mode in ['all', 'c', 'cx2y'] and \
                'x' in self.generator.encoder.marginal_encoders and 'y' in self.generator.decoders:
            err_cx2y = _classify_c(self.dataset,
                                   self.generator,
                                   classifier_y,
                                   self.test_loader,
                                   n_per_ref=1,
                                   from_to='x2y',
                                   device=self.device)
            agg['err_cx2y'].append(err_cx2y)

        if self.evaluation_mode in ['all', 'c', 'cy2x'] and \
                'y' in self.generator.encoder.marginal_encoders and 'x' in self.generator.decoders:
            err_cy2x = _classify_c(self.dataset,
                                   self.generator,
                                   classifier_x,
                                   self.test_loader,
                                   n_per_ref=1,
                                   from_to='y2x',
                                   device=self.device)
            agg['err_cy2x'].append(err_cy2x)

    # latent representation quality evaluation by classification
    def classify_from_representation(self, agg):
        assert self.dataset in ['mnist-mnist', 'mnist-svhn']
        epochs = self.config.n_epochs_for_latent_classification

        agg['err_xy2z'].append(_classify_from_representation(self.config.dim_z,
                                                             self.generator,
                                                             epochs,
                                                             self.valid_loader,
                                                             self.test_loader,
                                                             mode='xy2z',
                                                             device=self.device))

        if self.config.lambda_cx2y > 0:
            agg['err_x2z'].append(_classify_from_representation(self.config.dim_z,
                                                                self.generator,
                                                                epochs,
                                                                self.valid_loader,
                                                                self.test_loader,
                                                                mode='x2z',
                                                                device=self.device))

        if self.config.lambda_cy2x > 0:
            agg['err_y2z'].append(_classify_from_representation(self.config.dim_z,
                                                                self.generator,
                                                                epochs,
                                                                self.valid_loader,
                                                                self.test_loader,
                                                                mode='y2z',
                                                                device=self.device))

    def compute_fd(self, agg):
        # Warning: all samples/dataloaders fed into FD score computation routines must be [0,1]-valued
        xyref_path, xyref_loader, xref_path, xref_paths_by_label, yref_path, yref_paths_by_label, \
        xref_loader, xref_loaders_by_label, yref_loader, yref_loaders_by_label = \
            get_reference_for_fd(
                self.config, self.dataset,
                self.config.fd_model_types, self.config.fd_feature_dims,
                self.config.main_path, self.device)
        xref_images_by_digit, yref_images_by_digit = self.ref_samples_for_cond_fd

        def compute_joint_fd():
            fd_j = compute_frechet_distance(
                [xyref_path, None],
                [xyref_loader, self.generator.get_joint_sample_generator(n_samples=self.config.n_samples_for_joint_eval,
                                                                         batch_size=self.config.batch_size,
                                                                         device=self.device,
                                                                         postprocessor=lambda x: (x + 1) / 2)],
                model_type='mnist_svhn_ae',
                dims=0,
                batch_size=self.config.batch_size,
                device=self.device,
                num_workers=self.config.num_workers)

            return fd_j

        def compute_marginal_fd(var, ref_path, ref_loader):
            ix = 0 if var == 'x' else 1
            fd_m = compute_frechet_distance(
                [ref_path, None],
                [ref_loader, self.generator.get_marginal_sample_generator(var,
                                                                          n_samples=self.config.n_samples_for_joint_eval,
                                                                          batch_size=self.config.batch_size,
                                                                          device=self.device,
                                                                          postprocessor=lambda x: (x + 1) / 2)],
                model_type=self.config.fd_model_types[ix],
                dims=self.config.fd_feature_dims[ix],
                batch_size=self.config.batch_size,
                device=self.device,
                num_workers=self.config.num_workers)

            return fd_m

        def compute_cond_fd(from_to):
            cond_ref_images_by_digit = xref_images_by_digit if from_to == 'x2y' else yref_images_by_digit
            target_ref_paths_by_label = yref_paths_by_label if from_to == 'x2y' else xref_paths_by_label
            target_ref_loaders_by_label = yref_loaders_by_label if from_to == 'x2y' else xref_loaders_by_label
            iy = 1 if from_to == 'x2y' else 0
            fd_by_digit = defaultdict(list)
            for digit in tqdm(range(10), desc="Calculating FD scores ({})".format(from_to)):
                for i, ref in enumerate(tqdm(cond_ref_images_by_digit[digit], leave=False)):
                    fd = compute_frechet_distance(
                        [target_ref_paths_by_label[digit], None],
                        [target_ref_loaders_by_label[digit],
                         self.generator.get_conditional_sample_generator(condition=ref,
                                                                         n_samples=self.config.n_per_ref_for_cond_fd,
                                                                         batch_size=self.config.batch_size,
                                                                         from_to=from_to,
                                                                         postprocessor=lambda x: (x + 1) / 2)],
                        model_type=self.config.fd_model_types[iy],
                        dims=self.config.fd_feature_dims[iy],
                        batch_size=self.config.batch_size,
                        device=self.device,
                        num_workers=self.config.num_workers)
                    fd_by_digit[digit].append(fd.item())

            fd = np.array(list(fd_by_digit.values()))

            return fd

        if self.evaluation_mode in ['all', 'j'] and \
                'x' in self.generator.decoders and 'y' in self.generator.decoders:
            if self.dataset == 'mnist-svhn':
                # Joint generation
                fd_j = compute_joint_fd()
                agg['fd_j'].append(fd_j)
                print("FD (j): {}".format(agg['fd_j'][-1]))

        if self.evaluation_mode in ['all', 'c', 'cx2y'] and \
                'x' in self.generator.encoder.marginal_encoders and 'y' in self.generator.decoders:
            # Conditional generation (x->y)
            fd_cx2y = compute_cond_fd('x2y')
            agg['fd_cx2y_raw'].append(fd_cx2y)
            agg['fd_cx2y_mean'].append(fd_cx2y.mean())
            print("FD (x2y): {}".format(fd_cx2y.mean()))

            # Marginal generation (y)
            fd_my = compute_marginal_fd('y', yref_path, yref_loader)
            agg['fd_my'].append(fd_my)
            print("FD (m,y): {}".format(agg['fd_my'][-1]))

        if self.evaluation_mode in ['all', 'c', 'cy2x'] and \
                'y' in self.generator.encoder.marginal_encoders and 'x' in self.generator.decoders:
            # Conditional generation (y->x)
            fd_cy2x = compute_cond_fd('y2x')
            agg['fd_cy2x_raw'].append(fd_cy2x)
            agg['fd_cy2x_mean'].append(fd_cy2x.mean())
            print("FD (y2x): {}".format(fd_cy2x.mean()))

            # Marginal generation (x)
            fd_mx = compute_marginal_fd('x', xref_path, xref_loader)
            agg['fd_mx'].append(fd_mx)
            print("FD (m,x): {}".format(agg['fd_mx'][-1]))

    def evaluate_sketchy_retrieval(self, epoch, agg):
        precision_Ks, average_precisions = self.sketchy_retrieval.evaluate(self.model, epoch,
                                                                           self.config.save_retrieved_images)
        agg['test_P@K'].append(precision_Ks.mean())
        agg['test_mAP@all'].append(average_precisions.mean())

    def analyze_cub_cca(self, agg):
        print("CCA:")
        print("\tground truth\t{:10.9f}".format(self.cub_cca_gt))

        if self.evaluation_mode == 'all':
            modes = ['j', 'cx2y', 'cy2x']
        elif self.evaluation_mode == 'c':
            modes = ['cx2y', 'cy2x']
        else:
            modes = [self.config.evaluation_mode]

        corrs = self.cub_cca.compute_correlation(self.generator, modes)
        if self.evaluation_mode in ['all', 'j']:
            agg['cca_j'].append(corrs['j'])
            print("\tjoint\t{:10.9f}".format(corrs['j']))

        if self.evaluation_mode in ['all', 'c', 'cx2y']:
            agg['cca_cx2y'].append(corrs['cx2y'])
            print("\timage->sentence\t{:10.9f}".format(corrs['cx2y']))

        if self.evaluation_mode in ['all', 'c', 'cy2x']:
            agg['cca_cy2x'].append(corrs['cy2x'])
            print("\tsentence->image\t{:10.9f}".format(corrs['cy2x']))


def _classify_j(dataset,
                generator,
                classifier_x, classifier_y,
                n_samples, batch_size,
                device='cuda'):
    """
    Test [mnist-mnist, mnist-svhn] models with pretrained classifiers

    Parameters
    ----------
    dataset
    generator
    classifier_x
    classifier_y

    Returns
    -------

    """
    assert dataset in ['mnist-mnist', 'mnist-svhn']
    correct_pred = 0.
    assert n_samples % batch_size == 0, "assumed for simplicity"
    n_batch = n_samples // batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batch), desc="Evaluating classification error of joint generation"):
            zp, up, vp = generator.draw_from_prior(n=batch_size, device=device)
            xp, yp = generator.decode_zuv_xy(zp, up, vp)

            if dataset.split('-')[0] == 'mnist':
                xp = xp[..., 2:30, 2:30]
            _, labelx_soft = classifier_x(xp)
            _, labelx_hard = torch.max(labelx_soft, 1)

            if dataset.split('-')[1] == 'mnist':
                yp = yp[..., 2:30, 2:30]
            _, labely_soft = classifier_y(yp)
            _, labely_hard = torch.max(labely_soft, 1)

            correct_pred += ((labelx_hard - labely_hard + 1).cpu().numpy().astype(int) % 10 == 0).sum()

    err = 1 - correct_pred / n_samples
    print("err_j={} (n={})".format(err, n_samples))
    return err


def _classify_c(dataset,
                generator,
                classifier,
                test_loader,
                n_per_ref=30,
                from_to='x2y',
                device='cuda'):
    ix, iy = (0, 1) if from_to == 'x2y' else (1, 0)
    target_dataset = dataset.split('-')[iy]
    correct_pred = 0.
    n = 0

    with torch.no_grad():
        for dataT in tqdm(test_loader,
                          desc="Evaluating classification error of conditional generation ({})".format(from_to)):
            x = dataT[ix][0].to(device)  # (B,)
            y_label = dataT[iy][1].to(device)  # (B,)
            for j in range(n_per_ref):
                yhat = generator.draw_conditional_samples(x, from_to=from_to)  # (B * nuv, *img_size)

                if target_dataset == 'mnist':
                    yhat = yhat[..., 2:30, 2:30]
                _, yhat_label_soft = classifier(yhat)
                _, yhat_label_hard = torch.max(yhat_label_soft, 1)
                yhat_label_hard = yhat_label_hard.view(x.shape[0])
                correct_pred += (y_label == yhat_label_hard).sum().item()
                n += x.shape[0]

    err = 1 - correct_pred / n
    print("err_c{}={:.3f} (total#={} / n_per_ref={})".format(from_to, err, n, n_per_ref))

    return err


def _classify_from_representation(dim_z, generator, epochs,
                                  train_loader, test_loader,
                                  mode, device=None):
    assert mode in ['xy2z', 'x2z', 'y2z']
    print("\nEvaluating latent classification error for {}...".format(mode))

    def get_latent_z(batch):
        if mode == 'xy2z':
            z = generator.encoder.encode_xy_z(*batch)
        elif mode == 'x2z':
            z = generator.encoder.marginal_encoders['x'].encode_x_z(batch)
        elif mode == 'y2z':
            z = generator.encoder.marginal_encoders['y'].encode_x_z(batch)
        else:
            raise ValueError
        return z

    classifier = LatentClassifier(dim_z, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for _ in tqdm(range(1, epochs + 1), desc="Training latent classifier"):  # loop over the dataset for multiple epochs
        for i, batch in enumerate(train_loader):
            # get the inputs
            batch, targets = unpack_data_for_classifier(batch, device, mode)
            with torch.no_grad():
                z = get_latent_z(batch)
            optimizer.zero_grad()
            probs = classifier(z)[1]
            loss = criterion(probs, targets)
            loss.backward()
            optimizer.step()

    classifier.eval()
    total = 0
    wrong = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Testing latent classifier')):
            batch, targets = unpack_data_for_classifier(data, device, mode)
            z = get_latent_z(batch)
            probs = classifier(z)[1]
            _, predicted = torch.max(probs, 1)
            total += targets.size(0)
            wrong += (predicted != targets).sum().item()

    err = wrong / total
    print('err_{}={:.3f} ({}/{})\n'.format(mode, err, wrong, total))
    return err


# fd score evaluation
def get_reference_for_fd(config, dataset, fd_model_types, fd_feature_dims, main_path, device):
    # Warning: all dataloaders must produce [0,1]-valued tensors
    fd_model_types = fd_model_types.copy()

    # in any case, dat_x == 'mnist'
    stat_npz_filenames = [None, None]
    for i in range(2):
        if fd_model_types[i] == 'inception':
            stat_npz_filenames[i] = '{}_stats_dims{}.npz'.format(fd_model_types[i], fd_feature_dims[i])
        else:
            assert dataset.split('-')[i] in fd_model_types[i], list(zip(dataset.split('-'), fd_model_types))
            stat_npz_filenames[i] = '{}_ae_stats.npz'.format(dataset.split('-')[i])

    mnist_loader = get_dataloaders_mnist(config.batch_size, shuffle=False, device=device, binarize=False,
                                         stretch=False, drop_last=False, root_path=main_path, size=32)[0]  # train split
    mnist_loaders_by_label = get_dataloaders_mnist_by_label(config.batch_size, device,
                                                            binarize=False,
                                                            root_path=main_path,
                                                            split='train',
                                                            size=32,
                                                            stretch=False)
    xref_path = '{}/data/MNIST/fd_stats/{}'.format(main_path, stat_npz_filenames[0])
    xref_loader = None
    if not os.path.exists(xref_path):
        xref_loader = mnist_loader
    xref_paths_by_label = {digit: '{}/data/MNIST/fd_stats/{}/{}'.format(
        main_path, digit, stat_npz_filenames[0]) for digit in range(10)}
    xref_loaders_by_label = {digit: None for digit in range(10)}
    for i in range(10):
        if not os.path.exists(xref_paths_by_label[i]):
            xref_loaders_by_label[i] = mnist_loaders_by_label[i]

    xyref_path = xyref_loader = None
    if dataset == 'mnist-mnist':
        yref_path = xref_path
        yref_loader = xref_loader
        yref_paths_by_label = {digit: xref_paths_by_label[(digit + 1) % 10] for digit in range(10)}
        yref_loaders_by_label = {digit: xref_loaders_by_label[(digit + 1) % 10] for digit in range(10)}
    elif dataset == 'mnist-svhn':
        svhn_loaders_by_label = get_dataloaders_svhn_by_label(config.batch_size, device,
                                                              root_path=main_path,
                                                              split='extra',
                                                              stretch=False)
        yref_path = '{}/data/SVHN/fd_stats/{}'.format(main_path, stat_npz_filenames[1])
        yref_loader = None
        if not os.path.exists(yref_path):
            svhn_loader = get_dataloaders_svhn(config.batch_size, shuffle=False, device=device,
                                               drop_last=False, train_split='extra',
                                               root_path=main_path,
                                               stretch=False)[0]
            yref_loader = svhn_loader
        yref_paths_by_label = {digit: '{}/data/SVHN/fd_stats/{}/{}'.format(
            main_path, (digit + 1) % 10, stat_npz_filenames[1]) for digit in range(10)}
        yref_loaders_by_label = {digit: None for digit in range(10)}
        for i in range(10):
            if not os.path.exists(yref_paths_by_label[i]):
                yref_loaders_by_label[i] = svhn_loaders_by_label[(i + 1) % 10]

        xyref_path = '{}/data/mnist-svhn/fd_stats/{}'.format(main_path, 'mnist_svhn_ae_stats.npz')
        xyref_loader = get_dataloaders_mnistsvhn(config.batch_size, shuffle=False, device=device,
                                                 binarize=False, root_path=main_path,
                                                 dc=config.dc, dm=config.dm, drop_last=False,
                                                 stretch=False)[0]
    else:
        raise ValueError

    return xyref_path, xyref_loader, xref_path, xref_paths_by_label, yref_path, yref_paths_by_label, \
        xref_loader, xref_loaders_by_label, yref_loader, yref_loaders_by_label
