import warnings
from collections import defaultdict
from functools import wraps

import torch
import torch.nn as nn

from datasets.cub import CUBImgFt
from datasets.sketchy import SketchyVGGDataLoader
from helpers.cub import CUBSentHelper
from helpers.generic import dotdict, repeat_interleave
from models.adversarial.losses import GENERATOR_LOSS_FUNCTIONS, DISCRIMINATOR_LOSS_FUNCTIONS, RECON_LOSS_FUNCTIONS
from models.adversarial.unimodal.model import MarginalEncoder, Decoder
from models.networks.base import weights_init, get_fc, RandomFunction
from models.networks.cub import get_networks as get_cub_networks
from models.networks.image2image import get_networks as get_image2image_networks
from models.networks.sketchy import get_networks as get_sketchy_networks

IMAGE_DATASETS = ['mnist-mnist', 'mnist-svhn', 'mnist-cdcb', 'mnist-multiply']


def add_epsilon(method):
    @wraps(method)
    def wrapper(self, *args):
        return method(self, *args) + self.epsilon

    return wrapper


class Discriminator(RandomFunction):
    r"""
    Model discriminators (4)
        Joint:                q(x,y)q(z|x,y)q(u|z,x)q(v|z,y)   vs. p(z)p(u)p(v)\d(x|z,u)\d(y|z,v)
        Conditional (xy->y):  q(x,y)q(z|x,y)q(v|z,y)           vs. q(x)q(z|x)p(v)\d(y|z,v)
        Conditional (xy->x):  q(x,y)q(z|x,y)q(u|z,x)           vs. q(y)q(z|y)p(u)\d(x|z,u)
        Cross:                q(x)q(z|x)q(u|z,x)p(v)\d(y|z,v)  vs. q(y)q(z|y)q(v|z,y)p(u)\d(x|z,u)

    Note: we ignore the following ``marginal'' discriminators
        Marginal (x->x):      q(x)q(z|x)q(u|z,x)               vs. p(z)p(u)\d(x|z,u)
        Marginal (y->y):      q(y)q(z|y)q(v|z,y)               vs. p(z)p(v)\d(y|z,v)

    Regularization discriminators (2)
        CI(enc):              q(x,y)q(z|x,y)                   vs. q(x,y)q(z)
        CI(joint):            p(z)p(x|z)p(y|z)                 vs. p(z)p(x,y)
        CI(x->y):             q(x)q(z|x)p(y|z)                 vs. q(x)\qt(y|x)\qt(z)
        CI(y->x):             q(y)q(z|y)p(x|z)                 vs. q(y)\qt(x|y)\qt(z)
    """
    def __init__(self,
                 dim_z=128 * 2,
                 dim_u=128,
                 dim_v=128,
                 to_feature_xy=None,
                 n_hidden_layers=2,
                 hidden_dim=1000,
                 negative_slope=0.2,
                 use_batchnorm=False,
                 use_batchnorm_first_layer=False,
                 batchnorm_kwargs=None,
                 use_spectralnorm=False,
                 noise_dim=100,
                 add_noise_channel=True,
                 additional_noise_sigma_x=None,
                 additional_noise_sigma_y=None,
                 turn_off_noise_x=False,
                 turn_off_noise_y=False,
                 additional_noise_sigma_z=0.,
                 additional_noise_sigma_uv=0.,
                 implicit=False,
                 marginalize_bottleneck=False,
                 additional_epsilon_to_ratio=0.,
                 loss_weights=None,
                 device=None):
        super(Discriminator, self).__init__(noise_dim, add_noise_channel)
        self.sigma_x = additional_noise_sigma_x
        self.sigma_y = additional_noise_sigma_y
        self.sigma_z = additional_noise_sigma_z
        self.sigma_uv = additional_noise_sigma_uv

        self.dim_z = dim_z
        self.implicit = implicit
        self.marginalize_bottleneck = marginalize_bottleneck
        assert not (self.implicit and self.marginalize_bottleneck)
        self.epsilon = additional_epsilon_to_ratio
        self.loss_weights = loss_weights

        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}

        if self.implicit:
            dim_u = dim_v = 0
        self.dim_u = dim_u
        self.dim_v = dim_v

        self.to_feature_xy = to_feature_xy
        feature_dim_xy = to_feature_xy.output_dim
        self.device = device

        dim_zuv = int(not marginalize_bottleneck) * dim_z + dim_u + dim_v
        dim_zu = int(not marginalize_bottleneck) * dim_z + dim_u
        dim_zv = int(not marginalize_bottleneck) * dim_z + dim_v

        self.networks = torch.nn.ModuleDict()

        self.turn_off_noise_x = turn_off_noise_x
        self.turn_off_noise_y = turn_off_noise_y

        def get_fc_input_to_hidden(input_dim):
            return get_fc(input_dim=input_dim + noise_dim,
                          hidden_dims=[],
                          output_dim=hidden_dim,
                          lrelu_negative_slope=negative_slope,
                          use_batchnorm=use_batchnorm,
                          use_batchnorm_first_layer=use_batchnorm_first_layer,
                          batchnorm_kwargs=batchnorm_kwargs,
                          use_spectralnorm=use_spectralnorm,
                          last_activation_type='lrelu')

        def get_fc_hidden_to_ratio(input_dim):
            return get_fc(input_dim=input_dim + noise_dim,
                          hidden_dim=hidden_dim,
                          n_hidden_layers=n_hidden_layers,
                          output_dim=1,
                          lrelu_negative_slope=negative_slope,
                          use_batchnorm=use_batchnorm,
                          batchnorm_kwargs=batchnorm_kwargs,
                          use_spectralnorm=use_spectralnorm)

        def get_fc_latent_to_ratio(latent_dim):
            return get_fc(input_dim=latent_dim,
                          hidden_dim=hidden_dim,
                          n_hidden_layers=n_hidden_layers,
                          output_dim=1,
                          lrelu_negative_slope=negative_slope,
                          use_batchnorm=use_batchnorm,
                          use_batchnorm_first_layer=use_batchnorm_first_layer,
                          batchnorm_kwargs=batchnorm_kwargs,
                          use_spectralnorm=use_spectralnorm)

        # Networks for joint / cross / marginal discrimination
        for key in ['j', 'cx2y', 'cy2x', 'c', 'mx', 'my']:
            if self.loss_weights[key] > 0:
                dim = dim_zuv if key in ['j', 'c', 'mx', 'my'] else (dim_zv if key == 'cx2y' else dim_zu)
                self.networks['{}_latent'.format(key)] = get_fc_input_to_hidden(dim)
                self.networks['{}_ratio'.format(key)] = get_fc_hidden_to_ratio(feature_dim_xy + hidden_dim)

        # Networks for common information discrimination
        if not self.marginalize_bottleneck:
            for key in ['j', 'cx2y', 'cy2x', 'var']:
                if self.loss_weights['{}_ci'.format(key)] >= 0:
                    self.networks['{}_ci_latent'.format(key)] = get_fc_input_to_hidden(dim_z)
                    self.networks['{}_ci_ratio'.format(key)] = get_fc_hidden_to_ratio(feature_dim_xy + hidden_dim)

        for key in ['cx2y', 'cy2x', 'c', 'var']:
            if self.loss_weights['{}_agg'.format(key)] > 0:
                self.networks['{}_agg_ratio'.format(key)] = get_fc_latent_to_ratio(dim_z)

    def get_random_feature_xy(self, x, y):
        if not self.turn_off_noise_x:
            x = self.add_gaussian_noise(x, self.sigma_x)
            x = self.append_noise_and_concat(x)
        if not self.turn_off_noise_y:
            y = self.add_gaussian_noise(y, self.sigma_y)
            y = self.append_noise_and_concat(y)

        hxy = self.to_feature_xy(x, y)  # (B, feature_dim)
        hxy = torch.flatten(hxy, start_dim=1)
        return hxy

    def add_noise_and_concat_zuv(self, z, u, v):
        z = self.add_gaussian_noise(z, self.sigma_z)
        u = self.add_gaussian_noise(u, self.sigma_uv)
        v = self.add_gaussian_noise(v, self.sigma_uv)
        if self.marginalize_bottleneck:
            zuv = self.append_noise_and_concat(u, v)
        elif self.implicit:
            zuv = self.append_noise_and_concat(z)
        else:  # explicit bottleneck
            zuv = self.append_noise_and_concat(z, u, v)

        return zuv

    def add_noise_and_concat_zu(self, z, u):
        # Get feature of (z,u,v) with additional noise
        z = self.add_gaussian_noise(z, self.sigma_z)
        u = self.add_gaussian_noise(u, self.sigma_uv)
        if self.marginalize_bottleneck:
            zu = self.append_noise_and_concat(u)
        elif self.implicit:
            zu = self.append_noise_and_concat(z)
        else:  # explicit bottleneck
            zu = self.append_noise_and_concat(z, u)

        return zu

    @add_epsilon
    def get_ratio_xyzuv(self, key, hxy, z, u, v):
        assert key in ['j', 'c', 'mx', 'my']
        # j: q(xy)q(zuv|xy) vs. p(zuv)\d(x|zu)\d(y|zv); q(xy)q(u|xy)q(v|xy) vs. p(uv)p(xy|uv) if marginalize_bottleneck
        # c: q(x)q(zu|x)p(v)\d(y|zv) vs. q(y)q(zv|y)p(u)\d(x|zu); q(x)p(yuv|x) vs. q(y)p(xuv|y) if marginal_bottleneck
        # mx: q(x)q(z|x)q(u|z,x)p(v)p(y|z,v) vs. p(z)p(u)p(v)p(x|z,u)p(y|z,v) (p(v) is omitted)
        # my: q(y)q(z|v)q(v|z,y)p(u)p(x|z,u) vs. p(z)p(u)p(v)p(x|z,u)p(y|z,v) (p(u) is omitted)
        zuv = self.add_noise_and_concat_zuv(z, u, v)
        hzuv = self.networks['{}_latent'.format(key)](zuv)
        hxyzuv = self.append_noise_and_concat(hxy, hzuv)
        ratio = self.networks['{}_ratio'.format(key)](hxyzuv)

        return ratio

    @add_epsilon
    def get_ratio_xyzu(self, key, hxy, z, u):
        assert key in ['cx2y', 'cy2x']
        # cy2x: q(x|y)q(z|x,y)q(u|z,x) vs. q(z|y)p(u)\d(x|z,u)
        # cx2y: q(y|x)q(z|x,y)q(v|z,y) vs. q(z|x)p(v)\d(y|z,v)
        zu = self.add_noise_and_concat_zu(z, u)
        hzu = self.networks['{}_latent'.format(key)](zu)
        hxyzu = self.append_noise_and_concat(hxy, hzu)
        ratio = self.networks['{}_ratio'.format(key)](hxyzu)

        return ratio

    @add_epsilon
    def get_ratio_xyz(self, key, hxy, z):
        assert key in ['j_ci', 'cx2y_ci', 'cy2x_ci', 'var_ci']
        z = self.add_gaussian_noise(z, self.sigma_z)
        z = self.append_noise_and_concat(z)
        hz = self.networks['{}_latent'.format(key)](z)
        hxyz = self.append_noise_and_concat(hxy, hz)
        ratio = self.networks['{}_ratio'.format(key)](hxyz)

        return ratio

    @add_epsilon
    def get_ratio_z(self, key, z):
        assert key in ['cx2y_agg', 'cy2x_agg', 'c_agg', 'var_agg']
        # cx2y: pz_cx2y=[q(x)q(z|x)] vs. p(z)
        # cy2x: pz_cy2x=[q(y)q(z|y)] vs. p(z)
        # c:    pz_cx2y vs. pz_cy2x
        # var:  qz_var=[q(xy)q(z|xy)] vs. p(z)
        z = self.add_gaussian_noise(z, self.sigma_z)
        ratio = self.networks['{}_ratio'.format(key)](z)
        return ratio

    def forward(self, gener_outputs, discriminator_forward=False):
        (xq, yq, zqxy, uqxy, vqxy,
         xp, yp, zp_j, up_j, vp_j,
         zqxq, uqxm, vp_cx2y, ypx,
         zqyq, vqym, up_cy2x, xpy) = gener_outputs

        batch_size = xq.shape[0]

        # -- (2) Compute approximate density log ratios --
        # In the following r[model](_[suffix])_[{var,model}] stands for density ratio estimates of corresponding models,
        # evaluated with samples drawn from s \in {var, model}, accordingly;
        log_ratios = defaultdict(lambda: None)

        hxy_disc_var = self.get_random_feature_xy(xq, yq)  # variational distribution
        if self.loss_weights.var_ci >= 0 and not self.marginalize_bottleneck:
            # q(x,y)q(z|x,y) vs. q(x,y)q(z)
            hxy_disc_var_ci = hxy_disc_var.detach() if discriminator_forward else hxy_disc_var
            log_ratios['var_ci_joint'] = self.get_ratio_xyz('var_ci', hxy_disc_var_ci, zqxy)
            log_ratios['var_ci_prod'] = self.get_ratio_xyz('var_ci', hxy_disc_var_ci, zqxy[torch.randperm(batch_size)])

        if self.loss_weights.j > 0:
            hxy_disc_j = self.get_random_feature_xy(xp, yp)

            # q(x,y,z) || p(z)p(x|z)p(y|z)
            log_ratios['j_var'] = self.get_ratio_xyzuv('j', hxy_disc_var, zqxy, uqxy, vqxy)
            log_ratios['j_model'] = self.get_ratio_xyzuv('j', hxy_disc_j, zp_j, up_j, vp_j)

            if self.loss_weights.j_ci >= 0 and not self.marginalize_bottleneck:
                # p(z)p(x|z)p(y|z) vs. p(z)p(x,y)
                hxy_disc_j_ci = hxy_disc_j.detach() if discriminator_forward else hxy_disc_j
                log_ratios['j_ci_joint'] = self.get_ratio_xyz('j_ci', hxy_disc_j_ci, zp_j)
                log_ratios['j_ci_prod'] = self.get_ratio_xyz('j_ci', hxy_disc_j_ci, zp_j[torch.randperm(batch_size)])

        if self.loss_weights.cx2y > 0:
            hxy_disc_cx2y = self.get_random_feature_xy(xq, ypx)

            # q(y|x)q(z|xy) || q(z|x)p(y|z)
            log_ratios['cx2y_var'] = self.get_ratio_xyzu('cx2y', hxy_disc_var, zqxy, vqxy)
            log_ratios['cx2y_model'] = self.get_ratio_xyzu('cx2y', hxy_disc_cx2y, zqxq, vp_cx2y)

            if self.loss_weights.cx2y_ci >= 0 and not self.marginalize_bottleneck:
                # q(x)q(z|x)p(y|z) vs. q(x)\qt(y|x)\qt(z)
                hxy_disc_cx2y_ci = hxy_disc_cx2y.detach() if discriminator_forward else hxy_disc_cx2y
                log_ratios['cx2y_ci_joint'] = self.get_ratio_xyz('cx2y_ci', hxy_disc_cx2y_ci, zqxq)
                log_ratios['cx2y_ci_prod'] = self.get_ratio_xyz('cx2y_ci', hxy_disc_cx2y_ci, zqxq[torch.randperm(batch_size)])

            if self.loss_weights.cx2y_agg > 0:
                log_ratios['cx2y_agg_model'] = self.get_ratio_z('cx2y_agg', zqxq)
                log_ratios['cx2y_agg_prior'] = self.get_ratio_z('cx2y_agg', zp_j)

        if self.loss_weights.cy2x > 0:
            hxy_disc_cy2x = self.get_random_feature_xy(xpy, yq)

            # q(x|y)q(z|xy) || q(z|y)p(x|z)
            log_ratios['cy2x_var'] = self.get_ratio_xyzu('cy2x', hxy_disc_var, zqxy, uqxy)
            log_ratios['cy2x_model'] = self.get_ratio_xyzu('cy2x', hxy_disc_cy2x, zqyq, up_cy2x)

            if self.loss_weights.cy2x_ci >= 0 and not self.marginalize_bottleneck:
                # q(x)q(z|x)p(y|z) vs. q(x)\qt(y|x)\qt(z)
                hxy_disc_cy2x_ci = hxy_disc_cy2x.detach() if discriminator_forward else hxy_disc_cy2x
                log_ratios['cy2x_ci_joint'] = self.get_ratio_xyz('cy2x_ci', hxy_disc_cy2x_ci, zqyq)
                log_ratios['cy2x_ci_prod'] = self.get_ratio_xyz('cy2x_ci', hxy_disc_cy2x_ci, zqyq[torch.randperm(batch_size)])

            if self.loss_weights.cy2x_agg > 0:
                log_ratios['cy2x_agg_model'] = self.get_ratio_z('cy2x_agg', zqyq)
                log_ratios['cy2x_agg_prior'] = self.get_ratio_z('cy2x_agg', zp_j)

        if self.loss_weights.cx2y > 0 and self.loss_weights.cy2x > 0:
            if self.loss_weights.c > 0:
                # q(x)q(z|x)p(y|z) || q(y)q(z|y)p(x|z)
                log_ratios['c_cx2y'] = self.get_ratio_xyzuv('c', hxy_disc_cx2y, zqxq, uqxm, vp_cx2y)
                log_ratios['c_cy2x'] = self.get_ratio_xyzuv('c', hxy_disc_cy2x, zqyq, up_cy2x, vqym)
            if self.loss_weights.c_agg > 0:
                log_ratios['c_agg_cx2y'] = self.get_ratio_z('c_agg', zqxq)
                log_ratios['c_agg_cy2x'] = self.get_ratio_z('c_agg', zqyq)

        if self.loss_weights.mx > 0:
            log_ratios['mx_cx2y'] = self.get_ratio_xyzuv('mx', hxy_disc_cx2y, zqxq, uqxm, vp_cx2y)
            log_ratios['mx_j'] = self.get_ratio_xyzuv('mx', hxy_disc_j, zp_j, up_j, vp_j)

        if self.loss_weights.my > 0:
            log_ratios['my_cy2x'] = self.get_ratio_xyzuv('my', hxy_disc_cy2x, zqyq, up_cy2x, vqym)
            log_ratios['my_j'] = self.get_ratio_xyzuv('my', hxy_disc_j, zp_j, up_j, vp_j)

        return log_ratios


# joint encoder + marginal encoders
class JointEncoder(RandomFunction):
    def __init__(self,
                 dim_z=128 * 2,
                 prior_type='gaussian',
                 to_feature_xy=None,
                 marginal_encoders=None,
                 n_hidden_layers=0,
                 hidden_dim=0,
                 negative_slope=0.0,
                 use_batchnorm=False,
                 batchnorm_kwargs=None,
                 noise_dim=20,
                 add_noise_channel=True,
                 additional_noise_sigma=0.0,
                 device=None):
        super(JointEncoder, self).__init__(noise_dim, add_noise_channel)
        self.sigma = additional_noise_sigma
        self.dim_z = dim_z
        self.dim_u = marginal_encoders[0].dim_u
        self.dim_v = marginal_encoders[1].dim_u
        self.marginal_encoders = nn.ModuleDict(dict(x=marginal_encoders[0], y=marginal_encoders[1]))

        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}

        self.to_feature_xy = to_feature_xy
        self.to_latent = nn.ModuleDict()
        self.to_latent['hxy_z'] = get_fc(input_dim=to_feature_xy.output_dim + noise_dim,
                                         hidden_dim=hidden_dim,
                                         n_hidden_layers=n_hidden_layers,
                                         output_dim=dim_z,
                                         lrelu_negative_slope=negative_slope,
                                         last_activation_type='tanh' if prior_type == 'uniform' else None,
                                         use_batchnorm=use_batchnorm,
                                         batchnorm_kwargs=batchnorm_kwargs)  # (B, dim_z); joint encoding
        self.device = device

    def set_additional_gaussian_noise(self, switch):
        self.marginal_encoders['x'].set_additional_gaussian_noise(switch)
        self.marginal_encoders['y'].set_additional_gaussian_noise(switch)
        super().set_additional_gaussian_noise(switch)
        return self

    def get_random_feature_xy(self, x, y):
        x = self.add_gaussian_noise(x, self.sigma)
        x = self.append_noise_and_concat(x)
        y = self.add_gaussian_noise(y, self.sigma)
        y = self.append_noise_and_concat(y)

        hxy = self.to_feature_xy(x, y)  # (B, feature_dim)
        hxy = torch.flatten(hxy, start_dim=1)
        return hxy

    def encode_xy_zuv(self, x, y):
        hxy = self.get_random_feature_xy(x, y)
        hx = self.marginal_encoders['x'].get_random_feature(x)
        hy = self.marginal_encoders['y'].get_random_feature(y)
        return self.encode_hxy_zuv(hxy, hx, hy)

    def encode_hxy_zuv(self, hxy, hx, hy):  # q(zuv|xy)
        z = self.encode_hxy_z(hxy)
        u = self.marginal_encoders['x'].encode_zhx_u(z, hx)
        v = self.marginal_encoders['y'].encode_zhx_u(z, hy)
        return z, u, v

    def encode_xy_z(self, x, y):  # q(z|xy)
        hxy = self.get_random_feature_xy(x, y)
        return self.encode_hxy_z(hxy)

    def encode_hxy_z(self, hxy):  # q(z|xy)
        hxy = self.append_noise_and_concat(hxy)
        return self.to_latent['hxy_z'](hxy)  # (B, dim_z)


class Prior:
    def __init__(self, dim, prior_type='gaussian'):
        """
        Standard Gaussian prior

        Parameters
        ----------
        dim
        """
        self.dim = dim
        assert prior_type in ['gaussian', 'uniform']
        self.prior_type = prior_type

    def _draw_from_base(self, *size, device=None):
        if self.prior_type == 'gaussian':
            return torch.randn(size).to(device)
        else:
            return 2 * torch.rand(size).to(device) - 1

    def draw(self, n, device=None):
        """
        Draw n of random Z's

        Parameters
        ----------
        n
        device

        Returns
        -------
        zp
        """
        return self._draw_from_base(n, self.dim, device=device)

    def draw_like(self, x):
        return self.draw(x.shape[0], device=x.device)


class Generator(nn.Module):
    def __init__(self, encoder, decoders, priors,
                 loss_weights, implicit):
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.priors = priors

        self.loss_weights = loss_weights
        self.implicit = implicit

        self.dim_z = self.encoder.dim_z
        self.dim_u = self.encoder.dim_u
        self.dim_v = self.encoder.dim_v
        self.latent_dim = self.dim_z + self.dim_u + self.dim_v  # compatibility for BDMC evaluation

    def draw_from_prior(self, n, device=None):
        return self.priors['z'].draw(n, device), self.priors['u'].draw(n, device), self.priors['v'].draw(n, device)

    def draw_joint_samples(self, n, device=None):
        zp, up, vp = self.draw_from_prior(n, device)
        xp, yp = self.decode_zuv_xy(zp, up, vp)
        return xp, yp

    def decode_zuv_xy(self, z, u, v):
        return self.decoders['x'](z, u), self.decoders['y'](z, v)

    def decode(self, zuv):
        """
        Decode method compatible to AIS evaluation

        Parameters
        ----------
        zuv

        Returns
        -------
        xy
        """
        z = zuv[..., :self.dim_z]
        u = zuv[..., self.dim_z:self.dim_z + self.dim_u]
        v = zuv[..., self.dim_z + self.dim_u:]
        x, y = self.decode_zuv_xy(z, u, v)
        xy = torch.cat([x.view(x.shape[0], -1), y.view(y.shape[0], -1)], dim=-1)
        return xy

    def encode_xv_yz(self, xref, vref, from_to='x2y'):
        assert from_to in ['x2y', 'y2x']
        ix, iy = from_to.split('2')
        zqx = self.encoder.marginal_encoders[ix].encode_x_z(xref)  # (ncond, dim_z)
        ypx = self.decoders[iy](zqx, vref)
        return ypx, zqx

    def encode_xv_yzu(self, x, v, from_to='x2y'):
        assert from_to in ['x2y', 'y2x']
        ix, iy = from_to.split('2')
        zqx, uqx = self.encoder.marginal_encoders[ix].encode_x_zu(x)  # (ncond, dim_z), (ncond, dim_u)
        ypx = self.decoders[iy](zqx, v)
        return ypx, zqx, uqx

    def draw_conditional_samples(self, conditions, from_to='x2y'):
        # for each ref in conditions, generate "one" conditional sample
        assert from_to in ['x2y', 'y2x']
        iv = 'v' if from_to[-1] == 'y' else 'u'
        vp = self.priors[iv].draw(conditions.shape[0], device=conditions.device)
        return self.encode_xv_yz(conditions, vp, from_to=from_to)[0]

    def get_joint_sample_generator(self, n_samples, batch_size, device,
                                   postprocessor=lambda x: x):
        def joint_sample_generator():
            for i in range(n_samples // batch_size):
                zp, up, vp = self.draw_from_prior(n=batch_size, device=device)
                yield postprocessor(self.decoders['x'](zp, up)), postprocessor(self.decoders['y'](zp, vp))
        return joint_sample_generator()

    def get_conditional_sample_generator(self, condition, n_samples, batch_size, from_to='x2y',
                                         postprocessor=lambda x: x):
        """
        From one reference, draw n_samples
        Note: here we implicitly assume that q(z|x) and q(z|y) are deterministic and thus we only need to draw z once

        :param condition: ref_shape
        :param n_samples: int
        :param batch_size: int
        :param from_to: ['x2y', 'y2x']
        :return: (n_samples, *sample_shape)
        """
        assert from_to in ['x2y', 'y2x']
        ix, iy = from_to.split('2')
        iv = 'v' if from_to[-1] == 'y' else 'u'

        # FIXME: Here is a possible culprit of performance degradation,
        #        considering that BN uses batch statistics even during inference
        # draw z
        zqx = self.encoder.marginal_encoders[ix].encode_x_z(condition.unsqueeze(0))  # (1, dim_z)
        zqx_rep = repeat_interleave(zqx, dim=0, n_tile=batch_size)  # (n_samples, dim_z)

        def sample_generator():
            for i in range(n_samples // batch_size):
                # draw v and combine
                vp = self.priors[iv].draw(batch_size, device=condition.device)  # (n_samples, dim_v)
                ypx = self.decoders[iy](zqx_rep, vp)
                yield postprocessor(ypx)

        return sample_generator()

    def get_marginal_sample_generator(self, var, n_samples, batch_size, device,
                                      postprocessor=lambda x: x):
        def marginal_sample_generator():
            for i in range(n_samples // batch_size):
                zp, up, vp = self.draw_from_prior(n=batch_size, device=device)
                if var == 'x':
                    yield postprocessor(self.decoders['x'](zp, up))
                else:
                    yield postprocessor(self.decoders['y'](zp, vp))
        return marginal_sample_generator()

    def forward_var(self, xq, yq):
        # "variational": (xq, yq, zq, uq, vq) ~ q(x,y)q(z|x,y)q(u|z,x)q(v|z,y)
        hxyq = self.encoder.get_random_feature_xy(xq, yq)
        hxq = self.encoder.marginal_encoders['x'].get_random_feature(xq)
        hyq = self.encoder.marginal_encoders['y'].get_random_feature(yq)
        if not self.implicit:
            zqxy, uqxy, vqxy = self.encoder.encode_hxy_zuv(hxyq, hxq, hyq)
        else:
            zqxy = self.encoder.encode_hxy_z(hxyq)
            uqxy = vqxy = torch.zeros(xq.shape[0], 0).to(self.encoder.device)

        return hxyq, hxq, hyq, zqxy, uqxy, vqxy

    def forward_j(self, batch_size, device):
        # (zp, up, vp) ~ p(z)p(u)p(v)\d(x|z,u)\d(y|z,v)
        xp, yp, zp_j, up_j, vp_j = None, None, None, None, None
        if self.loss_weights.j > 0:
            zp_j, up_j, vp_j = self.draw_from_prior(n=batch_size, device=device)
            xp, yp = self.decode_zuv_xy(zp_j, up_j, vp_j)

        return xp, yp, zp_j, up_j, vp_j

    def forward_cx2y(self, xq):
        # (xq, zqx, uqx, vp, ypx) ~ q(x)q(z|x)q(u|z,x)p(v)\d(y|z,v)
        zqxq, uqxm, vp_cx2y, ypx = None, None, None, None
        if self.loss_weights.cx2y > 0:
            vp_cx2y = self.priors['v'].draw_like(xq)
            ypx, zqxq, uqxm = self.encode_xv_yzu(x=xq, v=vp_cx2y, from_to='x2y')

        return zqxq, uqxm, vp_cx2y, ypx

    def forward_cy2x(self, yq):
        # (yq, zqy, vqy, up, xpy) ~ q(y)q(z|y)q(v|z,y)p(u)\d(x|z,u)
        zqyq, vqym, up_cy2x, xpy = None, None, None, None
        if self.loss_weights.cy2x > 0:
            up_cy2x = self.priors['u'].draw_like(yq)
            xpy, zqyq, vqym = self.encode_xv_yzu(x=yq, v=up_cy2x, from_to='y2x')

        return zqyq, vqym, up_cy2x, xpy

    def forward(self, xq, yq, reconstruction=False):
        # reconstruction=True if generator is to be trained, else if discriminator is to be trained
        # Note: if self.implicit=True, hide (u, v)
        batch_size = xq.shape[0]

        # --- sampling ---
        # 0) "variational": (xq, yq, zq, uq, vq) ~ q(x,y)q(z|x,y)q(u|z,x)q(v|z,y)
        hxyq, hxq, hyq, zqxy, uqxy, vqxy = self.forward_var(xq, yq)
        # 1) "joint" (zp, up, vp) ~ p(z)p(u)p(v)\d(x|z,u)\d(y|z,v)
        xp, yp, zp_j, up_j, vp_j = self.forward_j(batch_size, xq.device)
        # 2) "conditional"
        zqxq, uqxm, vp_cx2y, ypx = self.forward_cx2y(xq)
        zqyq, vqym, up_cy2x, xpy = self.forward_cy2x(yq)

        # --- reconstruction ---
        xq_hat_j, yq_hat_j, yq_hat_cx2y, xq_hat_cy2x, xq_hat_c, yq_hat_c = None, None, None, None, None, None
        zp_hat_j, up_hat_j, vp_hat_j = None, None, None

        if reconstruction and not self.implicit:
            # 1) pure reconstruction terms (variational + one model)
            #   - for each of the following reconstruction terms, q(z|x,y)q(u|z,x)q(v|z,y) is properly used

            # joint: (xq, yq) reconstruction (xq, yq) -> (zq, uq, vq)
            # (zq, uq) -> xq_hat
            if self.loss_weights.j > 0 or self.loss_weights.cy2x > 0:
                # The decoder p(x|z,u) is required for (x,y)->z->x reconstruction
                if self.loss_weights.j_rec_x > 0:
                    xq_hat_j = self.decoders['x'](zqxy, uqxy)
            # (zq, uq) -> yq_hat
            if self.loss_weights.j > 0 or self.loss_weights.cx2y > 0:
                # The decoder p(y|z,v) is required for (x,y)->z->x reconstruction
                if self.loss_weights.j_rec_y > 0:
                    yq_hat_j = self.decoders['y'](zqxy, vqxy)

            # cx2y: yq reconstruction
            if self.loss_weights.cx2y > 0:
                if self.loss_weights.cx2y_rec_y > 0:
                    vqzxqyq = self.encoder.marginal_encoders['y'].encode_zhx_u(zqxq, hyq)  # v(z(x), y)
                    yq_hat_cx2y = self.decoders['y'](zqxq, vqzxqyq)  # yhat = y(z(x), v(z(x), y))

            # cy2x: xq reconstruction
            if self.loss_weights.cy2x > 0:
                if self.loss_weights.cy2x_rec_x > 0:
                    uqzyqxq = self.encoder.marginal_encoders['x'].encode_zhx_u(zqyq, hxq)  # u(z(y), x)
                    xq_hat_cy2x = self.decoders['x'](zqyq, uqzyqxq)  # xhat = y(z(x), v(z(x), y))

            # 2) hybrid reconstruction terms
            # "marginal reconstruction" is possible only if cx2y > 0 and cy2x > 0
            if self.loss_weights.cx2y > 0 and self.loss_weights.cy2x > 0:
                # m1x) marginal data reconstruction xq -> (zq, uq) -> xq_hat
                if self.loss_weights.mx_rec_x > 0:
                    xq_hat_c = self.decoders['x'](zqxq, uqxm)
                # m1y) marginal data reconstruction yq -> (zq, vq) -> yq_hat
                if self.loss_weights.my_rec_y > 0:
                    yq_hat_c = self.decoders['y'](zqyq, vqym)

            if self.loss_weights.j > 0 and self.loss_weights.j_rec_zuv > 0:
                zp_hat_j, up_hat_j, vp_hat_j = self.encoder.encode_xy_zuv(xp, yp)

        gener_outputs = [xq, yq, zqxy, uqxy, vqxy,
                         xp, yp, zp_j, up_j, vp_j,
                         zqxq, uqxm, vp_cx2y, ypx,
                         zqyq, vqym, up_cy2x, xpy]
        recons = [xq_hat_j, yq_hat_j,
                  zp_hat_j, up_hat_j, vp_hat_j,
                  yq_hat_cx2y,
                  xq_hat_cy2x,
                  xq_hat_c, yq_hat_c]

        return gener_outputs, recons


class Model(nn.Module):
    def __init__(self, generator: Generator, discriminator: Discriminator,
                 gener_loss_fn, recon_loss_fn_x, recon_loss_fn_y, recon_loss_fn_zuv, discr_loss_fn,
                 marginalize_bottleneck, uniform_discr_weights=True):
        super(Model, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.gener_loss_fn = gener_loss_fn
        self.recon_loss_fn_x = recon_loss_fn_x
        self.recon_loss_fn_y = recon_loss_fn_y
        self.recon_loss_fn_zuv = recon_loss_fn_zuv
        self.discr_loss_fn = discr_loss_fn

        self.loss_weights = self.discriminator.loss_weights

        self.dim_z = self.generator.dim_z
        self.dim_u = self.generator.dim_u
        self.dim_v = self.generator.dim_v

        self.marginalize_bottleneck = marginalize_bottleneck
        self.implicit = discriminator.implicit
        self.uniform_discr_weights = uniform_discr_weights
        
    def forward(self, xq, yq, train_discriminator=False):
        gener_outputs, recons = self.generator.forward(xq, yq, reconstruction=not train_discriminator)
        # detach generator outputs if it is for discriminator training
        gener_outputs = [arg.detach() if (train_discriminator and arg is not None) else arg
                         for arg in gener_outputs]
        log_ratios = self.discriminator.forward(gener_outputs, train_discriminator)

        return gener_outputs, recons, log_ratios

    def compute_discriminator_losses(self, xq, yq):
        *_, log_ratios = self.forward(xq, yq, train_discriminator=True)

        discr_losses = defaultdict(float)

        if self.loss_weights.var_ci >= 0 and not self.marginalize_bottleneck:
            discr_losses['var_ci'] = - self.discr_loss_fn(log_ratios['var_ci_joint'], log_ratios['var_ci_prod'])
        if self.loss_weights.var_agg > 0:
            discr_losses['var_agg'] = - self.discr_loss_fn(log_ratios['var_agg_model'], log_ratios['var_agg_prior'])

        if self.loss_weights.j > 0:
            discr_losses['j'] = - self.discr_loss_fn(log_ratios['j_var'], log_ratios['j_model'])
            if self.loss_weights.j_ci >= 0 and not self.marginalize_bottleneck:
                discr_losses['j_ci'] = - self.discr_loss_fn(log_ratios['j_ci_joint'], log_ratios['j_ci_prod'])

        if self.loss_weights.cx2y > 0:
            discr_losses['cx2y'] = - self.discr_loss_fn(log_ratios['cx2y_var'], log_ratios['cx2y_model'])
            if self.loss_weights.cx2y_ci >= 0 and not self.marginalize_bottleneck:
                discr_losses['cx2y_ci'] = - self.discr_loss_fn(log_ratios['cx2y_ci_joint'], log_ratios['cx2y_ci_prod'])
            if self.loss_weights.cx2y_agg > 0:
                discr_losses['cx2y_agg'] = - self.discr_loss_fn(log_ratios['cx2y_agg_model'], log_ratios['cx2y_agg_prior'])

        if self.loss_weights.cy2x > 0:
            discr_losses['cy2x'] = - self.discr_loss_fn(log_ratios['cy2x_var'], log_ratios['cy2x_model'])
            if self.loss_weights.cy2x_ci >= 0 and not self.marginalize_bottleneck:
                discr_losses['cy2x_ci'] = - self.discr_loss_fn(log_ratios['cy2x_ci_joint'], log_ratios['cy2x_ci_prod'])
            if self.loss_weights.cy2x_agg > 0:
                discr_losses['cy2x_agg'] = - self.discr_loss_fn(log_ratios['cy2x_agg_model'], log_ratios['cy2x_agg_prior'])

        if self.loss_weights.cx2y > 0 and self.loss_weights.cy2x > 0:
            if self.loss_weights.c > 0:
                discr_losses['c'] = - self.discr_loss_fn(log_ratios['c_cx2y'], log_ratios['c_cy2x'])
            if self.loss_weights.c_agg > 0:
                discr_losses['c_agg'] = - self.discr_loss_fn(log_ratios['c_agg_cx2y'], log_ratios['c_agg_cy2x'])

        if self.loss_weights.mx > 0:
            discr_losses['mx'] = - self.discr_loss_fn(log_ratios['mx_cx2y'], log_ratios['mx_j'])

        if self.loss_weights.my > 0:
            discr_losses['my'] = - self.discr_loss_fn(log_ratios['my_cy2x'], log_ratios['my_j'])
        
        discr_losses['total'] = torch.stack(
            [(1 if self.uniform_discr_weights else self.loss_weights[key]) * discr_losses[key]
             for key in discr_losses]).sum()

        return discr_losses

    def compute_generator_losses(self, xq, yq):
        gener_outputs, recons, log_ratios = self.forward(xq, yq, train_discriminator=False)

        gener_losses = defaultdict(float)

        if self.loss_weights.var_ci >= 0 and not self.marginalize_bottleneck:
            gener_losses['var_ci'] = log_ratios['var_ci_joint'].mean()
        if self.loss_weights.var_agg > 0:
            gener_losses['var_agg'] = self.gener_loss_fn(log_ratios['var_agg_model'], log_ratios['var_agg_prior'])

        if self.loss_weights.j > 0:
            gener_losses['j'] = self.gener_loss_fn(log_ratios['j_var'], log_ratios['j_model'])
            if self.loss_weights.j_ci >= 0 and not self.marginalize_bottleneck:
                gener_losses['j_ci'] = log_ratios['j_ci_joint'].mean()

        if self.loss_weights.cx2y > 0:
            gener_losses['cx2y'] = self.gener_loss_fn(log_ratios['cx2y_var'], log_ratios['cx2y_model'])
            if self.loss_weights.cx2y_ci >= 0 and not self.marginalize_bottleneck:
                gener_losses['cx2y_ci'] = log_ratios['cx2y_ci_joint'].mean()
            if self.loss_weights.cx2y_agg > 0:
                gener_losses['cx2y_agg'] = self.gener_loss_fn(log_ratios['cx2y_agg_model'], log_ratios['cx2y_agg_prior'])

        if self.loss_weights.cy2x > 0:
            gener_losses['cy2x'] = self.gener_loss_fn(log_ratios['cy2x_var'], log_ratios['cy2x_model'])
            if self.loss_weights.cy2x_ci >= 0 and not self.marginalize_bottleneck:
                gener_losses['cy2x_ci'] = log_ratios['cy2x_ci_joint'].mean()
            if self.loss_weights.cy2x_agg > 0:
                gener_losses['cy2x_agg'] = self.gener_loss_fn(log_ratios['cy2x_agg_model'], log_ratios['cy2x_agg_prior'])

        if self.loss_weights.cx2y > 0 and self.loss_weights.cy2x > 0:
            if self.loss_weights.c > 0:
                gener_losses['c'] = self.gener_loss_fn(log_ratios['c_cx2y'], log_ratios['c_cy2x'])
            if self.loss_weights.c_agg > 0:
                gener_losses['c_agg'] = self.gener_loss_fn(log_ratios['c_agg_cx2y'], log_ratios['c_agg_cy2x'])

        if self.loss_weights.mx > 0:
            gener_losses['mx'] = self.gener_loss_fn(log_ratios['mx_cx2y'], log_ratios['mx_j'])

        if self.loss_weights.my > 0:
            gener_losses['my'] = self.gener_loss_fn(log_ratios['my_cy2x'], log_ratios['my_j'])

        # reconstruction losses for explicit model
        if not self.implicit:
            (xq, yq, zqxy, uqxy, vqxy,
             xp, yp, zp_j, up_j, vp_j,
             zqxq, uqxm, vp_cx2y, ypx,
             zqyq, vqym, up_cy2x, xpy) = gener_outputs

            (xq_hat_j, yq_hat_j,
             zp_hat_j, up_hat_j, vp_hat_j,
             yq_hat_cx2y,
             xq_hat_cy2x,
             xq_hat_c, yq_hat_c) = recons

            if self.loss_weights.j > 0 or self.loss_weights.cy2x > 0:
                if self.loss_weights.j_rec_x > 0:
                    gener_losses['j_rec_x'] = self.recon_loss_fn_x(xq_hat_j, xq)

            if self.loss_weights.j > 0 or self.loss_weights.cx2y > 0:
                if self.loss_weights.j_rec_y > 0:
                    gener_losses['j_rec_y'] = self.recon_loss_fn_y(yq_hat_j, yq)

            # cx2y: yq reconstruction
            if self.loss_weights.cx2y > 0:
                if self.loss_weights.cx2y_rec_y > 0:
                    gener_losses['cx2y_rec_y'] = self.recon_loss_fn_y(yq_hat_cx2y, yq)

            # cy2x: xq reconstruction
            if self.loss_weights.cy2x > 0:
                if self.loss_weights.cy2x_rec_x > 0:
                    gener_losses['cy2x_rec_x'] = self.recon_loss_fn_x(xq_hat_cy2x, xq)  # NOTE: yq must be paired with xq

            if self.loss_weights.cx2y > 0 and self.loss_weights.cy2x > 0:
                if self.loss_weights.mx_rec_x > 0:
                    gener_losses['mx_rec_x'] = self.recon_loss_fn_x(xq_hat_c, xq)

                if self.loss_weights.my_rec_y > 0:
                    gener_losses['my_rec_y'] = self.recon_loss_fn_y(yq_hat_c, yq)

            if self.loss_weights.j > 0:
                if self.loss_weights.j_rec_zuv > 0:
                    gener_losses['j_rec_zuv'] = self.recon_loss_fn_zuv(zp_hat_j, zp_j) + \
                                                self.recon_loss_fn_zuv(up_hat_j, up_j) + \
                                                self.recon_loss_fn_zuv(vp_hat_j, vp_j)

        # model loss (total)
        gener_losses['total'] = torch.stack([self.loss_weights[key] * gener_losses[key] for key in gener_losses]).sum()

        return gener_losses


def build_models(config, device):
    dim_z = config.dim_z
    dim_u = config.dim_u
    dim_v = config.dim_v
    batchnorm_kwargs = dict(momentum=config.batchnorm_momentum, eps=config.batchnorm_eps)

    additional_noise_sigma_x = config.discriminator_additional_noise_sigma_xy
    additional_noise_sigma_y = config.discriminator_additional_noise_sigma_xy
    turn_off_noise_x = False
    turn_off_noise_y = False
    
    dec_to_feature_x_batchnorm2d = False
    dec_to_feature_y_batchnorm2d = False
    dec_to_feature_x_batchnorm2d_input_shape = None
    dec_to_feature_y_batchnorm2d_input_shape = None
    
    if config.dataset == 'sketchy-vgg':
        print("Warning: Some network parameters of SketchyVGG networks are hard-coded...")
        config.encoder_negative_slope = 0.2
        config.decoder_negative_slope = 0
        config.encoder_hidden_dim = 0
        config.encoder_n_hidden_layers = 0

        discr_to_feature_xy, \
        to_feature_x, to_feature_y,  to_feature_xy, \
        to_data_x, to_data_y = get_sketchy_networks(config)

        train_loader = SketchyVGGDataLoader(batch_size=config.batch_size, shuffle=True, drop_last=False,
                                            root_path=config.main_path, split=config.sketchy_split,
                                            train_or_test='train')
        additional_noise_sigma_x = config.instance_noise_scale_x * torch.Tensor(train_loader.sketch_features_stdev).to(device)
        additional_noise_sigma_y = config.instance_noise_scale_y * torch.Tensor(train_loader.photo_features_stdev).to(device)

    elif config.dataset in ['cub', 'cub-imgft2sent']:
        discr_to_feature_xy, \
        to_feature_x, to_feature_y, to_feature_xy, \
        to_data_x, to_data_y = get_cub_networks(config, device)

        if 'imgft' in config.dataset:
            imgft_stdev = CUBImgFt(config.main_path, 'train', device).features_stdev
            additional_noise_sigma_x = config.instance_noise_scale_x * torch.Tensor(imgft_stdev).to(device)
        additional_noise_sigma_y = 0.
        turn_off_noise_y = True  # noise to sentence is injected directly in discr_to_feature_xy from get_cub_networks
        dec_to_feature_y_batchnorm2d = config.cub_sent_use_conv2d and config.generator_core_use_batchnorm and config.cub_imgft_decoder_use_batchnorm
        dec_to_feature_y_batchnorm2d_input_shape = (32 * 16, 4, 4) if config.cub_sent_use_conv2d else None

    elif config.dataset in IMAGE_DATASETS:
        base_cc_x = base_cc_y = base_cc_xy = config.base_cc
        if config.dataset == 'mnist-mnist':
            cc_x, cc_y = 1, 1
        elif config.dataset == 'mnist-svhn':
            cc_x, cc_y = 1, 3
        elif config.dataset == 'mnist-cdcb':
            cc_x, cc_y = 3, 3
        elif config.dataset == 'mnist-multiply':
            cc_x, cc_y = 2, 2
        else:
            raise Exception('check config.dataset')
        img_size = 32

        discriminator_base_cc_x = 2 * base_cc_x
        discriminator_base_cc_y = 2 * base_cc_y
        discriminator_base_cc_xy = 2 * base_cc_xy
        decoder_base_cc_x = 2 * base_cc_x
        decoder_base_cc_y = 2 * base_cc_y

        *_, discr_to_feature_xy, \
        to_feature_x, to_feature_y, to_feature_xy, \
        to_data_x, to_data_y = get_image2image_networks(
            cc_x=cc_x, cc_y=cc_y,
            discriminator_base_cc_x=discriminator_base_cc_x,
            discriminator_base_cc_y=discriminator_base_cc_y,
            discriminator_base_cc_xy=discriminator_base_cc_xy,
            discriminator_negative_slope=config.discriminator_negative_slope,
            discriminator_add_noise_channel=config.discriminator_add_noise_channel,
            encoder_base_cc_x=base_cc_x,
            encoder_base_cc_y=base_cc_y,
            encoder_base_cc_xy=base_cc_xy,
            encoder_negative_slope=config.encoder_negative_slope,
            joint_encoder_add_noise_channel=config.joint_encoder_add_noise_channel,
            marginal_encoder_add_noise_channel=config.marginal_encoder_add_noise_channel,
            decoder_base_cc_x=decoder_base_cc_x,
            decoder_base_cc_y=decoder_base_cc_y,
            decoder_negative_slope=config.decoder_negative_slope,
            generator_use_batchnorm=config.generator_feature_use_batchnorm,
            discriminator_use_batchnorm=config.discriminator_feature_use_batchnorm,
            discriminator_use_batchnorm_first_layer=config.discriminator_feature_use_batchnorm_first_layer,
            batchnorm_kwargs=batchnorm_kwargs,
            use_spectralnorm=config.discriminator_feature_use_spectralnorm,
            last_activation=config.feature_map_last_activation,
            n_resblocks=config.n_resblocks,
            kernel_size=config.kernel_size,
            img_size=img_size,
        )
        dec_to_feature_x_batchnorm2d = True
        dec_to_feature_x_batchnorm2d_input_shape = (decoder_base_cc_x * 8, 32 // 8, 32 // 8)
        dec_to_feature_y_batchnorm2d = True
        dec_to_feature_y_batchnorm2d_input_shape = (decoder_base_cc_y * 8, 32 // 8, 32 // 8)

    else:
        raise ValueError('Check config.dataset {}'.format(config.dataset))

    if config.marginalize_bottleneck:
        assert not (config.lambda_var_ci >= 0)
        assert not (config.lambda_j_ci >= 0)
        assert not (config.lambda_cx2y_ci >= 0)
        assert not (config.lambda_cy2x_ci >= 0)

    # loss_weights
    loss_keys = ['j', 'cx2y', 'cy2x', 'c', 'mx', 'my',
                 'var_ci', 'j_ci', 'cx2y_ci', 'cy2x_ci',
                 'var_agg', 'cx2y_agg', 'cy2x_agg', 'c_agg',
                 'j_rec_x', 'cy2x_rec_x', 'mx_rec_x',
                 'j_rec_y', 'cx2y_rec_y', 'my_rec_y',
                 'j_rec_zuv',
                 ]
    loss_weights = dotdict({loss_key: getattr(config, 'lambda_{}'.format(loss_key)) for loss_key in loss_keys})

    discriminator = Discriminator(dim_z=dim_z, dim_u=dim_u, dim_v=dim_v,
                                  to_feature_xy=discr_to_feature_xy,
                                  n_hidden_layers=config.discriminator_n_hidden_layers,
                                  hidden_dim=config.discriminator_hidden_dim,
                                  negative_slope=config.discriminator_negative_slope,
                                  use_batchnorm=config.discriminator_core_use_batchnorm,
                                  use_batchnorm_first_layer=config.discriminator_feature_use_batchnorm_first_layer,
                                  batchnorm_kwargs=batchnorm_kwargs,
                                  use_spectralnorm=False,
                                  noise_dim=config.discriminator_noise_dim,
                                  add_noise_channel=config.discriminator_add_noise_channel,
                                  additional_noise_sigma_x=additional_noise_sigma_x,
                                  additional_noise_sigma_y=additional_noise_sigma_y,
                                  additional_noise_sigma_z=config.discriminator_additional_noise_sigma_z,
                                  additional_noise_sigma_uv=config.discriminator_additional_noise_sigma_uv,
                                  turn_off_noise_x=turn_off_noise_x,
                                  turn_off_noise_y=turn_off_noise_y,
                                  implicit=config.implicit,
                                  marginalize_bottleneck=config.marginalize_bottleneck,
                                  loss_weights=loss_weights,
                                  device=device)

    encoder_x = MarginalEncoder(dim_z=dim_z, dim_u=dim_u,
                                prior_type=config.prior_type,
                                degenerate=config.degenerate_local_encoder,
                                simple_local=config.simple_local_encoder,
                                to_feature=to_feature_x,
                                n_hidden_layers=config.encoder_n_hidden_layers,
                                hidden_dim=config.encoder_hidden_dim,
                                negative_slope=config.encoder_negative_slope,
                                use_batchnorm=config.generator_core_use_batchnorm,
                                batchnorm_kwargs=batchnorm_kwargs,
                                noise_dim=config.marginal_encoder_noise_dim,
                                add_noise_channel=config.marginal_encoder_add_noise_channel,
                                additional_noise_sigma=config.marginal_encoder_additional_noise_sigma,
                                device=device)

    encoder_y = MarginalEncoder(dim_z=dim_z, dim_u=dim_v,
                                prior_type=config.prior_type,
                                degenerate=config.degenerate_local_encoder,
                                simple_local=config.simple_local_encoder,
                                to_feature=to_feature_y,
                                n_hidden_layers=config.encoder_n_hidden_layers,
                                hidden_dim=config.encoder_hidden_dim,
                                negative_slope=config.encoder_negative_slope,
                                use_batchnorm=config.generator_core_use_batchnorm,
                                batchnorm_kwargs=batchnorm_kwargs,
                                noise_dim=config.marginal_encoder_noise_dim,
                                add_noise_channel=config.marginal_encoder_add_noise_channel,
                                additional_noise_sigma=config.marginal_encoder_additional_noise_sigma,
                                device=device)

    encoder = JointEncoder(dim_z=dim_z,
                           prior_type=config.prior_type,
                           to_feature_xy=to_feature_xy,
                           marginal_encoders=[encoder_x, encoder_y],
                           n_hidden_layers=config.encoder_n_hidden_layers,
                           hidden_dim=config.encoder_hidden_dim,
                           negative_slope=config.encoder_negative_slope,
                           use_batchnorm=config.generator_core_use_batchnorm,
                           batchnorm_kwargs=batchnorm_kwargs,
                           noise_dim=config.joint_encoder_noise_dim,
                           add_noise_channel=config.joint_encoder_add_noise_channel,
                           additional_noise_sigma=config.joint_encoder_additional_noise_sigma,
                           device=device)

    dec_to_feature_x = get_fc(input_dim=dim_z + dim_u,
                              hidden_dim=config.encoder_hidden_dim,
                              n_hidden_layers=config.encoder_n_hidden_layers,
                              output_dim=to_data_x.input_dim,
                              lrelu_negative_slope=config.decoder_negative_slope,
                              use_batchnorm=config.generator_core_use_batchnorm,
                              batchnorm_kwargs=batchnorm_kwargs,
                              last_activation_type='lrelu',
                              batchnorm2d=dec_to_feature_x_batchnorm2d,
                              batchnorm2d_input_shape=dec_to_feature_x_batchnorm2d_input_shape,
                              )
    dec_to_feature_y = get_fc(input_dim=dim_z + dim_v,
                              hidden_dim=config.encoder_hidden_dim,
                              n_hidden_layers=config.encoder_n_hidden_layers,
                              output_dim=to_data_y.input_dim,
                              lrelu_negative_slope=config.decoder_negative_slope,
                              use_batchnorm=config.generator_core_use_batchnorm,
                              batchnorm_kwargs=batchnorm_kwargs,
                              last_activation_type='lrelu',
                              batchnorm2d=dec_to_feature_y_batchnorm2d,
                              batchnorm2d_input_shape=dec_to_feature_y_batchnorm2d_input_shape,
                              )

    if config.dataset == 'cub-imgft2sent':
        dec_to_feature_x = get_fc(input_dim=dim_z + dim_u,
                                  hidden_dim=config.encoder_hidden_dim,
                                  n_hidden_layers=config.encoder_n_hidden_layers,
                                  output_dim=to_data_x.input_dim,
                                  lrelu_negative_slope=config.decoder_negative_slope,
                                  activation_type='elu',
                                  use_batchnorm=False,
                                  batchnorm_kwargs=batchnorm_kwargs,
                                  last_activation_type='lrelu',
                                  batchnorm2d=dec_to_feature_x_batchnorm2d,
                                  batchnorm2d_input_shape=dec_to_feature_x_batchnorm2d_input_shape,
                                  )
    decoders = dict()
    if config.lambda_j + config.lambda_cy2x > 0:
        decoders['x'] = Decoder(to_data_x, to_feature=dec_to_feature_x)
    if config.lambda_j + config.lambda_cx2y > 0:
        decoders['y'] = Decoder(to_data_y, to_feature=dec_to_feature_y)
    decoders = torch.nn.ModuleDict(decoders)

    priors = dict(z=Prior(dim_z, prior_type=config.prior_type),
                  u=Prior(dim_u, prior_type=config.prior_type),
                  v=Prior(dim_v, prior_type=config.prior_type))

    generator = Generator(encoder, decoders, priors, loss_weights, config.implicit)

    gener_loss_fn = GENERATOR_LOSS_FUNCTIONS[config.gener_loss_fn]
    recon_loss_fn_x = recon_loss_fn_y = RECON_LOSS_FUNCTIONS['l1']
    recon_loss_fn_zuv = RECON_LOSS_FUNCTIONS['l2']
    if config.dataset in ['cub', 'cub-imgft2sent']:
        if config.dataset == 'cub-imgft2sent':
            recon_loss_fn_x = RECON_LOSS_FUNCTIONS['l2']

        # for CUB dataset,
        #   yhat is logits (B, max_sentence_length, vocab_size) and
        #   y is sents_in_idx (B, max_sentence_length)
        def recon_loss_fn_y(yhat, y):
            embedding = generator.encoder.marginal_encoders['y'].to_feature.embedding
            if config.cub_sent_recon_fn == 'log':
                if yhat.shape[-1] == embedding.embedding_dim:
                    yhat = generator.decoders['y'].to_data.embedding_inverse(yhat)
                assert yhat.shape[-1] == embedding.num_embeddings  # vocab_size
                return RECON_LOSS_FUNCTIONS['log'](yhat, y)
            else:
                embed = embedding(y)  # (B, msl, embedding_dim)
                embed_hat = embedding(yhat.argmax(-1))
                return RECON_LOSS_FUNCTIONS[config.cub_sent_recon_fn](embed_hat, embed)

        # sanity check
        if config.cub_sent_recon_fn != 'log':
            if config.cub_sent_use_pretrained_embedding and not config.cub_sent_trainable_embedding:
                warnings.warn('If the sentence reconstruction function is acted on embedding domain, '
                              'it is unclear for now if the embedding can be trained')

    def discr_loss_fn(rp, rq, positive_label_smothing=config.positive_label_smoothing):
        return DISCRIMINATOR_LOSS_FUNCTIONS[config.discr_loss_fn](
            rp, rq, positive_label_smoothing=positive_label_smothing)

    model = Model(generator, discriminator,
                  gener_loss_fn, recon_loss_fn_x, recon_loss_fn_y, recon_loss_fn_zuv, discr_loss_fn,
                  config.marginalize_bottleneck,
                  config.uniform_discr_weights)

    # weight initialization
    model.apply(weights_init)

    # for CUB dataset, if pretrained_embedding is True, load weights
    if config.dataset in ['cub', 'cub-imgft2sent']:
        if config.cub_sent_use_pretrained_embedding:
            # the following parameters are for training word2vec embedding;
            # please note that these are independent from the language model we actually train.
            max_sentence_length = 32
            window_length = 3
            min_occur = 3

            pretrained_embedding = CUBSentHelper(config.main_path, max_sentence_length, min_occur, config.cub_sent_embedding_dim,
                                                 window_length, reset=False, device=None).word2vec_embedding
            discr_to_feature_xy.to_feature_y.embedding.weight = nn.Parameter(
                pretrained_embedding.detach().clone()).requires_grad_(config.cub_sent_trainable_embedding)
            to_feature_y.embedding.weight = nn.Parameter(
                pretrained_embedding.detach().clone()).requires_grad_(config.cub_sent_trainable_embedding)

        to_feature_xy.to_feature_y.embedding = to_feature_y.embedding

        # Tie embedding weights
        #   - Both embedding.weight and to_vocab_size weight have dimension (vocab_size, embedding_dim).
        #   - This simply forces the model to use the same embedding weight matrix in decoding.
        if config.cub_sent_tie_embedding_inverse:
            to_data_y.embedding_inverse.weight = to_feature_y.embedding.weight

    if config.cub_sent_tie_embedding:
        discriminator.to_feature_xy.to_feature_y.embedding = to_feature_xy.to_feature_y.embedding

    model = model.to(device)
    return generator, discriminator, model
