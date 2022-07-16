from collections import defaultdict

import torch
import torch.nn as nn

from models.networks.base import weights_init, get_fc, RandomFunction


class Discriminator(RandomFunction):
    def __init__(self,
                 dim_z=128 * 2,
                 dim_u=128,
                 to_feature=None,
                 n_hidden_layers=2,
                 hidden_dim=1000,
                 negative_slope=0.0,
                 use_batchnorm=False,
                 batchnorm_kwargs=None,
                 noise_dim=100,
                 add_noise_channel=True,
                 additional_noise_sigma_x=0.0,
                 additional_noise_sigma_z=0.0,
                 add_gaussian_noise_to_latent=False,
                 device=None):
        super(Discriminator, self).__init__(noise_dim, add_noise_channel)
        self.sigma_x = additional_noise_sigma_x
        self.sigma_z = additional_noise_sigma_z
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.noise_dim = noise_dim
        self.add_gaussian_noise_to_latent = add_gaussian_noise_to_latent

        self.to_feature = to_feature
        self.feature_dim = to_feature.output_dim
        self.device = device

        self.to_ratio = torch.nn.ModuleDict()
        self.to_ratio['zu'] = get_fc(input_dim=dim_z + dim_u + noise_dim,
                                     hidden_dims=[],
                                     output_dim=self.feature_dim,
                                     lrelu_negative_slope=negative_slope,
                                     use_batchnorm=use_batchnorm,
                                     batchnorm_kwargs=batchnorm_kwargs,
                                     last_activation_type='lrelu')
        self.to_ratio['hxzu'] = get_fc(input_dim=2 * self.feature_dim + noise_dim,
                                       hidden_dim=hidden_dim,
                                       n_hidden_layers=n_hidden_layers,
                                       output_dim=1,
                                       lrelu_negative_slope=negative_slope,
                                       use_batchnorm=use_batchnorm,
                                       batchnorm_kwargs=batchnorm_kwargs)

    def forward(self, hx, z, u):
        if self.add_gaussian_noise_to_latent:
            z = self.add_gaussian_noise(z, self.sigma_z)
            u = self.add_gaussian_noise(u, self.sigma_u)
        zu = self.append_noise_and_concat(z, u)
        hzu = self.to_ratio['zu'](zu)

        hxzu = self.append_noise_and_concat(hx, hzu)
        hxzu = self.to_ratio['hxzu'](hxzu)

        return hxzu


# marginal encoder + local encoder
class MarginalEncoder(RandomFunction):
    def __init__(self,
                 dim_z=128 * 2,
                 dim_u=128,
                 prior_type='gaussian',
                 degenerate=False,
                 simple_local=True,
                 to_feature=None,
                 n_hidden_layers=0,
                 hidden_dim=0,
                 negative_slope=0.0,
                 use_batchnorm=False,
                 batchnorm_kwargs=None,
                 noise_dim=20,
                 add_noise_channel=True,
                 additional_noise_sigma=0.0,
                 device=None):
        super(MarginalEncoder, self).__init__(noise_dim, add_noise_channel)
        self.sigma = additional_noise_sigma
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.degenerate = degenerate
        self.simple_local = simple_local
        self.to_feature = to_feature
        feature_dim = self.to_feature.output_dim

        self.to_latent = nn.ModuleDict()
        self.to_latent['hx_z'] = get_fc(input_dim=feature_dim + noise_dim,
                                        hidden_dim=hidden_dim,
                                        n_hidden_layers=n_hidden_layers,
                                        output_dim=dim_z,
                                        lrelu_negative_slope=negative_slope,
                                        last_activation_type='tanh' if prior_type == 'uniform' else None,
                                        use_batchnorm=use_batchnorm,
                                        batchnorm_kwargs=batchnorm_kwargs)  # (B, dim_z)
        if simple_local or degenerate:
            self.to_latent['zhx_u'] = get_fc(input_dim=feature_dim + int(not self.degenerate) * dim_z + noise_dim,
                                             hidden_dim=hidden_dim,
                                             n_hidden_layers=n_hidden_layers,
                                             output_dim=dim_u,
                                             lrelu_negative_slope=negative_slope,
                                             last_activation_type='tanh' if prior_type == 'uniform' else None,
                                             use_batchnorm=use_batchnorm,
                                             batchnorm_kwargs=batchnorm_kwargs)  # (B, dim_u)
        else:
            self.to_latent['z_hz'] = get_fc(input_dim=dim_z + noise_dim,
                                            hidden_dim=hidden_dim,
                                            n_hidden_layers=n_hidden_layers,
                                            output_dim=feature_dim,
                                            lrelu_negative_slope=negative_slope,
                                            use_batchnorm=use_batchnorm,
                                            batchnorm_kwargs=batchnorm_kwargs,
                                            last_activation_type='lrelu')  # (B, feature_dim)
            self.to_latent['hzx_u'] = get_fc(input_dim=2 * feature_dim,
                                             hidden_dim=hidden_dim,
                                             n_hidden_layers=n_hidden_layers,
                                             output_dim=dim_u,
                                             lrelu_negative_slope=negative_slope,
                                             last_activation_type='tanh' if prior_type == 'uniform' else None,
                                             use_batchnorm=use_batchnorm,
                                             batchnorm_kwargs=batchnorm_kwargs)  # (B, dim_u)

        self.device = device

    def get_random_feature(self, x):
        x = self.add_gaussian_noise(x, self.sigma)
        x = self.append_noise_and_concat(x)  # (B, cc_x [+ 1], 32, 32)
        hx = self.to_feature(x)  # (B, feature_dim)
        hx = torch.flatten(hx, start_dim=1)
        return hx

    def encode_x_z(self, x):  # q(z|x)
        hx = self.get_random_feature(x)
        return self.encode_hx_z(hx)

    def encode_hx_z(self, hx):
        return self.to_latent['hx_z'](self.append_noise_and_concat(hx))  # (B, dim_z)

    def encode_zx_u(self, z, x):  # q(u|zx)
        hx = self.get_random_feature(x)
        return self.encode_zhx_u(z, hx)

    def encode_zhx_u(self, z, hx):
        if self.simple_local or self.degenerate:
            if self.degenerate:
                zhx = self.append_noise_and_concat(hx)
            else:
                zhx = self.append_noise_and_concat(z, hx)
            return self.to_latent['zhx_u'](zhx)
        else:
            z = self.append_noise_and_concat(z)
            hz = self.to_latent['z_hz'](self.append_noise_and_concat(z))
            hzx = self.append_noise_and_concat(hz, hx)
            return self.to_latent['hzx_u'](hzx)

    def encode_x_zu(self, x):  # q(z|x)q(u|zx)
        hx = self.get_random_feature(x)
        return self.encode_hx_zu(hx)

    def encode_hx_zu(self, hx):
        z = self.encode_hx_z(hx)  # (B, dim_z)
        u = self.encode_zhx_u(z, hx)
        return z, u


class Decoder(nn.Module):
    def __init__(self, to_data, to_feature=None):
        """Parameters
        ----------
        to_data
        """
        super().__init__()
        self.to_data = to_data
        self.to_feature = to_feature

    def forward(self, z, u):
        zu = torch.cat([z, u], dim=1)
        if self.to_feature is not None:
            zu = self.to_feature(zu)
        return self.to_data(zu)

    def decode(self, zu):
        """
        Decode method for compatibility with AIS evaluation
        """
        return self.to_data(self.to_feature(zu))


class Generator(nn.Module):
    def __init__(self, encoder, decoder, prior):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.dim_z = self.encoder.dim_z
        self.dim_u = self.encoder.dim_u
        self.latent_dim = self.dim_z + self.dim_u
        self.device = encoder.device

    def draw_from_prior(self, n, device=None):
        return self.priors['z'].draw(n, device), self.priors['u'].draw(n, device)

    def forward(self, xq, reconstruction=True):
        batch_size = xq.shape[0]

        # model distribution
        zp, up = self.draw_from_prior(n=batch_size, device=xq.device)
        xp = self.decoder(zp, up)

        # data + variational distribution
        zqx, uqx = self.encoder.encode_x_zu(xq)

        # reconstruction
        if reconstruction:
            xq_hat = self.decoder(zqx, uqx)
            zp_hat, up_hat = self.encoder.encode_x_zu(xp)
        else:
            xq_hat = None
            zp_hat, up_hat = None, None

        return zqx, uqx, xp, zp, up, \
               xq_hat, zp_hat, up_hat


class Model(nn.Module):
    def __init__(self, generator, discriminator,
                 gener_loss_fn, recon_loss_fn, discr_loss_fn,
                 loss_weights):
        super(Model, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gener_loss_fn = gener_loss_fn
        self.recon_loss_fn = recon_loss_fn
        self.discr_loss_fn = discr_loss_fn
        self.loss_weights = loss_weights

        self.dim_z = self.generator.dim_z
        self.dim_u = self.generator.dim_u
        self.dim_v = self.generator.dim_v

        self.apply(weights_init)

    def forward(self, xq, reconstruction=True):
        gener_outputs = self.generator(xq, reconstruction)
        (zqx, uqx, xp, zp, up,
         xq_hat, zp_hat, up_hat) = gener_outputs

        # ratios
        hxq_disc = self.discriminator.get_random_feature(xq)
        hxp_disc = self.discriminator.get_random_feature(xp)

        ratios = defaultdict(lambda: None)
        ratios['var'] = self.discriminator(hxq_disc, zqx, uqx)
        ratios['model'] = self.discriminator(hxp_disc, zp, up)

        return ratios

    def compute_discriminator_losses(self, xq):
        *_, ratios = self(xq, reconstruction=False)
        discr_loss = - self.discr_loss_fn(ratios['var'], ratios['model'])

        return discr_loss

    def compute_generator_losses(self, xq):
        gener_outputs, ratios = self(xq, reconstruction=True)
        (zqx, uqx, xp, zp, up,
         xq_hat, zp_hat, up_hat) = gener_outputs

        model_losses = defaultdict(float)
        model_losses['dist'] = self.model_loss_fn(ratios['var'], ratios['model'])
        if self.loss_weights.rec > 0:
            model_losses['rec_zu'] = self.recon_loss_fn(zp, zp_hat) + self.recon_loss_fn(up, up_hat)
            model_losses['rec_x'] = self.recon_loss_fn(xq, xq_hat)
        model_losses['rec'] = model_losses['rec_zu'] + model_losses['rec_x']
        model_losses['total'] = model_losses['dist'] + self.loss_weights.rec * model_losses['rec']

        return model_losses
