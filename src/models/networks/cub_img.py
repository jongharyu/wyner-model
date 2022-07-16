# CUB Image model specification
import torch.nn as nn


class CUBImagesFeatureMap(nn.Module):
    """ Convolutional network for CUB images
    Note:
        - The architecture is adapted from the MMVAE paper.
        - cin=3, base_cc=64, negative_slope=0., use_batchnorm=False gives the original architecture used in MMVAE
    """
    def __init__(self, cin=3, base_cc=64, negative_slope=0.2,
                 use_batchnorm=False,
                 batchnorm_kwargs=None,
                 last_activation=True):
        super(CUBImagesFeatureMap, self).__init__()
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        modules = [
            # input size: 3 x 128 x 128
            nn.Conv2d(cin, base_cc, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # input size: base_cc x 64 x 64
            nn.Conv2d(base_cc, base_cc * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # size: (base_cc * 2) x 32 x 32
            nn.Conv2d(base_cc * 2, base_cc * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # size: (base_cc * 4) x 16 x 16
            nn.Conv2d(base_cc * 4, base_cc * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc * 8, **batchnorm_kwargs) if use_batchnorm and last_activation else None,
            nn.LeakyReLU(negative_slope=negative_slope) if last_activation else None,
            # size: (base_cc * 8) x 4 x 4
        ]
        self.network = nn.Sequential(*[module for module in modules if module is not None])

    def forward(self, x):
        return self.network(x)


class CUBImagesDecoderMap(nn.Module):
    """ Generate an image given a latent variable. """
    def __init__(self, dim_z, base_cc=64, cout=3, negative_slope=0.2,
                 batchnorm_kwargs=None,
                 use_batchnorm=False):
        super(CUBImagesDecoderMap, self).__init__()
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        self.dim_z = dim_z
        self.base_cc = base_cc
        self.network = nn.Sequential(
            nn.ConvTranspose2d(dim_z, base_cc * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(num_features=base_cc * 8, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # size: (base_cc * 8) x 16 x 16
            nn.ConvTranspose2d(base_cc * 8, base_cc * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # size: (base_cc * 4) x 16 x 16
            nn.ConvTranspose2d(base_cc * 4, base_cc * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # size: (base_cc * 2) x 32 x 32
            nn.ConvTranspose2d(base_cc * 2, base_cc, 4, 2, 1, bias=True),
            nn.BatchNorm2d(num_features=base_cc, **batchnorm_kwargs) if use_batchnorm else None,
            nn.LeakyReLU(negative_slope=negative_slope),
            # size: (base_cc) x 64 x 64
            nn.ConvTranspose2d(base_cc, cout, 4, 2, 1, bias=True),
            nn.Tanh(),
            # Output size: 3 x 128 x 128
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers (B, dim_z, 1, 1)
        return self.network(z)
