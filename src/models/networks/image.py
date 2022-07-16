from torch import nn as nn

from models.networks.base import ResBlock


def get_networks(cc,
                 discriminator_base_cc=32,
                 discriminator_negative_slope=0.2,
                 discriminator_add_noise_channel=True,
                 encoder_base_cc=16,
                 encoder_negative_slope=0.2,
                 encoder_add_noise_channel=True,
                 decoder_base_cc=32,
                 decoder_negative_slope=0.0,
                 discriminator_use_batchnorm=False,
                 discriminator_use_batchnorm_first_layer=False,
                 generator_use_batchnorm=True,
                 batchnorm_kwargs=None,
                 use_spectralnorm=False,
                 last_activation=True,
                 n_resblocks=0,
                 kernel_size=5,
                 img_size=32):
    if batchnorm_kwargs is None:
        batchnorm_kwargs = {}
    discriminator_conv_net = ConvNet(
        cin=cc + int(discriminator_add_noise_channel),
        base_cc=discriminator_base_cc,
        negative_slope=discriminator_negative_slope,
        use_batchnorm=discriminator_use_batchnorm,
        use_batchnorm_first_layer=discriminator_use_batchnorm_first_layer,
        batchnorm_kwargs=batchnorm_kwargs,
        use_spectralnorm=use_spectralnorm,
        n_resblocks=n_resblocks,
        kernel_size=kernel_size,
        last_activation=last_activation,
    )

    encoder_conv_net = ConvNet(
        cin=cc + int(encoder_add_noise_channel),
        base_cc=encoder_base_cc,
        negative_slope=encoder_negative_slope,
        use_batchnorm=generator_use_batchnorm,
        batchnorm_kwargs=batchnorm_kwargs,
        use_spectralnorm=False,
        n_resblocks=n_resblocks,
        kernel_size=kernel_size,
        last_activation=last_activation,
    )

    deconv_net = DeconvNet(
        base_cc=decoder_base_cc,
        cout=cc,
        negative_slope=decoder_negative_slope,
        use_batchnorm=generator_use_batchnorm,
        batchnorm_kwargs=batchnorm_kwargs,
        n_resblocks=n_resblocks,
        kernel_size=kernel_size,
        img_size=img_size,
    )

    return discriminator_conv_net, encoder_conv_net, deconv_net


class ConvNet(nn.Module):
    def __init__(self, cin=3, base_cc=32, negative_slope=0.2,
                 use_batchnorm=True,
                 use_batchnorm_first_layer=None,
                 batchnorm_kwargs=None,
                 use_spectralnorm=False,
                 n_resblocks=0, kernel_size=5,
                 last_activation=True):
        super(ConvNet, self).__init__()
        if use_batchnorm_first_layer is None:
            use_batchnorm_first_layer = use_batchnorm
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        assert kernel_size in [4, 5]
        padding = 1 if kernel_size == 4 else 2

        self.output_dim = (32 // (2 ** 4)) ** 2 * (base_cc * (2 ** 3))  # 1024 for base_cc=32

        def get_conv_layer(*args, **kwargs):
            conv_layer = nn.Conv2d(*args, **kwargs)
            if use_spectralnorm:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            return conv_layer

        modules = [get_conv_layer(cin, base_cc, kernel_size, stride=2, padding=padding),
                   nn.BatchNorm2d(num_features=base_cc, **batchnorm_kwargs) if use_batchnorm_first_layer else None,
                   nn.LeakyReLU(negative_slope=negative_slope),
                   # (img_size//2, img_size//2)
                   get_conv_layer(base_cc, 2 * base_cc, kernel_size, stride=2, padding=padding),
                   nn.BatchNorm2d(num_features=2 * base_cc, **batchnorm_kwargs) if use_batchnorm else None,
                   nn.LeakyReLU(negative_slope=negative_slope),
                   # (img_size//4, img_size//4)
                   get_conv_layer(2 * base_cc, 4 * base_cc, kernel_size, stride=2, padding=padding),
                   nn.BatchNorm2d(num_features=4 * base_cc, **batchnorm_kwargs) if use_batchnorm else None,
                   nn.LeakyReLU(negative_slope=negative_slope),
                   # (img_size // 8, img_size // 8)
                   *[ResBlock(3, base_cc=4 * base_cc, negative_slope=negative_slope,
                              batchnorm_kwargs=batchnorm_kwargs) for _ in range(n_resblocks)],
                   get_conv_layer(4 * base_cc, 8 * base_cc, kernel_size, stride=2, padding=padding),
                   nn.BatchNorm2d(num_features=8 * base_cc,
                                  **batchnorm_kwargs) if use_batchnorm and last_activation else None,
                   nn.LeakyReLU(negative_slope=negative_slope) if last_activation else None,
                   # (img_size//16, img_size//16)
                   ]
        self.network = nn.Sequential(*list(filter(None, modules)))

    def forward(self, x):
        return self.network(x)


class DeconvNet(nn.Module):
    def __init__(self, base_cc=64, cout=3, negative_slope=0.2,
                 use_batchnorm=True, batchnorm_kwargs=None,
                 n_resblocks=0, kernel_size=5, img_size=32):
        super(DeconvNet, self).__init__()
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        assert kernel_size in [4, 5]
        padding = 1 if kernel_size == 4 else 2

        self.base_cc = base_cc

        # CIFAR decoder
        self.img_size = img_size
        self.input_dim = (img_size // 8) ** 2 * (8 * base_cc)

        self.deconv1 = nn.ConvTranspose2d(base_cc * 8, base_cc * 4, kernel_size, stride=2, padding=padding)
        self.resblock = nn.Sequential(*[ResBlock(3, base_cc=4 * base_cc, negative_slope=negative_slope,
                                                 batchnorm_kwargs=batchnorm_kwargs) for _ in range(n_resblocks)])
        self.activation1 = nn.Sequential(
            *list(filter(None, [nn.BatchNorm2d(num_features=base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
                                nn.LeakyReLU(negative_slope=negative_slope)])),
        )
        self.deconv2 = nn.ConvTranspose2d(base_cc * 4, base_cc * 2, kernel_size, stride=2, padding=padding)
        self.activation2 = nn.Sequential(
            *list(filter(None, [nn.BatchNorm2d(num_features=base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
                                nn.LeakyReLU(negative_slope=negative_slope)])),
        )
        self.deconv3 = nn.ConvTranspose2d(base_cc * 2, cout, kernel_size, stride=2, padding=padding)

    def forward(self, z):
        z = z.reshape([-1, self.base_cc * 8, self.img_size // 8, self.img_size // 8])
        h = self.deconv1(z, output_size=(self.img_size // 4, self.img_size // 4))
        h = self.resblock(h)
        h = self.activation1(h)
        h = self.deconv2(h, output_size=(self.img_size // 2, self.img_size // 2))
        h = self.activation2(h)
        h = self.deconv3(h, output_size=(self.img_size, self.img_size))
        h = nn.Tanh()(h)

        return h


class SymmetricDeconvNet(nn.Module):
    def __init__(self, base_cc=64, cout=3, negative_slope=0.2,
                 use_batchnorm=True, batchnorm_kwargs=None,
                 n_resblocks=0, kernel_size=5, img_size=32):
        super(SymmetricDeconvNet, self).__init__()
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        assert kernel_size in [4, 5]
        padding = 1 if kernel_size == 4 else 2

        self.base_cc = base_cc

        # CIFAR decoder
        self.img_size = img_size
        self.input_dim = (img_size // 16) ** 2 * (8 * base_cc)

        self.deconv1 = nn.ConvTranspose2d(base_cc * 8, base_cc * 4, kernel_size, stride=2, padding=padding)
        self.resblock = nn.Sequential(*[ResBlock(3, base_cc=4 * base_cc, negative_slope=negative_slope,
                                                 batchnorm_kwargs=batchnorm_kwargs) for _ in range(n_resblocks)])
        self.activation1 = nn.Sequential(
            *list(filter(None, [nn.BatchNorm2d(num_features=base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
                                nn.LeakyReLU(negative_slope=negative_slope)])),
        )
        self.deconv2 = nn.ConvTranspose2d(base_cc * 4, base_cc * 2, kernel_size, stride=2, padding=padding)
        self.activation2 = nn.Sequential(
            *list(filter(None, [nn.BatchNorm2d(num_features=base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
                                nn.LeakyReLU(negative_slope=negative_slope)])),
        )
        self.deconv3 = nn.ConvTranspose2d(base_cc * 2, base_cc, kernel_size, stride=2, padding=padding)
        self.activation3 = nn.Sequential(
            *list(filter(None, [nn.BatchNorm2d(num_features=base_cc, **batchnorm_kwargs) if use_batchnorm else None,
                                nn.LeakyReLU(negative_slope=negative_slope)])),
        )
        self.deconv4 = nn.ConvTranspose2d(base_cc, cout, kernel_size, stride=2, padding=padding)

    def forward(self, z):
        z = z.reshape([-1, self.base_cc * 8, self.img_size // 16, self.img_size // 16])
        h = self.deconv1(z, output_size=(self.img_size // 8, self.img_size // 8))
        h = self.resblock(h)
        h = self.activation1(h)
        h = self.deconv2(h, output_size=(self.img_size // 4, self.img_size // 4))
        h = self.activation2(h)
        h = self.deconv3(h, output_size=(self.img_size // 2, self.img_size // 2))
        h = self.activation3(h)
        h = self.deconv4(h, output_size=(self.img_size, self.img_size))
        h = nn.Tanh()(h)

        return h
