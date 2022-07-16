from models.networks.base import JointFeatureMap
from models.networks.image import ConvNet
from models.networks.image import get_networks as get_image_networks


def get_networks(cc_x=1, cc_y=3,
                 discriminator_base_cc_x=32,
                 discriminator_base_cc_y=32,
                 discriminator_base_cc_xy=32,
                 discriminator_negative_slope=0.2,
                 discriminator_add_noise_channel=True,
                 encoder_base_cc_x=16,
                 encoder_base_cc_y=16,
                 encoder_base_cc_xy=16,
                 encoder_negative_slope=0.2,
                 decoder_negative_slope=0.0,
                 joint_encoder_add_noise_channel=True,
                 marginal_encoder_add_noise_channel=True,
                 decoder_base_cc_x=16,
                 decoder_base_cc_y=16,
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
    (discriminator_conv_net_x, encoder_conv_net_x, deconv_net_x) = \
        get_image_networks(cc=cc_x,
                           discriminator_base_cc=discriminator_base_cc_x,
                           discriminator_negative_slope=discriminator_negative_slope,
                           discriminator_add_noise_channel=discriminator_add_noise_channel,
                           encoder_base_cc=encoder_base_cc_x,
                           encoder_negative_slope=encoder_negative_slope,
                           encoder_add_noise_channel=marginal_encoder_add_noise_channel,
                           decoder_base_cc=decoder_base_cc_x,
                           decoder_negative_slope=decoder_negative_slope,
                           discriminator_use_batchnorm=discriminator_use_batchnorm,
                           discriminator_use_batchnorm_first_layer=discriminator_use_batchnorm_first_layer,
                           generator_use_batchnorm=generator_use_batchnorm,
                           batchnorm_kwargs=batchnorm_kwargs,
                           use_spectralnorm=use_spectralnorm,
                           last_activation=last_activation,
                           n_resblocks=n_resblocks,
                           kernel_size=kernel_size,
                           img_size=img_size)
    (discriminator_conv_net_y, encoder_conv_net_y, deconv_net_y) = \
        get_image_networks(cc=cc_y,
                           discriminator_base_cc=discriminator_base_cc_y,
                           discriminator_negative_slope=discriminator_negative_slope,
                           discriminator_add_noise_channel=discriminator_add_noise_channel,
                           encoder_base_cc=encoder_base_cc_y,
                           encoder_negative_slope=encoder_negative_slope,
                           encoder_add_noise_channel=marginal_encoder_add_noise_channel,
                           decoder_base_cc=decoder_base_cc_y,
                           decoder_negative_slope=decoder_negative_slope,
                           discriminator_use_batchnorm=discriminator_use_batchnorm,
                           discriminator_use_batchnorm_first_layer=discriminator_use_batchnorm_first_layer,
                           generator_use_batchnorm=generator_use_batchnorm,
                           batchnorm_kwargs=batchnorm_kwargs,
                           use_spectralnorm=use_spectralnorm,
                           last_activation=last_activation,
                           n_resblocks=n_resblocks,
                           kernel_size=kernel_size,
                           img_size=img_size)

    # joint discriminator / encoder
    discriminator_conv_net_xy = JointFeatureMap(
        aggregate=ConvNet(cin=cc_x + cc_y + 2 * int(discriminator_add_noise_channel),
                          base_cc=discriminator_base_cc_xy,
                          negative_slope=discriminator_negative_slope,
                          use_batchnorm=discriminator_use_batchnorm,
                          use_batchnorm_first_layer=discriminator_use_batchnorm_first_layer,
                          batchnorm_kwargs=batchnorm_kwargs,
                          use_spectralnorm=use_spectralnorm,
                          last_activation=last_activation,
                          n_resblocks=n_resblocks,
                          kernel_size=kernel_size))
    encoder_conv_net_xy = JointFeatureMap(
        aggregate=ConvNet(cin=cc_x + cc_y + 2 * int(joint_encoder_add_noise_channel),
                          base_cc=encoder_base_cc_xy,
                          negative_slope=encoder_negative_slope,
                          use_batchnorm=generator_use_batchnorm,
                          batchnorm_kwargs=batchnorm_kwargs,
                          use_spectralnorm=False,
                          last_activation=last_activation,
                          n_resblocks=n_resblocks,
                          kernel_size=kernel_size))

    return discriminator_conv_net_x, discriminator_conv_net_y, discriminator_conv_net_xy, \
           encoder_conv_net_x, encoder_conv_net_y, encoder_conv_net_xy, \
           deconv_net_x, deconv_net_y
