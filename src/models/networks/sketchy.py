from models.networks.base import get_fc, JointFeatureMap


def get_networks(config):
    # define models
    batchnorm_kwargs = dict(momentum=config.batchnorm_momentum, eps=config.batchnorm_eps)
    dim_x = dim_y = 512

    # hardcoding discriminator / encoder networks
    enc_feature_dim = 512
    discriminator_to_feature_xy = JointFeatureMap(
        get_fc(input_dim=dim_x + dim_y + config.discriminator_noise_dim,
               hidden_dims=[],
               output_dim=2 * enc_feature_dim,
               lrelu_negative_slope=config.discriminator_negative_slope,
               last_activation_type='lrelu' if config.feature_map_last_activation else None,
               use_batchnorm_first_layer=config.discriminator_feature_use_batchnorm_first_layer,
               use_batchnorm=config.discriminator_feature_use_batchnorm,
               batchnorm_kwargs=batchnorm_kwargs,
               use_spectralnorm=config.discriminator_feature_use_batchnorm))
    encoder_to_feature_x = get_fc(input_dim=dim_x + config.marginal_encoder_noise_dim,
                                  hidden_dims=[],
                                  output_dim=enc_feature_dim,
                                  lrelu_negative_slope=config.encoder_negative_slope,
                                  last_activation_type='lrelu' if config.feature_map_last_activation else None,
                                  use_batchnorm=config.generator_feature_use_batchnorm,
                                  batchnorm_kwargs=batchnorm_kwargs)
    encoder_to_feature_y = get_fc(input_dim=dim_y + config.marginal_encoder_noise_dim,
                                  hidden_dims=[],
                                  output_dim=enc_feature_dim,
                                  lrelu_negative_slope=config.encoder_negative_slope,
                                  last_activation_type='lrelu' if config.feature_map_last_activation else None,
                                  use_batchnorm=config.generator_feature_use_batchnorm,
                                  batchnorm_kwargs=batchnorm_kwargs)
    encoder_to_feature_xy = JointFeatureMap(
        get_fc(input_dim=dim_x + dim_y + config.joint_encoder_noise_dim,
               hidden_dims=[],
               output_dim=enc_feature_dim,
               lrelu_negative_slope=config.encoder_negative_slope,
               last_activation_type='lrelu' if config.feature_map_last_activation else None,
               use_batchnorm=config.generator_feature_use_batchnorm,
               batchnorm_kwargs=batchnorm_kwargs))

    # decoder maps
    dec_feature_dim = 128
    to_data_x = get_fc(input_dim=dec_feature_dim,
                       hidden_dims=[],
                       output_dim=dim_x,
                       lrelu_negative_slope=config.decoder_negative_slope,
                       use_batchnorm=config.generator_feature_use_batchnorm,
                       batchnorm_kwargs=batchnorm_kwargs)
    to_data_y = get_fc(input_dim=dec_feature_dim,
                       hidden_dims=[],
                       output_dim=dim_y,
                       lrelu_negative_slope=config.decoder_negative_slope,
                       use_batchnorm=config.generator_feature_use_batchnorm,
                       batchnorm_kwargs=batchnorm_kwargs)

    return discriminator_to_feature_xy, \
           encoder_to_feature_x, encoder_to_feature_y, encoder_to_feature_xy, \
           to_data_x, to_data_y
