import os

import torch

from models.autoencoders.main import build_cub_sent_ae
from models.networks.base import get_fc, JointFeatureMap
from models.networks.cub_sent import CUBSentFeatureMap, CUBSentDecoderMap
from models.networks.resnet import ResnetEncoderMap, ResnetDecoderMap

vocab_size = 1590


def get_pretrained_sent_ae(base_cc=32, device=None):
    pretrained_sent_ae = build_cub_sent_ae(base_cc=base_cc, conv2d=False)
    pretrained_model_path = os.path.join(os.path.dirname(__file__),
                                         '../../../pretrained/autoencoders/cub-sent-conv1d-ptemb-lr1e-4/model.rar')
    pretrained_sent_ae.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    pretrained_sent_ae.encoder.to_feature.freeze_conv = True
    pretrained_sent_ae.decoder.freeze_conv = True

    return pretrained_sent_ae


def get_networks(config, device):
    batchnorm_kwargs = dict(momentum=config.batchnorm_momentum, eps=config.batchnorm_eps)
    # vocab related parameters
    # max_sentence_length = 32
    # dim_y = max_sentence_length
    conv2d = config.cub_sent_use_conv2d
    base_cc_y = 32

    if config.cub_sent_use_pretrained_ae:
        assert not conv2d
        assert not config.cub_sent_tie_embedding
        assert not config.cub_sent_use_pretrained_embedding
        pretrained_sent_ae = get_pretrained_sent_ae(base_cc_y, device)
        to_feature_y = pretrained_sent_ae.encoder.to_feature.freeze()
        to_data_y = pretrained_sent_ae.decoder.to_data.freeze()  # freeze() is needed to avoid random initialization!
    else:
        to_feature_y = CUBSentFeatureMap(vocab_size=vocab_size,
                                         embedding_dim=config.cub_sent_embedding_dim,
                                         base_cc=base_cc_y,
                                         negative_slope=config.encoder_negative_slope,
                                         normalize_embedding=config.cub_sent_normalize_embedding,
                                         embedding_noise_scale=0.,
                                         use_batchnorm=config.generator_feature_use_batchnorm,
                                         batchnorm_kwargs=batchnorm_kwargs,
                                         last_activation=config.feature_map_last_activation,
                                         conv2d=conv2d)
        to_data_y = CUBSentDecoderMap(vocab_size=vocab_size,
                                      embedding_dim=config.cub_sent_embedding_dim,
                                      normalize_embedding=config.cub_sent_normalize_embedding,
                                      base_cc=base_cc_y,
                                      negative_slope=config.decoder_negative_slope,
                                      use_batchnorm=config.generator_feature_use_batchnorm,
                                      batchnorm_kwargs=batchnorm_kwargs,
                                      conv2d=conv2d)

    feature_dim_xy = 512  # feature_dim_x + feature_dim_y
    if 'imgft' in config.dataset:
        dim_x = 2048
        encoder_hidden_dims_x = (1024, 512)  # imgft
        feature_dim_x = 256
        to_feature_x = get_fc(input_dim=dim_x + config.marginal_encoder_noise_dim,
                              hidden_dims=encoder_hidden_dims_x,
                              output_dim=feature_dim_x,
                              lrelu_negative_slope=config.encoder_negative_slope,
                              use_batchnorm=config.cub_imgft_encoder_use_batchnorm,
                              batchnorm_kwargs=batchnorm_kwargs,
                              last_activation_type=config.cub_imgft_activation if config.feature_map_last_activation else None,
                              use_spectralnorm=False,
                              activation_type=config.cub_imgft_activation)

        to_feature_xy = JointFeatureMap(
            to_feature_x=get_fc(input_dim=dim_x + config.marginal_encoder_noise_dim,
                                hidden_dims=encoder_hidden_dims_x,
                                output_dim=feature_dim_x,
                                lrelu_negative_slope=config.encoder_negative_slope,
                                use_batchnorm=config.cub_imgft_encoder_use_batchnorm,
                                batchnorm_kwargs=batchnorm_kwargs,
                                last_activation_type='lrelu' if config.feature_map_last_activation else None,
                                use_spectralnorm=False,
                                activation_type=config.cub_imgft_activation),
            to_feature_y=CUBSentFeatureMap(vocab_size=vocab_size,
                                           embedding_dim=config.cub_sent_embedding_dim,
                                           base_cc=base_cc_y,
                                           negative_slope=config.encoder_negative_slope,
                                           normalize_embedding=config.cub_sent_normalize_embedding,
                                           embedding_noise_scale=0.,
                                           use_batchnorm=config.generator_feature_use_batchnorm,
                                           batchnorm_kwargs=batchnorm_kwargs,
                                           last_activation=config.feature_map_last_activation,
                                           use_spectralnorm=False,
                                           conv2d=conv2d) if not config.cub_sent_use_pretrained_ae else to_feature_y,
            aggregate=get_fc(input_dim=feature_dim_x + to_feature_y.output_dim,
                             hidden_dims=2 * (2 * feature_dim_xy,),
                             output_dim=feature_dim_xy,
                             lrelu_negative_slope=config.encoder_negative_slope,
                             last_activation_type='lrelu' if config.feature_map_last_activation else None,
                             use_batchnorm=config.generator_feature_use_batchnorm,
                             batchnorm_kwargs=batchnorm_kwargs,
                             use_spectralnorm=False),
        )

        discriminator_feature_dim_xy = 2 * feature_dim_xy  # feature_dim_x + discriminator_feature_dim_y
        discriminator_base_cc_y = 2 * base_cc_y
        if config.cub_sent_use_pretrained_ae:
            pretrained_sent_ae = get_pretrained_sent_ae(base_cc_y, device)
            pretrained_sent_ae.encoder.to_feature.embedding_noise_scale = config.instance_noise_scale_y
            discriminator_to_feature_y = pretrained_sent_ae.encoder.to_feature.freeze()
        else:
            discriminator_to_feature_y = CUBSentFeatureMap(vocab_size=vocab_size,
                                                           embedding_dim=config.cub_sent_embedding_dim,
                                                           base_cc=discriminator_base_cc_y,
                                                           negative_slope=config.discriminator_negative_slope,
                                                           normalize_embedding=config.cub_sent_normalize_embedding,
                                                           embedding_noise_scale=config.instance_noise_scale_y,
                                                           adaptive_embedding_noise=config.cub_sent_adaptive_embedding_noise,
                                                           use_batchnorm=config.discriminator_feature_use_batchnorm,
                                                           use_batchnorm_first_layer=config.discriminator_feature_use_batchnorm_first_layer,
                                                           batchnorm_kwargs=batchnorm_kwargs,
                                                           use_spectralnorm=config.discriminator_feature_use_spectralnorm,
                                                           last_activation=config.feature_map_last_activation,
                                                           conv2d=conv2d)

        discriminator_hidden_dims_x = (2048, 1024)
        discriminator_to_feature_xy = JointFeatureMap(
            to_feature_x=get_fc(input_dim=dim_x,
                                hidden_dims=discriminator_hidden_dims_x,
                                output_dim=2 * feature_dim_x,
                                lrelu_negative_slope=config.discriminator_negative_slope,
                                last_activation_type='lrelu' if config.feature_map_last_activation else None,
                                # use_batchnorm=config.cub_imgft_use_batchnorm,
                                use_batchnorm=config.discriminator_feature_use_batchnorm,
                                use_batchnorm_first_layer=config.discriminator_feature_use_batchnorm_first_layer,
                                batchnorm_kwargs=batchnorm_kwargs,
                                use_spectralnorm=config.discriminator_feature_use_spectralnorm,
                                activation_type=config.cub_imgft_activation),
            to_feature_y=discriminator_to_feature_y,
            aggregate=get_fc(input_dim=2 * feature_dim_x + discriminator_to_feature_y.output_dim,
                             hidden_dims=2 * (2 * discriminator_feature_dim_xy,),
                             output_dim=discriminator_feature_dim_xy,
                             lrelu_negative_slope=config.discriminator_negative_slope,
                             last_activation_type='lrelu' if config.feature_map_last_activation else None,
                             use_batchnorm=config.discriminator_feature_use_batchnorm,
                             batchnorm_kwargs=batchnorm_kwargs,
                             use_spectralnorm=config.discriminator_feature_use_spectralnorm),
        )

        decoder_hidden_dims = [512, 1024]
        to_data_x = get_fc(input_dim=feature_dim_x,
                           hidden_dims=decoder_hidden_dims,
                           output_dim=dim_x,
                           lrelu_negative_slope=config.decoder_negative_slope,
                           use_batchnorm=config.cub_imgft_decoder_use_batchnorm,
                           batchnorm_kwargs=batchnorm_kwargs,
                           activation_type=config.cub_imgft_activation,
                           last_activation_type=None)

    else:  # raw CUB image
        base_cc_x = config.base_cc
        to_feature_x = ResnetEncoderMap(imgsize=config.cub_imgsize, nfilter=base_cc_x)
        to_data_x = ResnetDecoderMap(imgsize=config.cub_imgsize, nfilter=base_cc_x)

        to_feature_xy = JointFeatureMap(
            to_feature_x=ResnetEncoderMap(imgsize=config.cub_imgsize, nfilter=base_cc_x),
            to_feature_y=CUBSentFeatureMap(vocab_size=vocab_size,
                                           embedding_dim=config.cub_sent_embedding_dim,
                                           base_cc=base_cc_y,
                                           negative_slope=config.encoder_negative_slope,
                                           normalize_embedding=config.cub_sent_normalize_embedding,
                                           embedding_noise_scale=0.,
                                           use_batchnorm=config.generator_feature_use_batchnorm,
                                           batchnorm_kwargs=batchnorm_kwargs,
                                           last_activation=config.feature_map_last_activation,
                                           use_spectralnorm=False,
                                           conv2d=conv2d),
            aggregate=get_fc(input_dim=to_feature_x.output_dim + to_feature_y.output_dim,
                             hidden_dims=2 * (2 * feature_dim_xy,),
                             output_dim=feature_dim_xy,
                             lrelu_negative_slope=config.encoder_negative_slope,
                             last_activation_type='lrelu' if config.feature_map_last_activation else None,
                             use_batchnorm=config.generator_feature_use_batchnorm,
                             batchnorm_kwargs=batchnorm_kwargs,
                             use_spectralnorm=False),
        )

        discriminator_feature_dim_xy = 2 * feature_dim_xy
        discriminator_to_feature_xy = JointFeatureMap(
            to_feature_x=ResnetEncoderMap(imgsize=config.cub_imgsize, nfilter=base_cc_x),
            to_feature_y=CUBSentFeatureMap(vocab_size=vocab_size,
                                           embedding_dim=config.cub_sent_embedding_dim,
                                           base_cc=2 * base_cc_y,
                                           negative_slope=config.discriminator_negative_slope,
                                           normalize_embedding=config.cub_sent_normalize_embedding,
                                           embedding_noise_scale=0.,
                                           use_batchnorm=config.discriminator_feature_use_batchnorm,
                                           batchnorm_kwargs=batchnorm_kwargs,
                                           last_activation=config.feature_map_last_activation,
                                           use_spectralnorm=config.discriminator_feature_use_spectralnorm,
                                           conv2d=conv2d),
            aggregate=get_fc(input_dim=(to_feature_x.output_dim + 2 * to_feature_y.output_dim),
                             hidden_dims=2 * (2 * discriminator_feature_dim_xy,),
                             output_dim=discriminator_feature_dim_xy,
                             lrelu_negative_slope=config.discriminator_negative_slope,
                             last_activation_type='lrelu' if config.feature_map_last_activation else None,
                             use_batchnorm=config.discriminator_feature_use_batchnorm,
                             batchnorm_kwargs=batchnorm_kwargs,
                             use_spectralnorm=config.discriminator_feature_use_spectralnorm),
        )

    return discriminator_to_feature_xy, \
           to_feature_x, to_feature_y, to_feature_xy, \
           to_data_x, to_data_y
