import torch
import torch.nn as nn


class CUBSentFeatureMap(nn.Module):
    """ Generate latent parameters for (image, sentence) pair.
    Note:
        - The architecture is adapted from the MMVAE paper.
        - {negative_slope=0, additional_noise_sigma=0., use_batchnorm=True} gives the MMVAE architecture.
        - This architecture is hard-coded for max_sentence_length = 32.
    """
    def __init__(self, vocab_size=1590, embedding_dim=128,
                 base_cc=32,
                 negative_slope=0.2,
                 normalize_embedding=False,
                 embedding_noise_scale=0.,
                 adaptive_embedding_noise=False,
                 use_batchnorm=False,
                 use_batchnorm_first_layer=None,
                 batchnorm_kwargs=None,
                 use_spectralnorm=False,
                 last_activation=True,
                 conv2d=False,
                 freeze_conv=False):
        super(CUBSentFeatureMap, self).__init__()
        if use_batchnorm_first_layer is None:
            use_batchnorm_first_layer = use_batchnorm
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}

        self.base_cc = base_cc
        self.max_sentence_length = 32
        self.output_dim = base_cc * 16 * 4 * 4 if conv2d else base_cc * 16 * 4

        self.conv2d = conv2d
        self.freeze_conv = freeze_conv

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.normalize_embedding = normalize_embedding
        # (B, max_sentence_size=32) -> (B, max_sentence_size=32, embedding_dim=128)

        def get_conv_layer(*args, **kwargs):
            conv_layer = nn.Conv2d(*args, **kwargs) if conv2d else nn.Conv1d(*args, **kwargs)
            if use_spectralnorm:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            return conv_layer

        if conv2d:
            # input size: 1 x 32 x 128
            modules = [get_conv_layer(1, base_cc, 4, 2, 1, bias=True),
                       nn.BatchNorm2d(base_cc, **batchnorm_kwargs) if use_batchnorm_first_layer else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc) x 16 x 64
                       get_conv_layer(base_cc, base_cc * 2, 4, 2, 1, bias=True),
                       nn.BatchNorm2d(base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 2) x 8 x 32
                       get_conv_layer(base_cc * 2, base_cc * 4, 4, 2, 1, bias=True),
                       nn.BatchNorm2d(base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 4) x 4 x 16
                       get_conv_layer(base_cc * 4, base_cc * 8, (1, 4), (1, 2), (0, 1), bias=True),
                       nn.BatchNorm2d(base_cc * 8, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 8) x 4 x 8
                       get_conv_layer(base_cc * 8, base_cc * 16, (1, 4), (1, 2), (0, 1), bias=True),
                       nn.BatchNorm2d(base_cc * 16, **batchnorm_kwargs) if use_batchnorm and last_activation else None,
                       nn.LeakyReLU(negative_slope=negative_slope) if last_activation else None,
                       # size: (base_cc * 16) x 4 x 4
                       ]
        else:
            # input size: embedding_dim(=128) x max_sentence_size(=32)
            modules = [get_conv_layer(embedding_dim, base_cc, 4, 2, 1, bias=True),
                       nn.BatchNorm1d(base_cc, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc) x 16
                       get_conv_layer(base_cc, base_cc * 2, 4, 2, 1, bias=True),
                       nn.BatchNorm1d(base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 2) x 8
                       get_conv_layer(base_cc * 2, base_cc * 4, 4, 2, 1, bias=True),
                       nn.BatchNorm1d(base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 4) x 4
                       get_conv_layer(base_cc * 4, base_cc * 8, 1, 1, 0, bias=True),
                       nn.BatchNorm1d(base_cc * 8, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 8) x 4
                       get_conv_layer(base_cc * 8, base_cc * 16, 1, 1, 0, bias=True),
                       nn.BatchNorm1d(base_cc * 16, **batchnorm_kwargs) if use_batchnorm and last_activation else None,
                       nn.LeakyReLU(negative_slope=negative_slope) if last_activation else None,
                       # size: (base_cc * 16) x 4
                       ]
        self.network = nn.Sequential(*list(filter(None, modules)))
        self.embedding_noise_scale = torch.tensor(embedding_noise_scale)
        self.adaptive_embedding_noise = adaptive_embedding_noise
        self.freeze()

    def freeze(self):
        if self.freeze_conv:
            self.requires_grad_(False)
            self.eval()
        return self

    def forward(self, x):
        self.freeze()
        if len(x.shape) == 3 and x.shape[-1] == self.embedding.embedding_dim:
            xembed = x
        else:
            if len(x.shape) == 3 and x.shape[-1] == self.embedding.num_embeddings:
                # if x is (B, max_sentence_length, vocab_size)
                x = x.argmax(-1)  # (B, max_sentence_length)
            else:
                assert len(x.shape) == 2 and x.shape[-1] == self.max_sentence_length
            xembed = self.embedding(x.long())  # (B, max_sentence_size=32, embedding_dim=128)

        if self.normalize_embedding:
            # Note: adaptive_embedding_noise is ignored
            xembed = xembed / (torch.norm(xembed, p=2, dim=-1, keepdim=True) + 1e-10)
            sigma = self.embedding_noise_scale / torch.sqrt(torch.Tensor([xembed.shape[-1]])[0])
            xembed = xembed + sigma * torch.randn(xembed.shape).to(xembed.device)
            xembed = xembed / (torch.norm(xembed, p=2, dim=-1, keepdim=True) + 1e-10)
        else:
            sigma = self.embedding_noise_scale
            if self.adaptive_embedding_noise:
               sigma = self.embedding_noise_scale * self.embedding.weight.std(dim=0)
            xembed = xembed + sigma * torch.randn(xembed.shape).to(xembed.device)

        xembed = xembed.unsqueeze(1)
        if not self.conv2d:
            xembed = xembed.squeeze().permute(0, 2, 1)
        return self.network(xembed).view(x.shape[0], -1)  # (B, base_cc * 16, 4(, 4))


class CUBSentDecoderMap(nn.Module):
    """ Generate a sentence given a sample from the latent space.
    Note:
        - The architecture is adapted from the MMVAE paper.
        - {negative_slope=0, additional_noise_sigma=0., use_batchnorm=True} gives the MMVAE architecture.
        - This architecture is hard-coded for max_sentence_length = 32.
    """
    def __init__(self,
                 vocab_size=1590, embedding_dim=128,
                 normalize_embedding=False,
                 base_cc=32,
                 negative_slope=0.2,
                 use_batchnorm=True, batchnorm_kwargs=None,
                 conv2d=False,
                 freeze_conv=False):
        super(CUBSentDecoderMap, self).__init__()
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        assert embedding_dim == 128
        self.max_sentence_length = 32
        self.embedding_dim = 128
        self.vocab_size = vocab_size
        self.conv2d = conv2d
        self.freeze_conv = freeze_conv
        self.normalize_embedding = normalize_embedding
        self.output_embedding = False  # by default, decoder outputs (B, max_sentence_length, vocab_size)

        self.base_cc = base_cc
        self.input_dim = (16 * base_cc) * 4 * 4 if conv2d else (16 * base_cc) * 4

        if conv2d:
            modules = [# size: (base_cc * 16) x 4 x 4
                       nn.ConvTranspose2d(base_cc * 16, base_cc * 8, (1, 4), (1, 2), (0, 1), bias=True),
                       nn.BatchNorm2d(base_cc * 8, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 8) x 4 x 8
                       nn.ConvTranspose2d(base_cc * 8, base_cc * 4, (1, 4), (1, 2), (0, 1), bias=True),
                       nn.BatchNorm2d(base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 4) x 4 x 16
                       nn.ConvTranspose2d(base_cc * 4, base_cc * 2, 4, 2, 1, bias=True),
                       nn.BatchNorm2d(base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 2) x 8 x 32
                       nn.ConvTranspose2d(base_cc * 2, base_cc, 4, 2, 1, bias=True),
                       nn.BatchNorm2d(base_cc, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc) x 16 x 64
                       nn.ConvTranspose2d(base_cc, 1, 4, 2, 1, bias=True)]
            # output size: 1 x 32(=max_sentence_length) x 128(=embedding_dim)
        else:
            # input_size: 1 x dim_z x 1
            modules = [# size: (base_cc * 16) x 4
                       nn.ConvTranspose1d(base_cc * 16, base_cc * 8, 1, 1, 0, bias=True),
                       nn.BatchNorm1d(base_cc * 8, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 8) x 4
                       nn.ConvTranspose1d(base_cc * 8, base_cc * 4, 1, 1, 0, bias=True),
                       nn.BatchNorm1d(base_cc * 4, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 4) x 8
                       nn.ConvTranspose1d(base_cc * 4, base_cc * 2, 4, 2, 1, bias=True),
                       nn.BatchNorm1d(base_cc * 2, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc * 2) x 16
                       nn.ConvTranspose1d(base_cc * 2, base_cc, 4, 2, 1, bias=True),
                       nn.BatchNorm1d(base_cc, **batchnorm_kwargs) if use_batchnorm else None,
                       nn.LeakyReLU(negative_slope=negative_slope),
                       # size: (base_cc) x 32
                       nn.ConvTranspose1d(base_cc, embedding_dim, 4, 2, 1, bias=True)]
            # output size: 128(=embedding_dim) x 32(=max_sentence_length)
        self.network = nn.Sequential(*list(filter(None, modules)))

        # invert the 'embedding' module upto one-hotness
        self.embedding_inverse = nn.Linear(self.embedding_dim, self.vocab_size)  # the output encodes logits

        self.freeze()

    def freeze(self):
        if self.freeze_conv:
            self.eval()
            self.requires_grad_(False)
        return self

    def forward(self, z):
        self.freeze()
        if self.conv2d:
            z = z.reshape([-1, self.base_cc * 16, 4, 4])
            embeds = self.network(z).squeeze(1)  # (B, max_sentence_length, embedding_dim)
        else:
            z = z.reshape([-1, self.base_cc * 16, 4])
            embeds = self.network(z).permute(0, 2, 1)  # (B, max_sentence_length, embedding_dim)

        if self.output_embedding:
            return embeds  # (B, max_sentence_length, embedding_dim)
        else:
            return self.embedding_inverse(embeds)  # (B, max_sentence_length, vocab_size)
