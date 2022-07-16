import torch
import torch.nn as nn

from models.networks.base import weights_init, get_fc


class Encoder(nn.Module):
    def __init__(self, to_feature, to_latent=None):
        super().__init__()
        self.to_feature = to_feature
        self.to_latent = lambda x: x if to_latent is None else to_latent

    def forward(self, x):
        h = self.to_feature(x).view(x.shape[0], -1)
        z = self.to_latent(h)
        return z


class Decoder(nn.Module):
    def __init__(self, to_data, to_feature=None):
        super().__init__()
        self.to_data = to_data
        self.to_feature = (lambda x: x) if to_feature is None else to_feature

    def forward(self, x):
        return self.to_data(self.to_feature(x))


class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.apply(weights_init)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            xhat = self.forward(x)

        return xhat
