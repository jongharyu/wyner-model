import torch
import torch.nn as nn
import torch.utils.data


class CUBImgFtFeatureMap(nn.Module):
    def __init__(self, feature_dim=256, input_dim=2048):
        super(CUBImgFtFeatureMap, self).__init__()
        self.hidden_dim = feature_dim
        self.base_cc = input_dim
        modules = []
        for i in range(int(torch.tensor(input_dim / feature_dim).log2())):
            modules.extend([nn.Linear(input_dim // (2 ** i), input_dim // (2 ** (i + 1))),
                            nn.ELU(inplace=True)])
        self.network = nn.Sequential(*modules)
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.network(x)


class CUBImgFtDecoderMap(nn.Module):
    """ Generate a CUB image feature given a sample from the latent space. """

    def __init__(self, dim_z, hidden_dim=256, base_cc=2048):
        super(CUBImgFtDecoderMap, self).__init__()
        self.hidden_dim = hidden_dim
        self.base_cc = base_cc
        self.network = nn.Sequential()
        modules = []
        for i in range(int(torch.tensor(base_cc / hidden_dim).log2())):
            dim_in = dim_z if i == 0 else hidden_dim * i
            dim_out = hidden_dim if i == 0 else hidden_dim * (2 * i)
            modules.extend([nn.Linear(dim_in, dim_out),
                            nn.ELU(inplace=True)])
        modules.append(nn.Linear(base_cc // 2, base_cc))
        self.network = nn.Sequential(*modules)  # relies on above terminating at base_cc // 2
        self.feature_dim = base_cc // 2

    def forward(self, z):
        return self.network(z)
