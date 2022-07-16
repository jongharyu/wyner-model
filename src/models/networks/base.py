import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:  # to exclude pretrained layers with requires_grad=False
            nn.init.xavier_uniform_(m.weight)  # an alternative: nn.init.normal_(m.weight, 0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


def get_fc(input_dim, output_dim, hidden_dim=0, n_hidden_layers=2, hidden_dims=None,
           lrelu_negative_slope=0.,
           elu_alpha=1.,
           use_batchnorm=False,
           use_batchnorm_first_layer=None,
           batchnorm_kwargs=None,
           use_spectralnorm=False,
           activation_type='lrelu',
           last_activation_type=None,
           batchnorm2d=False,
           batchnorm2d_input_shape=None):

    assert activation_type in ['lrelu', 'elu']
    assert last_activation_type in [None, 'lrelu', 'elu', 'relu', 'tanh'], last_activation_type
    # Warning: if last_activation_type in ['lrelu', 'elu'], then it is considered as a "feature" and thus we apply bn;
    #          otherwise, i.e., if last_activation_type in ['relu', 'tanh', None], we do not put batchnorm

    # Note: these are default BN parameters from the tensorflow implementation of Symmetric VAE
    if use_batchnorm_first_layer is None:
        use_batchnorm_first_layer = use_batchnorm
    if batchnorm_kwargs is None:
        batchnorm_kwargs = {}
    if hidden_dims is None:
        hidden_dims = [hidden_dim] * n_hidden_layers
    else:
        n_hidden_layers = len(hidden_dims)

    def get_linear_layer(dim_in, dim_out, spectralnorm=use_spectralnorm):
        linear_layer = nn.Linear(dim_in, dim_out)
        if spectralnorm:
            linear_layer = nn.utils.spectral_norm(linear_layer)
        return linear_layer

    def get_activation(act_type):
        if act_type == 'lrelu':
            return nn.LeakyReLU(negative_slope=lrelu_negative_slope)
        elif act_type == 'elu':
            return nn.ELU(alpha=elu_alpha)
        elif act_type == 'tanh':
            return nn.Tanh()
        elif act_type == 'relu':
            return nn.ReLU()
        elif act_type is None:
            return None

    # Note: if hidden_dims is not None, it overloads (hidden_dim, n_hidden_layers)
    modules = []
    if n_hidden_layers == 0:
        modules.append(get_linear_layer(input_dim, output_dim, spectralnorm=last_activation_type is not None))
    else:
        modules.extend([get_linear_layer(input_dim, hidden_dims[0]),
                        nn.BatchNorm1d(num_features=hidden_dims[0], **batchnorm_kwargs) if use_batchnorm_first_layer else None,
                        get_activation(activation_type)])
        for i in range(n_hidden_layers - 1):
            modules.extend([get_linear_layer(hidden_dims[i], hidden_dims[i + 1]),
                            nn.BatchNorm1d(num_features=hidden_dims[i + 1], **batchnorm_kwargs) if use_batchnorm else None,
                            get_activation(activation_type)])
        modules.append(get_linear_layer(hidden_dims[-1], output_dim, spectralnorm=last_activation_type is not None))

    if last_activation_type not in [None, 'tanh', 'relu'] and \
            ((use_batchnorm and n_hidden_layers > 0) or
             (use_batchnorm_first_layer and n_hidden_layers == 0)):
        if batchnorm2d:
            network_ = nn.Sequential(*list(filter(None, modules)))

            class NetworkWithBatchnorm2d(nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network
                    self.batchnorm = nn.BatchNorm2d(num_features=batchnorm2d_input_shape[0], **batchnorm_kwargs)
                    self.activation = get_activation(last_activation_type)

                def forward(self, x):
                    h = self.network(x).reshape(-1, *batchnorm2d_input_shape)
                    return self.activation(self.batchnorm(h))
            network = NetworkWithBatchnorm2d(network_)
        else:
            modules.extend([nn.BatchNorm1d(num_features=output_dim, **batchnorm_kwargs),
                            get_activation(last_activation_type)])
            network = nn.Sequential(*list(filter(None, modules)))
    else:
        modules.append(get_activation(last_activation_type))
        network = nn.Sequential(*list(filter(None, modules)))

    network.input_dim = input_dim
    network.output_dim = output_dim

    return network


class ResBlock(nn.Module):
    def __init__(self, kernel_size, base_cc, negative_slope=0.,
                 batchnorm_kwargs=None):
        super(ResBlock, self).__init__()
        # default BN parameters from the tensorflow implementation of Symmetric VAE
        if batchnorm_kwargs is None:
            batchnorm_kwargs = {}
        self.block = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(kernel_size=kernel_size, stride=1, in_channels=base_cc, out_channels=base_cc, ),
            nn.BatchNorm2d(base_cc, **batchnorm_kwargs),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(kernel_size=kernel_size, stride=1, in_channels=base_cc, out_channels=base_cc, ),
            nn.BatchNorm2d(base_cc, **batchnorm_kwargs),
        )

    def forward(self, x):
        return x + self.block(x)
    

class Stack(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return torch.stack(args, dim=-1)


class RandomFunction(nn.Module):
    def __init__(self, noise_dim=0, add_noise_channel=True):
        super(RandomFunction, self).__init__()
        self.noise_dim = noise_dim
        self.add_noise_channel = add_noise_channel
        self.additional_gaussian_noise_switch = True

    def set_additional_gaussian_noise(self, switch):
        self.additional_gaussian_noise_switch = switch
        return self

    def append_noise_and_concat(self, *vars):
        if len(vars[0].shape) > 2:
            if len(vars[0].shape) == 4:
                # assume that all vars are images of the same shape
                noise = self.draw_uniform_noise(vars[0].shape[0], int(self.add_noise_channel), *vars[0].shape[-2:])
            else:
                # an exceptional case made for CUB sentence
                noise = self.draw_uniform_noise(vars[0].shape[0], int(self.add_noise_channel), *vars[0].shape[-1:])
            return torch.cat([*vars, noise], dim=1)  # (B, var_ccs + add_noise_channel, 32, 32)
        else:
            noise = self.draw_gaussian_noise(vars[0].shape[0], self.noise_dim)
            return torch.cat([*vars, noise], dim=1)  # (B, var_dims + noise_dim)

    def add_gaussian_noise(self, x, sigma):
        if self.additional_gaussian_noise_switch:
            return x + sigma * self.draw_gaussian_noise(*x.shape)
        else:
            return x

    def draw_uniform_noise(self, *size):
        return (2 * torch.rand(size) - 1).to(self.device)

    def draw_gaussian_noise(self, *size):
        return torch.randn(size).to(self.device)


class JointFeatureMap(nn.Module):
    def __init__(self, aggregate=None, to_feature_x=None, to_feature_y=None):
        super().__init__()
        self.to_feature_x = to_feature_x
        self.to_feature_y = to_feature_y
        self.aggregate = aggregate
        self.output_dim = aggregate.output_dim if aggregate is not None else 0

    def forward(self, x, y):
        hx = x if self.to_feature_x is None else self.to_feature_x(x)
        hy = y if self.to_feature_y is None else self.to_feature_y(y)
        if self.to_feature_x is None and self.to_feature_y is None:
            hxy = torch.cat([hx, hy], dim=1)
        else:
            hxy = torch.cat([hx.view(x.shape[0], -1), hy.view(y.shape[0], -1)], dim=1)
        return hxy if self.aggregate is None else self.aggregate(hxy)
