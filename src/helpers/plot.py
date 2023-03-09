import math
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

mpl.style.use('ggplot')


irange = range
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def normalize_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    return ax


def channel2width(samples):
    img_size = samples.shape[-3:]
    width_scale = 1
    if img_size[0] == 2:
        # phantom_channel = torch.zeros(*batch_size, 1, *img_size[1:]).to(samples.device)
        # samples = torch.cat([samples, phantom_channel], dim=-3)
        samples = torch.cat([samples[..., :1, :, :], samples[..., 1:, :, :]], dim=-1)
        width_scale = 2
    return samples, width_scale


def plot_grid_samples(samples, path, m=8):
    samples, width_scale = channel2width(samples)

    fig = plt.figure(figsize=(width_scale * m, m))
    gs = gridspec.GridSpec(m, m)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        normalize_ax(ax)
        sample[sample < 0] = 0
        sample[sample > 1] = 1
        plt.imshow(sample.permute([1, 2, 0]).squeeze().detach().cpu().numpy())

    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_grid_two_image_samples(samples_x, samples_y, filepath, m=8, dpi=50):
    # scale pixel values
    samples_x = (samples_x + 1.) / 2
    samples_y = (samples_y + 1.) / 2

    samples_x, width_scale = channel2width(samples_x)
    samples_y, width_scale = channel2width(samples_y)

    samples_x[samples_x < 0] = 0
    samples_x[samples_x > 1] = 1
    samples_y[samples_y < 0] = 0
    samples_y[samples_y > 1] = 1
    samples = [samples_x.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy(),
               samples_y.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()]

    fig = plt.figure(figsize=(width_scale * 2 * m, m))
    outer = gridspec.GridSpec(1, 2, wspace=0.05, hspace=0.05)

    for i in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(m, m, subplot_spec=outer[i], wspace=0.05, hspace=0.05)
        for j, sample in enumerate(samples[i][:m ** 2]):
            ax = plt.Subplot(fig, inner[j])
            normalize_ax(ax)
            fig.add_subplot(ax)
            plt.imshow(sample)

    plt.savefig(filepath, bbox_inches='tight', pad_inches=0., dpi=dpi)
    plt.close(fig)


def permute_images(samples):
    """
    Parameters
    ----------
    samples: torch.tensor
        shape (nz, nuv, channel, width, height)

    Returns
    -------
    samples: np.array
        shape (nz, nuv, width, height, channel)
    width_scale: 1 or 2
    """
    if len(samples.shape) == 4:
        samples = samples.unsqueeze(dim=1)

    samples, width_scale = channel2width(samples)

    # samples: shape (nz, nuv, *img_shape)
    samples[samples < 0] = 0
    samples[samples > 1] = 1

    samples = samples.permute([0, 1, 3, 4, 2])
    samples = torch.squeeze(samples, dim=-1).detach().cpu().numpy()  # squeeze the last channel, if possible

    return samples, width_scale


def save_images(samples, path, prefix='', output_filetype='jpg', mode=None):
    if not os.path.isdir(path):
        os.makedirs(path)
    # assume pixel values are scaled within [-1, 1]
    samples, _ = permute_images((samples + 1.) / 2)
    samples = samples.reshape((samples.shape[0] * samples.shape[1], *samples.shape[2:]))

    for i in range(len(samples)):
        im = Image.fromarray(samples[i] * 255, mode)
        im.convert("L" if mode is None else mode).save('{}/img{}_{:05d}.{}'.format(path, prefix, i, output_filetype))


def save_summary_joint_image_samples(samples_x, samples_y, filepath, dpi=50):
    samples_x, width_scale = permute_images((samples_x + 1.) / 2)
    samples_y, width_scale = permute_images((samples_y + 1.) / 2)
    nz, nuv = samples_x.shape[:2]

    fig = plt.figure(figsize=(width_scale * nuv, 2 * nz))
    outer = gridspec.GridSpec(nz, 1, wspace=.05, hspace=.05)

    for i in range(nz):
        inner = gridspec.GridSpecFromSubplotSpec(2, nuv, subplot_spec=outer[i], wspace=.05, hspace=.0)
        for j in range(nuv):
            ax = plt.Subplot(fig, inner[j])
            normalize_ax(ax)
            fig.add_subplot(ax)
            plt.imshow(samples_x[i][j])

            ax = plt.Subplot(fig, inner[j + nuv])
            normalize_ax(ax)
            fig.add_subplot(ax)
            plt.imshow(samples_y[i][j])

    plt.savefig(filepath, bbox_inches='tight', pad_inches=0., dpi=dpi)
    plt.close(fig)


def save_summary_conditional_image_samples(conditions, samples, filepath, dpi=50):
    # conditions: shape (nz, *img_shape)
    # scale pixel values
    conditions = (conditions + 1.) / 2
    samples = (samples + 1.) / 2

    # reshape
    conditions, width_scale = channel2width(conditions)
    conditions = conditions.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()

    # samples: shape (nz, nuv, *img_shape)
    samples, width_scale = permute_images(samples)
    nz, nuv = samples.shape[:2]

    fig = plt.figure(figsize=(width_scale * (nuv + 1), nz))

    outer = gridspec.GridSpec(nz, nuv + 1, wspace=.05, hspace=.05)
    gs = np.array(list(outer)).reshape((nz, nuv + 1))

    for i in range(nz):
        ax = plt.Subplot(fig, gs[i][0])
        normalize_ax(ax)
        fig.add_subplot(ax)
        plt.imshow(conditions[i])

        for j in range(nuv):
            ax = plt.Subplot(fig, gs[i][j + 1])
            normalize_ax(ax)
            fig.add_subplot(ax)
            plt.imshow(samples[i][j])

    plt.savefig(filepath, bbox_inches='tight', pad_inches=0., dpi=dpi)
    plt.close(fig)


def save_summary_stylish_image_samples(conditions, references, samples, filepath, dpi=50):
    # conditions: shape (nz, *img_shape)
    # references: shape (nuv, *img_shape)

    # scale pixel values
    conditions = (conditions + 1.) / 2
    references = (references + 1.) / 2
    samples = (samples + 1.) / 2

    conditions, width_scale = channel2width(conditions)
    references, width_scale = channel2width(references)
    conditions = conditions.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()
    references = references.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()

    # samples: shape (nz, nuv, *img_shape)
    samples, width_scale = permute_images(samples)
    nz, nuv = samples.shape[:2]

    fig = plt.figure(figsize=(width_scale * (nuv + 1), nz + 1))

    outer = gridspec.GridSpec(nz + 1, nuv + 1, wspace=.05, hspace=.05)
    gs = np.array(list(outer)).reshape((nz + 1, nuv + 1))

    for j in range(nuv):
        ax = plt.Subplot(fig, gs[0][j + 1])
        normalize_ax(ax)
        fig.add_subplot(ax)
        plt.imshow(references[j])

    for i in range(nz):
        ax = plt.Subplot(fig, gs[i + 1][0])
        normalize_ax(ax)
        fig.add_subplot(ax)
        plt.imshow(conditions[i])

        for j in range(nuv):
            ax = plt.Subplot(fig, gs[i + 1][j + 1])
            normalize_ax(ax)
            fig.add_subplot(ax)
            plt.imshow(samples[i][j])

    plt.savefig(filepath, bbox_inches='tight', pad_inches=0., dpi=dpi)
    plt.close(fig)
