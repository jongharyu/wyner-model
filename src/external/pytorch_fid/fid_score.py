"""Calculates the Frechet Distance (FD) to evalulate GANs

The FD metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FD is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from .inception import InceptionV3
from models.autoencoders.main import build_image_ae
from models.classifiers.model import MNISTClassifier, SVHNClassifier
from datasets.utils import unpack_data


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int, default=8,
                    help='Number of processes to use for data loading')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(dataloader, model, dims, model_type='inception', device='cpu'):
    """Calculates the activations of the pool_3 layer for all images given dataloader

    Params:
    -- dataloader  : List of image files paths
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- model_type  : Either 'inception' (default) or 'mnist' or 'svhn'

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    assert model_type in ['inception', 'mnist', 'svhn', 'mnist_ae', 'svhn_ae', 'mnist_svhn_ae']
    model.eval()
    pred_list = []

    def process_mnist_batch(batch):
        batch = batch[:, 0:1]
        # mnist autoencoder assumes size 32
        if batch.shape[-1] == 28 and model_type == 'mnist_ae':
            batch = F.pad(batch, [2, 2, 2, 2], "constant", 0)
        # mnist classifier assumes size 28
        if batch.shape[-1] == 32 and model_type == 'mnist':
            batch = batch[..., 2:30, 2:30]

        return batch

    dataset = 'mnist-svhn' if model_type == 'mnist_svhn_ae' else None
    for batch in dataloader:
        batch = unpack_data(batch, device=device, dataset=dataset)
        if not model_type == 'mnist_svhn_ae':
            assert batch.min() >= 0 and batch.max() <= 1
        else:
            assert batch[0].min() >= 0 and batch[0].max() <= 1
            assert batch[1].min() >= 0 and batch[1].max() <= 1

        if model_type == 'inception':
            with torch.no_grad():
                if batch.shape[1] == 1:
                    batch = torch.cat([batch, batch, batch], dim=1)
                pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        elif model_type == 'mnist_svhn_ae':
            with torch.no_grad():
                mnist_batch, svhn_batch = batch
                mnist_batch = process_mnist_batch(mnist_batch)
                pred = model(2 * mnist_batch - 1, 2 * svhn_batch - 1)

        else:  # model_type in ['mnist', 'svhn'] + ['mnist_ae', 'svhn_ae']
            with torch.no_grad():
                if 'mnist' in model_type:
                    batch = process_mnist_batch(batch)
                # Caution: all classifiers and autoencoders are trained with [-1,1]-inputs
                batch = 2 * batch - 1
                if model_type.endswith('ae'):
                    pred = model.encoder(batch)
                else:
                    pred = model.extract_features(batch, dims)

        pred_list.append(pred)

    pred_arr = torch.cat(pred_list)
    return pred_arr


def compute_activation_statistics(dataloader, model,
                                  model_type='inception', dims=2048,
                                  device='cpu',
                                  verbose=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculation of the statistics used by the FD.
    Params:
    -- dataloader  : Dataloader
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : mean over all activations from the encoder.
    -- sigma : covariance matrix over all activations from the encoder.
    """
    act: torch.Tensor = get_activations(dataloader, model, dims, model_type, device)
    mu, sigma = _compute_2nd_order_stats(act)

    if verbose:
        dims = 1024 if model_type.endswith('ae') else dims
        if act.shape[0] < dims + 1:
            print("Warning: number of samples {} <= dimensions {} + 1, causing a singular covariance matrix estimate".format(
                act.shape[0], dims))

    return mu, sigma


def compute_statistics(path, dataloader, model, batch_size, model_type, dims, device,
                       num_workers=8) -> Tuple[torch.Tensor, torch.Tensor]:
    if path:
        if path.endswith('.npz'):
            if os.path.exists(path):
                with np.load(path) as f:
                    m, s = f['mu'][:], f['sigma'][:]
                m = torch.from_numpy(m).to(device)
                s = torch.from_numpy(s).to(device)
            else:
                print("Warning: the npz file {} does not exist, so we compute and save now...".format(path))
                path_to_folder = '/'.join(path.rstrip('/').split('/')[:-1])
                if dataloader:
                    m, s = compute_statistics(None, dataloader, model, batch_size, model_type, dims, device, num_workers)
                else:
                    m, s = compute_statistics(path_to_folder, model, batch_size, model_type, dims, device, num_workers)
                if not os.path.exists(path_to_folder):
                    os.makedirs(path_to_folder)
                np.savez(path, mu=m.cpu().numpy(), sigma=s.cpu().numpy())
        else:
            assert os.path.exists(path), "{} does not exist!".format(path)
            path = pathlib.Path(path)
            files = sorted([file for ext in IMAGE_EXTENSIONS
                           for file in path.glob('*.{}'.format(ext))])

            if len(files) == 0:
                raise RuntimeError("{} does not contain any files of extensions in {}".format(path, IMAGE_EXTENSIONS))

            if batch_size > len(files):
                print(('Warning: batch size is bigger than the data size. '
                       'Setting batch size to data size'))
                batch_size = len(files)

            dataset = ImagePathDataset(files, transforms=TF.ToTensor())
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)

            m, s = compute_activation_statistics(dataloader, model,
                                                 model_type, dims,
                                                 device)
    else:
        assert dataloader
        m, s = compute_activation_statistics(dataloader, model,
                                             model_type, dims,
                                             device)

    return m, s


def get_model_instance(model_type, dims, device):
    if model_type == 'inception':
        assert dims in [64, 192, 768, 2048]
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
    elif model_type in ['mnist', 'svhn']:
        if model_type == 'mnist':
            assert dims in [320, 50]
            model = MNISTClassifier().to(device)
        elif model_type == 'svhn':
            assert dims in [500, 50]
            model = SVHNClassifier().to(device)
        else:
            raise ValueError()
        dirname = os.path.dirname(__file__)
        pretrained_model_path = os.path.join(dirname, '../../../pretrained/classifiers/{}.rar'.format(model_type))
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    elif model_type in ['mnist_ae', 'svhn_ae']:
        cc = 1 if model_type == 'mnist_ae' else 3
        base_cc = 32
        negative_slope = 0.2
        use_batchnorm = True
        n_resblocks = 0

        model = build_image_ae(cc, base_cc, negative_slope, use_batchnorm, n_resblocks)
        dirname = os.path.dirname(__file__)
        pretrained_model_path = os.path.join(dirname,
                                             '../../../pretrained/autoencoders/{}-cc32nres0/model.rar'.format(
                                                 model_type.split('_')[0]))
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    elif model_type == 'mnist_svhn_ae':
        # class
        from models.networks.base import JointFeatureMap
        mnist_ae = get_model_instance('mnist_ae', dims, device)
        svhn_ae = get_model_instance('svhn_ae', dims, device)
        model = JointFeatureMap(to_feature_x=mnist_ae.encoder, to_feature_y=svhn_ae.encoder)
    else:
        raise ValueError

    return model.to(device)


def compute_frechet_distance(paths, dataloaders, batch_size, device, dims, num_workers=8, model_type='inception',
                             compute_auxiliary_scores=False, use_numpy=False):
    """Calculates the FD of two paths or dataloaders"""
    if paths is None:
        paths = [None, None]
    model = get_model_instance(model_type, dims, device)
    m1, s1 = compute_statistics(paths[0], dataloaders[0], model, batch_size,
                                model_type, dims, device, num_workers)
    m2, s2 = compute_statistics(paths[1], dataloaders[1], model, batch_size,
                                model_type, dims, device, num_workers)
    fd = compute_frechet_distance_from_stats(m1, s1, m2, s2, use_numpy=use_numpy)

    if not compute_auxiliary_scores:
        return fd
    else:
        ent1 = _compute_entropy(s1)
        ent2 = _compute_entropy(s2)
        kl_div_12 = _compute_kl_divergence(m1, s1, m2, s2)
        kl_div_21 = _compute_kl_divergence(m2, s2, m1, s1)
        return fd, ent1, ent2, kl_div_12, kl_div_21


def compute_frechet_distance_from_stats(*args, use_numpy):
    if use_numpy:
        return _compute_frechet_distance_from_stats_numpy(*args)
    else:
        return _compute_frechet_distance_from_stats_torch(*args)


def _compute_frechet_distance_from_stats_numpy(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fd calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def _compute_entropy(sigma):
    """Numpy implementation of the differential entropy for gaussian distribution.
    The differential entropy of a multivariate Gaussian X ~ N(m, C) is
            h(N(m, C)) = 1 / 2 * log det( 2 * pi * e * C)

    Params:
    -- sigma: The covariance matrix over activations of samples

    Returns:
    --   : The differential entropy
    """
    dim = sigma.shape[0]
    sigma = np.atleast_2d(sigma)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        print('Warning: cov estimate is not positive definite, and -inf is returned')
        return -np.inf
    else:
        return .5 * (logdet + dim * np.log(2 * np.pi * np.exp(1)))


def _compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Kullback--Leibler divergence for two gaussian distributions.
    The KL divergence between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is
            1 / 2 * (Tr(C_2^{-1} * C_1) + (mu_2 - mu_1)^T C_2^{-1} (mu_2 - mu_1) - dim + log ( det(C_2) / det(C_1))

    Params:
    -- mu1
    -- mu2
    -- sigma1
    -- sigma2

    Returns:
    --   : The KL divergence D( N(mu1, sigma1) || N(mu2, sigma2) )
    """
    dim = mu1.shape[0]

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    sign1, logdet1 = np.linalg.slogdet(sigma2)
    sign2, logdet2 = np.linalg.slogdet(sigma2)

    if sign1 <= 0 or sign2 <= 0:
        print("Warning: sigma1 or sigma2 are not positive definite, and -inf is returned")
        return -np.inf

    try:
        sigma2inv = linalg.inv(sigma2)
    except linalg.LinAlgError:
        print("Warning: sigma2 is singular, and -inf is returned")
        return -np.inf

    return .5 * (np.trace(sigma2inv @ sigma1) +
                 diff.T @ sigma2inv @ diff - dim +
                 logdet2 - logdet1)


# The following four functions are from
# Reference: https://github.com/photosynthesis-team/piq/blob/60abd2f22eafa2c8af57006525e8dff4dbbb43eb/piq/fd.py
def _compute_2nd_order_stats(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the statistics used by Frechet distance
    Args:
        samples:  Low-dimension representation of image set.
            Shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
    Returns:
        mu: mean over all samples
        sigma: covariance matrix over all samples
    """
    mu = torch.mean(samples, dim=0)
    sigma = _cov(samples, rowvar=False)
    return mu, sigma


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.], device=error.device, dtype=error.dtype), atol=1e-5):
            break

    return s_matrix, error


def _compute_frechet_distance_from_stats_torch(mu1: torch.Tensor, sigma1: torch.Tensor,
                                               mu2: torch.Tensor, sigma2: torch.Tensor,
                                               eps=1e-6) -> torch.Tensor:
    r"""
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean, _ = _sqrtm_newton_schulz(sigma1.double().mm(sigma2.double()))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f'FD calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def _cov(m: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    r"""Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('Tensor for covariance computations has more than 2 dimensions. '
                         'Only 1 or 2 dimensional arrays are allowed')

    if m.dim() < 2:
        m = m.view(1, -1)

    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    fd, ent1, ent2, kl_div_12, kl_div_21 = compute_frechet_distance(args.path,
                                                                     args.batch_size,
                                                                     device,
                                                                     args.dims,
                                                                     args.num_workers,
                                                                     compute_auxiliary_scores=True)
    print('FD={}, ent1={}, ent2={}, kl_div_12={}, kl_div_21={}'.format(fd, ent1, ent2, kl_div_12, kl_div_21))


if __name__ == '__main__':
    main()
