import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision import transforms
from torchvision.datasets import SVHN, CelebA, ImageFolder
from torchvision.datasets.mnist import MNIST

from datasets.cub import CUBImgFt, CUBSent
from datasets.mnistcdcb import MNISTCdCb
from datasets.sketchy import SketchyVGGDataLoader
from datasets.transforms import StaticallyBinarize, StretchZeroOne, TensorDatasetWithTransform, FourDigitsToTwoImages


def get_dataloaders_mnist(batch_size, shuffle=False, device="cuda", binarize=False, root_path='..',
                          valid_split=0.1, size=28, drop_last=True, stretch=True, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

    transform = [transforms.ToTensor()]
    if size != 28:
        transform = [transforms.Resize(size)] + transform
    if binarize:
        transform.append(StaticallyBinarize())
    if stretch:
        transform.append(StretchZeroOne())
    transform = transforms.Compose(transform)

    # split train into (train, valid)
    train_dataset = MNIST('{}/data'.format(root_path), train=True, download=True, transform=transform)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    train_indices, valid_indices = indices[split:], indices[:split]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                              drop_last=drop_last, **kwargs)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices),
                              drop_last=drop_last, **kwargs)

    test_dataset = MNIST('{}/data'.format(root_path), train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, **kwargs)

    return train_loader, valid_loader, test_loader


def get_dataloaders_mnist_by_label(batch_size, device="cuda", binarize=False, root_path='..',
                                   split='train', size=28, stretch=True):
    assert split in ['train', 'test']
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
    transform = [transforms.ToTensor()]
    if size != 28:
        transform = [transforms.Resize(size)] + transform
    if binarize:
        transform.append(StaticallyBinarize())
    if stretch:
        transform.append(StretchZeroOne())
    transform = transforms.Compose(transform)

    train = split == 'train'
    dataset = MNIST('{}/data'.format(root_path), train=train, download=True, transform=transform)
    indices = np.arange(len(dataset))
    indices_by_label = {digit: indices[dataset.targets == digit] for digit in range(10)}
    dataloaders = {}
    for i in range(10):
        dataloaders[i] = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(indices_by_label[i]),
                                    drop_last=False, **kwargs)

    return dataloaders


def get_dataloaders_svhn(batch_size, shuffle=True, device="cuda", root_path='..', valid_split=0.1, drop_last=True,
                         train_split='extra', stretch=True, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    transform = [transforms.ToTensor()]
    if stretch:
        transform.append(StretchZeroOne())
    transform = transforms.Compose(transform)

    # split train into (train, valid)
    train_dataset = SVHN('{}/data/SVHN'.format(root_path), split=train_split, download=True, transform=transform)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    train_indices, valid_indices = indices[split:], indices[:split]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                              drop_last=drop_last, **kwargs)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices),
                              drop_last=drop_last, **kwargs)

    test_dataset = SVHN('{}/data/SVHN'.format(root_path), split='test', download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, **kwargs)

    return train_loader, valid_loader, test_loader


def get_dataloaders_svhn_by_label(batch_size, device="cuda", root_path='..', split='extra',
                                  stretch=True, **kwargs):
    assert ['train', 'test', 'extra']
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    transform = [transforms.ToTensor()]
    if stretch:
        transform.append(StretchZeroOne())
    transform = transforms.Compose(transform)

    # split train into (train, valid)
    dataset = SVHN('{}/data/SVHN'.format(root_path), split=split, download=True, transform=transform)
    indices = np.arange(len(dataset))
    indices_by_label = {digit: indices[dataset.labels == digit] for digit in range(10)}
    dataloaders = {}
    for i in range(10):
        dataloaders[i] = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(indices_by_label[i]),
                                    drop_last=False, **kwargs)

    return dataloaders


def get_dataloaders_celeba(batch_size, shuffle=True, device="cuda", root_path='..', valid_split=0.1, stretch=True,
                           crop_size=128, img_size=128, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    transform = [transforms.CenterCrop(crop_size),
          transforms.Resize((img_size, img_size)),
          transforms.ToTensor()]
    if stretch:
        transform.append(StretchZeroOne())
    transform = transforms.Compose(transform)

    # split train into (train, valid)
    train_dataset = CelebA('{}/data'.format(root_path), split='train', download=True, transform=transform)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    train_indices, valid_indices = indices[split:], indices[:split]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                              shuffle=shuffle, **kwargs)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices),
                              shuffle=shuffle, **kwargs)

    test_dataset = CelebA('{}/data'.format(root_path), split='test', download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# CUB Image / Sentence dataloaders
# remember that when combining with captions, this should be x10
def get_dataloaders_cub_img(batch_size, shuffle=True, device="cuda", root_path='..', imgsize=128, stretch=True, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    transform = [transforms.Resize([imgsize, imgsize]), transforms.ToTensor()]
    if stretch:
        transform.append(StretchZeroOne())
    transform = transforms.Compose(transform)

    train_dataset = ImageFolder(os.path.join(root_path, 'data', 'cub', 'img', 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    test_dataset = ImageFolder(os.path.join(root_path, 'data', 'cub', 'img', 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_loader


# remember that when combining with captions, this should be x10
def get_dataloaders_cub_imgft(batch_size, shuffle=True, device="cuda", root_path='..',
                              preprocess=False, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    train_dataset = CUBImgFt(root_path, 'train', device)
    test_dataset = CUBImgFt(root_path, 'test', device)

    if preprocess:
        train_dataset.features = (train_dataset.features ** 0.25) - (train_dataset.features ** 0.25).mean()
        test_dataset.features = (test_dataset.features ** 0.25) - (train_dataset.features ** 0.25).mean()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_loader


def get_dataloaders_cub_sent(batch_size, shuffle=True, device="cuda", root_path='..',
                             max_sentence_length=32, min_occur=3, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    transform = lambda x: torch.tensor(x)
    train_dataset = CUBSent(root_path, split='train', transform=transform,
                            max_sequence_length=max_sentence_length, min_occur=min_occur)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    test_dataset = CUBSent(root_path, split='test', transform=transform,
                           max_sequence_length=max_sentence_length, min_occur=min_occur)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_loader


def get_dataloaders_cub(batch_size, shuffle=True, device='cuda', root_path='..',
                        imgft=True, imgsize=128, cub_imgft_scaling=False,
                        max_sentence_length=32, min_occur=3, **kwargs):
    kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else kwargs

    # load base datasets
    if imgft:
        train_loader1, _, test_loader1 = get_dataloaders_cub_imgft(batch_size, shuffle, device, root_path,
                                                                   preprocess=cub_imgft_scaling)
    else:
        train_loader1, _, test_loader1 = get_dataloaders_cub_img(batch_size, shuffle, device, root_path, imgsize)
    train_loader2, _, test_loader2 = get_dataloaders_cub_sent(batch_size, shuffle, device, root_path,
                                                              max_sentence_length, min_occur)

    train_imgft_dataset = train_loader1.dataset
    test_imgft_dataset = test_loader1.dataset
    train_sent_dataset = train_loader2.dataset
    test_sent_dataset = test_loader2.dataset

    def resampler(dataset, idx):
        # This is required because there are 10 captions per image.
        # Allows easier reuse of the same image for the corresponding set of captions.
        return idx // 10

    train_loader = DataLoader(
        TensorDataset([ResampleDataset(train_imgft_dataset, resampler, size=len(train_imgft_dataset) * 10),
                       train_sent_dataset]),
        batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(
        TensorDataset([ResampleDataset(test_imgft_dataset, resampler, size=len(test_imgft_dataset) * 10),
                       test_sent_dataset]),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, test_loader


def get_dataloaders_mnistadd1(batch_size, shuffle=True, device='cuda', binarize=False, root_path='..',
                              dc=10000, dm=1, size=28, **kwargs):
    kwargs = {'num_workers': 2, 'pin_memory': True, **kwargs} if device == 'cuda' else kwargs

    path_mnistadd1 = os.path.join(root_path, 'data', 'mnist-add1')
    path_train_idx0 = os.path.join(path_mnistadd1, 'train-dc{}-m{}-idx0.pt'.format(dc, dm))
    path_train_idx1 = os.path.join(path_mnistadd1, 'train-dc{}-m{}-idx1.pt'.format(dc, dm))
    path_valid_idx0 = os.path.join(path_mnistadd1, 'valid-dc{}-m{}-idx0.pt'.format(dc, dm))
    path_valid_idx1 = os.path.join(path_mnistadd1, 'valid-dc{}-m{}-idx1.pt'.format(dc, dm))
    path_test_idx0 = os.path.join(path_mnistadd1, 'test-dc{}-m{}-idx0.pt'.format(dc, dm))
    path_test_idx1 = os.path.join(path_mnistadd1, 'test-dc{}-m{}-idx1.pt'.format(dc, dm))

    if not all([os.path.isfile(f) for f in [path_train_idx0, path_train_idx1,
                                            path_valid_idx0, path_valid_idx1,
                                            path_test_idx0, path_test_idx1]]):
        raise RuntimeError('Generate transformed indices with the script in datasets.pair_images')

    # get transformed indices
    train_mnist0 = torch.load(path_train_idx0)
    valid_mnist0 = torch.load(path_valid_idx0)
    test_mnist0 = torch.load(path_test_idx0)
    train_mnist1 = torch.load(path_train_idx1)
    valid_mnist1 = torch.load(path_valid_idx1)
    test_mnist1 = torch.load(path_test_idx1)

    # load base dataloaders
    train0, valid0, test0 = get_dataloaders_mnist(batch_size, shuffle, device, binarize, root_path, size=size, **kwargs)
    train1, valid1, test1 = get_dataloaders_mnist(batch_size, shuffle, device, binarize, root_path, size=size, **kwargs)

    train_dataset = TensorDataset([
        ResampleDataset(train0.dataset, lambda d, i: train_mnist0[i], size=len(train_mnist0)),
        ResampleDataset(train1.dataset, lambda d, i: train_mnist1[i], size=len(train_mnist1))
    ])
    valid_dataset = TensorDataset([
        ResampleDataset(valid0.dataset, lambda d, i: valid_mnist0[i], size=len(valid_mnist0)),
        ResampleDataset(valid1.dataset, lambda d, i: valid_mnist1[i], size=len(valid_mnist1))
    ])
    test_dataset = TensorDataset([
        ResampleDataset(test0.dataset, lambda d, i: test_mnist0[i], size=len(test_mnist0)),
        ResampleDataset(test1.dataset, lambda d, i: test_mnist1[i], size=len(test_mnist1))
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader


def get_dataloaders_mnistmultiply(batch_size, shuffle=True, device='cuda', binarize=False, root_path='..',
                                  dc=1000, dc_valid=100, dc_test=100, dm=1, size=28, **kwargs):
    kwargs = {'num_workers': 2, 'pin_memory': True, **kwargs} if device == 'cuda' else kwargs

    dc_train = dc
    datapath = '{}/data/mnist-multiply'.format(root_path)
    if not (os.path.exists('{}/train-dc{}-m{}-idx00.pt'.format(datapath, dc_train, dm))
            and os.path.exists('{}/train-dc{}-m{}-idx01.pt'.format(datapath, dc_train, dm))
            and os.path.exists('{}/test-dc{}-m{}-idx10.pt'.format(datapath, dc_test, dm))
            and os.path.exists('{}/test-dc{}-m{}-idx11.pt'.format(datapath, dc_test, dm))
    ):
        raise RuntimeError('Generate transformed indices with the script in datasets.pair_images')

    # get transformed indices
    train_mnist00 = torch.load('{}/train-dc{}-m{}-idx00.pt'.format(datapath, dc_train, dm))
    train_mnist01 = torch.load('{}/train-dc{}-m{}-idx01.pt'.format(datapath, dc_train, dm))
    train_mnist10 = torch.load('{}/train-dc{}-m{}-idx10.pt'.format(datapath, dc_train, dm))
    train_mnist11 = torch.load('{}/train-dc{}-m{}-idx11.pt'.format(datapath, dc_train, dm))
    valid_mnist00 = torch.load('{}/valid-dc{}-m{}-idx00.pt'.format(datapath, dc_valid, dm))
    valid_mnist01 = torch.load('{}/valid-dc{}-m{}-idx10.pt'.format(datapath, dc_valid, dm))
    valid_mnist10 = torch.load('{}/valid-dc{}-m{}-idx01.pt'.format(datapath, dc_valid, dm))
    valid_mnist11 = torch.load('{}/valid-dc{}-m{}-idx11.pt'.format(datapath, dc_valid, dm))
    test_mnist00 = torch.load('{}/test-dc{}-m{}-idx00.pt'.format(datapath, dc_test, dm))
    test_mnist01 = torch.load('{}/test-dc{}-m{}-idx01.pt'.format(datapath, dc_test, dm))
    test_mnist10 = torch.load('{}/test-dc{}-m{}-idx10.pt'.format(datapath, dc_test, dm))
    test_mnist11 = torch.load('{}/test-dc{}-m{}-idx11.pt'.format(datapath, dc_test, dm))

    # load base dataloaders
    train0, valid0, test0 = get_dataloaders_mnist(batch_size, shuffle, device, binarize, root_path, size=size, **kwargs)
    train1, valid1, test1 = get_dataloaders_mnist(batch_size, shuffle, device, binarize, root_path, size=size, **kwargs)

    train_dataset = TensorDatasetWithTransform([
        ResampleDataset(train0.dataset, lambda d, i: train_mnist00[i], size=len(train_mnist00)),
        ResampleDataset(train0.dataset, lambda d, i: train_mnist01[i], size=len(train_mnist01)),
        ResampleDataset(train1.dataset, lambda d, i: train_mnist10[i], size=len(train_mnist10)),
        ResampleDataset(train1.dataset, lambda d, i: train_mnist11[i], size=len(train_mnist11)),
    ], transform=FourDigitsToTwoImages())
    valid_dataset = TensorDatasetWithTransform([
        ResampleDataset(valid0.dataset, lambda d, i: valid_mnist00[i], size=len(valid_mnist00)),
        ResampleDataset(valid0.dataset, lambda d, i: valid_mnist01[i], size=len(valid_mnist01)),
        ResampleDataset(valid1.dataset, lambda d, i: valid_mnist10[i], size=len(valid_mnist10)),
        ResampleDataset(valid1.dataset, lambda d, i: valid_mnist11[i], size=len(valid_mnist11)),
    ], transform=FourDigitsToTwoImages())
    test_dataset = TensorDatasetWithTransform([
        ResampleDataset(test0.dataset, lambda d, i: test_mnist00[i], size=len(test_mnist00)),
        ResampleDataset(test0.dataset, lambda d, i: test_mnist01[i], size=len(test_mnist01)),
        ResampleDataset(test1.dataset, lambda d, i: test_mnist10[i], size=len(test_mnist10)),
        ResampleDataset(test1.dataset, lambda d, i: test_mnist11[i], size=len(test_mnist11)),
    ], transform=FourDigitsToTwoImages())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader, valid_loader, test_loader


def get_dataloaders_mnistsvhn(batch_size, shuffle=False, device='cuda', binarize=False, root_path='..',
                              dc=10000, dm=20, size=32, train_split='extra', stretch=True, **kwargs):
    kwargs = {'num_workers': 2, 'pin_memory': True, **kwargs} if device == 'cuda' else kwargs

    if not (os.path.exists('{}/data/mnist-svhn/train-dc{}-m{}-idx0.pt'.format(root_path, dc, dm))
            and os.path.exists('{}/data/mnist-svhn/train-dc{}-m{}-idx1.pt'.format(root_path, dc, dm))
            and os.path.exists('{}/data/mnist-svhn/valid-dc{}-m{}-idx0.pt'.format(root_path, dc, dm))
            and os.path.exists('{}/data/mnist-svhn/valid-dc{}-m{}-idx1.pt'.format(root_path, dc, dm))
            and os.path.exists('{}/data/mnist-svhn/test-dc{}-m{}-idx0.pt'.format(root_path, dc, dm))
            and os.path.exists('{}/data/mnist-svhn/test-dc{}-m{}-idx1.pt'.format(root_path, dc, dm))
    ):
        raise RuntimeError('Generate transformed indices with the script in datasets.pair_images')

    # get transformed indices
    train_mnist = torch.load('{}/data/mnist-svhn/train-dc{}-m{}-idx0.pt'.format(root_path, dc, dm))
    train_svhn = torch.load('{}/data/mnist-svhn/train-dc{}-m{}-idx1.pt'.format(root_path, dc, dm))
    valid_mnist = torch.load('{}/data/mnist-svhn/valid-dc{}-m{}-idx0.pt'.format(root_path, dc, dm))
    valid_svhn = torch.load('{}/data/mnist-svhn/valid-dc{}-m{}-idx1.pt'.format(root_path, dc, dm))
    test_mnist = torch.load('{}/data/mnist-svhn/test-dc{}-m{}-idx0.pt'.format(root_path, dc, dm))
    test_svhn = torch.load('{}/data/mnist-svhn/test-dc{}-m{}-idx1.pt'.format(root_path, dc, dm))

    # load base dataloaders
    train1, valid1, test1 = get_dataloaders_mnist(batch_size, shuffle, device, binarize, root_path,
                                                  size=size, stretch=stretch, **kwargs)
    train2, valid2, test2 = get_dataloaders_svhn(batch_size, shuffle, device, root_path,
                                                 train_split=train_split, stretch=stretch, **kwargs)

    train_dataset = TensorDataset([
        ResampleDataset(train1.dataset, lambda d, i: train_mnist[i], size=len(train_mnist)),
        ResampleDataset(train2.dataset, lambda d, i: train_svhn[i], size=len(train_svhn))
    ])
    valid_dataset = TensorDataset([
        ResampleDataset(valid1.dataset, lambda d, i: valid_mnist[i], size=len(valid_mnist)),
        ResampleDataset(valid2.dataset, lambda d, i: valid_svhn[i], size=len(valid_svhn))
    ])
    test_dataset = TensorDataset([
        ResampleDataset(test1.dataset, lambda d, i: test_mnist[i], size=len(test_mnist)),
        ResampleDataset(test2.dataset, lambda d, i: test_svhn[i], size=len(test_svhn))
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader, valid_loader, test_loader


def get_dataloaders_mnistcdcb(batch_size, shuffle=True, device="cuda", root_path='..', valid_split=0.1, drop_last=True,
                              size=32, **kwargs):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else kwargs

    # split train into (train, valid)
    train_dataset = MNISTCdCb('{}/data/MNISTCdCb'.format(root_path), split='train', height=size)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    train_indices, valid_indices = indices[split:], indices[:split]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
                              shuffle=shuffle, drop_last=drop_last, **kwargs)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices),
                              shuffle=shuffle, drop_last=drop_last, **kwargs)

    test_dataset = MNISTCdCb('{}/data/MNISTCdCb'.format(root_path), split='test', height=size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=drop_last, **kwargs)

    return train_loader, valid_loader, test_loader


def get_bimodal_dataloaders(config, generator, worker_init_fn, device):
    batch_size = config.batch_size
    train_loaders_marginal = [None, None]

    if config.dataset == 'mnist-mnist':
        train_loader, valid_loader, test_loader = get_dataloaders_mnistadd1(
            batch_size, shuffle=True, device=device, binarize=False, root_path=config.main_path, size=32,
            dc=config.dc, dm=config.dm, drop_last=True, generator=generator, worker_init_fn=worker_init_fn,
        )
        train_loaders_marginal[0], *_ = get_dataloaders_mnist(batch_size, shuffle=True, device=device,
                                                              binarize=False, root_path=config.main_path,
                                                              size=32, drop_last=True,
                                                              generator=generator, worker_init_fn=worker_init_fn,)
        train_loaders_marginal[1], *_ = get_dataloaders_mnist(batch_size, shuffle=True, device=device,
                                                              binarize=False, root_path=config.main_path,
                                                              size=32, drop_last=True,
                                                              generator=generator, worker_init_fn=worker_init_fn,)

    elif config.dataset == 'mnist-svhn':
        train_loader, valid_loader, test_loader = get_dataloaders_mnistsvhn(
            batch_size, shuffle=True, device=device, binarize=False, root_path=config.main_path, size=32,
            dc=config.dc, dm=config.dm, drop_last=True,
            generator=generator, worker_init_fn=worker_init_fn,
        )

        train_loaders_marginal[0], *_ = get_dataloaders_mnist(batch_size, shuffle=False, device=device,
                                                              binarize=False, root_path=config.main_path,
                                                              size=32, drop_last=True,
                                                              generator=generator, worker_init_fn=worker_init_fn,)
        train_loaders_marginal[1], *_ = get_dataloaders_svhn(batch_size, shuffle=False, device=device,
                                                             root_path=config.main_path,
                                                             drop_last=True,
                                                             generator=generator, worker_init_fn=worker_init_fn,)

    elif config.dataset == 'mnist-cdcb':
        train_loader, valid_loader, test_loader = get_dataloaders_mnistcdcb(
            batch_size, shuffle=True, device=device, binarize=False, root_path=config.main_path, size=32,
            dc=config.dc, dm=config.dm, drop_last=True, valid_split=0.05,
            generator=generator, worker_init_fn=worker_init_fn,
        )
        assert not config.ssl, 'MNISTCdCb dataset does not support semi-supervised data'

    elif config.dataset == 'mnist-multiply':
        train_loader, valid_loader, test_loader = get_dataloaders_mnistmultiply(
            batch_size, shuffle=True, device=device, binarize=False, root_path=config.main_path, size=32,
            dc=config.dc, dm=config.dm, drop_last=True, dc_valid=100,
            generator=generator, worker_init_fn=worker_init_fn,
        )
        assert not config.ssl, 'MNIST-multiply dataset does not support semi-supervised data'

    elif config.dataset == 'sketchy-vgg':
        train_loader = SketchyVGGDataLoader(batch_size, shuffle=True, drop_last=False,
                                            root_path=config.main_path, split=config.sketchy_split,
                                            train_or_test='train')
        test_loader = SketchyVGGDataLoader(batch_size, shuffle=False, drop_last=False,
                                           root_path=config.main_path, split=config.sketchy_split,
                                           train_or_test='test')
        valid_loader = test_loader
        assert not config.ssl, 'SketchyVGG dataset does not support semi-supervised data'

    elif 'cub' in config.dataset:
        imgft = True if 'ft' in config.dataset else False
        max_sentence_length = 32  # max length of any description for birds dataset
        min_occur = 3
        train_loader, valid_loader, test_loader = get_dataloaders_cub(
            batch_size, True, device,
            config.main_path,
            imgft=imgft, imgsize=config.cub_imgsize,
            cub_imgft_scaling=config.cub_imgft_scaling,
            max_sentence_length=max_sentence_length, min_occur=min_occur,
            generator=generator, worker_init_fn=worker_init_fn,
        )

    else:
        raise Exception('check config.dataset')

    return train_loader, valid_loader, test_loader, train_loaders_marginal
