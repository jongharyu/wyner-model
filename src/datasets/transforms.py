# custom transforms
import torch
from torchnet.dataset import TensorDataset


class CutHalf(object):
    """Cut (C, H, W) into two pieces of size (C, H, W//2)"""
    def __call__(self, sample):
        w = sample.shape[-1]
        assert w % 2 == 0
        sample = sample.squeeze(0)
        return sample[..., :w//2], sample[..., w//2:]


class StaticallyBinarize(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return (img > .5).float()


class StretchZeroOne(object):
    def __init__(self):
        pass

    def __call__(self, img):
        assert torch.min(img) >= 0 and torch.max(img) <= 1
        return 2 * img - 1


class TwoDigitsToOneImage(object):
    def __init__(self):
        pass

    def __call__(self, data):
        x0, x1 = data
        return torch.cat([x0[0], x1[0]], dim=-1), [x0[1], x1[1]]


class FourDigitsToTwoImages(object):
    def __init__(self):
        pass

    def __call__(self, data):
        x0, x1, y0, y1 = data
        return [(torch.cat([x0[0], x1[0]], dim=-3), [x0[1], x1[1]]),
                (torch.cat([y0[0], y1[0]], dim=-3), [y0[1], y1[1]])]


class TensorDatasetWithTransform(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        super().__init__(tensors)
        # assert all(tensors[0].shape == tensor.shape for tensor in tensors), [tensor.shape for tensor in tensors]
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        return self.transform([tensor[index] for tensor in self.tensors])

    def __len__(self):
        return len(self.tensors[0])