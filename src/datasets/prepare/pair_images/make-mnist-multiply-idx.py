import torch
from torchvision import datasets, transforms

max_d_train = 1000  # maximum number of data points per class
max_d_valid = 100  # maximum number of data points per class
max_d_test = 100  # maximum number of data points per class
dm = 1  # data multiplier: random permutations to match


def rand_match_on_idx(l_x, idx_x, l_y, idx_y, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx_x0, _idx_x1, _idx_y0, _idx_y1 = [], [], [], []
    for i in l_x.unique():
        for j in l_x.unique():  # assuming both have same idxs
            k = i * j
            y0, y1 = k // 10, k % 10

            l_idx_x0, l_idx_x1 = idx_x[l_x == i], idx_x[l_x == j]
            l_idx_y0, l_idx_y1 = idx_y[l_y == y0], idx_y[l_y == y1]
            n = min(l_idx_x0.size(0), l_idx_y0.size(0), max_d)
            l_idx_x0, l_idx_x1 = l_idx_x0[:n], l_idx_x1[:n]
            l_idx_y0, l_idx_y1 = l_idx_y0[:n], l_idx_y1[:n]
            for _ in range(dm):
                _idx_x0.append(l_idx_x0[torch.randperm(n)])
                _idx_x1.append(l_idx_x1[torch.randperm(n)])
                _idx_y0.append(l_idx_y0[torch.randperm(n)])
                _idx_y1.append(l_idx_y1[torch.randperm(n)])
    return torch.cat(_idx_x0), torch.cat(_idx_x1), torch.cat(_idx_y0), torch.cat(_idx_y1)


#%%
datapath = '../data/mnist-multiply'

# get the individual datasets
tx = transforms.ToTensor()
train_mnist = datasets.MNIST('../data/', train=True,  download=True, transform=tx)
test_mnist = datasets.MNIST('../data/', train=False, download=True, transform=tx)

mnist_l, mnist_li = train_mnist.targets.sort()
idx00, idx01, idx10, idx11 = rand_match_on_idx(mnist_l, mnist_li, mnist_l, mnist_li, max_d=max_d_train, dm=dm)
print('len train idx:', len(idx00))
torch.save(idx00, '{}/train-dc{}-m{}-idx00.pt'.format(datapath, max_d_train, dm))
torch.save(idx01, '{}/train-dc{}-m{}-idx01.pt'.format(datapath, max_d_train, dm))
torch.save(idx10, '{}/train-dc{}-m{}-idx10.pt'.format(datapath, max_d_train, dm))
torch.save(idx11, '{}/train-dc{}-m{}-idx11.pt'.format(datapath, max_d_train, dm))

mnist_l, mnist_li = train_mnist.targets.sort()
idx00, idx01, idx10, idx11 = rand_match_on_idx(mnist_l, mnist_li, mnist_l, mnist_li, max_d=max_d_valid, dm=dm)
print('len valid idx:', len(idx00))
torch.save(idx00, '{}/valid-dc{}-m{}-idx00.pt'.format(datapath, max_d_valid, dm))
torch.save(idx01, '{}/valid-dc{}-m{}-idx01.pt'.format(datapath, max_d_valid, dm))
torch.save(idx10, '{}/valid-dc{}-m{}-idx10.pt'.format(datapath, max_d_valid, dm))
torch.save(idx11, '{}/valid-dc{}-m{}-idx11.pt'.format(datapath, max_d_valid, dm))

mnist_l, mnist_li = test_mnist.targets.sort()
idx00, idx01, idx10, idx11 = rand_match_on_idx(mnist_l, mnist_li, mnist_l, mnist_li, max_d=max_d_test, dm=dm)
print('len test idx:', len(idx00))
torch.save(idx00, '{}/test-dc{}-m{}-idx00.pt'.format(datapath, max_d_test, dm))
torch.save(idx01, '{}/test-dc{}-m{}-idx01.pt'.format(datapath, max_d_test, dm))
torch.save(idx10, '{}/test-dc{}-m{}-idx10.pt'.format(datapath, max_d_test, dm))
torch.save(idx11, '{}/test-dc{}-m{}-idx11.pt'.format(datapath, max_d_test, dm))

