import argparse
import os

import numpy as np
import scipy.io
from tqdm import tqdm
from PIL import Image


def main(config):
    path_svhn = os.path.join(config.root_path, 'data', 'SVHN')
    path_svhn_jpg = os.path.join(config.root_path, 'data', 'SVHNjpg')

    # Reference: https://gist.github.com/aferriss/e80a120e0f9dc39b307f
    for idx, data_type in enumerate(config.data_types):
        path = os.path.join(path_svhn_jpg, '{}Set'.format(data_type))
        mat = scipy.io.loadmat('{}/{}_32x32.mat'.format(path_svhn, data_type))
        imgs = np.array(mat["X"])
        labels = np.array(mat["y"])
        print(data_type, [np.sum(labels % 10 == label) for label in range(10)])
        for label in range(10):
            path_label = os.path.join(path, str(label))
            if not os.path.exists(path_label):
                os.makedirs(path_label)
        for i in tqdm(range(labels.shape[0])):
            img = Image.fromarray(imgs[:, :, :, i], 'RGB')
            label = int(labels[i]) % 10
            filename = "img_{:06d}.{}".format(i, config.file_type)
            img.save(os.path.join(path, filename))
            img.save(os.path.join(path, str(label), filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAVE SVHN in jpg files')

    parser.add_argument('--root-path', type=str, default=".", help='SVHN data lives in root_path/data/SVHN/')
    parser.add_argument('--data-types', type=str, nargs='*', default=['extra'], choices=['train', 'test', 'extra'])
    parser.add_argument('--file-type', type=str, default='jpg', choices=['jpg', 'png'])

    config = parser.parse_args()
    main(config)
