# -*- coding: utf-8 -*-
import argparse
import os
import random
from collections import defaultdict
from itertools import cycle
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class SketchyVGGDataLoader:
    def __init__(self, batch_size, shuffle=True, drop_last=False,
                 root_path='..', split=1, train_or_test='train'):
        self.path_sketchy = os.path.join(root_path, 'data', 'Sketchy')
        self.path_sketch = os.path.join(self.path_sketchy, 'sketch', 'tx_000000000000')  # 75,471 sketches
        self.path_photo = os.path.join(self.path_sketchy, 'extended_photo')  # 12,500 + 60,502 = 73,002 photos

        assert not drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.root_path = root_path
        self.split = split
        self.train_or_test = train_or_test

        self.sketch_features, self.sketch_classes, self.sketch_paths, self.sketch_idx_per_class = \
            self.load_sketchy_features(root_path=root_path, split=split, train_or_test=train_or_test, data_type='sketch')
        self.photo_features, self.photo_classes, self.photo_paths, self.photo_idx_per_class = \
            self.load_sketchy_features(root_path=root_path, split=split, train_or_test=train_or_test, data_type='photo')

        self.sketch_features_mean = self.sketch_features.mean(axis=0, keepdims=True)
        self.sketch_features_stdev = self.sketch_features.std(axis=0, keepdims=True)
        self.photo_features_mean = self.photo_features.mean(axis=0, keepdims=True)
        self.photo_features_stdev = self.photo_features.std(axis=0, keepdims=True)

        self.dataset = self.sketch_features  # for compatibility
        self.max_steps = np.ceil(self.sketch_features.shape[0] / batch_size).astype(int)

        assert set(self.sketch_classes.tolist()) == set(self.photo_classes.tolist())
        self.classes = sorted(list(set(self.sketch_classes.tolist())))

    def __iter__(self):
        self._step = 0
        return self

    def __next__(self):
        if self._step < self.max_steps:
            self._step += 1
            train_sketch_idx, train_photo_idx = self._pick_random_pairs()
            batch_sketch = np.take(self.sketch_features, train_sketch_idx, axis=0)
            batch_photo = np.take(self.photo_features, train_photo_idx, axis=0)

            batch_sketch = torch.Tensor(batch_sketch)
            batch_photo = torch.Tensor(batch_photo)

            return batch_sketch, batch_photo
        else:
            raise StopIteration

    def _pick_random_pairs(self):
        sketch_idx_list = []
        photo_idx_list = []
        random.shuffle(self.classes)
        for i, cls in enumerate(cycle(self.classes)):
            if i >= self.batch_size:
                break

            sketch_idx = random.choice(self.sketch_idx_per_class[cls])
            photo_idx = random.choice(self.photo_idx_per_class[cls])
            assert cls == self.sketch_classes[sketch_idx] == self.photo_classes[photo_idx]

            sketch_idx_list.append(sketch_idx)
            photo_idx_list.append(photo_idx)

        return np.array(sketch_idx_list), np.array(photo_idx_list)

    @staticmethod
    def load_sketchy_features(root_path, split, train_or_test, data_type):
        # Note: 75,479 sketches, 73,002 (12,500 + 60,502) photos
        #   Split 1 (100/25; a.k.a. SEM-PCYC split)
        #   Split 2 (104/21; a.k.a. ECCV 2018 split)
        assert split in [1, 2], 'The split argument {} must be either 1 or 2.'.format(split)
        assert train_or_test in ['train', 'test']
        assert data_type in ['sketch', 'photo']
        split_path = 'split{}'.format(split)
        path_sketchy_features = os.path.join(root_path, 'data', 'SketchyVGG', split_path)

        features = np.load(os.path.join(path_sketchy_features, '{}_{}_features.npy'.format(train_or_test, data_type)))
        paths = np.load(os.path.join(path_sketchy_features, '{}_{}_paths.npy'.format(train_or_test, data_type)))
        classes = []
        indices_per_class = defaultdict(list)
        for i, path in enumerate(paths.tolist()):
            cls = path.split('/')[-2]
            indices_per_class[cls].append(i)
            classes.append(cls)
        classes = np.array(classes)

        print("No. {} images ({}): {}".format(data_type, train_or_test, features.shape[0]))

        return features, classes, paths, indices_per_class


class SketchyRetrieval:
    def __init__(self, test_loader,
                 n_images_to_save=10, n_retrievals=100,
                 metric='euclidean',
                 run_path=None,
                 device=None):
        self.test_loader = test_loader
        self.batch_size = test_loader.batch_size
        self.n_images_to_save = n_images_to_save
        self.n_retrievals = n_retrievals
        self.metric = metric
        self.run_path = run_path
        self.device = device

    @staticmethod
    def parse_class(path):
        return path.split('/')[-2]

    def evaluate(self, model, epoch, save_retrieved_images=False):
        sketch_features = self.test_loader.sketch_features
        photo_features = self.test_loader.photo_features
        sketch_classes = self.test_loader.sketch_classes
        photo_classes = self.test_loader.photo_classes

        # Step 1: Convert all test sketches (X) to shared representation.
        zxs = []
        for step in tqdm(range(np.ceil(sketch_features.shape[0] / self.batch_size).astype(int))):
            sketch_batch = torch.Tensor(sketch_features[step * self.batch_size: (step + 1) * self.batch_size]).to(self.device)
            zx = model.encoder.marginal_encoders['x'].encode_x_z(sketch_batch)
            zxs.append(zx)
        zxs = torch.cat(zxs, dim=0).detach().cpu().numpy()

        # Step 2: Convert all test photos (Y) to shared representation.
        zys = []
        for step in tqdm(range(np.ceil(photo_features.shape[0] / self.batch_size).astype(int))):
            photo_batch = torch.Tensor(photo_features[step * self.batch_size: (step + 1) * self.batch_size]).to(self.device)
            zy = model.encoder.marginal_encoders['y'].encode_x_z(photo_batch)
            zys.append(zy)
        zys = torch.cat(zys, dim=0).detach().cpu().numpy()

        # Step 3: compute Precision@K
        relevances_K, _ = self.get_retrievals(zxs, zys,
                                              xclss=sketch_classes, yclss=photo_classes,
                                              K=self.n_retrievals,
                                              metric=self.metric)
        precision_Ks = self.compute_precisions_at_k(relevances_K)
        precision_K = precision_Ks.mean()
        print('P@{} ({})\t{:.4f}'.format(self.n_retrievals, self.metric, precision_K))

        # Step 4: compute mAP@all
        # Note: for computing mAP@K, see https://stackoverflow.com/questions/54966320/mapk-computation
        relevances, retrieved_zys_idxs = self.get_retrievals(zxs, zys,
                                                             xclss=sketch_classes, yclss=photo_classes,
                                                             metric=self.metric)
        average_precisions = self.compute_average_precisions(relevances)
        mAP = average_precisions.mean()
        print('mAP ({})\t{:.4f}'.format(self.metric, mAP))

        # Step 5: (optional) save retrieved images
        if save_retrieved_images:
            self.save_retrieved_images(retrieved_zys_idxs, epoch)

        return precision_Ks, average_precisions

    @staticmethod
    def get_retrievals(zxs, zys, xclss, yclss, K=None, metric='euclidean'):
        if K is None:
            K = len(yclss)

        # find nearest neighbors of zxs (query sketches) with respect to zys (photos)
        nbrs = NearestNeighbors(n_neighbors=K, metric=metric, algorithm='auto').fit(zys)
        _, retrieved_zys_idxs = nbrs.kneighbors(zxs)
        retrieved_yclss = yclss[retrieved_zys_idxs]
        relevances = (retrieved_yclss == xclss[:, np.newaxis])
        # Note: (n_queries, K) matrix 'relevances' contains every information for computing
        #       "precision@K" and "average precision" for each query
        #       relevances[i, j] = (j-th retrieval is relevant for query i)

        return relevances, retrieved_zys_idxs

    def save_retrieved_images(self, retrieved_zys_idxs, epoch):
        path_retrievals = os.path.join(self.run_path, 'retrievals', 'e{:03d}'.format(epoch))

        sketch_paths = self.test_loader.sketch_paths
        photo_paths = self.test_loader.photo_paths

        sketch_idx_per_class = self.test_loader.sketch_idx_per_class
        sketch_classes = self.test_loader.sketch_classes

        path_sketch = self.test_loader.path_sketch
        path_photo = self.test_loader.path_photo

        retrieved_paths = photo_paths[retrieved_zys_idxs[..., :self.n_images_to_save]]
        for cls in tqdm(sorted(set(sketch_classes.tolist())),
                        desc='Saving retrieved images for sketch sample queries...'):
            # create folder
            path_retrievals_per_class = os.path.join(path_retrievals, cls)
            os.makedirs(path_retrievals_per_class)

            # given a class, pick one query sketch image at random
            sketch_idx = random.choice(sketch_idx_per_class[cls])  # alternative: sketch_idx_per_class[label][0]

            # find the path of the query sketch image
            query_path = os.path.join(path_sketch, sketch_paths[sketch_idx])
            assert os.path.exists(query_path)  # sanity check
            copyfile(query_path, os.path.join(path_retrievals_per_class, 'query.jpg'))  # save the query image

            # for the selected query sketch image
            for rank, retrieved_path in enumerate(retrieved_paths[sketch_idx]):
                abs_retrieved_path = os.path.join(path_photo, retrieved_path)
                assert os.path.exists(abs_retrieved_path), 'Error: retrieved photo does not exist!'
                match_suffix = '' if int(self.parse_class(retrieved_path) == cls) else '_f'
                copyfile(abs_retrieved_path,
                         os.path.join(path_retrievals_per_class, '{}{}.jpg'.format(rank, match_suffix)))
        else:
            print('done!')

    @staticmethod
    def compute_precisions_at_k(relevances):
        return relevances.mean(axis=1)

    @staticmethod
    def compute_average_precisions(relevances):
        n_queries = relevances.shape[0]
        prs = relevances.cumsum(axis=1) / np.ones_like(relevances).cumsum(axis=1)  # note: prs[:, K] = Precision@K's
        # perform "optimistic interpolation", a convention in information retrieval
        max_prs = np.maximum.accumulate(prs[..., ::-1], axis=1)[..., ::-1]
        avg_prs = np.zeros(prs.shape[0])
        for i in range(n_queries):
            avg_prs[i] = max_prs[i][relevances[i] == 1].sum() / relevances[i].sum()

        return avg_prs  # (n_queries, )


class VGGNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super(VGGNetFeats, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Linear(4096, 512)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class InvertImage:
    def __init__(self):
        pass

    def __call__(self, x):
        return 1 - x


def main(config):
    # cuda
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    path_sketchy = os.path.join(config.main_path, 'data', 'Sketchy')

    path_sketch_model = os.path.join(path_sketchy, 'pretrained', 'vgg16_sketch.pth')
    path_photo_model = os.path.join(path_sketchy, 'pretrained', 'vgg16_photo.pth')

    # Sketch model: pre-trained on ImageNet
    sketch_model = VGGNetFeats(pretrained=False, finetune=False).to(device)
    sketch_model.load_state_dict(torch.load(path_sketch_model, map_location=device)['state_dict_sketch'])

    # Photo model: pre-trained on ImageNet
    photo_model = VGGNetFeats(pretrained=False, finetune=False).to(device)
    photo_model.load_state_dict(torch.load(path_photo_model, map_location=device)['state_dict_image'])

    transform_sketch = transforms.Compose([transforms.Resize((config.image_size, config.image_size)),
                                           transforms.ToTensor(),
                                           InvertImage()])
    transform_photo = transforms.Compose([transforms.Resize((config.image_size, config.image_size)),
                                          transforms.ToTensor()])

    sketch_dataset = ImageFolder(os.path.join(path_sketchy, 'sketch', 'tx_000000000000'), transform=transform_sketch)
    photo_dataset = ImageFolder(os.path.join(path_sketchy, 'extended_photo'), transform=transform_photo)

    # all the unique classes
    assert set(sketch_dataset.classes) == set(photo_dataset.classes)
    classes = sorted(sketch_dataset.classes)

    # divide the classes
    if config.split == 1:
        # According to Shen et al., "Zero-Shot Sketch-Image Hashing", CVPR 2018.
        np.random.seed(0)
        train_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False)
        test_classes = np.setdiff1d(classes, train_classes)
    else:
        # According to Yelamarthi et al., "A Zero-Shot Framework for Sketch Based Image Retrieval", ECCV 2018.
        with open(os.path.join(path_sketchy, "test_split_eccv2018.txt")) as fp:
            test_classes = fp.read().splitlines()
            train_classes = np.setdiff1d(classes, test_classes)

    split_path = 'split{}'.format(config.split)
    path_features = os.path.join(config.main_path, 'data', 'SketchyVGG', split_path)
    if not os.path.exists(path_features):
        os.makedirs(path_features)

    # 1) Compute and save sketch features
    sketch_loader = DataLoader(sketch_dataset,
                               batch_size=config.batch_size,
                               num_workers=config.num_workers,
                               shuffle=False, drop_last=False,
                               pin_memory=True)
    sketch_features, sketch_classes, sketch_paths = get_features(sketch_model, sketch_loader, train_classes, test_classes, device)
    for train_or_test in ['train', 'test']:
        np.savez_compressed(os.path.join(path_features, '{}_sketch.npz'.format(train_or_test)),
                            features=sketch_features[train_or_test],
                            classes=sketch_classes[train_or_test],
                            paths=sketch_paths[train_or_test])

    # 2) Compute and save photo features
    photo_loader = DataLoader(photo_dataset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=False, drop_last=False,
                              pin_memory=True)
    photo_features, photo_classes, photo_paths = get_features(photo_model, photo_loader, train_classes, test_classes, device)
    for train_or_test in ['train', 'test']:
        np.savez_compressed(os.path.join(path_features, '{}_photo.npz'.format(train_or_test)),
                            features=photo_features[train_or_test],
                            classes=photo_classes[train_or_test],
                            paths=photo_paths[train_or_test])


def get_features(model, dataloader, train_classes, test_classes, device):
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    all_classes = np.array(dataset.classes)

    features = defaultdict(list)
    paths = defaultdict(list)
    classes = defaultdict(list)

    for i, batch in tqdm(enumerate(dataloader)):
        imgs, cls_idxs = batch
        fts = model(imgs.to(device))
        rel_pths = np.array([os.path.join(*path.split('/')[-2:])
                             for (path, _) in dataset.imgs[i * batch_size: (i + 1) * batch_size]])
        clss = all_classes[cls_idxs.numpy()]
        train_idxs = [i for (i, cls) in enumerate(clss) if cls in train_classes]
        test_idxs = [i for (i, cls) in enumerate(clss) if cls in test_classes]

        features['train'].append(fts[train_idxs])
        paths['train'].append(rel_pths[train_idxs])
        classes['train'].append(clss[train_idxs])

        features['test'].append(fts[test_idxs])
        paths['test'].append(rel_pths[test_idxs])
        classes['test'].append(clss[test_idxs])

    for train_or_test in ['train', 'test']:
        features[train_or_test] = torch.cat(features[train_or_test], dim=0).detach().cpu().numpy()
        classes[train_or_test] = np.concatenate(classes[train_or_test], axis=0)
        paths[train_or_test] = np.concatenate(paths[train_or_test], axis=0)

    return features, classes, paths


if __name__ == '__main__':
    # Parse options for processing
    parser = argparse.ArgumentParser(description='Extracting pretrained VGG16 features of Sketchy dataset')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA use')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
    parser.add_argument('--main-path', type=str, default="..",
                        help='main path where datasets live and loggings are saved')
    parser.add_argument('--split', type=int, default=1, choices=[1, 2],
                        help='split1=(SEM-PCYC); split2=(ECCV 2018)')
    parser.add_argument('--image-size', default=224, type=int, help='image size for VGG16 input')
    config = parser.parse_args()

    main(config)
