import io
import json
import os
import pickle
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import torch
from nltk import sent_tokenize, word_tokenize
from torch import nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
from tqdm import tqdm


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class CUBSent(Dataset):
    def __init__(self, root_path, split, transform=None, max_sequence_length=32, min_occur=3):
        super().__init__()
        assert split in ['train', 'test'], "Only train or test split is available"
        self.path_cub_sent_root = os.path.join(root_path, 'data', 'cub', 'sent')
        self.split = split

        self.max_sequence_length = max_sequence_length
        self.min_occur = min_occur

        self.transform = transform

        if split == 'train':
            self.path_cub_sent_data_raw = os.path.join(self.path_cub_sent_root, 'text_trainvalclasses.txt')
        elif split == 'test':
            self.path_cub_sent_data_raw = os.path.join(self.path_cub_sent_root, 'text_testclasses.txt')
        else:
            raise ValueError

        os.makedirs(self.path_vocab, exist_ok=True)
        self.path_vocab_file = os.path.join(self.path_vocab, 'cub.vocab')
        self.path_cub_sent_data = os.path.join(self.path_vocab, 'cub.{}.s{}'.format(split, self.max_sequence_length))

        if os.path.exists(self.path_cub_sent_data):
            self._load_data()
        else:
            print("Preprocessed data file not found for {} split at {}. "
                  "Creating new... (this may take a while)".
                  format(split.upper(), self.path_cub_sent_data))
            self._save_preprocessed_data()

    @property
    def path_vocab(self):
        return os.path.join(self.path_cub_sent_root, "oc={}_sl={}".format(self.min_occur, self.max_sequence_length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[str(idx)]['idx']
        if self.transform is not None:
            sent = self.transform(sent)
        return sent, self.data[str(idx)]['length']

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def _load_data(self, load_vocab=True):
        with open(self.path_cub_sent_data, 'rb') as file:
            self.data = json.load(file)

        if load_vocab:
            self._load_vocab()

        return self

    def _load_vocab(self):
        if os.path.exists(self.path_vocab_file):
            with open(self.path_vocab_file, 'r') as vocab_file:
                vocab = json.load(vocab_file)
            self.word2idx, self.idx2word = vocab['word2idx'], vocab['idx2word']
            return self
        else:
            return self._build_vocab()

    def _save_preprocessed_data(self):
        if os.path.exists(self.path_vocab_file) or self.split == 'test':
            self._load_vocab()
        else:
            self._build_vocab()

        with open(self.path_cub_sent_data_raw, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.word2idx.get(w, self.word2idx['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(self.path_cub_sent_data, 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        return self._load_data(load_vocab=False)

    def _build_vocab(self):
        assert self.split == 'train', "Vocabulary can be created only for training file."

        with open(self.path_cub_sent_data_raw, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        counter = OrderedCounter()
        word2idx = dict()
        idx2word = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for token in special_tokens:
            idx2word[len(word2idx)] = token
            word2idx[token] = len(word2idx)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            counter.update(words)
            texts.append(words)

        for word, count in counter.items():
            if count > self.min_occur and word not in special_tokens:
                idx2word[len(word2idx)] = word
                word2idx[word] = len(word2idx)
            else:
                unq_words.append(word)

        assert len(word2idx) == len(idx2word)

        print("Vocabulary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(word2idx), len(unq_words), self.min_occur))

        vocab = dict(word2idx=word2idx, idx2word=idx2word)
        with io.open(self.path_vocab_file, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.path_vocab, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.path_vocab, 'cub.all'), 'wb') as a_file:
            pickle.dump(counter, a_file)

        return self._load_vocab()


class CUBImgFt(Dataset):
    def __init__(self, root_path, split, device):
        super().__init__()
        assert split in ['train', 'test']
        path_cub_img_root = os.path.join(root_path, 'data', 'cub', 'img')
        self.path_cub_img = os.path.join(path_cub_img_root, split)
        path_cub_imgft_root = os.path.join(path_cub_img_root, 'resnet101_2048')
        self.path_cub_imgft = os.path.join(path_cub_imgft_root, '{}.ft'.format(split))
        self.path_cub_img_paths = os.path.join(path_cub_img_root, '{}.paths.npy'.format(split))
        self.split = split

        tx = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        self.dataset = datasets.ImageFolder(self.path_cub_img, transform=tx)
        self.paths = ['/'.join(img[0].split('/')[-2:]) for img in self.dataset.imgs]

        os.makedirs(path_cub_imgft_root, exist_ok=True)
        if os.path.exists(self.path_cub_imgft):
            self._load_features()
        else:
            print("Data file not found for CUB image features at `{}`. "
                  "Extracting resnet101 features from CUB image dataset... "
                  "(this may take a while)".format(self.path_cub_imgft))
            self._save_features(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def _load_features(self):
        self.features = torch.load(self.path_cub_imgft)
        self.features_stdev = self.features.std(dim=0, keepdims=True)
        self.paths = np.load(self.path_cub_img_paths)
        return self

    def _save_features(self, device):
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.model.eval()

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=256,
                                                 shuffle=False, drop_last=False, **kwargs)
        pathsloader = torch.utils.data.DataLoader(self.paths, batch_size=256,
                                                  shuffle=False, drop_last=False, **kwargs)
        features =[]
        paths = []
        with torch.no_grad():
            for i, ((batch_imgs, _), batch_paths) in tqdm(enumerate(zip(dataloader, pathsloader))):
                features.append(self.model(batch_imgs).squeeze())
                paths.append(batch_paths)

            features = torch.cat(features)
            paths = np.concatenate(paths)

        torch.save(features, self.path_cub_imgft)
        np.save(self.path_cub_img_paths, paths)
        del features, paths

        self._load_features()

        return self
