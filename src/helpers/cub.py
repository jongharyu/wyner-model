import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from gensim.models import FastText
from matplotlib import pyplot as plt
from nltk import sent_tokenize, word_tokenize
from scipy.linalg import eig
from skimage.filters import threshold_yen as threshold
from torch.nn import functional as F
from tqdm import tqdm

from datasets.cub import CUBImgFt, OrderedCounter
from datasets.utils import unpack_data


def find_nearest_neighbors(queries, database):
    # TODO: check if the following is better than sklearn + cpu?
    indices = compute_pairwise_distances(database.to(queries.device), queries).argmin(dim=0)
    return indices


def compute_pairwise_distances(sample1, sample2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n1, d)``.
    sample2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n2, d)``.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n1, n2). The [i, j]-th entry is equal to
        ``|| sample1[i, :] - sample2[j, :] ||_p``."""
    # TODO: implement cosine distance
    if len(sample1.shape) == 2:
        sample1, sample2 = sample1.unsqueeze(0), sample2.unsqueeze(0)
    B, n1, n2 = sample1.shape[0], sample1.shape[1], sample2.shape[1]
    norms_1 = torch.sum(sample1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n1, n2)
             + norms_2.transpose(1, 2).expand(B, n1, n2))
    distances_squared = norms - 2 * sample1.matmul(sample2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


class CUBImgHelper:
    def __init__(self, imgft: bool, root_path, device):
        self.imgft = imgft
        self.path_cub = os.path.join(root_path, 'data', 'cub')
        self.path_cub_img_root = os.path.join(root_path, 'data', 'cub', 'img')
        self.train_dataset = CUBImgFt(root_path, 'train', device)
        self.test_dataset = CUBImgFt(root_path, 'test', device)

    def imgft2imgpaths(self, features, search_split):
        dataset = self.train_dataset if search_split == 'train' else self.test_dataset
        idxs = find_nearest_neighbors(features, dataset.features).cpu().numpy()
        return dataset.paths[idxs]

    def retrieve_img(self, imgft, search_split):
        if self.imgft:
            imgpath = self.imgft2imgpaths(imgft.unsqueeze(0), search_split=search_split)
            return Image.open(os.path.join(self.path_cub_img_root, search_split, imgpath))
        else:
            return ((imgft + 1) / 2).permute(1, 2, 0).detach().cpu().numpy()

    def save_imgft_recon(self, imgfts, imgfts_hat, filepath, n_refs=8):
        imgfts = imgfts.data.cpu()

        fig, axes = plt.subplots(ncols=2, nrows=n_refs, figsize=(4 * 2, 4 * n_refs))
        for i, (imgft, imgft_hat) in enumerate(zip(imgfts, imgfts_hat)):
            if i >= n_refs:
                break
            img = self.retrieve_img(imgft, search_split='test')
            axes[i][0].imshow(img)
            axes[i][0].axis('off')
            img_hat = self.retrieve_img(imgft_hat, search_split='test')
            axes[i][1].imshow(img_hat)
            axes[i][1].axis('off')

        plt.savefig(filepath, bbox_inches='tight')
        plt.close()


class CUBSentHelper:
    def __init__(self, root_path, max_sentence_length, min_occur,
                 embedding_dim, window_length,
                 reset=False, device=None):
        self.min_occur = min_occur
        self.max_sentence_length = max_sentence_length

        self.embedding_dim = embedding_dim
        self.window_length = window_length

        self.root_path = root_path
        self.path_cub_sent_root = os.path.join(root_path, 'data', 'cub', 'sent')
        self.RESET = reset
        self.device = device

        self.word2vec_embedding = self.get_word2vec_embedding()
        self.weights = self.get_weights()

    @property
    def vocab(self):
        # call dataloader function to create vocab file
        path_vocab_file = os.path.join(self.path_vocab, 'cub.vocab')
        if not os.path.exists(path_vocab_file):
            raise ValueError("Generate vocabulary first by running get_dataloaders_cub_sentences(batch_size)")
        with open(path_vocab_file, 'r') as file:
            vocab = json.load(file)
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab['word2idx'])

    def truncate(self, sent):
        # truncate after <eos>
        sent_ = sent.cpu().numpy() if isinstance(sent, torch.Tensor) else sent
        return sent[:np.where(sent_ == self.eos_idx)[0][0] + 1] if self.eos_idx in sent else sent

    def get_word2vec_embedding(self):
        # load word embeddings
        if os.path.exists(self.path_word2vec_embedding) and not self.RESET:
            with open(self.path_word2vec_embedding, 'rb') as file:
                embedding = pickle.load(file)
        else:
            embedding = self._generate_save_word2vec_embedding()

        return torch.from_numpy(embedding).to(self.device)

    def get_weights(self):
        # load sentence weights
        path_weights = os.path.join(self.path_vocab, 'cub.weights')
        if os.path.exists(path_weights) and not self.RESET:
            with open(path_weights, 'rb') as file:
                weights = pickle.load(file)
        else:
            weights = self._generate_save_weights()

        return torch.from_numpy(weights).to(self.device).type(self.word2vec_embedding.dtype)

    def _generate_save_word2vec_embedding(self):
        with open(os.path.join(self.path_cub_sent_root, 'text_trainvalclasses.txt'), 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        texts = []
        for i, line in tqdm(enumerate(sentences)):
            words = word_tokenize(line)
            texts.append(words)

        model = FastText(vector_size=self.embedding_dim, window=self.window_length, min_count=self.min_occur)
        model.build_vocab(corpus_iterable=texts)
        model.train(corpus_iterable=texts, total_examples=len(texts), epochs=10)

        base = np.ones((self.embedding_dim,), dtype=np.float32)
        embedding = [base * (i - 1) for i in range(3)]
        # Note: first three words are {<exc>, <pad>, <eos>}
        for word in list(self.idx2word.values())[3:]:
            embedding.append(model.wv[word])

        embedding = np.array(embedding)
        with open(self.path_word2vec_embedding, 'wb') as file:
            pickle.dump(embedding, file)

        return embedding

    def _generate_save_weights(self, a=1e-3):
        with open(os.path.join(self.path_cub_sent_root, 'text_trainvalclasses.txt'), 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

            counter = OrderedCounter()

            for i, line in enumerate(sentences):
                words = word_tokenize(line)
                counter.update(words)

        weights = np.zeros(self.vocab_size)
        total_cnt = sum(list(counter.values()))
        exc_cnt = 0  # count for non-existing words
        for word, cnt in counter.items():
            if word in self.vocab['word2idx'].keys():
                weights[self.vocab['word2idx'][word]] = a / (a + cnt / total_cnt)
            else:
                exc_cnt += cnt
        weights[0] = a / (a + exc_cnt / total_cnt)  # put these weights to the first slot, which is vacant

        with open(os.path.join(self.path_vocab, 'cub.weights'), 'wb') as file:
            pickle.dump(weights, file)

        return weights

    @property
    def pad_idx(self):
        return self.vocab['word2idx']['<pad>']

    @property
    def eos_idx(self):
        return self.vocab['word2idx']['<eos>']

    @property
    def exc_idx(self):
        return self.vocab['word2idx']['<exc>']

    @property
    def path_vocab(self):
        return os.path.join(self.path_cub_sent_root, 'oc={}_sl={}'.format(self.min_occur, self.max_sentence_length))

    @property
    def path_word2vec_embedding(self):
        return os.path.join(self.path_vocab, 'cub.w2v_emb.dim={}_wl={}'.format(self.embedding_dim, self.window_length))

    def get_weighted_embeddings(self, sentences):
        batch = []
        for sent_in_idxs in sentences:
            embeddings_stacked = torch.stack([self.word2vec_embedding[idx] for idx in self.truncate(sent_in_idxs)])
            weights_stacked = torch.stack([self.weights[idx] for idx in self.truncate(sent_in_idxs)])
            batch.append(torch.sum(embeddings_stacked * weights_stacked.unsqueeze(-1), dim=0) / embeddings_stacked.shape[0])

        return torch.stack(batch, dim=0).to(self.device)

    @property
    def idx2word(self):
        return self.vocab['idx2word']

    def truncate_after_eos(self, sents_in_idx):
        assert len(sents_in_idx.shape) == 2
        return [self.truncate(sent) for sent in sents_in_idx.long()]

    def convert_idxs_to_words(self, sent_in_idxs, skip_pad=False):
        if skip_pad:
            idxs = [self.idx2word[str(idx)] for idx in sent_in_idxs if idx != 0]
        else:
            idxs = [self.idx2word[str(idx)] for idx in sent_in_idxs]
        return ' '.join(idxs)

    def save_sent_recon(self, sents, logits_hat, filepath, n_refs=8):
        fig, axes = plt.subplots(ncols=2, nrows=n_refs, figsize=(4 * 2, 4 * n_refs))
        for iref, (sent, sent_hat) in enumerate(zip(sents, logits_hat.argmax(-1))):
            if iref >= n_refs:
                break
            self.plot_caption(axes[iref][0], sent_in_idx=sent)
            self.plot_caption(axes[iref][1], sent_in_idx=sent_hat)

        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

    def plot_caption(self, ax, sent_in_idx, fontsize=12):
        sent_in_idx = self.truncate_after_eos(sent_in_idx.unsqueeze(0))[0].cpu().numpy()  # truncate after <eos>
        caption = ' '.join(self.idx2word[str(idx)] + '\n' if (n + 1) % 5 == 0
                           else self.idx2word[str(idx)] for n, idx in enumerate(sent_in_idx))
        ax.text(x=0, y=1,
                s=caption,
                fontsize=fontsize,
                verticalalignment='top',
                horizontalalignment='left')
        ax.axis('off')
        return ax

    def reconstruct(self, generator, data, epoch, run_path, n_examples=3, n_saves=8):
        # Node: generated samples need to be processed with embedding_inverse
        embed_recons = generator.reconstruct(data[:n_saves])
        recons = generator.decoders['y'].to_data.embedding_inverse(embed_recons).argmax(-1)
        data_sents_in_idx = self.truncate_after_eos(data)
        recons_sents_in_idx = self.truncate_after_eos(recons)

        print("\n Reconstruction examples (excluding <PAD>):")
        for data_sent, recon_sent in zip(data_sents_in_idx[:n_examples], recons_sents_in_idx[:n_examples]):
            print('[DATA]:\t {}'.format(self.convert_idxs_to_words(data_sent, skip_pad=True)))
            print('[RECON]:\t {}\n'.format(self.convert_idxs_to_words(recon_sent, skip_pad=True)))

        with open('{}/recon_{:03d}.txt'.format(run_path, epoch), "w+") as txt_file:
            for data_sent, recon_sent in zip(data_sents_in_idx, recons_sents_in_idx):
                txt_file.write('[DATA]:\t {}'.format(self.convert_idxs_to_words(data_sent)))
                txt_file.write('[RECON]:\t {}\n'.format(self.convert_idxs_to_words(recon_sent)))

    def generate(self, model, epoch, run_path, n_examples=3, n_samples=8):
        samples = model.joint_generation(n_samples).argmax(dim=-1).squeeze()  # (n_samples, max_sentence_length)
        samples = [[self.truncate(sent) for sent in sents] for sents in samples.long()]  # TODO: What does it mean?

        print("\n Generated examples (excluding <PAD>):")
        for sent_in_idxs in samples[0][:n_examples]:
            print('[GEN]   ==> {}'.format(self.convert_idxs_to_words(sent_in_idxs, skip_pad=True)))

        with open('{}/gen_samples_{:03d}.txt'.format(run_path, epoch), "w+") as txt_file:
            for sents in samples:
                for sent in sents:
                    txt_file.write('{}\n'.format(self.convert_idxs_to_words(sent)))
                txt_file.write('\n')


class CUBHelper:
    def __init__(self, img_helper: CUBImgHelper, sent_helper: CUBSentHelper, run_path):
        self.img_helper = img_helper
        self.sent_helper = sent_helper
        self.run_path = run_path

    def save_joint_samples(self, imgfts, logits, filepath):
        sents_in_idx = logits.argmax(-1)
        nz, n_per_ref = imgfts.shape[:2]
        fig, axes = plt.subplots(ncols=n_per_ref, nrows=2 * nz, figsize=(4 * n_per_ref, 4 * (2 * nz)))
        for i, (imgfts_, sents_in_idx_) in enumerate(zip(imgfts, sents_in_idx)):
            for j, (imgft, sent_in_idx) in enumerate(zip(imgfts_, sents_in_idx_)):
                img = self.img_helper.retrieve_img(imgft, search_split='train')
                axes[2 * i][j].imshow(img)
                axes[2 * i][j].axis('off')
                self.sent_helper.plot_caption(axes[2 * i + 1][j], sent_in_idx=sent_in_idx, fontsize=8)

        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

    def save_conditional_samples(self, conditions, samples, from_to, filepath, n_refs=8, n_samples=8, ground_truths=None):
        assert from_to in ['x2y', 'y2x']
        imgfts = conditions if from_to == 'x2y' else samples
        imgfts = imgfts.data.cpu()
        sents_in_idx = samples.argmax(-1) if from_to == 'x2y' else conditions

        if from_to == 'x2y':
            fig, axes = plt.subplots(ncols=n_samples + 1, nrows=n_refs, figsize=(4 * (n_samples + 1), 4 * n_refs))
            # - x2y: one image, multiple sentences
            for iref in range(n_refs):
                if iref >= n_refs:
                    break
                img = self.img_helper.retrieve_img(imgfts[iref], search_split='test')
                axes[iref][0].imshow(img)
                axes[iref][0].axis('off')
                for i, sent_in_idx in enumerate(sents_in_idx[iref]):
                    if i >= n_samples:
                        break
                    self.sent_helper.plot_caption(axes[iref][i + 1], sent_in_idx=sent_in_idx)
        else:
            fig, axes = plt.subplots(ncols=n_samples + 2, nrows=n_refs, figsize=(4 * (n_samples + 2), 4 * n_refs))
            # - y2x: one sentence, multiple images
            assert ground_truths is not None
            for iref in range(n_refs):
                if iref >= n_refs:
                    break
                self.sent_helper.plot_caption(axes[iref][0], sent_in_idx=sents_in_idx[iref])
                img_gt = self.img_helper.retrieve_img(ground_truths[iref], search_split='test')
                axes[iref][1].imshow(img_gt)
                axes[iref][1].axis('off')
                for i, imgft in enumerate(imgfts[iref]):
                    if i >= n_samples:
                        break
                    img = self.img_helper.retrieve_img(imgft, search_split='test')
                    axes[iref][i + 2].imshow(img)
                    axes[iref][i + 2].axis('off')

        plt.savefig(filepath, bbox_inches='tight')
        plt.close()


class CUBCanonicalCorrelationAnalysis:
    def __init__(self, train_loader, test_loader, sent_helper, batch_size, run_path, device):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.sent_helper = sent_helper

        self.batch_size = batch_size
        self.run_path = run_path
        self.device = device

        self.principal_comps = self.get_principal_comps()

        path_cca_stat = os.path.join(self.sent_helper.root_path, 'data', 'cub', 'cca')
        if not os.path.exists(os.path.join(path_cca_stat, 'img_mean.pt')):
            print("Saving statistics of training data for CCA...")
            self._save_cca_projs()
        self.img_mean = torch.load(os.path.join(path_cca_stat, 'img_mean.pt')).to(device)
        self.embed_mean = torch.load(os.path.join(path_cca_stat, 'embed_mean.pt')).to(device)
        self.img_proj = torch.load(os.path.join(path_cca_stat, 'img_proj.pt')).to(device)
        self.embed_proj = torch.load(os.path.join(path_cca_stat, 'embed_proj.pt')).to(device)

    def apply_principal_comps(self, weighted_embeddings, batch_size=2048):
        return torch.cat([embed - torch.matmul(self.principal_comps, embed.unsqueeze(-1)).squeeze()
                          for embed in weighted_embeddings.split(batch_size, 0)]).to(self.device)

    def to_embedding(self, sents):
        # map sentence to a projection of its weighted embedding onto principal components
        return self.apply_principal_comps(self.sent_helper.get_weighted_embeddings(sents.int()))

    def get_principal_comps(self):
        # load principal components
        path_principal_comps = os.path.join(self.sent_helper.path_vocab, 'cub.pc')
        if os.path.exists(path_principal_comps) and not self.sent_helper.RESET:
            with open(path_principal_comps, 'rb') as file:
                principal_comps = pickle.load(file)
        else:
            principal_comps = self._generate_save_principal_comps()

        return principal_comps.to(self.device)

    def _generate_save_principal_comps(self):
        sents = torch.cat([batch[1][0] for batch in self.train_loader]).int()  # note: batch[1][1] is sentence length up to <eos>
        weighted_embeddings = self.sent_helper.get_weighted_embeddings(sents)
        _, _, V = torch.svd(weighted_embeddings - weighted_embeddings.mean(dim=0), some=True)
        v = V[:, 0].unsqueeze(-1)
        u = v.mm(v.t())
        with open(os.path.join(self.sent_helper.path_vocab, 'cub.pc'), 'wb') as file:
            pickle.dump(u, file)

        return u

    def compute_corrs(self, imgs, sents):
        embeds = self.to_embedding(sents)
        with torch.no_grad():
            corr = F.cosine_similarity((imgs - self.img_mean) @ self.img_proj,
                                       (embeds - self.embed_mean) @ self.embed_proj).mean()
        return corr.item()

    def _save_cca_projs(self, k=40):
        imgs, sents = [torch.cat(pair) for pair in zip(*[(batch[0], batch[1][0]) for batch in self.train_loader])]
        embeds = self.to_embedding(sents.int())

        corr, (img_proj, embed_proj) = self.compute_cca([imgs, embeds], k=k)
        img_mean = imgs.mean(dim=0)
        embed_mean = embeds.mean(dim=0)

        print("Largest eigenvalue from CCA: {:.3f}".format(corr[0]))

        torch.save(img_mean, os.path.join(self.run_path, 'img_mean.pt'))
        torch.save(embed_mean, os.path.join(self.run_path, 'embed_mean.pt'))
        torch.save(img_proj, os.path.join(self.run_path, 'img_proj.pt'))
        torch.save(embed_proj, os.path.join(self.run_path, 'embed_proj.pt'))

    def compute_correlation(self, generator, modes=('gt',)):
        assert set(modes).issubset(['gt', 'j', 'cx2y', 'cy2x'])
        corrs = defaultdict(list)
        # TODO: implement raw image version
        with torch.no_grad():
            if 'j' in modes:
                for i in tqdm(range(min(16, len(self.test_loader)))):
                    zp, up, vp = generator.draw_from_prior(n=self.test_loader.batch_size, device=self.device)
                    imgfts, logits = generator.decode_zuv_xy(zp, up, vp)
                    generated_sents = logits.argmax(-1)
                    # drop the last batch
                    if imgfts.shape[0] < self.batch_size:
                        break
                    # compute correlation with CCA
                    corr = self.compute_corrs(imgfts, generated_sents)
                    corrs['j'].append(corr)
            if set(modes).intersection(['cx2y', 'cy2x', 'gt']):
                dataloader = self.test_loader
                for i, batch in tqdm(enumerate(dataloader), total=len(self.test_loader)):
                    imgfts, sents = unpack_data(batch, device=self.device)
                    # drop the last batch
                    if imgfts.shape[0] < self.batch_size:
                        break
                    # compute correlation with CCA
                    if 'cx2y' in modes:  # img2sent
                        logits = generator.draw_conditional_samples(imgfts, from_to='x2y')
                        # since yp is in embedding domain (B, msl, embedding_dim),
                        # we should make it of size (B, msl)
                        generated_sents = logits.argmax(-1)
                        corrs['cx2y'].append(self.compute_corrs(imgfts, generated_sents))
                    if 'cy2x' in modes:  # sent2img
                        generated_imgs = generator.draw_conditional_samples(sents, from_to='y2x')
                        corrs['cy2x'].append(self.compute_corrs(generated_imgs, sents))
                    if 'gt' in modes:
                        corrs['gt'].append(self.compute_corrs(imgfts, sents))

        return {mode: np.array(corrs[mode]).mean() for mode in corrs}

    @staticmethod
    def compute_cca(views, k=None, eps=1e-12):
        """Compute (multi-view) CCA

        Args:
            views (list): list of views where each view `v_i` is of size `N x o_i`
            k (int): joint projection dimension | if None, find using Otsu
            eps (float): regularizer [default: 1e-12]

        Returns:
            correlations: correlations along each of the k dimensions
            projections: projection matrices for each view
        """
        n_views = len(views)  # number of views
        n_observations = views[0].size(0)  # number of observations (same across views)
        os = [v.size(1) for v in views]
        kmax = np.min(os)
        ocum = np.cumsum([0] + os)
        os_sum = sum(os)
        A, B = np.zeros([os_sum, os_sum]), np.zeros([os_sum, os_sum])

        for i in range(n_views):
            v_i = views[i]
            v_i_bar = v_i - v_i.mean(0).expand_as(v_i)  # centered, N x o_i
            C_ij = (1.0 / (n_observations - 1)) * torch.mm(v_i_bar.t(), v_i_bar)
            # A[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
            B[ocum[i]:ocum[i + 1], ocum[i]:ocum[i + 1]] = C_ij
            for j in range(i + 1, n_views):
                v_j = views[j]  # N x o_j
                v_j_bar = v_j - v_j.mean(0).expand_as(v_j)  # centered
                C_ij = (1.0 / (n_observations - 1)) * torch.mm(v_i_bar.t(), v_j_bar)
                A[ocum[i]:ocum[i + 1], ocum[j]:ocum[j + 1]] = C_ij
                A[ocum[j]:ocum[j + 1], ocum[i]:ocum[i + 1]] = C_ij.t()

        A[np.diag_indices_from(A)] += eps
        B[np.diag_indices_from(B)] += eps

        eigenvalues, eigenvectors = eig(A, B)
        # TODO: sanity check to see that all eigenvalues are e+0i
        # sort eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]  # arrange in descending order

        if k is None:
            t = threshold(eigenvalues.real[:kmax])
            k = np.abs(np.asarray(eigenvalues.real[0::10]) - t).argmin() * 10  # closest k % 10 == 0 idx
            print('Since k unspecified, choose k={}'.format(k))

        eigenvalues = eigenvalues[idx[:k]]
        eigenvectors = eigenvectors[:, idx[:k]]

        correlations = torch.from_numpy(eigenvalues.real).type_as(views[0])
        projections = torch.split(torch.from_numpy(eigenvectors.real).type_as(views[0]), os)

        return correlations, projections
