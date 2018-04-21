import torch
from torch.autograd import Variable

import pickle
import argparse
import numpy as np

from tqdm import tqdm
from os.path import abspath
from collections import defaultdict


def PGD(weights, def_embs, e_star, eta=0.0001, lr=1, max_epoch=500, lamb=0, patience_init=5, norm_ord=2):
    def simplex_project(y):
        D = len(y)
        u = np.sort(y)[::-1]
        x_tmp = (1. - np.cumsum(u)) / np.arange(1, D + 1)
        lmd = x_tmp[np.sum(u + x_tmp > 0) - 1]
        return np.maximum(y + lmd, 0)

    epoch = 0
    patience = patience_init
    loss_diff = eta + 1
    loss = np.linalg.norm(np.dot(weights, def_embs) - e_star, ord=2) + lamb * np.linalg.norm(weights, ord=norm_ord)
    
    while abs(loss_diff) > eta and epoch < max_epoch:
        epoch += 1
        gradient = 2 * np.dot(np.dot(weights, def_embs) - e_star, def_embs.T) + norm_ord * lamb * weights
        weights -= lr * gradient
        weights = simplex_project(weights)
        loss_new = np.linalg.norm(np.dot(weights, def_embs) - e_star, ord=2) + lamb * np.linalg.norm(weights, ord=norm_ord)
        # print(loss_new)
        loss_diff = loss_new - loss
        if loss_diff > eta:
            lr /= 2
            patience -= 1
            if patience < 0:
                break
        else:
            loss = loss_new
            patience = patience_init
    # assert False
    return weights


def get_weights(target_emb, def_embs, lamb, norm_ord):
    # import torch.nn.functional as F

    # target_emb = F.normalize(target_emb, dim=0).numpy()
    # def_embs = F.normalize(def_embs, dim=1).numpy()
    # weights = np.random.rand(def_embs.shape[0])
    # weights_projected, loss = PGD(weights, def_embs, target_emb, lamb=lamb, norm_ord=norm_ord)
    # return weights_projected, loss
    # 
    import torch.nn.functional as F

    target_emb = F.normalize(target_emb, dim=0).numpy()
    def_embs = F.normalize(def_embs, dim=1).numpy()
    b = target_emb - def_embs[0]
    a = def_embs[1:, :] - def_embs[0]
    num_def = a.shape[0]
    X = np.zeros((num_def, num_def))
    i = 0
    j = 0
    for i in range(num_def):
        for j in range(i, num_def):
            X[i][j] = X[j][i] = np.dot(a[i], a[j])
    y = np.dot(a, b)
    X = np.asmatrix(X)
    y = np.asmatrix(y).T
    weights = np.linalg.solve(X, y)
    weights = np.array([1 - weights.sum()] + weights.T.tolist()[0])
    if weights.min() >= 0 and weights.max() <= 1:
        return weights
    e_star = np.dot(weights, def_embs)
    weights = np.random.rand(def_embs.shape[0])
    weights_projected = PGD(weights, def_embs, e_star, lamb=lamb, norm_ord=norm_ord)
    # weights_projected, loss = PGD(weights, def_embs, target_emb, lamb=lamb, norm_ord=norm_ord)
    return weights_projected


if __name__ == '__main__':
    import sys
    sys.path.append(abspath('../src'))

    import model.util as util
    from model.rnn import RNNEncoder

    parser = argparse.ArgumentParser('dictionary generation')
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--data', type=str, required=True, help='dataset path')
    parser.add_argument('-e', '--encoder', type=str, default='', help='load checkpoint')
    parser.add_argument('-o', '--output', type=str, required=True, help='output path')
    parser.add_argument('-l', '--lamb', type=float, default=0, help='l2-norm coef for weight')
    parser.add_argument('-n', '--norm', type=float, default=2, help='weight normalization ord')
    args = parser.parse_args()
    print(args)

    # load data
    old_data = pickle.load(open(args.data, 'rb'))

    dic_embed = old_data['dic_embed']
    def_embed = old_data['def_embed']
    dic_word2ix = old_data['dic_word2ix']
    def_word2ix = old_data['def_word2ix']
    train_pairs = old_data['train_pairs']
    valid_pairs = old_data['valid_pairs']
    rvd_candidates = old_data['rvd_candidates']
    rvd_pairs_seen = old_data['rvd_pairs_seen']
    rvd_pairs_unseen = old_data['rvd_pairs_unseen']
    rvd_pairs_concept = old_data['rvd_pairs_concept']
    print('load data ok')

    train_word2sens = defaultdict(list)
    for word, sen in train_pairs:
        train_word2sens[word].append(sen)

    new_train_pairs = []

    if args.encoder:
        # set GPU device
        if args.gpu > -1:
            torch.cuda.set_device(args.gpu)

        # load checkpoint
        print('loading checkpoint:', args.encoder)
        checkpoint = torch.load(args.encoder, map_location=lambda storage, loc: storage)
        model_args = checkpoint['args']

        encoder = RNNEncoder(def_word2ix, model_args).cuda()
        encoder.load_state_dict(checkpoint['state_dict'])

        for word, sens in tqdm(train_word2sens.items(), ncols=10):
            sens = [util.pad_sentence(s, model_args.pad_size, def_word2ix['</s>']) for s in sens]
            sens = {tuple(s) for s in sens}
            sens = [list(s) for s in sens]
            numsens = len(sens)
            if numsens == 1:
                new_train_pairs.append((word, sens[0], 1))
                continue
            target_emb = dic_embed[word].cuda().view(1, -1)
            target_emb = target_emb.expand(numsens, dic_embed.size(1))
            out_embs = encoder.estimate_from_defsens(sens)
            weights = get_weights(target_emb[0].cpu(), out_embs.cpu().data, args.lamb, args.norm)
            for i, sen in enumerate(sens):
                new_train_pairs.append((word, sen, weights[i]))
    else:
        for word, sens in train_word2sens.items():
            weight = 1.0 / len(sens)
            for sen in sens:
                new_train_pairs.append((word, sen, weight))

    # dump to pickle
    # outpath = '../data/noraset/rank_{}'.format(basename(args.data))
    with open(args.output, 'wb') as wf:
        data = {
            'dic_embed': dic_embed.cpu(),
            'def_embed': def_embed.cpu(),
            'dic_word2ix': dic_word2ix,
            'def_word2ix': def_word2ix,
            'valid_pairs': valid_pairs,
            'train_pairs': new_train_pairs,
            'rvd_candidates': rvd_candidates,
            'rvd_pairs_seen': rvd_pairs_seen,
            'rvd_pairs_unseen': rvd_pairs_unseen,
            'rvd_pairs_concept': rvd_pairs_concept,
        }
        pickle.dump(data, wf)
