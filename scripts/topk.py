import torch
import random
import pickle
import argparse
from tqdm import tqdm
from os.path import abspath, basename, dirname, join
from multiprocessing import Pool, Manager
from collections import defaultdict


if __name__ == '__main__':
    import sys
    sys.path.append(abspath('../src'))
    import model.util as util
    from model.rnn import RNNEncoder
    from model.bow import BOWEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-k', '--topk', type=float, default=0.5)
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    args = parser.parse_args()

    # load checkpoint
    print('[0] loading checkpoint:', args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

    # load data
    model_args = checkpoint['args']
    with open(model_args.data, 'rb') as rf:
        data = pickle.load(rf)
    dic_embed = data['dic_embed']
    def_embed = data['def_embed']
    dic_word2ix = data['dic_word2ix']
    def_word2ix = data['def_word2ix']
    train_pairs = data['train_pairs']
    print('[1] load data OK:', model_args.data)

    # load model
    model_args.gpu = args.gpu
    encoder = RNNEncoder(def_word2ix, model_args)
    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)
        encoder = encoder.cuda()
    encoder.load_state_dict(checkpoint['state_dict'])
    print('[2] load model OK!')

    # get word2sens
    word2sens = defaultdict(list)
    for word, sen in train_pairs:
        word2sens[word].append(sen)
    word2sens = dict(word2sens)

    # get topk
    new_train_pairs = list() 
    new_word2sens = defaultdict(list)
    dic_embed = util.normalize_matrix_by_row(dic_embed.numpy())

    def rank_word(word):
        global new_train_pairs
        sens = word2sens[word]
        topk = max(1, int(len(sens) * args.topk))
        out_embs = encoder.estimate_from_idxsens(sens, normalize=True)
        grd_embs = dic_embed[word].reshape(1, -1)
        simi_matrix = out_embs.dot(grd_embs.transpose())
        assert simi_matrix.shape == (len(sens), 1)
        simi_matrix = simi_matrix.reshape(-1)
        rank_indices = sorted(range(simi_matrix.shape[0]), key=lambda i: simi_matrix[i], reverse=True)
        new_train_pairs += [(word, sens[i]) for i in rank_indices[:topk]]
        new_word2sens[word] += [(sens[i], simi_matrix[i]) for i in rank_indices]

    for word in tqdm(word2sens.keys(), ncols=10):
        rank_word(word)
    print('[3] rank OK!')

    random.shuffle(new_train_pairs)

    # dump
    data['train_pairs'] = new_train_pairs
    origin_path = model_args.data
    origin_dir = dirname(origin_path)
    output_path = join(origin_dir, 'top{}-{}'.format(args.topk, basename(origin_path)))
    with open(output_path, 'wb') as wf:
        pickle.dump(data, wf)

    with open('top{}-word2sens.pk'.format(args.topk), 'wb') as wf:
        data = {
            'dic_word2ix': dic_word2ix,
            'def_word2ix': def_word2ix,
            'word2sens': new_word2sens
        }
        pickle.dump(data, wf)

















