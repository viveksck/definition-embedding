import torch
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, Manager
from os.path import basename, join, dirname, abspath


if __name__ == '__main__':
    import sys
    sys.path.append(abspath('../src'))
    import model.util as util
    from model.rnn import RNNEncoder
    from model.bow import BOWEncoder

    parser = argparse.ArgumentParser('dictionary generation')
    parser.add_argument('-p', '--process', type=int, default=20)
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    parser.add_argument('-t', '--test', choices=['rvd_pairs_seen', 'rvd_pairs_unseen', 'rvd_pairs_concept'], default='rvd_pairs_unseen')
    args = parser.parse_args()

    # load checkpoint
    print('[0] loading checkpoint:', args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    
    # load data
    model_args = checkpoint['args']
    with open(model_args.data, 'rb') as rf:
        data = pickle.load(rf)
    rvd_pairs = data[args.test]
    dic_embed = data['dic_embed']
    def_embed = data['def_embed']
    dic_word2ix = data['dic_word2ix']
    def_word2ix = data['def_word2ix']
    rvd_candidates = set(data['rvd_candidates'])
    print('#candi:', len(rvd_candidates))
    print('#pairs:', len(rvd_pairs))
    rvd_pairs = [(w, s) for w, s in rvd_pairs if w in rvd_candidates]
    print('#pairs:', len(rvd_pairs))
    print('[1] load data OK:', model_args.data, len(rvd_pairs))

    # load model
    model_args.gpu = -1
    encoder = {
        'gru': RNNEncoder(def_word2ix, model_args),
        'bow': BOWEncoder(def_word2ix, model_args)
    }[model_args.rnn]
    encoder.load_state_dict(checkpoint['state_dict'])
    print('[2] load model OK!')

    # load ground embed; calculate output embed
    sens = [s for _, s in rvd_pairs]
    grd_words = [w for w, _ in rvd_pairs]
    out_embs = encoder.estimate_from_defsens(sens).data.numpy()
    out_embs = util.normalize_matrix_by_row(out_embs)
    print('[3] calculate embedding OK!')

    # output file
    outdir = '../output'
    util.mkdir(outdir)
    outpath = join(outdir, basename(encoder.cp_path) + '.txt')
    wf = open(outpath, 'w')

    # evaluation starts
    rvd_candidates = list(rvd_candidates)
    rvd_word2ix = {w: i for i, w in enumerate(rvd_candidates)}

    rvd_candi_embs = dic_embed[rvd_candidates].numpy()
    rvd_norm_candi_embs = util.normalize_matrix_by_row(rvd_candi_embs)
    print('[4] normalize rvd embeddings OK!')

    rvd_candi_ranges = range(len(rvd_candidates))
    simi_matrix = out_embs.dot(rvd_norm_candi_embs.transpose())
    print('[5] calculate similarity in batch OK!')

    # multi process ranking
    poses = Manager().list()

    def single_sort(i):
        rank = sorted(range(len(rvd_candidates)), key=lambda j: simi_matrix[i][j], reverse=True)
        poses.append(rank.index(rvd_word2ix[grd_words[i]]))

    pool = Pool(processes=args.process)
    pool.map(single_sort, list(range(out_embs.shape[0])))
    print('[6] ranking in batch OK!')
    print('----# samples:', len(poses))

    # calculate accuracy score
    acc_1 = acc_10 = acc_100 = 0.0
    for pos in poses:
        acc_1 += float(pos == 0)
        acc_10 += float(pos < 10)
        acc_100 += float(pos < 100)
    acc_1 /= out_embs.shape[0]
    acc_10 /= out_embs.shape[0]
    acc_100 /= out_embs.shape[0]
    med_pos = sorted(poses)[int(out_embs.shape[0] / 2)]
    print('acc@1:', acc_1)
    print('acc@10:', acc_10)
    print('acc@100:', acc_100)
    print('med_pos:', med_pos)