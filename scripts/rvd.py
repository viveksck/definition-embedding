import torch
import pickle
import argparse
from os.path import basename, join, dirname, abspath
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from multiprocessing import Pool, Value, Lock, Manager

if __name__ == '__main__':
    import sys
    sys.path.append(abspath('../src'))
    import util
    import model

    parser = argparse.ArgumentParser('dictionary generation')
    parser.add_argument('-g', '--gpu', type=int, help='gpu device')
    parser.add_argument('-p', '--process', type=int, default=20, help='process num')
    parser.add_argument('-t', '--test', choices=['test_pairs', 'test_pairs_seen', 'test_pairs_unseen'], default='test_pairs_unseen', help='test key in the data dump')
    parser.add_argument('-c', '--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('-l', '--shortlist', type=str, default='../data/rvd/output_shortlist.pk', help='path to shortlist pickle')
    myargs = parser.parse_args()

    print('loading checkpoint:', myargs.checkpoint)
    checkpoint = torch.load(myargs.checkpoint, map_location=lambda storage, loc: storage)
    args = checkpoint['args']
    if 'temp' not in args:
        args.temp = 1.0

    print('loading data:', args.data)
    with open(args.data, 'rb') as rf:
        data = pickle.load(rf)
    print('loading data: OK!')
    dic_embed = data['dic_embed']
    def_embed = data['def_embed']
    dic_word2ix = data['dic_word2ix']
    def_word2ix = data['def_word2ix']
    test_pairs = data[myargs.test]

    torch.cuda.set_device(myargs.gpu)
    dic_embed.requires_grad = False
    args.emb_dim = dic_embed.weight.size(1)

    print('initialize model')
    encoder = model.RNNEncoder(dic_word2ix, def_word2ix, dic_embed, def_embed, args).cuda()
    encoder.load_state_dict(checkpoint['state_dict'])

    with open(myargs.shortlist, 'rb') as rf:
        word2emb_shortlist = pickle.load(rf)

    word2sens = defaultdict(list)
    for word, sen in test_pairs:
        word2sens[word].append(sen)

    dic_embed = encoder.dic_embed.cpu().weight.data.numpy()
    word2sen = dict()
    for word, sen in test_pairs:
        if word not in word2sen:
            word2sen[word] = sen

    sens = []
    words = []
    for word, sen in word2sen.items():
        sens.append(sen)
        words.append(word)
    grd_embs = [dic_embed[w] for w in words]
    out_embs = encoder.estimate_from_defsens(sens).cpu().data.numpy()
    assert len(grd_embs) == out_embs.shape[0]

    acc_1 = 0.0
    acc_10 = 0.0
    acc_100 = 0.0

    outdir = '../rvd_out'
    util.mkdir(outdir)
    outpath = join(outdir, basename(encoder.cp_path) + '.txt')
    wf = open(outpath, 'w')
    shortlist = word2emb_shortlist.keys()
    dic_ix2word = {v: k for k, v in encoder.dic_word2ix.items()}
    def_ix2word = {v: k for k, v in encoder.def_word2ix.items()}
    for w in words:
        word2emb_shortlist[dic_ix2word[w]] = dic_embed[w]

    poses = Manager().list()

    def test(i):
        out_emb = out_embs[i]
        rank = sorted(shortlist, key=lambda w: cosine(out_emb, word2emb_shortlist[w]))
        pos = rank.index(dic_ix2word[words[i]])
        poses.append(pos)

    pool = Pool(processes=myargs.process)
    pool.map(test, list(range(out_embs.shape[0])))
    acc_1 = acc_10 = acc_100 = 0.0
    print('# samples:', len(poses))
    for pos in poses:
        acc_1 += float(pos == 0)
        acc_10 += float(pos < 10)
        acc_100 += float(pos < 100)
    acc_1 /= out_embs.shape[0]
    acc_10 /= out_embs.shape[0]
    acc_100 /= out_embs.shape[0]
    print('acc@1:', acc_1)
    print('acc@10:', acc_10)
    print('acc@100:', acc_100)

    med_pos = sorted(poses)[int(out_embs.shape[0] / 2)]
    print('med_pos:', med_pos)
    print('var(pos):', np.var(poses))