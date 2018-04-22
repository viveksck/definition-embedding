import re
import torch
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, Manager
from os.path import basename, join, dirname, abspath


if __name__ == '__main__':
    import sys
    sys.path.append(abspath('../src'))
    import model.util as util
    from model.rnn import RNNEncoder

    parser = argparse.ArgumentParser('dictionary generation')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
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
    rvd_candidates = set(data['rvd_candidates'])
    print('[1] load data OK:', model_args.data)

    # load model
    model_args.gpu = args.gpu
    encoder = RNNEncoder(def_word2ix, model_args)
    if args.gpu > -1:
        encoder = encoder.cuda()
    encoder.load_state_dict(checkpoint['state_dict'])
    print('[2] load model OK!')

    # load definition data
    with open(args.data, 'rb') as rf:
        word2strsen = pickle.load(rf)
    word2emb = dict()
    for word, strsen in tqdm(word2strsen.items(), ncols=10):
        strsen = strsen.replace('-', ' ')
        strsen = re.sub(' +', ' ', strsen)
        strsen = ' '.join(word_tokenize(strsen))
        word2emb[word] = encoder.estimate_from_strsens(strsen)[0]

    with open('output.pk', 'wb') as wf:
        pickle.dump(word2emb, wf)












