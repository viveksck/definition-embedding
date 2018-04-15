import torch
import random
import pickle
import argparse
import numpy as np
from os.path import join
from functools import reduce
from collections import Counter
from collections import defaultdict

"""
args:
    --voc_size: voc_size of no 'unk' words
    --emb_file: pickle file of word2emb dict
    --dic_file: dic_word \t def_sentence (tokenized)
    --out_file: output pickle file path
"""


def encode_dict_file(file_path, dic_word2ix, def_word2ix):
    pairs = list()
    with open(file_path) as rf:
        for line in rf:
            try:
                dic_word, def_sent = line.rstrip().split('\t')
            except ValueError:
                continue

            # exclude oov dic_word
            if dic_word not in dic_word2ix:
                continue

            # encode dic_words
            i_dic_word = dic_word2ix[dic_word]
            
            # tokenize definition sentence
            def_words = def_sent.split()
            def_words = [w if w in def_word2ix else '<unk>' for w in def_words]
            def_words = ['<s>'] + def_words + ['</s>']
            
            # encode def_words
            i_def_words = [def_word2ix[w] for w in def_words]

            pairs.append((i_dic_word, i_def_words))
    print('----encode {} lines: {}'.format(len(pairs), file_path))
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data preprocess')
    parser.add_argument('--valid_ratio', type=float, default=0.05)
    parser.add_argument('-r', '--rvd_dir', type=str, required=True)
    parser.add_argument('-v', '--voc_size', type=int, default=10000)
    parser.add_argument('-e', '--emb_file', type=str, required=True)
    parser.add_argument('-d', '--dic_file', type=str, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)
    args = parser.parse_args()

    """ load pretrained embedding from pickle """
    with open(args.emb_file, 'rb') as rf:
        word2emb = pickle.load(rf)

    """ decide word2ix """

    # count def_words in training file
    dic_vocab = set()
    with open(join(args.rvd_dir, 'rvd_candidates.txt')) as rf:
        rvd_candidates = rf.read().splitlines()
        rvd_candidates = {w for w in rvd_candidates if w in word2emb}
    dic_vocab |= rvd_candidates

    word_counter = Counter()
    with open(args.dic_file) as rf:
        for line in rf:
            try:
                dic_word, def_sent = line.rstrip().split('\t')
            except ValueError:
                continue

            # exclude oov dic_word        
            if dic_word not in word2emb:
                continue
            
            # add to dic_vocab
            dic_vocab.add(dic_word)

            # count def_word in vocabulary
            def_words = def_sent.split()
            for w in def_words:
                if w in word2emb:
                    word_counter[w] += 1
    print('[0] # dic_vocab:', len(dic_vocab))

    # decide def vocabulary based on frequency
    def_comm = word_counter.most_common(args.voc_size)
    def_vocb = {w for w, _ in def_comm}
    print('[1] min_freq(def_word):', def_comm[-1][1])

    # encode def_word, retrieve embedding
    def_embed = []
    def_word2ix = dict()
    for i, w in enumerate(def_vocb):
        def_word2ix[w] = i
        def_embed.append(word2emb[w])
    print('[2] def_embed OK!')

    # encode dic_word, retrieve embedding
    dic_embed = []
    dic_word2ix = dict()
    for i, w in enumerate(dic_vocab):
        dic_word2ix[w] = i
        dic_embed.append(word2emb[w])
    print('[3] dic_embed OK!')

    # add special tokens and embeddings to def_word
    def_word2ix['<s>'] = len(def_word2ix)
    def_word2ix['</s>'] = len(def_word2ix)
    def_word2ix['<unk>'] = len(def_word2ix)
    emb_dim = len(def_embed[0])
    def_embed += np.random.rand(3, emb_dim).tolist()
    print('[4] special token OK!')

    # store def/dic_embed as torch.FloatTensor
    def_embed = torch.FloatTensor(def_embed)
    dic_embed = torch.FloatTensor(dic_embed)
    assert def_embed.shape[0] == len(def_word2ix)

    # encode training file
    train_pairs_all = encode_dict_file(args.dic_file, dic_word2ix, def_word2ix)

    # split train, valid
    word2pairs = defaultdict(list)
    for p in train_pairs_all:
        word2pairs[p[0]].append(p)
    dic_words = list(word2pairs.keys())
    random.shuffle(dic_words)

    num_valid = int(len(dic_words) * args.valid_ratio)
    valid_words = dic_words[:num_valid]
    train_words = dic_words[num_valid:]
    valid_pairs = []
    train_pairs = []
    for word in valid_words:
        valid_pairs += word2pairs[word]
    for word in train_words:
        train_pairs += word2pairs[word]
    print('[5] split {} train, {} valid'.format(len(train_pairs), len(valid_pairs)))

    # encode reverse dictionary data
    rvd_path_seen = join(args.rvd_dir, 'rvd_seen.txt')
    rvd_path_unseen = join(args.rvd_dir, 'rvd_unseen.txt')
    rvd_path_concept = join(args.rvd_dir, 'rvd_concept.txt')
    rvd_pairs_seen = encode_dict_file(rvd_path_seen, dic_word2ix, def_word2ix)
    rvd_pairs_unseen = encode_dict_file(rvd_path_unseen, dic_word2ix, def_word2ix)
    rvd_pairs_concept = encode_dict_file(rvd_path_concept, dic_word2ix, def_word2ix)

    rvd_candidates = [dic_word2ix[w] for w in rvd_candidates]
    print('[6] encode rvd data OK!')

    # output
    with open(args.out_file, 'wb') as wf:
        pickle.dump(
            {
                'args': vars(args),
                'dic_embed': dic_embed,
                'def_embed': def_embed,
                'dic_word2ix': dic_word2ix,
                'def_word2ix': def_word2ix,
                'train_pairs': train_pairs,
                'valid_pairs': valid_pairs,
                'rvd_candidates': rvd_candidates,
                'rvd_pairs_seen': rvd_pairs_seen,
                'rvd_pairs_unseen': rvd_pairs_unseen,
                'rvd_pairs_concept': rvd_pairs_concept,
            }, wf
        )
    print('[7] preprocessing OK!')


    