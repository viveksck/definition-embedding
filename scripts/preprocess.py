import torch
import random
import pickle
import argparse
import numpy as np
from collections import Counter
from collections import defaultdict

"""
args:
    --voc_size: voc_size of no 'unk' words
    --emb_file: pickle file of word2emb dict
    --dic_file: dic_word \t def_sentence (tokenized)
    --out_file: output pickle file path
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser('data preprocess')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('-r', '--rvd_dir', type=str, default='')
    parser.add_argument('-v', '--voc_size', type=int, default=10000)
    parser.add_argument('-e', '--emb_file', type=str, required=True)
    parser.add_argument('-d', '--dic_file', type=str, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)
    args = parser.parse_args()

    """ load pretrained embedding from pickle """
    with open(args.emb_file, 'rb') as rf:
        word2emb = pickle.load(rf)

    """ encode training dictionary file """
    word_counter = Counter()
    word2defsens = defaultdict(list)

    with open(args.dic_file) as rf:
        for line in rf:
            try:
                dic_word, def_sent = line.rstrip().split('\t')
            except:
                continue
            # exclude oov dic_word        
            if dic_word not in word2emb:
                continue
            
            # count def_word in vocabulary
            def_words = def_sent.split()
            for w in def_words:
                if w in word2emb:
                    word_counter[w] += 1

            # add to word2defsens
            word2defsens[dic_word].append(def_words)

    # decide def vocabulary based on frequency
    def_comm = word_counter.most_common(args.voc_size)
    def_vocb = {w for w, _ in def_comm}
    print('min frequency of def word:', def_comm[-1][1])

    # encode def_word and dic_word, retrieve embedding
    def_embed = []
    dic_embed = []
    def_word2ix = dict()
    dic_word2ix = dict()
    for i, w in enumerate(def_vocb):
        def_word2ix[w] = i
        def_embed.append(word2emb[w])
    for i, w in enumerate(word2defsens.keys()):
        dic_word2ix[w] = i
        dic_embed.append(word2emb[w])
    def_word2ix['<s>'] = len(def_word2ix)
    def_word2ix['</s>'] = len(def_word2ix)
    def_word2ix['<unk>'] = len(def_word2ix)
    emb_dim = len(def_embed[0])
    def_embed += np.random.rand(3, emb_dim).tolist()
    def_embed = torch.FloatTensor(def_embed)
    dic_embed = torch.FloatTensor(dic_embed)
    assert def_embed.shape[0] == len(def_word2ix)
    print('encode def/dic_word ok')

    # shuffle dict_words, encode (dic_word, def_words) pairs
    pairs = list()
    dic_words = list(word2defsens.keys())
    random.shuffle(dic_words)
    for dic_word in dic_words:
        def_sens = word2defsens[dic_word]

        # encode dic_words
        i_dic_word = dic_word2ix[dic_word]
        
        for def_words in def_sens:
            # replace oov def_word with <unk>, add <s> </s>
            def_words = [w if w in def_word2ix else '<unk>' for w in def_words]
            def_words = ['<s>'] + def_words + ['</s>']
            
            # encode def_words
            i_def_words = [def_word2ix[w] for w in def_words]

            pairs.append((i_dic_word, i_def_words))

    # split train, valid
    num_valid = int(len(pairs) * args.valid_ratio)
    valid_pairs = pairs[:num_valid]
    train_pairs = pairs[num_valid:]
    print('split train/valid pairs ok')

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
                'valid_pairs': valid_pairs
            }, wf
        )


    