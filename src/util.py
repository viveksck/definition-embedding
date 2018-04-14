import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from collections import defaultdict


def getval(variable):
    return variable.data.storage()[0]


def encode_sentence(sen, word2ix):
    return [word2ix[w] if w in word2ix else word2ix['<unk>'] for w in sen]


def pad_sentence(sen, pad_size, ixpad):
    padsen = sen[:]
    padsen += [ixpad] * (pad_size - len(padsen))
    return padsen[:pad_size]


def get_dataset(word_sen_pairs, dic_embed, pad_size, ixend, is_train):
    class Pairs(Dataset):
        def __init__(self, pairs):
            self.len = len(pairs)
            self.dic_embed = dic_embed
            self.pairs = [(word, torch.LongTensor(sen)) for word, sen in pairs]

        def __getitem__(self, index):
            word, padsen = self.pairs[index]
            return dic_embed[word], padsen

        def __len__(self):
            return self.len

    pairs = [(word, pad_sentence(sen, pad_size, ixend)) for word, sen in word_sen_pairs]
    return Pairs(pairs)


def get_batches(word_sen_weight_pairs, pad_size, ixpad, batch_size):  
    import torch
    import random

    # get word2padsens
    word2padsen_weights = defaultdict(list)
    for word, sen, weight in word_sen_weight_pairs:
        padsen = pad_sentence(sen, pad_size, ixpad)
        word2padsen_weights[word].append((padsen, weight))
    word2padsen_weights = dict(word2padsen_weights)
    
    # shuffle word list
    words = list(word2padsen_weights.keys())
    random.shuffle(words)
    num_words = len(words)

    # assign to batches
    i = 0
    batches = []
    while i < num_words:
        batch_sens = []
        batch_weis = []
        batch_words = []
        batch_sen_nums = []
        batch_vocab = words[i: i + batch_size]
        for word in batch_vocab:
            padsen_weights = word2padsen_weights[word]
            sen_num = len(padsen_weights)
            batch_sens += [sen for sen, wei in padsen_weights]
            batch_weis += [wei for sen, wei in padsen_weights]
            batch_words += [word] * sen_num 
            batch_sen_nums.append(sen_num)
        batch_sens = torch.LongTensor(batch_sens)
        batch_weis = Variable(torch.FloatTensor(batch_weis))
        batch_words = torch.LongTensor(batch_words)
        batches.append((batch_words, batch_sens, batch_sen_nums, batch_weis))
        i += batch_size
    return batches


def topk_pairs(pairs, k):
    w2c = defaultdict(int)
    new_pairs = []
    for w, s in pairs:
        if w2c[w] >= k:
            continue
        w2c[w] += 1
        new_pairs.append((w, s))
    return new_pairs


def mkdir(dirpath):
    import os
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


if __name__ == '__main__':
    word_sen_pairs = [
        [1, [20, 30, 40]],
        [2, [12, 13, 14]],
        [3, [22, 23, 24]],
        [4, [32, 33, 34]],
        [5, [42, 43, 44]],
    ]
    batches = get_batches(word_sen_pairs, 10, 11, 2)
    for words, sens in batches:
        print('words', words, 'sens', sens)


















