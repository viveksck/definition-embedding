import torch
import random
from collections import defaultdict
from model.util import pad_sentence
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


def get_padded_dataset(word_sen_pairs, dic_embed, pad_size, ixpad, batch_size, is_train):
    class PaddedDataset(Dataset):
        def __init__(self, pairs):
            self.len = len(pairs)
            self.dic_embed = dic_embed
            self.pairs = [(word, torch.LongTensor(sen)) for word, sen in pairs]

        def __getitem__(self, index):
            word, padsen = self.pairs[index]
            return dic_embed[word], padsen

        def __len__(self):
            return self.len

    pairs = [(word, pad_sentence(sen, pad_size, ixpad)) for word, sen in word_sen_pairs]
    dataset = PaddedDataset(pairs)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)


def get_parameter_batches(word_sen_pairs, dic_embed, pad_size, ixpad, batch_size, use_gpu):
    weights = torch.randn(len(word_sen_pairs))
    if use_gpu:
        weights = weights.cuda()
    weights = torch.nn.parameter.Parameter(weights)
    padded_pairs = [(word, pad_sentence(sen, pad_size, ixpad)) for word, sen in word_sen_pairs]

    i = 0
    batches = []
    random.shuffle(padded_pairs)
    num_pairs = len(padded_pairs)
    while i < num_pairs:
        batch_pairs = padded_pairs[i: i + batch_size]
        batch_words = [w for w, _ in batch_pairs]

        batch_grdemb = dic_embed[batch_words]
        batch_sens = [s for _, s in batch_pairs]
        batch_weights = weights[i: i + batch_size]

        batches.append((batch_grdemb, torch.LongTensor(batch_sens), batch_weights))
        i += batch_size
    return batches, weights


def get_bow_dataset(word_sen_pairs, dic_embed, def_embed, def_word2ix, batch_size):
    i = 0
    batches = []
    stop_words = ['<s>', '</s>', '<unk>', 'a', 'of', 'the', ',', '.']
    stop_words = {def_word2ix[w] for w in stop_words if w in def_word2ix}
    random.shuffle(word_sen_pairs)
    num_pairs = len(word_sen_pairs)
    while i < num_pairs:
        batch_pairs = word_sen_pairs[i: i + batch_size]
        batch_words = []
        batch_senemb = []
        for word, sen in batch_pairs:
            sen = sen[1: -1]
            sen = [w for w in sen if w not in stop_words]
            if not sen:
                continue
            sen_mat = def_embed[sen]
            sen_emb = sen_mat.sum(0) / len(sen)
            batch_words.append(word)
            batch_senemb.append(sen_emb)
        batch_grdemb = dic_embed[batch_words]
        batch_senemb = torch.stack(batch_senemb)
        batches.append((batch_grdemb, batch_senemb))
        i += batch_size
    return batches


def get_word_batches(word_sen_pairs, dic_emb, pad_size, ixpad, batch_size, use_gpu):
    weights = torch.randn(len(word_sen_pairs))
    if use_gpu:
        weights = weights.cuda()
    weights = torch.nn.parameter.Parameter(weights)

    # get word2padsens
    word2padsens = defaultdict(list)
    for word, sen in word_sen_pairs:
        padsen = pad_sentence(sen, pad_size, ixpad)
        word2padsens[word].append(padsen)
    word2padsens = dict(word2padsens)
    
    # shuffle word list
    words = list(word2padsens.keys())
    random.shuffle(words)
    num_words = len(words)

    # assign to batches
    i = 0
    j = 0
    batches = []
    while i < num_words:
        batch_sens = []
        batch_grdembs = []
        batch_sen_nums = []
        batch_vocab = words[i: i + batch_size]
        for word in batch_vocab:
            padsens = word2padsens[word]
            sen_num = len(padsens)
            batch_sens += padsens
            batch_grdembs += [dic_emb[word]] * sen_num
            batch_sen_nums.append(sen_num)
        batch_sens = torch.LongTensor(batch_sens)
        batch_grdembs = torch.stack(batch_grdembs)
        '''
        batch_grdembs [list] num_sens * emb_dim
        batch_sens    [LongTensor] num_sens * pad_size
        batch_sen_nums [list] num_worsd
        '''
        assert sum(batch_sen_nums) == batch_sens.size(0)
        assert batch_grdembs.size(0) == batch_sens.size(0)

        batch_weights = weights[j: j + batch_sens.size(0)]
        batches.append((batch_grdembs, batch_sens, batch_sen_nums, batch_weights))
        i += batch_size
        j += batch_sens.size(0)
    return batches, weights


def get_weighted_batches(word_sen_weight_pairs, dic_embed, pad_size, ixpad, batch_size):
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
        batch_grdemb = dic_embed[batch_words]
        batch_sens = torch.LongTensor(batch_sens)
        batch_weis = Variable(torch.FloatTensor(batch_weis))
        assert batch_weis.size(0) == batch_sens.size(0) == batch_grdemb.size(0) == sum(batch_sen_nums)
        batches.append((batch_grdemb, batch_sens, batch_sen_nums, batch_weis))
        i += batch_size
    return batches


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