import numpy as np
from collections import defaultdict


def getval(variable):
    return variable.data.storage()[0]


def encode_sentence(sen, word2ix):
    return [word2ix[w] if w in word2ix else word2ix['<unk>'] for w in sen]


def pad_sentence(sen, pad_size, ixpad):
    padsen = sen[:]
    padsen += [ixpad] * (pad_size - len(padsen))
    return padsen[:pad_size]


def topk_pairs(pairs, k):
    w2c = defaultdict(int)
    new_pairs = []
    for w, s in pairs:
        if w2c[w] >= k:
            continue
        w2c[w] += 1
        new_pairs.append((w, s))
    return new_pairs


def normalize_matrix_by_row(np_mat):
    from numpy import linalg
    row_norm = linalg.norm(np_mat, axis=1)
    return np_mat / row_norm.reshape(-1, 1)


def mkdir(dirpath):
    import os
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


if __name__ == '__main__':
    pass


















