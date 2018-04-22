from tqdm import tqdm
from os.path import basename, join
from model.base import BaseEncoder
from model.util import pad_sentence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, FloatTensor
from torch.autograd import Variable


class BOWEncoder(BaseEncoder):
    def __init__(self, def_word2ix, args):
        super(BOWEncoder, self).__init__(def_word2ix, args)

        # input parameters
        self.drop_ratio = args.drop_ratio

        # pipeline
        self.linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.emb_dropout = nn.Dropout(p=args.emb_drop)

        # checkpoint and loss record file
        self.cp_path = '../checkpoint/bow_{data}_d{drop}_b{batch}_{opt}_lr{lr}'.format(
            data=basename(args.data).replace('.pk', ''),
            drop='{}'.format(args.emb_drop),
            opt=args.optim,
            batch=args.batch_size,
            lr=args.lr
        )
        if not args.fine_tune:
            self.cp_path += '_fix'

    def forward(self, sen_emb):
        '''
        Arguments:
            sen_emb (batch_size, emb_dim) Variable
        Returns:
            (batch_size, emb_dim)
        '''
        out = self.linear(F.relu(sen_emb))
        return out

    def estimate_from_defsens(self, def_sens):
        senembs = []
        stop_words = ['<s>', '</s>', '<unk>', 'a', 'of', 'the', ',', '.']
        stop_words = {self.def_word2ix[w] for w in stop_words if w in self.def_word2ix}
        for sen in def_sens:
            sen = [w for w in sen if w not in stop_words]
            if not sen:
                sen = [self.def_word2ix['<unk>']]
            senemb = self.embed(Variable(torch.LongTensor(sen))).sum(0) / len(sen)
            senembs.append(senemb)
        senembs = torch.stack(senembs)

        # go through the encoder
        self.eval()
        if self.use_gpu:
            senembs = senembs.cuda()
        out_embs = self(senembs)
        return out_embs

