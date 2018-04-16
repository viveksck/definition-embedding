from tqdm import tqdm
from os.path import basename, join
from model.base import BaseEncoder
from model.util import pad_sentence

import torch.nn as nn
from torch import LongTensor, FloatTensor
from torch.autograd import Variable


class RNNEncoder(BaseEncoder):
    def __init__(self, def_word2ix, args):
        super(RNNEncoder, self).__init__(def_word2ix, args)

        # input parameters
        self.hid_dim = args.hid_dim
        self.num_layers = args.num_layers
        self.drop_ratio = args.drop_ratio

        # pipeline
        rnn = nn.GRU if args.rnn == 'gru' else nn.LSTM
        self.rnn = rnn(
            bidirectional=False,
            dropout=self.drop_ratio,
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.num_layers,
        )
        self.dropout = nn.Dropout(p=self.drop_ratio)
        self.emb_dropout = nn.Dropout(p=args.emb_drop)
        self.hidden2embed = nn.Linear(self.hid_dim, self.emb_dim)

        # checkpoint and loss record file
        self.cp_path = '../checkpoint/en_{data}_h{hid}_p{pad}_l{layer}_d{drop}_b{batch}_{rnn}_{opt}_lr{lr}'.format(
            emb=args.emb_dim,
            hid=args.hid_dim,
            data=basename(args.data).replace('.pk', ''),
            layer=args.num_layers,
            drop='{}-{}'.format(args.emb_drop, args.drop_ratio),
            rnn=args.rnn,
            opt=args.optim,
            batch=args.batch_size,
            pad=args.pad_size,
            lr=args.lr
        )
        if not args.fine_tune:
            self.cp_path += '_fix'

    def forward(self, def_sens):
        '''
        Arguments:
            def_sens (batch_size, pad_size) Variable
        Returns:
            (batch_size, emb_dim)
        '''
        hid_layer = self.forward_before_linear(def_sens)
        return self.hidden2embed(hid_layer)

    def forward_before_linear(self, def_sens):
        '''
        Arguments:
            def_sens (batch_size, pad_size) Variable
        Returns:
            (batch_size, emb_dim)
        '''
        in_embeds = self.embed(def_sens)
        in_embeds = self.emb_dropout(in_embeds)

        lstm_out, _ = self.rnn(in_embeds.transpose(0, 1))
        lstm_out = self.dropout(lstm_out)
        return lstm_out[-1]


