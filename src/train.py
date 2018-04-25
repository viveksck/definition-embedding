import torch
import torch.optim as optim

import random
import pickle
import dataset
import argparse
import model.util as util

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from model.rnn import RNNEncoder
from model.bow import BOWEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser('dictionary generation')
    # configuration
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--load', type=str, default='', help='load checkpoint')
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='output checkpoint')
    # network parameter
    parser.add_argument('-r', '--rnn', choices=['lstm', 'gru', 'bow'], default='gru', help='rnn')
    parser.add_argument('-i', '--hid_dim', type=int, default=512, help='hidden layer dimension')
    parser.add_argument('-p', '--pad_size', type=int, default=20, help='padding_size')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-d', '--drop_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--emb_drop', type=float, default=0.0, help='embedding dropout rate')
    parser.add_argument('-l', '--num_layers', type=int, default=2, help='number of lstm layers')
    # training setting
    parser.add_argument('-s', '--num_train', type=int, default=-1, help='number of training pairs used. -1 means all data.')
    parser.add_argument('-f', '--fine_tune', action='store_false', help='whether to fine tune word embedding')
    parser.add_argument('-o', '--optim', choices=['SGD', 'Adam', 'Adadelta'], default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patience', type=int, default=4, help='patience for early stop')
    args = parser.parse_args()

    # set GPU device
    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    # load checkpoint
    if args.load:
        print('loading checkpoint:', args.load)
        checkpoint = torch.load(args.load, map_location=lambda storage, loc: storage)
        ckargs = checkpoint['args']
        args.rnn = ckargs.rnn
        # args.data = ckargs.data
        args.hid_dim = ckargs.hid_dim
        args.pad_size = ckargs.pad_size
        args.num_train = ckargs.num_train
        args.fine_tune = ckargs.fine_tune
        # args.batch_size = ckargs.batch_size
        args.drop_ratio = ckargs.drop_ratio
        args.num_layers = ckargs.num_layers
    pprint(vars(args))

    # load data
    print('loading data from:', args.data)
    with open(args.data, 'rb') as rf:
        data = pickle.load(rf)

    dic_embed = data['dic_embed']
    def_embed = data['def_embed']
    dic_word2ix = data['dic_word2ix']
    def_word2ix = data['def_word2ix']
    train_pairs = data['train_pairs']
    valid_pairs = data['valid_pairs']
    args.emb_dim = dic_embed.size(1)
    assert args.emb_dim == dic_embed.size(1)
    if args.gpu > -1:
        def_embed = def_embed.cuda()
        dic_embed = dic_embed.cuda()

    if args.num_train > 0:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:args.num_train]
        valid_pairs = valid_pairs[:args.num_train]
    num_train_pairs = len(train_pairs)
    num_train_vocab = len({word for word, _ in train_pairs})

    train_grd_embed = [dic_embed[w] for w, _ in train_pairs]
    valid_grd_embed = [dic_embed[w] for w, _ in valid_pairs]

    # init model and dataset
    if args.rnn == 'bow':
        encoder = BOWEncoder(def_word2ix, args)
        validset = dataset.get_bow_dataset(valid_pairs, dic_embed, def_embed, def_word2ix, args.batch_size)
        trainset = dataset.get_bow_dataset(train_pairs, dic_embed, def_embed, def_word2ix, args.batch_size)
    elif args.rnn == 'gru' or args.rnn == 'lstm':
        encoder = RNNEncoder(def_word2ix, args)
        trainset = dataset.get_padded_dataset(train_pairs, dic_embed, args.pad_size, def_word2ix, args.batch_size, is_train=True)
        validset = dataset.get_padded_dataset(valid_pairs, dic_embed, args.pad_size, def_word2ix, args.batch_size, is_train=False)
    encoder.init_def_embedding(def_embed)
    print('load data: {} train, {} valid'.format(len(train_pairs), len(valid_pairs)))

    # optimizer
    lr = args.lr
    params = list(filter(lambda t: t.requires_grad, encoder.parameters()))
    print('# params:', len(params))
    opts = {
        'Adam': optim.Adam(params, lr=lr),
        'Adadelta': optim.Adadelta(params, lr=1),
        'SGD': optim.SGD(params, lr=lr, momentum=0.9),
    }
    optimizer = opts[args.optim]

    # GPU setting
    if args.gpu > -1:
        encoder = encoder.cuda()

    # some variables
    start_epoch = 0
    best_valid_loss = 99
    patience = args.patience

    # load checkpoint
    if args.load:
        start_epoch = checkpoint['epoch'] + 1
        encoder.load_state_dict(checkpoint['state_dict'])
        print('best valid loss from loaded model:', checkpoint['best_valid_loss'])

    # start training
    util.mkdir('../checkpoint')
    for epoch in range(start_epoch, 300):
        print('epoch', epoch)
        print(args)

        epoch_train_loss = 0.0
        epoch_valid_loss = 0.0

        for i, train_tup in tqdm(enumerate(trainset), total=len(trainset), ncols=50):
        # for i, (words, sens, sen_nums, weights) in tqdm(enumerate(train_batches), total=len(train_batches), ncols=30):
            if train_tup[0].size(0) == 1:
                continue
            encoder.zero_grad()
            # train_loss = encoder.get_batch_loss(words, sens, batch_sen_nums=sen_nums, weights=weights)
            train_loss = encoder.get_loss(train_tup)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += util.getval(train_loss)

        # get validation loss
        encoder.eval()
        for j, valid_tup in enumerate(validset):
            valid_loss = encoder.get_loss(valid_tup)
            epoch_valid_loss += util.getval(valid_loss)
        encoder.train()

        epoch_train_loss /= i
        epoch_valid_loss /= j

        print('train_loss: {:.4f} valid_loss: {:.4f} diff: {:.4f}'.format(epoch_train_loss, epoch_valid_loss, epoch_valid_loss - epoch_train_loss))

        # early stop
        if epoch_valid_loss >= best_valid_loss - 0.0001:
            patience -= 1
            # adjust learning rate
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('adjust lr ->', lr)
        else:
            patience = args.patience
            best_valid_loss = epoch_valid_loss

        if patience < 0 or lr < 0.000001:
            break

        encoder.save_checkpoint(epoch, epoch_train_loss, epoch_valid_loss, best_valid_loss)

print('loss: ' + str(best_valid_loss))
