from tqdm import tqdm
from os.path import basename, join
from model.util import pad_sentence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, FloatTensor
from torch.autograd import Variable


class CosineDistanceLoss(nn.Module):
    def __init__(self, batch_size, use_gpu, margin=0, size_average=True):
        super(CosineDistanceLoss, self).__init__()
        self.margin = margin
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.size_average = size_average
        self.y = Variable(torch.ones(batch_size))
        if use_gpu:
            self.y = self.y.cuda()

    def forward(self, input1, input2, batch_size):
        if batch_size == self.batch_size:
            y = self.y
        else:
            y = Variable(torch.ones(batch_size))
            if self.use_gpu:
                y = y.cuda()
        return F.cosine_embedding_loss(input1, input2, y, self.margin, self.size_average)


class BaseEncoder(nn.Module):
    def __init__(self, def_word2ix, args):
        super(BaseEncoder, self).__init__()

        # input parameters
        self.args = args
        self.use_gpu = args.gpu > -1
        self.emb_dim = args.emb_dim

        # word embedding
        self.def_word2ix = def_word2ix
        self.def_vocab_size = len(def_word2ix)

        self.embed = nn.Embedding(self.def_vocab_size, self.emb_dim)
        self.embed.weight.requires_grad = args.fine_tune
        if self.use_gpu:
            self.embed = self.embed.cuda()

        # loss function
        self.loss_cos = CosineDistanceLoss(args.batch_size, self.use_gpu)

        # checkpoint path
        self.cp_path = None


    def init_def_embedding(self, def_embed):
        assert self.emb_dim == def_embed.size(1)
        assert self.def_vocab_size == def_embed.size(0)
        self.embed.weight.data.copy_(def_embed)

    def save_checkpoint(self, epoch, train_loss, valid_loss, best_valid_loss):
        data = {
            'args': self.args,
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'best_valid_loss': best_valid_loss,
            'state_dict': self.state_dict(),
        }
        if self.args.checkpoint:
            self.cp_path = self.args.checkpoint
        torch.save(data, self.cp_path)

    def forward(self, def_sens):
        raise NotImplementedError

    def get_loss(self, tup):
        grd_emb, def_sens = tup
        def_sens = Variable(def_sens)
        grd_emb = Variable(grd_emb)
        if self.use_gpu:
            def_sens = def_sens.cuda()
        out_emb = self(def_sens)
        assert out_emb.size() == grd_emb.size(), 'out {}, grd {}'.format(out_emb.size(), grd_emb.size())

        batch_size = def_sens.size(0)
        loss = self.loss_cos(out_emb, grd_emb, batch_size)

        return loss

    # def get_batch_loss(self, dic_words, def_sens, batch_sen_nums, weights):
    #     weights = weights.cuda()
    #     def_sens = Variable(def_sens).cuda()
    #     dic_words = Variable(dic_words).cuda()
    #     out_embs = self(def_sens)
    #     # target = self.dic_embed(dic_words).cuda()
    #     loss = Variable(FloatTensor([0])).cuda()
    #     assert def_sens.size(0) == sum(batch_sen_nums)
    #     # cos_losses = 1 - F.cosine_similarity(out_embs, target, dim=1, eps=1e-8)
    #     # print(cos_losses.grad)
    #     i = 0
    #     for sen_num in batch_sen_nums:
    #         def_embs = out_embs[i: i + sen_num]
    #         # word_losses = cos_losses[i: i + sen_num]
    #         word_weights = weights[i: i + sen_num].view(1, -1)
    #         weighted_emb = torch.mm(word_weights, def_embs).view(1, -1)
    #         # target_emb = target[i].view(1, -1)
    #         target_emb = self.dic_embed(dic_words[i].view(1, -1)).cuda().view(1, -1)
    #         word_loss = self.loss_cos(target_emb, weighted_emb, 1)
    #         loss += word_loss
    #         # print('grad', loss.grad)
    #         # assert False
    #         i += sen_num
    #     loss /= len(batch_sen_nums)
    #     return loss

    def estimate_from_defsens(self, def_sens):
        # sentence padding
        pad_size = self.args.pad_size
        pad_token = self.def_word2ix['</s>']
        pad_sens = [pad_sentence(s, pad_size, pad_token) for s in def_sens]

        # go through the encoder
        self.eval()
        pad_sens = Variable(LongTensor(pad_sens))
        if self.use_gpu:
            pad_sens = pad_sens.cuda()
        out_embs = self(pad_sens)
        return out_embs

    def estimate_from_strsen(self, sen):
        sen = ['<s>'] + sen.lower().split()
        sen = [self.def_word2ix[w] if w in self.def_word2ix else self.def_word2ix['<unk>'] for w in sen]
        sens = [sen]
        out_embs = self.estimate_from_defsens(sens)
        return out_embs[0].cpu().data.numpy()
