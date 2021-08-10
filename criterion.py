# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function, Variable

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label-1
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)


    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class ReguOhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, Lambda=0.8,
        min_kept=100000, weight=None):
        super(ReguOhemCrossEntropy, self).__init__()
        self.Lambda = Lambda
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')


    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        # pred = F.softmax(score, dim=1)
        pred = score.clone()
        pixel_crossentropy = self.criterion(score, target)
        #entropy =tf.reduce_mean( tf.reduce_sum(-probabilities * tf.log(probabilities), axis=1,))
        pixel_entropy = ((-1*pred)*pred.log()).mean(0)
        pixel_losses = pixel_crossentropy - self.Lambda * pixel_entropy
        pixel_losses = pixel_losses.contiguous().view(-1)
        # mask = target.contiguous().view(-1) != self.ignore_label

        # tmp_target = target.clone()
        # tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, target.unsqueeze(1))
        # pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        pred, ind = pred.contiguous().view(-1,).contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        # pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
        min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label-1
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=self.ignore_label,
                                             reduction='none')


    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        # pred = F.softmax(score, dim=1)
        pred = score.clone()
        pixel_losses = self.criterion(score, target).contiguous().view(-1)

        tmp_target = target.clone()
        # print('before', torch.min(tmp_target))
        # print('before', torch.max(tmp_target))
        tmp_target.masked_fill_(tmp_target.eq(self.ignore_label), 0)
        # tmp_target[tmp_target == self.ignore_label] = 0
        # print('after', torch.min(tmp_target))
        # print('after', torch.max(tmp_target))
        mask = target.contiguous().view(-1) != self.ignore_label

        pred = pred.gather(1, tmp_target.unsqueeze(1))
        # print('after', pred)
        # print(mask)
        pred, ind = pred.contiguous().view(-1,).clone().masked_select(mask).contiguous().sort()
        # pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        # pred, ind = pred.contiguous().view(-1,).contiguous().sort()
        if pred.size(0) == 0:
            min_value = 0
        else:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        if len(pixel_losses[mask][ind]) != 0:
            # pixel_losses = pixel_losses[mask][ind]
            pixel_losses = pixel_losses.clone().masked_select(mask)[ind]
            # pixel_losses = pixel_losses[ind]
            pixel_losses = pixel_losses.clone().masked_select(pred < threshold)
            # pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class OhemCrossEntropy_HRnet(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
        min_kept=100000, weight=None):
        super(OhemCrossEntropy_HRnet, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        # pred = F.softmax(score, dim=1)
        pred = score.clone()
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        # mask = target.contiguous().view(-1) != self.ignore_label

        # tmp_target = target.clone()
        # tmp_target[tmp_target == self.ignore_label] = 0
        # pred = pred.gather(1, target.unsqueeze(1))
        # pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        pred, ind = pred.contiguous().view(-1,).contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        # pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, score, target):
        # https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        num_classes = score.shape[1]
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        if num_classes == 1:
            target_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            target_1_hot_f = target_1_hot[:, 0:1, :, :]
            target_1_hot_s = target_1_hot[:, 1:2, :, :]
            target_1_hot = torch.cat([target_1_hot_s, target_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(score)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            target_1_hot = torch.eye(num_classes)[target.squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            # probas = F.softmax(score, dim=1)
            probas = score.clone()
        target_1_hot = target_1_hot.type(score.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * target_1_hot, dims)
        cardinality = torch.sum(probas + target_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore_label=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore_label


    # def one_hot(index, classes):
    #     # index is not flattened (pypass ignore) ############
    #     # size = index.size()[:1] + (classes,) + index.size()[1:]
    #     # view = index.size()[:1] + (1,) + index.size()[1:]
    #     #####################################################
    #     # index is flatten (during ignore) ##################
    #     size = index.size()[:1] + (classes,)
    #     view = index.size()[:1] + (1,)
    #     #####################################################
    #
    #     # mask = torch.Tensor(size).fill_(0).to(device)
    #     mask = torch.Tensor(size).fill_(0).cuda()
    #     index = index.view(view)
    #     ones = 1.
    #
    #     return mask.scatter_(1, index, ones)


    def forward(self, input, target):
        # ph, pw = input.size(2), input.size(3)
        # h, w = target.size(1), target.size(2)
        # if ph != h or pw != w:
        #     input = F.upsample(input=input, size=(h, w), mode='bilinear')
        # pred = F.softmax(input, dim=1).to(input.device)
        # # pred_t = pred.gather(1, target.unsqueeze(1))
        # pred_t = pred[:,0,:,:]
        #
        # mask = target != self.ignore
        # pred_t_mask = pred_t.clone()
        # # pred_t_mask = pred_t.clone().masked_select(mask)
        # pred_t_mask = pred_t_mask.clamp(self.eps, 1. - self.eps)
        # pixel_loss = pred_t_mask
        # # pixel_loss = (-1 * (1 - pred_t_mask) ** self.gamma) * (pred_t_mask.log())
        # return pixel_loss.mean()


        '''
        only support ignore at 0
        '''
        ph, pw = input.size(2), input.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            input = F.upsample(input=input, size=(h, w), mode='bilinear')
        # print(target)
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        # print('max_target', max(target))
        # print('min_target', min(target))
        if self.ignore is not None:
            # if target.sum() == 0:
            #     target[0] = 1
            # if min(target) != 0:
            #     target[-1] = 0
            valid = (target != self.ignore)
            # input = input[valid]
            # target = target[valid]
            masked_input = torch.zeros((target[valid].size()[0],input.size(1))).to(input.device)
            for c in range(input.size(1)):
                masked_input[:,c] = input[:,c].masked_select(valid)
            input = masked_input
            # input = input.clone().masked_select(valid)
            target = target.clone().masked_select(valid)

        # if self.one_hot: target = one_hot(target, input.size(1))

        if self.one_hot:
            index = target.clone()
            classes = input.size(1)
            size = index.size()[:1] + (classes,)
            view = index.size()[:1] + (1,)
            mask = (torch.Tensor(size).fill_(0)).to(input.device)
            index = index.view(view)
            ones = 1.
            target_local = mask.scatter_(1, index, ones)
        probs = F.softmax(input, dim=1)

        # probs = input.clone()
        # print('probs', probs)
        # print('target_local', target_local)
        probs = (probs * target_local).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        # print(1 - probs)
        # print(torch.pow((1 - probs), self.gamma))
        # print(log_p)
        # print(-(torch.pow((1 - probs), self.gamma)) * log_p)
        batch_loss = (-(torch.pow((1 - probs), self.gamma)) * log_p)
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
            # print('got average batch_loss:', loss)
        else:
            loss = batch_loss.sum()
        # loss = loss.to(input.device)
        # print(loss)
        return loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        target = target.float()
		# print(input.size(), target.size())
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
