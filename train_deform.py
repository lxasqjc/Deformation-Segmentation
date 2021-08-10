# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
import pandas as pd
import numpy as np
from scipy.io import loadmat
from utils import colorEncode
from lib.utils import as_numpy
import torchvision.utils as vutils
# Numerical libs
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy
# Our libs
from config import cfg
from dataset import TrainDataset, imresize, b_imresize #, ValDataset
from models import ModelBuilder, SegmentationModule, FovSegmentationModule, SegmentationModule_plain, SegmentationModule_fov_deform, DeformSegmentationModule
from utils import AverageMeter, parse_devices, setup_logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from eval import eval_during_train_deform, eval_during_train
from eval_multipro import eval_during_train_multipro
from criterion import OhemCrossEntropy, DiceCoeff, FocalLoss
from pytorch_toolbelt.losses.dice import DiceLoss

from saliency_network import saliency_network_resnet18, saliency_network_resnet18_nonsyn, saliency_network_resnet10_nonsyn, fov_simple, fov_simple_nonsyn
# import wandb



# train one epoch
def train(segmentation_module, iterator, optimizers, epoch, cfg, history=None, foveation_module=None, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_edge_loss = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        # print('type_batch_data: {}\n'.format(type(batch_data)))
        # print('len_batch_data: {}\n'.format(len(batch_data)))
        if len(batch_data) == 1:
            batch_data = batch_data[0]
            single_gpu_mode = True
            print('single gpu mode ON \n')
            # print('type batch_data: {}\n'.format(type(batch_data)))
            batch_data['img_data'] = batch_data['img_data'].cuda()
            batch_data['seg_label'] = batch_data['seg_label'].cuda()
            batch_data = [batch_data]
        else:
            single_gpu_mode = False

        if cfg.DATASET.check_dataload and i == 0:
            aug_img = batch_data[0]['img_data'][0].unsqueeze(0)
            print('aug_img shape:{}'.format(aug_img.shape))
            aug_img = vutils.make_grid(aug_img, normalize=True, scale_each=True)
            writer.add_image('train/aug_img', aug_img, epoch)

            img_ori = batch_data[0]['img_ori'].unsqueeze(0)
            print('img_ori shape:{}'.format(img_ori.shape))
            img_ori = vutils.make_grid(img_ori, normalize=True, scale_each=True)
            writer.add_image('train/img_ori', img_ori, epoch)


        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        if cfg.MODEL.fov_deform:
            adjust_learning_rate(optimizers, cur_iter, cfg, epoch=epoch)

            if cfg.TRAIN.stage_adjust_edge_loss != 1.0 and epoch >= cfg.TRAIN.adjust_edge_loss_start_epoch and epoch <= cfg.TRAIN.adjust_edge_loss_end_epoch:
                cfg.TRAIN.edge_loss_scale = cfg.TRAIN.stage_adjust_edge_loss
                print('stage adjusted edge_loss_scale: ', cfg.TRAIN.edge_loss_scale)
            elif cfg.TRAIN.fixed_edge_loss_scale > 0.0:
                adjust_edge_loss_scale(cur_iter, cfg)

        else:
            adjust_learning_rate(optimizers, cur_iter, cfg)

        print_grad = None

        if cfg.MODEL.fov_deform:
            if single_gpu_mode:
                loss, acc, _ = segmentation_module(batch_data[0], writer=writer, count=cur_iter, epoch=epoch)
            elif 'single' in cfg.DATASET.list_train:
                if cfg.TRAIN.deform_joint_loss:
                    loss, acc, y_print, y_sampled, edge_loss = segmentation_module(batch_data)
                else:
                    loss, acc, y_print, y_sampled = segmentation_module(batch_data)
                # print(type(y_print))
                print(y_print.shape)
                # print(type(y_sampled))
                print(y_sampled.shape)
                y_print = y_print[0]
                y_sampled = y_sampled[0]
                colors = loadmat('data/color150.mat')['colors']
                y_color = colorEncode(as_numpy(y_print), colors)
                y_print = torch.from_numpy(y_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                print('train/y_original size: {}'.format(y_print.shape))
                y_print = vutils.make_grid(y_print, normalize=False, scale_each=True)

                if not cfg.MODEL.naive_upsample:
                    y_sampled_color = colorEncode(as_numpy(y_sampled), colors)
                    y_sampled_print = torch.from_numpy(y_sampled_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                    print('train/y_sampled size: {}'.format(y_sampled_print.shape))
                    y_sampled_print = vutils.make_grid(y_sampled_print, normalize=False, scale_each=True)

                writer.add_image('train/Label Original', y_print, cur_iter)
                writer.add_image('train/Deformed Label', y_sampled_print, cur_iter)
            else:
                if cfg.TRAIN.opt_deform_LabelEdge and epoch >= cfg.TRAIN.fix_seg_start_epoch and epoch <= cfg.TRAIN.fix_seg_end_epoch:
                    loss, acc, edge_loss = segmentation_module(batch_data)
                    # if epoch == cfg.TRAIN.opt_deform_LabelEdge_sepTrainPre:
                    #     cfg.TRAIN.opt_deform_LabelEdge = False
                elif cfg.TRAIN.separate_optimise_deformation and epoch >= cfg.TRAIN.fix_seg_start_epoch and epoch <= cfg.TRAIN.fix_seg_end_epoch:
                    loss, acc, save_print_grad = segmentation_module(batch_data)
                    if cur_iter > 0:
                        if save_print_grad[0]['saliency_grad'] is not None:
                            writer.add_histogram('Saliency gradient histogram', save_print_grad[0]['saliency_grad'], cur_iter)
                        if save_print_grad[0]['check1_grad'] is not None:
                            writer.add_histogram('Check1 gradient histogram', save_print_grad[0]['check1_grad'], cur_iter)
                        if save_print_grad[0]['check2_grad'] is not None:
                            writer.add_histogram('Check2 gradient histogram', save_print_grad[0]['check2_grad'], cur_iter)
                    if epoch == cfg.TRAIN.fix_seg_end_epoch:
                        cfg.TRAIN.opt_deform_LabelEdge = False
                        cfg.TRAIN.separate_optimise_deformation = False
                        cfg.MODEL.rev_deform_interp = 'nearest'
                        cfg.VAL.y_sampled_reverse = True
                # elif cfg.TRAIN.opt_deform_LabelEdge and epoch > cfg.TRAIN.opt_deform_LabelEdge_sepTrainPre:
                #     loss, acc, save_print_grad = segmentation_module(batch_data)
                elif cfg.TRAIN.deform_joint_loss:
                    if cfg.DATASET.check_dataload:
                        loss, acc, edge_loss = segmentation_module(batch_data, writer=writer)
                    else:
                        loss, acc, edge_loss = segmentation_module(batch_data)
                else:
                    loss, acc = segmentation_module(batch_data)
        else:
            if single_gpu_mode:
                loss, acc = segmentation_module(batch_data[0], writer=writer, count=cur_iter)
            elif 'single' in cfg.DATASET.list_train:
                if cfg.TRAIN.deform_joint_loss:
                    loss, acc, y_print, y_sampled_print, edge_loss = segmentation_module(batch_data)
                else:
                    loss, acc, y_print, y_sampled_print = segmentation_module(batch_data)
                writer.add_image('train/Label Original', y_print, count)
                writer.add_image('train/Deformed Label', y_sampled_print, count)
            else:
                loss, acc = segmentation_module(batch_data)
        if loss is None and acc is None:
            print('A-skip iter: {}\n'.format(i))
            continue
        # print()
        loss_step = loss.mean()
        acc_step = acc.mean()

        # Backward
        if not (cfg.MODEL.gt_gradient and cfg.MODEL.gt_gradient_intrinsic_only):
            loss_step.backward()

            if cfg.MODEL.fov_deform:
                # (optimizer) = optimizers
                for optimizer in optimizers:
                    if cfg.TRAIN.fix_deform_aft_pretrain and epoch >= cfg.TRAIN.fix_deform_start_epoch and epoch <= cfg.TRAIN.fix_deform_end_epoch:
                        if optimizer.param_groups[0]['zoom']==False: # update segmentation module only
                            optimizer.step()
                    elif cfg.TRAIN.opt_deform_LabelEdge and epoch >= cfg.TRAIN.fix_seg_start_epoch and epoch <= cfg.TRAIN.fix_seg_end_epoch:
                        if optimizer.param_groups[0]['zoom']==True: # update deformation module only
                            optimizer.step()
                    else:
                        optimizer.step()
            else:
                for optimizer in optimizers:
                    optimizer.step()

        # update average loss and acc
        ave_total_loss.update(loss_step.data.item())
        ave_acc.update(acc_step.data.item()*100)
        if cfg.TRAIN.deform_joint_loss:
            ave_edge_loss.update(edge_loss.mean().data.item())

        # if loss is None and acc is None:
        #     print('B-skip iter: {}\n'.format(i))
        #     continue

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()



        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            if cfg.TRAIN.deform_joint_loss:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                      'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                      'Accuracy: {:4.2f}, Seg_Loss: {:.6f}, Edge_Loss: {:.6f}'
                      .format(epoch, i, cfg.TRAIN.epoch_iters,
                              batch_time.average(), data_time.average(),
                              cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                              ave_acc.average(), ave_total_loss.average(), ave_edge_loss.average()))
            else:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                      'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                      'Accuracy: {:4.2f}, Seg_Loss: {:.6f}'
                      .format(epoch, i, cfg.TRAIN.epoch_iters,
                              batch_time.average(), data_time.average(),
                              cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                              ave_acc.average(), ave_total_loss.average()))

        fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
        if history is not None:
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(ave_total_loss.average())
            history['train']['acc'].append(ave_acc.average()/100)
            if cfg.TRAIN.deform_joint_loss:
                history['train']['edge_loss'].append(ave_edge_loss.average())
            if print_grad is not None:
                history['train']['print_grad'] = print_grad


def checkpoint(nets, cfg, epoch):
    print('Saving checkpoints...')
    if cfg.MODEL.fov_deform:
        (net_encoder, net_decoder, crit, net_saliency, net_compress) = nets
        dict_saliency = net_saliency.state_dict()
        torch.save(
            dict_saliency,
            '{}/saliency_epoch_{}.pth'.format(cfg.DIR, epoch))
        dict_compress = net_compress.state_dict()
        torch.save(
            dict_compress,
            '{}/compress_epoch_{}.pth'.format(cfg.DIR, epoch))
    else:
        (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))



def checkpoint_last(nets, cfg, epoch):
    print('Saving checkpoints...')
    if cfg.MODEL.fov_deform:
        (net_encoder, net_decoder, crit, net_saliency, net_compress) = nets
        dict_saliency = net_saliency.state_dict()
        torch.save(
            dict_saliency,
            '{}/saliency_epoch_last.pth'.format(cfg.DIR))
        dict_compress = net_compress.state_dict()
        torch.save(
            dict_compress,
            '{}/compress_epoch_last.pth'.format(cfg.DIR))
    else:
        (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_last.pth'.format(cfg.DIR))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_last.pth'.format(cfg.DIR))

def checkpoint_history(history, cfg, epoch):
    print('Saving history...')
    # print('epoch: {}\ntrain_loss: {}\ntrain_acc: {}\nval_miou: {}\nval_acc: {}\nval_deformed_miou: {}\nval_y_reverse_miou: {}\nnum_valid_samples: {}\n'.format(history['save']['epoch']
    # , history['save']['train_loss']
    # , history['save']['train_acc']
    # , history['save']['val_iou']
    # , history['save']['val_acc']
    # , history['save']['val_iou_deformed']
    # , history['save']['val_iou_y_reverse']
    # , history['save']['num_valid_samples']))
    # save history as csv
    data_frame = pd.DataFrame(
        data={'epoch': history['save']['epoch']
            , 'train_loss': history['save']['train_loss']
            , 'train_acc': history['save']['train_acc']
            , 'val_miou': history['save']['val_iou']
            , 'val_acc': history['save']['val_acc']
            , 'val_deformed_miou': history['save']['val_iou_deformed']
            , 'val_y_reverse_miou': history['save']['val_iou_y_reverse']
            , 'num_valid_samples': history['save']['num_valid_samples']
              }
    )
    if cfg.VAL.dice:
        data_frame['val_dice'] = history['save']['val_dice']
        data_frame['val_dice_deformed'] = history['save']['val_dice_deformed']
    if cfg.TRAIN.deform_joint_loss:
        data_frame['train_edge_loss'] = history['save']['train_edge_loss']

    for c in range(cfg.DATASET.num_class):
        data_frame['val_iou_class_'+str(c)] = history['save']['val_iou_class_'+str(c)]
        data_frame['val_iou_deformed_class_'+str(c)] = history['save']['val_iou_deformed_class_'+str(c)]
        if cfg.TRAIN.opt_deform_LabelEdge:
            data_frame['val_iou_y_reverse_class_'+str(c)] = 0.0
        elif cfg.VAL.y_sampled_reverse:
            data_frame['val_iou_y_reverse_class_'+str(c)] = history['save']['val_iou_y_reverse_class_'+str(c)]

    data_frame.to_csv('{}/history_epoch_last.csv'.format(cfg.DIR),
                      index_label='epoch')

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    if cfg.MODEL.fov_deform:
        (net_encoder, net_decoder, crit, net_saliency, net_compress) = nets
    else:
        (net_encoder, net_decoder, crit) = nets

    if cfg.TRAIN.optim.lower() == 'sgd':
        if cfg.MODEL.fov_deform:
            optimizer = torch.optim.SGD([
                    {'params': net_encoder.parameters(),'lr_mult':cfg.TRAIN.lr_mult_encoder,'zoom':False},
                    {'params': net_decoder.parameters(),'lr_mult':cfg.TRAIN.lr_mult_decoder,'zoom':False},
                    {'params': net_saliency.parameters(),'lr_mult':cfg.TRAIN.lr_mult_saliency,'zoom':True},
                    {'params': net_compress.parameters(),'lr_mult':cfg.TRAIN.lr_mult_compress,'zoom':True}
                    ],lr =cfg.TRAIN.lr_encoder,momentum=cfg.TRAIN.beta1,weight_decay=cfg.TRAIN.weight_decay)
        else:
            optimizer_encoder = torch.optim.SGD(
                group_weight(net_encoder),
                lr=cfg.TRAIN.lr_encoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizer_decoder = torch.optim.SGD(
                group_weight(net_decoder),
                lr=cfg.TRAIN.lr_decoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)

    elif cfg.TRAIN.optim.lower() == 'adam':
        if cfg.MODEL.fov_deform:
            # optimizer = torch.optim.Adam([
            #         {'params': net_encoder.parameters(),'lr_mult':cfg.TRAIN.lr_mult_encoder,'zoom':False},
            #         {'params': net_decoder.parameters(),'lr_mult':cfg.TRAIN.lr_mult_decoder,'zoom':False},
            #         {'params': net_saliency.parameters(),'lr_mult':cfg.TRAIN.lr_mult_saliency,'zoom':True},
            #         {'params': net_compress.parameters(),'lr_mult':cfg.TRAIN.lr_mult_compress,'zoom':True}
            #         ],lr =cfg.TRAIN.lr_encoder,weight_decay=cfg.TRAIN.weight_decay)
            optimizer_encoder = torch.optim.Adam(
                [{'params': net_encoder.parameters(),'lr_mult':cfg.TRAIN.lr_mult_encoder,'zoom':False}],
                lr=cfg.TRAIN.lr_encoder,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizer_decoder = torch.optim.Adam(
                [{'params': net_decoder.parameters(),'lr_mult':cfg.TRAIN.lr_mult_decoder,'zoom':False}],
                lr=cfg.TRAIN.lr_encoder,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizer_saliency = torch.optim.Adam(
                [{'params': net_saliency.parameters(),'lr_mult':cfg.TRAIN.lr_mult_saliency,'zoom':True}],
                lr=cfg.TRAIN.lr_encoder,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizer_compress = torch.optim.Adam(
                [{'params': net_compress.parameters(),'lr_mult':cfg.TRAIN.lr_mult_compress,'zoom':True}],
                lr=cfg.TRAIN.lr_encoder,
                weight_decay=cfg.TRAIN.weight_decay)
        else:
            optimizer_encoder = torch.optim.Adam(
                group_weight(net_encoder),
                lr=cfg.TRAIN.lr_encoder,
                weight_decay=cfg.TRAIN.weight_decay)
            optimizer_decoder = torch.optim.Adam(
                group_weight(net_decoder),
                lr=cfg.TRAIN.lr_decoder,
                weight_decay=cfg.TRAIN.weight_decay)


    if cfg.MODEL.fov_deform:
        # return (optimizer)
        return (optimizer_encoder, optimizer_decoder, optimizer_saliency, optimizer_compress)
    else:
        return (optimizer_encoder, optimizer_decoder)

def adjust_gms_tau(cur_iter, cfg, r=1e5):
    cfg.MODEL.gumbel_tau = max(0.1, float(np.exp(-1.*r*float(cur_iter))))
    print('adjusted_tau: ', cfg.MODEL.gumbel_tau)


def adjust_edge_loss_scale(cur_iter, cfg):
    scale_running_el = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.edge_loss_pow)
    cfg.TRAIN.edge_loss_scale = cfg.TRAIN.fixed_edge_loss_scale * scale_running_el

    if cfg.TRAIN.edge_loss_scale < cfg.TRAIN.edge_loss_scale_min:
        cfg.TRAIN.edge_loss_scale = cfg.TRAIN.edge_loss_scale_min
    print('scaled edge_loss_scale: ', cfg.TRAIN.edge_loss_scale)

def adjust_learning_rate(optimizers, cur_iter, cfg, lr_mbs = False, f_max_iter=1, lr_scale=1, wd_scale=1, epoch=None):
    # print('adjusted max_iter:', cfg.TRAIN.max_iters)
    scale_running_lr = ((1. - float(cur_iter) / f_max_iter) ** cfg.TRAIN.lr_pow)
    if not lr_mbs:
        scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    if cfg.TRAIN.fov_scale_lr != '':
        lr_scale = pow(lr_scale, cfg.TRAIN.fov_scale_pow)
        wd_scale = pow(wd_scale, cfg.TRAIN.fov_scale_pow)
        print('after fov_pow lr_scale={}, wd_scale={}'.format(lr_scale, wd_scale))
        print('original scale_running_lr: ', scale_running_lr)
        scale_running_lr *= lr_scale
        print('scaled scale_running_lr: ', scale_running_lr)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr
    if cfg.TRAIN.fov_scale_seg_only:
        scale_running_lr /= lr_scale
    cfg.TRAIN.running_lr_foveater = cfg.TRAIN.lr_foveater * scale_running_lr
    if cfg.MODEL.fov_deform:
        # (optimizer) = optimizers
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = param_group['lr_mult']*cfg.TRAIN.running_lr_foveater
        base_lr = 0.1
        N_pretraining_base = cfg.TRAIN.deform_pretrain

        if cfg.TRAIN.scale_by_iter:
            N_pretraining = N_pretraining_base*cfg.TRAIN.epoch_iters
            lr_idx = cur_iter
        else:
            N_pretraining = N_pretraining_base
            lr_idx = epoch

        # lr_class -> for segmentation_module; lr_zoom -> DeformSegmentationModule
        # pretrain stage, same lr for both deform and seg module
        # non-pretrain stage, larger lr for seg module by larger lr_class (i.e. segmentation_module)
        if cfg.TRAIN.deform_pretrain_bol:
            lr_class = base_lr*0.1**(lr_idx//N_pretraining)
            lr_zoom = base_lr*0.1**(lr_idx//N_pretraining)
        elif lr_idx>=N_pretraining:
            lr_class = base_lr*0.1**((lr_idx-N_pretraining)//N_pretraining)
            lr_zoom = base_lr*0.1**(lr_idx//N_pretraining)
        else:
            lr_class = base_lr*0.1**(lr_idx//N_pretraining)
            lr_zoom = base_lr*0.1**(lr_idx//N_pretraining)

        if cfg.TRAIN.fix_deform_aft_pretrain and epoch >= cfg.TRAIN.fix_deform_start_epoch and epoch <= cfg.TRAIN.fix_deform_end_epoch:
            lr_zoom = 0.0
        if cfg.TRAIN.opt_deform_LabelEdge and epoch >= cfg.TRAIN.fix_seg_start_epoch and epoch <= cfg.TRAIN.fix_seg_end_epoch:
            lr_class = 0.0

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                if param_group['zoom']==True:
                    param_group['lr'] = param_group['lr_mult']*lr_zoom
                    if cfg.TRAIN.opt_deform_LabelEdge:
                        param_group['zoom'] *= cfg.TRAIN.opt_deform_LabelEdge_accrate
                else:
                    param_group['lr'] = param_group['lr_mult']*lr_class
    else:
        (optimizer_encoder, optimizer_decoder) = optimizers
        for param_group in optimizer_encoder.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_encoder
            if cfg.TRAIN.fov_scale_weight_decay != '':
                param_group['weight_decay'] = cfg.TRAIN.weight_decay * wd_scale
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_decoder
            if cfg.TRAIN.fov_scale_weight_decay != '':
                param_group['weight_decay'] = cfg.TRAIN.weight_decay * wd_scale


def main(cfg, gpus):
    # wandb.init()
    # ###============== IMPORT SALIENCY LIBS ===========###
    # if cfg.MODEL.rev_deform_opt == 1:
    #     from saliency_sampler_normal_reverse_saliency import Saliency_Sampler
    # elif cfg.MODEL.rev_deform_opt == 2:
    #     from saliency_sampler_normal_reverse_saliency_deconv import Saliency_Sampler
    # elif cfg.MODEL.rev_deform_opt == 3:
    #     from saliency_sampler_normal_reverse_deconv_pad import Saliency_Sampler
    # elif cfg.MODEL.rev_deform_opt == 4 or cfg.MODEL.rev_deform_opt == 5 or cfg.MODEL.rev_deform_opt == 6 or cfg.MODEL.rev_deform_opt == 51:
    #     from saliency_sampler_inv_fill_NN import Saliency_Sampler

    ###============== DEFINE LOSSES ===========###
    if cfg.DATASET.root_dataset == '/scratch0/chenjin/GLEASON2019_DATA/Data/':
        if cfg.TRAIN.loss_fun == 'DiceLoss':
            crit = DiceLoss('multiclass')
        elif cfg.TRAIN.loss_fun == 'FocalLoss':
            crit = FocalLoss()
        elif cfg.TRAIN.loss_fun == 'DiceCoeff':
            crit = DiceCoeff()
        elif cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=-2)
        else:
            crit = OhemCrossEntropy(ignore_label=-1,
                                     thres=0.9,
                                     min_kept=100000,
                                     weight=None)
    elif 'ADE20K' in cfg.DATASET.root_dataset:
        crit = nn.NLLLoss(ignore_index=-2)
    elif 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        elif cfg.TRAIN.loss_fun == 'DiceLoss':
            # crit = DiceLoss(ignore_index=19)
            crit = DiceLoss('multiclass', ignore_index=19)
        else:
            if cfg.TRAIN.loss_weight != []:
                class_weights = torch.FloatTensor(list(cfg.TRAIN.loss_weight)).cuda()
            else:
                class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                                1.0166, 0.9969, 0.9754, 1.0489,
                                                0.8786, 1.0023, 0.9539, 0.9843,
                                                1.1116, 0.9037, 1.0865, 1.0955,
                                                1.0865, 1.1529, 1.0507]).cuda()
            if cfg.TRAIN.scale_weight != "":
                if 'power' in cfg.TRAIN.scale_weight:
                    p = int(cfg.TRAIN.scale_weight.split('_')[-1])
                    class_weights = class_weights.pow(p)

            if cfg.DATASET.binary_class != -1:
                class_weights = None

            crit = OhemCrossEntropy(ignore_label=20,
                                         thres=0.9,
                                         min_kept=131072,
                                         weight=class_weights)
    elif 'DeepGlob' in cfg.DATASET.root_dataset and (cfg.TRAIN.loss_fun == 'FocalLoss' or cfg.TRAIN.loss_fun == 'OhemCrossEntropy'):
        if cfg.TRAIN.loss_fun == 'FocalLoss':
            crit = FocalLoss(gamma=6, ignore_label=cfg.DATASET.ignore_index)
        elif cfg.TRAIN.loss_fun == 'OhemCrossEntropy':
            crit = OhemCrossEntropy(ignore_label=cfg.DATASET.ignore_index,
                                         thres=0.9,
                                         min_kept=131072)
    else:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            if cfg.DATASET.ignore_index != -2:
                crit = nn.NLLLoss(ignore_index=cfg.DATASET.ignore_index)
            else:
                crit = nn.NLLLoss(ignore_index=-2)
        else:
            if cfg.DATASET.ignore_index != -2:
                crit = nn.CrossEntropyLoss(ignore_index=cfg.DATASET.ignore_index)
            else:
                crit = nn.CrossEntropyLoss(ignore_index=-2)
    # crit = DiceCoeff()

    ###============== Network Builders ===========###
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
        dilate_rate=cfg.DATASET.segm_downsampling_rate)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)
    if cfg.MODEL.fov_deform:
        net_saliency = ModelBuilder.build_net_saliency(
            cfg=cfg,
            weights=cfg.MODEL.weights_net_saliency)
        net_compress = ModelBuilder.build_net_compress(
            cfg=cfg,
            weights=cfg.MODEL.weights_net_compress)
        if cfg.MODEL.arch_decoder.endswith('deepsup'):
            segmentation_module = DeformSegmentationModule(net_encoder, net_decoder, net_saliency, net_compress, crit, cfg, deep_sup_scale=cfg.TRAIN.deep_sup_scale)
        else:
            segmentation_module = DeformSegmentationModule(net_encoder, net_decoder, net_saliency, net_compress, crit, cfg)
    else:
        if cfg.MODEL.arch_decoder.endswith('deepsup'):
            segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg, deep_sup_scale=cfg.TRAIN.deep_sup_scale)
        else:
            segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg)
    total = sum([param.nelement() for param in segmentation_module.parameters()])
    print('Number of SegmentationModule params: %.2fM \n' % (total / 1e6))



    ###============== Load deformation model if start_epoch > 0===========###
    # if cfg.TRAIN.start_epoch > 0 and cfg.MODEL.fov_deform:
    #     weights = os.path.join(cfg.DIR, 'model_epoch_{}.pth'.format(fg.TRAIN.start_epoch))
    #     segmentation_module.load_state_dict(
    #         torch.load(weights, map_location=lambda storage, loc: storage), strict=True)

    ###============== SET UP OPTIMIZERS ===========###
    if cfg.MODEL.fov_deform:
        nets = (net_encoder, net_decoder, crit, net_saliency, net_compress)
    else:
        nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    ###============== LOAD NETS INTO GPUs ===========###
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)

    segmentation_module.cuda()




    ###============== SET UP WRITER ===========###
    writer = SummaryWriter('{}/tensorboard'.format(cfg.DIR))
    if cfg.VAL.dice:
        history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'save': {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_iou': [], 'val_dice': [], 'val_acc': [], 'val_iou_deformed': [], 'val_dice_deformed': [], 'val_acc_deformed': [], 'val_iou_y_reverse': [], 'val_dice_y_reverse': [], 'val_acc_y_reverse': [], 'num_valid_samples': [], 'print_grad': None}}
    else:
        if cfg.TRAIN.deform_joint_loss:
            history = {'train': {'epoch': [], 'loss': [], 'edge_loss': [], 'acc': []}, 'save': {'epoch': [], 'train_loss': [], 'train_edge_loss': [], 'train_acc': [], 'val_iou': [], 'val_dice': [], 'val_acc': [], 'val_iou_deformed': [], 'val_dice_deformed': [], 'val_acc_deformed': [], 'val_iou_y_reverse': [], 'val_dice_y_reverse': [], 'val_acc_y_reverse': [], 'num_valid_samples': [], 'print_grad': None}}
        else:
            history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'save': {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_iou': [], 'val_dice': [], 'val_acc': [], 'val_iou_deformed': [], 'val_dice_deformed': [], 'val_acc_deformed': [], 'val_iou_y_reverse': [], 'val_dice_y_reverse': [], 'val_acc_y_reverse': [], 'num_valid_samples': [], 'print_grad': None}}
    for c in range(cfg.DATASET.num_class):
        history['save']['val_iou_class_'+str(c)] = []
        history['save']['val_iou_deformed_class_'+str(c)] = []
        history['save']['val_iou_y_reverse_class_'+str(c)] = []
    if cfg.TRAIN.start_epoch > 0:
        # epoch,train_loss,train_acc,val_miou,val_acc,val_deformed_miou,val_y_reverse_miou,num_valid_samples,
        # val_iou_class_0,val_iou_deformed_class_0,val_iou_y_reverse_class_0,val_iou_class_1,val_iou_deformed_class_1,
        # val_iou_y_reverse_class_1,val_iou_class_2,val_iou_deformed_class_2,val_iou_y_reverse_class_2,val_iou_class_3,
        # val_iou_deformed_class_3,val_iou_y_reverse_class_3
        history_previous_epoches = pd.read_csv('{}/history_epoch_{}.csv'.format(cfg.DIR, cfg.TRAIN.start_epoch))
        history['save']['epoch'] = list(history_previous_epoches['epoch'])
        history['save']['train_loss'] = list(history_previous_epoches['train_loss'])
        if cfg.TRAIN.deform_joint_loss:
            history['save']['train_edge_loss'] = list(history_previous_epoches['train_edge_loss'])
        history['save']['train_acc'] = list(history_previous_epoches['train_acc'])
        history['save']['val_iou'] = list(history_previous_epoches['val_miou'])
        history['save']['val_acc'] = list(history_previous_epoches['val_acc'])
        history['save']['num_valid_samples'] = list(history_previous_epoches['num_valid_samples'])
        if 'val_iou_deformed' in history_previous_epoches:
            history['save']['val_iou_deformed'] = list(history_previous_epoches['val_deformed_miou'])
        else:
            history['save']['val_iou_deformed'] = ['' for i in range(len(history['save']['epoch']))]
            # history['save']['val_acc_deformed'] = list(history_previous_epoches['val_acc_deformed'])
        if 'val_iou_y_reverse' in history_previous_epoches:
            history['save']['val_iou_y_reverse'] = list(history_previous_epoches['val_y_reverse_miou'])
        else:
            history['save']['val_iou_y_reverse'] = ['' for i in range(len(history['save']['epoch']))]
            # history['save']['val_acc_y_reverse'] = list(history_previous_epoches['val_acc_y_reverse'])
        if cfg.VAL.dice:
            if 'val_dice' in history_previous_epoches:
                history['save']['val_dice'] = list(history_previous_epoches['val_dice'])
            else:
                history['save']['val_dice'] = ['' for i in range(len(history['save']['epoch']))]
            if 'val_dice_deformed' in history_previous_epoches:
                history['save']['val_dice_deformed'] = list(history_previous_epoches['val_dice_deformed'])
            else:
                history['save']['val_dice_deformed'] = ['' for i in range(len(history['save']['epoch']))]
        for c in range(cfg.DATASET.num_class):
            if ('val_iou_class_'+str(c)) in history_previous_epoches:
                history['save']['val_iou_class_'+str(c)] = list(history_previous_epoches['val_iou_class_'+str(c)])
            else:
                history['save']['val_iou_class_'+str(c)] = ['' for i in range(len(history['save']['epoch']))]
            if ('val_iou_deformed_class_'+str(c)) in history_previous_epoches:
                history['save']['val_iou_deformed_class_'+str(c)] = list(history_previous_epoches['val_iou_deformed_class_'+str(c)])
            else:
                history['save']['val_iou_deformed_class_'+str(c)] = ['' for i in range(len(history['save']['epoch']))]
            if cfg.VAL.y_sampled_reverse:
                if ('val_iou_y_reverse_class_'+str(c)) in history_previous_epoches:
                    history['save']['val_iou_y_reverse_class_'+str(c)] = list(history_previous_epoches['val_iou_y_reverse_class_'+str(c)])
                else:
                    history['save']['val_iou_y_reverse_class_'+str(c)] = ['' for i in range(len(history['save']['epoch']))]

    ###============== Dataset and Loader ===========###
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))
    # create loader iterator
    iterator_train = iter(loader_train)
    initial_relative_eval_y_ysample_last = True
    relative_eval_y_ysample_last = None
    ###============== MAIN LOOP ===========###
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        cfg.TRAIN.global_epoch = epoch+1
        # if cfg.TRAIN.stage_adjust_edge_loss != 1.0 and epoch >= cfg.TRAIN.adjust_edge_loss_start_epoch and epoch <= cfg.TRAIN.adjust_edge_loss_end_epoch:
        #     cfg.TRAIN.fixed_edge_loss_scale = cfg.TRAIN.stage_adjust_edge_loss

        if not cfg.TRAIN.skip_train_for_eval:

            train(segmentation_module, iterator_train,
            optimizers, epoch+1, cfg, history, writer=writer)
            ## checkpointing
            if (epoch+1) % cfg.TRAIN.checkpoint_per_epoch == 0:
                checkpoint(nets, cfg, epoch+1)
                checkpoint_last(nets, cfg, epoch+1)
            else:
                checkpoint_last(nets, cfg, epoch+1)

        ## eval during train
        if (epoch+1) % cfg.TRAIN.eval_per_epoch == 0:
            if cfg.VAL.multipro:
                val_iou, val_acc = eval_during_train_multipro(cfg, gpus)
            elif not cfg.MODEL.fov_deform:
                val_iou, val_acc = eval_during_train(cfg, gpus)
            else:
                if cfg.VAL.dice:
                    if cfg.VAL.y_sampled_reverse:
                        val_iou, val_dice, val_acc, val_iou_deformed, val_dice_deformed, val_acc_deformed, val_iou_y_reverse, val_dice_y_reverse, val_acc_y_reverse, relative_eval_y_ysample, ious = eval_during_train_deform(cfg, writer=writer, count=epoch+1)
                    else:
                        val_iou, val_dice, val_acc, val_iou_deformed, val_dice_deformed, val_acc_deformed, relative_eval_y_ysample, ious = eval_during_train_deform(cfg, writer=writer, count=epoch+1)
                else:
                    if cfg.VAL.y_sampled_reverse:
                        val_iou, val_acc, val_iou_deformed, val_acc_deformed, val_iou_y_reverse, val_acc_y_reverse, relative_eval_y_ysample, ious = eval_during_train_deform(cfg, writer=writer, count=epoch+1)
                    else:
                        val_iou, val_acc, val_iou_deformed, val_acc_deformed, relative_eval_y_ysample, ious = eval_during_train_deform(cfg, writer=writer, count=epoch+1)
            # deform y-y_sampled
            for i in range(len(relative_eval_y_ysample)):
                writer.add_scalars('Eval step Y_sampled-Y distribution', {'Class {}'.format(i): relative_eval_y_ysample[i]}, epoch+1)
            if initial_relative_eval_y_ysample_last:
                relative_eval_y_ysample_last = relative_eval_y_ysample
                initial_relative_eval_y_ysample_last = False
            else:
                # print(type(relative_eval_y_ysample_last))
                for i in range(len(relative_eval_y_ysample_last)):
                    # print(relative_eval_y_ysample_last[i] - relative_eval_y_ysample[i])
                    writer.add_scalars('Eval step incremental Y_sampled-Y distribution', {'Class {}'.format(i): relative_eval_y_ysample_last[i] - relative_eval_y_ysample[i]}, epoch+1)
                relative_eval_y_ysample_last = relative_eval_y_ysample

            # unfold ious
            if cfg.VAL.y_sampled_reverse:
                iou, iou_deformed, iou_y_reverse, num_valid_samples = ious
            else:
                iou, iou_deformed, num_valid_samples = ious

            # save history
            history['save']['epoch'].append(epoch+1)
            if history['train']['loss'] == []:
                history['save']['train_loss'].append('')
            else:
                history['save']['train_loss'].append(history['train']['loss'][-1])
                writer.add_scalar('Loss/train', history['train']['loss'][-1], epoch+1)
            if cfg.TRAIN.deform_joint_loss:
                if history['train']['edge_loss'] == []:
                    history['save']['train_edge_loss'].append('')
                else:
                    history['save']['train_edge_loss'].append(history['train']['edge_loss'][-1])
                    writer.add_scalar('edge_loss/train', history['train']['edge_loss'][-1], epoch+1)
            if history['train']['acc'] == []:
                history['save']['train_acc'].append('')
            else:
                history['save']['train_acc'].append(history['train']['acc'][-1]*100)
                writer.add_scalar('Acc/train', history['train']['acc'][-1]*100, epoch+1)
            history['save']['val_iou'].append(val_iou)
            print('val_iou_deformed: {}'.format(val_iou_deformed))
            history['save']['val_iou_deformed'].append(val_iou_deformed)
            if cfg.VAL.dice:
                history['save']['val_dice'].append(val_dice)
                history['save']['val_dice_deformed'].append(val_dice)
            history['save']['val_acc'].append(val_acc)
            history['save']['val_acc_deformed'].append(val_acc_deformed)
            history['save']['num_valid_samples'].append(num_valid_samples)
            for c in range(cfg.DATASET.num_class):
                history['save']['val_iou_class_'+str(c)].append(iou[c])
                history['save']['val_iou_deformed_class_'+str(c)].append(iou_deformed[c])
                if cfg.VAL.y_sampled_reverse:
                    history['save']['val_iou_y_reverse_class_'+str(c)].append(iou_y_reverse[c])
            # write to tensorboard


            # writer.add_scalar('Acc/val', val_acc, epoch+1)
            # writer.add_scalar('Acc/val_deformed', val_acc_deformed, epoch+1)
            writer.add_scalars('Acc', {'val': val_acc}, epoch+1)
            writer.add_scalars('Acc', {'val_deformed': val_acc_deformed}, epoch+1)
            # writer.add_scalar('mIoU/val', val_iou, epoch+1)
            # writer.add_scalar('mIoU/val_deformed', val_iou_deformed, epoch+1)
            writer.add_scalars('mIoU', {'val': val_iou}, epoch+1)
            writer.add_scalars('mIoU', {'val_deformed': val_iou_deformed}, epoch+1)
            if cfg.VAL.y_sampled_reverse:
                history['save']['val_iou_y_reverse'].append(val_iou_y_reverse)
                history['save']['val_acc_y_reverse'].append(val_acc_y_reverse)
                if cfg.VAL.dice:
                    history['save']['val_dice_y_reverse'].append(val_dice)
                writer.add_scalars('Acc', {'val_y_reverse': val_acc_y_reverse}, epoch+1)
                writer.add_scalars('mIoU', {'val_y_reverse': val_iou_y_reverse}, epoch+1)
            else:
                history['save']['val_iou_y_reverse'].append('n/a')
                history['save']['val_acc_y_reverse'].append('n/a')
            # writer.add_scalar('Acc_deformed/val', val_acc_deformed, epoch+1)
            # writer.add_scalar('mIoU_deformed/val', val_iou_deformed, epoch+1)
            if cfg.VAL.dice:
                writer.add_scalar('mDice/val', val_dice, epoch+1)
                writer.add_scalar('mDice/val_deformed', val_dice_deformed, epoch+1)
                # writer.add_scalar('mDice_deformed/val', val_dice_deformed, epoch+1)
        else:
            history['save']['epoch'].append(epoch+1)
            history['save']['train_loss'].append(history['train']['loss'][-1])
            history['save']['train_acc'].append(history['train']['acc'][-1]*100)
            history['save']['val_iou'].append('')
            history['save']['val_iou_deformed'].append('')
            history['save']['val_iou_y_reverse'].append('')
            if cfg.VAL.dice:
                history['save']['val_dice'].append('')
                history['save']['val_dice_deformed'].append('')
                history['save']['val_dice_y_reverse'].append('')
            history['save']['val_acc'].append('')
            history['save']['val_acc_deformed'].append('')
            history['save']['val_acc_y_reverse'].append('')
            history['save']['num_valid_samples'].append('')
            if cfg.TRAIN.deform_joint_loss:
                history['save']['train_edge_loss'].append('')
            # write to tensorboard
            writer.add_scalar('Loss/train', history['train']['loss'][-1], epoch+1)
            writer.add_scalar('Acc/train', history['train']['acc'][-1]*100, epoch+1)
            if cfg.TRAIN.deform_joint_loss:
                writer.add_scalar('edge_loss/train', history['train']['edge_loss'][-1], epoch+1)
            # writer.add_scalar('Acc/val', '', epoch+1)
            # writer.add_scalar('mIoU/val', '', epoch+1)
            for c in range(cfg.DATASET.num_class):
                history['save']['val_iou_class_'+str(c)].append('')
                history['save']['val_iou_deformed_class_'+str(c)].append('')
                if cfg.VAL.y_sampled_reverse:
                    history['save']['val_iou_y_reverse_class_'+str(c)].append('')

        # saving history
        checkpoint_history(history, cfg, epoch+1)


    if not cfg.TRAIN.save_checkpoint:
        os.remove('{}/encoder_epoch_last.pth'.format(cfg.DIR))
        os.remove('{}/decoder_epoch_last.pth'.format(cfg.DIR))
    print('Training Done!')
    writer.close()

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/deform-cityscape.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    if cfg.TRAIN.auto_batch == 'auto10':
        # asign 10G per gpu estimated by: can take about 10e6 pixels with hrnetv2
        cfg.TRAIN.batch_size_per_gpu = int((1e6*0.65) // (cfg.DATASET.imgSizes[0]*cfg.DATASET.imgSizes[0]))
        gpus = parse_devices(args.gpus)
        num_gpu = len(gpus)
        cfg.TRAIN.num_gpus = num_gpu
        num_data = len([x for x in open(cfg.DATASET.list_train, 'r')])
        print(num_data, num_gpu, cfg.TRAIN.batch_size_per_gpu)
        cfg.TRAIN.epoch_iters = int(num_data // (num_gpu*cfg.TRAIN.batch_size_per_gpu))

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
        if cfg.MODEL.fov_deform:
            cfg.MODEL.weights_net_saliency = os.path.join(
                cfg.DIR, 'saliency_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            assert os.path.exists(cfg.MODEL.weights_net_saliency), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed) # Python
    torch.manual_seed(cfg.TRAIN.seed) # pytorch cpu vars
    np.random.seed(cfg.TRAIN.seed)  # cpu vars

    # if cuda:
    torch.cuda.manual_seed(cfg.TRAIN.seed)
    torch.cuda.manual_seed_all(cfg.TRAIN.seed)  # pytorch gpu vars
    torch.backends.cudnn.deterministic = True   # needed
    torch.backends.cudnn.benchmark = False


    main(cfg, gpus)
