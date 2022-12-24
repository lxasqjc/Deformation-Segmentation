# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat
import pandas as pd
# Our libs
from config import cfg
from dataset import ValDataset, imresize, b_imresize
from models import ModelBuilder, SegmentationModule, DeformSegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from criterion import OhemCrossEntropy, DiceCoeff, DiceLoss, FocalLoss
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from PIL import ImageFilter
from scipy import ndimage
from tqdm import tqdm

from saliency_network import saliency_network_resnet18, saliency_network_resnet10_nonsyn, saliency_network_resnet18_nonsyn, fov_simple, saliency_network_resnet18_stride1

colors = loadmat('data/color150.mat')['colors']

def trim_accuracy(preds, label, dia_factor, cfg):
    valid = (label >= 0)
    trimap_img = {}

    seg_label_norm = (label - label.min())/(label.max() - label.min())
    seg_label_grey = np.array((seg_label_norm*255)).astype(np.uint8)
    seg_label_img = Image.fromarray(seg_label_grey, 'L')
    seg_label_Edges = seg_label_img.filter(ImageFilter.FIND_EDGES)
    seg_label_Edges_ay = np.array(seg_label_Edges.convert('L'))
    seg_label_Edges_ay[(seg_label_Edges_ay > 0)] = 1

    for i in range(dia_factor+1):
        seg_label_Edges_dil = ndimage.binary_dilation(seg_label_Edges_ay, iterations=2**i)
        acc_sum = (valid * seg_label_Edges_dil * (preds == label)).sum()
        valid_sum = (valid * seg_label_Edges_dil).sum()
        acc = float(acc_sum) / (valid_sum + 1e-10)
        trimap_img['trim_width_{}_acc'.format(str(2**i))] = acc
        if cfg.VAL.trimap_visual_check:
            if not os.path.isdir(os.path.join(cfg.DIR, 'trimap_visual_check')):
                os.makedirs(os.path.join(cfg.DIR, 'trimap_visual_check'))
            np.save('{}/seg_label_Edges_dil_{}.npy'.format(os.path.join(cfg.DIR, 'trimap_visual_check'), 2**i), seg_label_Edges_dil)
            masked_label = (valid * seg_label_Edges_dil * label)
            masked_pred = (valid * seg_label_Edges_dil * preds)
            np.save('{}/masked_label_dil_{}.npy'.format(os.path.join(cfg.DIR, 'trimap_visual_check'), 2**i), masked_label)
            np.save('{}/masked_pred_dil_{}.npy'.format(os.path.join(cfg.DIR, 'trimap_visual_check'), 2**i), masked_pred)

    return trimap_img


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu, foveation_module=None, writer=None, count=None):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    acc_meter_deformed = AverageMeter()
    intersection_meter_deformed = AverageMeter()
    union_meter_deformed = AverageMeter()

    eval_y_distribution_meter = AverageMeter()
    eval_y_sampled_distribution_meter = AverageMeter()

    if cfg.VAL.report_per_img_iou:
        img_iou_dic = {'image_name': [], 'image_mIoU': []}
        for c in range(cfg.DATASET.num_class):
            img_iou_dic['img_iou_class_'+str(c)] = []

    if cfg.VAL.trimap:
        trimap_all = {}
        for w in range(cfg.VAL.trimap_dia_factor+1):
            trimap_all['trim_width_{}_acc'.format(str(2**w))] = []

    if cfg.VAL.y_sampled_reverse:
        acc_meter_y_reverse = AverageMeter()
        intersection_meter_y_reverse = AverageMeter()
        union_meter_y_reverse = AverageMeter()
    
    # Add creation of 'result' folder for visualization in case it doesn't exist
    if cfg.VAL.visualize:
        if not os.path.isdir(os.path.join(cfg.DIR, "result")):
            os.makedirs(os.path.join(cfg.DIR, "result"))

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    feed_batch_count = 0
    for batch_data in loader:
        # print('feed_batch_count: {}\n'.format(feed_batch_count))
        if feed_batch_count == (len(loader)//cfg.TRAIN.num_gpus-1):
            feed_batch_count = -1
        if cfg.VAL.batch_size > 1:
            info_list = []
            for b in range(len(batch_data)): # over all batches
                info_list.append(batch_data[b]['info'])
                if b == 0:
                    continue
                if not cfg.VAL.test:
                    batch_data[0]['seg_label'] = torch.cat([batch_data[0]['seg_label'], batch_data[b]['seg_label']])
                for s in range(len(batch_data[0]['img_data'])): # over all resized images
                    batch_data[0]['img_data'][s] = torch.cat([batch_data[0]['img_data'][s], batch_data[b]['img_data'][s]])

            batch_data[0]['info'] = info_list

        batch_data = batch_data[0]
        if cfg.VAL.no_upsample:
            segm = Image.fromarray(as_numpy(batch_data['seg_label'][0]).astype('uint8'))
            segm_down = imresize(
                segm,
                (segm.size[0] // cfg.DATASET.segm_downsampling_rate, \
                 segm.size[1] // cfg.DATASET.segm_downsampling_rate), \
                interp='nearest')
            seg_label = as_numpy(segm_down)
        else:
            if cfg.VAL.batch_size > 1:
                seg_label = as_numpy(batch_data['seg_label'])
            else:
                seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[-2], seg_label.shape[-1])
            bs = img_resized_list[0].shape[0]
            scores = torch.zeros(bs, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                feed_dict_info = feed_dict['info']
                if not cfg.TRAIN.train_eval_visualise:
                    del feed_dict['img_ori']
                feed_dict_copy = feed_dict['info']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                feed_dict['info'] = feed_dict_copy

                # forward pass
                if cfg.VAL.y_sampled_reverse:
                    scores_tmp, deformed_score, y_sampled, y_sampled_reverse = segmentation_module(feed_dict, segSize=segSize, count=count, writer=writer, feed_dict_info=feed_dict_info, feed_batch_count=feed_batch_count)
                else:
                    scores_tmp, deformed_score, y_sampled = segmentation_module(feed_dict, segSize=segSize, count=count, writer=writer, feed_dict_info=feed_dict_info, feed_batch_count=feed_batch_count)
                if scores_tmp is None and deformed_score is None and y_sampled is None:
                    continue
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                y_sampled = as_numpy(y_sampled.squeeze(0))

                (y_distribution_current, _) = np.histogram(
                    feed_dict['seg_label'].cpu(), bins=cfg.DATASET.num_class, range=(0, cfg.DATASET.num_class-1))
                (y_sampled_distribution_current, _) = np.histogram(
                    y_sampled, bins=cfg.DATASET.num_class, range=(0, cfg.DATASET.num_class-1))
                eval_y_distribution_meter.update((y_distribution_current / np.sum(y_distribution_current)).astype(float))
                eval_y_sampled_distribution_meter.update((y_sampled_distribution_current / np.sum(y_sampled_distribution_current)).astype(float))

            if scores_tmp is None and deformed_score is None and y_sampled is None:
                continue
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            _, pred_deformed = torch.max(deformed_score, dim=1)
            pred_deformed = as_numpy(pred_deformed.squeeze(0).cpu())
            if cfg.VAL.y_sampled_reverse:
                y_sampled_reverse = as_numpy(y_sampled_reverse.squeeze(0).cpu())
        if scores_tmp is None and deformed_score is None and y_sampled is None:
            continue
        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        acc_deformed, pix_deformed = accuracy(pred_deformed, y_sampled)
        if cfg.VAL.y_sampled_reverse:
            acc_y_reverse, pix_y_reverse = accuracy(y_sampled_reverse, seg_label)


        if 'CITYSCAPES' in cfg.DATASET.root_dataset or 'CITYSCAPE' in cfg.DATASET.list_train:
            intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=20-1)
            # print('pred_deformed shape: {}'.format(pred_deformed.shape))
            # print('y_sampled shape: {}'.format(y_sampled.shape))
            intersection_deformed, union_deformed, area_lab_deformed = intersectionAndUnion(pred_deformed, y_sampled, cfg.DATASET.num_class, ignore_index=20-1)
            if cfg.VAL.y_sampled_reverse:
                # assert (cfg.MODEL.rev_deform_interp == 'nearest'), "y_sampled_reverse only appliable to nearest rev_deform_interp"
                intersection_y_reverse, union_y_reverse, area_lab_y_reverse = intersectionAndUnion(y_sampled_reverse, seg_label, cfg.DATASET.num_class, ignore_index=20-1)
        else:
            if cfg.DATASET.ignore_index != -2:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
                intersection_deformed, union_deformed, area_lab_deformed = intersectionAndUnion(pred_deformed, y_sampled, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
                if cfg.VAL.y_sampled_reverse:
                    # assert (cfg.MODEL.rev_deform_interp == 'nearest'), "y_sampled_reverse only appliable to nearest rev_deform_interp"
                    intersection_y_reverse, union_y_reverse, area_lab_y_reverse = intersectionAndUnion(y_sampled_reverse, seg_label, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
            else:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
                intersection_deformed, union_deformed, area_lab_deformed = intersectionAndUnion(pred_deformed, y_sampled, cfg.DATASET.num_class)
                if cfg.VAL.y_sampled_reverse:
                    # assert (cfg.MODEL.rev_deform_interp == 'nearest'), "y_sampled_reverse only appliable to nearest rev_deform_interp"
                    intersection_y_reverse, union_y_reverse, area_lab_y_reverse = intersectionAndUnion(y_sampled_reverse, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_meter_deformed.update(acc_deformed, pix_deformed)
        intersection_meter_deformed.update(intersection_deformed)
        union_meter_deformed.update(union_deformed)
        if cfg.VAL.report_per_img_iou:
            img_iou = intersection / (union + 1e-10)
            img_name = feed_dict['info'].split('/')[-1]
            img_iou_dic['image_name'].append(img_name)
            img_iou_dic['image_mIoU'].append(img_iou.mean())
            for c in range(cfg.DATASET.num_class):
                img_iou_dic['img_iou_class_'+str(c)].append(img_iou[c])
        if cfg.VAL.trimap:
            assert cfg.VAL.batch_size == 1, "VAL.batch_size > 1 not currently supported"
            trimap_img = trim_accuracy(pred, seg_label, cfg.VAL.trimap_dia_factor, cfg)
            for w in range(cfg.VAL.trimap_dia_factor+1):
                trimap_all['trim_width_{}_acc'.format(str(2**w))].append(trimap_img['trim_width_{}_acc'.format(str(2**w))])


        if cfg.VAL.y_sampled_reverse:
            acc_meter_y_reverse.update(acc_y_reverse, pix_y_reverse)
            intersection_meter_y_reverse.update(intersection_y_reverse)
            # print('sample #-{} IoU: {}\n'.format(intersection_meter_y_reverse.count, intersection_y_reverse/union_y_reverse))
            union_meter_y_reverse.update(union_y_reverse)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)
        feed_batch_count += 1

    # summary
    if cfg.VAL.report_per_img_iou:
        data_frame = pd.DataFrame(
            data={'image_name': img_iou_dic['image_name']
                , 'image_mIoU': img_iou_dic['image_mIoU']
                  }
        )
        for c in range(cfg.DATASET.num_class):
            data_frame['img_iou_class_'+str(c)] = img_iou_dic['img_iou_class_'+str(c)]
        data_frame = data_frame.sort_values('image_mIoU')
        data_frame.to_csv('{}/image_IoUs_list.csv'.format(cfg.DIR),
                          index_label='idx')

    if cfg.VAL.trimap:
        print('Saving trimap...')

        trimap_save = {'trim_acc': [], 'trim_width': []}
        for w in range(cfg.VAL.trimap_dia_factor+1):
            print('trim_width_{} total count: {}'.format(2**w, len(trimap_all['trim_width_{}_acc'.format(str(2**w))])))
            trimap_save['trim_acc'].append((sum(trimap_all['trim_width_{}_acc'.format(str(2**w))]) / len(trimap_all['trim_width_{}_acc'.format(str(2**w))])))
            trimap_save['trim_width'].append(2**w)

        data_frame = pd.DataFrame(
            data={'trim_acc': trimap_save['trim_acc']
                , 'trim_width': trimap_save['trim_width']}
        )

        data_frame.to_csv('{}/trimap_last_count_{}.csv'.format(cfg.DIR, len(trimap_all['trim_width_{}_acc'.format(str(2**0))])),
                          index_label='idx')


    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    if cfg.VAL.dice:
        dice = (2 * intersection_meter.sum) / (union_meter.sum + intersection_meter.sum + 1e-10)
    iou_deformed = intersection_meter_deformed.sum / (union_meter_deformed.sum + 1e-10)
    if cfg.VAL.y_sampled_reverse:
        iou_y_reverse = intersection_meter_y_reverse.sum / (union_meter_y_reverse.sum + 1e-10)
    if cfg.VAL.dice:
        dice_deformed = (2 * intersection_meter_deformed.sum) / (union_meter_deformed.sum + intersection_meter_deformed.sum + 1e-10)
        if cfg.VAL.y_sampled_reverse:
            dice_y_reverse = (2 * intersection_meter_y_reverse.sum) / (union_meter_y_reverse.sum + intersection_meter_y_reverse.sum + 1e-10)
    writer.add_histogram('Eval Label Original distribution', eval_y_distribution_meter.average()*100.0, count)
    writer.add_histogram('Eval Deformed Label distribution', eval_y_sampled_distribution_meter.average()*100.0, count)
    writer.add_histogram('Eval Deformed Label Distribution - Label Original Distribution', eval_y_sampled_distribution_meter.average()*100.0 - eval_y_distribution_meter.average()*100.0, count)
    relative_eval_y_ysample = eval_y_sampled_distribution_meter.average()*100.0 - eval_y_distribution_meter.average()*100.0
    y_sampled_distribution = eval_y_sampled_distribution_meter.average()*100.0
    y_distribution = eval_y_distribution_meter.average()*100.0
    for i in range(len(relative_eval_y_ysample)):
        writer.add_scalars('Eval Deformed Label vs Label Original distribution Class {}'.format(i), {'Label Original distribution': y_distribution[i]}, count)
        writer.add_scalars('Eval Deformed Label vs Label Original distribution Class {}'.format(i), {'Deformed Label': y_sampled_distribution[i]}, count)


    print('[Eval Summary]:')
    print('intersection_meter: \n')
    print(intersection_meter)
    if cfg.VAL.dice:
        print('Mean IoU: {:.4f}, Mean Dice: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s, Valid Samples: {}'
          .format(iou.mean(), dice.mean(), acc_meter.average()*100, time_meter.average(), intersection_meter.count))
        print('Mean IoU_deformed: {:.4f}, Mean Dice_deformed: {:.4f}, Accuracy_deformed: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou_deformed.mean(), dice_deformed.mean(), acc_meter_deformed.average()*100, time_meter.average()))
        if cfg.VAL.y_sampled_reverse:
            print('Mean iou_y_reverse: {:.4f}, Mean dice_y_reverse: {:.4f}, Accuracy_y_reverse: {:.2f}%'
              .format(iou_y_reverse.mean(), dice_y_reverse.mean(), acc_meter_y_reverse.average()*100))
    else:
        print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s, Valid Samples: {}'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average(), intersection_meter.count))
        print('Mean IoU_deformed: {:.4f}, Accuracy_deformed: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou_deformed.mean(), acc_meter_deformed.average()*100, time_meter.average()))
        if cfg.VAL.y_sampled_reverse:
            print('Mean iou_y_reverse: {:.4f}, Accuracy_y_reverse: {:.2f}%'
              .format(iou_y_reverse.mean(), acc_meter_y_reverse.average()*100))

    if cfg.VAL.y_sampled_reverse:
        ious = [iou, iou_deformed, iou_y_reverse, intersection_meter.count]
    else:
        ious = [iou, iou_deformed, intersection_meter.count]

    # implemented for eval during trainig
    if cfg.VAL.dice:
        if cfg.VAL.y_sampled_reverse:
            return iou.mean(), dice.mean(), acc_meter.average()*100, iou_deformed.mean(), dice_deformed.mean(), acc_meter_deformed.average()*100, iou_y_reverse.mean(), dice_y_reverse.mean(), acc_meter_y_reverse.average()*100, relative_eval_y_ysample, ious
        else:
            return iou.mean(), dice.mean(), acc_meter.average()*100, iou_deformed.mean(), dice_deformed.mean(), acc_meter_deformed.average()*100, relative_eval_y_ysample, ious
    else:
        if cfg.VAL.y_sampled_reverse:
            return iou.mean(), acc_meter.average()*100, iou_deformed.mean(), acc_meter_deformed.average()*100, iou_y_reverse.mean(), acc_meter_y_reverse.average()*100, relative_eval_y_ysample, ious
        else:
            return iou.mean(), acc_meter.average()*100, iou_deformed.mean(), acc_meter_deformed.average()*100, relative_eval_y_ysample, ious

def eval_during_train_deform(cfg, writer=None, gpu=0, count=None):
    torch.cuda.set_device(gpu)

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    # absolute paths of seg_deform weights
    cfg.MODEL.weights_net_saliency = os.path.join(
        cfg.DIR, 'saliency_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_net_saliency), "checkpoint does not exitst!"
    net_saliency = ModelBuilder.build_net_saliency(
        cfg=cfg,
        weights=cfg.MODEL.weights_net_saliency)

    cfg.MODEL.weights_net_compress = os.path.join(
        cfg.DIR, 'compress_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_net_compress), "checkpoint does not exitst!"
    net_compress = ModelBuilder.build_net_compress(
        cfg=cfg,
        weights=cfg.MODEL.weights_net_compress)

    if 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=19)
    elif 'Digest' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=-2)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=-2)
    elif cfg.TRAIN.loss_fun == 'FocalLoss' and 'DeepGlob' in cfg.DATASET.root_dataset:
        crit = FocalLoss(gamma=6, ignore_label=cfg.DATASET.ignore_index)
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

    segmentation_module = DeformSegmentationModule(net_encoder, net_decoder, net_saliency, net_compress, crit, cfg)
    # develop check if parameters properly loaded
    for name, param in segmentation_module.net_compress.named_parameters():
        if 'conv_last.bias' in name:
            print ('EVAL\n')
            print (name, param.data)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    if cfg.VAL.dice:
        if cfg.VAL.y_sampled_reverse:
            mIoU, dice, acc, mIoU_deformed, dice_deformed, acc_deformed, mIoU_y_reverse, dice_y_reverse, acc_y_reverse, relative_eval_y_ysample, ious = evaluate(segmentation_module, loader_val, cfg, gpu, writer=writer, count=count)
        else:
            mIoU, dice, acc, mIoU_deformed, dice_deformed, acc_deformed, relative_eval_y_ysample, ious = evaluate(segmentation_module, loader_val, cfg, gpu, writer=writer, count=count)
    else:
        if cfg.VAL.y_sampled_reverse:
            mIoU, acc, mIoU_deformed, acc_deformed, mIoU_y_reverse, acc_y_reverse, relative_eval_y_ysample, ious = evaluate(segmentation_module, loader_val, cfg, gpu, writer=writer, count=count)
        else:
            mIoU, acc, mIoU_deformed, acc_deformed, relative_eval_y_ysample, ious = evaluate(segmentation_module, loader_val, cfg, gpu, writer=writer, count=count)

    print('Evaluation Done!')
    if cfg.VAL.dice:
        if cfg.VAL.y_sampled_reverse:
            return mIoU, dice, acc, mIoU_deformed, dice_deformed, acc_deformed, mIoU_y_reverse, dice_y_reverse, acc_y_reverse, relative_eval_y_ysample, ious
        else:
            return mIoU, dice, acc, mIoU_deformed, dice_deformed, acc_deformed, relative_eval_y_ysample, ious
    else:
        if cfg.VAL.y_sampled_reverse:
            return mIoU, acc, mIoU_deformed, acc_deformed, mIoU_y_reverse, acc_y_reverse, relative_eval_y_ysample, ious
        else:
            return mIoU, acc, mIoU_deformed, acc_deformed, relative_eval_y_ysample, ious

def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    if 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=19)
    elif 'Digest' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=-2)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=-2)
    elif cfg.TRAIN.loss_fun == 'FocalLoss' and 'DeepGlob' in cfg.DATASET.root_dataset:
        crit = FocalLoss(gamma=6, ignore_label=cfg.DATASET.ignore_index)
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

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/deform-cityscape.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
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

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    main(cfg, args.gpu)
