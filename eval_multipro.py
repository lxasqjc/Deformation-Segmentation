# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat
# from sklearn.metrics import confusion_matrix
# Our libs
from config import cfg
from dataset import ValDataset, imresize, b_imresize, patch_loader
from models import ModelBuilder, SegmentationModule, SegmentationModule_plain, SegmentationModule_fov_deform, FovSegmentationModule, DeformSegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger, confusion_matrix, hd95
from criterion import OhemCrossEntropy, DiceCoeff, DiceLoss, FocalLoss
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


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
    Image.fromarray(im_vis).save(os.path.join(dir_result, os.path.splitext(img_name)[0] + '_seg.png'))

def visualize_result_fov(data, foveated_expection, dir_result):
    (img, F_Xlr, info) = data

    # aggregate images and save
    im_vis = np.concatenate((img, foveated_expection),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(F_Xlr.astype(np.uint8),mode='L').save(os.path.join(dir_result, os.path.splitext(img_name)[0] + '_F_Xlr.png'))
    Image.fromarray(im_vis).save(os.path.join(dir_result, os.path.splitext(img_name)[0] + '_fov_exp.png'))
def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue, foveation_module=None):
    segmentation_module.eval()
    if cfg.MODEL.foveation:
        foveation_module.eval()
    patch_bank = list((float(cfg.VAL.expand_prediection_rate_patch)*np.array(cfg.MODEL.patch_bank)).astype(int))
    # initialize a confusion matrix
    confusion = np.zeros((cfg.DATASET.num_class, cfg.DATASET.num_class))
    for batch_data in loader:
        # process data
        if batch_data is None:
            continue

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
        if not cfg.VAL.test:
            if cfg.VAL.batch_size > 1:
                seg_label = as_numpy(batch_data['seg_label'])
            else:
                seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        img_resized_list_unnorm = batch_data['img_data_unnorm']

        if cfg.VAL.visualize and cfg.MODEL.foveation and cfg.VAL.foveated_expection:
            foveated_expection = torch.zeros(batch_data['img_ori'].shape)
            foveated_expection_temp = torch.cat([foveated_expection.unsqueeze(0), foveated_expection.unsqueeze(0)])
            foveated_expection_weight =  torch.zeros(foveated_expection_temp.shape[0:-1]) # 2,w,h

        with torch.no_grad():
            segSize = (seg_label.shape[-2], seg_label.shape[-1])
            bs = img_resized_list[0].shape[0]
            scores = torch.zeros(bs, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores_tmp = torch.zeros(bs, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)
            scores_tmp = async_copy_to(scores_tmp, gpu_id)

            if cfg.VAL.max_score:
                scores_tmp_2 = torch.cat([scores_tmp.unsqueeze(0), scores_tmp.unsqueeze(0)])
                scores_tmp_2 = async_copy_to(scores_tmp_2, gpu_id)

            if cfg.VAL.approx_pred_Fxlr_by_ensemble or cfg.VAL.F_Xlr_low_scale != 0:
                fov_map_scale_temp = cfg.MODEL.fov_map_scale
                if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                    scores_ensemble = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
                    scores_ensemble = async_copy_to(scores_ensemble, gpu_id)
                    approx_pred_Fxlr_iter = len(patch_bank)
                # create fake feed_dict
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img_resized_list[0]
                feed_dict['img_data_unnorm'] = img_resized_list_unnorm[0]
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)
                # get F_Xlr at original high resolution fov_map_scale # b,d,w,h
                X = feed_dict['img_data'] # NOTE only support test image = 1
                fov_map_scale = cfg.MODEL.fov_map_scale
                X_lr = b_imresize(X, (round(X.shape[2]/fov_map_scale), round(X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))), interp='bilinear')
                feed_dict['cor_info'] = (tuple([0]), tuple([0]))
                if cfg.VAL.visualize:
                    patch_data, F_Xlr, Y_patch_cord, X_patches_cords, X_patches_unnorm = foveation_module(feed_dict, train_mode=False)
                else:
                    patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                # patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                F_Xlr_ori = F_Xlr.clone()
                print(F_Xlr.size())
                # scale F_Xlr to size of score b,d,W,H
                if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                    F_Xlr_scale = b_imresize(F_Xlr, (segSize[0], segSize[1]), interp='nearest')
                if cfg.VAL.F_Xlr_low_scale != 0:
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!Fist detect F_Xlr_low_scale')
                    F_Xlr_low_res = b_imresize(F_Xlr, (round(X.shape[2]/cfg.VAL.F_Xlr_low_scale), round(X.shape[3]/(cfg.VAL.F_Xlr_low_scale*cfg.MODEL.patch_ap))), interp='bilinear')
                    cfg.MODEL.fov_map_scale = cfg.VAL.F_Xlr_low_scale
                    approx_pred_Fxlr_iter = 1
                    # print('cfg.VAL.F_Xlr_low_scale:', cfg.VAL.F_Xlr_low_scale)
            else:
                approx_pred_Fxlr_iter = 1

            for pred_iter in range(approx_pred_Fxlr_iter):
                if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                    cfg.MODEL.fov_map_scale = patch_bank[0]
                    cfg.MODEL.one_hot_patch = [0]*len(patch_bank)
                    cfg.MODEL.one_hot_patch[pred_iter] = 1
                for idx in range(len(img_resized_list)):
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img_resized_list[idx]
                    feed_dict['img_data_unnorm'] = img_resized_list_unnorm[idx]
                    if cfg.VAL.F_Xlr_low_scale != 0:
                        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!ADD')
                        feed_dict['F_Xlr_low_res'] = F_Xlr_low_res
                        # print('F_Xlr_low_res_size:', feed_dict['F_Xlr_low_res'].size())
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, gpu_id)

                    # Foveation
                    if cfg.MODEL.foveation:
                        X, Y = feed_dict['img_data'], feed_dict['seg_label']
                        X_unnorm = feed_dict['img_data_unnorm']
                        with torch.no_grad():
                            patch_segSize = (patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                            patch_scores = torch.zeros(cfg.VAL.batch_size, cfg.DATASET.num_class, patch_segSize[0], patch_segSize[1])
                            patch_scores = async_copy_to(patch_scores, gpu_id)

                        fov_map_scale = cfg.MODEL.fov_map_scale
                        # NOTE: although here we use batch imresize yet in practical batch size for X = 1
                        X_lr = b_imresize(X, (round(X.shape[2]/fov_map_scale), round(X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))), interp='bilinear')

                        # foveation (crop as you go)
                        if cfg.VAL.F_Xlr_only:
                            feed_dict['cor_info'] = (tuple([0]), tuple([0]))
                            patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                        else:
                            if cfg.VAL.F_Xlr_acc_map_only:
                                Xlr_miou_map = torch.zeros(X_lr.shape[2], X_lr.shape[3])
                                Xlr_loss_map = torch.zeros(X_lr.shape[2], X_lr.shape[3])
                            print('Percentage of fov_location finished:')
                            pbar_X_lr = tqdm(total=X_lr.shape[2])
                            for xi in range(X_lr.shape[2]):
                                for yi in range(X_lr.shape[3]):
                                    # feed_dict['cor_info'] = (xi, yi)
                                    feed_dict['cor_info'] = (tuple([xi]), tuple([yi]))
                                    if cfg.VAL.visualize:
                                        patch_data, F_Xlr, Y_patch_cord, X_patches_cords, X_patches_unnorm = foveation_module(feed_dict, train_mode=False)
                                    else:
                                        patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                                    # TODO: foveation (pre_cropped available)

                                    if cfg.VAL.F_Xlr_acc_map_only:
                                        patch_scores, patch_loss = segmentation_module(patch_data, segSize=patch_segSize, F_Xlr_acc_map=cfg.VAL.F_Xlr_acc_map_only)
                                        _, patch_pred = torch.max(patch_scores, dim=1)
                                        # w,h
                                        patch_pred = as_numpy(patch_pred.squeeze(0).cpu())

                                        # calculate accuracy and SEND THEM TO MASTER
                                        # acc, pix = accuracy(pred, seg_label)
                                        if 'CITYSCAPES' in cfg.DATASET.root_dataset:
                                            intersection, union, area_lab = intersectionAndUnion(patch_pred, patch_data['seg_label'].squeeze(0).cpu(), cfg.DATASET.num_class, ignore_index=20-1)
                                        else:
                                            if cfg.DATASET.ignore_index != -2:
                                                intersection, union, area_lab = intersectionAndUnion(patch_pred, patch_data['seg_label'].squeeze(0).cpu(), cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
                                            else:
                                                intersection, union, area_lab = intersectionAndUnion(patch_pred, patch_data['seg_label'].squeeze(0).cpu(), cfg.DATASET.num_class)
                                        patch_iou = intersection.sum() / union.sum()
                                        Xlr_miou_map[xi,yi] = patch_iou
                                        Xlr_loss_map[xi,yi] = patch_loss
                                        continue
                                    else:
                                        patch_scores = segmentation_module(patch_data, segSize=patch_segSize)

                                    cx_Y, cy_Y, patch_size_Y, p_y_w, p_y_h = Y_patch_cord
                                    if cfg.MODEL.fov_padding:
                                        # p_y = max(patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                                        scores_tmp_pad = torch.zeros(scores_tmp.shape)
                                        scores_tmp_pad = F.pad(scores_tmp_pad, (p_y_w,p_y_w,p_y_h,p_y_h))
                                        scores_tmp_pad = async_copy_to(scores_tmp_pad, gpu_id)
                                        # print('scores_tmp_pad shape: ', scores_tmp_pad.shape)
                                    patch_size_Y_x = patch_size_Y
                                    patch_size_Y_y = patch_size_Y*cfg.MODEL.patch_ap
                                    if not cfg.VAL.max_score:
                                        if cfg.MODEL.fov_padding:
                                            scores_tmp_pad = scores_tmp_pad*0
                                            scores_tmp_pad[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores.clone()
                                            scores_tmp = torch.add(scores_tmp, scores_tmp_pad[:, :, p_y_h:scores_tmp_pad.shape[2]-p_y_h, p_y_w:scores_tmp_pad.shape[3]-p_y_w])
                                        else:
                                            scores_tmp[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = torch.add(scores_tmp[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y], patch_scores)
                                    else:
                                        if cfg.MODEL.fov_padding:
                                            scores_tmp_pad = scores_tmp_pad*0
                                            scores_tmp_pad[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores
                                            scores_tmp_2[1] = scores_tmp_pad[:, :, p_y_h:scores_tmp_pad.shape[2]-p_y_h, p_y_w:scores_tmp_pad.shape[3]-p_y_w]
                                        else:
                                            scores_tmp_2[1, :, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores
                                        max_class_scores_tmp_2_0, _ = torch.max(scores_tmp_2[0], dim=1)
                                        max_class_scores_tmp_2_1, _ = torch.max(scores_tmp_2[1], dim=1)
                                        # 2,B,W,H, B=1
                                        max_class_scores_tmp_2 = torch.cat([max_class_scores_tmp_2_0.unsqueeze(0), max_class_scores_tmp_2_1.unsqueeze(0)])
                                        # get patch idx of max(max(score))
                                        # patch_idx_by_score.shape = B,W,H; B=1
                                        _, patch_idx_by_score = torch.max(max_class_scores_tmp_2, dim=0)
                                        scores_tmp_2_patch_idx = patch_idx_by_score.unsqueeze(1).unsqueeze(0).expand(scores_tmp_2.shape)
                                        scores_tmp_2[0] = scores_tmp_2.gather(0, scores_tmp_2_patch_idx)[0]
                                        scores_tmp_2[1] = torch.zeros(scores_tmp_2[0].shape)

                                    if cfg.VAL.visualize:
                                        if cfg.VAL.central_crop:
                                            cx_0, cy_0, patch_size_0, p_y_w, p_y_h = Y_patch_cord
                                        if cfg.VAL.hard_max_fov:
                                            weight_s, max_s = torch.max(F_Xlr[0,:,xi,yi], dim=0)
                                            if cfg.MODEL.hard_fov or cfg.MODEL.categorical:
                                                max_s = 0
                                            cx, cy, patch_size, p_w, p_h = X_patches_cords[max_s]
                                            X_patch = b_imresize(X_patches_unnorm[:,max_s,:,:,:], (patch_size, patch_size), interp='nearest')
                                            X_patch = X_patch[0]
                                            print('X_patch_shape: ', X_patch.shape)
                                            # c,w,h
                                            weighed_patch = X_patch.permute(1,2,0).cpu()
                                            # w,h
                                            patch_weight = weight_s.unsqueeze(-1).expand(*weighed_patch.shape[0:-1])
                                        else: # soft fov - max_score=False mode not currently supported
                                            cx_w, cy_w, patch_size_w = 0, 0, 0
                                            for i in range(len(X_patches_cords)):
                                                cx, cy, patch_size, p_w, p_h = X_patches_cords[i]
                                                w = F_Xlr[0,i,xi,yi]
                                                cx_w += w*cx
                                                cy_w += w*cy
                                                patch_size_w += w*patch_size
                                            cx, cy, patch_size = int(cx_w), int(cy_w), int(patch_size_w)
                                            # patch_size = int(torch.sum(F_Xlr[0,:,xi,yi] * torch.FloatTensor(cfg.MODEL.patch_bank)))
                                            if cfg.MODEL.fov_padding:
                                                fov_map_scale = cfg.MODEL.fov_map_scale
                                                # p = patch_size
                                                cx_p = xi*fov_map_scale + patch_size_Y//2 - patch_size//2 + p_h
                                                cy_p = yi*(fov_map_scale*cfg.MODEL.patch_ap) + patch_size_Y//2 - patch_size//2 + p_w
                                                X_unnorm_pad = F.pad(X_unnorm, (p_w,p_w,p_h,p_h))
                                                crop_patch = X_unnorm_pad[:, :, cx_p:cx_p+patch_size, cy_p:cy_p+patch_size]
                                            else:
                                                crop_patch = X_unnorm[:, :, cx:cx+patch_size, cy:cy+patch_size]
                                            X_patch = b_imresize(crop_patch, (patch_size_0,patch_size_0), interp='bilinear')
                                            X_patch = b_imresize(X_patch, (patch_size, patch_size), interp='nearest')
                                            X_patch = X_patch[0]
                                            print('X_patch_shape: ', X_patch.shape)
                                            # c,w,h
                                            weighed_patch = X_patch.permute(1,2,0).cpu()

                                        if cfg.VAL.foveated_expection:
                                            if cfg.MODEL.fov_padding:
                                                fov_map_scale = cfg.MODEL.fov_map_scale
                                                # p = patch_size
                                                cx_p = xi*fov_map_scale + patch_size_Y//2 - patch_size//2 + p_h
                                                cy_p = yi*(fov_map_scale*cfg.MODEL.patch_ap) + patch_size_Y//2 - patch_size//2 + p_w
                                                # C,W,H
                                                foveated_expection_temp_pad = torch.zeros(foveated_expection_temp.shape[3],foveated_expection_temp.shape[1],foveated_expection_temp.shape[2])
                                                foveated_expection_temp_pad = F.pad(foveated_expection_temp_pad, (p_w,p_w,p_h,p_h))
                                                # print('foveated_expection_temp_pad:', foveated_expection_temp_pad.shape)
                                                # print('cx_p, cy_p, patch_size:', cx_p, cy_p, patch_size)
                                                # W,H,C
                                                foveated_expection_temp_pad = foveated_expection_temp_pad.permute(1,2,0)
                                                foveated_expection_temp_pad[cx_p:cx_p+patch_size, cy_p:cy_p+patch_size, :] = weighed_patch
                                                foveated_expection_temp[1] = foveated_expection_temp_pad[p_h:-p_h, p_w:-p_w, :]
                                                if cfg.VAL.central_crop:
                                                    # p_y = max(patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                                                    foveated_expection_temp_pad_y = foveated_expection_temp[1].clone() # W,H,C
                                                    foveated_expection_temp_pad_y = foveated_expection_temp_pad_y.permute(2,0,1) # C,W,H
                                                    foveated_expection_temp_pad_y = F.pad(foveated_expection_temp_pad_y, (p_y_w,p_y_w,p_y_h,p_y_h))
                                                    foveated_expection_temp_temp = foveated_expection_temp_pad_y[:, cx_0:cx_0+patch_size_0, cy_0:cy_0+patch_size_0].clone()
                                                    foveated_expection_temp_pad_y = foveated_expection_temp_pad_y*0
                                                    foveated_expection_temp_pad_y[:, cx_0:cx_0+patch_size_0, cy_0:cy_0+patch_size_0] = foveated_expection_temp_temp
                                                    foveated_expection_temp_pad_y = foveated_expection_temp_pad_y.permute(1,2,0) # W,H,C
                                                    foveated_expection_temp[1] = foveated_expection_temp_pad_y[p_y_h:foveated_expection_temp_pad_y.shape[0]-p_y_h, p_y_w:foveated_expection_temp_pad_y.shape[1]-p_y_w, :]
                                                    print('max: ', torch.max(foveated_expection_temp_temp))
                                                    print('min: ', torch.min(foveated_expection_temp_temp))
                                                    # if torch.min(foveated_expection_temp_temp) == 0:
                                                    #     print(foveated_expection_temp_temp)
                                                    #     raise Exception('weighted patch may wrong')
                                                if cfg.VAL.hard_max_fov:
                                                    # W,H
                                                    foveated_expection_weight_pad = torch.zeros(foveated_expection_temp_pad.shape[0:-1])
                                                    foveated_expection_weight_pad[cx_p:cx_p+patch_size, cy_p:cy_p+patch_size] = patch_weight
                                                    foveated_expection_weight[1] = foveated_expection_weight_pad[p_h:-p_h, p_w:-p_w]
                                            else:
                                                # W,H,C
                                                foveated_expection_temp[1, cx:cx+patch_size, cy:cy+patch_size, :] = weighed_patch
                                                if cfg.VAL.hard_max_fov:
                                                    # W,H
                                                    foveated_expection_weight[1, cx:cx+patch_size, cy:cy+patch_size] = patch_weight
                                            if cfg.VAL.hard_max_fov:
                                                foveated_expection_weight[0], max_w_idx = torch.max(foveated_expection_weight, dim=0)
                                            if not cfg.VAL.max_score:
                                                max_w_idx = max_w_idx.unsqueeze(0).unsqueeze(-1).expand(*foveated_expection_temp.shape)
                                                # max_w_idx_w = max_w_idx.unsqueeze(0).expand(*foveated_expection_weight.shape)
                                                foveated_expection = foveated_expection_temp.gather(0, max_w_idx)[0]
                                            else:
                                                max_s_idx = patch_idx_by_score.unsqueeze(-1).expand(*foveated_expection_temp.shape).cpu()
                                                foveated_expection = foveated_expection_temp.gather(0, max_s_idx)[0]

                                            # foveated_expection_weight[0] = foveated_expection_weight.gather(0, max_w_idx_w).squeeze(0)
                                            foveated_expection_temp[0] = foveated_expection
                                            foveated_expection_temp[1] = torch.zeros(foveated_expection_temp[0].shape)
                                            if cfg.VAL.hard_max_fov:
                                                foveated_expection_weight[1] = torch.zeros(foveated_expection_weight[0].shape)



                                pbar_X_lr.update(1)
                                # print('{}/{} foveate points, xi={}, yi={}\n'.format(xi*X_lr.shape[3]+yi, X_lr.shape[2]*X_lr.shape[3], xi, yi))
                            if cfg.VAL.max_score:
                                scores_tmp = scores_tmp_2[0]
                        # print('F_Xlr: ', F_Xlr.shape)
                        # print(F_Xlr)
                    # non foveation mode
                    else:
                        # forward pass
                        scores_tmp, deformed_score, y_sampled = segmentation_module(feed_dict, segSize=segSize)
                    if scores_tmp.shape != scores.shape:
                        print('scores_tmp shape: {}\n'.format(scores_tmp.shape))
                        print('scores shape: {}\n'.format(scores.shape))
                    scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                    y_sampled = as_numpy(y_sampled.squeeze(0))
                    if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                        scores_ensemble = scores_ensemble + scores * F_Xlr_scale[:,pred_iter,:,:]

            if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                scores = scores_ensemble
                cfg.MODEL.fov_map_scale = fov_map_scale_temp
            if cfg.VAL.F_Xlr_low_scale != 0:
                cfg.MODEL.fov_map_scale = fov_map_scale_temp
                F_Xlr = F_Xlr_ori
            if cfg.VAL.ensemble:
                if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'scores')):
                    os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'scores'))
                np.save(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'scores', batch_data['info'].split('/')[-1]), scores.cpu())
            if cfg.VAL.F_Xlr_acc_map_only:
                if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'Xlr_miou_map')):
                    os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'Xlr_miou_map'))
                np.save(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'Xlr_miou_map', batch_data['info'].split('/')[-1]), Xlr_miou_map.cpu())

                if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'Xlr_loss_map')):
                    os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'Xlr_loss_map'))
                np.save(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'Xlr_loss_map', batch_data['info'].split('/')[-1]), Xlr_loss_map.cpu())

            _, pred = torch.max(scores, dim=1)
            # w,h
            pred = as_numpy(pred.squeeze(0).cpu())
            # print('pred shape: {}\n'.format(scores.shape))
            _, pred_deformed = torch.max(deformed_score, dim=1)
            pred_deformed = as_numpy(pred_deformed.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        acc_deformed, pix_deformed = accuracy(pred_deformed, y_sampled)
        if 'CITYSCAPES' in cfg.DATASET.root_dataset:
            intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=20-1)
            intersection_deformed, union_deformed, area_lab_deformed = intersectionAndUnion(pred_deformed, y_sampled, cfg.DATASET.num_class, ignore_index=20-1)
        else:
            if cfg.DATASET.ignore_index != -2:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
                intersection_deformed, union_deformed, area_lab_deformed = intersectionAndUnion(pred_deformed, y_sampled, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
            else:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
                intersection_deformed, union_deformed, area_lab_deformed = intersectionAndUnion(pred_deformed, y_sampled, cfg.DATASET.num_class)
        # calculate Hausdorff distance
        h_dists = []
        if cfg.VAL.hd95:
            if 'CITYSCAPES' in cfg.DATASET.root_dataset:
                ig_class = 19
            else:
                ig_class = cfg.DATASET.ignore_index
            for cur_class in range(cfg.DATASET.num_class):
                if cur_class == ig_class:
                    h_dists.append(np.nan)
                    continue
                pred_c = pred.copy()
                seg_label_c = seg_label.copy()
                mask_pred_b = pred_c != cur_class
                mask_pred_f = pred_c == cur_class
                pred_c[mask_pred_b] = 0
                pred_c[mask_pred_f] = 1

                mask_seg_label_b = seg_label_c != cur_class
                mask_seg_label_f = seg_label_c == cur_class
                seg_label_c[mask_seg_label_b] = 0
                seg_label_c[mask_seg_label_f] = 1

                if (pred_c == 1).sum() > 1 and (seg_label_c == 1).sum() > 1:
                    dist_ = hd95(pred_c, seg_label_c)
                    h_dists.append(dist_)
                else:
                    h_dists.append(np.nan)

        if 'CITYSCAPES' in cfg.DATASET.root_dataset:
            confusion += confusion_matrix(seg_label, pred, seg_label.shape, cfg.DATASET.num_class, ignore=20-1)
        else:
            if cfg.DATASET.ignore_index != -2:
                confusion += confusion_matrix(seg_label, pred, seg_label.shape, cfg.DATASET.num_class, ignore=cfg.DATASET.ignore_index)
            else:
                confusion += confusion_matrix(seg_label, pred, seg_label.shape, cfg.DATASET.num_class)
        if cfg.MODEL.foveation:
            if cfg.MODEL.gumbel_softmax:
                F_Xlr = F_Xlr.exp()
            F_Xlr_cp = F_Xlr.clone()
            F_Xlr_score = as_numpy(F_Xlr.clone().cpu())


            patch_bank_F_Xlr = torch.tensor(patch_bank).to(F_Xlr.device)
            F_Xlr = patch_bank_F_Xlr.unsqueeze(-1).unsqueeze(-1).float()*(F_Xlr.squeeze(0)).float()
            F_Xlr = as_numpy(F_Xlr.cpu())
            # t,b,d,w,h
            F_Xlr = np.sum(F_Xlr,axis=0)
            F_Xlr = np.expand_dims(F_Xlr,axis=0)
            if cfg.VAL.all_F_Xlr_time:
                F_Xlr_info = (F_Xlr, batch_data['info'].split('/')[-1].split('.')[0], F_Xlr_score)
                result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab, F_Xlr_info, h_dists))
            else:
                result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab, F_Xlr, F_Xlr_score, h_dists))
        else:
            # result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab))
            result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab, acc_deformed, pix_deformed, intersection_deformed, union_deformed))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint))
                )
            if cfg.MODEL.foveation and cfg.VAL.foveated_expection:
                foveated_expection = as_numpy(foveated_expection.cpu())
                # b,d,w,h
                F_Xlr = F_Xlr_cp
                F_Xlr = (F_Xlr-torch.min(F_Xlr))/(torch.max(F_Xlr)-torch.min(F_Xlr))
                F_Xlr = b_imresize((1-F_Xlr), (segSize[0], segSize[1]), interp='nearest')
                # b,d,w,h -> d,w,h -> w,h,d
                F_Xlr = as_numpy(F_Xlr.squeeze(0).permute(1,2,0).cpu())

                for idx in range(F_Xlr.shape[2]):
                    F_Xlr[:,:,idx] = F_Xlr[:,:,idx]*(255//F_Xlr.shape[2]*idx)
                F_Xlr = np.sum(F_Xlr,axis=2)
                visualize_result_fov(
                    (batch_data['img_ori'], F_Xlr, batch_data['info']),
                    foveated_expection*255,
                    os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint))
                )

def evaluate_train(segmentation_module, loader, cfg, gpu_id, result_queue, foveation_module=None):
    segmentation_module.eval()
    if cfg.MODEL.foveation:
        foveation_module.eval()
    patch_bank = list((float(cfg.VAL.expand_prediection_rate_patch)*np.array(cfg.MODEL.patch_bank)).astype(int))
    # initialize a confusion matrix
    confusion = np.zeros((cfg.DATASET.num_class, cfg.DATASET.num_class))
    for batch_data in loader:
        # process data
        if batch_data is None:
            continue
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        img_resized_list_unnorm = batch_data['img_data_unnorm']
        # note for foveation resize not applied, i.e. both seg_label and img_data are at original size
        if cfg.VAL.visualize and cfg.MODEL.foveation:
            foveated_expection = torch.zeros(batch_data['img_ori'].shape)
            foveated_expection_temp = torch.cat([foveated_expection.unsqueeze(0), foveated_expection.unsqueeze(0)])
            foveated_expection_weight =  torch.zeros(foveated_expection_temp.shape[0:-1]) # 2,w,h

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(cfg.VAL.batch_size, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores_tmp = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])

            if cfg.VAL.max_score:
                scores_tmp_2 = torch.cat([scores_tmp.unsqueeze(0), scores_tmp.unsqueeze(0)])
                # scores_tmp_2 = async_copy_to(scores_tmp_2, gpu_id)

            for idx in range(len(img_resized_list)):
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img_resized_list[idx]
                feed_dict['img_data_unnorm'] = img_resized_list_unnorm[idx]
                del feed_dict['img_ori']
                del feed_dict['info']

                # Foveation
                if cfg.MODEL.foveation:
                    X, Y = feed_dict['img_data'], feed_dict['seg_label']
                    X_unnorm = feed_dict['img_data_unnorm']
                    with torch.no_grad():
                        patch_segSize = (patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                        patch_scores = torch.zeros(cfg.VAL.batch_size, cfg.DATASET.num_class, patch_segSize[0], patch_segSize[1])

                    fov_map_scale = cfg.MODEL.fov_map_scale
                    # NOTE: although here we use batch imresize yet in practical batch size for X = 1
                    X_lr = b_imresize(X, (round(X.shape[2]/fov_map_scale), round(X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))), interp='bilinear')

                    # foveation (crop as you go)
                    pbar_X_lr = tqdm(total=X_lr.shape[2])
                    for xi in range(X_lr.shape[2]):
                        for yi in range(X_lr.shape[3]):
                            # feed_dict['cor_info'] = (xi, yi)
                            feed_dict['cor_info'] = (tuple([xi]), tuple([yi]))
                            if cfg.VAL.visualize:
                                patch_data, F_Xlr, Y_patch_cord, X_patches_cords, X_patches_unnorm = foveation_module(feed_dict, train_mode=False)
                            else:
                                patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                            patch_scores = segmentation_module(patch_data, segSize=patch_segSize)

                            cx_Y, cy_Y, patch_size_Y, p_y_w, p_y_h = Y_patch_cord
                            if cfg.MODEL.fov_padding:
                                # p_y = max(patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                                scores_tmp_pad = torch.zeros(scores_tmp.shape)
                                scores_tmp_pad = F.pad(scores_tmp_pad, (p_y_w,p_y_w,p_y_h,p_y_h))
                            patch_size_Y_x = patch_size_Y
                            patch_size_Y_y = patch_size_Y*cfg.MODEL.patch_ap
                            if not cfg.VAL.max_score:
                                if cfg.MODEL.fov_padding:
                                    scores_tmp_pad = scores_tmp_pad*0
                                    scores_tmp_pad[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores.clone()
                                    scores_tmp = torch.add(scores_tmp, scores_tmp_pad[:, :, p_y_h:scores_tmp_pad.shape[2]-p_y_h, p_y_w:scores_tmp_pad.shape[3]-p_y_w])
                                else:
                                    scores_tmp[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = torch.add(scores_tmp[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y], patch_scores)
                            else:
                                if cfg.MODEL.fov_padding:
                                    scores_tmp_pad = scores_tmp_pad*0
                                    scores_tmp_pad[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores
                                    scores_tmp_2[1] = scores_tmp_pad[:, :, p_y_h:scores_tmp_pad.shape[2]-p_y_h, p_y_w:scores_tmp_pad.shape[3]-p_y_w]
                                else:
                                    scores_tmp_2[1, :, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores
                                max_class_scores_tmp_2_0, _ = torch.max(scores_tmp_2[0], dim=1)
                                max_class_scores_tmp_2_1, _ = torch.max(scores_tmp_2[1], dim=1)
                                # 2,B,W,H, B=1
                                max_class_scores_tmp_2 = torch.cat([max_class_scores_tmp_2_0.unsqueeze(0), max_class_scores_tmp_2_1.unsqueeze(0)])
                                _, patch_idx_by_score = torch.max(max_class_scores_tmp_2, dim=0)
                                scores_tmp_2_patch_idx = patch_idx_by_score.unsqueeze(1).unsqueeze(0).expand(scores_tmp_2.shape)
                                scores_tmp_2[0] = scores_tmp_2.gather(0, scores_tmp_2_patch_idx)[0]
                                scores_tmp_2[1] = torch.zeros(scores_tmp_2[0].shape)

                            if cfg.VAL.visualize:
                                if cfg.VAL.central_crop:
                                    cx_0, cy_0, patch_size_0, p_y_w, p_y_h = Y_patch_cord
                                if cfg.VAL.hard_max_fov:
                                    weight_s, max_s = torch.max(F_Xlr[0,:,xi,yi], dim=0)
                                    cx, cy, patch_size, p_w, p_h = X_patches_cords[max_s]
                                    X_patch = b_imresize(X_patches_unnorm[:,max_s,:,:,:], (patch_size, patch_size), interp='nearest')
                                    X_patch = X_patch[0]
                                    print('X_patch_shape: ', X_patch.shape)
                                    # c,w,h
                                    weighed_patch = X_patch.permute(1,2,0).cpu()
                                    # w,h
                                    patch_weight = weight_s.unsqueeze(-1).expand(*weighed_patch.shape[0:-1])
                                else: # soft fov - max_score=False mode not currently supported
                                    cx_w, cy_w, patch_size_w = 0, 0, 0
                                    for i in range(len(X_patches_cords)):
                                        cx, cy, patch_size, p_w, p_h = X_patches_cords[i]
                                        w = F_Xlr[0,i,xi,yi]
                                        cx_w += w*cx
                                        cy_w += w*cy
                                        patch_size_w += w*patch_size
                                    cx, cy, patch_size = int(cx_w), int(cy_w), int(patch_size_w)
                                    # patch_size = int(torch.sum(F_Xlr[0,:,xi,yi] * torch.FloatTensor(cfg.MODEL.patch_bank)))
                                    if cfg.MODEL.fov_padding:
                                        fov_map_scale = cfg.MODEL.fov_map_scale
                                        # p = patch_size
                                        cx_p = xi*fov_map_scale + patch_size_Y//2 - patch_size//2 + p_h
                                        cy_p = yi*(fov_map_scale*cfg.MODEL.patch_ap) + patch_size_Y//2 - patch_size//2 + p_w
                                        X_unnorm_pad = F.pad(X_unnorm, (p_w,p_w,p_h,p_h))
                                        crop_patch = X_unnorm_pad[:, :, cx_p:cx_p+patch_size, cy_p:cy_p+patch_size]
                                    else:
                                        crop_patch = X_unnorm[:, :, cx:cx+patch_size, cy:cy+patch_size]
                                    X_patch = b_imresize(crop_patch, (patch_size_0,patch_size_0), interp='bilinear')
                                    X_patch = b_imresize(X_patch, (patch_size, patch_size), interp='nearest')
                                    X_patch = X_patch[0]
                                    print('X_patch_shape: ', X_patch.shape)
                                    # c,w,h
                                    weighed_patch = X_patch.permute(1,2,0).cpu()


                                if cfg.MODEL.fov_padding:
                                    fov_map_scale = cfg.MODEL.fov_map_scale
                                    # p = patch_size
                                    cx_p = xi*fov_map_scale + patch_size_Y//2 - patch_size//2 + p_h
                                    cy_p = yi*(fov_map_scale*cfg.MODEL.patch_ap) + patch_size_Y//2 - patch_size//2 + p_w
                                    # C,W,H
                                    foveated_expection_temp_pad = torch.zeros(foveated_expection_temp.shape[3],foveated_expection_temp.shape[1],foveated_expection_temp.shape[2])
                                    foveated_expection_temp_pad = F.pad(foveated_expection_temp_pad, (p_w,p_w,p_h,p_h))
                                    # W,H,C
                                    foveated_expection_temp_pad = foveated_expection_temp_pad.permute(1,2,0)
                                    foveated_expection_temp_pad[cx_p:cx_p+patch_size, cy_p:cy_p+patch_size, :] = weighed_patch
                                    foveated_expection_temp[1] = foveated_expection_temp_pad[p_h:-p_h, p_w:-p_w, :]
                                    if cfg.VAL.central_crop:
                                        # p_y = max(patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                                        foveated_expection_temp_pad_y = foveated_expection_temp[1].clone() # W,H,C
                                        foveated_expection_temp_pad_y = foveated_expection_temp_pad_y.permute(2,0,1) # C,W,H
                                        foveated_expection_temp_pad_y = F.pad(foveated_expection_temp_pad_y, (p_y_w,p_y_w,p_y_h,p_y_h))
                                        foveated_expection_temp_temp = foveated_expection_temp_pad_y[:, cx_0:cx_0+patch_size_0, cy_0:cy_0+patch_size_0].clone()
                                        foveated_expection_temp_pad_y = foveated_expection_temp_pad_y*0
                                        foveated_expection_temp_pad_y[:, cx_0:cx_0+patch_size_0, cy_0:cy_0+patch_size_0] = foveated_expection_temp_temp
                                        foveated_expection_temp_pad_y = foveated_expection_temp_pad_y.permute(1,2,0) # W,H,C
                                        foveated_expection_temp[1] = foveated_expection_temp_pad_y[p_y_h:foveated_expection_temp_pad_y.shape[0]-p_y_h, p_y_w:foveated_expection_temp_pad_y.shape[1]-p_y_w, :]
                                        print('max: ', torch.max(foveated_expection_temp_temp))
                                        print('min: ', torch.min(foveated_expection_temp_temp))
                                    if cfg.VAL.hard_max_fov:
                                        # W,H
                                        foveated_expection_weight_pad = torch.zeros(foveated_expection_temp_pad.shape[0:-1])
                                        foveated_expection_weight_pad[cx_p:cx_p+patch_size, cy_p:cy_p+patch_size] = patch_weight
                                        foveated_expection_weight[1] = foveated_expection_weight_pad[p_h:-p_h, p_w:-p_w]
                                else:
                                    # W,H,C
                                    foveated_expection_temp[1, cx:cx+patch_size, cy:cy+patch_size, :] = weighed_patch
                                    if cfg.VAL.hard_max_fov:
                                        # W,H
                                        foveated_expection_weight[1, cx:cx+patch_size, cy:cy+patch_size] = patch_weight
                                if cfg.VAL.hard_max_fov:
                                    foveated_expection_weight[0], max_w_idx = torch.max(foveated_expection_weight, dim=0)
                                if not cfg.VAL.max_score:
                                    max_w_idx = max_w_idx.unsqueeze(0).unsqueeze(-1).expand(*foveated_expection_temp.shape)
                                    # max_w_idx_w = max_w_idx.unsqueeze(0).expand(*foveated_expection_weight.shape)
                                    foveated_expection = foveated_expection_temp.gather(0, max_w_idx)[0]
                                else:
                                    max_s_idx = patch_idx_by_score.unsqueeze(-1).expand(*foveated_expection_temp.shape).cpu()
                                    foveated_expection = foveated_expection_temp.gather(0, max_s_idx)[0]

                                # foveated_expection_weight[0] = foveated_expection_weight.gather(0, max_w_idx_w).squeeze(0)
                                foveated_expection_temp[0] = foveated_expection
                                foveated_expection_temp[1] = torch.zeros(foveated_expection_temp[0].shape)
                                if cfg.VAL.hard_max_fov:
                                    foveated_expection_weight[1] = torch.zeros(foveated_expection_weight[0].shape)

                        # print('{}/{} foveate points, xi={}, yi={}'.format(xi*X_lr.shape[3]+yi, X_lr.shape[2]*X_lr.shape[3], xi, yi))
                        pbar_X_lr.update(1)
                    if cfg.VAL.max_score:
                        scores_tmp = scores_tmp_2[0]
                else:
                    # forward pass
                    scores_tmp = segmentation_module(feed_dict, segSize=segSize)

                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            # w,h
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        if 'CITYSCAPES' in cfg.DATASET.root_dataset:
            intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=20-1)
        else:
            if cfg.DATASET.ignore_index != -2:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
            else:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)

        # flatten for confusion confusion_matrix
        pred_flatten = pred.flatten()
        seg_label_flatten = seg_label.flatten()
        # calculate the confusion matrix and add to the accumulated matrix
        if 'CITYSCAPES' in cfg.DATASET.root_dataset:
            confusion += confusion_matrix(seg_label, pred, seg_label.shape, cfg.DATASET.num_class, ignore=20-1)
        else:
            if cfg.DATASET.ignore_index != -2:
                confusion += confusion_matrix(seg_label, pred, seg_label.shape, cfg.DATASET.num_class, ignore=cfg.DATASET.ignore_index)
            else:
                confusion += confusion_matrix(seg_label, pred, seg_label.shape, cfg.DATASET.num_class)
        if cfg.MODEL.foveation:

            patch_bank_F_Xlr = torch.tensor(patch_bank).to(F_Xlr.device)
            F_Xlr = patch_bank_F_Xlr.unsqueeze(-1).unsqueeze(-1).float()*(F_Xlr.squeeze(0)).float()
            F_Xlr = as_numpy(F_Xlr.cpu())
            # t,b,d,w,h
            F_Xlr = np.sum(F_Xlr,axis=0)
            F_Xlr = np.expand_dims(F_Xlr,axis=0)
            if cfg.VAL.all_F_Xlr_time:
                F_Xlr_info = (F_Xlr, batch_data['info'].split('/')[-1].split('.')[0])
                result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab, F_Xlr_info))
            else:
                result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab, F_Xlr))
        else:
            result_queue.put_nowait((acc, pix, intersection, union, confusion, area_lab))

        # visualization
        if cfg.VAL.visualize:
            # seg_label = convert_label(label=seg_label, inverse=True)
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint))
                )
            if cfg.MODEL.foveation:
                # if not cfg.VAL.hard_max_fov:
                #     foveated_expection = foveated_expection / overlap_count
                foveated_expection = as_numpy(foveated_expection.cpu())
                # b,d,w,h
                F_Xlr = (F_Xlr-torch.min(F_Xlr))/(torch.max(F_Xlr)-torch.min(F_Xlr))
                F_Xlr = b_imresize((1-F_Xlr), (segSize[0], segSize[1]), interp='nearest')
                # b,d,w,h -> d,w,h -> w,h,d
                F_Xlr = as_numpy(F_Xlr.squeeze(0).permute(1,2,0).cpu())

                for idx in range(F_Xlr.shape[2]):
                    F_Xlr[:,:,idx] = F_Xlr[:,:,idx]*(255//F_Xlr.shape[2]*idx)
                F_Xlr = np.sum(F_Xlr,axis=2)
                visualize_result_fov(
                    (batch_data['img_ori'], F_Xlr, batch_data['info']),
                    foveated_expection*255,
                    os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint))
                )

def worker_train(cfg, gpu_id, start_idx, end_idx, result_queue):
    # torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

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
    if cfg.MODEL.foveation:
        net_foveater = ModelBuilder.build_foveater(
        in_channel=cfg.MODEL.in_dim,
        out_channel=len(cfg.MODEL.patch_bank),
        weights=cfg.MODEL.weights_foveater,
        cfg=cfg)

    if 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=20-1)
    elif 'Digest' in cfg.DATASET.root_dataset:
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
    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)


    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)

    # Main loop
    if cfg.MODEL.foveation:
        evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue, foveation_module)
    else:
        evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)


def worker_deform(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

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
    if cfg.MODEL.fov_deform:
        net_saliency = ModelBuilder.build_net_saliency(
            cfg=cfg,
            weights=cfg.MODEL.weights_net_saliency)
        net_compress = ModelBuilder.build_net_compress(
            cfg=cfg,
            weights=cfg.MODEL.weights_net_compress)

    if cfg.MODEL.foveation:
        net_foveater = ModelBuilder.build_foveater(
        in_channel=cfg.MODEL.in_dim,
        out_channel=len(cfg.MODEL.patch_bank),
        weights=cfg.MODEL.weights_foveater,
        cfg=cfg)


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

    if cfg.MODEL.fov_deform:
        segmentation_module = DeformSegmentationModule(net_encoder, net_decoder, net_saliency, net_compress, crit, cfg)
    else:
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg)
    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)

    segmentation_module.cuda()
    if cfg.MODEL.foveation:
        foveation_module.cuda()

    # if not os.path.exists(os.path.join(cfg.DIR, 'network_summary.txt')):
    f = open(os.path.join(cfg.DIR, 'network_summary.txt'), 'w')
    if cfg.MODEL.foveation:
        print(foveation_module, file = f)
    print(segmentation_module, file = f)

    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)
        total_fov = sum([param.nelement() for param in foveation_module.parameters()])
        print('Number of FoveationModule params: %.2fM \n' % (total_fov / 1e6))

    total = sum([param.nelement() for param in segmentation_module.parameters()])
    f.write('Number of SegmentationModule params: %.2fM \n' % (total / 1e6))

    # Main loop
    if cfg.MODEL.foveation:
        evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue, foveation_module)
    else:
        evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)
    f.write('Max memory allocated: %.2fM' % (torch.cuda.max_memory_allocated() / 1e6))
    f.close()


def worker(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

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
    if cfg.MODEL.foveation:
        net_foveater = ModelBuilder.build_foveater(
        in_channel=cfg.MODEL.in_dim,
        out_channel=len(cfg.MODEL.patch_bank),
        weights=cfg.MODEL.weights_foveater,
        cfg=cfg)


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
    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)

    segmentation_module.cuda()
    if cfg.MODEL.foveation:
        foveation_module.cuda()

    # if not os.path.exists(os.path.join(cfg.DIR, 'network_summary.txt')):
    f = open(os.path.join(cfg.DIR, 'network_summary.txt'), 'w')
    if cfg.MODEL.foveation:
        print(foveation_module, file = f)
    print(segmentation_module, file = f)

    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)
        total_fov = sum([param.nelement() for param in foveation_module.parameters()])
        print('Number of FoveationModule params: %.2fM \n' % (total_fov / 1e6))

    total = sum([param.nelement() for param in segmentation_module.parameters()])
    f.write('Number of SegmentationModule params: %.2fM \n' % (total / 1e6))

    # Main loop
    if cfg.MODEL.foveation:
        evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue, foveation_module)
    else:
        evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)
    f.write('Max memory allocated: %.2fM' % (torch.cuda.max_memory_allocated() / 1e6))
    f.close()


def eval_during_train_multipro(cfg, gpus):

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    # load foveation weights
    if cfg.MODEL.foveation:
        weights=cfg.MODEL.weights_foveater = os.path.join(
            cfg.DIR, 'foveater_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_foveater), "checkpoint does not exitst!"
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    if cfg.VAL.all_F_Xlr_time:
        F_Xlr_all = []
    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker_train, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        if cfg.MODEL.foveation:
            if cfg.VAL.all_F_Xlr_time:
                (acc, pix, intersection, union, confusion, area_lab, F_Xlr_info) = result_queue.get()
            else:
                (acc, pix, intersection, union, confusion, area_lab, F_Xlr) = result_queue.get()
        else:
            (acc, pix, intersection, union, confusion, area_lab) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        if cfg.VAL.all_F_Xlr_time:
            F_Xlr_all.append(F_Xlr_info)
        processed_counter += 1*cfg.VAL.batch_size
        pbar.update(1*cfg.VAL.batch_size)

    for p in procs:
        p.join()

    # summary
    # and write result_log.txt file
    # f_result= open(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "result_log.txt"),"w+")

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))
        # f_result.write('class [{}], IoU: {:.4f}\n'.format(i, _iou))

    print('[Eval Summary]:')
    # f_result.write('[Eval Summary]:\n')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%\n'
          .format(iou.mean(), acc_meter.average()*100))
    # f_result.write('Mean IoU: {:.4f}, Accuracy: {:.2f}%\n'
          # .format(iou.mean(), acc_meter.average()*100))
    if cfg.MODEL.foveation:
        if cfg.VAL.all_F_Xlr_time:
            return iou.mean(), acc_meter.average()*100, F_Xlr_all
        else:
            return iou.mean(), acc_meter.average()*100, F_Xlr
    else:
        return iou.mean(), acc_meter.average()*100

def main(cfg, gpus):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    area_lab_meter = AverageMeter()
    confusion_meter = np.zeros((cfg.DATASET.num_class, cfg.DATASET.num_class))
    if cfg.VAL.hd95:
        h_dists_meter = []
    if cfg.VAL.all_F_Xlr_time:
        F_Xlr_all = []
    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        if cfg.MODEL.fov_deform:
            proc = Process(target=worker_deform, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        else:
            proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        if cfg.MODEL.foveation:
            if cfg.VAL.all_F_Xlr_time:
                (acc, pix, intersection, union, confusion, area_lab, F_Xlr_info, h_dists) = result_queue.get()
            else:
                (acc, pix, intersection, union, confusion, area_lab, F_Xlr, F_Xlr_score, h_dists) = result_queue.get()
        elif cfg.MODEL.fov_deform:
            (acc, pix, intersection, union, confusion, area_lab, acc_deformed, pix_deformed, intersection_deformed, union_deformed) = result_queue.get()
        else:
            (acc, pix, intersection, union, confusion, area_lab) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        area_lab_meter.update(area_lab)
        if cfg.VAL.all_F_Xlr_time:
            F_Xlr_all.append(F_Xlr_info)
        confusion_meter = confusion_meter + confusion
        if cfg.VAL.hd95:
            h_dists_meter.append(h_dists)
        processed_counter += 1*cfg.VAL.batch_size
        pbar.update(1*cfg.VAL.batch_size)

    for p in procs:
        p.join()

    # summary
    # and write result_log.txt file
    f_result= open(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "result_log.txt"),"w+")

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    if cfg.VAL.hd95:
        h_dists_mean = np.nanmean(np.array(h_dists_meter), axis=0)

    pos = confusion_meter.sum(1)
    res = confusion_meter.sum(0)
    tp = np.diag(confusion_meter)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))
        f_result.write('class [{}], IoU: {:.4f}\n'.format(i, _iou))
    for i, _iou in enumerate(IoU_array):
        print('class_cm [{}], IoU_cm: {:.4f}'.format(i, _iou))
        f_result.write('class_cm [{}], IoU_cm: {:.4f}\n'.format(i, _iou))
    acc_class = intersection_meter.sum / (area_lab_meter.sum + 1e-10) * 100
    for i, _class_acc in enumerate(acc_class):
        print('class [{}], class_acc: {:.4f}'.format(i, _class_acc))
        f_result.write('class [{}], class_acc: {:.4f}\n'.format(i, _class_acc))

    if cfg.VAL.dice:
        dice = (2 * intersection_meter.sum) / (union_meter.sum + intersection_meter.sum + 1e-10)
        for i, _dice in enumerate(dice):
            print('class [{}], Dice: {:.4f}'.format(i, _dice))
            f_result.write('class [{}], Dice: {:.4f}\n'.format(i, _dice))

    if cfg.VAL.hd95:
        for i, _class_h_dists in enumerate(h_dists_mean):
            print('class [{}], class_h_dists: {:.4f}'.format(i, _class_h_dists))
            f_result.write('class [{}], class_h_dists: {:.4f}\n'.format(i, _class_h_dists))


    print('[Eval Summary]:')
    f_result.write('[Eval Summary]:\n')
    if cfg.VAL.dice:
        if cfg.VAL.hd95:
            print('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Mean Dice: {:.4f}, Mean h_dists: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, dice.mean(), np.nanmean(h_dists_mean), acc_meter.average()*100))
            f_result.write('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Mean Dice: {:.4f}, Mean h_dists: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, dice.mean(), np.nanmean(h_dists_mean), acc_meter.average()*100))
        else:
            print('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Mean Dice: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, dice.mean(), acc_meter.average()*100))
            f_result.write('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Mean Dice: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, dice.mean(), acc_meter.average()*100))
    else:
        if cfg.VAL.hd95:
            print('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Mean h_dists: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, np.nanmean(h_dists_mean), acc_meter.average()*100))
            f_result.write('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Mean h_dists: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, np.nanmean(h_dists_mean), acc_meter.average()*100))
        else:
            print('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, acc_meter.average()*100))
            f_result.write('Mean IoU: {:.4f}, Mean IoU_cm: {:.4f}, Accuracy: {:.2f}%\n'
                  .format(iou.mean(), mean_IoU, acc_meter.average()*100))

    print('IoU Confusion Matrix:\n')
    print(confusion_meter)
    cm = confusion_meter / np.maximum(1.0, pos + res - tp)
    # cm = cm / cm.sum(axis=1)[:, np.newaxis]
    cm = np.around(cm, decimals=3)
    np.savetxt(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "confusion_matrix.csv"), cm, delimiter=",")

    if cfg.MODEL.foveation:
        if cfg.VAL.all_F_Xlr_time:
            for val_idx in range(len(F_Xlr_all)):
                print('F_Xlr_{}'.format(F_Xlr_all[val_idx][1]), F_Xlr_all[val_idx][0].shape)
                if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "F_Xlr_all_vals")):
                    os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "F_Xlr_all_vals"))
                np.save('{}/F_Xlr_last_{}.npy'.format(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "F_Xlr_all_vals"), F_Xlr_all[val_idx][1]), F_Xlr_all[val_idx][0])

                if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "F_Xlr_all_scores")):
                    os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "F_Xlr_all_scores"))
                np.save('{}/F_Xlr_last_{}.npy'.format(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), "F_Xlr_all_scores"), F_Xlr_all[val_idx][1]), F_Xlr_all[val_idx][-1])
        else:
            print('F_Xlr_time', F_Xlr.shape)
            np.save('{}/F_Xlr_time_last.npy'.format(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint)), F_Xlr))
            np.save('{}/F_Xlr_score_last.npy'.format(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint)), F_Xlr))

    print('Evaluation Done!')
    f_result.close()


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/foveation-cityscape-hrnetv2.yaml",
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

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    # load foveation weights
    if cfg.MODEL.foveation:
        cfg.MODEL.weights_foveater = os.path.join(
            cfg.DIR, 'foveater_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_foveater), "checkpoint does not exitst!"
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if cfg.MODEL.fov_deform:
        # absolute paths of seg_deform weights
        cfg.MODEL.weights_net_saliency = os.path.join(
            cfg.DIR, 'saliency_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_net_saliency), "weights_net_saliency checkpoint does not exitst!"

        cfg.MODEL.weights_net_compress = os.path.join(
            cfg.DIR, 'compress_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_net_compress), "weights_net_compress checkpoint does not exitst!"


    if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint))):
        os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint)))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
