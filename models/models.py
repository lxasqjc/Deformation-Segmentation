import torch
import random
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
import torchsnooper
from . import resnet, resnext, mobilenet, hrnetv2_nodownsp
from lib.nn import SynchronizedBatchNorm2d
from dataset import imresize, b_imresize
from builtins import any as b_any

from lib.utils import as_numpy
from utils import colorEncode
from scipy.io import loadmat
import numpy as np
from PIL import Image
from PIL import ImageFilter
import time
import os

from scipy import ndimage
import scipy.interpolate
import cv2
import torchvision.models as models
from saliency_network import saliency_network_resnet18, fov_simple, saliency_network_resnet18_stride1
from models.model_utils import Resnet, ResnetDilated, MobileNetV2Dilated, C1DeepSup, C1, PPM, PPMDeepsup, UPerNet

BatchNorm2d = SynchronizedBatchNorm2d

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def fillMissingValues_tensor(target_for_interp, copy=False, interp_mode='tri'):
    """
    fill missing values in a tenor

    input shape: [num_classes, h, w]
    output shape: [num_classes, h, w]
    """

    if copy:
        target_for_interp = target_for_interp.clone()

    def getPixelsForInterp(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.

        input shape: [num_classes, h, w]
        output shape: [num_classes, h, w]
        """

        invalid_mask = torch.isnan(img)
        kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), device=invalid_mask.device).unsqueeze(0).unsqueeze(0).expand(1,invalid_mask.shape[0],3,3).float()

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = invalid_mask.float().unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = input_scaled.unsqueeze(0) # b,c,w,h

            dilated_mask_scaled = torch.clamp(F.conv2d(invalid_mask_scaled, kernel, padding=(1, 1)), 0, 1)
            dilated_mask_scaled_t = dilated_mask_scaled.float()
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
        else:

            dilated_mask = torch.clamp(F.conv2d(invalid_mask.float().unsqueeze(0),
                                                kernel, padding=(1, 1)), 0, 1).squeeze(0)

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  (~invalid_mask).float()
        # Add 4 zeros corner points required for interp2d
        masked_for_interp[:,0,0] *= 0
        masked_for_interp[:,0,-1] *= 0
        masked_for_interp[:,-1,0] *= 0
        masked_for_interp[:,-1,-1] *= 0
        masked_for_interp[:,0,0] += 1
        masked_for_interp[:,0,-1] += 1
        masked_for_interp[:,-1,0] += 1
        masked_for_interp[:,-1,-1] += 1

        return masked_for_interp.bool(), invalid_mask

    def getPixelsForInterp_NB(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.
        """
        # mask invalid pixels
        invalid_mask = np.isnan(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = torch.tensor(invalid_mask.astype('float')).unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = np.array(input_scaled).astype('bool')
            dilated_mask_scaled = cv2.dilate(invalid_mask_scaled.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))
            dilated_mask_scaled_t = torch.tensor(dilated_mask_scaled.astype('float')).unsqueeze(0)
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
            dilated_mask = np.array(dilated_mask).astype('uint8')
        else:
            dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask

    # Mask pixels for interpolation
    if interp_mode == 'nearest':
        interpolator=scipy.interpolate.NearestNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    elif interp_mode == 'BI':
        interpolator=scipy.interpolate.LinearNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    else:
        interpolator=Interp2D(target_for_interp.shape[-2], target_for_interp.shape[-1])
        mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)
        if invalid_mask.float().sum() == 0:
            return target_for_interp

    if interp_mode == 'nearest' or interp_mode == 'BI':
        points = np.argwhere(mask_for_interp)
        values = target_for_interp[mask_for_interp]
    else:
        points = torch.where(mask_for_interp[0]) # tuple of 2 for (h, w) indices
        points = torch.cat([t.unsqueeze(0) for t in points]) # [2, number_of_points]
        points = points.permute(1,0) # shape: [number_of_points, 2]
        values = target_for_interp.clone()[mask_for_interp].view(mask_for_interp.shape[0],-1).permute(1,0) # shape: [number_of_points, num_classes]
    interp = interpolator(points, values) # return [num_classes, h, w]

    if interp_mode == 'nearest' or interp_mode == 'BI':
        target_for_interp[invalid_mask] = interp(np.argwhere(np.array(invalid_mask)))
    else:
        if not (interp.shape == target_for_interp.shape == invalid_mask.shape and interp.device == target_for_interp.device == invalid_mask.device):
            print('SHAPE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.shape, target_for_interp.shape, invalid_mask.shape))
            print('DEVICE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.device, target_for_interp.device, invalid_mask.device))
        try:
            target_for_interp[invalid_mask] = interp[torch.where(invalid_mask)].clone()
        except:
            print('interp: {}\n'.format(interp))
            print('invalid_mask: {}\n'.format(invalid_mask))
            print('target_for_interp: {}\n'.format(target_for_interp))
        else:
            pass
    return target_for_interp

def unorm(img):
    if 'GLEASON' in self.cfg.DATASET.list_train:
        mean=[0.748, 0.611, 0.823]
        std=[0.146, 0.245, 0.119]
    elif 'Digest' in self.cfg.DATASET.list_train:
        mean=[0.816, 0.697, 0.792]
        std=[0.160, 0.277, 0.198]
    elif 'ADE' in self.cfg.DATASET.list_train:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif 'CITYSCAPE' in self.cfg.DATASET.list_train or 'Cityscape' in self.cfg.DATASET.list_train:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif 'histo' in self.cfg.DATASET.list_train:
        mean=[0.8223, 0.7783, 0.7847]
        std=[0.210, 0.216, 0.241]
    elif 'DeepGlob' in self.cfg.DATASET.list_train:
        mean=[0.282, 0.379, 0.408]
        std=[0.089, 0.101, 0.127]
    elif 'Face_single_example' in self.cfg.DATASET.root_dataset or 'Face_single_example' in self.cfg.DATASET.list_train:
        mean=[0.282, 0.379, 0.408]
        std=[0.089, 0.101, 0.127]
    elif 'Histo' in self.cfg.DATASET.root_dataset or 'histomri' in self.cfg.DATASET.list_train or 'histomri' in self.cfg.DATASET.root_dataset:
        mean=[0.8223, 0.7783, 0.7847]
        std=[0.210, 0.216, 0.241]
    else:
        raise Exception('Unknown root for normalisation!')
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

class CompressNet(nn.Module):
    def __init__(self, cfg):
        super(CompressNet, self).__init__()
        if cfg.MODEL.saliency_net == 'fovsimple':
            self.conv_last = nn.Conv2d(24,1,kernel_size=1,padding=0,stride=1)
        else:
            self.conv_last = nn.Conv2d(256,1,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(x)
        out = self.conv_last(x)
        return out

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

class DeformSegmentationModule(SegmentationModuleBase):
    def __init__(self, net_encoder, net_decoder, net_saliency, net_compress, crit, cfg, deep_sup_scale=None):
        super(DeformSegmentationModule, self).__init__()
        self.encoder = net_encoder
        self.decoder = net_decoder
        self.localization = net_saliency
        self.crit = crit
        if cfg.TRAIN.opt_deform_LabelEdge or cfg.TRAIN.deform_joint_loss:
            self.crit_mse = nn.MSELoss()
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.print_original_y = True
        self.net_compress = net_compress
        if self.cfg.MODEL.saliency_output_size_short == 0:
            self.grid_size_x = cfg.TRAIN.saliency_input_size[0]
        else:
            self.grid_size_x = self.cfg.MODEL.saliency_output_size_short
        self.grid_size_y = cfg.TRAIN.saliency_input_size[1] // (cfg.TRAIN.saliency_input_size[0]//self.grid_size_x)
        self.padding_size_x = self.cfg.MODEL.gaussian_radius
        if self.cfg.MODEL.gaussian_ap == 0.0:
            gaussian_ap = cfg.TRAIN.saliency_input_size[1] // cfg.TRAIN.saliency_input_size[0]
        else:
            gaussian_ap = self.cfg.MODEL.gaussian_ap
        self.padding_size_y = int(gaussian_ap * self.padding_size_x)
        self.global_size_x = self.grid_size_x+2*self.padding_size_x
        self.global_size_y = self.grid_size_y+2*self.padding_size_y
        self.input_size = cfg.TRAIN.saliency_input_size
        self.input_size_net = cfg.TRAIN.task_input_size
        self.input_size_net_eval = cfg.TRAIN.task_input_size_eval
        if len(self.input_size_net_eval) == 0:
            self.input_size_net_infer = self.input_size_net
        else:
            self.input_size_net_infer = self.input_size_net_eval
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size_x+1, fwhm = self.cfg.MODEL.gaussian_radius)) # TODO: redo seneitivity experiments on gaussian radius as corrected fwhm (effective gaussian radius)
        gaussian_weights = b_imresize(gaussian_weights.unsqueeze(0).unsqueeze(0), (2*self.padding_size_x+1,2*self.padding_size_y+1), interp='bilinear')
        gaussian_weights = gaussian_weights.squeeze(0).squeeze(0)

        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size_x+1,2*self.padding_size_y+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y).cuda()
        # initialization of u(x,y),v(x,y) range from 0 to 1
        for k in range(2):
            for i in range(self.global_size_x):
                for j in range(self.global_size_y):
                    self.P_basis[k,i,j] = k*(i-self.padding_size_x)/(self.grid_size_x-1.0)+(1.0-k)*(j-self.padding_size_y)/(self.grid_size_y-1.0)

        self.save_print_grad = [{'saliency_grad': 0.0, 'check1_grad': 0.0, 'check2_grad': 0.0} for _ in range(cfg.TRAIN.num_gpus)]

    def re_initialise(self, cfg, this_size):# dealing with varying input image size such as pcahisto dataset
        this_size_short = min(this_size)
        this_size_long = max(this_size)
        scale_task_rate_1 = this_size_short // min(cfg.TRAIN.dynamic_task_input)
        scale_task_rate_2 = this_size_long // max(cfg.TRAIN.dynamic_task_input)
        scale_task_size_1 = tuple([int(x//scale_task_rate_1) for x in this_size])
        scale_task_size_2 = tuple([int(x//scale_task_rate_2) for x in this_size])
        if scale_task_size_1[0]*scale_task_size_1[1] < scale_task_size_2[0]*scale_task_size_2[1]:
            scale_task_size = scale_task_size_1
        else:
            scale_task_size = scale_task_size_2

        cfg.TRAIN.task_input_size = scale_task_size
        cfg.TRAIN.saliency_input_size = tuple([int(x*cfg.TRAIN.dynamic_saliency_relative_size) for x in scale_task_size])

        if self.cfg.MODEL.saliency_output_size_short == 0:
            self.grid_size_x = cfg.TRAIN.saliency_input_size[0]
        else:
            self.grid_size_x = self.cfg.MODEL.saliency_output_size_short
        self.grid_size_y = cfg.TRAIN.saliency_input_size[1] // (cfg.TRAIN.saliency_input_size[0]//self.grid_size_x)
        self.global_size_x = self.grid_size_x+2*self.padding_size_x
        self.global_size_y = self.grid_size_y+2*self.padding_size_y

        self.input_size = cfg.TRAIN.saliency_input_size
        self.input_size_net = cfg.TRAIN.task_input_size

        if len(self.input_size_net_eval) == 0:
            self.input_size_net_infer = self.input_size_net
        else:
            self.input_size_net_infer = self.input_size_net_eval
        self.P_basis = torch.zeros(2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y).cuda()
        # initialization of u(x,y),v(x,y) range from 0 to 1
        for k in range(2):
            for i in range(self.global_size_x):
                for j in range(self.global_size_y):
                    self.P_basis[k,i,j] = k*(i-self.padding_size_x)/(self.grid_size_x-1.0)+(1.0-k)*(j-self.padding_size_y)/(self.grid_size_y-1.0)

    def create_grid(self, x, segSize=None, x_inv=None):
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y, device=x.device),requires_grad=False)

        P[0,:,:,:] = self.P_basis.to(x.device) # [1,2,w,h], 2 corresponds to u(x,y) and v(x,y)
        P = P.expand(x.size(0),2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y)
        # input x is saliency map xs
        x_cat = torch.cat((x,x),1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size_x,self.global_size_y)
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size_x,self.grid_size_y)
        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size_x,self.grid_size_y)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size_x,self.grid_size_y)
        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter
        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)
        xgrids = xgrids.view(-1,1,self.grid_size_x,self.grid_size_y)
        ygrids = ygrids.view(-1,1,self.grid_size_x,self.grid_size_y)
        grid = torch.cat((xgrids,ygrids),1)

        if len(self.input_size_net_eval) != 0 and segSize is not None:# inference
            grid = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(grid)
        else:
            grid = nn.Upsample(size=self.input_size_net, mode='bilinear')(grid)
        if segSize is None:# training
            grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(grid)
        else:# inference
            grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net_infer)), mode='bilinear')(grid)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        grid_y = torch.transpose(grid_y,1,2)
        grid_y = torch.transpose(grid_y,2,3)

        #### inverse deformation
        if segSize is not None and x_inv is not None:
            grid_reorder = grid.permute(3,0,1,2)
            grid_inv = torch.autograd.Variable(torch.zeros((2,grid_reorder.shape[1],segSize[0],segSize[1]), device=grid_reorder.device))
            grid_inv[:] = float('nan')
            u_cor = (((grid_reorder[0,:,:,:]+1)/2)*(segSize[1]-1)).int().long().view(grid_reorder.shape[1],-1)
            v_cor = (((grid_reorder[1,:,:,:]+1)/2)*(segSize[0]-1)).int().long().view(grid_reorder.shape[1],-1)
            x_cor = torch.arange(0,grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
            x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
            y_cor = torch.arange(0,grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
            y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
            grid_inv[0][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(x_cor)
            grid_inv[1][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(y_cor)
            grid_inv[0] = grid_inv[0]/grid_reorder.shape[3]*2-1
            grid_inv[1] = grid_inv[1]/grid_reorder.shape[2]*2-1
            grid_inv = grid_inv.permute(1,2,3,0)
            return grid, grid_inv
        else:
            return grid, grid_y

    def ignore_label(self, label, ignore_indexs):
        label = np.array(label)
        temp = label.copy()
        for k in ignore_indexs:
            label[temp == k] = 0
        return label

    def forward(self, feed_dict, *, writer=None, segSize=None, F_Xlr_acc_map=False, count=None, epoch=None, feed_dict_info=None, feed_batch_count=None):

        # EXPLAIN: re initialise apply only when input image has varying size, e.g. pcahisto dataset
        if self.cfg.TRAIN.dynamic_task_input[0] != 1:
            this_size = tuple(feed_dict['img_data'].shape[-2:])
            print('this_size: {}'.format(this_size))
            self.re_initialise(self.cfg, this_size)
            print('task_input_size after re_initialise: {}'.format(self.input_size_net))
            print('saliency_input_size after re_initialise: {}'.format(self.input_size))

        # EXPLAIN: for each high-resolution image X
        x = feed_dict['img_data']
        del feed_dict['img_data']
        t = time.time()
        ori_size = (x.shape[-2],x.shape[-1])

        # EXPLAIN: compute its lower resolution version Xlr
        x_low = b_imresize(x, self.input_size, interp='bilinear')
        epoch = self.cfg.TRAIN.global_epoch
        if segSize is None and ((self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss) and epoch <= self.cfg.TRAIN.deform_pretrain):
            min_saliency_len = min(self.input_size)
            s = random.randint(min_saliency_len//3, min_saliency_len)
            x_low = nn.AdaptiveAvgPool2d((s,s))(x_low)
            x_low = nn.Upsample(size=self.input_size,mode='bilinear')(x_low)

        # EXPLAIN: calculate deformation/saliency map d=Dθ(Xlr)
        xs = self.localization(x_low)
        xs = self.net_compress(xs)
        xs = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(xs)
        xs = xs.view(-1,self.grid_size_x*self.grid_size_y) # N,1,W,H
        xs = nn.Softmax()(xs) # N,W*H
        xs = xs.view(-1,1,self.grid_size_x,self.grid_size_y) #saliency map

        y = feed_dict['seg_label'].clone()

        # EXPLAIN: calculate the target deformation map dt = fedge(fgaus(Ylr)) from the uniformly downsampled segmentation labelYlr
        if self.cfg.MODEL.gt_gradient or (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
        # EXPLAIN: if motivational study (gt_gradient) or uniform downsample (uniform_sample == 'BI')
            xsc = xs.clone().detach()
            for j in range(y.shape[0]):
                if segSize is not None:
                    # i.e. if training
                    (y_j_dist, _) = np.histogram(y[j].cpu(), bins=2, range=(0, 1))
                if self.cfg.MODEL.fix_gt_gradient and not (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
                    # EXPLAIN: for motivational study: simulating a set "edge-based" samplers each at different sampling density around edge
                    y_clone = y.clone().cpu()
                    if self.cfg.MODEL.ignore_gt_labels != []:
                        y_clone = torch.tensor(self.ignore_label(y_clone, self.cfg.MODEL.ignore_gt_labels))
                    y_norm = (y_clone[j] - y_clone[j].min()).float()/(y_clone[j].max() - y_clone[j].min()).float()
                    y_low = b_imresize(y_norm.unsqueeze(0).unsqueeze(0).float(), self.input_size, interp='bilinear') # [N,1,W,H]
                    y_gradient = y_low.clone() # [N,C,W,H]
                    y_low_cpu = y_low.cpu()
                    y_low_img_ay = np.array((y_low_cpu[0][0]*255)).astype(np.uint8)
                    y_low_img = Image.fromarray(y_low_img_ay, 'L')
                    # apply gaussian blur to avoid not having enough saliency for sampling
                    y_low_img = y_low_img.filter(ImageFilter.GaussianBlur(radius=self.cfg.MODEL.gt_grad_gaussian_blur_r )) # default radius=2
                    y_low_Edges = y_low_img.filter(ImageFilter.FIND_EDGES)
                    y_gradient[0][0] = torch.tensor(np.array(y_low_Edges.convert('L'))/255.).to(y_low.device)
                    xs_j = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(y_gradient)
                    xsc_j = xs_j[0] # 1,W,H
                    xsc[j] = xsc_j
                if segSize is not None and y_j_dist[1]/y_j_dist.sum() <= 0.001 and self.cfg.DATASET.binary_class != -1:
                    # exclude corner case
                    print('y_{} do not have enough forground class, skip this sample\n'.format(j))
                    if self.cfg.VAL.y_sampled_reverse:
                        return None, None, None, None
                    else:
                        return None, None, None

            if self.cfg.TRAIN.deform_zero_bound:
                xsc_mask = xsc.clone()*0.0
                bound = self.cfg.TRAIN.deform_zero_bound_factor
                xsc_mask[:,:,1*bound:-1*bound,1*bound:-1*bound] += 1.0
                xsc *= xsc_mask
            xs.data = xsc.data.to(xs.device)
        elif self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss:
        # EXPLAIN: if ours - calculate the target deformation map xs_target for edge_loss
            xs_target = xs.clone().detach()
            for j in range(y.shape[0]):
                (y_j_dist, _) = np.histogram(y[j].cpu(), bins=2, range=(0, 1))
                if not (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
                    y_norm = (y[j] - y[j].min()).float()/(y[j].max() - y[j].min()).float()
                    y_low = b_imresize(y_norm.unsqueeze(0).unsqueeze(0).float(), self.input_size, interp='bilinear') # [N,1,W,H]
                    y_gradient = y_low.clone() # [N,C,W,H]
                    y_low_cpu = y_low.cpu()
                    y_low_img_ay = np.array((y_low_cpu[0][0]*255)).astype(np.uint8)
                    y_low_img = Image.fromarray(y_low_img_ay, 'L')
                    y_low_img = y_low_img.filter(ImageFilter.GaussianBlur(radius=self.cfg.MODEL.gt_grad_gaussian_blur_r )) # default radius=2
                    y_low_Edges = y_low_img.filter(ImageFilter.FIND_EDGES)
                    y_gradient[0][0] = torch.tensor(np.array(y_low_Edges.convert('L'))/255.).to(y_low.device)
                    xs_j = nn.Upsample(size=(xs_target.shape[-2],xs_target.shape[-1]), mode='bilinear')(y_gradient)
                    (xs_j_dist, _) = np.histogram(xs_j.cpu(), bins=10, range=(0, 1))
                    xsc_j = xs_j[0] # 1,W,H
                    if self.cfg.TRAIN.opt_deform_LabelEdge_softmax:
                        xsc_j = xsc_j.view(1,xs_target.shape[-2]*xs_target.shape[-1])
                        xsc_j = nn.Softmax()(xsc_j)
                    xs_target[j] = xsc_j.view(1,xs_target.shape[-2],xs_target.shape[-1])
                elif y_j_dist[1]/y_j_dist.sum() <= 0.001 and self.cfg.DATASET.binary_class != -1:
                    print('y_{} do not have enough forground class, skip this sample\n'.format(j))
                    if segSize is not None:
                        if self.cfg.VAL.y_sampled_reverse:
                            return None, None, None, None
                        else:
                            return None, None, None
            if self.cfg.TRAIN.deform_zero_bound:
                xs_target_mask = xs_target.clone()*0.0
                bound = self.cfg.TRAIN.deform_zero_bound_factor
                xs_target_mask[:,:,1*bound:-1*bound,1*bound:-1*bound] += 1.0
                xs_target *= xs_target_mask

        # EXPLAIN: pad to avoid boundary artifact following A. Recasens et,al. (2018)
        if self.cfg.MODEL.uniform_sample != '':
            xs = xs*0 + 1.0/(self.grid_size_x*self.grid_size_y)
        if self.cfg.TRAIN.def_saliency_pad_mode == 'replication':
            xs_hm = nn.ReplicationPad2d((self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x))(xs) # padding
        elif self.cfg.TRAIN.def_saliency_pad_mode == 'reflect':
            xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='reflect')
        elif self.cfg.TRAIN.def_saliency_pad_mode == 'zero':
            xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='constant')

        # EXPLAIN: if training
        if segSize is None:
            if self.cfg.MODEL.gt_gradient and self.cfg.MODEL.gt_gradient_intrinsic_only:
                return None, None

            # EXPLAIN: pretraining trick following A. Recasens et,al. (2018)
            N_pretraining = self.cfg.TRAIN.deform_pretrain
            epoch = self.cfg.TRAIN.global_epoch
            if self.cfg.TRAIN.deform_pretrain_bol or (epoch>=N_pretraining and (epoch<self.cfg.TRAIN.smooth_deform_2nd_start or epoch>self.cfg.TRAIN.smooth_deform_2nd_end)):
                p=1 # non-pretain stage: no random size pooling to x_sampled
            else:
                p=0 # pretrain stage: random size pooling to x_sampled

            # EXPLAIN: construct the deformed sampler Gd (Eq. 3)
            grid, grid_y = self.create_grid(xs_hm)
            if self.cfg.MODEL.loss_at_high_res:
                xs_inv = 1-xs_hm
                _, grid_inv_train = self.create_grid(xs_hm, segSize=tuple(np.array(ori_size)//self.cfg.DATASET.segm_downsampling_rate), x_inv=xs_inv)

            # EXPLAIN: during training the labelY is downsampled with the same deformed sampler to get Yˆ = Gd(Y,d) (i.e. y_low)
            if self.cfg.MODEL.uniform_sample == 'BI':
                y_sampled = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(y.float().unsqueeze(1)).long().squeeze(1)
            else:
                y_sampled = F.grid_sample(y.float().unsqueeze(1), grid_y).squeeze(1)

            # EXPLAIN: calculate the edge loss Le(θ;Xlr,Ylr)=fMSE(d,dt)
            if self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss:
                assert (xs.shape == xs_target.shape), "xs shape ({}) not equvelent to xs_target shape ({})\n".format(xs.shape, xs_target.shape)
                if self.cfg.TRAIN.opt_deform_LabelEdge_norm:
                    xs_norm = ((xs - xs.min()) / (xs.max() - xs.min()))
                    xs_target_norm = ((xs_target - xs_target.min()) / (xs_target.max() - xs_target.min()))
                    edge_loss = self.crit_mse(xs_norm, xs_target_norm)
                    edge_acc = self.pixel_acc(xs_norm.long(), xs_target_norm.long())
                else:
                    edge_loss = self.crit_mse(xs, xs_target)
                    edge_acc = self.pixel_acc(xs.long(), xs_target.long())
                edge_loss *= self.cfg.TRAIN.edge_loss_scale

                # EXPLAIN: for staged training when traing with only the edge loss
                if self.cfg.TRAIN.opt_deform_LabelEdge and epoch >= self.cfg.TRAIN.fix_seg_start_epoch and epoch <= self.cfg.TRAIN.fix_seg_end_epoch:
                    return edge_loss, edge_acc, edge_loss

            # EXPLAIN: computes the downsampled image X^ =Gd(X,d)
            if self.cfg.MODEL.uniform_sample == 'BI':
                x_sampled = nn.Upsample(size=self.input_size_net, mode='bilinear')(x)
            else:
                x_sampled = F.grid_sample(x, grid)

            # EXPLAIN: pretraining trick following A. Recasens et,al. (2018)
            if random.random()>p:
                min_saliency_len = min(self.input_size)
                s = random.randint(min_saliency_len//3, min_saliency_len)
                x_sampled = nn.AdaptiveAvgPool2d((s,s))(x_sampled)
                x_sampled = nn.Upsample(size=self.input_size_net,mode='bilinear')(x_sampled)

            # EXPLAIN: The downsampled image X^ is then fed into the segmentation network to
            # estimate the corresponding segmentation probabilities Pˆ =Sϕ(Xˆ)
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(x_sampled, return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(x_sampled, return_feature_maps=True))
            torch.cuda.reset_max_memory_allocated(0)

            # EXPLAIN: ablation, if calculate loss at high resolution, inverse upsample the prediction to high-res space
            if self.cfg.MODEL.loss_at_high_res and self.cfg.MODEL.uniform_sample == 'BI':
                pred_sampled_train = nn.Upsample(size=ori_size, mode='bilinear')(x)
            elif self.cfg.MODEL.loss_at_high_res:
                unfilled_mask_2d = torch.isnan(grid_inv_train[:,:,:,0])
                grid_inv_train[torch.isnan(grid_inv_train)] = 0
                pred_sampled_train = F.grid_sample(x, grid_inv_train.float())
                pred_sampled_train[unfilled_mask_2d.unsqueeze(1).expand(pred_sampled_train.shape)] = float('nan')
                for n in range(pred_sampled_train.shape[0]):
                    pred_sampled_train[n] = fillMissingValues_tensor(pred_sampled_train[n], interp_mode=self.cfg.MODEL.rev_deform_interp)

            # change variable naming
            if self.deep_sup_scale is not None: # use deep supervision technique
                pred, pred_deepsup = pred, pred_deepsup
            if self.cfg.MODEL.loss_at_high_res:
                pred,image_output,hm,_,pred_sampled = pred,x_sampled,xs,y_sampled,pred_sampled_train
            else:
                pred,image_output,hm,feed_dict['seg_label'] = pred,x_sampled,xs,y_sampled.long()
            # del xs, x_sampled, x
            if self.cfg.MODEL.loss_at_high_res:
                del pred_sampled_train
            del y_sampled

            # EXPLAIN: end of training, calculate loss and return
            if self.cfg.MODEL.loss_at_high_res:
                pred_sampled[torch.isnan(pred_sampled)] = 0 # assign residual missing with 0 probability
                loss = self.crit(pred_sampled, feed_dict['seg_label'])
            else:
                loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale
            if self.cfg.TRAIN.deform_joint_loss:
                # print('seg loss: {}, scaled edge_loss: {}\n'.format(loss, edge_loss))
                loss = loss + edge_loss
            if self.cfg.MODEL.loss_at_high_res:
                acc = self.pixel_acc(pred_sampled, feed_dict['seg_label'])
            else:
                acc = self.pixel_acc(pred, feed_dict['seg_label'])
            if self.cfg.TRAIN.deform_joint_loss:
                return loss, acc, edge_loss
            else:
                return loss, acc

        # EXPLAIN: if inference
        else:
            t = time.time()
            # EXPLAIN: at inference, calculate both the non-uniform downsampler (grid) and upsampler (grid_inv)
            xs_inv = 1-xs_hm
            grid, grid_inv = self.create_grid(xs_hm, segSize=segSize, x_inv=xs_inv)
            _, grid_y = self.create_grid(xs_hm, segSize=segSize)

            # EXPLAIN: computes the downsampled image X^ =Gd(X,d)
            if self.cfg.MODEL.uniform_sample == 'BI':
                x_sampled = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(x)
            else:
                x_sampled = F.grid_sample(x, grid)
                x_sampled = nn.Upsample(size=self.input_size_net_infer,mode='bilinear')(x_sampled)

            segSize_temp = tuple(self.input_size_net_infer)
            print('eval segSize_temp: {}'.format(segSize_temp))

            # EXPLAIN: The downsampled image X^ is then fed into the segmentation network to
            # estimate the corresponding segmentation probabilities Pˆ =Sϕ(Xˆ)
            x = self.decoder(self.encoder(x_sampled, return_feature_maps=True), segSize=segSize_temp)

            # EXPLAIN: downsample and upsample label y for calculating the intrinsic upsampling error IoU(Y′,Y)
            if self.cfg.MODEL.uniform_sample == 'BI':
                y_sampled = nn.Upsample(size=tuple(np.array(self.input_size_net_infer)), mode='bilinear')(y.float().unsqueeze(1)).long().squeeze(1)
            else:
                y_sampled = F.grid_sample(y.float().unsqueeze(1), grid_y, mode='nearest').long().squeeze(1)

            # EXPLAIN: inverse upsample the prediction and low resolution label to high-res space
            if self.cfg.MODEL.uniform_sample == 'BI' or self.cfg.MODEL.uniform_sample == 'nearest':
                # uniform case
                if self.cfg.MODEL.uniform_sample == 'BI':
                    pred_sampled = nn.Upsample(size=segSize, mode='bilinear')(x)
                elif self.cfg.MODEL.uniform_sample == 'nearest':
                    pred_sampled = nn.Upsample(size=segSize, mode='nearest')(x)
                pred_sampled_unfilled_mask_2d = torch.isnan(grid_inv[:,:,:,0])
                if self.cfg.VAL.y_sampled_reverse:
                    assert (self.cfg.MODEL.rev_deform_interp == 'nearest'), "y_sampled_reverse only appliable to nearest rev_deform_interp"
                    y_sampled_reverse =  nn.Upsample(size=segSize, mode='nearest')(y_sampled.float().unsqueeze(1)).squeeze(1)
                if self.cfg.VAL.x_sampled_reverse:
                    x_sampled_unorm = unorm(x_sampled)
                    x_sampled_reverse =  nn.Upsample(size=segSize, mode='bilinear')(x_sampled_unorm)
            elif self.cfg.MODEL.rev_deform_opt == 51:
                # ours deformed case
                pred_sampled_unfilled_mask_2d = torch.isnan(grid_inv[:,:,:,0])
                grid_inv[torch.isnan(grid_inv)] = 0
                pred_sampled = F.grid_sample(x, grid_inv.float())
                pred_sampled[pred_sampled_unfilled_mask_2d.unsqueeze(1).expand(pred_sampled.shape)] = float('nan')
                if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                    pred_sampled_cpu = pred_sampled.cpu()
                for n in range(pred_sampled.shape[0]):
                    if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                        pred_sampled[n] = torch.tensor(fillMissingValues_tensor(np.array(pred_sampled_cpu[n]), interp_mode=self.cfg.MODEL.rev_deform_interp))
                    else:
                        pred_sampled[n] = fillMissingValues_tensor(pred_sampled[n], interp_mode=self.cfg.MODEL.rev_deform_interp)

                # for visualisation purpose: downsample and upsample image x
                if self.cfg.VAL.x_sampled_reverse:
                    x_sampled_unorm = unorm(x_sampled)
                    x_sampled_reverse = F.grid_sample(x_sampled_unorm, grid_inv.float())
                    x_sampled_reverse[pred_sampled_unfilled_mask_2d.unsqueeze(1).expand(x_sampled_reverse.shape)] = float('nan')
                    if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                        x_sampled_reverse_cpu = x_sampled_reverse.cpu()
                    for n in range(x_sampled_reverse.shape[0]):
                        if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                            x_sampled_reverse[n] = torch.tensor(fillMissingValues_tensor(np.array(x_sampled_reverse_cpu[n]), interp_mode=self.cfg.MODEL.rev_deform_interp))
                        else:
                            x_sampled_reverse[n] = fillMissingValues_tensor(x_sampled_reverse[n], interp_mode=self.cfg.MODEL.rev_deform_interp)
                            x_sampled_reverse[n][torch.isnan(x_sampled_reverse[n])] = x_sampled_reverse[n][~torch.isnan(x_sampled_reverse[n])].mean()

                # for visualisation and calculating the intrinsic upsampling error IoU(Y′,Y) purpose
                if self.cfg.VAL.y_sampled_reverse:
                    if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                        y_sampled_reverse = F.grid_sample(y_sampled.float().unsqueeze(1), grid_inv.float(), mode='nearest').squeeze(1)
                        y_sampled_reverse[pred_sampled_unfilled_mask_2d.expand(y_sampled_reverse.shape)] = float('nan')
                        if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                            y_sampled_reverse_cpu = y_sampled_reverse.unsqueeze(1).cpu()
                        for n in range(y_sampled_reverse.shape[0]):
                            if self.cfg.MODEL.rev_deform_interp == 'nearest' or self.cfg.MODEL.rev_deform_interp == 'BI':
                                y_sampled_reverse[n] = torch.tensor(fillMissingValues_tensor(np.array(y_sampled_reverse_cpu[n]), interp_mode=self.cfg.MODEL.rev_deform_interp)).squeeze(1)
                            else:
                                y_sampled_reverse[n] = fillMissingValues_tensor(y_sampled_reverse[n], interp_mode=self.cfg.MODEL.rev_deform_interp)
                    else:
                        y_sampled_score = y_sampled.unsqueeze(1).expand(y_sampled.shape[0], self.cfg.DATASET.num_class, y_sampled.shape[-2], y_sampled.shape[-1])
                        y_sampled_score = y_sampled_score.float()
                        for c in range(self.cfg.DATASET.num_class):
                            mask = (y_sampled_score[:,c,:,:] == c) # nate make sure classes are mapped to the range [0,num_class-1]
                            un_mask = (y_sampled_score[:,c,:,:] != c)
                            y_sampled_score[:,c,:,:][mask] = 1.0
                            y_sampled_score[:,c,:,:][un_mask] = 0.0
                        y_sampled_score_reverse = F.grid_sample(y_sampled_score, grid_inv.float())
                        y_sampled_score_reverse[pred_sampled_unfilled_mask_2d.unsqueeze(1).expand(y_sampled_score_reverse.shape)] = float('nan')
                        for n in range(y_sampled_score_reverse.shape[0]):
                            y_sampled_score_reverse[n] = fillMissingValues_tensor(y_sampled_score_reverse[n], interp_mode=self.cfg.MODEL.rev_deform_interp)
                        _, y_sampled_reverse = torch.max(y_sampled_score_reverse, dim=1)

            # post processing FILL residual missing
            if feed_batch_count != None:
                if feed_batch_count < self.cfg.VAL.batch_size:
                    self.num_res_nan_percentage = []
                self.num_res_nan_percentage.append(float(torch.isnan(pred_sampled).sum()*100.0) / float(pred_sampled.shape[0]*pred_sampled.shape[1]*pred_sampled.shape[2]*pred_sampled.shape[3]))
            pred_sampled[torch.isnan(pred_sampled)] = 0 # assign residual missing with 0 probability

            # change variable naming
            if self.cfg.VAL.no_upsample:
                pred,image_output,hm = x,x_sampled,xs
            else:
                if self.cfg.MODEL.rev_deform_opt == 51:
                    pred,image_output,hm,pred_sampled,pred_sampled_unfilled_mask_2d = x,x_sampled,xs,pred_sampled,pred_sampled_unfilled_mask_2d
                else:
                    pred,image_output,hm,pred_sampled = x,x_sampled,xs,pred_sampled

            # EXPLAIN: for visualisation only
            if (writer is not None and feed_batch_count < 4 and feed_batch_count != None) or self.cfg.TRAIN.train_eval_visualise:
                if self.cfg.TRAIN.train_eval_visualise:
                    dir_result = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch))
                    if not os.path.isdir(dir_result):
                        os.makedirs(dir_result)

                    dir_result_lb_ori = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'label_original')
                    dir_result_sal = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'saliency_map')
                    dir_result_edge = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'label_edge_target')
                    dir_result_grid = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'deformed_grid')
                    dir_result_def_img = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'deformed_image')
                    dir_result_def_pred = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'deformed_pred')
                    dir_result_lb_samp = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'label_sampled')
                    dir_result_int_lb_samp = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'interp_label_sampled')
                    dir_result_int_def_pred = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'interpolated_deformed_pred')
                    dir_result_int_def_pred_unfill = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'interpolated_deformed_pred_unfilled')
                    dir_result_sampling_mask = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'sampling_masked_on_original')
                    if self.cfg.VAL.x_sampled_reverse:
                        dir_result_int_def_img = os.path.join(self.cfg.DIR, "visual_epoch_{}".format(epoch),'interpolated_deformed_image')
                        if not os.path.isdir(dir_result_int_def_img):
                            os.makedirs(dir_result_int_def_img)

                    if not os.path.isdir(dir_result_lb_ori):
                        os.makedirs(dir_result_lb_ori)
                    if not os.path.isdir(dir_result_sal):
                        os.makedirs(dir_result_sal)
                    if not os.path.isdir(dir_result_edge):
                        os.makedirs(dir_result_edge)
                    if not os.path.isdir(dir_result_grid):
                        os.makedirs(dir_result_grid)
                    if not os.path.isdir(dir_result_def_img):
                        os.makedirs(dir_result_def_img)
                    if not os.path.isdir(dir_result_def_pred):
                        os.makedirs(dir_result_def_pred)
                    if not os.path.isdir(dir_result_lb_samp):
                        os.makedirs(dir_result_lb_samp)
                    if not os.path.isdir(dir_result_int_lb_samp):
                        os.makedirs(dir_result_int_lb_samp)
                    if not os.path.isdir(dir_result_int_def_pred):
                        os.makedirs(dir_result_int_def_pred)
                    if not os.path.isdir(dir_result_int_def_pred_unfill):
                        os.makedirs(dir_result_int_def_pred_unfill)
                    if not os.path.isdir(dir_result_sampling_mask):
                        os.makedirs(dir_result_sampling_mask)

                if self.cfg.DATASET.grid_path != '':
                    grid_img = Image.open(self.cfg.DATASET.grid_path).convert('RGB')
                    grid_resized = grid_img.resize(segSize, Image.BILINEAR)
                    del grid_img

                    grid_resized = np.float32(np.array(grid_resized)) / 255.
                    grid_resized = grid_resized.transpose((2, 0, 1))
                    grid_resized = torch.from_numpy(grid_resized.copy())

                    grid_resized = torch.unsqueeze(grid_resized, 0).expand(grid.shape[0],grid_resized.shape[-3],grid_resized.shape[-2],grid_resized.shape[-1])
                    grid_resized = grid_resized.to(feed_dict['seg_label'].device)

                    if self.cfg.MODEL.uniform_sample == 'BI':
                        grid_output = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(grid_resized)
                    else:
                        grid_output = F.grid_sample(grid_resized, grid)
                    del grid_resized

                image_output = image_output.data
                image_output = unorm(image_output)

                hm = hm.data
                hm_max, _ = hm.view(hm.shape[0],-1).max(dim=1)
                hm_shape = hm.shape
                hm = (hm.view(hm.shape[0],-1)/hm_max.view(hm.shape[0],-1)).view(hm_shape)
                for i in range(hm.shape[0]):
                    xhm = vutils.make_grid(hm[i].unsqueeze(0), normalize=True, scale_each=True)
                    writer.add_image('eval_{}/Saliency Map'.format(feed_batch_count*self.cfg.VAL.batch_size+i), xhm, count)
                    if self.cfg.TRAIN.train_eval_visualise:
                        img_name = feed_dict['info'].split('/')[-1]
                        Image.fromarray(np.array(hm[i].squeeze(0).squeeze(0).cpu()*255.0).astype(np.uint8)).save(os.path.join(dir_result_sal, os.path.splitext(img_name)[0] + '.png'))

                    if self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss:
                        xs_target_p = vutils.make_grid(xs_target[i].unsqueeze(0), normalize=True, scale_each=True)
                        writer.add_image('eval_{}/Label Edge (Target)'.format(feed_batch_count*self.cfg.VAL.batch_size+i), xs_target_p, count)
                        if self.cfg.TRAIN.train_eval_visualise:
                            Image.fromarray(np.array(xs_target[i].squeeze(0).squeeze(0).cpu()*255.0).astype(np.uint8)).save(os.path.join(dir_result_edge, os.path.splitext(img_name)[0] + '.png'))

                    deformed_grid = vutils.make_grid(grid_output[i].unsqueeze(0), normalize=True, scale_each=True)
                    writer.add_image('eval_{}/Deformed Grid'.format(feed_batch_count*self.cfg.VAL.batch_size+i), deformed_grid, count)
                    if self.cfg.TRAIN.train_eval_visualise:
                        Image.fromarray(np.array(grid_output[i].permute(1,2,0).cpu()*255.0).astype(np.uint8)).save(os.path.join(dir_result_grid, os.path.splitext(img_name)[0] + '.png'))

                    x = vutils.make_grid(image_output[i].unsqueeze(0), normalize=True, scale_each=True)
                    writer.add_image('eval_{}/Deformed Image'.format(feed_batch_count*self.cfg.VAL.batch_size+i), x, count)
                    if self.cfg.TRAIN.train_eval_visualise:
                        Image.fromarray(np.array(((image_output[i]-image_output[i].min())/(image_output[i].max()-image_output[i].min())).permute(1,2,0).cpu()*255.0).astype(np.uint8)).save(os.path.join(dir_result_def_img, os.path.splitext(img_name)[0] + '.png'))
                        if self.cfg.VAL.x_sampled_reverse:
                            Image.fromarray(np.array(((x_sampled_reverse[i]-x_sampled_reverse[i].min())/(x_sampled_reverse[i].max()-x_sampled_reverse[i].min())).permute(1,2,0).cpu()*255.0).astype(np.uint8)).save(os.path.join(dir_result_int_def_img, os.path.splitext(img_name)[0] + '.png'))
                    _, pred_print = torch.max(pred, dim=1)
                    colors = loadmat('data/color150.mat')['colors']

                    pred_color = colorEncode(as_numpy(pred_print[i].unsqueeze(0).squeeze(0)), colors)
                    pred_print_temp = torch.from_numpy(pred_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                    if self.cfg.TRAIN.train_eval_visualise:
                        Image.fromarray(np.array(pred_color)).save(os.path.join(dir_result_def_pred, os.path.splitext(img_name)[0] + '.png'))
                    pred_print_temp = vutils.make_grid(pred_print_temp, normalize=False, scale_each=True)
                    writer.add_image('eval_{}/Deformed pred'.format(feed_batch_count*self.cfg.VAL.batch_size+i), pred_print_temp, count)
                    if self.cfg.TRAIN.train_eval_visualise:
                        y_color = colorEncode(as_numpy(feed_dict['seg_label'][i].unsqueeze(0).squeeze(0)), colors)
                        Image.fromarray(y_color).save(os.path.join(dir_result_lb_ori, os.path.splitext(img_name)[0] + '.png'))
                    if self.print_original_y:
                        y_color = colorEncode(as_numpy(feed_dict['seg_label'][i].unsqueeze(0).squeeze(0)), colors)
                        y_print = torch.from_numpy(y_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                        y_print = vutils.make_grid(y_print, normalize=False, scale_each=True)
                        writer.add_image('eval_{}/Label Original'.format(feed_batch_count*self.cfg.VAL.batch_size+i), y_print, count)

                    y_sampled_color = colorEncode(as_numpy(y_sampled[i].unsqueeze(0).squeeze(0)), colors)
                    y_sampled_print = torch.from_numpy(y_sampled_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                    if self.cfg.TRAIN.train_eval_visualise:
                        Image.fromarray(y_sampled_color).save(os.path.join(dir_result_lb_samp, os.path.splitext(img_name)[0] + '.png'))
                    y_sampled_print = vutils.make_grid(y_sampled_print, normalize=False, scale_each=True)
                    writer.add_image('eval_{}/Label Sampled'.format(feed_batch_count*self.cfg.VAL.batch_size+i), y_sampled_print, count)

                    if self.cfg.VAL.y_sampled_reverse:
                        y_sampled_reverse_color = colorEncode(as_numpy(y_sampled_reverse[i].unsqueeze(0).squeeze(0)), colors)
                        y_sampled_reverse_print = torch.from_numpy(y_sampled_reverse_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                        if self.cfg.TRAIN.train_eval_visualise:
                            Image.fromarray(y_sampled_reverse_color).save(os.path.join(dir_result_int_lb_samp, os.path.splitext(img_name)[0] + '.png'))
                        y_sampled_reverse_print = vutils.make_grid(y_sampled_reverse_print, normalize=False, scale_each=True)
                        writer.add_image('eval_{}/Interpolated Label Sampled'.format(feed_batch_count*self.cfg.VAL.batch_size+i), y_sampled_reverse_print, count)

                    _, pred_print_sampled = torch.max(pred_sampled[i].unsqueeze(0), dim=1)
                    pred_sampled_color = colorEncode(as_numpy(pred_print_sampled.squeeze(0)), colors)
                    if self.cfg.MODEL.rev_deform_opt == 51:
                        pred_print_sampled_unfilled = pred_print_sampled.clone()
                        pred_print_sampled_unfilled[pred_sampled_unfilled_mask_2d[i].unsqueeze(0)] = 30
                        pred_sampled_color_unfilled = colorEncode(as_numpy(pred_print_sampled_unfilled.squeeze(0)), colors)
                    pred_print_sampled = torch.from_numpy(pred_sampled_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                    if self.cfg.TRAIN.train_eval_visualise:
                        Image.fromarray(pred_sampled_color).save(os.path.join(dir_result_int_def_pred, os.path.splitext(img_name)[0] + '.png'))
                    pred_print_sampled = vutils.make_grid(pred_print_sampled, normalize=False, scale_each=True)
                    writer.add_image('eval_{}/Interpolated Deformed Pred'.format(feed_batch_count*self.cfg.VAL.batch_size+i), pred_print_sampled, count)

                    if self.cfg.MODEL.rev_deform_opt == 51:
                        pred_print_sampled_unfilled = torch.from_numpy(pred_sampled_color_unfilled.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                        if self.cfg.TRAIN.train_eval_visualise:
                            Image.fromarray(pred_sampled_color_unfilled).save(os.path.join(dir_result_int_def_pred_unfill, os.path.splitext(img_name)[0] + '.png'))
                            red_spot_mask_2d = torch.zeros(pred_sampled_unfilled_mask_2d[i].shape)
                            red_spot_mask_2d[~pred_sampled_unfilled_mask_2d[i]] = 1
                            red_spot_mask_2d = np.array(red_spot_mask_2d)
                            red_spot_mask_2d_dilate = ndimage.binary_dilation(red_spot_mask_2d, iterations=2)
                            img_sampling_masked = feed_dict['img_ori']
                            img_sampling_masked[:,:,0][red_spot_mask_2d_dilate] = 255.0
                            img_sampling_masked[:,:,1][red_spot_mask_2d_dilate] = 0.0
                            img_sampling_masked[:,:,2][red_spot_mask_2d_dilate] = 0.0
                            Image.fromarray(img_sampling_masked.astype(np.uint8)).save(os.path.join(dir_result_sampling_mask, os.path.splitext(img_name)[0] + '.png'))
                        pred_print_sampled_unfilled = vutils.make_grid(pred_print_sampled_unfilled, normalize=False, scale_each=True)
                        writer.add_image('eval_{}/Interpolated Deformed Pred unfilled'.format(feed_batch_count*self.cfg.VAL.batch_size+i), pred_print_sampled_unfilled, count)

                self.print_original_y = False

            if writer is not None and feed_batch_count == -1 and feed_batch_count != None:
                print('EVAL: pred_sampled num residual nan percentage: {} %\n'.format(np.array(self.num_res_nan_percentage).mean()))
                writer.add_scalar('Residual_nan_percentage_eval', np.array(self.num_res_nan_percentage).mean(), count)

            # finish inference and return
            if F_Xlr_acc_map:
                loss = self.crit(pred_sampled, feed_dict['seg_label'])
                return pred_sampled, loss
            else:
                if self.cfg.VAL.y_sampled_reverse:
                    return pred_sampled, pred, y_sampled, y_sampled_reverse.long()
                else:
                    return pred_sampled, pred, y_sampled

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, cfg, deep_sup_scale=None, net_fov_res=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.net_fov_res = net_fov_res

    # @torchsnooper.snoop()
    def forward(self, feed_dict, *, segSize=None, F_Xlr_acc_map=False, writer=None, count=None, feed_dict_info=None, feed_batch_count=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            elif self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            if self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize, res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize)
            if self.cfg.VAL.write_pred:
                _, pred_print = torch.max(pred, dim=1)
                colors = loadmat('data/color150.mat')['colors']
                pred_color = colorEncode(as_numpy(pred_print.squeeze(0)), colors)
                pred_print = torch.from_numpy(pred_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                print('train/pred size: {}'.format(pred_print.shape))
                pred_print = vutils.make_grid(pred_print, normalize=False, scale_each=True)
                writer.add_image('train/pred', pred_print, count)

            if F_Xlr_acc_map:
                loss = self.crit(pred, feed_dict['seg_label'])
                return pred, loss
            else:
                return pred

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    @staticmethod
    def build_encoder(arch='resnet50', fc_dim=2048, weights='', dilate_rate=4):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=dilate_rate)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2_nodownsp':
            net_encoder = hrnetv2_nodownsp.__dict__['hrnetv2_nodownsp'](pretrained=False)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='upernet',
                      fc_dim=2048, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def build_net_saliency(cfg=None,
                        weights=''):
        # define saliency network
        # Spatial transformer localization-network
        if cfg.MODEL.track_running_stats:
            if cfg.MODEL.saliency_net == 'resnet18':
                net_saliency = saliency_network_resnet18()
            elif cfg.MODEL.saliency_net == 'resnet18_stride1':
                net_saliency = saliency_network_resnet18_stride1()
            elif cfg.MODEL.saliency_net == 'fovsimple':
                net_saliency = fov_simple(cfg)

        if len(weights) == 0:
            net_saliency.apply(ModelBuilder.weights_init)
        else:
            print('Loading weights for net_saliency')
            net_saliency.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_saliency

    @staticmethod
    def build_net_compress(cfg=None,
                        weights=''):
        net_compress = CompressNet(cfg)

        if len(weights) == 0:
            net_compress.apply(ModelBuilder.weights_init)
        else:
            print('Loading weights for net_compress')
            net_compress.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_compress
