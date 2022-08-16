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

BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1

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
        self.epoch_record = 0

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

        if self.cfg.MODEL.img_gradient:
            sobel_weight = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            self.sobel_filter = nn.Conv2d(1, 1, kernel_size=(3,3),bias=False)
            self.sobel_filter.weight[0][0].data[:,:] = sobel_weight
            self.compress_filter = nn.Conv2d(3,1,kernel_size=1,padding=0,stride=1)
            self.compress_filter.weight.data = torch.tensor([0.2989, 0.5870, 0.1140]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        self.save_print_grad = [{'saliency_grad': 0.0, 'check1_grad': 0.0, 'check2_grad': 0.0} for _ in range(cfg.TRAIN.num_gpus)]

    def re_initialise(self, cfg, this_size):
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

    def fillMissingValues_tensor(self, target_for_interp, copy=False, interp_mode='tri'):
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
            invalid_mask = np.isnan(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            #dilate to mark borders around invalid regions
            if max(invalid_mask.shape) > 512:
                dr = max(invalid_mask.shape)/512
                input = torch.tensor(invalid_mask.astype('float')).unsqueeze(0)
                shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
                shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
                # print('input shape: {}\n'.format(input.shape))
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
            # Interpolate only holes, only using these pixels
            # missing locations should be same for all classes (first dim)
            points = torch.where(mask_for_interp[0])
            points = torch.cat([t.unsqueeze(0) for t in points])
            points = points.permute(1,0)
            values = target_for_interp.clone()[mask_for_interp].view(mask_for_interp.shape[0],-1).permute(1,0) # shape: [number_of_points, num_classes]
        interp = interpolator(points, values)

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
            # The normalize code -> t.sub_(m).div_(s)
        return img

    def ignore_label(self, label, ignore_indexs):
        label = np.array(label)
        temp = label.copy()
        for k in ignore_indexs:
            label[temp == k] = 0
        return label

    # # @torchsnooper.snoop()
    def forward(self, feed_dict, *, writer=None, segSize=None, F_Xlr_acc_map=False, count=None, epoch=None, feed_dict_info=None, feed_batch_count=None):
        if self.cfg.TRAIN.dynamic_task_input[0] != 1:
            this_size = tuple(feed_dict['img_data'].shape[-2:])
            print('this_size: {}'.format(this_size))
            self.re_initialise(self.cfg, this_size)
            print('task_input_size after re_initialise: {}'.format(self.input_size_net))
            print('saliency_input_size after re_initialise: {}'.format(self.input_size))

        x = feed_dict['img_data']
        del feed_dict['img_data']

        t = time.time()
        ori_size = (x.shape[-2],x.shape[-1])
        x_low = b_imresize(x, self.input_size, interp='bilinear')
        epoch = self.cfg.TRAIN.global_epoch
        if segSize is None and ((self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss) and epoch <= self.cfg.TRAIN.deform_pretrain):
            min_saliency_len = min(self.input_size)
            s = random.randint(min_saliency_len//3, min_saliency_len)
            x_low = nn.AdaptiveAvgPool2d((s,s))(x_low)
            x_low = nn.Upsample(size=self.input_size,mode='bilinear')(x_low)
        xs = self.localization(x_low)
        xs = self.net_compress(xs)
        xs = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(xs)
        xs = xs.view(-1,self.grid_size_x*self.grid_size_y) # N,1,W,H
        xs = nn.Softmax()(xs) # N,W*H
        xs = xs.view(-1,1,self.grid_size_x,self.grid_size_y) #saliency map

        y = feed_dict['seg_label'].clone()
        if self.cfg.TRAIN.rescale_regressed_xs:
            xs_shape = xs.shape
            xs_reshape = xs.clone().view(xs.shape[0],-1)
            xs_reshape_max, _ = xs_reshape.max(dim=1)
            xs_reshape_min, _ = xs_reshape.min(dim=1)
            xs_scale = ((xs_reshape-xs_reshape_min.view(xs.shape[0],-1))/(xs_reshape_max.view(xs.shape[0],-1)-xs_reshape_min.view(xs.shape[0],-1))).view(xs_shape)
            xs.data = xs_scale.data.to(xs.device)

        if self.cfg.MODEL.fix_saliency:
            if self.cfg.MODEL.fix_saliency == 1: # central one gaussian
                gaussian_saliency = torch.FloatTensor(makeGaussian(self.grid_size_x, fwhm = self.grid_size_x//4))
            elif self.cfg.MODEL.fix_saliency == 2: # two eye gaussian
                path = './data/Face_single_example/EYE_saliency.png'
                eye_saliency = cv2.imread(path)
                eye_saliency_g = cv2.cvtColor(eye_saliency, cv2.COLOR_BGR2GRAY)
                eye_saliency_g = torch.FloatTensor(eye_saliency_g/255)
                eye_saliency_g = F.interpolate(eye_saliency_g.unsqueeze(0).unsqueeze(0), (self.grid_size_x,self.grid_size_y), mode='bilinear').squeeze(0).squeeze(0)
                gaussian_saliency = eye_saliency_g

            m = nn.Softmax()
            gaussian_saliency = m(gaussian_saliency)
            gaussian_saliency = b_imresize(gaussian_saliency.unsqueeze(0).unsqueeze(0), (self.grid_size_x,self.grid_size_y), interp='bilinear').to(xs.device)
            xs = gaussian_saliency
        elif self.cfg.MODEL.img_gradient and (epoch == 1 or self.cfg.MODEL.fix_img_gradient):
            # opt1: PIL implementation of edge detection
            x_gradient = x_low.clone()[:,0,:,:] # [N,C,W,H] -> [N,W,H]
            for i in range(x_low.shape[0]):
                x_low_cpu = x_low.cpu()
                x_low_img = Image.fromarray(np.array(x_low_cpu[i].permute(1,2,0)*255).astype(np.uint8))
                x_low_Edges = x_low_img.filter(ImageFilter.FIND_EDGES)
                x_gradient[i] = torch.tensor(np.array(x_low_Edges.convert('L'))/255.).to(x_low.device)
            x_gradient = x_gradient.unsqueeze(1)
            xs = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(x_gradient)
        elif self.cfg.MODEL.gt_gradient or (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
            xsc = xs.clone().detach()
            for j in range(y.shape[0]):
                if segSize is not None:
                    (y_j_dist, _) = np.histogram(y[j].cpu(), bins=2, range=(0, 1))
                if self.cfg.MODEL.fix_gt_gradient and not (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
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
            # Note we batch_size_per_gpu = 1 is suggested for opt_deform_LabelEdge
            # because when it's larger than 1, those samples need to be skipped would have 0 loss
            # thus averaged down the loss with foreground to much smaller value
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

        if self.cfg.MODEL.uniform_sample != '':
            xs = xs*0 + 1.0/(self.grid_size_x*self.grid_size_y)
        if self.cfg.TRAIN.def_saliency_pad_mode == 'replication':
            xs_hm = nn.ReplicationPad2d((self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x))(xs) # padding
        elif self.cfg.TRAIN.def_saliency_pad_mode == 'reflect':
            xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='reflect')
        elif self.cfg.TRAIN.def_saliency_pad_mode == 'zero':
            xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='constant')
        #
        # training
        if segSize is None:
            if self.cfg.MODEL.gt_gradient and self.cfg.MODEL.gt_gradient_intrinsic_only:
                if 'single' in self.cfg.DATASET.list_train:
                    return None, None, None, None
                else:
                    return None, None
            if 'single' in self.cfg.DATASET.list_train:
                y_print = feed_dict['seg_label'][0].unsqueeze(0)
                print('y_print size: {}'.format(y_print.shape))

            N_pretraining = self.cfg.TRAIN.deform_pretrain
            epoch = self.cfg.TRAIN.global_epoch
            # pretrain stage, simplify task of segmentation_module by smoothing x_sampled
            # non-pretain stage, no smoothing applied
            if self.cfg.TRAIN.deform_pretrain_bol or (epoch>=N_pretraining and (epoch<self.cfg.TRAIN.smooth_deform_2nd_start or epoch>self.cfg.TRAIN.smooth_deform_2nd_end)):
                p=1 # no random size pooling to x_sampled ->
            else:
                p=0 # random size pooling to x_sampled -> simplify task of segmentation_module

            grid, grid_y = self.create_grid(xs_hm)
            if self.cfg.MODEL.loss_at_high_res or self.cfg.TRAIN.separate_optimise_deformation:
                xs_inv = 1-xs_hm
                _, grid_inv_train = self.create_grid(xs_hm, segSize=tuple(np.array(ori_size)//self.cfg.DATASET.segm_downsampling_rate), x_inv=xs_inv)
            if self.cfg.TRAIN.separate_optimise_deformation:
                y = y.float()
                y.requires_grad=True
            if not self.cfg.MODEL.naive_upsample:
                if self.cfg.MODEL.uniform_sample == 'BI':
                    y_sampled = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(y.float().unsqueeze(1)).long().squeeze(1)
                else:
                    y_sampled = F.grid_sample(y.float().unsqueeze(1), grid_y).squeeze(1)
            if self.cfg.TRAIN.separate_optimise_deformation and self.cfg.VAL.y_sampled_reverse and not (self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss):
                unfilled_mask_2d = torch.isnan(grid_inv_train[:,:,:,0])
                mask = torch.isnan(grid_inv_train)
                grid_inv_train = grid_inv_train.clone().masked_fill_(mask.eq(1), 0.0)
                y_sampled = y_sampled.unsqueeze(1)
                y_sampled_score = y_sampled.clone().expand(y_sampled.shape[0], self.cfg.DATASET.num_class, y_sampled.shape[-2], y_sampled.shape[-1]).float()
                y_sampled_score = y_sampled_score.clone() + 1.0
                class_ar = torch.tensor(np.arange(self.cfg.DATASET.num_class)).float().to(y_sampled.device)
                class_ar += 1.0
                class_ar = class_ar.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                y_sampled_score = y_sampled_score / class_ar
                mask = y_sampled_score != 1
                y_sampled_score.masked_fill_(mask, 0.0)
                if self.cfg.MODEL.rev_deform_interp == 'nearest':
                    y_sampled_score_reverse = F.grid_sample(y_sampled_score, grid_inv_train.float(), mode='nearest')
                else:
                    y_sampled_score_reverse = F.grid_sample(y_sampled_score, grid_inv_train.float())
                y_sampled_score_reverse[unfilled_mask_2d.unsqueeze(1).expand(y_sampled_score_reverse.shape)] = float('nan')

                for n in range(y_sampled_score_reverse.shape[0]):
                    y_sampled_score_reverse[n] = self.fillMissingValues_tensor(y_sampled_score_reverse[n], interp_mode=self.cfg.MODEL.rev_deform_interp)

                mask = torch.isnan(y_sampled_score_reverse)
                y_sampled_score_reverse = y_sampled_score_reverse.clone().masked_fill_(mask.eq(1), 0.0)

                loss = self.crit(y_sampled_score_reverse, feed_dict['seg_label'])
                acc = self.pixel_acc(y_sampled_score_reverse, feed_dict['seg_label'])
                return loss, acc, self.save_print_grad

            elif self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss:
                assert (xs.shape == xs_target.shape), "xs shape ({}) not equvelent to xs_target shape ({})\n".format(xs.shape, xs_target.shape)
                if self.cfg.TRAIN.opt_deform_LabelEdge_norm:
                    xs_norm = ((xs - xs.min()) / (xs.max() - xs.min()))
                    xs_target_norm = ((xs_target - xs_target.min()) / (xs_target.max() - xs_target.min()))
                    edge_loss = self.crit_mse(xs_norm, xs_target_norm)
                    edge_acc = self.pixel_acc(xs_norm.long(), xs_target_norm.long())
                else:
                    edge_loss = self.crit_mse(xs, xs_target)
                    edge_acc = self.pixel_acc(xs.long(), xs_target.long())
                print("Edge loss: {}\n".format(edge_loss))
                print("Epoch {} edge_loss_scale={}".format(epoch, self.cfg.TRAIN.edge_loss_scale))
                edge_loss *= self.cfg.TRAIN.edge_loss_scale
                print("Scaled Edge loss: {}\n".format(edge_loss))
                if self.cfg.TRAIN.opt_deform_LabelEdge and epoch >= self.cfg.TRAIN.fix_seg_start_epoch and epoch <= self.cfg.TRAIN.fix_seg_end_epoch:
                    return edge_loss, edge_acc, edge_loss

            if self.cfg.MODEL.uniform_sample == 'BI':
                x_sampled = nn.Upsample(size=self.input_size_net, mode='bilinear')(x)
            else:
                x_sampled = F.grid_sample(x, grid)
            if random.random()>p:
                min_saliency_len = min(self.input_size)
                s = random.randint(min_saliency_len//3, min_saliency_len)
                x_sampled = nn.AdaptiveAvgPool2d((s,s))(x_sampled)
                x_sampled = nn.Upsample(size=self.input_size_net,mode='bilinear')(x_sampled)

            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(x_sampled, return_feature_maps=True))
            elif self.cfg.TRAIN.separate_optimise_deformation:
                pred = self.encoder(x_sampled, return_feature_maps=True)
                pred = self.decoder(x)
            else:
                pred = self.decoder(self.encoder(x_sampled, return_feature_maps=True))
            torch.cuda.reset_max_memory_allocated(0)

        #     # Y sampling
            if not self.cfg.MODEL.naive_upsample:
                if self.cfg.MODEL.loss_at_high_res and self.cfg.MODEL.uniform_sample == 'BI':
                    pred_sampled_train = nn.Upsample(size=ori_size, mode='bilinear')(x)
                elif self.cfg.MODEL.loss_at_high_res:
                    unfilled_mask_2d = torch.isnan(grid_inv_train[:,:,:,0])
                    grid_inv_train[torch.isnan(grid_inv_train)] = 0
                    pred_sampled_train = F.grid_sample(x, grid_inv_train.float())
                    pred_sampled_train[unfilled_mask_2d.unsqueeze(1).expand(pred_sampled_train.shape)] = float('nan')
                    for n in range(pred_sampled_train.shape[0]):
                        pred_sampled_train[n] = self.fillMissingValues_tensor(pred_sampled_train[n], interp_mode=self.cfg.MODEL.rev_deform_interp)

            if self.deep_sup_scale is not None: # use deep supervision technique
                pred, pred_deepsup = pred, pred_deepsup
            if self.cfg.MODEL.naive_upsample:
                pred,image_output,hm = pred,x_sampled,xs
            else: # deform y training
                if self.cfg.MODEL.loss_at_high_res:
                    pred,image_output,hm,_,pred_sampled = pred,x_sampled,xs,y_sampled,pred_sampled_train
                else:
                    pred,image_output,hm,feed_dict['seg_label'] = pred,x_sampled,xs,y_sampled.long()

                # del xs, x_sampled, x
                if self.cfg.MODEL.loss_at_high_res:
                    del pred_sampled_train
                if 'single' not in self.cfg.DATASET.list_train:
                    del y_sampled
            if self.cfg.MODEL.deconv:
                loss = self.crit(ys_deconv, feed_dict['seg_label'])
            else:
                if self.cfg.MODEL.loss_at_high_res:
                    pred_sampled[torch.isnan(pred_sampled)] = 0 # assign residual missing with 0 probability
                    loss = self.crit(pred_sampled, feed_dict['seg_label'])
                else:
                    loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale
            if self.cfg.TRAIN.deform_joint_loss:
                print('seg loss: {}, scaled edge_loss: {}\n'.format(loss, edge_loss))
                loss = loss + edge_loss

            if self.cfg.MODEL.deconv:
                acc = self.pixel_acc(ys_deconv, feed_dict['seg_label'])
            else:
                if self.cfg.MODEL.loss_at_high_res:
                    acc = self.pixel_acc(pred_sampled, feed_dict['seg_label'])
                else:
                    acc = self.pixel_acc(pred, feed_dict['seg_label'])

            if 'single' in self.cfg.DATASET.list_train:
                if self.cfg.TRAIN.deform_joint_loss:
                    return loss, acc, y_print, y_sampled[0].unsqueeze(0), edge_loss
                else:
                    return loss, acc, y_print, y_sampled[0].unsqueeze(0)
            else:
                if self.cfg.TRAIN.deform_joint_loss:
                    return loss, acc, edge_loss
                else:
                    return loss, acc
        # # inference
        else:
            t = time.time()
            xs_inv = 1-xs_hm
            grid, grid_inv = self.create_grid(xs_hm, segSize=segSize, x_inv=xs_inv)
            _, grid_y = self.create_grid(xs_hm, segSize=segSize)
            if self.cfg.MODEL.uniform_sample == 'BI':
                x_sampled = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(x)
            else:
                x_sampled = F.grid_sample(x, grid)
                x_sampled = nn.Upsample(size=self.input_size_net_infer,mode='bilinear')(x_sampled)
            if self.cfg.MODEL.naive_upsample:
                x = self.decoder(self.encoder(x_sampled, return_feature_maps=True), segSize=segSize)
            else:
                segSize_temp = tuple(self.input_size_net_infer)
                print('eval segSize_temp: {}'.format(segSize_temp))
                x = self.decoder(self.encoder(x_sampled, return_feature_maps=True), segSize=segSize_temp)
        #     # Y sampling
            if not self.cfg.MODEL.naive_upsample:
                if self.cfg.MODEL.uniform_sample == 'BI':
                    y_sampled = nn.Upsample(size=tuple(np.array(self.input_size_net_infer)), mode='bilinear')(y.float().unsqueeze(1)).long().squeeze(1)
                else:
                    y_sampled = F.grid_sample(y.float().unsqueeze(1), grid_y, mode='nearest').long().squeeze(1)

                if self.cfg.MODEL.uniform_sample == 'BI' or self.cfg.MODEL.uniform_sample == 'nearest':
                    if self.cfg.MODEL.uniform_sample == 'BI':
                        pred_sampled = nn.Upsample(size=segSize, mode='bilinear')(x)
                    elif self.cfg.MODEL.uniform_sample == 'nearest':
                        pred_sampled = nn.Upsample(size=segSize, mode='nearest')(x)
                    pred_sampled_unfilled_mask_2d = torch.isnan(grid_inv[:,:,:,0])
                    if self.cfg.VAL.y_sampled_reverse:
                        assert (self.cfg.MODEL.rev_deform_interp == 'nearest'), "y_sampled_reverse only appliable to nearest rev_deform_interp"
                        y_sampled_reverse =  nn.Upsample(size=segSize, mode='nearest')(y_sampled.float().unsqueeze(1)).squeeze(1)
                    if self.cfg.VAL.x_sampled_reverse:
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
                                # The normalize code -> t.sub_(m).div_(s)
                            return img
                        x_sampled_unorm = unorm(x_sampled)
                        x_sampled_reverse =  nn.Upsample(size=segSize, mode='bilinear')(x_sampled_unorm)

                elif self.cfg.MODEL.rev_deform_opt == 51:
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

                    if self.cfg.VAL.x_sampled_reverse:
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
                                y_sampled_score_reverse[n] = self.fillMissingValues_tensor(y_sampled_score_reverse[n], interp_mode=self.cfg.MODEL.rev_deform_interp)
                            _, y_sampled_reverse = torch.max(y_sampled_score_reverse, dim=1)
                    del fillMissingValues_tensor
                ## FILL residual missing
                if feed_batch_count != None and not 'single' in self.cfg.DATASET.list_train:
                    if feed_batch_count < self.cfg.VAL.batch_size:
                        self.num_res_nan_percentage = []
                    self.num_res_nan_percentage.append(float(torch.isnan(pred_sampled).sum()*100.0) / float(pred_sampled.shape[0]*pred_sampled.shape[1]*pred_sampled.shape[2]*pred_sampled.shape[3]))
                pred_sampled[torch.isnan(pred_sampled)] = 0 # assign residual missing with 0 probability
            #================ transfer original ... = self.seg_deform ...
            if self.cfg.VAL.no_upsample:
                pred,image_output,hm = x,x_sampled,xs
            elif self.cfg.MODEL.naive_upsample:
                pred,image_output,hm = x,x_sampled,xs
            else:
                if self.cfg.MODEL.rev_deform_opt == 51:
                    pred,image_output,hm,pred_sampled,pred_sampled_unfilled_mask_2d = x,x_sampled,xs,pred_sampled,pred_sampled_unfilled_mask_2d
                else:
                    pred,image_output,hm,pred_sampled = x,x_sampled,xs,pred_sampled
            #================ transfer original ... = self.seg_deform ...
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


                def unorm(img):
                    if 'GLEASON' in self.cfg.DATASET.root_dataset:
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
                        # The normalize code -> t.sub_(m).div_(s)
                    return img

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
                    if self.cfg.MODEL.img_gradient:
                        xhm = vutils.make_grid(xs[i].unsqueeze(0), normalize=True, scale_each=True)
                    else:
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

                    if not self.cfg.MODEL.naive_upsample:
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


                    if not self.cfg.MODEL.naive_upsample:
                        if self.cfg.MODEL.deconv:
                            _, y_deconv = torch.max(ys_deconv, dim=1)
                            pred_sampled_color = colorEncode(as_numpy(y_deconv.squeeze(0)), colors)
                        else:
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

            if writer is not None and feed_batch_count == -1 and feed_batch_count != None and not 'single' in self.cfg.DATASET.list_train:
                print('EVAL: pred_sampled num residual nan percentage: {} %\n'.format(np.array(self.num_res_nan_percentage).mean()))
                writer.add_scalar('Residual_nan_percentage_eval', np.array(self.num_res_nan_percentage).mean(), count)

            if F_Xlr_acc_map:
                if self.cfg.MODEL.naive_upsample:
                    loss = self.crit(pred, feed_dict['seg_label'])
                    return pred, loss
                elif self.cfg.MODEL.deconv:
                    loss = self.crit(ys_deconv, feed_dict['seg_label'])
                else:
                    loss = self.crit(pred_sampled, feed_dict['seg_label'])
                    return pred_sampled, loss
            else:
                if self.cfg.MODEL.naive_upsample:
                    return pred
                elif self.cfg.MODEL.deconv:
                    return ys_deconv
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



def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 2:
            orig_resnet.conv1.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer2.apply(
                partial(self._nostride_dilate, dilate=4))
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=8))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=16))
        elif dilate_scale == 4:
            orig_resnet.layer2.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=4))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=8))
        elif dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)

# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    # def print_decoder_grad_check(self, grad):
    #     # print('decoder_grad:', grad.data)
    #     print('decoder_grad max:', torch.max(grad.data))

    def forward(self, conv_out, segSize=None, res=None):
        conv5 = conv_out[-1]
        # print('debug check got here - deconv? \n')
        # if not self.use_softmax:
        #     conv5.register_hook(self.print_decoder_grad_check)
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if res is not None:
            x += res

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)

# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
