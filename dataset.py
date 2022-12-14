import os, os.path, random
import json
import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import albumentations as A

def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy())
    return img

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def b_imresize(im, size, interp='bilinear'):
    return F.interpolate(im, size, mode=interp)

# from HRnet
def multi_scale_aug(image, label=None):
    # print('image_shape: ', image.shape)
    # print('label_shape: ', label.shape)
    rand_scale = 0.5 + random.randint(0, 16) / 10.0
    long_size = np.int(2048 * rand_scale + 0.5)
    w, h = image.shape[-2:]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)

    image = F.interpolate(image, (new_w, new_h), mode='bilinear')

    if label is not None:
        label = F.interpolate(label.unsqueeze(1).float(), (new_w, new_h), mode='nearest').squeeze(1).long()
    else:
        return image
    return image, label


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # NOTE: dataset classes SHOULD be mapped with minimum starting from [1:] NOT [0:]
        # remapping labels reflecting servie degree of GS
        if opt.root_dataset == '/scratch0/chenjin/GLEASON2019_DATA/Data/' or \
        opt.root_dataset == '/home/chenjin/Chen_UCL/Histo-MRI-mapping/GLEASON2019_DATA/Data/' or \
        opt.root_dataset == '/SAN/medic/Histo_MRI_GPU/chenjin/Data/GLEASON2019_DATA/Data/' or \
        'GLEASON2019_DATA' in opt.root_dataset:
            # four class mapping
            if opt.class_mapping == 0:
                self.label_mapping = {0: 1,
                                      1: 1, 2: 1,
                                      3: 2, 4: 3,
                                      5: 4, 6: 1,
                                      }
            # three class mapping exclude class5
            elif opt.class_mapping == 30:
                self.label_mapping = {0: 1,
                                      1: 1, 2: 1,
                                      3: 2, 4: 3,
                                      5: 1, 6: 1,
                                      }
            # gs3 vs all
            elif opt.class_mapping == 3:
                self.label_mapping = {0: 1,
                                      1: 1, 2: 1,
                                      3: 2, 4: 1,
                                      5: 1, 6: 1,
                                      }
            # gs4 vs all
            elif opt.class_mapping == 4:
                self.label_mapping = {0: 1,
                                      1: 1, 2: 1,
                                      3: 1, 4: 2,
                                      5: 1, 6: 1,
                                      }
            # gs5 vs all
            elif opt.class_mapping == 5:
                self.label_mapping = {0: 1,
                                      1: 1, 2: 1,
                                      3: 1, 4: 1,
                                      5: 2, 6: 1,
                                      }
            # benine vs all
            elif opt.class_mapping == 6:
                self.label_mapping = {0: 1,
                                      1: 1, 2: 1,
                                      3: 2, 4: 2,
                                      5: 2, 6: 1,
                                      }
            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.748, 0.611, 0.823],
                std=[0.146, 0.245, 0.119])


        elif opt.root_dataset == '/home/chenjin/Chen_UCL/Histo-MRI-mapping/DigestPath2019/' or 'Digest' in opt.list_train:
            self.label_mapping = {0: 1,
                                  255: 2,
                                  }
            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.816, 0.697, 0.792],
                std=[0.160, 0.277, 0.198])
        elif 'ADE20K' in opt.root_dataset or 'ADE' in opt.list_train:
            self.label_mapping = {}
            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        elif 'CITYSCAPES' in opt.root_dataset or 'CITYSCAPE' in opt.list_train:
            # following HRNet-Semantic-Segmentation setting
            # but starting from 1 instead of 0, seems 0 leads to bug in criterion.OhemCrossEntropy implementation
            # debug note 24/12/19 seems label must start from 1 and must be continues, otherwise lead inconsistence between pred by view(-1) and seg_label
            ignore_label=20
            if opt.binary_class != -1:
                self.label_mapping = {}
                for i in range(-1,34):
                    if i in (-1,0,1,2,3,4,5,6,9,10,14,15,16,18,29,30):
                        self.label_mapping[i] = ignore_label
                    if i == opt.binary_class:
                        self.label_mapping[i] = 2
                    else:
                        self.label_mapping[i] = 1
            else:
                self.label_mapping = {-1: ignore_label, 0: ignore_label,
                                      1: ignore_label, 2: ignore_label,
                                      3: ignore_label, 4: ignore_label,
                                      5: ignore_label, 6: ignore_label,
                                      7: 1, 8: 2, 9: ignore_label,
                                      10: ignore_label, 11: 3, 12: 4,
                                      13: 5, 14: ignore_label, 15: ignore_label,
                                      16: ignore_label, 17: 6, 18: ignore_label,
                                      19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12,
                                      25: 13, 26: 14, 27: 15, 28: 16,
                                      29: ignore_label, 30: ignore_label,
                                      31: 17, 32: 18, 33: 19}
            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        elif 'Histo' in opt.root_dataset or 'histomri' in opt.list_train or 'histomri' in opt.root_dataset:
            self.label_mapping = {}
            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.8223, 0.7783, 0.7847],
                std=[0.210, 0.216, 0.241])
        elif 'DeepGlob' in opt.root_dataset or 'deepglob' in opt.root_dataset or 'DeepGlob' in opt.list_train:
            # ignore_label=7
            if opt.ignore_index == 0:
                self.label_mapping = {0: 2,
                                      1: 3, 2: 4,
                                      3: 5, 4: 6,
                                      5: 7, 6: 1,
                                      }
            elif opt.ignore_index == 6:
                self.label_mapping = {0: 1,
                                      1: 2, 2: 3,
                                      3: 4, 4: 5,
                                      5: 6, 6: 7,
                                      }
            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.282, 0.379, 0.408],
                std=[0.089, 0.101, 0.127])
        elif 'Face_single_example' in opt.root_dataset or 'Face_single_example' in opt.list_train:
            # ignore_label=7
            self.label_mapping = {201: 1,
                                  184: 2, 173: 3,
                                  191: 4, 194: 5
                                  }

            # mean and std
            self.normalize = transforms.Normalize(
                # gleason2019 322 train mean and std applied
                mean=[0.282, 0.379, 0.408],
                std=[0.089, 0.101, 0.127])
        else:
            raise Exception('Unknown root for mapping and normalisation!')
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)



    def convert_label(self, label, inverse=False):
        label = np.array(label)
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def convert_label_single_face(self, label, inverse=False):
        label = np.array(label)
        temp = label.copy()
        if inverse:
            pass
            # for v, k in self.label_mapping.items():
            #     label[temp == k] = v
        else:
            print('max label before mapping: {}'.format(np.max(label)))
            for k, v in self.label_mapping.items():
                label[temp == k] = v
            label[label == temp] = 1
            print('max label after mapping: {}'.format(np.max(label)))

            # label_base = np.zeros(label.shape)
            # label = np.maximum(label, label_base)

        return label

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def img_transform_unnorm(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())
        return img

    def img_transform_rev(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        # !!!!! JC: This is why all data need to mapped to 1-numClass
        # and because of this, ignore_index (in CrossEntropy/OhemCrossEntropy/IoU) = ignore_label (in dataset class_mapping)-1
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, cal_REV=False, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

        # augmentation
        self.augmentation = opt.random_flip
        self.balance_sam_idx = 0
        self.num_class = opt.num_class

        self.cal_REV = cal_REV
        self.opt = opt

        if self.opt.repeat_sample != 0:
            self.repeat_sample_count = 0

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]

            # when gt_gradient, make sure loaded sample got enough foreground pixels
            if self.opt.gt_gradient_rm_under_repre > 0 and self.opt.num_class == 2:
                search_well_repre_class = True
                s_idx = self.cur_idx
                while search_well_repre_class:
                    search_sample = self.list_sample[s_idx]
                    s_idx += 1
                    if s_idx >= self.num_sample:
                        s_idx = 0
                    segm_path = os.path.join(self.root_dataset, search_sample['fpath_segm'])
                    segm = self.convert_label(Image.open(segm_path))
                    segm -= 1
                    hist, _ = np.histogram(segm, bins=self.num_class, range=(0, self.num_class-1))
                    if (hist[-1] / np.sum(hist)) > self.opt.gt_gradient_rm_under_repre:
                        this_sample = search_sample
                        search_well_repre_class = False
                self.cur_idx = s_idx
            if self.augmentation == 'balance_sample' and self.balance_sam_idx > 2:
                # search gs-5 and reset idx every 3 steps represent
                # severe rare gs-4 in contrast to balanced other 3 classes
                search_rare_class = True
                s_idx = self.cur_idx
                while search_rare_class:
                    search_sample = self.list_sample[s_idx]
                    s_idx += 1
                    if s_idx >= self.num_sample:
                        s_idx = 0
                    segm_path = os.path.join(self.root_dataset, search_sample['fpath_segm'])
                    segm = self.convert_label(Image.open(segm_path))
                    hist, _ = np.histogram(segm, bins=self.num_class, range=(0, self.num_class-1))
                    if (hist[-1] / np.sum(hist)) > 0.25:
                        this_sample = search_sample
                        search_rare_class = False
                self.balance_sam_idx = 0
            self.balance_sam_idx += 1
            # print('this sample fpath_segm: {}'.format(this_sample['fpath_segm']))
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            if self.opt.repeat_sample != 0:
                self.repeat_sample_count += 1
                if self.repeat_sample_count % self.opt.repeat_sample == 0:
                    self.cur_idx += 1
                    self.repeat_sample_count = 0
            else:
                self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled and self.opt.shuffle_list:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            # print('batch_records_height: {}, batch_records_width: {}'.format(batch_records[i]['height'], batch_records[i]['width']))
            if self.imgMaxSize == 1:
                # discard rescalel, keep original image size
                this_scale = 1
            else:
                this_scale = min(
                    this_short_size / min(img_height, img_width), \
                    self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale
            # print('this_scale: {}'.format(this_scale))
            # print('batch_widths[i]: {}, batch_heights[i]: {}'.format(batch_widths[i], batch_heights[i]))

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'


        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            # skip non exitst Training
            if not os.path.isfile(segm_path):
                continue

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)

            if self.opt.check_dataload and i == 0:
                img_ori = self.img_transform(img)


            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])
            # print(img.size)
            # random_flip


            if self.augmentation == 'Flip':
                if np.random.choice([0, 1]):
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.augmentation == 'Flip_Pixel':
                aug = A.Compose([
                    A.GaussNoise(),
                    A.RandomBrightnessContrast(),
                    A.Flip()
                ],p=1)
                img = np.array(img)
                segm = np.array(segm)
                augmented = aug(image=img, mask=segm)
                img = Image.fromarray(augmented['image'])
                segm = Image.fromarray(augmented['mask'])
            elif self.augmentation == 'cityHRaug':
                aug = A.Compose([
                    A.RandomScale(scale_limit=[0.5,2.0]),
                    A.RandomCrop(512, 1024),
                    A.HorizontalFlip()
                ],p=1)
                img = np.array(img)
                segm = np.array(segm)
                augmented = aug(image=img, mask=segm)
                img = Image.fromarray(augmented['image'])
                segm = Image.fromarray(augmented['mask'])
            elif self.augmentation == 'balance_sample' and (i+1) % 4 == 0:
                aug = A.Compose([
                    A.RandomCrop(self.imgSizes[0], self.imgSizes[1]),
                    A.Flip()
                ],p=1)
                img = np.array(img)
                segm = np.array(segm)
                search_rare = True
                while search_rare:
                    augmented = aug(image=img, mask=segm)
                    segm_s = self.convert_label(augmented['mask'])
                    hist, _ = np.histogram(segm_s, bins=self.num_class, range=(0, self.num_class-1))
                    if (hist[-1] / np.sum(hist)) > 0.25:
                        img = Image.fromarray(augmented['image'])
                        segm = Image.fromarray(augmented['mask'])
                        search_rare = False
            elif self.augmentation == 'fullFoV_balance_sample' and (i+1) % 4 == 0:
                img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
                segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')
                aug = A.Compose([
                    A.RandomCrop(batch_widths[i], batch_heights[i]),
                    A.Flip()
                ],p=1)
                img = np.array(img)
                segm = np.array(segm)
                search_rare = True
                while search_rare:
                    augmented = aug(image=img, mask=segm)
                    segm_s = self.convert_label(augmented['mask'])
                    hist, _ = np.histogram(segm_s, bins=self.num_class, range=(0, self.num_class-1))
                    if (hist[-1] / np.sum(hist)) > 0.25:
                        img = Image.fromarray(augmented['image'])
                        segm = Image.fromarray(augmented['mask'])
                        search_rare = False
            elif self.augmentation.split("_")[0] == 'Crop':
                if self.augmentation == 'Crop_Flip':
                    crop_w = self.imgSizes[0]
                    crop_h = self.imgSizes[1]
                else:
                    crop = int(self.augmentation.split("_")[-1])
                    if img.size[0] < crop or img.size[1] < crop:
                        crop_h, crop_w = img.size[0], img.size[1]
                    else:
                        crop_w, crop_h = crop, crop
                if self.augmentation.split("_")[1] == 'aug':
                    aug = A.Compose([
                        # A.RandomSizedCrop((1250, 2500), 2500, 2500),
                        # A.ShiftScaleRotate(),
                        A.RandomCrop(crop_w, crop_h),
                        A.RandomBrightnessContrast(),
                        A.MultiplicativeNoise(),
                        A.Flip()
                        # A.RGBShift(),
                        # A.Blur(),
                        # A.GaussNoise(),
                        # A.ElasticTransform(),
                        # A.Cutout(p=1)
                    ],p=1)
                else:
                    aug = A.Compose([
                        # A.RandomSizedCrop((1250, 2500), 2500, 2500),
                        # A.ShiftScaleRotate(),
                        A.RandomCrop(crop_w, crop_h),
                        A.Flip()
                        # A.RGBShift(),
                        # A.Blur(),
                        # A.GaussNoise(),
                        # A.ElasticTransform(),
                        # A.Cutout(p=1)
                    ],p=1)
                img = np.array(img)
                segm = np.array(segm)
                augmented = aug(image=img, mask=segm)
                img = Image.fromarray(augmented['image'])
                segm = Image.fromarray(augmented['mask'])

            # print(img.size)
            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor CxHxW
            if self.cal_REV:
                img = self.img_transform_rev(img)
            else:
                img = self.img_transform(img)
            # JC: re-ordering label according to servie degree
            if 'Face_single_example' in self.opt.root_dataset or 'Face_single_example' in self.opt.list_train:
                segm = self.convert_label_single_face(segm)
            else:
                segm = self.convert_label(segm)
            # JC: for dataset like gleason2019 that background labelled 0 need to be considered
            if ('GLEASON' in self.root_dataset and np.min(segm) == 0) or 'histomri' in self.opt.list_train or 'histomri' in self.opt.root_dataset:
                segm = Image.fromarray(np.add(segm, 1))
            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            # print('train batch_image_{}_size: {}\n'.format(i, img.shape))
            # print('train batch_segm_{}_size: {}\n'.format(i, segm.shape))
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms # torch.nn.functional.one_hot(batch_segms, num_classes=7)
        if self.opt.check_dataload:
            output['img_ori'] = img_ori
        # print('seg_label shape: {}'.format(output['seg_label'].shape))
        # print('seg_label size: {}'.format(output['seg_label'].shape))
        if self.cal_REV:
            return batch_images
        else:
            return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, cfg, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.cfg = cfg
        if cfg.VAL.expand_prediection_rate != 1:
            self.imgSizes_val = tuple(cfg.VAL.expand_prediection_rate*np.array(self.imgSizes))
            self.imgMaxSize_val = cfg.VAL.expand_prediection_rate*self.imgMaxSize
        else:
            self.imgSizes_val = self.imgSizes
            self.imgMaxSize_val = self.imgMaxSize

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)

        if self.cfg.DATASET.val_central_crop:
            width, height = img.size
            new_width, new_height = self.cfg.DATASET.val_central_crop_shape

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))
            segm = segm.crop((left, top, right, bottom))

            self.imgMaxSize_val = 1


        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size
        # print('ori_width = {}, ori_height = {}'.format(ori_width, ori_height))

        img_resized_list = []
        img_resized_list_unnorm = []
        for this_short_size in self.imgSizes_val:
            if self.imgMaxSize_val == 1:
                # discard 1st downsample in foveation model, i.e. creat foveation map on original image
                scale = 1
            else:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            self.imgMaxSize_val / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)
            # print('scaled target_height = {}, target_width = {}'.format(target_height, target_width))

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            # print('rounded target_height = {}, target_width = {}'.format(target_height, target_width))

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized_unnorm = self.img_transform_unnorm(img_resized)
            img_resized = self.img_transform(img_resized)

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
            img_resized_unnorm = torch.unsqueeze(img_resized_unnorm, 0)
            img_resized_list_unnorm.append(img_resized_unnorm)

        # JC: re-ordering label according to servie degree
        if 'Face_single_example' in self.cfg.DATASET.root_dataset or 'Face_single_example' in self.cfg.DATASET.list_train:
            segm = self.convert_label_single_face(segm)
        else:
            segm = self.convert_label(segm)
        # JC: for dataset like gleason2019 that background labelled 0 need to be considered
        if ('GLEASON' in self.root_dataset and np.min(segm) == 0) or 'histomri' in self.cfg.DATASET.list_train or 'histomri' in self.cfg.DATASET.root_dataset:
            segm = Image.fromarray(np.add(segm, 1))
        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)

        batch_segms = torch.unsqueeze(segm, 0)
        # batch_segms = torch.nn.functional.one_hot(batch_segms, num_classes=7)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['img_data_unnorm'] = [x.contiguous() for x in img_resized_list_unnorm]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, cfg, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)
        self.opt = opt
        self.cfg = cfg

    def __getitem__(self, index):
        def crop_image(image_path):
            list_crop_imgs = []
            coordinate_list = []
            img_path = image_path
            if 'gleason2019' in self.opt.list_train:
                patch_size = int(self.opt.list_train.split('train268_')[1].split('_')[0])
            else:
                patch_size = 5000
            patch_size_x = patch_size
            patch_size_y = patch_size
            overlap = 0
            count = 0
            print(image_path)
            image = cv2.imread(image_path)
            ori_size = (image.shape[1], image.shape[0])

            x_iter_num = (image.shape[1]+overlap)//(patch_size_x-overlap)
            y_iter_num = (image.shape[0]+overlap)//(patch_size_y-overlap)
            for xi in range(x_iter_num+1):
                for yi in range(y_iter_num+1):
                    if xi == 0 or image.shape[1] < patch_size_x:
                        cx = 0
                    elif xi == x_iter_num:
                        cx = image.shape[1]-patch_size_x
                    else:
                        cx = xi*patch_size_x-overlap
                    if yi == 0 or image.shape[0] < patch_size_y:
                        cy = 0
                    elif yi == y_iter_num:
                        cy = image.shape[0]-patch_size_y
                    else:
                        cy = yi*patch_size_y-overlap
                    if image.shape[1] < patch_size_x:
                        patch_size_x_cur = image.shape[1]
                    else:
                        patch_size_x_cur = patch_size_x
                    if image.shape[0] < patch_size_y:
                        patch_size_y_cur = image.shape[0]
                    else:
                        patch_size_y_cur = patch_size_y

                    crop_ti = image[cy:cy+patch_size_y_cur, cx:cx+patch_size_x_cur]
                    crop_ti = Image.fromarray(crop_ti)
                    coordinate_list.append([cx, cy])
                    list_crop_imgs.append(crop_ti)

            return list_crop_imgs, coordinate_list, ori_size

        this_record = self.list_sample[index]
        # load image
        image_path = os.path.join(self.opt.root_dataset, this_record['fpath_img'])
        list_crop_imgs, coordinate_list, ori_size = crop_image(image_path)
        crop_outputs = []
        for crop_img in list_crop_imgs:
            img = crop_img.convert('RGB')

            ori_width, ori_height = img.size

            img_resized_list = []
            for this_short_size in self.imgSizes:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            self.imgMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)

                # to avoid rounding in network
                target_width = self.round2nearest_multiple(target_width, self.padding_constant)
                target_height = self.round2nearest_multiple(target_height, self.padding_constant)

                # resize images
                img_resized = imresize(img, (target_width, target_height), interp='bilinear')

                # image transform, to torch float tensor 3xHxW
                img_resized = self.img_transform(img_resized)
                img_resized = torch.unsqueeze(img_resized, 0)
                img_resized_list.append(img_resized)

            output = dict()
            output['img_ori'] = np.array(img)
            output['img_data'] = [x.contiguous() for x in img_resized_list]
            output['info'] = this_record['fpath_img']
            crop_outputs.append(output)
        return [crop_outputs, coordinate_list, ori_size]

    def __len__(self):
        return self.num_sample
