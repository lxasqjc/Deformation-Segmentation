from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.grid_path = ""
_C.DATASET.list_test = ""
_C.DATASET.class_mapping = 0
_C.DATASET.ignore_index = -2
_C.DATASET.num_class = 150
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = "Flip"
_C.DATASET.multi_scale_aug = False
_C.DATASET.adjust_crop_range = False
_C.DATASET.mirror_padding = False
_C.DATASET.binary_class = -1
_C.DATASET.gt_gradient_rm_under_repre = 0.0
_C.DATASET.repeat_sample = 0
_C.DATASET.shuffle_list = True
_C.DATASET.val_central_crop = False
_C.DATASET.val_central_crop_shape = (300, 300)
_C.DATASET.check_dataload = False

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# weights to finetune net_saliency
_C.MODEL.weights_net_saliency = ""
# weights to finetune net_compress
_C.MODEL.weights_net_compress = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# option to disable track_running_stats
_C.MODEL.track_running_stats = True
_C.MODEL.fov_deform = False
_C.MODEL.naive_upsample = True
_C.MODEL.deconv = False
_C.MODEL.rev_deform_opt = 51 # 51-fill missing by tensor (diffrienable) based triangular filling;
_C.MODEL.rev_deform_interp = 'tri' # different reverse deformation solutions
_C.MODEL.loss_at_high_res = False # choose whether calculate loss after reverse deformation at high resolution domain
_C.MODEL.img_gradient = False
_C.MODEL.saliency_net = 'fovsimple'
# set uniform_sample = 'Saliency' to sample/rev_sample based on uniform saliency
# set uniform_sample = 'BI' to sample/rev_sample by nn.upsample
_C.MODEL.uniform_sample = ''
# optional adjust saliency output size, default same size as saliency_input_size
_C.MODEL.saliency_output_size_short = 0
# fixed gaussian saliency for debug purpose
# 1: central gaussian, 2: two eye gaussian
_C.MODEL.fix_saliency = 0
# optional gaussian kernel radius (gaussian kernel size = 2*radius+1)
_C.MODEL.gaussian_radius = 30
# optional change gaussian aspect ratio
_C.MODEL.gaussian_ap = 0.0
# input image channel
_C.MODEL.in_dim = 3
_C.MODEL.fix_img_gradient = False
_C.MODEL.gt_gradient = False
_C.MODEL.gt_gradient_intrinsic_only = True
_C.MODEL.fix_gt_gradient = False
_C.MODEL.ignore_gt_labels = []
_C.MODEL.gt_grad_gaussian_blur_r = 1

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.auto_batch = 'manual'
_C.TRAIN.gpu_threshold = 0.65e6
_C.TRAIN.batch_size_per_gpu = 2
_C.TRAIN.num_gpus = 1
# number of iterations per batch
_C.TRAIN.fov_location_step = 1
# default fov_location_step = num of pixels in Xlr
_C.TRAIN.auto_fov_location_step = False
_C.TRAIN.sync_location = 'mean_mbs'
_C.TRAIN.mini_batch_size = 1
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
_C.TRAIN.loss_fun = "FocalLoss"
_C.TRAIN.loss_weight = []
_C.TRAIN.scale_weight = ""
# s_entropy_weight should be negative as we want high entropy (more uniform distribution)
_C.TRAIN.s_entropy_weight = -1.0
_C.TRAIN.optim = "SGD"
_C.TRAIN.fov_scale_pow = 1 # scale distribution of lr_scale/wd_scale
# ***use fov average patch size to scale learning rate (ini imp at 4th Mar 2020)
_C.TRAIN.fov_scale_lr = ''
# ***use fov average patch size to scale weight decay (ini imp at 4th Mar 2020)
_C.TRAIN.fov_scale_weight_decay = ''
_C.TRAIN.fov_scale_seg_only = False
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
_C.TRAIN.lr_foveater = 0.02
# lr_mult
_C.TRAIN.lr_mult_encoder = 0.0001
_C.TRAIN.lr_mult_decoder = 0.0001
_C.TRAIN.lr_mult_saliency = 0.001
_C.TRAIN.lr_mult_compress = 0.001
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# optianl switch to scale by iter
_C.TRAIN.scale_by_iter = False
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.weight_decay_fov = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16
# global epoch count
_C.TRAIN.global_epoch = 1

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304
# whether save checkpoint of last epoch
_C.TRAIN.save_checkpoint = True
# number of epochs to perform eval_during_train
_C.TRAIN.eval_per_epoch = 1
# number of epochs to save checkpoints
_C.TRAIN.checkpoint_per_epoch = 2000

# entropy regularisation
_C.TRAIN.entropy_regularisation = False
_C.TRAIN.entropy_regularisation_weight = 1.0

# deformation
_C.TRAIN.task_input_size = (1024,2048)
_C.TRAIN.task_input_size_eval = ()
_C.TRAIN.saliency_input_size = (256,512)
_C.TRAIN.deform_pretrain_bol = True
_C.TRAIN.deform_pretrain = 100
_C.TRAIN.fix_deform_aft_pretrain = False
_C.TRAIN.fix_deform_start_epoch = 2000
_C.TRAIN.fix_deform_end_epoch = 2001
_C.TRAIN.smooth_deform_2nd_start = 2001
_C.TRAIN.smooth_deform_2nd_end = 2001

_C.TRAIN.separate_optimise_deformation = False
_C.TRAIN.opt_deform_LabelEdge = False
# _C.TRAIN.opt_deform_LabelEdge_sepTrainPre = 999
_C.TRAIN.fix_seg_start_epoch = 2000
_C.TRAIN.fix_seg_end_epoch = 2001
_C.TRAIN.opt_deform_LabelEdge_accrate = 1.0
_C.TRAIN.opt_deform_LabelEdge_softmax = True
_C.TRAIN.opt_deform_LabelEdge_norm = True
_C.TRAIN.deform_joint_loss = False
_C.TRAIN.edge_loss_scale = 100.0
_C.TRAIN.rescale_regressed_xs = False
_C.TRAIN.fixed_edge_loss_scale = -1.0
_C.TRAIN.edge_loss_pow = 0.9
_C.TRAIN.edge_loss_scale_min = 0.0
_C.TRAIN.stage_adjust_edge_loss = 1.0
_C.TRAIN.adjust_edge_loss_start_epoch = 2000
_C.TRAIN.adjust_edge_loss_end_epoch = 2001
_C.TRAIN.def_saliency_pad_mode = 'replication'
_C.TRAIN.dynamic_task_input = (1,1)
_C.TRAIN.dynamic_saliency_relative_size = 1.0
_C.TRAIN.deform_zero_bound = False
_C.TRAIN.deform_zero_bound_factor = 1
_C.TRAIN.skip_train_for_eval = False
_C.TRAIN.train_eval_visualise = False

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"
# visualize hard max fov maps
_C.VAL.hard_max_fov = False
_C.VAL.max_score = False
_C.VAL.central_crop = False
# save F_Xlr_time for all val data
_C.VAL.all_F_Xlr_time = False
_C.VAL.rename_eval_folder = ""
_C.VAL.multipro = False # currently not supported
_C.VAL.dice = False
_C.VAL.hd95 = False
_C.VAL.F_Xlr_only = False
_C.VAL.F_Xlr_acc_map_only = False
_C.VAL.foveated_expection = True
# option to output score_maps of fixed patch baselines for later ensemble, currently used
_C.VAL.ensemble = False
# option to ensemble fixed patch baselines, NEED TEST
_C.VAL.approx_pred_Fxlr_by_ensemble = False
# downsample F_Xlr for efficient inference
_C.VAL.F_Xlr_low_scale = 0
_C.VAL.expand_prediection_rate = 1 # =2 for HRnet in cityscapes
_C.VAL.expand_prediection_rate_patch = 1.0 # =2 for HRnet in cityscapes
# option to disable upsample
_C.VAL.no_upsample = False
# option to write pred to tensorboard
_C.VAL.write_pred = False
# use existing evaluation script as test
_C.VAL.test = False
# optional evaluate y_sampled_reverse
_C.VAL.y_sampled_reverse = False
_C.VAL.x_sampled_reverse = False
_C.VAL.report_per_img_iou = False
_C.VAL.trimap = False
_C.VAL.trimap_dia_factor = 5
_C.VAL.trimap_visual_check = False
# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
