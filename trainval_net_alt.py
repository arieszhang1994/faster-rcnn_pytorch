# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN 
# 4-step Alternating Training version
# Licensed under The MIT License [see LICENSE for details]
# Written by Aries Zhang, based on code from Jianwei Yang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.alt_parts.vgg16 import vgg16_step1, vgg16_step2, vgg16_step3, vgg16_step4
from model.faster_rcnn.alt_parts.resnet import resnet_step1, resnet_step2, resnet_step3, resnet_step4 

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=[1,1,1,1], nargs='+', type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)

# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  '''
  decide the order of the training data 
  '''
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    # we need leftover_flag to see whether training data can be devided exactly by batch_size
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  # imdb(not used)
  # roidb[0]: example:{'boxes': array([[262, 210, 323, 338],
  #     [164, 263, 252, 371],
  #     [  4, 243,  66, 373],
  #     [240, 193, 294, 298],
  #     [276, 185, 311, 219]], dtype=uint16), 
  # 'gt_classes': array([9, 9, 9, 9, 9], dtype=int32), 
  # 'gt_ishard': array([0, 0, 1, 0, 1], dtype=int32), 
  # 'gt_overlaps': <5x21 sparse matrix of type '<class 'numpy.float32'>'with 5 stored elements in Compressed Sparse Row format>, 
  # 'flipped': False, 'seg_areas': array([7998., 9701., 8253., 5830., 1260.], dtype=float32), 
  # 'img_id': 0, 
  # 'image': '/home/arieszhang/workspace/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg', 
  # 'width': 500, 'height': 375, 
  # 'max_classes': array([9, 9, 9, 9, 9]), 
  # 'max_overlaps': array([1., 1., 1., 1., 1.], dtype=float32), 
  # 'need_crop': 0}
  # ratio_list: rank roidb based on the ratio between width and height. after sorted
  # ratio_index: np.argsort(ratio_list)
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/alt"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  step_of_train  = 1

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  iters_per_epoch = int(train_size / args.batch_size)

  #------------------------------------------
  #4-step alternating training
  #------------------------------------------
  if step_of_train == 1:
    #--------------------------------------------------
    # step 1: train the RPN for the region proposal task
    #---------------------------------------------------
    # initilize the network here.
    if args.net == 'vgg16':
      fasterRCNN_step1 = vgg16_step1(imdb.classes, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN_step1 = resnet_step1(imdb.classes, num_layers=101, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN_step1 = resnet_step1(imdb.classes, num_layers=50, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN_step1 = resnet_step1(imdb.classes, num_layers=152, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()

    params_step1 = []
    for key, value in dict(fasterRCNN_step1.named_parameters()).items():
      if value.requires_grad and 'RCNN_rpn.' in key:
        if 'bias' in key:
          params_step1 += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                  'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params_step1 += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
      lr = lr * 0.1
      optimizer_step1 = torch.optim.Adam(params_step1)
    elif args.optimizer == "sgd":
      optimizer_step1 = torch.optim.SGD(params_step1, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
      fasterRCNN_step1.cuda()

    if args.mGPUs:
      fasterRCNN_step1 = nn.DataParallel(fasterRCNN_step1)
    lr_start = lr

    for epoch in range(args.start_epoch, args.max_epochs[0] + 1):
      # setting to train mode
      fasterRCNN_step1.train()
      loss_temp = 0
      start = time.time()

      if epoch % (args.lr_decay_step + 1) == 0:
          adjust_learning_rate(optimizer_step1, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      data_iter = iter(dataloader)
      for step in range(iters_per_epoch):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        fasterRCNN_step1.zero_grad()
        rois, \
        rpn_loss_cls, rpn_loss_box = fasterRCNN_step1(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
        loss_temp += loss.item()

        # backward
        optimizer_step1.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN_step1, 10.)
        optimizer_step1.step()

        if step % args.disp_interval == 0:
          end = time.time()
          if step > 0:
            loss_temp /= (args.disp_interval + 1)

          if args.mGPUs:
            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_box = rpn_loss_box.mean().item()
          else:
            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box = rpn_loss_box.item()

          print("[session %d][step %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                  % (args.session, step_of_train, epoch, step, iters_per_epoch, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (0, 0, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, 0, 0))
          if args.use_tfboard:
            info = {
              'step_of_train': step_of_train,
              'loss': loss_temp,
              'loss_rpn_cls': loss_rpn_cls,
              'loss_rpn_box': loss_rpn_box,
              'loss_rcnn_cls': 0,
              'loss_rcnn_box': 0
            }
            logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          loss_temp = 0
          start = time.time()
    
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_step{}_{}_{}.pth'.format(args.session, step_of_train,epoch, step))
      save_checkpoint({
        'session': args.session,
        'step_of_train':step_of_train,
        'epoch': epoch + 1,
        'model': fasterRCNN_step1.module.state_dict() if args.mGPUs else fasterRCNN_step1.state_dict(),
        'optimizer': optimizer_step1.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))
    step_of_train += 1

  if step_of_train == 2:
    #--------------------------------------------------
    # step 2: train the Fast-RCNN using the propapsals generated by step-1 RPN
    #---------------------------------------------------
    # initilize the network here.
    if args.net == 'vgg16':
      fasterRCNN_step2 = vgg16_step2(imdb.classes, step1_model_path=os.path.join(output_dir,'faster_rcnn_1_step1_20_5010.pth'), \
                  fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN_step2 = resnet_step2(imdb.classes, step1_model_path=os.path.join(output_dir,'faster_rcnn_1_step1_20_5010.pth'), \
                  num_layers=101, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN_step2 = resnet_step2(imdb.classes, step1_model_path=os.path.join(output_dir,'faster_rcnn_1_step1_20_5010.pth'), \
                  num_layers=50, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN_step2 = resnet_step2(imdb.classes, step1_model_path=os.path.join(output_dir,'faster_rcnn_1_step1_20_5010.pth'), \
                  num_layers=152, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()
      
    params_step2 = []
    for key, value in dict(fasterRCNN_step2.named_parameters()).items():
      if value.requires_grad and 'detector.' in key:
        if 'bias' in key:
          params_step2 += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                  'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params_step2 += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
      lr = lr * 0.1
      optimizer_step2 = torch.optim.Adam(params_step2)
    elif args.optimizer == "sgd":
      optimizer_step2 = torch.optim.SGD(params_step2, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
      fasterRCNN_step2.cuda()

    if args.mGPUs:
      fasterRCNN_step2 = nn.DataParallel(fasterRCNN_step2)
    lr_start = lr

    for epoch in range(args.start_epoch, args.max_epochs[1] + 1):
      # setting to train mode
      fasterRCNN_step2.train()
      loss_temp = 0
      start = time.time()

      if epoch % (args.lr_decay_step + 1) == 0:
          adjust_learning_rate(optimizer_step2, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      data_iter = iter(dataloader)
      for step in range(iters_per_epoch):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        fasterRCNN_step2.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN_step2(im_data, im_info, gt_boxes, num_boxes)

        loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()

        # backward
        optimizer_step2.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN_step2, 10.)
        optimizer_step2.step()

        if step % args.disp_interval == 0:
          end = time.time()
          if step > 0:
            loss_temp /= (args.disp_interval + 1)

          if args.mGPUs:
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_box = RCNN_loss_bbox.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
          else:
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

          print("[session %d][step %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                  % (args.session, step_of_train, epoch, step, iters_per_epoch, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (0, 0, loss_rcnn_cls, loss_rcnn_box))
          if args.use_tfboard:
            info = {
              'step_of_train': step_of_train,
              'loss': loss_temp,
              'loss_rpn_cls': 0,
              'loss_rpn_box': 0,
              'loss_rcnn_cls': loss_rcnn_cls,
              'loss_rcnn_box': loss_rcnn_box
            }
            logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          loss_temp = 0
          start = time.time()
    
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_step{}_{}_{}.pth'.format(args.session, step_of_train,epoch, step))
      save_checkpoint({
        'session': args.session,
        'step_of_train':step_of_train,
        'epoch': epoch + 1,
        'model': fasterRCNN_step2.module.state_dict() if args.mGPUs else fasterRCNN_step2.state_dict(),
        'optimizer': optimizer_step2.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))
    step_of_train += 1

  if step_of_train == 3:
    #--------------------------------------------------
    # step 3: train the RPN for the region proposal task 
    #---------------------------------------------------
    if args.net == 'vgg16':
      fasterRCNN_step3 = vgg16_step3(imdb.classes, step2_model_path=os.path.join(output_dir, 'faster_rcnn_1_step2_20_5010.pth'), \
                  fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN_step3 = resnet_step3(imdb.classes, step2_model_path=os.path.join(output_dir, 'faster_rcnn_1_step2_20_5010.pth'), \
                  num_layers=101, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN_step3 = resnet_step3(imdb.classes, step2_model_path=os.path.join(output_dir, 'faster_rcnn_1_step2_20_5010.pth'), \
                  num_layers=50, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN_step3 = resnet_step3(imdb.classes, step2_model_path=os.path.join(output_dir, 'faster_rcnn_1_step2_20_5010.pth'), \
                  num_layers=152, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()

    params_step3 = []
    for key, value in dict(fasterRCNN_step3.named_parameters()).items():
      if value.requires_grad and 'RCNN_rpn.' in key:
        if 'bias' in key:
          params_step3 += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                  'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params_step3 += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
      lr = lr * 0.1
      optimizer_step3 = torch.optim.Adam(params_step3)
    elif args.optimizer == "sgd":
      optimizer_step3 = torch.optim.SGD(params_step3, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
      fasterRCNN_step3.cuda()

    if args.mGPUs:
      fasterRCNN_step3 = nn.DataParallel(fasterRCNN_step3)
    lr_start = lr

    for epoch in range(args.start_epoch, args.max_epochs[2] + 1):
      # setting to train mode
      fasterRCNN_step3.train()
      loss_temp = 0
      start = time.time()

      if epoch % (args.lr_decay_step + 1) == 0:
          adjust_learning_rate(optimizer_step3, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      data_iter = iter(dataloader)
      for step in range(iters_per_epoch):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        fasterRCNN_step3.zero_grad()
        rois, \
        rpn_loss_cls, rpn_loss_box = fasterRCNN_step3(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
        loss_temp += loss.item()

        # backward
        optimizer_step3.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN_step3, 10.)
        optimizer_step3.step()

        if step % args.disp_interval == 0:
          end = time.time()
          if step > 0:
            loss_temp /= (args.disp_interval + 1)

          if args.mGPUs:
            loss_rpn_cls = rpn_loss_cls.mean().item()
            loss_rpn_box = rpn_loss_box.mean().item()
          else:
            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box = rpn_loss_box.item()

          print("[session %d][step %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                  % (args.session, step_of_train, epoch, step, iters_per_epoch, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (0, 0, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, 0, 0))
          if args.use_tfboard:
            info = {
              'step_of_train': step_of_train,
              'loss': loss_temp,
              'loss_rpn_cls': loss_rpn_cls,
              'loss_rpn_box': loss_rpn_box,
              'loss_rcnn_cls': 0,
              'loss_rcnn_box': 0
            }
            logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          loss_temp = 0
          start = time.time()
    
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_step{}_{}_{}.pth'.format(args.session, step_of_train,epoch, step))
      save_checkpoint({
        'session': args.session,
        'step_of_train':step_of_train,
        'epoch': epoch + 1,
        'model': fasterRCNN_step3.module.state_dict() if args.mGPUs else fasterRCNN_step3.state_dict(),
        'optimizer': optimizer_step3.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))
    step_of_train += 1    

  if step_of_train == 4:
    #--------------------------------------------------
    # step 4: train the Fast-RCNN using the propapsals generated by step-3 RPN
    #---------------------------------------------------
    # initilize the network here.
    if args.net == 'vgg16':
      fasterRCNN_step4 = vgg16_step4(imdb.classes, step2_model_path=os.path.join(output_dir,'faster_rcnn_1_step2_20_5010.pth'), \
                      step3_model_path=os.path.join(output_dir,'faster_rcnn_1_step3_20_5010.pth'), \
                      fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN_step4 = resnet_step4(imdb.classes, step2_model_path=os.path.join(output_dir,'faster_rcnn_1_step2_20_5010.pth'), \
                      step3_model_path=os.path.join(output_dir,'faster_rcnn_1_step3_20_5010.pth'), \
                      num_layers=101, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN_step4 = resnet_step4(imdb.classes, step2_model_path=os.path.join(output_dir,'faster_rcnn_1_step2_20_5010.pth'), \
                      step3_model_path=os.path.join(output_dir,'faster_rcnn_1_step3_20_5010.pth'), \
                      num_layers=50, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN_step4 = resnet_step4(imdb.classes, step2_model_path=os.path.join(output_dir,'faster_rcnn_1_step2_20_5010.pth'), \
                      step3_model_path=os.path.join(output_dir,'faster_rcnn_1_step3_20_5010.pth'), \
                      num_layers=152, fix_cnn_base=True, pretrained=True, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()
      
    params_step4 = []
    for key, value in dict(fasterRCNN_step4.named_parameters()).items():
      if value.requires_grad and 'detector.' in key:
        if 'bias' in key:
          params_step4 += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                  'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params_step4 += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
      lr = lr * 0.1
      optimizer_step4 = torch.optim.Adam(params_step4)
    elif args.optimizer == "sgd":
      optimizer_step4 = torch.optim.SGD(params_step4, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
      fasterRCNN_step4.cuda()

    if args.mGPUs:
      fasterRCNN_step4 = nn.DataParallel(fasterRCNN_step4)
    lr_start = lr

    for epoch in range(args.start_epoch, args.max_epochs[3] + 1):
      # setting to train mode
      fasterRCNN_step4.train()
      loss_temp = 0
      start = time.time()

      if epoch % (args.lr_decay_step + 1) == 0:
          adjust_learning_rate(optimizer_step4, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      data_iter = iter(dataloader)
      for step in range(iters_per_epoch):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        fasterRCNN_step4.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN_step4(im_data, im_info, gt_boxes, num_boxes)

        loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()

        # backward
        optimizer_step4.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN_step4, 10.)
        optimizer_step4.step()

        if step % args.disp_interval == 0:
          end = time.time()
          if step > 0:
            loss_temp /= (args.disp_interval + 1)

          if args.mGPUs:
            loss_rcnn_cls = RCNN_loss_cls.mean().item()
            loss_rcnn_box = RCNN_loss_bbox.mean().item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
          else:
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

          print("[session %d][step %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                  % (args.session, step_of_train, epoch, step, iters_per_epoch, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (0, 0, loss_rcnn_cls, loss_rcnn_box))
          if args.use_tfboard:
            info = {
              'step_of_train': step_of_train,
              'loss': loss_temp,
              'loss_rpn_cls': 0,
              'loss_rpn_box': 0,
              'loss_rcnn_cls': loss_rcnn_cls,
              'loss_rcnn_box': loss_rcnn_box
            }
            logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

          loss_temp = 0
          start = time.time()
    
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_step{}_{}_{}.pth'.format(args.session, step_of_train,epoch, step))
      save_checkpoint({
        'session': args.session,
        'step_of_train':step_of_train,
        'epoch': epoch + 1,
        'model': fasterRCNN_step4.module.state_dict() if args.mGPUs else fasterRCNN_step4.state_dict(),
        'optimizer': optimizer_step4.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))   
      
  if args.use_tfboard:
    logger.close()
