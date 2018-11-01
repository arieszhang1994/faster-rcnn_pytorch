# --------------------------------------------------------
# Pytorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Aries Zhang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.faster_rcnn.detector import _detector

import math
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from model.faster_rcnn.resnetparts import resnet101, resnet50, resnet152

class resnet_step1(nn.Module):
    '''step1 for resnet'''
    def __init__(self, classes, num_layers=101, fix_cnn_base=False, \
                    class_agnostic=False, pretrained=False, base_model='resnet101'):
        super(resnet_step1, self).__init__()
        if num_layers == 101:
            self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
        elif num_layers == 50:
            self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        elif num_layers == 152:
            self.model_path = 'data/pretrained_model/resnet152_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.num_layer = num_layers
        self.fix_cnn_base = fix_cnn_base

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        #define base
        if self.num_layer == 101:
            resnet = resnet101()
        elif self.num_layer == 50:
            resnet = resnet50()
        elif self.num_layer == 152:
            resnet = resnet152()
        if self.pretrained:
            print("Step1: Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
        # Build resnet.
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, \
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        
        # Fix blocks
        if fix_cnn_base:
            for p in self.RCNN_base[0].parameters(): p.requires_grad=False
            for p in self.RCNN_base[1].parameters(): p.requires_grad=False
            
            assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
            if cfg.RESNET.FIXED_BLOCKS >= 3:
                for p in self.RCNN_base[6].parameters(): p.requires_grad=False
            if cfg.RESNET.FIXED_BLOCKS >= 2:
                for p in self.RCNN_base[5].parameters(): p.requires_grad=False
            if cfg.RESNET.FIXED_BLOCKS >= 1:
                for p in self.RCNN_base[4].parameters(): p.requires_grad=False
            
            def set_bn_fix(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    for p in m.parameters(): p.requires_grad=False
            self.RCNN_base.apply(set_bn_fix)

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)

        #init weight
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        return rois, rpn_loss_cls, rpn_loss_bbox

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode and self.fix_cnn_base:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            #self.detector.RCNN_top.apply(set_bn_eval)

class resnet_step2(nn.Module):
    '''step2 for resnet'''
    def __init__(self, classes, step1_model_path, num_layers=101, fix_cnn_base=False, \
                        class_agnostic=False, pretrained=False, base_model='resnet101'):
        super(resnet_step2, self).__init__()
        if num_layers == 101:
            self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
        elif num_layers == 50:
            self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        elif num_layers == 152:
            self.model_path = 'data/pretrained_model/resnet152_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.num_layer = num_layers
        self.fix_cnn_base = fix_cnn_base
        self.base_model = 'resnet'+str(num_layer)
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        #define base
        if self.num_layer == 101:
            resnet = resnet101()
        elif self.num_layer == 50:
            resnet = resnet50()
        elif self.num_layer == 152:
            resnet = resnet152()
        if self.pretrained:
            print("Step2: Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
        # Build resnet.
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, \
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

        # Fix blocks
        if fix_cnn_base:
            for p in self.RCNN_base[0].parameters(): p.requires_grad=False
            for p in self.RCNN_base[1].parameters(): p.requires_grad=False
            
            assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
            if cfg.RESNET.FIXED_BLOCKS >= 3:
                for p in self.RCNN_base[6].parameters(): p.requires_grad=False
            if cfg.RESNET.FIXED_BLOCKS >= 2:
                for p in self.RCNN_base[5].parameters(): p.requires_grad=False
            if cfg.RESNET.FIXED_BLOCKS >= 1:
                for p in self.RCNN_base[4].parameters(): p.requires_grad=False
            
            def set_bn_fix(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    for p in m.parameters(): p.requires_grad=False
            self.RCNN_base.apply(set_bn_fix)

        
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        #init weight of rpn
        state_dict_rpn = torch.load(step1_model_path)
        self.RCNN_rpn.load_state_dict({k.replace('RCNN_rpn.',''):v for k,v in state_dict_rpn['model'].items() if 'RCNN_rpn' in k})
        for key, value in dict(self.RCNN_rpn.named_parameters()).items():
            value.requires_grad = False

        # define detector
        self.detector = _detector(self.classes, self.class_agnostic,pretrained, base_model=self.base_model)
        #init weight of detector
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.detector.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.detector.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, alt=True)

        if not self.training:
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # detector part
        rois, cls_prob, bbox_pred, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.detector(base_feat, rois, batch_size,gt_boxes, num_boxes)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode and self.fix_cnn_base:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.detector.RCNN_top.apply(set_bn_eval)

class resnet_step3(nn.Module):
    '''step3 for resnet'''
    def __init__(self, classes, step2_model_path, num_layers=101, fix_cnn_base=False, \
                class_agnostic=False, pretrained=False, base_model='resnet101'):
        super(resnet_step3, self).__init__()
        self.model_path = step2_model_path
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.num_layer = num_layers
        self.fix_cnn_base = fix_cnn_base
        self.base_model = 'resnet'+str(num_layer)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        if self.num_layer == 101:
            resnet = resnet101()
        elif self.num_layer == 50:
            resnet = resnet50()
        elif self.num_layer == 152:
            resnet = resnet152()

        print("Step3: Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)

        # Build resnet.
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,   
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        self.RCNN_base.load_state_dict({k.replace('RCNN_base.',''):v for k,v in state_dict['model'].items() if 'RCNN_base' in k})
        for key, value in dict(self.RCNN_base.named_parameters()).items():
            value.requires_grad = False

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_rpn.load_state_dict({k.replace('RCNN_rpn.',''):v for k,v in state_dict['model'].items() if 'RCNN_rpn' in k})

    
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        return rois, rpn_loss_cls, rpn_loss_bbox

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode and self.fix_cnn_base:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            #self.detector.RCNN_top.apply(set_bn_eval)

class resnet_step4(nn.Module):
    '''step4 for resnet'''
    def __init__(self, classes, step2_model_path, step3_model_path, num_layers=101, fix_cnn_base=False, \
                class_agnostic=False, pretrained=False, base_model='resnet101'):
        super(resnet_step4, self).__init__()
        self.step2_model_path = step2_model_path
        self.step3_model_path = step3_model_path
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.num_layer = num_layers
        self.fix_cnn_base = fix_cnn_base
        self.base_model = 'resnet'+str(num_layer)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        #define base
        if self.num_layer == 101:
            resnet = resnet101()
        elif self.num_layer == 50:
            resnet = resnet50()
        elif self.num_layer == 152:
            resnet = resnet152()

        print("Step4: Loading pretrained weights from %s and %s" %(step2_model_path,step3_model_path))
        state_dict_step2 = torch.load(self.step2_model_path)
        state_dict_step3 = torch.load(self.step3_model_path)

        # not using the last maxpool layer
         # Build resnet.
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,   
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

        self.RCNN_base.load_state_dict({k.replace('RCNN_base.',''):v for k,v in state_dict_step3['model'].items() if 'RCNN_base' in k})
        for key, value in dict(self.RCNN_base.named_parameters()).items():
            value.requires_grad = False
        
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        #init weight of rpn
        self.RCNN_rpn.load_state_dict({k.replace('RCNN_rpn.',''):v for k,v in state_dict_step3['model'].items() if 'RCNN_rpn' in k})
        for key, value in dict(self.RCNN_rpn.named_parameters()).items():
            value.requires_grad = False
        
        # define detector
        self.detector = _detector(self.classes, self.class_agnostic,pretrained, base_model=base_model)
        self.detector.load_state_dict({k.replace('detector.',''):v for k,v in state_dict_step2['model'].items() if 'detector' in k})

    
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, alt=True)

        if not self.training:
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # detector part
        rois, cls_prob, bbox_pred, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.detector(base_feat, rois, batch_size,gt_boxes, num_boxes)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode and self.fix_cnn_base:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.detector.RCNN_top.apply(set_bn_eval)
