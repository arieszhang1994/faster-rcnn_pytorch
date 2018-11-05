# --------------------------------------------------------
# Pytorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Aries Zhang, based on code from Jiasen Lu, Jianwei Yang
# --------------------------------------------------------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from model.faster_rcnn.resnetparts import resnet101,resnet50,resnet152
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _detector(nn.Module):
    '''detecotor'''
    def __init__(self, classes, class_agnostic, pretrained=False, base_model='vgg16'):
        super(_detector, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.pretrained = pretrained

        #processing of roi get from the rpn 
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # roi pool, roi align or roicrop
        # 1.0/16.0 is because size of input img is 16 times larger than feature map
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        self.base_model = base_model

        # top
        if base_model == 'vgg16':
            self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
            vgg = models.vgg16()
            if self.pretrained:
                state_dict = torch.load(self.model_path)
                vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
            self.RCNN_top = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
            # not using the last maxpool layer
            self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

            if self.class_agnostic:
                self.RCNN_bbox_pred = nn.Linear(4096, 4)
            else:
                self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
        elif base_model == 'resnet101' or base_model == 'resnet50' or base_model == 'resnet152':
            if base_model == 'resnet101':
                self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
                resnet = resnet101()
            elif base_model == 'resnet50':
                self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
                resnet = resnet50()
            elif base_model == 'resnet152':
                self.model_path = 'data/pretrained_model/resnet152_caffe.pth'
                resnet = resnet152()
            if self.pretrained:
                print("Detector: Loading pretrained weights from %s" %(self.model_path))
                state_dict = torch.load(self.model_path)
                resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
            self.RCNN_top = nn.Sequential(resnet.layer4)
            self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
            if self.class_agnostic:
                self.RCNN_bbox_pred = nn.Linear(2048, 4)
            else:
                self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)
            def set_bn_fix(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    for p in m.parameters(): p.requires_grad=False
            self.RCNN_top.apply(set_bn_fix)
        else:
            print("no support for other CNN model")
            exit()
    
    def forward(self, base_feat, rois, batch_size, gt_boxes, num_boxes):
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            # 1.0/16.0 is done in _affine_grid_gen
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        if self.base_model == 'vgg16':
            def _head_to_tail_vgg16(pool5):
                pool5_flat = pool5.view(pool5.size(0), -1)
                fc7 = self.RCNN_top(pool5_flat)
                return fc7
            pooled_feat = _head_to_tail_vgg16(pooled_feat)

        elif self.base_model == 'resnet101' or self.base_model == 'resnet50' or self.base_model == 'resnet152':
            def _head_to_tail_resnet(pool5):
                fc7 = self.RCNN_top(pool5).mean(3).mean(2)
                return fc7
            pooled_feat = _head_to_tail_resnet(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label

