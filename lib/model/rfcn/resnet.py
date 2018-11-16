# --------------------------------------------------------
# Pytorch RFCN
# Licensed under The MIT License [see LICENSE for details]
# Written by Aries Zhang, based on code from Jiasen Lu, Jianwei Yang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.rfcn.rfcn import _RFCN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from model.rfcn.resnetparts import resnet101, resnet50, resnet152
import torch.utils.model_zoo as model_zoo
import pdb

class resnet(_RFCN):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        self.num_layers = num_layers
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _RFCN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        resnet = eval('resnet{}()'.format(self.num_layers))
        model_path = 'data/pretrained_model/resnet{}_caffe.pth'.format(self.num_layers)

        if self.pretrained:
            print("Loading pretrained weights from %s" % model_path)
            state_dict = torch.load(model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet.
        self.RCNN_base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        self.RCNN_conv_remain = resnet.layer4

        # Fix blocks
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
        self.RCNN_conv_remain.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            for fix_layer in range(6, 3 + cfg.RESNET.FIXED_BLOCKS, -1):
                self.RCNN_base[fix_layer].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_conv_remain.apply(set_bn_eval)

if __name__ == '__main__':
    import torch
    import numpy as np
    from torch.autograd import Variable

    input = torch.randn(3, 3, 600, 800)

    model = resnet101().cuda()
    input = Variable(input.cuda())
    out = model(input)
    print(out.size())
