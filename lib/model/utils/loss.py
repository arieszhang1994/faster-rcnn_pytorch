import torch.nn.functional as F
import torch
from model.utils.net_utils import _smooth_l1_loss

def detect_loss(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):
    # classification loss
    RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

    # bounding box regression L1 loss
    RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

    return RCNN_loss_cls, RCNN_loss_bbox

def ohem_detect_loss(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):

    def log_sum_exp(x):
        x_max = x.data.max()
        return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max

    num_hard = cfg.TRAIN.BATCH_SIZE * self.batch_size
    pos_idx = rois_label > 0
    num_pos = pos_idx.int().sum()

    # classification loss
    num_classes = cls_score.size(1)
    weight = cls_score.data.new(num_classes).fill_(1.)
    weight[0] = num_pos.data[0] / num_hard

    conf_p = cls_score.detach()
    conf_t = rois_label.detach()

    # rank on cross_entropy loss
    loss_c = log_sum_exp(conf_p) - conf_p.gather(1, conf_t.view(-1,1))
    loss_c[pos_idx] = 100. # include all positive samples
    _, topk_idx = torch.topk(loss_c.view(-1), num_hard)
    loss_cls = F.cross_entropy(cls_score[topk_idx], rois_label[topk_idx], weight=weight)

    # bounding box regression L1 loss
    pos_idx = pos_idx.unsqueeze(1).expand_as(bbox_pred)
    loc_p = bbox_pred[pos_idx].view(-1, 4)
    loc_t = rois_target[pos_idx].view(-1, 4)
    loss_box = F.smooth_l1_loss(loc_p, loc_t)

    return loss_cls, loss_box