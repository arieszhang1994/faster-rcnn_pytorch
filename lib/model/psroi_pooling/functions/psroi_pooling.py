import torch
from torch.autograd import Function
from .._ext import psroi_pooling 


class PSRoIPoolingFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        ctx.pooled_width = int(pooled_width)
        ctx.pooled_height = int(pooled_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.group_size = int(group_size)
        ctx.output_dim = int(output_dim)
        ctx.mappingchannel = None
        ctx.rois = None
        ctx.feature_size = None

    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, ctx.output_dim, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.mappingchannel =  features.new(num_rois, ctx.output_dim, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        psroi_pooling.psroi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, ctx.group_size, ctx.output_dim, \
        features, rois, output, ctx.mappingchannel)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        psroi_pooling.psroi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, ctx.output_dim,  \
        grad_output, ctx.rois, grad_input, ctx.mappingchannel)
        return grad_input, None
