# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
from ..utils import common_layers


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        polylines = polylines.clone()
        polylines_mask = polylines_mask.clone()
        
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature_cat = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid_mlp = self.mlps(polylines_feature_cat[polylines_mask])

        feature_buffers = polylines_feature_cat.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid_mlp.shape[-1])

        feature_buffers_copy = feature_buffers.clone()
        feature_buffers_copy[polylines_mask] = polylines_feature_valid_mlp

       
        # max-pooling
        # feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            #valid_mask = (polylines_mask > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers_copy[polylines_mask])  # (N, C)
            
            feature_buffers_out = feature_buffers_copy.new_zeros(batch_size ,num_polylines, num_points_each_polylines, feature_buffers_valid.shape[-1]) # add 11
            
            feature_buffers_out[polylines_mask] = feature_buffers_valid

            return feature_buffers_out
            
        return feature_buffers
