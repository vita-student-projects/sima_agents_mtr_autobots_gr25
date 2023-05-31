import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os

from mtr.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtr.models.utils import polyline_encoder
from mtr.utils import common_utils
from mtr.ops.knn import knn_utils



from mtr.models.utils.train_helpers import nll_loss_multimodes
from datetime import datetime

def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
    obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
        angle=-center_heading
    ).view(num_center_objects, num_objects, num_timestamps, 2)

    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    # rotate direction of velocity
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

    return obj_trajs

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(self, d_k=64, dim_out = 6):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        self.dim_out = dim_out
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, dim_out))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        z_mean = pred_obs[:, :, 2]
        x_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 4]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 5]) * 3.14
        # for stability
        return  torch.stack([x_mean, y_mean, z_mean, x_sigma, y_sigma, rho], dim=2)


class MapEncoderPts(nn.Module):
    '''
    This class operates on the road lanes provided as a tensor with shape
    (B, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False  # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        '''
        :param roads: (B, S, P, k_attr+1)  where B is batch size, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, d_k) where T_obs is the observation horizon. THis tensor is obtained from
        AutoBot's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B*S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask

class PositionalEncoding(nn.Module):
    '''
    Standard positional encoding.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=20):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AutoBotsModel(nn.Module):
    def __init__(self, config, L_enc=1, L_dec=1, d_k=256, dropout=0.0, tx_hidden_size=384, num_heads=16):
        super().__init__()
        
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.model_cfg = config
        
        

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1, # self.model_cfg.NUM_INPUT_ATTR_AGENT + 1
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        print('self.model_cfg.NUM_INPUT_ATTR_AGENT',self.model_cfg.NUM_INPUT_ATTR_AGENT)
        print('self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT',self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT)
        print('self.model_cfg.NUM_LAYER_IN_MLP_AGENT',self.model_cfg.NUM_LAYER_IN_MLP_AGENT)
        print('self.model_cfg.D_MODEL',self.model_cfg.D_MODEL)
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )
        
        # ============================== AutoBot-Ego ENCODER ==============================
        print('CREATION of AUTOBOTS ENCODER')
        self._M = 0
        self.emb_size = self.model_cfg.D_K  # self.d_k
        
        self.L_dec= self.model_cfg.L_DEC
        self.L_enc = self.model_cfg.L_ENC
        self.dropout = self.model_cfg.DROPOUT
        self.tx_hidden_size = self.model_cfg.TX_HIDDEN_SIZE
        self.num_heads = self.model_cfg.NUM_HEADS
        self.num_modes = self.model_cfg.NUM_MODES
        self.time_seq = self.model_cfg.TIME_SEQ

        
        
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for layer in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)
        
        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.emb_size, dropout=0.0)
        
        # ============================== MAP ENCODER ==========================

        self.map_encoder = MapEncoderPts(d_k=self.emb_size, map_attr=3, dropout=self.dropout)
        self.map_attn_layers = nn.MultiheadAttention(self.emb_size, num_heads=self.num_heads, dropout=0.3)
        
        # ============================== AutoBot-Ego DECODER ==============================
       
        self.Q = nn.Parameter(torch.Tensor( self.time_seq, 1, self.num_modes, self.emb_size), requires_grad=True) 
        nn.init.xavier_uniform_(self.Q)
        
        self.map_attn_layers = nn.MultiheadAttention(self.emb_size, num_heads=self.num_heads, dropout=0.3)

        self.tx_decoder = []
        for dec_layer in range(self.L_dec):
            self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.emb_size, nhead=self.num_heads,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)
        
       
        
        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.emb_size, dim_out=4) # dimension to predict
        
        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(self.num_modes, 1, self.emb_size), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)
        
        self.prob_decoder = nn.MultiheadAttention(self.emb_size, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear( self.emb_size, 1))
        
        self.mode_map_attn = nn.MultiheadAttention(self.emb_size, num_heads=self.num_heads)
        
        
    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        AutoBots
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        # print('temporal_attn_fn')
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks[:, -1][temp_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)
    
    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        AutoBots
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''

        T_obs = agents_emb.size(0)

        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)

        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.contiguous().view(B*T_obs, num_agents))
        agents_soc_emb = agents_soc_emb.view(num_agents, B, T_obs, -1).permute(2, 1, 0, 3)
        
        return agents_soc_emb

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder



    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask
    
    def get_autobots_loss(self, batch_dict):
        '''
        Do the loss 
        out_dists: [64, 80, 5, 5]
        mode_probs: [5, 64]
        ego_out: need [5, 80, 3]
        '''
        # loss for 6 features
        # print(' LOSS ')
        input_dict = batch_dict['input_dict']
        # gt_world_small = input_dict['center_gt_trajs']
        # center_objects_world = input_dict['center_objects_world']

        # mode_probs = batch_dict['pred_scores']
        # pred_obs = batch_dict['pred_trajs']
        # num_center = mode_probs.shape[0]

        # gt_world = input_dict['center_gt_trajs_src'] # cx, cy, cz , heading, vx, vy


        # gt_relative = transform_trajs_to_center_coords(
        #     obj_trajs=gt_world[:,:,:],
        #     center_xyz=center_objects_world[:, 0:3],
        #     center_heading=center_objects_world[:, 6],
        #     heading_index=6, rot_vel_index=[7, 8]
        # )

        # center_gt_relative =  gt_relative[torch.arange(num_center), torch.arange(num_center)]
        # center_gt_relative = center_gt_relative[:, 11:,[0,1,2,6]]

        # ego_out = torch.cat((center_gt_relative, gt_world_small[:,:,2:4]), dim=-1).cuda() # x,y,z,yaw,vx,vy
        ego_out = input_dict['center_gt_trajs']

        entropy_weight = 1.0
        kl_weight = 1.0
        use_FDEADE_aux_loss = True
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(
                                                                        pred_obs,
                                                                        ego_out,
                                                                        mode_probs,
                                                                        entropy_weight=entropy_weight,
                                                                        kl_weight=kl_weight,
                                                                        use_FDEADE_aux_loss=use_FDEADE_aux_loss)
        tb_dict = {}
        disp_dict = {}
        
        loss_sum = (nll_loss + adefde_loss + kl_loss)
        
        return loss_sum, tb_dict, disp_dict


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dict = batch_dict['input_dict']
        obj_trajs = input_dict['obj_trajs'].to(device)  
        obj_trajs_mask= input_dict['obj_trajs_mask'].to(device) 
        map_polylines  = input_dict['map_polylines'].to(device)
        map_polylines_mask = input_dict['map_polylines_mask'].to(device)
        track_index_to_predict = input_dict['track_index_to_predict'].to(device)
        
        scenario_id = input_dict['scenario_id']


        env_masks_orig = torch.zeros((obj_trajs_mask.shape[0], obj_trajs_mask.shape[2])).cuda()

        env_masks_orig = obj_trajs_mask[torch.arange(len(track_index_to_predict)),track_index_to_predict]
        # for agent_i in range(track_index_to_predict.shape[0]):
        #     env_masks_orig[agent_i, :] = obj_trajs_mask[agent_i, track_index_to_predict[agent_i], :]


        env_masks = torch.zeros((env_masks_orig.shape))
        # env_masks = (1.0 - env_masks_orig) > 0
    
        env_masks = (~env_masks_orig).type(torch.BoolTensor)#.to(obj_trajs.device) # B, c , To
        env_masks = env_masks.unsqueeze(1)
        env_masks = env_masks.repeat(1, self.num_modes, 1)
        env_masks = env_masks.view(obj_trajs.shape[0] * self.num_modes, -1)
        env_masks = env_masks.cuda()

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda() 
        map_polylines_center = input_dict['map_polylines_center'].cuda() 

        

        # assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        
        self._M = num_objects - 1 
        num_polylines = map_polylines.shape[1]
        

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)
        
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]


        # apply self-attn
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0) # (num_center_objects, num_objects)
        

        
        # AutoBot's Encoder
        agents_emb = obj_polylines_feature.permute(2, 0, 1, 3)
        agents_emb_mask = ~obj_trajs_mask.permute(0, 2, 1)
        
        for i in range(self.L_enc):
            
            agents_emb = self.temporal_attn_fn(agents_emb, 
                                               agents_emb_mask, 
                                               layer=self.temporal_attn_layers[i])

            agents_emb = self.social_attn_fn(agents_emb, 
                                             agents_emb_mask, 
                                             layer=self.social_attn_layers[i])

       
        
        selected_agents_tensor = agents_emb[:, :, track_index_to_predict, :]
        
        ego_soctemp_emb = torch.zeros((agents_emb.size(0), agents_emb.size(1), self.emb_size)).cuda()

        
        for agent_i in range(track_index_to_predict.shape[0]):
            ego_soctemp_emb[:, agent_i, :] = selected_agents_tensor[:, agent_i, agent_i, :]

                
        # ego_soctemp_emb = selected_agents_tensor[torch.arange(track_index_to_predict.shape[0]), track_index_to_predict]
            

        num_modes = self.num_modes # self.c
        hidden_size = self.emb_size # self.d_k or hidden size
        time_seq = self.time_seq
        batch_size = ego_soctemp_emb.shape[1]

        # Repeat the tensors for the number of modes for efficient forward pass.
        # context = agents_emb.unsqueeze(2).repeat(1, 1, self.c, 1, 1)
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, num_modes, 1)
        
        context = context.view(-1, num_center_objects*num_modes, hidden_size)

        
        # Process map information
        orig_map_features, orig_road_segs_masks = self.map_encoder(map_polylines_feature, ego_soctemp_emb)

        map_features = orig_map_features.unsqueeze(2).repeat(1, 1, num_modes, 1)

        map_features = map_features.reshape(-1, batch_size*num_modes, hidden_size)
        road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, num_modes, 1).view(batch_size*num_modes, -1)

        # AutoBot-Ego Decoding
        out_seq = self.Q.repeat(1, num_center_objects, 1, 1).view(time_seq, num_center_objects*num_modes, -1)
        time_masks = self.generate_decoder_mask(seq_len=time_seq, device=obj_trajs.device)
        

        
        for d in range(self.L_dec):
            
            if d == 1:
                ego_dec_emb_map = self.map_attn_layers(query=out_seq, key=map_features, value=map_features,
                                        key_padding_mask=road_segs_masks)[0]
                
                out_seq = out_seq + ego_dec_emb_map

            out_seq = self.tx_decoder[d](tgt=out_seq, memory=context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)

        out_dists = self.output_model(out_seq).reshape(time_seq, num_center_objects, num_modes, -1).permute(2, 0, 1, 3)
        
        # Mode prediction  
        mode_params_emb = self.P.repeat(1, num_center_objects, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]
        mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                                 key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb
            
            

        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        batch_dict['pred_scores'] = mode_probs
        batch_dict['pred_trajs'] = out_dists
        
        return batch_dict