import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import sys
import math

#TDmodule
class TD_Spotlight_Attention(nn.Module):
    def __init__(self, 
                 bottom_channels, 
                 top_channels, 
                 rep_reduction_tuple = ('r',16),
                 rep_pooling_techniques = ['avp'],
                 rep_combine_technique = 'sum',
                 spotlight_gen_technique = 'top_attention_only', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
                 
                 ##Optional for neuro analysis exps
                 operation_seq = 'channel->spatial', 
                 spatial_computation_method = 'att_scale', #att_scale, crop, sample, 
                 channel_nonlin = 'sigmoid',
                 spatial_searchlight_nonlin = None,
                 dropout = 0,
                 
                ):
        super(TD_Spotlight_Attention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax2d()
        
        
        self.bottom_channels = bottom_channels
        self.top_channels = top_channels
        self.rep_reduction_tuple = rep_reduction_tuple
        self.rep_pooling_techniques = rep_pooling_techniques
        self.rep_combine_technique = rep_combine_technique 
        self.spotlight_gen_technique = spotlight_gen_technique
        self.operation_seq = operation_seq
        self.spatial_computation_method = spatial_computation_method
        self.channel_nonlin = channel_nonlin
        self.spatial_searchlight_nonlin = spatial_searchlight_nonlin
        self.dropout = dropout
        
        #0. get reduced channels
        if(self.rep_reduction_tuple[0] == 'r'):
            rd_channels = top_channels//self.rep_reduction_tuple[1]
            rd_channels_bottom = bottom_channels//self.rep_reduction_tuple[1] #incase required for recurrent models
        else:
            rd_channels = self.rep_reduction_tuple[1]
            rd_channels_bottom = self.rep_reduction_tuple[1]
                
        #1. Create variables based on rep_pooling_techniques and spotlight_gen_technique method 
        
        
        
        #Parameter 1 for gen spotlight: Ug that maps top (or joint) spatial map to searchlight
        self.Ug_mlp = None
        if(self.spotlight_gen_technique in ['top_attention_only']):
            self.Ug_mlp = nn.Sequential(
                    nn.Conv2d(top_channels, rd_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rd_channels, bottom_channels, 1, bias=False)
                    )
        
        elif(self.spotlight_gen_technique in ['joint_top_bottom_concat_attention']):
           # self.Wg_top_mlp 
            self.Ug_top_mlp = nn.Sequential(
                    nn.Conv2d(top_channels, rd_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                )
                
            self.Ug_bottom_mlp = nn.Sequential(
                    nn.Conv2d(bottom_channels, rd_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                )
                
            self.Ug_mlp = nn.Sequential(
                    nn.Conv2d(rd_channels*2, bottom_channels, 1, bias=False)
                )
        
        elif(self.spotlight_gen_technique in ['joint_top_bottom_bilinear_attention']):
            self.Ug_top_mlp = nn.Sequential(
                    nn.Conv2d(top_channels, rd_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                )
                
            self.Ug_bottom_mlp = nn.Sequential(
                    nn.Conv2d(bottom_channels, rd_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                )
                
            self.Ug_mlp = nn.Sequential(
                    nn.Conv2d(rd_channels, bottom_channels, 1, bias=False)
                )
        else:
            print("Error: spotlight gen technique: {} not found".format(self.spotlight_gen_technique))
            sys.exit(0)
            
        
            
        #Parameter  for dropout 
        if(self.dropout>0):
            self.dropout_layer = nn.Dropout2d(self.dropout)
        else:
            self.dropout_layer = nn.Identity()
        
        
            
    def generate_searchlight(self, top_x, bottom_x, prev_searchlight=None):
        ###################1. Perform mapping of Xt -> Ht####################
        top_reps = []
        bottom_reps = []
        
        if('mp' in self.rep_pooling_techniques):
            top_reps.append(self.max_pool(top_x))
        if('avp' in self.rep_pooling_techniques):
            top_reps.append(self.avg_pool(top_x))
        
        if(self.spotlight_gen_technique in ['joint_top_bottom_concat_attention','joint_top_bottom_bilinear_attention']):
            if('mp' in self.rep_pooling_techniques):
                bottom_reps.append(self.max_pool(bottom_x))
            if('avp' in self.rep_pooling_techniques):
                bottom_reps.append(self.avg_pool(bottom_x))
        
            
        if(self.spotlight_gen_technique == 'top_attention_only'):
            h_ts = [self.Ug_mlp(top_rep) for top_rep in top_reps]
            
            
        elif(self.spotlight_gen_technique == 'joint_top_bottom_concat_attention'):
            h_ts = []
            for i, bt_rep in enumerate(bottom_reps):
                top_rep = top_reps[i]
                h_ts.append(self.Ug_mlp(torch.cat((self.Ug_top_mlp(top_rep),self.Ug_bottom_mlp(bt_rep)),1)))
        
        elif(self.spotlight_gen_technique == 'joint_top_bottom_bilinear_attention'):
            h_ts = []
            for i, bt_rep in enumerate(bottom_reps):
                top_rep = top_reps[i]
                h_ts.append(self.Ug_mlp(self.Ug_top_mlp(top_rep)+self.Ug_bottom_mlp(bt_rep)))
            
        else:
            print("Basic input to h_t map mechanism: {} not found".format(self.channel_mech))
            sys.exit(0)     
            
        if(self.rep_combine_technique == 'sum'):
            h_t = torch.stack(h_ts, dim=0).sum(dim=0)
        elif(self.rep_combine_technique == 'avg'):
            h_t = torch.stack(h_ts, dim=0).mean(dim=0)
            
        s_t = h_t
        if(self.dropout>0):
            s_t = self.dropout_layer(s_t) 
            
        return s_t
            
    
    def compute_channel_modified(self, bottom_x, searchlight):
        if(self.channel_nonlin in ['sigmoid']):
            searchlight = self.sigmoid(searchlight)
        elif(self.channel_nonlin in ['softmax']):
            searchlight = self.softmax(searchlight)
        return searchlight * bottom_x
        
    def compute_spatial_modified(self, bottom_x, searchlight):
        if(self.spatial_searchlight_nonlin in ['sigmoid']):
            searchlight = self.sigmoid(searchlight)
        elif(self.spatial_searchlight_nonlin in ['softmax']):
            searchlight = self.softmax(searchlight)
           
       
        elif(self.spatial_computation_method in ['att_scale']):
            #print('using alt method')
            spatial_scale_map = torch.sum(searchlight * bottom_x, dim=[1])#self.compute_batch_conv2d(bottom_x, searchlight.view(searchlight.shape[0], 1, searchlight.shape[1],1,1)).squeeze(1)
            spatial_scale_map = self.sigmoid(spatial_scale_map)
            spatial_scale_map = torch.unsqueeze(spatial_scale_map,1)
            bottom_x = spatial_scale_map * bottom_x
            
        elif(self.spatial_computation_method in ['att_scale_alt_impl']):
            spatial_scale_map = self.compute_batch_conv2d(bottom_x, searchlight.view(searchlight.shape[0], 1, searchlight.shape[1],1,1)).squeeze(1)
            spatial_scale_map = self.sigmoid(spatial_scale_map)
            spatial_scale_map = torch.unsqueeze(spatial_scale_map,1)
            bottom_x = spatial_scale_map * bottom_x
        
            
        else:
            print("Error: Spatial computation method {} not implemented or found".format(self.spatial_computation_method))
        
        
        return bottom_x
    
    def compute_batch_conv2d(self, inp, weight):
    #inp of shape batch_size*in_channels*res1*res2
    #weight of shape batch_size*out_channels*in_channels*f1*f2
    #out of shape batch_size*out_channels*res1'*res2' (res' indicates dependent on padding)
        x = inp.view(inp.shape[0],1,inp.shape[1],inp.shape[2],inp.shape[3])
        b_i, b_j, c, h, w = x.shape
        b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape
        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        weight = weight.view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)
        out = F.conv2d(out, weight=weight, bias=None, stride=1, dilation=1, padding=0, groups=b_i)
        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])
        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])
        out = out.permute([1, 0, 2, 3, 4])
        out = torch.squeeze(out,dim=1)
        return out 
    
    
    
    def static_forward(self, bottom_x, top_x, prev_searchlight):
        #1. Generate searchlight 
        s_t = self.generate_searchlight(top_x, bottom_x, prev_searchlight)
        #print(s_t.shape, bottom_x.shape)
        #2. Do operations with searchlight based on sequence
        if(self.operation_seq == 'channel->spatial'):
            bottom_x = self.compute_channel_modified(bottom_x, s_t)
            bottom_x = self.compute_spatial_modified(bottom_x, s_t)
        elif(self.operation_seq == 'spatial->channel'):
            bottom_x = self.compute_spatial_modified(bottom_x, s_t)
            bottom_x = self.compute_channel_modified(bottom_x, s_t)
        elif(self.operation_seq == 'spatial'):
            bottom_x = self.compute_spatial_modified(bottom_x, s_t)  
        elif(self.operation_seq == 'channel'):
            bottom_x = self.compute_channel_modified(bottom_x, s_t)  
        else:
            print("Error; Operation sequence/computation mechanism: {} not found:".format(self.operation_seq))
            sys.exit(0)
        
        #3. Return bottom_x and new searchlight
        return bottom_x, s_t#, None
    
     
    def forward(self, bottom_x, top_x, prev_searchlight=None, prev_stop_flags=None, prev_stop_vals=None, prev_stop_times=None):
        #print(prev_searchlight)
        if(prev_searchlight is None): #create searchlight
            prev_searchlight = torch.zeros(bottom_x.shape[0], bottom_x.shape[1],1,1, device=bottom_x.device) #batch*num_channels
        
        return self.static_forward(bottom_x, top_x, prev_searchlight)
            
        



            
        
        
        
        
        

            
        
        
        
        
        

            
        
        
        
        
        
