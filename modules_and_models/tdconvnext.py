#Adapted from official code for https://github.com/facebookresearch/ConvNeXt


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

#TDmodule
class TD_Spotlight_Attention(nn.Module):
    def __init__(self, 
                 bottom_channels, 
                 top_channels, 
                 rep_reduction_tuple = ('r',16),
                 rep_pooling_techniques = ['avp'],
                 rep_combine_technique = 'sum',
                 spotlight_gen_technique = 'joint_top_bottom_concat_attention', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
                 
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
        elif(self.spotlight_gen_technique in ['identity']):
            None
        elif(self.spotlight_gen_technique in ['simple_mapping']):
            self.Ug_conv = nn.Conv2d(top_channels, bottom_channels, 1, bias=False)
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
        #0. If identity recurrence, then return new bottom_x as top_x
        if(self.spotlight_gen_technique == 'identity'):
            bottom_x = top_x
            return bottom_x, None
        elif(self.spotlight_gen_technique == 'simple_mapping'):
            bottom_x = self.Ug_conv(top_x)
            return bottom_x, None
        
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
    
    
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                attn_layer=None, #set to existing attention blocks (se, eca, cbam, etc)
                 use_fca=False,
                 is_td_block=False,
                  ##TD block arguments
        #is_td_block = False,
        time_steps: int = 3,
        m: int = 2,
        use_bnt = False,
        
        ##TD block specific arguments
        rep_reduction_tuple = ('r',16),
        rep_pooling_techniques = ['avp'],
        rep_combine_technique = 'sum',
        spotlight_gen_technique = 'top_attention_only', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
        operation_seq = 'channel->spatial',
        channel_nonlin = 'sigmoid',
        spatial_searchlight_nonlin = None, 
                
                ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_td_block = is_td_block
        if(is_td_block):
            self.use_bnt = use_bnt
            self.time_steps = time_steps
            self.m = m
            top_channels = dim
            if(self.m==0):
                bottom_channels = dim
            elif(self.m==1):
                bottom_channels = int(4 * dim)
            elif(self.m==2):
                bottom_channels = dim
            elif(self.m==3):
                bottom_channels = dim
                    
            if(self.use_bnt):
             #   for t in range(1, self.time_steps):
                   # if(self.m>=3):
                    #    setattr(self,"bn1_{}".format(t), norm_layer(first_planes))
                    #if(self.m>=2):
                     #   setattr(self,"bn2_{}".format(t), norm_layer(width))
                    #if(self.m>=1):
                     #   setattr(self,"bn3_{}".format(t), norm_layer(outplanes))
                        
                print("BNT Not implemented yet")
                #self.gammas_over_t = nn.ModuleList([nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None for t in range(1, self.time_steps)])
                    
        
                        
            self.RA_block = TD_Spotlight_Attention(bottom_channels = bottom_channels,
                                                               top_channels = top_channels, 
                                                               rep_reduction_tuple = rep_reduction_tuple,
                                                               rep_pooling_techniques = rep_pooling_techniques,
                                                               rep_combine_technique = rep_combine_technique,
                                                               spotlight_gen_technique = spotlight_gen_technique,
                                                               operation_seq = operation_seq,
                                                               #spatial_computation_method = spatial_computation_method, 
                                                               channel_nonlin = channel_nonlin,
                                                               spatial_searchlight_nonlin = spatial_searchlight_nonlin,
                                                               #dropout = dropout
                                                              )

            
    def forward(self, x):
        if(not self.is_td_block):
            x = self._default_feed_forward(x)
    
        elif(self.is_td_block):
            if(self.m==0):
                x = self._static_recur_attention_forward_m0(x)
            elif(self.m == 1):
                x = self._static_recur_attention_forward_m1(x)
            elif(self.m == 2):
                x = self._static_recur_attention_forward_m2(x)
            elif(self.m == 3):
                x = self._static_recur_attention_forward_m3(x)
                    
        else:
            print("Error: Block type: {} does not exist".format(self.block_type))
            sys.exit(0)
            
        return x
    
    def _default_feed_forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
    def _static_recur_attention_forward_m1(self, x):
        input = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
            #x = self.drop1(x)
        prev_t_x = x
        s_t = None
        for t in range(self.time_steps):
            x = self.pwconv2(prev_t_x)
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2), s_t)
                prev_t_x = prev_t_x.permute(0, 2, 3, 1)
                
            #x = self.drop2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        x = input + self.drop_path(x)
        return x
    
    def _static_recur_attention_forward_m2(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        prev_t_x = x
        s_t = None
        for t in range(self.time_steps):
            x = self.pwconv1(prev_t_x)
            x = self.act(x)
            x = self.pwconv2(x)
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2), s_t)
                prev_t_x = prev_t_x.permute(0, 2, 3, 1)
                
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        x = input + self.drop_path(x)
        return x

class TD_ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 
                 ##Attention block arguments
        attn_layer = None, #'se','eca'
        use_fca=False, 
        #TD specific arguments
                 apply_td_at_layers_list= [2,3], #apply at last few layers
                 #apply_td_at_stages = [2,3],
                 apply_td_at_blocks_of_stages = [[0,1,2,3,4,5],
                                                 [0,1,2,3,4,5],
                                                 [0,1,2,3,4,5,6,7,8,9,10],
                                                 [0,1,2,3,4,5],
                                                ],  
                 output_over_t = False,
                 output_over_t_blocks = [0,1,2],
                 output_over_t_method = 'pred_over_t', 
                
                 ##TD block specific arguments
                 time_steps: int = 3,
                 m: int = 2,
                 use_bnt = False,
                 rep_reduction_tuple = ('r',16),
                 rep_pooling_techniques = ['avp'],
                 rep_combine_technique = 'sum',
                 spotlight_gen_technique = 'joint_top_bottom_concat_attention', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
                 operation_seq = 'channel->spatial',
                 channel_nonlin = 'sigmoid',
                 spatial_searchlight_nonlin = None,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        self.time_steps = time_steps
        self.output_over_t = output_over_t
        self.output_over_t_blocks = output_over_t_blocks
        self.output_over_t_method = output_over_t_method
        self.apply_td_at_layers_list = apply_td_at_layers_list
        self.apply_td_at_blocks_of_stages = apply_td_at_blocks_of_stages
        
        for i in range(4):
            if(i in apply_td_at_layers_list):
                #Make custom TD blocks
                blocks_list = []
                apply_td_at_blocks = apply_td_at_blocks_of_stages[i]
                for j in range(depths[i]):
                    #is_td_block = False
                    if(j in apply_td_at_blocks):
                        blocks_list.append(Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value,
                                                attn_layer = attn_layer,
                use_fca=use_fca,
                is_td_block=True,
                
                time_steps = time_steps,
                 m = m,
                 #use_bnt = use_bnt,
                 rep_reduction_tuple = rep_reduction_tuple,
                 rep_pooling_techniques= rep_pooling_techniques,
                 rep_combine_technique = rep_combine_technique,
                 spotlight_gen_technique = spotlight_gen_technique, #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
                 operation_seq = operation_seq,
                 channel_nonlin = channel_nonlin,
                 spatial_searchlight_nonlin = spatial_searchlight_nonlin,
                                                
                                                ))
                        
                    else:
                        blocks_list.append(Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value))
                        
                stage = nn.Sequential(*blocks_list)
            
                
            else:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if(m.bias is not None):
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_over_t(self, x): #only changes time steps for last stage 
        #x = self.stem(x)
        #x = self.stages[0:3](x)
        
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        out_over_t = []
        #print(x.shape)
        for t in range(1, self.time_steps+1):
            for i in self.output_over_t_blocks:
                if(i>=len(self.stages[3])):
                    break
                self.stages[3][i].time_steps = t
            x_ = self.downsample_layers[3](x)
            x_ = self.stages[3](x_)
            x_ = self.norm(x_.mean([-2, -1]))
            out_over_t.append(x_)
        return out_over_t
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
    
    def _output_pred_over_t_forward(self, x):
        outs_over_t = self.forward_features_over_t(x)
        outs_over_t = [self.head(x) for x in outs_over_t]
        return outs_over_t
    
    def _default_forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def forward(self, x):
        if(not self.output_over_t):
            return self._default_forward(x)
        else:
            if(self.output_over_t_method in ['pred_over_t']):
                return self._output_pred_over_t_forward(x)
            else:
                print("Error: over t method {} not found".format(self.output_over_t_method))
                sys.exit(0)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def td_convnext_tiny(pretrained=False, **kwargs):
    model = TD_ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def td_convnext_small(pretrained=False, **kwargs):
    model = TD_ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def td_convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = TD_ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def td_convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = TD_ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def td_convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = TD_ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
