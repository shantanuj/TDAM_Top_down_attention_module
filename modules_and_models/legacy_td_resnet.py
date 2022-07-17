import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from .modules import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        
        ##Custom block arguments
        is_custom_block: bool = False,
        block_type: str = 'td_spotlight', #td_spotlight_concat_t
        time_steps: int = 3,
        m: int = 2,
        use_bnt = True,
        use_fca = False,
        
        ##TD block specific arguments
        rep_reduction_tuple: tuple = ('r',16),
        rep_pooling_techniques: List[str] = ['avp'],
        rep_combine_technique: str = 'sum',
        spotlight_gen_technique: str = 'top_attention_only', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
        operation_seq = 'channel->spatial',
        spatial_computation_method = 'att_scale', #att_scale, crop, sample, 
        channel_nonlin = 'sigmoid',
        spatial_searchlight_nonlin = None,
        dropout = 0,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.is_custom_block = is_custom_block
        self.use_bnt = use_bnt
        self.use_fca = use_fca 
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)]) #Only used in cases wherein fca is used 
        
        
        if(self.is_custom_block):
            self.block_type = block_type
            self.time_steps = time_steps
            self.m = m
            top_channels = planes
            
            if(self.block_type == 'cbam'):
                self.att_block = CBAM(planes, rep_reduction_tuple, use_fca=self.use_fca, fca_wh=c2wh[planes])     
            
            elif(self.block_type == 'se'):
                self.att_block = SE(planes, rep_reduction_tuple, use_fca=self.use_fca, fca_wh=c2wh[planes])
            elif(self.block_type == 'eca'):
                self.att_block = ECA(planes, k_size=3, use_fca=self.use_fca, fca_wh=c2wh[planes])
            
            elif(self.block_type.split('_')[0] == 'td'):
                if(self.m==0):
                    bottom_channels = planes
                elif(self.m==1):
                    bottom_channels = planes
                elif(self.m==2):
                    bottom_channels = inplanes
                elif(self.m==3):
                    print("Error. For BasicBlock m<=2")
                    sys.exit(0)
                if(self.use_bnt):
                    for t in range(1, self.time_steps):
                        if(self.m>=2):
                            setattr(self,"bn1_{}".format(t), norm_layer(planes))
                        if(self.m>=1):
                            setattr(self,"bn2_{}".format(t), norm_layer(planes))
                        #if(self.m==0):
                          #  setattr(self,"bn3_{}".format(t), norm_layer(planes))
                        
                    #if(self.m==0):
                      #  setattr(self,"bn3_{}".format(0), norm_layer(planes))

                
                
                    
                if(self.block_type.split('_')[1] == 'spotlight'):
                        self.RA_block = TD_Spotlight_Attention(bottom_channels = bottom_channels,
                                                               top_channels = top_channels, 
                                                               rep_reduction_tuple = rep_reduction_tuple,
                                                               rep_pooling_techniques = rep_pooling_techniques,
                                                               rep_combine_technique = rep_combine_technique,
                                                               spotlight_gen_technique = spotlight_gen_technique,
                                                               operation_seq = operation_seq,
                                                               spatial_computation_method = spatial_computation_method, 
                                                               channel_nonlin = channel_nonlin,
                                                               spatial_searchlight_nonlin = spatial_searchlight_nonlin,
                                                               dropout = dropout
                                                              )
                        
                        
                            
                else:
                    print("Error: TD block type {} not found".format(self.block_type))
                    sys.exit(0)
            else:
                print("Error: block type {} not found".format(self.block_type))
                sys.exit(0)
    
    def forward(self, x:Tensor) -> Tensor:
        if(not self.is_custom_block or self.block_type in ['None',None]):
            out = self._default_forward(x)
            
        elif(self.block_type in ['cbam','se','eca','fca']):
            out = self._att_feed_forward(x)
        
    
        elif(self.block_type.split('_')[0] == 'td' and self.block_type.split('_')[1] in ['spotlight']):
            
            if(self.m==0):
                out = self._static_recur_attention_forward_m0(x)
            elif(self.m == 1):
                out = self._static_recur_attention_forward_m1(x)
            elif(self.m == 2):
                out = self._static_recur_attention_forward_m2(x)
                    
        else:
            print("Error: Block type: {} does not exist".format(self.block_type))
            sys.exit(0)
            
        return out
    
    
    
    
    def _static_recur_attention_forward_m0(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        #prev_t_x = out
        s_t = None
        for t in range(self.time_steps):
            out, s_t = self.RA_block(out, out, s_t) 
            #if(self.use_bnt):
               # out = getattr(self,"bn3_{}".format(t))(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
    def _static_recur_attention_forward_m1(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        prev_t_x = out
        s_t = None
        
        for t in range(self.time_steps):
            out = self.conv2(prev_t_x)
            if(t==0):
                out = self.bn2(out)
            elif(self.use_bnt):
                out = getattr(self,"bn2_{}".format(t))(out)
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x, out, s_t)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out  
    
    def _static_recur_attention_forward_m2(self, x: Tensor) -> Tensor:
        identity = x
        prev_t_x = x 
        s_t = None
        for t in range(self.time_steps):
            out = self.conv1(prev_t_x)
            if(t==0):
                out = self.bn1(out)
            elif(self.use_bnt):
                out = getattr(self,"bn1_{}".format(t))(out)
            out = self.relu(out)

            out = self.conv2(out)
            if(t==0):
                out = self.bn2(out)
            elif(self.use_bnt):
                out = getattr(self,"bn2_{}".format(t))(out)
                
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x, out, s_t) 
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
    def _default_forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def _att_feed_forward(self, x: Tensor) -> Tensor:
        identity = x
       # print("I", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att_block(out)
        
        #print("A", x.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        
        ##Custom block arguments
        is_custom_block: bool = False,
        block_type: str = 'td_spotlight',
        time_steps: int = 3,
        m: int = 2,
        use_bnt = True,
        use_fca = False,
        
        ##TD block specific arguments
        rep_reduction_tuple: tuple = ('r',16),
        rep_pooling_techniques: List[str] = ['avp'],
        rep_combine_technique: str = 'sum',
        spotlight_gen_technique: str = 'top_attention_only', 
        operation_seq = 'channel->spatial',
        spatial_computation_method = 'att_scale', 
        channel_nonlin = 'sigmoid',
        spatial_searchlight_nonlin = None,
        dropout = 0,
        
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        #print(self.downsample)
        self.stride = stride
        
        self.is_custom_block = is_custom_block
        self.use_bnt = use_bnt
        
        self.use_fca = use_fca
        
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)]) #Only used in cases wherein fca is used 
        
        if(self.is_custom_block):
            self.block_type = block_type
            self.time_steps = time_steps
            self.m = m
            top_channels = planes
            
            if(self.block_type == 'cbam'):
                self.att_block = CBAM(planes * self.expansion, rep_reduction_tuple, use_fca=self.use_fca, fca_wh=c2wh[planes])     
            
            elif(self.block_type == 'se'):
                self.att_block = SE(planes * self.expansion, rep_reduction_tuple, use_fca=self.use_fca, fca_wh=c2wh[planes])
                
            elif(self.block_type == 'eca'):
                self.att_block = ECA(planes * self.expansion, k_size=3, use_fca=self.use_fca, fca_wh=c2wh[planes])
            
            elif(self.block_type.split('_')[0] == 'td'):
                top_channels = planes*self.expansion
                
                if(self.m==0):
                    bottom_channels = planes*self.expansion
                elif(self.m==1):
                    bottom_channels = width
                elif(self.m==2):
                    bottom_channels = width
                elif(self.m==3):
                    bottom_channels = inplanes
                    
                if(self.use_bnt):
                    for t in range(1, self.time_steps):
                        if(self.m>=3):
                            setattr(self,"bn1_{}".format(t), norm_layer(width))
                        if(self.m>=2):
                            setattr(self,"bn2_{}".format(t), norm_layer(width))
                        if(self.m>=1):
                            setattr(self,"bn3_{}".format(t), norm_layer(planes*self.expansion))
                        
                    
                    
                if(self.block_type.split('_')[1] == 'spotlight'):
                        self.RA_block = TD_Spotlight_Attention(bottom_channels = bottom_channels,
                                                               top_channels = top_channels, 
                                                               rep_reduction_tuple = rep_reduction_tuple,
                                                               rep_pooling_techniques = rep_pooling_techniques,
                                                               rep_combine_technique = rep_combine_technique,
                                                               spotlight_gen_technique = spotlight_gen_technique,
                                                               operation_seq = operation_seq,
                                                               spatial_computation_method = spatial_computation_method, 
                                                               channel_nonlin = channel_nonlin,
                                                               spatial_searchlight_nonlin = spatial_searchlight_nonlin,
                                                               dropout = dropout,
                                                               )
                        
                else:
                    print("Error: TD block type {} not found".format(self.block_type))
                    sys.exit(0)
            else:
                print("Error: block type {} not found".format(self.block_type))
                sys.exit(0)

    def forward(self, x:Tensor) -> Tensor:
        if(not self.is_custom_block or self.block_type in ['None',None]):
            out = self._default_forward(x)
            
        elif(self.block_type in ['cbam','se','eca','fca']):
            out = self._att_feed_forward(x)
        
        elif(self.block_type.split('_')[0] == 'td' and self.block_type.split('_')[1] in ['spotlight']):
            
            if(self.m==0):
                out = self._static_recur_attention_forward_m0(x)
            elif(self.m == 1):
                out = self._static_recur_attention_forward_m1(x)
            elif(self.m == 2):
                out = self._static_recur_attention_forward_m2(x)
            elif(self.m == 3):
                out = self._static_recur_attention_forward_m3(x)
                    
        else:
            print("Error: Block type: {} does not exist".format(self.block_type))
            sys.exit(0)
            
        return out
    
    
    
    def _static_recur_attention_forward_m0(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        s_t = None
        #prev_t_x = out
        for t in range(self.time_steps):
            out, s_t = self.RA_block(out, out, s_t) 
            #if(self.use_bnt):
              #  out = getattr(self,"bn4_{}".format(t))(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
    def _static_recur_attention_forward_m1(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
            
        prev_t_x = out
        s_t = None
        
        for t in range(self.time_steps):
            out = self.conv3(prev_t_x)
            if(t==0):
                out = self.bn3(out)
            elif(self.use_bnt):
                out = getattr(self,"bn3_{}".format(t))(out)
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x, out, s_t)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out  
    
    def _static_recur_attention_forward_m2(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        prev_t_x = out
        s_t = None
        
        for t in range(self.time_steps):
            out = self.conv2(prev_t_x)
            if(t==0):
                out = self.bn2(out)
            elif(self.use_bnt):
                out = getattr(self,"bn2_{}".format(t))(out)
            out = self.relu(out)

            out = self.conv3(out)
            if(t==0):
                out = self.bn3(out)
            elif(self.use_bnt):
                out = getattr(self,"bn3_{}".format(t))(out)
                
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x, out, s_t)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
    def _static_recur_attention_forward_m3(self, x: Tensor) -> Tensor:
        identity = x
        prev_t_x = x 
        s_t = None
        
        for t in range(self.time_steps):
            out = self.conv1(prev_t_x)
            if(t==0):
                out = self.bn1(out)
            elif(self.use_bnt):
                out = getattr(self,"bn1_{}".format(t))(out)
            out = self.relu(out)

            out = self.conv2(out)
            if(t==0):
                out = self.bn2(out)
            elif(self.use_bnt):
                out = getattr(self,"bn2_{}".format(t))(out)
            out = self.relu(out)
            
            out = self.conv3(out)
            if(t==0):
                out = self.bn3(out)
            elif(self.use_bnt):
                out = getattr(self,"bn3_{}".format(t))(out)
            #out = self.relu(out)
                
            if(t<self.time_steps-1):
                prev_t_x, s_t = self.RA_block(prev_t_x, out, s_t)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
   
    def _default_forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def _att_feed_forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.att_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
   

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        
        ##Custom block arguments
        custom_blocks_list: List[int] = [1,2,3,4],
        att_layer_wise_blocks_list: List[int] = [[0,1,2,3,4,5]*4],
        output_over_t = False,
        output_over_t_blocks = [0,1,2],
        
        ##Custom block arguments
        is_custom_block: bool = False,
        block_type: str = 'td_spotlight',
        time_steps: int = 3,
        m: int = 2,
        use_bnt = True,
        use_fca = False,
        
        ##TD block specific arguments
        rep_reduction_tuple: tuple = ('r',16),
        rep_pooling_techniques: List[str] = ['avp'],
        rep_combine_technique: str = 'sum',
        spotlight_gen_technique: str = 'top_attention_only', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
        operation_seq = 'channel->spatial',
        spatial_computation_method = 'att_scale', #att_scale, crop, sample, 
        channel_nonlin = 'sigmoid',
        spatial_searchlight_nonlin = None,
        dropout = 0,
        
        output_over_t_method = 'pred_over_t', 
        ###Other variables
        
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #####################Custom block variables#########################
        self.custom_blocks_list = custom_blocks_list
        #print(1,self.custom_blocks_list)
        self.att_layer_wise_blocks_list = att_layer_wise_blocks_list
        #print(2,self.block_layers_list)
        #self.block_type = block_type
        self.time_steps = time_steps
        self.output_over_t = output_over_t
        self.output_over_t_blocks = output_over_t_blocks
        ##Custom block arguments
        #self.is_custom_block = is_custom_block#: bool = False,
        self.block_type = block_type#: str = 'td_spotlight',
        #self.time_steps = time_steps#: int = 3,
        self.m =m #: int = 2,
        self.use_bnt = use_bnt# = True,
        self.use_fca = use_fca
        
        ##TD block specific arguments
        self.rep_reduction_tuple = rep_reduction_tuple#: tuple = ('r',16),
        self.rep_pooling_techniques = rep_pooling_techniques#: List[str] = ['avp'],
        self.rep_combine_technique = rep_combine_technique#: str = 'sum',
        self.spotlight_gen_technique = spotlight_gen_technique#: str = 'top_attention_only', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
        self.operation_seq  = operation_seq #= 'channel->spatial',
        self.spatial_computation_method = spatial_computation_method # = 'att_scale', #att_scale, crop, sample, 
        self.channel_nonlin = channel_nonlin #= 'sigmoid',
        self.spatial_searchlight_nonlin = spatial_searchlight_nonlin# = None,
        self.dropout = dropout
        
        self.output_over_t_method = output_over_t_method
        #####################################################
        
        
        
        
        self.layer1 = self._make_layer(block, 64, layers[0], is_custom_block = 1 in self.custom_blocks_list, att_blocks = self.att_layer_wise_blocks_list[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], is_custom_block = 2 in self.custom_blocks_list, att_blocks = self.att_layer_wise_blocks_list[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], is_custom_block = 3 in self.custom_blocks_list, att_blocks = self.att_layer_wise_blocks_list[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], is_custom_block = 4 in self.custom_blocks_list, att_blocks = self.att_layer_wise_blocks_list[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, is_custom_block: bool = True, att_blocks = []) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            #print(downsample)

        layers = []
        if(0 in att_blocks and is_custom_block):
            #Make custom block 
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer,
                                is_custom_block = True,
                                ##Custom block arguments
                                block_type = self.block_type,
                                time_steps = self.time_steps,
                                m = self.m,
                                use_bnt = self.use_bnt,
                                use_fca = self.use_fca,
                                ##TD block specific arguments
                                rep_reduction_tuple = self.rep_reduction_tuple,
                                rep_pooling_techniques = self.rep_pooling_techniques,
                                rep_combine_technique = self.rep_combine_technique,
                                spotlight_gen_technique = self.spotlight_gen_technique,
                                operation_seq = self.operation_seq,
                                spatial_computation_method = self.spatial_computation_method, #att_scale, crop, sample, 
                                channel_nonlin = self.channel_nonlin,
                                spatial_searchlight_nonlin = self.spatial_searchlight_nonlin,
                                dropout = self.dropout,
                                ))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, is_custom_block=False))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if(i in att_blocks and is_custom_block):
                #Make custom block
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation, 
                                    norm_layer=norm_layer,
                                    is_custom_block = True,
                                ##Custom block arguments
                                block_type = self.block_type,
                                time_steps = self.time_steps,
                                m = self.m,
                                use_bnt = self.use_bnt,
                                use_fca = self.use_fca,
                                ##TD block specific arguments
                                rep_reduction_tuple = self.rep_reduction_tuple,
                                rep_pooling_techniques = self.rep_pooling_techniques,
                                rep_combine_technique = self.rep_combine_technique,
                                spotlight_gen_technique = self.spotlight_gen_technique,
                                operation_seq = self.operation_seq,
                                spatial_computation_method = self.spatial_computation_method, #att_scale, crop, sample, 
                                channel_nonlin = self.channel_nonlin,
                                spatial_searchlight_nonlin = self.spatial_searchlight_nonlin,
                                dropout = self.dropout,
        
                                ))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, is_custom_block=False))

        return nn.Sequential(*layers)

    def get_features(self):
        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
    
    def _output_pred_over_t_forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        out_over_t = []
        for t in range(1, self.time_steps+1):
            for i in self.output_over_t_blocks:
                if(i>=len(self.layer4)):
                    break
                self.layer4[i].time_steps = t
            x_ = self.layer4(x)
            x_ = self.avgpool(x_)
            x_ = torch.flatten(x_,1)
            x_ = self.fc(x_)
            out_over_t.append(x_)

        #x_over_t = [torch.flatten(self.avgpool(x),1) for x in x_over_t]
        #x = torch.flatten(x, 1)
        #x_over_t = self.fc(x)

        return out_over_t
    

    def forward(self, x: Tensor) -> Tensor:
        #print("JHH:",self.output_over_t_method, self.output_over_t)
        if(not self.output_over_t):
            return self._forward_impl(x)
        else:
            if(self.output_over_t_method in ['pred_over_t']):
                return self._output_pred_over_t_forward_impl(x)
            else:
                print("Error: over t method {} not found".format(self.output_over_t_method))
                sys.exit(0)

    
        
def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    *args,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, *args, **kwargs)
    if pretrained:
        print("LOADING PRETRAINED WEIGHTS")
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
                                              #strict=False)
        model.load_state_dict(state_dict)
    return model



def load_custom_resnet(config, 
                       resnet_block_type = 'basic',
                       pretrained=False,
                       progress=True,
                       custom_blocks_list = [1,2,3,4],
                       att_layer_wise_blocks_list = [[0,1,2,3,4,5]*4], 
                       block_type: str = 'td_spotlight_concat_t',
                       time_steps: int = 3,
                       output_over_t = False,
                       output_over_t_blocks = [0,1,2], #currently we assume only for layer4
                       m: int = 2,
                       use_bnt = True,
                       use_fca = False,
                     ##TD block specific arguments
                       rep_reduction_tuple: tuple = ('r',16),
                       rep_pooling_techniques: List[str] = ['avp'],
                       rep_combine_technique: str = 'sum',
                       spotlight_gen_technique: str = 'top_attention_only', #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
                       operation_seq = 'channel->spatial',
                       spatial_computation_method = 'att_scale', #att_scale, crop, sample, 
                       channel_nonlin = 'sigmoid',
                       spatial_searchlight_nonlin = None,
                       dropout = 0.0,
        
                    
                       
                    ###Output over t method specific:
                       output_over_t_method = 'pred_over_t', #concat_over_t
                    ###
                       num_classes: int = 1000,
                       zero_init_residual: bool = False,
                       groups: int = 1,
                       width_per_group: int = 64,
                       replace_stride_with_dilation: Optional[List[bool]] = None,
                       norm_layer: Optional[Callable[..., nn.Module]] = None
                      ):
    
     return general_resnet(config, resnet_block_type, pretrained, progress,
                           custom_blocks_list = custom_blocks_list,
                           att_layer_wise_blocks_list = att_layer_wise_blocks_list,
                           ##Custom block arguments
                                block_type = block_type,
                                time_steps = time_steps,
                                m = m,
                                use_bnt = use_bnt,
                                use_fca = use_fca, 
                                output_over_t = output_over_t,
                                output_over_t_blocks = output_over_t_blocks,
                                
                               ##TD block specific arguments
                                rep_reduction_tuple = rep_reduction_tuple,
                                rep_pooling_techniques = rep_pooling_techniques,
                                rep_combine_technique = rep_combine_technique,
                                spotlight_gen_technique = spotlight_gen_technique,
                                operation_seq = operation_seq,
                                spatial_computation_method = spatial_computation_method, #att_scale, crop, sample, 
                                channel_nonlin = channel_nonlin,
                                spatial_searchlight_nonlin = spatial_searchlight_nonlin,
                                dropout = dropout,
        
                                
                           
                                output_over_t_method = output_over_t_method,
                                ##Basic settings
                                num_classes = num_classes
                          )#, number_of_conv_layers, scale_wise_channels, feature_extractor_out_channels
    
def general_resnet(config, resnet_block_type = 'basic', pretrained: bool = False, progress: bool = True, *args, **kwargs: Any) -> ResNet:
    r"""ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    scale_wise_channels = []
    feature_extractor_out_channels = 512
    if(resnet_block_type=='basic'):
        block = BasicBlock
        convlayers_per_scale_layer = 2
        scale_wise_channels = [64,128,256,512]
        
    elif(resnet_block_type in ['bneck','bottleneck']):
        block = Bottleneck
        convlayers_per_scale_layer = 3
        scale_wise_channels = [256,512,1024,2048]
        feature_extractor_out_channels = 2048
        
    number_of_conv_layers = 0
    for scale_layers in config:
        number_of_conv_layers += convlayers_per_scale_layer*scale_layers
        
    #print(block, config)
    return _resnet('resnet{}'.format(number_of_conv_layers+2), block, config, pretrained, progress, *args, **kwargs), number_of_conv_layers, scale_wise_channels, feature_extractor_out_channels

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
