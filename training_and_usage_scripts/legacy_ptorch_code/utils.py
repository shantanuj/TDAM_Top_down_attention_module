from collections import defaultdict, deque, OrderedDict
import copy
import datetime
import hashlib
import time
import torch
import torch.distributed as dist
import pickle
import errno
#import torch.nn.functional as F
import os
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pycocotools import mask as maskUtils
from torch import topk
import cv2
import numpy as np
import pickle
from collections import OrderedDict
import xmltodict
import os
import torch


with open('./imnet_id_to_name.pkl','rb') as handle:
    imnet_id_to_name = pickle.load(handle)
    
#with open('./imnet_ids_sim_np.pkl','rb') as handle:
 #   imnet_sim = pickle.load(handle)
    
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

    
def get_sorted_max_over_t(outs_over_t):
    """
    Given predictions over time (list of len=Time and each tensor of dim = Batch x Classes), this sorts predictions by putting the least confident as the first row and most confident as last row
    Note: Returns as a list of tensors 
    """
    outs_over_t = torch.stack(outs_over_t)
    outs_over_t = outs_over_t.transpose(1,0)
    for i in range(outs_over_t.shape[0]):
        outs_over_t[i] = torch.stack(sorted(outs_over_t[i], key=lambda x:x.max(), reverse=False))
    outs_over_t = outs_over_t.transpose(1,0)
    outs_over_t_list = []
    for i in range(outs_over_t.shape[0]):
        outs_over_t_list.append(outs_over_t[i])
    return outs_over_t_list



def accuracy_over_t(outputs, target, topk=(1,), wordnet_class_threshold = 0.5, sort_by_confidence=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        correct_t = []
        preds_over_t = []
        if(sort_by_confidence):
            #print("sorting")
            for output in outputs:
                output = torch.nn.Softmax(dim=-1)(output)
            outputs = get_sorted_max_over_t(outputs) #Sort to get last time step being equivalent to most confident predictions
        #list_of_i_early_correct = []
        for output in outputs[::-1]:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            preds_over_t.append(pred)
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_t.append(correct)
        
        correct_max = torch.max(torch.stack(correct_t), dim=0)#[0]
        correct = correct_max[0]
        correct_ids = correct_max[1]
        
        #print(correct_ids.shape)
        cases_where_earlier_time_steps = list(torch.where(correct_ids[0]>0)) #the 2nd 0 indicates the last time step pred
        
        threshold_earlier_ts = []
        for i_early_t in cases_where_earlier_time_steps[0]:
            correct_t_step = correct_ids[0][i_early_t]
            pred_last = preds_over_t[0][0][i_early_t]
            pred_at_t = preds_over_t[correct_t_step][0][i_early_t]
            if(imnet_sim[pred_last][pred_at_t]>=wordnet_class_threshold): #this means that it might be a potential overlap in class (same object diff. prediction), hence we disregard if a lesser confident output is correct here
                correct[0][i_early_t] = False
                correct[1][i_early_t] = True
                #print('hi')
            else:
                threshold_earlier_ts.append(i_early_t)
        
        res = []
        
        for k in range(maxk): #Fix top5 such that if True in earlier k then false in other ks
            for i in range(k):
                correct[k][correct[i]] = False
            
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        
        return res, cases_where_earlier_time_steps, threshold_earlier_ts
    
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key='model', strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(pretrained=False)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(pretrained=False, quantize=False)
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side-effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc)
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path
#{"mode":"full","isActive":false}

from pycocotools import mask as maskUtils
def get_iou(bb1, bb2):
    #Coco expects format to be: xmin,ymin,width,height
    return maskUtils.iou([bb1],[bb2],[False])[0][0]

def getCAM_single(feature_conv, weight_fc, class_idx, normalize=True):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    if(normalize):
        cam = cam - np.min(cam)
        cam= cam / np.max(cam)
    cam_img = cam
    return [cam_img]

def getCAM_batch(feature_convs, weight_fcs, class_ids, normalize=True):
    num_b, nc, h, w = feature_convs.shape
    cam_imgs = []
    #print(h,w)
    for i in range(num_b):
        #print(weight_fc[class_ids].shape)
        #print(feature_conv[i].shape)
        cam = weight_fcs[class_ids[i]].dot(feature_convs[i].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        if(normalize):
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
        cam_img = cam
        cam_imgs.append(cam_img)
    return cam_imgs


#Code adapted from: https://github.com/zalkikar/BBOX_GradCAM/blob/85feb617d40e8261301abe35aa3cc36a4b89340a/BBOXES_from_GRADCAM.py#L70
def heatmap_smoothing(og_img, heatmap, resize_list=(224,224)):
        #og_img
        heatmap = cv2.resize(heatmap, (resize_list[0],resize_list[1])) # Resizing
        og_img = cv2.resize(og_img, (resize_list[0],resize_list[1])) # Resizing
        '''
        The minimum pixel value will be mapped to the minimum output value (alpha - 0)
        The maximum pixel value will be mapped to the maximum output value (beta - 155)
        Linear scaling is applied to everything in between.
        These values were chosen with trial and error using COLORMAP_JET to deliver the best pixel saturation for forming contours.
        '''
        heatmapshow = cv2.normalize(heatmap, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        
        return og_img, heatmapshow
    
def form_bboxes(smooth_heatmap, bbox_scale_list = (1,1,1,1), threshold_ratio=0.5):
        grey_img = cv2.cvtColor(smooth_heatmap, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey_img,int(threshold_ratio*255),255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        max_cnt = 0
        min_cnt = 200 
        x = 0
        y = 0 
        w = 0
        h = 0
        poly_coords = None
        for item in range(len(contours)):
            cnt = contours[item]
            #print(len(cnt))
            if len(cnt)>15: #should be atleast 20cm?
                if(len(cnt)>max_cnt):
                    max_cnt = len(cnt)
                    #print("max cnt now: ", max_cnt)
                #if(len(cnt)<min_cnt):
                 #   min_cnt = len(cnt)
                  #  print("min cnt now: ", min_cnt)
                #print(len(cnt))
                    x,y,w,h = cv2.boundingRect(cnt) # x, y is the top left corner, and w, h are the width and height respectively
                    poly_coords = [cnt] # polygon coordinates are based on contours            
                    x = int(x*bbox_scale_list[0]) # rescaling the boundary box based on user input
                    y = int(y*bbox_scale_list[1])
                    w = int(w*bbox_scale_list[2])
                    h = int(h*bbox_scale_list[3])
    
            else: 
                None
                #print("contour error (too small) for thresh ratio:{}".format(threshold_ratio))
                
        return [x,y,w,h], poly_coords, grey_img, contours
                
def show_bboxrectangle(og_img, bbox_coords):
    return cv2.rectangle(og_img,
                      (bbox_coords[0],bbox_coords[1]),
                      (bbox_coords[0]+bbox_coords[2],bbox_coords[1]+bbox_coords[3]),
                      (0,0,0),3)


def heatmap_smoothing_or(heatmap, resize_list=(224,224)):
        #og_img
        heatmap = cv2.resize(heatmap, (resize_list[0],resize_list[1])) # Resizing
        #og_img = cv2.resize(og_img, (resize_list[0],resize_list[1])) # Resizing
        '''
        The minimum pixel value will be mapped to the minimum output value (alpha - 0)
        The maximum pixel value will be mapped to the maximum output value (beta - 155)
        Linear scaling is applied to everything in between.
        These values were chosen with trial and error using COLORMAP_JET to deliver the best pixel saturation for forming contours.
        '''
        heatmapshow = cv2.normalize(heatmap, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        
        return heatmapshow
    
class SaveFeatures():
    features=None
    def __init__(self, m): 
        #print(m)
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = ((output.cpu()).data).numpy()
    def remove(self): 
        self.hook.remove()
        
def get_CAM_map_with_prediction(model, images, mod_t_for_layers=['layer4'], target_layer='layer4', is_recurrent=True, t=3, change_t_blocks=[0,1,2,3,4], normalize=False, kval=1, return_pred_prob=False):
    #print('hi')
    #print(model.module._modules)
    final_layer = model.module._modules.get(target_layer)
    #print("HEY", final_layer)
    #print(model, final_layer)
    if(is_recurrent):
        #model.time_steps = t
        for mod_layer in mod_t_for_layers:
            #print("Recurrent t for each block of layer {}".format(mod_layer))
            layer = model.module._modules.get(mod_layer)
            for i, mod_bl in enumerate(layer): 
                if(i in change_t_blocks):
                    mod_bl.time_steps = t
                else:
                    mod_bl.time_steps = model.module.time_steps
                #print(model.layer4[i].time_steps)
    activated_features = SaveFeatures(final_layer)
    prediction = model(images)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    weight_softmax_params = list(model.module._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    class_idx = topk(pred_probabilities,kval)[1][:,kval-1].view(pred_probabilities.shape[0],1).int()
    pred_probs = topk(pred_probabilities,kval)[0][:,kval-1]
    #print(class_idx.shape,topk(pred_probabilities,kval)[0].shape)
    #kval=3
    #torch.topk(x, kval)[1][:,kval-1].view(x.shape[0],1)
    #print(class_idx)
    overlays = getCAM_batch(activated_features.features, weight_softmax, class_idx, normalize=normalize )
    if(return_pred_prob):
        return overlays, class_idx, pred_probs, prediction
    else:
        return overlays, class_idx
    
def filter_pred_with_iou(model, images, time_steps, image_res=224):
    #we output a filtered list of outputs 
    #time_step_outs = []
    #existing_bboxes = []
    time_step_img_dict = {} #keyed by outs (list of time step outputs) and bboxes 
    
    for current_time_step in range(1, time_steps+1):
        or_model_l4, pred_or, pred_prob, prediction_all = get_CAM_map_with_prediction(model, images, is_recurrent=True, mod_t_for_layers=['layer4'], target_layer='layer4',
                                                      normalize=False, kval=1, 
                                                      t=current_time_step, return_pred_prob=True)
        #print(len(or_model_l4), or_model_l4[0].shape)
        #img_bboxes = []
        #time_step_predictions.append(pred_or)
        for i_img, or_img in enumerate(images):
            #if(i_img<i_img+1):
             #   existing_bboxes.append()
            #1. CAM map smoothing
            #print(i_img, len(images))
            heatmap = heatmap_smoothing_or(or_model_l4[i_img], resize_list = (image_res, image_res))
            prediction = int(pred_or[i_img][0])
            new_bbox_coords, poly_coords, grey_img, contours = form_bboxes(heatmap, threshold_ratio=0.15)
            #print(new_bbox_coords)
            new_bboxes = []
            #if(len(exis))
            if(i_img not in time_step_img_dict):
                time_step_img_dict[i_img] = {}
                time_step_img_dict[i_img]['out'] = [prediction_all[i_img]]
                time_step_img_dict[i_img]['bbox'] = [new_bbox_coords]
                time_step_img_dict[i_img]['prob'] = [pred_prob[i_img]]
            else:
                #add_new_item=True
                for j, existing_bbox_out in enumerate(time_step_img_dict[i_img]['bbox']):
                    #bbox = existing_bbox_out[0]
                    #prob = existing_bbox_out[1]
                    if(get_iou(existing_bbox_out, new_bbox_coords)>0.5):
                        #add_new_item = False
                        if(pred_prob[i_img]>time_step_img_dict[i_img]['prob'][j]): #Either replace or duplicate previous matching prediction based on probability score
                            #Replace existing
                            time_step_img_dict[i_img]['prob'][j] = pred_prob[i_img]#[i_img]
                            time_step_img_dict[i_img]['out'][j] = prediction_all[i_img]
                            time_step_img_dict[i_img]['bbox'][j] = new_bbox_coords#[i_img]
                        else:
                            pred_prob[i_img] = time_step_img_dict[i_img]['prob'][j]
                            prediction_all[i_img] = time_step_img_dict[i_img]['out'][j]
                            new_bbox_coords = time_step_img_dict[i_img]['bbox'][j]
                
                time_step_img_dict[i_img]['prob'].append(pred_prob[i_img])
                time_step_img_dict[i_img]['out'].append(prediction_all[i_img])
                time_step_img_dict[i_img]['bbox'].append(new_bbox_coords)
                #else:
                 #   existing_bboxes[i_img].append((new_bbox_coords, pred_prob[i_img]))
           
        #img_bboxes.append(bbox_coords)
    outputs = [[] for t in range(time_steps)]
    for img_id in time_step_img_dict:
        #print(time_step_img_dict[img_id])
        for i, out in enumerate(time_step_img_dict[img_id]['out']):
            outputs[i].append(out)
            #print(out.shape)
            #input()
    outputs = [torch.stack(out) for out in outputs]
    return outputs


def accuracy_over_t_simple(outputs, target, topk=(1,5)):
    #Outputs = outputs over time
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        correct_t = []
        preds_over_t = []
        for output in outputs[::-1]:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            preds_over_t.append(pred)
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_t.append(correct)
        
        correct_max = torch.max(torch.stack(correct_t), dim=0)#[0]
        correct = correct_max[0]
        correct_ids = correct_max[1]
        
        res = []
        
        for k in range(maxk): #Fix top5 such that if True in earlier k then false in other ks
            for i in range(k):
                correct[k][correct[i]] = False
            
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        
        return res#, cases_where_earlier_time_steps, threshold_earlier_ts
    
def get_processed_output_non_recurrent(output_struct_for_ml, 
                                       or_model, 
                                       images, synset_list, 
                                       imnet_id_to_name, 
                                       image_ids,
                                       top_k=1
                                      ):
    output = or_model(images)
    out_probs, output_classes = output.topk(top_k)
    
    for i, img_id in enumerate(image_ids):
        if(img_id in output_struct_for_ml):
            print("error: duplicate processing of image: {}".format(img_id))
            continue
        output_struct_for_ml[img_id] = [] #0th element is top1
        for j, output_class_num in enumerate(output_classes[i]):
            output_prob = float(out_probs[i][j])
            #print(synset_list[output_class_num], output_prob)
            output_struct_for_ml[img_id].append((synset_list[output_class_num], output_prob))
        output_struct_for_ml[img_id] = sorted(output_struct_for_ml[img_id], key=lambda x:x[1], reverse=True)
    return output_struct_for_ml
        
    
def get_processed_output_recurrent(output_struct_for_ml, 
                                       or_model, 
                                       images, synset_list, 
                                       imnet_id_to_name, 
                                       image_ids,
                                       total_t = 3,
                                        top_k=1
                                  ):
    #1. Filter pred with IOU
    outputs_over_t = filter_pred_with_iou(or_model, images, total_t)
    
    #2. 
    
    for i, img_id in enumerate(image_ids):
        if(img_id in output_struct_for_ml):
            print("error: duplicate processing of image: {}".format(img_id))
            continue
        output_struct_for_ml[img_id] = {} #0th element is top1
        for t, output in enumerate(outputs_over_t):
            output_struct_for_ml[img_id][t] = []
            out_probs, output_classes = output.topk(top_k)
            for j, output_class_num in enumerate(output_classes[i]):
                output_prob = float(out_probs[i][j])
                output_struct_for_ml[img_id][t].append((synset_list[output_class_num], output_prob))
            output_struct_for_ml[img_id][t] = sorted(output_struct_for_ml[img_id][t], key=lambda x:x[1], reverse=True)
    return output_struct_for_ml

def compute_ml_metrics(gt_ml_annot, output_struct_for_ml, is_recurrent=False, topk=1):
    #Currently just check if ground truth is in output or not
    top1_acc = 0.0
    count=0
    results_dict = {}
    for img_id in output_struct_for_ml:
        prediction_struct = output_struct_for_ml[img_id]
        if(img_id not in gt_ml_annot):
            print("Error: img {} not found".format(img_id))
            continue
        if('correct' not in gt_ml_annot[img_id]):
            continue
        if(not is_recurrent):
            top_prediction = prediction_struct[0][0] #first element and its synset name
            if(top_prediction in gt_ml_annot[img_id]['correct']):
                top1_acc+=1
        else:
            for t in prediction_struct:
                top_prediction = prediction_struct[t][0][0]
                if(top_prediction in gt_ml_annot[img_id]['correct']):
                    top1_acc+=1
                    break
        count+=1
    results_dict['accuracy'] = top1_acc/count
    return results_dict

import os
from torch.utils.data import Dataset
from PIL import Image
class Custom_ImageNet_Val(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs#natsorted(all_imgs)
        #print(all_imgs[:10])
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_imgs[idx]
        