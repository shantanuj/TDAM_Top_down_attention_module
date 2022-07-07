import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

import presets
import utils
import json

try:
    from apex import amp
except ImportError:
    amp = None
    
import sys
sys.path.append("../")
from ptflops import get_model_complexity_info

def train_one_epoch_static(model, criterion, optimizer, data_loader, device, epoch, print_freq, apex=False):
    #Todo: can add increased penalty 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    #print(optimizer)
    header = 'Epoch: [{}]'.format(epoch)
    #torch.autograd.set_detect_anomaly(True)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        if(args.output_over_t and args.out_over_t_method in ['pred_over_t']):
            #1. multiple outputs generated
            outputs = model(image) 
            
            #2. For each output, get loss and predictions
            losses_over_t = []
            preds_over_t = []
            for output in outputs:
                losses_over_t.append(criterion(output, target))
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                preds_over_t.append(pred)
                
            #3. Get minimum loss time indices for each batch 
            argmin_loss_over_t = torch.argmin(torch.stack(losses_over_t),dim=0)
            
            #4. For each image in batch, get min loss prediction / can also use ground truth
            min_loss_preds = []
            for batch_i, min_loss_tstep in enumerate(argmin_loss_over_t):
                min_loss_preds.append(preds_over_t[min_loss_tstep][0][batch_i])
                
            #5. Then go through each time step and each batch and see if prediction overlaps with prediction of min loss
            #If yes, then set loss_t_weight for that time step for that batch index to be 1 or just add to overall loss calculation.. (optionally with higher penalty for those cases with overlap)
            loss_t_weights = torch.zeros_like(torch.stack(losses_over_t))
            for time_i, pred_at_t in enumerate(preds_over_t):
                for batch_i, batch_pred_at_t in enumerate(pred_at_t[0]): #assuming pred_at_t is of shape 1*batch
                    min_loss_pred = min_loss_preds[batch_i]
                    pred_sim = utils.imnet_sim[min_loss_pred][batch_pred_at_t]
                    if(pred_sim>=args.wnet_thresh):
                        if(pred_sim==1):
                            loss_t_weights[time_i][batch_i] = 1.0 #hard threshold / alt. can use similarity 
                        else:
                            loss_t_weights[time_i][batch_i] = args.overlap_pred_pen
                        
            #Calculate final loss as weighted sum
            loss = torch.mean(loss_t_weights*torch.stack(losses_over_t))
            
        else:
            output = model(image)
            loss = torch.mean(criterion(output, target))
        
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

def evaluate(model, criterion, data_loader, device, print_freq=100, use_min_loss_indices_for_acc_over_t = False, wordnet_class_threshold = 0.5, sort_by_confidence=True, over_t=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if(not args.output_over_t or not (args.out_over_t_method in ['pred_over_t'])):
                output = model(image)
                loss = torch.mean(criterion(output, target))
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                outputs = model(image)
                loss = torch.mean(criterion(outputs[0], target))
                (acc1, acc5), cases_where_earlier_time_steps, threshold_earlier_ts = utils.accuracy_over_t(outputs, target, topk=(1,5), wordnet_class_threshold = args.wnet_thresh, sort_by_confidence=args.sort_pred_by_softmax)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setupq
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



    
def evaluate_with_iou_filtering(model, criterion, data_loader, device, print_freq=100, predict_over_t = True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            #output = model(image)
            if(predict_over_t): #means output is of over time predictions
                output = utils.filter_pred_with_iou(model, image, model.module.time_steps, image_res=224)
                #print(type(output))
                #print(output[0].shape)
                loss = torch.mean(criterion(output[0], target))
            else:
                output = model(image)
                loss = torch.mean(criterion(output, target))
            if(type(output)==list):
                acc1, acc5 = utils.accuracy_over_t_simple(output, target, topk=(1,5))
            else:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            
            
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setupq
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (args.resize_size, args.crop_size)

    
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        #random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                             ))
        if args.cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    print("Image size: Resize {} Crop {}".format(args.resize_size, args.crop_size))
    print("Image other vars: Random erase prob {} Rande scale {} Rande ratio {} Rande val {}".format(args.random_erase, args.random_erase_scale, args.random_erase_ratio, args.random_erase_val))
    cache_path = _get_cache_path(valdir)
    
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
        #print('HI2')
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size,
                                             random_erase_prob=args.random_erase,
                                              random_erase_scale = args.random_erase_scale, 
                                              random_erase_ratio = args.random_erase_ratio, 
                                              random_erase_val = args.random_erase_val
                                            ))
        if args.cache_dataset:
            #print('HI2')
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        #print("HI")
        #input()
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args, output_exp_val_log_path, output_exp_model_ckpt_path, output_exp_model_best_path, model_details_file):
    
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    #print(args.res)
    if(args.v2_dataset is not None):
        #from imagenetv2_pytorch import ImageNetV2Dataset
        #from torch.utils.data import DataLoader
        val_dir = os.path.join(args.data_path, args.v2_dataset)
        train_dir = val_dir
        num_classes = 1000
    else:
    
        train_dir = os.path.join(args.data_path, 'train')
        if(args.dataset == 'imnet1k'):
            val_dir = os.path.join(args.data_path, 'val_12')
            num_classes = 1000
        elif(args.dataset == 'imnet200h'):
            val_dir = os.path.join(args.data_path, 'val_ptorch_format')
            num_classes = 200
        else:
            print("ERROR: Dataset: {} not found".format(args.dataset))
            sys.exit(0)
    
        
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    from modules_and_models.custom_resnet import load_custom_resnet
    
    print("=========LOADING MODEL=========")
    if(args.finetune and args.out_over_t_method in ['pred_over_t']):
        output_over_t_method = 'pred_over_t'
    else:
        output_over_t_method = args.out_over_t_method
    
    model = load_custom_resnet(config = args.rnet_config, 
                       resnet_block_type = args.rnet_bl_type,
                       pretrained=args.pretrained,
                       progress=True,
                       custom_blocks_list = args.custom_blocks,
                       att_layer_wise_blocks_list = args.block_layers,
                       block_type = args.block_type,
                       time_steps = args.time_steps,
                       output_over_t = args.output_over_t,
                       output_over_t_blocks = args.out_over_t_blocks, #currently we assume only for layer4
                       m = args.mdist,
                       use_bnt = not args.nbnt,
                     ##TD block specific arguments
                       rep_reduction_tuple = args.rep_reduction,
                       rep_pooling_techniques = args.rep_pooling_ops,
                       rep_combine_technique = args.rep_combine_tech,
                       spotlight_gen_technique = args.spotlight_gen_technique, #top_attention_only, joint_top_bottom_concat, joint_top_bottom_bilinear
                       operation_seq = args.operation_seq,
                       spatial_computation_method = args.sp_tech, #att_scale, crop, sample, 
                       dropout = args.dropout,
                       use_fca = args.use_fca,
                       output_over_t_method = output_over_t_method,
                       num_classes = num_classes,)
    model = model[0] 
    
    if(args.out_over_t_method in ['concat_over_t'] and not args.finetune): #to allow from scratch training for concat or loading a model wit already concat 
        model.change_fc_for_concat_over_t_impl()#The function will check if fc is already ok (in cases where concat over t model is loaded)
        model.output_over_t_method = args.out_over_t_method
        
    if(args.load_custom_pretrained is not None):
        print("LOADING CUSTOM PRETRAINED MODEL from path: {} ".format(args.load_custom_pretrained))
        checkpoint = torch.load(args.load_custom_pretrained, map_location='cpu')
        strict=True
        model.load_state_dict(checkpoint['model'], strict=strict)
        if(args.out_over_t_method in ['concat_over_t'] and args.finetune):
            model.change_fc_for_concat_over_t_impl()#The function will check if fc is already ok (in cases where concat over t model is loaded)
            model.output_over_t_method = args.out_over_t_method
    
            
    
    

       
    print("========LOADED BASE MODEL=========")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("\n Total parameters: {}".format(pytorch_total_params))
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True) 
    with open(model_details_file,'w') as f:
        f.write('\n{:<30}  {:<8}'.format('Computational complexity: ', flops))
        f.write('{:<30}  {:<8}'.format('Number of parameters: ', params))
        f.write("\n Total parameters (method2): {}".format(pytorch_total_params))
        f.write(str(model))
        
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(reduce=False)

    opt_name = args.opt.lower()
    
    #Set optimizer for finetuning (if finetune_lr_fc is provided)
    if(args.finetune and args.finetune_lr_fc is not None):
        backbone_params = list(filter(lambda kv: 'fc' not in kv[0].split('.'), model.named_parameters()))
        backbone_params = list(map(lambda x:x[1], backbone_params))
        fc_params = list(filter(lambda kv: 'fc' in kv[0].split('.'), model.named_parameters()))
        fc_params = list(map(lambda x:x[1], fc_params))
        
        if opt_name == 'sgd':
            optimizer = torch.optim.SGD([{'params':backbone_params},
                                         {'params':fc_params, 'lr':args.finetune_lr_fc}],
                 lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop([{'params':backbone_params},
                                             {'params':fc_params, 'lr':args.finetune_lr_fc}], lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
        else:
            raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))
    
    else:
        if opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
        else:
            raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    
    
    if args.distributed:
        if(args.out_over_t_method in ['concat_over_t']):
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    print("Image size: Resize {} Crop {}".format(args.resize_size, args.crop_size))
    print("Image other vars: Random erase prob {} Rande scale {} Rande ratio {} Rande val {}".format(args.random_erase, args.random_erase_scale, args.random_erase_ratio, args.random_erase_val))   
    if args.test_only:
        if(args.eval_over_iou):
            new_acc1, new_acc5 = evaluate_with_iou_filtering(model, criterion, data_loader_test, device, print_freq=100, predict_over_t = True)
        else:
            new_acc1, new_acc5 = evaluate(model, criterion, data_loader_test, device=device, use_min_loss_indices_for_acc_over_t = False, wordnet_class_threshold = args.wnet_thresh, sort_by_confidence=True)
        return

    print("Start training")
    start_time = time.time()
    top_acc1 = 0.0
    is_best=False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_one_epoch_static(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, args.apex)
        
        lr_scheduler.step()
        if(args.eval_over_iou and args.use_custom_block_model and epoch>88): #filter for multiple objects
            new_acc1, new_acc5 = evaluate_with_iou_filtering(model, criterion, data_loader_test, device, print_freq=100, predict_over_t = True)
        else:
            new_acc1, new_acc5 = evaluate(model, criterion, data_loader_test, device=device, use_min_loss_indices_for_acc_over_t = False, wordnet_class_threshold = args.wnet_thresh, sort_by_confidence=True)
            
        if(new_acc1 > top_acc1):
            is_best = True
            with open(output_exp_val_log_path,'a') as f:
                f.write("NEW_BEST: epoch: {} , top1: {}, top5: {}\n".format(epoch, new_acc1, new_acc5))
                print("NEW_BEST: epoch: {} , top1: {}, top5: {}".format(epoch, new_acc1, new_acc5))
            top_acc1 = new_acc1
        else:
            is_best = False
            with open(output_exp_val_log_path,'a') as f:
                f.write("epoch: {} , top1: {}, top5: {}\n".format(epoch, new_acc1, new_acc5))
                print("epoch: {} , top1: {}, top5: {}".format(epoch, new_acc1, new_acc5))
                
        if args.output_dir or args.resume:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            if(is_best):
                utils.save_on_master(
                    checkpoint,
                    output_exp_model_best_path)
                
            utils.save_on_master(
                checkpoint,
                output_exp_model_ckpt_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='/data/Image_datasets/Imagenet/ImageNet2012-pre/images/', help='dataset')
    parser.add_argument('--dataset','--ds', default='imnet1k', help='dataset')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--rnet_config','--rconfig', default=[3,4,6,3], help='model')
    parser.add_argument('--rnet_bl_type','--rbl', default='bottleneck', help='model')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/data/00_Vision_models/ImageNet', help='path where to save')
    parser.add_argument('--save_name', default='basic')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune', action='store_true', help='specify finetune from checkpoint')
    parser.add_argument('--finetune_lr_fc','--lr_fc', default=None)
    #finetune_lr_fc
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
    parser.add_argument('--random-erase','--rep', default=0.0, type=float, help='random erasing probability (default: 0.0)')
    parser.add_argument('--random-erase-scale','--res', default=(0.02,0.33), type=str, help='random erasing scale (default: 0.0)')
    parser.add_argument('--random-erase-ratio','--rer', default=(0.3,3.3), type=str, help='random erasing scale (default: 0.0)')
    parser.add_argument('--random-erase-val','--rev', default=0.0, type=float, help='random erasing value (default: 0.0)')


    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    
    #######New arguments specific to TD model 
    #parser.add_argument('--resnet_block_type','--rbt', type=str, default='bottleneck')
    parser.add_argument('--load_custom_pretrained', '--lcpt', type=str, default=None)
    parser.add_argument('--use_custom_block_model','--cbm', action = 'store_true', help="Use custom blocks")

    parser.add_argument('--custom_blocks','--custom_layers','--cb', default=[0,1,2,3,4], type =str, help='Specifically for resnet, specify which Layer is a custom block')
    
    parser.add_argument('--block_layers','--custom_bls','--cbs', default=[[0,1,2,3,4,5],
                                                                          [0,1,2,3,4,5],
                                                                          [0,1,2,3,4,5],
                                                                          [0,1,2,3,4,5],
                                                                         ], type =str, help='Specifically for resnet, specify for each layer which sub-layer/block is a custom block') 

    parser.add_argument('--block_type', '--bt', default = 'td_spotlight', type=str, help= "cbam, td_cbam, se, td_se")
    parser.add_argument('--time_steps', default = 3, type=int, help="1 means default forward pass")
    parser.add_argument('--output_over_t','--out_ot', action = 'store_true', help="Use custom blocks")
    
    parser.add_argument('--out_over_t_method','--oot_m',default='pred_over_t', help='concat_over_t, pred_over_t')
    #parser.add_argument('--pred_over_t','--oot_m',default='pred_over_t', help='concat_over_t, pred_over_t')
    
    parser.add_argument('--out_over_t_blocks','--out_ot_blocks', default=[0,1,2], type =str, help='Specifically for resnet, specify which Layer is a custom block')
    parser.add_argument('--mdist', default =3, help='distance b/w top and bottom layer', type=int)
    parser.add_argument('--resize_size','--rsz', default =256, help='image resize size', type=int)
    parser.add_argument('--crop_size','--csz', default =224, help='image crop size', type=int)
    
    parser.add_argument('--nbnt','--not_bnt', action = 'store_true') 
    parser.add_argument('--rep_reduction', default=('r',16), type=str)
    parser.add_argument('--rep_pooling_ops', default=['avp'], type=str)
    parser.add_argument('--rep_combine_tech', default='sum', type=str, help='sum, mean')
    
    parser.add_argument('--spotlight_gen_technique', '--st_gen', default = 'joint_top_bottom_concat_attention', type=str, help="top_attention_only, joint_top_bottom_concat_attention,")
    
    ##For neuro analysis experiments
    parser.add_argument('--operation_seq', default='channel->spatial', type=str)
    parser.add_argument('--sp_tech','--spcm',  default = 'att_scale', type=str, help="att_scale, stn_only_from_feature_map, stn_only_from_searchlight, att_scale_then_stn_from_feature_map, att_scale_then_stn_from_searchlight")
    
    parser.add_argument('--channel_nonlin','--cnlin',default='sigmoid')
    parser.add_argument('--spatial_searchlight_nonlin','--ssnlin',default=None)
    
    
    parser.add_argument('--dropout','--dp',type=float, default=0)
    parser.add_argument('--use_fca', '--ufca', action='store_true')
    
    
    parser.add_argument('--del_dir','--ddir', action='store_true')
    parser.add_argument('--local_rank')
    
    parser.add_argument('--wnet_thresh', type=float, default=0.5)
    parser.add_argument('--sort_pred_by_softmax', '--spbs', action='store_true')
    parser.add_argument('--overlap_pred_pen', '--opp', default=1.0,type=float)
    
    parser.add_argument('--eval_over_iou', '--eou', action='store_true')
    parser.add_argument('--v2_dataset', '--v2d', type=str, default=None)

    return parser

def process_args(args):
    args.resize_size = int((256/224)*args.crop_size)
    if(args.use_custom_block_model and 'td' in args.block_type):
        args.eval_over_iou = True
    #1. Convert list inputs into list
    
    if(type(args.custom_blocks) is not list):
        args.custom_blocks = [int(i) for i in args.custom_blocks.split(',')]
        
    if(type(args.block_layers) is not list):
        args.block_layers = [int(i) for i in args.block_layers.split(',')]
        
    if(type(args.rep_reduction) is not tuple):
        args.rep_reduction = [i for i in args.rep_reduction.split(',')]
        args.rep_reduction[1] = int(args.rep_reduction[1])
        args.rep_reduction = tuple(args.rep_reduction)
     
    if(type(args.rep_pooling_ops) is not list):
        args.rep_pooling_ops = [i for i in args.rep_pooling_ops.split(',')]
    
    if(type(args.out_over_t_blocks) is not list):
        args.out_over_t_blocks = [i for i in args.out_over_t_blocks.split(',')]
    
    if(type(args.rnet_config) is not list):
        args.rnet_config = [int(i) for i in args.rnet_config.split(',')]
    
    if(type(args.random_erase_scale) is not tuple):
        args.random_erase_scale = tuple([float(i) for i in args.random_erase_scale.split(',')])
    if(type(args.random_erase_ratio) is not tuple):
        args.random_erase_ratio = tuple([float(i) for i in args.random_erase_ratio.split(',')])
    
    if(not args.use_custom_block_model):
        args.block_layers = [[],[],[],[]]
        args.custom_blocks = []
        args.block_type = None
    
    #2. Store important vars for path name
    output_name = "{}_{}_{}".format(args.model, args.rnet_config, args.rnet_bl_type)
    
    dataset_dir = args.data_path
    if(dataset_dir.split('/')[-1]==''):
        dataset_name = dataset_dir.split('/')[-2]#'imnet_{}'.format(num_classes)
    else:
        dataset_name = dataset_dir.split('/')[-1]
    
    output_dir = os.path.join(args.output_dir, dataset_name)
    output_dir = os.path.join(output_dir, output_name)
    
    
    
        
    
    if(args.use_custom_block_model):
        output_name = "{}_{}_bl{}_t{}_m{}_d{}".format(str(args.rnet_config)+str(args.rnet_bl_type), args.save_name, args.block_type, args.time_steps, args.mdist,args.dropout)
        
        if(args.output_over_t):
            output_name+= "_OUTOt"
        
            
    else:    
        output_name = "{}_basic_{}".format(str(args.rnet_config)+str(args.rnet_bl_type), args.save_name)
        
    
    if(args.finetune):
        output_name+="_FT"
    
    if(args.use_fca):
        output_name+="_FCA"
    #use_fca = args.use_fca,
    
    if(args.load_custom_pretrained is not None):
        output_name += "_cpretrained"
    elif(args.pretrained):
        output_name += "_pretrained"
        
    output_exp_val_log_path = os.path.join("./train_logs/val_logs", "{}_val_best.txt".format(output_name))
    exp_config_file = os.path.join("./train_logs/setups/", '{}_Settings.json'.format(output_name))
    model_details_file_path = os.path.join("./train_logs/setups/", '{}_model_details.txt'.format(output_name))
    #output_exp_val_log_path = os.path.join("./train_logs", "{}_val_best.txt".format(output_name))
    
    output_exp_model_ckpt_path = os.path.join(output_dir, "{}_checkpoint.pth".format(output_name))
    output_exp_model_best_path = os.path.join(output_dir, "{}_model_best.pth".format(output_name))
    
    
    if(not (args.resume or args.test_only)):
        if(os.path.isfile(exp_config_file)):
            if(args.del_dir):
                if(os.path.isfile(output_exp_val_log_path)):
                    os.remove(output_exp_val_log_path)
                if(os.path.isfile(output_exp_model_ckpt_path)):
                    os.remove(output_exp_model_ckpt_path)
                if(os.path.isfile(output_exp_model_best_path)):
                    os.remove(output_exp_model_best_path)
                if(os.path.isfile(model_details_file_path)):
                    os.remove(model_details_file_path)
                os.remove(exp_config_file)
                sys.exit(0)
            
            print("*************Warning: Path already exists: {}. End process to prevent deletion...*****************".format(exp_config_file))
            #import time
            #time.sleep(5) 
            z = input()
       
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir)
            
        
        with open(exp_config_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return args, output_exp_val_log_path, output_exp_model_ckpt_path, output_exp_model_best_path, model_details_file_path
    
    
    #3. Return output 
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args, output_exp_val_log_path, output_exp_model_ckpt_path, output_exp_model_best_path, model_details_file_path = process_args(args)
    main(args, output_exp_val_log_path, output_exp_model_ckpt_path, output_exp_model_best_path, model_details_file_path)
