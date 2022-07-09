1. Source code adapted from: https://github.com/pytorch/vision/tree/main/references/classification and https://github.com/rwightman/pytorch-image-models/tree/master/timm 

2. To see examples of loading/creating TD models, see examples in jupyter notebook "Model_loading_and_visualization_examples.ipynb" (requires jupyter to run)

3. Our experiments require following datasets:
    - Download original ImageNet-1k ILSVRC-12 dataset from: https://image-net.org/download-images
    - Extract and convert training and validation images to Pytorch dataloader format (with each class having a subdirectory) as detailed in https://github.com/pytorch/vision/tree/main/references/classification
    - For ImageNet-V2 validation, download dataset from https://github.com/modestyachts/ImageNetV2
    - Additionally, for weakly-supervised object localization, download ImageNet ILSVRC-12 localization annotations for validation data from: https://image-net.org/download-images 
    
4. Training commands: 
    - To train an original ResNet50 model (which has block configuration of 3,4,6,3 bottleneck blocks) with distributed GPU training:
        - CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 25903 --use_env main_train.py --data-path /data/Image_datasets/Imagenet/ImageNet2012-pre/images/ -b 64 --save_name basic_ResNet50 --rnet_config 3,4,6,3 --rbl bottleneck --model resnet50 --output-dir /data/Vision_models/
        - Incase of 'path already exists' warning (due to distributed usage), simply press enter
        - Make sure both 'data-path' refers to ImageNet dierctory containing train and eval and 'output-dir' refer to existing directories
        - Similarly, to train a ResNet18 (or other variant), simply change rnet_config to 2,2,2,2 and rbl to basic (as per original ResNet18 config)
        
    - To train a TD based ResNet50 model with distributed GPU training:
        - CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 25903 --use_env main_train.py --data-path /data/Image_datasets/Imagenet/ImageNet2012-pre/images/ -b 64 --save_name tdTOP_m2_t2 --rnet_config 3,4,6,3 --rbl bottleneck --cbm --cb 3,4 --bt td_spotlight --mdist 1 --time_steps 2 --sp_tech att_scale --epochs 125  --model resnet50 --output-dir /data/Vision_models/ --spotlight_gen_technique top_attention_only
        - cb refers to layers at which top-down feedback is enabled
        - if specific blocks are to be set for each layer, then go to main_train.py and change argument 'custom_bls' 
        
    - To train a TD based model with timm library training, please refer to timm_td_train.py and timm_td_resnet.py which can be directly integrated with timm code (https://github.com/rwightman/pytorch-image-models/tree/master/timm).
    - To train a TD based ConvNext model, please refer to td_convnext_main.py and tdconvnext.py which can be directly integrated with official ConvNeXt code (https://github.com/facebookresearch/ConvNeXt).   

5. Evaluation commands:
    - To only evaluate a model, simply add 'test-only' and --lcpt with existing model path as shown below  
        - CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 25903 --use_env main_train.py --data-path /data/Image_datasets/Imagenet/ImageNet2012-pre/images/ -b 64 --save_name tdTOP_m2_t2 --rnet_config 3,4,6,3 --rbl bottleneck --cbm --cb 3,4 --cbs 0,1,2,3,4,5 --bt td_spotlight --mdist 1 --time_steps 3 --sp_tech att_scale --epochs 125  --model resnet50 --output-dir /data/Vision_models/ --test-only --lcpt /data/Vision_models/tdTOP_m2_t2_model_best.ckpt
        
    - To evaluate on V2 validation sets, modify --data-path to refer to ImageNetv2 directory and modify --v2_dataset with name of  split type (e.g. imagenetv2-top-images-format-val)
        - CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 25903 --use_env main_train.py --data-path /data/Image_datasets/Imagenet/ImageNet2012-pre/V2/ -b 64 --save_name tdTOP_m2_t2 --rnet_config 3,4,6,3 --rbl bottleneck --cbm --cb 3,4 --cbs 0,1,2,3,4,5 --bt td_spotlight --mdist 1 --time_steps 2 --sp_tech att_scale --epochs 125  --model resnet50 --output-dir /data/Vision_models/ --test-only --lcpt /data/Vision_models/tdTOP_m2_t2_model_best.ckpt --v2_dataset imagenetv2-top-images-format-val

    - For grad-CAM/localization maps over computation steps, pls see usage in the jupyter notebook _"Model_loading_and_visualization_examples.ipynb"_ and associated scripts in _utils.py_ 
    
