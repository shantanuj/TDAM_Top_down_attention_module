### Training/evaluation and usage scripts for TDAM
This directory is adapted from the original timm library code: https://github.com/rwightman/pytorch-image-models 

### Direct usage:
1. For direct usage of TDAM models, please see examples in jupyter notebook `TDAM_usage_and_visualization.ipynb` for model loading and optionally, for how to operate the model to get outputs over each computation/time step.
2. Integrating with "timm":
    - Currently, we have provided our integration with "timm" in the `local_timm` directory. 
    - However, if you wish to integrate with your own version of "timm", please do the following:
        1. Add `tdresnet.py` (provided in this directory) to timm.models
        2. Add `tdresnet` to model registry by modifying timm.models.__init__.py to include line `from .tdresnet import *`
        3. Add `custom_utils.py`, `distributed_td_train_and_eval.sh` and `td_train_and_evaluate.py` to the main directory.
        
### Training and evaluation experiments:
1. Our experiments require following datasets:
    - Download original ImageNet-1k ILSVRC-12 dataset from: https://image-net.org/download-images
    - Extract and convert training and validation images to Pytorch dataloader format (with each class having a subdirectory) as detailed in https://github.com/pytorch/vision/tree/main/references/classification
    - For ImageNet-V2 validation, download dataset from https://github.com/modestyachts/ImageNetV2
    - Additionally, for weakly-supervised object localization, download ImageNet ILSVRC-12 localization annotations for validation data from: https://image-net.org/download-images 
    - For additional tasks of fine-grained and multi-label image classification, please see file `additional_tasks_links.md`
        
2. To train a TDAM based ResNet50 model with distributed GPU training (variants of TDAM-models can be specified by changing arguments such as time_steps, mdist and additional arguments specified in `td_train_and_eval.py`:    
```
./distributed_td_train_and_eval.sh {NUM_GPUs} {dataset_path} --model tdresnet50 --time_steps 2 --mdist 1 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --remode pixel --reprob 0.5 --batch-size 128 --amp -j 4 
```

For example to train a TDAM-ResNet34 with t=2, m=2 and joint attention for 100 epochs:
```
./distributed_td_train_and_eval.sh 4 /data/shantanu/Image_datasets/Imagenet/ImageNet2012-pre/images/ --model tdresnet34 --time_steps 2 --mdist 2 --sched cosine --epochs 100 --warmup-epochs 5 --lr 0.4 --remode pixel --reprob 0.5 --batch-size 128 --amp -j 4 --spotlight_gen_technique joint_top_bottom_concat_attention
```

3. To evaluate a TDAM based model, simply specify the path to the model under the --resume argument and add --validate-only, as done below:
```
./distributed_td_train_and_eval.sh 4 /data/shantanu/Image_datasets/Imagenet/ImageNet2012-pre/images/ --model tdresnet50 --time_steps 2 --mdist 1  --remode pixel --reprob 0.5 --batch-size 128 --amp -j 4 --resume /data/shantanu/TDAM_Models/TD_RNet50_t2_m1_joint.pth.tar --validate-only
```

4. Operating TDAM-models on multiple computation/time steps:
    - By default TDAM-models output only the last computation step. To get outputs over each computation step, set `model.output_over_t` to be True. 
    - For evaluation on ImageNet-1k, to filter cases wherein multiple objects may be present in an image, the predictions of TDAM-models are by default filtered over unique time-steps by only keeping predictions with IOU<0.5.     
 
5. Applying model at different layers/blocks:
    - Currently, for ResNet based models it is recommended to apply TDAM at layers 3 and 4 (the last two layers) as they are deeper in the hierarchy, and the features are more semantically meaningful for top-down attention to operate. 
    - However, if one wants to apply at earlier layers or specific blocks within each layer, change args `apply_td_at_layers_list` and `apply_td_at_blocks_for_layers` as appropriate (note for ResNet101, where TD is not applied at every block of layer3 for efficiency, this is specified on L448 of `td_train_and_evaluate.py`)
    