# "TDAM: Top-Down Attention Module for Contextually Guided Feature Selection in CNNs" 

- PyTorch implementation for paper "Top-Down Attention Module for Contextually Guided Feature Selection in CNNs".

- To run code, ideally create a virtual/conda environment and install requirements listed in [`requirements.txt`](requirements.txt) by running: 
```
pip install -r requirements.txt
```

- For module usage and performing training/analysis, please see provided scripts in `training_and_analysis_scripts` directory (specifically [`TDAM_usage_and_visualization.ipynb`](TDAM_usage_and_visualization.ipynb) with instructions in that directory's `README.md`. 

- For just the module and model integration/implementation code, please see `modules_and_models` directory.

## ImageNet-1k pre-trained models

|Model|Top-1(%)|Top-5(%)|GoogleDrive|
|:-------:|:----:|:---:|:----------------:|
|TDAM(t2,m2)-RNet18 |72.16|90.61|[TD_ResNet18](https://drive.google.com/file/d/1_dko76uh6YjQG9o_vw6LkXTB_3abYbQG/view?usp=sharing)|
|TDAM(t2,m2)-RNet34 |75.75|92.58|[TD_ResNet34](https://drive.google.com/file/d/1v1DOkjbtXAMUgQLzuox9xCFpG4XgCZGF/view?usp=sharing)|
|TDAM(t2,m1)-RNet50 |78.96|94.19|[TD_ResNet50](https://drive.google.com/file/d/1teK0HxyP_3P1pDLePwesiO8xG_ZAJVR_/view?usp=sharing)|
|TDAM(t2,m1)-RNet101|81.62|95.76|[TD_ResNet101](https://drive.google.com/file/d/1bbUztG6NpL2vUKmTfpjKRiSCjzQvtYzt/view?usp=sharing)|

## Code environment
The codebase and associated experiments are performed in following environment:
- OS: Ubuntu
- CUDA: 11.4
- GPU: NVIDIA Tesla V100 DGXS (16GB)
- Python 3.8.10

## Acknowledgement
The codebase utilizes the [timm](https://github.com/rwightman/pytorch-image-models) and [torchvision](https://github.com/pytorch/vision) libraries.

## License
This project's codebase is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information
In case of any suggestions or questions, please leave a message here or contact me directly at jaiswals@ihpc.a-star.edu.sg, thanks!