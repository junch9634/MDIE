## Updates & TODO Lists
- [X] (2023.02.06) README_TEMPLATES.md is released.


## Getting Started

### Environment Setup

Tested on Titan RTX with python 3.7, pytorch 1.13.0, torchvision 0.14.0, CUDA 11.7

1. Install dependencies
```
sudo apt update && sudo apt upgrade
```

2. Set up a python environment
```
conda create -n uien python=3.7
conda activate uien
pip install torch torchvision
pip install opencv-python matplotlib scikit-learn
```

## Train & Evaluation

### Dataset Preparation
1. Download `NYUv2 dataset' given link or from MAT .

```
cd /ailab_mat/dataset/NYU2v
```

2. Organize the folders as follows
```
NYU2v
├── gt
       └──1_0_0_0_0
              ├──0.jpg
              ├──1.jpg
              └──...
├── gt
       └──pretrain_data
              ├──4_0_0_0_0
                     ├──0.jpg
                     ├──1.jpg
                     └──...
              ├──4_1_0_0_0
                     ├──0.jpg
                     ├──1.jpg
                     └──...
              ├──4_0_1_0_0
                     └──...
              ├──4_0_0_1_0
                     └──...
              └──4_0_0_0_1
                     └──...
└── datasets
       ├── multi_degrade
              ├──0_0.jpg
              ├──0_1.jpg
              ├──0_2.jpg
              ├──0_3.jpg
              ├──0_4.jpg     
              ├──1_0.jpg
              ├──1_1.jpg
              └──...
       ├── multi_degrade_0
              ├──0.jpg
              ├──1.jpg
              └──...
```
### Train on sample dataset
```
python train.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 0
python train_sd.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 0   # Super-Resolution and Deblurring
python train_single.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 0   # Only one method 
# Super-Resolution, Deblurring, Denoising, Dehazing, Deraining
python train_mult.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 0 --model_type single   # no branch
python train_mult.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 0 --model_type multi-last-avg   # 5 branch (each branch ) play a role in each task / each branch output sum to fusion
python train_mult.py --lr 1e-5 --nEpochs 100 --step 30 --gpus 0 --model_type multi-tot-avg    # 5 branch (each branch ) play a role in each task / sum loss output of branch / each branch output sum to fusion
```

### Evaluation on test dataset
```
python eval.py --cuda --gpus 0 --model model_srresnet.pth
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.

## Notes

Some codes are rewritten from
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template/edit/master/BLANK_README.md)


## Authors
- **Changhyun Jun** [junch9634](https://github.com/junch9634)

## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.
