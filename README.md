# Transforming_CV

## Getting started 

```python
git clone https://github.com/JavierTham/Transforming_CV.git
cd Transforming_CV
conda env create -f environment.yml
conda activate transforming_cv
```

_Use conda environments as it is easier to download required cuda dependencies for pytorch_

_if using venv, remember to install CUDA for pytorch! https://pytorch.org/get-started/locally/_

## Train a model
Models are loaded from the timm or torchvision library.

### Search for available models

**timm**

```python
import timm 

timm.list_models()[:5]
>>> ['adv_inception_v3',
 'cspdarknet53',
 'cspdarknet53_iabn',
 'cspresnet50',
 'cspresnet50d']
```

**torchvision**

Check out their [docs](https://pytorch.org/vision/0.13/models.html) or [source code](https://github.com/pytorch/vision/tree/main/torchvision/models) to find all possible models and their available pretrained weights

_*Torchvision has updated the way we load a pretrained model_

<br>

Basic syntax for training a model
```python
python train.py data_dir model num_classes [OPTIONS]
```

Training script for pretrained timm models
```python
python train.py data/cifar100 mobilevitv2_075 100 --timm --pretrained --epochs 20 --workers 4 --pin-mem --batch-size 32
```

Training script for pretrained torchvision models (copy string for weights directly from the official docs)
```python
python train.py data/cifar100 resnet50 100 --weights ResNet50_Weights.IMAGENET1K_V1 --lr 0.0001 --workers 4 --pin-mem --batch-size 32
```

**Training script trains the model and tests on the validation set in data/cifar100/validation (for eg.)**

<br>

for more help
```python
python train.py --help
```

## Validate model

We can validate our trained model with saved 

```python
python validate.py data/cifar100 mobilevitv2_075 100 --timm --checkpoint-path states/model_epoch9.pt
```

```python
python validate.py data/cifar100 resnet50 100 --checkpoint-path states/model_epoch5.pt
```

## Directory structure

```
.
├── data
│   └── cifar100
│       ├── train
│       │   ├── X.npy
│       │   └── y.npy
│       ├── validation
│       │   ├── X.npy
│       │   └── y.npy 
│       └── test
│           ├── X.npy
│           └── y.npy
├── output
├── src
│   ├── functions.py
│   ├── ImageDataset.py
│   └── ...
├── states
├── train.py
└── validate.py
```

## References
Models are taken from [timm](https://github.com/rwightman/pytorch-image-models) and [torchvision](https://github.com/pytorch/vision)

## Useful links
Repo of collated papers for vision transformer and attention
https://github.com/cmhungsteve/Awesome-Transformer-Attention

# Weights and biases
[Weights & Biases](https://wandb.ai/site) is a free (for personal use) MLOps platform that can be integrated easily with pytorch/tensorflow/keras/fastai and other frameworks easily

### Further work
Create corresponding docker image
