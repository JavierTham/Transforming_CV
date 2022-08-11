# Transforming_CV

## Getting started 

```python
git clone https://github.com/JavierTham/Transforming_CV.git
cd Transforming_CV
pip install -r requirements.txt 
```

## Train a model
Models are loaded from the timm or torchvision library. Check out the list of available models

### Search for available models

_timm_
```python
import timm 

timm.list_models()[:5]
>>> ['adv_inception_v3',
 'cspdarknet53',
 'cspdarknet53_iabn',
 'cspresnet50',
 'cspresnet50d']
```

_torchvision_
```python
from functions import list_torch_models

list_torch_models()
>>> [
```

Torchvision has updated the way we load a pretrained model

Script for timm models
```python
python train ../data/cifar100 mobilevitv2_075 100 --timm --pretrained --epochs 20 --workers 4 --pin-mem
```

Script for torchvision models
```python
python train ../data/cifar100 resnet50 100 --weights ResNet50_Weights.DEFAULT --lr 0.0001 --workers 4 --pin-mem
```

## Validate model

## Further work


## References
Models are taken from [timm](https://github.com/rwightman/pytorch-image-models) and [torchvision](https://github.com/pytorch/vision)

## Using weights and biases
[Weights & Biases](https://wandb.ai/site) is a free (for personal use) MLOps platform that can be integrated easily with pytorch/tensorflow/keras/fastai and other frameworks easily
