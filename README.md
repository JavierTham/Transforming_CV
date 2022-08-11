# Transforming_CV

## Getting started 

```python
git clone https://github.com/JavierTham/Transforming_CV.git
cd Transforming_CV
pip install -r requirements.txt 
```

## Train a model
Models are loaded from the timm or torchvision library. Check out the list of available models
timm
```python
import timm 

timm.list_models()[:5]
>>> ['adv_inception_v3',
 'cspdarknet53',
 'cspdarknet53_iabn',
 'cspresnet50',
 'cspresnet50d']
```

torchvision
```python
from functions import list_torch_models

list_torch_models()
>>> [
```

## Validate model

## Further work


## References
Models are taken from [timm](https://github.com/rwightman/pytorch-image-models) and [torchvision](https://github.com/pytorch/vision)

## Using weights and biases
[Weights & Biases](https://wandb.ai/site) is a free (for personal use) MLOps platform that can be integrated easily with pytorch/tensorflow/keras/fastai and other frameworks easily
