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


Training script for timm models
```python
python train ../data/cifar100 mobilevitv2_075 100 --timm --pretrained --epochs 20 --workers 4 --pin-mem
```

Training script for torchvision models (copy string for weights directly from the official docs)
```python
python train ../data/cifar100 resnet50 100 --weights ResNet50_Weights.IMAGENET1K_V1 --lr 0.0001 --workers 4 --pin-mem
```

## Validate model

## Further work


## References
Models are taken from [timm](https://github.com/rwightman/pytorch-image-models) and [torchvision](https://github.com/pytorch/vision)

## Using weights and biases
[Weights & Biases](https://wandb.ai/site) is a free (for personal use) MLOps platform that can be integrated easily with pytorch/tensorflow/keras/fastai and other frameworks easily
