import numpy as np

import argparse
import time
import os
from scipy.__config__ import get_info

import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from functions import trainer, validation
from ImageDataset import *

from pprint import pprint

parser = argparse.ArgumentParser(description='PyTorch classification model training')

group = parser.add_argument_group("Dataset")
parser.add_argument("data_dir",
                    help="path to folder containing train and validation folders")

group = parser.add_argument_group("Model parameters")                    
parser.add_argument("model",
                    help="name of model (from timm or torchvision)")
parser.add_argument("num_classes", type=int,
                    help="number of class labels")
group.add_argument("--pretrained", action="store_true",
                    help="use pretrained model for timm models (default: false)")
group.add_argument("--img-size", default=224, type=int, metavar="int",
                    help="Input image dimension, uses model default if empty")                    
group.add_argument("--batch-size", default=128, type=int, metavar="int",
                    help="mini-batch size (default: 128)")
group.add_argument("--checkpoint-path", default="", metavar="str",
                    help="path to model state dict")
group.add_argument("--timm", action="store_true",
                    help="Use timm model (default: false - import from torchvision)")
group.add_argument("--weights", default=None, type=str, metavar="str",
                    help="pretrained weight for torchvision models from API (eg. 'IMAGENET1K_V1', 'DEFAULT' etc.)")

group = parser.add_argument_group("LR parameters")
group.add_argument("--lr", default=0.001, metavar="float",
                    help="learning rate (default:0.001)")
group.add_argument("--epochs", default=10, type=int, metavar="int",
                    help="number of epochs (default: 10)")

group = parser.add_argument_group("Optimizer parameters")
group.add_argument("--optimizer", default=torch.optim.Adam, metavar="str",
                    help="optimizer (default: Adam)")

group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument("--workers", default=0, type=int, metavar="int",
                    help="number of workers for dataloader")

def list_timm_models(filter='', pretrained=False):
    pprint(timm.list_models(filter=filter, pretrained=pretrained))

def list_torch_models():
    import torchvision.models
    pprint(dir(torchvision.models))

def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
    
    Keyword Args:
        num_classes (int): number of output nodes for classification head
        **: other kwargs are model specific
    """

    model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **kwargs)
    return model

# def train_model(
#         model,
#         criterion,
#         optimizer,
#         train_dataloader,
#         val_dataloader,
#         epochs=10):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     for epoch in range(epochs):
#         model, losses, scores = trainer(
#             model,
#             device,
#             train_dataloader,
#             criterion,
#             optimizer,
#             epoch)

#         val_loss, val_score = validation(
#             model,
#             device,
#             val_dataloader,
#             criterion,
#             optimizer,
#             epoch)
 
def get_dataloaders(X_train, y_train, dataset_name):
    train_dataloader = None 
    test_dataloader = None
    return train_dataloader, test_dataloader

def parse_data(data_dir):
    
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "validation")

    X_train = np.load(os.path.join(train_path, "X.npy"))
    y_train = np.load(os.path.join(train_path, "y.npy"))
    X_val = np.load(os.path.join(val_path, "X.npy"))
    y_val = np.load(os.path.join(val_path, "y.npy"))
    return X_train, X_val, y_train, y_val

def get_in_features(layers):
    for layer in layers:
        print(layer)
        if isinstance(layer, nn.Linear):
            return layer.in_features
    print("No in_features found")
    raise Exception

def main():
    args = parser.parse_args()
    print(args)

    X_train, X_val, y_train, y_val = parse_data(args.data_dir)

    # time_now = time.strftime("%D %X")
    # wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="cifar", **{"name": "MobileViTv2_050" + time_now})
    # wandb.config = config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model from timm
    if args.timm:
        model = timm.create_model(
            args.model,
            num_classes=int(args.num_classes),
            pretrained=args.pretrained,
            checkpoint_path=args.checkpoint_path)
    # load model from torchvision
    else:
        model = eval(f"torchvision.models.{args.model}(weights='{args.weights}')")
        # change classifier head

        last_layer_name = list(model.named_children())[-1][0]
        layer = getattr(model, last_layer_name)
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
        else:
            in_features = get_in_features(layer.children())
        setattr(model, last_layer_name, nn.Linear(in_features, int(args.num_classes)))
    model.to(device)

    params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "shuffle": True}

    # change custom Dataset class if needed
    train_dataset = ImageDataset(X_train, y_train, size=int(args.img_size))
    train_dataloader = DataLoader(train_dataset, **params)
    val_dataset = ImageDataset(X_val, y_val, size=int(args.img_size))
    val_dataloader = DataLoader(val_dataset, **params)

    criterion = nn.CrossEntropyLoss()
    optimizer = args.optimizer(model.parameters(), lr=float(args.lr))
    for epoch in range(args.epochs):
        model, losses, scores = trainer(model, device, train_dataloader, criterion, optimizer, epoch)
        val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)

        # if args.save:
        #     pass

if __name__ == "__main__":
    main()