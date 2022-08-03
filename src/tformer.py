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

parser = argparse.ArgumentParser(description='PyTorch model training')
parser.add_argument("model",
                    help="name of model")
parser.add_argument("data_dir",
                    help="path to folder containing train and validation data")
parser.add_argument("num_classes", type=int,
                    help="number of class labels")
parser.add_argument("--pretrained", action="store_true",
                    help="use pretrained model for timm models (default: false)")
parser.add_argument("--checkpoint-path", default="",
                    help="path to model state dict")
parser.add_argument("--batch-size", default=128, type=int,
                    metavar='', help="mini-batch size (default: 128)")
parser.add_argument("--lr", default=0.001,
                    help="learning rate (default:0.001)")
parser.add_argument("--workers", default=0, type=int,
                    help="number of workers for dataloader")
parser.add_argument("--epochs", default=10, type=int,
                    help="number of epochs (default: 10)")
parser.add_argument("--img-size", default=224, type=int,
                    metavar="", help="Input image dimension, uses model default if empty")
parser.add_argument("--optimizer", default=torch.optim.Adam,
                    help="optimizer (default: Adam)")
parser.add_argument("--criterion", default=nn.CrossEntropyLoss(),
                    help="criterion (default: CrossEntropyLoss)")
parser.add_argument("--timm", action="store_true",
                    help="Use timm model (default: false - import from torchvision)")
parser.add_argument("--weights", default=None, type=str,
                    help="pretrained weight for torchvision models from API (eg. 'IMAGENET1K_V1', 'DEFAULT' etc.)")

def list_timm_models(filter='', pretrained=False):
    return timm.list_models(filter=filter, pretrained=pretrained)

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

        ### FIX THIS
        last_layer_name = list(model.children())
        layers = getattr(model, last_layer_name)
        if isinstance(layers, nn.Linear):
            in_features = layers.in_features
        else:
            in_features = get_in_features(layers.children())
        setattr(model, last_layer_name, nn.Linear(in_features, int(args.num_classes)))
    model.to(device)

    params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "shuffle": True}

    train_dataset = ImageDataset(X_train, y_train, size=int(args.img_size))
    train_dataloader = DataLoader(train_dataset, **params)
    val_dataset = ImageDataset(X_val, y_val, size=int(args.img_size))
    val_dataloader = DataLoader(val_dataset, **params)

    optimizer = args.optimizer(model.parameters(), lr=float(args.lr))
    for epoch in range(args.epochs):
        model, losses, scores = trainer(model, device, train_dataloader, args.criterion, optimizer, epoch)
        val_loss, val_score = validation(model, device, val_dataloader, args.criterion, optimizer, epoch)

if __name__ == "__main__":
    main()