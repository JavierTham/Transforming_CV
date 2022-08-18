import numpy as np

import argparse
import time
import os

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from src.functions import predict, get_in_features, create_torch_model
from src.ImageDataset import *

# import wandb

parser = argparse.ArgumentParser(description='PyTorch classification model training')

group = parser.add_argument_group("Dataset")
parser.add_argument("data_dir",
                    help="path to folder containing train and validation folders")

group = parser.add_argument_group("Model parameters")                    
parser.add_argument("model",
                    help="name of model (from timm or torchvision)")
parser.add_argument("num_classes", type=int,
                    help="number of class labels")
group.add_argument("--timm", action="store_true",
                    help="Use timm model (default: false - import from torchvision)")
group.add_argument("--pretrained", action="store_true",
                    help="use pretrained model for timm models (default: false)")
group.add_argument("--weights", default=None, type=str, metavar="str",
                    help="pretrained weight for torchvision models from API (eg. 'IMAGENET1K_V1', 'DEFAULT' etc.)")
group.add_argument("--img-size", default=224, type=int, metavar="int",
                    help="Input image dimension, uses model default if empty")                    
group.add_argument("--batch-size", default=64, type=int, metavar="int",
                    help="mini-batch size (default: 64)")
group.add_argument("--checkpoint-path", default="", metavar="str",
                    help="path to model state dict")

group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument("--workers", default=0, type=int, metavar="int",
                    help="number of workers for dataloader")
group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

def parse_data(data_dir):
    path = data_dir.split("/")
    test_path = os.path.join(*path, "test")

    X_test = np.load(os.path.join(test_path, "X.npy"))
    y_test = np.load(os.path.join(test_path, "y.npy"))
    
    return X_test, y_test

def main():
    args = parser.parse_args()
    print(args)

    ### Can change to whatever logging software you use
    # time_now = time.strftime("%D %X")
    # log_params = dict(
    #     project = "Transforming_CV",
    #     entity = "javiertham",
    #     config = vars(args),
    #     group = "cifar",
    #     name = args.model + "_" + time_now
    # )
    # wandb.init(**log_params)
    # wandb.config = config

    X_test, y_test = parse_data(args.data_dir)

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
        model = create_torch_model(args.model, args.weights)
        # change classifier head
        last_layer_name = list(model.named_children())[-1][0]
        layer = getattr(model, last_layer_name)
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
        else:
            in_features = get_in_features(layer.children())
        setattr(model, last_layer_name, nn.Linear(in_features, int(args.num_classes)))

        if args.checkpoint_path:
            state_dict = torch.load(args.checkpoint_path)
            model.load_state_dict(state_dict)

    model.to(device)

    # change preprocessing for image if required
    preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.pin_mem,
        "shuffle": True}

    # change custom Dataset class if required
    test_dataset = ImageDataset(X_test, y_test, preprocess)
    test_dataloader = DataLoader(test_dataset, **params)

    try:
        all_y_true, all_y_pred = predict(model, device, test_dataloader)
        output_path = "output"
        torch.save(all_y_true, os.path.join(output_path, "all_y_true.pt"))
        torch.save(all_y_pred, os.path.join(output_path, "all_y_pred.pt"))
        score = accuracy_score(all_y_true, all_y_pred)
        print("score:", score)

        # log score
        # wandb.init(project="Transforming_CV", entity="javiertham", group="cifar", **{"name": f"{model}"})
        # wandb.log({"Test Accuracy": score})
        
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()