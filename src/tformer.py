import timm
import time
import torch
import torch.nn as nn
import argparse
from functions import trainer, validation
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch model training')
parser.add_argument("model",
                    help="name of model")
parser.add_argument("-p", "--pretrained", action="store_true",
                    help="Use pretrained model")
parser.add_argument("--checkpoint-path", default="", metavar="",
                    help="path to model state dict")
parser.add_argument("--num-classes", type=int, metavar="",
                    help="number of class labels")
parser.add_argument("-b", "--batch-size", default=128, type=int,
                    metavar='', help="mini-batch size (default: 128)")
parser.add_argument("--lr", default=0.001,
                    help="learning rate (default:0.001)")
parser.add_argument("-e", "--epochs", default=10, type=int,
                    help="number of epochs (default: 10)")
parser.add_argument("--img-size", default=None, type=int,
                    metavar="", help="Input image dimension, uses model default if empty")
parser.add_argument("-o", "--optimizer", default=torch.optim.Adam,
                    help="optimizer (default: Adam)")
parser.add_argument("-c", "--criterion", default=nn.CrossEntropyLoss,
                    help="criterion (default: CrossEntropyLoss)")

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

def train_model(
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model, losses, scores = trainer(
            model,
            device,
            train_dataloader,
            criterion,
            optimizer,
            epoch)

        val_loss, val_score = validation(
            model,
            device,
            val_dataloader,
            criterion,
            optimizer,
            epoch)

def get_dataloaders(X_train, y_train, dataset_name):
    train_dataloader = None 
    test_dataloader = None
    return train_dataloader, test_dataloader

def main():
    args = parser.parse_args()
    print(args)
    # TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/train"
    # NUM_CLASSES = 100

    # time_now = time.strftime("%D %X")
    # wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="cifar", **{"name": "MobileViTv2_050" + time_now})
    # wandb.config = config

    # train_data = unpickle(TRAIN_DATA_PATH)
    # X_train = train_data["data"]
    # y_train = train_data["fine_labels"]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mobilevit = timm.create_model(
    #     "mobilevitv2_075",
    #     pretrained=True,
    #     num_classes=NUM_CLASSES)
    # mobilevit.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(mobilevit.parameters(), lr=config['learning_rate'])

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    # train_dataset = CIFARDataset(X_train, y_train, size=224)
    # train_dataloader = DataLoader(train_dataset, **params)
    # val_dataset = CIFARDataset(X_val, y_val, size=224)
    # val_dataloader = DataLoader(val_dataset, **params)

    # for epoch in range(config['epochs']):
    #     model, losses, scores = trainer(mobilevit, device, train_dataloader, criterion, optimizer, epoch)
    #     val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)

if __name__ == "__main__":
    main()