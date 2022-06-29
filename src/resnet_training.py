from Resnet import *
from functions import *
from CIFARDataset import *

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import time
import wandb

TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/train"
META_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/meta"
NUM_CLASSES = 100

torch.cuda.empty_cache()

config = {
    "learning_rate": 1e-03,
    "epochs": 20,
    "batch_size": 256,
    "num_workers": 4,
    "pre-train": False
}

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="cifar", **{"name": "ResNet50_" + time_now})
wandb.config = config

params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True,
}

train_data = unpickle(TRAIN_DATA_PATH)
meta_data = unpickle(META_DATA_PATH)

fine_labels = meta_data["fine_label_names"]
coarse_labels = meta_data["coarse_label_names"] 

X_train = train_data["data"]
y_train = train_data["fine_labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = Resnet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=config['learning_rate'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

train_dataset = CIFARDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = CIFARDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(resnet, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)