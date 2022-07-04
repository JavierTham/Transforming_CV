import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from functions import *
from config.DeiT_config import *
from ImagenetDataset import *

import os
import time
import random
import wandb

random.seed(42)

TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/imagenet/train"
NUM_CLASSES = 1000

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="cifar", **{"name": "ViT_" + time_now})
wandb.config = config

d = {"labels": [],
    "data": np.empty((1281167, 3072)),
    "mean": np.empty((1281167, 3072))}

start_idx = 0
end_idx = 0
for i in range(1, 11):
    train_path = os.path.join(TRAIN_DATA_PATH, f"train_data_batch_{i}")

    train_data = unpickle(train_path)
    labels = list(map(lambda x: x-1, train_data["labels"]))
    d["labels"].extend(labels)
 
    data = train_data["data"]
    means = train_data["mean"]

    end_idx = start_idx + data.shape[0]
    d["data"][start_idx:end_idx, :] = data
    d["mean"][start_idx:end_idx, :] = means
    start_idx += len(train_data)

X = d["data"]
y = d["labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deit = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
deit.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deit.parameters(), lr=config['learning_rate'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

train_dataset = ImagenetDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = ImagenetDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(deit, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)