import time
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config.MobileViTv2_config import *
from functions import *
from CIFARDataset import *

import wandb

STATE_DICT_PATH = "states/mobilevitv2-0.75.pt"
TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/train"
NUM_CLASSES = 100

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="cifar", **{"name": "MobileViTv2_075" + time_now})
wandb.config = config

train_data = unpickle(TRAIN_DATA_PATH)
X_train = train_data["data"]
y_train = train_data["fine_labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilevit = timm.create_model(
	"mobilevitv2_075",
	pretrained=True,
	num_classes=NUM_CLASSES)
mobilevit.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilevit.parameters(), lr=config['learning_rate'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

train_dataset = CIFARDataset(X_train, y_train, size=224)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = CIFARDataset(X_val, y_val, size=224)

val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(mobilevit, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)