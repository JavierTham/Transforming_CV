import torch
import pandas as pd
from torch.utils.data import DataLoader
from functions import *
from CNN_LSTM import *
from config.CNN_LSTM_config import *
from VideoDataset import *

import wandb
import time

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="UCF101", **{"name": "MobileNetv2_LSTM_"+time_now})

### Charades ###
# data_path = "../data/"
# TRAIN_DATA_PATH = os.path.join(data_path, "train_data2.csv")
# TEST_DATA_PATH = os.path.join(data_path, "test_data2.csv")
# FRAMES_PATH = "/media/kayne/SpareDisk/data/video_frames/"

### UCF101 ###
TRAIN_DATA_PATH = "../data/UCF_train.csv"
VAL_DATA_PATH = "../data/UCF_val.csv"

FRAMES_PATH = "/media/kayne/SpareDisk/data/UCF101/video_frames/"

wandb.config = config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnnlstm = CNNLSTM(config['lstm_hidden_size'], config['lstm_num_layers']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnnlstm.parameters(), lr=config['learning_rate'])

train_df = pd.read_csv(TRAIN_DATA_PATH)
val_df = pd.read_csv(VAL_DATA_PATH)

train_dataset = VideoDataset(FRAMES_PATH, train_df["idx"], train_df["action"])
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = VideoDataset(FRAMES_PATH, val_df["idx"], val_df["action"])
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)