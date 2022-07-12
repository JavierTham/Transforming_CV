import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from functions import *
from CNN_LSTM import *
from VideoDataset import *
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
import wandb
import time
import random

config = {
    "learning_rate": 1e-04,
    "epochs": 10,
    "batch_size": 16,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1, 
    "drop out:": True
}

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="UCF101", **{"name": "CNN_LSTM_"+time_now})

### Charades ###
# data_path = "../data/"
# train_data_path = os.path.join(data_path, "train_data2.csv")
# test_data_path = os.path.join(data_path, "test_data2.csv")
# frames_path = "/media/kayne/SpareDisk/data/video_frames/"

### UCF101 ###
train_data_path = "../data/UCF_df2.csv"
frames_path = "/media/kayne/SpareDisk/data/UCF101/video_frames/"
videos_path = "/media/kayne/SpareDisk/data/UCF101/videos/"

wandb.config = config

params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnnlstm = CNNLSTM(config['lstm_hidden_size'], config['lstm_num_layers']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnnlstm.parameters(), lr=config['learning_rate'])

df = pd.read_csv(train_data_path)

groups = df.loc[:, "group_id"]
vid_classes = df.loc[:, "action"]

classes = sorted(os.listdir(videos_path))
le = preprocessing.LabelEncoder()
le.fit(classes)
vid_classes = le.transform(vid_classes)

group_le = preprocessing.LabelEncoder()
groups = group_le.fit_transform(groups)

gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
for train_idx, test_idx in gss.split(np.ones((df.shape[0],1)), vid_classes, groups):
    pass

train_dataset = VideoDataset(frames_path, train_idx, vid_classes[train_idx])
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = VideoDataset(frames_path, test_idx, vid_classes[test_idx])
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)