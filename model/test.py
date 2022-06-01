import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from functions import *
from CNN_LSTM import *

train_data_path = "../data/train_data.csv"
video_path = "../data/Charades_v1"
frames_path = "/media/kayne/SpareDisk/data/video_frames/"

def get_df(train_data_path, frames_path):
	df = pd.read_csv(train_data_path)
	df = df.loc[:, ["id", "vid_class"]]
	df['id'] = df.loc[:, 'id'].apply(lambda x: os.path.join(frames_path, f"{x}.npz"))
	return df

### ------------ test custom Dataset ------------ ###
# df = get_df(train_data_path, frames_path)
# print(df.head())

# dataset = VideoDataset(df)
# for data in dataset:
# 	X, y = data[0], data[1]
# 	print("Size/shape of frames:", X.shape)
# 	print("Class:", y)
# 	break

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for data in dataset:
# 	X, y = data[0], data[1]
# 	print("X type:", type(X))
# 	print("y type:", type(y))
# 	X, y = X.to(device), y.to(device)
# 	print("X device:", X.device)
# 	print("y device:", y.device)
# 	break

### ------------ test training ------------- ###
config = {
    "learning_rate": 0.001,
    "epochs": 1,
    "batch_size": 16,
    "sequence_len": 30,
    "lstm_hidden_size": 10, # for benchmarking
    "lstm_num_layers": 1   # for benchmarking
}

params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': 4,
	'pin_memory': True
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnnlstm = CNNLSTM(config['lstm_hidden_size'], config['lstm_num_layers']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnnlstm.parameters(), lr=config['learning_rate'])

df = get_df(train_data_path, frames_path).head()
train_dataset = VideoDataset(df)
train_dataloader = DataLoader(train_dataset, **params)

for epoch in range(config['epochs']):
	train(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)