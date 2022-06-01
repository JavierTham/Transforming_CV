import os
import torch
from functions import *
from CNN_LSTM import *

import wandb
# wandb.init(project="Transforming_CV", entity="javiertham")

data_path = "../data/"
train_data_path = os.path.join(data_path, "train_data.csv")
test_data_path = os.path.join(data_path, "test_data.csv")
frames_path = "/media/kayne/SpareDisk/data/video_frames/"

config = {
    "learning_rate": 0.001,
    "epochs": 1,
    "batch_size": 16,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 10, # for benchmarking
    "lstm_num_layers": 1   # for benchmarking
}
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

df = get_df(train_data_path, frames_path)
train_dataset = VideoDataset(df)
train_dataloader = DataLoader(train_dataset, **params)

for epoch in range(config['epochs']):
	losses, scores = train(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)

	# wandb.log({"loss": losses, "scores": scores})