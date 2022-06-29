import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_tester import *

class VideoDataset(Dataset):
    def __init__(self, frames_path, video_ids, video_classes):
        super(VideoDataset, self).__init__()
        self.frames_path = frames_path
        self.video_ids = video_ids
        self.video_classes = video_classes
        
    def __getitem__(self, idx):
        video_class = self.video_classes[idx]
        video_id = self.video_ids[idx]
        path = os.path.join(self.frames_path, f"{video_id}.npz")

        frames = np.load(path)[f"{video_id}.npy"]

        ### TEST ###
        # test_data_class(video_id, video_class)

        return torch.tensor(frames), torch.tensor(video_class)
    
    def __len__(self):
        return len(self.video_ids)


class CNNLSTM(nn.Module):
    """
    Creates a CNN-LSTM model from pretrained MobileNetv2
    
    @params
    ---
    lstm_hidden_size: hidden size for the lstm model
    lstm_num_layers: number of layers for the lstm model
    """
    
    def __init__(self, lstm_hidden_size, lstm_num_layers):
        super(CNNLSTM, self).__init__()
        self.cnn = torchvision.models.mobilenet_v2(pretrained=True).features
        # cnn = torchvision.models.alexnet(pretrained=True)
        # self.cnn = nn.Sequential(*list(cnn.children())[:-1]) # remove last layer

        self.AvgPool2d = nn.AvgPool2d(7)
        self.lstm = nn.LSTM(
            1280,
            # 256, # for AlexNet
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True)
        self.fc1 = nn.Linear(1280, 1280)
        self.fc2 = nn.Linear(lstm_hidden_size, 512)
        # 157 classes
        self.fc3 = nn.Linear(512, 157)
        self.dropout = nn.Dropout(0.2)
        
        # freeze entire CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

        # # initialize weights for lstm
        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #          nn.init.constant_(param, 0.0)
        #     elif 'weight_ih' in name:
        #          nn.init.kaiming_normal_(param)
        #     elif 'weight_hh' in name:
        #          nn.init.orthogonal_(param)
                
    def forward(self, x):
        # batch_size, sequence_length, num_channels, height, width
        B, L, C, H, W = x.size()
        # CNN
        output = []
        for i in range(L):
            #input one frame at a time into the basemodel
            x_t = self.cnn(x[:, i, :, :, :])
            x_t = self.AvgPool2d(x_t)

            # Flatten the output
            x_t = x_t.view(x_t.size(0), -1)

            #make a list of tensors for the given smaples 
            output.append(x_t)

        # reshape to (batch_size, sequence_length, output_size)
        x = torch.stack(output, dim=0).transpose_(0, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # LSTM
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :].view(x.size(0), -1)
        # FC
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        
        return x