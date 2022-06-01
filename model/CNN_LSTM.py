import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

## ---------------------- Dataloader ---------------------- ##
class VideoDataset(Dataset):
    """
    df - dataframe of path to each video frames and their labels
    """

    def __init__(self, df):
        super(VideoDataset, self).__init__()
        self.df = df
        
    def __getitem__(self, idx):
        frames_path = self.df.iloc[idx, 0]
        video_class = self.df.iloc[idx, 1]
        
        frames = np.load(frames_path)
        frames = np.transpose(frames['arr_0'], (0, 3, 1, 2)) # each compressed .npz file only has 1 "arr_0.npy" file
        frames = torch.Tensor(frames)

        ### FOR TESTING PURPOSES ###
        # print(type(video_class))
            
        return frames, torch.tensor(video_class)
    
    def __len__(self):
        return len(self.df)


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
        self.cnn = torchvision.models.MobileNetV2().features

        # output of MobileNetv2 feature layer (1280, 7, 7)
        self.lstm = nn.LSTM(
            1280*7*7,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True)

        # 157 classes
        self.fc = nn.Linear(lstm_hidden_size, 157)
        
        for param in self.cnn.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        # batch_size, sequence_length, num_channels, height, width
        B, L, C, H, W = x.size()
        x = x.view(B * L, C, H, W)
        # CNN
        x = self.cnn(x)
        x = x.view(x.size(0), -1)    # x.size(0): B*L
        x = x.view(B, L, x.size(-1))
        # LSTM
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :].view(x.size(0), -1)
        # FC
        x = self.fc(x)
        
        return x

########################################################
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred