import pandas as pd
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

## ---------------------- Dataloader ---------------------- ##
class VideoDataset(Dataset):
    """
    df - dataframe of path to each video and their labels
    """
    
    def __init__(self, df, seq_len=100):
        super(VideoDataset, self).__init__()
        self.df = df
        self.seq_len = seq_len
        self.transform = self.get_transforms()
    
    def get_transforms(self):
        "for MobileNetv2" 
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __getitem__(self, idx):
        '''
        adapted from bleedai
        '''
        
        frames_list = []
        video_classes = []
        
        video_path = self.df.iloc[idx, 0]
        video_reader = cv2.VideoCapture(video_path)
        video_class = self.df.iloc[idx, 1]
        
        video_classes.append(video_class)
        
        # Get the total number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count / self.seq_len), 1)

        for frame_counter in range(self.seq_len):
            # Set the current frame position of the video, loop video if video too short
            frame_position = frame_counter * skip_frames_window % video_frames_count 
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            success, frame = video_reader.read() 

            if not success:
                break

            processed_frame = self.transform(Image.fromarray(frame))
            frames_list.append(processed_frame)

        video_reader.release()
        

        ### FOR TESTING PURPOSES ###
        print(video_path)
            
        return torch.stack(frames_list), video_classes[0]
    
    def __len__(self):
        return len(self.df)


class CNNLSTM(nn.Module):
    """
    Creates a CNN-LSTM model from pretrained MobileNetv2
    
    @params
    ---
    freeze_layers: freeze the cnn model parameters from 0:freeze_layers
    lstm_hidden_size: hidden size for the lstm model
    lstm_num_layers: number of layers for the lstm model
    """
    
    def __init__(self, batch_size, freeze_layers, lstm_hidden_size, lstm_num_layers):
        super(CNNLSTM, self).__init__()
        self.cnn = torchvision.models.MobileNetV2().features
        self.lstm = nn.LSTM(1280*7*7, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 157)
        self.batch_size = batch_size
        
        for param in self.cnn[:freeze_layers].parameters():
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
        x = x[:, -1, :].view(self.batch_size, -1)
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