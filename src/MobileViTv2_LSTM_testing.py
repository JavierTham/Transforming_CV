import pandas as pd

import torch
import torch.nn as nn

from VideoDataset import *
from CNN_LSTM import CNNLSTM
from functions import *
from config.CNN_LSTM_config import *
from torch.utils.data import DataLoader

NUM_CLASSES = 100
STATE_DICT_PATH = "states/MobileViTv2_LSTM_epoch10.pth"
FRAMES_PATH = "/media/kayne/SpareDisk/data/UCF101/video_frames/"
TEST_DATA_PATH = "../data/UCF_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### model ###
cnnlstm = CNNLSTM(config['lstm_hidden_size'], config['lstm_num_layers'])

state_dict = torch.load(STATE_DICT_PATH)
cnnlstm.load_state_dict(state_dict)
cnnlstm.to(device)
cnnlstm.eval()
###

### data ###
test_df = pd.read_csv(TEST_DATA_PATH)

test_dataset = VideoDataset(FRAMES_PATH, test_df["idx"], test_df["action"])
test_dataloader = DataLoader(test_dataset, **params)
###

criterion = nn.CrossEntropyLoss()
output = predict(cnnlstm, device, test_dataloader)
torch.save(output[0], "../output/MobileViTv2_LSTM/pictures.pt")
torch.save(output[1], "../output/MobileViTv2_LSTM/all_y_true.pt")
torch.save(output[2], "../output/MobileViTv2_LSTM/all_y_pred.pt")
torch.save(output[3], "../output/MobileViTv2_LSTM/y_true_examples.pt")
torch.save(output[4], "../output/MobileViTv2_LSTM/y_pred_examples.pt")