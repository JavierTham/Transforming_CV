import os
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from CNN_LSTM import *
from functions import *
import random

random.seed(42)

### Charades ###
# train_data_path = "../data/train_data2.csv"
# video_path = "../data/Charades_v1"
# frames_path = "/media/kayne/SpareDisk/data/video_frames/"

### UCF101 ###
train_data_path = "../data/UCF_df.csv"
frames_path = "/media/kayne/SpareDisk/data/UCF101/video_frames/"
videos_path = "/media/kayne/SpareDisk/data/UCF101/videos/"

config = {
    "learning_rate": 1e-03,
    "epochs": 20,
    "batch_size": 16,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 1
}

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

### Charades ###
# num_examples = 3
# df = pd.read_csv(train_data_path).iloc[:num_examples, :] # test a few, shld overfit quickly
# print(df)
# video_ids = list(range(num_examples))
# vid_classes = np.asarray(df['vid_class'])

### UCF101 ###
num_examples = 5
df = pd.read_csv(train_data_path) #.iloc[:num_examples, :]
video_ids = [random.randrange(0, len(df)) for _ in range(num_examples)]
vid_classes = df.loc[video_ids, "action"]

print(video_ids)

classes = sorted(os.listdir(videos_path))
le = preprocessing.LabelEncoder()
le.fit(classes)
vid_classes = le.transform(vid_classes)

train_dataset = VideoDataset(frames_path, video_ids, vid_classes)
train_dataloader = DataLoader(train_dataset, **params)

# X_train, X_val, y_train, y_val = train_test_split(video_id, vid_classes, test_size=0.20, random_state=42)
# train_dataset = VideoDataset(frames_path, X_train, y_train)
# train_dataloader = DataLoader(train_dataset, **params)
# val_dataset = VideoDataset(frames_path, X_val, y_val)
# val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
    model, losses, scores = trainer(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
    print(f"Epoch {epoch} loss:", sum(losses) / len(losses))

    # val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)