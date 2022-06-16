import os
import torch
from functions import *
from CNN_LSTM import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="Training", **{"name": time_now})

### Charades ###
# data_path = "../data/"
# train_data_path = os.path.join(data_path, "train_data2.csv")
# test_data_path = os.path.join(data_path, "test_data2.csv")
# frames_path = "/media/kayne/SpareDisk/data/video_frames/"

### UCF101 ###
train_data_path = "../data/UCF_df.csv"
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

# num_examples = 3000
# df = pd.read_csv(train_data_path)
# video_ids = list(range(num_examples))
# vid_classes = np.asarray(df['vid_class'])

num_examples = 5000
df = pd.read_csv(train_data_path)
video_ids = [random.randrange(0, num_examples) for _ in range(num_examples)]
vid_classes = df.loc[video_ids, "action"]

classes = sorted(os.listdir(videos_path))
le = preprocessing.LabelEncoder()
le.fit(classes)
vid_classes = le.transform(vid_classes)

X_train, X_val, y_train, y_val = train_test_split(video_ids, vid_classes, test_size=0.20, random_state=42)

# train_dataset = VideoDataset(frames_path, video_ids, vid_classes)
# train_dataloader = DataLoader(train_dataset, **params)

train_dataset = VideoDataset(frames_path, X_train, y_train)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = VideoDataset(frames_path, X_val, y_val)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)