import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from CNN_LSTM import *

train_data_path = "../data/train_data.csv"
video_path = "../data/Charades_v1"
frames_path = "/media/kayne/SpareDisk/data/video_frames/"

config = {
    "learning_rate": 1e-03,
    "epochs": 100,
    "batch_size": 10,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 256,
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

df = pd.read_csv(train_data_path).iloc[:2,] # test a few, shld overfit quickly
frames_paths = list(df['id'].apply(lambda x: os.path.join(frames_path, f"{x}.pt")))
vid_classes = np.asarray(df['vid_class'])
train_dataset = VideoDataset(frames_paths, vid_classes)
train_dataloader = DataLoader(train_dataset, **params)

def trainer(model, device, train_dataloader, criterion, optimizer, epoch):
    model.train()
    losses = []
    scores = []
    for data in train_dataloader:
        
        X, y = data
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(X)

        loss = criterion(output, y)
        losses.append(loss.item())

        y_pred = torch.max(output, 1)[1]

        y_true = y.cpu().data.squeeze().numpy()
        y_pred = y_pred.cpu().data.squeeze().numpy()

        step_score = accuracy_score(y_true, y_pred)
        scores.append(step_score)
        
        loss.backward()
        optimizer.step()
    return losses, scores

for epoch in range(config['epochs']):
    losses, scores = trainer(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
    print(losses, scores)