import os
import torch
from functions import *
from CNN_LSTM import *
import wandb

config = {
    "learning_rate": 1e-06,
    "epochs": 1,
    "batch_size": 16,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 256, # for benchmarking
    "lstm_num_layers": 1   # for benchmarking
}

# wandb.init(project="Transforming_CV", entity="javiertham", config=config)

data_path = "../data/"
train_data_path = os.path.join(data_path, "train_data.csv")
test_data_path = os.path.join(data_path, "test_data.csv")
frames_path = "/media/kayne/SpareDisk/data/video_frames/"


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
val_criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(cnnlstm.parameters(), lr=config['learning_rate'])

df = pd.read_csv(train_data_path)
frames_paths = list(df['id'].apply(lambda x: os.path.join(frames_path, f"{x}.npz")))
vid_classes = np.asarray(df['vid_class'])

train_X, val_X, train_y, val_y = train_test_split(frames_paths, vid_classes, test_size=0.20, random_state=42)

train_dataset = VideoDataset(train_X, train_y)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = VideoDataset(val_X, val_y)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	losses, scores = train(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
	epoch_test_loss, epoch_test_score = validation(cnnlstm, device, val_dataloader, val_criterion, optimizer, epoch)

	# wandb.log({"epoch_test_loss": epoch_test_loss, "epoch_test_score": epoch_test_score})

# wandb.finish()