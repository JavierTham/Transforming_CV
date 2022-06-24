import time
import torch
from torch.utils.data import DataLoader
from Resnet import *
from functions import *
from sklearn.model_selection import train_test_split
import wandb

TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/train"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"
META_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/meta"

config = {
    "learning_rate": 1e-02,
    "epochs": 10,
    "batch_size": 256,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1, 
    "drop out:": True
}
wandb.config = config
params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="resnet", **{"name": time_now})

train_data = unpickle(TRAIN_DATA_PATH)
test_data = unpickle(TEST_DATA_PATH)
meta_data = unpickle(META_DATA_PATH)

fine_labels = meta_data["fine_label_names"]
coarse_labels = meta_data["coarse_label_names"] 

X_train = train_data["data"]
X_test = test_data["data"]
y_train = train_data["fine_labels"]
y_test = test_data["fine_labels"]
# y_train = test_data["coarse_labels"]
# y_test = test_data["coarse_labels"]

# X_train = X_train.reshape(len(X_train), 3, 32, 32)
# print(train_data.keys())
# print(fine_labels[train_data["fine_labels"][120]])

# pic = X_train[120]

# print(pic.shape)
# plt.imshow(pic.transpose(1,2,0))
# plt.show()
# pic2 = preprocess(torch.tensor(pic / 255, dtype=torch.float)) # remember to scale pixel values
# print(pic2.shape)
# plt.imshow(pic2.permute(1,2,0))
# plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = Resnet(100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=config['learning_rate'])
# optimizer = torch.optim.SGD(resnet.parameters(), lr=config["learning_rate"], momentum=0.9)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

num_examples = 100
train_dataset = CIFARDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = CIFARDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(resnet, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)
