from Resnet import *
from functions import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_data_path = "/media/kayne/SpareDisk/data/cifar100/train"
test_data_path = "/media/kayne/SpareDisk/data/cifar100/test"
meta_data_path = "/media/kayne/SpareDisk/data/cifar100/meta"

train_data = unpickle(train_data_path)
test_data = unpickle(test_data_path)
meta_data = unpickle(meta_data_path)

# print(train_data.keys())

fine_labels = meta_data["fine_label_names"]
coarse_labels = meta_data["coarse_label_names"]

X_train = train_data["data"]
X_test = train_data["data"]
y_train = test_data["fine_labels"]
y_test = test_data["fine_labels"]
y_train = test_data["coarse_labels"]
y_test = test_data["coarse_labels"]

X_train = X_train.reshape(len(X_train), 3, 32, 32)

from torchvision import transforms
import torch
import numpy as np
preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

# print(train_data.keys())
# print(fine_labels[train_data["fine_labels"][120]])

pic = X_train[120] 

print(pic.shape)
plt.imshow(pic.transpose(1,2,0))
plt.show()
pic2 = preprocess(torch.tensor(pic / 255, dtype=torch.float)) # remember to scale pixel values
print(pic2.shape)
plt.imshow(pic2.permute(1,2,0))
plt.show()

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

params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}

# train_dataset = VideoDataset(X_train, y_train)
# train_dataloader = DataLoader(train_dataset, **params)

# for X, y in train_dataset:
#     print(X.shape, y)
#     break

# for epoch in range(config['epochs']):
	# model, losses, scores = trainer(cnnlstm, device, train_dataloader, criterion, optimizer, epoch)
	# val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)