import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset

class CIFARDataset(Dataset):
    def __init__(self, X_train, y_train):
        super(CIFARDataset, self).__init__()
        self.X_train = X_train.reshape(len(X_train), 3, 32, 32) 
        self.y_train = y_train
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, idx):        
        # scale to [0,1]
        X = torch.tensor(self.X_train[idx] / 255, dtype=torch.float) 
        X = self.preprocess(X)
        y = torch.tensor(self.y_train[idx])
        return X, y
    
    def __len__(self):
        return len(self.X_train)

class Resnet(nn.Module):
    def __init__(self, num_classes):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.num_classes = num_classes
        
        # freeze all layers and change last layer
        self.n_inputs = self.resnet.fc.in_features
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.n_inputs, self.num_classes)

    def forward(self, x):
        return self.resnet(x)