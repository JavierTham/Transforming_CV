import torch
from torchvision import transforms
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