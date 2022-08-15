import torch
from torch.utils.data import Dataset

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, X, y, preprocess):
        super(ImageDataset, self).__init__()
        # for CIFAR100 dataset and ImageNet (32x32)
        self.X = X.reshape(len(X), 3, 32, 32)
        self.y = y
        self.preprocess = preprocess

    def __getitem__(self, idx):
        X = self.preprocess(Image.fromarray(self.X[idx].transpose(1,2,0)))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y
    
    def __len__(self):
        return len(self.X)