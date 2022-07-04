import torch
from torchvision import transforms
from torch.utils.data import Dataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ImagenetDataset(Dataset):
    def __init__(self, X, y):
        super(ImagenetDataset, self).__init__()
        self.X = X.reshape(len(X), 3, 32, 32) 
        self.y = y
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float)
        X = self.preprocess(X)
        y = torch.tensor(self.y[idx])
        return X, y
    
    def __len__(self):
        return len(self.X)