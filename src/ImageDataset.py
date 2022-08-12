import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, X, y, preprocess):
        super(ImageDataset, self).__init__()
        self.X = X.reshape(len(X), 3, 32, 32) 
        self.y = y
        self.preprocess = preprocess

    def __getitem__(self, idx):
        X = self.preprocess(Image.fromarray(self.X[idx].transpose(1,2,0)))
        y = torch.tensor(self.y[idx])
        return X, y
    
    def __len__(self):
        return len(self.X)