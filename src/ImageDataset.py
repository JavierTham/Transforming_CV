import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, X, y, size=224):
        super(ImageDataset, self).__init__()
        self.X = X.reshape(len(X), 3, 32, 32) 
        self.y = y
        # standard image preprocessing for classification
        # change accordingly
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    def __getitem__(self, idx):
        X = self.preprocess(Image.fromarray(self.X[idx].transpose(1,2,0)))
        y = torch.tensor(self.y[idx])
        return X, y
    
    def __len__(self):
        return len(self.X)