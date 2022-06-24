import torch
from torchvision import transforms
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, X_train, y_train):
        super(VideoDataset, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, idx):
        print(self.X_train[idx].shape)
        X = self.preprocess(self.X_train[idx])
        y = self.y_train[idx]
        return torch.tensor(X), torch.tensor(y)
    
    def __len__(self):
        return len(self.X_train)