import numpy as np
import os
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, frames_path, video_ids, video_classes):
        super(VideoDataset, self).__init__()
        self.frames_path = frames_path
        self.video_ids = video_ids
        self.video_classes = video_classes
        
    def __getitem__(self, idx):
        video_class = self.video_classes[idx]
        video_id = self.video_ids[idx]
        path = os.path.join(self.frames_path, f"{video_id}.npz")

        frames = np.load(path)[f"{video_id}.npy"]

        return torch.tensor(frames), torch.tensor(video_class)
    
    def __len__(self):
        return len(self.video_ids)