import pandas as pd
import numpy as np
import os
import torch
from CNN_LSTM import *
import matplotlib.pyplot as plt

train_data_path = "../data/train_data.csv"
video_path = "../data/Charades_v1"
frames_path = "/media/kayne/SpareDisk/data/video_frames/"

df = pd.read_csv(train_data_path).iloc[:1, :]
frames_paths = list(df['id'].apply(lambda x: os.path.join(frames_path, f"{x}.pt")))
vid_classes = np.asarray(df['vid_class'])

dataset = VideoDataset(frames_paths, vid_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(frames_paths[0])
for data in dataset:
	X, y = data[0], data[1]
	print("Size/shape of frames:", X.shape)
	print("Class:", y)
	print("X type:", type(X))
	print("y type:", type(y))

	plt.imshow(np.array(X[0,:,:,:]).transpose(1,2,0))
	plt.show()

	X, y = X.to(device), y.to(device)
	print("X device:", X.device)
	print("y device:", y.device)