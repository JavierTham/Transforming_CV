import pandas as pd
import numpy as np
import torch
from functions import *

train_data_path = "../data/train_data.csv"
video_path = "../data/Charades_v1"
frames_path = "/media/kayne/SpareDisk/data/video_frames/"

### test if custom Dataset returns output in correct format ###
def get_df(train_data_path, frames_path):
	df = pd.read_csv(train_data_path)
	df = df.loc[:, ["id", "vid_class"]]
	df['id'] = df.loc[:, 'id'].apply(lambda x: os.path.join(frames_path, f"{x}.npz"))
	return df

df = get_df(train_data_path, frames_path)
print(df.head())

dataset = VideoDataset(df)
for data in dataset:
	X, y = data[0], data[1]
	print("Size/shape of frames:", X.shape)
	print("Class:", y)
	break

use_cuda = torch.cuda.is_available()    
device = torch.device("cuda" if use_cuda else "cpu")
for data in dataset:
	X, y = data[0], data[1]
	print("X type:", type(X))
	print("y type:", type(y))
	X, y = X.to(device), y.to(device)
	print("X device:", X.device)
	print("y device:", y.device)
	break