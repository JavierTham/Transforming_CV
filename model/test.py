import pandas as pd
import numpy as np
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
for i, data in enumerate(dataset):
	print("Size/shape of frames:", data[0]['arr_0'].shape) # each compressed .npz file only has 1 "arr_0.npy" file
	print("Class:", data[1])
	break
