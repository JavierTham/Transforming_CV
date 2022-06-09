import pandas as pd
import numpy as np
import re
import torch

def test_data_class(video_id, video_class):
	df = pd.read_csv("../data/Charades_v1_train.csv")
	actions = df[df['id'] == video_id]['actions'].item()
	print("Correct class!") if bool(re.search(str(video_class), actions)) \
							else print("Wrong class")

def test_data_shape(frames, seq_len=50, channels=3, height=224, width=224):
	print("Frames shape:\n[seq_len, channels, height, width]\n", frames.shape)
	if frames.shape != (seq_len, channels, height, width):
		print("WRONG SHAPE!")
	print("Mean:", torch.mean(frames, 3))
	print("Std:", torch.std(frames, 3))