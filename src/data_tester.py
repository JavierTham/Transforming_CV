import pandas as pd
import numpy as np
import re
import os
import torch
import matplotlib.pyplot as plt

frames_path = "/media/kayne/SpareDisk/data/video_frames/"

def test_data_class(video_id, video_class):
    df = pd.read_csv("../data/train_data2.csv")
    action = df.iloc[video_id, -1]
    print(f"Correct class! {video_class}") if action == video_class\
                            else print(f"Wrong class, {action} not {video_class}")

def test_data_shape(video_id, seq_len=50, channels=3, height=224, width=224):   
    frames_path = "/media/kayne/SpareDisk/data/video_frames/"
    path = os.path.join(frames_path, f"{video_id}.npz")
    frames = np.load(path)[f"{video_id}.npy"]

    # print("Frames shape:\n[seq_len, channels, height, width]\n", frames.shape)
    if frames.shape != (seq_len, channels, height, width):
        return (video_id, frames.shape)

    return None

def check_data(video_id):
    df = pd.read_csv("../data/train_data2.csv")
    print(df.iloc[video_id, :])

    path = os.path.join(frames_path, f"{video_id}.npz")
    frames = np.load(path)[f"{video_id}.npy"]
    
    for frame in frames:
        plt.imshow(np.array(frame).transpose(1,2,0))
        plt.show()

if __name__ == "__main__":
    num_examples = 16000

    wrong_data = []    
    for video_id in range(num_examples):
        wrong_data.append(test_data_shape(video_id))

    print(wrong_data)