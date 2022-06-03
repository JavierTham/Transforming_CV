import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import torch

import matplotlib.pyplot as plt

def process_video(video_id, video_path, seq_len=50):
        frames_list = []
        
        video_path = os.path.join(video_path, f'{video_id}.mp4')
        video_reader = cv2.VideoCapture(video_path)
        
        # Get the total number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count / seq_len), 1)

        for frame_counter in range(seq_len):
            # Set the current frame position of the video, loop video if video too short
            frame_position = frame_counter * skip_frames_window % video_frames_count 
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            success, frame = video_reader.read() 

            if not success:
                break

            img = Image.fromarray(frame)
            img = img.resize((256, 256))
            img = img.crop((16, 16, 240, 240)) # center crop 224x224
            frame = np.array(img)
            frame[:,:,0] = (frame[:,:,0] - 0.485) / 0.229 # normalize mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            frame[:,:,1] = (frame[:,:,1] - 0.456) / 0.224
            frame[:,:,2] = (frame[:,:,2] - 0.406) / 0.225

            frames_list.append(frame)

        video_reader.release()
            
        np.savez_compressed(f'/media/kayne/SpareDisk/data/video_frames/{video_id}.npz', frames_list)

train_data_path = "../data/train_data.csv"
video_path = "../data/Charades_v1"

df = pd.read_csv(train_data_path)
vid_id_list = df['id']

# for vid_id in vid_id_list:
#     print(vid_id)
#     process_video(vid_id, video_path)