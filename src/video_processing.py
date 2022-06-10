import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import torch
from torchvision import transforms

import matplotlib.pyplot as plt

def process_video(video_id, video_path, start_time, end_time, seq_len=50):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        video_path = os.path.join(video_path, f'{video_id}.mp4')
        video_reader = cv2.VideoCapture(video_path)

        fps = video_reader.get(cv2.CAP_PROP_FPS)

        num_frames = round(end_time - start_time) * fps
        # sample frames with skipping
        required_frame_length = seq_len * 2

        frames_list = []
        skip_frames_window = 2

        starting_frame = start_time * fps
        # sample from middle if duration is long enough
        if required_frame_length <= num_frames:
            offset = (num_frames - required_frame_length) / 2
            starting_frame += offset

        for frame_counter in range(seq_len):
            # Set the current frame position of the video, loop video if video too short
            frame_position = (frame_counter * skip_frames_window) % num_frames + starting_frame
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            success, frame = video_reader.read()

            if not success:
                break

            # print(frame.shape)
            # plt.imshow(frame)
            # plt.show() 

            transformed_frame = preprocess(Image.fromarray(frame))
            transformed_frame = transformed_frame.detach().cpu().numpy()

            # plt.imshow(np.array(transformed_frame).transpose(1,2,0))
            # plt.show()

            frames_list.append(transformed_frame)

        video_reader.release()
        
        frames_list = np.stack(frames_list, axis = 0)

        # print(type(frames_list))
        # print(frames_list.shape)

        print(video_id)
        np.savez_compressed(f"/media/kayne/SpareDisk/data/video_frames/{video_id}.npz", **{video_id: frames_list})

train_data_path = "../data/train_data2.csv"
video_path = "../data/Charades_v1"
frames_path = "/media/kayne/SpareDisk/data/video_frames/"

df = pd.read_csv(train_data_path).iloc[:100, :]

for i in range(len(df)):
    vid_id, start_time, end_time = df.loc[i, ["id", "start_time", "end_time"]]
    process_video(vid_id, video_path, start_time, end_time)