import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import json
import torch
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt

frames_path = "/media/kayne/SpareDisk/data/UCF101/video_frames/"

def process_video(idx, vid_path, fps, total_frame_count, seq_len=50):
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        video_reader = cv2.VideoCapture(vid_path)

        # sample frames with skipping
        required_frame_length = seq_len * 2

        frames_list = []
        skip_frames_window = 2

        # start from the front
        starting_frame = 0
        # sample from middle if duration is long enough
        if required_frame_length <= total_frame_count:
            offset = (total_frame_count - required_frame_length) / 2
            starting_frame += offset

        # some videos do not have constant fps, might have less frames than (duration*fps)
        max_frame_counter = 1
        num_frames = 0
        frame_counter = 0
        while num_frames < seq_len:
            # Set the current frame position of the video,
            frame_position = (frame_counter / max_frame_counter * skip_frames_window) % total_frame_count + starting_frame
            print(frame_position)

            # start from beginning if video too short
            if frame_position >= total_frame_count:
                max_frame_counter = frame_counter
                frame_position = starting_frame

            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            success, frame = video_reader.read()

            # if bad frame, restart from front
            if not success:
                print("NOT SUCCESS:", frame_position)
                frame_counter = 0
                continue

            num_frames += 1
            frame_counter += 1

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
        print(frames_list.shape)

        save_path = os.path.join(frames_path, f"{idx}.npz")
        np.savez_compressed(save_path, **{str(idx): frames_list})

if __name__ == "__main__":
    df = pd.read_csv("../data/UCF_df.csv")
    # df.apply(lambda x: process_video(x["idx"], x["path_to_vid"], x["fps"], x["total_frame_count"]), axis=1)

    ### FOR MESSED UP DATA ###
    with open(f"wrong_data_1.json", "r") as f:
        wrong_data = json.load(f)

    for i, frame_shape in tqdm(wrong_data):
        temp_df = df.iloc[[i], :]
        temp_df.apply(lambda x: process_video(x["idx"], x["path_to_vid"], x["fps"], x["total_frame_count"]), axis=1)