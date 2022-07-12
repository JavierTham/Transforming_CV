import pandas as pd
import os
import cv2
import re
import tqdm

videos_path = "/media/kayne/SpareDisk/data/UCF101/videos/"
classes = sorted(os.listdir(videos_path))

def create_df(videos_path, classes):
	df = []
	idx = 0
	for action in classes:
		action_path = os.path.join(videos_path, action)

		for vid in sorted(os.listdir(action_path)):
			vid_path = os.path.join(action_path, vid)

			video_reader = cv2.VideoCapture(vid_path)
			fps = int(video_reader.get(cv2.CAP_PROP_FPS))
			total_frame_count = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
			group_id = action + re.findall("g\d+", vid)[0]

			df.append((idx, group_id, vid_path, action, fps, total_frame_count))
			idx += 1

	df = pd.DataFrame(df, columns=["idx", "group_id", "path_to_vid", "action", "fps", "total_frame_count"])
	print(df.iloc[0,:])
	df.to_csv("../data/UCF_df2.csv", index=False)

if __name__ == "__main__":
	create_df(videos_path, classes)
	df = pd.read_csv("../data/UCF_df.csv")
	print(df.describe())