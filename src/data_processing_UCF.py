import pandas as pd
import os
import cv2
import re

from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit, train_test_split

VIDEO_PATH = "/media/kayne/SpareDisk/data/UCF101/videos/"
LABELS = sorted(os.listdir(VIDEO_PATH))

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
	return df

def encode_labels(df, target, labels):
	le = preprocessing.LabelEncoder()
	le.fit(labels)
	encoded_labels = le.transform(target)
	return encoded_labels

def split(df, *, X, y, train_size=None, test_size=None, group_id, random_state=42):
	gss = GroupShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=42)
	train_idx, test_idx = next(gss.split(df.loc[:, X], df.loc[:, y], groups=df[group_id]))
	return df.iloc[train_idx, :], df.iloc[test_idx, :]

if __name__ == "__main__":
	df = create_df(VIDEO_PATH, LABELS)

	encoded_labels = encode_labels(df, df.loc[:, "action"], LABELS)

	groups = df.loc[:, "group_id"].unique()
	encoded_groups = encode_labels(df, df.loc[:, "group_id"], groups)
	
	df.loc[:, "action"] = encoded_labels
	df.loc[:, "group_id"] = encoded_groups
	# df.to_csv("../data/UCF_df.csv", index=False)

	train_df, test_df = split(df, X="idx", y="idx", group_id="group_id", train_size=0.7)
	val_df, test_df = split(test_df, X="idx", y="idx", group_id="group_id", test_size=0.5)
	
	train_df.to_csv("../data/UCF_train.csv", index=False)
	val_df.to_csv("../data/UCF_val.csv", index=False)
	test_df.to_csv("../data/UCF_test.csv", index=False)