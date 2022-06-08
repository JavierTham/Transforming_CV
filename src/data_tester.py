import pandas as pd
import re

def test_data(video_id, video_class):
	df = pd.read_csv("../data/Charades_v1_train.csv")
	actions = df[df['id'] == video_id]['actions'].item()
	print("Correct class!") if bool(re.search(video_class, actions)) \
							else print("Wrong class")

test_data('46GP8', 'c147')