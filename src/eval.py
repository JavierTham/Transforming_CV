import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from functions import inv_normalize

all_y_true_path = "../output/DeiT/all_y_true.pt"
all_y_pred_path = "../output/DeiT/all_y_pred.pt"
picture_path = "../output/DeiT/pictures.pt"
y_pred_examples_path = "../output/DeiT/y_pred_examples.pt"
y_true_examples_path = "../output/DeiT/y_true_examples.pt"

all_y_true = torch.load(all_y_true_path)
all_y_pred = torch.load(all_y_pred_path)
pics = torch.load(picture_path)
y_pred_examples = torch.load(y_pred_examples_path)
y_true_examples = torch.load(y_true_examples_path)

score = accuracy_score(all_y_true, all_y_pred)
print("Accuracy score:", score)

with open("/media/kayne/SpareDisk/data/imagenet/data_labels.txt", "r") as f:
    data_labels = [s.split(" ") for s in f.readlines()]

with open("/media/kayne/SpareDisk/data/imagenet/actual_labels.txt", "r") as f:
    actual_labels = [s.split(" ", 1)[1] for s in f.readlines()]

mapping = pd.read_csv("mapping.csv")

for i in range(len(pics)):
    pic = inv_normalize(pics[i])
    print("y_true:\n", actual_labels[y_true_examples[i]])
    print("y_pred:\n", actual_labels[y_pred_examples[i]])
    plt.imshow(pic.permute(1,2,0))
    plt.show()