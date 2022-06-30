from pytorch_pretrained_vit import ViT
from functions import *
from CIFARDataset import *
from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# NUM_CLASSES = 100

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vit = ViT('B_16_imagenet1k', pretrained=True)
# n_inputs = vit.fc.in_features
# vit.fc = nn.Linear(n_inputs, NUM_CLASSES)
# vit.to(device)

# x = torch.rand(8, 3, 224, 224).to(device)
# output = vit(x)
# print(output)

TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"
META_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/meta"

# test_data = unpickle(TEST_DATA_PATH)
meta_data = unpickle(META_DATA_PATH)
# X_test = test_data["data"]
# y_test = test_data["fine_labels"]
fine_labels = meta_data["fine_label_names"]

# pic = X_test.reshape(len(X_test), 3, 32, 32)[0]
# label = fine_labels[y_test[0]]

# dataset = CIFARDataset(X_test, y_test)

# for x, y in dataset:
#     print(y)
#     plt.imshow(x.permute(1,2,0))
#     plt.show()
#     break

all_y_pred = torch.load("../output/ResNet/all_y_pred.pt")
all_y_true = torch.load("../output/ResNet/all_y_true.pt")
score = accuracy_score(all_y_true, all_y_pred)
print("accuracy:", score)

pics = torch.load("../output/ResNet/pictures.pt")
y_pred = torch.load("../output/ResNet/y_pred_examples.pt")
y_true = torch.load("../output/ResNet/y_true_examples.pt")

for idx in range(len(pics)):
    print("y_true:", fine_labels[y_true[idx]], "\ny_pred:", fine_labels[y_pred[idx]])

    pic = pics[idx]
    re_normalize_pic = inv_normalize(pic)
    plt.imshow(re_normalize_pic.permute(1,2,0))
    plt.show()

print(pics[0].size())