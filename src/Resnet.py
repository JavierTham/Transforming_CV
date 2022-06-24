import torch
from torchvision import models, datasets, transforms
import pickle
import numpy as np
import matplotlib.pyplot as plt

# model = models.resnet50()
# preprocess = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

# imagenet_data = datasets.ImageNet('../data/imagenet_data')
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=args.nThreads)

train_data_path = "/media/kayne/SpareDisk/data/cifar100/train"
test_data_path = "/media/kayne/SpareDisk/data/cifar100/test"
meta_data_path = "/media/kayne/SpareDisk/data/cifar100/meta"

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    return d

train_data = unpickle(train_data_path)
test_data = unpickle(test_data_path)
meta_data = unpickle(meta_data_path)

fine_labels = meta_data["fine_label_names"]
coarse_labels = meta_data["coarse_label_names"]

X_train = train_data["data"]
X_test = train_data["data"]
y_train = test_data["fine_labels"]
y_test = test_data["fine_labels"]
# y_train = train_data["coarse_labels"]
# y_test = test_data["coarse_labels"]

X_train = X_train.reshape(len(X_train), 3, 32, 32)

# print(train_data.keys())
# print(fine_labels[train_data["fine_labels"][120]])
# pic = X_train[120].transpose(1,2,0)
# plt.imshow(pic)
# plt.show()

