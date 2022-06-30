# from ViT import *
from pytorch_pretrained_vit import ViT
from functions import *
from config import *
from CIFARDataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

NUM_CLASSES = 100
STATE_DICT_PATH = "states/ViTmodel_epoch20.pth"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### model ###
vit = ViT('B_16_imagenet1k', pretrained=True)
for param in vit.parameters():
    param.requires_grad = False
n_inputs = vit.fc.in_features
vit.fc = nn.Linear(n_inputs, NUM_CLASSES)

state_dict = torch.load(STATE_DICT_PATH, map_location="cpu")
vit.load_state_dict(state_dict)
vit.to(device)
###

### data ###
test_data = unpickle(TEST_DATA_PATH)
X_test = test_data["data"]
y_test = test_data["fine_labels"]

test_dataset = CIFARDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, **params)
###

criterion = nn.CrossEntropyLoss()
output = predict(vit, device, test_dataloader)
torch.save(output[0], "../output/ViT/pictures.pt")
torch.save(output[1], "../output/ViT/all_y_true.pt")
torch.save(output[2], "../output/ViT/all_y_pred.pt")
torch.save(output[3], "../output/ViT/y_true_examples.pt")
torch.save(output[4], "../output/ViT/y_pred_examples.pt")