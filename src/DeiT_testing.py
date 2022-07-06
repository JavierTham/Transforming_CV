import torch
from torch.utils.data import DataLoader

import timm
from CIFARDataset import CIFARDataset

from config.DeiT_config import *
from ImagenetDataset import *
from functions import *

TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"
STATE_DICT_PATH = "states/DeiT_epoch20.pth"
NUM_CLASSES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### model ###
deit = timm.create_model(
    'deit_tiny_patch16_224',
    num_classes=NUM_CLASSES,
    checkpoint_path=STATE_DICT_PATH)
deit.eval()
deit.to(device)
###

### data ###
test_data = unpickle(TEST_DATA_PATH)
X_test = test_data["data"]
y_test = test_data["fine_labels"]

# --- Imagenet --- #
"""
# label from data and label used by pretrained model are different
mapping = pd.read_csv("mapping.csv")
y = [mapping.iloc[i - 1, 2] for i in y_test]

test_dataset = ImagenetDataset(X_test, y)
test_dataloader = DataLoader(test_dataset, **params)
"""
# ---------------- #

# --- CIFAR100 --- #
test_dataset = CIFARDataset(X_test, y_test, size=224)
test_dataloader = DataLoader(test_dataset, **params)
# ---------------- #

output = predict(deit, device, test_dataloader)
torch.save(output[0], "../output/DeiT/pictures.pt")
torch.save(output[1], "../output/DeiT/all_y_true.pt")
torch.save(output[2], "../output/DeiT/all_y_pred.pt")
torch.save(output[3], "../output/DeiT/y_true_examples.pt")
torch.save(output[4], "../output/DeiT/y_pred_examples.pt")