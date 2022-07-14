import timm

from functions import *
from config.Swin_config import *
from CIFARDataset import *
from torch.utils.data import DataLoader
import torch.nn as nn

NUM_CLASSES = 100
STATE_DICT_PATH = "states/Swin_tiny_epoch5.pth"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### model ###
vit = timm.create_model(
    'swin_tiny_patch4_window7_224',
    num_classes=NUM_CLASSES,
    checkpoint_path=STATE_DICT_PATH)
vit.eval()
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
torch.save(output[0], "../output/Swin/pictures.pt")
torch.save(output[1], "../output/Swin/all_y_true.pt")
torch.save(output[2], "../output/Swin/all_y_pred.pt")
torch.save(output[3], "../output/Swin/y_true_examples.pt")
torch.save(output[4], "../output/Swin/y_pred_examples.pt")