import timm
import torch
import torch.nn as nn

from CIFARDataset import *
from functions import *
from config.MobileViTv2_config import *
from torch.utils.data import DataLoader

NUM_CLASSES = 100
STATE_DICT_PATH = "states/MobileViTv2_075_model_epoch9.pth"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### model ###
mobilevit = timm.create_model(
	"mobilevitv2_075",
	pretrained=True,
	num_classes=NUM_CLASSES)
mobilevit.to(device)
mobilevit.eval()
###

### data ###
test_data = unpickle(TEST_DATA_PATH)
X_test = test_data["data"]
y_test = test_data["fine_labels"]

test_dataset = CIFARDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, **params)
###

criterion = nn.CrossEntropyLoss()
output = predict(mobilevit, device, test_dataloader)
torch.save(output[0], "../output/MobileViTv2/pictures.pt")
torch.save(output[1], "../output/MobileViTv2/all_y_true.pt")
torch.save(output[2], "../output/MobileViTv2/all_y_pred.pt")
torch.save(output[3], "../output/MobileViTv2/y_true_examples.pt")
torch.save(output[4], "../output/MobileViTv2/y_pred_examples.pt")