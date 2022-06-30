import torch
from CIFARDataset import *
from functions import *
from Resnet import *
from config import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

NUM_CLASSES = 100
STATE_DICT_PATH = "states/ResNet_model_epoch5.pth"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### model ###
resnet = Resnet(NUM_CLASSES)
resnet.load_state_dict(torch.load(STATE_DICT_PATH))
resnet.to(device)
resnet.eval()
###

### data ###
test_data = unpickle(TEST_DATA_PATH)
X_test = test_data["data"]
y_test = test_data["fine_labels"]

test_dataset = CIFARDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, **params)
###

criterion = nn.CrossEntropyLoss()
output = predict(resnet, device, test_dataloader)
torch.save(output[0], "../output/ResNet/pictures.pt")
torch.save(output[1], "../output/ResNet/all_y_true.pt")
torch.save(output[2], "../output/ResNet/all_y_pred.pt")
torch.save(output[3], "../output/ResNet/y_true_examples.pt")
torch.save(output[4], "../output/ResNet/y_pred_examples.pt")