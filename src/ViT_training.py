# from ViT import *
from functions import *
from config import *
from Resnet import *
from CIFARDataset import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

from pytorch_pretrained_vit import ViT

TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/train"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"
META_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/meta"
NUM_CLASSES = 100

time_now = time.strftime("%D %X")
wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="cifar", **{"name": "ViT_" + time_now})
wandb.config = config

train_data = unpickle(TRAIN_DATA_PATH)
test_data = unpickle(TEST_DATA_PATH)
meta_data = unpickle(META_DATA_PATH)

fine_labels = meta_data["fine_label_names"]
coarse_labels = meta_data["coarse_label_names"] 

X_train = train_data["data"]
X_test = test_data["data"]
y_train = train_data["fine_labels"]
y_test = test_data["fine_labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vit = ViT(image_size=224, patch_size=16, num_classes=100, dim=config["dim"], depth=config["depth"], heads=config["heads"], mlp_dim=config["mlp_dim"]).to(device)
vit = ViT('B_16_imagenet1k', pretrained=True)
for param in vit.parameters():
    param.requires_grad = False
n_inputs = vit.fc.in_features
vit.fc = nn.Linear(n_inputs, NUM_CLASSES)
vit.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit.parameters(), lr=config['learning_rate'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

train_dataset = CIFARDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = CIFARDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(vit, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)