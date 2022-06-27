from ViT import *
from functions import *
from Resnet import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/train"
TEST_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/test"
META_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/meta"
NUM_CLASSES = 100

config = {
    "learning_rate": 1e-03,
    "epochs": 5,
    "batch_size": 256,
    "sequence_len": 50,
    "num_workers": 4,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 1, 
    "drop out:": True
}

# time_now = time.strftime("%D %X")
# wandb.init(project="Transforming_CV", entity="javiertham", config=config, group="resnet", **{"name": time_now})
# wandb.config = config

params = {
	'batch_size': config['batch_size'],
	'shuffle': True,
	'num_workers': config['num_workers'],
	'pin_memory': True
}

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

vit = ViT(image_size=224, patch_size=16, num_classes=100, dim=256, depth=6, heads=8, mlp_dim=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit.parameters(), lr=config['learning_rate'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

num_examples = 100
train_dataset = CIFARDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, **params)
val_dataset = CIFARDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, **params)

for epoch in range(config['epochs']):
	model, losses, scores = trainer(vit, device, train_dataloader, criterion, optimizer, epoch)
	val_loss, val_score = validation(model, device, val_dataloader, criterion, optimizer, epoch)