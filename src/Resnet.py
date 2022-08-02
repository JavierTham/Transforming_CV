import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Resnet(nn.Module):
    def __init__(self, num_classes):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        
        self.n_inputs = self.resnet.fc.in_features
        # freeze all layers and change last layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.n_inputs, self.num_classes)

    def forward(self, x):
        return self.resnet(x)