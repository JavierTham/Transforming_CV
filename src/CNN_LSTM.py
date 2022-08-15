import timm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights

class CNNLSTM(nn.Module):
    """
    Creates a CNN-LSTM model from pretrained backbone
    
    @params
    ---
    lstm_hidden_size: hidden size for the lstm model
    lstm_num_layers: number of layers for the lstm model
    """
    
    def __init__(self, lstm_hidden_size, lstm_num_layers, num_classes):
        super(CNNLSTM, self).__init__()
        # self.cnn = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        # self.AvgPool2d = nn.AvgPool2d(7)
        self.cnn = timm.create_model(
	                    "mobilevitv2_075",
	                    pretrained=True,
	                    num_classes=0)
        # self.cnn = nn.Sequential(*list(cnn.children())[:-1]) # remove last layer

        self.lstm = nn.LSTM(
            384, # change to dim size of output of feature map
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True)
        self.fc1 = nn.Linear(384, 384) # layer between feature extraction and lstm
        self.fc2 = nn.Linear(lstm_hidden_size, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)
        
        # freeze entire CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

        # # initialize weights for lstm
        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #          nn.init.constant_(param, 0.0)
        #     elif 'weight_ih' in name:
        #          nn.init.kaiming_normal_(param)
        #     elif 'weight_hh' in name:
        #          nn.init.orthogonal_(param)
                
    def forward(self, x):
        # batch_size, sequence_length, num_channels, height, width
        B, L, C, H, W = x.size()
        # CNN
        output = []
        for i in range(L):
            #input one frame at a time into the basemodel
            x_t = self.cnn(x[:, i, :, :, :])
            # x_t = self.AvgPool2d(x_t)        # | for mobilenet-v2
                                               # |
            # Flatten the output               # |  
            # x_t = x_t.view(x_t.size(0), -1)  # |

            #make a list of tensors for the given smaples 
            output.append(x_t)

        # reshape to (batch_size, sequence_length, output_size)
        x = torch.stack(output, dim=0).transpose_(0, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # LSTM
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :].view(x.size(0), -1)
        # FC
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        
        return x