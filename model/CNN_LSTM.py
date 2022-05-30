import pandas as pd
from functions import *
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
# wandb.init(project="Transforming_CV", entity="javiertham")

# hyperparameters
config = {
    "learning_rate": 0.01,
    "epochs": 1,
    "batch_size": 5,
    "sequence_len": 50,
    "freeze_layers": 26,
    "lstm_hidden_size": 3, # for benchmarking
    "lstm_num_layers": 1   # for benchmarking
}
wandb.config = config

data_path = "../data/"

def train(model, device, train_loader, criterion, optimizer):
    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        X, y = data
        # X, y = X.to(device), y.to(device)
        # model.to(device)
        N_count += X.size(0)
        
        optimizer.zero_grad()
        output = model(X)
        
        loss = criterion(output, y)
        losses.append(loss.item())
        
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)
        
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        
#         wandb.log({"loss": loss})
        
        ### FOR TESTING PURPOSES ###
        print("prediction:", y_pred)
        print("actual class:", y)
        break
        
    return losses, scores

def validation(model, device, optimizer, test_loader):
    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            # X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = criterion(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100 * test_score))

    # save Pytorch models of best record
    # torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

params = {'batch_size': config['batch_size'], 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}

df_train = pd.read_csv(os.path.join(data_path, "cleaned_data.csv"))
df_train = df_train.loc[:, ['id', 'vid_class']]
df_train['id'] = df_train.loc[:, 'id'].apply(lambda x: os.path.join(data_path, "Charades_v1", f"{x}.mp4"))

train_data = VideoDataset(df_train, config['sequence_len'])
# test_data = VideoDataset(df_test)
train_loader = DataLoader(train_data, **params)
# test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

cnnlstm = CNNLSTM(config['batch_size'], config['freeze_layers'], config['lstm_hidden_size'], config['lstm_num_layers'])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnnlstm.parameters(), lr=config['learning_rate'])

for epoch in range(config['epochs']):
    train_losses, train_scores = train(cnnlstm, device, train_loader, criterion, optimizer)

    print("train loss:", train_losses)
    print("train_scores:", train_scores)