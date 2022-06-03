import pandas as pd
import os
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    
    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        N_count += X.size(0)
        
        optimizer.zero_grad()
        output = model(X)
        
        loss = criterion(output, y)
        losses.append(loss.item())
        
        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)
        
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Train Epoch: {epoch + 1} [{N_count}/{len(train_loader)} ({100. * (batch_idx + 1) / len(train_loader)})]\
                \tLoss: {loss.item():.6f}, Accu: {100 * step_score:.2f}')
        
        # wandb.log({"loss": loss, "score": step_score})

        ### FOR TESTING PURPOSES ###
        print("prediction:", y_pred)
        print("actual class:", y)
        
    return losses, scores

def validation(model, device, test_loader, criterion, optimizer, epoch):
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            output = model(X)

            loss = criterion(output, y)
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
    print(f'\nValidation set ({len(all_y)} samples): Average loss: {test_loss:.4f}, Accuracy: {100 * test_score:.2f}')

    # save Pytorch models of best record
    torch.save(model.state_dict(), f'cnnlstm_epoch{epoch + 1}.pth')
    torch.save(optimizer.state_dict(), f'optimizer_epoch{epoch + 1}.pth')      # save optimizer
    print(f"Epoch {epoch + 1} model saved!")

    # wandb.log({"Accuracy": 100 * test_score})

    return test_loss, test_score

def predict(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred