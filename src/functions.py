import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score

import pickle
from tqdm import tqdm 

# import wandb

def trainer(
        model,
        device,
        train_loader,
        criterion,
        optimizer,
        epoch):
    """train the model
    
    Args:
        model - model to train
        device - cuda or cpu
        train_loader (torch.utils.data.DataLoader) - DataLoader for training data
        criterion - criterion for loss function
        optimizer - optimizer to use for backprop
    """

    model.train()
    model.to(device)
    
    losses = []
    scores = []

    for batch_idx, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        
        loss = criterion(output, y)
        losses.append(loss.item())
        
        y_pred = torch.max(output, 1)[1]
        y_pred = y_pred.cpu().data.squeeze().numpy()
        y_true = y.cpu().data.squeeze().numpy()

        step_score = accuracy_score(y_true, y_pred)
        scores.append(step_score)
        
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx} loss:", loss.item(), "\ntop 1 accuracy:", step_score)
        
        # log metrics every 10 batches
        # if (batch_idx + 1) % 10 == 0:
        #     wandb.log({"Batch": batch_idx + 1, "Training loss": loss.item(),
        #                 "Training top 1 accuracy": step_score})

    return model, losses, scores

def validation(
        model,
        device,
        val_loader,
        criterion,
        optimizer,
        epoch,
        save=True,
        model_name=""):
    """validate the model
    
    Args:
        model - model to train
        device - cuda or cpu
        val_loader (torch.utils.data.DataLoader) - DataLoader for validation data
        criterion - criterion for loss function
        optimizer - optimizer to use for backprop
        epoch - current epoch (for logging purposes)
        save - Save model state dictionaries
        model_name - Model name used for saving state dict
    """

    model.eval()

    val_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            X, y = data
            X, y = X.to(device), y.to(device)

            output = model(X)

            loss = criterion(output, y)
            val_loss += loss.item()                 
            y_pred = torch.max(output, 1)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

            print(f"Validation batch {batch_idx} loss:", loss.item())

            # if (batch_idx + 1) % 10 == 0:
            #     wandb.log({"Batch": batch_idx + 1, "Validation loss": loss.item()})

    # average loss per batch
    val_loss /= len(val_loader)

    # compute accuracy
    all_y_true = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    
    all_y_true = all_y_true.cpu().data.squeeze().numpy()
    all_y_pred = all_y_pred.cpu().data.squeeze().numpy()
    
    test_score = accuracy_score(all_y_true, all_y_pred)

    print(f'\nValidation set ({len(all_y)} samples): Average Validation loss: {val_loss:.4f}, Accuracy: {100 * test_score:.2f}')

    if save:
        save_dir = os.path.join("states")
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, f"{model_name}_optimizer_epoch{epoch+1}.pt"))
        print(f"Epoch {epoch+1} model saved!")

    # wandb.log({"Epoch": epoch, "Validation top 1 Accuracy": test_score})

    return val_loss, test_score

def predict(
        model,
        device,
        loader):
    model.eval()

    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            X = X.to(device)
            
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y.extend(y)
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

        all_y_true = torch.stack(all_y, dim=0)

    return all_y_true, all_y_pred

def unpickle(file, mode='rb', encoding="latin1"):
    with open(file, mode) as fo:
        d = pickle.load(fo, encoding=encoding)
    return d

def inv_normalize(img):
    """'undo' normalization (standard imagenet preprocessing) of image for viewing"""
    inv = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    return inv(img)

def get_in_features(layers):
    '''
    returns the in_features attribute of the first linear layer
    to change classifier head
    '''
    for layer in layers:
        if isinstance(layer, nn.Linear):
            return layer.in_features
    raise Exception("No in_features found")

def create_torch_model(model_name, weights=None):
    '''Create (pretrained) torchvision models'''
    if weights:
        return eval(f"torchvision.models.{model_name}(weights='{weights}')")
    else:
        return eval(f"torchvision.models.{model_name}()")