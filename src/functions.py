import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
import pickle
import wandb

def trainer(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    
    losses = []
    scores = []
    top_5 = []

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

        # top_5_score = top_k_accuracy_score(y_true, output.detach().cpu(), k=5, labels=range(157))
        # top_5.append(top_5_score)
        
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx} loss:", loss.item(), "\ntop 1 accuracy:", step_score) #, "\ntop 5 accuracy:", top_5_score)
        
        if (batch_idx + 1) % 10 == 0:
            wandb.log({"Batch": batch_idx + 1, "Training loss": loss.item(),
                        "Training top 1 accuracy": step_score})#, "Training top 5 accuracy": top_5_score})

    return model, losses, scores

def validation(model, device, test_loader, criterion, optimizer, epoch, save=True):
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            X, y = data
            X, y = X.to(device), y.to(device)

            output = model(X)

            loss = criterion(output, y)
            test_loss += loss.item()                 
            y_pred = torch.max(output, 1)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

            print(f"Validation batch {batch_idx} loss:", loss.item())

            if (batch_idx + 1) % 10 == 0:
                wandb.log({"Batch": batch_idx + 1, "Validation loss": loss.item()})

    # average loss per batch
    test_loss /= len(test_loader)

    # compute accuracy
    all_y_true = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    
    all_y_true = all_y_true.cpu().data.squeeze().numpy()
    all_y_pred = all_y_pred.cpu().data.squeeze().numpy()
    
    test_score = accuracy_score(all_y_true, all_y_pred)
    # top_5_score = top_k_accuracy_score(all_y_true, output.detach().cpu(), k=5, labels=range(157))

    print(f'\nValidation set ({len(all_y)} samples): Average loss: {test_loss:.4f}, Accuracy: {100 * test_score:.2f}')

    if save:
        torch.save(model.state_dict(), f'states/model_epoch{epoch + 1}.pth')
        torch.save(optimizer.state_dict(), f'states/optimizer_epoch{epoch + 1}.pth')      # save optimizer
        print(f"Epoch {epoch + 1} model saved!")

    wandb.log({"Epoch": epoch, "Validation top 1 Accuracy": test_score})#, "Validation top 5 Accuracy": top_5_score})

    return test_loss, test_score

def predict(model, device, loader, show_pic=True):
    model.eval()

    X_examples = []
    y_pred_examples = []
    y_true_examples = []
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y.extend(y)
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())
            
            if show_pic:
                # show first example from each batch
                X_examples.append(X[0].cpu())
                y_pred_examples.append(y_pred[0].cpu())
                y_true_examples.append(y[0])

        all_y_true = torch.stack(all_y, dim=0)

    return X_examples, all_y_true, all_y_pred, y_true_examples, y_pred_examples

def unpickle(file, mode='rb', encoding="latin1"):
    with open(file, mode) as fo:
        d = pickle.load(fo, encoding=encoding)
    return d

def inv_normalize(img):
    inv = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    return inv(img)