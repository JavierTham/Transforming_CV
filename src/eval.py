import torch
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from functions import inv_normalize, unpickle

META_DATA_PATH = "/media/kayne/SpareDisk/data/cifar100/meta"

def get_accuracy(model_name):
    all_y_true_path = f"../output/{model_name}/all_y_true.pt"
    all_y_pred_path = f"../output/{model_name}/all_y_pred.pt"
    picture_path = f"../output/{model_name}/pictures.pt"
    y_pred_examples_path = f"../output/{model_name}/y_pred_examples.pt"
    y_true_examples_path = f"../output/{model_name}/y_true_examples.pt"

    all_y_true = torch.load(all_y_true_path)
    all_y_pred = torch.load(all_y_pred_path)
    pics = torch.load(picture_path)
    y_pred_examples = torch.load(y_pred_examples_path)
    y_true_examples = torch.load(y_true_examples_path)

    print(y_pred_examples)
    print(y_true_examples)

    score = accuracy_score(all_y_true, all_y_pred)
    return score
    

    # labels = unpickle(META_DATA_PATH)["fine_label_names"]
    
    # for i in range(len(pics)):
    #     pic = inv_normalize(pics[i])
    #     print(y_true_examples[i])
    #     print("y_true:\n", labels[y_true_examples[i]])
    #     print("y_pred:\n", labels[y_pred_examples[i]])
    #     plt.imshow(pic.permute(1,2,0))
    #     plt.show()

# with open("/media/kayne/SpareDisk/data/imagenet/data_labels.txt", "r") as f:
#     data_labels = [s.split(" ") for s in f.readlines()]

# with open("/media/kayne/SpareDisk/data/imagenet/actual_labels.txt", "r") as f:
#     actual_labels = [s.split(" ", 1)[1] for s in f.readlines()]

# mapping = pd.read_csv("mapping.csv")

if __name__ == "__main__":
    model = "ResNet"
    wandb.init(project="Transforming_CV", entity="javiertham", group="cifar", **{"name": f"{model}"})
    score = get_accuracy(model)
    wandb.log({"Test Accuracy": score})
    print("Accuracy score:", score)
    