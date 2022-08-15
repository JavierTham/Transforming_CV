import os
import numpy as np
from sklearn.model_selection import train_test_split

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

if __name__ == "__main__":
    train_path = os.path.join("..", "data", "cifar100", "train_data")
    test_path = os.path.join("..", "data", "cifar100", "test_data")
    
    train_data = unpickle(train_path)
    test_data = unpickle(test_path)

    X_train, X_val, y_train, y_val = \
        train_test_split(train_data["data"], train_data["fine_labels"], random_state=42)
    X_test, y_test = test_data["data"], test_data["fine_labels"]

    train_dir = os.path.join("..", "data", "cifar100", "train")
    val_dir = os.path.join("..", "data", "cifar100", "validation")
    test_dir = os.path.join("..", "data", "cifar100", "test")

    np.save(os.path.join(train_dir, "X.npy"), X_train)
    np.save(os.path.join(train_dir, "y.npy"), y_train)
    np.save(os.path.join(val_dir, "X.npy"), X_val)
    np.save(os.path.join(val_dir, "y.npy"), y_val)
    np.save(os.path.join(test_dir, "X.npy"), X_test)
    np.save(os.path.join(test_dir, "y.npy"), y_test)