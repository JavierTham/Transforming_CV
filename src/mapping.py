import pandas as pd

with open("/media/kayne/SpareDisk/data/imagenet/actual_labels.txt", "r") as f:
    actual_labels = [s.split(" ", 1) for s in f.readlines()]

actual_labels = pd.DataFrame(actual_labels, columns=["synset", "label"])
actual_labels["idx"] = list(range(1000))

with open("/media/kayne/SpareDisk/data/imagenet/data_labels.txt", "r") as f:
    data_labels = [s.split(" ")[0] for s in f.readlines()]

mapping = pd.DataFrame(data_labels, columns=["synset"])

mapping = mapping.merge(actual_labels, on="synset")
print(mapping)

mapping.to_csv("mapping.csv", index=None)