# PART 3

## 1. Environment setup

Install al requires package via `pip install -r requirements.txt`

## 2. File Structure

## 2.1 dataset.py

Files contain a `Dataset` object which can be initialized by:

```python
class Dataset():
    def __init__(self, root:str, img_size: Tuple[int, int] = (32, 32)):
        ...
```

The `Dataset` object contains method

```python
def get_data(self):
    """
    Return the images and labels
    """

def shuffle_data(self, random_state: int = 0):
    """
    Shuffle the dataset (images and labels) in unison.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    """
def __len__(self):
    """
    Return length of the dataset
    """

def __getitem__(self, idx = 0):
    """
    Get image and the label
    """
```

## 2.2 siftbovw.py

This file contains the object `SiftBoVW` which perform Scale-Invariant Feature Transform and construct Bag of Visual Word, can be initialized by:

```python
class SiftBoVW:
    def __init__(self, n_words=100, max_sample = 100000, batch_size = 1000, random_state=0):
        ...
```

## 3. Example

Below show how to perform classification using a k-nearest neighbours classifier.

```python
from dataset import Dataset
from siftbovw import SiftBoVW
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

N_NEIGHBORS = 20

# Compute dataset
training = Dataset(r"training",img_size=(256, 256))
training.shuffle_data()
imgs, labels = training.get_data()
split_ratio = 0.9
split_index = int(len(imgs) * split_ratio)
imgs_train, labels_train = imgs[:split_index], labels[:split_index]
imgs_test, labels_test = imgs[split_index:], labels[split_index:]

sift_bovw = SiftBoVW(n_words=100, max_sample=100000, batch_size=100, random_state=0)
x_train = sift_bovw.fit_transform(imgs_train)

x_test = sift_bovw.transform(imgs_test)

knn = KNeighborsClassifier(n_neighbors= N_NEIGHBORS)
knn.fit(x_train, labels_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(labels_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
```
