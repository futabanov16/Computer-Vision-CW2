from torch.utils.data import Dataset
from torchvision import transforms

from typing import Tuple
import os
import numpy as np
import cv2 

class CNN_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5,), (0.5,))])
        self._imgs, self._labels = self._get_img_class(data_dir)
        self.classes = {label.item(): float(idx) for idx, label in enumerate(np.unique(self._labels))}

    def _get_img_class(self, root:str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load grayscale images and labels from subdirectories.

        This method assumes that `root` contains one subdirectory per class.
        Each subdirectory name is used as the class label, and all `.jpg`
        images inside that subdirectory belong to that class.

        Images are:
        - read in grayscale using OpenCV
        - resized to `img_size`
        - stacked into a single NumPy array

        Parameters
        ----------
        root : str
            Path to the root directory containing class subdirectories.
        img_size : Tuple[int, int], optional
            Target image size (width, height) for resizing each image.

        Returns
        -------
        imgs : numpy.ndarray
            Array of shape (N, H, W) containing all grayscale images.
        labels : numpy.ndarray
            Array of shape (N,) containing the corresponding labels
            (subdirectory names as strings).

        Raises
        ------
        FileNotFoundError
            If the `root` directory does not exist.
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"File not found: {root}")
        
        imgs = []
        labels = []
        for name in os.listdir(root):
            sub_path = os.path.join(root, name)

            if not os.path.isdir(sub_path):
                continue
            
            for img_name in os.listdir(sub_path):
                if not img_name.endswith('.jpg'):
                    continue

                img_path = os.path.join(sub_path, img_name)
                img      = np.expand_dims(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), axis=-1)
                img     = self.transform(img)

                imgs.append(img)
                labels.append(name)

        return np.stack(imgs), np.array(labels)    

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        img = self._imgs[idx]
        label = self._labels[idx]

        return img, self.classes[label]
    
if __name__ == "__main__":
    dataset = CNN_Dataset(data_dir='training')
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Image type: {img.dtype}, Label: {label}")
    print(dataset.classes)
    print(f"Max: {img.max()}, Min: {img.min()}")