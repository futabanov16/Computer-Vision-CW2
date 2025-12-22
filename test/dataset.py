import os
import cv2
import numpy as np
from typing import Tuple

class Dataset():
    def __init__(self, root:str, img_size: Tuple[int, int] = (32, 32)):
        """
        Initialize the Dataset.

        Parameters
        ----------
        root : str
            Path to the root directory. Each subdirectory under `root`
            is treated as one class and should contain `.jpg` images.
        img_size : Tuple[int, int], optional
            Target image size (width, height) to resize each image to.
            Default is (32, 32).
        """
        self._imgs, self._labels = self._get_img_class(root, img_size)

    
    def _get_img_class(self, root:str, img_size: Tuple[int, int] = (32, 32)) -> Tuple[np.ndarray, np.ndarray]:
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
                img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img      = cv2.resize(img, img_size)


                imgs.append(img)
                labels.append(name)

        return np.stack(imgs), np.array(labels)

    def get_data(self):
        return self._imgs, self._labels
    
    def shuffle_data(self, random_state: int = 0):
        """
        Shuffle the dataset (images and labels) in unison.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility. Default is 0.
        """
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(self._imgs))
        self._imgs = self._imgs[perm]
        self._labels = self._labels[perm]

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of images loaded.
        """
        return len(self.imgs)
    
    def __getitem__(self, idx: int):
        """
        Get the (image, label) pair at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve. Must be in the range
            [0, len(self) - 1].

        Returns
        -------
        image : numpy.ndarray
            Grayscale image array of shape (H, W).
        label : str
            Class label (subdirectory name) corresponding to this image.
        """
        return self._imgs[idx], self._labels[idx]