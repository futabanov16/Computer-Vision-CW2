import os
import cv2
import numpy as np
from typing import Tuple

class Dataset():
    def __init__(self, root:str, img_size: Tuple[int, int] = (32, 32), label_exist: bool = True):
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
        label_exist : bool, optional
            If False, output of self._labels will be the image filenames
            instead of subdirectory names (class labels). Default is True. 
        """
        self._label_exist = label_exist
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
            # get the subdirectory path
            sub_path = os.path.join(root, name)

            # if the sub_path is not a directory, check if it's an image file
            # if it's not an image file, skip it
            if not os.path.isdir(sub_path) and not sub_path.endswith('.jpg'):
                continue
            
            # else check if self._label_exist is False, then read the image
            elif not os.path.isdir(sub_path) and self._label_exist is False:
                img_path = sub_path
                img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img      = cv2.resize(img, img_size)

                imgs.append(img)
                labels.append(name)

            # the sub_path is a directory
            else:
                # iterate through all image files in the subdirectory
                for img_name in os.listdir(sub_path):
                    if not img_name.endswith('.jpg'):
                        continue

                    img_path = os.path.join(sub_path, img_name)
                    img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img      = cv2.resize(img, img_size)


                    imgs.append(img)
                    labels.append(name)


        if self._label_exist is False:
            paired = list(zip(imgs, labels))
            paired.sort(key=lambda x: int(x[1].split('.')[0]))  # sort by filename
            
            # unzip the sorted pairs
            imgs_sorted, labels_sorted = zip(*paired)

            return np.stack(imgs_sorted), np.array(labels_sorted)
        
        else:    
            return np.stack(imgs), np.array(labels)

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
    
    @property
    def label_exist(self) -> bool:
        return self._label_exist
    
    def runtxt_write(self, predicts: np.ndarray, filepath: str = None):
        """
        Write the dataset information to a text file.

        Each line in the file contains the image name and its predicted label,
        separated by a space.

        Parameters
        ----------
        predicts : numpy.ndarray
            Array of predicted labels corresponding to the images.
        filepath : str, optional
            Path to the output text file. If None, defaults to 'runtxt.txt'.
        """

        if self._label_exist:
            raise ValueError("self._label_exist is True. Cannot write runtxt for labeled data.")
        if filepath is None:
            filepath = 'runtxt.txt'
        
        with open(filepath, 'w') as f:
            for img_name, pred in zip(self._labels, predicts):
                f.write(f"{img_name} {pred}\n")