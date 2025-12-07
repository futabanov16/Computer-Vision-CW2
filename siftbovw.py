import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

class SiftBoVW:
    def __init__(self, n_words=100, max_sample = 100000, batch_size = 1000, random_state=0):
        """
        Initialize the SIFTBoVW class.

        Parameters
        ----------
        n_words : int, optional
            The number of visual words to use. Default is 100.
        max_sample : int, optional
            The maximum number of descriptors to sample. Default is 100000.
        batch_size : int, optional
            The batch size for MiniBatchKMeans. Default is 1000.
        random_state : int, optional
            The random seed for MiniBatchKMeans. Default is 0.
        """
        self._n_words = n_words
        self._max_sample = max_sample
        self._batch_size = batch_size # batch size for MiniBatchKMeans on fitting the codebook
        self._random_state = random_state

        self.sift = cv2.SIFT_create()
        self.kmeans = None


    @property
    def n_words(self):
        return self._n_words

    @property
    def max_sample(self):
        return self._max_sample

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def random_state(self):
        return self._random_state

    def _extract_sift_single_img(self, img: np.ndarray) -> np.ndarray:
        """
        Extract SIFT descriptors from a single image.

        Parameters
        ----------
        img : numpy.ndarray
            A grayscale image (numpy array).

        Returns
        -------
        numpy.ndarray
            A numpy array containing the extracted SIFT descriptors.
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        keypoints, desc = self.sift.detectAndCompute(img, None)
        
        if desc is None:
            return np.zeros((0, 128), dtype=np.float32)

        return desc.astype(np.float32)
    
    def _extract_sift(self, imgs: np.ndarray) -> list[np.ndarray]:
        """
        Extract SIFT descriptors from a list of images.

        Parameters
        ----------
        imgs : list
            A list of images (numpy arrays).

        Returns
        -------
        list
            A list of SIFT descriptors for each image.
        """
        desc_list = []
        for _, img in enumerate(imgs):
            desc = self._extract_sift_single_img(img)
            desc_list.append(desc)
        
        return desc_list
    
    def _build_histogram(self, desc):
        K = self._n_words

        if desc is None or desc.shape[0] == 0:
            hist = np.zeros((1,K), dtype=np.float32)
        else:
            word_indices = self.kmeans.predict(desc)
            hist, _ = np.histogram(word_indices, bins=np.arange(K+1))
            hist = hist.reshape(1,-1).astype(np.float32)

        hist = normalize(hist, norm = 'l2')

        return hist.squeeze(0) # (K,)
    
    def fit(self, imgs):
        # 1) extract SIFT descriptors for all images
        desc_list = self._extract_sift(imgs)

        # 2) stack descriptors
        all_desc = np.vstack([d for d in desc_list if d.shape[0]> 0])
        print(f"[SIFTBoVW] Total descriptors before sampling: {all_desc.shape[0]}")
        
        # 3) optional sampling
        if all_desc.shape[0] > self._max_sample:
            sample_indices = np.random.choice(all_desc.shape[0], size=self._max_sample, replace=False)
            sample_desc = all_desc[sample_indices]
            print(f"[SIFTBoVW] Subsampled to {sample_desc.shape[0]} descriptors")

        else:
            sample_desc = all_desc

        # 4) k-means clustering
        self.kmeans = MiniBatchKMeans(n_clusters=self._n_words, batch_size=self._batch_size, random_state=self._random_state, verbose=1)
        self.kmeans.fit(sample_desc)

        center = self.kmeans.cluster_centers_
        labels = self.kmeans.predict(sample_desc)

        # print(center.shape)  # (n_words, 128
        # print(labels.shape)  # (n_samples,)
        # counts = np.bincount(labels, minlength=self._n_words)
        # print(f"Visual word counts: {counts}")
        # print(counts.min(), counts.max(), counts.mean())
        return self
    
    def transform(self, imgs):
        if self.kmeans is None:
            raise Exception("SIFTBoVW is not fitted yet!")
        
        # 1) extract SIFT descriptors for all images
        desc_list = self._extract_sift(imgs)

        X = []
        for desc in desc_list:
            h = self._build_histogram(desc)
            X.append(h)
        X = np.vstack(X)  # (N, n_words)
        return X
    
    def fit_transform(self, imgs):
        self.fit(imgs)
        return self.transform(imgs)
            
if __name__ == '__main__':
    siftbovw = SiftBoVW()
