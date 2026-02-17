import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

class PHOW:
    """
    Pyramid Histogram of Words (PHOW) - Dense SIFT with Spatial Pooling
    Implements Bag of Visual Words with spatial pyramid pooling
    """
    def __init__(self, n_words=250, max_sample=200000, batch_size=100, 
                 pyramid_levels=[0, 1, 2], random_state=0):
        """
        Parameters
        ----------
        n_words : int
            Visual vocabulary size
        max_sample : int
            Maximum number of descriptors to sample
        batch_size : int
            Batch size for MiniBatchKMeans
        pyramid_levels : list
            Pyramid levels, e.g., [0, 1, 2] represents 1x1, 2x2, 4x4 grids
        random_state : int
            Random seed
        """
        self._n_words = n_words
        self._max_sample = max_sample
        self._batch_size = batch_size
        self._pyramid_levels = pyramid_levels
        self._random_state = random_state
        
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        
        # Calculate feature dimension
        self._feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """Calculate final feature dimension"""
        total_dim = 0
        for level in self._pyramid_levels:
            n_cells = (2 ** level) ** 2  # level 0: 1, level 1: 4, level 2: 16
            total_dim += n_cells * self._n_words
        return total_dim
    
    @property
    def feature_dim(self):
        return self._feature_dim
    
    def _extract_dense_sift_single_img(self, img: np.ndarray) -> np.ndarray:
        """
        Extract Dense SIFT descriptors from a single image
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Dense SIFT: Use grid sampling with fixed step size
        step_size = 8  # Sample every 8 pixels
        keypoints = [cv2.KeyPoint(x, y, step_size) 
                    for y in range(0, img.shape[0], step_size) 
                    for x in range(0, img.shape[1], step_size)]
        
        keypoints, desc = self.sift.compute(img, keypoints)
        
        if desc is None:
            return np.zeros((0, 128), dtype=np.float32), []
        
        return desc.astype(np.float32), keypoints
    
    def _extract_pyramid_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Extract histogram features from spatial pyramid
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            Concatenated spatial pyramid feature vector
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        h, w = img.shape[:2]
        all_histograms = []
        
        for level in self._pyramid_levels:
            # Number of grid cells in this level
            n_cells = 2 ** level  # level 0: 1, level 1: 2, level 2: 4
            cell_h = h // n_cells
            cell_w = w // n_cells
            
            level_histograms = []
            
            # Iterate through each grid cell
            for i in range(n_cells):
                for j in range(n_cells):
                    # Calculate boundaries of current grid
                    y_start = i * cell_h
                    y_end = (i + 1) * cell_h if i < n_cells - 1 else h
                    x_start = j * cell_w
                    x_end = (j + 1) * cell_w if j < n_cells - 1 else w
                    
                    # Extract grid region
                    cell = img[y_start:y_end, x_start:x_end]
                    
                    # Extract SIFT within this grid
                    desc, kps = self._extract_dense_sift_single_img(cell)
                    
                    # Build histogram
                    if desc.shape[0] == 0:
                        hist = np.zeros(self._n_words, dtype=np.float32)
                    else:
                        word_indices = self.kmeans.predict(desc)
                        hist, _ = np.histogram(word_indices, bins=np.arange(self._n_words + 1))
                        hist = hist.astype(np.float32)
                    
                    # L2 normalization
                    hist = normalize(hist.reshape(1, -1), norm='l2').squeeze(0)
                    level_histograms.append(hist)
            
            # Weight by pyramid level (optional)
            # level 0 weight: 1/(2^0) = 1
            # level 1 weight: 1/(2^1) = 0.5
            # level 2 weight: 1/(2^2) = 0.25
            weight = 1.0 / (2 ** level)
            level_histograms = [h * weight for h in level_histograms]
            
            all_histograms.extend(level_histograms)
        
        # Concatenate histograms from all levels
        feature = np.concatenate(all_histograms)
        
        # Overall L2 normalization
        feature = normalize(feature.reshape(1, -1), norm='l2').squeeze(0)
        
        return feature
    
    def _extract_all_descriptors(self, imgs):
        """Extract SIFT descriptors from all images for training codebook"""
        all_desc = []
        for img in imgs:
            desc, _ = self._extract_dense_sift_single_img(img)
            if desc.shape[0] > 0:
                all_desc.append(desc)
        
        if len(all_desc) == 0:
            raise ValueError("No descriptors extracted!")
        
        return np.vstack(all_desc)
    
    def fit(self, imgs):
        """
        Train visual vocabulary (codebook)
        """
        print(f"[PHOW] Extracting Dense SIFT descriptors...")
        all_desc = self._extract_all_descriptors(imgs)
        print(f"[PHOW] Total descriptors: {all_desc.shape[0]}")
        

        # Fix the random seed
        rng = np.random.default_rng(self._random_state)
        # Sampling
        if all_desc.shape[0] > self._max_sample:
            sample_indices = rng.choice(all_desc.shape[0], 
                                            size=self._max_sample, 
                                            replace=False)
            sample_desc = all_desc[sample_indices]
            print(f"[PHOW] Sampled {sample_desc.shape[0]} descriptors")
        else:
            sample_desc = all_desc
        
        # K-Means clustering
        print(f"[PHOW] Training K-Means (n_clusters={self._n_words})...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self._n_words,
            batch_size=self._batch_size,
            random_state=self._random_state,
            verbose=0
        )
        self.kmeans.fit(sample_desc)
        
        print(f"[PHOW] Training complete! Feature dimension: {self._feature_dim}")
        return self
    
    def transform(self, imgs):
        """
        Transform images to PHOW features
        """
        if self.kmeans is None:
            raise Exception("PHOW not trained yet, please call fit() first!")
        
        X = []
        for i, img in enumerate(imgs):
            feature = self._extract_pyramid_histogram(img)
            X.append(feature)
            
            if (i + 1) % 100 == 0:
                print(f"[PHOW] Processed {i + 1}/{len(imgs)} images")
        
        return np.vstack(X)
    
    def fit_transform(self, imgs):
        """
        Train and transform
        """
        self.fit(imgs)
        return self.transform(imgs)


if __name__ == '__main__':
    # Test
    phow = PHOW(n_words=250, pyramid_levels=[0, 1, 2])
    print(f"Feature dimension: {phow.feature_dim}")

