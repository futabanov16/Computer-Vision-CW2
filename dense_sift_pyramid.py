import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class DenseSIFTPyramid:
    """
    Dense SIFT in a Gaussian Pyramid
    Extract Dense SIFT features at multiple scales of Gaussian pyramid
    """
    def __init__(self, n_words=250, max_sample=200000, batch_size=100,
                 n_scales=3, scale_factor=0.7, step_size=8, random_state=0):
        """
        Parameters
        ----------
        n_words : int
            Visual vocabulary size
        max_sample : int
            Maximum number of descriptors to sample
        batch_size : int
            Batch size for MiniBatchKMeans
        n_scales : int
            Number of pyramid scales (including original image)
        scale_factor : float
            Pyramid scale factor (scale ratio of each level relative to previous level)
        step_size : int
            Step size for dense sampling
        random_state : int
            Random seed
        """
        self._n_words = n_words
        self._max_sample = max_sample
        self._batch_size = batch_size
        self._n_scales = n_scales
        self._scale_factor = scale_factor
        self._step_size = step_size
        self._random_state = random_state
        
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        
        # Feature dimension: each scale contributes n_words histogram bins
        self._feature_dim = n_scales * n_words
    
    @property
    def feature_dim(self):
        return self._feature_dim
    
    def _build_gaussian_pyramid(self, img: np.ndarray) -> list:
        """
        Build Gaussian pyramid
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image
            
        Returns
        -------
        list
            Pyramid levels list, from large to small
        """
        pyramid = [img]
        current_img = img.copy()
        
        for i in range(1, self._n_scales):
            # Calculate new size
            new_size = (int(current_img.shape[1] * self._scale_factor),
                       int(current_img.shape[0] * self._scale_factor))
            
            # Gaussian blur first, then downsample
            blurred = cv2.GaussianBlur(current_img, (5, 5), 1.0)
            scaled = cv2.resize(blurred, new_size, interpolation=cv2.INTER_LINEAR)
            
            pyramid.append(scaled)
            current_img = scaled
        
        return pyramid
    
    def _extract_dense_sift_single_img(self, img: np.ndarray) -> np.ndarray:
        """
        Extract Dense SIFT descriptors from a single image
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            SIFT descriptors
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        h, w = img.shape[:2]
        
        # Dense sampling: use grid with fixed step size
        keypoints = [cv2.KeyPoint(x, y, self._step_size) 
                    for y in range(0, h, self._step_size) 
                    for x in range(0, w, self._step_size)]
        
        if len(keypoints) == 0:
            return np.zeros((0, 128), dtype=np.float32)
        
        keypoints, desc = self.sift.compute(img, keypoints)
        
        if desc is None:
            return np.zeros((0, 128), dtype=np.float32)
        
        return desc.astype(np.float32)
    
    def _extract_pyramid_descriptors(self, img: np.ndarray) -> list:
        """
        Extract SIFT descriptors at all levels of Gaussian pyramid
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image
            
        Returns
        -------
        list
            List of descriptors for each scale
        """
        pyramid = self._build_gaussian_pyramid(img)
        pyramid_descriptors = []
        
        for level, pyr_img in enumerate(pyramid):
            desc = self._extract_dense_sift_single_img(pyr_img)
            pyramid_descriptors.append(desc)
        
        return pyramid_descriptors
    
    def _build_pyramid_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Build pyramid histogram features for a single image
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            Concatenated pyramid feature vector
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        pyramid_descriptors = self._extract_pyramid_descriptors(img)
        all_histograms = []
        
        for level, desc in enumerate(pyramid_descriptors):
            # Build histogram for each scale
            if desc.shape[0] == 0:
                hist = np.zeros(self._n_words, dtype=np.float32)
            else:
                word_indices = self.kmeans.predict(desc)
                hist, _ = np.histogram(word_indices, bins=np.arange(self._n_words + 1))
                hist = hist.astype(np.float32)
            
            # L2 normalization
            hist = normalize(hist.reshape(1, -1), norm='l2').squeeze(0)
            
            # Optional: weight different scales
            # Smaller scales (coarser) typically have lower weight
            weight = 1.0 / (2 ** level)
            hist = hist * weight
            
            all_histograms.append(hist)
        
        # Concatenate histograms from all scales
        feature = np.concatenate(all_histograms)
        
        # Overall L2 normalization
        feature = normalize(feature.reshape(1, -1), norm='l2').squeeze(0)
        
        return feature
    
    def _extract_all_descriptors(self, imgs):
        """
        Extract SIFT descriptors from all images for training codebook
        
        Parameters
        ----------
        imgs : list
            List of images
            
        Returns
        -------
        numpy.ndarray
            Stacked array of all descriptors
        """
        all_desc = []
        
        for i, img in enumerate(imgs):
            pyramid_descriptors = self._extract_pyramid_descriptors(img)
            
            # Collect descriptors from all scales
            for desc in pyramid_descriptors:
                if desc.shape[0] > 0:
                    all_desc.append(desc)
            
            if (i + 1) % 100 == 0:
                print(f"[DenseSIFTPyramid] Processed {i + 1}/{len(imgs)} images")
        
        if len(all_desc) == 0:
            raise ValueError("No descriptors extracted!")
        
        return np.vstack(all_desc)
    
    def fit(self, imgs):
        """
        Train visual vocabulary (codebook)
        
        Parameters
        ----------
        imgs : list
            List of training images
            
        Returns
        -------
        self
        """
        print(f"[DenseSIFTPyramid] Extracting Dense SIFT descriptors from Gaussian pyramid...")
        print(f"[DenseSIFTPyramid] Pyramid parameters: {self._n_scales} scales, scale_factor={self._scale_factor}")
        
        all_desc = self._extract_all_descriptors(imgs)
        print(f"[DenseSIFTPyramid] Total descriptors: {all_desc.shape[0]}")
        
        # Fix random seed
        rng = np.random.default_rng(self._random_state)
        
        # Sampling
        if all_desc.shape[0] > self._max_sample:
            sample_indices = rng.choice(all_desc.shape[0], 
                                       size=self._max_sample, 
                                       replace=False)
            sample_desc = all_desc[sample_indices]
            print(f"[DenseSIFTPyramid] Sampled {sample_desc.shape[0]} descriptors")
        else:
            sample_desc = all_desc
        
        # K-Means clustering
        print(f"[DenseSIFTPyramid] Training K-Means (n_clusters={self._n_words})...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self._n_words,
            batch_size=self._batch_size,
            random_state=self._random_state,
            verbose=0
        )
        self.kmeans.fit(sample_desc)
        
        print(f"[DenseSIFTPyramid] Training complete! Feature dimension: {self._feature_dim}")
        return self
    
    def transform(self, imgs):
        """
        Transform images to pyramid SIFT features
        
        Parameters
        ----------
        imgs : list
            List of images
            
        Returns
        -------
        numpy.ndarray
            Feature matrix (n_images, feature_dim)
        """
        if self.kmeans is None:
            raise Exception("Model not trained yet, please call fit() first!")
        
        X = []
        for i, img in enumerate(imgs):
            feature = self._build_pyramid_histogram(img)
            X.append(feature)
            
            if (i + 1) % 100 == 0:
                print(f"[DenseSIFTPyramid] Processed {i + 1}/{len(imgs)} images")
        
        return np.vstack(X)
    
    def fit_transform(self, imgs):
        """
        Train and transform
        
        Parameters
        ----------
        imgs : list
            List of images
            
        Returns
        -------
        numpy.ndarray
            Feature matrix
        """
        self.fit(imgs)
        return self.transform(imgs)


if __name__ == '__main__':
    # Test code
    print("=" * 60)
    print("Dense SIFT in Gaussian Pyramid Test")
    print("=" * 60)
    
    # Create test instance
    pyramid_sift = DenseSIFTPyramid(
        n_words=250,
        n_scales=3,
        scale_factor=0.7,
        step_size=8
    )
    
    print(f"\nConfiguration:")
    print(f"  - Visual vocabulary: {pyramid_sift._n_words}")
    print(f"  - Pyramid scales: {pyramid_sift._n_scales}")
    print(f"  - Scale factor: {pyramid_sift._scale_factor}")
    print(f"  - Step size: {pyramid_sift._step_size}")
    print(f"  - Feature dimension: {pyramid_sift.feature_dim}")
    print(f"\nDescription:")
    print(f"  This method extracts Dense SIFT features at {pyramid_sift._n_scales} different scales,")
    print(f"  each scale contributes a {pyramid_sift._n_words}-dimensional BoVW histogram,")
    print(f"  final feature dimension is {pyramid_sift._n_scales} × {pyramid_sift._n_words} = {pyramid_sift.feature_dim}")
    print("=" * 60)

