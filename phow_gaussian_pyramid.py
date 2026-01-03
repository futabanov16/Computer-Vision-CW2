import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


class PHOWGaussianPyramid:
    """
    PHOW with Gaussian Pyramid
    Combines Dense SIFT at multiple Gaussian pyramid scales with spatial pooling
    This method extracts spatial pyramid features at each Gaussian pyramid level,
    capturing both multi-scale and spatial layout information
    """
    def __init__(self, n_words=250, max_sample=200000, batch_size=100,
                 gaussian_scales=2, scale_factor=0.7, 
                 spatial_levels=[0, 1], step_size=8, random_state=0):
        """
        Parameters
        ----------
        n_words : int
            Visual vocabulary size
        max_sample : int
            Maximum number of descriptors to sample
        batch_size : int
            Batch size for MiniBatchKMeans
        gaussian_scales : int
            Number of Gaussian pyramid scales (including original image)
        scale_factor : float
            Gaussian pyramid scale factor (each level relative to previous level)
        spatial_levels : list
            Spatial pyramid levels, e.g., [0, 1] represents 1x1, 2x2 grids
        step_size : int
            Step size for dense sampling
        random_state : int
            Random seed
        """
        self._n_words = n_words
        self._max_sample = max_sample
        self._batch_size = batch_size
        self._gaussian_scales = gaussian_scales
        self._scale_factor = scale_factor
        self._spatial_levels = spatial_levels
        self._step_size = step_size
        self._random_state = random_state
        
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        
        # Calculate feature dimension
        self._feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """
        Calculate total feature dimension
        Each Gaussian scale contributes spatial pyramid histograms
        """
        total_dim = 0
        for spatial_level in self._spatial_levels:
            n_cells = (2 ** spatial_level) ** 2  # level 0: 1, level 1: 4, level 2: 16
            total_dim += n_cells * self._n_words
        
        # Multiply by number of Gaussian scales
        total_dim *= self._gaussian_scales
        
        return total_dim
    
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
        
        for i in range(1, self._gaussian_scales):
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
        list
            Keypoints
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        h, w = img.shape[:2]
        
        # Dense sampling: use grid with fixed step size
        keypoints = [cv2.KeyPoint(x, y, self._step_size) 
                    for y in range(0, h, self._step_size) 
                    for x in range(0, w, self._step_size)]
        
        if len(keypoints) == 0:
            return np.zeros((0, 128), dtype=np.float32), []
        
        keypoints, desc = self.sift.compute(img, keypoints)
        
        if desc is None:
            return np.zeros((0, 128), dtype=np.float32), []
        
        return desc.astype(np.float32), keypoints
    
    def _extract_spatial_pyramid_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Extract spatial pyramid histogram from a single image scale
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image at a specific Gaussian scale
            
        Returns
        -------
        numpy.ndarray
            Spatial pyramid feature vector for this image scale
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        h, w = img.shape[:2]
        all_histograms = []
        
        for spatial_level in self._spatial_levels:
            # Number of grid cells in this spatial level
            n_cells = 2 ** spatial_level  # level 0: 1, level 1: 2, level 2: 4
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
                    
                    # Extract Dense SIFT within this grid
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
            
            # Weight by spatial pyramid level (optional)
            # level 0 weight: 1/(2^0) = 1
            # level 1 weight: 1/(2^1) = 0.5
            weight = 1.0 / (2 ** spatial_level)
            level_histograms = [h * weight for h in level_histograms]
            
            all_histograms.extend(level_histograms)
        
        # Concatenate histograms from all spatial levels
        feature = np.concatenate(all_histograms)
        
        # Overall L2 normalization
        feature = normalize(feature.reshape(1, -1), norm='l2').squeeze(0)
        
        return feature
    
    def _extract_full_pyramid_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract features from both Gaussian pyramid and spatial pyramid
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            Complete feature vector combining Gaussian and spatial pyramids
        """
        # Build Gaussian pyramid
        gaussian_pyramid = self._build_gaussian_pyramid(img)
        
        all_features = []
        
        # Extract spatial pyramid features at each Gaussian scale
        for gaussian_level, pyr_img in enumerate(gaussian_pyramid):
            spatial_features = self._extract_spatial_pyramid_histogram(pyr_img)
            
            # Optional: weight by Gaussian pyramid level
            # Coarser scales get lower weight
            weight = 1.0 / (2 ** gaussian_level)
            spatial_features = spatial_features * weight
            
            all_features.append(spatial_features)
        
        # Concatenate features from all Gaussian scales
        feature = np.concatenate(all_features)
        
        # Final L2 normalization
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
            # Build Gaussian pyramid for this image
            gaussian_pyramid = self._build_gaussian_pyramid(img)
            
            # Extract descriptors from all Gaussian scales
            for pyr_img in gaussian_pyramid:
                desc, _ = self._extract_dense_sift_single_img(pyr_img)
                if desc.shape[0] > 0:
                    all_desc.append(desc)
            
            if (i + 1) % 100 == 0:
                print(f"[PHOWGaussianPyramid] Processed {i + 1}/{len(imgs)} images")
        
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
        print(f"[PHOWGaussianPyramid] Extracting Dense SIFT from Gaussian + Spatial pyramids...")
        print(f"[PHOWGaussianPyramid] Gaussian scales: {self._gaussian_scales}, Spatial levels: {self._spatial_levels}")
        
        all_desc = self._extract_all_descriptors(imgs)
        print(f"[PHOWGaussianPyramid] Total descriptors: {all_desc.shape[0]}")
        
        # Fix random seed
        rng = np.random.default_rng(self._random_state)
        
        # Sampling
        if all_desc.shape[0] > self._max_sample:
            sample_indices = rng.choice(all_desc.shape[0], 
                                       size=self._max_sample, 
                                       replace=False)
            sample_desc = all_desc[sample_indices]
            print(f"[PHOWGaussianPyramid] Sampled {sample_desc.shape[0]} descriptors")
        else:
            sample_desc = all_desc
        
        # K-Means clustering
        print(f"[PHOWGaussianPyramid] Training K-Means (n_clusters={self._n_words})...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self._n_words,
            batch_size=self._batch_size,
            random_state=self._random_state,
            verbose=0
        )
        self.kmeans.fit(sample_desc)
        
        print(f"[PHOWGaussianPyramid] Training complete! Feature dimension: {self._feature_dim}")
        return self
    
    def transform(self, imgs):
        """
        Transform images to PHOW-Gaussian pyramid features
        
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
            feature = self._extract_full_pyramid_features(img)
            X.append(feature)
            
            if (i + 1) % 100 == 0:
                print(f"[PHOWGaussianPyramid] Processed {i + 1}/{len(imgs)} images")
        
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
    print("PHOW with Gaussian Pyramid Test")
    print("=" * 60)
    
    # Create test instance
    phow_gauss = PHOWGaussianPyramid(
        n_words=250,
        gaussian_scales=2,      # 2 Gaussian scales
        scale_factor=0.7,
        spatial_levels=[0, 1],  # 2 spatial levels (1x1, 2x2)
        step_size=8
    )
    
    print(f"\nConfiguration:")
    print(f"  - Visual vocabulary: {phow_gauss._n_words}")
    print(f"  - Gaussian scales: {phow_gauss._gaussian_scales}")
    print(f"  - Scale factor: {phow_gauss._scale_factor}")
    print(f"  - Spatial levels: {phow_gauss._spatial_levels}")
    print(f"  - Step size: {phow_gauss._step_size}")
    print(f"  - Feature dimension: {phow_gauss.feature_dim}")
    print(f"\nDescription:")
    print(f"  This method combines Gaussian pyramid and spatial pyramid:")
    print(f"  - {phow_gauss._gaussian_scales} Gaussian scales capture multi-scale information")
    print(f"  - {len(phow_gauss._spatial_levels)} spatial levels capture spatial layout")
    print(f"  - Total cells per image: {phow_gauss._gaussian_scales} × {sum(4**l for l in phow_gauss._spatial_levels)} = {phow_gauss._gaussian_scales * sum(4**l for l in phow_gauss._spatial_levels)}")
    print(f"  - Final feature dimension: {phow_gauss.feature_dim}")
    print("=" * 60)

