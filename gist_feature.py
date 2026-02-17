import numpy as np
import cv2
from scipy.ndimage import convolve


class GISTFeature:
    """
    GIST Feature Extractor
    GIST is a global scene descriptor that captures the overall texture and spatial structure of images
    """
    def __init__(self, n_scales=4, n_orientations=8, n_blocks=4):
        """
        Parameters
        ----------
        n_scales : int
            Number of scales (Gaussian pyramid levels)
        n_orientations : int
            Number of orientations (Gabor filter orientations)
        n_blocks : int
            Number of spatial grid divisions (n_blocks x n_blocks)
        """
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.n_blocks = n_blocks
        
        # Feature dimension
        self.feature_dim = n_scales * n_orientations * (n_blocks ** 2)
    
    def _create_gabor_filters(self, img_size):
        """
        Create Gabor filter bank
        """
        filters = []
        
        # Gabor filters with different scales and orientations
        for scale in range(self.n_scales):
            for orientation in range(self.n_orientations):
                theta = np.pi * orientation / self.n_orientations
                
                # Gabor parameters
                wavelength = 2 ** (scale + 2)  # Wavelength
                sigma = wavelength * 0.56  # Standard deviation
                gamma = 0.5  # Aspect ratio
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(31, 31),
                    sigma=sigma,
                    theta=theta,
                    lambd=wavelength,
                    gamma=gamma,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                
                filters.append(kernel)
        
        return filters
    
    def _apply_gabor_filters(self, img, filters):
        """
        Apply Gabor filter bank to image
        """
        responses = []
        
        for kernel in filters:
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        return responses
    
    def _extract_grid_features(self, response):
        """
        Extract grid features from filter responses
        Divide image into n_blocks x n_blocks grid, compute average response for each grid
        """
        h, w = response.shape
        block_h = h // self.n_blocks
        block_w = w // self.n_blocks
        
        features = []
        
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                y_start = i * block_h
                y_end = (i + 1) * block_h if i < self.n_blocks - 1 else h
                x_start = j * block_w
                x_end = (j + 1) * block_w if j < self.n_blocks - 1 else w
                
                block = response[y_start:y_end, x_start:x_end]
                
                # Compute average response intensity for this block
                mean_response = np.abs(block).mean()
                features.append(mean_response)
        
        return np.array(features)
    
    def extract_single(self, img):
        """
        Extract GIST features from a single image
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image (grayscale)
            
        Returns
        -------
        numpy.ndarray
            GIST feature vector
        """
        # Ensure grayscale and float32
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        
        # Resize image to improve efficiency
        target_size = 256
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = cv2.resize(img, (target_size, target_size))
        
        # Create Gabor filters
        filters = self._create_gabor_filters(img.shape)
        
        # Apply filters
        responses = self._apply_gabor_filters(img, filters)
        
        # Extract grid features from each response
        all_features = []
        for response in responses:
            grid_features = self._extract_grid_features(response)
            all_features.extend(grid_features)
        
        # Convert to numpy array and normalize
        gist_feature = np.array(all_features, dtype=np.float32)
        
        # L2 normalization
        norm = np.linalg.norm(gist_feature)
        if norm > 0:
            gist_feature = gist_feature / norm
        
        return gist_feature
    
    def extract(self, imgs):
        """
        Batch extract GIST features
        
        Parameters
        ----------
        imgs : list or numpy.ndarray
            List of images
            
        Returns
        -------
        numpy.ndarray
            Feature matrix (n_images, feature_dim)
        """
        features = []
        
        for i, img in enumerate(imgs):
            feature = self.extract_single(img)
            features.append(feature)
            
            if (i + 1) % 100 == 0:
                print(f"[GIST] Processed {i + 1}/{len(imgs)} images")
        
        return np.vstack(features)


if __name__ == '__main__':
    # Test
    gist = GISTFeature(n_scales=4, n_orientations=8, n_blocks=4)
    print(f"GIST feature dimension: {gist.feature_dim}")

