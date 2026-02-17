"""
Generate run3.txt predictions using the best method: PHOW-Gaussian Pyramid + Chi2-SVM

This script:
1. Loads all training data (1500 images)
2. Trains the PHOW-Gaussian Pyramid feature extractor with optimal parameters
3. Trains Chi2-SVM classifier with C=0.1
4. Loads test data (2985 images, unlabeled)
5. Generates predictions and writes to run3.txt
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import AdditiveChi2Sampler
from dataset import Dataset
from phow_gaussian_pyramid import PHOWGaussianPyramid

def main():
    print("=" * 60)
    print("Generating Run 3 Predictions")
    print("=" * 60)
    print("\nMethod: PHOW-Gaussian Pyramid + Chi2-LinearSVC")
    print("Parameters:")
    print("  - n_words: 250")
    print("  - gaussian_scales: 2")
    print("  - spatial_levels: [0, 1]")
    print("  - C: 0.1")
    print("  - sample_steps: 3")
    print()
    
    # Load full training data
    print("[1/5] Loading training data...")
    training = Dataset(r"training", img_size=(256, 256))
    train_imgs, train_labels = training.get_data()
    print(f"  Loaded {len(train_imgs)} training images")
    print(f"  Classes: {len(np.unique(train_labels))}")
    
    # Initialize PHOW-Gaussian Pyramid feature extractor
    print("\n[2/5] Training PHOW-Gaussian Pyramid feature extractor...")
    phow_gaussian = PHOWGaussianPyramid(
        n_words=250,
        gaussian_scales=2,
        scale_factor=0.7,
        spatial_levels=[0, 1],
        step_size=8,
        max_sample=100000,
        random_state=42
    )
    
    # Learn vocabulary from all training images
    print("  Learning visual vocabulary...")
    phow_gaussian.fit(train_imgs)
    print(f"  Vocabulary size: {phow_gaussian._n_words}")
    
    # Transform training images
    print("  Transforming training images...")
    X_train = phow_gaussian.transform(train_imgs)
    print(f"  Feature dimension: {X_train.shape[1]}")
    
    # Train Chi2-SVM classifier
    print("\n[3/5] Training Chi2-LinearSVC classifier...")
    pipeline = Pipeline([
        ('chi2', AdditiveChi2Sampler(sample_steps=3)),
        ('svm', LinearSVC(C=0.1, max_iter=10000, random_state=42))
    ])
    
    print("  Fitting classifier on training data...")
    pipeline.fit(X_train, train_labels)
    print("  Training complete!")
    
    # Load test data (unlabeled)
    print("\n[4/5] Loading test data...")
    testing = Dataset(r"testing", img_size=(256, 256), label_exist=False)
    test_imgs, _ = testing.get_data()
    print(f"  Loaded {len(test_imgs)} test images")
    
    # Generate predictions
    print("\n[5/5] Generating predictions...")
    X_test = phow_gaussian.transform(test_imgs)
    predictions = pipeline.predict(X_test)
    print(f"  Generated {len(predictions)} predictions")
    
    # Write predictions to run3.txt
    output_file = "run3.txt"
    testing.runtxt_write(predictions, filepath=output_file)
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Predictions saved to {output_file}")
    print("=" * 60)
    
    # Show sample predictions
    print("\nSample predictions (first 10):")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 10:
                print(f"  {line.strip()}")
            else:
                break
    
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Unique classes predicted: {len(np.unique(predictions))}")
    
    # Show class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} images ({count/len(predictions)*100:.1f}%)")
    
    print("\n[DONE] Run 3 predictions ready for submission!")

if __name__ == "__main__":
    main()

