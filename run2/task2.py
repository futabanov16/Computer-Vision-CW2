import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.model_selection import train_test_split


def extract_patches(image):
    # Extracting pixel blocks through dense sampling from images
    patch_size = 8
    stride = 4

    h, w = image.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch.flatten())
    
    return np.array(patches)

def normalize_patches(patches):
    # Centered mean
    mean_centered = patches - patches.mean(axis=1, keepdims=True)
    
    # L2 normalisation
    norms = np.linalg.norm(mean_centered, axis=1, keepdims=True)

    # avoid zero devision
    norms[norms == 0] = 1
    normalized = mean_centered / norms
    
    return normalized





def extract_bow_features(image, kmeans):
    # Extract blocks
    patches = extract_patches(image)
    
    # normalisation
    normalized_patches = normalize_patches(patches)
    
    # mapping each block to the nearest
    visual_words = kmeans.predict(normalized_patches)
    
    # histogram
    histogram = np.bincount(visual_words, minlength=kmeans.n_clusters)
    
    # L2 normalised histogram
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm
    
    return histogram

def predict_image(image, classifiers, scaler, kmeans):
    # Extract and standardise features
    features = extract_bow_features(image, kmeans)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Compute scores from all one-vs-all classifiers
    scores = []
    for clf in classifiers:
        score = clf.predict_proba(features_scaled)[0][1]
        scores.append(score)
    
    predicted_class_idx = np.argmax(scores)
    return predicted_class_idx



def train_classifiers(images, labels, kmeans, class_names):
    X_train = []
    for img in images:
        features = extract_bow_features(img, kmeans)
        X_train.append(features)

    X_train = np.array(X_train)

    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train 15 one-vs-all logistic regression classifiers
    classifiers = []
    for class_idx in range(len(class_names)):
        # Binary labels for the current class
        y_binary = (labels == class_idx).astype(int)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_binary)
        classifiers.append(clf)

    return classifiers, scaler



def build_vocabulary(images):
    all_patches = []
    n_clusters = 800
    max_patches_per_image = 100

    for img in images:
        patches = extract_patches(img)
        # Random sampling to speed up (if there are too many blocks)
        if len(patches) > max_patches_per_image:
            indices = np.random.choice(len(patches), max_patches_per_image, replace=False)
            patches = patches[indices]
        all_patches.append(patches)

    all_patches = np.vstack(all_patches)
    # print(f"Total extract {len(all_patches)} blocks")

    # Normalise blocks
    normalized_patches = normalize_patches(all_patches)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    kmeans.fit(normalized_patches)

    return kmeans



def load_training_images(training_dir):
    images = []
    labels = []

    all_items = os.listdir(training_dir)
    class_names = sorted([d for d in all_items
                          if os.path.isdir(os.path.join(training_dir, d))])

    # print(f"Find {len(class_names)} classes: {class_names}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(training_dir, class_name)
        image_files = sorted(glob.glob(os.path.join(class_dir, '*.jpg')))

        for img_file in image_files:
            img = Image.open(img_file).convert('L')  # convert to grayscale image
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
            labels.append(class_idx)

    return images, np.array(labels), class_names



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    training_dir = os.path.join(script_dir, 'training')
    testing_dir = os.path.join(script_dir, 'testing')
    output_file = os.path.join(script_dir, 'run2.txt')

    # training
    #images, labels, class_names = load_training_images(training_dir)
    #kmeans = build_vocabulary(images)
    #classifiers, scaler = train_classifiers(images, labels, kmeans, class_names)
    # load full training data

    images, labels, class_names = load_training_images(training_dir)

    # split training data into train / validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    # build vocabulary using TRAINING images only
    kmeans = build_vocabulary(train_images)

    # train classifiers using TRAINING images only
    classifiers, scaler = train_classifiers(
        train_images,
        train_labels,
        kmeans,
        class_names
    )

    test_images = sorted(glob.glob(os.path.join(testing_dir, '*.jpg')))

    # validation accuracy evaluation
    correct = 0
    total = len(val_images)

    for img, true_label in zip(val_images, val_labels):
        pred_label = predict_image(img, classifiers, scaler, kmeans)
        if pred_label == true_label:
            correct += 1

    val_accuracy = correct / total
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # prediction
    predictions = []
    for img_path in test_images:
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        
        predicted_idx = predict_image(img_array, classifiers, scaler, kmeans)
        predicted_class = class_names[predicted_idx]
        
        img_name = os.path.basename(img_path)
        predictions.append((img_name, predicted_class))
    
    # sort the image number
    def get_image_number(item):
        img_name = item[0]
        try:
            return int(img_name.replace('.jpg', ''))
        except ValueError:
            return float('inf')

    predictions.sort(key=get_image_number)
    
    # write into result
    with open(output_file, 'w') as f:
        for img_name, predicted_class in predictions:
            f.write(f"{img_name} {predicted_class}\n")


