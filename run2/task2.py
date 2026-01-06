import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.model_selection import StratifiedKFold


def extract_patches(image):
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
    # mean centering
    mean_centered = patches - patches.mean(axis=1, keepdims=True)

    # L2 normalisation
    norms = np.linalg.norm(mean_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return mean_centered / norms

def extract_bow_features(image, kmeans):
    patches = extract_patches(image)
    normalized_patches = normalize_patches(patches)

    visual_words = kmeans.predict(normalized_patches)
    histogram = np.bincount(visual_words, minlength=kmeans.n_clusters)

    # L2 normalised histogram
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm

    return histogram

def predict_image(image, classifiers, scaler, kmeans):
    features = extract_bow_features(image, kmeans)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    scores = []
    for clf in classifiers:
        score = clf.predict_proba(features_scaled)[0][1]
        scores.append(score)

    return np.argmax(scores)

def train_classifiers(images, labels, kmeans, class_names):
    X = []
    for img in images:
        X.append(extract_bow_features(img, kmeans))
    X = np.array(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = []
    for class_idx in range(len(class_names)):
        y_binary = (labels == class_idx).astype(int)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_scaled, y_binary)
        classifiers.append(clf)

    return classifiers, scaler

def build_vocabulary(images, n_clusters):
    all_patches = []
    max_patches_per_image = 100

    for img in images:
        patches = extract_patches(img)
        if len(patches) > max_patches_per_image:
            idx = np.random.choice(len(patches), max_patches_per_image, replace=False)
            patches = patches[idx]
        all_patches.append(patches)

    all_patches = np.vstack(all_patches)
    normalized_patches = normalize_patches(all_patches)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        verbose=1
    )
    kmeans.fit(normalized_patches)

    return kmeans

def load_training_images(training_dir):
    images = []
    labels = []

    class_names = sorted([
        d for d in os.listdir(training_dir)
        if os.path.isdir(os.path.join(training_dir, d))
    ])

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(training_dir, class_name)
        image_files = sorted(glob.glob(os.path.join(class_dir, '*.jpg')))

        for img_file in image_files:
            img = Image.open(img_file).convert('L')
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
            labels.append(class_idx)

    return images, np.array(labels), class_names

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    training_dir = os.path.join(script_dir, 'training')
    testing_dir = os.path.join(script_dir, 'testing')
    output_file = os.path.join(script_dir, 'run2.txt')

    # load training data
    images, labels, class_names = load_training_images(training_dir)


    # k-fold Cross Validation
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(images, labels), 1):
        print(f"\nFold {fold_idx}")

        train_images = [images[i] for i in train_idx]
        val_images   = [images[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels   = labels[val_idx]

        # build vocabulary & train classifiers on training fold
        kmeans = build_vocabulary(train_images, 2000)
        classifiers, scaler = train_classifiers(
            train_images, train_labels, kmeans, class_names
        )

        # evaluate
        correct = 0
        for img, true_label in zip(val_images, val_labels):
            if predict_image(img, classifiers, scaler, kmeans) == true_label:
                correct += 1

        acc = correct / len(val_images)
        fold_accuracies.append(acc)

        print(f"Fold {fold_idx} accuracy: {acc:.4f}")

    print(f"Mean CV accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std  CV accuracy: {np.std(fold_accuracies):.4f}")
    
    

    #print("\nTraining final model on full training set...")
    kmeans = build_vocabulary(images, 2000)
    classifiers, scaler = train_classifiers(
        images, labels, kmeans, class_names
    )

    test_images = sorted(glob.glob(os.path.join(testing_dir, '*.jpg')))
    predictions = []

    for img_path in test_images:
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32)

        pred_idx = predict_image(img_array, classifiers, scaler, kmeans)
        pred_class = class_names[pred_idx]

        predictions.append((os.path.basename(img_path), pred_class))

    # sort by image number
    def get_image_number(item):
        try:
            return int(item[0].replace('.jpg', ''))
        except ValueError:
            return float('inf')

    predictions.sort(key=get_image_number)

    with open(output_file, 'w') as f:
        for img_name, pred_class in predictions:
            f.write(f"{img_name} {pred_class}\n")

