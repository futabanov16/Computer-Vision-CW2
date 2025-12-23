from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import random

import numpy as np
from PIL import Image


# Tiny image feature extraction
def tiny_image_feature(img_path: Path, size: int = 16) -> np.ndarray:
    """
    Tiny image feature:
    - grayscale
    - centre crop to square
    - resize to size x size
    - flatten
    - zero mean, unit length (L2)
    """
    img = Image.open(img_path).convert("L")
    w, h = img.size
    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.resize((size, size), resample=Image.BILINEAR)

    x = np.asarray(img, dtype=np.float32).reshape(-1)
    x = x - x.mean()
    norm = np.linalg.norm(x)
    if norm > 1e-12:
        x = x / norm
    return x


# Data loading
def load_training_set(train_root: Path, size: int = 16) -> Tuple[np.ndarray, List[str]]:
    """
    Expects train_root/class_name/*.jpg
    Keeps class label EXACTLY as folder name (case-sensitive).
    """
    if not train_root.exists():
        raise FileNotFoundError(f"Training folder not found: {train_root}")

    class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under: {train_root}")

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for cdir in class_dirs:
        label = cdir.name
        img_paths = sorted([p for p in cdir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        for p in img_paths:
            X_list.append(tiny_image_feature(p, size=size))
            y_list.append(label)

    X_train = np.vstack(X_list).astype(np.float32)
    return X_train, y_list


def make_train_val_split(
    train_root: Path,
    val_per_class: int = 20,
    seed: int = 42,
) -> Tuple[List[Path], List[str], List[Path], List[str]]:
    """
    Stratified split from training folder.
    For each class folder: randomly pick val_per_class images for validation,
    use the rest for training.
    """
    if not train_root.exists():
        raise FileNotFoundError(f"Training folder not found: {train_root}")

    rng = random.Random(seed)

    train_paths: List[Path] = []
    train_labels: List[str] = []
    val_paths: List[Path] = []
    val_labels: List[str] = []

    class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under: {train_root}")

    for cdir in class_dirs:
        label = cdir.name
        img_paths = [p for p in cdir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        img_paths.sort()

        if len(img_paths) <= val_per_class:
            raise ValueError(
                f"Not enough images in class '{label}': {len(img_paths)} (need > {val_per_class})"
            )

        rng.shuffle(img_paths)

        val = img_paths[:val_per_class]
        trn = img_paths[val_per_class:]

        val_paths.extend(val)
        val_labels.extend([label] * len(val))

        train_paths.extend(trn)
        train_labels.extend([label] * len(trn))

    return train_paths, train_labels, val_paths, val_labels


def featurize(paths: List[Path], size: int = 16) -> np.ndarray:
    """
    Convert a list of image paths into a feature matrix.
    """
    X_list = [tiny_image_feature(p, size=size) for p in paths]
    return np.vstack(X_list).astype(np.float32)


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Simple classification accuracy.
    """
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def _numeric_stem(p: Path) -> int:
    """
    For test images like '0.jpg', '12.jpg' -> sort numerically.
    Falls back to lexicographic if not numeric.
    """
    try:
        return int(p.stem)
    except ValueError:
        return 10**12


def load_test_images(test_root: Path) -> List[Path]:
    """
    Expects test_root/*.jpg
    Returns images sorted numerically by filename stem if possible.
    """
    if not test_root.exists():
        raise FileNotFoundError(f"Testing folder not found: {test_root}")

    imgs = [p for p in test_root.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not imgs:
        raise FileNotFoundError(f"No test images found under: {test_root}")

    imgs.sort(key=lambda p: (_numeric_stem(p), p.name))
    return imgs

# kNN classifier
def knn_predict(
    X_train: np.ndarray,
    y_train: List[str],
    x_test: np.ndarray,
    k: int = 5,
) -> str:
    """
    Euclidean kNN majority vote.
    Tie-break: class with smallest sum of distances among its neighbours.
    """
    dists = np.linalg.norm(X_train - x_test[None, :], axis=1)

    k_eff = min(k, len(dists))
    nn_idx = np.argpartition(dists, kth=k_eff - 1)[:k_eff]

    nn_labels = [y_train[i] for i in nn_idx]
    nn_dists = dists[nn_idx]

    # Count votes
    counts: Dict[str, int] = {}
    for lab in nn_labels:
        counts[lab] = counts.get(lab, 0) + 1

    max_votes = max(counts.values())
    candidates = [lab for lab, c in counts.items() if c == max_votes]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break by distance sum
    best_label = None
    best_score = float("inf")
    nn_labels_arr = np.array(nn_labels)

    for lab in candidates:
        score = nn_dists[nn_labels_arr == lab].sum()
        if score < best_score:
            best_score = score
            best_label = lab

    return str(best_label)

# Main
def main(
    train_root: str = "training",
    test_root: str = "testing",
    out_path: str = "run1.txt",
    size: int = 16,
    k: int = 5,
    tune_k: bool = True,
    val_per_class: int = 20,
    seed: int = 42,
) -> None:
    train_root_p = Path(train_root)
    test_root_p = Path(test_root)

    print(f"Loading training set from: {train_root_p.resolve()}")

    # Show the discovered class names (case-sensitive)
    class_dirs = sorted([p for p in train_root_p.iterdir() if p.is_dir()])
    classes = [p.name for p in class_dirs]
    print(f"Classes ({len(classes)}): {classes}")

    chosen_k = k

    # 1) Tune k using ONLY training data (split into train/val)
    if tune_k:
        train_paths, train_labels, val_paths, val_labels = make_train_val_split(
            train_root_p, val_per_class=val_per_class, seed=seed
        )
        print(f"Train split: {len(train_paths)} images | Val split: {len(val_paths)} images")

        X_tr = featurize(train_paths, size=size)
        X_va = featurize(val_paths, size=size)

        k_grid = [1, 3, 5, 7, 9, 11, 15, 21]
        best_k = None
        best_acc = -1.0

        for kk in k_grid:
            preds = [knn_predict(X_tr, train_labels, X_va[i], k=kk) for i in range(X_va.shape[0])]
            acc = accuracy(val_labels, preds)
            print(f"k={kk:>2}  val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_k = kk

        chosen_k = int(best_k) if best_k is not None else k
        print(f"Selected k={chosen_k} from validation (acc={best_acc:.4f})")

    # 2) Final training on ALL training data, then predict test
    X_train_all, y_train_all = load_training_set(train_root_p, size=size)
    print(f"Training examples (ALL): {len(y_train_all)} | Feature dim: {X_train_all.shape[1]}")

    print(f"Loading test images from: {test_root_p.resolve()}")
    test_imgs = load_test_images(test_root_p)
    print(f"Test images: {len(test_imgs)}")

    lines: List[str] = []
    for i, img_path in enumerate(test_imgs, start=1):
        x = tiny_image_feature(img_path, size=size)
        pred = knn_predict(X_train_all, y_train_all, x, k=chosen_k)
        lines.append(f"{img_path.name} {pred}")

        if i % 250 == 0:
            print(f"Processed {i}/{len(test_imgs)}...")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f" Wrote predictions to: {Path(out_path).resolve()}")


if __name__ == "__main__":
    main(
        train_root="training",
        test_root="testing",
        out_path="run1.txt",
        size=16,
        k=1,              # fallback if tuning disabled
        tune_k=false,       # set False if you want fixed k (for final test run)
        val_per_class=20,  # 20 val images per class (so 80 train per class)
        seed=42,           # makes the split reproducible
    )
