
"""
Utility functions and grading helpers for Week 7 (CNNs).
This file includes helper functions for image classification (CNN) exercises.
"""

import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# --------------------
# Display helpers
# --------------------


def show_result(name: str, res: dict):
    status = "PASS" if res.get("passed") else "FAIL"
    msg = res.get("message", "")
    print(f"[{status}] {name}" + (f" | {msg}" if msg else ""))


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / max(1, len(y_true))

# --------------------
# Week 7: CIFAR-10 dataset loading and baseline for CNN exercises
# --------------------


def load_cifar10_dataset(n_train=5000, n_test=1000, seed=0, grayscale=True):
    """
    Load CIFAR-10 dataset from HuggingFace and prepare it for the exercises.

    Args:
        n_train: Number of training samples to use
        n_test: Number of test samples to use
        seed: Random seed for reproducibility
        grayscale: If True, convert images to grayscale

    Returns:
        train_images: NumPy array of shape (n_train, H, W) with float32 values in [0,1]
        train_labels: NumPy array of shape (n_train,) of integer labels
        test_images: NumPy array of shape (n_test, H, W) with float32 values
        test_labels: NumPy array of shape (n_test,) of integer labels
        class_names: List of class names
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the datasets library: pip install datasets")

    # Load CIFAR-10 from HuggingFace
    print("Loading CIFAR-10 dataset from HuggingFace...")
    train_dataset = load_dataset(
        "uoft-cs/cifar10", split="train")
    test_dataset = load_dataset(
        "uoft-cs/cifar10", split="test")

    # Set random seed
    rng = random.Random(seed)
    np.random.seed(seed)

    # Sample indices
    train_indices = rng.sample(
        range(len(train_dataset)), min(n_train, len(train_dataset)))
    test_indices = rng.sample(range(len(test_dataset)),
                              min(n_test, len(test_dataset)))

    # Extract images and labels
    def process_images(dataset, indices, grayscale=True):
        images = []
        labels = []
        for idx in indices:
            example = dataset[idx]
            img = np.array(example['img'])  # Shape: (32, 32, 3)

            if grayscale:
                # Convert RGB to grayscale using standard luminosity formula
                img = (0.299 * img[:, :, 0] +
                       0.587 * img[:, :, 1] +
                       0.114 * img[:, :, 2])

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(example['label'])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

    train_images, train_labels = process_images(
        train_dataset, train_indices, grayscale)
    test_images, test_labels = process_images(
        test_dataset, test_indices, grayscale)

    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(
        f"Loaded {len(train_images)} training images and {len(test_images)} test images")
    print(f"Image shape: {train_images.shape[1:]}")

    return train_images, train_labels, test_images, test_labels, class_names


def pca_logistic_baseline(train_images, train_labels, test_images, test_labels, n_components=20):
    """
    Baseline classifier using PCA followed by logistic regression.

    Flattens the input images, reduces dimensionality with PCA, trains a multinomial logistic regression
    classifier, and computes the test accuracy.

    Args:
        train_images: Array of shape (N_train, H, W) with float32 values in [0,1].
        train_labels: Array of shape (N_train,) of integer class labels.
        test_images: Array of shape (N_test, H, W) with float32 values.
        test_labels: Array of shape (N_test,) of integer class labels.
        n_components: Number of principal components to keep (limited by number of features).

    Returns:
        Test accuracy as a float in [0,1].
    """
    # Flatten images to (N, D)
    X_tr = train_images.reshape(train_images.shape[0], -1)
    X_te = test_images.reshape(test_images.shape[0], -1)
    # Choose n_components not exceeding the number of features
    k = min(n_components, X_tr.shape[1])
    pca = PCA(n_components=k)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_tr_pca, train_labels)
    preds = clf.predict(X_te_pca)
    return accuracy(test_labels, preds)


# --------------------
# Week 7 grading tests for CNN exercises
# --------------------

def test_exercise_7_pca(student_pca_func):
    """
    Validate the student's PCA baseline implementation for Week 7.

    The student_pca_func should have signature:

        def student_pca_func(train_images, train_labels, test_images, test_labels, n_components):
            # return test accuracy (float between 0 and 1)

    This test loads a small subset of CIFAR-10 and checks that the function returns a
    float in [0,1] without raising an exception.
    """
    try:
        Xtr, ytr, Xte, yte, _ = load_cifar10_dataset(
            n_train=100, n_test=30, seed=42, grayscale=True)
    except Exception as e:
        return {"passed": False, "message": f"dataset loading error: {e}"}

    try:
        acc = student_pca_func(Xtr, ytr, Xte, yte, 5)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}
    # Type check
    if not isinstance(acc, (int, float)):
        return {"passed": False, "message": f"returned type must be float, got {type(acc)}"}
    # Range check
    if not (0.0 <= acc <= 1.0):
        return {"passed": False, "message": f"accuracy out of range: {acc}"}
    return {"passed": True}


def _test_cnn_student_func(student_fn):
    """
    Helper for CNN-related tests. Loads a tiny CIFAR-10 subset, calls the student function,
    and ensures the return value is a float in [0,1].
    """
    try:
        Xtr, ytr, Xte, yte, _ = load_cifar10_dataset(
            n_train=50, n_test=20, seed=99, grayscale=True)
    except Exception as e:
        return {"passed": False, "message": f"dataset loading error: {e}"}

    try:
        acc = student_fn(Xtr, ytr, Xte, yte)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}
    if not isinstance(acc, (int, float)):
        return {"passed": False, "message": f"returned type must be float, got {type(acc)}"}
    if not (0.0 <= acc <= 1.0):
        return {"passed": False, "message": f"accuracy out of range: {acc}"}
    return {"passed": True}


def test_exercise_7_simple_cnn(student_fn):
    """
    Test for the simple CNN implementation in Week 7.
    Expects the student's function signature:

        def student_fn(train_images, train_labels, test_images, test_labels):
            # return test accuracy
    """
    return _test_cnn_student_func(student_fn)


def test_exercise_7_proper_cnn(student_fn):
    """
    Test for the proper CNN implementation in Week 7.
    Expects the same signature as the simple CNN function. Uses the same helper.
    """
    return _test_cnn_student_func(student_fn)


def test_exercise_7_data_aug_cnn(student_fn):
    """
    Test for the data-augmented CNN implementation in Week 7.
    Uses the same helper to validate the return type and range.
    """
    return _test_cnn_student_func(student_fn)


def test_exercise_7_advanced_cnn(student_fn):
    """
    Test for the advanced CNN implementation in Week 7.
    Uses the same helper to validate the return type and range.
    """
    return _test_cnn_student_func(student_fn)
