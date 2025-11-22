
"""
Utility functions and simple grading helpers for the Naive Bayes lab.
This file is self-contained (numpy-only) for easy use in Google Colab.
"""

from collections import Counter, defaultdict
import math
import random
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# --------------------
# Display helpers (same interface as the NN baseline)
# --------------------
def show_result(name: str, res: dict):
    status = "PASS" if res.get("passed") else "FAIL"
    msg = res.get("message", "")
    print(f"[{status}] {name}" + (f" | {msg}" if msg else ""))

# --------------------
# Text preprocessing
# --------------------
def tokenize(text: str):
    """
    Tiny tokenizer: lowercase, keep alphanumerics, split on non-alnum.
    """
    out = []
    w = []
    for ch in text.lower():
        if ch.isalnum():
            w.append(ch)
        else:
            if w:
                out.append("".join(w))
                w = []
    if w:
        out.append("".join(w))
    return out

def build_vocab(texts, min_freq=1, max_size=None):
    """
    Build a word -> index vocabulary.
    """
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    items = [(w, c) for w, c in cnt.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        items = items[:max_size]
    vocab = {w: i for i, (w, _) in enumerate(items)}
    return vocab

def vectorize_bow(texts, vocab, binary=True):
    """
    Convert list of raw texts into BOW vectors (list of lists).
    If binary=True, values are 0/1 (Bernoulli NB); else counts (Multinomial NB).
    """
    X = []
    for t in texts:
        toks = tokenize(t)
        vec = [0]*len(vocab)
        if binary:
            seen = set()
            for tok in toks:
                if tok in vocab and tok not in seen:
                    vec[vocab[tok]] = 1
                    seen.add(tok)
        else:
            for tok in toks:
                if tok in vocab:
                    vec[vocab[tok]] += 1
        X.append(vec)
    return X

def train_test_split(X, y, test_size=0.25, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_test = max(1, int(len(X)*test_size))
    test_idx = set(idx[:n_test])
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for i in range(len(X)):
        if i in test_idx:
            X_te.append(X[i]); y_te.append(y[i])
        else:
            X_tr.append(X[i]); y_tr.append(y[i])
    return X_tr, X_te, y_tr, y_te

# --------------------
# Naive Bayes for text
# --------------------
class NaiveBayesText:
    """
    Simple Naive Bayes for binary classification with either:
      - 'bernoulli': uses presence/absence of words
      - 'multinomial': uses word counts
    Laplace smoothing with alpha > 0.
    """
    def __init__(self, mode='bernoulli', alpha=1.0):
        assert mode in ('bernoulli', 'multinomial')
        self.mode = mode
        self.alpha = float(alpha)
        self.vocab_size = None
        self.class_priors = {}   # {c: log p(c)}
        # For bernoulli: feature likelihoods store P(x_j=1|c)
        # For multinomial: feature likelihoods store P(w=j | c)
        self.feature_likelihoods = {}  # {c: [prob per vocab index]}

    def fit(self, X, y):
        """
        X: list of lists (n_samples x vocab_size)
        y: list of labels (0/1)
        """
        n = len(X)
        self.vocab_size = len(X[0]) if n > 0 else 0

        # priors
        counts = Counter(y)
        for c in counts:
            self.class_priors[c] = math.log((counts[c] + self.alpha) / (n + 2*self.alpha))

        # likelihoods
        if self.mode == 'bernoulli':
            # count #docs where feature j = 1 for each class
            doc_counts = {0: [0]*self.vocab_size, 1: [0]*self.vocab_size}
            class_counts = {0: 0, 1: 0}
            for xi, yi in zip(X, y):
                class_counts[yi] += 1
                for j, v in enumerate(xi):
                    if v > 0:
                        doc_counts[yi][j] += 1
            # P(x_j = 1 | c) with Laplace smoothing
            self.feature_likelihoods = {}
            for c in (0,1):
                denom = class_counts[c] + 2*self.alpha
                probs = [ (doc_counts[c][j] + self.alpha) / denom for j in range(self.vocab_size) ]
                self.feature_likelihoods[c] = probs

        else:  # multinomial
            word_counts = {0: [0]*self.vocab_size, 1: [0]*self.vocab_size}
            total_words = {0: 0, 1: 0}
            for xi, yi in zip(X, y):
                for j, cnt in enumerate(xi):
                    if cnt > 0:
                        word_counts[yi][j] += cnt
                        total_words[yi] += cnt
            self.feature_likelihoods = {}
            for c in (0,1):
                denom = total_words[c] + self.alpha*self.vocab_size
                probs = [ (word_counts[c][j] + self.alpha) / denom for j in range(self.vocab_size) ]
                self.feature_likelihoods[c] = probs

    def _log_likelihood(self, x, c):
        if self.mode == 'bernoulli':
            # log P(x|c) = sum_j [ x_j*log p_jc + (1-x_j)*log(1-p_jc) ]
            probs = self.feature_likelihoods[c]
            s = 0.0
            for j, v in enumerate(x):
                pj = probs[j]
                if v > 0:
                    s += math.log(pj)
                else:
                    s += math.log(1.0 - pj)
            return s
        else:
            # multinomial: log P(x|c) = sum_j x_j * log p(w=j | c)
            probs = self.feature_likelihoods[c]
            s = 0.0
            for j, cnt in enumerate(x):
                if cnt > 0:
                    s += cnt * math.log(probs[j])
            return s

    def predict_proba(self, X):
        """
        Returns posterior P(y=1|x) for each sample (and P(y=0|x) = 1 - p)
        using log-space computations with Bayes rule up to a proportionality constant.
        """
        out = []
        for x in X:
            logp0 = self.class_priors.get(0, float("-inf")) + self._log_likelihood(x, 0)
            logp1 = self.class_priors.get(1, float("-inf")) + self._log_likelihood(x, 1)
            # log-sum-exp for normalization
            m = max(logp0, logp1)
            denom = m + math.log(math.exp(logp0 - m) + math.exp(logp1 - m))
            p1 = math.exp(logp1 - denom)
            out.append([1.0 - p1, p1])
        return out

    def predict(self, X):
        preds = []
        for x in X:
            logp0 = self.class_priors.get(0, float("-inf")) + self._log_likelihood(x, 0)
            logp1 = self.class_priors.get(1, float("-inf")) + self._log_likelihood(x, 1)
            preds.append(1 if logp1 >= logp0 else 0)
        return preds

def accuracy(y_true, y_pred):
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / max(1, len(y_true))

def confusion_matrix(y_true, y_pred):
    # y in {0,1}
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return [[tn, fp],
            [fn, tp]]

# --------------------
# Week 7: Synthetic shape dataset and baseline for CNN exercises
# --------------------

def generate_shape_dataset(n_samples_per_class=50, image_size=28, seed=0):
    """
    Generates a simple synthetic image dataset consisting of three classes of shapes: circles,
    squares, and triangles. Each image is grayscale and normalized to [0,1].

    Args:
        n_samples_per_class: Number of images to generate for each class.
        image_size: Width/height of the square images.
        seed: Random seed for reproducibility.

    Returns:
        images: NumPy array of shape (N, image_size, image_size) containing float32 pixel values.
        labels: NumPy array of shape (N,) with integer class labels {0,1,2} corresponding to
                classes ["circle", "square", "triangle"] in alphabetical order.
    """
    rng = random.Random(seed)
    classes = ['circle', 'square', 'triangle']
    images, labels = [], []
    for c_idx, shape in enumerate(classes):
        for _ in range(n_samples_per_class):
            # start with a blank image
            img = np.zeros((image_size, image_size), dtype=np.uint8)
            # choose a random center away from the borders
            margin = image_size // 4
            cx = rng.randint(margin, image_size - margin - 1)
            cy = rng.randint(margin, image_size - margin - 1)
            # choose a random size proportional to the image
            size = rng.randint(max(2, image_size // 8), max(3, image_size // 5))
            if shape == 'circle':
                cv2.circle(img, (cx, cy), size, (255,), -1)
            elif shape == 'square':
                top_left = (cx - size, cy - size)
                bottom_right = (cx + size, cy + size)
                cv2.rectangle(img, top_left, bottom_right, (255,), -1)
            else:  # triangle
                pts = np.array([
                    [cx, cy - size],
                    [cx - size, cy + size],
                    [cx + size, cy + size]
                ], np.int32)
                cv2.fillPoly(img, [pts], (255,))
            images.append(img.astype(np.float32) / 255.0)
            labels.append(c_idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    idx = np.arange(len(images))
    np.random.shuffle(idx)
    return images[idx], labels[idx]


def train_test_split_images(images, labels, test_ratio=0.3, seed=0):
    """
    Splits a set of images and labels into train and test subsets.

    Args:
        images: NumPy array of shape (N, H, W) containing image data.
        labels: NumPy array of shape (N,) containing integer labels.
        test_ratio: Fraction of examples to put in the test set.
        seed: Random seed for reproducibility.

    Returns:
        (train_images, test_images, train_labels, test_labels) as NumPy arrays.
    """
    rng = random.Random(seed)
    idx = list(range(len(images)))
    rng.shuffle(idx)
    n_test = max(1, int(len(images) * test_ratio))
    test_idx = set(idx[:n_test])
    tr_images, tr_labels, te_images, te_labels = [], [], [], []
    for i in range(len(images)):
        if i in test_idx:
            te_images.append(images[i])
            te_labels.append(labels[i])
        else:
            tr_images.append(images[i])
            tr_labels.append(labels[i])
    return (np.array(tr_images), np.array(te_images),
            np.array(tr_labels), np.array(te_labels))


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
    Validate the student's PCA baseline implementation for Week 7.

    The student_pca_func should have signature:

        def student_pca_func(train_images, train_labels, test_images, test_labels, n_components):
            # return test accuracy (float between 0 and 1)

    This test generates a small synthetic shape dataset and checks that the function returns a
    float in [0,1] without raising an exception.
    """
    images, labels = generate_shape_dataset(n_samples_per_class=10, image_size=20, seed=42)
    Xtr, Xte, ytr, yte = train_test_split_images(images, labels, test_ratio=0.3, seed=1)
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
    Helper for CNN-related tests. Generates a tiny shape dataset, calls the student function,
    and ensures the return value is a float in [0,1].
    """
    images, labels = generate_shape_dataset(n_samples_per_class=5, image_size=20, seed=99)
    Xtr, Xte, ytr, yte = train_test_split_images(images, labels, test_ratio=0.3, seed=2)
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

# --------------------
# Tiny offline datasets
# --------------------
def tiny_spam_dataset():
    """
    Returns texts, labels for a tiny spam/ham dataset.
    Labels: 1 = spam, 0 = ham
    """
    spam = [
        "win cash now free prize claim",
        "you have been selected to win a $1000 gift card",
        "urgent offer click link to claim reward",
        "congratulations you are a winner claim your prize",
        "limited time deal win big now",
        "free entry win vacation now",
        "claim free bonus click today",
    ]
    ham = [
        "hey are we still on for lunch tomorrow",
        "please review the meeting notes from today",
        "can you send the updated report",
        "let us know your availability for next week",
        "see you at the skating practice tonight",
        "thanks for the help with the project",
        "remember to bring snacks for movie night",
    ]
    texts = ham + spam
    labels = [0]*len(ham) + [1]*len(spam)
    return texts, labels

# --------------------
# Grading tests (mirrors the baseline utils style)
# --------------------
def _safe_call(fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        return {"ok": True, "out": out, "err": None}
    except Exception as e:
        return {"ok": False, "out": None, "err": str(e)}

def test_exercise_1_probability(student_functions):
    """
    student_functions should be a dict with keys:
      - 'joint': joint(p_a, p_b) -> p(a,b) assuming independence
      - 'conditional': conditional(p_ab, p_b) -> p(a|b)
      - 'bayes': bayes(p_ba, p_a, p_b) -> p(a|b)
    """
    try:
        joint = student_functions["joint"]
        conditional = student_functions["conditional"]
        bayes = student_functions["bayes"]
    except KeyError as e:
        return {"passed": False, "message": f"missing function: {e}"}

    # Simple checks
    res1 = _safe_call(joint, 0.3, 0.5)
    if not res1["ok"]:
        return {"passed": False, "message": f"joint runtime error: {res1['err']}"}
    if abs(res1["out"] - 0.15) > 1e-8:
        return {"passed": False, "message": "joint is incorrect for (0.3, 0.5)"}

    res2 = _safe_call(conditional, 0.12, 0.4)
    if not res2["ok"]:
        return {"passed": False, "message": f"conditional runtime error: {res2['err']}"}
    if abs(res2["out"] - 0.3) > 1e-8:
        return {"passed": False, "message": "conditional is incorrect for (0.12, 0.4)"}

    res3 = _safe_call(bayes, 0.6, 0.2, 0.5)
    if not res3["ok"]:
        return {"passed": False, "message": f"bayes runtime error: {res3['err']}"}
    # p(a|b) = p(b|a)p(a)/p(b) = 0.6*0.2/0.5 = 0.24
    if abs(res3["out"] - 0.24) > 1e-8:
        return {"passed": False, "message": "bayes rule result incorrect for (0.6, 0.2, 0.5)"}

    return {"passed": True}

def test_exercise_2_nb_fit_predict(student_fit_func):
    """
    student_fit_func should be a function:
        def student_fit_func(texts, labels, mode, alpha):
            # return acc on held-out set using NaiveBayesText
    We check it on the tiny_spam_dataset for reproducibility.
    """
    texts, labels = tiny_spam_dataset()
    try:
        acc = student_fit_func(texts, labels, mode='bernoulli', alpha=1.0)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}
    if not (0.5 <= acc <= 1.0):
        return {"passed": False, "message": f"unexpected accuracy: {acc:.3f}"}
    return {"passed": True}

def test_exercise_3_smoothing(student_train_eval):
    """
    student_train_eval should be a function (alpha)->(train_acc, test_acc)
    We check that changing alpha does not crash and produces floats in [0,1].
    """
    try:
        a_small = student_train_eval(0.1)
        a_big = student_train_eval(5.0)
    except Exception as e:
        return {"passed": False, "message": f"runtime error: {e}"}

    for name, pair in [("alpha=0.1", a_small), ("alpha=5.0", a_big)]:
        if (not isinstance(pair, (tuple, list))) or len(pair) != 2:
            return {"passed": False, "message": f"{name} must return (train_acc, test_acc)"}
        tr, te = pair
        if not (0.0 <= tr <= 1.0 and 0.0 <= te <= 1.0):
            return {"passed": False, "message": f"accuracy out of range for {name}"}

    return {"passed": True}
