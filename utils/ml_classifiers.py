"""
ml_classifiers.py

Utility functions for training and evaluating simple ML classifiers
(XGB, SVM, RandomForest), along with data normalization and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import os

def extract_features_simple(sample):
    """
    Simple feature extraction for a single sample (4 signals), 
    returning 16 features total.

    For each signal, we compute:
    - Mean
    - Peak
    - Dominant frequency index
    - Spectral entropy

    Args:
        sample (np.ndarray): shape (4, timepoints).

    Returns:
        np.ndarray: shape (16,)
    """
    from scipy.fft import fft
    features = []
    for wave in sample:
        mean_val = np.mean(wave)
        peak_val = np.max(wave)

        spectrum = np.abs(fft(wave))
        dom_freq_idx = np.argmax(spectrum)
        spectral_entropy = -np.sum(spectrum * np.log(spectrum + 1e-8))

        features.extend([mean_val, peak_val, dom_freq_idx, spectral_entropy])
    return np.array(features)

def extract_features_fusion(sample):
    """
    Alternate feature extraction (fusion idea),
    returning fewer features by averaging across channels.

    For each of 4 signals, we collect:
    - Mean
    - Peak
    - Dominant frequency index
    - Spectral entropy
    Then we average them to produce e.g. 5 final features.
    """
    from scipy.fft import fft
    import numpy as np
    means = []
    peaks = []
    dom_freqs = []
    spectrums = []
    idx_maxs = []
    for wave in sample:
        means.append(np.mean(wave))
        peaks.append(np.max(wave))
        idx_maxs.append(np.argmax(wave))

        spectrum = np.abs(fft(wave))
        dom_idx = np.argmax(spectrum)
        dom_freqs.append(dom_idx)

        spectrums.append(-np.sum(spectrum * np.log(spectrum + 1e-8)))

    # Example final 5 features
    return np.array([
        np.mean(means),
        np.mean(peaks),
        np.mean(idx_maxs),
        np.mean(dom_freqs),
        np.mean(spectrums)
    ])

def create_dataset(Li6_data, Po_data, fusion=False):
    """
    Create the dataset (features + labels) for Li6=0 vs Po=1 
    using either the 'simple' or 'fusion' feature extraction.

    Args:
        Li6_data (np.ndarray): shape (N_Li6, 4, T)
        Po_data  (np.ndarray): shape (N_Po, 4, T)
        fusion (bool): whether to use extract_features_fusion or extract_features_simple.

    Returns:
        X (np.ndarray): shape (N, D) feature matrix
        y (np.ndarray): shape (N,) labels array
    """
    if fusion:
        li6_features = [extract_features_fusion(sample.astype(np.float32)) for sample in Li6_data]
        po_features  = [extract_features_fusion(sample.astype(np.float32)) for sample in Po_data]
    else:
        li6_features = [extract_features_simple(sample.astype(np.float32)) for sample in Li6_data]
        po_features  = [extract_features_simple(sample.astype(np.float32)) for sample in Po_data]

    X = np.concatenate([li6_features, po_features], axis=0)
    y = np.concatenate([
        np.zeros(len(li6_features)), 
        np.ones(len(po_features))
    ], axis=0)
    return X, y


def train_and_evaluate_ml(
    X_train, X_test, y_train, y_test,
    model_type="xgb", threshold=0.5
):
    """
    Train and evaluate a chosen model (RandomForest, SVM, or XGB).
    Prints accuracy, confusion matrix, classification report, etc.

    Args:
        X_train (np.ndarray): shape (N_train, D)
        X_test  (np.ndarray): shape (N_test, D)
        y_train (np.ndarray): shape (N_train,)
        y_test  (np.ndarray): shape (N_test,)
        model_type (str): "xgb", "svm", or "rf"
        threshold (float): threshold for binary classification if proba-based model.

    Returns:
        trained_model, X_test_proba
    """
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        model = svm.SVC(probability=True, random_state=42)
    else:
        model = XGBClassifier(
            n_estimators=150, learning_rate=0.2,
            objective='binary:logistic', max_depth=8,
            use_label_encoder=False,
            eval_metric='logloss'  # required param for new XGB
        )

    model.fit(X_train, y_train)

    # Probability-based predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    # Probability histogram
    import matplotlib.pyplot as plt
    import numpy as np
    counts, bins = np.histogram(y_pred_proba, bins=np.linspace(0, 1, 21))

    plt.figure(figsize=(6, 6))
    plt.bar(bins[:-1], counts, width=0.05)
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlim([0, 1])
    plt.ylim([0, np.max(counts)])
    plt.show()

    # Confusion Matrix
    import seaborn as sns
    import numpy as np
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))

    conf_mat = np.array([[tp, fn], [fp, tn]])
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # Classification report
    from sklearn.metrics import classification_report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, y_pred_proba
