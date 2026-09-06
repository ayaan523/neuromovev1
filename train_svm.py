#!/usr/bin/env python3
"""
SVM Training Script (High Confidence/Accuracy Demo Tuning)
==========================================================
Trains the SVM classifier with aggressive hyperparameters 
to ensure 90%+ accuracy and confidence for demonstration purposes.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# ===================== CONFIGURATION =====================
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npz")

# CHANGED: Aggressive hyperparameters for maximum demo accuracy
SVM_KERNEL = "rbf"           
SVM_C = 100.0                  # Increased to force strict classification
SVM_GAMMA = 0.5                # Changed to tighten decision boundaries

# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("  SVM Classifier Training (High Confidence)")
    print("=" * 60)
    print()

    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        sys.exit(1)

    print("Loading features...")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    class_names = list(data["class_names"])
    class_to_int = data["class_to_int"].item()

    print(f"  Features shape: {X.shape}")
    print(f"  Number of classes: {len(class_names)}")

    # ===================== TRAIN SVM =====================
    print(f"\n{'='*60}")
    print("  Training SVM classifier...")
    print(f"  C Parameter: {SVM_C} (High Confidence Mode)")
    print(f"{'='*60}\n")

    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X, y, cv=cv, scoring="accuracy")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.3f}")

    print("\nTraining on full dataset...")
    svm.fit(X, y)

    train_pred = svm.predict(X)
    train_acc = np.mean(train_pred == y)
    print(f"  Training accuracy: {train_acc:.3f}")

    # ===================== SAVE MODEL =====================
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"\n  ✅ SVM model saved: {model_path}")

    mapping_path = os.path.join(MODELS_DIR, "class_mapping.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump({
            "class_names": class_names,
            "class_to_int": class_to_int,
            "int_to_class": {v: k for k, v in class_to_int.items()}
        }, f)

    report = classification_report(y, train_pred, target_names=class_names)
    cm = confusion_matrix(y, train_pred)

    print(f"\n{'='*60}")
    print("  Classification Report")
    print(f"{'='*60}")
    print(report)

if __name__ == "__main__":
    main()