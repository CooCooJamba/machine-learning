#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
A comprehensive benchmarking script for comparing classification models on the Iris dataset.
Compares performance of Logistic Regression, K-Nearest Neighbors, and Support Vector Machines.
Includes detailed metrics, visualization, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report,
                             accuracy_score,
                             confusion_matrix,
                             precision_recall_curve,
                             roc_curve,
                             auc)
from sklearn.preprocessing import label_binarize, StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Set a clean visual style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_and_explore_data():
    """
    Loads the Iris dataset and performs initial exploratory data analysis.
    
    Returns:
        tuple: (Features DataFrame, Target DataFrame, target names list)
    """
    print("=" * 50)
    print("1. DATA LOADING AND EXPLORATION")
    print("=" * 50)

    data = load_iris()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['target'])
    target_names = data.target_names

    print(f"Feature space dimensions: {x.shape}")
    print(f"Target variable dimensions: {y.shape}")
    print(f"Feature names: {list(x.columns)}")
    print(f"Class names: {list(target_names)}")
    print("\n")

    # Initial data exploration
    print("First look at features (X):")
    print(x.head())
    print("\n")

    print("First look at target variable (y):")
    print(y['target'].value_counts().to_frame('Count'))
    print("\n")

    print("Missing values check:")
    print("X:\n", x.isna().sum())
    print("y:\n", y.isna().sum())
    print("\n")

    print("Feature statistics:")
    print(x.describe())
    print("\n")

    # Class distribution visualization
    plt.figure()
    sns.countplot(x='target', data=y, palette='viridis')
    plt.title('Class Distribution in Target Variable')
    plt.xticks(ticks=[0, 1, 2], labels=target_names)
    plt.show()

    # Correlation matrix visualization
    plt.figure()
    data_for_corr = x.copy()
    data_for_corr['target'] = y['target']
    sns.heatmap(data_for_corr.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    return x, y, target_names


def prepare_data(x, y, test_size=0.2, random_state=123):
    """
    Splits data into train/test sets and scales features.
    
    Args:
        x (DataFrame): Feature matrix
        y (DataFrame): Target variable
        test_size (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Scaled and split data (X_train, X_test, Y_train, Y_test)
    """
    print("=" * 50)
    print("2. DATA PREPARATION")
    print("=" * 50)

    # Train/test split with stratification
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, 
        shuffle=True, stratify=y
    )

    # Feature scaling (critical for distance-based algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for better readability
    X_train = pd.DataFrame(X_train_scaled, columns=x.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=x.columns)

    # Convert y to 1D array to avoid sklearn warnings
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print("\n")

    return X_train, X_test, Y_train, Y_test


def evaluate_model(model, model_name, X_test, Y_test, target_names):
    """
    Evaluates model performance and generates comprehensive visualizations.
    
    Args:
        model: Trained scikit-learn model
        model_name (str): Model name for display purposes
        X_test (DataFrame): Test features
        Y_test (array): True test labels
        target_names (list): List of class names
    """
    print(f"MODEL EVALUATION: {model_name}")
    print("-" * 30)

    # Generate predictions
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Performance metrics
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=target_names))
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
    print("\n")

    # Confusion matrix
    plt.figure()
    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # ROC and PR curves (only for models that support probability predictions)
    if Y_pred_proba is not None:
        # Binarize target for multiclass evaluation
        Y_test_bin = label_binarize(Y_test, classes=[0, 1, 2])
        n_classes = Y_test_bin.shape[1]

        # ROC Curve
        plt.figure()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(Y_test_bin[:, i], Y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{target_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Random classifier line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves ({model_name})')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        # Precision-Recall Curve
        plt.figure()
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(Y_test_bin[:, i], Y_pred_proba[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{target_names[i]} (AUC = {pr_auc:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves ({model_name})')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()

    print("\n" * 2)


def main():
    """Main function executing the complete benchmarking pipeline."""
    # 1. Load and explore data
    x, y, target_names = load_and_explore_data()

    # 2. Prepare data
    X_train, X_test, Y_train, Y_test = prepare_data(x, y)

    # 3. Train and evaluate models
    print("=" * 50)
    print("3. MODEL TRAINING AND EVALUATION")
    print("=" * 50)

    # Logistic Regression
    print("\n>>> Logistic Regression <<<")
    lr_model = LogisticRegression(random_state=123, max_iter=200)
    lr_model.fit(X_train, Y_train)
    evaluate_model(lr_model, "Logistic Regression", X_test, Y_test, target_names)

    # K-Nearest Neighbors
    print("\n>>> K-Nearest Neighbors (KNN) <<<")
    knn_param_grid = {'n_neighbors': np.arange(1, 26)}
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
    knn_grid.fit(X_train, Y_train)

    print(f"Best KNN parameters: {knn_grid.best_params_}")
    print(f"Best cross-validation accuracy: {knn_grid.best_score_:.4f}")

    evaluate_model(knn_grid.best_estimator_, "K-Nearest Neighbors", X_test, Y_test, target_names)

    # Support Vector Machine
    print("\n>>> Support Vector Machine (SVM) <<<")
    svm_param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(SVC(probability=True, random_state=123),
                            svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, Y_train)

    print(f"Best SVM parameters: {svm_grid.best_params_}")
    print(f"Best cross-validation accuracy: {svm_grid.best_score_:.4f}")

    evaluate_model(svm_grid.best_estimator_, "Support Vector Machine", X_test, Y_test, target_names)


if __name__ == "__main__":
    main()

