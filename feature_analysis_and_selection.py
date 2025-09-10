"""
Feature Analysis and Selection for Near-Earth Objects Classification

This script performs comprehensive feature importance analysis using various methods
(linear models, ensemble techniques, SHAP) and dimensionality reduction (PCA, UMAP)
on the NEO dataset to identify hazardous objects.
"""

# Standard library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine learning imports
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Statistical analysis imports
import statsmodels.api as sm

# Visualization and advanced analysis imports
import umap
import shap


def load_and_preprocess_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads and preprocesses the NEO dataset.

    Parameters:
        filepath (str): Path to the CSV file containing the data.

    Returns:
        tuple: Feature matrix (X) and target variable vector (y).
    """
    # Load data from CSV
    data = pd.read_csv(filepath)
    
    # Convert target variable to integer (binary classification: hazardous or not)
    y = data['hazardous'].astype(int)
    
    # Remove non-informative and identifier columns
    columns_to_drop = ['id', 'hazardous', 'orbiting_body', 'name', 'sentry_object']
    X = data.drop(columns=columns_to_drop)

    return X, y


def evaluate_with_linear_models(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Evaluates feature importance using linear models and statistical methods.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable vector.
    """
    print("=== FEATURE IMPORTANCE: LINEAR MODELS ===\n")

    # 1. Linear regression with p-value analysis using statsmodels
    print("1. Significant features based on p-value (OLS):")
    X_with_intercept = sm.add_constant(X)
    model_sm = sm.OLS(y, X_with_intercept)
    results = model_sm.fit()
    p_values = results.pvalues[1:]  # Exclude intercept
    significant_features_pval = X.columns[p_values < 0.05]
    print(f"   Features with p-value < 0.05: {list(significant_features_pval)}\n")

    # 2. Feature importance based on linear regression coefficients
    print("2. Feature importance based on absolute coefficients (Linear Regression):")
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    weights = np.abs(lr_model.coef_)
    # Create DataFrame for better visualization: feature - its weight
    weights_df = pd.DataFrame({'feature': X.columns, 'weight': weights})
    weights_df = weights_df.sort_values('weight', ascending=False)
    print(weights_df.to_string(index=False), '\n')

    # 3. Feature selection using Lasso regularization
    print("3. Significant features based on Lasso (non-zero coefficients):")
    lasso_model = Lasso(alpha=0.1, random_state=42)
    lasso_model.fit(X, y)
    non_zero_coef = lasso_model.coef_ != 0
    significant_features_lasso = X.columns[non_zero_coef]
    print(f"   Features selected by Lasso: {list(significant_features_lasso)}\n")

    # Visualize linear model feature importance
    plot_feature_importance(weights_df['weight'], weights_df['feature'],
                           "Absolute Coefficient Value", "Linear Regression")


def evaluate_with_ensemble_models(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Evaluates feature importance using ensemble methods.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable vector.

    Returns:
        tuple: Trained Gradient Boosting model and feature importances DataFrame.
    """
    print("=== FEATURE IMPORTANCE: ENSEMBLE METHODS ===\n")

    # 1. Random Forest feature importance
    print("1. Feature importance from RandomForest:")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    # Create DataFrame for better visualization
    importances_rf = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
    importances_rf = importances_rf.sort_values('importance', ascending=False)
    print(importances_rf.to_string(index=False), '\n')

    # 2. Gradient Boosting feature importance
    print("2. Feature importance from GradientBoosting:")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X, y)
    importances_gb = pd.DataFrame({'feature': X.columns, 'importance': gb_model.feature_importances_})
    importances_gb = importances_gb.sort_values('importance', ascending=False)
    print(importances_gb.to_string(index=False), '\n')

    # Visualize feature importance from both ensemble methods
    plot_feature_importance(importances_rf['importance'], importances_rf['feature'],
                           "Feature Importance", "Random Forest")
    plot_feature_importance(importances_gb['importance'], importances_gb['feature'],
                           "Feature Importance", "Gradient Boosting")

    return gb_model, importances_gb


def evaluate_with_shap(model, X: pd.DataFrame) -> None:
    """
    Analyzes and visualizes feature importance using SHAP values.

    Parameters:
        model: Trained model (expects GradientBoosting).
        X (pd.DataFrame): Feature matrix.
    """
    print("=== FEATURE IMPORTANCE: SHAP ANALYSIS ===\n")

    # Calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # 1. Summary bar plot
    print("1. SHAP summary plot (bar):")
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()

    # 2. Detailed summary plot showing feature impact distribution
    print("2. Detailed SHAP summary plot (showing impact distribution):")
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.show()

    # 3. Calculate mean absolute SHAP values for numerical comparison
    feature_importance_shap = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
    })
    feature_importance_shap = feature_importance_shap.sort_values('mean_abs_shap', ascending=False)
    print("3. Mean absolute SHAP values:")
    print(feature_importance_shap.to_string(index=False), '\n')

    # Visualize SHAP-based feature importance
    plot_feature_importance(feature_importance_shap['mean_abs_shap'], feature_importance_shap['feature'],
                           "Mean |SHAP value|", "SHAP Analysis")


def plot_feature_importance(importance: np.ndarray, feature_names: list,
                           xlabel: str, title: str) -> None:
    """
    Creates horizontal bar plot for feature importance visualization.

    Parameters:
        importance (np.ndarray): Array of importance values.
        feature_names (list): List of feature names.
        xlabel (str): X-axis label.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importance, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel(xlabel)
    plt.title(f"Feature Importance: {title}")
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(X: pd.DataFrame, y: pd.Series, top_n: int = 3) -> None:
    """
    Compares model performance using all features vs. top-N most important features.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable vector.
        top_n (int): Number of top features to select.
    """
    print("=== MODEL PERFORMANCE EVALUATION ===")

    # Identify top features based on GradientBoosting importance
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X, y)
    importances = gb_model.feature_importances_
    top_features = X.columns[np.argsort(importances)[-top_n:]]

    print(f"\nUsing top-{top_n} features: {list(top_features)}")

    # Create model for testing
    model = LinearRegression()

    # Cross-validation with all features
    scores_all = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_all = -scores_all.mean()

    # Cross-validation with selected features
    scores_selected = cross_val_score(model, X[top_features], y, cv=5, scoring='neg_mean_squared_error')
    mse_selected = -scores_selected.mean()

    print(f"\nMSE with all features: {mse_all:.4f}")
    print(f"MSE with selected features: {mse_selected:.4f}")

    # Performance comparison
    if mse_selected <= mse_all:
        print("✓ Feature selection IMPROVED model performance.")
    else:
        print("✗ Feature selection DID NOT improve model performance.")


def apply_dimensionality_reduction(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Applies dimensionality reduction techniques (PCA and UMAP) and visualizes results.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable vector.
    """
    print("=== DIMENSIONALITY REDUCTION ===\n")

    # Standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # 1. Principal Component Analysis (PCA)
    print("1. Principal Component Analysis (PCA):")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"   Explained variance ratio: "
          f"PC1: {explained_variance[0]:.3f}, PC2: {explained_variance[1]:.3f}")

    # PCA visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Hazardous Object')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
    plt.title('Data Visualization using PCA')
    plt.tight_layout()
    plt.show()

    # 2. UMAP for non-linear dimensionality reduction
    print("2. UMAP Dimensionality Reduction:")
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # UMAP visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Hazardous Object')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('Data Visualization using UMAP')
    plt.tight_layout()
    plt.show()


def main():
    """Main function that executes the complete analysis pipeline."""
    # Load and preprocess data
    print("Loading data...")
    X, y = load_and_preprocess_data('neo_v2.csv')

    print(f"Data shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target variable 'hazardous' distribution:\n{y.value_counts()}\n")

    # Feature importance analysis
    evaluate_with_linear_models(X, y)
    gb_model, importances_gb = evaluate_with_ensemble_models(X, y)
    evaluate_with_shap(gb_model, X)

    # Model performance evaluation with feature selection
    evaluate_model_performance(X, y, top_n=3)

    # Dimensionality reduction and visualization
    apply_dimensionality_reduction(X, y)


if __name__ == "__main__":
    main()

