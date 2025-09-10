# Install required packages
get_ipython().system('pip install kaggle plotly')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def linear_prediction(X, w):
    """Calculate linear prediction: y = Xw"""
    return np.dot(X, w)

def mean_squared_error_loss(X, y, w):
    """Calculate Mean Squared Error loss"""
    return np.mean((linear_prediction(X, w) - y) ** 2)

def gradient_descent_optimization(X, w, y, learning_rate=0.001, n_iter=10000):
    """
    Implement gradient descent optimization for linear regression
    
    Parameters:
    X: Feature matrix with bias term
    w: Initial weight vector
    y: Target values
    learning_rate: Step size for gradient descent
    n_iter: Number of iterations
    
    Returns:
    w: Optimized weight vector
    """
    n_samples = len(y)
    
    for _ in range(n_iter):
        # Calculate predictions and error
        y_pred = linear_prediction(X, w)
        error = y_pred - y
        
        # Calculate gradient (vectorized)
        gradient = (2/n_samples) * np.dot(X.T, error)
        
        # Update weights
        w = w - learning_rate * gradient
        
    return w

# Sample dataset (single feature with bias term)
X_data = np.array([
    6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764, 8.5781, 6.4862, 5.0546,
    5.7107, 14.164, 5.734, 8.4084, 5.6407, 5.3794, 6.3654, 5.1301, 6.4296, 7.0708,
    6.1891, 20.27, 5.4901, 6.3261, 5.5649, 18.945, 12.828, 10.957, 13.176, 22.203,
    5.2524, 6.5894, 9.2482, 5.8918, 8.2111, 7.9334, 8.0959, 5.6063, 12.836, 6.3534,
    5.4069, 6.8825, 11.708, 5.7737, 7.8247, 7.0931, 5.0702, 5.8014, 11.7, 5.5416,
    7.5402, 5.3077, 7.4239, 7.6031, 6.3328, 6.3589, 6.2742, 5.6397, 9.3102, 9.4536,
    8.8254, 5.1793, 21.279, 14.908, 18.959, 7.2182, 8.2951, 10.236, 5.4994, 20.341,
    10.136, 7.3345, 6.0062, 7.2259, 5.0269, 6.5479, 7.5386, 5.0365, 10.274, 5.1077,
    5.7292, 5.1884, 6.3557, 9.7687, 6.5159, 8.5172, 9.1802, 6.002, 5.5204, 5.0594,
    5.7077, 7.6366, 5.8707, 5.3054, 8.2934, 13.394, 5.4369
])

y_target = np.array([
    17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483, 12, 6.5987, 3.8166,
    3.2522, 15.505, 3.1551, 7.2258, 0.71618, 3.5129, 5.3048, 0.56077, 3.6518, 5.3893,
    3.1386, 21.767, 4.263, 5.1875, 3.0825, 22.638, 13.501, 7.0467, 14.692, 24.147,
    -1.22, 5.9966, 12.134, 1.8495, 6.5426, 4.5623, 4.1164, 3.3928, 10.117, 5.4974,
    0.55657, 3.9115, 5.3854, 2.4406, 6.7318, 1.0463, 5.1337, 1.844, 8.0043, 1.0179,
    6.7504, 1.8396, 4.2885, 4.9981, 1.4233, -1.4211, 2.4756, 4.6042, 3.9624, 5.4141,
    5.1694, -0.74279, 17.929, 12.054, 17.054, 4.8852, 5.7442, 7.7754, 1.0173, 20.992,
    6.6799, 4.0259, 1.2784, 3.3411, -2.6807, 0.29678, 3.8845, 5.7014, 6.7526, 2.0576,
    0.47953, 0.20421, 0.67861, 7.5435, 5.3436, 4.2415, 6.7981, 0.92695, 0.152, 2.8214,
    1.8451, 4.2959, 7.2029, 1.9869, 0.14454, 9.0551, 0.61705
])

# Prepare feature matrix with bias term
X_with_bias = np.column_stack((np.ones(len(X_data)), X_data))
initial_weights = np.array([1.0, 1.0])

# Numerical solution using gradient descent
weights_numerical = gradient_descent_optimization(X_with_bias, initial_weights, y_target)
print("Numerically optimized weights:", weights_numerical)

# Plot results for numerical solution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.grid(True)
plt.scatter(X_data, y_target, color='green', label='Data points', alpha=0.7)
plt.plot(X_data, linear_prediction(X_with_bias, weights_numerical), 
         color='orange', linewidth=2, label='Numerical fit')
plt.xlabel('Feature value')
plt.ylabel('Target value')
plt.title('Numerical Solution (Gradient Descent)')
plt.legend()

# Analytical solution (Normal Equation)
weights_analytical = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_target
print("Analytically optimized weights:", weights_analytical)

# Plot results for analytical solution
plt.subplot(1, 2, 2)
plt.grid(True)
plt.scatter(X_data, y_target, color='black', label='Data points', alpha=0.7)
plt.plot(X_data, X_with_bias @ weights_analytical, 
         color='red', linewidth=2, label='Analytical fit')
plt.xlabel('Feature value')
plt.ylabel('Target value')
plt.title('Analytical Solution (Normal Equation)')
plt.legend()
plt.tight_layout()
plt.show()

# Compare numerical and analytical solutions
print("\nComparison of solutions:")
print(f"Numerical weights: {weights_numerical}")
print(f"Analytical weights: {weights_analytical}")
print(f"Difference: {np.abs(weights_numerical - weights_analytical)}")

# Download and prepare dataset (commented out for clarity)
"""
# Set up Kaggle API
from google.colab import files
files.upload()  # Upload kaggle.json

!rm -rf ~/.kaggle
!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d nancyalaswad90/diamonds-prices
!unzip diamonds-prices.zip

# Load and prepare data
data = pd.read_csv('diamonds.csv')
print(f"Dataset shape: {data.shape}")
data.head()
"""

# For demonstration, we'll use the sample data path as in the original code
# In practice, you would use the downloaded diamonds dataset
try:
    data = pd.read_csv('/content/sample_data/business.retailsales.csv')
    print(f"Dataset shape: {data.shape}")
    print("\nDataset info:")
    data.info()
    
    # Check for missing values
    print("\nMissing values per column:")
    print(data.isna().sum())
    
    # Prepare features and target
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")
    
    # Display unique values in categorical columns
    unique_counts = pd.DataFrame([
        [col, data[col].nunique()] for col in categorical_cols
    ], columns=['Column', 'Unique Values']).sort_values('Unique Values')
    
    print("\nUnique values in categorical columns:")
    print(unique_counts)
    
    # One-hot encode categorical variables
    dummies = pd.get_dummies(data[categorical_cols], drop_first=True)
    X = data.drop(categorical_cols, axis=1).join(dummies)
    
    # Define target variable
    target_name = 'Total Net Sales'
    y = data[target_name]
    X = X.drop(target_name, axis=1)
    
    # Visualize target distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Total Net Sales')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Variable')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(y)
    plt.ylabel('Total Net Sales')
    plt.title('Boxplot of Target Variable')
    
    plt.tight_layout()
    plt.show()
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and compare models
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(random_state=42),
        'Ridge Regression': Ridge(random_state=42)
    }
    
    results = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items(), 1):
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {'MSE': mse, 'R2': r2, 'MAE': mae}
        
        # Plot predictions vs actual
        plt.subplot(1, 3, i)
        plt.scatter(y_pred, y_test, alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name}\nMAE: {mae:.2f}, R²: {r2:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    # Print results comparison
    print("\nModel Comparison Results:")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  MSE: {metrics['MSE']:.3f}")
        print(f"  R²: {metrics['R2']:.3f}")
        print(f"  MAE: {metrics['MAE']:.3f}")
        print("-" * 30)
    
except FileNotFoundError:
    print("Sample data file not found. Skipping regularization comparison.")
    print("Please upload the dataset or adjust the file path.")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("1. Implemented linear regression from scratch with:")
print("   - Gradient descent (numerical solution)")
print("   - Normal equation (analytical solution)")
print("2. Compared numerical and analytical approaches")
print("3. Demonstrated regularization techniques with scikit-learn:")
print("   - Linear Regression (no regularization)")
print("   - Lasso Regression (L1 regularization)")
print("   - Ridge Regression (L2 regularization)")
print("4. Evaluated models using MSE, R², and MAE metrics")

