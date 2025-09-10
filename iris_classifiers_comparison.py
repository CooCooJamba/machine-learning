#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Comparison of different classifier performance on the Iris dataset.
Includes: Decision Tree, Stacking, Bagging (Random Forest), Boosting (Gradient Boosting)
with hyperparameter tuning using GridSearch.
"""

# Import libraries
import warnings
from sklearn.datasets import load_iris
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
X, y = load_iris(return_X_y=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print("=" * 50)
print("CLASSIFIER COMPARISON ON IRIS DATASET")
print("=" * 50)

# 1. Baseline Decision Tree
print("\n1. BASELINE DECISION TREE")

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=['sepal length', 'sepal width', 
                                           'petal length', 'petal width'],
          class_names=['setosa', 'versicolor', 'virginica'])
plt.title("Baseline Decision Tree")
plt.show()

# 2. Decision Tree with Hyperparameter Tuning
print("\n2. DECISION TREE WITH HYPERPARAMETER TUNING")

# Parameters for GridSearch
params = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                          param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')
print(f'Best cross-validation accuracy: {grid_search.best_score_:.4f}')

# Train with best parameters
best_tree = DecisionTreeClassifier(**best_params, random_state=42)
best_tree.fit(X_train, y_train)
y_pred = best_tree.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Visualize optimized tree
plt.figure(figsize=(12, 8))
plot_tree(best_tree, filled=True, feature_names=['sepal length', 'sepal width', 
                                                'petal length', 'petal width'],
          class_names=['setosa', 'versicolor', 'virginica'])
plt.title("Optimized Decision Tree")
plt.show()

# 3. Stacking Classifier
print("\n3. STACKING CLASSIFIER")

# Base classifiers
base_classifiers = [
    ('lr', LogisticRegression(random_state=42, max_iter=200)),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(random_state=42, probability=True))
]

stacking = StackingClassifier(estimators=base_classifiers, 
                             final_estimator=LogisticRegression(),
                             cv=5)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# 4. Random Forest (Bagging)
print("\n4. RANDOM FOREST (BAGGING)")

bagging = RandomForestClassifier(**best_params, random_state=42)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# 5. Gradient Boosting
print("\n5. GRADIENT BOOSTING")

boosting = GradientBoostingClassifier(**best_params, random_state=42)
boosting.fit(X_train, y_train)
y_pred = boosting.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

print("\n" + "=" * 50)
print("ANALYSIS COMPLETED")
print("=" * 50)

