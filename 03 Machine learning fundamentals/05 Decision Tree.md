# Day 21–22: Decision Trees – Intuitive and Interpretable Models

Welcome to the next topic in our machine learning journey! **Decision trees** are powerful, intuitive models that mimic human decision-making. They can be used for both classification and regression tasks. Their structure is easy to interpret, making them valuable for understanding feature importance and model reasoning.

We'll cover:
- What decision trees are and how they work.
- Splitting criteria: Gini impurity, entropy, information gain.
- Building and visualizing trees.
- Decision trees for classification and regression.
- Overfitting and pruning.
- Feature importance.
- Hyperparameter tuning.
- Advantages and disadvantages.
- Practice tasks.

Let's get started!

---

## 1. What is a Decision Tree?

A decision tree is a flowchart-like structure where each internal node represents a "test" on a feature (e.g., "is age > 30?"), each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a predicted value (for regression).

They are called **decision trees** because we start at the root and follow branches based on feature values until we reach a leaf, which gives the decision.

**Example**: Predicting whether to play tennis based on weather conditions. The tree might ask: "Is it sunny? If yes, check humidity; if no, check if it's rainy, etc."

---

## 2. How Decision Trees Work

The goal is to create a tree that best separates the data into pure subsets. The tree is built **recursively**:

1. Start with all data at the root.
2. Select the best feature and split point that most reduces impurity (or increases homogeneity).
3. Split the data into subsets.
4. Repeat recursively on each subset until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or pure leaves).

### 2.1 Splitting Criteria

To decide which feature to split on, we need a measure of **impurity**. The most common are:

#### Gini Impurity (for classification)
Measures how often a randomly chosen element would be incorrectly classified if randomly labeled according to the class distribution in the node.

\[
\text{Gini}(t) = 1 - \sum_{i=1}^{C} p_i^2
\]

where \(p_i\) is the proportion of class \(i\) in node \(t\). A pure node (all same class) has Gini = 0. The worst case (all classes equally mixed) gives maximum Gini.

#### Entropy and Information Gain
Entropy measures disorder:

\[
\text{Entropy}(t) = -\sum_{i=1}^{C} p_i \log_2 p_i
\]

Pure node: entropy = 0. Maximum when classes equally mixed.

**Information Gain** is the reduction in entropy after a split:

\[
\text{IG}(split) = \text{Entropy}(parent) - \sum_{j} \frac{n_j}{n} \text{Entropy}(child_j)
\]

We choose the split that maximizes information gain.

#### Variance Reduction (for regression)
For regression trees, we minimize the variance within child nodes (or equivalently, maximize variance reduction).

### 2.2 Building the Tree

The tree is built greedily: at each node, we consider all features and all possible split points, choose the one that best reduces impurity, and repeat.

---

## 3. Decision Tree Classification Example

Let's use the Iris dataset to build a decision tree classifier.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create decision tree (default hyperparameters)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
```

---

## 4. Decision Tree Regression Example

Decision trees can also predict continuous values.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic regression data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit regression tree
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train, y_train)

# Predict
y_pred = dt_reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Plot
X_plot = np.linspace(0, 5, 100).reshape(-1,1)
y_plot = dt_reg.predict(X_plot)

plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test')
plt.plot(X_plot, y_plot, color='red', label='Decision Tree Prediction')
plt.legend()
plt.show()
```

---

## 5. Visualizing Decision Trees

One of the biggest advantages of decision trees is interpretability. We can visualize the tree structure.

### Using `plot_tree` (scikit-learn)

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=feature_names, class_names=target_names, rounded=True)
plt.show()
```

### Using `export_text` (text representation)

```python
from sklearn.tree import export_text

tree_rules = export_text(dt, feature_names=feature_names)
print(tree_rules)
```

This prints a textual description of the tree's decision rules.

---

## 6. Feature Importance

Decision trees can rank features by how useful they are for splitting. Importance is computed based on how much each feature reduces impurity across all splits.

```python
importance = dt.feature_importances_
for name, imp in zip(feature_names, importance):
    print(f"{name}: {imp:.4f}")

# Plot
plt.barh(feature_names, importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Decision Tree')
plt.show()
```

---

## 7. Advantages and Disadvantages of Decision Trees

| Advantages | Disadvantages |
|------------|---------------|
| Easy to understand and interpret | Prone to overfitting (especially deep trees) |
| Requires little data preprocessing (no scaling needed) | Can be unstable: small variations in data may produce different trees |
| Handles both numerical and categorical data | Greedy algorithm may not find optimal tree |
| Non-parametric – captures complex relationships | Biased toward features with many levels (for categorical) |
| Provides feature importance | High variance – often outperformed by ensembles (Random Forest, Gradient Boosting) |

---

## 8. Overfitting and Pruning

Decision trees tend to overfit if grown too deep. They memorize noise in the training data. To prevent overfitting, we use:

- **Pre-pruning (early stopping)**: Stop growing the tree before it becomes too complex. Hyperparameters like `max_depth`, `min_samples_split`, `min_samples_leaf` control this.
- **Post-pruning (pruning)**: Grow a deep tree, then remove branches that have little predictive power. Scikit-learn doesn't implement post-pruning directly, but `ccp_alpha` (cost complexity pruning) can be used.

### Effect of `max_depth`

```python
# Train trees with different depths and compare performance
depths = range(1, 10)
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.plot(depths, train_scores, 'b-', label='Train accuracy')
plt.plot(depths, test_scores, 'r-', label='Test accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

You'll see train accuracy increasing with depth, but test accuracy may peak and then decline – a sign of overfitting.

---

## 9. Hyperparameter Tuning

Key hyperparameters for decision trees:

- `max_depth`: Maximum depth of the tree. Deeper trees can model more complex patterns but risk overfitting.
- `min_samples_split`: Minimum number of samples required to split an internal node. Higher values prevent splits on small subsets.
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node. Smoother boundaries.
- `max_features`: Number of features to consider for best split. Reduces overfitting.
- `criterion`: 'gini' or 'entropy'.
- `ccp_alpha`: Complexity parameter for minimal cost-complexity pruning.

We can use `GridSearchCV` or `RandomizedSearchCV` to find optimal parameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
```

---

## 10. Practical Example: Titanic Survival with Decision Tree

Let's apply decision trees to the Titanic dataset we used earlier.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler  # Actually not needed for trees, but we keep for consistency

# Load and preprocess Titanic
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Simplified preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = (df['Sex'] == 'male').astype(int)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a default tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(f"Default tree accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Tune hyperparameters
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

# Evaluate on test
best_dt = grid.best_estimator_
y_pred_best = best_dt.predict(X_test)
print(f"Test accuracy after tuning: {accuracy_score(y_test, y_pred_best):.4f}")

# Feature importance
importance = best_dt.feature_importances_
for name, imp in zip(features, importance):
    print(f"{name}: {imp:.4f}")
```

---

## 11. Summary

| Concept | Key Idea |
|---------|----------|
| Decision tree | Recursive partitioning based on feature tests |
| Splitting criteria | Gini, entropy, information gain (classification); variance reduction (regression) |
| Interpretability | Easy to visualize and explain |
| Overfitting | Deep trees memorize noise; use pruning or hyperparameter tuning |
| Feature importance | Based on impurity reduction |
| Hyperparameters | `max_depth`, `min_samples_split`, `min_samples_leaf`, etc. |
| Advantages | No scaling needed, handles mixed data, interpretable |
| Disadvantages | High variance, unstable, can overfit |

---

## 12. Practice Tasks

1. **Manual Decision Tree**
   - Draw a small dataset of 10 points with two features and two classes. Manually build a decision tree by choosing splits (you can use Gini impurity). Compare with scikit-learn's output.

2. **Iris with Different Criteria**
   - Train decision trees on Iris using Gini and entropy. Compare accuracy. Are the trees different? Visualize them.

3. **Effect of `max_depth`**
   - Generate a synthetic dataset (e.g., make_classification from sklearn) and plot training vs test accuracy for different max_depth values. Identify the point of overfitting.

4. **Regression Tree**
   - Use the California housing dataset (or any regression dataset). Train a decision tree regressor. Tune `max_depth`. Evaluate using RMSE.

5. **Feature Importance**
   - On the Titanic dataset, extract feature importance. Which features are most predictive of survival? Does this match your intuition?

6. **Grid Search**
   - Perform a grid search on the Iris dataset to find the best hyperparameters. Report the best parameters and test accuracy.

7. **Visualize Tree**
   - Train a shallow tree (max_depth=3) on any dataset and visualize it. Explain the decision path for a couple of test samples.

8. **Mini Project: Predict Credit Risk**
   - Find a credit risk dataset (e.g., from UCI or Kaggle). Preprocess it (handle missing values, encode categoricals). Build a decision tree classifier. Tune it. Evaluate using accuracy, precision, recall. Interpret the tree and feature importance.

---

You've now added decision trees to your machine learning toolbox. Next, we can explore **ensemble methods** like Random Forest and Gradient Boosting, or move to **SVM**. Let me know your preference!
