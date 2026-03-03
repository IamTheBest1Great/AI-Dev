# Day 19–20: K-Nearest Neighbors (KNN) – Learning from Neighbors

Welcome to the next topic in our machine learning journey! Today we're exploring **K-Nearest Neighbors (KNN)**, one of the simplest and most intuitive algorithms. It's a **non-parametric**, **instance-based** learning algorithm used for both classification and regression. Instead of learning a model from training data, it stores the entire training dataset and makes predictions based on similarity.

We'll cover:
- The intuition behind KNN.
- How KNN works for classification and regression.
- Choosing the right K and distance metrics.
- The critical importance of feature scaling.
- Pros and cons.
- Implementation with scikit-learn.
- Practical examples.
- Evaluating KNN models.
- Practice tasks.

Let's get started!

---

## 1. What is K-Nearest Neighbors (KNN)?

KNN is based on a simple idea: **"Birds of a feather flock together."** To predict the label of a new data point, look at the K closest points in the training set (its neighbors) and let them vote.

- For **classification**: The predicted class is the majority class among the K nearest neighbors.
- For **regression**: The predicted value is the average (or median) of the K nearest neighbors' values.

KNN is called a **lazy learner** because it does not build an explicit model during training. It just memorizes the training data. All computation happens at prediction time.

---

## 2. How KNN Works – Step by Step

1. Choose the number of neighbors, K.
2. For a new data point, calculate the distance to every point in the training set.
3. Select the K training points with the smallest distances.
4. For classification: take a majority vote of their labels. For regression: take the average of their values.
5. Assign that result to the new point.

### Distance Metrics
The most common distance metric is **Euclidean distance**:

\[
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

Other metrics include:
- **Manhattan distance**: sum of absolute differences.
- **Minkowski distance**: a generalization (Euclidean is p=2, Manhattan is p=1).
- **Cosine similarity**: often used for text data.

Scikit-learn's KNN allows you to choose the metric via the `metric` parameter.

---

## 3. Choosing K – The Hyperparameter

The choice of K is crucial:

- **Small K** (e.g., K=1): The model is sensitive to noise and may overfit. The decision boundary is very wiggly.
- **Large K**: The model is smoother, less sensitive to noise, but may underfit if K becomes too large and includes points from other classes.

Typically, we choose K using cross-validation (e.g., try odd values to avoid ties in binary classification).

**Rule of thumb**: K is often chosen as the square root of the number of training samples, but always validate.

---

## 4. The Importance of Feature Scaling

KNN relies on distances. If features have different scales (e.g., age 0-100 vs. salary 30k-200k), the distance will be dominated by the feature with the largest scale. This can lead to poor performance.

**Always scale features** before applying KNN. Common scaling methods:
- **Standardization**: (x - mean) / std (zero mean, unit variance).
- **Min-Max scaling**: (x - min) / (max - min) (range [0,1]).

Use `StandardScaler` or `MinMaxScaler` from scikit-learn.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 5. KNN for Classification – Example with Iris

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN classifier (K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Finding the Best K with Cross-Validation

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k} with accuracy {max(cv_scores):.4f}")

# Plot
import matplotlib.pyplot as plt
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('K')
plt.ylabel('Cross-validated accuracy')
plt.title('KNN Performance on Iris')
plt.show()
```

---

## 6. KNN for Regression – Example with Synthetic Data

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic regression data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (though only one feature, scaling still good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)

# Predict
y_pred = knn_reg.predict(X_test_scaled)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Plot
X_plot = np.linspace(0, 5, 100).reshape(-1,1)
X_plot_scaled = scaler.transform(X_plot)
y_plot = knn_reg.predict(X_plot_scaled)

plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test')
plt.plot(X_plot, y_plot, color='red', label='KNN Regression (K=5)')
plt.legend()
plt.show()
```

---

## 7. Advantages and Disadvantages of KNN

| Advantages | Disadvantages |
|------------|---------------|
| Simple, intuitive, easy to implement | Prediction can be slow for large datasets (need to compute distances to all training points) |
| No training phase (lazy learner) | Sensitive to irrelevant features and feature scales |
| Naturally handles multi-class problems | Requires feature scaling |
| Can be used for both classification and regression | Memory-intensive (stores all training data) |
| Non-parametric – can model complex boundaries | Choice of K and distance metric impacts performance |
| | Curse of dimensionality: performance degrades in high dimensions |

---

## 8. Curse of Dimensionality

As the number of dimensions increases, points become sparse, and distances become less meaningful – nearly all points appear equally far. KNN suffers greatly in high dimensions. **Dimensionality reduction** (PCA, feature selection) is often necessary before applying KNN in high-dimensional spaces.

---

## 9. Weighted KNN

Instead of a simple majority vote, we can weight neighbors by their distance (closer neighbors have more influence). In scikit-learn, set `weights='distance'` (default is 'uniform').

```python
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

---

## 10. Practical Example: Classifying Handwritten Digits

The digits dataset (8x8 images of digits) is a classic KNN application.

```python
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 11. Summary

| Concept | Key Idea |
|---------|----------|
| KNN | Predicts based on majority (classification) or average (regression) of K nearest neighbors |
| Distance metric | Euclidean is common; others available |
| Feature scaling | Essential for meaningful distances |
| Choosing K | Trade-off between bias and variance; use cross-validation |
| Curse of dimensionality | KNN struggles in high dimensions |
| Weighted KNN | Closer neighbors get more influence |

---

## 12. Practice Tasks

1. **Manual KNN (Conceptual)**
   - Given a small 2D dataset with 5 points and labels, manually compute the Euclidean distances from a new point (2,3) and classify with K=3.

2. **KNN on Iris**
   - Load Iris, split, scale.
   - Find the best K using cross-validation (1-20).
   - Evaluate on test set with best K.
   - Plot decision boundaries for K=1, K=5, K=15 (using two features for visualization).

3. **Effect of Scaling**
   - Create a dataset with two features on very different scales (e.g., age 0-100, income 20k-200k).
   - Fit KNN without scaling and with scaling.
   - Compare accuracy and decision boundaries.

4. **KNN for Regression**
   - Generate nonlinear data (e.g., y = x³ + noise).
   - Fit KNN regression for different K values.
   - Plot predictions and compute MSE.
   - Discuss how K affects smoothness.

5. **Weighted vs Uniform**
   - On Iris, compare KNN with uniform weights vs distance weights. Does it improve accuracy?

6. **Mini Project: Classify Fruits**
   - Find a fruit dataset (or create one with features like weight, color, size) or use the built-in wine dataset.
   - Preprocess, scale, split.
   - Use cross-validation to choose best K.
   - Report accuracy, confusion matrix.
   - Discuss the impact of feature scaling.

---

You've now added KNN to your toolkit – a simple yet powerful algorithm. Next, we can explore **decision trees and random forests**, or move to **model evaluation and validation** in more depth. Let me know your preference!
