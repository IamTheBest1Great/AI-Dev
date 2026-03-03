# Day 23–24: Random Forest – The Power of Many Trees

Welcome to the next topic in our machine learning journey! Today we're exploring **Random Forest**, one of the most popular and powerful ensemble methods. It builds upon the decision trees we learned earlier by combining many trees to create a model that is more accurate, robust, and less prone to overfitting.

We'll cover:
- What ensemble learning is.
- Bagging (Bootstrap Aggregating) – the foundation of Random Forest.
- How Random Forest adds extra randomness.
- The algorithm step-by-step.
- Key hyperparameters and their effects.
- Feature importance.
- Out-of-bag (OOB) evaluation.
- Implementation with scikit-learn.
- Advantages and disadvantages.
- Practical examples.
- Comparison with single decision trees.
- Practice tasks.

Let's get started!

---

## 1. What is Ensemble Learning?

Ensemble learning combines multiple models to produce a single, stronger model. The idea is that by aggregating the predictions of several base models, we can reduce errors and improve generalization. Two main types:

- **Averaging / Voting**: Train several models independently and average their predictions (e.g., Random Forest).
- **Boosting**: Train models sequentially, each focusing on the mistakes of the previous (e.g., Gradient Boosting).

Random Forest belongs to the **bagging** family.

---

## 2. Bagging (Bootstrap Aggregating)

**Bagging** is a technique that:
1. Creates multiple **bootstrap samples** (random samples with replacement) from the training data.
2. Trains a separate model (often a decision tree) on each bootstrap sample.
3. Aggregates predictions: for classification, by majority voting; for regression, by averaging.

This reduces variance without increasing bias much, leading to a more stable model.

### Why does bagging work?
Different bootstrap samples produce slightly different trees. Each tree may overfit in different ways, but averaging their predictions smooths out the noise, reducing overall variance.

---

## 3. Random Forest – Bagging + Random Feature Selection

Random Forest improves upon bagging by adding **random feature selection** at each split. When building a tree, at each node, only a random subset of features is considered for splitting (not all features). This decorrelates the trees even further.

### The Algorithm Step-by-Step

For a Random Forest with `n_estimators` trees:

1. For each tree from 1 to `n_estimators`:
   - Draw a bootstrap sample (random sample with replacement) of size N from the training data.
   - Grow a decision tree on this bootstrap sample:
     - At each node, randomly select `max_features` features from the total features.
     - Choose the best split among those features using a criterion (e.g., Gini).
     - Grow the tree to its full depth (or until `min_samples_split` etc., but typically no pruning).
   - Do not prune the tree.
2. Aggregate predictions from all trees:
   - **Classification**: majority vote.
   - **Regression**: average.

---

## 4. Why Random Forest Works So Well

- **Bagging** reduces variance by averaging many trees.
- **Random feature selection** decorrelates the trees. If all trees used all features, they'd be more similar, and averaging would reduce variance less.
- Trees are grown deep, so they have low bias (they fit the bootstrap sample well). The averaging then reduces variance without increasing bias much.

The result: a model that often performs very well out-of-the-box with minimal tuning.

---

## 5. Key Hyperparameters in Random Forest

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `n_estimators` | Number of trees in the forest | More trees = better performance, but diminishing returns and higher computation cost. |
| `max_depth` | Maximum depth of each tree | Deeper trees can capture more complex patterns but may overfit. Often left default (None) to grow fully. |
| `min_samples_split` | Min samples required to split an internal node | Higher values prevent overfitting. |
| `min_samples_leaf` | Min samples required at a leaf | Smoother boundaries, prevents overfitting. |
| `max_features` | Number of features to consider at each split | Smaller values increase randomness, reduce correlation between trees. Common defaults: sqrt(p) for classification, p/3 for regression. |
| `bootstrap` | Whether to use bootstrap samples | Usually True. If False, all trees trained on whole dataset (less randomness). |
| `oob_score` | Whether to compute out-of-bag score | Useful for validation without a separate validation set. |
| `random_state` | Seed for reproducibility | |

---

## 6. Out-of-Bag (OOB) Score

Since each tree is trained on a bootstrap sample, about 1/3 of the data is **not used** for that tree (these are called out-of-bag samples). We can use these OOB samples as a built-in validation set:

- For each data point, collect predictions from trees where that point was OOB.
- Aggregate these predictions (vote or average) to get an OOB prediction.
- Compute the overall OOB score (accuracy for classification, R² for regression).

This provides an unbiased estimate of the model's performance without needing a separate validation set.

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")
```

---

## 7. Feature Importance

Random Forest provides two types of feature importance:

### Mean Decrease in Impurity
For each feature, sum the reduction in impurity (Gini or entropy) over all splits where that feature was used, weighted by the number of samples split. This is automatically computed and available in `feature_importances_`.

### Permutation Importance
For a given feature, randomly shuffle its values and measure the drop in model performance. This is more reliable but computationally expensive.

```python
importances = rf.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

---

## 8. Implementation with scikit-learn

### Classification Example: Iris Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, oob_score=True)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
importances = rf.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")

# Plot
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### Regression Example: Boston Housing (or synthetic)

```python
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
boston = load_boston()
X, y = boston.data, boston.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)

# Predict
y_pred = rf_reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

---

## 9. Hyperparameter Tuning for Random Forest

While Random Forest works well with default parameters, tuning can squeeze out extra performance. Key parameters to tune:

- `n_estimators` (more is usually better, but diminishing returns)
- `max_depth` (often left None, but can be limited to reduce overfitting)
- `min_samples_split` and `min_samples_leaf`
- `max_features`

Use `GridSearchCV` or `RandomizedSearchCV`.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

---

## 10. Advantages and Disadvantages of Random Forest

| Advantages | Disadvantages |
|------------|---------------|
| High accuracy (often top performer) | Less interpretable than a single tree (though feature importance helps) |
| Robust to overfitting (due to averaging) | Can be slow to predict with many trees (but training can be parallelized) |
| Handles high-dimensional data well | Requires more memory and storage |
| Provides feature importance | Less effective on very sparse data (e.g., text) |
| Handles missing values reasonably | |
| No need for feature scaling | |
| Can handle both classification and regression | |

---

## 11. Random Forest vs. Single Decision Tree

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| Variance | High (prone to overfitting) | Low (averaging reduces variance) |
| Bias | Low (if grown deep) | Slightly higher, but usually better overall |
| Interpretability | High (can visualize) | Low (ensemble of trees) |
| Training speed | Fast | Slower (many trees) |
| Prediction speed | Fast | Slower (need to aggregate all trees) |
| Accuracy | Good, but often overfits | Usually better, more robust |

---

## 12. Practical Example: Classifying Handwritten Digits

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Digit classification accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 13. Summary

| Concept | Key Idea |
|---------|----------|
| Ensemble | Combine multiple models for better performance |
| Bagging | Train on bootstrap samples, average predictions |
| Random Forest | Bagging + random feature selection at each split |
| OOB Score | Built-in validation using out-of-bag samples |
| Feature Importance | Measure of feature contribution (impurity-based) |
| Hyperparameters | `n_estimators`, `max_features`, `max_depth`, etc. |
| Advantages | High accuracy, robust, no scaling needed, feature importance |
| Disadvantages | Less interpretable, slower prediction |

---

## 14. Practice Tasks

1. **Random Forest vs Single Tree**
   - Load the Iris dataset. Train a single decision tree (max_depth=5) and a random forest (n_estimators=100). Compare their test accuracies and visualize feature importances.

2. **Effect of `n_estimators`**
   - On a dataset of your choice, plot the OOB score or cross-validated accuracy as a function of `n_estimators` (from 1 to 200). At what point does adding more trees yield diminishing returns?

3. **Tuning `max_features`**
   - Experiment with different `max_features` values ('sqrt', 'log2', a fraction, None). How does it affect performance and tree correlation? (You can compute average correlation between tree predictions.)

4. **OOB Score vs Test Score**
   - On a dataset, split into train and test. Train a random forest with `oob_score=True`. Compare the OOB score with the test accuracy. How well does OOB estimate test performance?

5. **Feature Importance**
   - Use the Titanic dataset. Train a random forest. Extract feature importances. Which features are most important? Compare with the decision tree feature importance.

6. **Regression with Random Forest**
   - Use the California housing dataset (from sklearn.datasets.fetch_california_housing). Train a random forest regressor. Tune hyperparameters using `RandomizedSearchCV`. Evaluate with RMSE.

7. **Missing Values Experiment**
   - Introduce some missing values in a dataset (e.g., set 10% of values to NaN). Train a random forest (which can handle missing values by using surrogate splits in scikit-learn? Actually, sklearn's RandomForest does not natively handle NaNs; you'd need to impute first. But you can experiment with imputation strategies and see how robust RF is.)

8. **Mini Project: Predict Customer Churn**
   - Use a telecom churn dataset (available on Kaggle or UCI). Preprocess (encode categoricals, handle missing). Train a random forest classifier. Tune it. Evaluate using ROC-AUC. Report feature importance and discuss which factors drive churn.

---

You've now mastered Random Forest – a powerful, go-to algorithm for many machine learning tasks. Next, we could explore **Gradient Boosting** (XGBoost, LightGBM) or move to **Support Vector Machines**. Let me know your preference!
