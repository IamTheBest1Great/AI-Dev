# Week 3: Machine Learning Basics
## Day 15–16: Train/Test Split

Welcome to the first topic in our Machine Learning Basics week! Today we'll cover one of the most fundamental concepts in machine learning: **splitting data** into training and testing sets. This simple idea is the cornerstone of evaluating how well our models will perform on unseen data.

We'll explore:
- Why we need to split data.
- The concepts of overfitting and underfitting.
- How to perform train/test split using scikit-learn.
- Stratified splitting for classification.
- Special considerations for time series data.
- The role of validation sets.
- Brief introduction to cross-validation.

Let's get started!

---

## 1. Why Split Data?

Imagine you're studying for an exam. You practice on sample questions, but the real test has new questions. If you only practiced and then took the same questions on the exam, you'd get a perfect score – but that wouldn't mean you truly understood the material. Similarly, in machine learning, we want our model to perform well on **new, unseen data**, not just memorize the training examples.

If we train and evaluate on the same data, we risk **overfitting** – the model learns the noise and details of the training set too well, and fails to generalize. To get an honest estimate of performance, we need to test the model on data it has never seen.

Thus, we split our dataset into:
- **Training set**: Used to train the model (learn the parameters).
- **Test set**: Held back until the very end, used only once to evaluate the final model's performance.

This gives us an unbiased estimate of how the model will perform on new data.

---

## 2. Overfitting and Underfitting

Understanding train/test split requires understanding two common problems:

### Overfitting
- The model learns the training data too well, including its noise and random fluctuations.
- It performs exceptionally well on training data but poorly on test data.
- Analogy: memorizing answers without understanding concepts.

### Underfitting
- The model is too simple to capture the underlying structure of the data.
- It performs poorly on both training and test data.
- Analogy: not studying enough to grasp even the basics.

The goal is to find a balance – a model that generalizes well.

---

## 3. The Holdout Method: Train/Test Split

The simplest approach is to randomly split the dataset into two parts: a training set (typically 70-80%) and a test set (20-30%). The model is trained on the training set, and its performance is evaluated on the test set.

### Important Rules
- The test set should only be used **once** – at the very end. Never use it for tuning or decision-making.
- Both sets should be representative of the overall data distribution.
- The split should be random, but sometimes we need to preserve class proportions (stratification).

---

## 4. Implementing Train/Test Split in Python

Scikit-learn provides the convenient `train_test_split` function.

### Basic Example
```python
import numpy as np
from sklearn.model_selection import train_test_split

# Sample dataset: 100 samples, 2 features
X = np.random.rand(100, 2)  # features
y = np.random.randint(0, 2, size=100)  # binary labels

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

- `test_size`: fraction of data to use as test set (can also be an integer count).
- `random_state`: seed for reproducibility (so you get the same split each time).
- The function returns four arrays: `X_train, X_test, y_train, y_test`.

### With pandas DataFrames
```python
import pandas as pd

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 5. Stratified Split

When dealing with classification problems, especially with imbalanced classes, we want the training and test sets to have roughly the same class proportions as the original dataset. This is called **stratified splitting**.

In `train_test_split`, set `stratify=y`:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Now the percentage of each class in train and test will match the original.

---

## 6. Splitting Time Series Data

For time series, random splitting would leak future information into the training set. Instead, we must split **chronologically**: train on past, test on future.

```python
# Assume data is sorted by date
split_index = int(0.8 * len(data))
train = data[:split_index]
test = data[split_index:]
```

No `train_test_split` here – use simple slicing.

---

## 7. The Need for a Validation Set

In practice, we often need to tune model hyperparameters (like learning rate, regularization strength). If we use the test set for tuning, we risk overfitting to the test set. The solution: create a **validation set** (also called development set).

Common split: 60% train, 20% validation, 20% test.

We can do this with two splits:

```python
# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Then split temp into train and val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2 of total
```

Alternatively, scikit-learn's `train_test_split` can't do three-way directly, but this two-step method works.

---

## 8. Cross-Validation: A More Robust Approach

Instead of a single split, **k-fold cross-validation** divides data into k folds, trains on k-1 folds, and validates on the remaining fold, repeating k times. This gives a more stable estimate of model performance.

We'll cover cross-validation in detail later, but here's a quick preview:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

---

## 9. Practical Example: Complete Workflow

Let's put it all together with a real dataset (Iris) and a simple classifier.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Further split temp into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)  # 0.25*0.8=0.2

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# Try different k values on validation set
best_k = 1
best_acc = 0
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"k={k}, val accuracy={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

# Train final model with best k on full training+validation data
X_train_full = X_temp  # combined train+val
y_train_full = y_temp
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_full, y_train_full)

# Evaluate once on test set
y_test_pred = final_knn.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nBest k={best_k}, Test accuracy={test_acc:.4f}")
```

This workflow ensures we never use the test set for decision-making.

---

## 10. Common Pitfalls

- **Data leakage**: Accidentally using test data during training (e.g., scaling before split).
- **Not shuffling**: If data has order, random split without shuffling might create unrepresentative sets. `train_test_split` shuffles by default.
- **Using test set repeatedly**: If you evaluate multiple models on the test set and pick the best, you're effectively using test data for model selection, invalidating its purpose.

---

## 11. Summary

| Concept | Key Takeaway |
|---------|--------------|
| Training set | Data used to learn model parameters |
| Validation set | Data used to tune hyperparameters |
| Test set | Final, held-out data for unbiased evaluation |
| train_test_split | Scikit-learn function for random splitting |
| stratify | Ensures class proportions are preserved |
| Time series split | Must preserve temporal order |

---

## 12. Practice Tasks

1. **Basic split**: Generate a synthetic dataset with 200 samples and 5 features. Split into 70% train, 30% test using `train_test_split`. Print the shapes.

2. **Stratification**: Create an imbalanced dataset (e.g., 90% class 0, 10% class 1). Split with and without stratification. Compare the class proportions in train and test sets.

3. **Three-way split**: Using the Iris dataset, create a 60-20-20 train/validation/test split. Train a simple model (e.g., logistic regression) on the training set, tune one hyperparameter (e.g., C value) using validation, then evaluate on test.

4. **Time series split**: Load a time series dataset (e.g., stock prices) or create a date-ordered dataset. Write code to split it chronologically into train (first 80%) and test (last 20%).

5. **Cross-validation preview**: Use `cross_val_score` with 5 folds on the Iris dataset and a KNN classifier. Compare the mean accuracy with a single train/test split. Which estimate do you trust more?

6. **Experiment with split ratios**: For a dataset of your choice, try different train/test ratios (e.g., 50/50, 90/10). How does the performance estimate vary? (Keep in mind that small test sets give high variance estimates.)

---

You've now mastered the essential first step of any machine learning project: properly splitting data. Next, we'll dive into **linear regression**, one of the simplest yet most powerful models. Let me know when you're ready to proceed!
