# Day 17–18: Logistic Regression – From Linear to Classification

Welcome to the next step in your machine learning journey! Today we're moving from predicting continuous values (regression) to predicting **categories** – that's **classification**. And the go-to algorithm for binary classification is **logistic regression**.

Despite its name, logistic regression is a **classification** algorithm. It's simple, interpretable, and forms the foundation for many advanced techniques (like neural networks). We'll cover:

- The intuition behind logistic regression.
- The sigmoid function and probability estimation.
- Decision boundaries.
- Training with log loss (cross-entropy).
- Evaluation metrics for classification.
- Multiclass logistic regression (softmax).
- Implementation with scikit-learn.
- Regularization.
- Practical examples.

Let's dive in!

---

## 1. What is Logistic Regression?

Logistic regression is used when the target variable is **categorical**. Most commonly, it handles **binary classification** (two classes, e.g., spam/not spam, disease/no disease). It estimates the probability that an instance belongs to a particular class.

The name "regression" comes from its linear model core: it still computes a weighted sum of input features, but then passes that sum through a special function – the **sigmoid** – to squash the output into a probability between 0 and 1.

---

## 2. The Sigmoid Function

The sigmoid (or logistic) function is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

It takes any real number and maps it to a value between 0 and 1.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axvline(x=0, color='gray', linestyle='--')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.show()
```

**Interpretation**: 
- When \(z=0\), \(\sigma(z)=0.5\).
- Large positive \(z\) → σ close to 1.
- Large negative \(z\) → σ close to 0.

In logistic regression, \(z\) is the linear combination: \(z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p\).

Thus, the model predicts:

\[
P(y=1 | X) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p)}}
\]

---

## 3. Decision Boundary

We typically classify an instance as class 1 if the predicted probability is greater than 0.5, otherwise class 0. The **decision boundary** is where \(P(y=1) = 0.5\), i.e., \(z=0\):

\[
\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p = 0
\]

This is a linear boundary in feature space – hence the name "logistic regression" (linear decision boundary). For more complex boundaries, we can add polynomial features.

---

## 4. Training: Maximizing Likelihood / Minimizing Log Loss

Instead of squared error (used in linear regression), logistic regression uses **maximum likelihood estimation**. Intuitively, we want to choose parameters that make the observed data most probable.

The loss function minimized is **log loss** (also called cross-entropy):

\[
J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
\]

- If true class is 1, we want \(\hat{p}_i\) close to 1 (so \(-\log(\hat{p}_i)\) small).
- If true class is 0, we want \(\hat{p}_i\) close to 0 (so \(-\log(1-\hat{p}_i)\) small).

There is no closed-form solution, so we use iterative optimization like **gradient descent**.

---

## 5. Implementation with scikit-learn

Scikit-learn makes logistic regression easy.

### Binary Classification Example: Titanic Survival Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Titanic dataset (simplified)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Quick preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = (df['Sex'] == 'male').astype(int)  # male=1, female=0
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train model
model = LogisticRegression(max_iter=1000)  # increase max_iter if needed
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probabilities for class 1

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Coefficients
coeff_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0]
})
print(coeff_df.sort_values('coefficient', ascending=False))
```

**Interpretation**: Positive coefficients increase the log-odds of survival (i.e., increase probability). For example, being male (Sex=1) has a negative coefficient, meaning lower survival probability.

---

## 6. Evaluation Metrics for Classification

Accuracy alone can be misleading, especially with imbalanced classes. Important metrics:

- **Confusion Matrix**: TP, TN, FP, FN.
- **Precision**: TP / (TP + FP) – of predicted positives, how many are correct.
- **Recall (Sensitivity)**: TP / (TP + FN) – of actual positives, how many are found.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve – measures ability to distinguish between classes.

```python
from sklearn.metrics import roc_auc_score, roc_curve

# ROC AUC
auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {auc:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'Logistic (AUC = {auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 7. Multiclass Logistic Regression

Logistic regression naturally extends to multiple classes via **softmax regression** (multinomial logistic regression). Instead of sigmoid, we use the softmax function, which gives probabilities for each class that sum to 1.

In scikit-learn, just set `multi_class='multinomial'` (default is 'auto' which picks multinomial for multiclass). Also use a solver that supports multinomial (e.g., 'lbfgs').

### Example: Iris Dataset (3 classes)

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 8. Regularization in Logistic Regression

Like linear regression, logistic regression can be regularized to prevent overfitting:

- **L2 regularization** (Ridge): Adds penalty \(\alpha \sum \beta_j^2\) (default in scikit-learn).
- **L1 regularization** (Lasso): Adds penalty \(\alpha \sum |\beta_j|\) (can shrink some coefficients to zero).

In scikit-learn, the `penalty` parameter controls this: `penalty='l2'` (default), `penalty='l1'` (requires solver 'liblinear' or 'saga'). The strength is controlled by `C` (inverse of regularization strength; smaller C = stronger regularization).

```python
# L1 regularized logistic regression
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model_l1.fit(X_train, y_train)
```

---

## 9. Advantages and Disadvantages

| Pros | Cons |
|------|------|
| Simple, fast, interpretable | Assumes linear decision boundary |
| Outputs well-calibrated probabilities | Can underfit with complex relationships |
| Works well with small datasets | Sensitive to outliers |
| Easy to regularize | Requires careful feature engineering |

---

## 10. Practical Example: Predicting Customer Churn

Let's simulate a churn dataset and build a logistic regression model.

```python
# Generate synthetic churn data
np.random.seed(42)
n_samples = 1000
tenure = np.random.randint(1, 72, n_samples)  # months
monthly_charges = np.random.uniform(30, 120, n_samples)
contract_type = np.random.choice([0,1], n_samples)  # 0=month-to-month, 1=yearly
churn_prob = 1 / (1 + np.exp(-(-3 + 0.02*tenure - 0.5*monthly_charges/100 + 2*contract_type)))
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

# Create DataFrame
df_churn = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'contract_type': contract_type,
    'churn': churn
})

# Split
X = df_churn[['tenure', 'monthly_charges', 'contract_type']]
y = df_churn['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

## 11. Summary

| Concept | Key Idea |
|---------|----------|
| Sigmoid | Maps linear output to probability |
| Decision boundary | Linear surface where P=0.5 |
| Log loss (cross-entropy) | Loss function minimized during training |
| Coefficients | Interpret as log-odds change |
| Regularization | Prevents overfitting (L1, L2) |
| Multiclass | Softmax extension |
| Evaluation | Accuracy, confusion matrix, precision/recall, ROC-AUC |

---

## 12. Practice Tasks

1. **Binary Classification from Scratch (Conceptual)**
   - Write a simple function that computes the sigmoid.
   - Explain in your own words why we can't use squared error for logistic regression.

2. **Titanic Survival**
   - Load the Titanic dataset (as above).
   - Experiment with different feature combinations.
   - Try adding interaction terms or polynomial features. Does accuracy improve?
   - Compare with L1 and L2 regularization.

3. **ROC Curve Analysis**
   - For the Titanic model, plot the ROC curve.
   - Find the threshold that maximizes the F1-score (you can vary the decision threshold manually).

4. **Multiclass on Iris**
   - Use logistic regression on Iris.
   - Print the coefficients for each class (3 sets). Interpret which features are most important for each species.

5. **Imbalanced Data**
   - Create an imbalanced dataset (e.g., 95% class 0, 5% class 1).
   - Train a logistic regression model and evaluate accuracy. Why is accuracy misleading?
   - Use `class_weight='balanced'` in LogisticRegression and observe the change in precision/recall.

6. **Mini Project: Predict Loan Default**
   - Find a dataset on loan default (e.g., from Kaggle) or create a synthetic one.
   - Preprocess (handle missing values, encode categorical).
   - Build a logistic regression model.
   - Evaluate using AUC and confusion matrix.
   - Interpret the coefficients to identify risk factors.

---

You've now mastered logistic regression – a fundamental tool for classification. Next, we'll explore **evaluation metrics and model selection** in more depth, or move to **decision trees**. Let me know your preference!
