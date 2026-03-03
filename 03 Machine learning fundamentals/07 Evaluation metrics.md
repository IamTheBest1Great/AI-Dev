# Day 25: Evaluation Metrics – Measuring Classification Performance

Welcome to Day 25! After building classification models like Logistic Regression, KNN, Decision Trees, and Random Forest, we need to answer a crucial question: **How good is our model?** Evaluation metrics help us quantify performance and guide model selection.

Today we'll focus on the essential metrics for classification:
- **Confusion Matrix** – the foundation.
- **Accuracy** – overall correctness.
- **Precision** – how many predicted positives are actually positive.
- **Recall** (Sensitivity) – how many actual positives are captured.
- **F1-Score** – harmonic mean of precision and recall.

We'll cover:
- Definitions and formulas.
- Intuitive interpretations.
- When to use each metric.
- Python implementation with scikit-learn.
- The precision-recall trade-off.
- Practice tasks.

Let's dive in!

---

## 1. Why Evaluation Metrics Matter

A model's performance isn't just about "did it guess right?" Different problems require different measures. For example:
- In spam detection, we want to avoid marking important emails as spam (high precision).
- In cancer screening, we want to catch as many actual cancers as possible (high recall), even if it means some false alarms.

Understanding these metrics helps you choose the right model and tune it for your specific needs.

---

## 2. Confusion Matrix

The confusion matrix is a table that summarizes the performance of a classification model. For binary classification, it has four entries:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| Actual Positive| True Positive (TP) | False Negative (FN)|
| Actual Negative| False Positive (FP)| True Negative (TN) |

- **True Positive (TP)**: Correctly predicted positive class.
- **True Negative (TN)**: Correctly predicted negative class.
- **False Positive (FP)**: Incorrectly predicted positive (Type I error).
- **False Negative (FN)**: Incorrectly predicted negative (Type II error).

From these four numbers, we derive all other metrics.

### Example in Python
```python
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# Output:
# [[2 1]   (TN=2, FP=1)
#  [1 4]]  (FN=1, TP=4)
```

You can also visualize it with a heatmap.

---

## 3. Accuracy

**Accuracy** is the simplest metric: the proportion of correct predictions among total predictions.

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")  # 0.75
```

### When to Use Accuracy
- When classes are **balanced** (roughly equal number of samples per class).
- When all misclassifications have **equal cost**.

### Limitation
Accuracy can be misleading with **imbalanced datasets**. For example, if 95% of samples are negative, a model that always predicts negative will have 95% accuracy but is useless for detecting positives.

---

## 4. Precision

**Precision** answers: "Of all instances predicted as positive, how many are actually positive?"

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

- High precision means few false positives.
- Also called **Positive Predictive Value (PPV)**.

```python
from sklearn.metrics import precision_score

prec = precision_score(y_true, y_pred)
print(f"Precision: {prec:.2f}")  # 0.80 (4 TP / (4 TP + 1 FP))
```

### When to Focus on Precision
- When false positives are costly (e.g., spam detection: marking legitimate email as spam is bad).
- In medical testing: a positive result leads to further investigation, so we want high precision to avoid unnecessary stress and procedures.

---

## 5. Recall (Sensitivity)

**Recall** answers: "Of all actual positive instances, how many did we correctly identify?"

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

- High recall means few false negatives.
- Also called **Sensitivity**, **True Positive Rate (TPR)**.

```python
from sklearn.metrics import recall_score

rec = recall_score(y_true, y_pred)
print(f"Recall: {rec:.2f}")  # 0.80 (4 TP / (4 TP + 1 FN))
```

### When to Focus on Recall
- When false negatives are costly (e.g., cancer screening: missing a cancer is far worse than a false alarm).
- In fraud detection: we want to catch as many fraudulent transactions as possible.

---

## 6. F1-Score

**F1-score** is the harmonic mean of precision and recall, providing a single score that balances both.

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

- Ranges from 0 to 1. High F1 means both precision and recall are high.
- Harmonic mean penalizes extreme values more than arithmetic mean.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-score: {f1:.2f}")  # 0.80
```

### When to Use F1-Score
- When you need a balance between precision and recall, especially with imbalanced classes.
- When you want a single metric to compare models.

---

## 7. The Precision-Recall Trade-off

There's often a tension: increasing precision tends to decrease recall, and vice versa. This is controlled by the **decision threshold** of the classifier.

- By default, many classifiers use 0.5 as threshold for positive class.
- Lowering the threshold increases recall (catch more positives) but may decrease precision (more false positives).
- Raising the threshold does the opposite.

We can plot **Precision-Recall curve** to visualize this trade-off.

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assume we have a classifier with predict_proba
y_scores = model.predict_proba(X_test)[:, 1]  # probabilities for positive class
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

The area under this curve (PR-AUC) is another useful metric.

---

## 8. Putting It All Together – A Complete Example

Let's use the Titanic dataset to compute all metrics.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Load and preprocess Titanic
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = (df['Sex'] == 'male').astype(int)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Compute metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))
```

---

## 9. Multi-Class Metrics

For multi-class problems, we can compute these metrics per class and then average:

- **Macro-average**: Compute metric for each class independently and average (unweighted).
- **Micro-average**: Aggregate TP, FP, FN across all classes and compute metric.
- **Weighted-average**: Average per-class metric weighted by number of true instances.

Scikit-learn's `classification_report` shows all three.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 10. Summary Table

| Metric | Formula | Focus | When to Use |
|--------|---------|-------|-------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | Balanced classes |
| Precision | TP/(TP+FP) | Accuracy of positive predictions | When FP cost is high |
| Recall | TP/(TP+FN) | Coverage of actual positives | When FN cost is high |
| F1-score | 2*(Prec*Recall)/(Prec+Recall) | Balance between precision and recall | Imbalanced classes, need single metric |
| Confusion Matrix | Table of TP,TN,FP,FN | Detailed breakdown | Always useful for diagnosis |

---

## 11. Practice Tasks

1. **Manual Calculation**
   - Given a confusion matrix with TP=50, TN=40, FP=10, FN=5, compute accuracy, precision, recall, and F1-score.

2. **Imbalanced Dataset**
   - Create an imbalanced dataset (e.g., 90% class 0, 10% class 1). Train a dummy classifier that always predicts 0. Compute accuracy, precision, recall. Why is accuracy misleading?

3. **Threshold Tuning**
   - Train a logistic regression on a binary dataset. Vary the decision threshold from 0.1 to 0.9 and plot precision and recall vs threshold. Find the threshold that maximizes F1-score.

4. **Multi-Class Metrics**
   - Use the Iris dataset. Train a classifier and compute precision, recall, and F1 for each class manually (using formulas) and verify with scikit-learn.

5. **Precision-Recall Curve**
   - Generate a synthetic dataset using `make_classification` with some overlap between classes. Plot the precision-recall curve and compute the area under the curve (using `auc` from sklearn.metrics).

6. **Mini Project: Compare Models**
   - Choose a dataset (e.g., Pima Indians Diabetes from UCI). Split into train/test. Train at least three different classifiers (Logistic Regression, KNN, Random Forest). For each, compute accuracy, precision, recall, F1, and confusion matrix. Write a short paragraph comparing their performance and recommending which metric is most important for this problem (e.g., is FN worse than FP?).

---

You now have a solid understanding of the core classification metrics. These will be your tools for evaluating and improving models throughout your AI journey. Next, we could dive into **ROC curves and AUC** or move to **cross-validation techniques**. Let me know your preference!
