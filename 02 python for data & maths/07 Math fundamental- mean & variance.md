# Day 13: Math Fundamentals – Mean and Variance

Welcome to Day 13! Today we're diving into essential statistical concepts that form the backbone of many machine learning algorithms and data preprocessing steps: **mean** and **variance**. Understanding these measures is crucial for:

- **Data normalization**: Centering and scaling features.
- **Evaluating model performance**: Mean Squared Error (MSE) measures average squared deviation.
- **Understanding data distributions**: Spread and central tendency.
- **Feature engineering**: Creating statistical features.

We'll cover:
- What mean and variance represent.
- Formulas and interpretation.
- Computing them with Python (NumPy, pandas).
- Sample vs. population variance.
- Standard deviation.
- Practical AI examples.

Let's get started!

---

## 1. Mean (Average)

The **mean** (or average) is the most common measure of central tendency. It represents the "typical" value in a dataset.

### Definition
For a dataset of \(n\) values \(x_1, x_2, \ldots, x_n\), the mean \(\bar{x}\) is:

\[
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
\]

### Interpretation
- The mean is the balancing point of the data.
- It is sensitive to outliers (extreme values can pull the mean significantly).

### Computing Mean in Python

```python
import numpy as np
import pandas as pd

# Simple list
data = [10, 20, 30, 40, 50]
mean = np.mean(data)
print(f"Mean: {mean}")  # 30.0

# With NumPy array
arr = np.array([2, 4, 6, 8, 10])
print(np.mean(arr))      # 6.0

# With pandas Series
s = pd.Series([1, 2, 3, 4, 5])
print(s.mean())          # 3.0

# With pandas DataFrame (column-wise)
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
print(df.mean())         # A: 2.0, B: 5.0
```

---

## 2. Variance

**Variance** measures the spread or dispersion of data points around the mean. A high variance means data points are spread out; low variance means they are clustered closely.

### Population Variance
If you have the entire population (all possible data), variance \(\sigma^2\) is:

\[
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
\]
where \(\mu\) is the population mean, \(N\) is population size.

### Sample Variance
In practice, we often work with a sample from a larger population. To get an unbiased estimate of the population variance, we use:

\[
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
\]
where \(n\) is the sample size, \(\bar{x}\) is the sample mean. The denominator \(n-1\) is called **Bessel's correction**, which corrects the bias in estimating population variance from a sample.

### Interpretation
- Variance is in squared units of the original data.
- Larger variance → more spread.

### Computing Variance in Python

```python
data = [10, 20, 30, 40, 50]

# Population variance (ddof=0)
var_pop = np.var(data, ddof=0)
print(f"Population variance: {var_pop}")  # 200.0

# Sample variance (ddof=1, default in many contexts)
var_sample = np.var(data, ddof=1)
print(f"Sample variance: {var_sample}")   # 250.0

# Using pandas (default ddof=1 for sample variance)
s = pd.Series(data)
print(s.var())  # 250.0

# For DataFrame columns
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
print(df.var())  # A: 1.0, B: 1.0
```

**Note:** `np.var` defaults to `ddof=0` (population). In statistics and machine learning, we often use sample variance, so specify `ddof=1`. pandas `var` uses `ddof=1` by default.

---

## 3. Standard Deviation

The **standard deviation** is the square root of variance. It brings the measure back to the original units, making interpretation easier.

\[
\sigma = \sqrt{\sigma^2} \quad \text{(population)}, \quad s = \sqrt{s^2} \quad \text{(sample)}
\]

### Computing Standard Deviation

```python
data = [10, 20, 30, 40, 50]

# Population std
std_pop = np.std(data, ddof=0)
print(f"Population std: {std_pop}")  # 14.142...

# Sample std
std_sample = np.std(data, ddof=1)
print(f"Sample std: {std_sample}")   # 15.811...

# pandas
s = pd.Series(data)
print(s.std())  # 15.811... (sample std)
```

---

## 4. Why Mean and Variance Matter in AI

### 4.1 Feature Scaling
Many machine learning algorithms perform better when features are on a similar scale. Two common techniques:
- **Standardization (Z-score normalization)**: \( z = \frac{x - \mu}{\sigma} \). This transforms data to have mean 0 and variance 1.
- **Min-Max scaling**: uses min and max, not mean/variance.

```python
# Standardization example
data = np.array([10, 20, 30, 40, 50])
mean = np.mean(data)
std = np.std(data, ddof=0)  # or ddof=1 depending on context
z_scores = (data - mean) / std
print(f"Z-scores: {z_scores}")
print(f"New mean: {np.mean(z_scores):.2f}")   # ~0
print(f"New variance: {np.var(z_scores):.2f}") # ~1
```

### 4.2 Loss Functions
- **Mean Squared Error (MSE)** is the average of squared differences between predictions and targets. It's directly related to variance (if predictions are unbiased, MSE equals variance of predictions plus squared bias).

### 4.3 Understanding Data Distributions
- Variance helps detect outliers (data points far from mean).
- In exploratory data analysis (EDA), we examine mean and variance across groups.

### 4.4 Assumptions in Models
- Some models (e.g., Linear Discriminant Analysis) assume equal variance across classes.
- In Bayesian inference, we often assume prior distributions with specified mean and variance.

---

## 5. Practical Example: Analyzing a Real Dataset

Let's load the Iris dataset and compute mean and variance for each feature, then standardize.

```python
import pandas as pd
import numpy as np

# Load Iris dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=columns)

# Compute mean and variance for numeric columns
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("Means:")
print(iris[numeric_cols].mean())
print("\nVariances (sample):")
print(iris[numeric_cols].var())

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris[numeric_cols])
iris_scaled_df = pd.DataFrame(iris_scaled, columns=numeric_cols)
print("\nAfter standardization - means (approx 0):")
print(iris_scaled_df.mean())
print("\nAfter standardization - variances (approx 1):")
print(iris_scaled_df.var())
```

**Output observation:** After scaling, means are very close to 0 and variances close to 1.

---

## 6. Visualizing Mean and Variance

We can use Matplotlib to see how mean and variance relate to data spread.

```python
import matplotlib.pyplot as plt

# Generate three datasets with same mean but different variance
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)    # mean 0, std 1
data2 = np.random.normal(0, 2, 1000)    # mean 0, std 2
data3 = np.random.normal(0, 0.5, 1000)  # mean 0, std 0.5

plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, alpha=0.5, label='std=1')
plt.hist(data2, bins=30, alpha=0.5, label='std=2')
plt.hist(data3, bins=30, alpha=0.5, label='std=0.5')
plt.axvline(x=0, color='black', linestyle='--', label='Mean')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distributions with Same Mean (0) but Different Variances')
plt.legend()
plt.show()
```

This plot shows that variance controls the spread around the mean.

---

## 7. Important Notes

- **Outliers** affect both mean and variance significantly. Median and interquartile range are robust alternatives.
- For machine learning, when computing variance for standardization, use **population variance** (ddof=0) if you're standardizing the entire dataset; but if you're estimating from a sample to apply to new data, use sample variance (ddof=1) for unbiased estimation. Scikit-learn's `StandardScaler` uses `ddof=0` by default (population variance).
- Variance is always non-negative. Zero variance means all values are identical.

---

## 8. Summary

| Concept | Formula | Python (NumPy) | Python (pandas) |
|---------|---------|----------------|------------------|
| Mean | \(\frac{1}{n}\sum x_i\) | `np.mean(x)` | `x.mean()` |
| Population variance | \(\frac{1}{N}\sum (x_i - \mu)^2\) | `np.var(x, ddof=0)` | `x.var(ddof=0)` |
| Sample variance | \(\frac{1}{n-1}\sum (x_i - \bar{x})^2\) | `np.var(x, ddof=1)` | `x.var()` (default) |
| Population std | \(\sqrt{\text{pop variance}}\) | `np.std(x, ddof=0)` | `x.std(ddof=0)` |
| Sample std | \(\sqrt{\text{sample variance}}\) | `np.std(x, ddof=1)` | `x.std()` (default) |

---

## 9. Practice Tasks

1. **Basic Computations**
   - Create an array of 20 random integers between 1 and 100.
   - Compute its mean, population variance, sample variance, and standard deviation.
   - Verify that population variance < sample variance (since denominator smaller for sample).

2. **Effect of Outliers**
   - Take the array [10, 12, 11, 13, 12, 100].
   - Compute mean and variance.
   - Remove the outlier 100, recompute, and observe the change.

3. **Standardization**
   - Generate 1000 random numbers from a normal distribution with mean 50 and std 10.
   - Standardize them (z-scores) manually using mean and std.
   - Verify the new mean is ~0 and std ~1.

4. **Real Data Exploration**
   - Load the Boston housing dataset (or any dataset) using `sklearn.datasets` or pandas.
   - For each numeric column, compute mean and variance.
   - Identify which feature has the highest variance – what does that tell you?
   - Standardize the features and verify the results.

5. **Visualization**
   - Create two datasets: one with low variance (e.g., mean 10, std 1) and one with high variance (mean 10, std 5).
   - Plot histograms of both on the same figure.
   - Add vertical lines for the means.

---

You now have a solid grasp of mean and variance – fundamental concepts that appear everywhere in AI. Next, we could dive deeper into **probability distributions** or move into **linear algebra review** for machine learning. Let me know your preference!
