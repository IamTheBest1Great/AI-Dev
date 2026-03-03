# Day 15–16: Linear Regression – The Foundation of Predictive Modeling

Welcome to the core of machine learning! **Linear regression** is one of the simplest yet most powerful algorithms. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data. Despite its simplicity, it forms the basis for many advanced techniques and is widely used in forecasting, trend analysis, and understanding feature importance.

We'll cover:
- The intuition behind linear regression.
- Simple linear regression (one feature).
- Multiple linear regression.
- The math behind Ordinary Least Squares (conceptually).
- Implementing linear regression with scikit-learn.
- Evaluating model performance.
- Assumptions and diagnostics.
- Extending to polynomial regression.
- Regularization (brief introduction).

Let's get started!

---

## 1. What is Linear Regression?

Linear regression tries to answer: **"Can we predict a continuous value (like house price, temperature, sales) based on other variables?"**

It assumes a linear relationship between the input features and the target. For example, we might model house price as a combination of size, number of bedrooms, and location.

The goal is to find the "best" line (or hyperplane in higher dimensions) that minimizes the error between predictions and actual values.

---

## 2. Simple Linear Regression

### The Equation

For a single feature \(x\), the model is:

\[
y = \beta_0 + \beta_1 x + \varepsilon
\]

Where:
- \(y\) is the target (dependent variable).
- \(x\) is the feature (independent variable).
- \(\beta_0\) is the **intercept** (value of \(y\) when \(x=0\)).
- \(\beta_1\) is the **slope** (change in \(y\) per unit change in \(x\)).
- \(\varepsilon\) is the error term (unexplained variation).

We estimate \(\beta_0\) and \(\beta_1\) from the data.

### Interpretation
- If \(\beta_1 > 0\), as \(x\) increases, \(y\) increases.
- If \(\beta_1 < 0\), as \(x\) increases, \(y\) decreases.
- The magnitude of \(\beta_1\) tells us how strong the effect is.

### Example: Predicting Salary from Years of Experience
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
years_exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
salary = np.array([30, 35, 38, 42, 48, 53, 60, 65, 70, 78])

# Create and fit model
model = LinearRegression()
model.fit(years_exp, salary)

# Coefficients
print(f"Intercept (β0): {model.intercept_:.2f}")
print(f"Slope (β1): {model.coef_[0]:.2f}")

# Predict for new value
pred = model.predict([[12]])
print(f"Predicted salary for 12 years: {pred[0]:.2f}")

# Plot
plt.scatter(years_exp, salary, color='blue', label='Actual')
plt.plot(years_exp, model.predict(years_exp), color='red', label='Fitted line')
plt.xlabel('Years Experience')
plt.ylabel('Salary (k$)')
plt.legend()
plt.show()
```

---

## 3. Multiple Linear Regression

When we have multiple features, the model extends to:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon
\]

Each \(\beta_j\) represents the expected change in \(y\) for a one-unit change in \(x_j\), holding all other features constant.

### Example: Predicting House Price with Size and Bedrooms
```python
from sklearn.linear_model import LinearRegression

# Features: size (sq ft), bedrooms
X = np.array([[1500, 3],
              [1800, 4],
              [1200, 2],
              [2000, 4],
              [1600, 3]])
y = np.array([300, 360, 240, 400, 320])  # price in k$

model = LinearRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")
print(f"Size coefficient: {model.coef_[0]:.2f} (price per sq ft)")
print(f"Bedrooms coefficient: {model.coef_[1]:.2f} (price per bedroom)")
```

---

## 4. How Linear Regression Learns: Ordinary Least Squares (OLS)

The most common method to find the best coefficients is **Ordinary Least Squares**. It minimizes the sum of squared residuals:

\[
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

where \(\hat{y}_i\) is the predicted value. The coefficients are computed using linear algebra (the normal equation):

\[
\hat{\beta} = (X^T X)^{-1} X^T y
\]

In practice, scikit-learn handles this for us.

---

## 5. Assumptions of Linear Regression

For valid inference, linear regression relies on several assumptions:

1. **Linearity**: The relationship between features and target is linear.
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: Constant variance of errors across all levels of features.
4. **Normality**: Errors are normally distributed (mainly for hypothesis testing, less critical for prediction).
5. **No multicollinearity**: Features are not highly correlated with each other.

Violations can lead to biased estimates or poor predictions.

---

## 6. Model Evaluation Metrics

### R-squared (Coefficient of Determination)
Proportion of variance in the target explained by the model:

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

- Ranges from 0 to 1 (higher is better).
- Can be artificially inflated by adding irrelevant features (adjusted R-squared corrects for this).

### Mean Squared Error (MSE)
Average of squared errors:

\[
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
\]

### Root Mean Squared Error (RMSE)
Square root of MSE – in same units as target.

### Mean Absolute Error (MAE)
Average absolute error:

\[
MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
\]

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
```

---

## 7. Overfitting and Underfitting in Linear Regression

- **Underfitting**: Model too simple (e.g., using only one feature when relationship is complex). High bias, low variance.
- **Overfitting**: Model too complex (e.g., adding many irrelevant features or polynomial terms). Low bias, high variance.

We use train/test split and cross-validation to detect these.

---

## 8. Polynomial Regression – Extending Linearity

If the relationship is curved, we can still use linear regression by adding polynomial features. For example:

\[
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon
\]

This is still linear in the coefficients, so we can use the same method.

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit linear regression on polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
```

Be careful: higher-degree polynomials can easily overfit.

---

## 9. Regularization – Preventing Overfitting

When we have many features, coefficients can become large and unstable. **Regularization** adds a penalty to the loss function to shrink coefficients.

- **Ridge Regression (L2)**: Adds penalty \(\alpha \sum \beta_j^2\).
- **Lasso Regression (L1)**: Adds penalty \(\alpha \sum |\beta_j|\) (can shrink some to zero, useful for feature selection).

We'll cover these in more detail later.

---

## 10. Practical Example: Boston Housing Dataset

Let's apply linear regression to the Boston Housing dataset (a classic).

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
print(f"R² on test: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# Coefficients
coeff_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_
})
print(coeff_df.sort_values('coefficient', ascending=False))
```

---

## 11. Summary

| Concept | Key Idea |
|---------|----------|
| Simple linear regression | One feature, fits a line |
| Multiple linear regression | Multiple features, fits a hyperplane |
| OLS | Minimizes sum of squared errors |
| R² | Proportion of variance explained |
| MSE / RMSE | Average squared error (RMSE in original units) |
| MAE | Average absolute error |
| Polynomial regression | Adds polynomial terms to capture curvature |
| Regularization | Shrinks coefficients to prevent overfitting |

---

## 12. Practice Tasks

1. **Simple Linear Regression**
   - Generate synthetic data with `y = 3x + 5 + noise` (use `np.random.randn` for noise).
   - Fit a linear regression model.
   - Print coefficients and compare with true values.
   - Plot data and regression line.

2. **Multiple Linear Regression**
   - Load the California housing dataset (from `sklearn.datasets.fetch_california_housing`).
   - Split into train/test.
   - Fit a linear regression model.
   - Evaluate using R² and RMSE.
   - Interpret the coefficients: which features affect price most?

3. **Polynomial Regression**
   - Create data following `y = x² + noise`.
   - Fit a simple linear regression (degree 1) and a polynomial regression (degree 2).
   - Compare their performance on test data.
   - Plot both fits.

4. **Effect of Outliers**
   - Create a small dataset with one obvious outlier.
   - Fit linear regression with and without the outlier.
   - Observe how coefficients change. (This shows sensitivity to outliers.)

5. **Feature Scaling**
   - Does linear regression require feature scaling? (Answer: no, but it can help with interpretation and regularization.)
   - Experiment: fit linear regression on unscaled vs scaled features. Compare coefficients.

6. **Mini Project: Predict Student Performance**
   - Find a dataset (e.g., Student Performance from UCI or Kaggle) with a continuous target (e.g., final grade).
   - Perform exploratory analysis.
   - Build a linear regression model.
   - Evaluate and interpret.

---

You've now mastered linear regression – the gateway to more complex models. Next, we'll explore **classification** with logistic regression. Let me know when you're ready to proceed!
