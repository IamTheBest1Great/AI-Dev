# Day 12: Data Visualization with Matplotlib

Welcome to Day 12! Today we dive into **data visualization** using Matplotlib, the most widely used plotting library in Python. Visualization is crucial in AI and data science for:
- **Exploring data** – understanding distributions, patterns, and outliers.
- **Debugging models** – monitoring training curves, evaluating predictions.
- **Communicating results** – presenting findings clearly to stakeholders.

Matplotlib is highly customizable and integrates seamlessly with NumPy and pandas. We'll cover:
- Installing and importing Matplotlib.
- Basic plot types: line, scatter, bar, histogram.
- Customizing plots: labels, titles, legends, colors, styles.
- Creating subplots.
- Plotting directly from pandas DataFrames.
- Saving figures.
- Practical examples relevant to AI.

Let's get started!

---

## 1. What is Matplotlib?

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It was inspired by MATLAB's plotting capabilities. The main module is `matplotlib.pyplot`, a collection of functions that make matplotlib work like MATLAB.

**Why for AI?**  
- Visualize datasets to understand feature distributions and relationships.
- Plot loss and accuracy curves during model training.
- Display images, confusion matrices, and feature maps.
- Create publication-quality figures for reports.

---

## 2. Installation

Make sure your virtual environment is active, then:

```bash
pip install matplotlib
```

Import convention:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

---

## 3. Basic Plot Types

### Line Plot
Line plots are ideal for showing trends over time or continuous data.

```python
# Simple line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.show()
```

### Scatter Plot
Scatter plots show relationships between two continuous variables.

```python
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y)
plt.show()
```

### Bar Plot
Bar plots compare categorical data.

```python
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]

plt.bar(categories, values)
plt.show()
```

### Histogram
Histograms show the distribution of a single variable.

```python
data = np.random.randn(1000)  # 1000 random numbers from normal distribution
plt.hist(data, bins=30, edgecolor='black')
plt.show()
```

---

## 4. Customizing Plots

Adding labels, titles, legends, and colors makes plots informative.

```python
# Data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create plot
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='--', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linestyle='-', linewidth=2)

# Add labels and title
plt.xlabel('X axis (time)')
plt.ylabel('Y axis (value)')
plt.title('Sine and Cosine Functions')

# Add legend
plt.legend()

# Add grid
plt.grid(True, alpha=0.3)

# Show plot
plt.show()
```

**Common customization options:**
- `color`: 'red', 'blue', 'green', '#FF5733' (hex), etc.
- `linestyle`: '-', '--', '-.', ':'
- `marker`: 'o', 's', '^', 'D' for scatter points
- `alpha`: transparency (0 to 1)

---

## 5. Working with Figures and Axes

Matplotlib has two interfaces:
- **pyplot (state-based)**: simple, good for quick plots.
- **Object-oriented (OO)**: more control, recommended for complex plots.

### Using OO Interface
```python
# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))  # width, height in inches

# Plot on axes
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
ax.plot(x, y, marker='o')

# Customize
ax.set_title('Sales Over Time')
ax.set_xlabel('Quarter')
ax.set_ylabel('Sales (thousands)')
ax.grid(True)

plt.show()
```

---

## 6. Subplots

Display multiple plots in a single figure.

```python
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten axes for easy indexing (optional)
axes = axes.flatten()

# Data
x = np.linspace(0, 5, 100)

# Plot on each subplot
axes[0].plot(x, np.sin(x))
axes[0].set_title('sin(x)')

axes[1].plot(x, np.cos(x))
axes[1].set_title('cos(x)')

axes[2].plot(x, np.exp(-x))
axes[2].set_title('exp(-x)')

axes[3].plot(x, x**2)
axes[3].set_title('x^2')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
```

---

## 7. Plotting with Pandas DataFrames

Pandas integrates with Matplotlib, allowing quick plots directly from DataFrames.

```python
# Sample DataFrame
df = pd.DataFrame({
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Sales': [100, 120, 110, 150, 180],
    'Profit': [20, 25, 22, 30, 35]
})

# Line plot of all numeric columns (Year as x)
df.plot(x='Year', y=['Sales', 'Profit'], marker='o')
plt.title('Sales and Profit Over Years')
plt.ylabel('Amount')
plt.show()

# Bar plot
df.plot(x='Year', y='Sales', kind='bar', color='green')
plt.show()

# Histogram of a column
df['Sales'].plot(kind='hist', bins=5, edgecolor='black')
plt.show()
```

**Common `kind` options:** 'line', 'bar', 'barh', 'hist', 'box', 'kde', 'scatter'.

---

## 8. Scatter Matrix (Pair Plot) with pandas

For exploring relationships between multiple variables.

```python
from pandas.plotting import scatter_matrix

# Load Iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Scatter matrix
scatter_matrix(iris, alpha=0.5, figsize=(10, 10), diagonal='hist')
plt.show()
```

---

## 9. Saving Figures

```python
plt.plot(x, y)
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')  # dpi for resolution
plt.savefig('my_plot.pdf')  # vector format for publications
```

---

## 10. Practical AI Examples

### Example 1: Visualizing Training History
Simulate loss and accuracy over epochs.

```python
epochs = np.arange(1, 21)
loss = 1.0 / (epochs + 0.5) + 0.05 * np.random.randn(20)
accuracy = 1 - loss + 0.1 * np.random.randn(20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, loss, 'r-', label='Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()

ax2.plot(epochs, accuracy, 'b-', label='Training Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curve')
ax2.legend()

plt.tight_layout()
plt.show()
```

### Example 2: Confusion Matrix Visualization
Though we'll cover this more in ML, here's a simple heatmap using `imshow`.

```python
# Simulated confusion matrix
cm = np.array([[50, 2, 5],
               [3, 45, 4],
               [6, 3, 42]])

plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0,1,2], ['Class0', 'Class1', 'Class2'])
plt.yticks([0,1,2], ['Class0', 'Class1', 'Class2'])

# Add text annotations
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.show()
```

### Example 3: Feature Distributions by Class
Using Iris dataset, compare sepal length across species.

```python
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Box plot
iris.boxplot(column='sepal_length', by='species')
plt.suptitle('')  # remove automatic title
plt.title('Sepal Length Distribution by Species')
plt.show()

# Histogram with transparency
species = iris['species'].unique()
for sp in species:
    subset = iris[iris['species'] == sp]
    plt.hist(subset['sepal_length'], alpha=0.5, label=sp, bins=10)

plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

---

## 11. Styling and Themes

Matplotlib offers several style sheets.

```python
print(plt.style.available)  # list available styles

plt.style.use('seaborn-v0_8-darkgrid')
# or
plt.style.use('ggplot')

# Then plot as usual
```

---

## 12. Summary – Key Matplotlib Functions

| Function | Purpose |
|----------|---------|
| `plt.plot()` | Line plot |
| `plt.scatter()` | Scatter plot |
| `plt.bar()` | Bar plot |
| `plt.hist()` | Histogram |
| `plt.xlabel()`, `plt.ylabel()` | Axis labels |
| `plt.title()` | Plot title |
| `plt.legend()` | Show legend |
| `plt.grid()` | Show grid |
| `plt.subplots()` | Create figure and axes for subplots |
| `plt.savefig()` | Save figure to file |
| `df.plot()` | Quick plot from pandas |

---

## 13. Practice Tasks

1. **Basic Plots**
   - Generate 100 random x points from 0 to 10, and y = sin(x) + random noise.
   - Create a scatter plot of x vs y.
   - Create a line plot of x vs sin(x) (without noise) on the same figure, with different colors and a legend.

2. **Customization**
   - Plot a bar chart of your favorite fruits and their quantities (make up data).
   - Add title, axis labels, and change bar colors.
   - Add a horizontal line at the average quantity.

3. **Subplots**
   - Create a 2x2 grid of subplots showing:
     - Top-left: line plot of x^2
     - Top-right: scatter plot of random points
     - Bottom-left: histogram of 1000 random normal numbers
     - Bottom-right: bar chart of 5 categories
   - Give each subplot a title.

4. **Pandas Plotting**
   - Load the Iris dataset into a pandas DataFrame.
   - Create a boxplot of petal length for each species using pandas `boxplot`.
   - Create a scatter plot of sepal length vs petal length, colored by species (hint: loop through species or use seaborn later, but try using matplotlib scatter with color mapping).

5. **Mini Project: Visualize a Dataset**
   - Choose any dataset (e.g., Titanic, Boston Housing, or a dataset of your interest).
   - Perform the following visualizations:
     - Histogram of a numeric column.
     - Bar chart of a categorical column.
     - Scatter plot of two numeric columns.
     - Box plots comparing a numeric column across categories.
   - Save the figure as a PNG.

---

You've now learned the fundamentals of data visualization with Matplotlib. These skills will help you explore data, debug models, and present results effectively. Next, we could dive into **Seaborn** for statistical visualizations or move into **machine learning with scikit-learn**. Let me know your preference!
