# Day 9: Pandas Basics – Data Manipulation for AI

Welcome to Day 9! Today we're diving into **Pandas**, the most popular Python library for data manipulation and analysis. In any AI project, the first step is understanding and preparing your data – and Pandas is the tool that makes this efficient and intuitive. Built on top of NumPy, Pandas introduces two powerful data structures: **Series** and **DataFrame**.

We'll cover:
- What Pandas is and why it's essential for AI.
- Installing and importing Pandas.
- Series and DataFrames – creation and inspection.
- Reading and writing data (CSV, JSON, Excel).
- Selecting, filtering, and modifying data.
- Handling missing data.
- Grouping and aggregating.
- Merging datasets.
- Applying functions.
- Practical examples relevant to machine learning.

Let's get started!

---

## 1. What is Pandas?

Pandas provides high-level data structures and functions designed to make working with structured (tabular, multidimensional, potentially heterogeneous) data fast and easy. It's built on NumPy, so it inherits fast array computations, but adds:

- **Labeled axes** (rows and columns have names).
- **Handling missing data** gracefully.
- **Powerful grouping and aggregation** (similar to SQL GROUP BY).
- **Merging and joining** datasets.
- **Time series functionality**.

**Why for AI?**  
- Datasets are often in CSV or Excel files; Pandas loads them effortlessly.
- Data cleaning (handling missing values, removing duplicates, filtering) is essential before feeding data into models.
- Exploratory data analysis (EDA) uses Pandas to understand distributions, correlations, and patterns.
- Feature engineering (creating new columns from existing ones) is done with Pandas.

---

## 2. Installation

Make sure your virtual environment is active, then:

```bash
pip install pandas
```

Import convention:

```python
import pandas as pd
import numpy as np  # often used alongside pandas
```

---

## 3. Series – The One-Dimensional Workhorse

A **Series** is like a column in a spreadsheet: a one-dimensional array with axis labels (called the index).

### Creating a Series

```python
import pandas as pd

# From a list
s1 = pd.Series([10, 20, 30, 40])
print(s1)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# With custom index
s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s2)
# a    10
# b    20
# c    30
# d    40
# dtype: int64

# From a dictionary (keys become index)
data = {'apple': 3, 'banana': 5, 'cherry': 2}
s3 = pd.Series(data)
print(s3)
# apple     3
# banana    5
# cherry    2
# dtype: int64

# With a scalar (repeated value)
s4 = pd.Series(5, index=['x', 'y', 'z'])
print(s4)
# x    5
# y    5
# z    5
# dtype: int64
```

### Basic Attributes and Methods

```python
print(s2.values)        # [10 20 30 40]
print(s2.index)         # Index(['a', 'b', 'c', 'd'], dtype='object')
print(s2['b'])          # 20
print(s2[['a', 'c']])   # a    10, c    30
print(s2.mean())        # 25.0
```

---

## 4. DataFrame – The Heart of Pandas

A **DataFrame** is a two-dimensional labeled data structure with columns of potentially different types. Think of it as a spreadsheet or SQL table.

### Creating a DataFrame

```python
# From a dictionary of lists/arrays
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)
print(df)
#       name  age  salary
# 0    Alice   25   50000
# 1      Bob   30   60000
# 2  Charlie   35   70000

# From a list of dictionaries
data2 = [
    {'name': 'Alice', 'age': 25, 'salary': 50000},
    {'name': 'Bob', 'age': 30, 'salary': 60000},
    {'name': 'Charlie', 'age': 35, 'salary': 70000}
]
df2 = pd.DataFrame(data2)

# From a NumPy array with column names
arr = np.random.randn(4, 3)
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

### Inspecting a DataFrame

```python
print(df.head(2))           # first 2 rows
print(df.tail(2))           # last 2 rows
print(df.info())            # concise summary (column types, non-null counts)
print(df.describe())        # statistical summary for numeric columns
print(df.shape)             # (3, 3)
print(df.columns)           # Index(['name', 'age', 'salary'], dtype='object')
print(df.index)             # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)            # data type of each column
```

---

## 5. Reading and Writing Data

Pandas can read from and write to many formats. The most common are CSV, JSON, and Excel.

### Reading CSV
```python
df = pd.read_csv('data.csv')
# If there's no header, use header=None and provide names
df = pd.read_csv('data.csv', names=['col1', 'col2', 'col3'])
```

### Writing CSV
```python
df.to_csv('output.csv', index=False)   # index=False prevents writing row numbers
```

### Reading JSON
```python
df = pd.read_json('data.json')
```

### Writing JSON
```python
df.to_json('output.json', orient='records')  # orient='records' gives list of dicts
```

### Reading Excel (requires `openpyxl` or `xlrd`)
```python
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

---

## 6. Basic Operations on DataFrames

### Selecting Columns
```python
# Single column -> Series
ages = df['age']
print(type(ages))   # <class 'pandas.core.series.Series'>

# Multiple columns -> DataFrame
subset = df[['name', 'salary']]
```

### Selecting Rows by Position (iloc) and Label (loc)
```python
# iloc: integer-based indexing
first_row = df.iloc[0]           # first row as Series
first_two_rows = df.iloc[0:2]    # first two rows
element = df.iloc[1, 2]          # row 1, column 2 (salary for Bob)

# loc: label-based indexing (uses index labels)
# If index is default (0,1,2,...), loc works similarly
row_1 = df.loc[1]                # row with index label 1
# More useful with custom index
df_custom = df.set_index('name')
print(df_custom.loc['Alice'])     # row for Alice
```

### Adding a New Column
```python
df['bonus'] = df['salary'] * 0.1
print(df)
```

### Removing Columns
```python
df.drop('bonus', axis=1, inplace=True)   # axis=1 means columns
# or df = df.drop('bonus', axis=1)
```

### Renaming Columns
```python
df.rename(columns={'age': 'years', 'salary': 'income'}, inplace=True)
```

---

## 7. Filtering and Conditional Selection

Use boolean masks.

```python
# Select rows where age > 30
mask = df['age'] > 30
print(df[mask])

# Combine conditions with & (and), | (or), ~ (not)
mask2 = (df['age'] > 25) & (df['salary'] < 65000)
print(df[mask2])

# Using query method (more readable for complex conditions)
print(df.query('age > 25 and salary < 65000'))
```

---

## 8. Handling Missing Data

Real-world data often has missing values (NaN).

```python
# Create a DataFrame with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# Detect missing
print(df_missing.isnull())
print(df_missing.isnull().sum())   # count missing per column

# Drop rows with any missing
df_dropped = df_missing.dropna()   # default axis=0 (rows)

# Drop columns with any missing
df_dropped_cols = df_missing.dropna(axis=1)

# Fill missing with a constant
df_filled = df_missing.fillna(0)

# Fill with mean of column
df_filled_mean = df_missing.fillna(df_missing.mean())
```

---

## 9. Aggregations and Grouping

### Simple Aggregations
```python
print(df['salary'].mean())
print(df[['age', 'salary']].sum())
print(df.agg({'age': 'mean', 'salary': 'sum'}))  # different functions per column
```

### GroupBy – Split-Apply-Combine
```python
# Example dataset: sales by product category
sales_data = {
    'product': ['A', 'B', 'A', 'B', 'A', 'C'],
    'region': ['North', 'North', 'South', 'South', 'North', 'South'],
    'sales': [100, 150, 200, 120, 130, 180]
}
df_sales = pd.DataFrame(sales_data)

# Group by product, compute mean sales
grouped = df_sales.groupby('product')['sales'].mean()
print(grouped)

# Group by product and region, then sum
grouped_multi = df_sales.groupby(['product', 'region'])['sales'].sum()
print(grouped_multi)

# Agg with multiple functions
grouped_agg = df_sales.groupby('product')['sales'].agg(['mean', 'sum', 'count'])
```

### value_counts() – Frequency table
```python
print(df_sales['product'].value_counts())
```

---

## 10. Merging and Joining DataFrames

Often you need to combine multiple DataFrames.

### Concatenation (stacking)
```python
df1 = pd.DataFrame({'A': [1,2], 'B': [3,4]})
df2 = pd.DataFrame({'A': [5,6], 'B': [7,8]})

# Row-wise
concat_rows = pd.concat([df1, df2], axis=0, ignore_index=True)
# Column-wise
concat_cols = pd.concat([df1, df2], axis=1)
```

### Merging (like SQL JOIN)
```python
left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2'],
    'value_left': [10, 20, 30]
})
right = pd.DataFrame({
    'key': ['K0', 'K1', 'K3'],
    'value_right': [40, 50, 60]
})

# Inner join
inner = pd.merge(left, right, on='key', how='inner')
print(inner)
#   key  value_left  value_right
# 0  K0          10           40
# 1  K1          20           50

# Left join
left_join = pd.merge(left, right, on='key', how='left')
print(left_join)
#   key  value_left  value_right
# 0  K0          10         40.0
# 1  K1          20         50.0
# 2  K2          30          NaN
```

### Joining on Index
```python
left = left.set_index('key')
right = right.set_index('key')
joined = left.join(right, how='inner')
```

---

## 11. Applying Functions

Apply a function to each element, row, or column.

### apply() on Series
```python
df['age'] = df['age'].apply(lambda x: x + 1)   # add 1 to every age
```

### apply() on DataFrame (row or column wise)
```python
# Row-wise: axis=1
df['total'] = df.apply(lambda row: row['age'] + row['salary'], axis=1)

# Column-wise: axis=0 (default)
max_vals = df.apply(np.max, axis=0)
```

### map() for Series (replacement based on dictionary)
```python
df['gender'] = df['name'].map({'Alice': 'F', 'Bob': 'M', 'Charlie': 'M'})
```

### applymap() element-wise on entire DataFrame
```python
df[['age', 'salary']] = df[['age', 'salary']].applymap(lambda x: x * 1.1)
```

---

## 12. Practical AI Examples

### Example 1: Loading and Exploring a Dataset
```python
# Load the famous Iris dataset from a URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df_iris = pd.read_csv(url, names=columns)

print(df_iris.head())
print(df_iris.info())
print(df_iris['species'].value_counts())
print(df_iris.describe())
```

### Example 2: Data Cleaning – Handling Missing Values
```python
# Simulate missing values in Iris dataset
df_iris_missing = df_iris.copy()
df_iris_missing.loc[0:5, 'sepal_length'] = np.nan

# Drop rows with any missing (simple)
df_clean = df_iris_missing.dropna()

# Or fill with column mean
df_filled = df_iris_missing.fillna(df_iris_missing.mean(numeric_only=True))
```

### Example 3: Feature Engineering – Creating New Columns
```python
# Create a new feature: sepal area = sepal_length * sepal_width
df_iris['sepal_area'] = df_iris['sepal_length'] * df_iris['sepal_width']

# Create a categorical column based on petal length
df_iris['petal_size_category'] = pd.cut(df_iris['petal_length'],
                                         bins=[0, 2, 5, 10],
                                         labels=['small', 'medium', 'large'])
```

### Example 4: Preparing Data for Machine Learning
```python
# Separate features and labels
X = df_iris.drop('species', axis=1)   # feature matrix
y = df_iris['species']                 # target vector

# Convert categorical labels to numbers (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Now X and y_encoded are ready for model training
```

---

## 13. Summary – Key Pandas Concepts for AI

| Concept | Why It Matters |
|---------|----------------|
| DataFrame | Primary data structure for datasets |
| read_csv / to_csv | Load and save datasets |
| info(), describe() | Quick data understanding |
| isnull(), dropna(), fillna() | Handle missing data – crucial for real-world data |
| groupby() | Aggregate statistics per category |
| merge() | Combine multiple data sources |
| apply() | Feature engineering |
| value_counts() | Explore categorical distributions |
| iloc / loc | Flexible data selection |

---

## 14. Practice Tasks

1. **Series Practice**
   - Create a Series from a list `[5, 10, 15, 20]` with index `['a','b','c','d']`.
   - Multiply each value by 2.
   - Extract values with index 'b' and 'd'.

2. **DataFrame Creation**
   - Create a DataFrame from a dictionary of three students: name, test_score, study_hours.
   - Add a column `passed` where test_score >= 60.
   - Compute the average test_score.

3. **Reading and Writing**
   - Download a sample CSV (e.g., from Kaggle or use `pd.read_csv` on a URL) and load it.
   - Save the first 10 rows to a new CSV file.

4. **Filtering and Grouping**
   - Use the Iris dataset (or any dataset) and filter rows where petal_length > 1.5.
   - Group by species and compute the mean of each numeric column.

5. **Handling Missing Data**
   - Create a small DataFrame with at least 2 missing values.
   - Experiment with dropping missing rows, filling with 0, and filling with column mean.

6. **Merging**
   - Create two DataFrames: one with student IDs and names, another with student IDs and scores.
   - Merge them to get a complete table.

7. **Mini Project: Analyze a Dataset**
   - Load the Titanic dataset from a URL (e.g., `pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')`).
   - Explore the data: shape, info, missing values.
   - Compute survival rate by gender and by passenger class.
   - Fill missing ages with the median age.
   - Create a new feature `family_size` = `sibsp + parch + 1`.
   - Save the cleaned DataFrame to a CSV.

---

You've now covered the essentials of Pandas – enough to handle most data manipulation tasks you'll encounter in AI projects. The key is practice. Work through the tasks, and soon you'll feel at home wrangling any dataset.

Next, we could explore **data visualization with Matplotlib and Seaborn** or dive into **scikit-learn for machine learning basics**. Let me know your preference!
