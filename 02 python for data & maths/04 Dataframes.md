# Day 10: Mastering DataFrames in Pandas

Welcome to Day 10! Today we'll focus exclusively on **DataFrames** – the most important data structure in Pandas. A DataFrame is a two-dimensional, labeled data structure with columns that can hold different types (like a spreadsheet or SQL table). Mastering DataFrames is essential for data manipulation, cleaning, and preparation in any AI project.

We'll cover:
- What a DataFrame is and its key components.
- Various ways to create DataFrames.
- Inspecting and understanding your DataFrame.
- Selecting, filtering, and modifying data.
- Handling missing values within DataFrames.
- Grouping, aggregating, and summarizing data.
- Combining multiple DataFrames.
- Applying functions and transformations.
- Practical examples relevant to machine learning.

Let's dive deep!

---

## 1. What is a DataFrame?

A DataFrame is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Think of it as a dictionary of Series objects (each column is a Series), all sharing the same index.

**Key components:**
- **Index**: row labels (can be integers, strings, etc.)
- **Columns**: column labels
- **Data**: the actual values, stored in a NumPy array or similar structure.

```python
import pandas as pd
import numpy as np

# A simple DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})

print(df)
```

---

## 2. Creating DataFrames

There are many ways to create a DataFrame. Here are the most common:

### From a dictionary of lists/arrays
```python
data = {
    'product': ['A', 'B', 'C'],
    'price': [100, 200, 300],
    'in_stock': [True, False, True]
}
df = pd.DataFrame(data)
```

### From a list of dictionaries
```python
data = [
    {'product': 'A', 'price': 100, 'in_stock': True},
    {'product': 'B', 'price': 200, 'in_stock': False},
    {'product': 'C', 'price': 300, 'in_stock': True}
]
df = pd.DataFrame(data)
```

### From a NumPy array
```python
arr = np.random.randn(4, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

### From a CSV file (most common for real datasets)
```python
df = pd.read_csv('data.csv')
```

### From a list of lists with specified column names
```python
data = [[1, 'Alice', 25],
        [2, 'Bob', 30],
        [3, 'Charlie', 35]]
df = pd.DataFrame(data, columns=['ID', 'Name', 'Age'])
```

### From a single Series (becomes one column)
```python
s = pd.Series([10, 20, 30], name='values')
df = pd.DataFrame(s)  # or s.to_frame()
```

### Empty DataFrame
```python
df_empty = pd.DataFrame()
```

---

## 3. Inspecting a DataFrame

Once you have a DataFrame, you need to understand its structure and contents.

```python
# Assume df is loaded with some data

# First few rows
print(df.head(2))

# Last few rows
print(df.tail(3))

# Random sample
print(df.sample(5))

# Shape (rows, columns)
print(df.shape)

# Column names
print(df.columns)

# Index (row labels)
print(df.index)

# Data types of each column
print(df.dtypes)

# Concise summary (including memory usage)
print(df.info())

# Statistical summary (numeric columns only)
print(df.describe())

# Count of unique values in a column
print(df['column_name'].value_counts())

# Basic information about the DataFrame
print(df.info())
```

---

## 4. Selecting Data from a DataFrame

### Selecting Columns
```python
# Single column -> Series
col_series = df['column_name']
# or
col_series = df.column_name   # only if column name is a valid Python identifier

# Multiple columns -> DataFrame
subset_df = df[['col1', 'col2']]
```

### Selecting Rows by Position (iloc) and Label (loc)
`iloc` uses integer positions, `loc` uses index labels.

```python
# First row
row0 = df.iloc[0]

# Last row
last = df.iloc[-1]

# First two rows
first_two = df.iloc[0:2]

# Specific cell (row 1, column 2)
cell = df.iloc[1, 2]

# Using loc (if index is default integer, similar to iloc)
row1 = df.loc[1]

# With custom index
df_custom = df.set_index('Name')
row_alice = df_custom.loc['Alice']
```

### Selecting Specific Rows and Columns
```python
# All rows, columns 'Age' and 'Salary'
df[['Age', 'Salary']]

# Rows 0 to 2, columns 'Name' and 'Salary'
df.loc[0:2, ['Name', 'Salary']]

# Using iloc: rows 0 to 2, columns 1 to 2
df.iloc[0:3, 1:3]
```

### Boolean Indexing (Filtering)
```python
# Rows where Age > 30
df[df['Age'] > 30]

# Multiple conditions: use & (and), | (or), ~ (not)
df[(df['Age'] > 25) & (df['Salary'] < 65000)]

# Using query method (cleaner for complex conditions)
df.query('Age > 25 and Salary < 65000')

# Filter based on list of values
df[df['Name'].isin(['Alice', 'Charlie'])]
```

### Selecting with `iat` and `at` for fast scalar access
```python
# at: label-based scalar access
value = df.at[1, 'Age']

# iat: integer-based scalar access
value = df.iat[1, 1]
```

---

## 5. Modifying DataFrames

### Adding a New Column
```python
# Constant value
df['Bonus'] = 5000

# Based on existing columns
df['Total'] = df['Salary'] + df['Bonus']

# Using assign (returns a new DataFrame)
df_new = df.assign(Bonus_Ratio=lambda x: x['Bonus'] / x['Salary'])
```

### Inserting a Column at a Specific Position
```python
df.insert(1, 'Department', ['HR', 'IT', 'Finance'])  # after first column
```

### Replacing Values
```python
# Replace specific values in a column
df['Department'] = df['Department'].replace('HR', 'Human Resources')

# Replace using dictionary
df.replace({'Department': {'IT': 'Information Technology'}}, inplace=True)
```

### Renaming Columns
```python
df.rename(columns={'Age': 'Years', 'Salary': 'Income'}, inplace=True)
```

### Dropping Columns or Rows
```python
# Drop columns
df.drop('Bonus', axis=1, inplace=True)

# Drop rows by index label
df.drop([0, 2], axis=0, inplace=True)

# Drop rows by condition (using boolean indexing, then drop)
df.drop(df[df['Age'] < 30].index, inplace=True)
```

### Sorting
```python
# Sort by column
df_sorted = df.sort_values(by='Age', ascending=False)

# Sort by multiple columns
df_sorted = df.sort_values(by=['Department', 'Age'])
```

### Setting and Resetting Index
```python
# Set a column as index
df.set_index('Name', inplace=True)

# Reset index to default integer
df.reset_index(inplace=True)
```

---

## 6. Handling Missing Data in DataFrames

Real-world data often has missing values (NaN).

### Detecting Missing Data
```python
# Boolean DataFrame where values are missing
print(df.isnull())

# Count missing per column
print(df.isnull().sum())

# Check if any missing in entire DataFrame
print(df.isnull().any().any())
```

### Dropping Missing Data
```python
# Drop rows with any missing
df_clean = df.dropna()  # default axis=0

# Drop columns with any missing
df_clean = df.dropna(axis=1)

# Drop rows where all values are missing
df_clean = df.dropna(how='all')

# Drop rows with less than 2 non-NA values
df_clean = df.dropna(thresh=2)
```

### Filling Missing Data
```python
# Fill with a constant
df_filled = df.fillna(0)

# Fill with mean of column
df_filled = df.fillna(df.mean(numeric_only=True))

# Forward fill (propagate last valid observation)
df_filled = df.fillna(method='ffill')

# Backward fill
df_filled = df.fillna(method='bfill')

# Interpolate (for numeric columns)
df_interpolated = df.interpolate()
```

### Handling Missing Data in Machine Learning
In ML, you often impute missing values with mean/median/mode or use models that handle missing data.

---

## 7. Grouping and Aggregating Data

The `groupby` operation is used to split data into groups based on some criteria, apply a function, and combine the results.

### Basic GroupBy
```python
# Example dataset
df_sales = pd.DataFrame({
    'Product': ['A', 'B', 'A', 'B', 'A', 'C'],
    'Region': ['North', 'North', 'South', 'South', 'North', 'South'],
    'Sales': [100, 150, 200, 120, 130, 180]
})

# Group by Product and compute mean sales
product_mean = df_sales.groupby('Product')['Sales'].mean()
print(product_mean)

# Group by multiple columns
grouped = df_sales.groupby(['Product', 'Region'])['Sales'].sum()
print(grouped)
```

### Multiple Aggregations
```python
# Agg with list of functions
agg_result = df_sales.groupby('Product')['Sales'].agg(['mean', 'sum', 'count'])
print(agg_result)

# Different functions per column (if multiple columns)
agg_multi = df_sales.groupby('Product').agg({
    'Sales': 'sum',
    'Region': 'nunique'   # number of unique regions per product
})
```

### Transforming and Filtering Groups
```python
# Transform: compute group-wise statistics and return DataFrame of same shape
df_sales['Sales_centered'] = df_sales.groupby('Product')['Sales'].transform(lambda x: x - x.mean())

# Filter groups based on condition
filtered = df_sales.groupby('Product').filter(lambda x: x['Sales'].sum() > 200)
```

### Using `groupby` with `.apply`
```python
def top_two(df):
    return df.nlargest(2, 'Sales')

top_two_per_product = df_sales.groupby('Product').apply(top_two)
```

---

## 8. Merging, Joining, and Concatenating DataFrames

Often you need to combine multiple DataFrames.

### Concatenation (Stacking)
```python
df1 = pd.DataFrame({'A': [1,2], 'B': [3,4]})
df2 = pd.DataFrame({'A': [5,6], 'B': [7,8]})

# Row-wise (vertical)
concat_rows = pd.concat([df1, df2], axis=0, ignore_index=True)

# Column-wise (horizontal)
concat_cols = pd.concat([df1, df2], axis=1)
```

### Merging (SQL-like joins)
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

# Left join
left_join = pd.merge(left, right, on='key', how='left')

# Outer join
outer = pd.merge(left, right, on='key', how='outer')
```

### Joining on Index
```python
left = left.set_index('key')
right = right.set_index('key')
joined = left.join(right, how='inner')
```

---

## 9. Applying Functions to DataFrames

### `apply` on rows or columns
```python
# Apply to each column (axis=0) – default
df[['Age', 'Salary']].apply(np.mean)

# Apply to each row (axis=1)
df['Age_Salary_sum'] = df.apply(lambda row: row['Age'] + row['Salary'], axis=1)
```

### `applymap` – element-wise on entire DataFrame
```python
# Square all numeric values
df[['Age', 'Salary']] = df[['Age', 'Salary']].applymap(lambda x: x**2)
```

### `map` on a Series (for value replacement)
```python
df['Gender'] = df['Name'].map({'Alice': 'F', 'Bob': 'M', 'Charlie': 'M'})
```

---

## 10. Pivot Tables and Cross-Tabulation

Pandas provides `pivot_table` for creating spreadsheet-style pivot tables, and `crosstab` for frequency tables.

### Pivot Table
```python
# Using the sales dataset
pivot = pd.pivot_table(df_sales,
                        values='Sales',
                        index='Product',
                        columns='Region',
                        aggfunc='sum',
                        fill_value=0)
print(pivot)
```

### Cross-Tabulation
```python
# Frequency of product by region
ct = pd.crosstab(df_sales['Product'], df_sales['Region'])
print(ct)
```

---

## 11. Working with Text Data in DataFrames

Pandas provides string methods via `.str` accessor.

```python
df = pd.DataFrame({'text': ['Hello World', 'Python is great', 'AI is future']})

# Convert to lowercase
df['text_lower'] = df['text'].str.lower()

# Check if contains 'AI'
df['has_ai'] = df['text'].str.contains('AI', case=False)

# Split
df['words'] = df['text'].str.split()
```

---

## 12. Practical AI Examples with DataFrames

### Example 1: Loading and Exploring a Real Dataset
```python
# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print(df.head())
print(df.info())
print(df.describe(include='all'))  # include non-numeric
```

### Example 2: Data Cleaning and Feature Engineering
```python
# Handle missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Extract title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplify rare titles
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
```

### Example 3: Grouped Analysis
```python
# Survival rate by gender
survival_by_gender = df.groupby('Sex')['Survived'].mean()
print(survival_by_gender)

# Survival rate by passenger class and gender
survival_by_class_gender = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
print(survival_by_class_gender)
```

### Example 4: Preparing Data for Machine Learning
```python
# Convert categorical columns to dummy variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# Select features and target
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize',
            'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Mr', 'Title_Mrs', 'Title_Rare']
X = df[features]
y = df['Survived']

# Now X and y are ready for model training
```

---

## 13. Performance Tips for DataFrames

- **Use vectorized operations** instead of `apply` with loops whenever possible.
- **Avoid chained indexing** like `df[df['a'] > 0]['b']` – use `.loc` instead.
- **Use `inplace=False` (default) for most operations**, as it's clearer and avoids subtle bugs.
- **For large datasets, use `dtypes` efficiently**: convert object columns to categorical if they have few unique values.
- **Use `pd.read_csv` with `usecols` and `dtype`** to load only needed columns and save memory.

---

## 14. Summary – Key DataFrame Operations for AI

| Operation | Purpose |
|-----------|---------|
| `pd.read_csv()` | Load dataset |
| `df.head()`, `df.info()`, `df.describe()` | Explore data |
| `df.isnull().sum()` | Identify missing values |
| `df.fillna()`, `df.dropna()` | Handle missing data |
| `df.groupby().agg()` | Aggregate statistics |
| `pd.get_dummies()` | One-hot encode categorical variables |
| `df.apply()` | Feature engineering |
| `pd.merge()` | Combine datasets |
| `df.to_csv()` | Save cleaned data |

---

## 15. Practice Tasks

1. **Creating DataFrames**
   - Create a DataFrame from a dictionary of your choice with at least 5 rows and 4 columns (including some text and numeric).
   - Load the Iris dataset from a URL (or use `pd.read_csv` with the Iris URL).

2. **Inspecting and Selecting**
   - For the Iris dataset, display the first 8 rows, the column names, and the data types.
   - Select only the `sepal_length` and `species` columns.
   - Filter rows where `petal_length` > 1.5.

3. **Handling Missing Data**
   - Introduce some missing values into a copy of the Iris dataset (set a few cells to `np.nan`).
   - Use `isnull().sum()` to verify.
   - Fill missing values with the column mean.
   - Drop rows that still have missing values (if any).

4. **Grouping and Aggregation**
   - Group the Iris dataset by `species` and compute the mean, min, and max of each numeric column.
   - Create a pivot table showing average `sepal_length` for each species.

5. **Merging**
   - Create two small DataFrames: one with student IDs and names, another with student IDs and scores.
   - Perform an inner join and a left join. Observe the differences.

6. **Feature Engineering**
   - In the Titanic dataset, create a new feature `IsAlone` that is 1 if `FamilySize == 1` else 0.
   - Create a feature `AgeGroup` by binning age into categories: Child (0-12), Teen (13-19), Adult (20-59), Senior (60+).
   - Compute survival rate for each age group.

7. **Mini Project: Data Cleaning Pipeline**
   - Load a dataset of your choice (e.g., from Kaggle) that has missing values, mixed types, etc.
   - Write a script that:
     - Displays basic info.
     - Handles missing values appropriately.
     - Converts categorical columns to numerical using one-hot encoding or label encoding.
     - Saves the cleaned DataFrame to a new CSV.

---

You've now gained a comprehensive understanding of DataFrames in Pandas – the Swiss Army knife of data manipulation in Python. With these skills, you can tackle any data preparation task for machine learning.

Next, we could explore **data visualization with Matplotlib and Seaborn** or move directly into **machine learning with scikit-learn**. Let me know your preference!
