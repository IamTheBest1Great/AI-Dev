# Day 11: Cleaning and Filtering Data with Pandas

Welcome to Day 11! Today we focus on two critical steps in any data analysis or machine learning pipeline: **cleaning** and **filtering**. Raw data is rarely ready for modeling – it often contains missing values, duplicates, incorrect types, outliers, or irrelevant records. Cleaning and filtering transform messy data into a reliable, structured format that algorithms can learn from.

We'll cover:
- Why cleaning and filtering matter for AI.
- Handling missing data (detection, removal, imputation).
- Removing duplicates.
- Converting data types.
- Cleaning text data.
- Filtering rows based on conditions.
- Using `query()`, `isin()`, `between()`.
- Detecting and handling outliers.
- Practical examples with real datasets.
- Practice tasks.

Let's get started!

---

## 1. Why Cleaning and Filtering are Essential for AI

- **Garbage in, garbage out**: Models trained on dirty data produce unreliable results.
- **Missing values** can cause errors or bias in algorithms.
- **Duplicates** can overrepresent certain patterns, leading to overfitting.
- **Incorrect data types** (e.g., numbers stored as strings) prevent mathematical operations.
- **Outliers** can skew statistics and model performance.
- **Filtering** removes irrelevant observations, focusing the model on what matters.

Cleaning and filtering typically consume 60-80% of a data scientist's time – mastering these skills is crucial.

---

## 2. Handling Missing Data

Pandas represents missing values as `NaN` (Not a Number) for numeric data, or `None`/`NaN` for other types.

### Detecting Missing Data
```python
import pandas as pd
import numpy as np

# Sample DataFrame with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12],
    'D': ['x', 'y', None, 'z']
})

print(df)

# Check for missing values
print(df.isnull())
print(df.isnull().sum())        # count missing per column
print(df.isnull().sum().sum())  # total missing values
```

### Dropping Missing Data
```python
# Drop rows with any missing values
df_drop_rows = df.dropna()
print(df_drop_rows)

# Drop rows where all values are missing (if any)
df_drop_all = df.dropna(how='all')

# Drop columns with any missing values
df_drop_cols = df.dropna(axis=1)

# Drop rows with fewer than 3 non-NA values
df_thresh = df.dropna(thresh=3)
```

### Filling Missing Data
```python
# Fill with a constant
df_fill_const = df.fillna(0)

# Fill with mean of column (numeric only)
df['A'].fillna(df['A'].mean(), inplace=True)

# Forward fill (propagate last valid observation)
df_ffill = df.fillna(method='ffill')

# Backward fill
df_bfill = df.fillna(method='bfill')

# Interpolate (for numeric columns)
df_interp = df.interpolate()

# Fill with mode for categorical columns
df['D'].fillna(df['D'].mode()[0], inplace=True)
```

**Choosing a strategy:**
- For time series, forward/backward fill often makes sense.
- For numeric columns, mean/median imputation is common.
- For categorical, mode (most frequent) or a new category "Unknown".
- Sometimes dropping is best if missing values are few and random.

---

## 3. Removing Duplicates

Duplicates can skew analysis and model training.

```python
# Sample with duplicates
df_dup = pd.DataFrame({
    'id': [1, 2, 2, 3, 3, 4],
    'value': [10, 20, 20, 30, 30, 40]
})

print(df_dup)

# Check for duplicate rows
print(df_dup.duplicated())           # boolean Series
print(df_dup.duplicated().sum())     # count duplicates

# Drop duplicate rows (keep first occurrence)
df_no_dup = df_dup.drop_duplicates()
print(df_no_dup)

# Drop duplicates based on specific columns
df_no_dup_subset = df_dup.drop_duplicates(subset=['id'])
print(df_no_dup_subset)

# Keep last occurrence instead of first
df_keep_last = df_dup.drop_duplicates(keep='last')
```

---

## 4. Converting Data Types

Often, data loaded from CSV may have incorrect types (e.g., numbers stored as strings).

```python
df = pd.DataFrame({
    'age': ['25', '30', '35'],        # string
    'salary': ['50000', '60000', '70000']  # string
})

print(df.dtypes)

# Convert to numeric
df['age'] = pd.to_numeric(df['age'])
df['salary'] = pd.to_numeric(df['salary'])
print(df.dtypes)

# Convert multiple columns at once
df = df.astype({'age': 'int32', 'salary': 'float64'})

# Handle errors during conversion
df['age'] = pd.to_numeric(df['age'], errors='coerce')  # invalid -> NaN
```

**Common conversions:**
- `pd.to_datetime()` for date/time columns.
- `astype('category')` for categorical variables.

---

## 5. Cleaning Text Data

Text columns often need cleaning before use (e.g., for NLP or categorical encoding).

```python
df = pd.DataFrame({
    'name': [' Alice ', 'BOB', 'charlie', '  DAVE  '],
    'city': ['New York', 'paris', 'london', 'BERLIN']
})

# Strip whitespace
df['name'] = df['name'].str.strip()

# Convert to lowercase
df['city'] = df['city'].str.lower()

# Replace
df['name'] = df['name'].str.replace('alice', 'Alice')  # case-sensitive

# Remove punctuation
df['name'] = df['name'].str.replace('[^\w\s]', '', regex=True)

# Split strings into lists
df['name_split'] = df['name'].str.split()

# Extract substrings with regex
df['first_letter'] = df['city'].str.extract('([a-z])')
```

---

## 6. Filtering Rows with Conditions

Filtering is selecting a subset of rows based on conditions.

### Basic Boolean Indexing
```python
# Load sample data
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'score': [85, 92, 78, 88]
})

# Filter where age > 28
df_filtered = df[df['age'] > 28]
print(df_filtered)

# Multiple conditions: use & (and) and | (or)
df_filtered = df[(df['age'] > 25) & (df['score'] > 80)]
print(df_filtered)

# Using | (or)
df_filtered = df[(df['age'] < 26) | (df['score'] > 90)]
print(df_filtered)

# Negation with ~
df_filtered = df[~(df['name'] == 'Bob')]
```

### Using `query()` – Cleaner Syntax
```python
# Equivalent to above
df_filtered = df.query('age > 25 and score > 80')
print(df_filtered)

# Using variables in query
threshold = 28
df_filtered = df.query('age > @threshold')
```

### Using `isin()` for Membership
```python
# Filter rows where name is in a list
names_list = ['Alice', 'Charlie']
df_filtered = df[df['name'].isin(names_list)]

# Filter out rows where name is in list
df_filtered = df[~df['name'].isin(names_list)]
```

### Using `between()` for Range
```python
# Filter rows where age is between 26 and 30 inclusive
df_filtered = df[df['age'].between(26, 30)]
```

### Filtering with String Methods
```python
# Rows where name starts with 'A'
df_filtered = df[df['name'].str.startswith('A')]

# Contains substring
df_filtered = df[df['name'].str.contains('li', case=False)]
```

---

## 7. Handling Outliers

Outliers are extreme values that can distort statistical analyses and model performance.

### Detecting Outliers
Common methods: IQR (Interquartile Range) and Z-score.

#### Using IQR
```python
# Generate data with outliers
np.random.seed(42)
data = pd.DataFrame({'value': np.random.randn(100)})
data.loc[10] = 100  # extreme outlier
data.loc[20] = -100  # extreme outlier

# Calculate IQR
Q1 = data['value'].quantile(0.25)
Q3 = data['value'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
data_no_outliers = data[(data['value'] >= lower_bound) & (data['value'] <= upper_bound)]
print(f"Removed {len(data) - len(data_no_outliers)} outliers")
```

#### Using Z-score
```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(data['value']))
threshold = 3
data_no_outliers = data[z_scores < threshold]
```

### Handling Outliers
- Remove them (if they are data entry errors).
- Cap them (winsorization) – replace extreme values with a specified percentile.
- Transform the variable (e.g., log transform).

---

## 8. Putting It All Together – A Comprehensive Cleaning Pipeline

Let's work through a realistic example using the Titanic dataset, applying multiple cleaning steps.

```python
import pandas as pd
import numpy as np

# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Step 1: Initial exploration
print(df.info())
print(df.head())

# Step 2: Handle missing values
# Check missing
print(df.isnull().sum())

# Age: fill with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Embarked: fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin: has too many missing, maybe drop or extract deck letter
df['Deck'] = df['Cabin'].str[0]  # extract first letter
df.drop('Cabin', axis=1, inplace=True)

# Step 3: Convert data types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

# Step 4: Feature engineering from text
# Extract title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Group rare titles
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Step 5: Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Step 6: Drop irrelevant columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Step 7: Convert categorical columns to dummy variables (one-hot encoding)
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# Step 8: Filter rows (optional) – e.g., keep only adults? For this example, we'll keep all.
# But if we wanted to focus on adults:
# df = df[df['Age'] >= 18]

# Step 9: Check for outliers in Fare (maybe cap)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df.loc[df['Fare'] > upper_bound, 'Fare'] = upper_bound  # cap outliers

print(df.head())
print(df.info())
```

This pipeline produces a clean DataFrame ready for machine learning.

---

## 9. Summary of Key Operations

| Task | Pandas Method |
|------|---------------|
| Detect missing | `df.isnull().sum()` |
| Drop missing | `df.dropna()` |
| Fill missing | `df.fillna()` |
| Remove duplicates | `df.drop_duplicates()` |
| Change data type | `df.astype()` or `pd.to_numeric()` |
| Clean text | `.str.strip()`, `.str.lower()`, `.str.replace()` |
| Filter rows | `df[condition]`, `df.query()`, `df['col'].isin()`, `df['col'].between()` |
| Handle outliers | IQR method, Z-score, capping |

---

## 10. Practice Tasks

1. **Missing Data**
   - Create a DataFrame with at least 10 rows and 4 columns, intentionally introduce missing values.
   - Use `isnull().sum()` to identify missing.
   - Fill numeric missing with column mean, categorical missing with mode.
   - Drop any rows that still have missing values.

2. **Duplicates**
   - Load any dataset or create one with duplicate rows.
   - Identify duplicates, count them, and remove them.
   - Experiment with `keep='last'` and `subset`.

3. **Data Type Conversion**
   - Load a CSV (or create a DataFrame) where a numeric column is read as string.
   - Convert it to numeric using `pd.to_numeric()`.
   - Handle errors with `errors='coerce'`.

4. **Text Cleaning**
   - Create a DataFrame with a column of messy strings (extra spaces, mixed case, punctuation).
   - Clean it: strip, lowercase, remove punctuation.
   - Extract the first word into a new column.

5. **Filtering**
   - Use the Iris dataset (or any dataset) and filter:
     - Rows where sepal length > 5 and species = 'setosa'.
     - Rows where petal length is between 1 and 2.
     - Rows where species is in ['versicolor', 'virginica'].

6. **Outliers**
   - Generate a column of 100 random numbers with some extreme values.
   - Detect outliers using IQR and Z-score.
   - Decide to remove or cap them.

7. **Mini Project: Clean a Real Dataset**
   - Find a dataset online (e.g., from Kaggle) that is known to be messy.
   - Write a script that:
     - Loads the data.
     - Prints basic info and missing counts.
     - Handles missing values appropriately.
     - Removes duplicates.
     - Converts data types as needed.
     - Filters out irrelevant rows (if any).
     - Saves the cleaned dataset to a new CSV.
   - Document each step with comments.

---

You've now mastered the essential skills of cleaning and filtering data – a superpower in the AI world. With clean data, your models will perform better and your analyses will be more reliable.

Next, we could move into **data visualization with Matplotlib and Seaborn** or start **machine learning with scikit-learn**. Let me know your choice!
