# Day 7: NumPy Basics – The Foundation of Numerical Computing in Python

Welcome to Day 7! Today we're diving into **NumPy** (Numerical Python), the most fundamental library for scientific computing in Python. Almost every AI/ML library (pandas, scikit-learn, TensorFlow, PyTorch) is built on top of NumPy or follows its conventions. Mastering NumPy is essential for efficient data manipulation, mathematical operations, and preparing data for machine learning models.

We'll cover:
- What NumPy is and why it's crucial for AI.
- Installing NumPy.
- Creating and inspecting arrays.
- Indexing, slicing, and reshaping.
- Element-wise operations and broadcasting.
- Universal functions (ufuncs).
- Aggregations and basic statistics.
- Linear algebra basics.
- Random number generation.

Let's get started!

---

## 1. What is NumPy?

NumPy provides:
- A powerful **N-dimensional array object** called `ndarray`.
- Fast **vectorized operations** (no explicit loops needed).
- Tools for integrating C/C++ and Fortran code.
- Linear algebra, Fourier transform, and random number capabilities.

**Why for AI?**
- Data in machine learning is typically represented as arrays (e.g., images as 3D arrays, tabular data as 2D arrays).
- Mathematical operations on these arrays (matrix multiplications, activations, gradients) are the core of model training.
- NumPy's speed (implemented in C) makes large-scale computations feasible.

---

## 2. Installing NumPy

Make sure you're in your project's virtual environment (see Day 6).

```bash
pip install numpy
```

Verify installation:

```python
import numpy as np
print(np.__version__)
```

We usually import NumPy as `np` for brevity.

---

## 3. Creating NumPy Arrays

The fundamental object is `ndarray`. You can create arrays from Python lists or using built-in functions.

### From Lists
```python
import numpy as np

# 1D array
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)          # [1 2 3 4 5]
print(type(arr1))    # <class 'numpy.ndarray'>

# 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# Specify data type
arr3 = np.array([1, 2, 3], dtype=np.float32)
print(arr3)          # [1. 2. 3.]
```

### Using Built-in Functions
```python
# Array of zeros
zeros = np.zeros((3, 4))  # 3 rows, 4 columns
print(zeros)

# Array of ones
ones = np.ones((2, 3))
print(ones)

# Identity matrix
identity = np.eye(3)
print(identity)

# Array with a range of values (like range but returns array)
range_arr = np.arange(0, 10, 2)  # start, stop, step
print(range_arr)                 # [0 2 4 6 8]

# Evenly spaced numbers over an interval
linspace_arr = np.linspace(0, 1, 5)  # 5 numbers from 0 to 1 inclusive
print(linspace_arr)                   # [0.   0.25 0.5  0.75 1.  ]

# Random arrays
random_arr = np.random.rand(3, 3)     # uniform in [0,1)
print(random_arr)

random_int = np.random.randint(0, 10, size=(2, 5))  # random integers
print(random_int)
```

---

## 4. Array Attributes

Once you have an array, you can inspect its properties.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Shape:", arr.shape)        # (2, 3) – 2 rows, 3 columns
print("Number of dimensions:", arr.ndim)   # 2
print("Number of elements:", arr.size)     # 6
print("Data type:", arr.dtype)             # int64 (or int32 depending on system)
print("Item size (bytes):", arr.itemsize)  # 8 for int64
```

---

## 5. Indexing and Slicing

NumPy arrays support indexing and slicing similar to Python lists, but extended to multiple dimensions.

### Basic Indexing
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Single element
print(arr[0, 1])     # 2 (row 0, column 1)
print(arr[1, -1])    # 6 (last column of row 1)

# Entire row
print(arr[0])        # [1 2 3]

# Entire column
print(arr[:, 1])     # [2 5]  (all rows, column 1)
```

### Slicing
```python
# First two rows, first two columns
print(arr[:2, :2])   # [[1 2]
                     #  [4 5]]

# Reverse rows
print(arr[::-1])     # [[4 5 6]
                     #  [1 2 3]]

# Extract a subarray (creates a view, not a copy – careful!)
sub = arr[0:2, 0:2]
print(sub)
```

**Important**: Slicing returns a **view** of the original data (no copy). Modifying the view affects the original. To get a separate copy, use `.copy()`.

```python
sub_copy = arr[0:2, 0:2].copy()
```

---

## 6. Basic Operations

NumPy supports **vectorized operations**, meaning you can perform element-wise operations without explicit loops.

### Arithmetic
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition
print(a + b)          # [5 7 9]

# Other operations
print(a * b)          # [4 10 18]
print(a - b)          # [-3 -3 -3]
print(a / b)          # [0.25 0.4  0.5 ]

# Scalar operations
print(a * 2)          # [2 4 6]
print(a ** 2)         # [1 4 9]
```

### Comparison
```python
print(a > 2)          # [False False  True]
```

### Broadcasting
Broadcasting allows NumPy to work with arrays of different shapes during arithmetic.

```python
# Add a scalar to a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + 10)
# [[11 12 13]
#  [14 15 16]]

# Add a 1D array to a 2D array (broadcasts along rows)
row = np.array([10, 20, 30])
print(arr + row)
# [[11 22 33]
#  [14 25 36]]
```

---

## 7. Universal Functions (ufuncs)

NumPy provides fast, element-wise functions called universal functions.

```python
arr = np.array([1, 2, 3, 4])

print(np.sqrt(arr))        # [1.         1.41421356 1.73205081 2.        ]
print(np.exp(arr))         # [ 2.71828183  7.3890561  20.08553692 54.59815003]
print(np.log(arr))         # [0.         0.69314718 1.09861229 1.38629436]
print(np.sin(arr))         # [ 0.84147098  0.90929743  0.14112001 -0.7568025 ]
```

---

## 8. Aggregations (Reductions)

Compute statistics across axes.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("Sum of all elements:", np.sum(arr))          # 21
print("Mean:", np.mean(arr))                        # 3.5
print("Min:", np.min(arr))                          # 1
print("Max:", np.max(arr))                          # 6
print("Standard deviation:", np.std(arr))           # 1.707825127659933

# Sum along rows (axis=1) – sum each row
print("Sum of each row:", np.sum(arr, axis=1))      # [6 15]

# Sum along columns (axis=0) – sum each column
print("Sum of each column:", np.sum(arr, axis=0))   # [5 7 9]
```

---

## 9. Reshaping and Transposing

Change the shape of arrays without changing data.

```python
arr = np.arange(12)        # [0 1 2 ... 11]

# Reshape to 3x4
reshaped = arr.reshape(3, 4)
print(reshaped)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Flatten to 1D
flattened = reshaped.flatten()  # returns a copy
print(flattened)                 # [0 1 2 ... 11]

# Transpose (swap rows and columns)
transposed = reshaped.T
print(transposed.shape)          # (4, 3)
```

**Note**: `reshape` returns a view if possible; use `copy` to ensure independence.

---

## 10. Linear Algebra Basics

NumPy includes a submodule `linalg` for linear algebra operations.

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Dot product (matrix multiplication)
C = np.dot(A, B)        # or A @ B (Python 3.5+)
print(C)
# [[19 22]
#  [43 50]]

# Determinant
det = np.linalg.det(A)
print(det)              # -2.0000000000000004

# Inverse
inv = np.linalg.inv(A)
print(inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Eigenvalues
eigvals = np.linalg.eigvals(A)
print(eigvals)          # [-0.37228132  5.37228132]
```

---

## 11. Random Number Generation

NumPy's `random` module is essential for generating random data (e.g., initializing neural network weights, splitting data, synthetic datasets).

```python
# Set seed for reproducibility
np.random.seed(42)

# Random numbers from uniform distribution [0,1)
uniform = np.random.rand(3, 3)

# Random numbers from normal distribution (mean 0, std 1)
normal = np.random.randn(1000)   # 1000 samples

# Random integers
integers = np.random.randint(0, 100, size=10)

# Shuffle an array
arr = np.arange(10)
np.random.shuffle(arr)
print(arr)

# Random choice from array
choices = np.random.choice([1, 2, 3, 4, 5], size=10, replace=True)
```

---

## 12. Practical AI Example: Data Normalization

Suppose you have a dataset of features and you want to standardize them (zero mean, unit variance).

```python
# Simulate a dataset: 100 samples, 5 features
np.random.seed(42)
data = np.random.randn(100, 5) * 10 + 5   # mean ~5, std ~10

# Compute mean and std along rows (feature-wise)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# Standardize: (data - mean) / std
data_normalized = (data - mean) / std

# Check: new mean ≈ 0, std ≈ 1
print("New mean:", np.mean(data_normalized, axis=0))
print("New std:", np.std(data_normalized, axis=0))
```

This is exactly what many ML algorithms require before training.

---

## 13. Summary: Key NumPy Concepts for AI

| Concept | Why It Matters in AI |
|---------|----------------------|
| Arrays | Represent data (images, tabular, sequences) |
| Shape & reshaping | Adjust data dimensions for models |
| Vectorization | Fast computations without loops |
| Broadcasting | Flexible operations on different shapes |
| Aggregations | Compute loss, accuracy, statistics |
| Linear algebra | Neural network operations (matrix multiplications) |
| Random | Initialize weights, shuffle data, generate synthetic data |

---

## 14. Practice Tasks

1. **Array Creation**
   - Create a 1D array of numbers from 10 to 50 (inclusive) with step 2.
   - Create a 3x3 matrix of random integers between 1 and 20.
   - Create a 4x4 identity matrix.

2. **Indexing and Slicing**
   - Given a 5x5 array of ones, set the central 3x3 region to zeros.
   - Extract every other element from a 1D array of 20 random numbers.

3. **Operations**
   - Create two 3x3 matrices with random values. Compute their element-wise product and matrix product.
   - Generate an array of 100 random numbers from a normal distribution. Compute its mean and standard deviation. Reshape it into a 10x10 matrix.

4. **Broadcasting**
   - Create a 4x3 array of random integers. Add a 1D array of three numbers to each row. (Hint: the 1D array will broadcast.)

5. **Linear Algebra**
   - Solve the linear system: 2x + y = 5, x - 3y = -1 using `np.linalg.solve`. (Represent as matrix A and vector b, then solve.)

6. **Mini Project: Image-like Data**
   - Create a 28x28 array (simulating a grayscale image) with random integers 0-255.
   - Normalize the pixel values to [0,1].
   - Compute the average pixel intensity.
   - Flip the image horizontally (reverse columns) using slicing.

---

Great job reaching Day 7! NumPy is a vast library, but these basics will carry you through most AI tasks. Practice these concepts until they feel natural – they are the building blocks of everything that follows.

What would you like to tackle next? Options: **pandas** for data manipulation, **matplotlib** for visualization, or perhaps **scikit-learn** for machine learning basics? Let me know!
