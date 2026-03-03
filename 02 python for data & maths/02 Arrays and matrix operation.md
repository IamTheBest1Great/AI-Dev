# Day 8: Arrays and Matrix Operations in NumPy

Welcome to Day 8! Now that you have a solid foundation in NumPy basics, we'll focus specifically on **arrays and matrix operations** – the core of numerical computing in AI. Matrices (2D arrays) are everywhere: they represent datasets, neural network weights, images, and more. Understanding how to manipulate them efficiently is essential.

We'll cover:
- Creating and inspecting matrices.
- Element-wise vs. matrix operations.
- Matrix multiplication (dot product).
- Transposing, reshaping, and manipulating matrices.
- Linear algebra operations (inverse, determinant, etc.).
- Broadcasting with matrices.
- Practical AI examples.

Let's dive deeper!

---

## 1. What is a Matrix in NumPy?

In NumPy, a **matrix** is simply a 2-dimensional array. You can create one using `np.array()` with a nested list, or using functions like `np.zeros()`, `np.ones()`, `np.random.rand()`, etc.

```python
import numpy as np

# Create a 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(A)
print("Shape:", A.shape)   # (3, 3)
print("Data type:", A.dtype)
```

**Note:** NumPy also has a separate `matrix` class, but it's deprecated. Use regular 2D arrays instead.

---

## 2. Creating Matrices: Common Patterns

```python
# Zeros matrix
Z = np.zeros((4, 3))          # 4 rows, 3 columns

# Ones matrix
O = np.ones((2, 5))

# Identity matrix
I = np.eye(4)                 # 4x4 identity

# Random matrix (uniform [0,1))
R = np.random.rand(3, 3)

# Random integers
R_int = np.random.randint(0, 10, size=(2, 4))

# Diagonal matrix from a list
diag = np.diag([1, 2, 3])     # 3x3 diagonal
```

---

## 3. Basic Matrix Properties

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Shape:", A.shape)       # (2, 3)
print("Number of dimensions:", A.ndim)  # 2
print("Total elements:", A.size)        # 6
print("Data type:", A.dtype)            # int64
```

---

## 4. Indexing and Slicing Matrices

(Review with emphasis on 2D)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Single element
print(A[1, 2])       # 6 (row 1, column 2)

# Entire row
print(A[0])          # [1 2 3]

# Entire column
print(A[:, 1])       # [2 5 8]

# Submatrix (rows 1-2, columns 1-2)
print(A[1:3, 1:3])   # [[5 6]
                     #  [8 9]]

# Using step
print(A[::2, ::2])   # rows 0 and 2, cols 0 and 2 → [[1 3]
                     #                                [7 9]]
```

**Important:** Slicing returns a **view** (not a copy). Modifying the view changes the original.

```python
sub = A[:2, :2]
sub[0, 0] = 999
print(A)             # A is modified!
```

To avoid this, use `.copy()`:

```python
sub = A[:2, :2].copy()
```

---

## 5. Element-wise Operations vs. Matrix Operations

### Element-wise Operations
These apply to each element independently.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
print(A + B)
# [[ 6  8]
#  [10 12]]

# Multiplication (element-wise)
print(A * B)
# [[ 5 12]
#  [21 32]]

# Division, subtraction, powers, etc. are all element-wise.
```

### Matrix Multiplication (Dot Product)
Matrix multiplication follows linear algebra rules (rows × columns). Use `np.dot(A, B)` or the `@` operator (Python 3.5+).

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

C = np.dot(A, B)   # or A @ B
print(C)
# [[19 22]
#  [43 50]]
```

**Why it matters in AI:** Neural network layers are essentially matrix multiplications followed by additions (biases).

---

## 6. Transpose, Reshape, and Flatten

### Transpose
Swap rows and columns.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### Reshape
Change the shape without changing data.

```python
B = np.arange(12).reshape(3, 4)
print(B)
```

### Flatten / Ravel
Convert to 1D.

```python
flat = B.flatten()   # returns a copy
raveled = B.ravel()  # returns a view (if possible)
```

---

## 7. Linear Algebra with `np.linalg`

NumPy's `linalg` module provides essential linear algebra functions.

```python
from numpy import linalg

A = np.array([[1, 2],
              [3, 4]])

# Determinant
det = linalg.det(A)
print("Determinant:", det)   # -2.0

# Inverse
inv = linalg.inv(A)
print("Inverse:\n", inv)

# Check: A @ inv ≈ I
print(A @ inv)                # [[1. 0.], [0. 1.]]

# Eigenvalues and eigenvectors
eigvals, eigvecs = linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# Solve linear system Ax = b
b = np.array([5, 6])
x = linalg.solve(A, b)
print("Solution x:", x)       # solves 1*x1 + 2*x2 = 5, 3*x1 + 4*x2 = 6
```

---

## 8. Stacking and Concatenating Matrices

Combine matrices in various ways.

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6]])

# Vertical stacking (row-wise)
C = np.vstack((A, B))        # B must have same number of columns
print(C)
# [[1 2]
#  [3 4]
#  [5 6]]

# Horizontal stacking (column-wise)
D = np.hstack((A, A))        # same number of rows
print(D)
# [[1 2 1 2]
#  [3 4 3 4]]

# Concatenate with axis control
E = np.concatenate((A, A), axis=0)  # same as vstack
F = np.concatenate((A, A), axis=1)  # same as hstack
```

---

## 9. Broadcasting with Matrices

Broadcasting allows operations between arrays of different shapes. This is extremely useful in AI for adding biases, scaling, etc.

```python
# Add a row vector to each row of a matrix
X = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])   # shape (3,)

Y = X + b                     # broadcasts b to each row
print(Y)
# [[11 22 33]
#  [14 25 36]]

# Add a column vector to each column
c = np.array([[100],
              [200]])         # shape (2,1)
Z = X + c
print(Z)
# [[101 102 103]
#  [204 205 206]]
```

**Rule:** Broadcasting works when dimensions are compatible (either equal or one is 1).

---

## 10. Practical AI Examples

### Example 1: Simple Linear Layer (Forward Pass)
In a neural network, a linear layer computes `output = input @ weights.T + bias`.

```python
# Simulate a batch of 4 samples, each with 3 features
X = np.random.randn(4, 3)      # input batch
W = np.random.randn(5, 3)       # weights for 5 output neurons
b = np.random.randn(5)          # biases

# Forward pass: (4,3) @ (3,5) -> (4,5)
output = X @ W.T + b
print("Output shape:", output.shape)  # (4, 5)
```

### Example 2: Mean Squared Error (MSE) Loss
MSE = mean((predictions - targets)^2)

```python
predictions = np.array([[0.2, 0.8],
                        [0.5, 0.5]])
targets = np.array([[0.1, 0.9],
                    [0.6, 0.4]])

mse = np.mean((predictions - targets) ** 2)
print("MSE:", mse)
```

### Example 3: Feature Scaling (Standardization)
Standardization: (X - mean) / std

```python
# Suppose we have a dataset of 100 samples, 4 features
np.random.seed(42)
X = np.random.randn(100, 4) * 10 + 5   # artificial data

# Compute mean and std per feature (column)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Standardize (broadcasting applies)
X_scaled = (X - mean) / std
print("New mean (approx 0):", np.mean(X_scaled, axis=0))
print("New std (approx 1):", np.std(X_scaled, axis=0))
```

### Example 4: Simple Image Processing
A grayscale image is a 2D matrix (height × width). Common operations:

```python
# Simulate a 5x5 image
image = np.random.randint(0, 256, size=(5, 5))
print("Original image:\n", image)

# Flip horizontally (mirror)
flipped = image[:, ::-1]
print("Horizontally flipped:\n", flipped)

# Crop to 3x3 center
crop = image[1:4, 1:4]
print("Center crop:\n", crop)

# Brighten (add constant)
bright = image + 20
```

---

## 11. Performance Tip: Use Vectorization

Always prefer vectorized operations over explicit Python loops. NumPy is implemented in C and highly optimized.

```python
# Slow (avoid)
A = np.random.randn(1000, 1000)
B = np.random.randn(1000, 1000)
C = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        C[i, j] = A[i, j] + B[i, j]

# Fast (vectorized)
C = A + B
```

---

## 12. Summary

| Operation | NumPy Code |
|-----------|------------|
| Matrix multiplication | `A @ B` or `np.dot(A, B)` |
| Element-wise multiply | `A * B` |
| Transpose | `A.T` |
| Reshape | `A.reshape(new_shape)` |
| Determinant | `np.linalg.det(A)` |
| Inverse | `np.linalg.inv(A)` |
| Solve linear system | `np.linalg.solve(A, b)` |
| Stack vertically | `np.vstack((A, B))` |
| Stack horizontally | `np.hstack((A, B))` |
| Broadcasting | `A + b` (with compatible shapes) |

---

## 13. Practice Tasks

1. **Matrix Creation and Properties**
   - Create a 4x6 matrix of random integers between 0 and 50.
   - Print its shape, size, and data type.
   - Extract the second column and compute its mean.

2. **Matrix Multiplication**
   - Create two matrices: A (3x2) and B (2x4) with random values.
   - Compute their product C = A @ B.
   - Verify that C has shape (3,4).

3. **Linear Algebra**
   - Create a 3x3 matrix with numbers 1 to 9 (reshape `np.arange(1,10).reshape(3,3)`).
   - Compute its determinant, inverse, and eigenvalues.
   - Verify that `A @ inv(A)` is approximately identity.

4. **Broadcasting Practice**
   - Create a 5x3 matrix X (random) and a 1x3 row vector b (random).
   - Add b to each row of X without using a loop.
   - Now create a 5x1 column vector c and add it to each column of X.

5. **Mini Project: Simple Neural Network Layer**
   - Write a function `linear_forward(X, W, b)` that returns the output.
   - Test with X of shape (10,4) (10 samples, 4 features), W of shape (3,4) (3 output neurons), and b of shape (3,).
   - Apply a ReLU activation (max(0, output)) element-wise.

6. **Image-like Manipulation**
   - Create an 8x8 matrix with random grayscale values (0-255).
   - Apply a 2x2 average pooling: divide the image into 2x2 blocks and replace each block with its average. (Hint: reshape and use mean.)
   - Hint: reshape to (4,2,4,2) and mean over axes 1 and 3.

---

You've now mastered the essential matrix operations that form the backbone of AI computations. Keep practicing, and soon these operations will become second nature.

Next, we could explore **pandas** for data manipulation or **matplotlib** for visualization – both crucial for AI workflows. Let me know your preference!
