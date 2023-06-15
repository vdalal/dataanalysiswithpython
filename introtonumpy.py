# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-numpy/blob/master/2.%20NumPy.ipynb

import sys
import numpy as np

np.array([1, 2, 3, 4])
print(np.array)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
b = np.array([0, .5, 1, 1.5, 2])

print(a)
print(b)

print(a[0], a[1])

print(a[0:])

print(a[1:3])

print(a[1:-1])

print(a[::2])

print(b)

print(b[0], b[2], b[-1])

print(b[[0, 2, -1]])

print(a.dtype)

print(b.dtype)

print(np.array([11, 12, 13, 14], dtype=float))

print(np.array([11, 12, 13, 14], dtype=np.int8))

c = np.array(['a', 'b', 'c'])

print(c.dtype)

d = np.array([{'a': 1}, sys])
print(d)
print(d.dtype)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.shape)
print(A.ndim)
print(A.size)

B = np.array([
    [
        [12, 11, 10],
        [9, 8, 7],
    ],
    [
        [6, 5, 4],
        [3, 2, 1]
    ]
])

print(B)
print(B.shape)
print(B.ndim)
print(B.size)

C = np.array([
    [
        [12, 11, 10],
        [9, 8, 7],
    ],
    [
        [6, 5, 4],
        [3, 2, 1]
    ]
])

print(C.dtype)
print(C.shape)
print(C.size)

# Square matrix
A = np.array([
    #.   0. 1. 2
    [1, 2, 3], # 0
    [4, 5, 6], # 1
    [7, 8, 9]  # 2
])

print(A[1])
print(A[1][0])

# rows = 0, 1 and not including row = 2
print(A[0:2])

# columns = 0, 1 and not including column = 2
print(A[:, :2])

# rows & columns = 0, 1 and not including row & column = 2
print(A[:2, :2])

print(A[:2, 2:])

print(A[:2, 2:])

print(A)

A[1] = np.array([10, 10, 10])

print(A)

A[2] = 99

print(A)

