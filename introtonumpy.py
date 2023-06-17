# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-numpy/blob/master/2.%20NumPy.ipynb
# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/numpy-operations

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

# Summary statistics

a = np.array([1, 2, 3, 4])
print(sum(a))

print(a.sum())
print(a.mean())
print(a.std())
print(a.var())

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(A.sum())
print(A.mean())
print(A.std())
print(A.var())

# sum rows
print(A.sum(axis=0))

# sum columns
print(A.sum(axis=1))

print(A.mean(axis=0))
print(A.std(axis=0))

print(A.mean(axis=1))
print(A.std(axis=1))

# Broadcasting and Vectorized operations
a = np.arange(4)
print(a)

print(a+10)
print(a*10)

a *= 100
print(a)

l = [0, 1, 2, 3]
[i * 10 for i in l]
print(l)
print([i * 10 for i in l])


a = np.arange(4)
b = np.array([10, 10, 10, 10])
print(a)
print(b)
print(a+b)
print(a*b)

# boolean arrays; also called masks
a = np.arange(4)
print(a)

# print 0th and last element in array
print(a[0], a[-1])

# multi-index method to select 0th and last element
print(a[[0, -1]])

# boolean method to select elements in the array; select the 0th & 4th element
print(a[[True, False, False, True]])

print(a)
# filtering | query mechanism; print true | false for indexes with values >= 2 in array a
print(a >= 2)

#print(a > 1)
#print(a > 0)
#print(a >= 0)

# filtering elements in the array
a[a >= 2]
print(a[a >= 2])

print(a.mean())

# print values > mean in array a
print(a[a > a.mean()])

# print invert of values > mean in array a
print(a[~(a > a.mean())])

# print values = 0 or 1
print(a[(a == 0) | (a == 1)])

# print is condition = a <= 2 AND a divisible by 2
print(a[(a <= 2) & (a % 2 == 0)])

A = np.random.randint(100, size=(3, 3))
print(A)

# print values in array A at locations masked by new array (values = true)
print(A[np.array([
    [True, False, True],
    [False, True, False],
    [True, False, True]
])])

# print mask for locations > 30 in array A; generates the true | false matrix which can be used for filtering
print(A > 30)

# print values > 30 in array A
print(A[A > 30])

# linear algebra
# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/numpy-algebra-and-size

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [6, 5],
    [4, 3],
    [2, 1]
])

# dot product
print(A.dot(B))

# cross product
print(A @ B)

# transform B
print(B.T)

print(A)

print(B.T @ A)

# Size of objects in Memory; Int, floats

# An integer in Python is > 24bytes
print(sys.getsizeof(1))

# Longs are even larger
print(sys.getsizeof(10**100))

# Numpy size is much smaller
print(np.dtype(int).itemsize)

# Numpy size is much smaller
print(np.dtype(np.int8).itemsize)

print(np.dtype(float).itemsize)

# Lists are even larger

# A one-element list
print(sys.getsizeof([1]))

# An array of one element in numpy
print(np.array([1]).nbytes)

# performance/speed
l = list(range(100000))
a = np.arange(100000)

# numpy will be extremely faster compared to python
# %time np.sum(a ** 2)

# %time sum([x == 2 for x in l])


# Useful Numpy functions
# random
print(np.random.random(size=2))

print(np.random.normal(size=2))

print(np.random.rand(2, 4))

# arange
print(np.arange(10))
print(np.arange(5, 10))

# range 0 to 1 in increments of 0.1
print(np.arange(0, 1, .1))

# reshape
print(np.arange(10).reshape(2, 5))
print(np.arange(10).reshape(5, 2))

# linspace
# 0 to 100, 5 values equally spaced
print(np.linspace(0, 100, 5))

print(np.linspace(0, 1, 5))

# 0 to 1, 20 values
print(np.linspace(0, 1, 20))

print(np.linspace(0, 1, 20, dtype=float))
print(np.linspace(0, 1, 20, dtype=int))

print(np.linspace(0, 1, 20, False))

# zeros, ones, empty
print(np.zeros(5))
print(np.zeros((3, 3)))
print(np.zeros((3, 3), dtype=int))

print(np.ones(5))
print(np.ones((3, 3)))

print(np.empty(5))
print(np.empty((2, 2)))

# identity and eye
print('identity & eye')
print(np.identity(3))
print(np.eye(3, 3))

print('identity & eye: eye 8,4')
print(np.eye(8, 4))

print('identity & eye: eye 8,4, k=1')
print(np.eye(8, 4, k=1))

print('identity & eye: eye 8,4, k=-3')
print(np.eye(8, 4, k=-3))

print("Hello World"[6])
















