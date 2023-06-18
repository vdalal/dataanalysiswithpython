# numpy exercise: https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-numpy/blob/master/3.%20NumPy%20exercises.ipynb
import numpy
# Import the numpy package under the name np
import numpy as np

# Print the numpy version and the configuration
print(np.__version__)

# Array creation
# Create a numpy array of size 10, filled with zeros.
a = np.array([0]*10)
print(a)

np.zeros(10)
print(a)

# Create a numpy array with values ranging from 10 to 49
b = np.arange(10, 49)
print(b)

# Create a numpy matrix of 2*2 integers, filled with ones.
# c = numpy.ones([2,2], dtype=np.int)
c = numpy.ones([2,2], dtype=int)
print(c)

# Create a numpy matrix of 3*2 float numbers, filled with ones.
d = numpy.ones([3,2])
print(d)

d = numpy.ones([3,2], dtype=float)
print(d)

# Given the X numpy array, create a new numpy array with the same shape and type as X, filled with ones.
X = np.arange(4, dtype=int)
Y = np.ones_like(X)
print(X)
print(Y)

# Given the X numpy matrix, create a new numpy matrix with the same shape and type as X, filled with zeros.
X = np.array([[1,2,3], [4,5,6]], dtype=int)
Y = np.zeros_like(X)
print(X)
print(Y)

# Create a numpy matrix of 4*4 integers, filled with fives.
a = np.ones([4,4], dtype=int) * 5
print(a)

# this below prints 4 2*3 matrices X (numerous) times
a = np.ones([2,3] * 4)
print(a)

# Given the X numpy matrix, create a new numpy matrix with the same shape and type as X, filled with sevens.
X = np.array([2, 3], dtype=int)
Y = np.ones_like(X) * 7

print(Y)

# Create a 3*3 identity numpy matrix with ones on the diagonal and zeros elsewhere.
#np.eye(3)
print(np.identity(3))

# Create a numpy array, filled with 3 random integer values between 1 and 10.
X = np.random.randint(10, size=3)
print(X)

# Create a 3*3*3 numpy matrix, filled with random float values.
X = np.random.random((3,3,3))
print(X)

Y = np.random.randn(3,3,3) # 0 to 1 floats
print(Y)

# Given the X python list convert it to an Y numpy array
X = [1, 2, 3]
print(X, type(X))

Y = np.array(X)
print(Y, type(Y)) # different type

# Given the X numpy array, make a copy and store it on Y.
X = np.array([5,2,3], dtype=int)
print(X, id(X))

Y = np.copy(X)
print(Y, id(Y)) # different id

# Create a numpy array with numbers from 1 to 10
x = np.arange(1, 11)
print(x)

# Create a numpy array with the odd numbers between 1 to 10
y = np.arange(1, 11, 2)
print(y)

# Create a numpy array with numbers from 1 to 10, in descending order.
z = np.arange(1, 11)[::-1]
print(z)

# Create a 3*3 numpy matrix, filled with values ranging from 0 to 8
x = np.arange(9).reshape(3,3)
print(x)

# Show the memory size of the given Z numpy matrix
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))

# Array indexation
# Given the X numpy array, show it's first element
X = np.array(['A','B','C','D','E'])
print(X[0])

# Given the X numpy array, show it's last element
print(X[len(X)-1])
print(X[-1])

# Given the X numpy array, show it's first three elements
X = np.array(['A','B','C','D','E'])
print(X[0:3])

# Given the X numpy array, show all middle elements
X = np.array(['A','B','C','D','E'])
print(X[1:-1])

# Given the X numpy array, show the elements in reverse position
X = np.array(['A','B','C','D','E'])
print(X[::-1])

# Given the X numpy array, show the elements in an odd position
X = np.array(['A','B','C','D','E'])
#X[[0, 2, -1]]
print(X[::2])

# Given the X numpy matrix, show the first row elements
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

print(X[0])

# Given the X numpy matrix, show the last row elements
print(X[-1])

# Given the X numpy matrix, show the first element on first row
print(X[0][0])
print(X[0,0])

# Given the X numpy matrix, show the last element on last row
print(X[-1, -1])

# Given the X numpy matrix, show the middle row elements
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
])
print(X[1:-1])

# Given the X numpy matrix, show the first two elements on the first two rows
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

#X[:2][:2] wrong!
#X[0:2, 0:2]
print(X[:2, :2])

# Given the X numpy matrix, show the last two elements on the last two rows
print(X[2:, 2:])

# Array manipulation
# Convert the given integer numpy array to float
X = [-5, -3, 0, 10, 40]
Y = np.array(X, dtype=float)
print(X)
print(Y)

# Reverse the given numpy array (first element becomes last)
X = [-5, -3, 0, 10, 40]

Y = X[::-1]
print(Y)

# Order (sort) the given numpy array
X = [0, 10, -5, 40, -3]
print(X)
X.sort()
print(X)

# Given the X numpy array, set the fifth element equal to 1
X = np.zeros(10)
print(X)
X[4] = 1
print(X)

# Given the X numpy array, change the 50 with a 40
X = np.array([10, 20, 30, 50])
X[3] = 40
print(X)

# Given the X numpy matrix, change the last row with all 1
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

X[-1] = np.array([1, 1, 1, 1])
print(X)

# Given the X numpy matrix, change the last item on the last row with a 0
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])
X[-1][-1] = 0
print(X)

# Given the X numpy matrix, add 5 to every element
X += 5
print(X)

# Boolean arrays (also called masks)
# Given the X numpy array, make a mask showing negative elements
X = np.array([-1, 2, 0, -4, 5, 6, 0, 0, -9, 10])

mask = X <= 0
print(mask)

# Given the X numpy array, get the negative elements
X = np.array([-1, 2, 0, -4, 5, 6, 0, 0, -9, 10])
print(X[X<0])

# method 2
mask = X < 0
print(X[mask])

# Given the X numpy array, get numbers higher than 5
print(X[X>5])

# method 2
mask = X > 5
print(X[mask])

# Given the X numpy array, get numbers higher than the elements mean
print(X[X > X.mean()])

# method 2
mask = X > X.mean()
print(X[mask])

# Given the X numpy array, get numbers equal to 2 or 10
X = np.array([-1, 2, 0, -4, 5, 6, 0, 0, -9, 10])
print(X[(X == 2) | (X == 10)])

# method 2
mask = (X == 2) | (X == 10)
print(X[mask])

# Logic functions
# Given the X numpy array, return True if none of its elements is zero
X = np.array([-1, 2, 0, -4, 5, 6, 0, 0, -9, 10])
print(X.all())

# Given the X numpy array, return True if any of its elements is zero
print(X.any())

# Summary statistics
# Given the X numpy array, show the sum of its elements
X = np.array([3, 5, 6, 7, 2, 3, 4, 9, 4])
# np.sum(X)
print(X.sum())

# Given the X numpy array, show the mean value of its elements
# np.mean(X)
print(X.mean())

# Given the X numpy matrix, show the sum of its columns
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

print(X.sum(axis=0)) # remember: axis=0 columns; axis=1 rows

# Given the X numpy matrix, show the mean value of its rows
X = np.array([
    [1,   2,  3,  4],
    [5,   6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
])

print(X.mean(axis=1)) # remember: axis=0 columns; axis=1 rows

# Given the X numpy array, show the max value of its elements
X = np.array([1, 2, 0, 4, 5, 6, 0, 0, 9, 10])

#np.max(X)
print(X.max())


























