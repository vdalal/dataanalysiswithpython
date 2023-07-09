# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-pandas/blob/master/2%20-%20Pandas%20Series%20exercises.ipynb

# Import the numpy package under the name np
import numpy as np

# Import the pandas package under the name pd
import pandas as pd

# Print the pandas version and the configuration
print(pd.__version__)

# Create an empty pandas Series
pd.Series()

# Given the X python list convert it to an Y pandas Series
X = ['A','B','C']
print(X, type(X))

Y = pd.Series(X)
print(Y, type(Y)) # different type

# Given the X pandas Series, name it 'My letters'
X = pd.Series(['A','B','C'])

X.name = 'My letters'
print(X)

# Given the X pandas Series, show its values
X = pd.Series(['A','B','C'])

print(X.values)

# Series indexation
# Assign index names to the given X pandas Series
X = pd.Series(['A','B','C'])
print('Series X:', X)
index_names = ['first', 'second', 'third']

X.index = index_names
print('Series X:', X)

# Given the X pandas Series, show its first element
X = pd.Series(['A','B','C'], index=['first', 'second', 'third'])
# #X[0] # by position
#X.iloc[0] # by position
print(X['first']) # by index

# Given the X pandas Series, show its last element
print(X[-1])
print(X.iloc[-1]) # by position
print(X['third']) # by index

# Given the X pandas Series, show all middle elements
X = pd.Series(['A','B','C','D','E'],
              index=['first','second','third','forth','fifth'])
print(X[1:-1])
print('By key', X[['second', 'third', 'forth']])
#X.iloc[1:-1] # by position
print(X[1:-1]) # by position

# Given the X pandas Series, show the elements in reverse position
#X.iloc[::-1]
print(X[::-1])

# Given the X pandas Series, show the first and last elements
X = pd.Series(['A','B','C','D','E'],
              index=['first','second','third','forth','fifth'])

print(X[['first', 'fifth']])
#X.iloc[[0, -1]]
print(X[[0, -1]])

# Series manipulation
# Convert the given integer pandas Series to float
X = pd.Series([1,2,3,4,5],
              index=['first','second','third','forth','fifth'])

pd.Series(X, dtype=float)
print(X)

# Reverse the given pandas Series (first element becomes last)
X = pd.Series([1,2,3,4,5],
              index=['first','second','third','forth','fifth'])

X = X[::-1]
print(X)

# Order (sort) the given pandas Series
X = pd.Series([4,2,5,1,3],
              index=['forth','second','fifth','first','third'])

X = X.sort_values()
print(X)

# Given the X pandas Series, set the fifth element equal to 10
X = pd.Series([1,2,3,4,5],
              index=['A','B','C','D','E'])

X[4] = 10
print(X)
X['D'] = 10
print(X)

# Given the X pandas Series, change all the middle elements to 0
X = pd.Series([1,2,3,4,5],
              index=['A','B','C','D','E'])

X[1:-1] = 0
print(X)

# Given the X pandas Series, add 5 to every element
X = pd.Series([1,2,3,4,5])
X + 5
print(X)

X = X + 5
print(X)

# Series boolean arrays (also called masks)
# Given the X pandas Series, make a mask showing negative elements
X = pd.Series([-1,2,0,-4,5,6,0,0,-9,10])
mask = X < 0
print(mask)

# Given the X pandas Series, get the negative elements
print(X * mask)
print(X[mask])

# Given the X pandas Series, get numbers higher than 5
X = pd.Series([-1,2,0,-4,5,6,0,0,-9,10])

mask = X > 5
print(X[mask])

# Given the X pandas Series, get numbers higher than the elements mean
mask = X > X.mean()
print(X[mask])

# Given the X pandas Series, get numbers equal to 2 or 10
mask = (X == 2) | (X == 10)
print(X[mask])

# Logic functions
# Given the X pandas Series, return True if none of its elements is zero
X = pd.Series([-1,2,0,-4,5,6,0,0,-9,10])

print(X.all())

# Given the X pandas Series, return True if any of its elements is zero
print(X.any())

# Summary statistics
# Given the X pandas Series, show the sum of its elements
X = pd.Series([3,5,6,7,2,3,4,9,4])

#np.sum(X)
print(X.sum())

# Given the X pandas Series, show the mean value of its elements
#np.mean(X)
print(X.mean())

# Given the X pandas Series, show the max value of its elements
X = pd.Series([1,2,0,4,5,6,0,0,9,10])
#np.max(X)
print(X.max())









