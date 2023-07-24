# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/data-cleaning-introduction
# https://github.com/ine-rmotr-curriculum/data-cleaning-rmotr-freecodecamp/blob/master/1%20-%20Missing%20Data.ipynb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# What does "missing data" mean? What is a missing value? It depends on the origin of the data and the
# context it was generated. For example, for a survey, a Salary field with an empty value, or a number 0,
# or an invalid value (a string for example) can be considered "missing data". These concepts are related
# to the values that Python will consider "Falsy"
falsy_values = (0, False, None, '', [], {})

# For Python, all the values above are considered "falsy":
any(falsy_values)

# Numpy has a special "nullable" value for numbers which is np.nan. It's NaN: "Not a number")
# The np.nan value is kind of a virus. Everything that it touches becomes np.nan:
print(3 + np.nan)

a = np.array([1, 2, 3, np.nan, np.nan, 4])

print(a.sum())
print(a.mean())

# This is better than regular None values, which in the previous examples would have raised an exception
# print(3 + None)

# For a numeric array, the None value is replaced by np.nan
a = np.array([1, 2, 3, np.nan, None, 4], dtype='float')
print(a)

# As we said, np.nan is like a virus. If you have any nan value in an array and you try to perform an operation on it,
# you'll get unexpected results
a = np.array([1, 2, 3, np.nan, np.nan, 4])

print(a.mean())

# Numpy also supports an "Infinite" type which also behaves as a virus
print(np.inf)

print(3 + np.inf)

print(np.inf / 3)

print(np.inf / np.inf)

b = np.array([1, 2, 3, np.inf, np.nan, 4], dtype='float')
print(b.sum())

# Checking for nan or inf
# There are two functions: np.isnan and np.isinf that will perform the desired check
print(np.isnan(np.nan))

print(np.isinf(np.inf))

# And the joint operation can be performed with np.isfinite
print(np.isfinite(np.nan), np.isfinite(np.inf))

# np.isnan and np.isinf also take arrays as inputs, and return boolean arrays as results
print(np.isnan(np.array([1, 2, 3, np.nan, np.inf, 4])))

print(np.isinf(np.array([1, 2, 3, np.nan, np.inf, 4])))

print(np.isfinite(np.array([1, 2, 3, np.nan, np.inf, 4])))

# Filtering them out
# Whenever you're trying to perform an operation with a Numpy array and you know there might be missing values,
# you'll need to filter them out before proceeding, to avoid nan propagation. We'll use a combination of the
# previous np.isnan + boolean arrays for this purpose
a = np.array([1, 2, 3, np.nan, np.nan, 4])

print(a[~np.isnan(a)])

# Which is equivalent to
print(a[np.isfinite(a)])

# And with that result, all the operation can be now performed
print(a[np.isfinite(a)].sum())

print(a[np.isfinite(a)].mean())

# Handling Missing Data with Pandas
# https://github.com/ine-rmotr-curriculum/data-cleaning-rmotr-freecodecamp/blob/master/2%20-%20Handling%20Missing%20Data%20with%20Pandas.ipynb




