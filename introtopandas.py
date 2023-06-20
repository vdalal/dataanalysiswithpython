# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-pandas/blob/master/1%20-%20Pandas%20-%20Series.ipynb
# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/pandas-introduction

import pandas as pd
import numpy as np

# Pandas Series
# We'll start analyzing "The Group of Seven". Which is a political formed by Canada, France, Germany, Italy, Japan, the United Kingdom and the United States. We'll start by analyzing population, and for that, we'll use a pandas.Series object.

# In millions
g7_pop = pd.Series([35.467, 63.951, 80.940, 60.665, 127.061, 64.511, 318.523])
print(g7_pop)

g7_pop.name = 'G7 Population in millions'
print(g7_pop)

print(g7_pop.dtype)

print(g7_pop.values)

print(type(g7_pop.values))

# And they look like simple Python lists or Numpy Arrays. But they're actually more similar to Python dicts.
# A Series has an index, that's similar to the automatic index assigned to Python's lists:
print(g7_pop)

print(g7_pop[0])

print(g7_pop[1])

print(g7_pop.index)

l = ['a', 'b', 'c']

g7_pop.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]

print(g7_pop)

pd.Series({
    'Canada': 35.467,
    'France': 63.951,
    'Germany': 80.94,
    'Italy': 60.665,
    'Japan': 127.061,
    'United Kingdom': 64.511,
    'United States': 318.523
}, name='G7 Population in millions')

print(pd.Series)

pd.Series(
    [35.467, 63.951, 80.94, 60.665, 127.061, 64.511, 318.523],
    index=['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom',
           'United States'],
    name='G7 Population in millions')

print(pd.Series)

print(pd.Series(g7_pop, index=['France', 'Germany', 'Italy', 'Spain']))

# Indexing
# Indexing works similarly to lists and dictionaries, you use the index of the element you're looking for

print(g7_pop)

# index 'Canada'
print(g7_pop['Canada'])

print(g7_pop['Japan'])

# Numeric positions can also be used, with the iloc attribute
print(g7_pop.iloc[0])

# last element
print(g7_pop.iloc[-1])

print(g7_pop[['Italy', 'France']])

print(g7_pop[['Italy', 'France']])

# range; from Canada to Italy
print(g7_pop['Canada': 'Italy'])

# Conditional selection (boolean arrays)
# The same boolean array techniques we saw applied to numpy arrays can be used for Pandas Series:
print(g7_pop)

# returns bool array with countries with population > 70 mil
print(g7_pop > 70)

# returns array with countries whose population is > 70 mil
print(g7_pop[g7_pop > 70])

# mean: returns mean
print(g7_pop.mean())

# returns array with countries whose populations are > mean
print(g7_pop[g7_pop > g7_pop.mean()])

# standard deviation
print(g7_pop.std())

# ~ not
# | or
# & and
print(g7_pop[(g7_pop > g7_pop.mean() - g7_pop.std() / 2) | (g7_pop > g7_pop.mean() + g7_pop.std() / 2)])

# Operations and methods
# Series also support vectorized operations and aggregation functions as Numpy
print(g7_pop)

g7_pop = g7_pop * 1000000
print(g7_pop)



