# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-pandas/blob/master/3%20-%20Pandas%20-%20DataFrames.ipynb

import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Population': [35.467, 63.951, 80.94 , 60.665, 127.061, 64.511, 318.523],
    'GDP': [
        1785387,
        2833687,
        3874437,
        2167744,
        4602367,
        2950039,
        17348075
    ],
    'Surface Area': [
        9984670,
        640679,
        357114,
        301336,
        377930,
        242495,
        9525067
    ],
    'HDI': [
        0.913,
        0.888,
        0.916,
        0.873,
        0.891,
        0.907,
        0.915
    ],
    'Continent': [
        'America',
        'Europe',
        'Europe',
        'Europe',
        'Asia',
        'Europe',
        'America'
    ]
}, columns=['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'])

print(df)

df.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]

print(df)

print(df.columns)

print(df.index)

print(df.info)

print(df.size)

print(df.shape)

print(df.describe())

print('df.dtypes:', df.dtypes)

print('df.dtypes.value_counts:', df.dtypes.value_counts())

# Indexing, Selection and Slicing
# Individual columns in the DataFrame can be selected with regular indexing. Each column is represented as a Series
print(df)

print('Index = Canada', df.loc['Canada'])

# print last element
print('Index = last', df.iloc[-1])

# print populations
print(df['Population'])

# index of the returned Series is the same as the DataFrame one. And its name is the name of the column.
# If you're working on a notebook and want to see a more DataFrame-like format you can use the to_frame method
print(df['Population'].to_frame())

# select multiple columns
print(df[['Population', 'GDP']])

# the result is another DataFrame. Slicing works differently, it acts at "row level"
print(df[1:3])

# Row level selection works better with loc and iloc which are recommended over regular "direct slicing" (df[:]).
# loc selects rows matching the given index
print(df.loc['Italy'])

# all rows between France and Italy; inclusive
print(df.loc['France': 'Italy'])

# As a second "argument", you can pass the column(s) you'd like to select
print(df.loc['France': 'Italy', 'Population'])

# select columns Population & GDP
print(df.loc['France': 'Italy', ['Population', 'GDP']])

# iloc works with the (numeric) "position" of the index
print(df.iloc[0])

print(df.iloc[-1])

print(df.iloc[[0, 1, -1]])

print(df.iloc[1:3])

# Filter on 3rd column
print(df.iloc[1:3, 3])

# Filter on 0th & 3rd columns
print(df.iloc[1:3, [0, 3]])

print(df.iloc[1:3, 1:3])

# Conditional selection (boolean arrays)
# We saw conditional selection applied to Series and it'll work in the same way for DataFrames.
# After all, a DataFrame is a collection of Series

print(df)

print(df['Population'] > 70)

print(df.loc[df['Population'] > 70])

# The boolean matching is done at Index level, so you can filter by any row, as long as it contains the right indexes.
# Column selection still works as expected
print(df.loc[df['Population'] > 70, 'Population'])

print(df.loc[df['Population'] > 70, ['Population', 'GDP']])

# Dropping stuff
# Opposed to the concept of selection, we have "dropping". Instead of pointing out which values you'd like to
# select you could point which ones you'd like to drop

print(df.drop('Canada'))

print(df.drop(['Canada', 'Japan']))

# Drop based on column indexes
print(df.drop(columns=['Population', 'HDI']))

# Drop based on rows, axis = 0
print(df.drop(['Italy', 'Canada'], axis=0))

# Drop columns, axis = 1
print(df.drop(['Population', 'HDI'], axis=1))

print(df.drop(['Population', 'HDI'], axis='columns'))

print(df.drop(['Canada', 'Germany'], axis='rows'))

# All these drop methods return a new DataFrame. If you'd like to modify it "in place", you can use the
# inplace attribute

# Operations
print(df[['Population', 'GDP']])

print(df[['Population', 'GDP']] / 100)

# Operations with Series work at a column level, broadcasting down the rows (which can be counter intuitive)
crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])
print(crisis)

print(df[['GDP', 'HDI']])

# Add crises from original
print(df[['GDP', 'HDI']] + crisis)

# Modifying DataFrames
# It's simple and intuitive, You can add columns, or replace values for columns without issues
# Adding a new column
langs = pd.Series(
    ['French', 'German', 'Italian'],
    index=['France', 'Germany', 'Italy'],
    name='Language'
)

print(langs)

df['Language'] = langs

print(df)

# Replacing values per column
df['Language'] = 'English'
print(df)

# Renaming Columns
print(df.rename(
    columns={
        'HDI': 'HD Index',
        'Annual Popcorn Consumption': 'APC'
    }, index={
        'United States': 'USA',
        'United Kingdom': 'UK',
        'Argentina': 'AR'
    }))

print(df)

print(df.rename(index=str.upper))

# ???
print(df.rename(index=lambda x: x.lower()))
print(df)

# Dropping columns
df.drop(columns='Language', inplace=True)
print(df)

# Adding values
print(df._append(pd.Series({
    'Population': 3,
    'GDP': 5
}, name='China')))

# df = df._append(pd.Series({
#    'Population': 3,
#    'GDP': 5
# }, name='China'))

print(df)

df.loc['China'] = pd.Series({'Population': 1_400_000_000, 'Continent': 'Asia'})
print(df)

df.drop('China', inplace=True)
print(df)

# More radical index changes
# ???
df.reset_index()
df.set_index('Population')
print(df)

# Creating columns from other columns
# Altering a DataFrame often involves combining different columns into another. For example,
# in our Countries analysis, we could try to calculate the "GDP per capita", which is just, GDP / Population
df[['Population', 'GDP']]

print(df)

print(df['GDP'] / df['Population'])

# The result of that operation is just another series that you can add to the original DataFrame
df['GDP Per Capita'] = df['GDP'] / df['Population']
print(df)

# Statistical info
# You've already seen the describe method, which gives you a good "summary" of the DataFrame
df.head()
print(df.head())

print(df.describe())

population = df['Population']

print(population)

print(population.min(), population.max())

print(population.sum())

print(population.sum() / len(population))

print(population.mean())

population.mean()
population.std()
population.median()
population.describe()

print(population.quantile(.25))

print(population.quantile([.2, .4, .6, .8, 1]))















