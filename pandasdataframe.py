# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-pandas/blob/master/3%20-%20Pandas%20-%20DataFrames.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

print('Dataframe columns:', df.columns)

print('Dataframe index:', df.index)

# Quick information about the structure of the dataframe
print('Dataframe info:', df.info)

print('Dataframe size:', df.size)

print('Dataframe shape:', df.shape)

# Summary stats of the dataframe; for all numeric columns e.g. continents is object (string) type and is thus missing
print('Dataframe summary stats for numeric columns:', df.describe())

print('df.dtypes:', df.dtypes)

print('df.dtypes.value_counts:', df.dtypes.value_counts())

# Indexing, Selection and Slicing
# Individual columns in the DataFrame can be selected with regular indexing. Each column is represented as a Series
# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/pandas-dataframes
print(df)

# Select rows by index with the loc attribute
print('Select rows by index with loc = Canada', df.loc['Canada'])

# print last element
# Select rows by position with the loc attribute
print('Select rows by position with iloc e.g. 1 = first, 2 = second & -1 = last', df.iloc[-1])

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

# all rows from France and Italy; inclusive
print(df.loc['France': 'Italy'])

# Now with a second dimension, i.e. population i.e. include column(s) in your sliced data
print(df.loc['France': 'Italy', 'Population'])

# select dimensions (columns = Population & GDP)
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
# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/pandas-conditional-selection-and-modifying-dataframes

print(df)

# Generate boolean selection
print('Generate boolean selection', df['Population'] > 70)

# Generate and insert boolean selection; output is a different dataframe
print(df.loc[df['Population'] > 70])

# The boolean matching is done at Index level, so you can filter by any row, as long as it contains the right indexes.
# Column selection still works as expected
# Multi dimension selection
print(df.loc[df['Population'] > 70, 'Population'])

# Multi dimension selection
print(df.loc[df['Population'] > 70, ['Population', 'GDP']])

# Dropping stuff
# Opposed to the concept of selection, we have "dropping". Instead of pointing out which values you'd like to
# select you could point which ones you'd like to drop

# Note: immutable; does NOT change the underlying dataframe
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

# Broadcasting Operations
print(df[['Population', 'GDP']])

print(df[['Population', 'GDP']] / 100)

# Operations with Series work at a column level, broadcasting down the rows (which can be counterintuitive)
crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])
print(crisis)

print(df[['GDP', 'HDI']])

# Add crises from original
# Result = subtract 1 mil for each GDP and 0.3 for each HDI
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

# Pandas matches the indexes and assigns the language
df['Language'] = langs

# Modifying original dataframe by adding new column (language)
# Typically with the = operator, you are modifying the original dataframe
print('Modified dataframe', df)

# Replacing values per column
df['Language'] = 'English'
print(df)

# Renaming Columns
# Changes index. Note: Argentina does not exist but it does not cause a problem
print('Modified dataframe index', df.rename(
    columns={
        'HDI': 'HD Index',
        'Annual Popcorn Consumption': 'APC'
    }, index={
        'United States': 'USA',
        'United Kingdom': 'UK',
        'Argentina': 'AR'
    }))

print('Original dataframe is unmodified (immutable operation)', df)

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
# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-course/pandas-creating-columns

df[['Population', 'GDP']]

print(df)

print(df['GDP'] / df['Population'])

# The result of that operation (a series) is just another series that you can add to the original DataFrame
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


# Test questions
certificates_earned = pd.DataFrame({
    'Certificates': [8, 2, 5, 6],
    'Time (in months)': [16, 5, 9, 12]
})

certificates_earned.index = ['Tom', 'Kris', 'Ahmad', 'Beau']
print(certificates_earned.iloc[2])

certificates_earned = pd.DataFrame({
    'Certificates': [8, 2, 5, 6],
    'Time (in months)': [16, 5, 9, 12]
})
names = ['Tom', 'Kris', 'Ahmad', 'Beau']

certificates_earned.index = names
longest_streak = pd.Series([13, 11, 9, 7], index=names)
certificates_earned['Longest streak'] = longest_streak

print(certificates_earned)

# Pandas: read_csv() function
# https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-pandas/blob/master/5%20-%20Pandas%20-%20Reading%20CSV%20and%20Basic%20Plotting.ipynb
# CSV file: https://github.com/ine-rmotr-curriculum/freecodecamp-intro-to-pandas/blob/master/data/btc-market-price.csv
df = pd.read_csv('data/btc-market-price.csv')
print('Print dataframe of BTC prices read from csv file', df)

print(df.head())

# The CSV file we're reading has only two columns: timestamp and price. It doesn't have a header, it contains
# whitespaces and has values separated by commas. pandas automatically assigned the first row of data as headers,
# which is incorrect. We can overwrite this behavior with the header parameter

df = pd.read_csv('data/btc-market-price.csv', header=None)
# First n rows; by default = 5
print(df.head())

# Set the names of each column explicitly by setting the df.columns attribute
df.columns = ['Timestamp', 'Price']
print(df.head())

# Little more information on the dataframe
print(df.info())

print(df.shape)

# Print last 3 rows
print(df.tail(3))

# Note: The type of the Price column was correctly interpreted as float, but the Timestamp was interpreted as a
# regular string (object in pandas notation)

print(df.dtypes)

# We can perform a vectorized operation to parse all the Timestamp values as Datetime objects
# Timestamp is injested as an object (string). It can be converted to a Timestamp
pd.to_datetime(df['Timestamp']).head()
print(pd.to_datetime(df['Timestamp']).head())

# Assignment operator = changes the original dataframe
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print(df.head())
print(df.dtypes)

# Change the autoincremented ID generated by pandas and use the Timestamp DS column as the Index
# Set index to timestamp
df.set_index('Timestamp', inplace=True)

print(df.head())

print(df.loc['2017-09-29'])

# Summary of all steps we performed above
df = pd.read_csv('data/btc-market-price.csv', header=None)
df.columns = ['Timestamp', 'Price']
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

print(df.head())

# Do all operations in one shot; like in a script
df = pd.read_csv(
    'data/btc-market-price.csv',
    header=None,
    names=['Timestamp', 'Price'],
    index_col=0,
    parse_dates=True
)

print(df.loc['2017-09-29'])

# Plotting basics
# pandas integrates with Matplotlib and creating a plot
df.plot()
print(df.plot())

# plt.plot() accepts many parameters, but the first two ones are the most important ones: the values for the X and Y axes
print(plt.plot(df.index, df['Price']))

x = np.arange(-10, 11)
print(plt.plot(x, x ** 2))

plt.plot(x, x ** 2)
plt.plot(x, -1 * (x ** 2))

# Each plt function alters the global state. If you want to set settings of your plot you can use the plt.figure function.
# Others like plt.title keep altering the global plot:

plt.figure(figsize=(12, 6))
plt.plot(x, x ** 2)
plt.plot(x, -1 * (x ** 2))

plt.title('My Nice Plot')

df.plot(figsize=(16, 9), title='Bitcoin Price 2017-2018')













