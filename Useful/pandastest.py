# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:49:40 2020
https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
@author: Mariusz
"""


import pandas as pd
import numpy as np
# Create Series by passing list of values
s = pd.Series([1,3,5,np.nan, 6,8])
print (s)

# Creating DataFram by passing NumPy array with datatime index
dates = pd.date_range('20130101', periods=6)
print (dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print (df)

# Creating DataFrame by passing dict of objects
df2 = pd.DataFrame({'A': 1.,
                        'B': pd.Timestamp('20130102'),
                        'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                        'D': np.array([3] * 4, dtype='int32'),
                        'E': pd.Categorical(["test", "train", "test", "train"]),
                        'F': 'foo'})
print (df2)

print (df2.dtypes)

# viewing data
# n rows from top
print (df.head(2))

# n rows from bottom
print (df.tail(2))

# display index
print (df.index)

# display column
print (df.columns)

# transfer Data Frame to numpy. If datatypes are different then numpy will 
# find datatype which fits all of them. It may be object. Every time data need 
# to be used it will have to be cast

print (df.to_numpy())

# In this case array will be of type object
print (df2.to_numpy())

# show quick statistic of data
print (df.describe())

# transposing data
print (df.T)

# sorting data by axis
print (df)
print (df.sort_index(axis=1, ascending=False))

# sorting by value
print (df.sort_values(by='B'))

# Selection
# it is recommended to use .at, .iat, .loc, .iloc

# Select single column
print ()
print (df['A'])

# Selection of rows
print()
print (df[0:3])
print (df['20130102':'20130104'])

# Selection using label
print (dates[0])
print (df.loc[dates[0]])

# Selecting multi axis by label
print (df.loc[:, ['A','B']])

# Showing label slicing
print (df.loc['20130102':'20130104',['A','B']])

# Reduction in the dimentions 
print (df.loc['20130102',['A','B']])

# Getting scalar value
print()
print (df.loc[dates[0], 'A'])

# Selection by position

# passed by integers. Selection of Row
print (df.iloc[3])

# interger slices, kind of like Range in excel
print (df.iloc[3:5, 0:2])

# by list of integers with selection of columns
print()
print (df.iloc[[1,2,4],[0,2]])

# slicing rows explicitly
print(df.iloc[1:3,:])

# slicing columns explicitly
print(df.iloc[:, 1:3])

# getting value explicitly
print()
print (df.iloc[1,1])

# boolean indexing
# single columnt to get data
print(df[df['A']>0])

# selecting values from Data Frame where boolean condition is met
print (df[df > 0])

# using isin() for filtering

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print (df2)
print(df2[df2['E'].isin(['two', 'four'])])

# Setting

# setting a new column automatically aligns the data by the indexes
print()
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
print (s1)
df['F'] = s1

# setting value by label
df.at[dates[0],'A'] = 0
print (df)

# setting value by position
df.iat[5,3] = 0
print (df)

# setting by assigning with NumPy array
df.loc[:,'D'] = np.array([5]*len(df))
print (df)

# setting with where operation
df2 = df.copy()
df2[df2 > 0] = -df2
print (df2)

# Missing Data. use of np.nan to represent it.

df1  = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
print (df1)
df1.loc[dates[0]:dates[1],'E'] = 1
print (df1)

# drop rows with missing data
print(df1.dropna(how='any'))

# filling missing data
print(df1.fillna(value=5))

# getting boolean mask where values are nan
print (pd.isna(df1))

# Operations

# Statistics

# Operations in general exclude missing data
print (df.mean())

# Operation on other axis
print (df.mean(1))

# operating with objects with different dimention and needing alignment.
# in addition pandas automatically broadcasts along the specified dimension
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print (s)

print (df.sub(s, axis = 'index'))

# Apply
print (df.apply(np.cumsum))
print ()
print (df.apply(lambda x: x.max() - x.min(), axis = 0))

# Histogramming 
s = pd.Series(np.random.randint(0,7, size=10))
print (s)
print (s.value_counts())

# String Method
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print (s.str.upper())
print (s.str.lower())
print (s.str.len())

# Merge
df = pd.DataFrame(np.random.randn(10, 4))
print (df)
print ()
# brake into pieces
pieces = [df[:3], df[3:7], df[7:]]
print (pieces[1])
print ()
print (pd.concat(pieces))

# Join
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print (left)
print ()
print (right)
print ()
print (pd.merge(left, right, on='key'))

# Altrnative
print()
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
print (left)
print ()
print (right)
print ()
print (pd.merge(left, right, on='key'))

# Grouping
print ()
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

print (df)
print()

# Grouping and applying sum
print (df.groupby('A').sum())
print ()
print (df.groupby(['A','B']).sum())

# Reshaping

# Stack
print ()
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))

print (tuples)
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print ()
print (index)
print ()

df = pd.DataFrame(np.random.randn(8,2), index=index, columns=['A','B'])
print (df)
print()
df2 = df[:4]
print (df2)

# stack method "compresses" level in the datafram
stacked = df2.stack()
print ()
print (stacked)

# unstack
unstack = stacked.unstack()
print()
print (unstack)
print()
print (stacked.unstack(1))
print()
print (stacked.unstack(0))

# Pivot Tables
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})

print ()
print (df)

# produce pivot table
print (pd.pivot_table(df, values='D', index=['A','B'],columns =['C']))

# Time Series
rng = pd.date_range('1/1/2012', periods=100, freq='S')
print ()
print (rng)
print()
ts = pd.Series(np.random.randint(0,500, len(rng)), index=rng)
print (ts)
print ()
print (ts.resample('5Min').sum())

# Time Zone representation
print()

rng = pd.date_range('3/6/2012 00:00', periods = 5, freq='D')
print (rng)

ts = pd.Series(np.random.randn(len(rng)), rng)
print ()
print (ts)

ts_utc = ts.tz_localize('UTC')
print ()
print (ts_utc)

# Converting to another time zone
print()
print (ts_utc.tz_convert('US/Eastern'))

# Converting between time span representations

rng = pd.date_range('1/1/2012', periods=5, freq='M')
print ()
print (rng)
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print()
print(ts)

ps = ts.to_period()
print ()
print (ps)
print ()
print (ps.to_timestamp())

# Converting between period and timestamp. Example of converting quarterly
# frequency with year ending November to 9ap of the end of the month following
# the quarter end

prng = pd.period_range('1990Q1','2000Q4',freq='Q-NOV')
print ()
print (prng)
print (len(prng))
ts = pd.Series(np.random.randn(len(prng)),prng)
print ()
print (ts)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H','s') + 9
print ()
print (ts.index)
print()
print (ts.head())

# Categoricals

df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})

# convert rad grades to catergorical data type

df['grade'] = df['raw_grade'].astype("category")

print()
print (df)
print()
print (df['grade'])

# rename categories 
df['grade'].cat.categories = ["very good", "good", "very bad"]
print ()
print (df)

# reorder categories and simultaneously add missing categories
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium",
                                              "good", "very good"])

print ()
print (df['grade'])

# sorting as per order in the category
print ()
print (df.sort_values(by="grade"))

# grouping by a categorical column also shows emtpy category

print()
print(df.groupby("grade").size())

# Plotting

import matplotlib.pyplot as plt

plt.close('all')

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

print ()
print (ts)
ts = ts.cumsum()
print ()
print (ts)
#ts.plot()

# plotting all the columns

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A','B','C','D'])

df = df.cumsum()
#plt.legend(loc='best')
#plt.figure()

#df.plot()

# Getting Data in and out

# writing to csv file

df.to_csv('foo.csv')

# reading from csv
df = pd.read_csv('foo.csv')
print (df)

# writing to HDF5 Stores

df.to_hdf('foo.h5','df')

# reading from HDF5 store

df = pd.read_hdf('foo.h5','df')
print()
print (df)

# write to Excel
df.to_excel('foo.xlsx', sheet_name='data')

# read from Excel
df = pd.read_excel('foo.xlsx', 'data', index_col=None, na_values=['NA'])
print ()
print (df)