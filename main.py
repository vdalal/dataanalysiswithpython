import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# wget https://raw.githubusercontent.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example/master/data/sales_d
# ata.csv -o ./sales_data.csv
sales = pd.read_csv(
    'sales_data.csv',
    parse_dates=['Date'])

sales.head()

sales.shape

sales.info()

sales.describe()

sales['Unit_Cost'].describe()

sales['Unit_Cost'].mean()

sales['Revenue'].describe()

sales['Revenue'].mean()

print('the end')


