import pandas as pd

data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.describe())

