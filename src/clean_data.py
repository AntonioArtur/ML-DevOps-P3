import pandas as pd

data = pd.read_csv("../data/census.csv")

data.columns = [column.strip() for column in data.columns]

data = data.dropna()

data.to_csv("../data/clean_census.csv", index=False)
