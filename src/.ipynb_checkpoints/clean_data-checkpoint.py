import pandas as pd

data = pd.read_csv("../data/census.csv")

data = data.dropna()

data.to_csv("../data/clean_census.csv", index=False)