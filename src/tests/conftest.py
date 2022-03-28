import pytest
import pandas as pd

@pytest.fixture(scope="session")
def data(request):

    data = pd.read_csv("./data/census.csv")

    return data

@pytest.fixture(scope="session")
def clean_data(request):

    clean_data = pd.read_csv("./data/clean_census.csv")

    return clean_data