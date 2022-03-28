from ml.data import *
from ml.model import * 

def test_null_removal(clean_data):
    assert len(clean_data)==len(clean_data.dropna())
    
def test_processing(clean_data):
    model = load_model()
    encoder = load_encoder()
    lb = load_lb()
    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                ]
    X, y, _, _ = process_data(
    clean_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    assert X.shape[0]==y.shape[0]
    
def test_infer(clean_data):
    model = load_model()
    encoder = load_encoder()
    lb = load_lb()
    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                ]
    X, y, _, _ = process_data(
    clean_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    predictions = inference(model, X)
    assert predictions.shape==y.shape