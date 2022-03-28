import json
from ml.data import *
from ml.model import * 

data = load_data()
model = load_model()
encoder=load_encoder()
lb=load_lb()

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

def slice_data(df, feature):
    assert feature in cat_features
    lines = []
    print(f"Slicing on feature: {feature}")
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        X, y, _, _ = process_data(df_temp, categorical_features=cat_features, label="salary", 
                     training=False, encoder=encoder, lb=lb)
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(lb.transform(y), preds)
        print(f"Precision for {feature} =={cls}:  {precision}")
        print(f"Recall for {feature} =={cls}:  {recall}")
        print(f"Fbeta for {feature} =={cls}:  {fbeta}")
        line = {"class":cls, "precision":precision, "recall":recall, "fbeta":fbeta}
        lines.append(line)
    return lines        
 
feature_dict = dict()

for feature in cat_features:
    feature_dict[feature] = slice_data(data, feature)
    
with open('../slice_output.txt', 'w') as file:
     file.write(json.dumps(feature_dict))