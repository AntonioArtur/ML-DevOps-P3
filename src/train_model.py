# Script to train machine learning model.
from ml.data import load_data, process_data
from ml.model import save_encoder, save_model, save_lb, inference,\
                     compute_model_metrics, train_model
from sklearn.model_selection import train_test_split

data = load_data()

train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

save_encoder(encoder)

save_lb(lb)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder
)

model = train_model(X_train, y_train)

save_model(model)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(lb.transform(y_test), preds)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")
