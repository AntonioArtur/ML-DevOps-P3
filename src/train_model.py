# Script to train machine learning model.
from ml.data import *
from ml.model import * 
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

# Add code to load in the data.

data = load_data()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

#Save encoder
save_encoder(encoder)

#Save lb
save_lb(lb)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder
)

# Train and save a model.
model = train_model(X_train, y_train)

#Save model
save_model(model)

#Do inference
preds = inference(model, X_test)

#Compute metrics
precision, recall, fbeta = compute_model_metrics(lb.transform(y_test), preds)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")