# Script to train machine learning model.
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Add the necessary imports for the starter code.

# Add code to load in the data.

data = pd.read_csv("../data/census.csv")

#Processing function

def process_data(dataset, categorical_features, label, training=True, encoder=None):

    if training:
        numerical_features = [feature for feature in dataset.columns if feature not in categorical_features + [label]]

        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
            ])

        encoder = ColumnTransformer([
                ("num", num_pipeline, numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ])
        
        X = encoder.fit_transform(dataset).todense()
        
    else:
        X = encoder.transform(dataset).todense()
        
    y = np.array(dataset[label].tolist())
    
    return X, y, encoder, dataset[label].unique().tolist()


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

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder
)

# Train and save a model.
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
joblib.dump(clf, "../model/model.pkl")

#Save encoder
joblib.dump(encoder, "../model/encoder.pkl")