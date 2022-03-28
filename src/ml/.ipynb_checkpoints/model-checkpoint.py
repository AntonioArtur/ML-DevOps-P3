import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

def load_model(path="../model/model.pkl"):
    return joblib.load(path)

def load_encoder(path="../model/encoder.pkl"):
    return joblib.load(path)

def load_lb(path="../model/lb.pkl"):
    return joblib.load(path)

def save_model(model):
    joblib.dump(model, "../model/model.pkl")

def save_encoder(encoder):
    joblib.dump(encoder, "../model/encoder.pkl")
    
def save_lb(lb):
    joblib.dump(lb, "../model/lb.pkl")

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
