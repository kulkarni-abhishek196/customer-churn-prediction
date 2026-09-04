from src.preprocess import load_and_preprocess
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def training(filepath):

    X, y, preprocessor = load_and_preprocess(filepath)
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def training_rf(filePath):

    X, y, preprocessor = load_and_preprocess(filePath)
    rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('randomForest', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf.fit(X_train, y_train)

    return rf, X_test, y_test

def training_xgboost(filePath):

    X, y, preprocessor = load_and_preprocess(filePath)
    xg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgboost', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    xg.fit(X_train, y_train)

    return xg, X_test, y_test

