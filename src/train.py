from src.preprocess import load_and_preprocess
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def training(filepath):

    X, y, preprocessor = load_and_preprocess(filepath)
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf.fit(X_train, y_train)

    return clf, X_test, y_test



