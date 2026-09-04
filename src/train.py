from src.preprocess import load_and_preprocess
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, roc_auc_score

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
    param_grid = {
        'randomForest__n_estimators': [50, 100, 200, 300],
        'randomForest__max_depth': [10, 15, 20, 30],
        'randomForest__min_samples_split': [2, 5, 10],
        'randomForest__min_samples_leaf': [2, 4, 6],
        'randomForest__max_features': ['sqrt', 'log2']
    }

    X, y, preprocessor = load_and_preprocess(filePath)
    rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('randomForest', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=5,
                               scoring='recall',
                               n_jobs=-1,
                               verbose=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf.fit(X_train, y_train)
    #grid_search.fit(X_train, y_train)
    print("best parameters found:")
    print(grid_search.best_params_)

    #best_clf = grid_search.best_estimator_

# Training performance
    # y_train_pred = best_clf.predict(X_train)
    # y_test_pred = best_clf.predict(X_test)

    # print("Train recall:", recall_score(y_train, y_train_pred))
    # print("Test recall:", recall_score(y_test, y_test_pred))
    # print()
    # print("Train ROC-AUC:", roc_auc_score(y_train, best_clf.predict_proba(X_train)[:,1]))
    # print("Test ROC-AUC:", roc_auc_score(y_test, best_clf.predict_proba(X_test)[:,1]))
    return rf, X_test, y_test

def training_xgboost(filePath):

    X, y, preprocessor = load_and_preprocess(filePath)
    xg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgboost', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    xg.fit(X_train, y_train)

    return xg, X_test, y_test

