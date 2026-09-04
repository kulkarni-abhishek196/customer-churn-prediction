
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC: ", roc_auc_score(y_test, y_prob))
    print(confusion_matrix(y_test, y_pred))

def evaluate_rf(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC: ", roc_auc_score(y_test, y_prob))
    print(confusion_matrix(y_test, y_pred))

def evaluate_xgboost(xg, X_test, y_test):
    y_pred = xg.predict(X_test)
    y_prob = xg.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC: ", roc_auc_score(y_test, y_prob))
    print(confusion_matrix(y_test, y_pred))
