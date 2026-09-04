from src.train import training, training_rf, training_xgboost
from src.evaluate import evaluate, evaluate_rf, evaluate_xgboost
from src.explain import explain_model

clf, X_test, y_test = training_xgboost('data/ecomerce_data_E_Comm.csv')
evaluate_xgboost(clf, X_test, y_test)
explain_model(clf, X_test)

