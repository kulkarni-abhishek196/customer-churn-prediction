from src.train import training
from src.evaluate import evaluate

clf, X_test, y_test = training('data/ecomerce_data_E_Comm.csv')
evaluate(clf, X_test, y_test)

