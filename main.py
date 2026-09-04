from src.train import training, training_rf
from src.evaluate import evaluate, evaluate_rf

clf, X_test, y_test = training_rf('data/ecomerce_data_E_Comm.csv')
evaluate_rf(clf, X_test, y_test)

