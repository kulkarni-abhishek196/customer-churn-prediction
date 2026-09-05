from src.train import training, training_rf, training_xgboost
from src.evaluate import evaluate, evaluate_rf, evaluate_xgboost
from src.explain import explain_model
import os
import joblib


clf, X_test, y_test = training_xgboost('data/ecomerce_data_E_Comm.csv')

os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/churn_model.pkl')
print("Model saved.")

evaluate_xgboost(clf, X_test, y_test)

