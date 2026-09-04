# Customer Churn Prediction
Predicting customer churn for an e-commerce platform using ML.
to help businesses identify at-risk customers before they leave.

#2. Problem Statement
E-commerce company losing customers without knowing why
Goal: build a model to identify customers likely to churn
Business value: proactive retention is cheaper than acquiring new customers

#3. Dataset
5,630 customers, 20 features
Source: Kaggle E-Commerce Customer Churn dataset
Target: binary churn flag (83% not churned, 17% churned)
Class imbalance noted and addressed

#4. Project Structure
customer-churn-prediction/
├── data/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── models/
├── outputs/
├── app.py
├── requirements.txt
└── README.md