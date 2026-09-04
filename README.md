# Customer Churn Prediction

Predicting customer churn for an e-commerce platform using ML to help businesses identify at-risk customers before they leave.

## 2. Problem Statement

E-commerce company losing customers without knowing why.

**Goal:** Build a model to identify customers likely to churn.

**Business value:** Proactive retention is cheaper than acquiring new customers.

## 3. Dataset

* **5,630 customers**
* **20 features**
* **Source:** Kaggle E-Commerce Customer Churn dataset
* **Target:** Binary churn flag

  * 83% not churned
  * 17% churned
* **Class imbalance:** Noted and addressed

## 4. Project Structure

```text
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
```

## 5. Approach / Methodology

EDA — correlation analysis, churn rate by category, class imbalance check
Imputation — domain-informed fills (Tenure→0, DaySinceLastOrder→max, CouponUsed→0)
Category merging — Phone/Mobile Phone, CC/Credit Card, Mobile/Mobile Phone
Encoding — OHE for nominal categoricals, StandardScaler for numericals
Pipeline — sklearn Pipeline + ColumnTransformer for clean preprocessing
Models — Logistic Regression → Random Forest (GridSearchCV) → XGBoost
