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

* **EDA** — Correlation analysis, churn rate by category, class imbalance check
* **Imputation** — Domain-informed fills (`Tenure → 0`, `DaySinceLastOrder → max`, `CouponUsed → 0`)
* **Category merging** — Phone/Mobile Phone, CC/Credit Card, Mobile/Mobile Phone
* **Encoding** — OHE for nominal categoricals, StandardScaler for numericals
* **Pipeline** — `sklearn Pipeline` + `ColumnTransformer` for clean preprocessing
* **Models** — Logistic Regression → Random Forest (`GridSearchCV`) → XGBoost

## 6. Results Table

| Model                 | ROC-AUC | Recall (Churners) | Precision (Churners) | False Negatives |
| --------------------- | ------: | ----------------: | -------------------: | --------------: |
| Logistic Regression   |    0.89 |              0.51 |                 0.72 |              93 |
| Random Forest (Tuned) |    0.97 |              0.63 |                 0.90 |              70 |
| XGBoost               |    0.98 |              0.84 |                 0.90 |              51 |

## 7. Why Recall Over Accuracy

Missing an actual churner (**False Negative**) = lost customer = lost lifetime value.

A false alarm (**False Positive**) = unnecessary retention offer = small cost.

Therefore, **recall on churners is the primary optimisation metric, not accuracy.**

## 8. Key EDA Findings

* Complaint rate is **3× higher among churners** compared to non-churners.
* Low-tenure customers churn significantly more than long-term customers.
* Single customers churn at higher rates than married customers.
* `DaySinceLastOrder` is positively correlated with churn — inactive customers are at higher risk.

## 9. SHAP Insights (Feature Importance)

* **Tenure** — Dominant feature, 3× more important than any other. New customers are at the highest risk.
* **Complain** — Second strongest signal. Every complaint is a churn warning.
* **NumberOfAddress** — Unexpected third place. More addresses = higher churn risk, possibly indicating lower platform commitment.
* **SatisfactionScore** — Low satisfaction strongly predicts churn.
* **DaySinceLastOrder** — Inactive customers are at elevated risk, confirming the imputation decision.

## 10. Unexpected Finding

`NumberOfAddress` was the **3rd most important churn predictor** — not obvious from domain knowledge alone.

Customers with more registered addresses churn at higher rates, possibly indicating lower platform commitment or customers testing the service across multiple locations.

## 11. Business Recommendations

* **New customer onboarding program** — First 0–6 months are critical; tenure is the biggest signal.
* **Complaint resolution fast-track** — Resolve complaints within 24 hours and follow up proactively.
* **Re-engagement campaign for inactive customers** — Target the high `DaySinceLastOrder` segment before they're gone.
* **Cashback as a retention tool** — Higher cashback slightly reduces churn; consider targeted offers for at-risk segments.

## 12. Technical Notes

* Random Forest showed overfitting (**Train recall 1.0, Test recall 0.91**) with default parameters — documented with constrained comparison.
* XGBoost's built-in regularisation handled this more elegantly.
* `GridSearchCV` was used with `scoring='recall'` to optimise for the business-relevant metric.

## 13. How to Run

```bash
git clone https://github.com/kulkarni-abhishek196/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
python main.py
```

**For Streamlit app:**

```bash
streamlit run app.py
```

## 14. Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Streamlit
