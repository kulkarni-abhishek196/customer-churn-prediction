import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Customer Churn prediction!")
st.write("Insert your inputs")

cust_tenure = st.number_input("Enter the customer Tenure")

device_options = ["Phone", "Computer"]
cust_device = st.selectbox(
    label="Select customer device:",
    options=device_options,
    index=0
)

cityTier_option = [1, 2, 3]
cust_cityTier = st.selectbox(
    label="Select customer CityTier",
    options=cityTier_option,
    index=0
)

cust_warehouseToHome = st.number_input("Enter warehouse distance to customer's house")

paymentOptions = ["Credit Card", "Debit Card", "E-Wallet", "UPI", "COD"]
cust_paymentMode = st.selectbox(
    label="Select payment options",
    options=paymentOptions,
    index=0
)

GenderOptions = ["Male", "Female"]
cust_Gender = st.selectbox(
    label="Select customer gender",
    options=GenderOptions,
    index=0
)

cust_HoursSpendOn = st.number_input("Enter customer number of hours spent on the App")

cust_NumberOfDeviceRegistered = st.number_input("Enter the number of Devices registered by Customer")

PreferedcategoryOptions = ["Phone", "Laptop & Accessory", "Fashion", "Grocery", "Others"]
cust_preferedCategoryOptions = st.selectbox(
    label="Select customer's prefered category options",
    options=PreferedcategoryOptions,
    index=0
)

cust_SatisfactionScore = st.number_input("Enter customer Satisfaction score:")

MaritalStatusOptions = ["Single", "Divorced", "Married"]
cust_MaritalStatus = st.selectbox(
    label="Enter customer's Marital Status",
    options=MaritalStatusOptions,
    index=0
)

cust_NumberOfAddress = st.number_input("Enter Customer number of addresses")

complainOptions = [0,1]
cust_Complain = st.selectbox(
    label="If customer has registered any complain select '1' for yes and '0' for no",
    options=complainOptions,
    index=0
)

cust_OrderAmountHike = st.number_input("Enter number of orders increased by customers in last year")

cust_CouponUsed = st.number_input("Enter number of coupons used by customer")

cust_OrderCount = st.number_input("Enter number of orders by customer")

cust_DaySinceLastOrder = st.number_input("Enter number of days since the customer has ordered")

cust_CashbackAmount = st.number_input("Enter Cashback Amount provided to customer")


model = joblib.load('models/churn_model.pkl')

if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        'Tenure_filled': cust_tenure,
        'PreferredLoginDevice': cust_device,
        'CityTier': cust_cityTier,
        'WarehouseToHome_filled': cust_warehouseToHome,
        'PreferredPaymentMode': cust_paymentMode,
        'Gender': cust_Gender,
        'HourSpendOnApp_filled': cust_HoursSpendOn,
        'NumberOfDeviceRegistered': cust_NumberOfDeviceRegistered,
        'PreferedOrderCat': cust_preferedCategoryOptions,
        'SatisfactionScore': cust_SatisfactionScore,
        'MaritalStatus': cust_MaritalStatus,
        'NumberOfAddress': cust_NumberOfAddress,
        'Complain': cust_Complain,
        'OrderAmountHikeFromlastYear_filled': cust_OrderAmountHike,
        'CouponUsed_filled': cust_CouponUsed,
        'OrderCount_filled': cust_OrderCount,
        'DaySinceLastOrder_filled': cust_DaySinceLastOrder,
        'CashbackAmount': cust_CashbackAmount
    }])

    #predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN (probability: {probability:.1%})")
    else:
        st.success(f"✅ Customer is likely to STAY (probability of churn: {probability:.1%})")
    

