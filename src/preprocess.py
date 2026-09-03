import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_and_preprocess(filePath):

    df = pd.read_csv(filePath)

    # 1. there are multiple columns which have null values

    #Tenure it has empty values customers must have just started so imputing the 0
    df['Tenure_filled'] = df['Tenure'].fillna(0)

    #WarehghouseToHome => The customers are categorised into the city tiers and that's why we will be using the median distance of the city tier
    df['WarehouseToHome_filled'] = df['WarehouseToHome'].fillna(df.groupby('CityTier')['WarehouseToHome'].transform('median'))

    #HourSpendOnApp => imputing the median values at empty places
    df['HourSpendOnApp_filled'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median())

    #OrderAmountHikeFromLastYear => 
    df['OrderAmountHikeFromlastYear_filled'] = df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median())

    #CouponUsed => empty values could simply mean that the coupon are not used at all
    df['CouponUsed_filled'] = df['CouponUsed'].fillna(0)

    #OrderCount
    df['OrderCount_filled'] = df['OrderCount'].fillna(0)

    #DaySinceLastOrder
    df['DaySinceLastOrder_filled'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].max())


    #2. There are multiple columns which have same/similar categorical values
    # PreferredLoginDevice has Mobile Phone and Phone so will merged it to one
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Mobile Phone': 'Phone', 'Phone':'Phone'})

    # PreferredPaymentMode has CC and Credit card and both mean the same thing
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'CC': 'Credit Card', 'Credit Card':'Credit Card', 'Cash on Delivery': 'COD', 'COD': 'COD'})

    # PreferedOrderCat has the Mobile and Mobile Phone will merge too
    df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone', 'Mobile Phone': 'Mobile Phone'})


    #dropping the customerID column since it's not going to contribute
    df = df.drop(columns=['CustomerID'])

    #dropping the columns which has Nan values since we've already imputed those with other columns
    df = df.drop(columns=['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder'])

    #Convert the categorical columns to numerical using one-hot encoding
    #Why one-hot encoding? because there is no proper order being followed with the categorical values
    numerical_cols = [
    'Tenure_filled', 'WarehouseToHome_filled', 'HourSpendOnApp_filled',
    'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear_filled', 'CouponUsed_filled',
    'OrderCount_filled', 'DaySinceLastOrder_filled', 'CashbackAmount',
    'CityTier'  # numeric ordinal, StandardScaler is fine
    ]

    categorical_cols = [
    'PreferredLoginDevice', 'PreferredPaymentMode',
    'Gender', 'PreferedOrderCat', 'MaritalStatus'
    ]

    #dropping the churn column
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    #pipeline amd applying transformations
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    return X, y, preprocessor
