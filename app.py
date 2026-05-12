import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load('random_forest_churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Telco Customer Churn Prediction")

st.write("Enter customer information to predict churn.")

# User Inputs
gender = st.selectbox("Gender", ['Male', 'Female'])

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

Partner = st.selectbox("Has Partner", ['Yes', 'No'])

Dependents = st.selectbox("Has Dependents", ['Yes', 'No'])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])

MultipleLines = st.selectbox(
    "Multiple Lines",
    ['Yes', 'No', 'No phone service']
)

InternetService = st.selectbox(
    "Internet Service",
    ['DSL', 'Fiber optic', 'No']
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ['Yes', 'No', 'No internet service']
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ['Yes', 'No', 'No internet service']
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ['Yes', 'No', 'No internet service']
)

TechSupport = st.selectbox(
    "Tech Support",
    ['Yes', 'No', 'No internet service']
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ['Yes', 'No', 'No internet service']
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ['Yes', 'No', 'No internet service']
)

Contract = st.selectbox(
    "Contract Type",
    ['Month-to-month', 'One year', 'Two year']
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ['Yes', 'No']
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Feature Engineering
NumServices = sum([
    OnlineSecurity == 'Yes',
    OnlineBackup == 'Yes',
    DeviceProtection == 'Yes',
    TechSupport == 'Yes',
    StreamingTV == 'Yes',
    StreamingMovies == 'Yes'
])

IsNewCustomer = int(tenure < 12)

HighMonthlyCharges = int(MonthlyCharges > 70)

HasFiberOptic = int(InternetService == 'Fiber optic')

LongTermContract = int(
    Contract in ['One year', 'Two year']
)

AutoPayment = int(
    PaymentMethod in [
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
)

# Create DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'NumServices': [NumServices],
    'IsNewCustomer': [IsNewCustomer],
    'HighMonthlyCharges': [HighMonthlyCharges],
    'HasFiberOptic': [HasFiberOptic],
    'LongTermContract': [LongTermContract],
    'AutoPayment': [AutoPayment]
})

# Prediction
if st.button("Predict Churn"):

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    if prediction == 1:
        st.error(
            f"Customer is likely to churn.\n\nProbability: {probability:.2%}"
        )

    else:
        st.success(
            f"Customer is likely to stay.\n\nProbability of churn: {probability:.2%}"
        )