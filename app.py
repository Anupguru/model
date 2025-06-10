import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('random_forest_model.pkl')

st.title("Customer Churn Prediction")

# User inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure_months = st.number_input('Tenure Months', min_value=0, value=1)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=50.0)
total_charges = st.number_input('Total Charges', min_value=0.0, value=100.0)

# Manual encoding — must match what you did during training
binary_map = {'Yes': 1, 'No': 0}
gender_map = {'Male': 1, 'Female': 0}
internet_map = {'DSL': 1, 'Fiber optic': 2, 'No': 0}
multi_line_map = {'No phone service': 0, 'No': 1, 'Yes': 2}
internet_service_map = {'No internet service': 0, 'No': 1, 'Yes': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_map = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}

# Prepare input dict with encoded values
input_dict = {
    'Gender': gender_map[gender],
    'Senior Citizen': senior_citizen,
    'Partner': binary_map[partner],
    'Dependents': binary_map[dependents],
    'Tenure Months': tenure_months,
    'Phone Service': binary_map[phone_service],
    'Multiple Lines': multi_line_map[multiple_lines],
    'Internet Service': internet_map[internet_service],
    'Online Security': internet_service_map[online_security],
    'Online Backup': internet_service_map[online_backup],
    'Device Protection': internet_service_map[device_protection],
    'Tech Support': internet_service_map[tech_support],
    'Streaming TV': internet_service_map[streaming_tv],
    'Streaming Movies': internet_service_map[streaming_movies],
    'Contract': contract_map[contract],
    'Paperless Billing': binary_map[paperless_billing],
    'Payment Method': payment_map[payment_method],
    'Monthly Charges': monthly_charges,
    'Total Charges': total_charges
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button('Predict Churn'):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    result = 'Yes' if prediction == 1 else 'No'
    st.success(f'Prediction: Churn = {result}')
    st.info(f'Probability of Churn: {proba:.2f}')