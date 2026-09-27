
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('churn_model.sav')

st.set_page_config(page_title="ABC Ltd Churn Predictor", page_icon=" ")
st.title("ABC Ltd — Customer Churn Predictor")
st.write("Fill in the customer's details and click **Predict** to see their churn risk.")

expected_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Same encodings used during training
gender_map = {'Female': 0, 'Male': 1}
yesno_map = {'No': 0, 'Yes': 1}
lines_map = {'No phone service': 0, 'No': 1, 'Yes': 2}
internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
addon_map = {'No internet service': 0, 'No': 1, 'Yes': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_map = {
    'Electronic check': 0, 'Mailed check': 1,
    'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
}

st.header("Customer profile")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Gender', list(gender_map.keys()))
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Has a Partner', list(yesno_map.keys()))
    dependents = st.selectbox('Has Dependents', list(yesno_map.keys()))
    tenure = st.number_input('Tenure (months with ABC Ltd)', min_value=0, max_value=100, value=12, step=1)
with col2:
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    total_charges = st.number_input('Total Charges to Date', min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
    contract = st.selectbox('Contract Type', list(contract_map.keys()))
    paperless_billing = st.selectbox('Paperless Billing', list(yesno_map.keys()))
    payment_method = st.selectbox('Payment Method', list(payment_map.keys()))

st.header("Services subscribed")
col3, col4 = st.columns(2)
with col3:
    phone_service = st.selectbox('Phone Service', list(yesno_map.keys()))
    multiple_lines = st.selectbox('Multiple Lines', list(lines_map.keys()))
    internet_service = st.selectbox('Internet Service', list(internet_map.keys()))
    online_security = st.selectbox('Online Security', list(addon_map.keys()))
with col4:
    online_backup = st.selectbox('Online Backup', list(addon_map.keys()))
    device_protection = st.selectbox('Device Protection', list(addon_map.keys()))
    tech_support = st.selectbox('Tech Support', list(addon_map.keys()))
    streaming_tv = st.selectbox('Streaming TV', list(addon_map.keys()))
    streaming_movies = st.selectbox('Streaming Movies', list(addon_map.keys()))

if st.button('Predict Churn Risk'):
    input_data = pd.DataFrame([[
        gender_map[gender],
        1 if senior_citizen == 'Yes' else 0,
        yesno_map[partner],
        yesno_map[dependents],
        tenure,
        yesno_map[phone_service],
        lines_map[multiple_lines],
        internet_map[internet_service],
        addon_map[online_security],
        addon_map[online_backup],
        addon_map[device_protection],
        addon_map[tech_support],
        addon_map[streaming_tv],
        addon_map[streaming_movies],
        contract_map[contract],
        yesno_map[paperless_billing],
        payment_map[payment_method],
        monthly_charges,
        total_charges
    ]], columns=expected_columns)

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error(f"This customer is **likely to churn** (probability: {prediction_proba[0][1]*100:.1f}%)")
    else:
        st.success(f"This customer is **likely to stay** (probability of staying: {prediction_proba[0][0]*100:.1f}%)")

st.write("""
---
### How to run this app
1. Open a terminal in the same folder as `app.py` and `churn_model.sav`.
2. Install the requirements once: `pip install streamlit pandas scikit-learn joblib`
3. Run: `streamlit run app.py`
4. A local URL opens in your browser — share it on your network, or deploy it for free on [Streamlit Community Cloud](https://streamlit.io/cloud) so non-technical teammates can use it without installing anything.
""")
