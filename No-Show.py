import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load("final_rf_model.pkl")

# Define preprocessing function
def preprocess_data(data):
    # New feature combining Gender and Age buckets
    data['Age_bucket'] = pd.cut(data['Age'], bins=[0, 12, 18, 30, 50, 120], labels=['0-12', '13-18', '19-30', '31-50', '51+'])
    data['Gender_Age'] = data['Gender'] + "_" + data['Age_bucket'].astype(str)
    
    # New feature for days till appointment
    data['Days_Till_Appointment'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days
    data['Days_Till_Appointment'] = data['Days_Till_Appointment'].apply(lambda x: 0 if x < 0 else x)
    
    # Encoding categorical features
    label_encoders = {}
    for column in ['Gender', 'Neighbourhood', 'Age_bucket', 'Gender_Age']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data

# Streamlit UI
st.title("Medical Appointment No-Show Predictor")

st.sidebar.header("Patient Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
neighbourhood = st.sidebar.selectbox("Neighbourhood", options=['JARDIM DA PENHA', 'MATA DA PRAIA', 'PONTAL DE CAMBURI', '...'])  # Add more neighbourhoods as per the dataset
scholarship = st.sidebar.checkbox("Scholarship")
hipertension = st.sidebar.checkbox("Hipertension")
diabetes = st.sidebar.checkbox("Diabetes")
alcoholism = st.sidebar.checkbox("Alcoholism")
handcap = st.sidebar.checkbox("Handcap")
sms_received = st.sidebar.slider("Number of SMS Received", 0, 5, 0)
scheduled_day = st.sidebar.date_input("Scheduled Day")
appointment_day = st.sidebar.date_input("Appointment Day")

input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Neighbourhood': [neighbourhood],
    'Scholarship': [scholarship],
    'Hipertension': [hipertension],
    'Diabetes': [diabetes],
    'Alcoholism': [alcoholism],
    'Handcap': [handcap],
    'SMS_received': [sms_received],
    'ScheduledDay': [scheduled_day],
    'AppointmentDay': [appointment_day]
})

# Display user input summary
st.subheader("Patient Details Summary")
st.write(input_data.T)

# Predict button
if st.button("Predict"):
    input_data_preprocessed = preprocess_data(input_data)
    prediction = model.predict_proba(input_data_preprocessed.drop(columns=['ScheduledDay', 'AppointmentDay']))[:, 1]
    
    st.subheader("Prediction Result")
    st.write(f"The probability of the patient being a no-show is: {prediction[0]:.2f}")

st.write("Note: The prediction is based on the provided data and the trained model, actual outcomes may vary.")

# Footer
st.write("----")
st.write("Developed with ❤️ by erick Yegon")

