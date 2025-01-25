import streamlit as st
import requests
import numpy as np

# Title
st.title('Customer Journey Analysis and Segmentation')

# Input fields for customer data
age = st.number_input('Enter Age', min_value=18, max_value=100, value=30)
income = st.number_input('Enter Income (in $)', min_value=1000, max_value=100000, value=50000)
spending_score = st.number_input('Enter Spending Score (0-100)', min_value=0, max_value=100, value=50)

# Button to get customer segmentation
if st.button('Get Customer Segmentation'):
    # Prepare input data
    input_data = {"age": age, "income": income, "spending_score": spending_score}
    # Send data to API
    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.write(f"Customer Cluster: {result['cluster']}")
        st.write(f"Encoded Data: {result['encoded_data']}")
    else:
        st.write(f"Error: {response.status_code}, {response.text}")
