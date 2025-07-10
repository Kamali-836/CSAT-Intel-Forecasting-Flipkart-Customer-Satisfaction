import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD

# Load the model and preprocessor
model = joblib.load('random_forest_model.joblib')
svd = joblib.load('svd_transformer.joblib')

# Define the features
categorical_features = ['channel_name', 'category', 'Agent Shift']
numerical_features = ['Item_price', 'response_time_mins']

# Recreate the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

def main():
    st.title("🛒 Flipkart CSAT Prediction App")
    
    # Create form
    with st.form("my_form"):
        # Input fields
        channel_name = st.selectbox("Channel Name", ['Inbound', 'Outcall', 'Email', 'Self-service'])
        category = st.selectbox("Issue Category", ['Returns', 'Order Related', 'Refunds', 'Product Queries', 'Payments'])
        agent_shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Night'])
        item_price = st.number_input("Item Price (INR)", min_value=0, value=1000)
        response_time = st.number_input("Response Time (minutes)", min_value=0, value=10)
        
        # THIS IS THE IMPORTANT LINE - Creates the submitted variable
        submitted = st.form_submit_button("Predict CSAT")
    
    # Check if form was submitted
    if submitted:
        # Create input data
        input_data = pd.DataFrame({
    'channel_name': [channel_name],
    'category': [category],
    'Agent Shift': [agent_shift],
    'Item_price': [item_price],
    'response_time_mins': [response_time]
})

        
        # Transform data
        processed_data = preprocessor.fit_transform(input_data)
        processed_data = svd.transform(processed_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        st.write(f"Predicted CSAT: {prediction[0]}")

if __name__ == '__main__':
    main()