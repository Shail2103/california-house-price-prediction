import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set up page configurations
st.set_page_config(page_title="California House Price Predictor", layout="centered")

st.title("🏡 California House Price Prediction Dashboard")
st.write("""
This interactive web app predicts the **median house value** for a California district 
based on the classic California Housing dataset.
""")
st.markdown("---")

# Load the pre-trained model pipeline safely
@st.cache_resource
def load_model():
    try:
        with open('housing_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model file 'housing_model.pkl' not found! Please run 'train_model.py' first.")
        return None

model = load_model()

if model is not None:
    st.header("🔧 Input District Features")
    
    # Layout splits for cleaner look
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.slider("Median Income (in $10k blocks)", 0.5, 15.0, 3.8, help="Median income for households in the block group.")
        house_age = st.slider("Median House Age", 1, 52, 28)
        ave_rooms = st.slider("Average Rooms per Household", 1.0, 10.0, 5.4)
        ave_bedrms = st.slider("Average Bedrooms per Household", 0.5, 5.0, 1.1)

    with col2:
        population = st.number_input("Block Population", min_value=3, max_value=35000, value=1425)
        ave_occup = st.slider("Average House Occupancy (Members/HH)", 1.0, 10.0, 3.0)
        latitude = st.slider("Latitude coordinate", 32.5, 42.0, 35.6)
        longitude = st.slider("Longitude coordinate", -124.3, -114.3, -119.5)

    # Compile the slider data into a Dataframe structured exactly like the training data
    input_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrms],
        'Population': [population],
        'AveOccup': [ave_occup],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    st.markdown("---")
    
    # Trigger Prediction
    if st.button("🔮 Estimate House Value", type="primary"):
        # The target variable is in units of $100,000
        prediction = model.predict(input_data)[0] * 100000
        
        # Display the result formatted nicely
        st.success(f"### 🎉 Estimated Median House Value: **${prediction:,.2f}**")
        
        # Micro context map feature using Streamlit's native map utility
        st.write("📍 **Target District Location Map:**")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=6)