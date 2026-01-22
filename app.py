import streamlit as st
import pickle
import numpy as np
import pandas as pd

#Load the saved model and scaler
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

#App title and description
st.title("California House Price Predictor")
st.write("Predict house prices based on California housing data using Machine Learning")
st.write("---")

#Create two columns for input
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income(in $10,000s)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
    house_age = st.number_input("House Age(years)", min_value=1, max_value=52, value=25)
    ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=2.0, step=0.1)
    ave_bedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.1)

with col2:
    population = st.number_input("Population", min_value=3, max_value=1000)
    ave_occup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0, step=0.01)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.0, step=0.01)

#Predict button
if st.button("Predict Price", type="primary"):
    #Create feature engineering(same as training)
    rooms_per_household = ave_rooms / ave_occup
    bedrooms_per_room = ave_bedrms / ave_rooms
    population_per_household = population / house_age

    #Create feature array in correct order
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, rooms_per_household, bedrooms_per_room, population_per_household]])

    #Scale features
    features_scaled = scaler.transform(features)

    #Make Prediction
    prediction = model.predict(features_scaled)[0]

    #Display result
    st.success(f"### Predicted House Price: ${prediction:,.2f}")
    st.balloons()

#Add info section
st.write("---")
st.write("### About This Model")
st.write("- **Model**: Random Forest Regressor")
st.write("- **R² Score**: 0.8041")
st.write("- **Mean Absolute Error**: $32,968")
st.write("- **Dataset**: California Housing (20,640 samples)")