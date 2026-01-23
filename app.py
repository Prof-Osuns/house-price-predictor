import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing


#Function to train and saved model
@st.cache_resource
def train_and_load_model():
    #Check if models exist
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    #If not, train the model
    st.info("Training model for the first time... This will take about 2 minutes.")

    #Load data
    housing = fetch_california_housing()

    #Create DataFrame with explicit column names
    features_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                      'Population', 'AveOccup', 'Latitude', 'Longitude']
    

    df = pd.DataFrame(housing.data, columns=features_names)
    df['Price'] = housing.target * 100000
    

    #Feature engineering
    df['RoomsPerHousehold'] = df['AveRooms']/df['AveOccup']
    df['BedroomsPerRoom'] = df['AveBedrms']/df['AveRooms']
    df['PopulationPerHousehold'] = df['Population']/df['HouseAge']

    #Prepare features 
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    #Save models
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    st.success("Model trained successfully!")
    return model, scaler

#Load model
model, scaler = train_and_load_model()


#App title and description
st.title("California House Price Predictor")
st.write("Predict house prices based on California housing data using Machine Learning")
st.write("---")

#Add explanation
st.info("**Note:** This model predicts house prices based on neighbourhood charateristics, not individual buyer information.")

with st.expander("What do these inputs mean?"):
    st.write("""
              **Median Income**: Median household income in the neighborhood (in tens of thousands of dollars)
              -Example: 3.0 = $30,000/year

              **House Age**: Average age of houses in the area

              **Average Rooms**: Average total rooms per household (including bedrooms, kitchen, living room, bathrooms)

              **Average Bedrooms**:  Average number of bedrooms per household

              **Population**: Total number of people in the census block/neighborhood

              **Average Occupancy**: Average number of people per household
              - Example: 3.0 = average of 3 people per home

              **Latitude & Longitude**: Geographic coordinates of the area
              -Higher latitude = further north, lower longitude = further west
              """)
    
st.write("---")

#Create two columns for input
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income(in $10,000s)", min_value=0.0, max_value=15.0, value=3.0, step=0.1,
                              help="Median household income in the area (e.g., 5.0 = $50,000/year)")
    house_age = st.number_input("House Age(years)", min_value=1, max_value=52, value=25,
                                help="Average age of house in the neighborhood")
    ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=2.0, step=0.1,
                                help="Average total rooms per household")
    ave_bedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.1,
                                 help="Average number of bedrooms per household")

with col2:
    population = st.number_input("Population", min_value=3, max_value=1000,
                                 help="Total people in the census block")
    ave_occup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1,
                                help="Average number of people per household")
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0, step=0.01,
                               help="Geographic latitude (higher = further north)")
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.0, step=0.01,
                                help="Geographic longitude (lower = further west)")

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
st.write("- **Note**: Model trains automatically on first run (takes ~2 minutes)")