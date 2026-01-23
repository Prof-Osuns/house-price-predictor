# California House Price Predictor

A machine learning web application that predicts house prices in California based on location and neighborhood charateristics.

## Live Demo
Try it here: https://house-price-predictor-ayomikun.streamlit.app/ 

## Model Performance
- **Algorithm**: Random Forest Regressor
- **R² Score**: 0.8041 (explains 80% of price variance)
- **Mean Absolute Error**: $32,968
- **RMSE**: $50,665

## Features
- Predicts prices based on 8 neighborhood charateristics
- Interactive web interface built with Streamlit
- Trained on 20,640 California housing samples
- Custom feature engineering for improved accuracy

## How It Works
The model uses census block data including:
- Median income of the area
- House age
- Average room and bedrooms
- Geographic location (latitude/longitude)

## Technical Implementation
- Built Random Forest, Gradient Boosting, and Linear Regression
- Compared model performance using RMSE, R², and MAE metrics
- Applied feature engineering (rooms per household, bedrooms per room ratios)
- Deployed using Streamlit Cloud

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

## Project Structure
- app.py - Streamlit web application
- train_model.py - Model training script
- best_model.pkl - Trained Random Forest model
- scaler.pkl - Feature scaler
- requirements.txt - Dependencies

## Author
Ayomikun Osunseyi - Aspiring AI/ML Engineer
