import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

#Load data
print("Loading data...")
df = pd.read_csv('housing_data.csv')

#Feature engineering - create new features
df['RoomsPerHousehold'] = df['AveRooms']/df['AveOccup']
df['BedroomsPerRoom'] = df['AveBedrms']/df['AveRooms']
df['PopulationPerHousehold'] = df['Population']/df['HouseAge']

#Prepare features and target
X = df.drop('Price', axis=1)
y = df['Price']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Training Models ---\n")

#Model 1: Linear Regresssion
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_squared_error(y_test, lr_pred)


print(f"Linear Regression - RMSE: ${lr_rmse:,.2f}, R²: {lr_r2:.4f}, MAE: ${lr_mae:,.2f}\n")

#Model 2: Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"Random Forest - RMSE: ${rf_rmse:,.2f}, R²: {rf_r2:.4f}, MAE: ${rf_mae:,.2f}\n")

#Model3: Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)

print(f"Gradient Boosting- RMSE: ${gb_rmse:,.2f}, R²: {gb_r2:.4f}, MAE: ${gb_mae:,.2f}\n")

#Pick best model (lowest RMSE)
models = {'Linear Regression':(lr, lr_rmse),
          'Random Forest': (rf, rf_rmse),
          'Gradient Boosting': (gb, gb_rmse)}

best_model_name = min(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]

print(f"--- BEST MODEL: {best_model_name} ---\n")

#Save best model and scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved!")
print(f"Feature names: {list(X.columns)}")