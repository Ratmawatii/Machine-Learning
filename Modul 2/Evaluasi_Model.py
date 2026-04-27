import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Data
X = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)
Y = np.array([300, 420, 500, 600, 750])

#Train Model 
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

#evaluasi
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")    
print(f"R-squared (R2 Score): {r2:.4f}")