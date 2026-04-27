import sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from preprocessing_Data import X_test_processed

#training modeel
model = LinearRegression() 


#Data aktual (y) dan prediksi (y_pred)
Y = np.array([300, 420, 500, 600, 750])
y_pred = model.predict(X_test_processed)  #misal: {300, 410, 490, 610, 740}

#Hitung MSE
mse = mean_squared_error(Y, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

