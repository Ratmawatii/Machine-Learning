from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np #tambahan untuk np
from sklearn.linear_model import LinearRegression
from Persiapan_data import X_train, y_train
from preprocessing_Data import X_test_processed
from Persiapan_data import y_test

#training modeel
model = LinearRegression() 
model.fit(X_train, y_train) #data input dan target

#prediksi 
y_pred = model.predict(X_test_processed)

#hitung metrik
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"""
      METRIK EVALUASI:
      - Mean Squared Error (MSE): {mse:.2f}
      - Root Mean Squared Error (RMSE): {rmse:.2f} juta Rp
      - R-squared (R2 Score): {r2:.2f}
      - Mean Absolute Error (MAE): {mae:.2f} juta Rp
""")
