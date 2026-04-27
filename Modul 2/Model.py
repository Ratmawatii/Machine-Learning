from sklearn.ensemble import RandomForestRegressor
from Persiapan_data import y_train
from preprocessing_Data import X_train_processed

#inilisialisasi model
model = RandomForestRegressor(n_estimators=100, random_state=42)

#training model
model.fit(X_train_processed, y_train)