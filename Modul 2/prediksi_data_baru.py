from sklearn.linear_model import LinearRegression
from Persiapan_data import X_train, y_train
from preprocessing_Data import X_test_processed
import pandas as pd
from preprocessing_Data import preprocessor

#training modeel
model = LinearRegression() 
model.fit(X_train, y_train) #data input dan target

#contoh data baru
data_baru = pd.DataFrame({
    'luas_tanah': [120, 80],
    'jumlah_kamar': [3, 2],
    'jarak_ke_pusat_kota': [5, 15],
    'tahun_dibangun': [2010, 2000]
})

#prepocess + prediksi
data_baru_processed = preprocessor.transform(data_baru)
prediksi_harga = model.predict(data_baru_processed)

print("Prediksi Harga Rumah (juta Rp):", prediksi_harga.round(2))