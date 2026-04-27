import pandas  as pd
from sklearn.linear_model import LinearRegresion
from sklearn.linear_model import LinearRegression

#data cotoh multifariat 
data = {
    'Luas (m2)': [50, 70, 90, 110, 130],
    'Kamar': [2, 3, 3, 4, 4],
    'jarak (km)': [10, 5, 3, 2, 1],
    'Harga (juta Rp)': [300, 420, 500, 600, 750]
}
#buat dataframe
df = pd.DataFrame(data)

#pisahkan fitur (x) dan target (y)
X = df[['Luas (m2)', 'Kamar', 'jarak (km)']]
y = df['Harga (juta Rp)']

#Latih model 
model = LinearRegression()
model.fit(X, y)

#KOefisien regresi
print(f"Intercept (B0): {model.intercept_}")
print(f"Koefisien (B1, B2, B3): {model.coef_}")

#prediksi untuk rumah: 100m2, 3 kamar, 4 km dari pusat kota
prediksi = model.predict([[100, 3, 4]])
print(f"Prediksi harga: Rp {prediksi[0]:.0f} juta")

