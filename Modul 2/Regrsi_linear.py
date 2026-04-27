import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Data contoh: luas tanah (X) dan harga rumah (Y)
X = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)  #fitur dalam meter persegi
Y = np.array([300, 450, 500, 600, 700])  # target (dalam rp)

#latih model regresi linear
model = LinearRegression()
model.fit(X, Y)

# prediksi harga untuk luas 100m2
prediksi = model.predict(np.array([[100]]))
print("Prediksi harga untuk luas 100 m2: Rp {prediksi[0]:.0f} juta")

#visualisasi 
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, model.predict(X), color='red', label='Garis Regrsi')
plt.xlabel('luas tanah (m2)')
plt.ylabel('Harga Rumah (juta RP)')
plt.legend()
plt.show()


