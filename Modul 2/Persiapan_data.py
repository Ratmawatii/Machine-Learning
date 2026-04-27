import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Gemertae data acak
np.random.seed(0)
data = {
    'luas_tanah': np.random.randint(50, 200, 100),
    'jumlah_kamar': np.random.randint(1, 5, 100),
    'jarak_ke_pusat_kota': np.round(np.random.randint(1, 20, 100), 1),
    'tahun_dibangun': np.random.randint(1990, 2020, 100),
    'harga': 0 #akan dihitung
}
# Buat DataFrame
df = pd.DataFrame(data)

#simulasikan harga berdasarkan rumus + noise
df['harga'] = (
    5* df['luas_tanah'] +
    50* df['jumlah_kamar'] -
    10* df['jarak_ke_pusat_kota'] +
    0.5* (2023 - df['tahun_dibangun']) +
    np.random.normal(0, 50, 100) 

)

#split data: 80% training, 20% testing
X = df.drop('harga', axis=1)
y = df['harga']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
