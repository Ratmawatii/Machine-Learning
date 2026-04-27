from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from Persiapan_data import X_test, X_train


# sementara (tanpa preprocessing)
X_test_processed = X_test

#Definisikan transromer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['luas_tanah', 'jarak_ke_pusat_kota', 'tahun_dibangun']),
        # jika ada kategorikal: ('cat', OneHotEncoder(), ['kategori_fitur'])
    ]
)

#apply prepocessing
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

#update variabel processed
X_train_processed = X_train_scaled
X_test_processed = X_test_scaled
