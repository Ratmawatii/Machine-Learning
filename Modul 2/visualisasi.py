from pyexpat import features

import matplotlib.pyplot as plt
from Persiapan_data import X
from Model import model


#feature importance
importances = model.feature_importances_
features_names = X.columns

plt.barh(features_names, importances)
plt.xlabel('importance Score')
plt.title('kontribusi fitur terhadap prediksi harga')
plt.show()
