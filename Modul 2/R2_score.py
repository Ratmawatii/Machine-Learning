from sklearn.metrics import r2_score
from Persiapan_data import y_pred, Y

r2 = r2_score(Y, y_pred)
print(f"R-squared (R2 Score): {r2:.4f}")