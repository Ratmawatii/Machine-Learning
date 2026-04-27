from sklearn.model_selection import GridSearchCV
from Persiapan_data import y_train
from preprocessing_Data import X_train_processed, X_test_processed
from sklearn.ensemble import RandomForestRegressor

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train_processed, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

