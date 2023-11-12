import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df = pd.read_csv('sandbox/data/processed_data.csv')

X = df.drop("GDP", axis=1)
y = df['GDP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models

# SVR
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Gradient boosting
gb_regressor = GradientBoostingRegressor(random_state=42)
gb_regressor.fit(X_train, y_train)

# Grid

svr_grid = {
    'C': [0.1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', ],
    'epsilon': [0.1,]
}

rf_grid = {
    'n_estimators': [50],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10],
    'min_samples_split': [2,],
    'min_samples_leaf': [1, 2]
}

gb_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1],
    'subsample': [0.8]
}

best_params_dict = {}

for model_name, model, grid in ('SVR', svr_model, svr_grid), ("RF", rf_model, rf_grid), ("GB", gb_regressor, gb_grid):
    print("Starting params search for model: ", model_name)

    grid_search = GridSearchCV(
      estimator=model,
      param_grid=grid,
      cv=3,
      n_jobs=-1,
      verbose=10,
      refit=False,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_params_dict[model_name] = best_params

    print(f"Los mejores parametros para el modelo '{model_name}' son:")
    print(best_params)

print(best_params_dict)
