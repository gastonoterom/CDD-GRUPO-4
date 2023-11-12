import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

df = pd.read_csv('sandbox/data/processed_data.csv')

X = df[['Chronic kidney disease', 'Meningitis', 'Diarrheal diseases']]
y = df['Chronic respiratory diseases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best params

best_params_dict = {
    "RF": {
        'random_state': 42,
        'n_estimators': 50,
        'max_features': 'sqrt',
        'max_depth':  10,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
    }
}

# Random Forest

model = RandomForestRegressor(**best_params_dict["RF"])
model.fit(X_train, y_train)

dump(model, 'sandbox/data/rf_model.joblib')
