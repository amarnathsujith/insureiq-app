"""
train_model.py — Train and save the insurance prediction pipeline.
Run this once before starting the server, or whenever you want to retrain.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, 'insurance.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'insurance_model.pkl')

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df = df.dropna()
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop(columns='charges')
Y = df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    ))
])

print("Training model...")
pipeline.fit(X_train, Y_train)

train_r2 = pipeline.score(X_train, Y_train)
test_r2  = pipeline.score(X_test, Y_test)
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")

joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
print("Done! You can now run: python app.py")
