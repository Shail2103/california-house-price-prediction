import pickle
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

print("Fetching California Housing Dataset...")
# Load dataset
cali_data = fetch_california_housing(as_frame=True)
X = cali_data.data
y = cali_data.target  # Expressed in hundreds of thousands of dollars ($100,000s)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor...")
# Build a pipeline combining feature scaling and the regressor
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Calculate Accuracy Score
train_score = model_pipeline.score(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

# Save the trained model pipeline
with open('housing_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Model saved successfully as 'housing_model.pkl'!")