
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Create project folders if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Generate reproducible demonstration data
rng = np.random.default_rng(42)
n = 1200

df = pd.DataFrame({
    "Attendance": rng.integers(30, 101, n),
    "Study_Hours": rng.integers(0, 9, n),
    "Previous_Marks": rng.integers(0, 101, n)
})

# Demonstration labels based on an explicit scoring rule
score = (
    df["Attendance"] * 0.30
    + np.minimum(df["Study_Hours"] / 8, 1) * 100 * 0.20
    + df["Previous_Marks"] * 0.50
)

df["Performance"] = np.where(
    score >= 75, "High",
    np.where(score >= 50, "Average", "Needs Support")
)

# Save the generated dataset
df.to_csv("data/student_training_data.csv", index=False)

X = df[["Attendance", "Study_Hours", "Previous_Marks"]]
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model training completed!")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the complete model pipeline
joblib.dump(model, "model/performance_model.joblib")

print("\nModel saved to model/performance_model.joblib")
print("Demo dataset saved to data/student_training_data.csv")
