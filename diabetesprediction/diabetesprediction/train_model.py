import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
]
TARGET_NAME = 'Outcome'

def find_dataset():
    current_dir = Path(__file__).resolve().parent
    local_path = current_dir / 'diabetes.csv'
    if local_path.exists():
        return local_path
    parent_path = current_dir.parent / 'diabetes.csv'
    if parent_path.exists():
        return parent_path
    raise FileNotFoundError(f"Dataset not found at {local_path} or {parent_path}")

def train_and_save_model(model_filename='diabetes_model.joblib'):
    dataset_path = find_dataset()
    print(f"Loading dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path)

    X = data[FEATURE_NAMES]
    y = data[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Training LogisticRegression pipeline with StandardScaler...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    model_path = Path(__file__).resolve().parent / model_filename
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully to: {model_path}")

    # Verify saved model can be loaded and predict
    loaded = joblib.load(model_path)
    test_sample = X_test.iloc[0:1]
    original_pred = pipeline.predict(test_sample)
    loaded_pred = loaded.predict(test_sample)
    assert (original_pred == loaded_pred).all(), "Loaded model predictions do not match!"
    print("Verification passed: Loaded model matches original predictions.")
    return model_path, accuracy

if __name__ == '__main__':
    train_and_save_model()
