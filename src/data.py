from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset_split():
    """Return the exact dataset split used to train the models.
    
    This ensures that scripts/main.py evaluates every model on the 
    same test set with identical preprocessing.
    """
    
    # ====================== 1. LOAD & CLEAN ======================
    df = pd.read_csv("../data/sleep-health-and-lifestyle-dataset/Sleep_health_and_lifestyle_dataset.csv")

    # Cleaning (must be identical to training script)
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna("Normal")

    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop(columns=['Person ID', 'Blood Pressure'])

    # ====================== 2. FEATURES & TARGET ======================
    num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']

    cat_cols = ['Gender', 'Occupation', 'BMI Category']

    X = df[num_cols + cat_cols]
    y = df['Sleep Disorder']

    # One-hot encoding (must match training)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Label encoding for target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ====================== 3. TRAIN/TEST SPLIT (exact same parameters) ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test
