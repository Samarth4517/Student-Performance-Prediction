import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 80)
print("STUDENT PERFORMANCE PREDICTION SYSTEM - MODEL TRAINING")
print("=" * 80)

# =========================================================
# STEP 1: LOAD DATASET
# =========================================================

print("\n[STEP 1] Loading Dataset...")

df = pd.read_csv("student_performance.csv")

print("✓ Dataset loaded successfully!")
print(f"Shape: {df.shape}")

print("\nDataset Preview:")
print(df.head())

# =========================================================
# STEP 2: DATA PREPROCESSING
# =========================================================

print("\n" + "=" * 80)
print("[STEP 2] Data Preprocessing...")
print("=" * 80)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Select numeric columns
numeric_df = df.select_dtypes(include=["number"])

# Remove Outliers (IQR)
print("\nRemoving outliers using IQR...")

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)

IQR = Q3 - Q1

mask = ~(
    ((numeric_df < (Q1 - 1.5 * IQR)) |
     (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
)

df = df[mask]

print(f"✓ Records Remaining: {len(df)}")

# =========================================================
# STEP 3: ENCODE TARGET
# =========================================================

print("\n[STEP 3] Encoding Target Variable...")

label_encoder = LabelEncoder()

df["Result"] = label_encoder.fit_transform(df["Result"])

print("\nClasses:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} → {i}")

# =========================================================
# STEP 4: FEATURE SELECTION
# =========================================================

print("\n[STEP 4] Selecting Features...")

X = df.drop("Result", axis=1)

y = df["Result"]

print("\nFeatures:")
print(X.columns.tolist())

# =========================================================
# STEP 5: SPLIT DATA
# =========================================================

print("\n[STEP 5] Train Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# =========================================================
# STEP 6: TRAIN DECISION TREE
# =========================================================

print("\n[STEP 6] Training Decision Tree Model...")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("✓ Model Trained Successfully!")

# =========================================================
# STEP 7: MODEL EVALUATION
# =========================================================

print("\n[STEP 7] Evaluating Model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# STEP 8: SAVE MODEL
# =========================================================

print("\n[STEP 8] Saving Model...")

with open(
    "student_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

with open(
    "label_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )

print("✓ student_model.pkl saved")
print("✓ label_encoder.pkl saved")

print("\n" + "=" * 80)
print("DECISION TREE MODEL TRAINING COMPLETED")
print("=" * 80)