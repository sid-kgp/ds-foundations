import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ── 1. LOAD & CLEAN ──────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Quick clean
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.dropna(subset=["Embarked"])
df = df.drop(columns=["Cabin"])

# ── 2. SELECT FEATURES ───────────────────────────────────
# Convert Sex to numbers — ML models can't read text
df["Sex_encoded"] = df["Sex"].map({"female": 1, "male": 0})

features = ["Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare"]
X = df[features]
y = df["Survived"]

print("=" * 55)
print("FEATURES (X):")
print(X.head())
print(f"\nTarget (y) first 5: {y.head().tolist()}")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── 3. TRAIN TEST SPLIT ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} passengers")
print(f"Test set    : {X_test.shape[0]} passengers")

# ── 4. SCALE THE FEATURES ────────────────────────────────
# Age ranges 0-80, Fare ranges 0-500 — very different scales
# Scaling brings everything to the same range
# Why? So Age doesn't get ignored just because Fare is bigger
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Note: transform only, not fit_transform

print("\n" + "=" * 55)
print("AFTER SCALING — first row:")
print(f"Before: {X_train.iloc[0].tolist()}")
print(f"After : {X_train_scaled[0].tolist()}")

# ── 5. TRAIN THE MODEL ───────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("\n✓ Model trained!")

# ── 6. MAKE PREDICTIONS ──────────────────────────────────
y_pred = model.predict(X_test_scaled)
print(f"\nFirst 10 predictions : {y_pred[:10].tolist()}")
print(f"First 10 actual      : {y_test[:10].tolist()}")

# ── 7. EVALUATE ──────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*55}")
print(f"MODEL ACCURACY: {accuracy*100:.1f}%")
print(f"\nDETAILED REPORT:")
print(classification_report(y_test, y_pred,
      target_names=["Died", "Survived"]))

# ── 8. WHAT DID THE MODEL LEARN? ─────────────────────────
print("=" * 55)
print("FEATURE IMPORTANCE (coefficients):")
for feature, coef in zip(features, model.coef_[0]):
    direction = "↑ survival" if coef > 0 else "↓ survival"
    print(f"  {feature:15} {coef:+.3f}  {direction}")