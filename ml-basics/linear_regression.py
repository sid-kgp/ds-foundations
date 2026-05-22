import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. LOAD DATASET ──────────────────────────────────────
# Built into scikit-learn — no download needed
housing = fetch_california_housing()

# Convert to DataFrame so it looks familiar
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target  # target is in $100,000s

print("=" * 55)
print("DATASET OVERVIEW:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nBasic stats:")
print(df.describe().round(2))

# ── 2. MISSING VALUES CHECK ──────────────────────────────
print("\n" + "=" * 55)
print("MISSING VALUES:")
print(df.isnull().sum())

# ── 3. FEATURES AND TARGET ───────────────────────────────
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
X = df[features]
y = df["MedHouseVal"]

print("\n" + "=" * 55)
print(f"Features shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"Target range   : ${y.min()*100000:.0f} to ${y.max()*100000:.0f}")

# ── 4. TRAIN TEST SPLIT ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]:,} houses")
print(f"Test set    : {X_test.shape[0]:,} houses")

# ── 5. SCALE ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 6. TRAIN ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\n✓ Model trained!")

# ── 7. PREDICT ───────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

# ── 8. EVALUATE ──────────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 55)
print("MODEL PERFORMANCE:")
print(f"  R² Score : {r2:.3f}   ← how much variance explained (1.0 = perfect)")
print(f"  RMSE     : {rmse:.3f}  ← avg error in $100k units (${rmse*100000:.0f})")
print(f"  MAE      : {mae:.3f}  ← avg absolute error (${mae*100000:.0f})")

# ── 9. PREDICTIONS VS ACTUAL ─────────────────────────────
print("\n" + "=" * 55)
print("SAMPLE PREDICTIONS vs ACTUAL:")
print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12}")
for actual, pred in zip(y_test[:8], y_pred[:8]):
    error = pred - actual
    print(f"${actual*100000:>10,.0f}  ${pred*100000:>10,.0f}  ${error*100000:>+10,.0f}")

# ── 10. FEATURE IMPORTANCE ───────────────────────────────
print("\n" + "=" * 55)
print("WHAT DRIVES HOUSE PRICES:")
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)
print(coef_df.to_string(index=False))

# ── 11. VISUALISE ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Linear Regression — California Housing", fontweight="bold")

# Chart 1 — Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, color="steelblue", s=10)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", linewidth=2, label="Perfect prediction")
axes[0].set_xlabel("Actual Price ($100k)")
axes[0].set_ylabel("Predicted Price ($100k)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# Chart 2 — Residuals (errors)
residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residuals Distribution\n(should be centred at 0)")

plt.tight_layout()
plt.savefig("ml-basics/housing_regression.png", dpi=150)
plt.show()
print("\nChart saved ✓")